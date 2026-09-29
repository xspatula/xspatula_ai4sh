'''
Created on 30 Sep 2026

@author: thomas gumbricht

Bulk import of eDNA ASV abundances to observation.edna_asv and
observation.edna_asv_abundance using postgres COPY.

Reached through the generic tabular processes with target process "manage_edna_asv":

- single step: insert_tabular_data (process: manage_edna_asv) - translate + load
- dual step:   translate_tabular_data (process: manage_edna_asv) - writes a canonical
               abundance csv and a manage_edna_asv json process file; running that
               file (process manage_edna_asv) loads it.

Input (excel or csv, long format, one row per ASV and sample, as written by
prepare_ai4sh_data/edna_final_results_to_xspatula.py):

    observation_log_id__observation_log_name, sample_id__sample_name, asv, taxa,
    rel_abundance [, read_count] [, raw_read_count]

asv is the delivery key of the ASV, taxa its lineage in single column notation
(k__...|p__...|...|s__...). Every row needs an existing observation (observation log +
sample) and every lineage an existing taxon (Stage 4a, manage_taxon); otherwise
nothing is loaded. An asv key already in the database under another lineage aborts
the load, since that means the laboratory changed the row order of the delivery.

Insert only: ASVs and abundances already in the database are kept, new ones added.
The whole load is one transaction.
'''

# Standard library imports
import csv

import io

import os

# Application package imports
from src.lib.pilot import Full_path_locate

from src.utils.json_read_write import Dump_json

from src.utils.csv_read_write import Read_csv, Read_excel

from src.ai4sh.import_data.import_taxon import Normalise_taxon_name, Parse_lineages, Rank_orders, Lineage_steps

TARGET_PROCESS = 'manage_edna_asv'

ABUNDANCE_FN = 'edna_asv_abundance.csv'

LOG_COL = 'observation_log_id__observation_log_name'

SAMPLE_COL = 'sample_id__sample_name'

ASV_COL = 'asv'

TAXA_COL = 'taxa'

VALUE_COL_L = ['rel_abundance', 'read_count', 'raw_read_count']

REQUIRED_COL_L = [LOG_COL, SAMPLE_COL, ASV_COL, TAXA_COL, 'rel_abundance']

CANONICAL_COL_L = [LOG_COL, SAMPLE_COL, ASV_COL, TAXA_COL] + VALUE_COL_L


def Number(value, integer=False):
    ''' Float (or int) from a tabular cell; None for blanks.
    '''

    if value is None:

        return None

    value = str(value).strip()

    if value in ('', 'null', 'nan', 'None'):

        return None

    number = float(value.replace(',', '.'))

    return int(round(number)) if integer else number


def Canonical_rows(column_L, data_L_L):
    '''
    @brief Checks the header and returns rows in CANONICAL_COL_L order, text normalised.

    @return (row_L, error_L)
    '''

    header_L = [str(c).strip().lower() for c in column_L]

    missing_L = [c for c in REQUIRED_COL_L if c not in header_L]

    if missing_L:

        return [], ['missing column(s): %s' % ', '.join(missing_L)]

    index_L = [header_L.index(c) if c in header_L else None for c in CANONICAL_COL_L]

    row_L = []

    error_L = []

    for row_nr, row in enumerate(data_L_L, start=2):

        values = [row[i] if i is not None else None for i in index_L]

        log, sample, asv, taxa = [str(v).strip().lower() if v is not None else '' for v in values[:4]]

        if not (log and sample and asv and taxa) or 'null' in (log, sample, asv, taxa):

            error_L.append('row %s: blank observation log, sample, asv or taxa' % row_nr)

            continue

        try:

            numbers = [Number(values[4]), Number(values[5], True), Number(values[6], True)]

        except ValueError:

            error_L.append('row %s: non-numeric abundance or read count' % row_nr)

            continue

        if numbers[0] is None and numbers[1] is None and numbers[2] is None:

            continue

        row_L.append([log, sample, asv, taxa] + numbers)

    return row_L, error_L


class Process_import_edna_asv():
    ''' Translate and/or load eDNA ASV abundances, see module docstring.
    '''

    def __init__(self, process_S, pg_session_C, project_root_FP, scheme_params_D=None):

        self.process_S = process_S

        self.pg_session_C = pg_session_C

        self.project_root_FP = project_root_FP

        self.scheme_params_D = scheme_params_D

        self.verbose = process_S.process.verbose

        self.parameters = process_S.process.parameters

    def _Report_failure(self, msg):

        print (msg)

        if self.pg_session_C is not None:

            self.pg_session_C.failed_process_count = getattr(self.pg_session_C, 'failed_process_count', 0) + 1

    def _Sub_process(self, json_file_key):

        process = self.process_S.process.process

        if process.startswith('translate'):

            return self._Translate()

        if not self.pg_session_C:

            print ('❌ ERROR - Inserting ASVs requires a database connection. Please define a Postgres database in scheme file.')

            return None

        if process.startswith('insert'):

            abundance_FPN = self._Translate()

            if not abundance_FPN:

                return None

            return abundance_FPN if self._Load(abundance_FPN) else None

        if process == TARGET_PROCESS:

            if self.process_S.process.delete:

                self._Report_failure('❌ ERROR - %s does not support delete; remove ASVs with SQL.' % TARGET_PROCESS)

                return None

            abundance_FPN = Full_path_locate(self.project_root_FP, self.parameters.asv_data_path)

            if not abundance_FPN:

                self._Report_failure('❌ ERROR - ASV data file not found:\n   %s' % self.parameters.asv_data_path)

                return None

            return json_file_key if self._Load(abundance_FPN) else None

        self._Report_failure('❌ ERROR - process %s not available for ASV import' % process)

        return None

    def _Method_pipeline(self):

        method_pipeline = Normalise_taxon_name(getattr(self.parameters, 'method_pipeline', None))

        if not method_pipeline:

            self._Report_failure('❌ ERROR - parameter method_pipeline is required for %s' % TARGET_PROCESS)

        return method_pipeline

    def _Translate(self):
        ''' Reads the tabular source and writes the canonical abundance csv and the
        manage_edna_asv process file to dst_path.
        '''

        method_pipeline = self._Method_pipeline()

        if not method_pipeline:

            return None

        src_FPN = Full_path_locate(self.project_root_FP, self.parameters.tabular_data_path)

        if not src_FPN:

            self._Report_failure('❌ ERROR - Tabular data file not found:\n   %s' % self.parameters.tabular_data_path)

            return None

        dst_FP = Full_path_locate(self.project_root_FP, self.parameters.dst_path, True)

        if src_FPN.endswith('.csv'):

            result = Read_csv(src_FPN)

        elif src_FPN.endswith('.xlsx'):

            result = Read_excel(src_FPN)

        else:

            self._Report_failure('❌ ERROR - ASV data file type not supported: %s' % src_FPN)

            return None

        if not result:

            return None

        row_L, error_L = Canonical_rows(*result)

        for msg in error_L[:20]:

            print ('    ❌ %s' % msg)

        if error_L:

            self._Report_failure('❌ ERROR - %s row(s) in %s could not be read, nothing written' % (len(error_L), src_FPN))

            return None

        abundance_FPN = os.path.join(dst_FP, ABUNDANCE_FN)

        with open(abundance_FPN, 'w', newline='') as f:

            writer = csv.writer(f)

            writer.writerow(CANONICAL_COL_L)

            writer.writerows([['' if v is None else v for v in row] for row in row_L])

        if self.verbose > 0:

            print ('    Input read: %s (%s rows)' % (src_FPN, len(row_L)))

            print ('    ✅ Created abundance file: %s' % abundance_FPN)

        if self.process_S.process.process.startswith('insert'):

            return abundance_FPN

        process_FPN = os.path.join(dst_FP, '%s.json' % TARGET_PROCESS)

        Dump_json(process_FPN, {'process': [{'process': TARGET_PROCESS,
                                             'overwrite': False,
                                             'delete': False,
                                             'parameters': {'asv_data_path': os.path.join(self.parameters.dst_path, ABUNDANCE_FN),
                                                            'method_pipeline': method_pipeline}}]})

        if self.verbose > 0:

            print ('    ✅ Created JSON: %s' % process_FPN)

        return process_FPN

    def _Load(self, abundance_FPN):
        ''' Resolves observations, taxa and ASVs, then COPYs new ASVs and abundances.
        '''

        method_pipeline = self._Method_pipeline()

        if not method_pipeline:

            return None

        result = Read_csv(abundance_FPN)

        if not result:

            self._Report_failure('❌ ERROR - could not read abundance file %s' % abundance_FPN)

            return None

        row_L, error_L = Canonical_rows(*result)

        if error_L:

            self._Report_failure('❌ ERROR - %s: %s' % (abundance_FPN, '; '.join(error_L[:5])))

            return None

        conn = self.pg_session_C.conn

        cursor = conn.cursor()

        try:

            ok = self._Resolve_and_copy(cursor, row_L, method_pipeline)

            if ok:

                conn.commit()

            else:

                conn.rollback()

            return ok

        except Exception as e:

            conn.rollback()

            self._Report_failure('❌ ERROR - ASV COPY failed, nothing inserted (rolled back):\n   %s' % e)

            return None

        finally:

            cursor.close()

    def _Resolve_and_copy(self, cursor, row_L, method_pipeline):

        # method pipeline
        cursor.execute('SELECT id FROM observation_utility.method_pipeline WHERE LOWER(name) = %s;', (method_pipeline,))

        rec = cursor.fetchone()

        if not rec:

            self._Report_failure('❌ ERROR - method_pipeline <%s> not in observation_utility.method_pipeline' % method_pipeline)

            return None

        pipeline_id = rec[0]

        # observations: (observation log, sample) -> observation id
        log_L = sorted({row[0] for row in row_L})

        cursor.execute('SELECT o.id, LOWER(ol.name), LOWER(s.name) FROM observation.observation o '
                       'JOIN observation.observation_log ol ON ol.id = o.observation_log_id '
                       'JOIN observation.sample s ON s.id = o.sample_id '
                       'WHERE LOWER(ol.name) = ANY(%s);', (log_L,))

        observation_D = {}

        for obs_id, log, sample in cursor.fetchall():

            observation_D.setdefault((log, sample), []).append(obs_id)

        wanted_S = {(row[0], row[1]) for row in row_L}

        missing_L = sorted(k for k in wanted_S if k not in observation_D)

        multiple_L = sorted(k for k in wanted_S if len(observation_D.get(k, [])) > 1)

        if missing_L or multiple_L:

            for log, sample in missing_L[:20]:

                print ('    ❌ no observation for %s / %s' % (log, sample))

            for log, sample in multiple_L[:20]:

                print ('    ❌ more than one observation for %s / %s' % (log, sample))

            self._Report_failure('❌ ERROR - %s sample(s) without and %s with more than one observation, nothing inserted'
                                 % (len(missing_L), len(multiple_L)))

            return None

        # ASVs: key -> lineage, one lineage per key
        asv_D = {}

        for row in row_L:

            if asv_D.setdefault(row[2], row[3]) != row[3]:

                self._Report_failure('❌ ERROR - asv %s has two lineages in the file: %s / %s' % (row[2], asv_D[row[2]], row[3]))

                return None

        # taxa: lineage -> deepest resolved taxon id
        cursor.execute('SELECT id, name, prefix FROM organism_utility.taxon_rank ORDER BY id;')

        rank_L = cursor.fetchall()

        species_order, genus_order = Rank_orders(rank_L)

        cursor.execute('SELECT id, parent_taxon_id, name FROM organism_utility.taxon;')

        taxon_D = {(parent, name): taxon_id for taxon_id, parent, name in cursor.fetchall()}

        lineage_taxon_D = {}

        unresolved_L = []

        for lineage in set(asv_D.values()):

            parsed_L, parse_error_L = Parse_lineages(['taxa'], [[lineage]], rank_L, 'none', 'none')

            if parse_error_L or not parsed_L:

                unresolved_L.append(lineage)

                continue

            taxon_id = None

            for _order, name in Lineage_steps(parsed_L[0][2:], species_order, genus_order):

                taxon_id = taxon_D.get((taxon_id, name))

                if taxon_id is None:

                    break

            if taxon_id is None:

                unresolved_L.append(lineage)

            else:

                lineage_taxon_D[lineage] = taxon_id

        if unresolved_L:

            for lineage in sorted(unresolved_L)[:20]:

                print ('    ❌ taxon not in organism_utility.taxon: %s' % lineage)

            self._Report_failure('❌ ERROR - %s lineage(s) not in organism_utility.taxon (run the taxon import, Stage 4a, first); nothing inserted'
                                 % len(unresolved_L))

            return None

        # edna_asv: keep existing, refuse a key that now carries another lineage
        cursor.execute('SELECT asv_key, id, lineage FROM observation.edna_asv WHERE method_pipeline_id = %s;', (pipeline_id,))

        existing_asv_D = {key: (asv_id, lineage) for key, asv_id, lineage in cursor.fetchall()}

        changed_L = [key for key, lineage in asv_D.items() if key in existing_asv_D and existing_asv_D[key][1] != lineage]

        if changed_L:

            for key in changed_L[:20]:

                print ('    ❌ asv %s: database %s, file %s' % (key, existing_asv_D[key][1], asv_D[key]))

            self._Report_failure('❌ ERROR - %s asv key(s) carry another lineage than in the database - the delivery row order has probably changed; nothing inserted'
                                 % len(changed_L))

            return None

        new_asv_L = [(pipeline_id, key, lineage_taxon_D[lineage], lineage) for key, lineage in sorted(asv_D.items()) if key not in existing_asv_D]

        self._Copy(cursor, 'observation.edna_asv', ('method_pipeline_id', 'asv_key', 'taxon_id', 'lineage'), new_asv_L)

        cursor.execute('SELECT asv_key, id FROM observation.edna_asv WHERE method_pipeline_id = %s;', (pipeline_id,))

        asv_id_D = dict(cursor.fetchall())

        # edna_asv_abundance: keep existing (observation, asv) pairs
        observation_id_L = sorted({observation_D[(row[0], row[1])][0] for row in row_L})

        cursor.execute('SELECT observation_id, edna_asv_id FROM observation.edna_asv_abundance WHERE observation_id = ANY(%s);', (observation_id_L,))

        existing_S = set(cursor.fetchall())

        new_abundance_L = []

        for log, sample, key, _lineage, rel_abundance, read_count, raw_read_count in row_L:

            pair = (observation_D[(log, sample)][0], asv_id_D[key])

            if pair not in existing_S:

                new_abundance_L.append(pair + (rel_abundance, read_count, raw_read_count))

        self._Copy(cursor, 'observation.edna_asv_abundance',
                   ('observation_id', 'edna_asv_id', 'rel_abundance', 'read_count', 'raw_read_count'), new_abundance_L)

        if self.verbose > 0:

            print ('    ✅ %s: %s ASVs (%s new), %s abundances (%s new) for %s observations'
                   % (method_pipeline, len(asv_D), len(new_asv_L), len(row_L), len(new_abundance_L), len(observation_id_L)))

        return True

    def _Copy(self, cursor, table, column_L, row_L):

        if not row_L:

            return

        buffer = io.StringIO()

        csv.writer(buffer, lineterminator='\n').writerows([['' if v is None else v for v in row] for row in row_L])

        buffer.seek(0)

        cursor.copy_expert('COPY %s (%s) FROM STDIN WITH (FORMAT csv)' % (table, ', '.join(column_L)), buffer)
