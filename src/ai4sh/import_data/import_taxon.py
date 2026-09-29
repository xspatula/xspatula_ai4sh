'''
Created on 29 Sep 2026

@author: thomas gumbricht

Bulk import of taxonomic lineages to organism_utility.taxon using postgres COPY.

Reached through the generic tabular processes with target process "manage_taxon":

- single step: insert_tabular_data (process: manage_taxon) - translate + load
- dual step:   translate_tabular_data (process: manage_taxon) - writes a canonical
               lineage csv and a manage_taxon json process file; running that file
               (process manage_taxon) loads it.

Two input layouts are accepted (excel or csv):

- single column: one lineage string per row, ranks separated by '|' and tagged with
  the taxon_rank.prefix (e.g. k__Bacteria|p__Bacillota|...|g__Niallia|s__)
- multi column: one column per rank, headed by the taxon_rank.name (kingdom, phylum,
  class, order, family, genus, species, subspecies, variety)

Either layout may carry the optional columns taxonomy_reference and
taxonomy_reference_version, overriding the process parameters of the same name per row.

Loading is done rank by rank (kingdom first): parents are resolved from the ranks
already loaded, taxa already in the database are skipped, and only new taxa are
COPYed. The whole load is one transaction - it either completes or leaves the table
untouched - and rerunning the same file inserts nothing.
'''

# Standard library imports
import csv

import io

import os

# Application package imports
from src.lib.pilot import Full_path_locate

from src.utils.json_read_write import Dump_json

from src.utils.csv_read_write import Read_csv, Read_excel

TARGET_PROCESS = 'manage_taxon'

LINEAGE_SEPARATOR = '|'

REFERENCE_COL = 'taxonomy_reference'

REFERENCE_VERSION_COL = 'taxonomy_reference_version'

DEFAULT_REFERENCE_VERSION = 'unspecified'

LINEAGE_FN = 'taxon_lineage.csv'

ACCEPTED_STATUS = 'accepted'

SPECIES_RANK = 'species'

GENUS_RANK = 'genus'


def Normalise_taxon_name(value):
    ''' Lowercase, underscores to spaces and collapsed whitespace; None for blanks.
    '''

    if value is None:

        return None

    value = str(value).strip()

    if value in ('', 'null', 'nan', 'None'):

        return None

    return ' '.join(value.replace('_', ' ').split()).lower()


def Parse_lineages(column_L, data_L_L, rank_L, default_reference, default_version):
    '''
    @brief Converts single or multi column taxon rows to canonical lineage rows.

    @param column_L Header of the tabular file.
    @param data_L_L Data rows of the tabular file.
    @param rank_L List of (rank_id, rank_name, prefix) ordered from kingdom downwards.
    @param default_reference Taxonomy reference name used where the file has none.
    @param default_version Taxonomy reference version used where the file has none.

    @return (lineage_L, error_L): lineage_L holds unique tuples
            (reference, version, name_rank_1, ..., name_rank_n) in rank_L order,
            error_L holds messages for rows that could not be parsed.
    '''

    header_L = [str(c).strip().lower() for c in column_L]

    rank_name_L = [r[1] for r in rank_L]

    ref_i = header_L.index(REFERENCE_COL) if REFERENCE_COL in header_L else None

    ver_i = header_L.index(REFERENCE_VERSION_COL) if REFERENCE_VERSION_COL in header_L else None

    rank_col_D = {name: header_L.index(name) for name in rank_name_L if name in header_L}

    if rank_col_D:

        layout = 'multicolumn'

    else:

        layout = 'singlecolumn'

        lineage_i_L = [i for i in range(len(header_L)) if i not in (ref_i, ver_i)]

        if len(lineage_i_L) != 1:

            return [], ['❌ ERROR - single column layout requires exactly one lineage column (plus optional %s, %s), found: %s'
                        % (REFERENCE_COL, REFERENCE_VERSION_COL, column_L)]

        lineage_i = lineage_i_L[0]

        prefix_D = {r[2]: n for n, r in enumerate(rank_L) if r[2]}

    lineage_S = set()

    lineage_L = []

    error_L = []

    for row_nr, row in enumerate(data_L_L, start=2):

        names = [None] * len(rank_L)

        if layout == 'multicolumn':

            for n, rank_name in enumerate(rank_name_L):

                if rank_name in rank_col_D:

                    names[n] = Normalise_taxon_name(row[rank_col_D[rank_name]])

        else:

            lineage = row[lineage_i]

            if Normalise_taxon_name(lineage) is None:

                continue

            bad_part = None

            for part in str(lineage).split(LINEAGE_SEPARATOR):

                part = part.strip()

                prefix = next((p for p in prefix_D if part.startswith(p)), None)

                if prefix is None:

                    bad_part = part

                    break

                names[prefix_D[prefix]] = Normalise_taxon_name(part[len(prefix):])

            if bad_part is not None:

                error_L.append('row %s: unknown rank prefix in <%s>' % (row_nr, bad_part))

                continue

        if not any(names):

            continue

        reference = Normalise_taxon_name(row[ref_i]) if ref_i is not None else None

        reference = reference or default_reference

        version = Normalise_taxon_name(row[ver_i]) if ver_i is not None else None

        version = version or default_version or DEFAULT_REFERENCE_VERSION

        if not reference:

            error_L.append('row %s: no taxonomy reference (set process parameter %s or add a %s column)'
                           % (row_nr, REFERENCE_COL, REFERENCE_COL))

            continue

        lineage = tuple([reference, version] + names)

        if lineage not in lineage_S:

            lineage_S.add(lineage)

            lineage_L.append(lineage)

    return lineage_L, error_L


def Rank_orders(rank_L):
    ''' Return (species_order, genus_order): positions of species and genus in rank_L.
    '''

    rank_name_L = [r[1] for r in rank_L]

    species_order = rank_name_L.index(SPECIES_RANK) if SPECIES_RANK in rank_name_L else len(rank_L)

    genus_order = rank_name_L.index(GENUS_RANK) if GENUS_RANK in rank_name_L else None

    return species_order, genus_order


def Lineage_steps(names, species_order, genus_order):
    '''
    @brief Named ranks of one lineage as [(rank_order, name)], empty ranks skipped.

    @details A species given as a binomial under its genus ("bradyrhizobium elkanii"
    under "bradyrhizobium") is reduced to the epithet, so both notations resolve to
    the same taxon. Shared by the taxon loader and every lookup of a lineage's taxon.
    '''

    step_L = []

    for order, name in enumerate(names):

        if not name:

            continue

        if order == species_order and step_L and step_L[-1][0] == genus_order and name.startswith(step_L[-1][1] + ' '):

            name = name[len(step_L[-1][1]) + 1:]

        step_L.append((order, name))

    return step_L


def Build_taxon_nodes(lineage_L, rank_L):
    '''
    @brief Expands lineages into unique taxon nodes, one per named rank.

    @details A node is identified by the path of named ancestors down to and
    including itself, so the same name under different parents is different taxa,
    and empty ranks are skipped (the parent is the nearest named ancestor).
    Species are stored with the bare epithet as name and "genus epithet" as
    scientific_name; infraspecific ranks append their name to the parent's
    scientific_name; ranks above species use the name as scientific_name.

    @return (node_D, warning_L) where node_D maps path tuple -> dict(rank_id,
            rank_order, name, scientific_name, reference, version, parent_path).
    '''

    rank_name_L = [r[1] for r in rank_L]

    species_order, genus_order = Rank_orders(rank_L)

    node_D = {}

    warning_L = []

    for lineage in lineage_L:

        reference, version = lineage[0], lineage[1]

        path = ()

        parent = None

        for order, name in Lineage_steps(lineage[2:], species_order, genus_order):

            path = path + (name,)

            if path in node_D:

                node = node_D[path]

                if node['rank_order'] != order:

                    warning_L.append('<%s> occurs as both %s and %s under the same parent - kept as %s'
                                     % (' | '.join(path), rank_name_L[node['rank_order']],
                                        rank_name_L[order], rank_name_L[node['rank_order']]))

                if (node['reference'], node['version']) != (reference, version):

                    warning_L.append('<%s> occurs in both %s %s and %s %s - kept %s %s'
                                     % (' | '.join(path), node['reference'], node['version'],
                                        reference, version, node['reference'], node['version']))

                parent = path

                continue

            if order < species_order or parent is None:

                scientific_name = name

            else:

                scientific_name = '%s %s' % (node_D[parent]['scientific_name'], name)

                if order == species_order and node_D[parent]['rank_order'] != genus_order:

                    warning_L.append('species <%s> has no genus - scientific name set to <%s>'
                                     % (' | '.join(path), scientific_name))

            node_D[path] = {'rank_id': rank_L[order][0],
                            'rank_order': order,
                            'name': name,
                            'scientific_name': scientific_name,
                            'reference': reference,
                            'version': version,
                            'parent_path': parent}

            parent = path

    return node_D, list(dict.fromkeys(warning_L))


def Load_taxon_nodes(conn, node_D, rank_L, reference_id_D, status_id, copy_FP=None, verbose=1):
    '''
    @brief COPYs new taxon nodes to organism_utility.taxon, one rank at a time.

    @param conn Open psycopg2 connection (not autocommit); committed on success,
           rolled back on any error.
    @param node_D Nodes from Build_taxon_nodes.
    @param rank_L List of (rank_id, rank_name, prefix) ordered from kingdom downwards.
    @param reference_id_D Maps (reference name, version) -> taxonomy_reference.id.
    @param status_id taxon_status.id given to new taxa.
    @param copy_FP If given, the csv source of each COPY is also written here.

    @return List of (rank_name, n_in_file, n_existing, n_new).
    '''

    columns = ('parent_taxon_id', 'taxon_rank_id', 'taxon_status_id', 'taxonomy_reference_id', 'name', 'scientific_name')

    copy_sql = 'COPY organism_utility.taxon (%s) FROM STDIN WITH (FORMAT csv)' % ', '.join(columns)

    id_D = {}

    summary_L = []

    cursor = conn.cursor()

    try:

        for order, (rank_id, rank_name, _prefix) in enumerate(rank_L):

            rank_node_L = [(path, node) for path, node in node_D.items() if node['rank_order'] == order]

            if not rank_node_L:

                continue

            parent_id_S = {id_D[node['parent_path']] for _, node in rank_node_L if node['parent_path'] is not None}

            has_root = any(node['parent_path'] is None for _, node in rank_node_L)

            def Existing():

                cursor.execute('SELECT id, parent_taxon_id, name FROM organism_utility.taxon '
                               'WHERE parent_taxon_id = ANY(%s) OR (%s AND parent_taxon_id IS NULL);',
                               (list(parent_id_S), has_root))

                return {(rec[1], rec[2]): rec[0] for rec in cursor.fetchall()}

            existing_D = Existing()

            new_L = []

            n_existing = 0

            for path, node in rank_node_L:

                parent_id = id_D[node['parent_path']] if node['parent_path'] is not None else None

                if (parent_id, node['name']) in existing_D:

                    n_existing += 1

                    continue

                new_L.append((parent_id, rank_id, status_id,
                              reference_id_D[(node['reference'], node['version'])],
                              node['name'], node['scientific_name']))

            if new_L:

                buffer = io.StringIO()

                csv.writer(buffer, lineterminator='\n').writerows(new_L)

                if copy_FP:

                    with open(os.path.join(copy_FP, 'taxon_copy_%s_%s.csv' % (rank_id, rank_name)), 'w', newline='') as f:

                        f.write(','.join(columns) + '\n' + buffer.getvalue())

                buffer.seek(0)

                cursor.copy_expert(copy_sql, buffer)

                existing_D = Existing()

            for path, node in rank_node_L:

                parent_id = id_D[node['parent_path']] if node['parent_path'] is not None else None

                id_D[path] = existing_D[(parent_id, node['name'])]

            summary_L.append((rank_name, len(rank_node_L), n_existing, len(new_L)))

            if verbose > 1:

                print ('        %s: %s in file, %s existing, %s new' % (rank_name, len(rank_node_L), n_existing, len(new_L)))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cursor.close()

    return summary_L


class Process_import_taxon():
    ''' Translate and/or load taxon lineages, see module docstring.
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

            print ('❌ ERROR - Inserting taxa requires a database connection. Please define a Postgres database in scheme file.')

            return None

        if process.startswith('insert'):

            lineage_FPN = self._Translate()

            if not lineage_FPN:

                return None

            return lineage_FPN if self._Load(lineage_FPN) else None

        if process == TARGET_PROCESS:

            if self.process_S.process.delete:

                self._Report_failure('❌ ERROR - %s does not support delete; remove taxa with SQL.' % TARGET_PROCESS)

                return None

            lineage_FPN = Full_path_locate(self.project_root_FP, self.parameters.taxon_data_path)

            if not lineage_FPN:

                self._Report_failure('❌ ERROR - taxon data file not found:\n   %s' % self.parameters.taxon_data_path)

                return None

            return json_file_key if self._Load(lineage_FPN) else None

        self._Report_failure('❌ ERROR - process %s not available for taxon import' % process)

        return None

    def _Rank_L(self):

        self.pg_session_C.cursor.execute('SELECT id, name, prefix FROM organism_utility.taxon_rank ORDER BY id;')

        return [(rec[0], rec[1], rec[2]) for rec in self.pg_session_C.cursor.fetchall()]

    def _Translate(self):
        ''' Reads the tabular source and writes the canonical lineage csv and the
        manage_taxon process file to dst_path. Returns the path of the process file
        (translate) - the lineage csv path is kept in self.lineage_FPN.
        '''

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

            self._Report_failure('❌ ERROR - taxon data file type not supported: %s' % src_FPN)

            return None

        if not result:

            return None

        column_L, data_L_L = result

        if self.pg_session_C:

            rank_L = self._Rank_L()

        else:

            self._Report_failure('❌ ERROR - reading taxon ranks requires a database connection.')

            return None

        lineage_L, error_L = Parse_lineages(column_L, data_L_L, rank_L,
                                            Normalise_taxon_name(getattr(self.parameters, REFERENCE_COL, None)),
                                            Normalise_taxon_name(getattr(self.parameters, REFERENCE_VERSION_COL, None)))

        for msg in error_L[:20]:

            print ('    ❌ %s' % msg)

        if error_L:

            self._Report_failure('❌ ERROR - %s row(s) in %s could not be parsed, nothing written' % (len(error_L), src_FPN))

            return None

        lineage_FPN = os.path.join(dst_FP, LINEAGE_FN)

        with open(lineage_FPN, 'w', newline='') as f:

            writer = csv.writer(f)

            writer.writerow([REFERENCE_COL, REFERENCE_VERSION_COL] + [r[1] for r in rank_L])

            writer.writerows([['' if v is None else v for v in lineage] for lineage in lineage_L])

        if self.verbose > 0:

            print ('    Input read: %s (%s rows, %s unique lineages)' % (src_FPN, len(data_L_L), len(lineage_L)))

            print ('    ✅ Created lineage file: %s' % lineage_FPN)

        if self.process_S.process.process.startswith('insert'):

            return lineage_FPN

        process_FPN = os.path.join(dst_FP, '%s.json' % TARGET_PROCESS)

        taxon_data_path = os.path.join(self.parameters.dst_path, LINEAGE_FN)

        Dump_json(process_FPN, {'process': [{'process': TARGET_PROCESS,
                                             'overwrite': False,
                                             'delete': False,
                                             'parameters': {'taxon_data_path': taxon_data_path,
                                                            'dst_path': self.parameters.dst_path}}]})

        if self.verbose > 0:

            print ('    ✅ Created JSON: %s' % process_FPN)

        return process_FPN

    def _Load(self, lineage_FPN):
        ''' Loads a canonical lineage csv to organism_utility.taxon.
        '''

        result = Read_csv(lineage_FPN)

        if not result:

            self._Report_failure('❌ ERROR - could not read lineage file %s' % lineage_FPN)

            return None

        column_L, data_L_L = result

        rank_L = self._Rank_L()

        expected_L = [REFERENCE_COL, REFERENCE_VERSION_COL] + [r[1] for r in rank_L]

        if column_L != expected_L:

            self._Report_failure('❌ ERROR - lineage file columns %s do not match taxon ranks %s' % (column_L, expected_L))

            return None

        lineage_L = [tuple(v if v != '' else None for v in row) for row in data_L_L]

        node_D, warning_L = Build_taxon_nodes(lineage_L, rank_L)

        for msg in warning_L:

            print ('    ⚠️  %s' % msg)

        cursor = self.pg_session_C.cursor

        reference_id_D = {}

        missing_L = []

        for reference, version in {(node['reference'], node['version']) for node in node_D.values()}:

            cursor.execute('SELECT id FROM organism_utility.taxonomy_reference WHERE LOWER(name) = %s AND LOWER(version) = %s;',
                           (reference, version))

            rec = cursor.fetchone()

            if rec:

                reference_id_D[(reference, version)] = rec[0]

            else:

                missing_L.append('%s %s' % (reference, version))

        if missing_L:

            self._Report_failure('❌ ERROR - taxonomy reference(s) not in organism_utility.taxonomy_reference: %s' % ', '.join(sorted(missing_L)))

            return None

        cursor.execute('SELECT id FROM organism_utility.taxon_status WHERE name = %s;', (ACCEPTED_STATUS,))

        rec = cursor.fetchone()

        if not rec:

            self._Report_failure('❌ ERROR - taxon_status <%s> missing' % ACCEPTED_STATUS)

            return None

        copy_FP = None

        if getattr(self.parameters, 'dst_path', None):

            copy_FP = Full_path_locate(self.project_root_FP, self.parameters.dst_path, True)

        try:

            summary_L = Load_taxon_nodes(self.pg_session_C.conn, node_D, rank_L, reference_id_D, rec[0], copy_FP, self.verbose)

        except Exception as e:

            self._Report_failure('❌ ERROR - taxon COPY failed, nothing inserted (rolled back):\n   %s' % e)

            return None

        if self.verbose > 0:

            print ('    ✅ Taxa loaded from %s' % lineage_FPN)

            for rank_name, n_file, n_existing, n_new in summary_L:

                print ('        %-10s %6s in file %6s existing %6s new' % (rank_name, n_file, n_existing, n_new))

        return True
