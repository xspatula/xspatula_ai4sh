'''
Created on 23 April 2026

@author: thomasgumbricht

Select data from the database for machine learning and save locally as Parquet.

One function (_Select_data) serves all select processes; the process sets the role of the selection:

    select_spectra_only    → spectra      spectra only
    select_indicator_only  → indicator    target features (e.g. wetlab indicators)
    select_covariate_only  → covariate    other covariates (e.g. field pH, land cover, eDNA taxa)
    select_spectra         → all          spectra + indicators (+ covariates) in one file

The kind of data behind an indicator or covariate name is looked up in the database, not set by the
process: numeric indicators (observation_measurement), nominal indicators (observation_classification)
and landscape classes joined by position (landscape.<name>_observation). The same indicator (e.g. pH)
can thus be an indicator in one selection and a covariate in another.

Separate selections are joined into one project dataset by the process compile_dataset (_Compile_dataset),
on the key column sample_id.

The samples in scope (dataset/campaign/sampling_log/sample_array) can be restricted by class filters
(texture_class_array, landuse_array, landcover_array, biogeo16_array, juxtaposition_array, composition_array,
proximity_array; see CLASS_FILTERS) and by sample_source, a saved selection whose sample ids are reused - e.g.
select the spectra for exactly the samples of a class filtered indicator selection. Every filter applies to all
roles, as all queries take their samples from PG_manage_AI4SH._Scope_sql. The process identify_indicator lists
the number of samples per indicator and spectra provision for a scope and filter, before anything is saved.

Output paths are built from tags (see _Ml_output_paths):

    {output_root_fp}/{output_sub_fp}/{output_name}.parquet (+ {output_name}.json)

Every saved Parquet file is registered as the previous dataframe of its directory (.previous_dataframe,
see src/ai4sh/previous_dataframe.py), so the next process can read it with an empty parquet_file.
'''

# Standard library imports
import os
import re
import json
import hashlib

# Third-party imports
import numpy as np
import pandas as pd

# Application package imports
from src.postgres import Get_schema_table

from src.postgres.pg_ai4sh import PG_manage_AI4SH

from src.lib.pilot import Get_project_path

from src.utils.json_read_write import Dump_json

from src.ai4sh.feature_symbols import Load_target_units

from src.ai4sh.parquet_units import Save_parquet_with_units, Read_parquet_with_units

from src.ai4sh.previous_dataframe import Register_parquet

from src.ai4sh.spectral_columns import Band_col_name, Normalize_signal_type, Set_signal_type, DEFAULT_SIGNAL_TYPE

# Process → role of the selection
SELECT_ROLE_D = {
    'select_spectra_only': 'spectra',
    'select_indicator_only': 'indicator',
    'select_covariate_only': 'covariate',
    'select_spectra': 'all',
}

# Key columns in every selected Parquet file; sample_id is the join key as sample names are only
# unique within a sampling_log (e.g. the same LUCAS point id in 2009 and 2015)
KEY_COLUMNS = ['sample_id', 'sample_name', 'campaign_tag', 'sampling_log_tag', 'latitude_dd', 'longitude_dd',
               'profile_min', 'profile_max']

# Class filters restricting the samples in scope: (process parameter, label, kind, abbreviation in output names).
# kind: 'classification' (nominal indicator, observation_classification), 'landscape' (landscape.<label>_observation,
# joined by position) or 'sample' (observation.sample_<label>). Every selected sample carries the filter entry it
# matched in the column CLASS_COLUMN_PREFIX + label (e.g. class_landuse)
CLASS_FILTERS = [
    ('texture_class_array', 'texture', 'classification', 'tx'),
    ('landuse_array', 'landuse', 'landscape', 'lu'),
    ('landcover_array', 'landcover', 'landscape', 'lc'),
    ('biogeo16_array', 'biogeo16', 'landscape', 'bg'),
    ('juxtaposition_array', 'juxtaposition', 'sample', 'jx'),
    ('composition_array', 'composition', 'sample', 'cp'),
    ('proximity_array', 'proximity', 'sample', 'px'),
]

CLASS_COLUMN_PREFIX = 'class_'

# Classification applied to texture_class_array when texture_classification is not given
DEFAULT_TEXTURE_CLASSIFICATION = 'soil texture usda'

# identify_indicator flags indicators with fewer samples than this (min_samples) as too few for machine learning
DEFAULT_MIN_SAMPLES = 30

# Max length of the indicator/covariate part of an output name before it is cut and given a hash
MAX_FEATURE_NAME_LENGTH = 60

# Codes that source datasets use for "not measured" (e.g. LUCAS 2009 complements: -999, -888)
MISSING_VALUE_CODES = {-9999.0, -999.0, -888.0}

# Physically valid value range per unit (None = open end); units not listed only get the missing-code check
_FRACTION_RANGE_D = {
    'weight percent': (0, 100), 'volume percent': (0, 100), 'percent': (0, 100),
    'weight permill': (0, 1000), 'volume permill': (0, 1000), 'permill': (0, 1000),
    'g*kg^-1': (0, 1000), 'mg*g^-1': (0, 1000), 'mg*kg^-1': (0, 1e6), 'ug*g^-1': (0, 1e6),
    'vol*vol^-1': (0, 1), 'ph': (0, 14), 'kelvin': (0, None), 'celsius': (-273.15, None),
}


def Valid_unit_range(unit):
    '''Return (min, max) of physically valid values in unit, or None if unknown.

    Contents, concentrations, conductivities, lengths and areas cannot be negative.
    '''
    if not unit:
        return None

    if unit in _FRACTION_RANGE_D:
        return _FRACTION_RANGE_D[unit]

    if re.search(r'\*(kg|g|l|ml|cm|dm|m)\^-\d$', unit) or unit in ('cm', 'dm', 'm', 'km', 'mm', 'um', 'nm',
                                                                   'cm^2', 'dm^2', 'm^2', 'km^2', 'hectare', 'acre'):
        return (0, None)

    return None


class Process_select(Get_schema_table):
    '''Select spectra, indicators and covariates from the database and save locally as Parquet.'''

    def __init__(self, process_S, pg_session_C, project_root_FP=None):

        self.verbose = process_S.process.verbose

        self.process_S = process_S

        self.pg_session_C = pg_session_C

        self.project_root_FP = project_root_FP

        self.pg_ai4sh_C = PG_manage_AI4SH(pg_session_C)

    def _Sub_process(self, json_file_key):

        process = self.process_S.process.process

        if process in SELECT_ROLE_D:

            self._Select_data(SELECT_ROLE_D[process])

        elif process == 'compile_dataset':

            self._Compile_dataset()

        elif process == 'identify_indicator':

            self._Identify_indicator()

        else:
            raise ValueError('Unknown process type: %s' % process)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _Print_available_indicators(self, dataset_name, campaign_name):
        '''Query and print indicators available for dataset/campaign. Returns the list.'''

        available = self.pg_ai4sh_C._Retrieve_available_indicators_for_dataset(
            {'dataset_name': dataset_name, 'campaign_name': campaign_name},
            self.pg_session_C
        )

        scope = '"%s"' % dataset_name
        if campaign_name:
            scope += ' / campaign "%s"' % campaign_name

        if available:
            print('    Available indicators for %s:' % scope)
            for name, alias, unit in available:
                unit_str = ' [%s]' % unit if unit else ''
                if alias:
                    print('        - %s (%s)%s' % (name, alias, unit_str))
                else:
                    print('        - %s%s' % (name, unit_str))
        else:
            print('    No indicators found in the DB for %s.' % scope)

        return available

    def _Parse_array(self, parameter):
        '''Return list of values from an array parameter (e.g. indicator_array).

        structure.py converts a csv string to "{a,b,c}" (postgres array format) or
        leaves a JSON list unchanged. Both forms are handled here.
        '''
        raw = getattr(self.process_S.process.parameters, parameter, '')

        if not raw:
            return []

        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]

        # Postgres array string: {a, b, c}
        s = str(raw).strip()

        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]

        return [x.strip().strip('"') for x in s.split(',') if x.strip()]

    def _Parse_data_range(self):
        '''Return dict of {indicator: {min, max}} from the data_range text parameter.

        Empty string → empty dict (no range filter applied).
        '''
        raw = getattr(self.process_S.process.parameters, 'data_range', '')

        if not raw:
            return {}

        try:
            return json.loads(raw)

        except json.JSONDecodeError as e:

            print('    ⚠️  Could not parse data_range JSON: %s' % e)

            return {}

    def _Covariate_rank(self, label):
        '''Return the requested rank (order, family, genus or lowest) for a hierarchical class covariate.

        covariate_rank is either one rank for all covariates ("family") or JSON per covariate
        ({"landcover": "family", "landuse": "order"}); covariates not listed keep the rank recorded (lowest).
        '''
        raw = getattr(self.process_S.process.parameters, 'covariate_rank', '') or 'lowest'

        if isinstance(raw, dict):
            rank = raw.get(label, 'lowest')

        elif str(raw).strip().startswith('{'):

            try:
                rank = json.loads(raw).get(label, 'lowest')

            except json.JSONDecodeError as e:

                print('    ⚠️  Could not parse covariate_rank JSON: %s' % e)

                rank = 'lowest'
        else:
            rank = raw

        return str(rank).strip().lower() or 'lowest'

    def _Param(self, parameter, default=''):
        '''Return a process parameter as stripped text, default if missing or empty.'''

        value = getattr(self.process_S.process.parameters, parameter, default)

        value = default if value is None else value

        return str(value).strip() if not isinstance(value, (int, float)) else value

    def _Find_one(self, schema, table, value, parent_D=None):
        '''Return (id, name, tag) of the single record in schema.table with name, tag or alias = value.

        Prints an error and returns None if there is no or more than one matching record.
        '''
        recs = self.pg_ai4sh_C._Retrieve_record_for_select(schema, table, value, parent_D, self.pg_session_C)

        if not recs:

            print('    ❌ ERROR: <%s> not found as name, tag or alias in %s.%s' % (value, schema, table))

            return None

        if len(recs) > 1:

            print('    ❌ ERROR: <%s> matches %d records in %s.%s - give the tag, or the parent (dataset/campaign)' % (
                value, len(recs), schema, table))

            return None

        return recs[0]

    def _Resolve_scope(self):
        '''Resolve dataset/campaign/sampling_log/sample_array to the scope of the selection.

        Returns scope_D with ids for the queries, the records (id, name, tag) for the params file and
        scope_tag: the tag of the most specific level given, or 'sample_select' for a sample list.
        Returns None on error.
        '''
        dataset = self._Find_one('observation', 'dataset', self._Param('dataset'))

        if not dataset:
            return None

        scope_D = {'dataset_id': dataset[0], 'dataset': dataset, 'campaign': None, 'sampling_log': None}

        campaign_value = self._Param('campaign')

        if campaign_value:

            scope_D['campaign'] = self._Find_one('observation', 'campaign', campaign_value, {'dataset_id': dataset[0]})

            if not scope_D['campaign']:
                return None

            scope_D['campaign_id'] = scope_D['campaign'][0]

        sampling_log_value = self._Param('sampling_log')

        if sampling_log_value:

            parent_D = {'campaign_id': scope_D['campaign_id']} if scope_D.get('campaign_id') else None

            scope_D['sampling_log'] = self._Find_one('observation', 'sampling_log', sampling_log_value, parent_D)

            if not scope_D['sampling_log']:
                return None

            scope_D['sampling_log_id'] = scope_D['sampling_log'][0]

            campaign_id = self.pg_session_C._Single_search({'id': scope_D['sampling_log_id']}, ['campaign_id'], 'observation', 'sampling_log')[0]

            dataset_id = self.pg_session_C._Single_search({'id': campaign_id}, ['dataset_id'], 'observation', 'campaign')[0]

            if dataset_id != dataset[0]:

                print('    ❌ ERROR: sampling_log <%s> does not belong to dataset <%s>' % (sampling_log_value, dataset[1]))

                return None

        scope_D['sample_names'] = self._Parse_array('sample_array')

        if scope_D['sample_names']:

            scope_D['scope_tag'] = 'sample_select'

        else:

            scope_D['scope_tag'] = (scope_D['sampling_log'] or scope_D['campaign'] or dataset)[2]

        min_profile, max_profile = self._Param('min_profile'), self._Param('max_profile')

        if min_profile != '' and max_profile != '':
            scope_D['profile'] = [int(min_profile), int(max_profile)]

        scope_D['source'] = self._Read_sample_source()

        if scope_D['source'] is None:
            return None

        if scope_D['source']:
            scope_D['sample_ids'] = scope_D['source']['df']['sample_id'].tolist()

        scope_D['filters'] = self._Resolve_filters()

        if scope_D['filters'] is None:
            return None

        # The class filter of the selection: inherited from the sample source, then this process's own filters
        scope_D['sample_filter'] = dict(scope_D['source'].get('params', {}).get('sample_filter', {}) if scope_D['source'] else {})

        for f in scope_D['filters']:

            scope_D['sample_filter'][f['label']] = f['values']

            if f['kind'] == 'classification':
                scope_D['sample_filter']['texture_classification'] = f['classification']

        return scope_D

    def _Read_sample_source(self):
        '''Read the saved selection given by sample_source (a Parquet file relative to the project root).

        Returns {} if no sample_source is given, None on error, else {file, df [sample_id + class columns],
        params (the selection's params JSON, {} if missing)}.
        '''
        source = self._Param('sample_source')

        if source in ('', 'none', 'default'):
            return {}

        fpn = source if os.path.isabs(source) or not self.project_root_FP else Get_project_path(self.project_root_FP, source)

        if not fpn.endswith('.parquet'):
            fpn += '.parquet'

        if not os.path.exists(fpn):

            print('    ❌ ERROR: sample_source not found: %s' % fpn)

            return None

        df, _ = Read_parquet_with_units(fpn)

        if 'sample_id' not in df.columns:

            print('    ❌ ERROR: sample_source %s has no sample_id column - re-run the select process that made it' % fpn)

            return None

        params_fpn = '%s.json' % fpn[:-len('.parquet')]

        params_D = {}

        if os.path.exists(params_fpn):

            with open(params_fpn) as fh:
                params_D = json.load(fh)

        class_cols = [c for c in df.columns if c.startswith(CLASS_COLUMN_PREFIX)]

        if self.verbose:
            print('    Sample source: %d samples from %s' % (df['sample_id'].nunique(), fpn))

        return {'file': source, 'df': df[['sample_id'] + class_cols].drop_duplicates('sample_id'), 'params': params_D}

    def _Resolve_filters(self):
        '''Resolve the class filter parameters (CLASS_FILTERS) to class ids.

        A filter entry is the name, tag, alias or display name of a class. For a hierarchical landscape
        classification (land use, land cover) an entry can be at any rank and includes all classes below it:
        "cropland" selects every cropland family and genus. Entries of one filter must not overlap.

        Returns a list of filter dicts {label, name, kind, abbrev, values, indicator_id, class_value_D, class_ids}
        where class_value_D maps every class id passing the filter to the entry it falls under; None on error.
        '''
        filters = []

        for parameter, label, kind, abbrev in CLASS_FILTERS:

            values = self._Parse_array(parameter)

            if not values:
                continue

            filter_D = {'label': label, 'name': label, 'kind': kind, 'abbrev': abbrev, 'values': values,
                        'indicator_id': None}

            if kind == 'classification':

                classification = self._Param('texture_classification') or DEFAULT_TEXTURE_CLASSIFICATION

                rec = self.pg_ai4sh_C._Retrieve_indicator_for_select(classification, self.pg_session_C)

                if not rec or rec[3] != 'class':

                    print('    ❌ ERROR: texture_classification <%s> is not a nominal (class) indicator' % classification)

                    return None

                filter_D.update({'indicator_id': rec[0], 'classification': rec[1]})

                schema, table = 'observation_utility', 'nominal_classification'

            elif kind == 'sample':

                schema, table = 'observation_utility', label

            else:

                filter_D['landscape'] = self.pg_ai4sh_C._Retrieve_landscape_table(label, self.pg_session_C)

                if not filter_D['landscape']:

                    print('    ❌ ERROR: no landscape table for the %s filter' % label)

                    return None

                schema, table = 'landscape_utility', filter_D['landscape']['lookup_table']

            lookup = self.pg_ai4sh_C._Retrieve_class_lookup(schema, table, filter_D['indicator_id'], self.pg_session_C)

            value_ids_D = {}

            for value in values:

                matches = [id_ for id_, names in lookup if value.lower() in (n.lower() for n in names)]

                if len(matches) != 1:

                    print('    ❌ ERROR: %s <%s> %s in %s.%s; valid classes: %s' % (
                        parameter, value, 'not found' if not matches else 'is ambiguous', schema, table,
                        ', '.join(names[0] for _, names in lookup)))

                    return None

                value_ids_D[value] = matches[0]

            if kind == 'landscape' and filter_D['landscape']['hierarchical']:

                class_value_D = {}

                for class_id, root_id in self.pg_ai4sh_C._Retrieve_class_descendants(
                        table, list(value_ids_D.values()), self.pg_session_C):

                    value = next(v for v, id_ in value_ids_D.items() if id_ == root_id)

                    if class_value_D.get(class_id, value) != value:

                        print('    ❌ ERROR: %s entries overlap (one lies within the other): %s and %s' % (
                            parameter, class_value_D[class_id], value))

                        return None

                    class_value_D[class_id] = value
            else:

                class_value_D = {id_: value for value, id_ in value_ids_D.items()}

            filter_D.update({'class_value_D': class_value_D, 'class_ids': list(class_value_D)})

            filters.append(filter_D)

        return filters

    def _Class_columns_df(self, scope_D):
        '''Return a DataFrame [sample_id + one class column per filter] for the samples in scope: the filter
        entry each sample matched (own filters) or the class columns of the sample source. None if no filter.'''

        frame_L = []

        for f in scope_D['filters']:

            col = CLASS_COLUMN_PREFIX + f['label']

            recs = self.pg_ai4sh_C._Retrieve_filter_classes_for_scope(scope_D, f, self.pg_session_C)

            frame = pd.DataFrame([(sample_id, f['class_value_D'][class_id]) for sample_id, class_id in recs
                                  if class_id in f['class_value_D']], columns=['sample_id', col])

            conflicting = frame.groupby('sample_id')[col].nunique()

            if (conflicting > 1).any():

                print('    ⚠️  %s: %d sample(s) recorded under more than one filter entry, first kept'
                      % (col, int((conflicting > 1).sum())))

            frame_L.append(frame.drop_duplicates('sample_id').set_index('sample_id'))

        if scope_D['source']:

            source_df = scope_D['source']['df'].set_index('sample_id')

            own = [CLASS_COLUMN_PREFIX + f['label'] for f in scope_D['filters']]

            source_df = source_df[[c for c in source_df.columns if c not in own]]

            if len(source_df.columns):
                frame_L.append(source_df)

        if not frame_L:
            return None

        return pd.concat(frame_L, axis=1).reset_index()

    def _Resolve_features(self, labels):
        '''Resolve indicator/covariate names or aliases to what they are in the database.

        Returns list of dicts {label, kind, id, alias, value_column, lookup_table}, where kind is
        'measurement' (numeric), 'classification' (nominal indicator) or 'landscape' (class joined by
        position), plus the list of labels not found.
        '''
        features, not_found = [], []

        for label in labels:

            rec = self.pg_ai4sh_C._Retrieve_indicator_for_select(label, self.pg_session_C)

            if rec:

                kind = 'classification' if rec[3] == 'class' else 'measurement'

                features.append({'label': label, 'kind': kind, 'id': rec[0], 'alias': rec[2] or rec[1]})

                continue

            landscape = self.pg_ai4sh_C._Retrieve_landscape_table(label, self.pg_session_C)

            if landscape:

                rank = self._Covariate_rank(label) if landscape['hierarchical'] else 'lowest'

                rank_id = None

                if rank != 'lowest':

                    rank_id = self.pg_ai4sh_C._Retrieve_class_rank_id(rank, self.pg_session_C)

                    if rank_id is None:

                        print('    ⚠️  Unknown covariate_rank "%s" for %s; using the rank recorded' % (rank, label))

                        rank = 'lowest'

                # The rank is part of the output name (landcover.family), not of the column name
                features.append({'label': label, 'kind': 'landscape', 'id': None,
                                 'alias': label if rank == 'lowest' else '%s.%s' % (label, rank),
                                 'landscape': landscape, 'rank': rank, 'rank_id': rank_id})

                continue

            not_found.append(label)

        return features, not_found

    def _Confirm_missing(self, missing, role):
        '''Ask whether to continue without the missing indicators/covariates. Returns True to continue.'''

        print('    ⚠️  The following requested %ss have no data in the DB: %s' % (role, missing))

        self._Print_available_indicators(self._Param('dataset'), self._Param('campaign'))

        print("\n⚠️    ✅  Continue without missing %ss?: y\n      ❌ To skip/quit: n\n" % role)

        confirm = input("⚠️ Continue without %s (y); stop(n): " % missing)

        if confirm.lower() != 'y':

            print('    Skipping — output not saved.')

            return False

        return True

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _Feature_name_part(self, features):
        '''Return the name part for a list of indicators/covariates: the aliases joined by '-', cut and
        given a hash of the full list if longer than MAX_FEATURE_NAME_LENGTH.'''

        aliases = [re.sub(r'[^a-z0-9.-]+', '-', f['alias'].lower()).strip('-') for f in features]

        joined = '-'.join(aliases)

        if len(joined) > MAX_FEATURE_NAME_LENGTH:

            digest = hashlib.sha1(joined.encode('utf-8')).hexdigest()[:8]

            joined = '%s-%s' % (joined[:MAX_FEATURE_NAME_LENGTH - 9].rstrip('-'), digest)

        return joined

    def _Filter_name_part(self, sample_filter):
        '''Return the name part for a class filter, e.g. lu-cropland_tx-loam-silt-loam: per filter its abbreviation
        (CLASS_FILTERS) and entries, cut and given a hash of the full part if longer than MAX_FEATURE_NAME_LENGTH.'''

        parts = []

        for _, label, _, abbrev in CLASS_FILTERS:

            if sample_filter.get(label):

                parts.append('-'.join([abbrev] + [re.sub(r'[^a-z0-9.]+', '-', str(v).lower()).strip('-')
                                                  for v in sample_filter[label]]))

        joined = '_'.join(parts)

        if len(joined) > MAX_FEATURE_NAME_LENGTH:

            digest = hashlib.sha1(joined.encode('utf-8')).hexdigest()[:8]

            joined = '%s-%s' % (joined[:MAX_FEATURE_NAME_LENGTH - 9].rstrip('-_'), digest)

        return joined

    def _Ml_output_paths(self, role, scope_D, spectra_D, indicators, covariates):
        '''Build the output paths from tags.

        Defaults (when the parameter is empty or "default"):
            output_root_fp  ./data (relative to the project root)
            output_sub_fp   {role}/{spectra provision tag}/{preparation alias}/{scope tag}   (spectra, all)
                            {role}/{scope tag}                                               (indicator, covariate)
            output_name     {begin}-{end}_{bandwidth}                                        (spectra)
                            {aliases}_{min_profile}-{max_profile}                            (indicator, covariate)
                            {begin}-{end}_{bandwidth}+{indicator aliases}[+{covariate aliases}]_{min}-{max}  (all)

        A class filtered selection gets __{filter part} appended to the default output_name (see _Filter_name_part);
        output_suffix replaces the filter part (and is appended also to a given output_name), "none" leaves it out.

        Returns (out_dir, params_fpn, data_fpn, output_name).
        '''
        p = self.process_S.process.parameters

        root_fp = self._Param('output_root_fp', 'default')

        if root_fp in ('', 'default'):
            root_fp = './data'

        if self.project_root_FP and not os.path.isabs(root_fp):
            root_fp = Get_project_path(self.project_root_FP, root_fp)

        sub_fp = self._Param('output_sub_fp', 'default')

        if sub_fp in ('', 'default'):

            sub_L = [role]

            if spectra_D:
                sub_L.extend([spectra_D['provision'][2], spectra_D['preparation'][2] if spectra_D['preparation'] else 'all-prep'])

            sub_L.append(scope_D['scope_tag'])

            sub_fp = os.path.join(*sub_L)

        output_name = self._Param('output_name', 'default')

        if output_name in ('', 'default'):

            parts = []

            if spectra_D:
                parts.append('%s-%s_%s' % (p.begin_wavelength, p.end_wavelength, p.output_bandwidth))

            parts.extend(self._Feature_name_part(f) for f in (indicators, covariates) if f)

            output_name = '+'.join(parts)

            if indicators or covariates:
                output_name += '_%s-%s' % (p.min_profile, p.max_profile)

            suffix = self._Param('output_suffix', 'default')

            if suffix in ('', 'default'):
                suffix = self._Filter_name_part(scope_D['sample_filter'])

        else:

            suffix = self._Param('output_suffix', 'default')

            if suffix in ('', 'default'):
                suffix = ''

        if suffix and suffix != 'none':
            output_name += '__%s' % suffix

        out_dir = os.path.join(root_fp, sub_fp)

        return (out_dir, os.path.join(out_dir, '%s.json' % output_name),
                os.path.join(out_dir, '%s.parquet' % output_name), output_name)

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _Resolve_spectra(self):
        '''Resolve the spectra provision and preparation. Returns spectra_D or None on error.'''

        provision = self._Find_one('observation_utility', 'provision', self._Param('spectra_provision'))

        if not provision:
            return None

        preparation = None

        preparation_value = self._Param('spectra_preparation_name')

        if preparation_value:

            preparation = self._Find_one('observation_utility', 'preparation', preparation_value)

            if not preparation:
                return None

        return {'provision': provision, 'preparation': preparation}

    def _Spectra_df(self, scope_D, spectra_D, output_signal_type):
        '''Retrieve, clip and resample the spectra of the samples in scope.

        Returns (DataFrame [sample_id + band columns], band column names, output wavelengths), or None.
        Replicate spectra of a sample are averaged.
        '''
        p = self.process_S.process.parameters

        provision = spectra_D['provision']

        input_wl = self.pg_ai4sh_C._Retrieve_spectrometer_wavelengths({'provision_name': provision[1]}, self.pg_session_C)

        if input_wl is None:

            print('    ⚠️  No spectrometer found for provision: %s' % provision[1])

            return None

        input_wl = np.array(input_wl, dtype=float)

        if p.begin_wavelength+1 < input_wl[0] or p.end_wavelength-1 > input_wl[-1]:

            print('    ⚠️  Wavelength range [%d–%d nm] exceeds spectrometer range [%.0f–%.0f nm]; skipping.' % (
                p.begin_wavelength, p.end_wavelength, input_wl[0], input_wl[-1]))

            return None

        # Build output wavelength grid based on the native array (no assumed input_bandwidth)
        output_wl = np.arange(p.begin_wavelength, p.end_wavelength + 1, p.output_bandwidth, dtype=int)

        if self.verbose:

            print('    Spectrometer wavelengths: %d bands (%.0f–%.0f nm)' % (len(input_wl), input_wl[0], input_wl[-1]))

            print('    Output grid: %d bands (%d–%d nm, step %d nm)' % (
                len(output_wl), output_wl[0], output_wl[-1], p.output_bandwidth))

        preparation_id = spectra_D['preparation'][0] if spectra_D['preparation'] else None

        spectra_recs = self.pg_ai4sh_C._Retrieve_spectra_for_scope(scope_D, provision[0], preparation_id, self.pg_session_C)

        if not spectra_recs:

            print('    ⚠️  No spectral data found for %s / provision %s' % (scope_D['scope_tag'], provision[1]))

            return None

        if self.verbose:

            print('    Spectral records from DB: %d' % len(spectra_recs))

        as_absorbance = DEFAULT_SIGNAL_TYPE == 'refl' and output_signal_type == 'absorb'

        # Clip native wavelengths to requested range
        mask = (input_wl >= p.begin_wavelength) & (input_wl <= p.end_wavelength)

        clipped_wl = input_wl[mask]

        if len(clipped_wl) < 2:

            print('    ⚠️  Less than 2 native bands in the requested range; skipping.')

            return None

        wl_col_names = [Band_col_name(wl) for wl in output_wl]

        rows = []

        for sample_id, signal_array in spectra_recs:

            # Resample to output grid using native spacing (no assumed input_bandwidth)
            resampled = np.interp(output_wl, clipped_wl, np.array(signal_array, dtype=float)[mask])

            if as_absorbance:
                # log10(1/R) — standard apparent absorbance used in soil/NIR spectroscopy
                resampled = -np.log10(np.maximum(resampled, 1e-6))

            rows.append([sample_id] + list(resampled))

        spectra_df = pd.DataFrame(rows, columns=['sample_id'] + wl_col_names)

        n_replicates = len(spectra_df) - spectra_df['sample_id'].nunique()

        if n_replicates:

            print('    Averaging %d replicate spectra (same sample)' % n_replicates)

            spectra_df = spectra_df.groupby('sample_id', as_index=False).mean()

        spectra_df[wl_col_names] = spectra_df[wl_col_names].round(6)

        return spectra_df, wl_col_names, output_wl

    def _Translate_units(self, lab_df):
        '''Translate recorded units to the target unit from targetfeaturesymbols.json, so indicator values
        are always saved in one consistent unit per indicator — required for merging Parquet subsets from
        different provisions/datasets later on.

        lab_df columns: sample_id, label, value, unit_name. Returns (lab_df without unit_name, units_D,
        skipped labels) or None if the user stops.
        '''
        target_units_D = Load_target_units(self._Param('targetfeaturesymbols', 'default') or 'default', self.verbose)

        unit_skip_labels = []

        units_D = {}

        for label in list(lab_df['label'].unique()):

            ind_mask = lab_df['label'] == label

            src_units = [u for u in lab_df.loc[ind_mask, 'unit_name'].unique() if u]

            dst_unit = target_units_D.get(label)

            # No target unit or target 'native': keep whatever unit the DB recorded
            if not dst_unit or dst_unit == 'native':

                if len(src_units) > 1:
                    print('    ⚠️  Indicator %s has mixed native units in the DB: %s — '
                          'using "%s" for the saved unit label' % (label, src_units, src_units[0]))

                if src_units:
                    units_D[label] = src_units[0]

                continue

            units_D[label] = dst_unit

            for src_unit in src_units:

                if src_unit == dst_unit:
                    continue

                row_mask = ind_mask & (lab_df['unit_name'] == src_unit)

                try:

                    lab_df.loc[row_mask, 'value'] = self.pg_ai4sh_C._Translate_unit_value(
                        lab_df.loc[row_mask, 'value'], src_unit, dst_unit, self.pg_session_C)

                except Exception as e:

                    print('    ⚠️  %s' % e)

                    confirm = input(
                        "⚠️ Continue without indicator '%s' (unit translation missing)? (y); stop(n): " % label)

                    if confirm.lower() != 'y':

                        print('    Skipping — output not saved.')

                        return None

                    unit_skip_labels.append(label)

                    units_D.pop(label, None)

                    break

        lab_df = lab_df[~lab_df['label'].isin(unit_skip_labels)]

        return lab_df.drop(columns=['unit_name']), units_D, unit_skip_labels

    def _Screen_values(self, lab_df, units_D):
        '''Drop numeric values that are missing-value codes or outside the valid range of their unit,
        and report them per label, so that e.g. -999 or a negative clay content never reach the models.

        lab_df columns: sample_id, label, value (already in the unit units_D[label]).
        '''
        values = pd.to_numeric(lab_df['value'], errors='coerce')

        bad = values.isna() | values.isin(MISSING_VALUE_CODES)

        for label, unit in units_D.items():

            valid_range = Valid_unit_range(unit)

            if valid_range is None:
                continue

            vmin, vmax = valid_range

            mask = lab_df['label'] == label

            if vmin is not None:
                bad |= mask & (values < vmin)

            if vmax is not None:
                bad |= mask & (values > vmax)

        if bad.any():

            for label, group in lab_df[bad].groupby('label'):

                print('    ⚠️  %s: %s value(s) are missing codes or outside the valid range for %s, set to missing: %s'
                      % (label, len(group), units_D.get(label, 'unknown unit'),
                         sorted(group['value'].unique().tolist())[:5]))

        return lab_df[~bad]

    def _Feature_df(self, scope_D, features, role):
        '''Retrieve indicators/covariates of the samples in scope as one wide DataFrame.

        Returns (DataFrame [sample_id + one column per label], units_D, labels with data), or None.
        Nominal values (classifications, landscape classes) are saved as text with unit "class".
        '''
        frames, units_D = [], {}

        numeric = [f for f in features if f['kind'] == 'measurement']

        if numeric:

            label_D = {f['id']: f['label'] for f in numeric}

            recs = self.pg_ai4sh_C._Retrieve_measurements_for_scope(scope_D, list(label_D), self.pg_session_C)

            if recs:

                lab_df = pd.DataFrame([(r[0], label_D[r[1]], r[2], r[3]) for r in recs],
                                      columns=['sample_id', 'label', 'value', 'unit_name'])

                translated = self._Translate_units(lab_df)

                if translated is None:
                    return None

                lab_df, numeric_units_D, _ = translated

                units_D.update(numeric_units_D)

                lab_df = self._Screen_values(lab_df, numeric_units_D)

                frames.append(lab_df.pivot_table(index='sample_id', columns='label', values='value', aggfunc='first', dropna=False))

        classes = [f for f in features if f['kind'] == 'classification']

        if classes:

            label_D = {f['id']: f['label'] for f in classes}

            recs = self.pg_ai4sh_C._Retrieve_classifications_for_scope(scope_D, list(label_D), self.pg_session_C)

            if recs:

                class_df = pd.DataFrame([(r[0], label_D[r[1]], r[2]) for r in recs], columns=['sample_id', 'label', 'value'])

                frames.append(class_df.pivot_table(index='sample_id', columns='label', values='value', aggfunc='first', dropna=False))

                units_D.update({label: 'class' for label in class_df['label'].unique()})

        for f in [f for f in features if f['kind'] == 'landscape']:

            recs = self.pg_ai4sh_C._Retrieve_landscape_for_scope(
                scope_D, f['label'], f['landscape'], f['rank_id'], self.pg_session_C)

            if recs:

                frames.append(pd.DataFrame([r[:2] for r in recs], columns=['sample_id', f['label']]).set_index('sample_id'))

                units_D[f['label']] = 'class'

                # Lookup from the class value (tag) to its name, display_name, alias and rank, for plotting
                self.class_labels_D[f['label']] = {r[1]: {'name': r[2], 'display_name': r[3], 'alias': r[4], 'rank': r[5]}
                                                   for r in recs}

                if f['rank'] != 'lowest':

                    order = ('order', 'family', 'genus')

                    coarser = sum(1 for r in recs if r[5] in order and order.index(r[5]) < order.index(f['rank']))

                    if coarser:

                        print('    ⚠️  %s: %d of %d samples recorded at a coarser rank than %s, kept as recorded'
                              % (f['label'], coarser, len(recs), f['rank']))

        if not frames:

            print('    ⚠️  No %s data found for %s: %s' % (role, scope_D['scope_tag'], [f['label'] for f in features]))

            return None

        # Samples with at least one value are kept; missing values are NaN (the machine learning steps mask
        # them per indicator), so a sparse indicator (e.g. cec) does not remove samples from the others
        feature_df = pd.concat(frames, axis=1)

        feature_df.columns.name = None

        # Keep the requested order
        labels = [f['label'] for f in features if f['label'] in feature_df.columns]

        missing = [f['label'] for f in features if f['label'] not in feature_df.columns]

        if missing and not self._Confirm_missing(missing, role):
            return None

        return feature_df[labels].reset_index(), units_D, labels

    # ------------------------------------------------------------------
    # Main select logic
    # ------------------------------------------------------------------

    def _Identify_indicator(self):
        '''Print the number of samples in scope (dataset/campaign/sampling_log, class filters, sample_source)
        per class filter entry, spectra provision and indicator, flagging counts below min_samples. No data saved.'''

        scope_D = self._Resolve_scope()

        if not scope_D:
            return

        min_samples = self._Param('min_samples', DEFAULT_MIN_SAMPLES)

        min_samples = int(min_samples) if str(min_samples).strip() not in ('', 'default') else DEFAULT_MIN_SAMPLES

        def flag(n):
            return '  ⚠️  < %d' % min_samples if n < min_samples else ''

        scope = ' / '.join('%s "%s"' % (level, scope_D[level][1]) for level in ('dataset', 'campaign', 'sampling_log')
                           if scope_D[level])

        print('\n    Querying available data for %s' % scope)

        if scope_D['source']:
            print('    Sample source: %s' % scope_D['source']['file'])

        for label, values in scope_D['sample_filter'].items():
            print('    Filter %s: %s' % (label, values))

        n_samples = self.pg_ai4sh_C._Retrieve_n_samples_for_scope(scope_D, self.pg_session_C)

        print('\n    Samples in scope: %d%s' % (n_samples, flag(n_samples)))

        if not n_samples:
            return

        # ---- Samples per class filter entry ----
        class_df = self._Class_columns_df(scope_D)

        if class_df is not None:

            for col in [c for c in class_df.columns if c != 'sample_id']:

                print('\n    Samples per %s:' % col)

                for value, n in class_df[col].value_counts().items():
                    print('        - %s: %d%s' % (value, n, flag(n)))

        # ---- Spectral sensors (provisions) ----
        provisions = self.pg_ai4sh_C._Retrieve_provision_counts_for_scope(scope_D, self.pg_session_C)

        print('\n    Spectral sensors (provisions):' if provisions else '\n    No spectra found.')

        for name, n in provisions:
            print('        - %s: %d samples%s' % (name, n, flag(n)))

        # ---- Indicators ----
        indicators = self.pg_ai4sh_C._Retrieve_indicator_counts_for_scope(
            scope_D, sorted(MISSING_VALUE_CODES), self.pg_session_C)

        print('\n    Indicators (samples with a value):' if indicators else '\n    No indicators found.')

        for name, alias, unit, n in indicators:

            alias_str = ' (%s)' % alias if alias and alias != name else ''

            unit_str = ' [%s]' % unit if unit else ''

            print('        - %s%s%s: %d%s' % (name, alias_str, unit_str, n, flag(n)))

    def _Select_data(self, role):
        '''Select spectra, indicators and/or covariates for the samples in scope, apply the depth and data
        range filters and save as Parquet + params JSON. role: spectra, indicator, covariate or all.'''

        p = self.process_S.process.parameters

        self.class_labels_D = {}

        # ---- 1. Resolve what to select ----
        scope_D = self._Resolve_scope()

        if not scope_D:
            return

        spectra_D = None

        output_signal_type = None

        if role in ('spectra', 'all'):

            spectra_D = self._Resolve_spectra()

            if not spectra_D:
                return

            # Stored spectra are reflectance (hard-coded until the sensor signal type is in the DB)
            try:

                output_signal_type = Normalize_signal_type(
                    self._Param('output_spectra_signal') or self._Param('output_unit') or DEFAULT_SIGNAL_TYPE)

            except ValueError as e:

                print('    ❌ ERROR: output_spectra_signal: %s' % e)

                return

            if output_signal_type not in (DEFAULT_SIGNAL_TYPE, 'absorb'):

                print('    ❌ ERROR: conversion from %s to %s is not supported' % (DEFAULT_SIGNAL_TYPE, output_signal_type))

                return

        indicators, covariates, not_found = [], [], []

        if role in ('indicator', 'all'):

            indicators, missing = self._Resolve_features(self._Parse_array('indicator_array'))

            not_found += missing

        if role in ('covariate', 'all'):

            covariates, missing = self._Resolve_features(self._Parse_array('covariate_array'))

            not_found += missing

        if role == 'indicator' and not indicators and not not_found:

            print('    ❌ ERROR: select_indicator_only requires indicator_array')

            return

        if role == 'covariate' and not covariates and not not_found:

            print('    ❌ ERROR: select_covariate_only requires covariate_array')

            return

        if not_found:

            print('    ❌ ERROR: not found as indicator or landscape covariate: %s' % not_found)

            if not indicators and not covariates and role != 'all':
                return

            if not self._Confirm_missing(not_found, 'indicator/covariate'):
                return

        out_dir, params_fpn, data_fpn, output_name = self._Ml_output_paths(role, scope_D, spectra_D, indicators, covariates)

        if os.path.exists(data_fpn) and not self.process_S.process.overwrite:

            print('    Subset already exists, skipping: %s' % data_fpn)

            return

        # ---- 2. Retrieve ----
        df = pd.DataFrame(self.pg_ai4sh_C._Retrieve_sample_keys_for_scope(scope_D, self.pg_session_C), columns=KEY_COLUMNS)

        if df.empty:

            print('    ⚠️  No samples found for %s' % scope_D['scope_tag'])

            return

        units_D, wl_col_names, output_wl = {}, [], []

        # Class columns (the filter entry each sample matched) follow the key columns
        class_df = self._Class_columns_df(scope_D)

        class_cols = []

        if class_df is not None:

            class_cols = [c for c in class_df.columns if c != 'sample_id']

            df = df.merge(class_df, on='sample_id', how='left')

            units_D.update({c: 'class' for c in class_cols})

        if spectra_D:

            spectra = self._Spectra_df(scope_D, spectra_D, output_signal_type)

            if spectra is None:
                return

            spectra_df, wl_col_names, output_wl = spectra

            df = df.merge(spectra_df, on='sample_id', how='inner')

            units_D = Set_signal_type(units_D, wl_col_names, output_signal_type)

        feature_labels_D = {}

        # Indicators are required (inner join); covariates added to an all-in-one selection are optional (left join)
        for kind, features, how in (('indicator', indicators, 'inner'),
                                    ('covariate', covariates, 'inner' if role == 'covariate' else 'left')):

            if not features:
                continue

            result = self._Feature_df(scope_D, features, kind)

            if result is None:
                return

            feature_df, feature_units_D, feature_labels_D[kind] = result

            duplicates = [c for c in feature_df.columns if c != 'sample_id' and c in df.columns]

            if duplicates:

                print('    ❌ ERROR: %s columns already selected: %s' % (kind, duplicates))

                return

            df = df.merge(feature_df, on='sample_id', how=how)

            units_D.update(feature_units_D)

        # ---- 3. Depth filter (only when profile data exists) ----
        before = len(df)

        df = df[~(df['profile_min'].notna() & (df['profile_min'] < p.min_profile))]

        df = df[~(df['profile_max'].notna() & (df['profile_max'] > p.max_profile))]

        if self.verbose and before - len(df):

            print('    Excluded by depth filter: %d' % (before - len(df)))

        # ---- 4. Data range filter ----
        data_range = self._Parse_data_range()

        range_excluded = 0

        for label, rng in data_range.items():

            if label not in df.columns:

                print('    ⚠️  data_range indicator not in results: %s' % label)

                continue

            before = len(df)

            if 'min' in rng:
                df = df[df[label] >= rng['min']]

            if 'max' in rng:
                df = df[df[label] <= rng['max']]

            range_excluded += before - len(df)

        if self.verbose and range_excluded:

            print('    Excluded by data range filter: %d' % range_excluded)

        if df.empty:

            print('    ⚠️  No samples remain after all filters — nothing to save')

            return

        # ---- 5. Save ----
        os.makedirs(out_dir, exist_ok=True)

        Save_parquet_with_units(df.reset_index(drop=True), data_fpn, units_D)

        dataset, campaign, sampling_log = scope_D['dataset'], scope_D['campaign'], scope_D['sampling_log']

        params_D = {
            'role': role,
            'dataset_name': dataset[1],
            'dataset_tag': dataset[2],
            'campaign_name': campaign[1] if campaign else '',
            'campaign_tag': campaign[2] if campaign else '',
            'sampling_log_name': sampling_log[1] if sampling_log else '',
            'sampling_log_tag': sampling_log[2] if sampling_log else '',
            'sample_array': scope_D['sample_names'],
            'min_profile': int(p.min_profile),
            'max_profile': int(p.max_profile),
            'indicator_array': feature_labels_D.get('indicator', []),
            'covariate_array': feature_labels_D.get('covariate', []),
            'covariate_rank': {f['label']: f['rank'] for f in covariates if f['kind'] == 'landscape'},
            'class_labels': {k: v for k, v in self.class_labels_D.items() if k in df.columns},
            'indicator_units': {k: v for k, v in units_D.items() if k not in wl_col_names and k not in class_cols},
            'data_range': data_range,
            'sample_filter': scope_D['sample_filter'],
            'sample_source': scope_D['source']['file'] if scope_D['source'] else '',
            'class_columns': class_cols,
            'output_name': output_name,
            'n_samples': len(df),
        }

        if spectra_D:

            params_D.update({
                'provision_name': spectra_D['provision'][1],
                'provision_tag': spectra_D['provision'][2],
                'preparation_name': spectra_D['preparation'][1] if spectra_D['preparation'] else '',
                'preparation_tag': spectra_D['preparation'][2] if spectra_D['preparation'] else '',
                'begin_wavelength': int(p.begin_wavelength),
                'end_wavelength': int(p.end_wavelength),
                'output_bandwidth': int(p.output_bandwidth),
                'output_unit': output_signal_type,
                'output_wavelengths': [int(w) for w in output_wl],
            })

        Dump_json(params_fpn, params_D, verbose=self.verbose)

        # The selection becomes the previous dataframe of its directory (read when parquet_file is empty)
        note = '%s: %d samples' % (role, len(df))

        if scope_D['sample_filter']:
            note += ', filter %s' % scope_D['sample_filter']

        Register_parquet(data_fpn, self.process_S.process.process, note)

        print('    Saved %d samples → %s' % (len(df), data_fpn))

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def _Compile_dataset(self):
        '''Join saved selections (e.g. spectra + indicators + covariates) into one project dataset.

        The inputs are inner joined on sample_id, so only samples present in all inputs are kept. Key
        columns are kept once (from the first input); a feature column found in two inputs is an error.
        Saves {output_name}.parquet + {output_name}.json under
        {output_root_fp}/{output_sub_fp}/{output_name}/, a folder ready to use as project_root_fp in the
        machine learning processes.
        '''
        inputs = self._Parse_array('input_array')

        if len(inputs) < 2:

            print('    ❌ ERROR: compile_dataset requires at least 2 files in input_array')

            return

        output_name = self._Param('output_name')

        if output_name in ('', 'default'):

            print('    ❌ ERROR: compile_dataset requires an output_name (the name of the compiled dataset)')

            return

        def resolve(path):

            return Get_project_path(self.project_root_FP, path) if self.project_root_FP and not os.path.isabs(path) else path

        root_fp = self._Param('output_root_fp', 'default')

        root_fp = resolve('./data' if root_fp in ('', 'default') else root_fp)

        sub_fp = self._Param('output_sub_fp', 'default')

        out_dir = os.path.join(root_fp, 'compiled' if sub_fp in ('', 'default') else sub_fp, output_name)

        data_fpn = os.path.join(out_dir, '%s.parquet' % output_name)

        params_fpn = os.path.join(out_dir, '%s.json' % output_name)

        if os.path.exists(data_fpn) and not self.process_S.process.overwrite:

            print('    Compiled dataset already exists, skipping: %s' % data_fpn)

            return

        df, units_D, params_D, inputs_L = None, {}, {}, []

        for path in inputs:

            fpn = resolve(path)

            if not fpn.endswith('.parquet'):
                fpn += '.parquet'

            if not os.path.exists(fpn):

                print('    ❌ ERROR: input not found: %s' % fpn)

                return

            in_df, in_units_D = Read_parquet_with_units(fpn)

            if 'sample_id' not in in_df.columns:

                print('    ❌ ERROR: %s has no sample_id column - re-run the select process that made it' % fpn)

                return

            stem = os.path.basename(fpn)[:-len('.parquet')]

            in_params_fpn = os.path.join(os.path.dirname(fpn), '%s.json' % (stem))

            in_params_D = {}

            if os.path.exists(in_params_fpn):

                with open(in_params_fpn) as fh:
                    in_params_D = json.load(fh)

            inputs_L.append({'file': path, 'role': in_params_D.get('role', ''), 'n_samples': len(in_df)})

            if df is None:

                df = in_df

            else:

                # Key columns and class columns already taken from an earlier input describe the same sample
                in_df = in_df.drop(columns=[c for c in in_df.columns if c != 'sample_id' and (
                    c in KEY_COLUMNS or (c.startswith(CLASS_COLUMN_PREFIX) and c in df.columns))])

                duplicates = [c for c in in_df.columns if c != 'sample_id' and c in df.columns]

                if duplicates:

                    print('    ❌ ERROR: columns in %s already in an earlier input: %s' % (path, duplicates))

                    return

                df = df.merge(in_df, on='sample_id', how='inner')

            units_D.update(in_units_D)

            # Spectra parameters (wavelengths, provision) are read downstream, so they are taken over as they are;
            # indicator/covariate lists and units are merged
            for key, value in in_params_D.items():

                if key in ('indicator_array', 'covariate_array'):

                    params_D[key] = params_D.get(key, []) + [v for v in value if v not in params_D.get(key, [])]

                elif key == 'class_columns':

                    params_D[key] = params_D.get(key, []) + [v for v in value if v not in params_D.get(key, [])]

                elif key in ('indicator_units', 'covariate_rank', 'class_labels'):

                    params_D.setdefault(key, {}).update(value)

                elif key not in params_D or key in ('provision_name', 'provision_tag', 'preparation_name',
                                                     'preparation_tag', 'begin_wavelength', 'end_wavelength',
                                                     'output_bandwidth', 'output_unit', 'output_wavelengths'):

                    params_D[key] = value

        if df.empty:

            print('    ⚠️  No samples in common between the inputs — nothing to save')

            return

        params_D.update({'role': 'compiled', 'output_name': output_name, 'inputs': inputs_L, 'n_samples': len(df)})

        os.makedirs(out_dir, exist_ok=True)

        Save_parquet_with_units(df.reset_index(drop=True), data_fpn, units_D)

        Dump_json(params_fpn, params_D, verbose=self.verbose)

        Register_parquet(data_fpn, 'compile_dataset', 'compiled: %d samples from %s' % (
            len(df), ', '.join(i['file'] for i in inputs_L)))

        print('    Compiled %d samples (inputs: %s) → %s' % (
            len(df), ', '.join(str(i['n_samples']) for i in inputs_L), data_fpn))
