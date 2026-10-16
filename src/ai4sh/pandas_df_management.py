'''
Created on 9 September 2026

@author: thomasgumbricht
'''

# Standard library imports
import os
import glob

# Third-party imports
import pandas as pd

# Application package imports
from src.postgres import Get_schema_table

from src.lib.pilot import Get_project_path

from src.ai4sh.parquet_units import Read_parquet_with_units


class Process_pandas(Get_schema_table):
    '''Inspect / display a saved pandas Parquet dataset.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):

        self.verbose = process_S.process.verbose

        self.process_S = process_S

        self.pg_session_C = pg_session_C

        self.project_root_FP = project_root_FP

    def _Sub_process(self, _json_file_key):

        if self.process_S.process.process == 'inspect_pandas_dataset':

            self._Inspect_pandas_dataset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _Parse_column_array(self, raw):
        '''Return list of column names from the column_array parameter.

        structure.py converts a csv string to "{a,b,c}" (postgres array format) or
        leaves a JSON list unchanged. Both forms are handled here.
        '''
        if not raw:
            return []

        if isinstance(raw, list):
            return [x.strip() for x in raw if x.strip()]

        s = str(raw).strip()

        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]

        return [x.strip() for x in s.split(',') if x.strip()]

    def _Load_parquet_data(self, project_root_fp, parquet_file):
        '''Load the given parquet_file from project_root_fp. Returns (df, units_D).'''
        fpn = os.path.join(project_root_fp, parquet_file)

        if not os.path.exists(fpn):
            available = glob.glob(os.path.join(project_root_fp, '*.parquet'))
            available_names = [os.path.basename(fp) for fp in available]
            raise FileNotFoundError(
                'parquet_file not found: %s\n'
                '    Available .parquet files in %s:\n      %s'
                % (fpn, project_root_fp,
                   '\n      '.join(available_names) if available_names else '(none)'))

        df, units_D = Read_parquet_with_units(fpn)

        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(fpn)))

        return df, units_D

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def _Inspect_pandas_dataset(self):
        '''Print the structure and/or contents of a saved Parquet dataset.

        The first of the following boolean parameters that is true selects the display mode
        (in this order — all remaining ones are skipped): columns_units_data, columns_data,
        columns_only, columns_units_only.
        '''
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)

        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        parquet_file = str(getattr(p, 'parquet_file', '')).strip()

        if not parquet_file:
            print('    ERROR: parquet_file parameter is required.')
            return

        try:
            df, units_D = self._Load_parquet_data(project_root_fp, parquet_file)
        except FileNotFoundError as e:
            print('    ERROR: %s' % e)
            return

        requested = self._Parse_column_array(getattr(p, 'column_array', []))

        if requested:
            missing = [c for c in requested if c not in df.columns]
            if missing:
                print('    WARNING: column(s) not found in "%s", skipping: %s' % (
                    parquet_file, missing))
                print('    Available columns: %s' % list(df.columns))
            cols = [c for c in requested if c in df.columns]
        else:
            cols = list(df.columns)

        if not cols:
            print('    ERROR: no valid columns to display.')
            return

        max_rows = int(getattr(p, 'max_rows', 0))

        columns_units_data = bool(getattr(p, 'columns_units_data', True))
        columns_data = bool(getattr(p, 'columns_data', False))
        columns_only = bool(getattr(p, 'columns_only', False))
        columns_units_only = bool(getattr(p, 'columns_units_only', False))

        if columns_units_data:

            disp = df[cols].copy()
            disp.columns = pd.MultiIndex.from_tuples(
                [(c, units_D.get(c, '')) for c in cols], names=['column', 'unit'])
            print(disp.head(max_rows) if max_rows > 0 else disp)

        elif columns_data:

            disp = df[cols]
            print(disp.head(max_rows) if max_rows > 0 else disp)

        elif columns_only:

            print(cols)

        elif columns_units_only:

            print([(c, units_D.get(c, '')) for c in cols])

        else:

            print('    WARNING: no display mode selected (all boolean parameters are false).')

        if bool(getattr(p, 'show_size', False)):

            display_rows = max_rows if max_rows > 0 else len(df)

            print()
            print('    display columns: %d  display rows: %d' % (len(cols), display_rows))
            print('    dataframe columns: %d  dataframe rows: %d' % (len(df.columns), len(df)))
