'''
Created on 23 April 2026

@author: thomasgumbricht
'''

# Standard library imports
import os
import json

# Third-party imports
import numpy as np
import pandas as pd

# Application package imports
from src.postgres import Get_schema_table

from src.postgres.pg_ai4sh import PG_manage_AI4SH

from src.utils.json_read_write import Dump_json


class Process_select(Get_schema_table):
    '''Select a filtered spectral subset from the database and save locally as Parquet.'''

    def __init__(self, process_S, pg_session_C):

        self.verbose = process_S.process.verbose

        self.process_S = process_S

        self.pg_session_C = pg_session_C

        self.pg_ai4sh_C = PG_manage_AI4SH(pg_session_C)

    def _Sub_process(self, json_file_key):

        if self.process_S.process.process == 'select_spectra':

            self._Select_spectra()

        elif self.process_S.process.process == 'select_indicator':

            self._Select_indicator()

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
            for name, alias in available:
                if alias:
                    print('        - %s (%s)' % (name, alias))
                else:
                    print('        - %s' % name)
        else:
            print('    No indicators found in the DB for %s.' % scope)

        return available

    def _Build_output_paths(self):
        '''Build output directory and file paths from process parameters.

        Returns (out_dir, params_fpn, data_fpn).
        '''
        p = self.process_S.process.parameters

        subset_dir = '%s_%s-%s_%s' % (p.dataset_name, p.begin_wavelength,
                                       p.end_wavelength, p.output_bandwidth)

        out_dir = os.path.join(p.output_root_fp, subset_dir)

        fname_base = '%s_%s-%s_%s' % (p.provision_name, p.begin_wavelength,
                                       p.end_wavelength, p.output_bandwidth)

        params_fpn = os.path.join(out_dir, 'params-%s.json' % fname_base)

        data_fpn = os.path.join(out_dir, 'data-%s.parquet' % fname_base)

        return out_dir, params_fpn, data_fpn

    def _Parse_indicator_array(self):
        '''Return list of indicator names from the indicator_array parameter.

        structure.py converts a csv string to "{a,b,c}" (postgres array format) or
        leaves a JSON list unchanged. Both forms are handled here.
        '''
        raw = getattr(self.process_S.process.parameters, 'indicator_array', '')

        if not raw:
            return []

        if isinstance(raw, list):
            return [x.strip() for x in raw if x.strip()]

        # Postgres array string: {a, b, c}
        s = str(raw).strip()

        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]

        return [x.strip() for x in s.split(',') if x.strip()]

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

    # ------------------------------------------------------------------
    # Main select logic
    # ------------------------------------------------------------------

    def _Select_indicator(self):
        '''Print all provisions and indicators available for a dataset/campaign. No data saved.'''

        p = self.process_S.process.parameters

        dataset_name = p.dataset_name
        campaign_name = getattr(p, 'campaign_name', '').strip()

        scope = '"%s"' % dataset_name
        if campaign_name:
            scope += ' / campaign "%s"' % campaign_name

        print('\n    Querying available data for %s\n' % scope)

        # ---- Spectral sensors (provisions) ----
        provisions = self.pg_ai4sh_C._Retrieve_available_provisions_for_dataset(
            {'dataset_name': dataset_name, 'campaign_name': campaign_name},
            self.pg_session_C
        )

        if provisions:
            print('    Spectral sensors (provisions):')
            for prov in provisions:
                print('        - %s' % prov)
        else:
            print('    No spectral provisions found for %s.' % scope)

        print()

        # ---- Indicators ----
        self._Print_available_indicators(dataset_name, campaign_name)

    def _Select_spectra(self):
        '''Query DB, apply filters, resample, and save Parquet + params JSON.'''

        p = self.process_S.process.parameters

        out_dir, params_fpn, data_fpn = self._Build_output_paths()

        if os.path.exists(data_fpn) and not self.process_S.process.overwrite:

            print('    Subset already exists, skipping: %s' % data_fpn)

            return

        indicators = self._Parse_indicator_array()

        data_range = self._Parse_data_range()

        as_absorbance = bool(getattr(p, 'as_absorbance', False))

        preparation_name = getattr(p, 'preparation_name', '').strip()

        campaign_name = getattr(p, 'campaign_name', '').strip()

        # ---- 1. Spectrometer wavelength array ----
        wl_query = {
            'dataset_name': p.dataset_name,
            'provision_name': p.provision_name,
        }

        input_wl = self.pg_ai4sh_C._Retrieve_spectrometer_wavelengths(wl_query, self.pg_session_C)

        if input_wl is None:

            print('    ⚠️  No spectrometer found for provision: %s' % p.provision_name)

            return

        input_wl = np.array(input_wl, dtype=float)

        if p.begin_wavelength+1 < input_wl[0] or p.end_wavelength-1 > input_wl[-1]:

            print('    ⚠️  Wavelength range [%d–%d nm] exceeds spectrometer range [%.0f–%.0f nm]; skipping.' % (
                p.begin_wavelength, p.end_wavelength, input_wl[0], input_wl[-1]))

            return

        # Build output wavelength grid based on the native array (no assumed input_bandwidth)
        output_wl = np.arange(p.begin_wavelength, p.end_wavelength + 1, p.output_bandwidth,
                               dtype=int)

        if self.verbose:

            print('    Spectrometer wavelengths: %d bands (%.0f–%.0f nm)' % (
                len(input_wl), input_wl[0], input_wl[-1]))

            print('    Output grid: %d bands (%d–%d nm, step %d nm)' % (
                len(output_wl), output_wl[0], output_wl[-1], p.output_bandwidth))

        # ---- 2. Spectral scans ----
        spectra_query = {
            'dataset_name': p.dataset_name,
            'provision_name': p.provision_name,
            'campaign_name': campaign_name,
            'preparation_name': preparation_name,
        }

        spectra_recs = self.pg_ai4sh_C._Retrieve_spectra_for_dataset(spectra_query, self.pg_session_C)

        if not spectra_recs:

            print('    ⚠️  No spectral data found for dataset/provision: %s / %s' % (
                p.dataset_name, p.provision_name))

            return

        if self.verbose:

            print('    Spectral records from DB: %d' % len(spectra_recs))

        # ---- 3. Indicator measurements ----
        if indicators:

            lab_recs = self.pg_ai4sh_C._Retrieve_lab_measurements_for_dataset(
                {'dataset_name': p.dataset_name, 'lab_indicators': indicators},
                self.pg_session_C
            )

            if not lab_recs:

                print('    ⚠️  No lab measurements found for any of the requested indicators: %s' % indicators)
                print('    Skipping — output not saved (no indicator data to plot or model).')

                return None

        else:

            lab_recs = []

        # ---- 4. Build spectral DataFrame ----
        wl_col_names = ['w_%d' % wl for wl in output_wl]

        rows = []

        depth_excluded = 0

        for rec in spectra_recs:

            sample_name, campaign_name_rec, lat, lon, prof_min, prof_max, signal_array = rec

            # Depth filter (only when profile data exists)
            if prof_min is not None and prof_min < p.min_profile:

                depth_excluded += 1

                continue

            if prof_max is not None and prof_max > p.max_profile:

                depth_excluded += 1

                continue

            # Clip native wavelengths to requested range
            mask = (input_wl >= p.begin_wavelength) & (input_wl <= p.end_wavelength)

            clipped_wl = input_wl[mask]

            clipped_signal = np.array(signal_array, dtype=float)[mask]

            if len(clipped_wl) < 2:

                continue

            # Resample to output grid using native spacing (no assumed input_bandwidth)
            resampled = np.interp(output_wl, clipped_wl, clipped_signal)

            if as_absorbance:
                # log10(1/R) — standard apparent absorbance used in soil/NIR spectroscopy
                resampled = -np.log10(np.maximum(resampled, 1e-6))

            resampled = np.round(resampled, 6)

            row = {
                'sample_name': sample_name,
                'campaign_name': campaign_name_rec,
                'latitude_dd': lat,
                'longitude_dd': lon,
                'profile_min': prof_min,
                'profile_max': prof_max,
            }

            for col, val in zip(wl_col_names, resampled):

                row[col] = float(val)

            rows.append(row)

        if self.verbose and depth_excluded:

            print('    Excluded by depth filter: %d' % depth_excluded)

        if not rows:

            print('    ⚠️  No samples passed depth filter — nothing to save')

            return

        spectra_df = pd.DataFrame(rows)

        # ---- 5. Indicator DataFrame (pivot) ----
        if lab_recs:

            lab_df = pd.DataFrame(lab_recs, columns=['sample_name', 'indicator_name', 'value'])

            lab_pivot = lab_df.pivot_table(index='sample_name', columns='indicator_name',
                                            values='value', aggfunc='first')

            lab_pivot.columns.name = None

            missing_indicators = [ind for ind in indicators if ind not in lab_pivot.columns]

            if missing_indicators:

                print('    ⚠️  The following requested indicators have no data in the DB: %s' % missing_indicators)

                self._Print_available_indicators(p.dataset_name, campaign_name)

                print ("\n⚠️    ✅  Continue without missing indicators?: y\n      ❌ To skip/quit: n\n")

                confirm = input("⚠️ Continue without %s (y); stop(n): " % missing_indicators)

                if confirm.lower() != 'y':

                    print('    Skipping — output not saved.')

                    return None

            df = spectra_df.merge(lab_pivot.reset_index(), on='sample_name', how='inner')

        else:

            df = spectra_df

        # ---- 6. Data range filter ----
        range_excluded = 0

        for indicator, rng in data_range.items():

            if indicator not in df.columns:

                print('    ⚠️  data_range indicator not in results: %s' % indicator)

                continue

            before = len(df)

            if 'min' in rng:

                df = df[df[indicator] >= rng['min']]

            if 'max' in rng:

                df = df[df[indicator] <= rng['max']]

            range_excluded += before - len(df)

        if self.verbose and range_excluded:

            print('    Excluded by data range filter: %d' % range_excluded)

        if df.empty:

            print('    ⚠️  No samples remain after all filters — nothing to save')

            return

        # ---- 7. Save ----
        os.makedirs(out_dir, exist_ok=True)

        df.to_parquet(data_fpn, index=False)

        params_D = {
            'dataset_name': p.dataset_name,
            'campaign_name': getattr(p, 'campaign_name', ''),
            'provision_name': p.provision_name,
            'preparation_name': preparation_name,
            'begin_wavelength': int(p.begin_wavelength),
            'end_wavelength': int(p.end_wavelength),
            'output_bandwidth': int(p.output_bandwidth),
            'min_profile': int(p.min_profile),
            'max_profile': int(p.max_profile),
            'indicator_array': indicators,
            'data_range': data_range,
            'as_absorbance': as_absorbance,
            'output_wavelengths': [int(w) for w in output_wl],
            'n_samples': len(df),
        }

        Dump_json(params_fpn, params_D, verbose=self.verbose)

        print('    Saved %d samples → %s' % (len(df), data_fpn))
