'''
Created on 25 April 2026

@author: thomasgumbricht
'''

import os
import json
import numbers
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import KFold

from src.ai4sh.chemometrics import (apply_derivative, apply_scatter_correction,
                                     apply_scaling, apply_decomposition,
                                     apply_chemometrics, _load_chemometric_config)
from src.ai4sh.filter import apply_filter, apply_multi_filter
from src.ai4sh.previous_dataframe import (
    Previous_df_path, Read_previous_df, Write_previous_df, Previous_parquet,
)
from src.ai4sh.parquet_units import (Read_parquet_with_units, Save_parquet_with_units,
                                    Find_selected_parquet, DERIVED_STEM_RE,
                                    SUFFIX_SEP, Derived_stem)
from src.ai4sh.spectral_columns import (Band_col_name, Band_value, Is_band_col,
                                       Set_signal_type, Signal_type_of)

from src.postgres import Get_schema_table
from src.lib.pilot import Get_project_path
from src.ai4sh.project_paths import Default_fpn
from src.ai4sh.provenance import Load_config_json
from src.ai4sh.ml_setting import Load_settings, Is_path, Spectra_colormap
from src.ai4sh.ml_transformers import Holdout_test_mask, Holdout_info, Ward_fit

_METADATA_COLS = frozenset([
    'sample_id', 'sample_name', 'campaign_tag', 'sampling_log_tag', 'campaign_name', 'latitude_dd', 'longitude_dd',
    'profile_min', 'profile_max',
])


def _is_metadata_col(col):
    '''True for key columns and the class columns (class_<filter>) of a class filtered selection.'''
    return col in _METADATA_COLS or col.startswith('class_')


def _build_detector(name, threshold):
    n = name.lower()
    if n in ('iforest', 'isolationforest'):
        return IsolationForest(contamination=threshold)
    if n in ('ee', 'eenvelope', 'ellipticenvelope'):
        return EllipticEnvelope(contamination=threshold)
    if n in ('lof', 'lofactor', 'localoutlierfactor'):
        return LocalOutlierFactor(contamination=threshold)
    if n in ('1csvm', '1c-svm', 'oneclasssvm'):
        return OneClassSVM(nu=threshold)
    return None


def _parse_array_param(raw):
    '''Coerce list / CSV string / postgres {a,b} string to a Python list.'''
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1]
    return [x.strip() for x in s.split(',') if x.strip()]


# Options offered by scaler/method "select": per-spectrum scatter corrections and band-wise variable scalings
SCATTER_CORRECTIONS = ['l1', 'l2', 'max', 'snv', 'msc']
VARIABLE_SCALINGS   = ['meancentring', 'autoscaling', 'paretoscaling', 'poissonscaling']
# Filter parameter sets compared by filter_fp "select" when no filter_array is given; key → file abbreviation
FILTER_PRESETS = ['moving-average', 'gauss', 'savitzky-golay', 'lowess']
_FILTER_ABBREV = {'moving_average': 'ma', 'gauss': 'gf', 'savitzky-golay': 'sg', 'lowess': 'lw'}

# Regex matching any known preprocessing suffix at the end of a parquet stem.
_DERIVED_RE = DERIVED_STEM_RE

# Spectral column patterns: wl_nm_1350 (band, see spectral_columns.py), d1350 (derivative),
# pc1 (PCA), wc1 (Ward cluster)
_DERIVED_SPECTRAL_COL_RE = re.compile(r'^(?:d\d+|pc\d+|wc\d+)$')


def _is_spectral_col(col):
    return Is_band_col(col) or bool(_DERIVED_SPECTRAL_COL_RE.match(str(col)))


def _col_to_index(col):
    '''Extract the numeric index from a spectral column name.'''
    s = str(col)
    if Is_band_col(s):
        return Band_value(s)
    if s.startswith('pc') or s.startswith('wc'):
        return int(s[2:])
    if s.startswith('d'):
        return int(s[1:])
    return int(s)


# ------------------------------------------------------------------ companion JSON helpers

def _normalize_params_to_companion(d):
    '''Convert a selection JSON (as written by select.py) to the companion format.'''
    out = dict(d)
    wavelengths = out.pop('output_wavelengths', [])
    out['output_data'] = {
        'data_type': 'wavelength',
        'column_format': 'wl_nm_{n} — n is wavelength in nm',
        'spectral_array': wavelengths,
    }
    out.setdefault('preprocessing_chain', [])
    return out


def _load_companion_json(project_root_fp, stem):
    '''Load the companion {stem}.json of a parquet stem, or None if there is none.'''
    fp = os.path.join(project_root_fp, stem + '.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        d = json.load(fh)
    if 'output_data' not in d:
        d = _normalize_params_to_companion(d)
    return d


def _build_output_data(out_cols, signal_type=None):
    '''Build output_data dict from output column names.'''
    out = _build_output_format(out_cols)
    if signal_type:
        out['signal_type'] = signal_type
    return out


def _build_output_format(out_cols):
    '''Describe the column format of the output column names.'''
    if not out_cols:
        return {'data_type': 'unknown', 'column_format': '', 'spectral_array': []}
    first = str(out_cols[0])
    if first.startswith('pc'):
        return {
            'data_type': 'pca',
            'column_format': 'pc{n} — n is component number starting at 1',
            'spectral_array': list(range(1, len(out_cols) + 1)),
        }
    if first.startswith('wc'):
        return {
            'data_type': 'ward_cluster',
            'column_format': 'wc{n} — n is cluster index (1-based)',
            'spectral_array': list(range(1, len(out_cols) + 1)),
        }
    if first.startswith('d'):
        return {
            'data_type': 'derivative',
            'column_format': 'd{n} — n is midpoint wavelength in nm',
            'spectral_array': [_col_to_index(c) for c in out_cols],
        }
    return {
        'data_type': 'wavelength',
        'column_format': 'wl_nm_{n} — n is wavelength in nm',
        'spectral_array': [_col_to_index(c) if isinstance(c, str) else int(c)
                           for c in out_cols],
    }


def _step_abbrev_from_info(process_name, parameters):
    '''Short abbreviation for a step given its process name and parameter dict.'''
    if process_name == 'spectra_derivative':
        n = parameters.get('derivative', 1)
        return 'd%dapp' % n if parameters.get('append', False) else 'd%d' % n
    if process_name == 'spectra_scatter_correction':
        return ''.join(parameters.get('scaler', ['snv']))
    if process_name == 'spectra_scaling':
        return {'meancentring': 'mc', 'autoscaling': 'as',
                'paretoscaling': 'ps', 'poissonscaling': 'poi'}.get(
                    parameters.get('method', ''), 'sc')
    if process_name == 'spectra_decomposition':
        return 'pca%d' % parameters.get('n_components', 5)
    if process_name == 'select_variance_threshold':
        return 'vt'
    if process_name == 'detect_outliers':
        return 'ol'
    if process_name == 'filter_spectra':
        cfg = parameters.get('filter_cfg', {})
        if 'savitzky-golay' in cfg: return 'sg'
        if 'gauss' in cfg:          return 'gf'
        if 'moving_average' in cfg: return 'ma'
        if 'lowess' in cfg:         return 'lw'
        return 'flt'
    if process_name in ('multi_filter_spectra', 'mulit_filter_spectra'):
        return 'mf'
    if process_name == 'ward_clustering':
        return 'wc'
    if process_name == 'spectra_indicator_univariate_selection':
        return parameters.get('_uv_abbrev', 'uv')
    return process_name[:4]


def _write_companion_json(project_root_fp, in_stem, out_stem, out_cols, step_info_or_list,
                          signal_type=None):
    '''Write companion JSON for an output parquet, inheriting input companion metadata.'''
    companion = _load_companion_json(project_root_fp, in_stem) or {}
    # The unprocessed parquet the chain starts from — regression modeling refits the chain on it
    companion.setdefault('source_parquet', in_stem + '.parquet')
    companion['output_data'] = _build_output_data(out_cols, signal_type)
    chain = list(companion.get('preprocessing_chain', []))
    step_num = len(chain) + 1
    steps = step_info_or_list if isinstance(step_info_or_list, list) else [step_info_or_list]
    # Build chain entries — support '_id' override and pass-through of extra keys
    for i, step in enumerate(steps):
        abbrev = step.get('_id') or _step_abbrev_from_info(
            step['process'], step.get('parameters', {}))
        entry = {
            'id': abbrev,
            'step': step_num + i,
            'process': step['process'],
            'parameters': step.get('parameters', {}),
        }
        for k, v in step.items():
            if k not in ('_id', 'process', 'parameters'):
                entry[k] = v
        chain.append(entry)
    companion['preprocessing_chain'] = chain
    # Enforce key order: metadata → preprocessing_chain → data_range → n_samples → output_data
    _tail = ('data_range', 'n_samples', 'output_data')
    ordered = {k: v for k, v in companion.items() if k not in _tail + ('preprocessing_chain',)}
    ordered['preprocessing_chain'] = companion['preprocessing_chain']
    for k in _tail:
        if k in companion:
            ordered[k] = companion[k]
    out_fp = os.path.join(project_root_fp, out_stem + '.json')
    with open(out_fp, 'w') as fh:
        json.dump(ordered, fh, indent=2)


def _Ward_tune(X, n_clusters_list, kfolds, connectivity=False):
    '''K-fold cross-validated reconstruction MSE for each candidate n_clusters.

    Returns {n_clusters: (mean_mse, std_mse)}.
    '''
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    scores = {}
    for nc in n_clusters_list:
        fold_errors = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            ward, _ = Ward_fit(X_train, nc, connectivity)
            X_rec = ward.inverse_transform(ward.transform(X_test))
            fold_errors.append(float(np.mean((X_test - X_rec) ** 2)))
        scores[nc] = (float(np.mean(fold_errors)), float(np.std(fold_errors)))
    return scores


def _Ward_pick_best(scores):
    '''Smallest n_clusters whose mean MSE is within 5 % of the global minimum.'''
    min_mse = min(v[0] for v in scores.values())
    return min(nc for nc, (m, _) in sorted(scores.items()) if m <= min_mse * 1.05)


def _resolve_input_parquet(project_root_fp, dataframe_param):
    '''Resolve the input parquet path from the dataframe parameter.

    "raw" → the selected parquet, i.e. no known preprocessing suffix (see Find_selected_parquet).
    Anything else → treat as filename (with or without .parquet extension).
    '''
    if dataframe_param.lower() == 'raw':
        return Find_selected_parquet(project_root_fp)[0]
    name = dataframe_param if dataframe_param.endswith('.parquet') else dataframe_param + '.parquet'
    fp = os.path.join(project_root_fp, name)
    if not os.path.exists(fp):
        raise FileNotFoundError('Dataframe file not found: %s' % fp)
    return fp


def _Ask_choice(question, labels):
    '''Ask "question (0 = none, 1 = a, 2 = b ...)" until a valid number is entered; return it (0 = none).'''
    menu = ', '.join('%d = %s' % (i + 1, label) for i, label in enumerate(labels))
    while True:
        raw = input('\n    %s (0 = none, %s): ' % (question, menu)).strip()
        if raw.isdigit() and 0 <= int(raw) <= len(labels):
            return int(raw)
        print('    Enter a number 0-%d.' % len(labels))


def _resolve_dataframe(project_root_fp, dataframe_param):
    '''Input parquet path from a dataframe parameter, or None (error printed).

    "raw"      → the selected dataset (see Find_selected_parquet), never a derived file
    "previous" → the latest dataframe in .previous_dataframe
    otherwise  → a parquet file name in project_root_fp, the .parquet ending is optional
    '''
    param = str(dataframe_param).strip()
    if param.lower() == 'previous':
        return _resolve_single_previous(project_root_fp)
    try:
        return _resolve_input_parquet(project_root_fp, param)
    except FileNotFoundError as e:
        print('    ERROR: %s' % e)
        return None


def _prev_df_path(project_root_fp):
    return Previous_df_path(project_root_fp)


def _read_previous_df(project_root_fp):
    '''Latest record of the .previous_dataframe history (see src/ai4sh/previous_dataframe.py).'''
    return Read_previous_df(project_root_fp)


def _write_previous_df(project_root_fp, process_name, entries):
    '''Append a record to the .previous_dataframe history.'''
    Write_previous_df(project_root_fp, process_name, entries)


def _resolve_single_previous(project_root_fp):
    '''For simple processes: return the path of the latest dataframe in .previous_dataframe, or None.'''
    return Previous_parquet(project_root_fp)


def _resolve_previous_for_arrays(project_root_fp, indicator_cols, reg_keys, sel_keys):
    '''For array processes: read .previous_dataframe, confirm, validate, return mapping or None.

    Returns either {'_single': abs_path} when all combos share one parquet,
    or {(indicator, regressor, selector): abs_path} for per-combo parquets.
    reg_keys / sel_keys may be empty (pass [] for dimensions not applicable).
    Returns None to abort.
    '''
    prev = _read_previous_df(project_root_fp)
    if prev is None:
        print('    ERROR: no .previous_dataframe found in %s' % project_root_fp)
        return None
    entries = prev.get('entries', [])
    if not entries:
        print('    ERROR: .previous_dataframe has no entries.')
        return None

    print('    Previous: %s  (%d entry/entries)' % (prev.get('process', '?'), len(entries)))

    # Common case: single null-keyed entry — apply to all combos
    if (len(entries) == 1 and
            entries[0].get('indicator') is None and
            entries[0].get('regressor') is None and
            entries[0].get('selector') is None):
        fp = os.path.join(project_root_fp, entries[0]['dataframe'])
        if not os.path.exists(fp):
            print('    ERROR: previous dataframe not found: %s' % fp)
            return None
        print('      %s  (shared for all combos)' % entries[0]['dataframe'])
        return {'_single': fp}

    # Per-combo entries: match requested combinations
    for e in entries:
        print('      %s  [ind=%s  reg=%s  sel=%s]' % (
            e.get('dataframe', '?'), e.get('indicator'), e.get('regressor'), e.get('selector')))
    req_regs = reg_keys if reg_keys else [None]
    req_sels = sel_keys if sel_keys else [None]
    lookup = {}
    for e in entries:
        k = (e.get('indicator'), e.get('regressor'), e.get('selector'))
        fp = os.path.join(project_root_fp, e.get('dataframe', ''))
        if os.path.exists(fp):
            lookup[k] = fp
    missing = []
    found = {}
    for ind in indicator_cols:
        for reg in req_regs:
            for sel in req_sels:
                k = (ind, reg, sel)
                if k in lookup:
                    found[k] = lookup[k]
                else:
                    missing.append(k)
    if missing:
        print('    WARNING: %d requested combo(s) not in .previous_dataframe:' % len(missing))
        for m in missing[:5]:
            print('      indicator=%s  regressor=%s  selector=%s' % m)
        ans = input('    Proceed with restricted set? [y/n]: ').strip().lower()
        if ans != 'y':
            return None
    if not found:
        print('    ERROR: no matching previous dataframes found.')
        return None
    return found


def _spectra_x_axis_ml(columns):
    '''Return (x_values, x_label) appropriate for the column list.'''
    if not columns:
        return [], 'x'
    first = str(columns[0])
    if first.startswith('pc'):
        return list(range(1, len(columns) + 1)), 'Component'
    try:
        vals = [int(str(c).lstrip('d')) for c in columns]
        return vals, 'Wavelength (nm)'
    except (ValueError, TypeError):
        return list(range(len(columns))), 'Band'


def _chem_abbrev(step_name, label, append=False):
    '''Short abbreviation for a chemometric step name + label.'''
    if step_name == 'derivative':
        try:
            n = int(label.split('order ')[-1].rstrip(')').strip())
        except (ValueError, IndexError):
            n = 1
        return 'd%dapp' % n if append else 'd%d' % n
    if step_name == 'scatter_correction':
        return label.replace('+', '')
    if step_name == 'scaling':
        return {'meancentring': 'mc', 'autoscaling': 'as',
                'paretoscaling': 'ps', 'poissonscaling': 'poi'}.get(label, label[:4])
    if step_name == 'decomposition':
        if 'pca' in label:
            try:
                n = int(label.split('(')[1].split(' ')[0])
            except (IndexError, ValueError):
                n = '?'
            return 'pca%s' % n
        return label[:6]
    return ''.join(c for c in label.lower() if c.isalnum())[:6]


class Process_ml_preprocess(Get_schema_table):
    '''Machine-learning preprocessing: outlier detection per indicator.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP

    def _Overwrite(self, p):
        '''"overwrite" is set on the process block (next to "parameters"); a parameter of that name also counts.'''
        return bool(getattr(self.process_S.process, 'overwrite', False)) or bool(getattr(p, 'overwrite', False))

    def _Sub_process(self, _json_file_key):
        if self.process_S.process.process == 'detect_outliers':
            self._Detect_outliers()
        elif self.process_S.process.process == 'select_variance_threshold':
            self._Variance_threshold()
        elif self.process_S.process.process == 'chemometric_default':
            self._Chemometric_default()
        elif self.process_S.process.process == 'spectra_derivative':
            self._Spectra_derivative()
        elif self.process_S.process.process == 'spectra_scatter_correction':
            self._Spectra_scatter_correction()
        elif self.process_S.process.process == 'spectra_scaling':
            self._Spectra_scaling()
        elif self.process_S.process.process == 'spectra_decomposition':
            self._Spectra_decomposition()
        elif self.process_S.process.process == 'filter_spectra':
            self._Filter_spectra()
        elif self.process_S.process.process in ('multi_filter_spectra', 'mulit_filter_spectra'):
            self._Multi_filter_spectra()
        elif self.process_S.process.process == 'ward_clustering':
            self._Ward_clustering()
        elif self.process_S.process.process == 'spectra_indicator_univariate_selection':
            self._Univariate_selection()
        elif self.process_S.process.process == 'spectra_indicator_permutation_selection':
            self._Permutation_rfe_selection()

    # ------------------------------------------------------------------ helpers

    def _Load_parquet_data(self, project_root_fp):
        '''Load the selected {name}.parquet + {name}.json (see Find_selected_parquet).'''
        parquet_fpn, params_fpn = Find_selected_parquet(project_root_fp)

        df, _units_D = Read_parquet_with_units(parquet_fpn)

        params_D = {}
        if params_fpn:
            with open(params_fpn) as fh:
                params_D = json.load(fh)

        if self.verbose >= 1:
            n_expected = params_D.get('n_samples', len(df))
            print('    Loaded %d rows (%d expected) from %s' % (
                len(df), n_expected, os.path.basename(parquet_fpn)))

        return df, params_D, parquet_fpn

    def _Build_plot_output_path(self, project_root_fp):
        plot_dir = os.path.join(project_root_fp, 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir

    # ------------------------------------------------------------------ plot

    def _Plot_outliers(self, df, indicator_cols, masks, actual_detectors,
                       detector_name, threshold, show, save, plot_dir):
        plt.close('all')
        n = len(indicator_cols)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 2.5 * n))
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, indicator_cols):
            mask = masks[col]
            vals = df[col].values.astype(float)
            valid = ~np.isnan(vals)

            status = np.where(mask[valid], 'outlier', 'inlier')
            plot_df = pd.DataFrame({'value': vals[valid], 'status': status, '_': ''})

            sns.stripplot(
                data=plot_df, x='value', y='_', hue='status',
                palette={'inlier': 'steelblue', 'outlier': 'red'},
                hue_order=['inlier', 'outlier'],
                alpha=0.55, jitter=True, size=4, orient='h', ax=ax,
            )
            ax.set_ylabel('')
            ax.set_xlabel(col)
            n_out = int(mask.sum())
            used = actual_detectors.get(col, detector_name)
            fallback_note = ' [fallback]' if used != detector_name else ''
            ax.set_title('%s  (%d outlier%s)' % (col, n_out, 's' if n_out != 1 else ''))
            ax.text(0.5, 0.97, 'detector: %s%s   threshold: %.3f' % (used, fallback_note, threshold),
                    transform=ax.transAxes, ha='center', va='top', fontsize=7,
                    color='dimgray', style='italic')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.6)

        plt.tight_layout()

        if save and plot_dir:
            fname = 'outliers_%s.png' % '_'.join(c.replace(' ', '_') for c in indicator_cols)
            out_fp = os.path.join(plot_dir, fname)
            fig.savefig(out_fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % out_fp)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------ main

    def _Detect_outliers(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        detector_name   = str(getattr(p, 'detector', 'iforest')).strip()
        threshold       = float(getattr(p, 'threshold', 0.1))
        show            = getattr(p, 'show', True)
        save            = getattr(p, 'save', False)
        overwrite       = self._Overwrite(p)
        dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()

        # Load parquet
        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return
        df, units_D = Read_parquet_with_units(parquet_fp)
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        # Derive .ol output path
        stem      = os.path.splitext(os.path.basename(parquet_fp))[0]
        ol_del_fp = os.path.join(project_root_fp, Derived_stem(stem, 'ol') + '.parquet')

        # Determine which indicators to process
        requested = _parse_array_param(getattr(p, 'indicator_array', []))

        all_indicators = [
            c for c in df.columns
            if not _is_metadata_col(c)
            and not _is_spectral_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if requested:
            for col in requested:
                if col not in df.columns or col not in all_indicators:
                    print('    WARNING: indicator "%s" not found in parquet, skipping.' % col)
            indicator_cols = [c for c in requested if c in all_indicators]
        else:
            indicator_cols = all_indicators

        if not indicator_cols:
            print('    ERROR: no valid indicator columns to process.')
            return

        if self.verbose >= 1:
            print('    Detecting outliers for: %s' % ', '.join(indicator_cols))
            print('    Detector: %s  threshold: %.3f' % (detector_name, threshold))

        if _build_detector(detector_name, threshold) is None:
            print('    ERROR: unknown detector "%s".' % detector_name)
            return

        def _run_detection(thr):
            m = {}; ad = {}
            for col in indicator_cols:
                vals  = df[col].values.astype(float)
                valid = ~np.isnan(vals)
                if valid.sum() < 5:
                    print('    WARNING: too few valid values for "%s" (%d), skipping.' % (col, valid.sum()))
                    m[col] = np.zeros(len(df), dtype=bool)
                    ad[col] = detector_name
                    continue
                X   = vals[valid].reshape(-1, 1)
                det = _build_detector(detector_name, thr)
                used_name = detector_name
                try:
                    yhat = det.fit_predict(X)
                except Exception:
                    det = IsolationForest(contamination=thr)
                    yhat = det.fit_predict(X)
                    used_name = 'iforest'
                ad[col] = used_name
                full_mask = np.zeros(len(df), dtype=bool)
                full_mask[np.where(valid)[0][yhat == -1]] = True
                m[col] = full_mask
            return m, ad

        # Interactive threshold loop — user can adjust until satisfied
        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None
        while True:
            masks, actual_detectors = _run_detection(threshold)
            summary = '  '.join('%s: %d' % (c, int(masks[c].sum())) for c in indicator_cols)
            print('    Outliers detected (threshold=%.3f) — %s' % (threshold, summary))
            self._Plot_outliers(df, indicator_cols, masks, actual_detectors,
                                detector_name, threshold, show, save, plot_dir)
            raw = input(
                '\n    New threshold? (0=skip outlier detection, Enter=accept %.3f): ' % threshold
            ).strip()
            if raw == '':
                break
            try:
                val = float(raw)
            except ValueError:
                print('    Invalid input — please enter a number or press Enter.')
                continue
            if val == 0:
                print('    Outlier detection skipped.')
                return
            if not (0 < val < 1):
                print('    Threshold must be between 0 and 1 (exclusive). Try again.')
                continue
            threshold = val

        # Interactive removal prompt
        total = sum(int(masks[c].sum()) for c in indicator_cols)
        if total == 0:
            print('    No outliers detected.')
            return

        choice = input(
            '\n    Remove outliers? [all / none / individual]: '
        ).strip().lower()

        approved = []
        if choice == 'all':
            approved = [c for c in indicator_cols if masks[c].any()]
        elif choice == 'individual':
            for col in indicator_cols:
                n_out = int(masks[col].sum())
                if n_out == 0:
                    continue
                ans = input(
                    '    Remove %d outlier%s for "%s"? [y/n]: ' % (
                        n_out, 's' if n_out != 1 else '', col)
                ).strip().lower()
                if ans == 'y':
                    approved.append(col)

        if not approved:
            print('    No outliers removed.')
            return

        # Apply NaN replacements and save
        if os.path.exists(ol_del_fp) and not overwrite:
            print('    Output already exists. Use overwrite=True to save. Skipping save.')
            _write_previous_df(project_root_fp, 'detect_outliers', [
                {'dataframe': os.path.basename(ol_del_fp),
                 'indicator': None, 'regressor': None, 'selector': None}
            ])
            return

        df_out = df.copy()
        for col in approved:
            df_out.loc[masks[col], col] = np.nan

        Save_parquet_with_units(df_out, ol_del_fp, units_D)
        _write_previous_df(project_root_fp, 'detect_outliers', [
            {'dataframe': os.path.basename(ol_del_fp),
             'indicator': None, 'regressor': None, 'selector': None}
        ])

        out_stem      = os.path.splitext(os.path.basename(ol_del_fp))[0]
        spectral_kept = [c for c in df.columns if _is_spectral_col(c)]
        _write_companion_json(
            project_root_fp, stem, out_stem, spectral_kept,
            {'process': 'detect_outliers',
             'parameters': {'detector': detector_name, 'threshold': threshold,
                            'indicator_array': list(approved)}},
            Signal_type_of(units_D, spectral_kept),
        )

        n_removed = sum(int(masks[c].sum()) for c in approved)
        print('    Removed %d outlier value%s across %d indicator%s. Saved: %s' % (
            n_removed, 's' if n_removed != 1 else '',
            len(approved), 's' if len(approved) != 1 else '',
            ol_del_fp,
        ))

    # ------------------------------------------------------------------ variance threshold plot

    def _Plot_variance_threshold(self, df, spectral_cols, wavelengths, variances_scaled,
                                 discard_mask, actual_threshold, scaler_name,
                                 max_spectra, show, save, plot_dir):
        fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6),
                                              sharex=False)

        wl = np.array(wavelengths)
        var = np.array(variances_scaled)

        # Top panel: variance per wavelength with threshold line and discard shading.
        # actual_threshold None = no threshold set yet: reference line at the lowest band variance
        ax_top.plot(wl, var, color='steelblue', linewidth=0.9, label='variance (scaled)')
        if actual_threshold is None:
            ax_top.axhline(var.min(), color='red', linestyle='--', linewidth=1.0,
                           label='min variance = %.4f' % var.min())
            title_thr = 'min variance: %.4f' % var.min()
        else:
            ax_top.axhline(actual_threshold, color='red', linestyle='--', linewidth=1.0,
                           label='threshold = %.4f' % actual_threshold)
            title_thr = 'threshold: %.4f' % actual_threshold
        # Same y-range in every redraw (variance range + 5 % margin); discard shading from its bottom
        margin = 0.05 * (var.max() - var.min())
        y_lo, y_hi = var.min() - margin, var.max() + margin
        ax_top.fill_between(wl, y_lo, var, where=discard_mask,
                            color='red', alpha=0.25, label='discard')
        ax_top.set_ylim(y_lo, y_hi)
        ax_top.set_ylabel('Variance (%s scaled)' % scaler_name)
        ax_top.set_title('Band variance   scaler: %s   %s' % (scaler_name, title_thr))
        ax_top.legend(fontsize=8, loc='upper right')

        # Bottom panel: spectra subsample with discard regions shaded
        n_total = len(df)
        skip = max(1, int(np.ceil(n_total / max_spectra)))
        df_sub = df.iloc[::skip]
        X_sub = df_sub[spectral_cols].values.astype(float)

        for row in X_sub:
            ax_bot.plot(wl, row, color='steelblue', linewidth=0.4, alpha=0.3)

        # Shade contiguous discard regions
        in_discard = False
        span_start = None
        for i, (w, disc) in enumerate(zip(wl, discard_mask)):
            if disc and not in_discard:
                span_start = w
                in_discard = True
            elif not disc and in_discard:
                ax_bot.axvspan(span_start, wl[i - 1], color='red', alpha=0.18)
                in_discard = False
        if in_discard:
            ax_bot.axvspan(span_start, wl[-1], color='red', alpha=0.18)

        n_discard = int(discard_mask.sum())
        ax_bot.set_xlabel('Wavelength (nm)')
        ax_bot.set_ylabel('Reflectance')
        ax_bot.set_title('Spectra (n=%d shown)   %d band%s to discard (red regions)' % (
            len(df_sub), n_discard, 's' if n_discard != 1 else ''))

        plt.tight_layout()

        if save and plot_dir:
            out_fp = os.path.join(plot_dir, 'variance_threshold_%s_%s.png' % (
                scaler_name, 'min' if actual_threshold is None else '%.4f' % actual_threshold))
            fig.savefig(out_fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % out_fp)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------ variance threshold

    def _Variance_threshold(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        scaler_name = str(getattr(p, 'scaler', 'none')).strip().lower()
        if scaler_name == 'none':
            print('    Variance threshold skipped (scaler=none).')
            return

        dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()
        show      = getattr(p, 'show', True)
        save      = getattr(p, 'save', False)
        overwrite = self._Overwrite(p)
        max_spectra = int(getattr(p, 'max_spectra', 100))

        # Resolve input parquet
        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return
        df, units_D = Read_parquet_with_units(parquet_fp)
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        # Derive .vt output path; check for an existing result before computing
        stem   = os.path.splitext(os.path.basename(parquet_fp))[0]
        vt_fp  = os.path.join(project_root_fp, Derived_stem(stem, 'vt') + '.parquet')

        if os.path.exists(vt_fp) and not overwrite:
            self._Accept_and_save(df, None, [], 'vt', stem, project_root_fp, overwrite,
                                  {'process': 'select_variance_threshold'}, units_D)
            return

        # Extract spectral columns
        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns (wl_*, wn_*, d*, pc*) found.')
            return

        wavelengths = [_col_to_index(c) for c in spectral_cols]
        X = df[spectral_cols].values.astype(float)

        # Scale
        scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}
        if scaler_name not in scalers:
            print('    ERROR: unknown scaler "%s". Use minmax, standard, or robust.' % scaler_name)
            return

        X_scaled = scalers[scaler_name].fit_transform(X)
        variances_scaled = VarianceThreshold(threshold=0.0).fit(X_scaled).variances_
        var_min, var_med, var_max = (float(np.min(variances_scaled)),
                                     float(np.median(variances_scaled)),
                                     float(np.max(variances_scaled)))

        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None

        # Interactive threshold loop: first plot shows the lowest band variance, every new
        # threshold is redrawn with the bands it removes; Enter accepts, 0 or < min skips.
        # VarianceThreshold keeps a band only if its variance is strictly above the threshold.
        threshold    = None
        discard_mask = np.zeros(len(spectral_cols), dtype=bool)
        redraw       = True
        while True:
            if redraw:
                self._Plot_variance_threshold(
                    df, spectral_cols, wavelengths, variances_scaled, discard_mask,
                    threshold, scaler_name, max_spectra, show, save, plot_dir,
                )
                n_discard = int(discard_mask.sum())
                if threshold is None:
                    print('    Band variance (%s scaled): min %.4f   median %.4f   max %.4f' % (
                        scaler_name, var_min, var_med, var_max))
                else:
                    print('    Threshold %.4f: %d band%s retained, %d discarded' % (
                        threshold, len(spectral_cols) - n_discard,
                        's' if len(spectral_cols) - n_discard != 1 else '', n_discard))
                redraw = False

            prompt = '\n    Variance threshold (0 = skip%s) [min %.4f, max %.4f]: ' % (
                '' if threshold is None else ', Enter = accept %.4f' % threshold, var_min, var_max)
            raw = input(prompt).strip()
            if raw == '':
                if threshold is None:
                    print('    Enter a threshold, or 0 to skip.')
                    continue
                break
            try:
                val = float(raw)
            except ValueError:
                print('    Enter a number.')
                continue
            if val <= 0 or val < var_min:
                print('    Variance threshold skipped — no bands removed.')
                return
            if val >= var_max:
                print('    Threshold %.4f would remove every band; set it below %.4f.' % (val, var_max))
                continue
            threshold    = val
            discard_mask = variances_scaled <= threshold
            redraw       = True

        support     = ~discard_mask
        retain_cols = [c for c, keep in zip(spectral_cols, support) if keep]
        step_info   = {'process': 'select_variance_threshold',
                       'parameters': {'scaler': scaler_name, 'threshold': threshold}}
        out_fp, _ = self._Accept_and_save(df, df[retain_cols], retain_cols, 'vt',
                                          stem, project_root_fp, overwrite, step_info, units_D,
                                          confirm=False)
        if out_fp and self.verbose >= 1:
            print('    Removed %d band%s, %d retained.' % (
                n_discard, 's' if n_discard != 1 else '', len(retain_cols)))

    # ================================================================== chemometric processing

    def _Load_spec_inputs(self, p, dataframe_param=None):
        '''Parse common params, load full parquet, return input dict or None on error.

        dataframe_param overrides the dataframe parameter (used for each entry of dataframe_array).
        '''
        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return None

        if dataframe_param is None:
            dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()
        show        = getattr(p, 'show', True)
        save        = getattr(p, 'save', False)
        overwrite   = self._Overwrite(p)
        max_spectra = int(getattr(p, 'max_spectra', 100))
        colormap    = Spectra_colormap(getattr(p, 'colormap', 'default'), self.verbose)

        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return None

        df, units_D = Read_parquet_with_units(parquet_fp)
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns (wl_*, wn_*, d*, pc*) found.')
            return None

        wavelengths = [_col_to_index(c) for c in spectral_cols]
        spec_df     = df[spectral_cols].copy()
        spec_df.columns = wavelengths

        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None
        stem     = os.path.splitext(os.path.basename(parquet_fp))[0]

        return {
            'df': df, 'spec_df': spec_df, 'spectral_cols': spectral_cols,
            'wavelengths': wavelengths, 'max_spectra': max_spectra,
            'colormap': colormap, 'ann': 'n=%d' % len(df), 'stem': stem,
            'project_root_fp': project_root_fp, 'units_D': units_D,
            'show': show, 'save': save, 'overwrite': overwrite, 'plot_dir': plot_dir,
        }

    def _Sub_spec(self, spec_df, max_spectra, colormap):
        '''Subsample spec_df rows for display; return (sub_df, colors, n_shown).'''
        n    = len(spec_df)
        skip = max(1, int(np.ceil(n / max_spectra)))
        sub  = spec_df.iloc[::skip].reset_index(drop=True)
        colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(sub)))
        return sub, colors, len(sub)

    def _Plot_input_output(self, spec_in, spec_out, cols_in, cols_out, colors,
                            title_in, title_out, ann, show, save, plot_dir, fname):
        '''2-panel spectra plot: top = input, bottom = output.'''
        wl_in,  xl_in  = _spectra_x_axis_ml(cols_in)
        wl_out, xl_out = _spectra_x_axis_ml(cols_out)

        fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(10, 4))

        for i, row in enumerate(spec_in.values):
            ax_in.plot(wl_in, row, color=colors[i], linewidth=0.5, alpha=0.6)
        ax_in.set_xlabel(xl_in)
        ax_in.set_ylabel('Value')
        ax_in.set_title(title_in)
        ax_in.text(0.98, 0.97, ann, transform=ax_in.transAxes,
                   ha='right', va='top', fontsize=7, color='dimgray')

        for i, row in enumerate(spec_out.values):
            ax_out.plot(wl_out, row, color=colors[i], linewidth=0.5, alpha=0.6)
        ax_out.set_xlabel(xl_out)
        ax_out.set_ylabel('Value')
        ax_out.set_title(title_out)

        plt.tight_layout()
        if save and plot_dir:
            fp = os.path.join(plot_dir, fname)
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % fp)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _Plot_chem_chain(self, steps, max_spectra, colormap, ann,
                          show, save, plot_dir, fname):
        '''Stacked 2-inch-per-panel plot for all steps in a chemometric chain.

        steps: list of (label, spec_df_full, cols) from apply_chemometrics.
        '''
        n    = len(steps)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n),
                                  sharex=False, sharey=False)
        if n == 1:
            axes = [axes]

        for ax, (label, spec_full, cols) in zip(axes, steps):
            sub, colors, _ = self._Sub_spec(spec_full, max_spectra, colormap)
            wl, xl = _spectra_x_axis_ml(cols)
            for i, row in enumerate(sub.values):
                ax.plot(wl, row, color=colors[i], linewidth=0.5, alpha=0.6)
            ax.set_xlabel(xl)
            ax.set_ylabel('Value')
            ax.set_title(label)

        axes[0].text(0.98, 0.97, ann, transform=axes[0].transAxes,
                     ha='right', va='top', fontsize=7, color='dimgray')
        plt.tight_layout()
        if save and plot_dir:
            fp = os.path.join(plot_dir, fname)
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % fp)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _Accept_and_save(self, df_orig, spec_out_full, out_cols, abbrev,
                          stem, project_root_fp, overwrite, step_info, units_D=None,
                          confirm=True):
        '''Prompt y/n (unless confirm=False), save parquet + companion JSON. Returns (out_fp, skipped).'''
        out_stem = Derived_stem(stem, abbrev)
        out_fp   = os.path.join(project_root_fp, out_stem + '.parquet')

        if os.path.exists(out_fp) and not overwrite:
            print('    "%s" already applied. Use overwrite=True to rerun.' % abbrev)
            _proc = (step_info[0].get('process', 'unknown') if isinstance(step_info, list)
                     else step_info.get('process', 'unknown'))
            _write_previous_df(project_root_fp, _proc, [
                {'dataframe': os.path.basename(out_fp),
                 'indicator': None, 'regressor': None, 'selector': None}
            ])
            return out_fp, True

        if confirm:
            ans = input('\n    Accept and save %s result? [y/n]: ' % abbrev).strip().lower()
            if ans != 'y':
                print('    Result discarded.')
                return None, False

        non_spectral = [c for c in df_orig.columns if not _is_spectral_col(c)]
        signal_type  = Signal_type_of(units_D or {}, [c for c in df_orig.columns if _is_spectral_col(c)])
        save_df = spec_out_full.copy()
        # Numeric wavelength columns → wl_nm_ storage convention; d/pc columns kept as-is
        save_df.columns = [Band_col_name(c) if isinstance(c, numbers.Real) else str(c)
                            for c in out_cols]
        saved_cols = list(save_df.columns)
        df_save = pd.concat([
            df_orig[non_spectral].reset_index(drop=True),
            save_df.reset_index(drop=True),
        ], axis=1)
        Save_parquet_with_units(df_save, out_fp, Set_signal_type(units_D, saved_cols, signal_type))
        _proc = (step_info[0].get('process', 'unknown') if isinstance(step_info, list)
                 else step_info.get('process', 'unknown'))
        _write_previous_df(project_root_fp, _proc, [
            {'dataframe': os.path.basename(out_fp),
             'indicator': None, 'regressor': None, 'selector': None}
        ])
        _write_companion_json(project_root_fp, stem, out_stem, saved_cols, step_info, signal_type)
        print('    Saved: %s' % out_fp)
        return out_fp, False

    # ------------------------------------------------------------------ filter steps

    def _Resolve_filter_fp(self, filter_fp_param):
        '''Resolve filter JSON path: relative ./... → project root; absolute as-is.'''
        s = str(filter_fp_param).strip()
        if not s or s.lower() in ('none', 'no'):
            return None
        if os.path.isabs(s):
            return s
        return Get_project_path(self.project_root_FP, s)

    def _Load_named_settings(self, family, param, default_fpn=None):
        '''Settings of a parameter set name (ml_setting), "default" or a JSON path relative to the project root;
        {} for none or a missing file (warning).'''
        try:
            return Load_settings(family, param, default_fpn, resolve_path=self._Resolve_filter_fp,
                                 verbose=self.verbose) or {}
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: %s' % e)
            return {}

    def _Filter_spectra(self):
        '''Single smoothing filter; filter_fp "select" compares the filters in filter_array and asks.

        filter_fp and every filter_array entry: a filter parameter set name (ml_setting) or a JSON path
        relative to the project root, the .json ending is optional.
        '''
        p   = self.process_S.process.parameters
        _json = lambda x: x + '.json' if Is_path(x) and not x.lower().endswith('.json') else x

        filter_fp_param = _json(str(getattr(p, 'filter_fp', 'none')).strip())
        if filter_fp_param.lower() in ('', 'none', 'no'):
            print('    ERROR: filter_fp (a filter parameter set name, JSON file or select) is required for filter_spectra.')
            return

        if filter_fp_param.lower() == 'select':
            entries = [_json(e) for e in _parse_array_param(getattr(p, 'filter_array', []))] or list(FILTER_PRESETS)
        else:
            entries = [filter_fp_param]
        filters = []                                   # (entry, cfg, abbrev, label)
        for e in entries:
            cfg = self._Load_named_settings('filter', e)
            if not cfg:
                print('    ERROR: filter "%s" could not be loaded.' % e)
                return
            kind = next((k for k in _FILTER_ABBREV if k in cfg), None)
            if kind is None:
                print('    ERROR: no known filter (%s) in "%s".' % (', '.join(_FILTER_ABBREV), e))
                return
            name = os.path.splitext(os.path.basename(e))[0] if Is_path(e) else e
            filters.append((e, cfg, _FILTER_ABBREV[kind], name))

        inp = self._Load_spec_inputs(p)
        if inp is None:
            return
        apply_fn = lambda i: apply_filter(inp['spec_df'], inp['wavelengths'], filters[i][1])

        confirm = True
        if filter_fp_param.lower() == 'select':
            picked = self._Select_method(inp, 'filter', [(i, f[3]) for i, f in enumerate(filters)], apply_fn)
            if picked is None:
                return
            i, df_filtered, cols_out, _ = picked
            confirm = False
        else:
            i = 0
            df_filtered, cols_out, _ = apply_fn(0)
        entry, filter_cfg, abbrev, name = filters[i]

        sub_in,  colors, n_shown = self._Sub_spec(inp['spec_df'], inp['max_spectra'], inp['colormap'])
        sub_out, _,      _       = self._Sub_spec(df_filtered,    inp['max_spectra'], inp['colormap'])
        ann_shown = '%s  shown=%d' % (inp['ann'], n_shown)

        self._Plot_input_output(
            sub_in, sub_out, inp['wavelengths'], list(cols_out), colors,
            'Input spectra', 'Filter: %s' % name,
            ann_shown, inp['show'], inp['save'], inp['plot_dir'],
            'filter_%s.png' % abbrev,
        )

        step_info = {
            '_id': abbrev,
            'process': 'filter_spectra',
            'parameters': {'filter_fp': entry},
            'filter': filter_cfg,
        }
        self._Accept_and_save(inp['df'], df_filtered, list(cols_out), abbrev,
                               inp['stem'], inp['project_root_fp'], inp['overwrite'], step_info,
                               inp['units_D'], confirm=confirm)

    def _Multi_filter_spectra(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        mf_fp_param = str(getattr(p, 'multi_filter_fp',
                           getattr(p, 'mulit_filter_fp', 'none'))).strip()
        mf_cfg = self._Load_named_settings('multi_filter', mf_fp_param)
        if not mf_cfg:
            print('    ERROR: multi_filter_fp (a multi_filter parameter set name or JSON file) is required.')
            return

        dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()
        show            = getattr(p, 'show', True)
        save            = getattr(p, 'save', False)
        overwrite       = self._Overwrite(p)
        max_spectra     = int(getattr(p, 'max_spectra', 100))
        colormap        = Spectra_colormap(getattr(p, 'colormap', 'default'), self.verbose)

        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return

        df, units_D = Read_parquet_with_units(parquet_fp)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        wavelengths = [_col_to_index(c) for c in spectral_cols]
        spec_df     = df[spectral_cols].copy()
        spec_df.columns = wavelengths

        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None
        ann      = 'n=%d' % len(df)

        df_filtered, cols_out, _ = apply_multi_filter(spec_df, wavelengths, mf_cfg)

        sub_in,  colors, n_shown = self._Sub_spec(spec_df,    max_spectra, colormap)
        sub_out, _,      _       = self._Sub_spec(df_filtered, max_spectra, colormap)
        ann_shown = '%s  shown=%d' % (ann, n_shown)

        self._Plot_input_output(
            sub_in, sub_out, wavelengths, list(cols_out), colors,
            'Input spectra', 'Multi-filter',
            ann_shown, show, save, plot_dir,
            'filter_mf.png',
        )

        step_info = {
            '_id': 'mf',
            'process': 'multi_filter_spectra',
            'parameters': {'multi_filter_fp': mf_fp_param},
            'multi_filter': mf_cfg,
        }
        self._Accept_and_save(df, df_filtered, list(cols_out), 'mf',
                               stem, project_root_fp, overwrite, step_info, units_D)

    # ------------------------------------------------------------------ ward clustering

    def _Plot_ward_tune(self, scores, best_nc, show, save, plot_dir):
        '''Seaborn line plot of reconstruction MSE vs n_clusters with best_nc marker.'''
        nc_vals  = sorted(scores.keys())
        means    = [scores[nc][0] for nc in nc_vals]
        stds     = [scores[nc][1] for nc in nc_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(nc_vals, means, marker='o', color='steelblue', linewidth=1.5, label='mean MSE')
        ax.fill_between(nc_vals,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color='steelblue')
        ax.axvline(best_nc, color='red', linestyle='--', linewidth=1,
                   label='best n_clusters = %d' % best_nc)
        ax.set_xlabel('n_clusters')
        ax.set_ylabel('Reconstruction MSE')
        ax.set_title('Ward clustering — reconstruction error vs n_clusters')
        ax.legend(fontsize=8)
        plt.tight_layout()

        if save and plot_dir:
            fp = os.path.join(plot_dir, 'ward_tune.png')
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % fp)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _Ward_clustering(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        wc_fp_param = str(getattr(p, 'ward_clustering_fp', 'none')).strip()
        wc_fp       = self._Resolve_filter_fp(wc_fp_param)
        if wc_fp is None:
            print('    ERROR: ward_clustering_fp is required for ward_clustering.')
            return
        if not os.path.exists(wc_fp):
            print('    ERROR: ward clustering file not found: %s' % wc_fp)
            return

        cfg = Load_config_json(wc_fp)
        wc = cfg.get('ward_clustering', cfg)

        dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()
        show            = getattr(p, 'show', True)
        save            = getattr(p, 'save', False)
        overwrite       = self._Overwrite(p)
        max_spectra     = int(getattr(p, 'max_spectra', 100))
        colormap        = Spectra_colormap(getattr(p, 'colormap', 'default'), self.verbose)

        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return

        df, units_D = Read_parquet_with_units(parquet_fp)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        wavelengths = [_col_to_index(c) for c in spectral_cols]
        spec_df     = df[spectral_cols].copy()
        spec_df.columns = wavelengths

        X        = spec_df.values.astype(float)
        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None
        ann      = 'n=%d' % len(df)

        connectivity = bool(getattr(p, 'connectivity', True))
        n_bands      = X.shape[1]

        # Check for an existing result before spending time on tuning
        if os.path.exists(os.path.join(project_root_fp, Derived_stem(stem, 'wc') + '.parquet')) and not overwrite:
            self._Accept_and_save(df, None, [], 'wc', stem, project_root_fp, overwrite,
                                  {'process': 'ward_clustering'}, units_D)
            return

        # Suggested n_clusters: tuned best, or n_clusters from the ward clustering file
        tune_cfg = wc.get('tune_ward_clustering', {})
        n_tuned  = None
        if tune_cfg.get('apply', False):
            clusters_list = [c for c in tune_cfg.get('clusters', []) if 0 < c < n_bands]
            kfolds        = int(tune_cfg.get('kfolds', 3))
            if not clusters_list:
                print('    ERROR: tune_ward_clustering.clusters list is empty or exceeds n_bands.')
                return
            print('    Tuning Ward clustering over %d candidate cluster counts (connectivity=%s) ...'
                  % (len(clusters_list), connectivity))
            scores  = _Ward_tune(X, clusters_list, kfolds, connectivity)
            n_tuned = _Ward_pick_best(scores)
            print('    Tuned: best n_clusters = %d' % n_tuned)
            self._Plot_ward_tune(scores, n_tuned, show, save, plot_dir)
            n_suggest = n_tuned
        else:
            n_suggest = int(wc.get('n_clusters', 0))
            if n_suggest <= 0:
                print('    ERROR: n_clusters must be > 0 when tune_ward_clustering.apply is false.')
                return
        n_suggest = min(n_suggest, n_bands - 1)

        n_clusters = self._Ward_ask_n_clusters(n_suggest, n_bands)
        if n_clusters == 0:
            print('    No clustering saved.')
            return

        ward, order = Ward_fit(X, n_clusters, connectivity)
        X_out    = ward.transform(X)[:, order]
        out_cols = ['wc%d' % (i + 1) for i in range(n_clusters)]
        rank     = np.empty(n_clusters, dtype=int)
        rank[order] = np.arange(n_clusters)
        band_labels = rank[ward.labels_]            # 0-based wc index of every band
        cluster_bands = {'wc%d' % (k + 1): [wavelengths[i] for i in np.where(band_labels == k)[0]]
                         for k in range(n_clusters)}

        spec_out  = pd.DataFrame(X_out, columns=out_cols, index=spec_df.index)
        spec_rec  = pd.DataFrame(X_out[:, band_labels], columns=wavelengths, index=spec_df.index)
        sub_in,  colors, n_shown = self._Sub_spec(spec_df, max_spectra, colormap)
        sub_out, _,      _       = self._Sub_spec(spec_out, max_spectra, colormap)
        sub_rec, _,      _       = self._Sub_spec(spec_rec, max_spectra, colormap)
        ann_shown = '%s  shown=%d' % (ann, n_shown)

        self._Plot_input_output(
            sub_in, sub_out, wavelengths, out_cols, colors,
            'Input spectra', 'Ward clustering (%d clusters)' % n_clusters,
            ann_shown, show, save, plot_dir,
            'ward_clustering_wc.png',
        )
        self._Plot_ward_wavelength(sub_in, sub_rec, wavelengths, band_labels, colors,
                                   n_clusters, connectivity, ann_shown, show, save, plot_dir)

        step_info = {
            '_id': 'wc',
            'process': 'ward_clustering',
            'parameters': {'ward_clustering_fp': wc_fp_param,
                           'connectivity': connectivity,
                           'n_clusters_tuned_best': n_tuned,
                           'n_clusters_selected': n_clusters,
                           'n_clusters_applied': n_clusters},
            'agglomeration': cfg,
            'cluster_bands': cluster_bands,
        }
        self._Accept_and_save(df, spec_out, out_cols, 'wc',
                               stem, project_root_fp, overwrite, step_info, units_D,
                               confirm=False)

    def _Ward_ask_n_clusters(self, n_suggest, n_bands):
        '''Ask for the number of clusters to save: Enter = n_suggest, 0 = save nothing.'''
        while True:
            raw = input('\n    Number of clusters to save (0 = no clustering, Enter = %d) [1-%d]: '
                        % (n_suggest, n_bands - 1)).strip()
            if raw == '':
                return n_suggest
            try:
                n = int(raw)
            except ValueError:
                print('    Enter an integer.')
                continue
            if n == 0 or 0 < n < n_bands:
                return n
            print('    n_clusters must be 0 or between 1 and %d.' % (n_bands - 1))

    def _Plot_ward_wavelength(self, spec_in, spec_rec, wavelengths, band_labels, colors,
                              n_clusters, connectivity, ann, show, save, plot_dir):
        '''2-panel plot on the wavelength axis: input spectra / every band replaced by its cluster mean.

        With connectivity the cluster borders are drawn as thin vertical lines.
        '''
        wl, xl = _spectra_x_axis_ml(wavelengths)

        fig, (ax_in, ax_rec) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
        for i, row in enumerate(spec_in.values):
            ax_in.plot(wl, row, color=colors[i], linewidth=0.5, alpha=0.6)
        for i, row in enumerate(spec_rec.values):
            ax_rec.plot(wl, row, color=colors[i], linewidth=0.5, alpha=0.6, drawstyle='steps-mid')
        if connectivity:
            for j in np.where(np.diff(band_labels) != 0)[0]:
                border = (wl[j] + wl[j + 1]) / 2.0
                ax_rec.axvline(border, color='lightgray', linewidth=0.4, zorder=0)
        ax_in.set_ylabel('Value')
        ax_in.set_title('Input spectra')
        ax_in.text(0.98, 0.97, ann, transform=ax_in.transAxes,
                   ha='right', va='top', fontsize=7, color='dimgray')
        ax_rec.set_xlabel(xl)
        ax_rec.set_ylabel('Cluster mean')
        ax_rec.set_title('Ward clustering (%d clusters%s) on the wavelength axis'
                         % (n_clusters, ', contiguous bands' if connectivity else ''))
        plt.tight_layout()

        if save and plot_dir:
            fp = os.path.join(plot_dir, 'ward_clustering_wl.png')
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % fp)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------ univariate selection

    def _Plot_uv_result(self, spec_df, spec_retained, spec_discarded,
                         wavelengths, retained_wl, discarded_wl,
                         title_suffix, ann, show, save, plot_dir, fname, colormap, max_spectra):
        '''3-panel univariate selection plot: input / retained (highest F) / discarded (lowest F).

        Each panel has a row of small dots along the bottom x-axis marking band positions:
        panel 1 (input): green = retained, red = discarded;
        panel 2 (retained): green dots; panel 3 (discarded): red dots.
        '''
        n    = len(spec_df)
        skip = max(1, int(np.ceil(n / max_spectra)))
        idx  = list(range(0, n, skip))
        colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(idx)))

        _dot_kw = dict(linestyle='none', marker='o', markersize=2.5,
                       alpha=0.8, clip_on=False)
        _zeros  = lambda n: [0] * n   # y=0 in axes-fraction coords (bottom)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

        for i, row_i in enumerate(idx):
            ax1.plot(wavelengths, spec_df.iloc[row_i].values,
                     color=colors[i], linewidth=0.5, alpha=0.6)
        # Band-position markers: green = retained, red = discarded
        t1 = ax1.get_xaxis_transform()
        ax1.plot(retained_wl,  _zeros(len(retained_wl)),  color='green', transform=t1, **_dot_kw)
        ax1.plot(discarded_wl, _zeros(len(discarded_wl)), color='red',   transform=t1, **_dot_kw)
        ax1.set_ylabel('Value')
        ax1.set_title('Input spectra  (%d bands)' % len(wavelengths))
        ax1.text(0.98, 0.97, ann, transform=ax1.transAxes,
                 ha='right', va='top', fontsize=7, color='dimgray')

        for i, row_i in enumerate(idx):
            ax2.plot(retained_wl, spec_retained.iloc[row_i].values,
                     color=colors[i], linewidth=0.5, alpha=0.6)
        t2 = ax2.get_xaxis_transform()
        ax2.plot(retained_wl, _zeros(len(retained_wl)), color='green', transform=t2, **_dot_kw)
        ax2.set_ylabel('Value')
        ax2.set_title('Retained  (%d of %d bands) — highest F-score  [%s]' % (
            len(retained_wl), len(wavelengths), title_suffix))

        for i, row_i in enumerate(idx):
            ax3.plot(discarded_wl, spec_discarded.iloc[row_i].values,
                     color=colors[i], linewidth=0.5, alpha=0.6)
        t3 = ax3.get_xaxis_transform()
        ax3.plot(discarded_wl, _zeros(len(discarded_wl)), color='red', transform=t3, **_dot_kw)
        ax3.set_ylabel('Value')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_title('Discarded  (%d of %d bands) — lowest F-score' % (
            len(discarded_wl), len(wavelengths)))

        plt.tight_layout()
        if save and plot_dir:
            fp = os.path.join(plot_dir, fname)
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print('    Plot saved: %s' % fp)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _Univariate_selection(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        uv_fp_param = str(getattr(p, 'univariate_selector_fp', 'none')).strip()
        uv_fp       = self._Resolve_filter_fp(uv_fp_param)
        if uv_fp is None:
            print('    ERROR: univariate_selector_fp is required.')
            return
        if not os.path.exists(uv_fp):
            print('    ERROR: univariate selector file not found: %s' % uv_fp)
            return

        cfg = Load_config_json(uv_fp)

        n_features          = int(cfg.get('SelectKBest', {}).get('n_features', 20))
        test_size           = float(getattr(p, 'test_size', 0.3))
        dataframe_param     = str(getattr(p, 'dataframe', 'raw')).strip()
        indicator_array     = _parse_array_param(getattr(p, 'indicator_array', []))
        separate_selections = bool(getattr(p, 'separate_selections', False))
        show                = getattr(p, 'show', True)
        save                = getattr(p, 'save', False)
        overwrite           = self._Overwrite(p)
        max_spectra         = int(getattr(p, 'max_spectra', 100))
        colormap            = Spectra_colormap(getattr(p, 'colormap', 'default'), self.verbose)

        parquet_fp = _resolve_dataframe(project_root_fp, dataframe_param)
        if parquet_fp is None:
            return

        df, units_D = Read_parquet_with_units(parquet_fp)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        wavelengths = [_col_to_index(c) for c in spectral_cols]
        spec_df     = df[spectral_cols].copy()
        spec_df.columns = wavelengths

        all_indicators = [
            c for c in df.columns
            if not _is_metadata_col(c)
            and not _is_spectral_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if indicator_array:
            for col in indicator_array:
                if col not in all_indicators:
                    print('    WARNING: indicator "%s" not found or not numeric, skipping.' % col)
            indicator_cols = [c for c in indicator_array if c in all_indicators]
        else:
            indicator_cols = all_indicators

        if not indicator_cols:
            print('    ERROR: no valid indicator columns found.')
            return

        plot_dir = self._Build_plot_output_path(project_root_fp) if save else None
        X_full   = spec_df.values.astype(float)
        n_bands  = X_full.shape[1]
        k        = min(n_features, n_bands)

        # Select from the training samples only — the test samples (same holdout as regression_modeling) stay unseen
        train_rows = ~Holdout_test_mask(df, test_size)[0]
        holdout    = Holdout_info(df, test_size)
        if self.verbose >= 1:
            print('    Selecting from %d training samples (test_size=%.2f held out)' % (int(train_rows.sum()), test_size))

        if separate_selections:
            for indicator in indicator_cols:
                abbrev     = 'uv-' + indicator.replace(' ', '-')
                valid_mask = ~df[indicator].isna() & train_rows
                n_valid    = int(valid_mask.sum())
                if n_valid < 5:
                    print('    WARNING: too few valid samples for "%s" (%d), skipping.' % (
                        indicator, n_valid))
                    continue

                X_valid = X_full[valid_mask.values]
                y_valid = df.loc[valid_mask, indicator].values.astype(float)

                selector = SelectKBest(f_regression, k=k)
                selector.fit(X_valid, y_valid)
                support  = selector.get_support()   # True = highest F-score = retained
                f_scores = np.nan_to_num(selector.scores_, nan=0.0)

                retained_wl   = [wl for wl, keep in zip(wavelengths,   support) if keep]
                retained_cols = [c  for c,  keep in zip(spectral_cols, support) if keep]
                discarded_wl  = [wl for wl, keep in zip(wavelengths,   support) if not keep]

                if self.verbose >= 1:
                    print('    "%s": retained F-score range [%.1f – %.1f],  '
                          'discarded range [%.1f – %.1f]' % (
                              indicator,
                              f_scores[support].min(), f_scores[support].max(),
                              f_scores[~support].min(), f_scores[~support].max()))

                spec_out_df = spec_df[retained_wl].copy()
                spec_dis_df = spec_df[discarded_wl].copy()
                ann_shown   = 'n=%d (valid=%d)' % (len(df), n_valid)

                self._Plot_uv_result(
                    spec_df, spec_out_df, spec_dis_df,
                    wavelengths, retained_wl, discarded_wl,
                    indicator, ann_shown, show, save, plot_dir,
                    'univariate_%s.png' % abbrev, colormap, max_spectra,
                )

                step_info = {
                    '_id': abbrev,
                    'process': 'spectra_indicator_univariate_selection',
                    'parameters': {
                        'univariate_selector_fp': uv_fp_param,
                        'indicator': indicator,
                        'n_features_selected': len(retained_cols),
                        'test_size': test_size,
                    },
                    'univariate_selector': cfg,
                    'selected_bands': retained_wl,
                    'holdout': holdout,
                }
                self._Accept_and_save(df, spec_out_df, retained_wl, abbrev,
                                       stem, project_root_fp, overwrite, step_info, units_D)

        else:
            # Combined: average F-scores across all indicators
            scores_sum  = np.zeros(n_bands)
            n_valid_ind = 0
            for indicator in indicator_cols:
                valid_mask = ~df[indicator].isna() & train_rows
                if valid_mask.sum() < 5:
                    print('    WARNING: too few valid samples for "%s" (%d), skipping.' % (
                        indicator, int(valid_mask.sum())))
                    continue
                X_valid = X_full[valid_mask.values]
                y_valid = df.loc[valid_mask, indicator].values.astype(float)
                sel = SelectKBest(f_regression, k='all')
                sel.fit(X_valid, y_valid)
                scores_sum  += np.nan_to_num(sel.scores_, nan=0.0)
                n_valid_ind += 1

            if n_valid_ind == 0:
                print('    ERROR: no valid indicators for combined univariate selection.')
                return

            mean_scores   = scores_sum / n_valid_ind
            top_k_idx     = np.argsort(mean_scores)[::-1][:k]   # highest F first
            top_k_sorted  = sorted(top_k_idx)                    # restore wavelength order
            discard_idx   = sorted(set(range(n_bands)) - set(top_k_sorted))

            retained_wl   = [wavelengths[i]   for i in top_k_sorted]
            retained_cols = [spectral_cols[i] for i in top_k_sorted]
            discarded_wl  = [wavelengths[i]   for i in discard_idx]

            if self.verbose >= 1:
                print('    Combined (%d indicators): retained F-score range [%.1f – %.1f],  '
                      'discarded range [%.1f – %.1f]' % (
                          n_valid_ind,
                          mean_scores[top_k_sorted].min(), mean_scores[top_k_sorted].max(),
                          mean_scores[discard_idx].min(),  mean_scores[discard_idx].max()))

            spec_out_df = spec_df[retained_wl].copy()
            spec_dis_df = spec_df[discarded_wl].copy()
            ann_shown   = 'n=%d' % len(df)

            self._Plot_uv_result(
                spec_df, spec_out_df, spec_dis_df,
                wavelengths, retained_wl, discarded_wl,
                'combined (%d indicators)' % n_valid_ind, ann_shown,
                show, save, plot_dir, 'univariate_uv.png', colormap, max_spectra,
            )

            step_info = {
                '_id': 'uv',
                'process': 'spectra_indicator_univariate_selection',
                'parameters': {
                    'univariate_selector_fp': uv_fp_param,
                    'indicators': indicator_cols,
                    'n_features_selected': len(retained_cols),
                    'test_size': test_size,
                },
                'univariate_selector': cfg,
                'selected_bands': retained_wl,
                'holdout': holdout,
            }
            self._Accept_and_save(df, spec_out_df, retained_wl, 'uv',
                                   stem, project_root_fp, overwrite, step_info, units_D)

    # ------------------------------------------------------------------ permutation / RFE / tree selector

    def _Permutation_rfe_selection(self):
        '''Band selection: permutation importance, RFE, or tree-based — across regressor × selector.'''
        # Lazy imports — machine_learning_model.py imports from this module (circular guard)
        from src.ai4sh.machine_learning_model import _build_regressor, _DEFAULT_MODEL_PARAMS_FP
        from sklearn.feature_selection import RFE, RFECV
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance as sk_perm_importance
        from sklearn.base import clone

        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        dataframe_param     = str(getattr(p, 'dataframe',    'raw')).strip()
        indicator_array     = _parse_array_param(getattr(p, 'indicator_array',  []))
        selector_array      = _parse_array_param(getattr(p, 'selector_array',   []))
        regressor_array     = _parse_array_param(getattr(p, 'regressor_array',  []))
        model_params_param  = str(getattr(p, 'model_parameters_fp', 'default')).strip()
        separate_selections = bool(getattr(p, 'separate_selections', False))
        n_top               = int(getattr(p, 'n_top_features_in_plot', 0))
        overwrite           = self._Overwrite(p)
        test_size           = float(getattr(p, 'test_size', 0.3))

        if not selector_array:
            print('    ERROR: selector_array is empty.')
            return

        # Determine which selectors need a regressor
        _NEEDS_REGRESSOR = {'permutation', 'permutation_selector', 'rfe'}

        if not regressor_array and any(s.lower() in _NEEDS_REGRESSOR for s in selector_array):
            print('    ERROR: regressor_array is required for permutation / rfe selectors.')
            return

        # Model hyperparameters and selector settings (parameter set name, default, none or JSON path)
        model_params = self._Load_named_settings('regressor', model_params_param or 'default', _DEFAULT_MODEL_PARAMS_FP)
        sel_defaults = self._Load_named_settings('selector', 'default',
                                                 Default_fpn('selector', 'selector_default_settings.json'))
        # keys of the settings file: permutation / tree_based (older code read permutation_selector / tree_based_selector)
        perm_cfg     = sel_defaults.get('permutation', sel_defaults.get('permutation_selector', {}))
        tree_cfg     = sel_defaults.get('tree_based', sel_defaults.get('tree_based_selector', {}))
        perm_repeats = int(perm_cfg.get('permutation_repeats', 6))
        rfe_cv       = int(sel_defaults.get('rfe', {}).get('CV', 0))
        rfe_step     = int(sel_defaults.get('rfe', {}).get('step', 1))
        tree_n_est   = int(tree_cfg.get('n_estimators', 20))

        # Load parquet
        if dataframe_param.lower() == 'previous':
            # Resolve after indicator_cols is known — use single-entry path for now;
            # multi-combo resolution happens inside the loop if prev_map is returned.
            prev_map = None  # resolved below after indicator_cols is determined
            parquet_fp = _resolve_single_previous(project_root_fp)
            if parquet_fp is None:
                return
        else:
            prev_map = None
            try:
                parquet_fp = _resolve_input_parquet(project_root_fp, dataframe_param)
            except FileNotFoundError as e:
                print('    ERROR: %s' % e)
                return
        df, units_D = Read_parquet_with_units(parquet_fp)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        wavelengths = [_col_to_index(c) for c in spectral_cols]
        X_all       = df[spectral_cols].values.astype(float)
        n_bands     = len(spectral_cols)

        # Select from the training samples only — the test samples (same holdout as regression_modeling) stay unseen
        train_rows = ~Holdout_test_mask(df, test_size)[0]
        holdout    = Holdout_info(df, test_size)
        if self.verbose >= 1:
            print('    Selecting from %d training samples (test_size=%.2f held out)' % (int(train_rows.sum()), test_size))

        all_indicators = [
            c for c in df.columns
            if not _is_metadata_col(c)
            and not _is_spectral_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if indicator_array:
            for col in indicator_array:
                if col not in all_indicators:
                    print('    WARNING: indicator "%s" not found, skipping.' % col)
            indicator_cols = [c for c in indicator_array if c in all_indicators]
        else:
            indicator_cols = all_indicators
        if not indicator_cols:
            print('    ERROR: no valid indicator columns.')
            return

        # Build regressor objects from regressor_array
        regressors = {}
        for reg_key in regressor_array:
            hp  = model_params.get(reg_key, {}).get('hyper_parameters', {})
            reg = _build_regressor(reg_key, hp)
            if reg is None:
                print('    WARNING: could not build regressor "%s", skipping.' % reg_key)
            else:
                regressors[reg_key] = reg
        if not regressors and any(s.lower() in _NEEDS_REGRESSOR for s in selector_array):
            print('    ERROR: no valid regressors built.')
            return

        # Output dirs
        plot_dir = os.path.join(project_root_fp, 'plot', 'selection_plot')
        cov_dir  = os.path.join(project_root_fp, 'covariate_importance')
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(cov_dir,  exist_ok=True)

        def _ind_safe(ind):
            return ind.replace(' ', '-').replace('/', '-')

        def _sel_suffix(sel_key):
            k = sel_key.lower()
            if k in ('permutation', 'permutation_selector'):
                return 'pms'
            if k == 'rfe':
                return 'rfe'
            return 'tree'

        def _compute_scores(sel_key, regressor, X, y):
            '''
            Returns (scores_1d, pi_or_None) — scores are higher-is-better.
            regressor is ignored for tree_based (uses its own ExtraTreesRegressor).
            '''
            k = sel_key.lower()
            if k in ('permutation', 'permutation_selector'):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=42, shuffle=True)
                model = clone(regressor)
                model.fit(X_tr, y_tr)
                pi = sk_perm_importance(model, X_te, y_te,
                                        n_repeats=perm_repeats, random_state=42)
                return pi.importances_mean, pi
            if k == 'rfe':
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=42, shuffle=True)

                # Probe-fit to check whether this regressor exposes coef_ or
                # feature_importances_ — required for RFE's importance_getter='auto'.
                # Models like knn and svr(rbf) don't, so RFE cannot use them.
                probe = clone(regressor)
                rfe_compatible = False
                try:
                    n_probe = min(20, len(X_tr))
                    probe.fit(X_tr[:n_probe], y_tr[:n_probe])
                    rfe_compatible = (
                        hasattr(probe, 'feature_importances_') or
                        hasattr(probe, 'coef_') or
                        (hasattr(probe, 'named_steps') and
                         hasattr(probe.named_steps.get('clf', object()), 'coef_'))
                    )
                except Exception:
                    pass

                if rfe_compatible:
                    if hasattr(probe, 'named_steps') and not hasattr(probe, 'coef_'):
                        def imp_getter(est):
                            return np.abs(est.named_steps['clf'].coef_).ravel()
                    else:
                        imp_getter = 'auto'
                    model = clone(regressor)
                    if rfe_cv > 0:
                        selector = RFECV(model, min_features_to_select=1,
                                         step=rfe_step, cv=rfe_cv,
                                         importance_getter=imp_getter)
                    else:
                        selector = RFE(model, n_features_to_select=1, step=rfe_step,
                                       importance_getter=imp_getter)
                    selector.fit(X_tr, y_tr)
                    return 1.0 / selector.ranking_.astype(float), None
                else:
                    # RFE cannot handle this model — skip RFE entirely and fall
                    # back to standalone permutation importance on the full set.
                    print('      ⚠️  %s is RFE-incompatible (no coef_/feature_importances_); '
                          'using permutation importance instead.'
                          % type(probe).__name__)
                    model = clone(regressor)
                    model.fit(X_tr, y_tr)
                    pi = sk_perm_importance(model, X_te, y_te,
                                            n_repeats=perm_repeats, random_state=42)
                    return pi.importances_mean, pi
            if k in ('tree_based', 'tree', 'treebased'):
                model = ExtraTreesRegressor(n_estimators=tree_n_est, random_state=42)
                model.fit(X, y)
                return model.feature_importances_, None
            print('    WARNING: unknown selector type "%s", skipping.' % sel_key)
            return None, None

        def _make_bar_plot(scores, pi_or_none, title):
            '''Horizontal bar of top-N scores; best bar at top. Returns (fig, order).'''
            order    = np.argsort(scores)[::-1]
            n_show   = min(n_top, n_bands) if n_top > 0 else n_bands
            show_idx = order[:n_show]
            plot_wl  = [wavelengths[i] for i in show_idx][::-1]
            plot_sc  = scores[show_idx][::-1]
            std_arr  = (pi_or_none.importances_std[show_idx][::-1]
                        if pi_or_none is not None else None)
            n_bars = len(plot_wl)
            fig, ax = plt.subplots(figsize=(6, max(4, n_bars * 0.15)))
            ax.barh(range(n_bars), plot_sc, xerr=std_arr,
                    color='steelblue', alpha=0.75, height=0.8)
            if n_bars <= 40:
                ax.set_yticks(range(n_bars))
                ax.set_yticklabels([str(w) for w in plot_wl], fontsize=7)
            else:
                ax.set_yticks([])
            ax.set_xlabel('Score (higher = more important)')
            ax.set_ylabel('Wavelength (nm)')
            ax.set_title(title)
            plt.tight_layout()
            return fig, order

        def _save_scores_json(scores, pi_or_none, indicator, reg_key, sel_key, ind_s):
            order     = np.argsort(scores)[::-1]
            ranked_wl = [wavelengths[i]            for i in order]
            ranked_sc = [round(float(scores[i]), 6) for i in order]
            data = {
                'indicator':  indicator,
                'regressor':  reg_key,
                'selector':   sel_key,
                'wavelengths': ranked_wl,
                'scores':      ranked_sc,
            }
            if pi_or_none is not None:
                data['importances_std'] = [
                    round(float(pi_or_none.importances_std[i]), 6) for i in order]
                data['importances_all'] = [
                    [round(float(v), 6) for v in pi_or_none.importances[i]] for i in order]
            fname = '%s_%s_%s_%s.json' % (stem, ind_s, reg_key, sel_key)
            with open(os.path.join(cov_dir, fname), 'w') as fh:
                json.dump(data, fh, indent=2)

        def _prompt_and_write(order, indicator, reg_key, sel_key, suffix, ind_s):
            '''Prompt for number of bands to retain, then save parquet + companion JSON.'''
            n_shown = min(n_top, n_bands) if n_top > 0 else n_bands
            try:
                ans = int(input(
                    '\n    How many top bands to retain for "%s" / %s / %s? '
                    '(0 = skip saving, %d bands shown in plot): '
                    % (indicator, reg_key, sel_key, n_shown)
                ).strip())
            except (ValueError, EOFError):
                ans = 0
            if ans <= 0:
                print('    Skipping save.')
                return
            n_retain = min(ans, n_bands)
            top_idx  = sorted(order[:n_retain])
            top_cols = [spectral_cols[i] for i in top_idx]
            top_wl   = [wavelengths[i]   for i in top_idx]
            abbrev   = '%s-%s-%s' % (ind_s, reg_key, suffix)
            out_stem = Derived_stem(stem, abbrev)
            out_fp   = os.path.join(project_root_fp, out_stem + '.parquet')
            if os.path.exists(out_fp) and not overwrite:
                print('    "%s" already exists. Use overwrite=True to rerun.' % out_stem)
                _prev_entries_acc.append({
                    'dataframe': os.path.basename(out_fp),
                    'indicator': indicator,
                    'regressor': reg_key,
                    'selector':  sel_key,
                })
                return
            non_spectral = [c for c in df.columns if not _is_spectral_col(c)]
            spec_out = df[top_cols].copy()
            spec_out.columns = [Band_col_name(wl) for wl in top_wl]
            signal_type = Signal_type_of(units_D, top_cols)
            df_save = pd.concat([
                df[non_spectral].reset_index(drop=True),
                spec_out.reset_index(drop=True),
            ], axis=1)
            Save_parquet_with_units(df_save, out_fp,
                                    Set_signal_type(units_D, list(spec_out.columns), signal_type))
            step_info = {
                '_id': abbrev,
                'process': 'spectra_indicator_permutation_selection',
                'parameters': {
                    'regressor':          reg_key,
                    'selector':           sel_key,
                    'indicator':          indicator,
                    'n_features_selected': n_retain,
                    'test_size':          test_size,
                },
                'selected_bands': top_wl,
                'holdout': holdout,
            }
            _write_companion_json(project_root_fp, stem, out_stem,
                                   list(spec_out.columns), step_info, signal_type)
            # Accumulate entry — caller (_prompt_and_write wrapper) collects and writes all at end
            _prev_entries_acc.append({
                'dataframe': os.path.basename(out_fp),
                'indicator': indicator,
                'regressor': reg_key,
                'selector':  sel_key,
            })
            print('    Saved: %s  (%d bands)' % (out_fp, n_retain))

        # Collects {dataframe, indicator, regressor, selector} for each saved parquet
        _prev_entries_acc = []

        # ---- main loop: indicator × regressor × selector ----
        if separate_selections:
            for indicator in indicator_cols:
                valid_mask = ~df[indicator].isna() & train_rows
                n_valid    = int(valid_mask.sum())
                if n_valid < 10:
                    print('    WARNING: too few samples for "%s" (%d), skipping.' % (
                        indicator, n_valid))
                    continue
                X     = X_all[valid_mask]
                y     = df.loc[valid_mask, indicator].values.astype(float)
                ind_s = _ind_safe(indicator)
                if self.verbose >= 1:
                    print('    Indicator: %s  (%d samples, %d bands)' % (
                        indicator, n_valid, n_bands))

                for sel_key in selector_array:
                    suffix = _sel_suffix(sel_key)
                    k      = sel_key.lower()
                    # tree_based needs no external regressor
                    reg_items = (list(regressors.items())
                                 if k not in ('tree_based', 'tree', 'treebased')
                                 else [('tree_based', None)])
                    for reg_key, regressor in reg_items:
                        if self.verbose >= 1:
                            print('      %s / %s' % (sel_key, reg_key))
                        scores, pi = _compute_scores(sel_key, regressor, X, y)
                        if scores is None:
                            continue
                        title = '%s — %s / %s / %s  (top %d of %d)' % (
                            suffix.upper(), indicator, reg_key, sel_key,
                            min(n_top, n_bands) if n_top > 0 else n_bands, n_bands)
                        fig, order = _make_bar_plot(scores, pi, title)
                        plt.show()
                        plot_fn = '%s_%s_%s_%s.png' % (ind_s, reg_key, sel_key, suffix)
                        fig.savefig(os.path.join(plot_dir, plot_fn), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        _save_scores_json(scores, pi, indicator, reg_key, sel_key, ind_s)
                        _prompt_and_write(order, indicator, reg_key, sel_key, suffix, ind_s)

        else:
            # Combined: average scores across indicators, one result per (regressor × selector)
            for sel_key in selector_array:
                suffix = _sel_suffix(sel_key)
                k      = sel_key.lower()
                reg_items = (list(regressors.items())
                             if k not in ('tree_based', 'tree', 'treebased')
                             else [('tree_based', None)])
                for reg_key, regressor in reg_items:
                    if self.verbose >= 1:
                        print('    %s / %s (combined, %d indicators)' % (
                            sel_key, reg_key, len(indicator_cols)))
                    scores_sum  = np.zeros(n_bands)
                    n_valid_ind = 0
                    for indicator in indicator_cols:
                        valid_mask = ~df[indicator].isna() & train_rows
                        if valid_mask.sum() < 10:
                            continue
                        X = X_all[valid_mask]
                        y = df.loc[valid_mask, indicator].values.astype(float)
                        sc, _ = _compute_scores(sel_key, regressor, X, y)
                        if sc is None:
                            continue
                        scores_sum  += sc
                        n_valid_ind += 1
                    if n_valid_ind == 0:
                        print('    WARNING: no valid indicators for %s / %s.' % (sel_key, reg_key))
                        continue
                    mean_scores = scores_sum / n_valid_ind
                    label = 'combined (%d indicators)' % n_valid_ind
                    title = '%s — %s / %s / %s  (top %d of %d)' % (
                        suffix.upper(), label, reg_key, sel_key,
                        min(n_top, n_bands) if n_top > 0 else n_bands, n_bands)
                    fig, order = _make_bar_plot(mean_scores, None, title)
                    plt.show()
                    plot_fn = 'combined_%s_%s_%s.png' % (reg_key, sel_key, suffix)
                    fig.savefig(os.path.join(plot_dir, plot_fn), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    _save_scores_json(mean_scores, None, label, reg_key, sel_key, 'combined')
                    _prompt_and_write(order, label, reg_key, sel_key, suffix, 'combined')

        if _prev_entries_acc:
            _write_previous_df(project_root_fp, 'spectra_indicator_permutation_selection',
                               _prev_entries_acc)

    # ------------------------------------------------------------------ individual steps

    def _Spectra_derivative(self):
        '''Derivative; with more than one dataframe in dataframe_array the derivative of every input is
        compared and the user chooses which one to save.'''
        p      = self.process_S.process.parameters
        inputs = _parse_array_param(getattr(p, 'dataframe_array', []))
        inp    = None
        if len(inputs) <= 1:
            inp = self._Load_spec_inputs(p, inputs[0] if inputs else None)
            if inp is None:
                return

        derive = int(getattr(p, 'derivative', 1))
        append = bool(getattr(p, 'append', False))

        def derivative(inp):
            spec_out, cols_out, label = apply_derivative(
                inp['spec_df'], inp['wavelengths'], {'derive': derive})
            if append:
                cols_all = inp['wavelengths'] + list(cols_out)
                spec_out = pd.concat([inp['spec_df'].reset_index(drop=True),
                                       spec_out.reset_index(drop=True)], axis=1)
                spec_out.columns = cols_all
                cols_out = cols_all
            return spec_out, cols_out, 'derivative %d' % derive

        confirm = True
        if len(_parse_array_param(getattr(p, 'dataframe_array', []))) > 1:
            picked = self._Compare_inputs(p, 'derivative %d' % derive, derivative)
            if picked is None:
                return
            inp, spec_out, cols_out, _ = picked
            confirm = False
        else:
            spec_out, cols_out, _ = derivative(inp)

        sub_in,  colors, n_shown = self._Sub_spec(inp['spec_df'], inp['max_spectra'], inp['colormap'])
        sub_out, _,      _       = self._Sub_spec(spec_out,       inp['max_spectra'], inp['colormap'])
        abbrev    = 'd%dapp' % derive if append else 'd%d' % derive
        ann_shown = '%s  shown=%d' % (inp['ann'], n_shown)

        self._Plot_input_output(
            sub_in, sub_out, inp['wavelengths'], list(cols_out), colors,
            'Input spectra: %s' % inp['stem'],
            'Derivative order %d%s' % (derive, ' (appended)' if append else ''),
            ann_shown, inp['show'], inp['save'], inp['plot_dir'],
            'chemometric_%s.png' % abbrev,
        )
        self._Accept_and_save(
            inp['df'], spec_out, list(cols_out), abbrev,
            inp['stem'], inp['project_root_fp'], inp['overwrite'],
            {'process': 'spectra_derivative',
             'parameters': {'derivative': derive, 'append': append}},
            inp['units_D'], confirm=confirm,
        )

    def _Select_method(self, inp, kind, options, apply_fn):
        '''Plot raw + every option stacked, ask 0..n for the one to apply.

        options: list of (option, label); apply_fn(option) → (spec_out, cols_out, label).
        Returns (option, spec_out, cols_out, label), or None for 0 = none.
        '''
        steps, results = [('raw', inp['spec_df'], inp['wavelengths'])], []
        for opt, label in options:
            spec_out, cols_out, lab = apply_fn(opt)
            steps.append(('%d: %s' % (len(results) + 1, label), spec_out, cols_out))
            results.append((opt, spec_out, cols_out, lab))
        self._Plot_chem_chain(steps, inp['max_spectra'], inp['colormap'], inp['ann'],
                              inp['show'], inp['save'], inp['plot_dir'],
                              'compare_%s.png' % kind.replace(' ', '_'))
        n = _Ask_choice('%s to apply' % kind.capitalize(), [label for _, label in options])
        if n == 0:
            print('    No %s applied.' % kind)
            return None
        return results[n - 1]

    def _Compare_inputs(self, p, kind, transform_fn, menu_label=None):
        '''Apply transform_fn to every dataframe in dataframe_array, plot the results stacked, ask 0..n.

        transform_fn(inp) → (spec_out, cols_out, label). Returns (inp, spec_out, cols_out, label) of the
        chosen input, or None for 0 = none or an error. menu_label(result) → prompt text (default: input stem).
        '''
        steps, results = [], []
        for name in _parse_array_param(getattr(p, 'dataframe_array', [])):
            inp = self._Load_spec_inputs(p, name)
            if inp is None:
                return None
            spec_out, cols_out, label = transform_fn(inp)
            steps.append(('%d: %s → %s' % (len(results) + 1, inp['stem'], label), spec_out, cols_out))
            results.append((inp, spec_out, cols_out, label))
        first = results[0][0]
        self._Plot_chem_chain(steps, first['max_spectra'], first['colormap'], '',
                              first['show'], first['save'], first['plot_dir'],
                              'compare_%s.png' % kind.replace(' ', '_'))
        n = _Ask_choice('Input dataframe to apply %s to' % kind,
                        [menu_label(r) if menu_label else r[0]['stem'] for r in results])
        if n == 0:
            print('    No %s applied.' % kind)
            return None
        return results[n - 1]

    def _Spectra_scatter_correction(self):
        '''Scatter correction per spectrum (l1, l2, max, snv, msc); scaler "select" compares all and asks.'''
        p   = self.process_S.process.parameters
        inp = self._Load_spec_inputs(p)
        if inp is None:
            return

        scalers = _parse_array_param(getattr(p, 'scaler', ['snv']))
        apply_fn = lambda sc: apply_scatter_correction(inp['spec_df'], inp['wavelengths'],
                                                       {'scaler': sc if isinstance(sc, list) else [sc]})
        confirm = True
        if scalers == ['select']:
            picked = self._Select_method(inp, 'scatter correction',
                                         [(sc, sc) for sc in SCATTER_CORRECTIONS], apply_fn)
            if picked is None:
                return
            sc, spec_out, cols_out, label = picked
            scalers, confirm = [sc], False
        else:
            spec_out, cols_out, label = apply_fn(scalers)

        sub_in,  colors, n_shown = self._Sub_spec(inp['spec_df'], inp['max_spectra'], inp['colormap'])
        sub_out, _,      _       = self._Sub_spec(spec_out,       inp['max_spectra'], inp['colormap'])
        abbrev    = label.replace('+', '')
        ann_shown = '%s  shown=%d' % (inp['ann'], n_shown)

        self._Plot_input_output(
            sub_in, sub_out, inp['wavelengths'], cols_out, colors,
            'Input spectra', 'Scatter correction: %s' % label,
            ann_shown, inp['show'], inp['save'], inp['plot_dir'],
            'chemometric_%s.png' % abbrev,
        )
        self._Accept_and_save(
            inp['df'], spec_out, cols_out, abbrev,
            inp['stem'], inp['project_root_fp'], inp['overwrite'],
            {'process': 'spectra_scatter_correction',
             'parameters': {'scaler': scalers}},
            inp['units_D'], confirm=confirm,
        )

    def _Spectra_scaling(self):
        '''Variable (band-wise) scaling; method "select" compares all four and asks.'''
        p   = self.process_S.process.parameters
        inp = self._Load_spec_inputs(p)
        if inp is None:
            return

        method   = str(getattr(p, 'method', 'meancentring')).strip()
        apply_fn = lambda m: apply_scaling(inp['spec_df'], inp['wavelengths'], {'method': m})
        confirm  = True
        if method == 'select':
            picked = self._Select_method(inp, 'variable scaling',
                                         [(m, m) for m in VARIABLE_SCALINGS], apply_fn)
            if picked is None:
                return
            method, spec_out, cols_out, label = picked
            confirm = False
        else:
            spec_out, cols_out, label = apply_fn(method)

        sub_in,  colors, n_shown = self._Sub_spec(inp['spec_df'], inp['max_spectra'], inp['colormap'])
        sub_out, _,      _       = self._Sub_spec(spec_out,       inp['max_spectra'], inp['colormap'])
        abbrev    = {'meancentring': 'mc', 'autoscaling': 'as',
                     'paretoscaling': 'ps', 'poissonscaling': 'poi'}.get(method, method[:4])
        ann_shown = '%s  shown=%d' % (inp['ann'], n_shown)

        self._Plot_input_output(
            sub_in, sub_out, inp['wavelengths'], cols_out, colors,
            'Input spectra', 'Variable scaling: %s' % method,
            ann_shown, inp['show'], inp['save'], inp['plot_dir'],
            'chemometric_%s.png' % abbrev,
        )
        self._Accept_and_save(
            inp['df'], spec_out, cols_out, abbrev,
            inp['stem'], inp['project_root_fp'], inp['overwrite'],
            {'process': 'spectra_scaling',
             'parameters': {'method': method}},
            inp['units_D'], confirm=confirm,
        )

    def _Spectra_decomposition(self):
        '''Decomposition (PCA); with more than one dataframe in dataframe_array the decomposition of every
        input is compared (scores + explained variance) and the user chooses which one to save.'''
        p      = self.process_S.process.parameters
        inputs = _parse_array_param(getattr(p, 'dataframe_array', []))
        inp    = None
        if len(inputs) <= 1:
            inp = self._Load_spec_inputs(p, inputs[0] if inputs else None)
            if inp is None:
                return

        method       = str(getattr(p, 'method', 'pca')).strip().lower()
        n_components = int(getattr(p, 'n_components', 5))

        def decomposition(inp):
            spec_out, cols_out, _ = apply_decomposition(
                inp['spec_df'], inp['wavelengths'],
                {'method': method, 'n_components': n_components})
            # Share of the total spectral variance captured by the components (PCA score variances)
            total = float(inp['spec_df'].values.astype(float).var(axis=0).sum())
            share = float(spec_out.values.var(axis=0).sum()) / total if total > 0 else float('nan')
            return spec_out, cols_out, '%s %d (%.1f %% variance)' % (method.upper(), len(cols_out), 100 * share)

        confirm = True
        if len(inputs) > 1:
            picked = self._Compare_inputs(p, '%s %d' % (method, n_components), decomposition,
                                          menu_label=lambda r: '%s (%s)' % (r[0]['stem'], r[3].split('(')[-1].rstrip(')')))
            if picked is None:
                return
            inp, spec_out, cols_out, label = picked
            confirm = False
        else:
            spec_out, cols_out, label = decomposition(inp)

        n_actual  = len(cols_out)
        abbrev    = 'pca%d' % n_actual

        sub_in,  colors, n_shown = self._Sub_spec(inp['spec_df'], inp['max_spectra'], inp['colormap'])
        sub_out, _,      _       = self._Sub_spec(spec_out,       inp['max_spectra'], inp['colormap'])
        ann_shown = '%s  shown=%d' % (inp['ann'], n_shown)

        self._Plot_input_output(
            sub_in, sub_out, inp['wavelengths'], cols_out, colors,
            'Input spectra: %s' % inp['stem'], label,
            ann_shown, inp['show'], inp['save'], inp['plot_dir'],
            'chemometric_%s.png' % abbrev,
        )
        self._Accept_and_save(
            inp['df'], spec_out, cols_out, abbrev,
            inp['stem'], inp['project_root_fp'], inp['overwrite'],
            {'process': 'spectra_decomposition',
             'parameters': {'method': method, 'n_components': n_actual}},
            inp['units_D'], confirm=confirm,
        )

    def _Chemometric_default(self):
        p   = self.process_S.process.parameters
        inp = self._Load_spec_inputs(p)
        if inp is None:
            return

        chemometrics_param = str(getattr(p, 'chemometrics', 'default')).strip() or 'default'
        chemometrics_array = _parse_array_param(getattr(p, 'chemometrics_array', []))

        if not chemometrics_array:
            print('    No chemometrics_array specified — nothing to do.')
            return

        steps = apply_chemometrics(inp['spec_df'], inp['wavelengths'],
                                    chemometrics_array, chemometrics_param, self.verbose)

        # Build chain abbreviation and step_info_list from all steps except 'raw'
        chain_abbrevs  = []
        step_info_list = []
        chem_config    = _load_chemometric_config(chemometrics_param, self.verbose)
        for step_name_key in chemometrics_array:
            cfg = chem_config.get(step_name_key, {})
            if step_name_key == 'derivative':
                d = int(cfg.get('derivative', cfg.get('derive', 1)))
                a = bool(cfg.get('append', False))
                chain_abbrevs.append('d%dapp' % d if a else 'd%d' % d)
                step_info_list.append({'process': 'spectra_derivative',
                                       'parameters': {'derivative': d, 'append': a}})
            elif step_name_key == 'scatter_correction':
                sc = cfg.get('scaler', ['snv'])
                chain_abbrevs.append(''.join(sc))
                step_info_list.append({'process': 'spectra_scatter_correction',
                                       'parameters': {'scaler': sc}})
            elif step_name_key == 'scaling':
                m = cfg.get('method', 'meancentring')
                chain_abbrevs.append({'meancentring': 'mc', 'autoscaling': 'as',
                                      'paretoscaling': 'ps', 'poissonscaling': 'poi'}.get(m, m[:4]))
                step_info_list.append({'process': 'spectra_scaling',
                                       'parameters': {'method': m}})
            elif step_name_key == 'decomposition':
                m = cfg.get('method', 'pca')
                n = int(cfg.get('n_components', 5))
                chain_abbrevs.append('pca%d' % n)
                step_info_list.append({'process': 'spectra_decomposition',
                                       'parameters': {'method': m, 'n_components': n}})

        abbrev = SUFFIX_SEP.join(chain_abbrevs) if chain_abbrevs else 'default'

        self._Plot_chem_chain(
            steps, inp['max_spectra'], inp['colormap'], inp['ann'],
            inp['show'], inp['save'], inp['plot_dir'],
            'chemometric_%s.png' % abbrev,
        )

        _, spec_final, cols_final = steps[-1]
        self._Accept_and_save(
            inp['df'], spec_final, cols_final, abbrev,
            inp['stem'], inp['project_root_fp'], inp['overwrite'],
            step_info_list,
            inp['units_D'],
        )
