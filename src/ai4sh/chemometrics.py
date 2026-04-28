'''
Created on 24 April 2026

@author: thomasgumbricht
'''

import os
import json

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, normalize
from sklearn.decomposition import PCA

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

_DEFAULT_STANDARDISE_FPN = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'transform', 'indicator_standardise_default.json')

_DEFAULT_TRANSFORMATION_FPN = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'transform', 'indicator_transformation_default.json')

_DEFAULT_CHEMOMETRIC_FPN = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'chemometric', 'chemometric_default.json')


def _load_transform_config(param, default_fpn, verbose):
    if not param or str(param).strip().lower() == 'none':
        return None
    fpn = default_fpn if str(param).strip().lower() == 'default' else str(param).strip()
    if not os.path.exists(fpn):
        if verbose >= 1:
            print('    Warning: transform config not found: %s' % fpn)
        return None
    with open(fpn) as f:
        return json.load(f)


def _load_chemometric_config(param, verbose=0):
    if not param or str(param).strip().lower() == 'none':
        return {}
    fpn = _DEFAULT_CHEMOMETRIC_FPN if str(param).strip().lower() == 'default' else str(param).strip()
    if not os.path.exists(fpn):
        if verbose >= 1:
            print('    Warning: chemometric config not found: %s' % fpn)
        return {}
    with open(fpn) as f:
        return json.load(f)


def _snv(X):
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        out[i, :] = (X[i, :] - np.mean(X[i, :])) / np.std(X[i, :])
    return out


def _msc(X, reference=None):
    X = X.copy().astype(float)
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
    ref = np.mean(X, axis=0) if reference is None else reference
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        fit = np.polyfit(ref, X[i, :], 1, full=True)
        out[i, :] = (X[i, :] - fit[0][1]) / fit[0][0]
    return out, ref


# ---------------------------------------------------------------------------
# Indicator (lab data) transformations
# ---------------------------------------------------------------------------

def apply_transformations(df, indicators, param, verbose=0):
    '''Apply per-indicator transformations to a copy of df.

    param: 'default' | 'none' | path to transformation JSON
    Returns (df_out, transform_dict) where transform_dict[col] names the applied transform.
    '''
    transform_dict = {col: 'linear' for col in indicators}

    raw = _load_transform_config(param, _DEFAULT_TRANSFORMATION_FPN, verbose)
    if raw is None:
        return df.copy(), transform_dict

    config = raw.get('transformation', {})
    df_out = df.copy()

    for col in indicators:
        if col not in df_out.columns:
            continue

        col_cfg = config.get(col, {})
        series = df_out[col].dropna()

        if col_cfg.get('log'):
            df_out[col] = np.log(df_out[col])
            transform_dict[col] = 'log'

        elif col_cfg.get('sqrt'):
            df_out[col] = np.sqrt(df_out[col])
            transform_dict[col] = 'sqrt'

        elif col_cfg.get('reciprocal'):
            df_out[col] = np.reciprocal(df_out[col].astype(float))
            transform_dict[col] = 'reciprocal'

        elif col_cfg.get('boxcox'):
            try:
                transformed, _ = boxcox(series)
                df_out.loc[series.index, col] = transformed
                transform_dict[col] = 'boxcox'
            except Exception:
                if verbose >= 1:
                    print('    Warning: boxcox failed for %s — keeping linear.' % col)

        elif col_cfg.get('yeojohnson'):
            try:
                pt = PowerTransformer(method='yeo-johnson')
                vals = df_out[[col]].copy()
                df_out[col] = pt.fit_transform(vals).flatten()
                transform_dict[col] = 'yeojohnson'
            except Exception:
                if verbose >= 1:
                    print('    Warning: yeojohnson failed for %s — keeping linear.' % col)

        elif col_cfg.get('quantile'):
            try:
                qt = QuantileTransformer(output_distribution='normal', random_state=0)
                vals = df_out[[col]].copy()
                df_out[col] = qt.fit_transform(vals).flatten()
                transform_dict[col] = 'quantile'
            except Exception:
                if verbose >= 1:
                    print('    Warning: quantile transform failed for %s — keeping linear.' % col)

        if verbose >= 2 and transform_dict[col] != 'linear':
            print('    Transform %s → %s' % (col, transform_dict[col]))

    return df_out, transform_dict


def apply_standardisation(df, indicators, param, verbose=0):
    '''Apply z-score standardisation to a copy of df for selected indicators.

    param: 'default' | 'none' | path to standardisation JSON
    Returns (df_out, standard_dict) where standard_dict[col] is a stats dict or 'none'.
    '''
    standard_dict = {col: 'none' for col in indicators}

    raw = _load_transform_config(param, _DEFAULT_STANDARDISE_FPN, verbose)
    if raw is None:
        return df.copy(), standard_dict

    config = raw.get('standardise', {})
    df_out = df.copy()

    for col in indicators:
        if col not in df_out.columns:
            continue

        if not config.get(col, False):
            continue

        mean = df_out[col].mean()
        std = df_out[col].std()

        if std == 0:
            if verbose >= 1:
                print('    Warning: std=0 for %s — skipping standardisation.' % col)
            continue

        df_out[col] = (df_out[col] - mean) / std
        standard_dict[col] = {'scaler': 'standardscaler', 'mean': float(mean), 'std': float(std)}

        if verbose >= 2:
            print('    Standardised %s  mean=%.4f  std=%.4f' % (col, mean, std))

    return df_out, standard_dict


# ---------------------------------------------------------------------------
# Spectral (chemometric) transformations
# ---------------------------------------------------------------------------

def apply_derivative(df, columns, cfg):
    '''Apply nth-order finite-difference derivative to spectral DataFrame.

    Columns must be numeric wavelengths (int) or 'd<int>' strings from a prior derivative step.
    Returns (df_out, columns_out, label).
    '''
    derive = int(cfg.get('derive', 1))
    if derive == 0:
        return df[columns].copy(), list(columns), 'raw'

    work_df = df[columns].copy()
    work_cols = list(columns)

    for _ in range(derive):
        diff_df = work_df.diff(axis=1, periods=1).iloc[:, 1:]
        try:
            nums = [int(str(c).lstrip('d')) for c in work_cols]
        except (ValueError, TypeError):
            nums = list(range(len(work_cols)))
        new_cols = ['d%d' % int((nums[i] + nums[i + 1]) / 2) for i in range(len(nums) - 1)]
        diff_df.columns = new_cols
        work_df = diff_df
        work_cols = new_cols

    return work_df, work_cols, 'derivative (order %d)' % derive


def apply_scatter_correction(df, columns, cfg):
    '''Apply one or more scatter corrections in series to spectral DataFrame.

    cfg['scaler']: single string or list — 'l1', 'l2', 'max', 'snv', 'msc'.
    Chained scalers are applied left-to-right, each using the previous output as input.
    Returns (df_out, columns_out, label).
    '''
    scalers = cfg.get('scaler', ['snv'])
    if isinstance(scalers, str):
        scalers = [scalers]

    work = df[columns].copy()

    for sc in scalers:
        X = np.array(work, dtype=float)
        if sc in ('l1', 'l2', 'max'):
            X = normalize(X, norm=sc)
        elif sc == 'snv':
            X = _snv(X)
        elif sc == 'msc':
            X, _ = _msc(X)
        work = pd.DataFrame(data=X, columns=columns, index=df.index)

    return work, list(columns), '+'.join(scalers)


def apply_scaling(df, columns, cfg):
    '''Apply one scaling method to spectral DataFrame.

    cfg['method']: 'meancentring' | 'autoscaling' | 'paretoscaling' | 'poissonscaling'
    Returns (df_out, columns_out, label).
    '''
    method = cfg.get('method', 'meancentring')
    X = df[columns].values.astype(float)

    if method == 'meancentring':
        X_out = StandardScaler(with_std=False).fit_transform(X)

    elif method == 'autoscaling':
        X_out = StandardScaler().fit_transform(X)

    elif method == 'paretoscaling':
        scaler = StandardScaler().fit(X)
        scaler.mean_ = np.zeros(X.shape[1])
        scaler.var_ = np.sqrt(scaler.var_)
        scaler.scale_ = np.sqrt(scaler.var_)
        X_out = scaler.transform(X)

    elif method == 'poissonscaling':
        # Divides each band by sqrt of its mean — Poisson noise model
        scaler = StandardScaler(with_mean=False).fit(X)
        scaler.var_ = scaler.mean_.copy()
        scaler.scale_ = np.sqrt(scaler.var_)
        X_out = scaler.transform(X)

    else:
        X_out = X.copy()

    df_out = pd.DataFrame(data=X_out, columns=columns, index=df.index)
    return df_out, list(columns), method


def apply_decomposition(df, columns, cfg):
    '''Decompose spectral DataFrame (PCA or future methods).

    Returns (df_out, columns_out, label).
    Output columns are named 'pc1', 'pc2', ... for PCA.
    '''
    method = cfg.get('method', 'pca').lower()
    n_components = int(cfg.get('n_components', 5))

    X = df[columns].values.astype(float)

    if method == 'pca':
        n_components = min(n_components, X.shape[0], X.shape[1])
        X_out = PCA(n_components=n_components).fit_transform(X)
        new_cols = ['pc%d' % (i + 1) for i in range(X_out.shape[1])]
        label = 'pca (%d components)' % X_out.shape[1]
    else:
        X_out = X.copy()
        new_cols = list(columns)
        label = method

    df_out = pd.DataFrame(data=X_out, columns=new_cols, index=df.index)
    return df_out, new_cols, label


_CHEMOMETRIC_FUNCS = {
    'derivative': (apply_derivative, 'derivatives'),
    'scatter_correction': (apply_scatter_correction, 'scatter_correction'),
    'scaling': (apply_scaling, 'scaling'),
    'decomposition': (apply_decomposition, 'decomposition'),
}


def apply_chemometrics(df, columns, chemometrics_array, config_param, verbose=0):
    '''Apply chemometric steps in user-defined order.

    chemometrics_array: list of step names, e.g. ['scatter_correction', 'scaling'].
    config_param: 'default' | 'none' | path to chemometric config JSON.

    Returns list of (label, df_out, columns_out).
    First entry is always ('raw', raw_df, columns) so the caller can plot all steps.
    Each subsequent entry is the output of the named step, fed the prior step's output.
    '''
    config = _load_chemometric_config(config_param, verbose)
    results = [('raw', df[columns].copy(), list(columns))]

    if not chemometrics_array:
        return results

    current_df, current_cols = results[0][1], results[0][2]

    for step in chemometrics_array:
        if step not in _CHEMOMETRIC_FUNCS:
            if verbose >= 1:
                print('    Warning: unknown chemometric step "%s" — skipping.' % step)
            continue
        func, cfg_key = _CHEMOMETRIC_FUNCS[step]
        step_cfg = config.get(cfg_key, {})
        try:
            df_out, cols_out, label = func(current_df, current_cols, step_cfg)
            results.append(('%s: %s' % (step, label), df_out, cols_out))
            current_df, current_cols = df_out, cols_out
        except Exception as exc:
            if verbose >= 1:
                print('    Warning: chemometric step "%s" failed: %s' % (step, exc))

    return results
