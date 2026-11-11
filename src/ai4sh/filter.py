'''
Created on 25 April 2026

@author: thomasgumbricht
'''

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import savgol_filter

from src.ai4sh.project_paths import Default_fpn
from src.ai4sh.ml_setting import Load_settings

_DEFAULT_FILTER_DIR = Path(Default_fpn('filter'))

_FILTER_DEFAULT_FILES = {
    'moving-average': 'moving-average_default.json',
    'gauss':          'gauss_default.json',
    'savitzky-golay': 'savitzky-golay_default.json',
    'lowess':         'lowess_default.json',
    'loess':          'loess_default.json',
}


def _load_filter_config(param, family='filter'):
    '''Load filter config dict from param.

    param=None / "none" / "no" → None (no filtering).
    param=set name              → the filter (or multi_filter) parameter set of that name (ml_setting; file
                                  default/filter/ until set up).
    param=path string           → load from that path.
    '''
    name = str(param).strip().lower() if param is not None else ''
    default_fpn = str(_DEFAULT_FILTER_DIR / _FILTER_DEFAULT_FILES[name]) \
        if family == 'filter' and name in _FILTER_DEFAULT_FILES else None
    return Load_settings(family, param, default_fpn)


def _local_poly_matrix(n, n_points, degree):
    '''Smoother matrix H (n x n, sparse) of tricube-weighted local polynomial regression.

    Row i holds the weights that give the fitted value at band i as a linear combination of
    the n_points nearest bands (including band i itself): degree 1 = LOWESS (local linear),
    degree 2 = LOESS (local quadratic).  The bands sit on a regular index grid, so H depends
    only on (n, n_points, degree) and is applied to all spectra at once: X_smooth = X @ H.T.
    '''
    from scipy.sparse import csr_matrix
    n_pts = max(1, min(int(n_points), n))
    x = np.arange(n, dtype=float)
    H = np.zeros((n, n))
    for i in range(n):
        d = np.abs(x - i)
        idx = np.argpartition(d, n_pts - 1)[:n_pts]
        d_i = d[idx]
        max_d = d_i.max()
        w = np.ones(n_pts) if max_d == 0.0 else (1.0 - (d_i / max_d) ** 3) ** 3
        sw = np.sqrt(w)
        A = np.vander(x[idx] - i, degree + 1, increasing=True)   # centred: fitted value at i = beta[0]
        H[i, idx] = np.linalg.pinv(sw[:, None] * A)[0] * sw       # pinv: rank-deficient windows degrade to a lower degree
    return csr_matrix(H)


def _local_poly_smooth(X, n_points, degree):
    '''Tricube-weighted local polynomial smoothing of every row of X (2-D array).'''
    H = _local_poly_matrix(X.shape[1], n_points, degree)
    return np.asarray((H @ X.T).T)


def apply_filter(df, columns, cfg):
    '''Apply a single smoothing filter to every row of df[columns].

    cfg is a dict whose top-level key identifies the filter type.

    Returns (df_out, columns_out, label).
    columns_out == columns (no resampling in single-filter mode).
    label format: "filter:<method>".
    '''
    X = df[columns].values.astype(float)

    if 'moving_average' in cfg:
        mc = cfg['moving_average']
        kernel = np.asarray(mc['kernel'], dtype=float)
        kernel /= kernel.sum()
        mode = mc.get('mode', 'nearest')
        X1 = convolve1d(X, kernel, axis=-1, mode=mode)
        label = 'filter:moving_average'

    elif 'gauss' in cfg:
        gc = cfg['gauss']
        sigma = float(gc['sigma'])
        mode = gc.get('mode', 'nearest')
        X1 = gaussian_filter1d(X, sigma, axis=-1, mode=mode)
        label = 'filter:gauss'

    elif 'savitzky-golay' in cfg:
        sgc = cfg['savitzky-golay']
        wl = int(sgc['window_length'])
        po = int(sgc.get('polyorder', 2))
        deriv = int(sgc.get('deriv', 0))
        mode = sgc.get('mode', 'nearest')
        X1 = savgol_filter(X, window_length=wl, polyorder=po, deriv=deriv, axis=-1, mode=mode)
        label = 'filter:savitzky-golay'

    elif 'lowess' in cfg:
        lc = cfg['lowess']
        n_pts = int(lc['n_points'])
        X1 = _local_poly_smooth(X, n_pts, 1)
        label = 'filter:lowess'

    elif 'loess' in cfg:
        lc = cfg['loess']
        n_pts = int(lc['n_points'])
        if n_pts < 4:
            raise ValueError('loess n_points must be >= 4 (quadratic fit), got %d' % n_pts)
        X1 = _local_poly_smooth(X, n_pts, 2)
        label = 'filter:loess'

    else:
        return df, list(columns), 'filter:none'

    df_out = pd.DataFrame(X1, index=df.index, columns=columns)
    return df_out, list(columns), label


def apply_multi_filter(df, columns, cfg):
    '''Apply per-region filters; wavelength gaps between regions are dropped.

    cfg must contain a "multi_filter" dict with parallel lists:
      begin_wavelength, end_wavelength, and at least one of:
      moving_average.kernel[], savitzky-golay.window_length[], gauss.sigma[],
      lowess.n_points[], loess.n_points[].

    Each region slice is filtered independently; results are concatenated in
    wavelength order.  Columns are integer wavelengths.

    Returns (df_out, cols_out, "filter:multi_filter").
    '''
    mf = cfg['multi_filter']
    begins = mf['begin_wavelength']
    ends   = mf['end_wavelength']
    mode   = mf.get('mode', 'nearest')

    col_vals = np.array([int(c) for c in columns])

    region_dfs  = []
    region_cols = []

    for r in range(len(begins)):
        bwl = begins[r]
        ewl = ends[r]
        mask   = (col_vals >= bwl) & (col_vals < ewl)
        r_cols = [c for c, m in zip(columns, mask) if m]
        if not r_cols:
            continue

        region_cfg = {}

        ma_kernels = mf.get('moving_average', {}).get('kernel', [])
        sg_windows = mf.get('savitzky-golay', {}).get('window_length', [])
        sg_polys   = mf.get('savitzky-golay', {}).get('polyorder', [])
        ga_sigmas  = mf.get('gauss', {}).get('sigma', [])
        lw_npts    = mf.get('lowess', {}).get('n_points', [])
        lo_npts    = mf.get('loess', {}).get('n_points', [])

        if r < len(ma_kernels) and ma_kernels[r]:
            region_cfg['moving_average'] = {'kernel': ma_kernels[r], 'mode': mode}
        elif r < len(sg_windows) and sg_windows[r]:
            po = sg_polys[r] if r < len(sg_polys) else 2
            region_cfg['savitzky-golay'] = {
                'window_length': sg_windows[r], 'polyorder': po, 'mode': mode,
            }
        elif r < len(ga_sigmas) and ga_sigmas[r]:
            region_cfg['gauss'] = {'sigma': ga_sigmas[r], 'mode': mode}
        elif r < len(lw_npts) and lw_npts[r]:
            region_cfg['lowess'] = {'n_points': lw_npts[r]}
        elif r < len(lo_npts) and lo_npts[r]:
            region_cfg['loess'] = {'n_points': lo_npts[r]}

        if region_cfg:
            df_r, cols_r, _ = apply_filter(df, r_cols, region_cfg)
        else:
            df_r  = df[r_cols].copy()
            cols_r = r_cols

        region_dfs.append(df_r)
        region_cols.extend(cols_r)

    if not region_dfs:
        return df, list(columns), 'filter:multi_filter'

    df_out = pd.concat(region_dfs, axis=1)
    return df_out, region_cols, 'filter:multi_filter'
