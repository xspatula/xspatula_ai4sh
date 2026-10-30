'''
Created on 30 October 2026

@author: thomasgumbricht

Chemometric chain: a sequence of spectral preprocessing steps defined in one array parameter (chemometrics_array),
used by the process chemometric_chain (machine_learning_preprocess) and by plot_spectra.

Every item is a method name (default settings) or an object {step: settings} (custom settings):

    "chemometrics_array": [
        "snv",
        {"filter": "./project_neospectra/filter/savitzky-golay_d1.json"},
        "meancentring",
        {"ward_clustering": {"n_clusters": 40}},
        {"pca": {"n_components": 5}}
    ]

Method names (default settings)
    filter              moving-average, gauss, savitzky-golay, lowess (ml_setting filter parameter sets)
    scatter correction  l1, l2, max, snv, msc
    derivative          derivative (1st)
    variable scaling    meancentring, autoscaling, paretoscaling, poissonscaling
    decomposition       pca (5 components)

Objects (custom settings)
    {"filter": <parameter set name | JSON path | {"savitzky-golay": {...}}>}
    {"multi_filter": <parameter set name | JSON path | {"multi_filter": {...}}>}
    {"scatter_correction": "snv" | ["snv", "snv"]}
    {"derivative": 2}
    {"scaling": "autoscaling"}
    {"variance_threshold": {"threshold": 0.03, "scaler": "minmax"}}        threshold required
    {"ward_clustering": {"n_clusters": 40, "connectivity": true}}          n_clusters required
    {"ward_clustering": <JSON path with n_clusters>}
    {"pca": 5} or {"pca": {"n_components": 5}}

Each step records the same companion-JSON chain entry and file suffix as the corresponding single process, so a chain
gives the same file (e.g. 1350-2500_10.snv.sg5.d1.mc.wc40.pca5) and the same model replay as running the steps one
by one. Parse_chain validates the whole array before anything is computed; Apply_chain runs it.
'''

# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Application package imports
from src.ai4sh.chemometrics import (apply_derivative, apply_scatter_correction, apply_scaling,
                                    apply_decomposition)
from src.ai4sh.filter import apply_filter, apply_multi_filter, _DEFAULT_FILTER_DIR, _FILTER_DEFAULT_FILES
from src.ai4sh.ml_setting import Load_settings, Is_path
from src.ai4sh.provenance import Load_config_json
from src.ai4sh.ml_transformers import Ward_fit

SCATTER_CORRECTIONS = ['l1', 'l2', 'max', 'snv', 'msc']
VARIABLE_SCALINGS   = {'meancentring': 'mc', 'autoscaling': 'as', 'paretoscaling': 'ps', 'poissonscaling': 'poi'}
FILTER_PRESETS      = ['moving-average', 'gauss', 'savitzky-golay', 'lowess']
FILTER_KINDS        = ['moving_average', 'gauss', 'savitzky-golay', 'lowess']
VT_SCALERS          = {'minmax': MinMaxScaler, 'standard': StandardScaler, 'robust': RobustScaler}

METHOD_NAMES = FILTER_PRESETS + SCATTER_CORRECTIONS + ['derivative'] + list(VARIABLE_SCALINGS) + ['pca']
OBJECT_KEYS  = ['filter', 'multi_filter', 'scatter_correction', 'derivative', 'scaling', 'variance_threshold',
                'ward_clustering', 'pca']


def Num_tag(v):
    '''Number as a suffix part: 5 → 5, 1.5 → 1p5, 0.028 → 028 (leading "0." dropped; no dot inside a suffix).'''
    t = np.format_float_positional(float(v), trim='-')
    if t.startswith('0.'):
        t = t[2:]
    return t.replace('.', 'p')


def Filter_suffix(kind, cfg):
    '''File suffix of a single filter from its parameters: ma<kernel length>, gf<sigma>, sg<window_length>
    (+ .d1 / .d2 when deriv > 0), lw<n_points>; e.g. ma5, gf1p5, sg7, sg7.d1, lw7.'''
    fc = cfg[kind]
    if kind == 'moving_average':
        return 'ma%d' % len(fc.get('kernel', []))
    if kind == 'gauss':
        return 'gf' + Num_tag(fc.get('sigma', 0))
    if kind == 'savitzky-golay':
        tag = 'sg%d' % int(fc.get('window_length', 0))
        deriv = int(fc.get('deriv', 0))
        return tag + ('.d%d' % deriv if deriv > 0 else '')
    return 'lw%d' % int(fc.get('n_points', 0))


def _Plain(v):
    '''Undo the attribute wrapping of process parameters: Struct → dict, recursively.'''
    if isinstance(v, (list, tuple)):
        return [_Plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _Plain(x) for k, x in v.items()}
    if hasattr(v, '__dict__') and not isinstance(v, type):
        return {k: _Plain(x) for k, x in vars(v).items()}
    return v


def _Settings(family, value, resolve_path):
    '''Settings dict from a parameter set name, a JSON path (.json optional) or an inline dict.'''
    if isinstance(value, dict):
        return value
    s = str(value).strip()
    if Is_path(s) and not s.lower().endswith('.json'):
        s += '.json'
    # filter presets fall back to default/filter/<name>_default.json when ml_setting is not set up
    default_fpn = str(_DEFAULT_FILTER_DIR / _FILTER_DEFAULT_FILES[s.lower()]) \
        if family == 'filter' and s.lower() in _FILTER_DEFAULT_FILES else None
    cfg = Load_settings(family, s, default_fpn, resolve_path=resolve_path, verbose=0)
    if not cfg:
        raise ValueError('no %s settings in "%s"' % (family, value))
    return cfg


def _Source(value):
    '''How a step's settings were given, recorded as the step parameter (file/set name or "inline").'''
    return 'inline' if isinstance(value, dict) else str(value)


def _Step(item, resolve_path):
    '''One chemometrics_array item → step dict {kind, label, ...settings}; ValueError if invalid.'''

    if isinstance(item, str):
        name = item.strip().lower()
        if name in FILTER_PRESETS:
            return _Step({'filter': name}, resolve_path)
        if name in SCATTER_CORRECTIONS:
            return _Step({'scatter_correction': [name]}, resolve_path)
        if name == 'derivative':
            return _Step({'derivative': 1}, resolve_path)
        if name in VARIABLE_SCALINGS:
            return _Step({'scaling': name}, resolve_path)
        if name == 'pca':
            return _Step({'pca': 5}, resolve_path)
        if name in ('ward_clustering', 'variance_threshold'):
            raise ValueError('"%s" needs its settings: {"%s": {%s}}' % (
                name, name, '"n_clusters": 40' if name == 'ward_clustering' else '"threshold": 0.03'))
        raise ValueError('unknown method "%s"; method names: %s; objects: %s' % (
            item, ', '.join(METHOD_NAMES), ', '.join(OBJECT_KEYS)))

    if not isinstance(item, dict) or len(item) != 1:
        raise ValueError('an item must be a method name or an object with one key, got %s' % (item,))

    key, val = next(iter(item.items()))
    key = key.strip().lower()

    if key == 'filter':
        cfg = _Settings('filter', val, resolve_path)
        kind = next((k for k in FILTER_KINDS if k in cfg), None)
        if kind is None:
            raise ValueError('filter %s: no %s settings' % (val, ', '.join(FILTER_KINDS)))
        return {'kind': 'filter', 'cfg': cfg, 'source': _Source(val), 'suffix': Filter_suffix(kind, cfg)}

    if key == 'multi_filter':
        cfg = _Settings('multi_filter', val, resolve_path)
        if 'multi_filter' not in cfg:
            raise ValueError('multi_filter %s: no "multi_filter" settings' % (val,))
        return {'kind': 'multi_filter', 'cfg': cfg, 'source': _Source(val), 'suffix': 'mf'}

    if key == 'scatter_correction':
        scalers = [str(x).strip().lower() for x in (val if isinstance(val, list) else [val])]
        bad = [x for x in scalers if x not in SCATTER_CORRECTIONS]
        if bad or not scalers:
            raise ValueError('scatter_correction %s: use %s' % (val, ', '.join(SCATTER_CORRECTIONS)))
        return {'kind': 'scatter_correction', 'scaler': scalers, 'suffix': ''.join(scalers)}

    if key == 'derivative':
        n = val.get('derivative', 1) if isinstance(val, dict) else val
        if not str(n).isdigit() or int(n) < 1:
            raise ValueError('derivative %s: use an integer ≥ 1' % (val,))
        return {'kind': 'derivative', 'derivative': int(n), 'suffix': 'd%d' % int(n)}

    if key == 'scaling':
        m = str(val.get('method', '') if isinstance(val, dict) else val).strip().lower()
        if m not in VARIABLE_SCALINGS:
            raise ValueError('scaling %s: use %s' % (val, ', '.join(VARIABLE_SCALINGS)))
        return {'kind': 'scaling', 'method': m, 'suffix': VARIABLE_SCALINGS[m]}

    if key == 'variance_threshold':
        if not isinstance(val, dict) or 'threshold' not in val:
            raise ValueError('variance_threshold needs {"threshold": <value>} (scaler minmax, standard or robust)')
        scaler = str(val.get('scaler', 'minmax')).strip().lower()
        if scaler not in VT_SCALERS:
            raise ValueError('variance_threshold scaler %s: use %s' % (scaler, ', '.join(VT_SCALERS)))
        thr = float(val['threshold'])
        if thr <= 0:
            raise ValueError('variance_threshold threshold must be > 0')
        return {'kind': 'variance_threshold', 'scaler': scaler, 'threshold': thr,
                'suffix': 'vt%s%s' % (scaler[0], Num_tag(thr))}

    if key == 'ward_clustering':
        source = _Source(val)
        if not isinstance(val, dict):
            fpn = resolve_path(str(val)) if resolve_path else str(val)
            if not os.path.exists(fpn):
                raise ValueError('ward_clustering file not found: %s' % fpn)
            cfg = Load_config_json(fpn)
            val = cfg.get('ward_clustering', cfg)
        if int(val.get('n_clusters', 0) or 0) < 1:
            raise ValueError('ward_clustering needs "n_clusters" (≥ 1)')
        n = int(val['n_clusters'])
        return {'kind': 'ward_clustering', 'n_clusters': n, 'connectivity': bool(val.get('connectivity', True)),
                'source': source, 'suffix': 'wc%d' % n}

    if key == 'pca':
        n = val.get('n_components', 5) if isinstance(val, dict) else val
        if not str(n).isdigit() or int(n) < 1:
            raise ValueError('pca %s: n_components must be an integer ≥ 1' % (val,))
        return {'kind': 'pca', 'n_components': int(n), 'suffix': 'pca%d' % int(n)}

    raise ValueError('unknown step "%s"; objects: %s; method names: %s' % (
        key, ', '.join(OBJECT_KEYS), ', '.join(METHOD_NAMES)))


def Parse_chain(raw, resolve_path=None):
    '''chemometrics_array → list of step dicts. Raises ValueError (naming the item) for any invalid item.

    raw: list (items as given in the process file) or a postgres/CSV string of method names.
    resolve_path: function resolving a relative settings path against the project root.
    '''
    raw = _Plain(raw)
    if not raw:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        while len(s) >= 2 and s[0] + s[-1] in ('{}', '[]'):
            s = s[1:-1].strip()
        raw = [x.strip().strip('"') for x in s.split(',') if x.strip().strip('"')]
    steps = []
    for i, item in enumerate(raw):
        try:
            steps.append(_Step(item, resolve_path))
        except (ValueError, FileNotFoundError) as e:
            raise ValueError('chemometrics_array item %d: %s' % (i + 1, e))
    return steps


def _Variance_select(df, cols, scaler, threshold):
    '''Bands kept by the variance threshold, as in select_variance_threshold / VarianceThresholdSelect.'''
    X = VT_SCALERS[scaler]().fit_transform(df[cols].values.astype(float))
    support = VarianceThreshold(threshold=0.0).fit(X).variances_ > threshold
    if not support.any():
        raise ValueError('variance threshold %s removes every band' % threshold)
    keep = [c for c, k in zip(cols, support) if k]
    return df[keep].copy(), keep


def Apply_chain(df, cols, steps):
    '''Run the parsed steps on df[cols]; returns a list of dicts {label, df, cols, suffix, step_info}.

    The first entry is the input ("raw", no suffix/step_info); each next entry is the output of a step fed with the
    output of the step before. step_info is the companion-JSON chain entry of the corresponding single process.
    '''
    out = [{'label': 'raw', 'df': df[cols].copy(), 'cols': list(cols), 'suffix': None, 'step_info': None}]
    cur_df, cur_cols = out[0]['df'], out[0]['cols']

    for st in steps:
        kind, sfx = st['kind'], st['suffix']

        if kind == 'filter':
            d, c, _ = apply_filter(cur_df, cur_cols, st['cfg'])
            info = {'process': 'filter_spectra', 'parameters': {'filter_fp': st['source']}, 'filter': st['cfg']}
            label = 'filter: %s' % sfx

        elif kind == 'multi_filter':
            d, c, _ = apply_multi_filter(cur_df, cur_cols, st['cfg'])
            info = {'process': 'multi_filter_spectra', 'parameters': {'multi_filter_fp': st['source']},
                    'multi_filter': st['cfg']}
            label = 'multi-filter'

        elif kind == 'scatter_correction':
            d, c, _ = apply_scatter_correction(cur_df, cur_cols, {'scaler': st['scaler']})
            info = {'process': 'spectra_scatter_correction', 'parameters': {'scaler': st['scaler']}}
            label = 'scatter correction: %s' % '+'.join(st['scaler'])

        elif kind == 'derivative':
            d, c, _ = apply_derivative(cur_df, cur_cols, {'derivative': st['derivative']})
            info = {'process': 'spectra_derivative', 'parameters': {'derivative': st['derivative'], 'append': False}}
            label = 'derivative %d' % st['derivative']

        elif kind == 'scaling':
            d, c, _ = apply_scaling(cur_df, cur_cols, {'method': st['method']})
            info = {'process': 'spectra_scaling', 'parameters': {'method': st['method']}}
            label = 'variable scaling: %s' % st['method']

        elif kind == 'variance_threshold':
            d, c = _Variance_select(cur_df, cur_cols, st['scaler'], st['threshold'])
            info = {'process': 'select_variance_threshold',
                    'parameters': {'scaler': st['scaler'], 'threshold': st['threshold']}}
            label = 'variance threshold %s (%d of %d bands)' % (st['threshold'], len(c), len(cur_cols))

        elif kind == 'ward_clustering':
            n = st['n_clusters']
            if n >= len(cur_cols):
                raise ValueError('ward_clustering n_clusters %d must be below the number of bands (%d)'
                                 % (n, len(cur_cols)))
            X = cur_df[cur_cols].values.astype(float)
            ward, order = Ward_fit(X, n, st['connectivity'])
            c = ['wc%d' % (i + 1) for i in range(n)]
            d = pd.DataFrame(ward.transform(X)[:, order], columns=c, index=cur_df.index)
            rank = np.empty(n, dtype=int)
            rank[order] = np.arange(n)
            labels = rank[ward.labels_]
            info = {'process': 'ward_clustering',
                    'parameters': {'ward_clustering_fp': st['source'], 'connectivity': st['connectivity'],
                                   'n_clusters_tuned_best': None, 'n_clusters_selected': n,
                                   'n_clusters_applied': n},
                    'cluster_bands': {'wc%d' % (k + 1): [cur_cols[i] for i in np.where(labels == k)[0]]
                                      for k in range(n)}}
            label = 'ward clustering (%d clusters)' % n

        else:  # pca
            d, c, _ = apply_decomposition(cur_df, cur_cols, {'method': 'pca', 'n_components': st['n_components']})
            sfx = 'pca%d' % len(c)
            info = {'process': 'spectra_decomposition', 'parameters': {'method': 'pca', 'n_components': len(c)}}
            label = 'pca (%d components)' % len(c)

        info['_id'] = sfx
        out.append({'label': label, 'df': d, 'cols': list(c), 'suffix': sfx, 'step_info': info})
        cur_df, cur_cols = d, list(c)

    return out
