'''
Created on 5 November 2026

@author: thomasgumbricht

Plots of regression_modeling: observed/predicted scatter, feature importance and permutation importance, drawn
either as single figures (one per indicator, regressor and mode — the saved PNGs) or composed into one grid figure
from a plot layout (ml_setting family plot_layout).

A plot layout (parameter set content, or a user's JSON file of the same shape):

    {
      "columns": ["scatter", "importance", "permutation"],   panel types of one row, in order
      "wrap": 0,                       0: one row per indicator × regressor; n > 0: panels flow, n units per row
      "group_by": "indicator",         indicator | regressor — the outer loop of the rows
      "draw": "indicator",             indicator: a grid per indicator, as soon as it is done; run: one grid at the end
      "panel_size": [3, 3],            inches of one panel unit
      "max_units_per_row": 4,          a wider row is split, first into train/test and k-fold
      "importance_width": 2,           units of a feature importance panel (scatter and permutation are 1)
      "max_permutation_bars": 6        the most important features shown in a permutation panel
    }

Widths are in units: a scatter or permutation panel is 1 unit, a feature importance panel importance_width units
(2: so wrap 4 fits two importance panels, or one importance + two others, per row).

A file only needs the keys it changes. Optional "base": a preset name (e.g. "importance") the other keys change. A
file without "base" and without "columns"/"wrap" changes the preset "default" would pick; a file with "columns" or
"wrap" but no "base" starts from the built-in defaults above.

Panel types: scatter, importance, permutation (every mode run) or with the mode: scatter_tt, scatter_kf,
importance_tt, … When every column is a bare type, the row is ordered mode by mode (all train/test panels, then
the k-fold panels); otherwise the columns are taken as listed.

wrap > 0 (the scatter-<n> presets): a new indicator starts a new row, unless there is only one regressor; then the
indicators flow, n units per row (group_by regressor: the same with regressors and indicators swapped).

"default" picks a preset from the process settings (Default_layout_name); "none" shows the single figures.
'''

# Standard library imports
import os

import numpy as np
import matplotlib.pyplot as plt

from src.ai4sh.ml_setting import Load_settings

PANEL_TYPES = ('scatter', 'importance', 'permutation')

MODES = ('tt', 'kf')

MODE_LABEL = {'tt': 'train/test', 'kf': 'k-fold'}

LAYOUT_DEFAULTS = {
    'columns':              ['scatter'],
    'wrap':                 0,
    'group_by':             'indicator',
    'draw':                 'indicator',
    'panel_size':           [3, 3],
    'max_units_per_row':    4,
    'importance_width':     2,
    'max_permutation_bars': 6,
}


# ------------------------------------------------------------------ panels drawn into given axes

def _Covariate_labels(features):
    '''Short tick labels for covariate features ("cv:<covariate>=<class>[|<class>…]" for one-hot classes, "cv:<name>"
    for numeric covariates): a group of classes shows its first class and the number of others ("clay loam +4"). With
    one nominal covariate its name is returned apart (for the title) and left out of the labels.
    Returns (labels for every feature, common covariate name or '').'''
    names = [f[len('cv:'):] if f.startswith('cv:') else f for f in features if isinstance(f, str)]
    covs = {n.partition('=')[0] for n in names if '=' in n}
    common = covs.pop() if len(covs) == 1 and all('=' in n for n in names) else ''
    labels = []
    for f in features:
        if not isinstance(f, str):
            labels.append(str(f))
            continue
        n = f[len('cv:'):] if f.startswith('cv:') else f
        cov, eq, cls = n.partition('=')
        if eq:
            classes = cls.split('|')
            cls = classes[0] + (' +%d' % (len(classes) - 1) if len(classes) > 1 else '')
            n = cls if common else '%s=%s' % (cov, cls)
        labels.append(n)
    return labels, common


def _Feature_axis_label(features):
    '''Wavelength (nm) for bands; Component for the numbered components of a decomposition (PCA), which come out
    of the preprocessing as small integers (1, 2, …) — wavelengths are hundreds of nm.'''
    numeric = [f for f in features if not isinstance(f, str)]
    if numeric and max(float(f) for f in numeric) < 100:
        return 'Component'
    return 'Wavelength (nm)'


def Draw_scatter(ax, obs, pred, model_key, indicator, mode_label, metrics, regr_sym_D, tgt_sym_D, compact=False):
    sym    = regr_sym_D.get(model_key, {})
    marker = sym.get('marker', 'o')
    size   = sym.get('size',   40)
    tgt    = tgt_sym_D.get(indicator, {})
    color  = tgt.get('color', 'steelblue')
    alpha  = float(tgt.get('alpha', 0.6))
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)

    ax.scatter(obs, pred, marker=marker, s=size * (0.5 if compact else 1), color=color, alpha=alpha,
               edgecolors='black', linewidths=0.3)

    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)

    if 'n_train' in metrics:
        n_str = 'n_train=%d  n_test=%d' % (metrics['n_train'], metrics['n_test'])
    else:
        n_str = 'n=%d' % metrics.get('n', len(obs))
    txt = ('r²=%(r2).3f\nRMSE=%(rmse).3f\nRPIQ=%(rpiq).3f\n%(n)s'
           % {'r2': metrics['r2'], 'rmse': metrics['rmse'],
              'rpiq': metrics['rpiq'], 'n': n_str})
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=6.5 if compact else 7.5, color='dimgray',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    if compact:
        ax.set_xlabel('Observed', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title('%s — %s (%s)' % (indicator, model_key, mode_label), fontsize=9)
    else:
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.set_title('%s — %s (%s)' % (indicator, model_key, mode_label))


def Draw_importance(ax, ax_cv, importances, wavelengths, model_key, indicator, label, color='steelblue',
                    compact=False):
    '''Bars per spectral feature in ax; covariates (cv:... names), if any, in ax_cv.'''
    imp_arr = np.asarray(importances)
    is_cov  = np.array([isinstance(w, str) for w in wavelengths], dtype=bool)
    wl_arr  = np.asarray([w for w in wavelengths if not isinstance(w, str)], dtype=float)
    spec_imp = imp_arr[~is_cov]
    fs = 7 if compact else None

    if ax_cv is not None and is_cov.any():
        cv_names, common = _Covariate_labels([w for w in wavelengths if isinstance(w, str)])
        ax_cv.barh(range(len(cv_names)), imp_arr[is_cov], color=color, alpha=0.75)
        ax_cv.set_yticks(range(len(cv_names)))
        ax_cv.set_yticklabels(cv_names, fontsize=7)
        ax_cv.set_xlabel('Importance', fontsize=fs)
        ax_cv.set_title(common or 'Covariates', fontsize=8 if compact else 9)
        if compact:
            ax_cv.tick_params(axis='x', labelsize=7)

    if len(wl_arr):
        bar_w = max(1, (wl_arr[1] - wl_arr[0]) * 0.8) if len(wl_arr) > 1 else 5
        ax.bar(wl_arr, spec_imp, width=bar_w, color=color, alpha=0.75)
    ax.set_xlabel(_Feature_axis_label(wavelengths), fontsize=fs)
    ax.set_ylabel('Importance', fontsize=fs)
    if compact:
        ax.tick_params(labelsize=7)
        ax.set_title('Importance [%s]: %s — %s' % (label, model_key, indicator), fontsize=9)
    else:
        ax.set_title('Feature importance [%s]: %s — %s' % (label, model_key, indicator))


def _Top_permutation(means, stds, wavelengths, max_bars):
    order = np.argsort(means)   # ascending — least important at bottom
    sorted_wl    = [wavelengths[i] for i in order]
    sorted_means = np.asarray(means)[order]
    sorted_stds  = np.asarray(stds)[order]
    if max_bars > 0 and len(sorted_wl) > max_bars:
        sorted_wl    = sorted_wl[-max_bars:]
        sorted_means = sorted_means[-max_bars:]
        sorted_stds  = sorted_stds[-max_bars:]
    return sorted_wl, sorted_means, sorted_stds


def Draw_permutation(ax, means, stds, wavelengths, model_key, indicator, mode_label, color='steelblue',
                     max_bars=0, compact=False):
    sorted_wl, sorted_means, sorted_stds = _Top_permutation(means, stds, wavelengths, max_bars)
    n_bars = len(sorted_wl)
    ax.barh(range(n_bars), sorted_means, xerr=sorted_stds,
            color=color, alpha=0.75, height=0.8)

    if n_bars <= 40:
        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(_Covariate_labels(sorted_wl)[0] if any(isinstance(w, str) for w in sorted_wl)
                           else [str(w) for w in sorted_wl], fontsize=7)
    else:
        ax.set_yticks([])

    fs = 7 if compact else None
    ax.set_xlabel('Mean decrease in score', fontsize=fs)
    ax.set_ylabel('Feature' if any(isinstance(w, str) for w in sorted_wl) else _Feature_axis_label(sorted_wl),
                  fontsize=fs)
    if compact:
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title('Permutation [%s]: %s — %s' % (mode_label, model_key, indicator), fontsize=9)
    else:
        ax.set_title('Permutation importance [%s]: %s — %s' % (mode_label, model_key, indicator))


# ------------------------------------------------------------------ single figures (saved PNGs)

def Plot_scatter(obs, pred, model_key, indicator, mode_label, metrics, regr_sym_D, tgt_sym_D):
    fig, ax = plt.subplots(figsize=(5, 5))
    Draw_scatter(ax, obs, pred, model_key, indicator, mode_label, metrics, regr_sym_D, tgt_sym_D)
    plt.tight_layout()
    return fig


def Plot_importance(importances, wavelengths, model_key, indicator, label, color='steelblue'):
    '''Bars per spectral feature; covariates (cv:... names), if any, in a second panel.'''
    if any(isinstance(w, str) for w in wavelengths):
        fig, (ax, ax_cv) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax_cv = None
    Draw_importance(ax, ax_cv, importances, wavelengths, model_key, indicator, label, color)
    plt.tight_layout()
    return fig


def Plot_permutation(pi_result, wavelengths, model_key, indicator, mode_label, color='steelblue', max_cov=0):
    n_bars = len(_Top_permutation(pi_result.importances_mean, pi_result.importances_std, wavelengths, max_cov)[0])
    fig, ax = plt.subplots(figsize=(5, max(4, n_bars * 0.04)))
    Draw_permutation(ax, pi_result.importances_mean, pi_result.importances_std, wavelengths, model_key, indicator,
                     mode_label, color, max_cov)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ plot layout

def Default_layout_name(show_scatter, show_feat, show_perm, modes, n_indicators, n_regressors):
    '''The preset "default" stands for, from what the process shows; None if nothing is shown.'''
    if show_feat or show_perm:
        return 'importance'
    if not show_scatter:
        return None
    if len(modes) > 1:
        return 'scatter-tt-kf'
    n = n_regressors if n_regressors > 1 else n_indicators
    return 'scatter-%d' % max(1, min(4, n))


_LAYOUT_KEYS = set(LAYOUT_DEFAULTS) | {'base'}


def _Layout_raw(raw, name):
    '''The layout keys of raw: a parameter set's content or a user's file, or from the fallback file holding every
    preset by name the one called name. Raises ValueError.'''
    if not isinstance(raw, dict):
        raise ValueError('plot layout "%s" is not a JSON object' % name)
    if set(raw) & _LAYOUT_KEYS:
        return raw
    key = os.path.splitext(os.path.basename(str(name)))[0].lower()
    if not isinstance(raw.get(key), dict):
        raise ValueError('plot layout "%s" not found (no layout keys in it: %s)' % (name, ', '.join(sorted(_LAYOUT_KEYS))))
    return raw[key]


def Layout_content(raw, name, base=None):
    '''The layout raw (see _Layout_raw) on top of base (a completed layout) or the built-in defaults, checked.
    Raises ValueError.'''
    raw = _Layout_raw(raw, name)
    layout = dict(base or LAYOUT_DEFAULTS, **{k: v for k, v in raw.items() if k != 'base'})

    columns = layout['columns']
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        ptype, _, mode = str(col).partition('_')
        if ptype not in PANEL_TYPES or (mode and mode not in MODES):
            raise ValueError('plot layout "%s": unknown panel type "%s" (%s, optionally _tt or _kf)'
                             % (name, col, ', '.join(PANEL_TYPES)))
    layout['columns'] = [str(c) for c in columns]
    if layout['group_by'] not in ('indicator', 'regressor'):
        raise ValueError('plot layout "%s": group_by must be indicator or regressor' % name)
    if layout['draw'] not in ('indicator', 'run'):
        raise ValueError('plot layout "%s": draw must be indicator or run' % name)
    layout['wrap'] = int(layout['wrap'] or 0)
    layout['max_units_per_row'] = max(1, int(layout['max_units_per_row']))
    layout['importance_width'] = max(1, int(layout['importance_width']))
    layout['max_permutation_bars'] = int(layout['max_permutation_bars'])
    layout['name'] = os.path.splitext(os.path.basename(str(name)))[0]
    return layout


def Load_layout(param, default_name, default_fpn, resolve_path, verbose=1):
    '''The plot layout named by param: "none" → None, "default" → default_name, a parameter set name or a JSON
    path. Raises ValueError / FileNotFoundError. A preset missing from the database (setup not rerun) is taken
    from the fallback file.'''
    s = str(param or 'default').strip()
    if s.lower() in ('none', 'no'):
        return None
    name = default_name if s.lower() == 'default' else s
    if name is None:
        return None
    raw = _Layout_raw(_Load_layout_raw(name, default_fpn, resolve_path, verbose), name)

    # A partial layout changes a preset: the one named by "base", else the one "default" picks
    base_name = raw.get('base') or (default_name if 'columns' not in raw and 'wrap' not in raw else None)
    base = None
    if base_name:
        base = Layout_content(_Load_layout_raw(base_name, default_fpn, resolve_path, verbose), base_name)
    layout = Layout_content(raw, name, base)
    if base_name:
        layout['base'] = base_name
    return layout


def _Load_layout_raw(name, default_fpn, resolve_path, verbose):
    '''The content of the layout parameter set or file name; a preset missing from the database is taken from the
    fallback file.'''
    try:
        return Load_settings('plot_layout', name, default_fpn, name, resolve_path, verbose)
    except ValueError:
        if os.path.exists(default_fpn) and name.lower() in Load_settings('plot_layout', default_fpn):
            if verbose >= 1:
                print('    ⚠️  plot layout "%s" not in ml_setting.parameter_set (rerun setup_db) — taken from %s'
                      % (name, os.path.basename(default_fpn)))
            return Load_settings('plot_layout', default_fpn)
        raise


def In_layout(layout, ptype, mode):
    '''True if the layout has a column for this panel.'''
    return layout is not None and any(c == ptype or c == '%s_%s' % (ptype, mode) for c in layout['columns'])


def _Expanded_columns(columns, modes):
    '''(type, mode) per panel of one row.'''
    if all('_' not in c for c in columns):
        return [(c, m) for m in modes for c in columns]
    out = []
    for c in columns:
        ptype, _, mode = c.partition('_')
        out += [(ptype, mode)] if mode else [(ptype, m) for m in modes]
    return [(t, m) for t, m in out if m in modes]


def _Width(panel, layout):
    return layout['importance_width'] if panel['type'] == 'importance' else 1


def _Pack(panels, layout, max_units):
    '''Greedy split of a panel sequence into rows of at most max_units units.'''
    rows, row, used = [], [], 0
    for panel in panels:
        w = _Width(panel, layout)
        if row and used + w > max_units:
            rows.append(row)
            row, used = [], 0
        row.append(panel)
        used += w
    if row:
        rows.append(row)
    return rows


def Grid_rows(panels, layout, indicators, regressors):
    '''Rows of panels for one grid figure.'''
    modes = [m for m in MODES if any(p['mode'] == m for p in panels)]
    expanded = _Expanded_columns(layout['columns'], modes)
    by_key = {(p['indicator'], p['regressor'], p['type'], p['mode']): p for p in panels}
    if layout['group_by'] == 'regressor':
        outer, inner = regressors, indicators
        key = lambda o, i, t, m: (i, o, t, m)
    else:
        outer, inner = indicators, regressors
        key = lambda o, i, t, m: (o, i, t, m)
    inner = [i for i in inner if any(key(o, i, t, m) in by_key for o in outer for t, m in expanded)]

    rows = []
    if layout['wrap'] > 0:
        flow = []
        for o in outer:
            group = [by_key[key(o, i, t, m)] for i in inner for t, m in expanded if key(o, i, t, m) in by_key]
            if len(inner) > 1:
                rows += _Pack(group, layout, layout['wrap'])
            else:
                flow += group
        rows += _Pack(flow, layout, layout['wrap'])
        return rows

    max_units = layout['max_units_per_row']
    for o in outer:
        for i in inner:
            unit = [by_key[key(o, i, t, m)] for t, m in expanded if key(o, i, t, m) in by_key]
            if not unit:
                continue
            if sum(_Width(p, layout) for p in unit) > max_units and len(modes) > 1:
                for m in modes:
                    rows += _Pack([p for p in unit if p['mode'] == m], layout, max_units)
            else:
                rows += _Pack(unit, layout, max_units)
    return [r for r in rows if r]


def Plot_grid(panels, layout, indicators, regressors, regr_sym_D, tgt_sym_D, title=''):
    '''One figure with the panels arranged by the layout, or None if there is nothing to draw.'''
    rows = Grid_rows(panels, layout, indicators, regressors)
    if not rows:
        return None
    n_units = layout['wrap'] or max(sum(_Width(p, layout) for p in r) for r in rows)
    pw, ph = layout['panel_size']
    fig = plt.figure(figsize=(n_units * pw, len(rows) * ph), layout='constrained')
    gs = fig.add_gridspec(len(rows), n_units)

    for r, row in enumerate(rows):
        c = 0
        for p in row:
            w = _Width(p, layout)
            cell = gs[r, c:c + w]
            c += w
            label = p.get('label') or MODE_LABEL[p['mode']]
            if p['type'] == 'scatter':
                Draw_scatter(fig.add_subplot(cell), p['obs'], p['pred'], p['regressor'], p['indicator'], label,
                             p['metrics'], regr_sym_D, tgt_sym_D, compact=True)
            elif p['type'] == 'importance':
                if any(isinstance(f, str) for f in p['features']):
                    sub = cell.subgridspec(1, 2, width_ratios=[3, 1])
                    ax, ax_cv = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
                else:
                    ax, ax_cv = fig.add_subplot(cell), None
                Draw_importance(ax, ax_cv, p['importances'], p['features'], p['regressor'], p['indicator'],
                                p['mode'], p['color'], compact=True)
            else:
                Draw_permutation(fig.add_subplot(cell), p['means'], p['stds'], p['features'], p['regressor'],
                                 p['indicator'], p['mode'], p['color'], layout['max_permutation_bars'], compact=True)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
