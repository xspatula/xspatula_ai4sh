'''
Created on 26 April 2026

@author: thomasgumbricht
'''

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    from cubist import Cubist
    _CUBIST_AVAILABLE = True
    # Cubist's _make_data_string uses _escapes() which iterates over
    # pd.Series.astype(str) and expects pure str elements. Pandas 3.0+
    # returns ArrowStringArray where NaN stays as float — patch it once.
    try:
        import cubist._make_data_string as _cubist_mds
        from cubist._make_names_string import _escapes as _cubist_escapes_orig
        def _cubist_escapes_compat(x):
            x = [c if isinstance(c, str) else str(c) for c in x]
            return _cubist_escapes_orig(x)
        _cubist_mds._escapes = _cubist_escapes_compat
    except Exception:
        pass
except ImportError:
    _CUBIST_AVAILABLE = False

from sklearn.linear_model import LinearRegression, TheilSenRegressor, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              mean_absolute_percentage_error, median_absolute_error)

from src.postgres import Get_schema_table
from src.lib.pilot import Get_project_path
from src.ai4sh.machine_learning_preprocess import (
    _resolve_input_parquet, _resolve_single_previous, _load_companion_json,
    _is_spectral_col, _col_to_index, _parse_array_param, _METADATA_COLS,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

_DEFAULT_MODEL_PARAMS_FP = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'regressors', 'regression_model_parameters.json')
_DEFAULT_REGR_SYMBOLS_FP = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'plot', 'regressionmodelsymbols.json')
_DEFAULT_TARGET_SYMBOLS_FP = os.path.join(
    _REPO_ROOT, 'ai4sh', 'default', 'plot', 'targetfeaturesymbols.json')


# ------------------------------------------------------------------ model registry

def _build_regressor(key, hp):
    k = key.lower()
    if k == 'ols':        return LinearRegression(**hp)
    if k == 'theil_sen':  return TheilSenRegressor(**hp)
    if k == 'huber':      return HuberRegressor(**hp)
    if k == 'knn':        return KNeighborsRegressor(**hp)
    if k == 'dectree':    return DecisionTreeRegressor(**hp)
    if k == 'svr':        return SVR(**hp)
    if k == 'randfor':    return RandomForestRegressor(**hp)
    if k == 'mlp':
        return Pipeline([('scl', StandardScaler()), ('clf', MLPRegressor(**hp))])
    if k == 'plsr':       return PLSRegression(**hp)
    if k == 'cubist':
        if not _CUBIST_AVAILABLE:
            return None
        return Cubist(**hp)
    return None


# ------------------------------------------------------------------ metrics

def _ccc(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    mean_obs, mean_pred = obs.mean(), pred.mean()
    cov   = np.mean((obs - mean_obs) * (pred - mean_pred))
    denom = obs.var() + pred.var() + (mean_obs - mean_pred) ** 2
    return float(2 * cov / denom) if denom != 0 else 0.0


def _compute_metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    iq   = float(np.percentile(obs, 75) - np.percentile(obs, 25))
    rpiq = iq / rmse if rmse > 0 else float('inf')
    return {
        'r2':    round(float(r2_score(obs, pred)), 4),
        'rmse':  round(rmse, 4),
        'mae':   round(float(mean_absolute_error(obs, pred)), 4),
        'mape':  round(float(mean_absolute_percentage_error(obs, pred)), 4),
        'medae': round(float(median_absolute_error(obs, pred)), 4),
        'bias':  round(float(np.mean(pred - obs)), 4),
        'rpiq':  round(rpiq, 4),
        'ccc':   round(_ccc(obs, pred), 4),
    }


# ------------------------------------------------------------------ feature importance

def _feature_importance(model, key, feature_names=None):
    k = key.lower()
    if k in ('ols', 'theil_sen', 'huber'):
        return np.abs(model.coef_)
    if k in ('dectree', 'randfor'):
        return model.feature_importances_
    if k == 'cubist':
        return None   # permutation importance too slow for 600+ bands
    if k == 'svr':
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        return None
    if k == 'plsr':
        return np.sum(model.x_loadings_ ** 2, axis=1)
    return None   # knn, mlp — permutation importance only


# ------------------------------------------------------------------ plot helpers

def _plot_scatter(obs, pred, model_key, indicator, mode_label, metrics,
                  regr_sym_D, tgt_sym_D):
    sym    = regr_sym_D.get(model_key, {})
    marker = sym.get('marker', 'o')
    size   = sym.get('size',   40)
    tgt    = tgt_sym_D.get(indicator, {})
    color  = tgt.get('color', 'steelblue')
    alpha  = float(tgt.get('alpha', 0.6))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, pred, marker=marker, s=size, color=color, alpha=alpha,
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
            va='top', ha='left', fontsize=7.5, color='dimgray',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('%s — %s (%s)' % (indicator, model_key, mode_label))
    plt.tight_layout()
    return fig


def _plot_importance(importances, wavelengths, model_key, indicator, label,
                     color='steelblue'):
    imp_arr = np.asarray(importances)
    wl_arr  = np.asarray(wavelengths)

    bar_w = max(1, (wl_arr[1] - wl_arr[0]) * 0.8) if len(wl_arr) > 1 else 5
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(wl_arr, imp_arr, width=bar_w, color=color, alpha=0.75)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Importance')
    ax.set_title('Feature importance [%s]: %s — %s' % (label, model_key, indicator))
    plt.tight_layout()
    return fig


def _plot_permutation(pi_result, wavelengths, model_key, indicator, mode_label,
                      color='steelblue', max_cov=0):
    means = pi_result.importances_mean
    stds  = pi_result.importances_std
    order = np.argsort(means)   # ascending — least important at bottom

    sorted_wl    = [wavelengths[i] for i in order]
    sorted_means = means[order]
    sorted_stds  = stds[order]

    if max_cov > 0 and len(sorted_wl) > max_cov:
        sorted_wl    = sorted_wl[-max_cov:]
        sorted_means = sorted_means[-max_cov:]
        sorted_stds  = sorted_stds[-max_cov:]

    n_bars = len(sorted_wl)
    fig, ax = plt.subplots(figsize=(5, max(4, n_bars * 0.04)))
    ax.barh(range(n_bars), sorted_means, xerr=sorted_stds,
            color=color, alpha=0.75, height=0.8)

    if n_bars <= 40:
        ax.set_yticks(range(n_bars))
        ax.set_yticklabels([str(w) for w in sorted_wl], fontsize=7)
    else:
        ax.set_yticks([])

    ax.set_xlabel('Mean decrease in score')
    ax.set_ylabel('Wavelength (nm)')
    ax.set_title('Permutation importance [%s]: %s — %s' % (mode_label, model_key, indicator))
    plt.tight_layout()
    return fig


def _save_permutation_data(pi_result, wavelengths, model_key, indicator, mode_label, out_dir):
    means = pi_result.importances_mean
    stds  = pi_result.importances_std
    order = np.argsort(means)[::-1]   # descending — most important first

    data = {
        'indicator':        indicator,
        'model':            model_key,
        'mode':             mode_label,
        'wavelengths':      [wavelengths[i] for i in order],
        'importances_mean': [round(float(means[i]), 6) for i in order],
        'importances_std':  [round(float(stds[i]),  6) for i in order],
        'importances_all':  [
            [round(float(v), 6) for v in pi_result.importances[i]]
            for i in order
        ],
    }
    ind_s  = indicator.replace(' ', '-').replace('/', '-')
    out_fp = os.path.join(out_dir, 'permutation_%s_%s.json' % (ind_s, model_key))
    with open(out_fp, 'w') as fh:
        json.dump(data, fh, indent=2)
    return out_fp


# ------------------------------------------------------------------ main class

class Process_regression_model(Get_schema_table):

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose        = process_S.process.verbose
        self.process_S      = process_S
        self.pg_session_C   = pg_session_C
        self.project_root_FP = project_root_FP

    def _Sub_process(self, _):
        if self.process_S.process.process == 'regression_modeling':
            self._Regression_modeling()

    def _resolve_symbols_fp(self, param_val, default_fp):
        s = str(param_val).strip()
        if not s or s.lower() == 'default':
            return default_fp
        if os.path.isabs(s):
            return s
        return Get_project_path(self.project_root_FP, s)

    def _load_json_safe(self, fp, label):
        if not os.path.exists(fp):
            if self.verbose >= 1:
                print('    WARNING: %s file not found: %s' % (label, fp))
            return {}
        with open(fp) as fh:
            return json.load(fh)

    def _print_summary(self, results_D, indicator_cols, regressor_keys):
        modes = []
        for ind in indicator_cols:
            for mode in ('traintest', 'kfold'):
                if mode in results_D.get(ind, {}):
                    if mode not in modes:
                        modes.append(mode)
        for indicator in indicator_cols:
            ind_res = results_D.get(indicator, {})
            if not ind_res:
                continue
            print('\n    ── %s ──' % indicator)
            header = '    %-12s' % ''
            for mode in modes:
                for key in regressor_keys:
                    header += '  %-18s' % ('%s/%s' % (key, mode[:2]))
            print(header)
            for metric in ('r2', 'rmse', 'rpiq'):
                row = '    %-12s' % metric
                for mode in modes:
                    for key in regressor_keys:
                        val = ind_res.get(mode, {}).get(key, {}).get(metric, '-')
                        row += '  %-18s' % (('%.4f' % val) if isinstance(val, float) else val)
                print(row)

    def _Regression_modeling(self):
        p = self.process_S.process.parameters

        project_root_fp = str(p.project_root_fp).strip()
        if not os.path.isabs(project_root_fp):
            project_root_fp = Get_project_path(self.project_root_FP, project_root_fp)
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        dataframe_param    = str(getattr(p, 'dataframe',    'raw')).strip()
        indicator_array    = _parse_array_param(getattr(p, 'indicator_array',  []))
        regressor_array    = _parse_array_param(getattr(p, 'regressor_array',  []))
        model_params_param = str(getattr(p, 'model_parameters_fp', 'default')).strip()
        traintest          = bool(getattr(p, 'traintest', True))
        test_size          = float(getattr(p, 'test_size', 0.3))
        do_kfold           = bool(getattr(p, 'kfold',     False))
        folds              = int(getattr(p, 'folds',      10))
        regr_sym_param     = str(getattr(p, 'regressionmodelsymbols', 'default')).strip()
        tgt_sym_param      = str(getattr(p, 'targetfeaturesymbols',   'default')).strip()
        max_cov            = int(getattr(p, 'n_top_features_in_plot', 0))
        show_scatter       = bool(getattr(p, 'show_observed_predicted',    True))
        save_scatter       = bool(getattr(p, 'save_observed_predicted',    True))
        show_perm          = bool(getattr(p, 'show_permutation_importance', False))
        save_perm          = bool(getattr(p, 'save_permutation_importance', True))
        show_feat          = bool(getattr(p, 'show_feature_importance',    False))
        save_feat          = bool(getattr(p, 'save_feature_importance',    True))

        if not regressor_array:
            print('    ERROR: regressor_array is empty.')
            return
        if not traintest and not do_kfold:
            print('    ERROR: both traintest and kfold are false — nothing to run.')
            return

        if dataframe_param.lower() == 'previous':
            parquet_fp = _resolve_single_previous(project_root_fp)
            if parquet_fp is None:
                return
        else:
            try:
                parquet_fp = _resolve_input_parquet(project_root_fp, dataframe_param)
            except FileNotFoundError as e:
                print('    ERROR: %s' % e)
                return
        df   = pd.read_parquet(parquet_fp)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        companion = _load_companion_json(project_root_fp, stem) or {}

        spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        wavelengths = [_col_to_index(c) for c in spectral_cols]
        col_names   = ['w_%d' % wl for wl in wavelengths]
        X_all       = pd.DataFrame(df[spectral_cols].values.astype(float), columns=col_names)

        all_indicators = [
            c for c in df.columns
            if c not in _METADATA_COLS
            and not _is_spectral_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if indicator_array:
            for col in indicator_array:
                if col not in all_indicators:
                    print('    WARNING: indicator "%s" not in parquet, skipping.' % col)
            indicator_cols = [c for c in indicator_array if c in all_indicators]
        else:
            indicator_cols = all_indicators
        if not indicator_cols:
            print('    ERROR: no valid indicator columns.')
            return

        if model_params_param and model_params_param.lower() not in ('default', 'none', ''):
            mp_fp = (model_params_param if os.path.isabs(model_params_param)
                     else Get_project_path(self.project_root_FP, model_params_param))
        else:
            mp_fp = _DEFAULT_MODEL_PARAMS_FP
        model_params = self._load_json_safe(mp_fp, 'model_parameters')

        regr_sym_fp  = self._resolve_symbols_fp(regr_sym_param, _DEFAULT_REGR_SYMBOLS_FP)
        tgt_sym_fp   = self._resolve_symbols_fp(tgt_sym_param,  _DEFAULT_TARGET_SYMBOLS_FP)
        regr_sym_raw = self._load_json_safe(regr_sym_fp, 'regressionmodelsymbols')
        tgt_sym_raw  = self._load_json_safe(tgt_sym_fp,  'targetfeaturesymbols')
        regr_sym_D   = regr_sym_raw.get('regressionModelSymbols', regr_sym_raw)
        tgt_sym_D    = tgt_sym_raw.get('targetFeatureSymbols',    tgt_sym_raw)

        regressors = {}
        for key in regressor_array:
            hp  = model_params.get(key, {}).get('hyper_parameters', {})
            reg = _build_regressor(key, hp)
            if reg is None:
                if key.lower() == 'cubist' and not _CUBIST_AVAILABLE:
                    print('    WARNING: cubist package not installed — skipping. '
                          'Install with: pip install cubist')
                else:
                    print('    WARNING: unknown regressor "%s", skipping.' % key)
                continue
            regressors[key] = reg
        if not regressors:
            print('    ERROR: no valid regressors after filtering.')
            return

        # mode-specific output subfolders
        models_dir_tt = os.path.join(project_root_fp, 'models',              stem + '_tt')
        models_dir_kf = os.path.join(project_root_fp, 'models',              stem + '_kf')
        plot_dir_tt   = os.path.join(project_root_fp, 'plot',                stem + '_tt')
        plot_dir_kf   = os.path.join(project_root_fp, 'plot',                stem + '_kf')
        cov_dir_tt    = os.path.join(project_root_fp, 'covariate_importance', stem + '_tt')
        cov_dir_kf    = os.path.join(project_root_fp, 'covariate_importance', stem + '_kf')
        if traintest:
            os.makedirs(models_dir_tt, exist_ok=True)
            os.makedirs(plot_dir_tt,   exist_ok=True)
            os.makedirs(cov_dir_tt,    exist_ok=True)
        if do_kfold:
            os.makedirs(models_dir_kf, exist_ok=True)
            os.makedirs(plot_dir_kf,   exist_ok=True)
            os.makedirs(cov_dir_kf,    exist_ok=True)

        def _ind_safe(ind):
            return ind.replace(' ', '-').replace('/', '-')

        results_D = {}

        def _handle_fig(fig, fp, do_show, do_save):
            if do_show:
                plt.figure(fig.number)
                plt.show()
            if do_save:
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                fig.savefig(fp, dpi=150, bbox_inches='tight')
                if self.verbose >= 1:
                    print('    Saved: %s' % fp)
            plt.close(fig)

        for indicator in indicator_cols:
            valid_mask = ~df[indicator].isna()
            n_valid    = int(valid_mask.sum())
            if n_valid < 10:
                print('    WARNING: too few samples for "%s" (%d), skipping.' % (indicator, n_valid))
                continue

            X = X_all[valid_mask]
            y = df.loc[valid_mask, indicator].values.astype(float)
            results_D[indicator] = {}
            ind_s     = _ind_safe(indicator)
            ind_color = tgt_sym_D.get(indicator, {}).get('color', 'steelblue')

            if self.verbose >= 1:
                print('    Indicator: %s  (%d samples, %d bands)' % (indicator, n_valid, X.shape[1]))

            # split once per indicator — deterministic with fixed random_state
            if traintest:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42, shuffle=True)

            for model_key, model_proto in regressors.items():

                if self.verbose >= 1:
                    print('      Model: %s' % model_key)

                # ---- train / test
                if traintest:
                    model_tt = clone(model_proto)
                    model_tt.fit(X_tr, y_tr)
                    pred_tt  = model_tt.predict(X_te).ravel()

                    metrics_tt = _compute_metrics(y_te, pred_tt)
                    metrics_tt['n_train'] = len(y_tr)
                    metrics_tt['n_test']  = len(y_te)

                    jl_fp = os.path.join(models_dir_tt,
                                         '%s_%s_%s_tt.joblib' % (stem, ind_s, model_key))
                    joblib.dump(model_tt, jl_fp)
                    metrics_tt['model_fp'] = jl_fp

                    if show_scatter or save_scatter:
                        fig = _plot_scatter(y_te, pred_tt, model_key, indicator,
                                            'train/test', metrics_tt, regr_sym_D, tgt_sym_D)
                        _handle_fig(fig,
                                    os.path.join(plot_dir_tt,
                                                 'scatter_%s_%s_tt.png' % (ind_s, model_key)),
                                    show_scatter, save_scatter)

                    if show_feat or save_feat:
                        imp = _feature_importance(model_tt, model_key, col_names)
                        if imp is not None:
                            fig2 = _plot_importance(imp, wavelengths, model_key, indicator, 'tt',
                                                    color=ind_color)
                            _handle_fig(fig2,
                                        os.path.join(plot_dir_tt,
                                                     'importance_%s_%s_tt.png' % (ind_s, model_key)),
                                        show_feat, save_feat)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        pi = sk_permutation_importance(
                            model_tt, X_te, y_te, n_repeats=10, random_state=42)
                        fig3 = _plot_permutation(pi, wavelengths, model_key, indicator, 'train/test',
                                                 color=ind_color, max_cov=max_cov)
                        _handle_fig(fig3,
                                    os.path.join(plot_dir_tt,
                                                 'permutation_%s_%s_tt.png' % (ind_s, model_key)),
                                    show_perm, save_perm)
                        _save_permutation_data(pi, wavelengths, model_key, indicator, 'tt', cov_dir_tt)

                    results_D[indicator].setdefault('traintest', {})[model_key] = metrics_tt
                    if self.verbose >= 1:
                        print('        [tt] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                            metrics_tt['r2'], metrics_tt['rmse'], metrics_tt['rpiq']))

                # ---- k-fold
                if do_kfold:
                    model_kf = clone(model_proto)
                    pred_kf  = cross_val_predict(model_kf, X, y, cv=folds)
                    metrics_kf = _compute_metrics(y, pred_kf)
                    metrics_kf['n'] = len(y)

                    _scoring_map = [
                        ('r2',                                'r2'),
                        ('neg_root_mean_squared_error',        'rmse'),
                        ('neg_mean_absolute_error',            'mae'),
                        ('neg_mean_absolute_percentage_error', 'mape'),
                        ('neg_median_absolute_error',          'medae'),
                    ]
                    for scoring, key_out in _scoring_map:
                        try:
                            cv_scores = cross_val_score(
                                clone(model_proto), X, y, cv=folds, scoring=scoring)
                            metrics_kf[key_out + '_folded_mean'] = round(
                                float(abs(cv_scores.mean())), 4)
                            metrics_kf[key_out + '_folded_std'] = round(
                                float(cv_scores.std()), 4)
                        except Exception:
                            pass

                    model_kf.fit(X, y)
                    jl_fp = os.path.join(models_dir_kf,
                                         '%s_%s_%s_kf.joblib' % (stem, ind_s, model_key))
                    joblib.dump(model_kf, jl_fp)
                    metrics_kf['model_fp'] = jl_fp

                    if show_scatter or save_scatter:
                        fig = _plot_scatter(y, pred_kf, model_key, indicator,
                                            'k-fold', metrics_kf, regr_sym_D, tgt_sym_D)
                        _handle_fig(fig,
                                    os.path.join(plot_dir_kf,
                                                 'scatter_%s_%s_kf.png' % (ind_s, model_key)),
                                    show_scatter, save_scatter)

                    if show_feat or save_feat:
                        imp = _feature_importance(model_kf, model_key, col_names)
                        if imp is not None:
                            fig2 = _plot_importance(imp, wavelengths, model_key, indicator, 'kf',
                                                    color=ind_color)
                            _handle_fig(fig2,
                                        os.path.join(plot_dir_kf,
                                                     'importance_%s_%s_kf.png' % (ind_s, model_key)),
                                        show_feat, save_feat)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        pi = sk_permutation_importance(
                            model_kf, X, y, n_repeats=10, random_state=42)
                        fig3 = _plot_permutation(pi, wavelengths, model_key, indicator, 'k-fold',
                                                 color=ind_color, max_cov=max_cov)
                        _handle_fig(fig3,
                                    os.path.join(plot_dir_kf,
                                                 'permutation_%s_%s_kf.png' % (ind_s, model_key)),
                                    show_perm, save_perm)
                        _save_permutation_data(pi, wavelengths, model_key, indicator, 'kf', cov_dir_kf)

                    results_D[indicator].setdefault('kfold', {})[model_key] = metrics_kf
                    if self.verbose >= 1:
                        print('        [kf] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                            metrics_kf['r2'], metrics_kf['rmse'], metrics_kf['rpiq']))

        if results_D:
            self._print_summary(results_D, indicator_cols, list(regressors.keys()))

        out_json = {
            'source_parquet':      os.path.basename(parquet_fp),
            'preprocessing_chain': companion.get('preprocessing_chain', []),
            'results':             results_D,
        }
        result_fp = os.path.join(project_root_fp, stem + '_regression.json')
        with open(result_fp, 'w') as fh:
            json.dump(out_json, fh, indent=2)
        print('    Results saved: %s' % result_fp)
