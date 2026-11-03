'''
Created on 26 April 2026

@author: thomasgumbricht
'''

import os
import re
import json
import hashlib

from datetime import datetime

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              mean_absolute_percentage_error, median_absolute_error)

from src.postgres import Get_schema_table
from src.lib.pilot import Get_project_path
from src.ai4sh.machine_learning_preprocess import (
    _resolve_input_parquet, _resolve_single_previous, _load_companion_json,
    _is_spectral_col, _col_to_index, _parse_array_param, _is_metadata_col,
)
from src.ai4sh.parquet_units import Read_parquet_with_units, Find_selected_parquet
from src.ai4sh.chemometrics import (_load_transform_config, _DEFAULT_TRANSFORMATION_FPN,
                                     _DEFAULT_STANDARDISE_FPN)
from src.ai4sh.ml_transformers import (Chain_to_steps, Column_index, CovariateBlocks, Group_classes,
                                       Holdout_test_mask, IndicatorTransformer, Indicator_target_settings,
                                       Model_blocks, Parse_class_grouping, RegressorFrame, Target_domain_mask)
from src.ai4sh.spectral_columns import Signal_type_of
from src.ai4sh.project_paths import Default_fpn
from src.ai4sh.provenance import Provenance_snapshot, File_sha256
from src.ai4sh.ml_setting import Load_settings, Is_path

_DEFAULT_MODEL_PARAMS_FP = Default_fpn('regressors', 'regression_model_parameters.json')
_DEFAULT_REGR_SYMBOLS_FP = Default_fpn('plot', 'regressionmodelsymbols.json')
_DEFAULT_TARGET_SYMBOLS_FP = Default_fpn('plot', 'targetfeaturesymbols.json')


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
    '''Bars per spectral feature; covariates (cv:... names), if any, in a second panel.'''
    imp_arr = np.asarray(importances)
    is_cov  = np.array([isinstance(w, str) for w in wavelengths], dtype=bool)
    wl_arr  = np.asarray([w for w in wavelengths if not isinstance(w, str)], dtype=float)
    spec_imp = imp_arr[~is_cov]

    if is_cov.any():
        fig, (ax, ax_cv) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [3, 1]})
        cv_names = [w[len('cv:'):] for w in wavelengths if isinstance(w, str)]
        ax_cv.barh(range(len(cv_names)), imp_arr[is_cov], color=color, alpha=0.75)
        ax_cv.set_yticks(range(len(cv_names)))
        ax_cv.set_yticklabels(cv_names, fontsize=7)
        ax_cv.set_xlabel('Importance')
        ax_cv.set_title('Covariates', fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(9, 3))

    if len(wl_arr):
        bar_w = max(1, (wl_arr[1] - wl_arr[0]) * 0.8) if len(wl_arr) > 1 else 5
        ax.bar(wl_arr, spec_imp, width=bar_w, color=color, alpha=0.75)
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
    ax.set_ylabel('Feature' if any(isinstance(w, str) for w in sorted_wl) else 'Wavelength (nm)')
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


# ------------------------------------------------------------------ fitted model helpers

def Manifest_fp(model_fp):
    '''Path of the manifest JSON saved next to a model: <name>.joblib → <name>.manifest.json.'''
    return os.path.splitext(model_fp)[0] + '.manifest.json'


def _Fitted_target(model):
    '''Fitted indicator transformation of a model (lambda, mean, std ...) as a JSON-ready dict.'''
    if isinstance(model, TransformedTargetRegressor):
        return model.transformer_.Describe()
    return {'transform': 'linear', 'standardise': False}


class _Fitted_tail(RegressorMixin, BaseEstimator):
    '''The fitted regressor (+ inverse indicator transformation) of a model, applied to preprocessed features.

    Feature and permutation importance are reported for the features that reach the regressor (bands, pc, wc),
    in the native unit of the indicator.
    '''

    def __init__(self, model):
        self.model = model
        self.pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
        self.regressor = self.pipe[-1]

    def Preprocess(self, X):
        return self.pipe[:-2].transform(X)

    def Features(self, X):
        return [Column_index(c) for c in self.Preprocess(X.iloc[:2]).columns]

    def fit(self, X, y=None):
        return self

    def predict(self, Xt):
        pred = np.ravel(self.regressor.predict(self.pipe[-2].transform(Xt)))
        if self.pipe is not self.model:
            pred = self.model.transformer_.inverse_transform(pred.reshape(-1, 1)).ravel()
        return pred


# ------------------------------------------------------------------ covariates

def _Companion_covariates(companion):
    '''Covariate columns of a dataset: column_blocks (compiled dataset) or covariate_array (selection).'''
    return list((companion.get('column_blocks') or {}).get('covariate') or companion.get('covariate_array') or [])


def _Covariate_tag(covariate_cols):
    '''Part of the model/plot names telling which covariates were used: _cv-<labels> (hashed if long).'''
    tag = '-'.join(re.sub(r'[^0-9a-z]+', '', c.lower()) for c in covariate_cols)
    if len(tag) > 40:
        tag = hashlib.md5('|'.join(covariate_cols).encode()).hexdigest()[:8]
    return '_cv-' + tag


def _Fitted_covariates(model, covariates_D):
    '''Covariates of a fitted model for its manifest: numeric labels, nominal labels with the classes (after
    grouping) seen in training, and the class grouping.'''
    blocks = Model_blocks(model)
    if blocks is None:
        return {}
    return dict(covariates_D, levels=blocks.levels_)


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

    def _Settings_fp(self, param):
        '''A relative settings file path resolved against the project root; anything else (none, default,
        parameter set name, absolute path) as is.'''
        s = str(param).strip()
        if not Is_path(s) or os.path.isabs(s):
            return s
        return Get_project_path(self.project_root_FP, s)

    def _Settings(self, family, param, default_fpn, default_name='standard'):
        '''Settings dict of a parameter set / file ({} for none; missing file → warning and {}).'''
        try:
            return Load_settings(family, param, default_fpn, default_name, self._Settings_fp, self.verbose) or {}
        except FileNotFoundError as e:
            if self.verbose >= 1:
                print('    WARNING: %s' % e)
            return {}

    def _Source_parquet(self, project_root_fp, companion):
        '''The unprocessed parquet a preprocessing chain starts from, or None (error printed).'''
        if companion.get('source_parquet'):
            fp = os.path.join(project_root_fp, companion['source_parquet'])
        else:
            fp = Find_selected_parquet(project_root_fp)[0]
            print('    ⚠️  source_parquet not recorded in the companion JSON (older chain) — using %s'
                  % os.path.basename(fp))
        if not os.path.exists(fp):
            print('    ERROR: source parquet of the preprocessing chain not found: %s' % fp)
            return None
        return fp

    def _Rows_aligned(self, df, src_df, source_fp):
        '''True if the source parquet has the same samples, in the same order, as the input parquet.'''
        if len(df) != len(src_df):
            print('    ERROR: %s has %d rows, the input parquet %d — can not refit the chain.'
                  % (os.path.basename(source_fp), len(src_df), len(df)))
            return False
        key = next((k for k in ('sample_id', 'sample_name') if k in df.columns and k in src_df.columns), None)
        if key and not (df[key].astype(str).values == src_df[key].astype(str).values).all():
            print('    ERROR: %s and the input parquet differ in %s order — can not refit the chain.'
                  % (os.path.basename(source_fp), key))
            return False
        return True

    def _Covariates(self, df, units_D, companion, covariate_array, class_grouping):
        '''(numeric, nominal) covariate columns for covariate_array ([] = none, ["all"] = every covariate of the
        dataset); None (error printed) for a covariate not in the dataset or an invalid class_grouping.'''
        if [c.lower() for c in covariate_array] == ['all']:
            covariate_cols = [c for c in _Companion_covariates(companion) if c in df.columns]
            if not covariate_cols:
                print('    ⚠️  covariate_array "all": the dataset has no covariates — spectra only')
        else:
            covariate_cols = list(covariate_array)
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            print('    ERROR: covariate(s) not in the dataset: %s' % ', '.join(missing))
            return None
        spectral = [c for c in covariate_cols if _is_spectral_col(c)]
        if spectral:
            print('    ERROR: spectral columns can not be covariates: %s' % ', '.join(spectral))
            return None
        nominal = [c for c in covariate_cols
                   if units_D.get(c) == 'class' or not pd.api.types.is_numeric_dtype(df[c])]
        numeric = [c for c in covariate_cols if c not in nominal]
        for label, class_D in class_grouping.items():
            if label not in nominal:
                print('    ERROR: class_grouping for "%s", which is not a nominal covariate of this model' % label)
                return None
            classes = set(Group_classes(df[label].values).dropna())
            ungrouped = sorted(classes - set(class_D))
            if ungrouped:
                print('    ERROR: class_grouping for "%s" does not place the class(es): %s'
                      % (label, ', '.join(ungrouped)))
                return None
        return numeric, nominal

    def _Drop_unseen_classes(self, df, tr_mask, te_mask, nominal_cov, class_grouping, indicator):
        '''Test mask without the samples whose class (after grouping) has no training sample.'''
        for label in nominal_cov:
            classes = Group_classes(df[label].values, class_grouping.get(label)).values
            seen = set(classes[tr_mask])
            unseen = te_mask & np.array([c not in seen for c in classes], dtype=bool)
            if unseen.any():
                print('    ⚠️  %s: %d test sample(s) with a %s class not in the training samples (%s), skipped'
                      % (indicator, int(unseen.sum()), label, ', '.join(sorted(set(classes[unseen])))))
                te_mask = te_mask & ~unseen
        return te_mask

    def _Warn_rare_classes(self, df, valid_mask, nominal_cov, class_grouping, folds, indicator):
        '''k-fold: a class with fewer samples than folds is missing from some training folds (all-zero encoding).'''
        for label in nominal_cov:
            counts = Group_classes(df.loc[valid_mask, label].values, class_grouping.get(label)).value_counts()
            rare = counts[counts < folds]
            if len(rare):
                print('    ⚠️  %s: %s class(es) with fewer samples than folds (%d): %s — consider class_grouping'
                      % (indicator, label, folds, ', '.join('%s (%d)' % (c, n) for c, n in rare.items())))

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
        covariate_array    = _parse_array_param(getattr(p, 'covariate_array',  []))
        regressor_array    = _parse_array_param(getattr(p, 'regressor_array',  []))
        model_params_param = str(getattr(p, 'model_parameters_fp', 'default')).strip()
        traintest          = bool(getattr(p, 'traintest', True))
        test_size          = float(getattr(p, 'test_size', 0.3))
        do_kfold           = bool(getattr(p, 'kfold',     False))
        folds              = int(getattr(p, 'folds',      10))
        regr_sym_param     = str(getattr(p, 'regressionmodelsymbols', 'default')).strip()
        tgt_sym_param      = str(getattr(p, 'targetfeaturesymbols',   'default')).strip()
        transformation_param  = str(getattr(p, 'indicator_transformation', 'none') or 'none').strip()
        standardisation_param = str(getattr(p, 'indicator_standardisation', 'none') or 'none').strip()
        refit_chain        = bool(getattr(p, 'refit_preprocessing', True))
        model_fp_param     = str(getattr(p, 'model_fp', '') or '').strip()
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
        try:
            class_grouping = Parse_class_grouping(getattr(p, 'class_grouping', []))
        except ValueError as e:
            print('    ERROR: %s' % e)
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
        df, units_D = Read_parquet_with_units(parquet_fp)
        df   = df.reset_index(drop=True)
        stem = os.path.splitext(os.path.basename(parquet_fp))[0]
        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        companion = _load_companion_json(project_root_fp, stem) or {}
        chain     = companion.get('preprocessing_chain', [])

        # Spectra the model starts from: with refit_preprocessing the unprocessed source parquet of the chain,
        # whose steps are then refitted on the training samples inside the pipeline; otherwise the input parquet.
        if refit_chain and chain:
            source_fp = self._Source_parquet(project_root_fp, companion)
            if source_fp is None:
                return
            src_df, src_units_D = Read_parquet_with_units(source_fp)
            if not self._Rows_aligned(df, src_df, source_fp):
                return
            spectral_cols = [c for c in src_df.columns if _is_spectral_col(c)]
            signal_type   = Signal_type_of(src_units_D, spectral_cols)
            X_all = src_df[spectral_cols].reset_index(drop=True)
            if self.verbose >= 1:
                print('    Refitting %d preprocessing step(s) on the training samples, from %s' % (
                    len(chain), os.path.basename(source_fp)))
        else:
            source_fp     = parquet_fp
            spectral_cols = [c for c in df.columns if _is_spectral_col(c)]
            signal_type   = Signal_type_of(units_D, spectral_cols)
            X_all = df[spectral_cols].reset_index(drop=True)
            if chain:
                print('    ⚠️  refit_preprocessing=false: the %d preprocessing step(s) of %s were fitted on all '
                      'samples, test metrics are optimistic.' % (len(chain), os.path.basename(parquet_fp)))
            chain = []

        if not spectral_cols:
            print('    ERROR: no spectral columns found.')
            return
        col_names = [str(c) for c in spectral_cols]
        # Header row 2 — stored on each saved model and in the results JSON
        spectral_header_D = {c: signal_type for c in col_names}

        covariates = self._Covariates(df, units_D, companion, covariate_array, class_grouping)
        if covariates is None:
            return
        numeric_cov, nominal_cov = covariates
        covariate_cols = numeric_cov + nominal_cov

        # Covariates (input parquet, same rows as the source parquet) next to the spectra; samples missing any of
        # them are skipped
        covariate_ok = np.ones(len(df), dtype=bool)
        cov_tag = ''
        if covariate_cols:
            X_all = pd.concat([X_all, df[covariate_cols].reset_index(drop=True)], axis=1)
            covariate_ok = df[covariate_cols].notna().all(axis=1).values
            cov_tag = _Covariate_tag(covariate_cols)
            if self.verbose >= 1:
                print('    Covariates: %s' % ', '.join(
                    covariate_cols if not class_grouping else
                    ['%s (%d groups)' % (c, len(set(class_grouping[c].values()))) if c in class_grouping else c
                     for c in covariate_cols]))
            if (~covariate_ok).any():
                print('    ⚠️  %d sample(s) without a value for every covariate, skipped' % int((~covariate_ok).sum()))
        covariates_D = {'numeric': numeric_cov, 'nominal': nominal_cov,
                        'class_grouping': {c: g for c, g in class_grouping.items() if c in nominal_cov}}

        # A covariate of this model is never its indicator; the other covariates of the dataset are only
        # indicators when named in indicator_array
        all_indicators = [
            c for c in df.columns
            if not _is_metadata_col(c)
            and not _is_spectral_col(c)
            and c not in covariate_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if indicator_array:
            for col in indicator_array:
                if col not in all_indicators:
                    print('    WARNING: indicator "%s" not in parquet (or a covariate of this model), skipping.' % col)
            indicator_cols = [c for c in indicator_array if c in all_indicators]
        else:
            dataset_cov = set(_Companion_covariates(companion))
            indicator_cols = [c for c in all_indicators if c not in dataset_cov]
        if not indicator_cols:
            print('    ERROR: no valid indicator columns.')
            return

        # Settings: parameter set name (ml_setting), "default", "none" or a JSON path relative to the project root
        model_params = self._Settings('regressor', model_params_param or 'default', _DEFAULT_MODEL_PARAMS_FP)

        # Indicator transformation and standardisation — only what the process sets, default none
        transformation_raw = _load_transform_config(self._Settings_fp(transformation_param),
                                                    _DEFAULT_TRANSFORMATION_FPN, self.verbose)
        standardise_raw    = _load_transform_config(self._Settings_fp(standardisation_param),
                                                    _DEFAULT_STANDARDISE_FPN, self.verbose, 'standardisation')

        regr_sym_raw = self._Settings('plot_symbol', regr_sym_param or 'default', _DEFAULT_REGR_SYMBOLS_FP,
                                      'regressionmodelsymbols')
        tgt_sym_raw  = self._Settings('plot_symbol', tgt_sym_param or 'default', _DEFAULT_TARGET_SYMBOLS_FP,
                                      'targetfeaturesymbols')
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
        # Saved models: model_fp (relative to the project root) or <project_root_fp>/models
        models_root   = (os.path.join(project_root_fp, 'models') if not model_fp_param
                         else self._Settings_fp(model_fp_param))
        models_dir_tt = os.path.join(models_root, stem + cov_tag + '_tt')
        models_dir_kf = os.path.join(models_root, stem + cov_tag + '_kf')
        plot_dir_tt   = os.path.join(project_root_fp, 'plot',                stem + cov_tag + '_tt')
        plot_dir_kf   = os.path.join(project_root_fp, 'plot',                stem + cov_tag + '_kf')
        cov_dir_tt    = os.path.join(project_root_fp, 'covariate_importance', stem + cov_tag + '_tt')
        cov_dir_kf    = os.path.join(project_root_fp, 'covariate_importance', stem + cov_tag + '_kf')
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

        data_D = {
            'input_parquet':  os.path.basename(parquet_fp),
            'input_sha256':   File_sha256(parquet_fp),
            'source_parquet': os.path.basename(source_fp),
            'source_sha256':  File_sha256(source_fp) if source_fp != parquet_fp else None,
            'directory':      project_root_fp,
        }

        def _save_model(model, fp, pipeline_D, indicator, model_key, mode, metrics, train_rows):
            '''Save the whole fitted chain (preprocessing + regressor + indicator transformation) and its manifest.'''
            model.spectral_header_ = spectral_header_D
            model.input_columns_   = col_names
            model.covariates_      = covariates_D
            model.pipeline_info_   = pipeline_D
            joblib.dump(model, fp)
            metrics['model_fp']     = fp
            metrics['model_sha256'] = File_sha256(fp)
            metrics['manifest_fp']  = Manifest_fp(fp)
            ids = (df.loc[train_rows, holdout_key].tolist() if holdout_key in df.columns
                   else np.flatnonzero(train_rows).tolist())
            manifest = {
                'manifest_version':    1,
                'created_at':          datetime.now().isoformat(timespec='seconds'),
                'model_file':          os.path.basename(fp),
                'model_sha256':        metrics['model_sha256'],
                'indicator':           indicator,
                'indicator_unit':      units_D.get(indicator),
                'regressor':           model_key,
                'hyper_parameters':    model_params.get(model_key, {}).get('hyper_parameters', {}),
                'mode':                mode,
                'input':               {'columns': col_names, 'signal_type': signal_type, 'n_bands': len(col_names),
                                        'covariates': _Fitted_covariates(model, covariates_D)},
                'pipeline':            pipeline_D,
                'preprocessing_chain': companion.get('preprocessing_chain', []),
                'data':                dict(data_D, sample_id_key=holdout_key, n_train=len(ids), train_sample_ids=ids),
                'metrics':             {k: v for k, v in metrics.items() if k not in ('model_fp', 'manifest_fp')},
                'provenance':          Provenance_snapshot(),
            }
            with open(metrics['manifest_fp'], 'w') as fh:
                json.dump(manifest, fh, indent=2, default=str)

        # Test samples: hash of the sample id — the same holdout as the band selection steps
        test_rows, holdout_key = Holdout_test_mask(df.reset_index(drop=True), test_size)

        for indicator in indicator_cols:
            y_all = df[indicator].values.astype(float)

            transform, standardise = Indicator_target_settings(indicator, transformation_raw, standardise_raw)
            in_domain  = Target_domain_mask(np.nan_to_num(y_all, nan=1.0), transform)
            n_domain   = int((~np.isnan(y_all) & ~in_domain).sum())
            valid_mask = ~np.isnan(y_all) & in_domain & covariate_ok
            if n_domain:
                print('    ⚠️  %s: %d value(s) can not be %s-transformed, excluded' % (indicator, n_domain, transform))

            n_valid = int(valid_mask.sum())
            if n_valid < 10:
                print('    WARNING: too few samples for "%s" (%d), skipping.' % (indicator, n_valid))
                continue

            try:
                steps, notes = Chain_to_steps(chain, indicator, test_size)
            except ValueError as e:
                print('    ERROR: %s' % e)
                return
            for note in notes:
                print('    ⚠️  %s' % note)

            target = IndicatorTransformer(transform, standardise)
            if covariate_cols:
                blocks = CovariateBlocks(Pipeline(steps), tuple(numeric_cov), tuple(nominal_cov),
                                         covariates_D['class_grouping'])
                pipe = Pipeline([('blocks', blocks), ('frame', RegressorFrame()), ('regressor', None)])
            else:
                pipe = Pipeline(steps + [('frame', RegressorFrame()), ('regressor', None)])

            def _make_model(regressor):
                model = clone(pipe).set_params(regressor=clone(regressor))
                if transform != 'linear' or standardise:
                    model = TransformedTargetRegressor(regressor=model, transformer=clone(target),
                                                       check_inverse=False)
                return model

            pipeline_D = {
                'steps':     [name for name, _ in steps],
                'indicator': {'transform': transform, 'standardise': standardise},
                'holdout':   {'test_size': test_size, 'key': holdout_key},
                'notes':     notes,
                'covariates': covariates_D,
            }

            X = X_all[valid_mask]
            y = y_all[valid_mask]
            results_D[indicator] = {'pipeline': pipeline_D}
            ind_s     = _ind_safe(indicator)
            ind_color = tgt_sym_D.get(indicator, {}).get('color', 'steelblue')

            if do_kfold:
                self._Warn_rare_classes(df, valid_mask, nominal_cov, class_grouping, folds, indicator)

            if self.verbose >= 1:
                print('    Indicator: %s  (%d samples, %d bands%s, transform=%s, standardise=%s)' % (
                    indicator, n_valid, len(col_names),
                    ', %d covariates' % len(covariate_cols) if covariate_cols else '', transform, standardise))

            if traintest:
                tr_mask = valid_mask & ~test_rows
                te_mask = valid_mask & test_rows
                te_mask = self._Drop_unseen_classes(df, tr_mask, te_mask, nominal_cov, class_grouping, indicator)
                X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
                X_te, y_te = X_all[te_mask], y_all[te_mask]
                if len(y_te) < 2 or len(y_tr) < 10:
                    print('    WARNING: too few train (%d) / test (%d) samples for "%s", skipping train/test.' % (
                        len(y_tr), len(y_te), indicator))

            for model_key, model_proto in regressors.items():

                if self.verbose >= 1:
                    print('      Model: %s' % model_key)

                # ---- train / test
                if traintest and len(y_te) >= 2 and len(y_tr) >= 10:
                    model_tt = _make_model(model_proto)
                    model_tt.fit(X_tr, y_tr)
                    pred_tt  = np.ravel(model_tt.predict(X_te))

                    metrics_tt = _compute_metrics(y_te, pred_tt)
                    metrics_tt['n_train'] = len(y_tr)
                    metrics_tt['n_test']  = len(y_te)

                    fitted_D = dict(pipeline_D, indicator=_Fitted_target(model_tt))
                    metrics_tt['indicator_transformation'] = fitted_D['indicator']
                    jl_fp = os.path.join(models_dir_tt,
                                         '%s%s_%s_%s_tt.joblib' % (stem, cov_tag, ind_s, model_key))
                    _save_model(model_tt, jl_fp, fitted_D, indicator, model_key, 'traintest', metrics_tt, tr_mask)

                    if show_scatter or save_scatter:
                        fig = _plot_scatter(y_te, pred_tt, model_key, indicator,
                                            'train/test', metrics_tt, regr_sym_D, tgt_sym_D)
                        _handle_fig(fig,
                                    os.path.join(plot_dir_tt,
                                                 'scatter_%s_%s_tt.png' % (ind_s, model_key)),
                                    show_scatter, save_scatter)

                    tail = _Fitted_tail(model_tt)
                    if show_feat or save_feat:
                        imp = _feature_importance(tail.regressor, model_key)
                        if imp is not None:
                            fig2 = _plot_importance(imp, tail.Features(X_tr), model_key, indicator, 'tt',
                                                    color=ind_color)
                            _handle_fig(fig2,
                                        os.path.join(plot_dir_tt,
                                                     'importance_%s_%s_tt.png' % (ind_s, model_key)),
                                        show_feat, save_feat)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        Xt_te = tail.Preprocess(X_te)
                        feat_wl = tail.Features(X_te)
                        pi = sk_permutation_importance(
                            tail, Xt_te, y_te, scoring='r2', n_repeats=10, random_state=42)
                        fig3 = _plot_permutation(pi, feat_wl, model_key, indicator, 'train/test',
                                                 color=ind_color, max_cov=max_cov)
                        _handle_fig(fig3,
                                    os.path.join(plot_dir_tt,
                                                 'permutation_%s_%s_tt.png' % (ind_s, model_key)),
                                    show_perm, save_perm)
                        _save_permutation_data(pi, feat_wl, model_key, indicator, 'tt', cov_dir_tt)

                    results_D[indicator].setdefault('traintest', {})[model_key] = metrics_tt
                    if self.verbose >= 1:
                        print('        [tt] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                            metrics_tt['r2'], metrics_tt['rmse'], metrics_tt['rpiq']))

                # ---- k-fold (the whole pipeline is refitted in every fold)
                if do_kfold:
                    pred_kf  = np.ravel(cross_val_predict(_make_model(model_proto), X, y, cv=folds))
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
                                _make_model(model_proto), X, y, cv=folds, scoring=scoring)
                            metrics_kf[key_out + '_folded_mean'] = round(
                                float(abs(cv_scores.mean())), 4)
                            metrics_kf[key_out + '_folded_std'] = round(
                                float(cv_scores.std()), 4)
                        except Exception:
                            pass

                    model_kf = _make_model(model_proto)
                    model_kf.fit(X, y)
                    fitted_D = dict(pipeline_D, indicator=_Fitted_target(model_kf))
                    metrics_kf['indicator_transformation'] = fitted_D['indicator']
                    jl_fp = os.path.join(models_dir_kf,
                                         '%s%s_%s_%s_kf.joblib' % (stem, cov_tag, ind_s, model_key))
                    _save_model(model_kf, jl_fp, fitted_D, indicator, model_key, 'kfold', metrics_kf, valid_mask)

                    if show_scatter or save_scatter:
                        fig = _plot_scatter(y, pred_kf, model_key, indicator,
                                            'k-fold', metrics_kf, regr_sym_D, tgt_sym_D)
                        _handle_fig(fig,
                                    os.path.join(plot_dir_kf,
                                                 'scatter_%s_%s_kf.png' % (ind_s, model_key)),
                                    show_scatter, save_scatter)

                    tail = _Fitted_tail(model_kf)
                    if show_feat or save_feat:
                        imp = _feature_importance(tail.regressor, model_key)
                        if imp is not None:
                            fig2 = _plot_importance(imp, tail.Features(X), model_key, indicator, 'kf',
                                                    color=ind_color)
                            _handle_fig(fig2,
                                        os.path.join(plot_dir_kf,
                                                     'importance_%s_%s_kf.png' % (ind_s, model_key)),
                                        show_feat, save_feat)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        feat_wl = tail.Features(X)
                        pi = sk_permutation_importance(
                            tail, tail.Preprocess(X), y, scoring='r2', n_repeats=10, random_state=42)
                        fig3 = _plot_permutation(pi, feat_wl, model_key, indicator, 'k-fold',
                                                 color=ind_color, max_cov=max_cov)
                        _handle_fig(fig3,
                                    os.path.join(plot_dir_kf,
                                                 'permutation_%s_%s_kf.png' % (ind_s, model_key)),
                                    show_perm, save_perm)
                        _save_permutation_data(pi, feat_wl, model_key, indicator, 'kf', cov_dir_kf)

                    results_D[indicator].setdefault('kfold', {})[model_key] = metrics_kf
                    if self.verbose >= 1:
                        print('        [kf] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                            metrics_kf['r2'], metrics_kf['rmse'], metrics_kf['rpiq']))

        if results_D:
            self._print_summary(results_D, indicator_cols, list(regressors.keys()))

        out_json = {
            'input_parquet':       os.path.basename(parquet_fp),
            'source_parquet':      os.path.basename(source_fp),
            'refit_preprocessing': bool(chain),
            'preprocessing_chain': companion.get('preprocessing_chain', []),
            'signal_type':         signal_type,
            'feature_columns':     col_names,
            'covariates':          covariates_D,
            'results':             results_D,
            'provenance':          Provenance_snapshot(),
        }
        result_fp = os.path.join(project_root_fp, stem + cov_tag + '_regression.json')
        with open(result_fp, 'w') as fh:
            json.dump(out_json, fh, indent=2)
        print('    Results saved: %s' % result_fp)
