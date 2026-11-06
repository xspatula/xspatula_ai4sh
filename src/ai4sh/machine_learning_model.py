'''
Created on 26 April 2026

@author: thomasgumbricht
'''

import os
import re
import json
import hashlib
import warnings

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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.exceptions import ConvergenceWarning
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
from src.ai4sh.ml_transformers import (CappedPLSRegression, Chain_to_steps, Column_index, CovariateBlocks,
                                       Group_classes, Holdout_test_mask, IndicatorTransformer,
                                       Indicator_target_settings, Model_blocks, Parse_class_grouping,
                                       Pls_cap_text, RegressorFrame, Target_domain_mask)
from src.ai4sh.spectral_columns import Signal_type_of
from src.ai4sh.project_paths import Default_fpn
from src.ai4sh.provenance import Provenance_snapshot, File_sha256
from src.ai4sh.ml_setting import Load_settings, Is_path
from src.ai4sh.ml_model_plot import (Default_layout_name, In_layout, Load_layout, Plot_grid,
                                     Plot_importance, Plot_permutation, Plot_scatter)

_DEFAULT_MODEL_PARAMS_FP = Default_fpn('regressors', 'regression_model_parameters.json')
_DEFAULT_REGR_SYMBOLS_FP = Default_fpn('plot', 'regressionmodelsymbols.json')
_DEFAULT_TARGET_SYMBOLS_FP = Default_fpn('plot', 'targetfeaturesymbols.json')
_DEFAULT_PLOT_LAYOUT_FP = Default_fpn('plot', 'plotlayouts.json')


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
    if k == 'plsr':       return CappedPLSRegression(**hp)
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

class _Pooled_permutation:
    '''Permutation importances (features × repeats) pooled over folds, with mean and std like sklearn's result.'''

    def __init__(self, importances):
        self.importances      = importances
        self.importances_mean = importances.mean(axis=1)
        self.importances_std  = importances.std(axis=1)


def _Permutation_out_of_fold(make_model, X, y, folds, features):
    '''Permutation importance on held-out samples: in every fold (the folds of cross_val_predict) a model fitted on
    the other folds is permuted on the fold; the repeats of all folds are pooled. None if a fold ends with other
    features than the model on all samples (e.g. a band selection step that selects per fold).'''
    parts = []
    for tr, te in KFold(folds).split(X):
        tail = _Fitted_tail(make_model().fit(X.iloc[tr], y[tr]))
        if list(tail.Features(X.iloc[te])) != list(features):
            return None
        parts.append(sk_permutation_importance(tail, tail.Preprocess(X.iloc[te]), y[te], scoring='r2',
                                               n_repeats=10, random_state=42).importances)
    return _Pooled_permutation(np.concatenate(parts, axis=1))


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


def _Fitted_regressor(model):
    '''The fitted regressor (last pipeline step) of a model, inside the indicator transformation if any.'''
    return (model.regressor_ if isinstance(model, TransformedTargetRegressor) else model)[-1]


def _Fitted_hyper_parameters(model):
    '''Hyper-parameters the fitted regressor used where they can differ from the requested ones
    (plsr: n_components capped to the training samples and the features left after the chain); {} otherwise.'''
    regressor = _Fitted_regressor(model)
    if hasattr(regressor, 'n_components_'):
        return {'n_components': int(regressor.n_components_)}
    return {}


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


def _Fitted_covariates(model, covariates_D, units_D=None):
    '''Covariates of a fitted model for its manifest: numeric labels with their unit and the training mean and
    standard deviation they are standardised with (6 significant digits), nominal labels with the classes (after
    grouping) seen in training, and the class grouping.'''
    blocks = Model_blocks(model)
    if blocks is None:
        return {}
    standardisation = {label: {'mean': float('%.6g' % mean), 'std': float('%.6g' % blocks.numeric_std_[label])}
                       for label, mean in getattr(blocks, 'numeric_mean_', {}).items()}
    units = {label: (units_D or {}).get(label) for label in blocks.numeric}
    return dict(covariates_D, levels=blocks.levels_, numeric_standardisation=standardisation, numeric_units=units)


class _Convergence_counter:
    '''Counts the ConvergenceWarnings of each model (every fold and fit warns) and prints one line per model instead.'''

    def __enter__(self):
        self.label, self.count = '', 0
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter('always', ConvergenceWarning)
        self._show = warnings.showwarning
        warnings.showwarning = self._Show
        return self

    def _Show(self, message, category, filename, lineno, file=None, line=None):
        if issubclass(category, ConvergenceWarning):
            self.count += 1
        else:
            self._show(message, category, filename, lineno, file, line)

    def Label(self, label):
        '''Start counting for the next model (the count of the previous one is printed).'''
        self.Flush()
        self.label = label

    def Flush(self):
        if self.count:
            print('        ⚠️  %s: %d fit(s) stopped at max_iter before converging — raise max_iter in the '
                  'hyper-parameters (model_parameters_fp)' % (self.label, self.count))
        self.count = 0

    def __exit__(self, *exc):
        self.Flush()
        warnings.showwarning = self._show
        self._catch.__exit__(*exc)
        return False


# ------------------------------------------------------------------ main class

class Process_regression_model(Get_schema_table):

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose        = process_S.process.verbose
        self.process_S      = process_S
        self.pg_session_C   = pg_session_C
        self.project_root_FP = project_root_FP

    def _Sub_process(self, _):
        if self.process_S.process.process == 'regression_modeling':
            with _Convergence_counter() as self._convergence:
                self._Regression_modeling()

    def _Settings_fp(self, param):
        '''A relative settings file path resolved against the project root; anything else (none, default,
        parameter set name, absolute path) as is.'''
        s = str(param).strip()
        if not Is_path(s) or os.path.isabs(s):
            return s
        return Get_project_path(self.project_root_FP, s)

    def _Settings(self, family, param, default_fpn, default_name='standard'):
        '''Settings dict of a parameter set / file ({} for none); None for a missing file or an unknown parameter set
        name (error printed: the caller stops instead of running with library defaults).'''
        try:
            return Load_settings(family, param, default_fpn, default_name, self._Settings_fp, self.verbose) or {}
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: %s' % e)
            return None

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
                    header += '  %-18s' % ('%s/%s' % (key, 'tt' if mode == 'traintest' else 'kf'))
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
        layout_param       = str(getattr(p, 'plot_layout', 'default') or 'default').strip()
        save_layout        = bool(getattr(p, 'save_plot_layout', False))
        importance_only    = bool(getattr(p, 'importance_only', False))

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
        if importance_only:
            # Fit and map the importance only: no model tests (scatter, metrics, k-fold predictions), nothing saved
            # but the importance plots and the permutation data
            show_scatter = save_scatter = False
            if not (show_feat or save_feat or show_perm or save_perm):
                print('    ERROR: importance_only needs show/save_feature_importance or show/save_permutation_importance.')
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

        col_names = [str(c) for c in spectral_cols]
        # Header row 2 — stored on each saved model and in the results JSON
        spectral_header_D = {c: signal_type for c in col_names}

        covariates = self._Covariates(df, units_D, companion, covariate_array, class_grouping)
        if covariates is None:
            return
        numeric_cov, nominal_cov = covariates
        covariate_cols = numeric_cov + nominal_cov

        # A model needs spectra and/or covariates
        if not spectral_cols and not covariate_cols:
            print('    ERROR: no spectral columns and no covariates (covariate_array) — nothing to model on.')
            return

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
            print('    ERROR: no valid indicator columns.%s' % (
                ' %s is a dataset for prediction (compiled without indicators).' % stem
                if companion.get('purpose') == 'prediction' else ''))
            return

        # Settings: parameter set name (ml_setting), "default", "none" or a JSON path relative to the project root
        model_params = self._Settings('regressor', model_params_param or 'default', _DEFAULT_MODEL_PARAMS_FP)

        # Indicator transformation and standardisation — only what the process sets, default none
        try:
            transformation_raw = _load_transform_config(self._Settings_fp(transformation_param),
                                                        _DEFAULT_TRANSFORMATION_FPN, self.verbose)
            standardise_raw    = _load_transform_config(self._Settings_fp(standardisation_param),
                                                        _DEFAULT_STANDARDISE_FPN, self.verbose, 'standardisation')
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: %s' % e)
            return

        regr_sym_raw = self._Settings('plot_symbol', regr_sym_param or 'default', _DEFAULT_REGR_SYMBOLS_FP,
                                      'regressionmodelsymbols')
        tgt_sym_raw  = self._Settings('plot_symbol', tgt_sym_param or 'default', _DEFAULT_TARGET_SYMBOLS_FP,
                                      'targetfeaturesymbols')
        if model_params is None or regr_sym_raw is None or tgt_sym_raw is None:
            return
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
        # Folders of what is saved (plot folders are made by _handle_fig when a plot is saved)
        for on, models_dir, cov_dir in ((traintest, models_dir_tt, cov_dir_tt), (do_kfold, models_dir_kf, cov_dir_kf)):
            if on and not importance_only:
                os.makedirs(models_dir, exist_ok=True)
            if on and save_perm:
                os.makedirs(cov_dir, exist_ok=True)

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

        # Plot layout: the shown plots composed into grid figures; the saved PNGs stay single panels
        modes = [m for m, on in (('tt', traintest), ('kf', do_kfold)) if on]
        try:
            layout = Load_layout(layout_param,
                                 Default_layout_name(show_scatter, show_feat, show_perm, modes,
                                                     len(indicator_cols), len(regressors)),
                                 _DEFAULT_PLOT_LAYOUT_FP, self._Settings_fp, self.verbose)
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: %s' % e)
            return
        if layout is not None and self.verbose >= 1:
            print('    Plot layout: %s%s (grid drawn per %s)' % (
                layout['name'], ' on %s' % layout['base'] if layout.get('base') else '', layout['draw']))
        grid_panels = []

        def _Panel(ptype, mode, shown, save, make_fig, fp, **data):
            '''A shown panel the layout has goes to the grid; the single figure is made when saved or shown alone.'''
            in_grid = shown and In_layout(layout, ptype, mode)
            if in_grid:
                grid_panels.append(dict(data, type=ptype, mode=mode))
            if save or (shown and not in_grid):
                _handle_fig(make_fig(), fp, shown and not in_grid, save)

        def _Draw_grid(panels, indicators, suffix):
            fig = Plot_grid(panels, layout, indicators, list(regressors), regr_sym_D, tgt_sym_D,
                            title=stem + cov_tag)
            if fig is not None:
                _handle_fig(fig, os.path.join(project_root_fp, 'plot', stem + cov_tag + '_grid',
                                              '%s%s.png' % (layout['name'], suffix)), True, save_layout)

        data_D = {
            'input_parquet':  os.path.basename(parquet_fp),
            'input_sha256':   File_sha256(parquet_fp),
            'source_parquet': os.path.basename(source_fp),
            'source_sha256':  File_sha256(source_fp) if source_fp != parquet_fp else None,
            'directory':      project_root_fp,
        }

        constant_noted = set()   # (indicator, mode): the "does not vary" note does not depend on the regressor

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
            fitted_hp = _Fitted_hyper_parameters(model)
            if fitted_hp:
                metrics['fitted_hyper_parameters'] = fitted_hp
            cap = Pls_cap_text(_Fitted_regressor(model))
            if cap:
                print('        ⚠️  %s %s [%s]: %s — fewer components than requested'
                      % (indicator, model_key, 'tt' if mode == 'traintest' else 'kf', cap))
            blocks = Model_blocks(model)
            constant = blocks.Constant_covariates() if blocks is not None else []
            if constant and (indicator, mode) not in constant_noted:
                constant_noted.add((indicator, mode))
                print('        ⚠️  %s [%s]: covariate(s) %s do not vary in the training samples — no effect'
                      % (indicator, 'tt' if mode == 'traintest' else 'kf', ', '.join(constant)))
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
                'fitted_hyper_parameters': fitted_hp,
                'mode':                mode,
                'input':               {'columns': col_names, 'signal_type': signal_type, 'n_bands': len(col_names),
                                        'covariates': _Fitted_covariates(model, covariates_D, units_D)},
                'pipeline':            pipeline_D,
                'preprocessing_chain': companion.get('preprocessing_chain', []),
                'data':                dict(data_D, sample_id_key=holdout_key, n_train=len(ids), train_sample_ids=ids),
                'metrics':             {k: v for k, v in metrics.items()
                                        if k not in ('model_fp', 'manifest_fp', 'fitted_hyper_parameters')},
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

                self._convergence.Label('%s %s' % (indicator, model_key))
                if self.verbose >= 1:
                    print('      Model: %s' % model_key)

                # ---- train / test
                if traintest and len(y_te) >= 2 and len(y_tr) >= 10:
                    model_tt = _make_model(model_proto)
                    model_tt.fit(X_tr, y_tr)

                    if not importance_only:
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
                        _Panel('scatter', 'tt', show_scatter, save_scatter,
                               lambda: Plot_scatter(y_te, pred_tt, model_key, indicator,
                                                    'train/test', metrics_tt, regr_sym_D, tgt_sym_D),
                               os.path.join(plot_dir_tt, 'scatter_%s_%s_tt.png' % (ind_s, model_key)),
                               indicator=indicator, regressor=model_key, obs=y_te, pred=pred_tt, metrics=metrics_tt)

                    tail = _Fitted_tail(model_tt)
                    if show_feat or save_feat:
                        imp = _feature_importance(tail.regressor, model_key)
                        if imp is not None:
                            feats = tail.Features(X_tr)
                            _Panel('importance', 'tt', show_feat, save_feat,
                                   lambda: Plot_importance(imp, feats, model_key, indicator, 'tt', color=ind_color),
                                   os.path.join(plot_dir_tt, 'importance_%s_%s_tt.png' % (ind_s, model_key)),
                                   indicator=indicator, regressor=model_key, importances=imp, features=feats,
                                   color=ind_color)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        Xt_te = tail.Preprocess(X_te)
                        feat_wl = tail.Features(X_te)
                        pi = sk_permutation_importance(
                            tail, Xt_te, y_te, scoring='r2', n_repeats=10, random_state=42)
                        _Panel('permutation', 'tt', show_perm, save_perm,
                               lambda: Plot_permutation(pi, feat_wl, model_key, indicator, 'tt',
                                                        color=ind_color, max_cov=max_cov),
                               os.path.join(plot_dir_tt, 'permutation_%s_%s_tt.png' % (ind_s, model_key)),
                               indicator=indicator, regressor=model_key, means=pi.importances_mean,
                               stds=pi.importances_std, features=feat_wl, color=ind_color)
                        if save_perm:
                            _save_permutation_data(pi, feat_wl, model_key, indicator, 'tt', cov_dir_tt)

                    if importance_only:
                        if self.verbose >= 1:
                            print('        [tt] importance mapped (fitted on %d, permutation on %d samples)'
                                  % (len(y_tr), len(y_te)))
                    else:
                        results_D[indicator].setdefault('traintest', {})[model_key] = metrics_tt
                        if self.verbose >= 1:
                            print('        [tt] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                                metrics_tt['r2'], metrics_tt['rmse'], metrics_tt['rpiq']))

                # ---- k-fold (the whole pipeline is refitted in every fold)
                if do_kfold and importance_only:
                    model_kf = _make_model(model_proto)
                    model_kf.fit(X, y)

                if do_kfold and not importance_only:
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

                if do_kfold:
                    if show_scatter or save_scatter:
                        _Panel('scatter', 'kf', show_scatter, save_scatter,
                               lambda: Plot_scatter(y, pred_kf, model_key, indicator,
                                                    'k-fold', metrics_kf, regr_sym_D, tgt_sym_D),
                               os.path.join(plot_dir_kf, 'scatter_%s_%s_kf.png' % (ind_s, model_key)),
                               indicator=indicator, regressor=model_key, obs=y, pred=pred_kf, metrics=metrics_kf)

                    tail = _Fitted_tail(model_kf)
                    if show_feat or save_feat:
                        imp = _feature_importance(tail.regressor, model_key)
                        if imp is not None:
                            feats = tail.Features(X)
                            _Panel('importance', 'kf', show_feat, save_feat,
                                   lambda: Plot_importance(imp, feats, model_key, indicator, 'kf', color=ind_color),
                                   os.path.join(plot_dir_kf, 'importance_%s_%s_kf.png' % (ind_s, model_key)),
                                   indicator=indicator, regressor=model_key, importances=imp, features=feats,
                                   color=ind_color)

                    if (show_perm or save_perm) and model_key.lower() != 'cubist':
                        # On held-out samples, fold by fold (on the training samples it overrates the features
                        # the model overfits)
                        feat_wl = tail.Features(X)
                        pi = _Permutation_out_of_fold(lambda: _make_model(model_proto), X, y, folds, feat_wl)
                        if pi is None:
                            print('        ⚠️  %s %s [kf]: the folds select different features — permutation '
                                  'importance on the training samples' % (indicator, model_key))
                            pi = sk_permutation_importance(
                                tail, tail.Preprocess(X), y, scoring='r2', n_repeats=10, random_state=42)
                        _Panel('permutation', 'kf', show_perm, save_perm,
                               lambda: Plot_permutation(pi, feat_wl, model_key, indicator, 'kf',
                                                        color=ind_color, max_cov=max_cov),
                               os.path.join(plot_dir_kf, 'permutation_%s_%s_kf.png' % (ind_s, model_key)),
                               indicator=indicator, regressor=model_key, means=pi.importances_mean,
                               stds=pi.importances_std, features=feat_wl, color=ind_color)
                        if save_perm:
                            _save_permutation_data(pi, feat_wl, model_key, indicator, 'kf', cov_dir_kf)

                    if importance_only:
                        if self.verbose >= 1:
                            print('        [kf] importance mapped (fitted on all %d samples, permuted per held-out fold)'
                                  % len(y))
                    else:
                        results_D[indicator].setdefault('kfold', {})[model_key] = metrics_kf
                        if self.verbose >= 1:
                            print('        [kf] r2=%.3f  rmse=%.3f  rpiq=%.3f' % (
                                metrics_kf['r2'], metrics_kf['rmse'], metrics_kf['rpiq']))

            if layout is not None and layout['draw'] == 'indicator':
                _Draw_grid([p for p in grid_panels if p['indicator'] == indicator], [indicator], '_' + ind_s)

        self._convergence.Flush()
        if layout is not None and layout['draw'] == 'run':
            _Draw_grid(grid_panels, indicator_cols, '')

        if importance_only:
            saved = [what for what, on in (('feature importance plots', save_feat),
                                           ('permutation importance plots and data', save_perm)) if on]
            print('    importance_only: %s; no models or results' % (
                'saved %s' % ' and '.join(saved) if saved else 'shown only, nothing saved'))
            return

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
