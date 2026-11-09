'''
Created on 21 October 2026

@author: thomasgumbricht

Apply saved regression models to independent spectra.

A model saved by regression_modeling (<name>.joblib) is the whole fitted chain: the preprocessing steps refitted
on the training samples, the regressor and the indicator transformation. Its manifest (<name>.manifest.json)
records the input bands and signal type, the fitted settings, the training sample ids, the metrics, the sha256
of the model file and the provenance of the run.

Load_model() checks the model file against the manifest sha256 and the software versions; Predict_spectra()
takes unprocessed spectra with the model's bands (any column order) and returns predictions in the native unit
of the indicator. Both are plain functions so that a server can use them without the process machinery.

model_array entries are model files (.joblib) or names of registered models (ml_model.model); a registered model
must also match the sha256 recorded when it was registered.

The predict_independent process predicts every sample of a parquet with one or more models and saves
<stem>_pred.parquet (key columns + one column per model) with a companion JSON in dst_fp (default
<project_root_fp>/predictions, so the .previous_dataframe of the input folder is not changed). If the parquet also holds
observed values of the indicator, the metrics are computed on the samples that were not used for training, and the
observed/predicted scatter plots are shown (and saved) as in regression_modeling: same plot_layout presets and
symbols, one panel per model, labelled "independent". The default layout puts one indicator per row when several
regressors predict it, train/test and k-fold models of the same indicator × regressor side by side.
'''

# Standard library imports
import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from importlib import metadata

from src.postgres import Get_schema_table
from src.lib.pilot import Get_project_path
from src.ai4sh.parquet_units import Read_parquet_with_units, Save_parquet_with_units, INDICATOR_STEPS
from src.ai4sh.previous_dataframe import Register_parquet
from src.ai4sh.provenance import File_sha256
from src.ai4sh.spectral_columns import Signal_type_of
from src.ai4sh.machine_learning_preprocess import (_resolve_input_parquet, _resolve_single_previous,
                                                   _is_spectral_col, _parse_array_param, _load_companion_json)
from src.ai4sh.machine_learning_model import (Manifest_fp, _compute_metrics, _DEFAULT_REGR_SYMBOLS_FP,
                                              _DEFAULT_TARGET_SYMBOLS_FP, _DEFAULT_PLOT_LAYOUT_FP)
from src.ai4sh.ml_model_plot import Default_layout_name, Load_layout, In_layout, Plot_grid, Plot_scatter
from src.ai4sh.ml_setting import Load_settings
from src.ai4sh.ml_transformers import Model_blocks
from src.ai4sh.ml_setting import Is_path
from src.ai4sh.ml_model import Registered_model, Record_independent_metrics


def Load_model(model_fp, verbose=1):
    '''Return (model, manifest) of a saved model; raise ValueError if the model can not be trusted.

    The model file must match the sha256 in its manifest. Different versions of scikit-learn / numpy than the
    ones the model was saved with give a warning (joblib models are not guaranteed to load across versions).
    '''
    manifest_fp = Manifest_fp(model_fp)

    if not os.path.exists(model_fp):
        raise ValueError('model not found: %s' % model_fp)

    if not os.path.exists(manifest_fp):
        raise ValueError('no manifest for %s — models saved before the pipeline refit (Part 2/3) can not be used,'
                         ' rerun regression_modeling' % os.path.basename(model_fp))

    with open(manifest_fp) as fh:
        manifest = json.load(fh)

    if File_sha256(model_fp) != manifest.get('model_sha256'):
        raise ValueError('%s does not match the sha256 in its manifest (file replaced or modified)'
                         % os.path.basename(model_fp))

    saved_env = manifest.get('provenance', {}).get('environment', {})

    for pkg in ('scikit-learn', 'numpy', 'scipy'):

        try:
            now = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            now = None

        if saved_env.get(pkg) and now != saved_env.get(pkg) and verbose >= 1:
            print('    ⚠️  %s saved with %s %s, running %s — check the predictions'
                  % (os.path.basename(model_fp), pkg, saved_env.get(pkg), now))

    return joblib.load(model_fp), manifest


def Check_spectra(manifest, columns, units_D=None):
    '''List of problems that prevent applying the model to spectra with these columns (and units); [] if none.'''
    issues = []
    wanted = manifest['input']['columns']
    missing = [c for c in wanted if c not in set(columns)]
    if missing:
        issues.append('%d of the %d model bands are missing (e.g. %s) — the spectra must have the same bands as '
                      'the unprocessed source parquet of the model' % (len(missing), len(wanted), missing[:3]))
    if units_D is not None and not missing:
        signal_type = Signal_type_of(units_D, wanted)
        if signal_type != manifest['input']['signal_type']:
            issues.append('signal type %s, the model was trained on %s' % (signal_type, manifest['input']['signal_type']))
    covariates = _Model_covariates(manifest)
    missing_cov = [c for c in covariates if c not in set(columns)]
    if missing_cov:
        issues.append('covariate(s) of the model missing in the input: %s' % ', '.join(missing_cov))
    if units_D is not None:
        # Units are compared as text: they come from the indicator definitions in the database (selection, compile),
        # so the same unit has the same spelling. A hand-made file must use those spellings ("weight percent", not "%").
        trained = (manifest['input'].get('covariates') or {}).get('numeric_units') or {}
        wrong = ['%s in %s, the model was trained on %s' % (c, units_D.get(c), u) for c, u in trained.items()
                 if u and units_D.get(c) and units_D.get(c) != u and c not in missing_cov]
        if wrong:
            issues.append('covariate unit differs: %s' % '; '.join(wrong))
    return issues


def _Model_covariates(manifest):
    '''Covariate columns a model needs next to the spectra (numeric + nominal).'''
    cov = manifest['input'].get('covariates') or {}
    return list(cov.get('numeric', [])) + list(cov.get('nominal', []))


def Predict_spectra(model, manifest, spectra_df, units_D=None):
    '''Predict the indicator (native unit) for every row of spectra_df (unprocessed spectra, model bands, plus the
    covariates of the model). Rows with a missing covariate or a class not seen in training get NaN.'''
    issues = Check_spectra(manifest, spectra_df.columns, units_D)
    if issues:
        raise ValueError('; '.join(issues))
    unchecked = Unchecked_covariate_units(manifest, units_D)
    if unchecked:
        print('    ⚠️  %s: no unit recorded in the input for covariate(s) %s — unit not checked'
              % (manifest.get('model_file', 'model'), ', '.join(unchecked)))
    X = spectra_df[manifest['input']['columns'] + _Model_covariates(manifest)]
    blocks = Model_blocks(model)
    if blocks is None:
        return np.ravel(model.predict(X))
    skip = blocks.Unseen(X)
    pred = np.full(len(X), np.nan)
    if skip.any():
        print('    ⚠️  %d sample(s) with a missing covariate or a class not in the training samples, not predicted'
              % int(skip.sum()))
    if (~skip).any():
        pred[~skip] = np.ravel(model.predict(X[~skip]))
    return pred


def Unchecked_covariate_units(manifest, units_D):
    '''Numeric covariates with a unit in the model but none in the input data: their unit could not be checked.'''
    trained = ((manifest['input'].get('covariates') or {}).get('numeric_units') or {})
    return ['%s (model: %s)' % (c, u) for c, u in trained.items() if u and not (units_D or {}).get(c)]


class Process_predict(Get_schema_table):
    '''Apply saved regression models to the spectra of a parquet file.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP

    def _Sub_process(self, _):
        if self.process_S.process.process == 'predict_independent':
            self._Predict_independent()

    def _Path(self, s):
        return s if os.path.isabs(s) else Get_project_path(self.project_root_FP, s)

    def _Settings_fp(self, param):
        '''A relative settings file path resolved against the project root; anything else as is.'''
        s = str(param).strip()
        return s if not Is_path(s) or os.path.isabs(s) else self._Path(s)

    def _Plot_observed_predicted(self, panels, plot_dir, title):
        '''Observed/predicted scatter of every model with independent metrics: single figures (saved, or shown when
        the layout is none) and grid figures from plot_layout (shown, optionally saved), as in regression_modeling.'''
        p = self.process_S.process.parameters
        show        = bool(getattr(p, 'show_observed_predicted', True))
        save        = bool(getattr(p, 'save_observed_predicted', False))
        save_layout = bool(getattr(p, 'save_plot_layout', False))
        if not panels or not (show or save):
            return

        try:
            regr_sym_D = Load_settings('plot_symbol', str(getattr(p, 'regressionmodelsymbols', 'default') or 'default'),
                                       _DEFAULT_REGR_SYMBOLS_FP, 'regressionmodelsymbols', self._Settings_fp, self.verbose) or {}
            tgt_sym_D  = Load_settings('plot_symbol', str(getattr(p, 'targetfeaturesymbols', 'default') or 'default'),
                                       _DEFAULT_TARGET_SYMBOLS_FP, 'targetfeaturesymbols', self._Settings_fp, self.verbose) or {}
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: plot symbols: %s — no plots' % e)
            return
        regr_sym_D = dict(regr_sym_D.get('regressionModelSymbols', regr_sym_D))
        tgt_sym_D  = tgt_sym_D.get('targetFeatureSymbols', tgt_sym_D)

        # Two models with the same indicator, regressor and mode (e.g. other preprocessing): numbered, same symbol
        seen = {}
        for panel in panels:
            k = (panel['indicator'], panel['regressor'], panel['mode'])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > 1:
                label = '%s-%d' % (panel['regressor'], seen[k])
                regr_sym_D[label] = regr_sym_D.get(panel['regressor'], {})
                panel['regressor'] = label

        indicators = list(dict.fromkeys(q['indicator'] for q in panels))
        regressors = list(dict.fromkeys(q['regressor'] for q in panels))
        modes      = [m for m in ('tt', 'kf') if any(q['mode'] == m for q in panels)]
        try:
            layout = Load_layout(str(getattr(p, 'plot_layout', 'default') or 'default'),
                                 Default_layout_name(show, False, False, modes, len(indicators), len(regressors)),
                                 _DEFAULT_PLOT_LAYOUT_FP, self._Settings_fp, self.verbose)
        except (FileNotFoundError, ValueError) as e:
            print('    ERROR: %s — no plots' % e)
            return

        def _Out(fig, fp, do_show, do_save):
            if do_show:
                plt.figure(fig.number)
                plt.show()
            if do_save:
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                fig.savefig(fp, dpi=150, bbox_inches='tight')
                if self.verbose >= 1:
                    print('    Saved: %s' % fp)
            plt.close(fig)

        grid = []
        for q in panels:
            in_grid = show and In_layout(layout, 'scatter', q['mode'])
            if in_grid:
                grid.append(q)
            if save or (show and not in_grid):
                fig = Plot_scatter(q['obs'], q['pred'], q['regressor'], q['indicator'], q['label'], q['metrics'],
                                   regr_sym_D, tgt_sym_D)
                _Out(fig, os.path.join(plot_dir, 'scatter_%s_%s_%s.png' % (
                    q['indicator'].replace(' ', '-').replace('/', '-'), q['regressor'], q['mode'])),
                     show and not in_grid, save)

        if not grid:
            return
        if self.verbose >= 1:
            print('    Plot layout: %s%s' % (layout['name'], ' on %s' % layout['base'] if layout.get('base') else ''))
        groups = ([(q_ind, [q for q in grid if q['indicator'] == q_ind], '_' + q_ind) for q_ind in indicators]
                  if layout['draw'] == 'indicator' else [(None, grid, '')])
        for ind, group, suffix in groups:
            fig = Plot_grid(group, layout, [ind] if ind else indicators, regressors, regr_sym_D, tgt_sym_D, title=title)
            if fig is not None:
                _Out(fig, os.path.join(plot_dir, 'grid', '%s%s.png' % (layout['name'], suffix.replace('/', '-'))),
                     True, save_layout)

    def _Predict_independent(self):
        p = self.process_S.process.parameters

        project_root_fp = self._Path(str(p.project_root_fp).strip())
        if not os.path.exists(project_root_fp):
            print('    ERROR: project_root_fp not found: %s' % project_root_fp)
            return

        dataframe_param = str(getattr(p, 'dataframe', 'raw')).strip()
        model_array     = _parse_array_param(getattr(p, 'model_array', []))
        overwrite       = bool(getattr(p, 'overwrite', False))
        dst_param       = str(getattr(p, 'dst_fp', '') or '').strip()

        if not model_array:
            print('    ERROR: model_array is empty.')
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

        dst_fp = self._Path(dst_param) if dst_param else os.path.join(project_root_fp, 'predictions')
        os.makedirs(dst_fp, exist_ok=True)
        stem   = os.path.splitext(os.path.basename(parquet_fp))[0]
        out_fp = os.path.join(dst_fp, stem + '_pred.parquet')

        if os.path.exists(out_fp) and not overwrite:
            print('    %s already exists. Use overwrite=True to rerun.' % os.path.basename(out_fp))
            return

        # The models refit their own preprocessing — preprocessed spectra (same band names!) would give wrong values
        changed = [st.get('process') for st in (_load_companion_json(os.path.dirname(parquet_fp), stem) or {})
                   .get('preprocessing_chain', []) if st.get('process') not in INDICATOR_STEPS]
        if changed:
            print('    ERROR: %s holds preprocessed spectra (%s); give the unprocessed spectra, the models '
                  'apply their own preprocessing.' % (os.path.basename(parquet_fp), ', '.join(changed)))
            return

        df, units_D = Read_parquet_with_units(parquet_fp)
        df = df.reset_index(drop=True)

        if self.verbose >= 1:
            print('    Loaded %d rows from %s' % (len(df), os.path.basename(parquet_fp)))

        out_df    = df[[c for c in df.columns if not _is_spectral_col(c)]].copy()
        out_units = {c: u for c, u in units_D.items() if c in out_df.columns}
        models_L  = []
        panels    = []   # observed/predicted scatter panels of the models with independent metrics

        for model_param in model_array:

            try:
                registered_sha = None
                if Is_path(model_param):
                    model_fp = self._Path(model_param)
                else:
                    model_fp, registered_sha = Registered_model(self.pg_session_C, model_param)
                model, manifest = Load_model(model_fp, self.verbose)
                if registered_sha and manifest['model_sha256'] != registered_sha:
                    raise ValueError('the model file is not the one registered as "%s"' % model_param)
                pred = Predict_spectra(model, manifest, df, units_D)
            except ValueError as e:
                print('    ERROR: %s: %s' % (model_param, e))
                continue

            indicator = manifest['indicator']
            col = '%s__%s_%s' % (indicator, manifest['regressor'], 'tt' if manifest['mode'] == 'traintest' else 'kf')
            out_df[col] = pred
            if manifest.get('indicator_unit'):
                out_units[col] = manifest['indicator_unit']

            model_D = {'column': col, 'model_fp': model_fp, 'model_sha256': manifest['model_sha256'],
                       'manifest_fp': Manifest_fp(model_fp), 'indicator': indicator,
                       'regressor': manifest['regressor'], 'mode': manifest['mode'],
                       'training_metrics': manifest.get('metrics', {})}

            # Independent validation: observed values of samples not used for training
            if indicator in df.columns and pd.api.types.is_numeric_dtype(df[indicator]):
                key = manifest['data'].get('sample_id_key')
                train_ids = set(map(str, manifest['data'].get('train_sample_ids', [])))
                in_train = (df[key].astype(str).isin(train_ids).values if key in df.columns
                            else np.zeros(len(df), dtype=bool))
                obs  = df[indicator].values.astype(float)
                eval_rows = ~np.isnan(obs) & ~np.isnan(pred) & ~in_train
                if in_train.any():
                    print('    ⚠️  %s: %d sample(s) were used for training, excluded from the metrics'
                          % (col, int(in_train.sum())))
                if eval_rows.sum() >= 2:
                    metrics = _compute_metrics(obs[eval_rows], pred[eval_rows])
                    metrics['n'] = int(eval_rows.sum())
                    model_D['independent_metrics'] = metrics
                    mode = 'tt' if manifest['mode'] == 'traintest' else 'kf'
                    panels.append({'type': 'scatter', 'mode': mode, 'indicator': indicator,
                                   'regressor': manifest['regressor'], 'obs': obs[eval_rows],
                                   'pred': pred[eval_rows], 'metrics': metrics,
                                   'label': 'independent, %s model' % mode})
                    Record_independent_metrics(self.pg_session_C, manifest['model_sha256'], parquet_fp, metrics, self.verbose)
                    if self.verbose >= 1:
                        print('    %s  independent: n=%d  r2=%.3f  rmse=%.3f  rpiq=%.3f'
                              % (col, metrics['n'], metrics['r2'], metrics['rmse'], metrics['rpiq']))

            elif self.verbose >= 1:
                print('    %s  predicted %d samples' % (col, int((~np.isnan(pred)).sum())))

            models_L.append(model_D)

        if not models_L:
            print('    ERROR: no model could be applied.')
            return

        Save_parquet_with_units(out_df, out_fp, out_units)

        with open(os.path.join(dst_fp, stem + '_pred.json'), 'w') as fh:
            json.dump({'source_parquet': parquet_fp, 'source_sha256': File_sha256(parquet_fp),
                       'n_samples': len(out_df), 'models': models_L}, fh, indent=2, default=str)

        Register_parquet(out_fp, 'predict_independent', '%d model(s) applied to %d samples' % (len(models_L), len(out_df)))

        print('    Predictions saved: %s' % out_fp)

        self._Plot_observed_predicted(panels, os.path.join(dst_fp, 'plot'), stem)
