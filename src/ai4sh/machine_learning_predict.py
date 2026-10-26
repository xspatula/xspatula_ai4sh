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
observed values of the indicator, the metrics are computed on the samples that were not used for training.
'''

# Standard library imports
import os
import json

import numpy as np
import pandas as pd
import joblib

from importlib import metadata

from src.postgres import Get_schema_table
from src.lib.pilot import Get_project_path
from src.ai4sh.parquet_units import Read_parquet_with_units, Save_parquet_with_units
from src.ai4sh.previous_dataframe import Register_parquet
from src.ai4sh.provenance import File_sha256
from src.ai4sh.spectral_columns import Signal_type_of
from src.ai4sh.machine_learning_preprocess import (_resolve_input_parquet, _resolve_single_previous,
                                                   _is_spectral_col, _parse_array_param, _load_companion_json)
from src.ai4sh.machine_learning_model import Manifest_fp, _compute_metrics
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

    return issues


def Predict_spectra(model, manifest, spectra_df, units_D=None):
    '''Predict the indicator (native unit) for every row of spectra_df (unprocessed spectra, model bands).'''

    issues = Check_spectra(manifest, spectra_df.columns, units_D)

    if issues:
        raise ValueError('; '.join(issues))

    return np.ravel(model.predict(spectra_df[manifest['input']['columns']]))


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
                   .get('preprocessing_chain', []) if st.get('process') != 'detect_outliers']
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
