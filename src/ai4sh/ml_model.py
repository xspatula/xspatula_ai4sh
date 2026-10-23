'''
Created on 23 October 2026

@author: thomasgumbricht

Registry of final machine learning models (database schema ml_model).

register_model takes a model saved by regression_modeling (<name>.joblib + <name>.manifest.json) and writes

    ml_model.model                path + sha256 of the model file, indicator, regressor, mode, input, manifest
    ml_model.model_step           the preprocessing chain the model refits (one row per step)
    ml_model.model_parameter_set  the ml_setting parameter sets used to build the model
    ml_model.model_metric         the test (train/test) or k-fold metrics

and freezes the parameter sets it used (ml_setting.parameter_set.frozen; database triggers then block changing or
deleting them, so the record of how the model was made can not change afterwards — an edit becomes a new version).

The parameter sets are taken from the provenance of the modelling run (manifest) and of every preprocessing step that
produced the input parquet (.previous_dataframe of the data folder). Settings read from files are not in ml_setting;
their content and sha256 are kept in the manifest.

A registered model is not changed or replaced; register the new model under another name. retire_model marks a model
as retired (it can no longer be applied by name; its record and frozen parameter sets stay). predict_independent
adds the metrics on independent samples of a registered model to ml_model.model_metric (dataset independent:<parquet>).
'''

# Standard library imports
import os
import json

from src.ai4sh.ml_setting import Is_path, SESSION_USER_SQL
from src.ai4sh.provenance import File_sha256
from src.ai4sh.previous_dataframe import Read_previous_history

SET_KEY_PREFIX = 'db:ml_setting.parameter_set/'

METRIC_SKIP = ('n_train', 'n_test', 'n', 'model_sha256', 'indicator_transformation')


def _Read_manifest(model_fp):

    manifest_fp = os.path.splitext(model_fp)[0] + '.manifest.json'

    if not os.path.exists(model_fp):
        raise ValueError('model not found: %s' % model_fp)

    if not os.path.exists(manifest_fp):
        raise ValueError('no manifest for %s (saved before Part 3) — rerun regression_modeling' % os.path.basename(model_fp))

    with open(manifest_fp) as fh:
        manifest = json.load(fh)

    if File_sha256(model_fp) != manifest.get('model_sha256'):
        raise ValueError('%s does not match the sha256 in its manifest' % os.path.basename(model_fp))

    return manifest


def _Set_ids(provenance):
    '''{set_id} of the ml_setting parameter sets recorded in a provenance snapshot.'''

    return {c['set_id'] for k, c in (provenance or {}).get('configs', {}).items()
            if k.startswith(SET_KEY_PREFIX) and c.get('set_id') is not None}


def Model_parameter_sets(manifest):
    '''[(set_id, used_by)] of the parameter sets used by the modelling run and the preprocessing steps behind it.'''

    used = [(i, 'regression_modeling') for i in sorted(_Set_ids(manifest.get('provenance')))]

    data = manifest.get('data', {})

    directory = data.get('directory')

    source = os.path.splitext(data.get('source_parquet') or '')[0]

    target = os.path.splitext(data.get('input_parquet') or '')[0]

    if not (directory and source and target and os.path.isdir(directory)):
        return used

    # Latest record per parquet of the chain source → input (each step adds a suffix to the stem)
    latest = {}

    for rec in Read_previous_history(directory):

        for entry in rec.get('entries', []):

            stem = os.path.splitext(str(entry.get('dataframe') or ''))[0]

            if stem.startswith(source) and target.startswith(stem):
                latest[stem] = rec

    for stem, rec in sorted(latest.items()):

        process = (rec.get('provenance') or {}).get('process') or rec.get('process')

        used += [(i, process) for i in sorted(_Set_ids(rec.get('provenance')))]

    return list(dict.fromkeys(used))


def Register_model(pg_session_C, model_fp, name='', abstract='', verbose=1):
    '''Register a saved model; returns the model id (None if it was not registered, reason printed).'''

    pg = pg_session_C

    try:
        manifest = _Read_manifest(model_fp)
    except ValueError as e:
        print('    ERROR: %s' % e)
        return None

    name = (name or os.path.splitext(os.path.basename(model_fp))[0]).strip().lower()

    rec = pg._Execute_search_single_sql("SELECT name FROM ml_model.model WHERE model_sha256 = %s;", (manifest['model_sha256'],))

    if rec:
        print('    %s is already registered as "%s"' % (os.path.basename(model_fp), rec[0]))
        return None

    if pg._Execute_search_single_sql("SELECT id FROM ml_model.model WHERE name = %s;", (name,)):
        print('    ERROR: a different model is already registered as "%s" — choose another name' % name)
        return None

    rec = pg._Execute_search_single_sql(
        "SELECT id FROM observation_utility.indicator WHERE alias = %s OR name = %s ORDER BY (alias = %s) DESC LIMIT 1;",
        (manifest['indicator'], manifest['indicator'], manifest['indicator']))

    if not rec:
        print('    ERROR: indicator "%s" of %s is not in observation_utility.indicator' % (manifest['indicator'], name))
        return None

    indicator_id = rec[0]

    data, inp = manifest.get('data', {}), manifest.get('input', {})

    metrics = manifest.get('metrics', {})

    sets = Model_parameter_sets(manifest)

    cur = pg.cursor

    try:
        cur.execute(
            "INSERT INTO ml_model.model (name, indicator_id, regressor, mode, signal_type, n_bands, n_train, n_test, "
            "model_fp, model_sha256, source_parquet, source_sha256, manifest, abstract, registered_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, " + SESSION_USER_SQL + ") RETURNING id;",
            (name, indicator_id, manifest['regressor'], manifest['mode'], inp.get('signal_type'), inp.get('n_bands'),
             metrics.get('n_train', data.get('n_train')), metrics.get('n_test'), os.path.abspath(model_fp),
             manifest['model_sha256'], data.get('source_parquet'), data.get('source_sha256'),
             json.dumps(manifest, default=str), abstract or None))

        model_id = cur.fetchone()[0]

        for i, step in enumerate(manifest.get('preprocessing_chain', []), 1):
            cur.execute("INSERT INTO ml_model.model_step (model_id, step_no, step_id, process, parameters) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb);",
                        (model_id, step.get('step', i), step.get('id'), step.get('process'),
                         json.dumps(step.get('parameters', {}), default=str)))

        dataset = 'test' if manifest['mode'] == 'traintest' else 'kfold'

        n = metrics.get('n_test') if dataset == 'test' else metrics.get('n_train')

        for metric, value in metrics.items():
            if metric not in METRIC_SKIP and isinstance(value, (int, float)) and not isinstance(value, bool):
                cur.execute("INSERT INTO ml_model.model_metric (model_id, dataset, metric, value, n) "
                            "VALUES (%s, %s, %s, %s, %s);", (model_id, dataset, metric, float(value), n))

        frozen = []

        for set_id, used_by in sets:

            cur.execute("SELECT family, name, version FROM ml_setting.parameter_set WHERE id = %s;", (set_id,))

            row = cur.fetchone()

            if row is None:
                print('    ⚠️  parameter set id %s (used by %s) no longer exists — not linked' % (set_id, used_by))
                continue

            cur.execute("INSERT INTO ml_model.model_parameter_set (model_id, set_id, used_by) VALUES (%s, %s, %s) "
                        "ON CONFLICT DO NOTHING;", (model_id, set_id, used_by))

            cur.execute("UPDATE ml_setting.parameter_set SET frozen = TRUE WHERE id = %s AND NOT frozen;", (set_id,))

            frozen.append('%s/%s@%s' % row)

        pg.conn.commit()

    except Exception:
        pg.conn.rollback()
        raise

    if verbose >= 1:
        print('    ✅ registered "%s" (id %s): %s %s, %s; %d step(s); parameter sets frozen: %s'
              % (name, model_id, manifest['indicator'], manifest['regressor'], manifest['mode'],
                 len(manifest.get('preprocessing_chain', [])), ', '.join(sorted(set(frozen))) or 'none (settings from files)'))

    return model_id


def Registered_model(pg_session_C, name):
    '''(model_fp, model_sha256) of a registered model (status registered); ValueError if there is none.'''

    rec = pg_session_C._Execute_search_single_sql(
        "SELECT model_fp, model_sha256, status FROM ml_model.model WHERE name = %s;", (str(name).strip().lower(),))

    if not rec:
        raise ValueError('no registered model named "%s"' % name)

    if rec[2] != 'registered':
        raise ValueError('model "%s" is %s' % (name, rec[2]))

    return rec[0], rec[1]


def Retire_model(pg_session_C, name, verbose=1):
    '''Set a registered model to retired; True if it was changed.'''

    pg = pg_session_C

    pg.cursor.execute("UPDATE ml_model.model SET status = 'retired' WHERE name = %s AND status = 'registered' RETURNING id;",
                      (str(name).strip().lower(),))

    changed = pg.cursor.fetchone() is not None

    pg.conn.commit()

    if verbose >= 1:
        print('    %s' % ('✅ model "%s" retired' % name if changed else '"%s" is not a registered model (unknown or already retired)' % name))

    return changed


def Record_independent_metrics(pg_session_C, model_sha256, parquet_name, metrics, verbose=1):
    '''Write metrics on independent samples for the registered model with this sha256 (no-op if not registered or
    the registry is not set up). Rerunning with the same parquet replaces the values. Never raises.'''

    pg = pg_session_C

    if pg is None:
        return False

    try:
        if not pg._Execute_search_single_sql("SELECT to_regclass('ml_model.model') IS NOT NULL;")[0]:
            return False

        rec = pg._Execute_search_single_sql("SELECT id, name FROM ml_model.model WHERE model_sha256 = %s;", (model_sha256,))

        if not rec:
            return False

        dataset = 'independent:%s' % os.path.splitext(os.path.basename(parquet_name))[0]

        for metric, value in metrics.items():
            if metric not in METRIC_SKIP and isinstance(value, (int, float)) and not isinstance(value, bool):
                pg.cursor.execute(
                    "INSERT INTO ml_model.model_metric (model_id, dataset, metric, value, n) VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (model_id, dataset, metric) DO UPDATE SET value = EXCLUDED.value, n = EXCLUDED.n;",
                    (rec[0], dataset, metric, float(value), metrics.get('n')))

        pg.conn.commit()

        if verbose >= 1:
            print('    independent metrics added to the registry for "%s" (%s)' % (rec[1], dataset))

        return True

    except Exception as e:
        try:
            pg.conn.rollback()
        except Exception:
            pass
        print('    ⚠️  independent metrics not written to the registry: %s' % e)
        return False


class Process_ml_model:
    '''Processes of the root process ml_model.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP

    def _Sub_process(self, _):

        if self.process_S.process.process == 'register_model':
            self._Register_model()

        elif self.process_S.process.process == 'retire_model':

            from src.ai4sh.machine_learning_preprocess import _parse_array_param

            for name in _parse_array_param(getattr(self.process_S.process.parameters, 'model_array', [])):
                Retire_model(self.pg_session_C, name, self.verbose)

    def _Register_model(self):

        from src.lib.pilot import Get_project_path
        from src.ai4sh.machine_learning_preprocess import _parse_array_param

        p = self.process_S.process.parameters

        model_array = _parse_array_param(getattr(p, 'model_array', []))

        name = str(getattr(p, 'name', '') or '').strip()

        abstract = str(getattr(p, 'abstract', '') or '').strip()

        if not model_array:
            print('    ERROR: model_array is empty.')
            return

        if name and len(model_array) > 1:
            print('    ERROR: name can only be given when registering a single model.')
            return

        for model_param in model_array:

            s = str(model_param).strip()

            model_fp = s if os.path.isabs(s) or not Is_path(s) else Get_project_path(self.project_root_FP, s)

            Register_model(self.pg_session_C, os.path.expanduser(model_fp), name, abstract, self.verbose)
