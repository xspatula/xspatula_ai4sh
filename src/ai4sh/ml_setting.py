'''
Created on 23 October 2026

@author: thomasgumbricht

Machine learning parameter sets from the database schema ml_setting.

A parameter set is one named, versioned group of settings of one family (transformation, standardisation,
filter, multi_filter, selector, regressor, spectra_plot, plot_symbol, plot_layout), in one of three
categories:

    raw     library defaults / no operation          (name "none")
    preset  curated settings                          (e.g. "standard", "savitzky-golay", "targetfeaturesymbols")
    custom  settings defined by a user (stratum 4-5)

ml_setting.parameter_set.content holds the settings in the same JSON shape as the files that used to be in
<project>/default/. For the per-indicator families (transformation, standardisation) the content is the seed
template keyed by indicator name or alias; the settings actually used are the rows of
ml_setting.indicator_transformation / indicator_standardisation, created by sync_indicator_parameter_sets.

A process parameter that names settings is resolved by Load_settings():

    "" / "none" / "no"     → None (the caller's no-op / library defaults)
    "default"              → the preset named default_name (normally "standard")
    "<name>[@<version>]"   → that parameter set (latest version if none given)
    path (contains "/", "~" or ends with .json) → the JSON file

Every set loaded from the database is recorded in the provenance (src/ai4sh/provenance.py) with its id, version
and content hash. Until the ml_setting schema exists the default files are used (one warning).

New indicators (Fill_new_indicators, run by Run_process after manage_indicator) get a row in every transformation /
standardisation set and a grey placeholder symbol ("needs_review": true) in every plot_symbol set of target features.

Custom sets (create_parameter_set, user stratum 4-5) start from an existing set (base_set) and apply the changes of a
settings JSON file in the same shape as the default files. A name that already exists gets a new version; raw and
preset names are reserved. Loading a name without @version gives its latest version.
'''

# Standard library imports
import os
import json
import hashlib

from src.ai4sh.provenance import Load_config_json, Record_config

TARGET_TRANSFORMS = ('linear', 'log', 'sqrt', 'reciprocal', 'boxcox', 'yeojohnson', 'quantile')

INDICATOR_FAMILIES = {'transformation': 'transformation', 'standardisation': 'standardise'}

SYNC_FAMILIES = ('transformation', 'standardisation', 'plot_symbol')

FAMILIES = ('transformation', 'standardisation', 'filter', 'multi_filter', 'selector', 'regressor',
            'spectra_plot', 'plot_symbol', 'plot_layout')

RESERVED_NAMES = ('', 'none', 'no', 'default')

# community.user.user_name of the logged-in individual (session GUC set at login), else the database role
SESSION_USER_SQL = ("COALESCE((SELECT user_name FROM community.user "
                    "WHERE id = NULLIF(current_setting('audit.app_user_id', true), '')::INTEGER), current_user)")

TARGET_SYMBOLS_KEY = 'targetFeatureSymbols'

PLACEHOLDER_SYMBOL = {'color': 'grey', 'alpha': 0.2, 'unit': 'native', 'needs_review': True}

_SESSION = {'pg': None, 'ready': None, 'warned': False}


def Set_settings_session(pg_session_C):
    '''Database session used to read parameter sets (set by Run_process).'''

    _SESSION.update(pg=pg_session_C, ready=None)


def _Db_ready():
    '''True if there is a session and the ml_setting schema exists (checked once per session).'''

    if _SESSION['ready'] is None:

        pg = _SESSION['pg']

        try:
            _SESSION['ready'] = bool(pg is not None and pg._Execute_search_single_sql(
                "SELECT to_regclass('ml_setting.parameter_set') IS NOT NULL;")[0])

        except Exception:
            _Rollback()
            _SESSION['ready'] = False

    return _SESSION['ready']


def _Rollback():
    try:
        _SESSION['pg'].conn.rollback()
    except Exception:
        pass


def Content_hash(content):
    '''sha256 of the content as stored in parameter_set.content_hash (sorted keys, compact JSON).'''

    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def Is_path(param):
    '''True if a settings parameter is a file path rather than a parameter set name.'''

    s = str(param).strip()

    return '/' in s or s.startswith('~') or s.lower().endswith('.json')


def Get_parameter_set(family, name, version=None):
    '''Return the parameter set row as a dict (latest version unless version is given), or None.'''

    sql = ("SELECT id, family, name, version, category, frozen, content, content_hash "
           "FROM ml_setting.parameter_set WHERE family = %s AND name = %s")

    params = [family, name.lower()]

    if version is not None:
        sql += " AND version = %s"
        params.append(int(version))

    rec = _SESSION['pg']._Execute_search_single_sql(sql + " ORDER BY version DESC LIMIT 1;", params)

    if rec is None:
        return None

    return dict(zip(('id', 'family', 'name', 'version', 'category', 'frozen', 'content', 'content_hash'), rec))


def _Indicator_content(row, verbose=0):
    '''Content in the file shape ({"transformation": {indicator: {transform: true}}} or {"standardise": {indicator: bool}})
    built from the per-indicator rows of the set, keyed by both indicator name and alias.'''

    pg = _SESSION['pg']

    if row['family'] == 'transformation':

        recs = pg._Execute_search_all_sql(
            "SELECT i.name, i.alias, t.name FROM ml_setting.indicator_transformation x "
            "JOIN observation_utility.indicator i ON i.id = x.indicator_id "
            "JOIN ml_setting.transform t ON t.id = x.transform_id WHERE x.set_id = %s;", (row['id'],))

        values = {r[:2]: ({} if r[2] == 'linear' else {r[2]: True}) for r in recs}

    else:

        recs = pg._Execute_search_all_sql(
            "SELECT i.name, i.alias, x.standardise FROM ml_setting.indicator_standardisation x "
            "JOIN observation_utility.indicator i ON i.id = x.indicator_id WHERE x.set_id = %s;", (row['id'],))

        values = {r[:2]: bool(r[2]) for r in recs}

    key = INDICATOR_FAMILIES[row['family']]

    if not values:

        if verbose >= 1 and row['category'] != 'raw':
            print('    ⚠️  %s set "%s" has no per-indicator rows — using its template; run sync_indicator_parameter_sets'
                  % (row['family'], row['name']))

        return row['content'] or {key: {}}

    out = {}

    for (name, alias), value in values.items():

        for k in (name, alias):
            if k:
                out[k] = value

    return {key: out}


def Load_settings(family, param, default_fpn=None, default_name='standard', resolve_path=None, verbose=1):
    '''Return the settings named by param (see module docstring) as a dict, or None for none.

    default_fpn: file used when the database is not available (transition from <project>/default/).
    resolve_path: function resolving a relative file path (e.g. against the project root).
    Raises ValueError for an unknown parameter set, FileNotFoundError for a missing file.
    '''
    s = str(param if param is not None else '').strip()

    if s.lower() in ('', 'none', 'no'):
        return None

    if Is_path(s):

        fpn = resolve_path(s) if resolve_path else os.path.expanduser(s)

        if not os.path.exists(fpn):
            raise FileNotFoundError('settings file not found: %s' % fpn)

        return Load_config_json(fpn)

    name, _, version = s.lower().partition('@')

    if name == 'default':
        name = default_name

    if not _Db_ready():

        if default_fpn and os.path.exists(default_fpn):

            if not _SESSION['warned'] and verbose >= 1:
                print('    ⚠️  ml_setting schema not available — using the default settings files')
                _SESSION['warned'] = True

            return Load_config_json(default_fpn)

        raise ValueError('%s parameter set "%s" requested, but the ml_setting schema is not available' % (family, s))

    try:
        row = Get_parameter_set(family, name, version or None)
    except Exception:
        _Rollback()
        raise

    if row is None:
        raise ValueError('no %s parameter set named "%s" in ml_setting.parameter_set' % (family, s))

    if family in INDICATOR_FAMILIES:
        content, sha256 = _Indicator_content(row, verbose), None   # built from the rows: hash what is used
    else:
        content, sha256 = row['content'] or {}, row['content_hash']

    Record_config('db:ml_setting.parameter_set/%s/%s@%s' % (family, row['name'], row['version']),
                  content, sha256, set_id=row['id'], category=row['category'])

    return content


def Spectra_colormap(param, verbose=1):
    '''Colour ramp for plotting spectra: the given matplotlib name, or the spectra_plot set for "" / "default".'''

    s = str(param if param is not None else '').strip()

    if s and s.lower() != 'default':
        return s

    try:
        return (Load_settings('spectra_plot', 'default', verbose=verbose) or {}).get('colormap', 'jet')
    except ValueError:
        return 'jet'


# ------------------------------------------------------------------ per-indicator rows

def Sync_indicator_parameter_sets(pg_session_C, families=('transformation', 'standardisation'),
                                  overwrite=False, verbose=1, set_ids=None):
    '''Create the per-indicator rows of every transformation / standardisation set.

    Each indicator gets the value of the set's template (looked up by indicator name or alias), otherwise
    linear / no standardisation. Existing rows are kept unless overwrite is True. Frozen sets (used by a registered
    model) only get rows for indicators that have none — a new indicator does not change what a model used.
    Returns {set name: number of rows written}.
    '''
    pg = pg_session_C

    indicators = pg._Execute_search_all_sql("SELECT id, name, alias FROM observation_utility.indicator;")

    transform_ids = dict(pg._Execute_search_all_sql("SELECT name, id FROM ml_setting.transform;"))

    sets = pg._Execute_search_all_sql(
        "SELECT id, family, name, version, frozen, content FROM ml_setting.parameter_set "
        "WHERE family = ANY(%s) ORDER BY family, name, version;", (list(families),))

    if set_ids is not None:
        sets = [s for s in sets if s[0] in set_ids]

    written = {}

    for set_id, family, name, version, frozen, content in sets:

        label = '%s/%s@%s' % (family, name, version)

        if frozen and overwrite and verbose >= 1:
            print('    %s is frozen (used by a registered model) — only rows for new indicators are added' % label)

        rewrite = overwrite and not frozen

        key = INDICATOR_FAMILIES[family]

        template = {str(k).lower(): v for k, v in ((content or {}).get(key, {}) or {}).items()}

        table = 'indicator_transformation' if family == 'transformation' else 'indicator_standardisation'

        existing = {r[0] for r in pg._Execute_search_all_sql(
            "SELECT indicator_id FROM ml_setting.%s WHERE set_id = %%s;" % table, (set_id,))}

        used_keys = set()

        n = 0

        for ind_id, ind_name, ind_alias in indicators:

            if ind_id in existing and not rewrite:
                continue

            match = next((k for k in (str(ind_name or '').lower(), str(ind_alias or '').lower()) if k in template), None)

            if match:
                used_keys.add(match)

            value = template.get(match) if match else None

            if family == 'transformation':

                tname = next((t for t in TARGET_TRANSFORMS[1:] if (value or {}).get(t)), 'linear')

                pg.cursor.execute(
                    "INSERT INTO ml_setting.indicator_transformation (set_id, indicator_id, transform_id) "
                    "VALUES (%s, %s, %s) ON CONFLICT (set_id, indicator_id) "
                    "DO UPDATE SET transform_id = EXCLUDED.transform_id;",
                    (set_id, ind_id, transform_ids[tname]))

            else:

                pg.cursor.execute(
                    "INSERT INTO ml_setting.indicator_standardisation (set_id, indicator_id, standardise) "
                    "VALUES (%s, %s, %s) ON CONFLICT (set_id, indicator_id) "
                    "DO UPDATE SET standardise = EXCLUDED.standardise;",
                    (set_id, ind_id, bool(value)))

            n += 1

        pg.conn.commit()

        written[label] = n

        if (verbose >= 1 and n) or verbose >= 2:

            print('    %s: %d indicator row(s) written (%d already present)' % (label, n, len(existing) if not rewrite else 0))

            unmatched = sorted(set(template) - used_keys - {str(i[1]).lower() for i in indicators} - {str(i[2]).lower() for i in indicators})

            if unmatched and verbose >= 2:
                print('        template keys without an indicator in the database: %s' % ', '.join(unmatched))

    return written


def Fill_placeholder_plot_symbols(pg_session_C, verbose=1):
    '''Add a grey placeholder symbol ("needs_review": true) for every indicator that has no entry (by name or alias)
    in a not frozen plot_symbol set of target features. Returns {set label: [indicator keys added]}.'''

    pg = pg_session_C

    indicators = pg._Execute_search_all_sql("SELECT name, alias, display_name FROM observation_utility.indicator ORDER BY name;")

    sets = pg._Execute_search_all_sql(
        "SELECT id, name, version, frozen, content FROM ml_setting.parameter_set "
        "WHERE family = 'plot_symbol' ORDER BY name, version;")

    added_D = {}

    for set_id, name, version, frozen, content in sets:

        symbols = (content or {}).get(TARGET_SYMBOLS_KEY)

        if not isinstance(symbols, dict):
            continue  # e.g. regressionmodelsymbols (keyed by regressor, not indicator)

        label = 'plot_symbol/%s@%s' % (name, version)

        known = {str(k).lower() for k in symbols}

        missing = [(n, a, d) for n, a, d in indicators
                   if str(n or '').lower() not in known and str(a or '').lower() not in known]

        if not missing:
            continue

        if frozen:
            if verbose >= 1:
                print('    %s is frozen (used by a registered model) — %d indicator(s) without symbol not added'
                      % (label, len(missing)))
            continue

        added = []

        for n, a, d in missing:
            key = str(a or n).lower()
            symbols[key] = dict(PLACEHOLDER_SYMBOL, label=d or a or n)
            added.append(key)

        pg.cursor.execute("UPDATE ml_setting.parameter_set SET content = %s::jsonb, content_hash = %s WHERE id = %s;",
                          (json.dumps(content), Content_hash(content), set_id))
        pg.conn.commit()

        added_D[label] = added

        if verbose >= 1:
            print('    ⚠️  %s: grey placeholder symbol added for %d indicator(s) — set colour and label: %s'
                  % (label, len(added), ', '.join(added)))

    return added_D


def Fill_new_indicators(pg_session_C, verbose=1):
    '''Give indicators without ml settings their per-indicator rows and placeholder plot symbols.

    Called by Run_process after manage_indicator. Never raises: a failure here must not hide that the indicator
    itself was inserted (run sync_indicator_parameter_sets to repeat).'''

    pg = pg_session_C

    try:
        if not pg._Execute_search_single_sql("SELECT to_regclass('ml_setting.parameter_set') IS NOT NULL;")[0]:
            return None

        return {'rows': Sync_indicator_parameter_sets(pg, verbose=verbose),
                'plot_symbols': Fill_placeholder_plot_symbols(pg, verbose)}

    except Exception as e:

        try:
            pg.conn.rollback()
        except Exception:
            pass

        print('    ⚠️  ml settings for new indicators not filled (%s) — run sync_indicator_parameter_sets' % e)

        return None


# ------------------------------------------------------------------ custom sets

def _Merge(base, change):
    '''base with change applied: nested dicts merged, anything else replaced.'''

    out = dict(base)

    for k, v in change.items():
        out[k] = _Merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v

    return out


def _Indicator_change(pg, family, change):
    '''Per-indicator content keyed by alias (indicator name or alias accepted); ValueError for unknown indicators
    or values.'''

    key = INDICATOR_FAMILIES[family]

    values = change.get(key, change) if isinstance(change, dict) else None

    if not isinstance(values, dict):
        raise ValueError('%s settings must be {"%s": {indicator: value}}' % (family, key))

    alias_of = _Alias_map(pg)

    out, unknown = {}, []

    for k, v in values.items():

        if str(k).lower() not in alias_of:
            unknown.append(k)
            continue

        if family == 'transformation':

            v = v or {}

            if not isinstance(v, dict) or len([t for t, on in v.items() if on]) > 1 or \
                    any(t not in TARGET_TRANSFORMS for t in v):
                raise ValueError('%s: transformation must be {} or one of {"%s": true}'
                                 % (k, '": true}, {"'.join(TARGET_TRANSFORMS[1:])))

            v = {t: True for t, on in v.items() if on and t != 'linear'}

        elif not isinstance(v, bool):
            raise ValueError('%s: standardise must be true or false' % k)

        out[alias_of[str(k).lower()]] = v

    if unknown:
        raise ValueError('not indicators in observation_utility.indicator: %s' % ', '.join(map(str, unknown)))

    return out


def Create_parameter_set(pg_session_C, family, name, base_set='standard', change=None, abstract='', verbose=1):
    '''Save base_set with change applied as custom set family/name (next version if the name exists).

    Returns (set_id, version). Raises ValueError for an invalid family, name, base set or change.
    '''
    pg = pg_session_C

    family, name = str(family).strip().lower(), str(name).strip().lower()

    change = change or {}

    if family not in FAMILIES:
        raise ValueError('family must be one of %s' % ', '.join(FAMILIES))

    if name in RESERVED_NAMES or '@' in name or Is_path(name):
        raise ValueError('"%s" can not be used as a set name' % name)

    if not isinstance(change, dict):
        raise ValueError('the settings must be a JSON object')

    rec = pg._Execute_search_single_sql(
        "SELECT category, max(version) FROM ml_setting.parameter_set WHERE family = %s AND name = %s GROUP BY category;",
        (family, name))

    if rec and rec[0] != 'custom':
        raise ValueError('%s/%s is a %s set — choose another name' % (family, name, rec[0]))

    version = rec[1] + 1 if rec else 1

    # Base content
    base_name, _, base_version = str(base_set or 'none').strip().lower().partition('@')

    base_name = 'standard' if base_name == 'default' else base_name

    base_row = Get_parameter_set(family, base_name or 'none', base_version or None)

    if base_row is None:
        raise ValueError('no %s parameter set named "%s"' % (family, base_set))

    if family in INDICATOR_FAMILIES:

        key = INDICATOR_FAMILIES[family]

        alias_of = _Alias_map(pg)

        # One key per indicator (its alias), so a change given by name or by alias replaces the base value
        base_values = {alias_of[str(k).lower()]: v for k, v in _Indicator_content(base_row, 0).get(key, {}).items()
                       if str(k).lower() in alias_of}

        changed = _Indicator_change(pg, family, change)

        content = {key: dict(base_values, **changed)}

    else:

        changed = change

        content = _Merge(base_row['content'] or {}, change)

    pg.cursor.execute(
        "INSERT INTO ml_setting.parameter_set (family, name, version, category, creator, abstract, content, content_hash) "
        "VALUES (%s, %s, %s, 'custom', " + SESSION_USER_SQL + ", %s, %s::jsonb, %s) RETURNING id;",
        (family, name, version, abstract or 'based on %s/%s@%s' % (family, base_row['name'], base_row['version']),
         json.dumps(content), Content_hash(content)))

    set_id = pg.cursor.fetchone()[0]

    pg.conn.commit()

    if family in INDICATOR_FAMILIES:
        Sync_indicator_parameter_sets(pg, [family], verbose=0, set_ids={set_id})

    if verbose >= 1:
        print('    ✅ custom set %s/%s@%s created from %s/%s@%s (changed: %s)'
              % (family, name, version, family, base_row['name'], base_row['version'], ', '.join(map(str, changed)) or 'nothing'))

    return set_id, version


def _Alias_map(pg):
    '''{lowercased indicator name or alias: lowercased alias (name if no alias)}.'''

    out = {}

    for name, alias in pg._Execute_search_all_sql("SELECT name, alias FROM observation_utility.indicator;"):
        for k in (name, alias):
            if k:
                out[str(k).lower()] = str(alias or name).lower()

    return out


def List_parameter_sets(pg_session_C, family=None):
    '''Print the parameter sets (optionally of one family).'''

    recs = pg_session_C._Execute_search_all_sql(
        "SELECT family, name, version, category, frozen, coalesce(creator, ''), coalesce(abstract, '') "
        "FROM ml_setting.parameter_set WHERE %s IS NULL OR family = %s ORDER BY family, name, version;",
        (family, family))

    print('    %-16s %-24s %3s  %-7s %-6s %-14s %s' % ('family', 'name', 'v', 'category', 'frozen', 'creator', 'abstract'))

    for f, n, v, c, fr, cr, ab in recs:
        print('    %-16s %-24s %3s  %-7s %-6s %-14s %s' % (f, n, v, c, 'yes' if fr else '', cr[:14], ab[:60]))

    return recs


class Process_ml_setting:
    '''Processes of the root process ml_setting.'''

    def __init__(self, process_S, pg_session_C, project_root_FP):
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP

    def _Sub_process(self, _):

        if self.process_S.process.process == 'sync_indicator_parameter_sets':

            p = self.process_S.process.parameters

            from src.ai4sh.machine_learning_preprocess import _parse_array_param

            families = _parse_array_param(getattr(p, 'family_array', [])) or list(SYNC_FAMILIES)

            unknown = [f for f in families if f not in SYNC_FAMILIES]

            if unknown:
                print('    ERROR: family_array can only hold %s, got %s' % (list(SYNC_FAMILIES), unknown))
                return

            row_families = [f for f in families if f in INDICATOR_FAMILIES]

            if row_families:
                Sync_indicator_parameter_sets(self.pg_session_C, row_families,
                                              bool(getattr(p, 'rebuild', False)), self.verbose)

            if 'plot_symbol' in families:
                Fill_placeholder_plot_symbols(self.pg_session_C, self.verbose)

        elif self.process_S.process.process == 'create_parameter_set':

            self._Create_parameter_set()

        elif self.process_S.process.process == 'list_parameter_sets':

            family = str(getattr(self.process_S.process.parameters, 'family', '') or '').strip().lower()

            List_parameter_sets(self.pg_session_C, family or None)

    def _Create_parameter_set(self):

        from src.lib.pilot import Get_project_path

        p = self.process_S.process.parameters

        settings_fp = str(getattr(p, 'settings_fp', '') or '').strip()

        change = {}

        if settings_fp:

            fpn = os.path.expanduser(settings_fp)

            if not os.path.isabs(fpn):
                fpn = Get_project_path(self.project_root_FP, settings_fp)

            if not os.path.exists(fpn):
                print('    ERROR: settings_fp not found: %s' % fpn)
                return

            with open(fpn) as fh:
                change = json.load(fh)

        try:
            Create_parameter_set(self.pg_session_C, getattr(p, 'family', ''), getattr(p, 'name', ''),
                                 getattr(p, 'base_set', 'standard'), change,
                                 str(getattr(p, 'abstract', '') or '').strip(), self.verbose)

        except ValueError as e:
            self.pg_session_C.conn.rollback()
            print('    ERROR: %s' % e)
