"""
 @file setup_db_audit.py

 @brief Automatic audit-trigger config assembly.

 @details Reads the inline "audit": {"INSERT":bool,"UPDATE":bool,"DELETE":bool}
 key that create_table blocks may carry, and uses it to assemble and
 sync-to-disk the per-schema audit_triggers_<schema>_v10_sql.json config,
 plus a pilot file (db_xspatula_ai4sh_audit.txt) listing every file that
 needs to run to bring the database's audit triggers in line with that
 config. Pure file I/O - no database connection, no DDL, safe to call
 unconditionally on every setup_db.ipynb run.

 Deliberately disentangled from actually applying the triggers to a database:
 this module only ever writes JSON/text config files. Running the generated
 pilot file (job_setup_audit.json -> db_xspatula_ai4sh_audit.txt) through
 Initiate_audit() in setup_db_initiate.py - a separate, optional notebook
 cell - is what executes the CREATE TRIGGER calls for real, via the same
 Setup_schemas_tables dispatcher the main setup pass already uses. That
 dispatcher's own create_table/create_trigger overwrite semantics are what
 make repeated runs idempotent; this module never talks to Postgres and so
 never needs to reason about DB state, bootstrap order, or a "does the audit
 system already exist" check.

 A create_table block with no "audit" key (or an all-false one) is simply not
 audited - opt-in, no implicit schema-group fallback at runtime. The
 CLAUDE.md schema-group convention (full INSERT/UPDATE/DELETE for catalog
 schemas, UPDATE/DELETE only for bulk-data schemas) is only used once, by
 setup/scripts/backfill_audit_keys.py, to seed the initial "audit" key on
 existing tables.

 *Version History*:
 - Created: 2026-08-18
 - Updated: 2026-08-19 (Disentangled from DB execution - pure file assembly,
   applied separately via Initiate_audit)

 @author Thomas Gumbricht

 @date Created: 2026-08-18
 @date Updated: 2026-08-19
"""

# Standard library imports
from os import path
from collections import Counter

# Package application imports
from src.utils import Read_json, Dump_json

EVENT_ORDER = ["INSERT", "UPDATE", "DELETE"]

BOOTSTRAP_FILES = ["audit_table_v10_sql.json", "audit_function_v10_sql.json", "audit_triggers_audit_v10_sql.json"]

AUDIT_PILOT_FILE = "db_xspatula_ai4sh_audit.txt"

# audit.if_modified_func() inserts into audit.logged_actions on every trigger
# fire; an INSERT trigger on that same table would make that internal insert
# re-fire itself, recursing infinitely. Never allow it, however it's declared.
RECURSION_UNSAFE_S = {("audit", "logged_actions")}


def _Audit_dir(json_process_file_FPN_L):
    """
    @brief Derive the audit/ config directory from the pilot list's own paths.

    @details Every process JSON file in the pilot list lives one directory
    level under the json_ai4sh root (e.g. .../json_ai4sh/observation/x.json) -
    true for every existing collection per CLAUDE.md. The json_ai4sh root is
    therefore each entry's grandparent directory; audit/ is a sibling of the
    per-schema subfolders under that root.

    @param json_process_file_FPN_L List of absolute process JSON file paths.

    @return Absolute path to the audit/ config directory, or None if the list is empty.
    """

    if not json_process_file_FPN_L:

        return None

    grandparent_L = [path.dirname(path.dirname(FPN)) for FPN in json_process_file_FPN_L]

    # Almost always a single common grandparent; if a future file ever breaks
    # the one-level-deep convention, fall back to the most common one rather
    # than failing outright.
    json_ai4sh_FP = Counter(grandparent_L).most_common(1)[0][0]

    return path.join(json_ai4sh_FP, 'audit')


def _Collect_table_audit_state(json_process_file_FPN_L, verbose):
    """
    @brief Scans every create_table block in the pilot list for its "audit" key.

    @details Re-reads the same pilot-list file paths Setup_schemas_tables
    already resolved, rather than a fresh directory scan - guarantees this
    pass never sees a table that isn't actually reachable from the pilot list.

    @return Tuple (found_S, deleted_S, desired_D):
    - found_S: set of every (schema, table) seen via any create_table block.
    - deleted_S: subset of found_S whose create_table block has delete:true.
    - desired_D: {(schema, table): [event, ...]} for every table with a
      non-empty "audit" key and delete not set.
    """

    found_S = set()

    deleted_S = set()

    desired_D = {}

    for FPN in json_process_file_FPN_L:

        if not path.exists(FPN):

            continue

        data = Read_json(FPN)

        if not data or 'process' not in data:

            continue

        for p in data['process']:

            if p.get('process_id') != 'create_table':

                continue

            params_D = p.get('parameters', {})

            schema, table = params_D.get('schema'), params_D.get('table')

            if not schema or not table:

                continue

            key = (schema, table)

            found_S.add(key)

            if p.get('delete'):

                deleted_S.add(key)

                continue

            audit_D = p.get('audit')

            if not audit_D:

                continue

            if key in RECURSION_UNSAFE_S and audit_D.get('INSERT'):

                print(' ⚠️ ignoring INSERT audit on %s.%s (would recurse)' % key)

                audit_D = dict(audit_D, INSERT=False)

            events_L = [e for e in EVENT_ORDER if audit_D.get(e)]

            if not events_L:

                continue

            desired_D[key] = events_L

    return found_S, deleted_S, desired_D


def _Trigger_file_FPN(audit_dir_FP, schema):

    return path.join(audit_dir_FP, 'audit_triggers_%s_v10_sql.json' % schema)


def _Build_trigger_block(schema, table, events_L):

    return {
        "process_id": "create_trigger",
        "overwrite": True,
        "delete": False,
        "parameters": {
            "schema": schema,
            "table": table,
            "trigger": "%s_audit" % table,
            "timing": "AFTER",
            "events": events_L,
            "function": "audit.if_modified_func"
        }
    }


def _Merge_schema_trigger_file(audit_dir_FP, schema, found_S, deleted_S, desired_D, dry_run, verbose):
    """
    @brief Loads, diffs and (unless dry_run) rewrites one schema's
    audit_triggers_<schema>_v10_sql.json - add/update/prune only, byte-for-byte
    untouched otherwise.

    @details Comparison is by parsed-dict equality, not text/byte diff, since
    Dump_json's formatting won't match some hand-written files. A block is
    pruned if: its table was deleted, its table is no longer found anywhere in
    the pilot list, or - covering legacy multi-schema files such as the
    original audit_triggers_landscape_v10_sql.json, which held both landscape
    and landscape_utility - it belongs to a different schema than this file's
    own name says it should.

    @note "changed" only ever gates whether this file gets rewritten on disk
    (to avoid pointless git churn) - it must never gate whether the resulting
    trigger config gets *applied*. Every schema with any desired table always
    belongs in the generated audit pilot file, changed or not: applying is
    Initiate_audit()'s job, and create_trigger's own overwrite/exists-check
    semantics already make repeat application idempotent. A "only apply what
    changed" shortcut would leave a fresh database with zero triggers on any
    schema whose file already happened to be correct on disk.

    @return Tuple (merged_D, changed, added_L, updated_L, removed_L).
    """

    FPN = _Trigger_file_FPN(audit_dir_FP, schema)

    on_disk_D = Read_json(FPN) if path.exists(FPN) else None

    if not on_disk_D:

        on_disk_D = {"process": []}

    by_key_D = {}

    for block in on_disk_D.get('process', []):

        params_D = block.get('parameters', {})

        by_key_D[(params_D.get('schema'), params_D.get('table'))] = block

    added_L, updated_L, removed_L = [], [], []

    for key in list(by_key_D):

        if key[0] != schema or key in deleted_S or key not in found_S:

            removed_L.append(key)

            del by_key_D[key]

    for (s, t), events_L in desired_D.items():

        if s != schema:

            continue

        block = _Build_trigger_block(s, t, events_L)

        if (s, t) not in by_key_D:

            by_key_D[(s, t)] = block

            added_L.append((s, t))

        elif by_key_D[(s, t)] != block:

            by_key_D[(s, t)] = block

            updated_L.append((s, t))

    merged_D = {"process": list(by_key_D.values())}

    changed = bool(added_L or updated_L or removed_L)

    if changed and not dry_run:

        Dump_json(FPN, merged_D, indent=2, verbose=verbose)

    return merged_D, changed, added_L, updated_L, removed_L


def _Write_audit_pilot_file(setup_db_dir_FP, schema_L, dry_run, verbose):
    """
    @brief Writes db_xspatula_ai4sh_audit.txt - the pilot file a separate,
    optional notebook cell (Initiate_audit) runs to actually apply the audit
    config assembled by this module.

    @details Always lists the 3 bootstrap files (audit_table, audit_function,
    audit_triggers_audit) unconditionally, followed by one line per schema in
    schema_L - every schema with at least one currently-audited table, not
    just the ones whose file changed this run (see _Merge_schema_trigger_file's
    note on why "changed" must never gate application). Regenerated in full on
    every call; this file is a build artifact, not hand-maintained.

    @return bool True if written (or would be, under dry_run).
    """

    FPN = path.join(setup_db_dir_FP, AUDIT_PILOT_FILE)

    line_L = [
        "# AUTO-GENERATED by setup_db_audit.Assemble_audit_config - do not hand-edit.",
        "# Regenerated on every run of the main setup_db.ipynb cell. Run this file's",
        "# contents via a separate, optional cell (job_setup_audit.json) any time",
        "# after that - see Initiate_audit() in setup_db_initiate.py.",
        "",
        "audit/audit_table_v10_sql.json",
        "audit/audit_function_v10_sql.json",
        "audit/audit_triggers_audit_v10_sql.json",
        "",
    ]

    line_L += ["audit/audit_triggers_%s_v10_sql.json" % schema for schema in schema_L]

    if verbose:

        print('.   [dry run] would write audit pilot file:\n     %s' % FPN if dry_run else
              '.   Writing audit pilot file:\n     %s' % FPN)

    if dry_run:

        return True

    with open(FPN, 'w') as f:

        f.write("\n".join(line_L) + "\n")

    return True


def Assemble_audit_config(json_process_file_FPN_L, dry_run=False, verbose=0):
    """
    @brief Assembles the audit-trigger config for every create_table block
    that declares an inline "audit" key: writes the per-schema
    audit_triggers_<schema>_v10_sql.json files and the audit pilot file that
    applies them.

    @details Pure file I/O - opens no database connection and runs no DDL.
    Intended to be called from Initiate_database right after
    Setup_schemas_tables, on every setup_db.ipynb run, so the generated
    config is always in sync with the current table definitions regardless of
    where in the pilot list a table was added or edited. Applying the result
    for real is a separate step - see Initiate_audit() in setup_db_initiate.py.

    @param json_process_file_FPN_L List of absolute process JSON file paths
    (same list already passed to Setup_schemas_tables).
    @param dry_run If True, computes and reports the diff but writes nothing
    to disk.
    @param verbose Verbosity level.

    @return {schema: {"changed": bool, "added": [...], "updated": [...],
    "removed": [...]}} per schema with any currently-desired audited table,
    or None if the audit/ config directory couldn't be located.
    """

    audit_dir_FP = _Audit_dir(json_process_file_FPN_L)

    if not audit_dir_FP or not path.isdir(audit_dir_FP):

        print('❌ ERROR - could not locate the audit/ config directory, skipping audit config assembly')

        return None

    found_S, deleted_S, desired_D = _Collect_table_audit_state(json_process_file_FPN_L, verbose)

    schema_S = {s for s, t in found_S} | {s for s, t in desired_D}

    report_D = {}

    for schema in sorted(schema_S):

        merged_D, changed, added_L, updated_L, removed_L = _Merge_schema_trigger_file(
            audit_dir_FP, schema, found_S, deleted_S, desired_D, dry_run, verbose)

        report_D[schema] = {"changed": changed, "added": added_L, "updated": updated_L, "removed": removed_L}

        if verbose and changed:

            print('.   audit[%s]: +%d ~%d -%d' % (schema, len(added_L), len(updated_L), len(removed_L)))

    schema_with_triggers_L = sorted({s for s, t in desired_D})

    setup_db_dir_FP = path.dirname(path.dirname(audit_dir_FP))

    _Write_audit_pilot_file(setup_db_dir_FP, schema_with_triggers_L, dry_run, verbose)

    return report_D
