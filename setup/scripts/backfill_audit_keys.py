"""One-off script: backfill the inline "audit" key onto every create_table block.

Reads the existing audit_triggers_<schema>_v10_sql.json files as ground truth for
which tables are audited and with which events, and falls back to the CLAUDE.md
schema-group convention (full INSERT/UPDATE/DELETE for community/process/utility/
observation_utility/landscape_utility; UPDATE/DELETE only for observation/landscape)
for any create_table block with no matching ground-truth entry.

Usage:
    python backfill_audit_keys.py            # dry run, report only
    python backfill_audit_keys.py --apply    # write the changes

Run from anywhere; paths are resolved relative to this script's location.
"""
import argparse
import glob
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_ROOT = os.path.join(SCRIPT_DIR, "..", "zzz", "ai4sh", "setup_db", "json_ai4sh")
AUDIT_DIR = os.path.join(JSON_ROOT, "audit")

EVENT_ORDER = ["INSERT", "UPDATE", "DELETE"]
FULL_IUD_SCHEMAS = {"community", "process", "utility", "observation_utility", "landscape_utility"}
UD_ONLY_SCHEMAS = {"observation", "landscape"}
SKIP_TABLES = {("audit", "logged_actions")}


def Load_ground_truth():
    ground_truth_D = {}
    for FPN in sorted(glob.glob(os.path.join(AUDIT_DIR, "audit_triggers_*_v10_sql.json"))):
        try:
            data = json.load(open(FPN))
        except json.JSONDecodeError as e:
            print("  ! skipping unparsable %s: %s" % (FPN, e))
            continue
        for p in data.get("process", []):
            if p.get("process_id") != "create_trigger":
                continue
            params = p["parameters"]
            key = (params["schema"], params["table"])
            events_D = {e: (e in params["events"]) for e in EVENT_ORDER}
            ground_truth_D[key] = events_D
    return ground_truth_D


def Desired_audit_for(schema, table, ground_truth_D):
    key = (schema, table)
    if key in SKIP_TABLES:
        return None
    if key in ground_truth_D:
        return ground_truth_D[key]
    if schema in FULL_IUD_SCHEMAS:
        return {"INSERT": True, "UPDATE": True, "DELETE": True}
    if schema in UD_ONLY_SCHEMAS:
        return {"INSERT": False, "UPDATE": True, "DELETE": True}
    print("  ? unknown schema group %r for table %s.%s, skipping" % (schema, schema, table))
    return None


def Backfill_file(FPN, ground_truth_D, apply, report_L):
    try:
        data = json.load(open(FPN))
    except json.JSONDecodeError as e:
        report_L.append((FPN, "SKIP (invalid JSON: %s)" % e))
        return
    changed = False
    for p in data.get("process", []):
        if p.get("process_id") != "create_table":
            continue
        params = p["parameters"]
        schema, table = params["schema"], params["table"]
        desired = Desired_audit_for(schema, table, ground_truth_D)
        if desired is None:
            continue
        existing = p.get("audit")
        if existing == desired:
            continue
        note = "add %s.%s -> %s" % (schema, table, desired) if existing is None \
            else "update %s.%s -> %s (was %s)" % (schema, table, desired, existing)
        report_L.append((FPN, note))
        p["audit"] = desired
        changed = True
    if changed and apply:
        with open(FPN, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    ground_truth_D = Load_ground_truth()
    print("Loaded %d ground-truth trigger definitions from %s\n" % (len(ground_truth_D), AUDIT_DIR))

    report_L = []
    for FPN in sorted(glob.glob(os.path.join(JSON_ROOT, "**", "*.json"), recursive=True)):
        if os.path.commonpath([os.path.abspath(FPN), os.path.abspath(AUDIT_DIR)]) == os.path.abspath(AUDIT_DIR):
            continue  # don't touch the audit_triggers_*.json files themselves
        Backfill_file(FPN, ground_truth_D, args.apply, report_L)

    for FPN, note in report_L:
        print("%s: %s" % (os.path.relpath(FPN, JSON_ROOT), note))
    print("\n%d change(s) across %d file(s)%s" % (
        len(report_L), len({f for f, _ in report_L}),
        "" if args.apply else " (dry run, nothing written -- pass --apply to write)"))


if __name__ == "__main__":
    main()
