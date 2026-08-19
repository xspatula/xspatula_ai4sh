"""Dry-run verification for the audit-trigger config assembly.

Resolves the same scheme/project/pilot-list chain setup_db.ipynb uses, then
calls Assemble_audit_config(..., dry_run=True): pure file I/O (no database
connection at all), reports what would change on disk, writes nothing.

Usage (from the setup/ directory, same layout as setup_db.ipynb):
    python scripts/verify_audit_system.py [scheme_file] [job_file]

Defaults match setup/setup_db.ipynb's own cell values.
"""
import sys
from os import path

SETUP_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(SETUP_DIR)
sys.path.append(path.dirname(SETUP_DIR))

from src.lib import Get_scheme_project_path_setup
from src_setup.lib_setup import Assemble_audit_config


def main():
    scheme_file = sys.argv[1] if len(sys.argv) > 1 else path.join(SETUP_DIR, 'zzz', 'scheme_ai4sh_local_setup.json')
    job_file = sys.argv[2] if len(sys.argv) > 2 else 'job_setup_db.json'

    success = Get_scheme_project_path_setup(scheme_file, job_file)

    if not success:
        print('❌ ERROR - could not resolve scheme/project/pilot-list chain, aborting')
        sys.exit(1)

    scheme_params_D, json_process_file_FPN_L = success

    report_D = Assemble_audit_config(json_process_file_FPN_L, dry_run=True, verbose=1)

    if report_D is None:
        print('\n. No report - could not locate the audit/ config directory.')
        sys.exit(1)

    any_change = False
    print('\n=== Dry-run audit config assembly report ===')
    for schema, r in sorted(report_D.items()):
        if not r['changed']:
            continue
        any_change = True
        print('%s: +%d ~%d -%d' % (schema, len(r['added']), len(r['updated']), len(r['removed'])))
        for key in r['added']:
            print('    + %s.%s' % key)
        for key in r['updated']:
            print('    ~ %s.%s' % key)
        for key in r['removed']:
            print('    - %s.%s' % key)

    if not any_change:
        print('No differences - audit_triggers_*.json files and the audit pilot file are fully in sync.')


if __name__ == '__main__':
    main()
