"""Point this checkout's database environment files at another database.

The code connects through src/postgres/environment/.<role>.env, and every file holds one
DB_NAME. Running setup_db for a database rewrites them all for that database, so after
setting up e.g. the database synthetic, the notebooks of every other project would also talk to synthetic.
Switch back and forth with:

    python setup/switch_db.py synthetic
    python setup/switch_db.py lucas

Roles and passwords are shared by all databases on the server, only DB_NAME changes.
Without an argument it prints the current DB_NAME.
"""

import glob
import os
import re
import sys

ENV_FP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'postgres', 'environment')


def main():
    files = sorted(glob.glob(os.path.join(ENV_FP, '.*.env')))
    if not files:
        sys.exit('No environment files in %s - run setup_db first' % ENV_FP)
    if len(sys.argv) < 2:
        for fpn in files:
            m = re.search(r'^DB_NAME=(.*)$', open(fpn).read(), re.M)
            print('%-24s %s' % (os.path.basename(fpn), m.group(1) if m else '?'))
        return
    db = sys.argv[1]
    if not re.fullmatch(r'[a-z0-9_]+', db):
        sys.exit('Not a database name: %s' % db)
    for fpn in files:
        text = open(fpn).read()
        new = re.sub(r'^DB_NAME=.*$', 'DB_NAME=%s' % db, text, flags=re.M)
        with open(fpn, 'w') as f:
            f.write(new)
    print('%d environment files now use DB_NAME=%s' % (len(files), db))


if __name__ == '__main__':
    main()
