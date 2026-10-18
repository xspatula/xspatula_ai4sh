"""
 @file dump_restore_db.py

 @brief Full-database dump and restore using pg_dump/pg_restore.

 @details Dumps the complete production database (all schemas, data, SERIAL
 sequence positions, functions, triggers and table privileges) to a single
 custom-format (-Fc) file, and restores such a file into a freshly (re)created
 database. Bypasses re-running the slow setup + import chain (e.g. LUCAS).

 Nothing here is project specific: the database name, credentials and dump file
 location all come from the "postgresdb" object of the scheme file.

 Scheme file keys used (children of "postgresdb"):
 - db, host, port: connection target.
 - host_netrc_id, or user_name + password: credentials (same as Initiate_session).
 - dump_path: the dump file. Relative paths resolve against the project root
   (same rule as every other relative path in the scheme chain); use "~/" or an
   absolute path to keep the dump outside the project. Defaults to
   ~/xspatula/db_dump/<db>.dump when missing.
 - pg_bin_path (optional): folder holding pg_dump/pg_restore, when they are not
   on PATH or in a standard install location.
 - db_users (optional): when present, the per-role .env files are rewritten and
   roles (re)created and granted after a restore, exactly as in the main setup.

 PostgreSQL login roles are cluster-level and not part of a per-database dump.
 On the same server they already exist and their privileges come back with the
 dump; on a new server pg_restore reports GRANT errors for missing roles, which
 the trailing Setup_prod_roles call then creates and grants.

 *Version History*:
 - Created: 2026-10-18

 @author Thomas Gumbricht

 @date Created: 2026-10-18
"""

# Standard library imports
from os import path, makedirs, environ, replace, remove

from base64 import b64decode

from glob import glob

from shutil import which

from time import time

import subprocess

# Third party imports
from psycopg2 import sql as pgsql

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Package application imports
from src.lib import Full_path_locate, Get_project_path

from src.utils import Read_json

from src.postgres import User_netrc_credentials, User_login_pswd

from .setup_db import Initiate_session, Setup_prod_roles, INITIAL_DATABASE_NAME

from .setup_db_initiate import Get_user_response_boolean, Create_db_environment_dot

DEFAULT_DUMP_FOLDER = '~/xspatula/db_dump'

# Searched in order when pg_bin_path is not given and the tools are not on PATH
PG_BIN_CANDIDATE_L = ['/opt/homebrew/opt/postgresql@*/bin',
                      '/usr/local/opt/postgresql@*/bin',
                      '/Applications/Postgres.app/Contents/Versions/latest/bin',
                      '/usr/lib/postgresql/*/bin']

RESTORE_JOBS = 4

def _Load_scheme(notebook_FP, scheme_file):
    """
    @brief Locates and reads the scheme file and resolves the project root.

    @param notebook_FP Full path to the calling notebook (anchor for scheme_file).
    @param scheme_file Path to the scheme file.

    @return scheme_params_D with 'project_root_FP' added, or None on failure.
    """

    scheme_FPN = Full_path_locate(None, scheme_file, False, notebook_FP)

    if not scheme_FPN:

        print('❌ ERROR the scheme file does not exist:', scheme_file)

        return None

    scheme_params_D = Read_json(scheme_FPN)

    if not scheme_params_D or 'postgresdb' not in scheme_params_D:

        print('❌ ERROR the scheme file lacks a <postgresdb> object:', scheme_FPN)

        return None

    scheme_params_D['project_root_FP'] = Get_project_path(path.split(scheme_FPN)[0], scheme_params_D['project_path'])

    return scheme_params_D

def _Dump_FPN(scheme_params_D):
    """
    @brief Resolves the dump file path from postgresdb.dump_path.

    @return Absolute path to the dump file (parent folder not guaranteed to exist).
    """

    postgresDB_D = scheme_params_D['postgresdb']

    dump_path = postgresDB_D.get('dump_path', '%s/%s.dump' % (DEFAULT_DUMP_FOLDER, postgresDB_D['db']))

    return Get_project_path(scheme_params_D['project_root_FP'], dump_path)

def _Pg_tool(scheme_params_D, tool):
    """
    @brief Finds a PostgreSQL client executable (pg_dump or pg_restore).

    @details Order: postgresdb.pg_bin_path, PATH, then PG_BIN_CANDIDATE_L (highest
    version first). pg_dump must be at least the server's major version.

    @return Full path to the executable, or None if not found.
    """

    pg_bin_path = scheme_params_D['postgresdb'].get('pg_bin_path')

    if pg_bin_path:

        tool_FPN = path.join(path.expanduser(pg_bin_path), tool)

        return tool_FPN if path.exists(tool_FPN) else None

    tool_FPN = which(tool)

    if tool_FPN:

        return tool_FPN

    for pattern in PG_BIN_CANDIDATE_L:

        for bin_FP in sorted(glob(pattern), reverse=True):

            tool_FPN = path.join(bin_FP, tool)

            if path.exists(tool_FPN):

                return tool_FPN

    return None

def _Pg_environment(scheme_params_D):
    """
    @brief Builds the environment for pg_dump/pg_restore from the scheme credentials.

    @details The password goes in PGPASSWORD, never on the command line, so it does
    not show up in process listings.

    @return Environment dict, or None if no credentials could be resolved.
    """

    postgresDB_D = scheme_params_D['postgresdb']

    if 'host_netrc_id' in postgresDB_D:

        credential_D = User_netrc_credentials(postgresDB_D['host_netrc_id'])

    elif 'user_name' in postgresDB_D and 'password' in postgresDB_D:

        credential_D = User_login_pswd(postgresDB_D['user_name'], postgresDB_D['password'])

    else:

        credential_D = None

    if not credential_D:

        print('❌ ERROR - No valid user credentials provided for Postgres connection')

        return None

    env_D = dict(environ)

    env_D['PGHOST'] = str(postgresDB_D['host'])

    env_D['PGPORT'] = str(postgresDB_D['port'])

    env_D['PGUSER'] = credential_D['user_name']

    env_D['PGPASSWORD'] = b64decode(credential_D['pswd']).decode('ascii')

    return env_D

def _Prepare(notebook_FP, scheme_file, tool):
    """
    @brief Shared preamble for dump and restore: scheme, dump path, tool, environment.

    @return Tuple (scheme_params_D, dump_FPN, tool_FPN, env_D), or None on failure.
    """

    print('Scheme file:', scheme_file)

    scheme_params_D = _Load_scheme(notebook_FP, scheme_file)

    if not scheme_params_D:

        return None

    tool_FPN = _Pg_tool(scheme_params_D, tool)

    if not tool_FPN:

        print('❌ ERROR - Could not find <%s>; set <pg_bin_path> under <postgresdb> in the scheme file' % (tool))

        return None

    env_D = _Pg_environment(scheme_params_D)

    if not env_D:

        return None

    return scheme_params_D, _Dump_FPN(scheme_params_D), tool_FPN, env_D

def _Database_exists(session, db):
    """
    @brief Checks if a database exists in the cluster.
    """

    session.cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db,))

    return session.cursor.fetchone() is not None

def Dump_database(notebook_FP, scheme_file):
    """
    @brief Dumps the full production database to the file given by postgresdb.dump_path.

    @details Writes to a temporary file first and only replaces an existing dump once
    pg_dump has succeeded, so a failed dump never destroys the previous one.

    @param notebook_FP Full path to the calling notebook.
    @param scheme_file Path to the scheme file (relative to the notebook, ~/ or absolute).

    @return Full path to the dump file on success, None otherwise.
    """

    success = _Prepare(notebook_FP, scheme_file, 'pg_dump')

    if not success:

        return None

    scheme_params_D, dump_FPN, pg_dump_FPN, env_D = success

    db = scheme_params_D['postgresdb']['db']

    verbose = scheme_params_D['process'][0]['verbose']

    makedirs(path.split(dump_FPN)[0], exist_ok=True)

    tmp_FPN = '%s.tmp' % (dump_FPN)

    print('\n. Dumping database %s to:\n    %s' % (db, dump_FPN))

    if verbose > 1:

        print('.   Using:', pg_dump_FPN)

    cmd_L = [pg_dump_FPN, '--format=custom', '--file=%s' % (tmp_FPN), '--dbname=%s' % (db)]

    start = time()

    result = subprocess.run(cmd_L, env=env_D, capture_output=True, text=True)

    if result.returncode != 0:

        print('❌ ERROR - pg_dump failed:\n', result.stderr)

        if path.exists(tmp_FPN):

            remove(tmp_FPN)

        return None

    replace(tmp_FPN, dump_FPN)

    print('.   Done in %.0f s, size %.1f MB' % (time() - start, path.getsize(dump_FPN) / 1e6))

    return dump_FPN

def Restore_database(notebook_FP, scheme_file):
    """
    @brief Restores the production database from the file given by postgresdb.dump_path.

    @details Destructive: an existing database with the same name is dropped (after
    confirmation) and recreated empty before pg_restore runs. Object ownership is
    assigned to the restoring user (--no-owner) so the dump also restores on a
    server with a different superuser name. Table privileges are kept, and if the
    scheme lists db_users, Setup_prod_roles runs afterwards to create any role
    missing from this server.

    @param notebook_FP Full path to the calling notebook.
    @param scheme_file Path to the scheme file (relative to the notebook, ~/ or absolute).

    @return True on success (possibly with reported warnings), None otherwise.
    """

    success = _Prepare(notebook_FP, scheme_file, 'pg_restore')

    if not success:

        return None

    scheme_params_D, dump_FPN, pg_restore_FPN, env_D = success

    db = scheme_params_D['postgresdb']['db']

    if not path.exists(dump_FPN):

        print('❌ ERROR - dump file does not exist:\n    %s' % (dump_FPN))

        return None

    session = Initiate_session(INITIAL_DATABASE_NAME, scheme_params_D)

    if not session:

        return None

    session.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    if _Database_exists(session, db):

        print("\n.   ⚠️  Database %s exists and will be DROPPED and replaced by:\n    %s\n" % (db, dump_FPN))

        confirm = Get_user_response_boolean("Drop and restore database %s?" % (db))

        if confirm.lower() != 'y':

            session._Close()

            return None

        # WITH (FORCE) terminates open connections, e.g. from other notebooks (PostgreSQL 13+)
        session.cursor.execute(pgsql.SQL("DROP DATABASE {} WITH (FORCE);").format(pgsql.Identifier(db)))

    session.cursor.execute(pgsql.SQL("CREATE DATABASE {};").format(pgsql.Identifier(db)))

    session._Close()

    print('\n. Restoring database %s from:\n    %s' % (db, dump_FPN))

    cmd_L = [pg_restore_FPN, '--dbname=%s' % (db), '--no-owner', '--jobs=%s' % (RESTORE_JOBS), dump_FPN]

    start = time()

    result = subprocess.run(cmd_L, env=env_D, capture_output=True, text=True)

    print('.   pg_restore finished in %.0f s' % (time() - start))

    if result.returncode != 0:

        # pg_restore continues past errors (e.g. GRANT to a role missing on this server)
        # and only reports them at the end - show them, then let the roles step fix grants.
        print('⚠️  pg_restore reported errors:\n', result.stderr)

    if 'db_users' in scheme_params_D['postgresdb']:

        print('\n. Creating/granting database roles')

        # Same as the main setup: per-role .env files, then CREATE USER/GRANT for missing roles
        Create_db_environment_dot(notebook_FP, scheme_params_D['postgresdb'])

        Setup_prod_roles(scheme_params_D)

    return True
