"""
 @file pg_session.py

 @brief PostgreSQL session and credential management module.

 @details Handles credential lookup, psycopg2 connections, session logging, and
 user status retrieval for package workflows that interact with PostgreSQL.

 *Version History*:
 - Created: 2018-02-21
 - Updated: 2025-09-09 (Separated common and logging helpers, added documentation)
 - Updated: 2026-03-11 (Parameterized queries and password handling fixes)
 - Updated: 2026-03-15 (Updated PG_user_status)
 - Updated: 2026-08-14 (PG_user_status verifies a bcrypt hash instead of comparing plaintext)

 @author Thomas Gumbricht

 @date Created: 2018-02-21
 @date Updated: 2025-09-09 (Separated common and logging helpers, added documentation)
 @date Updated: 2026-03-11 (Parameterized queries and password handling fixes)
 @date Updated: 2026-03-15 (Updated PG_user_status)
 @date Updated: 2026-08-14 (PG_user_status verifies a bcrypt hash instead of comparing plaintext)
"""

# Standard library imports
from os import path, getenv

import netrc

from base64 import b64encode, b64decode

from dotenv import load_dotenv

# Third party imports
import psycopg2

from psycopg2 import sql as pgsql

# Package application imports
from src.postgres.pg_common import PG_common

from src.utils import Log

from src.community.password import Verify_password

class PG_session(PG_common):
    """
    @brief PostgreSQL session wrapper with connection and logging support.

    @details Combines shared SQL helpers with connection state, cursor handling,
    and lightweight logging used throughout PostgreSQL workflows.
    """

    def __init__(self, environment_dot_file, verbose=0, session_id='unknown', app_user_id=None):
        """
        @brief Connect to PostgreSQL and initialize session parameters.

        @param environment_dot_file Name of the environment file containing connection settings.
        @param verbose Verbosity level for logging and output (default: 0).
        @param session_id Identifier for the session (default: 'unknown').
        @param app_user_id community.user.id of the logged-in individual (default: None). The
            Postgres login role is shared per user stratum (user_cat_0..5), so it cannot identify
            which individual made a change - only audit.if_modified_func's changed_by (current_user)
            can. Setting this GUC lets the trigger also record changed_by_user_id, so per-row
            individual attribution comes from the audit trail instead of needing created_by/updated_by
            columns on every table.

        @details
            - Loads environment variables for the specified database.
            - Establishes a connection and cursor to the PostgreSQL database using psycopg2.
            - If app_user_id is given, sets it as the audit.app_user_id session GUC so
              audit.if_modified_func can attribute changes to the individual, not just the role.
            - Initializes session ID and verbosity.
            - Calls initializers for PG_common and Log classes.
        """

        env_query_D = Get_env_var(environment_dot_file)

        if env_query_D is None:

            raise RuntimeError('❌ ERROR - Could not load environment <%s>' % environment_dot_file)

        self.conn, self.cursor = PG_psycopg2_connect( env_query_D )

        if app_user_id is not None:

            self.cursor.execute("SET SESSION audit.app_user_id = %s", (str(app_user_id),))

        self.session_id = session_id

        self.verbose = verbose

        PG_common.__init__(self)

        self.log = Log

    def _Close(self):
        """
        @brief Close the active cursor and database connection.

        @return None
        """
        self.cursor.close()

        self.conn.close()

def User_netrc_credentials(user_netrc_id):
    """
    @brief Retrieves user credentials from the .netrc file.

    @param user_netrc_id coded host login and password found in user .netrc file

    @note The user_netrc_id must exist in the .netrc file in the users home directory.

    @return Dictionary containing connection user credentials.
    """

    # Retrieve login and password from the .netrc file
    secrets = netrc.netrc()

    # Authenticate username, account and password
    try:
        username, account, password = secrets.authenticators( user_netrc_id )

    except Exception:
        print ('❌ ERROR - Could not retrieve credentials for <%s> from .netrc file' %(user_netrc_id))
        return None

    # Encode the password before sending it
    password = b64encode(password.encode())

    # Create a query dictionary for testing the user credentials
    query_D = {'user_name':username, 'pswd':password}

    return query_D

def User_login_pswd(username, password):
    """
    @brief Build encoded login credentials from an explicit username and password.

    @param username Username to authenticate.
    @param password Plain-text password to encode for transport.

    @note The password is base64-encoded before it is returned in the credential dictionary.

    @return Dictionary containing connection user credentials.
    """

    # Encode the password before sending it
    password = b64encode(password.encode())

    # Create a query dictionary for testing the user credentials
    query_D = {'user_name':username, 'pswd':password}

    return query_D

def PG_psycopg2_connect(env_query_D):
    """
    @brief Establishes a connection to a PostgreSQL database using psycopg2.

    @param env_query_D Dictionary containing connection parameters:
        - db: Database name
        - user: Username
        - pswd: Base64-encoded password
        - port: Port number
        - host: Host address

    @details
        - Decodes the base64-encoded password.
        - Creates a connection and cursor to the PostgreSQL database using psycopg2.

    @return Tuple (conn, cursor):
        - conn: psycopg2 connection object
        - cursor: psycopg2 cursor object
    """
    pg_connection_D = {'dbname': env_query_D['db'],
        'user': env_query_D['user_name'],
        'password': b64decode(env_query_D['pswd']).decode('ascii'),
        'port': env_query_D['port'],
        'host': env_query_D['host']}

    conn = psycopg2.connect(**pg_connection_D)

    # Enable autocommit mode - makes each statement commit automatically
    conn.autocommit = True

    cursor = conn.cursor()

    return conn, cursor

def PG_user_status(environment_dot_file, user_netrc_id, user_name=None, user_pswd=None):
    """
    @brief Retrieves the user status information from the PostgreSQL database.

    @param environment_dot_file Name of the environment file used to connect to PostgreSQL.
    @param user_netrc_id Identifier for user credentials in the .netrc file (optional).
    @param user_name Username for login (optional, used if netrc ID is not provided).
    @param user_pswd Password for login (optional, used if netrc ID is not provided).

    @details
        - Connects to the PostgreSQL server using the provided database name.
        - Retrieves user credentials either from the .netrc file or from explicit parameters.
        - Looks up the row by email or username only (never by password - community.user.password
          holds a bcrypt hash, not a value the database can match with a plain equality check),
          then verifies the supplied plaintext password against the stored hash in Python via
          src.community.password.Verify_password.
        - A row that doesn't exist and a row whose password doesn't match both return None,
          so callers can't distinguish "unknown user" from "wrong password".
        - Closes the database session after the query.

    @return Tuple containing user information (id, email, first_name, middle_name, last_name, user_name, stratum_code, status_code), or None if connection, credentials, or password verification fail.
    """

    try:

        # Connect to the Postgres Server
        session = PG_session(environment_dot_file)

    except Exception as e:

        print ('❌ ERROR - Could not connect to Postgres server, exiting')
        print ('    Please check if the Postgres server is running and that you have access rights')
        print ('    error:', e)

        return None

    if user_netrc_id:

        # Get the user login credentials
        user_login_query_D = User_netrc_credentials(user_netrc_id)

    elif user_name and user_pswd:
        # user login credentials explicitly given in scheme_file
        user_login_query_D = {'user_name':user_name, 'pswd':user_pswd}

    else:
        print ('❌ ERROR - No user credentials given, exiting')

        return None

    if not user_login_query_D or not user_login_query_D['user_name']:

        return None

    # Decode password — kept as a Python value, NEVER embedded in the SQL string
    plaintext_password = b64decode(user_login_query_D['pswd']).decode('ascii')

    select_cols = 'id, email, first_name, middle_name, last_name, user_name, stratum_code, status_code, password'

    if '@' in user_login_query_D['user_name']:

        sql = pgsql.SQL(
            'SELECT {} FROM community.user WHERE email = %s;'
        ).format(pgsql.SQL(select_cols))

    else:

        sql = pgsql.SQL(
            'SELECT {} FROM community.user WHERE user_name = %s;'
        ).format(pgsql.SQL(select_cols))

    session.cursor.execute(sql, (user_login_query_D['user_name'],))

    row = session.cursor.fetchone()

    if row is None or not Verify_password(plaintext_password, row[-1]):

        rec = None

    else:

        # Drop the trailing password hash so the returned shape matches the documented columns
        rec = row[:-1]

    session._Close()

    return rec

def Get_env_var(db):
    """
    @brief Loads environment variables for a PostgreSQL database connection.

    @param db Name of the database. Used to locate the environment file.

    @details
        - Sets the current path to the directory of this file.
        - Constructs the environment filename as .<db>.
        - Checks if the environment file exists in the 'environment' directory.
        - Loads environment variables from the file using dotenv.
        - Retrieves DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD from environment.
        - Encodes the password using base64.

    @return Dictionary with keys 'host', 'port', 'db', 'user', and 'pswd' (base64 encoded password), or None if the environment file does not exist.
    """

    # Set current path to the directory of this file
    BASEDIR = path.abspath(path.dirname(__file__))

    # Set the name of the environment file to the name of the database
    env_FN = '.%s.env' %(db)

    # Check if the environment file exists
    if not path.exists(path.join(BASEDIR, 'environment', env_FN)):

        print ('❌ ERROR: the database environment file <%s> does not exist' %(env_FN))

        return None

    load_dotenv(path.join(BASEDIR, 'environment', env_FN), override=True)

    DB_HOST = getenv("DB_HOST")
    DB_PORT = getenv("DB_PORT")
    DB_NAME = getenv("DB_NAME")
    DB_USER = getenv("DB_USER")
    DB_PASSWORD = getenv("DB_PASSWORD")

    missing = [k for k, v in (('DB_HOST', DB_HOST), ('DB_PORT', DB_PORT),
                               ('DB_NAME', DB_NAME), ('DB_USER', DB_USER),
                               ('DB_PASSWORD', DB_PASSWORD)) if v is None]
    if missing:
        print('❌ ERROR: missing environment variable(s) in <%s>: %s' % (env_FN, ', '.join(missing)))
        return None

    return {'host':DB_HOST, 'port': DB_PORT, 'db':DB_NAME, 'user_name':DB_USER,
            'pswd': b64encode(DB_PASSWORD.encode())}