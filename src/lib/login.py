"""
 @file login.py

 @brief User authentication module for process database access.

 @details Validates project login input, authenticates the active user against
 PostgreSQL, and returns user status information required by later processing
 steps.

 *Version History*:
 - Created: 2025-01-05
 - Updated: 2025-09-03 (Documentation added)

 @author Thomas Gumbricht

 @date Created: 2025-01-05
 @date Updated: 2025-09-03 (Documentation added)
"""

LOGIN_EVALUATION = 'login_evaluation'

# Third party imports
from base64 import b64encode

# Package application imports
from src.postgres import PG_user_status, PG_session

def Get_set_database_session(scheme_params_D):

    """
    @brief Initializes and returns the user status and PostgreSQL database session.

    This function performs the following steps:
    - Logs in the user and retrieves their status using the provided default parameters.
    - Updates the default parameters dictionary with the user status.
    - Initializes a PostgreSQL session object with the database name and verbosity level.
    - Returns the user status dictionary and the PostgreSQL session object.

    @param scheme_params_D Dictionary containing default parameters, including database and process information.
    @return Tuple (user_status_D, pg_session_C):
        - user_status_D: Dictionary with user status information, or None if login fails.
        - pg_session_C: PostgreSQL session object, or None if login fails.
    """
    # Get user status
    user_status_D = Project_login(scheme_params_D)

    if not user_status_D:

        return None, None

    # Set the login environment variable for the user, this is used by the db session to determine which environement (provedge level) the user is in and thus which db to connect to
    if 'stratum_code' in user_status_D:

        dot_env_var = 'user_cat_%s' %(user_status_D['stratum_code'])
  
    # Set user status
    scheme_params_D['user_status'] = user_status_D

    # Login to the database with the user assigned stratum and get a session object.
    # app_user_id carries the individual community.user.id into the session (as the
    # audit.app_user_id GUC) so audit.if_modified_func can record who - not just which
    # shared stratum role - made a change.
    try:
        #pg_session_C = PG_session(scheme_params_D['postgresdb']['db'], scheme_params_D['process'][0]['verbose'])
        pg_session_C = PG_session(dot_env_var, scheme_params_D['process'][0]['verbose'],
                                   app_user_id=user_status_D.get('id'))

    except Exception as e:
        print('❌ ERROR - Could not connect to Postgres server, exiting')
        print('    Please check if the Postgres server is running and that you have access rights')
        print('    db:', scheme_params_D['postgresdb']['db'])
        print('    error:', e)
        return None, None

    return user_status_D, pg_session_C

def Project_login(default_parameter_D):
    """
    @brief Logs in to the postgre database project and sets the user status.

    @details This function authenticates the user against the database using the provided default parameters.
    It checks for the presence of required user project information and retrieves the user status
    from the database. If authentication fails or required information is missing, it prints a warning
    and returns None.

    @param default_parameter_D Dictionary containing default parameters, including user credentials and project info.
    @return user_status_D Dictionary with user status information (id, email, name, stratum, etc.) if successful; None otherwise.

    @warning Prints warnings and returns None if login fails, user project info is missing, or user is unrecognized.
    """

    if not 'user_project' in default_parameter_D:

        print ('❌ ERROR: object <user_project> missing in user_project_file')

        return None

    if 'user_netrc_id' in default_parameter_D['user_project'] and \
        len(default_parameter_D['user_project']['user_netrc_id']) > 0:
        
        rec = PG_user_status(LOGIN_EVALUATION, default_parameter_D['user_project']['user_netrc_id'])

    elif 'user_name' in default_parameter_D['user_project'] and 'password' in default_parameter_D['user_project']:

        rec = PG_user_status(LOGIN_EVALUATION,
                             None,
                             default_parameter_D['user_project']['user_name'],
                             b64encode(default_parameter_D['user_project']['password'].encode()))
        
    else:

        print ('❌ ERROR: no user project info - exiting')
        print ('You must either state a user_netrc_id or a user_name and user_password under user_project in the default parameters')

        return None
    
    if not rec:

        print ('❌ ERROR: unrecognised user (%s) - exiting' %(default_parameter_D['user_project']['user_name']))

        return None
    # TG TODO - this param is duplicated in the PG_user_status function - should be defined in one place only
    params = ['id', 'email', 'first_name', 'middle_name', 'last_name', 'user_name', 'stratum_code', 'status_code']

    user_status_D = dict(zip(params, rec))

    return user_status_D