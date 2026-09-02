"""
 @file setup_db_initiate.py

 @brief Database setup orchestration module.

 @details Coordinates setup notebook execution, resolves configuration paths,
 manages output folders, and prompts users before destructive database actions.

 *Version History*:
 - Created: 2025-09-02
 - Updated: 2026-03-14 (Code cleanup)
 - Updated: 2026-03-15 (Added prompts for user confirmation before critical operations)

 @author Thomas Gumbricht

 @date Created: 2025-09-02
 @date Updated: 2026-03-14 (Code cleanup)
 @date Updated: 2026-03-15 (Added prompts for user confirmation before critical operations)
 @date Updated: 2026-08-19 (Added audit initiation from main database setup)
"""

# Standard library imports
from os import path, makedirs

from shutil import rmtree

import traceback

# Package application imports
from setup.src_setup.lib_setup.setup_db import Setup_prod_roles

from setup.src_setup.lib_setup.setup_db_audit import Assemble_audit_config

from src.lib import Full_path_locate, Get_scheme_project_path_setup

from src_setup.lib_setup import Setup_prod_DB, Setup_schemas_tables

def Get_user_response_boolean(prompt_text):
        """
        @brief Prompt user for yes/no confirmation with formatted output.
        
        This method displays a formatted prompt showing the options
        (yes/continue, no/quit) and waits for user input. Used throughout
        the device designation workflow to request user confirmation before critical operations.
        
        @param prompt_text (str): The text describing the action requiring user confirmation
        
        @return str: User's input response (typically 'y' for yes or 'n' for no)
        """
        print ("\n⚠️    ✅  %s?: y\n      ❌ To skip/quit: n\n" % prompt_text)

        return input("⚠️ %s (y); stop(n): "% prompt_text)

def Delete_db_environments(notebook_FP):
    """
    @brief Delete generated database environment files.

    @details Resolves the package environment directory relative to the calling
    notebook and removes it recursively when it exists.

    @param notebook_FP Full path to the notebook driving the setup workflow.

    @return None
    """

    environment_path = path.join(path.split(notebook_FP)[0], 'src','postgres','environment')

    if path.exists(environment_path):
        
        rmtree(environment_path)

def Create_db_environment_dot(notebook_FP, postgresDB_D):
    """
    @brief Create per pg_user database environment files for database access.

    @details Creates or recreates the PostgreSQL environment directory and writes one `.env`
    file per configured database pg_user with connection settings from the scheme.

    @param notebook_FP Full path to the notebook driving the setup workflow.
    @param postgresDB_D Dictionary containing database connection settings and
    configured database users.

    @return None
    """
  
    environment_path = path.join(path.split(notebook_FP)[0], 'src','postgres','environment')

    # Delete any existing environment path and create a new one with the .env files for each user in the database configuration
    Delete_db_environments(notebook_FP)

    makedirs(environment_path)

    # Loop over the users defined in the database configuration and create a .env file for each user with the database connection parameters
    for item in postgresDB_D['db_users']:

        user_id = item['user_id']

        password = item['password']

        dot_env_file = path.join(environment_path, f'.{user_id}.env')

        with open(dot_env_file, 'w') as f:

            f.write(f"DB_NAME={postgresDB_D['db']}\n")
            f.write(f"DB_USER={user_id}\n")
            f.write(f"DB_PASSWORD={password}\n")
            f.write(f"DB_HOST={postgresDB_D['host']}\n")
            f.write(f"DB_PORT={postgresDB_D['port']}\n")

    print(f'\n. Database environment files created in: {environment_path}')

def Initiate_database(notebook_FP,scheme_file, proj_proc_file):
    """
    Initiate the process by loading the scheme file and the process file.
    
    Parameters:
    - scheme_file: Path to the scheme file.
    - process_file: Path to the process file.
    
    Returns:
    Nothing. The function will print the status of the process.
    If the process file does not exist, it will print an error message.
    If the process file exists, it will load the user default parameters and process files,
    run the job processes loop, and manage the process.
    If the process is completed, it will print 'Done'.
    """

    print ('Scheme file:', scheme_file)

    # Locate the scheme file 
    # The following arguments are:
    # 'None'. the scheme file has no parent
    # scheme_file: the path to the scheme file
    # False: do not create path if not found
    # notebook_FP: the full path name of the notebook running this function
    # The path to the scheme file can be given as an absolute path or relative to the notebook path
    scheme_file = Full_path_locate(None, scheme_file, False, notebook_FP)
    # The existence of the scheme file is tested in Get_scheme_project_path_setup() 
    success = Get_scheme_project_path_setup( scheme_file, proj_proc_file )

    if not success:

        return None

    scheme_params_D, json_process_file_FPN_L = success

    if scheme_params_D['process'][0]['delete']:

        # The user is about to delete the database, so ask for confirmation before proceeding
        print ("\n.   ⚠️  Confirm that you want to delete parts of or the whole the database.\n")

        confirm = Get_user_response_boolean("Delete database %s?" % scheme_params_D['postgresdb']['db'])

        if confirm.lower() != 'y':

            return None
        
    else:

        if scheme_params_D['process'][0]['verbose'] > 0:

            print ("\n.   ⚠️  Confirm that you want to set up the database.\n")

            confirm = Get_user_response_boolean("Set up database %s?" % scheme_params_D['postgresdb']['db'])

            if confirm.lower() != 'y':

                return None

        print ('\n. Setting up system production database: %s' %(scheme_params_D['postgresdb']['db']))

        # Create the environment files for the database users, which will be used to connect to the database in the subsequent steps. This is done before setting up the database itself, so that the environment files are available for any database setup scripts that may need to connect to the database during setup.
        Create_db_environment_dot(notebook_FP, scheme_params_D['postgresdb'])

        # Set up the database itself
        success = Setup_prod_DB(scheme_params_D)

        if not success:

            #Delete any existing environment files if the scheme file cannot be loaded, to avoid confusion with old environment files if the scheme file is later fixed and reloaded
            Delete_db_environments(notebook_FP)

            return None
    
    if json_process_file_FPN_L:

        # Set up the schemas and tables
        Setup_schemas_tables(scheme_params_D, json_process_file_FPN_L)
        
        if not scheme_params_D['process'][0]['delete']:

            # Assemble and sync-to-disk (no DDL, no DB connection) the audit-
            # trigger config for every create_table block that declares an
            # inline "audit" key, now that every table across the whole pilot
            # list has been created/updated. This only writes config files -
            # run Initiate_audit() separately (a distinct, optional notebook
            # cell) to actually apply them to the database.
            try:
                Assemble_audit_config(json_process_file_FPN_L, verbose=scheme_params_D['process'][0]['verbose'])
            except Exception as e:
                traceback.print_exc()

            # Create and GRANT ROLES after the schemas and tables have been created, so that the roles can be granted permissions on the schemas and tables during their creation
            try:
                Setup_prod_roles(scheme_params_D)
            except Exception as e:
                traceback.print_exc()

    else:

        print('⚠️ No process to run')

def Initiate_audit(notebook_FP, scheme_file, proj_proc_file):
    """
    @brief Applies the audit-trigger config assembled by Assemble_audit_config.

    @details Lighter than Initiate_database: no "set up database?" prompt, no
    environment-file rewrite, no role grants - those don't belong to an
    audit-only pass. Just resolves the audit job/pilot file (job_setup_audit.json
    -> db_xspatula_ai4sh_audit.txt, regenerated by the main setup cell's own
    Assemble_audit_config call) and runs it through the same Setup_schemas_tables
    dispatcher the main setup pass uses, which is what actually issues the
    CREATE TRIGGER calls. Intended to be run from a separate, optional
    notebook cell, any time after the main "Setup database" cell.

    @param notebook_FP Full path to the notebook driving the setup workflow.
    @param scheme_file Path to the scheme file (same one used for the main setup).
    @param proj_proc_file Job file naming the audit pilot file, e.g. job_setup_audit.json.

    @return None
    """

    print('Scheme file:', scheme_file)

    scheme_file = Full_path_locate(None, scheme_file, False, notebook_FP)

    success = Get_scheme_project_path_setup(scheme_file, proj_proc_file)

    if not success:

        return None

    scheme_params_D, json_process_file_FPN_L = success

    if not json_process_file_FPN_L:

        print('⚠️ No audit process to run')

        return None

    print('\n. Applying audit-trigger config to database: %s' % (scheme_params_D['postgresdb']['db']))

    Setup_schemas_tables(scheme_params_D, json_process_file_FPN_L)