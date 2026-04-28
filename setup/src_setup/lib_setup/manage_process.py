"""
 @file manage_process.py
 
 @brief Module for managing process definitions in a PostgreSQL database.
 
 @details Coordinates creation and removal of root and sub-process definitions by
 building query payloads from process settings and delegating database updates to
 the PostgreSQL session manager during setup workflows.
 
 
*Version History*:
 - Created: 2025-01-05
 - Updated: 2025-03-11
 - Updated: 2025-09-02 (Doxygen-style documentation)
 - Updated: 2026-03-14 (Refactoring and code cleanup using Claude Code)
 
 @author Thomas Gumbricht

 @date Created: 2025-01-05
 @date Updated: 2025-03-11
 @date Updated: 2025-09-02 (Doxygen-style documentation)
 @date Updated: 2026-03-14 (Refactoring and code cleanup using Claude Code)
"""

# Standard library imports
from os import path

# Package application imports
from src.postgres import Pg_manage_process

class Process_manage_process():
    """
    @brief Manager for process-related database setup tasks.

    @details Provides the workflow entry points for adding or deleting root and
    sub-process definitions in the PostgreSQL process schema.
    """

    def __init__(self, process_S, process_schema='process', pg_session_C=None):
        """
        @brief Initialize process management for a setup run.

        Initializes the process management class and sets up the PostgreSQL session.
        Call run() to execute the appropriate subprocess handler.

        @param process_S Object containing process parameters and user/project information.
        @param process_schema Name of the process schema to use (default: "process").
        @param pg_session_C Optional pre-existing Pg_manage_process session to reuse.
               If None, a new session is created.

        @details
        - Sets up a PostgreSQL session for process management.
        - Stores process parameters and verbosity settings.

        @return None
        """

        self.pg_session_C = pg_session_C if pg_session_C is not None else Pg_manage_process(process_S)

        self.process_S = process_S

        self.verbose = self.process_S.process.verbose

        self.process_schema = process_schema

    def run(self):
        """
        @brief Dispatches execution to the appropriate subprocess handler.

        Determines which subprocess to execute based on process:
        - "add_root_process": calls _Add_root_process().
        - "add_process": calls process().
        - Otherwise, logs an error for unknown process.

        @return None
        """

        sub = self.process_S.process.process

        if sub == 'add_root_process':

            self._Add_root_process()

        elif sub == 'add_process':

            self._Add_process()

        else:

            error_msg = '❌ ERROR process %s not available in Process_manage_process' % sub

            self.pg_session_C.log(error_msg)
            
    def _Add_root_process(self):
        """
        @brief Adds a root process to the database

        This method constructs a query dictionary containing the root process parameters
        and delegates the insertion or update operation to the PostgreSQL session manager.
        The parameters include the root process ID, title, label, and creator (user ID).
        The operation respects the overwrite and delete flags provided in the process object.

        @details
        - Converts the root process ID to lowercase.
        - Collects title, label, and creator information from the process object.
        - Calls the PostgreSQL session manager's _Manage_root_process method with the query dictionary.
        - Passes overwrite and delete flags to control the database operation.

        @return None
        """

        if self.verbose:
           
            if self.process_S.process.delete:
                
                print('.   Deleting root process: %s' %(self.process_S.process.parameters.root_process))

            elif self.process_S.process.overwrite:
               
                print('.   Overwriting root process: %s' %(self.process_S.process.parameters.root_process))
            else:
            
                print('.   Adding root process: %s' %(self.process_S.process.parameters.root_process))
   
        queryD = {'root_process':self.process_S.process.parameters.root_process.lower(),
                    'title':self.process_S.process.parameters.title,
                  'label':self.process_S.process.parameters.label,
                  'creator':self.process_S.user_project.user_name}
        
        msg = self.pg_session_C._Manage_root_process(queryD,self.process_S.process.overwrite,self.process_S.process.delete )

        if self.verbose:

            print(msg)

    def _Add_process(self):
        """
        @brief Adds a process to the database.

        This method constructs a query dictionary containing the sub-process parameters
        and delegates the insertion or update operation to the PostgreSQL session manager.
        The parameters include root process ID, sub-process ID, version, minimum user stratum,
        title, label, creator (user ID), and access level. The operation respects the
        overwrite and delete flags provided in the process object.

        @details
        - Converts root and sub-process IDs to lowercase.
        - Collects version, minimum user stratum, title, label, creator, and access information.
        - Calls the PostgreSQL session manager's _Manage_process method with the query dictionary.
        - Passes overwrite and delete flags to control the database operation.

        @return None
        """

        if self.verbose:
           
            if self.process_S.process.delete:
                
                print('.   Deleting sub process: %s' %(self.process_S.process.parameters.process))

            elif self.process_S.process.overwrite:
               
                print('.   Overwriting sub process: %s' %(self.process_S.process.parameters.process))

            else:
            
                print('.   Adding sub process: %s' %(self.process_S.process.parameters.process))

        queryD = {'root_process':self.process_S.process.parameters.root_process.lower(),
                    'process':self.process_S.process.parameters.process.lower(),
                    'min_user_stratum':self.process_S.process.parameters.min_user_stratum,
                    'title':self.process_S.process.parameters.title,
                    'label':self.process_S.process.parameters.label,
                    'creator':self.process_S.user_project.user_name} 
        
        msg = self.pg_session_C._Manage_process(self.process_S, queryD, 
                                                self.process_S.process.overwrite,
                                                self.process_S.process.delete)
        
        if self.verbose:
            
            print(msg)

def Run_process(strcutured_process_D, scheme_params_D):
    """
    @brief Manages and executes a set of process jobs described in a JSON-like dictionary.

    Iterates over each job in the input dictionary, printing job information and dispatching
    process parameters to the appropriate handler based on the root_process. Supports
    process actions such as overwrite and delete, and logs warnings for unknown process types.

    @param structured_process_D Dictionary where keys are file paths and values are dictionaries of process objects.
                    Each process object must have a 'process' attribute with required parameters.

    @return None

    @details
    - Prints job and process information for each entry.
    - Calls Process_manage_process for jobs with root_process 'manage_process'.
    - Logs and returns on unknown root_process.
    - Handles process actions: overwrite, delete, or standard execution.
    """

    pg_session_C = None

    print ('\n########### STARTING PROCESSES ########### \n')

    for key in strcutured_process_D:

        # If there are no processes defined for this job, skip to the next one
        if not strcutured_process_D[key]:

            continue

        json_file_name = path.split(key)[1]

        msg = '. Command file: %s\n. (%s ready processes to run)' %(key, len(strcutured_process_D[key]))

        print (msg)

        for p_nr, process_S in strcutured_process_D[key].items():
        
            root_process = process_S.process.root_process

            if process_S.process.overwrite:

                msg = '\n. Running process nr: %s %s (overwriting)' %(p_nr, 
                    process_S.process.process)

            elif process_S.process.delete:

                msg = '\n. Running process nr: %s %s (deleting)' %(p_nr, 
                    process_S.process.process)
                
            else:

                msg = '\n. Running process nr: %s %s' %(p_nr,
                    process_S.process.process)
                    
            print (msg)

            # Send off the process parameters to the corresponding package - only manage_process available
            if root_process == 'manage_process':

                if pg_session_C is None:
                    pg_session_C = Pg_manage_process(process_S)

                Process_manage_process(process_S, pg_session_C=pg_session_C).run()

            else:
                
                error_msg = '❌ ERROR root_process %s not available in Manage_process\n \
                    (file: %s;  process nr %s)' %(root_process,
                                                json_file_name,
                                                p_nr)

                print (error_msg)

                return

    if pg_session_C is not None:

        try:

            pg_session_C._Close()

        except Exception as e:

            print('⚠️  Warning - could not close database session: %s' % e)