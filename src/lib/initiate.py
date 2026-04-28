"""
 @file initiate.py

 @brief Process initialization module for loading and structuring process configurations.

 @details Loads scheme and project configuration files, resolves their paths, and
 returns structured process objects ready for downstream execution.

 *Version History*:
 - Created: December 2025

 @author Thomas Gumbricht

 @date Created: December 2025
"""

# Package application imports
from src.lib import Structure_processes

from src.lib import Full_path_locate, Get_scheme_project_path_setup

def Initiate_process(notebook_FP,scheme_file, proj_proc_file):
    """
    @brief Initiate the process by loading the scheme file and the project/process file.

    @details This function orchestrates the initialization of a processing workflow by:
    1. Locating and validating the scheme file
    2. Loading scheme parameters and process file paths
    3. Structuring the process configuration for execution
    
    @param notebook_FP Full path to the notebook executing this function.
                       Used as base path for relative file path resolution.
    @type notebook_FP: str
    
    @param scheme_file Path to the scheme configuration file.
                       Can be absolute or relative to notebook_FP.
    @type scheme_file: str
    
    @param proj_proc_file Path to the project/process configuration file.
                         Specifies which processes to execute.
    @type proj_proc_file: str
    
    @return Dictionary containing structured process data, or None if initialization fails.
    @rtype: dict or None
    
    The function performs the following steps:
        1. Prints the scheme file path for logging
        2. Resolves the full path to the scheme file
        3. Loads scheme parameters and process file list
        4. Structures the processes for execution
        5. Returns None if no processes are defined or if initialization fails
    
    @note The scheme file path can be provided as:
        - Absolute path: /full/path/to/scheme.json
        - Relative path: relative/to/notebook/scheme.json
    """

    print('Scheme file:', scheme_file)

    # Locate the scheme file 
    # The following arguments are:
    # 'None': the scheme file has no parent
    # scheme_file: the path to the scheme file
    # False: do not create path if not found
    # notebook_FP: the full path name of the notebook running this function
    # The path to the scheme file can be given as an absolute path or relative to the notebook path
    scheme_file = Full_path_locate(None, scheme_file, False, notebook_FP)
    
    # The existence of the scheme file is tested in Get_scheme_project_path_setup() 
    success = Get_scheme_project_path_setup(scheme_file, proj_proc_file)

    if not success:

        return None, None

    scheme_params_D, json_process_file_FPN_L = success

    if json_process_file_FPN_L: 

        strcutured_process_D = Structure_processes(scheme_params_D, json_process_file_FPN_L)
        
        return strcutured_process_D, scheme_params_D

    else:

        print('⚠️  No process to run')
        
        return None, None