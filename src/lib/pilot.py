"""
 @file pilot.py

 @brief Project path resolution and JSON configuration management utilities for in-situ data management workflows.

 @details Resolves project, scheme, and process file paths across relative,
 absolute, and home-based notations, then loads configuration inputs for package
 workflows.

 Rule: every relative path anywhere in the scheme -> job(project) -> pilot -> process
 chain is relative to the project root (see the "Path resolution" section in
 .claude/CLAUDE.md for the full explanation and the two upstream exceptions -
 scheme_file itself and project_path inside it).

 *Version History*:
 - Created: 2024-01-04
 - Updated: 2025-09-01 (Home directory resolution improvements)
 - Updated: 2026-03-14 (Home path and documentation cleanup)
 - Updated: 2026-03-15 (Notebook-relative path handling)
 - Updated: 2026-08-15 (Single project-root anchor for job/pilot resolution;
   tabular_data_path/dst_path in process files also moved to this anchor, see
   src/ai4sh/import_data/import_data.py)

 @author Thomas Gumbricht

 @date Created: 2024-01-04
 @date Updated: 2025-09-01 (Home directory resolution improvements)
 @date Updated: 2026-03-14 (Home path and documentation cleanup)
 @date Updated: 2026-03-15 (Notebook-relative path handling)
 @date Updated: 2026-08-15 (Single project-root anchor for job/pilot resolution)
"""

# Standard library imports
from os import path, makedirs

from copy import deepcopy

from pathlib import Path

# Package imports
from src.utils import Read_json, Pprint_parameter

def Get_project_path(parent_FP, project_path):
    """
    @brief Resolves a path string to an absolute path using parent_FP as the anchor.

    @details
    Converts a path given in any of five supported notations into an absolute path string.
    Does NOT validate whether the resulting path exists — use Full_path_locate() for that.

    **Supported Path Formats**:

    1. **Parent-relative** (`../`):
       - Resolves relative to the directory of parent_FP, supporting multiple levels.
       - Example: `'../../config.json'` from `/a/b/c/file` → `/a/config.json`

    2. **Current directory** (`.` or `./`):
       - `.` alone returns parent_FP unchanged.
       - `./subpath` appends subpath to parent_FP.

    3. **Home directory** (`~/`):
       - Resolves against pathlib.Path.home() for cross-platform compatibility.

    4. **Absolute** (`/`):
       - Returned unchanged.

    5. **Bare name or unrecognised prefix**:
       - Joined with parent_FP via os.path.join().

    @param parent_FP  Absolute path used as the anchor for relative resolution.
                      Typically a notebook path or directory path.
                      Example: '/Users/user/projects/notebooks/analysis.ipynb'

    @param project_path  Path string to resolve, using one of the supported formats above.

    @return String containing the resolved absolute path. Always returns a string; never None.
            Existence of the path is not checked.

    @note Uses pathlib.Path.home() for home directory resolution (updated Sept 2025),
          replacing os.path.expanduser() for better cross-platform compatibility.

    @see Full_path_locate() for resolution with existence validation and optional directory creation.
    """

    # Normalise parent_FP to a directory regardless of whether a file or dir was passed
    parent_FP = str(Path(parent_FP).parent) if Path(parent_FP).is_file() else parent_FP

    if project_path.startswith('../'):

        project_path = str((Path(parent_FP) / project_path).resolve())

    elif project_path.startswith('./'):

        sub_path = project_path[2:]

        project_path = str(Path(parent_FP) / sub_path)

    elif project_path == '.':

        project_path = parent_FP

    elif project_path.startswith('~/'):

        project_path = str(Path.home() / project_path[2:])

    elif project_path.startswith('/'):

        pass  # already absolute path stated, returned unchanged

    else:

        project_path = str(Path(parent_FP) / project_path)

    return project_path
 
def Full_path_locate(parent_path, path_string, dir_make = False, notebook_FP = None):
    """
    @brief Resolves a two-level path and validates (or creates) the result.

    @details
    Performs a two-step resolution using Get_project_path():
    1. Resolves parent_path relative to notebook_FP to get an intermediate base (FP).
    2. Resolves path_string relative to FP to get the final path (FPN).

    Optionally creates the directory if it does not exist. Prints an error and returns
    None if the final path does not exist after optional creation.

    @param parent_path  First path segment, resolved relative to notebook_FP.
                        Pass 'None' (string) when there is no meaningful parent and
                        path_string is absolute or home-relative.
    @param path_string  Second path segment, resolved relative to the result of the
                        first resolution step.
    @param dir_make     If True, creates the directory at FPN when it does not exist.
                        Default: False.
    @param notebook_FP  Absolute path of the calling notebook or script, used as the
                        anchor for resolving parent_path. Default: 'None' (string).

    @return String containing the resolved absolute path if it exists, otherwise None.

    @see Get_project_path() for the underlying path resolution logic.
    """
    if parent_path:
    
        #parent_FP = deepcopy(parent_path)

        if notebook_FP:

            fp = Get_project_path(notebook_FP, parent_path)

        else:

            fp = parent_path

        #fp = Get_project_path(notebook_FP, parent_path)

    elif notebook_FP:

        fp = notebook_FP

    else:

        msg = '❌ ERROR Full_path_locate() requires either a parent_path or notebook_FP to resolve the path.'
        print(msg)
        return None

    FP = Get_project_path(fp, path_string)

    if dir_make and not path.exists(FP):

        makedirs(FP)

    if not path.exists(FP):

        msg = '.   Path does not exist:\n   %s' %(FP)

        print(msg)

        return None

    return FP

def Clean_pilot_list(user_json_process_file_FPN_L, project_root_FP, project_D):
    """
    @brief Cleans and validates a list of JSON process file paths by removing comments and whitespace.

    @details
    This function processes a list of JSON file paths (either from a pilot file or pilot_list array),
    filtering out invalid entries and constructing full file paths. It performs the following operations:
    - Constructs the full path to the JSON process files directory
    - Validates that the directory exists
    - Filters out comments (lines starting with '#')
    - Filters out entries that are too short to be valid file names (< 5 characters)
    - Strips whitespace from valid entries
    - Constructs full file paths by joining with the JSON directory path

    @param user_json_process_file_FPN_L  List of JSON file names or paths.
    @param project_root_FP  The project root file path.
    @param project_D  Project dictionary containing process configuration.

    @return A cleaned list of full file paths to valid JSON process files.
            Returns None if the JSON process directory does not exist.

    @note Prints an error message and returns None if the constructed JSON path does not exist.
    """

    job_folder_FP = Get_project_path(project_root_FP, project_D["process"]["job_folder"])

    json_path = Get_project_path(job_folder_FP, project_D["process"]["process_sub_folder"])

    if not path.exists(json_path):

        print('    ❌ ERROR the path to the json process file(s) does not exist:', json_path)

        return None

    # Clean the list of json objects from comments and white space and too short names
    cleaned_L = [Get_project_path(json_path,x.strip())  for x in user_json_process_file_FPN_L if len(x) > 5 and x[0] != '#']

    return cleaned_L

def Get_scheme_project_path_setup(scheme_file, project_file):
    """
    @brief Loads and validates the scheme file and project/process file, then returns
           the scheme parameters and the list of process JSON file paths to run.

    @details
    Performs the following steps:
    1. Validates and reads the scheme JSON file.
    2. Resolves the project root path from scheme_params_D['project_path'].
    3. Locates and reads the project/process file relative to the project root.
    4. Determines the list of process JSON files by looking for, in order:
       - `pilot_list` array in the project file,
       - `pilot_file` text file in the project file,
       - a single process file where `process` is a list containing `process`.

    @param scheme_file   Absolute path to the scheme JSON file.
    @param project_file  Filename of the project/process file, resolved relative to
                         the project root path defined in the scheme file.

    @return Tuple (scheme_params_D, json_process_file_FPN_L) on success, where:
            - scheme_params_D (dict): parsed contents of the scheme file.
            - json_process_file_FPN_L (list): absolute paths to process JSON files.
            Returns None on any validation or file-loading failure.
    """

    # Check if the user project file exists
    if not scheme_file or not path.exists(scheme_file):

        print('  ❌ ERROR the user scheme file does not exist:', scheme_file)

        return None
    
    #user_scheme_FP, user_scheme_FN = path.split(user_scheme_file)

    #user_scheme_absolut_FP = Get_project_path(notebook_FP, user_scheme_FP)

    #user_scheme_FPN = path.join(user_scheme_absolut_FP, user_scheme_FN)
    # Read the user scheme file to get the project path and other default parameters.
    scheme_params_D = Read_json(scheme_file)

    if not scheme_params_D:

        print('  ❌ ERROR the user scheme file is empty or not a valid JSON:', scheme_file)
        return None

    # Set verbosity
    verbose = scheme_params_D['process'][0]['verbose']
            
    if verbose > 2:
            
        print ('====== Parameters from user scheme file:')

        Pprint_parameter( scheme_params_D )

        print ('======') 

    # Get the project root path from the user scheme file
    project_root_FP = Get_project_path( path.split(scheme_file)[0], scheme_params_D['project_path'])

    if not project_root_FP:

        print('  ❌ ERROR the <project_path> (%s) defined in the user scheme file does not exist:\n.   %s' %(scheme_params_D['project_path'], scheme_file))

        return None

    # Make project_root_FP available to downstream consumers (e.g. process files whose own
    # parameters contain relative paths, which are resolved relative to the project root -
    # see the "Path resolution" section in .claude/CLAUDE.md) - not just used locally here.
    scheme_params_D['project_root_FP'] = project_root_FP

    # Get the full path to the project file
    project_file_FPN = Get_project_path(project_root_FP, project_file)

    if not path.exists(project_file_FPN):

        print('  ❌ ERROR the project file does not exist:\n    %s' %(project_file_FPN))
       
        return None
    
    if verbose > 0:

        print ('\n. Using project/process file:', project_file_FPN)

    # Read the project file to get the job/process parameters.
    proj_proc_D = Read_json(project_file_FPN)

    if not proj_proc_D:

        print('  ❌ ERROR the project file is empty or not a valid JSON:', project_file_FPN)

        return None

    if verbose > 2:

        print ('====== Parameters from project/process file:')

        Pprint_parameter( proj_proc_D )

        print ('======')      

    # look for 1) <pilot_list>, 2) <pilot_file> or 3) <process_file> in that order in the project file
    if "pilot_list" in proj_proc_D["process"]:

        if verbose > 0:

            print ('.   Reading process files to run from array <pilot_list> in project file')

        if not isinstance(proj_proc_D["process"]["pilot_list"], list):

            #print('    ❌ ERROR the object <pilot_list> in the project file should be a list of json files')

            msg = '    ❌ ERROR the scheme file object pilot_list should be a list of json files\n      You need to edit the scheme file\n      %s' % (scheme_file)
            
            print (msg)
            
            return None

        process_L = proj_proc_D["process"]["pilot_list"]

        json_process_file_FPN_L = Clean_pilot_list(process_L, project_root_FP, proj_proc_D) 

    elif "pilot_file" in proj_proc_D["process"]:

        job_folder_FP = Get_project_path(project_root_FP, proj_proc_D["process"]["job_folder"])

        pilot_FPN = Get_project_path(job_folder_FP, proj_proc_D["process"]["pilot_file"])

        if verbose > 0:

            print ('.   Running <pilot_file>: %s' %(pilot_FPN))

        if not path.exists(pilot_FPN):

            msg = '    ❌ ERROR pilot file does not exist: %s\n      Either create the pilot file or edit the scheme file:\n      %s' % (pilot_FPN, scheme_file)
            
            print (msg)

            return None

        # Open and read the pilot text file linking to all json files defining the project
        with open(pilot_FPN) as f:

            process_L = f.readlines()

        json_process_file_FPN_L = Clean_pilot_list(process_L, project_root_FP, proj_proc_D) 

    # If the object <process> exists in the project file, 
    # it is not a true project file but a process file, 
    # Read the processes to execcute from that single process.
    elif isinstance(proj_proc_D["process"], list) and'process' in proj_proc_D['process'][0]:

        if verbose:

            print ('.   Treating the project file as a single process file with <process> defined')

        process_file_FPN = project_file_FPN
        # Set the list of json process files to run to a list with the single process file
        json_process_file_FPN_L  = [process_file_FPN]
        # The structure of individual process files is tested and evaluated in the loop running the processes in manage_process.py
            
    else:

        print('    ❌ ERROR the user project file must contain one the objects <pilot_list>, <pilot_file> or at least one <process>:\n    ❌ %s' % project_file_FPN)

        return None                
    
    if verbose > 1:

        print ('.   Json process files:')
                
        for json_file in json_process_file_FPN_L:
                
            print ('      ',json_file)

    return scheme_params_D,  json_process_file_FPN_L