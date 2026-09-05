"""
 @file structure.py

 @brief Process structure module for validating and organizing configuration data.

 @details Parses nested process definitions, normalizes parameter content, and
 prepares structured objects that can run with or without an active database
 connection.

 *Version History*:
 - Created: 2025-01-05
 - Updated: 2025-03-11
 - Updated: 2025-08-21 (Added option for running without DB connection)
 - Updated: 2025-09-01 (Refactored large loops and added Doxygen comments)
 - Updated: 2026-03-14 (Code cleanup and expanded Doxygen coverage)

 @author Thomas Gumbricht

 @date Created: 2025-01-05
 @date Updated: 2025-03-11
 @date Updated: 2025-08-21 (Added option for running without DB connection)
 @date Updated: 2025-09-01 (Refactored large loops and added Doxygen comments)
 @date Updated: 2026-03-14 (Code cleanup and expanded Doxygen coverage)
"""

# Standard library imports
from os import path

# Third party imports
import re

# Package application imports
from src.utils import Read_json, Pprint_parameter, Update_dict, Struct, Log

from src.lib import Project_login

from src.utils import Today, yyyymmdd_str_to_date, Now_as_str_4_postgres, yyyymmdd_HH_MM_SS_s_as_str_4_postgres

# for versions that that do not have a DB connection comment out 
# --- IGNORE ---
from src.postgres import PG_session

from src.lib.login import Get_set_database_session

def Check_param_instance(p, typeD, process_D, json_file_FN, p_str):
    """
    @brief Checks the type and validity of a process parameter instance.

    This function validates the value of a parameter in a process dictionary against its expected type,
    as defined in the type dictionary. It prints warnings for invalid types and attempts to coerce certain
    string representations to their correct types (e.g., booleans).

    Supported type prefixes in `typeD` (first 3 chars, case-insensitive):
    - ``array`` / ``csv`` — wraps scalars in ``{…}``; lists pass through unchanged.
    - ``tex`` — must be a ``str``; paths containing ``_dir`` may not start/end with ``/``.
    - ``int`` — must be int or digit string; coerces string digits to int.
    - ``flo`` / ``rea`` — must be float or int.
    - ``boo`` — coerces common truthy/falsy representations to bool or ``'1'``/``'0'``.
    - ``dat`` — coerces ``''``/``0`` to today; otherwise parses YYYYMMDD.
    - ``tim`` — coerces ``''``/``0`` to now; otherwise parses YYYYMMDD HH:MM:SS.

    @param p  Parameter name (str).
    @param typeD  Dict mapping parameter names to expected type strings.
    @param process_D  Dict containing parameter values; modified in-place on coercion.
    @param json_file_FN  Filename of the JSON file containing the process definition (str).
    @param p_str  String identifier for the process (str).
    @return  1 if the parameter is valid (after any coercion), None otherwise.
    """

    # TG TODO - the function is written for preparing lists for db insertion, this needs to be made more general for other use cases, e.g. by adding an argument for the target use case and then having different type checks and coercions for different use cases
    def _Print_error_msg(error_msg):
        """
        @brief Prints an error message for an invalid process parameter instance.

        @param error_msg The error message to print (str)
        """

        error_msg += '              (file: %s;  process nr %s)' %(json_file_FN, p_str)

        print (error_msg)

    if 'array' in typeD[p].lower() or typeD[p].lower()[0:3] == 'csv':

        if isinstance(process_D[p], str):

                process_D[p] = '{%s}' % process_D[p]

        elif isinstance(process_D[p], int):

                process_D[p] = '{%s}' %(process_D[p])

        elif isinstance(process_D[p], float):

                process_D[p] = '{%s}' %(process_D[p])

        elif isinstance(process_D[p], list):

            pass  # correct type, no conversion needed
        
        else:

            error_msg = '          ❌ FIX FOR UNKNOWN array type of parameter %s (%s)\n' %(p,process_D[p])

            _Print_error_msg(error_msg)

            return None

    elif typeD[p].lower()[0:3] == 'tex':

        if isinstance(process_D[p], int):

                process_D[p] = '%s' % process_D[p]

        if not isinstance(process_D[p], str):

            error_msg = '          ❌ ERROR parameter %s is not a string (%s)\n' %(p,process_D[p])

            _Print_error_msg(error_msg)

            return None

        if '_dir' in p.lower() and process_D[p]:

            if process_D[p][0] in ['/'] or process_D[p][len(process_D[p])-1] in ['/']:

                error_msg = '          ❌ ERROR parameter %s can not start or end with a slash (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None

    elif typeD[p].lower()[0:3] == 'int':

        if not isinstance(process_D[p], int) and not (isinstance(process_D[p], str) and process_D[p].isdigit()):

            error_msg = '          ❌ ERROR parameter %s is not an integer (%s)\n' %(p,process_D[p])

            _Print_error_msg(error_msg)

            return None
        
        if isinstance(process_D[p], str):

            process_D[p] = int(process_D[p])

    elif typeD[p].lower()[0:3] == 'flo':

        if not isinstance(process_D[p], float):

            if not isinstance(process_D[p], int):

                error_msg = '          ❌ ERROR parameter %s is neither a float nor an integer (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None

    elif typeD[p].lower()[0:3] == 'rea':

        if not isinstance(process_D[p], float):

            if not isinstance(process_D[p], int):

                error_msg = '          ❌ ERROR parameter %s is neither a real nor an integer (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None

    elif typeD[p].lower()[0:3] == 'boo':

        if not isinstance(process_D[p], bool):

            if process_D[p] in ['Y','y','true','True', '1', 1, 1.0, '1.0']:

                if process_D[p] in [ '1', 1, 1.0, '1.0']:

                    process_D[p] = '1'

                else:

                    process_D[p] = True

            elif process_D[p] in ['N','n','false', 'False','0', 0, 0.0, '0.0']:

                if process_D[p] in ['0',0, 0.0, '0.0']:

                    process_D[p] = '0'

                else:

                    process_D[p] = False

            else:

                error_msg = '          ❌ ERROR parameter %s is not a boolean (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None
            
    elif typeD[p].lower()[0:3] == 'dat':

        if process_D[p] in ['','0', 0]:

            process_D[p] = '%s' % Today()

        else:

            if isinstance(process_D[p], int):

                process_D[p] = '%s' % process_D[p]

            try:
                process_D[p] = '%s' % yyyymmdd_str_to_date(str(process_D[p]))

            except:
                error_msg = '          ❌ ERROR parameter %s is not a date in the format YYYYMMDD (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None
            
    elif typeD[p].lower()[0:3] == 'tim': #timestamp

        if process_D[p] in ['','0', 0]:

            process_D[p] = '%s' % Now_as_str_4_postgres()

        else:

            if isinstance(process_D[p], int):

                process_D[p] = '%s' % process_D[p]

            try:
                process_D[p] = '%s' % yyyymmdd_HH_MM_SS_s_as_str_4_postgres(str(process_D[p]))

                if not process_D[p]:

                    error_msg = '          ❌ ERROR parameter %s is not a timestamp format in the format YYYYMMDD_HHMMSS_s (%s)\n' %(p,process_D[p])

                    _Print_error_msg(error_msg)

                    return None
            except:
                error_msg = '          ❌ ERROR parameter %s is not a timestamp format in the format YYYYMMDD (%s)\n' %(p,process_D[p])

                _Print_error_msg(error_msg)

                return None

    return 1

class Scheme_params():
    ''' @brief Class for extracting process parameters from user defined defaults.
    '''
    def __init__(self, scheme_params_D):
        """
        @brief Constructor for the Scheme_params class.

        This method initializes the Scheme_params object by processing the default parameters dictionary.
        It removes the "process" list and converts the first list item to a dictionary for easier access.
        If the verbosity level in the process parameters is greater than 2, it prints the default parameters.

        @param scheme_params_D Dictionary containing the default parameters for the process, including a "process" key with a list of process configurations.
        """

        #  Remove the "process" list and convert first list item to a dict
        self.scheme_params_D = self._Split_out_process(scheme_params_D, 0)

        self._Set_execute_verbose_overwrite_delete()

        if self.scheme_params_D['process']['verbose'] > 2:

            print ('\n===== scheme_params_D:')

            Pprint_parameter(self.scheme_params_D)

            print ('=====\n')

    def _Split_out_process(self, raw_D, process_nr):
        """
        @brief Extracts a single process configuration from a dictionary containing multiple processes.

        This method takes a dictionary `raw_D` that contains a 'process' key with a list of process configurations.
        It returns a new dictionary where the 'process' key is replaced by the process configuration at the specified index `process_nr`.
        All other keys and values are copied as-is.

        @param raw_D Dictionary containing process configurations, including a 'process' key with a list of processes.
        @param process_nr Index of the process configuration to extract from the 'process' list.
        @return Dictionary with the selected process configuration and all other keys from the input dictionary.
        """
        process_D = {}

        for key, value in raw_D.items():

            if key == 'process':

                process_D['process'] = raw_D['process'][process_nr]

            else:

                process_D[key] = value

        return process_D
    
    def _Set_execute_verbose_overwrite_delete(self):

         for item in ['overwrite', 'delete', 'execute', 'verbose']:
            # Note that is the scheme file the items are set as arra(0) but they were chenged to an object 
            if not item in self.scheme_params_D['process']:

                    if item in ['overwrite', 'delete']:

                        self.scheme_params_D['process'][item] = False

                    elif item == 'execute':

                        self.scheme_params_D['process'][item] = True

                    elif item == 'verbose':

                        self.scheme_params_D['process'][item] = 1

    def _Set_user_params(self,user_parameter_D,json_file_FN):
        """
        @brief Set user parameters for the process.

        This method updates the user parameter dictionary by filling in missing variables from the default parameter values.
        It also sets the filename of the JSON file containing the user parameters.

        @param user_parameter_D Dictionary containing user-defined parameters for the process.
        @param json_file_FN Filename of the JSON file containing the user parameters.
        """

        self.json_file_FN = json_file_FN

        # Delete stratum_code from user_parameter_D and scheme_params_D if it exists, as it is not a parameter that should be passed to the process but is only used for checking user access to the process
        if 'stratum_code' in user_parameter_D:

            user_parameter_D.delete('stratum_code')

        if 'stratum_code' in self.scheme_params_D:

            self.scheme_params_D.delete('stratum_code')

        # update user_parameter_D by filling in missing variables from the default parameter values
        Update_dict(user_parameter_D, self.scheme_params_D)

        self.user_parameter_D = user_parameter_D

    def _Assemble_single_process(self, p_str, process_D, json_file_FN):
        """
        @brief Assemble and compile a single process configuration.

        This method updates the process dictionary with default parameters, sets key identifiers,
        and compiles the process configuration into a structured object for further use.

        @param p_str String identifier for the process (str).
        @param process_D Dictionary containing process-specific parameters (dict).
        @param json_file_FN Filename of the JSON file containing the process definition (str).

        The function performs the following steps:
        - Updates the process dictionary with default parameters.
        - Sets the main parameters and identifiers for the process.
        - Compiles the process configuration into a Struct object for attribute-style access.
        - Optionally prints the compiled process dictionary if verbosity is set high.
        """

        self.p_str = p_str

        Update_dict(process_D, self.scheme_params_D['process'])

        # Set the main parameters
        compiled_process_D = self.user_parameter_D

        compiled_process_D['process']  = process_D

        compiled_process_D['process']['p_str'] = p_str

        compiled_process_D['process']['json_file_FN'] = json_file_FN

        if compiled_process_D['process']['verbose'] > 2:

            print ('\n   compiled_process_D (process_center.py, 108):')

            Pprint_parameter(compiled_process_D)

        self.process_S = Struct(compiled_process_D)

    def _Assemble_parameters(self, session, process_schema='process'):
        """
        @brief Assemble and validate process parameters from the database and user input.

        This method retrieves parameter definitions for a process from the database, checks for required fields,
        fills in missing parameters with system defaults, validates parameter types, and updates the process
        parameters structure. It logs warnings for missing or invalid parameters and returns a status code
        indicating success or failure.

        @param session Database session object used to query parameter definitions.
        @param process_schema Name of the process schema in the database (default: 'process').
        @return int Status code: 1 if parameters are successfully assembled and validated, 0 otherwise.

        The function performs the following steps:
        - Checks for the existence of 'process' and 'parameters' in the process object.
        - Queries the database for parameter definitions.
        - Fills in missing parameters with system defaults.
        - Validates the presence and type of compulsory parameters.
        - Removes parameters not defined in the database.
        - Updates the process parameters structure.
        - Logs warnings for any issues encountered.
        """

        def extract_concat_parts(expr: str):
            """
            Extract:
            - values inside the last parentheses
            - the string inside single quotes

            Returns:
                (format_string, values_list)
            """
            # Extract quoted string
            quote_match = re.search(r"'([^']*)'", expr)
            format_string = quote_match.group(1) if quote_match else None

            # Extract last parenthesis content
            paren_match = re.search(r'\(([^()]*)\)\s*$', expr)
            if paren_match:
                content = paren_match.group(1)
                values = [v.strip() for v in content.split(',') if v.strip()]
            else:
                values = []

            return format_string, values

        status_OK = 1

        if not hasattr(self.process_S.process,'process'):

            error_msg = '          ❌ ERROR process lacking process \n \
                (file: %s;  process nr %s)' %(self.json_file_FN,
                                            self.p_str)

            Log( error_msg )

            status_OK = 0

        if not hasattr(self.process_S.process, 'parameters') or self.process_S.process.parameters == None:

            self.process_S.process.parameters = None

            error_msg = '          ❌ ERROR process lacking parameters \n \
                (file: %s;  process nr %s)' %(self.json_file_FN,
                                            self.p_str)

            Log( error_msg )

            status_OK = 0

            return status_OK

        queryD = {'process':self.process_S.process.process, 'parent':'process',
                  'element': 'parameters'}

        paramL =['parameter', 'default_value', 'required', 'parameter_type']

        param_recs = session._Multi_search(queryD, paramL, process_schema,'process_parameter')

        # Reformat parameters to their type (int, float (real), text)
        system_default_param_L  = [ (i[0],int( i[1] )) for i in param_recs if not i[2] and i[3].lower()[0:3] == 'int' ]

        system_default_param_L.extend([ (i[0], float( i[1] )) for i in param_recs if not i[2] and i[3].lower()[0:3] in ['flo','rea'] ] )

        system_default_param_L.extend([ (i[0], i[1]) for i in param_recs if not i[2] and i[3].lower()[0:3] not in ['int','flo','rea'] ] )

        system_scheme_params_D = dict (system_default_param_L)

        if self.scheme_params_D['process']['verbose'] > 2:

            print ('\n=====  system_scheme_params_D')

            Pprint_parameter(system_scheme_params_D)

            print ('=====\n')

        typeD = dict ( [ ( i[0],i[3] ) for i in param_recs ] )

        # Create a dict with compulsory parameters
        compuls_parameter_D = dict( [ (i[0],i[1]) for i in param_recs if i[2] ] )

        # Create a dict with non-compulsory parameters
        non_compuls_parameter_D = dict( [ (i[0],i[1]) for i in param_recs if not i[2] ] )

        if self.scheme_params_D['process']['verbose'] > 2:

            print ('\n===== compuls_parameter_D')

            Pprint_parameter(compuls_parameter_D)

            print ('=====\n')

        # Check that all compulsory parameters are included
        for key in compuls_parameter_D:

            if not hasattr(self.process_S.process.parameters, key):

                error_msg = '\n          ❌ ERROR compulsary parameter <%s> missing in json for process <%s>\n \
            (file: %s;  process nr: %s)' %(key,self.process_S.process.process,
                                             self.json_file_FN,
                                            self.p_str)

                Log( error_msg)

                status_OK = 0

        # Check that all non-comulsory parameters that are included are not empty, if they are empty fill them with the default value from the database if it exists, if it does not exist log a warning but do not stop the process
        for key in non_compuls_parameter_D:

            if not hasattr(self.process_S.process.parameters, key):

                if system_scheme_params_D[key] not in [None, '', 'false']:

                    setattr(self.process_S.process.parameters, key, system_scheme_params_D[key])

        # Check if this process has auto naming set for parameters and if so add the auto name parameters to the process parameters, this is needed for the process to be able to get the auto name value for the parameter when it runs, but also for the type checking of the parameters as the auto name parameters are defined in the database as well and need to be included in the process parameters for the type checking to work
        queryD = {'process':self.process_S.process.process}

        paramL =['parameter', 'concat']

        auto_name_recs = session._Multi_search(queryD, paramL, process_schema,'process_parameter_auto_name')

        if auto_name_recs:

            for auto_name_rec in auto_name_recs:

                if getattr(self.process_S.process.parameters, auto_name_rec[0], None) in ['auto', 'auto_name']:

                    format_string, values = extract_concat_parts(auto_name_rec[1])
                        
                    values = [ getattr(self.process_S.process.parameters, v) for v in values]  # only keep values that are defined parameters in the database
               
                    if format_string and values:

                        setattr(self.process_S.process.parameters, auto_name_rec[0], format_string % tuple(values))
        # Check if this process has inherit parameters and if so fill in the inherit parameters, this is needed for the process to be able to get the inherited value for the parameter when it runs, but also for the type checking of the parameters as the inherited parameters are defined in the database as well and need to be included in the process parameters for the type checking to work
        paramL =['process_parameter','src_schema', 'src_table', 'src_column', 'search_column', 'search_object', 'filter_column', 'filter_value']

        inherit_recs = session._Multi_search(queryD, paramL, process_schema,'process_parameter_inherit')

        for inherit_rec in inherit_recs:

            if getattr(self.process_S.process.parameters, inherit_rec[0], None) == 'inherit':

                if '__' in inherit_rec[5]  and inherit_rec[4].endswith('id'):

                    #Find the foreign key first, then search
                    search_value = getattr(self.process_S.process.parameters, inherit_rec[5], None)

                    foreign_key = session._Check_get_foreign_key(inherit_rec[5], search_value)

                    if not foreign_key:

                        print('          ❌ ERROR: no foreign key found for %s.%s.%s = %s' % (inherit_rec[1], inherit_rec[2], inherit_rec[4], search_value))

                        continue

                    if inherit_rec[6] or inherit_rec[0].endswith('_array'):

                        # Multi-row inherit: fetch ALL matching source rows (optionally
                        # filtered) and aggregate into the '{v1,v2,...}' shape the array
                        # fan-out mechanism (_Define_specifics/Split_arrays) already
                        # expects. Taken whenever a filter_column was set OR the target
                        # parameter is itself array-typed (e.g.
                        # setting_system_id__setting_system_name_array inheriting from the
                        # multi-row observation.dataset_setting_system junction table) -
                        # every scalar (non-array) inherit is unaffected and keeps the
                        # single-row path below.
                        query_D = {inherit_rec[4]: foreign_key[0]}

                        if inherit_rec[6]:

                            query_D[inherit_rec[6]] = inherit_rec[7]

                        rows = session._Multi_search(query_D, [inherit_rec[3]], inherit_rec[1], inherit_rec[2])

                        # If src_column is itself a FK integer id (e.g. dataset_setting_system's
                        # setting_system_id), the array fan-out downstream expects NAME strings
                        # to re-resolve, not raw ids - look up the referenced schema.table once
                        # via information_schema, same technique as the single-row int branch
                        # below, and resolve each row's id to its name.
                        fk_ref = None

                        if rows and isinstance(rows[0][0], int):

                            fk_sql = """
                                SELECT ccu.table_schema, ccu.table_name
                                FROM information_schema.key_column_usage kcu
                                JOIN information_schema.referential_constraints rc
                                    ON rc.constraint_name = kcu.constraint_name
                                   AND rc.constraint_schema = kcu.table_schema
                                JOIN information_schema.constraint_column_usage ccu
                                    ON ccu.constraint_name = rc.unique_constraint_name
                                WHERE kcu.table_schema = '%s'
                                  AND kcu.table_name   = '%s'
                                  AND kcu.column_name  = '%s';
                            """ % (inherit_rec[1], inherit_rec[2], inherit_rec[3])

                            fk_ref = session._Execute_search_single_sql(fk_sql)

                        values_L = []

                        for row in (rows or []):

                            v = row[0]

                            if v is None or (isinstance(v, str) and v.strip().lower() in ('', 'none', 'null')):

                                continue

                            if isinstance(v, int) and fk_ref:

                                name_rec = session._Single_search({'id': v}, ['name'], fk_ref[0], fk_ref[1])

                                if not name_rec:

                                    continue

                                v = name_rec[0]

                            values_L.append(str(v))

                        setattr(self.process_S.process.parameters, inherit_rec[0], ','.join(values_L))

                        continue
                    try:
                        # For the special case where an id is searched from the id defining table itself
                        if inherit_rec[4].endswith('_id') and inherit_rec[2] == inherit_rec[4].split('_')[0]:
                            inherit_rec[4] = 'id'
                        inhereted_rec = session._Single_search( {inherit_rec[4]:foreign_key[0]}, [inherit_rec[3]], inherit_rec[1], inherit_rec[2])
                    except:

                        print('          ❌ ERROR: no record found for %s.%s.%s = %s' % (inherit_rec[1], inherit_rec[2], inherit_rec[4], search_value))

                        continue

                    if inhereted_rec is None:

                        print('          ❌ ERROR: no record found for %s.%s.%s = %s' % (inherit_rec[1], inherit_rec[2], inherit_rec[4], foreign_key[0]))

                        continue

                    if isinstance(inhereted_rec[0], int):

                        # inhereted_rec[0] is a FK integer id; use information_schema
                        # to find the referenced schema/table for inherit_rec[3]
                        # (src_column) and then fetch the text 'name' by id.
                        sql = """
                            SELECT ccu.table_schema, ccu.table_name
                            FROM information_schema.key_column_usage kcu
                            JOIN information_schema.referential_constraints rc
                                ON rc.constraint_name = kcu.constraint_name
                               AND rc.constraint_schema = kcu.table_schema
                            JOIN information_schema.constraint_column_usage ccu
                                ON ccu.constraint_name = rc.unique_constraint_name
                            WHERE kcu.table_schema = '%s'
                              AND kcu.table_name   = '%s'
                              AND kcu.column_name  = '%s';
                        """ % (inherit_rec[1], inherit_rec[2], inherit_rec[3])

                        fk_ref = session._Execute_search_single_sql(sql)

                        if fk_ref:

                            name_rec = session._Single_search(
                                {'id': inhereted_rec[0]},
                                ['name'],
                                fk_ref[0], fk_ref[1]
                            )

                            if name_rec:

                                setattr(self.process_S.process.parameters, inherit_rec[0], name_rec[0])

                            else:

                                print('          ❌ ERROR: no name for id <%s> in %s.%s' % (inhereted_rec[0], fk_ref[0], fk_ref[1]))

                        else:

                            print('          ❌ ERROR: no FK reference found for %s.%s.%s in information_schema' % (inherit_rec[1], inherit_rec[2], inherit_rec[3]))

                    elif isinstance(inhereted_rec[0], (list, tuple)):
                        #LOOP foreign key for the source as long as the result in numerical 
                        # Convert list/tuple to postgres array string
                        value = '{%s}' % ','.join(inhereted_rec[0])

                        setattr(self.process_S.process.parameters, inherit_rec[0], value)

                    else:
                        setattr(self.process_S.process.parameters, inherit_rec[0], inhereted_rec[0])
                
                else:

                    search_value = getattr(self.process_S.process.parameters, inherit_rec[4], None)

                    inhereted_rec = session._Single_search( {inherit_rec[4]:search_value}, [inherit_rec[3]], inherit_rec[1], inherit_rec[2])
                    
                    if not inhereted_rec:

                        print('          ❌ ERROR: no inheritable record found for %s.%s.%s = %s' % (inherit_rec[1], inherit_rec[2], inherit_rec[4], search_value))

                        continue

                    setattr(self.process_S.process.parameters, inherit_rec[0], inhereted_rec[0])

        # Create a process dict from process struct
        process_D = dict( list( self.process_S.process.parameters.__dict__.items() ) )

        # Update the parameters and fill in missing parameters from the system inherit parameters
        Update_dict(process_D, system_scheme_params_D)

        # Test the type of all params
        for p in process_D:

            if p.startswith('@'):

                continue

            if not p in typeD:

                error_msg = '\n          ❌ ERROR parameter <%s> is not defined for the \n          process <%s>\n \
            (file: %s; process nr: %s)' %(p,self.process_S.process.process,
                                            self.json_file_FN,
                                            self.p_str)

                Log( error_msg )

                return 0

            instance_OK = Check_param_instance(p, typeD, process_D, self.json_file_FN, self.p_str)

            if not instance_OK:

                return 0

        # Remove parenthesised suffixes from parameter keys, but leave @-prefixed
        # indicator keys (e.g. "@fungi(%)") intact - their "(%)" is meaningful and
        # must survive to match indicator names/aliases in the database.
        process_D = {(key if key.startswith('@') else re.sub(r"\s*\(.*?\)", "", key)): value
            for key, value in process_D.items()}
        
        # Recreate the process struct process with the updated parameters
        self.process_S.process.parameters = Struct(process_D)

        if self.scheme_params_D['process']['verbose'] > 2:

            print ('\n ===== process_D')

            Pprint_parameter(process_D)

            print ('=====\n')

        return status_OK

def Get_process_from_db(pg_session_C, process_schema, process_parameter_C, user_status_D, json_process_file_obj, p, p_str):
    """
    @brief Retrieve and assemble a single process configuration from the database.

    Queries the database for the sub-process record identified by `p['process']`,
    checks that the authenticated user's stratum meets the minimum required level,
    then delegates to `process_parameter_C` to assemble and validate the full parameter set.

    @param pg_session_C  Active PostgreSQL session object.
    @param process_schema  Name of the schema that holds the process tables (e.g. 'process').
    @param process_parameter_C  Params instance used for assembly and validation.
    @param user_status_D  Dict with at least {'stratum_code': int} for the authenticated user.
    @param json_process_file_obj  Full path to the JSON file that defines this process (used in error messages).
    @param p  Dict for a single process entry, must contain 'process'.
    @param p_str  String representation of the process index (used in error messages).
    @return  Assembled process_S Struct on success, or None if the sub-process is not found,
             the user's stratum is too low, or parameter assembly fails.
    """

    # Get the process and its root process in a single query
    query_D = {'process':p['process']}

    record = pg_session_C._Single_search(query_D, ['root_process','process','min_user_stratum'], process_schema, 'process')

    if not record:

        print ('          ❌ ERROR: process <%s> (nr %s) not in DB - skipping' %(p['process'], p_str))
        
        return None

    root_process, process, min_user_stratum = record

    if user_status_D['stratum_code'] < min_user_stratum:

        print ('.  ❌ ERROR: user stratum too low for process %s - skipping' %p['process'])

        return None

    process_parameter_C._Assemble_single_process(p_str, p, path.split(json_process_file_obj)[1])

    if not process_parameter_C.process_S.process.execute:

        return process_parameter_C.process_S
    
    status_OK = process_parameter_C._Assemble_parameters(pg_session_C)

    if not status_OK:

        print ('.  ❌ ERROR: process nr %s <%s> not ready - skipping' %(p_str, p['process']))

        return None

    process_parameter_C.process_S.process.root_process = root_process

    process_parameter_C.process_S.process.process = process

    return process_parameter_C.process_S

def Set_process_overwrite_execute_verbose(scheme_params_D, user_parameter_D, p_nr):
    """
    @brief Set overwrite, execute, and verbose parameters for a process.

    This function checks if the 'overwrite', 'delete', 'execute', and 'verbose' parameters are present
    in the user-defined parameters for a specific process. If any of these parameters are missing,
    it sets them to default values from the scheme parameters, falling back to hardcoded defaults.

    @param scheme_params_D Dictionary containing default parameters for the process.
    @param user_parameter_D Dictionary containing user-defined parameters for the process.
    @param p_nr Index of the process in the user-defined parameters list.
    """

    if not 'overwrite' in user_parameter_D['process'][p_nr]:

        if 'overwrite' in scheme_params_D['process'][0]:

            user_parameter_D['process'][p_nr]['overwrite'] = scheme_params_D['process'][0]['overwrite']

        else:

            user_parameter_D['process'][p_nr]['overwrite'] = False

    if not 'delete' in user_parameter_D['process'][p_nr]:

        if 'delete' in scheme_params_D['process'][0]:

            user_parameter_D['process'][p_nr]['delete'] = scheme_params_D['process'][0]['delete']

        else:

            user_parameter_D['process'][p_nr]['delete'] = False

    if not 'execute' in user_parameter_D['process'][p_nr]:

        if 'execute' in scheme_params_D['process'][0]:

            user_parameter_D['process'][p_nr]['execute'] = scheme_params_D['process'][0]['execute']

        else:

            user_parameter_D['process'][p_nr]['execute'] = True

    if not 'verbose' in user_parameter_D['process'][p_nr]:

        if 'verbose' in scheme_params_D['process'][0]:

            user_parameter_D['process'][p_nr]['verbose'] = scheme_params_D['process'][0]['verbose']

        else:

            user_parameter_D['process'][p_nr]['verbose'] = 1

def Job_processes_loop(scheme_params_D, process_file_FPN_L, process_parameter_C, user_status_D=None,pg_session_C=None,process_schema='process'):
    """
    @brief Main loop to process job configurations from JSON files and assemble process objects.

    This function iterates over a list of JSON process files, reads user-defined parameters, and assembles process configurations
    either by retrieving details from a database or directly from the provided parameters. It supports both database-backed and
    standalone operation modes. The assembled process objects are stored in a dictionary keyed by the JSON file name and process index.

    @details
    - For each JSON file in the input list, reads the process definitions and sets user parameters.
    - For each process entry, checks for required fields and either retrieves process details from the database (if available)
    or assembles them from user parameters.
    - Handles missing fields, insufficient user privileges, and file read errors with warnings.
    - Returns a dictionary of assembled process objects ready for execution.

    @param scheme_params_D  Dict of default parameters; must contain 'postgresdb', 'project_path', 'user_project'.
    @param process_file_FPN_L  List of full file paths to JSON process configuration files.
    @param process_parameter_C  Params instance for managing parameter assembly.
    @param user_status_D  Dict with authenticated user info (required when DB is used, else None).
    @param pg_session_C  Active PostgreSQL session (required when DB is used, else None).
    @param process_schema  Name of the schema holding process tables (default: ``'process'``).
    @return  Dict keyed by JSON file path → {process_index: process_S Struct}.
    """
    def Process_loop():
        """
        @brief Loop over all processes in the JSON file and assemble process configurations.

        This function iterates through each process entry in the user parameter dictionary, checks for required fields,
        and either retrieves process details from the database or assembles them from the provided parameters. The results
        are stored in the json_cmd_D dictionary for further use.

        @details
        - Checks for the presence of 'process' in each process entry and prints a warning if missing.
        - If a database connection is available, retrieves process details using Get_process_from_db and updates json_cmd_D.
        - If no database connection, assembles process details directly from user parameters and updates json_cmd_D.
        - Skips processes with missing required fields or insufficient user privileges.

        @note
        - Relies on global variables: user_parameter_D, scheme_params_D, pg_session_C, process_schema,
            process_parameter_C, user_status_D, json_process_file_obj, json_cmd_D.
        - Designed to be called within Job_processes_loop.

        @return None. Results are stored in json_cmd_D.
        """

        # Loop over all processes in the json process file
        for p_nr, p in enumerate(user_parameter_D['process']):

            p_str = str(p_nr)

            if not 'process' in p:

                error_msg = '          ❌ ERROR: process %s missing process object' %(p_str)

                print (error_msg)

                continue

            result = Get_process_from_db(pg_session_C,process_schema,process_parameter_C,user_status_D,json_process_file_obj, p, p_str)

            if result:

                if process_parameter_C.process_S.process.execute:

                    json_cmd_D[json_process_file_obj][p_nr] = process_parameter_C.process_S

                else:

                    print ('.  ⚠️ SKIPPING: process nr %s <%s> not set to execute' %(p_str, p['process']))

            else:

                continue

    # Main loop function
    verbose = scheme_params_D['process'][0]['verbose']

    # Dict to hold all processes ready to run
    json_cmd_D = {}

    # Loop over all json files
    for json_process_file_obj in process_file_FPN_L:

        json_cmd_D[json_process_file_obj] = {}

        if verbose > 0:

            msg = '\n    reading JSON:\n    %s' %(json_process_file_obj)

            print (msg)

        user_parameter_D = Read_json(json_process_file_obj)

        if not user_parameter_D:

            if verbose > 0:

                msg = ('          ❌ ERROR: json file not read - skipping' )

            else:
                
                msg = ('          ❌ ERROR: json file %s not read - skipping' %json_process_file_obj)
            
            print (msg)

            continue

        # Set the user defined parameters
        process_parameter_C._Set_user_params(user_parameter_D, path.split(json_process_file_obj)[1])

        # Loop over all processes in the json file
        Process_loop()

    return json_cmd_D

def Check_json_files(process_file_FPN_L):

    """
    @brief Checks the existence of JSON process files in a given list.

    This function iterates over a list of file paths and verifies that each file exists on disk.
    If any file is missing, it prints a warning message and returns None. If all files exist, it returns True.

    @param process_file_FPN_L List of file paths to JSON process configuration files.
    @return True if all files exist, None if any file is missing.
    """
    for json_process_file_obj in process_file_FPN_L:

        if not path.exists(json_process_file_obj):

            error_msg = '          ❌ ERROR - json process file not found:\n    %s' %(json_process_file_obj)

            print (error_msg)

            return None

    return True

def Structure_processes(scheme_params_D, process_file_FPN_L):
    """
    @brief Assemble and structure process jobs from JSON files and database parameters.

    This function checks the existence of provided JSON process files, optionally establishes a database session,
    sets up default parameters, and loops over all process files to assemble job configurations. If a database is required,
    it retrieves user status and closes the session after processing. Returns a dictionary of assembled job objects.

    @param scheme_params_D Dictionary containing default parameters for the process and database connection.
    @param process_file_FPN_L List of file paths to JSON process configuration files.
    @return Dictionary of assembled job objects, or None if any file is missing or user/database status is invalid.
    """
    if not Check_json_files(process_file_FPN_L):

        return None

    # If this is a database required process, get the user status and open a db session
    #if (scheme_params_D['postgresdb']['db']):

    # Get user status
    user_status_D, pg_session_C = Get_set_database_session(scheme_params_D)

    if not user_status_D:

        return None

    # Set the default parameters
    process_parameter_C  = Scheme_params(scheme_params_D)

    try:
        structured_processes_D = Job_processes_loop(scheme_params_D, process_file_FPN_L, process_parameter_C, user_status_D, pg_session_C)

    finally:
        pg_session_C._Close()

    return structured_processes_D