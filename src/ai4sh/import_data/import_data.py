'''
Created on 28 Jan 2026

@author: thomas gumbricht
'''

# Standard library imports
import os

from copy import deepcopy

# Application package imports
from src.lib.pilot import Full_path_locate

from src.lib import Structure_processes

from src.utils.json_read_write import Dump_json

from src.utils.csv_read_write import Read_csv, Read_excel

from src.postgres import Get_schema_table

from src.postgres.pg_ai4sh import PG_manage_AI4SH

# Column names whose values must never be forced to lowercase during import.
# Add entries here to extend the exclusion list.
NO_LOWER_COLS = frozenset({
    'display_name',
    'abstract',
    'title',
    'label'
})

# Define classfication levels for substances
CLASSIFICATION_TABLE_D  = {'manage_order':['manage_family','manage_genus','manage_species'], 
                           'manage_family':['manage_genus','manage_species'], 
                           'manage_genus':['manage_species'],
                           'manage_landcover_order':['manage_landcover_family','manage_landcover_genus'],
                           'manage_landcover_family':['manage_landcover_genus'],
                           'manage_landuse_order':['manage_landuse_family','manage_landuse_genus'],
                           'manage_landuse_family':['manage_landuse_genus']
                           }

CLASSIFICATION_CHILDREN_D = { 'manage_family': {'table': 'family', 'parent_id_name': 'order_id__order_name'},
                            'manage_genus': {'table': 'genus', 'parent_id_name': 'family_id__family_name'},
                            'manage_species': {'table': 'species', 'parent_id_name': 'genus_id__genus_name'},
                            'manage_landcover_family': {'table': 'landcover_family', 'parent_id_name': 'landcover_order_id__landcover_order_name'},
                            'manage_landcover_genus': {'table': 'landcover_genus', 'parent_id_name': 'landcover_family_id__landcover_family_name'},
                            'manage_landuse_family': {'table': 'landuse_family', 'parent_id_name': 'landuse_order_id__landuse_order_name'},
                            'manage_landuse_genus': {'table': 'landuse_genus', 'parent_id_name': 'landuse_family_id__landuse_family_name'}}

SPECIAL_SEARCH_TABLES_D = {'observation.campaign': '_Retrieve_dataset_alias',
                           'observation.observation': '_Retrieve_sample_id_from_observation_log',
                           'observation.observation_measurement': '_Retrieve_quantity_indicator_id_from_provision'}

class Process_import_JSON(Get_schema_table):
    '''class for managing processes'''

    def __init__(self, process_S, pg_session_C, project_root_FP, scheme_params_D=None):
        '''
        '''
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP
        self.scheme_params_D = scheme_params_D

        self.verbose = process_S.process.verbose

        self.process_S = process_S

        self.pg_ai4sh_C = PG_manage_AI4SH(pg_session_C)

    def _Lower_text_values(self, queryD, schema, table):
        """Lowercase string values for text/char/varchar columns, skipping NO_LOWER_COLS."""

        text_cols = self.pg_session_C._Get_text_columns(schema, table)

        return {
            k: v.lower() if isinstance(v, str) and k.lower() in text_cols and k.lower() not in NO_LOWER_COLS else v
            for k, v in queryD.items()
        }

    def _Report_failure(self, msg):
        ''' Print a process-failure message and count it. pg_session_C is the one
        object shared by every Process_import_JSON instance created within a single
        Run_process call (including the per-row instances _Insert_tabular_data spawns),
        so a counter kept on it there survives across the whole run for a final
        summary - see process.py. '''

        print (msg)

        self.pg_session_C.failed_process_count = getattr(self.pg_session_C, 'failed_process_count', 0) + 1

    def _Sub_process(self, json_file_key):

        # Direct to subprocess
        if self.process_S.process.process.startswith('translate'):

            return self._Translate_tabular_data(json_file_key)

        elif self.process_S.process.process.startswith('insert'):

            if not self.pg_session_C:

                print ('❌ ERROR - Inserting data to Postgres requires a database connection. Please define a Postgres database in scheme file.')

                return None

            return self._Insert_tabular_data(json_file_key)

        elif self.process_S.process.process.startswith('manage'):

            if not self.pg_session_C:

                print ('❌ ERROR - Adding data to Postgres requires a database connection. Please define a Postgres database in scheme file.')

                return None

            success = self._Add_JSON_data()

            return json_file_key if success else None

        else:

            error_msg = '\n❌ ERROR process %s \n     not available in Process_translate' %(self.process_S.process.process)
            if self.pg_session_C:
                self.pg_session_C.log( error_msg )

            return None 

    def _Dump_translation(self):
        ''' Dump the structured data to JSON
        '''
        if not self.process_D:

            print ('    ❌ ERROR - No data to dump')

            return None
        
        Dump_json(self.dst_FPN,self.process_D)

        if self.verbose > 0:

            print ('    ✅ Created JSON: %s' %(self.dst_FPN))

    def _Set_dst_FPN(self,json_file_key):

        dst_FN = '%s.json' %(self.process_S.process.parameters.process)

        self.dst_FPN = os.path.join(self.dst_FP, dst_FN)

        if os.path.exists(self.dst_FPN) and not self.process_S.process.overwrite:

            if self.verbose > 1:
            
                print ('    🟡 JSON destination file already exists. Use overwrite option to replace.')
                print ('    🟡 Existing JSON file: %s' %(self.dst_FPN))

            return None
        
        elif os.path.exists(self.dst_FPN) and self.process_S.process.delete:

            os.remove(self.dst_FPN)

            if self.verbose > 1:

                print ('    ✅ Existing JSON file deleted: %s' %(self.dst_FPN))

            return None
            
        elif os.path.exists(self.dst_FPN) and self.process_S.process.overwrite:

            os.remove(self.dst_FPN)
        
        return True
    
    def _Extract_tabular_data(self, column_L, data_L_L):
        ''' Extract the site data from the tabular data file
        '''
        self.record_D = {}

        for row,data_L in enumerate(data_L_L):

            self.record_D[row] = dict(zip(column_L, data_L))

            keys_to_remove = []

            for key in self.record_D[row]:

                if self.record_D[row][key] == 'null' or self.record_D[row][key] == '':
                    
                    keys_to_remove.append(key)

            for key in keys_to_remove:

                self.record_D[row].pop(key)

        return True
        
    def _Structure_data(self):
        """
        @brief Structures the extracted data into the format required for JSON export.

        @details
        This method takes the extracted data stored in self.record_D and organizes it into the hierarchical structure expected by the JSON export functions. 
        It may involve mapping fields, applying default values, and preparing the data for assembly into the final sample event structure.

        @return None. The structured data is stored in instance variables for later use in JSON assembly.
        """

        main_process_D = {"root_process_id": "import_tabular_data",
                          "process": self.process_S.process.parameters.process,
                          "delete": False,
                          "overwrite": False,
                           "parameters": {}}
        
        self.process_D = {"process": []}
      
        for rec in self.record_D:

            process_D = deepcopy(main_process_D)

            # Lowercase all string values at translation time so that exported
            # JSON already carries normalised values before DB insertion.
            # Columns listed in NO_LOWER_COLS are left as-is.
            process_D['parameters'] = {
                k: v.lower() if isinstance(v, str) and k.lower() not in NO_LOWER_COLS else v
                for k, v in self.record_D[rec].items()
            }

            self.process_D['process'].append(process_D) 

        # If the process is for a higher level of substance classification, add the generic term to all lower levels          
        if self.process_S.process.parameters.process in CLASSIFICATION_TABLE_D:

            for rec in self.record_D:

                for lower_level in CLASSIFICATION_TABLE_D[self.process_S.process.parameters.process]:
                    name = self.record_D[rec]['name']
                    parent_column_id_name = CLASSIFICATION_CHILDREN_D[lower_level]['parent_id_name']
                    process = lower_level
                    process_D = deepcopy(main_process_D)
                    process_D['process'] = process
                    process_D['parameters'] = {'name': name,
                                                parent_column_id_name: name}

                    for opt_key in ('alias', 'display_name', 'abstract'):

                        if opt_key in self.record_D[rec]:

                            process_D['parameters'][opt_key] = self.record_D[rec][opt_key]

                    self.process_D['process'].append(process_D)

    def _Translate_tabular_data(self,json_file_key):
        '''
        '''

        tabular_data_path = Full_path_locate(self.project_root_FP,self.process_S.process.parameters.tabular_data_path)

        if not tabular_data_path:

            print ('❌ ERROR - Tabular data file not found:\n   %s' %(self.process_S.process.parameters.tabular_data_path))

            return None
        
        self.dst_FP = Full_path_locate(self.project_root_FP,self.process_S.process.parameters.dst_path, True)
        
        if not self._Set_dst_FPN(json_file_key):

           return None

        if self.verbose > 1:

            print ('Translating tabular data process started',json_file_key)
         
        # Read the tabular data
        if self.process_S.process.parameters.tabular_data_path.endswith('.csv'):

            result = Read_csv(tabular_data_path)
        
        elif self.process_S.process.parameters.tabular_data_path.endswith('.xlsx'):
           
            result = Read_excel(tabular_data_path)

        else:

            error_msg = '\n❌ ERROR import tabular data file type not supported: %s' %(self.process_S.process.parameters.tabular_data_path)

            print ( error_msg )

            return None
        
        if not result:
            
            return None
        
        elif self.verbose > 0:

            print ('    Input read: %s' %(tabular_data_path))
        
        column_L, data_L_L = result

        if not self._Extract_tabular_data(column_L, data_L_L):

            return None

        self._Structure_data()

        self._Dump_translation()

        return self.dst_FPN

    def _Insert_tabular_data(self, json_file_key):
        ''' Translate tabular data to JSON and immediately apply it to the database
        in a single step. The staged JSON is always generated with delete/overwrite
        set to False per row (see _Structure_data), so this route is INSERT-only by
        construction - existing records are left untouched, never updated.
        '''

        dst_FPN = self._Translate_tabular_data(json_file_key)

        if not dst_FPN:

            return None

        structured_D = Structure_processes(self.scheme_params_D, [dst_FPN])

        if not structured_D:

            print ('❌ ERROR - could not structure generated process file for insert:\n   %s' %(dst_FPN))

            return None

        for sub_key in structured_D:

            for p_nr, sub_process_S in structured_D[sub_key].items():

                sub_import_C = Process_import_JSON(sub_process_S, self.pg_session_C, self.project_root_FP, self.scheme_params_D)

                sub_import_C._Sub_process(sub_key)

        return dst_FPN

    def _Insert(self, query_D, schema_S, table_S, model_name_S):
        ''' Insert record in table
        '''

        success = self.pg_session_C._Check_insert_single_record(
            self._Lower_text_values(query_D, schema_S, table_S),
            schema_S, table_S
        )

        if not success:

            msg = '   ❌ ERROR - could not insert record <%s> in table %s.%s' %(model_name_S,
                                                                                     schema_S,
                                                                                     table_S)

            self._Report_failure(msg)

            return success

        return success

    def _Update(self, query_D, schema_S, table_S, model_name_S):
        ''' Update record in table
        '''

        success = self.pg_session_C._Check_update_single_record(
            self._Lower_text_values(query_D, schema_S, table_S),
            schema_S, table_S
        )

        if not success:

            msg = '   ❌ ERROR - could not update record <%s> in table %s.%s' %(model_name_S,
                                                                                     schema_S,
                                                                                     table_S)

            self._Report_failure(msg)

            return success

        return success

    def _Delete(self, query_D, schema_S, table_S, model_name_S):
        ''' Delete record in table
        '''

        success = self.pg_session_C._Check_delete_single_record(
            self._Lower_text_values(query_D, schema_S, table_S),
            schema_S, table_S
        )

        if not success:

            msg = '   ❌ ERROR - could not delete record <%s> in table %s.%s' %(model_name_S,
                                                                                     schema_S,
                                                                                     table_S)

            self._Report_failure(msg)

            return success

        return success
    
    def _Measurement_record(self, main_query_D):
        ''' Special function to retrieve the sample id for a measurement record based on the sample name and observation log name provided in the query_D. This allows for more flexible queries for retrieving the sample id without needing to have the sample_id field in the measurement table.
        '''

        # it should be enought o find the indicators once for each measurement record,        
        # Retrieve the indicators name and idname for the observation log based on the query_D. This is needed to ensure that the sample_id is retrieved for the correct observation log in case there are multiple observation logs with the same name but different indicators.
        measurement_indicators_L = self.pg_session_C._Retrieve_observation_log_indicators_from_log_name(main_query_D['observation_log_id__observation_log_name'])

        if not measurement_indicators_L:

            self._Report_failure('.  ❌ ERROR: could not retrieve indicators for observation_log %s' %(main_query_D['observation_log_id__observation_log_name']))

            return None

        indicator_D = dict(measurement_indicators_L)

        if main_query_D['indicator_id__indicator_name'] not in indicator_D:

            self._Report_failure('.  ❌ ERROR: indicator %s not found for observation log %s' %(main_query_D['indicator_id__indicator_name'], main_query_D['observation_log_id__observation_log_name']))
            print ('.     Available indicators for this observation log are: %s' %(list(indicator_D.keys())))

            return None

        if not main_query_D['indicator_id__indicator_name'] in indicator_D:

            self._Report_failure('.  ❌ ERROR: indicator %s not found for observation log %s' %(main_query_D['indicator_id__indicator_name'], main_query_D['observation_log_id__observation_log_name']))
            print ('.     Available indicators for this observation log are: %s' %(list(indicator_D.keys())))

            return None

        indicator_id = indicator_D[main_query_D['indicator_id__indicator_name']]

        observation_id = self.pg_session_C._Retrieve_observation_id_from_observation(main_query_D)

        if not observation_id:

            self._Report_failure('.  ❌ ERROR: could not retrieve observation id for measurement record')

            return None
                
  
        update_main_query_D = {'observation_id': observation_id,
                               'indicator_id': indicator_id,
                               'value': main_query_D['value'],
                               'standard_deviation': main_query_D['standard_deviation'],
                               'n_repeat': main_query_D['n_repeat']}
        
        if update_main_query_D['standard_deviation'] == -999.999:

            update_main_query_D.pop('standard_deviation')

        return update_main_query_D

    def _Add_JSON_data(self):

        if self.verbose > 2:

            print ('.   Adding JSON data to Postgres')

        schema_table_query_D = self._Get_process_schema_table()

        if self.verbose > 2:

            print ('.      Schema table query dictionary retrieved')

        query_D = {'process': self.process_S.process.process}
        records = self.pg_session_C._Multi_search(query_D,
                                             ['parameter', 'in_schema', 'in_table', 'write'], 'process', 'process_parameter_schema_table')

        dst_schema = records[0][1]
        
        dst_tables = set([item[2] for item in records])
        # Sort the destination tables by length to ensure that parent tables are processed before child tables
        dst_tables = list(dst_tables)
        dst_tables.sort(key=len)

        dst_main_table = dst_tables[0]

        main_table_key = '%s.%s' %(dst_schema, dst_main_table)
       
        # move the main table query from schema_table_query_D to main_query_D
        main_query_D = schema_table_query_D.pop(main_table_key)

        if self.verbose > 1:

            print ('      Managing main schema.table: %s.%s' % (dst_schema, dst_main_table))

        if main_table_key in SPECIAL_SEARCH_TABLES_D:

            retrieve_function = getattr(self.pg_ai4sh_C, SPECIAL_SEARCH_TABLES_D[main_table_key])

            record = retrieve_function(main_query_D, self.pg_session_C)

            if not record:

                self._Report_failure('.  ❌ ERROR: could not retrieve record for %s.%s' % (dst_schema, dst_main_table))

                return None
            
            # Replace the code field in the main_query_D with the retrieved id values
            # this prevents the need to have the code field in the main table and allows for more flexible queries for retrieving the id values
            main_query_D.pop(record[0])

            main_query_D[record[1]] = record[2]

        elif main_table_key == 'observation.measurement':

            main_query_D = self._Measurement_record(main_query_D)

            if not main_query_D:

                return None
            
        # Remove all parameter where write is set to False in the process_parameter_schema_table. This allows for more flexible queries where not all parameters need to be included in the main_query_D, but can still be used for retrieving id values for the parameters that are included in the main_query_D.  
        #TG TODO: this should be done in a more elegant way, for example by having a separate table for the parameters that are used for retrieving id values and the parameters that are used for writing to the database, or by having a flag in the process_parameter_schema_table that indicates whether the parameter is used for retrieving id values or for writing to the database. This would prevent the need to remove parameters from the main_query_D and would allow for more flexible queries where some parameters are only used for retrieving id values and not for writing to the database.
        for rec in records:

            if not rec[3] and rec[0] in main_query_D:

                main_query_D.pop(rec[0])


            if not rec[3] and rec[0] in schema_table_query_D:

                schema_table_query_D.pop(rec[0])
            
        # Get the keys for this table to use for managing content
        table_keys = self.pg_session_C._Get_table_keys(dst_schema, dst_main_table)

        if not 'name' in main_query_D:

            # Tables without their own 'name' (e.g. junction tables like
            # provision_indicator) are identified by their FK-reference columns
            # (xxx_id__xxx_name); main_query_D still holds those as human-readable
            # strings at this point (FK resolution to raw ids happens later), so use
            # the values themselves for the report label instead of table_keys'
            # column NAMES (e.g. literally "id"), which don't identify a specific row.
            descriptive_items = [(k.split('__')[-1], v) for k, v in main_query_D.items() if '__' in k]

            if descriptive_items:

                column_report_name = ", ".join('%s=%s' % (k, v) for k, v in descriptive_items)

            else:

                column_report_name = ",".join([item[0] for item in table_keys])

        else:

            column_report_name = main_query_D['name']

        # Check it the record is already registered in the database
        record_id = self.pg_session_C._Single_search_tab_keys(
            self._Lower_text_values(main_query_D, dst_schema, dst_main_table),
            ['id'], dst_schema, dst_main_table
        )

        if record_id == 'fk_error':

            self._Report_failure('.  ❌ ERROR: could not retrieve foreign key for %s.%s' % (dst_schema, dst_main_table))

            return None

        elif record_id and self.process_S.process.delete:

            self._Delete({'id': record_id[0]}, dst_schema, dst_main_table, column_report_name)

            main_table_id = '%s_id' %(dst_main_table)

            for schema_table in schema_table_query_D:

                schema, table  = schema_table.split('.')

                self._Delete({main_table_id: record_id[0]}, schema, table, column_report_name)
                # TG TODO check if this is the correct printout
                print ('.   ✅ Record %s deleted from %s.%s' %(column_report_name, schema, table))

            return None

        elif not record_id and self.process_S.process.delete:

            if self.verbose > 1:

                print ('.   ✅ Nothing to delete, record %s not found' %(column_report_name))

            return None

        elif record_id and self.process_S.process.overwrite:

            success = self._Update(main_query_D, dst_schema, dst_main_table, column_report_name)

            if success and self.verbose > 1:

                print ('.   ✅ Record %s updated in %s.%s' %(column_report_name, dst_schema, dst_main_table))

            if not success:

                return None

        elif record_id and self.verbose > 1:

            print ('.   🟡 Record %s already registered in %s.%s, use overwrite to update' %(column_report_name, dst_schema, dst_main_table))

        elif not record_id:

            success = self._Insert(main_query_D,dst_schema, dst_main_table, column_report_name)

            if success and self.verbose > 1:

                print ('.   ✅ Record %s inserted in %s.%s' %(column_report_name, dst_schema, dst_main_table))

            if not success:

                return None
            
        # Recheck the record id after filling the main table 
        if not record_id:
            
             record_id = self.pg_session_C._Single_search_tab_keys(
            self._Lower_text_values(main_query_D, dst_schema, dst_main_table),
            ['id'], dst_schema, dst_main_table
        )
        
        if not record_id:

            self._Report_failure('.  ❌ ERROR: could not retrieve record_id after inserting to %s.%s' % (dst_schema, dst_main_table))

            return None
        
        main_table_id = '%s_id' %(dst_main_table)

        if 'name' in main_query_D:

            name = main_query_D['name']

        else:

            name = 'record %s' %(record_id[0])

        self._Define_specifics(main_query_D,record_id, schema_table_query_D, main_table_id, name)

        return True

    def _Legacy_array_alias(self, item):
        ''' Original x_id__x_name[_array] -> x_id__x_name alias inference, kept for
        parameters that never opted into array_column metadata (see _Resolve_array_column). '''

        column_alias = item.split('__')[1].split('_array')[0].replace('name','id')

        column_alias += '__%s' %(column_alias.replace('id','name'))

        return column_alias

    def _Resolve_array_column(self, schema_table, item):
        ''' Resolve the destination column (and any parameter-baked constant column/value)
        for one array-suffixed parameter key being fanned out into schema_table.
        Falls back to the legacy alias inference when no explicit array_column metadata
        was registered for this parameter. '''

        meta = getattr(self, 'array_meta_D', {}).get(schema_table, {}).get(item)

        if meta and meta.get('column'):

            constant_D = {}

            if meta.get('constant_column'):

                constant_D[meta['constant_column']] = meta['constant_value']

            return meta['column'], constant_D

        return self._Legacy_array_alias(item), {}

    def _Define_specifics(self, main_query_D, record_id, schema_table_query_D, main_table_id, name):
        ''' Define device model specifics
        '''

        def Split_arrays():

            return_bool = False

            keys = [k for k in schema_table_query_D[schema_table] if k != main_table_id]

            meta_keys = [k for k in keys
                         if isinstance(schema_table_query_D[schema_table][k], str) and k.endswith('_array')
                         and getattr(self, 'array_meta_D', {}).get(schema_table, {}).get(k)]

            # Independent, metadata-driven fan-out (e.g. dataset_tag's substance_array/
            # keyword_array: each array key produces its own rows via array_column/
            # array_constant_column, with no relation to sibling keys).
            for item in meta_keys:

                return_bool = True

                value = schema_table_query_D[schema_table][item]

                value_csv = value[value.index("{") + 1:value.rindex("}")]

                value_in_L = value_csv.split(',')

                column_alias, constant_D = self._Resolve_array_column(schema_table, item)

                for value_in in value_in_L:

                    v = value_in.strip()

                    if not v or v.lower() in ('none', 'null'):

                        continue

                    item_query_D = {main_table_id: schema_table_query_D[schema_table][main_table_id]}

                    item_query_D[column_alias] = v

                    item_query_D.update(constant_D)

                    self._Define_specifics(main_query_D, record_id, {schema_table: item_query_D}, main_table_id, name)

            # Legacy lockstep zip: any remaining array key(s) with no array_column metadata
            # get position-zipped together with every other remaining (non-meta) key in this
            # schema_table, array or not - the original behaviour that e.g.
            # sample_juxtaposition's setting_system_id__setting_system_name_array (array) +
            # juxtaposition_id__juxtaposition_name (scalar) rely on to combine into one row.
            zip_keys = [k for k in keys if k not in meta_keys]

            legacy_array_present = any(
                isinstance(schema_table_query_D[schema_table][k], str) and k.endswith('_array') and '__' in k
                for k in zip_keys
            )

            if legacy_array_present:

                return_bool = True

                new_query_D = {}

                for key in zip_keys:

                    raw = schema_table_query_D[schema_table][key]

                    new_query_D[key] = raw.strip("{}").split(",") if isinstance(raw, str) else [raw]

                for idx in range(len(new_query_D[zip_keys[0]])):

                    item_query_D = {main_table_id: schema_table_query_D[schema_table][main_table_id]}

                    for key in zip_keys:

                        column_alias = self._Legacy_array_alias(key)

                        v = new_query_D[key][idx]

                        item_query_D[column_alias] = v.strip() if isinstance(v, str) else v

                    self._Define_specifics(main_query_D, record_id, {schema_table: item_query_D}, main_table_id, name)

            return return_bool

        # Loop over all devie model specific tables and insert/update/delete
        for schema_table in schema_table_query_D:

            # Add record_id to the query
            schema_table_query_D[schema_table][main_table_id] = record_id[0]

            if self.verbose > 1:

                print ('      Managing specifics in sub schema.table:', schema_table)

            # split the schema.table string into schema and table
            schema, table  = schema_table.split('.')

            # Test if input arrays are also output arrays or should be split,
            # if it was split True is return, if not Manage specifics without split
            if not Split_arrays():

                self._Manage_specifics(main_query_D, schema_table_query_D[schema_table], main_table_id, schema, table, record_id[0], name)
     
    def _Manage_specifics(self,main_query_D,updated_query_D,main_table_id, schema, table, record_value, name):
                                
        schema_table = '%s.%s' %(schema, table)
        
        if schema_table in SPECIAL_SEARCH_TABLES_D:

            at_columns_D = {k: v for k, v in updated_query_D.items() if k.startswith('@')}

            retrieve_function = getattr(self.pg_ai4sh_C, SPECIAL_SEARCH_TABLES_D[schema_table])

            at_params_D = retrieve_function(updated_query_D, at_columns_D, self.pg_session_C)

            if not at_params_D:
    
                self._Report_failure('.  ❌ ERROR: could not retrieve @-record for %s.%s' % (schema, table))

                return None
            
            self._Insert_at_records(updated_query_D,at_params_D, schema, table, main_query_D['provision_id__provision_name'])
    
            return None
        
        # Quick and dirty for method tier

        if table == 'observation_log_method_tier':

            method_tier_L = ['field','home','laboratory','drone','satellite','document','senses','auxiliary']
            
            for item in method_tier_L:

                updated_query_D[item] = getattr(self.process_S.process.parameters, item)

                #self._Manage_specifics(main_query_D,updated_query_D,main_table_id, schema, table, record_value, name)
        test_rec = self.pg_session_C._Single_search_foreign_key(updated_query_D,
                                                [main_table_id], schema, table)
            
        if not test_rec and not self.process_S.process.delete:

            result = self._Insert(updated_query_D,schema, table, name)

        elif self.process_S.process.overwrite:

            # Update the data in the device_table
            self._Update(updated_query_D,schema,table, name)
            
        elif self.process_S.process.delete:

            self._Delete({main_table_id: record_value}, schema, table, name)

        elif self.verbose > 1:

            print ('.     🟡 Record %s already registered in table %s, use overwrite to update' %(name, table))

    def _Insert_at_records(self,updated_query_D,at_params_D, schema, table, provision_name):

        # split out parameters that start with @ from the 
        core_query_D = {k: v for k, v in updated_query_D.items() if not k.startswith('@')}

        for key in at_params_D:

            sql_query_D = deepcopy(core_query_D)
    
            sql_query_D.update(at_params_D[key])

            if isinstance(at_params_D[key]['value'], list):

                if not self._Check_provision_array(provision_name, len(at_params_D[key]['value'])):

                    return None

                array_table = '%s_array' %(table)

                self._Insert(sql_query_D, schema, array_table, 'at record')

            else:

                self._Insert(sql_query_D, schema, table, 'at record')


    def _Check_provision_array(self, provision_name, value_len):

        record = self.pg_ai4sh_C._Retrieve_wavelength_cardinality_from_provision(provision_name, self.pg_session_C)

        if not record:

            self._Report_failure('.  ❌ ERROR: could not retrieve wavelength cardinality for provision %s' % provision_name)

            return None

        if record != value_len:

            self._Report_failure('.  ❌ ERROR: length of input array does not match wavelength cardinality for provision %s. Expected %s values, got %s values.' % (provision_name, record, value_len))

            return None
        
        return True