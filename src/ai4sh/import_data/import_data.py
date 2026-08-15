'''
Created on 28 Jan 2026

@author: thomas gumbricht
'''

# Standard library imports
import os

from copy import deepcopy

# Application package imports
from src.lib.pilot import Full_path_locate

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
                           'manage_genus':['manage_species']}

CLASSIFICATION_CHILDREN_D = { 'manage_family': {'table': 'family', 'parent_id_name': 'order_id__order_name'},
                            'manage_genus': {'table': 'genus', 'parent_id_name': 'family_id__family_name'},
                            'manage_species': {'table': 'species', 'parent_id_name': 'genus_id__genus_name'}}

SPECIAL_SEARCH_TABLES_D = {'observation.campaign': '_Retrieve_dataset_alias',
                           'observation.observation': '_Retrieve_sample_id_from_observation_log',
                           'observation.observation_measurement': '_Retrieve_quantity_indicator_id_from_provision'}

class Process_import_JSON(Get_schema_table):
    '''class for managing processes'''

    def __init__(self, process_S, pg_session_C, project_root_FP):
        '''
        '''
        self.verbose = process_S.process.verbose
        self.process_S = process_S
        self.pg_session_C = pg_session_C
        self.project_root_FP = project_root_FP

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

    def _Sub_process(self, json_file_key):

        # Direct to subprocess
        if self.process_S.process.process.startswith('translate'):

            return self._Translate_tabular_data(json_file_key)

        elif self.process_S.process.process.startswith('manage'):

            if not self.pg_session_C:

                print ('❌ ERROR - Adding data to Postgres requires a database connection. Please define a Postgres database in scheme file.')

                return None

            return self._Add_JSON_data()

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
                    alias = self.record_D[rec]['alias'] if 'alias' in self.record_D[rec] else None
                    display_name = self.record_D[rec]['display_name'] if 'display_name' in self.record_D[rec] else None
                    abstract = self.record_D[rec]['abstract'] if 'abstract' in self.record_D[rec] else None
                    process = lower_level
                    process_D = deepcopy(main_process_D)
                    process_D['process'] = process
                    process_D['parameters'] = {'name': name,
                                                parent_column_id_name: name,
                                                'alias': alias,
                                                'display_name': display_name,
                                                'abstract': abstract}
            
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

            print (msg)

            return success
        
        return success
    
    def _Measurement_record(self, main_query_D):
        ''' Special function to retrieve the sample id for a measurement record based on the sample name and observation log name provided in the query_D. This allows for more flexible queries for retrieving the sample id without needing to have the sample_id field in the measurement table.
        '''

        # it should be enought o find the indicators once for each measurement record,        
        # Retrieve the indicators name and idname for the observation log based on the query_D. This is needed to ensure that the sample_id is retrieved for the correct observation log in case there are multiple observation logs with the same name but different indicators.
        measurement_indicators_L = self.pg_session_C._Retrieve_observation_log_indicators_from_log_name(main_query_D['observation_log_id__observation_log_name'])

        if not measurement_indicators_L:

            print ('.  ❌ ERROR: could not retrieve indicators for observation_log %s' %(main_query_D['observation_log_id__observation_log_name']))

            return None
        
        indicator_D = dict(measurement_indicators_L)
        
        if main_query_D['indicator_id__indicator_name'] not in indicator_D:

            print ('.  ❌ ERROR: indicator %s not found for observation log %s' %(main_query_D['indicator_id__indicator_name'], main_query_D['observation_log_id__observation_log_name']))
            print ('.     Available indicators for this observation log are: %s' %(list(indicator_D.keys())))

            return None
        
        if not main_query_D['indicator_id__indicator_name'] in indicator_D:

            print ('.  ❌ ERROR: indicator %s not found for observation log %s' %(main_query_D['indicator_id__indicator_name'], main_query_D['observation_log_id__observation_log_name']))
            print ('.     Available indicators for this observation log are: %s' %(list(indicator_D.keys())))

            return None
        
        indicator_id = indicator_D[main_query_D['indicator_id__indicator_name']]
        
        observation_id = self.pg_session_C._Retrieve_observation_id_from_observation(main_query_D)

        if not observation_id:

            print ('.  ❌ ERROR: could not retrieve observation id for measurement record')

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

        if self.verbose > 1:

            print ('Adding JSON data to Postgres')

        schema_table_query_D = self._Get_process_schema_table()

        if self.verbose > 1:

            print ('Schema table query dictionary retrieved')

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

                print ('.  ❌ ERROR: could not retrieve record for %s.%s' % (dst_schema, dst_main_table))

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

            column_report_name = ",".join([item[0] for item in table_keys])

        else:

            column_report_name = main_query_D['name']

        # Check it the record is already registered in the database
        record_id = self.pg_session_C._Single_search_tab_keys(
            self._Lower_text_values(main_query_D, dst_schema, dst_main_table),
            ['id'], dst_schema, dst_main_table
        )

        if record_id == 'fk_error':

            print ('.  ❌ ERROR: could not retrieve foreign key for %s.%s' % (dst_schema, dst_main_table))

            return None

        elif record_id and self.process_S.process.delete:

            self._Delete('id = %s' %(record_id[0]), dst_schema, dst_main_table, column_report_name)

            #replace the id with model_id in the wehere_statement
            where_statement = where_statement.replace('id', 'model_id')

            for schema_table in schema_table_query_D:

                schema, table  = schema_table.split('.')

                self._Delete(where_statement,schema, table , column_report_name)
                # TG TODO check if this is the correct printout
                print ('.   ✅ Record %s deleted from %s.%s' %(column_report_name, schema, table))

            return None

        elif not record_id and self.process_S.process.delete:

            if self.verbose > 1:

                print ('.   ✅ Nothing to delete, record %s not found' %(column_report_name))

            return None

        elif record_id and self.process_S.process.overwrite:

            success = self._Update(main_query_D,'hardware', 'model', column_report_name)

            if success and self.verbose > 1:

                print ('.   ✅ Nothing to delete, record %s not found' %(column_report_name))

            return None

        elif record_id:

            if self.verbose > 1:

                print ('.   ✅ Record %s already registered, use overwrite to update' %(column_report_name))

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

            print ('.  ❌ ERROR: could not retrieve record_id after inserting device model to %s.%s' % (dst_schema, dst_main_table))

            return None
        
        main_table_id = '%s_id' %(dst_main_table)

        if 'name' in main_query_D:

            name = main_query_D['name']

        else:

            name = 'record %s' %(record_id[0])

        self._Define_specifics(main_query_D,record_id, schema_table_query_D, main_table_id, name)

    def _Define_specifics(self, main_query_D, record_id, schema_table_query_D, main_table_id, name):
        ''' Define device model specifics
        '''

        def Split_arrays():

            updated_query_D =  deepcopy(schema_table_query_D[schema_table])

            return_bool = False
            for item in schema_table_query_D[schema_table]:

                if item.endswith('_array') and '__' in item:

                    return_bool = True

                    item_parts = item.split('__')

                    if item_parts[0].endswith('_array'):

                        continue

                    the_item = schema_table_query_D[schema_table][item]
                    
                    value_csv = the_item[the_item.index("{") + 1:the_item.rindex("}")]

                    value_in_L = value_csv.split(',')

                    # replace the old item with an item with '_array' removed in updated_query_D
                    updated_query_D.pop(item)

                    new_item = item.removesuffix("_array")

                    for value_in in value_in_L:

                        updated_query_D[new_item] = value_in

                        self._Manage_specifics(updated_query_D, updated_query_D, main_table_id, schema, table, record_id[0], name)                    
                
            return return_bool

        # Loop over all devie model specific tables and insert/update/delete 
        for schema_table in schema_table_query_D:

            # Add record_id to the query 
            schema_table_query_D[schema_table][main_table_id] = record_id[0]

            if self.verbose > 1:

                print ('      Managing specifics in sub schema.table:', schema_table)

            # split the schema.table string into schema and table
            schema, table  = schema_table.split('.')
            # TG TODO 
            if len(schema_table_query_D[schema_table]) <= 3 and next(iter(schema_table_query_D[schema_table])).endswith('_array') and \
                '__' in next(iter(schema_table_query_D[schema_table])):
                # This is an array record that should be split into multiple records for a 1 to many relationship. The query_D should contain the id of the parent record and the array field with the values to be split into multiple records.

                keys = list(schema_table_query_D[schema_table].keys())

                keys.remove(main_table_id)

                new_query_D = {}

                for key in keys:

                    new_query_D[key] = schema_table_query_D[schema_table][key].strip("{}").split(",")

                for item in range(len(new_query_D[keys[0]])):

                    item_query_D = {main_table_id: schema_table_query_D[schema_table][main_table_id]}

                    for key in keys:

                        column_alias = key.split('__')[1].split('_array')[0].replace('name','id')

                        #column_alias = key.split('_array')[0]

                        column_alias += '__%s' %(column_alias.replace('id','name'))

                        item_query_D[column_alias] = new_query_D[key][item].strip()

                    self._Define_specifics(main_query_D, record_id, {schema_table: item_query_D}, main_table_id, name)

                continue
            
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
    
                print ('.  ❌ ERROR: could not retrieve @-record for %s.%s' % (schema, table))

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

            where_statement = '%s = %s' %(main_table_id, record_value)

            self.pg_session_C._Delete_(schema, table, where_statement)

        elif self.process_S.process.verbose > 1:

            print ('.     ✅ Record already registered in table %s, use overwrite to update' %(table))

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

            print ('.  ❌ ERROR: could not retrieve wavelength cardinality for provision %s' % provision_name)

            return None

        if record != value_len:

            print ('.  ❌ ERROR: length of input array does not match wavelength cardinality for provision %s. Expected %s values, got %s values.' % (provision_name, record, value_len))

            return None
        
        return True