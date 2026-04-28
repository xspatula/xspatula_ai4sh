'''
pg_get_schema_table.py
Module for retrieving schema and table as defined in the system process defintion
Created on 18 Nov 2021
Updated 4 Jan 2024

@author: thomasgumbricht
'''

class Get_schema_table:
    """
    @brief Constructor for Get_schema_table class.
    """
    def __init__(self):
        """
        @brief Constructor for Get_schema_table class.

        @return None
        """

        pass

    def _Get_process_schema_table(self):
        """
        """
        schema_table_query_D = {}

        # Get the parameter names and values from the process structure variable
        parameter_L = []

        for parameter_id in self.process_S.process.parameters.__dict__:

            parameter_L.append(parameter_id)

        value_L = []

        for item in parameter_L:

            value_L.append(getattr(self.process_S.process.parameters, item))

        # create a dictionary of the parameter names and values
        parameter_value_D = dict(zip(parameter_L, value_L))

        # Select distinct schema.table that apply for the device data
        sql = "SELECT DISTINCT in_schema, in_table FROM process.process_parameter_schema_table WHERE process = '%s';" %(self.process_S.process.process)

        records = self.pg_session_C._Execute_search_all_sql(sql)

        # loop over the schema.table pairs and create a dictionary with the schema.table as key
        # and the parameter names and values as values.
        for record in records:

            schema_table = '%s.%s' %(record[0], record[1])

            schema_table_query_D[schema_table] = {}

            # create a query for selecting which schema and table each parameters belongs to
            queryD = {'s':record[0], 't':record[1], 'spid':self.process_S.process.process}

            # construct the sql for searching after the parameter id (parameter_id) that goes to this schema.table
            sql = "SELECT parameter FROM process.process_parameter_schema_table WHERE process = '%(spid)s' \
                AND in_schema = '%(s)s' AND in_table = '%(t)s';" %queryD

            recs = self.pg_session_C._Execute_search_all_sql(sql)

            # loop over the complete list of parameters
            for key in list(parameter_value_D):

                # the parameters is found in the schema.table of the sql search, add it to query for this schema.table
                if key in (item[0] for item in recs):

                    schema_table_query_D[schema_table][key] = parameter_value_D[key]

            # Check if the this is a special @ table with free parameters
            if '@%' in schema_table_query_D[schema_table]:

                for key in list(parameter_value_D):

                    if key.startswith('@%') and len(key) > 2:

                        #schema_table_query_D[schema_table][key[2:]] = parameter_value_D[key]
                        schema_table_query_D[schema_table][key] = parameter_value_D[key]
                
                schema_table_query_D[schema_table].pop('@%')
                parameter_value_D.pop('@%')


            if '@' in schema_table_query_D[schema_table]:

                for key in list(parameter_value_D):

                    if key.startswith('@') and len(key) > 1:

                        #schema_table_query_D[schema_table][key.strip('@')] = parameter_value_D[key]
                        schema_table_query_D[schema_table][key] = parameter_value_D[key]

                    #if key.startswith('@%'):

                    #    schema_table_query_D[schema_table][key[2:]] = parameter_value_D[key]

                schema_table_query_D[schema_table].pop('@')
                parameter_value_D.pop('@')

                    
        return schema_table_query_D