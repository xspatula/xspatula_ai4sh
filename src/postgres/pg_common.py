"""
 @file pg_common.py

 @brief Common PostgreSQL query helper module.

 @details Provides reusable SQL builders and record management helpers for select,
 insert, update, delete, and key lookup operations across schemas and tables.

 *Version History*:
 - Created: 2018-02-21
 - Updated: 2025-09-09 (Split common helpers into a dedicated module and added documentation)
 - Updated: 2026-03-11 (Added psycopg2.sql identifiers and parameterized queries)

 @author Thomas Gumbricht

 @date Created: 2018-02-21
 @date Updated: 2025-09-09 (Split common helpers into a dedicated module and added documentation)
 @date Updated: 2026-03-11 (Added psycopg2.sql identifiers and parameterized queries)
"""

from copy import deepcopy
from tabnanny import verbose

from psycopg2 import sql as pgsql


# Operators permitted in _Dict_to_select WHERE clauses
_VALID_OPS = frozenset({'=', '!=', '<>', '<', '>', '<=', '>=', 'LIKE', 'ILIKE', 'IS', 'IS NOT', 'BETWEEN'})

class PG_common():
    """
    @brief Base class with reusable PostgreSQL command helpers.

    @details Provides low-level query execution and SQL-building utilities shared
    by higher-level PostgreSQL session and process-management classes.
    """

    def __init__(self):
        """
        @brief Initialize shared PostgreSQL command helpers.
        """
        pass

    def _Execute_search_single_sql(self, sql, params=None):
        """
        @brief Execute the sql statement and return one record.

        @param sql The sql statement to execute (string or psycopg2.sql.Composed).
        @param params Optional tuple/list of query parameters.

        @return The first record found.
        """
        
        self.cursor.execute(sql, params)

        return self.cursor.fetchone()

    def _Execute_search_all_sql(self, sql, params=None):
        """
        @brief Execute the sql statement and return all records.

        @param sql The sql statement to execute (string or psycopg2.sql.Composed).
        @param params Optional tuple/list of query parameters.

        @return All records found.
        """

        self.cursor.execute(sql, params)

        return self.cursor.fetchall()

    def _Execute_commit_sql(self, sql, params=None):
        """
        @brief Execute and commit the sql statement.

        @param sql The sql statement to execute (string or psycopg2.sql.Composed).
        @param params Optional tuple/list of query parameters.

        @return None
        """

        self.cursor.execute(sql, params)

        self.conn.commit()

    def _Execute_return_commit_sql(self, sql, params=None):
        """
        @brief Execute, fetch the serial number, and commit.

        @param sql The sql statement to execute (string or psycopg2.sql.Composed).
        @param params Optional tuple/list of query parameters.

        @return The serial_nr of the inserted row.
        """

        self.cursor.execute(sql, params)

        serial_nr = self.cursor.fetchone()[0]

        self.conn.commit()

        return serial_nr

    def _Get_table_keys(self, schema, table):
        """
        @brief Get the primary keys of a table.

        @param schema The schema of the table.
        @param table The name of the table.

        @return A list of primary key column names.
        """

        sql = """
            SELECT column_name
            FROM information_schema.table_constraints
            JOIN information_schema.key_column_usage
                USING (constraint_catalog, constraint_schema, constraint_name,
                       table_catalog, table_schema, table_name)
            WHERE constraint_type = 'PRIMARY KEY'
              AND table_schema = %s
              AND table_name = %s;
        """

        self.tab_keys = self._Execute_search_all_sql(sql, (schema, table))

        return self.tab_keys
    
    def _Get_unique_keys(self, schema, table):
        """
        @brief Get the unique keys of a table.

        @param schema The schema of the table.
        @param table The name of the table.

        @return A list of primary key column names.
        """

        sql = """
            SELECT column_name
            FROM information_schema.table_constraints
            JOIN information_schema.key_column_usage
                USING (constraint_catalog, constraint_schema, constraint_name,
                       table_catalog, table_schema, table_name)
            WHERE constraint_type = 'UNIQUE'
              AND table_schema = %s
              AND table_name = %s;
        """

        self.unique_keys = self._Execute_search_all_sql(sql, (schema, table))

        return self.unique_keys

    def _Get_text_columns(self, schema, table):
        """
        @brief Get the set of text/char/varchar column names for a table.

        @param schema The schema of the table.
        @param table The name of the table.

        @return A set of lowercase column names whose type is text, character, or
                character varying.
        """

        sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND data_type IN ('text', 'character varying', 'character');
        """

        rows = self._Execute_search_all_sql(sql, (schema, table))

        return {row[0].lower() for row in rows}

    def _Get_table_columns(self, schema, table):
        """
        @brief Get all column names for a table.

        @param schema The schema of the table.
        @param table The name of the table.

        @return A set of lowercase column names.
        """

        sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s;
        """

        rows = self._Execute_search_all_sql(sql, (schema, table))

        return {row[0].lower() for row in rows}

    def _Dict_to_select(self, queryD):
        """
        @brief Converts a dictionary to parameterized WHERE conditions.

        @param queryD A dictionary where keys are column names and values are dicts
               with 'op' (SQL operator) and 'val' (value or tuple for BETWEEN).

        @details
        - Column names are wrapped with pgsql.Identifier to prevent injection.
        - Operators are validated against a whitelist.
        - Values are returned as parameters, never interpolated into SQL.

        @return Tuple (conditions_sql, params):
               - conditions_sql: psycopg2.sql.Composed fragment (no WHERE keyword).
               - params: list of parameter values for cursor.execute().

        @raises ValueError if an unsupported operator is supplied.
        """

        conditions = []
        params = []

        for key in queryD:

            col = key.replace('#', '')
            op = queryD[key]['op'].upper().strip()
            val = queryD[key]['val']

            if op not in _VALID_OPS:
                raise ValueError('_Dict_to_select: unsupported operator %r' % op)

            if op == 'BETWEEN':

                conditions.append(
                    pgsql.SQL('{} BETWEEN %s AND %s').format(pgsql.Identifier(col))
                )
                params.extend([val[0], val[1]])

            else:

                conditions.append(
                    pgsql.SQL('{} ').format(pgsql.Identifier(col))
                    + pgsql.SQL(op + ' %s')
                )
                params.append(val)

        if not conditions:
            return pgsql.SQL('TRUE'), []

        return pgsql.SQL(' AND ').join(conditions), params

    def _Dict_to_columns_values(self, queryD, schema, table):
        """
        @brief Converts a dictionary to column name and value lists for insertion.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to insert.
        @param schema The schema of the table.
        @param table The name of the table.

        @return None. Sets self.query with keys 'schema', 'table', 'cols' (list of
                column name strings), and 'vals' (list of Python values).
        """

        self.query = {
            'schema': schema,
            'table': table,
            'cols': list(queryD.keys()),
            'vals': list(queryD.values()),
        }

    def _Single_search(self, queryD, paramL, schema, table):
        """
        @brief Select a single record from any schema.table.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to search for (or dicts with 'op'/'val').
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.

        @return The first record found.
        """

        selectQuery = {}

        for item in queryD:

            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        return self._Execute_search_single_sql(sql, params)

    def _Single_search_between(self, queryD, paramL, schema, table, between_param_L, tolerance):
        """
        @brief Select a single record with BETWEEN for specified parameters.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.
        @param between_param_L A list of column names for which to apply BETWEEN.
        @param tolerance The tolerance value to define the range for BETWEEN.

        @return The first record found.
        """

        selectQuery = {}

        for item in queryD:

            if item in between_param_L:
                selectQuery[item] = {
                    'op': 'BETWEEN',
                    'val': (queryD[item] - tolerance, queryD[item] + tolerance)
                }
            elif isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        return self._Execute_search_single_sql(sql, params)

    def _Single_search_tab_keys(self, queryD, paramL, schema, table):
        """
        @brief Select a single record using the table's primary keys.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.

        @return A tuple containing the first record found and the WHERE clause used.
        """

        search_L = self._Create_search_sql_from_tab_keys(queryD, schema, table)

        if not search_L or search_L == 'fk_error':
            return search_L

        unique_keys, where_sql, where_params, _ = search_L

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        return self._Execute_search_single_sql(sql, where_params)
    
    def _Single_search_foreign_key(self, queryD, paramL, schema, table):
        """
        @brief Select a single record using the table's unique keys (exact-case match).

        @details Identical in behaviour to _Single_search_tab_keys but named
                 separately so callers can express intent (searching by a foreign-key
                 value rather than a primary key).  Values are passed to the database
                 exactly as supplied; no case conversion is applied.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.

        @return The first record found, or None.
        """

        search_L = self._Create_search_sql_from_tab_keys(queryD, schema, table)

        if not search_L:
            pass
            '''
            # Inverted foreign key: find keys whose last '_'-split segment is 'name'
            name_keys = [key for key in queryD if key.split('_')[-1] == 'name']

            for key in name_keys:
                queryD['name'] = queryD[key].lower()

                tab_cols = self._Get_table_columns(schema, table)
                filtered_queryD = {k: v for k, v in queryD.items() if k in tab_cols}
                
                paramL = list(filtered_queryD.keys())

                search_L = self._Create_search_sql_from_tab_keys(filtered_queryD, schema, table)

                if search_L:
                    break
                '''
        if not search_L or search_L == 'fk_error':
            return search_L

        _, where_sql, where_params, _ = search_L

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        return self._Execute_search_single_sql(sql, where_params)

    def _Insert_record(self, queryD, schema, table):
        """
        @brief Insert a new record into the specified schema and table.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to insert.
        @param schema The schema of the table.
        @param table The name of the table.

        @return True if the insertion was successful, False otherwise.
        """

        self._Dict_to_columns_values(queryD, schema, table)

        cols = self.query['cols']
        vals = self.query['vals']

        sql = pgsql.SQL('INSERT INTO {}.{} ({}) VALUES ({});').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            pgsql.SQL(', ').join(map(pgsql.Identifier, cols)),
            pgsql.SQL(', ').join([pgsql.Placeholder()] * len(vals))
        )

        try:
            self._Execute_commit_sql(sql, vals)
            return True

        except Exception as e:
            print ('    ⚠️ WARNING - Insertion failed for table %s.%s' % (schema, table))
            log = '      error: %s' % e
            self.log(log, '      ')
            return False

    def _Insert_return_record(self, queryD, schema, table):
        """
        @brief Insert a new record and return the generated serial_nr.

        @param queryD A dictionary where keys are column names and values are
               the corresponding values to insert.
        @param schema The schema of the table.
        @param table The name of the table.

        @return The serial_nr of the inserted row, or None on failure.
        """

        self._Dict_to_columns_values(queryD, schema, table)

        cols = self.query['cols']
        vals = self.query['vals']

        sql = pgsql.SQL('INSERT INTO {}.{} ({}) VALUES ({}) RETURNING serial_nr;').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            pgsql.SQL(', ').join(map(pgsql.Identifier, cols)),
            pgsql.SQL(', ').join([pgsql.Placeholder()] * len(vals))
        )

        try:
            return self._Execute_return_commit_sql(sql, vals)

        except Exception as e:
            print ('    ❌ ERROR - Insertion return failed for table %s.%s' % (schema, table))
            log = '      error: %s' % e
            self.log(log, '      ')
            return None

    def _Convert_id_code(self, queryD):
        """
        @brief Converts keys in a query dictionary from code/id/source/model notation
               to database foreign key values.

        @param queryD Dictionary containing query parameters, possibly with keys using
               '__code', '__id', '__source', or '__model' suffixes.

        @return Updated dictionary with keys converted to database foreign key values,
                or None if a required foreign key is not found.
        """

        def _Get_code(key):
            rec = self._Check_get_foreign_code_key(key, queryD[key])
            if not rec:
                error_msg = '❌ ERROR Can not find foreign key: %s' % (queryD[key])
                error_msg += '    queryD: %s' % (queryD)
                self.log(error_msg)
                return None
            updated_query_D.pop(key)
            updated_query_D[key.split('__')[0]] = rec[0]
            return rec

        def _Get_id(key, value):
            #rec = self._Check_get_foreign_id_key(key, value)
            rec = self._Check_get_foreign_key(key, value)
            if not rec:
                #error_msg = '❌ ERROR Can not find foreign key: %s\n' % (value)
                #error_msg += '    queryD: %s' % (queryD)
                #self.log(error_msg)
                return 'fk_error'
            updated_query_D.pop(key)
            updated_query_D[key.split('__')[0]] = rec[0]
            return rec

        def _Get_array_id(key, value):
            #rec = self._Check_get_foreign_id_key(key, value)
            rec = self._Check_get_foreign_key(key, value)
            if not rec:
                error_msg = '❌ ERROR Can not find foreign key: %s\n' % (value)
                error_msg += '    queryD: %s' % (queryD)
                self.log(error_msg)
                return None
            return rec

        updated_query_D = deepcopy(queryD)

        for key in queryD:

            if '__' in key:

                if key.endswith('_array'):

                    updated_query_D.pop(key)

                    id_key = key.removesuffix("_array")

                    value_csv = queryD[key][queryD[key].index("{") + 1:queryD[key].rindex("}")]

                    value_in_L = value_csv.split(',')

                    value_out_L = []

                    for value in value_in_L:

                        if not value or value in ('', 'none', 'null'):
                            continue

                        if value.endswith('__code'):
                            rec = _Get_code(id_key)
                        else:
                            rec = _Get_array_id(id_key, value)

                        if not rec:
                            return None

                        value_out_L.append(str(rec[0]))

                    if value_out_L:
                        updated_query_D[id_key.split('__')[0]] = '{%s}' % ','.join(value_out_L)

                elif key.endswith('__code'):

                    rec = _Get_code(key)

                    if not rec:
                        return None

                else:

                    rec = _Get_id(key, queryD[key])

                    if not rec:
                        
                        return 'fk_error'

        return updated_query_D

    def _Create_search_sql_from_tab_keys(self, queryD, schema, table):
        """
        @brief Build a parameterized WHERE clause based on the primary keys of a table.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param schema The schema of the table.
        @param table The name of the table.

        @return A list containing:
            - A list of primary key column names.
            - Composed WHERE conditions (psycopg2.sql.Composed, no WHERE keyword).
            - List of parameter values for those conditions.
            - The updated query dictionary with foreign keys converted.
        Or None if conversion fails.
        """

        updated_query_D = self._Convert_id_code(queryD)

        if not updated_query_D or updated_query_D == 'fk_error':
            return updated_query_D

        #tab_keys = self._Get_table_keys(schema, table)

        tab_unique = self._Get_unique_keys(schema, table)

        #if not tab_keys:
        #    tab_keys = list(updated_query_D.keys())
        #else:
        #    tab_keys = [item[0] for item in tab_keys]

        if not tab_unique:
            tab_unique = list(updated_query_D.keys())
        else:
            tab_unique = [item[0] for item in tab_unique]

        selectQuery = {}

        #for item in tab_keys:
        # Should be enough a single UNIQUE key to identify a record, so we use unique keys instead of primary keys here
        unique_key_flag = False
        for item in tab_unique:
            
            if item not in updated_query_D:
                # Special solution to look for alias instead of name
                # TG TODO DEMANDS MORE
                #if item == 'alias':
                #    continue
                #    unique_key_flag = True
                error_msg = '⚠️ WARNING unique key <%s> not found in queryD for table %s.%s' % (
                    item, schema, table
                )
                error_msg += '\n            searching for inverted name key instead...'
                self.log(error_msg)
                return None

            selectQuery[item] = {'op': '=', 'val': updated_query_D[item]}

        where_sql, where_params = self._Dict_to_select(selectQuery)

        #return tab_uniqe, where_sql, where_params, updated_query_D
        return tab_unique, where_sql, where_params, updated_query_D

    def _Check_get_foreign_code_key(self, key, value):
        """
        @brief Get foreign key based on code.

        @param key The key in the query dictionary that ends with '__code'.
        @param value The value associated with the key in the query dictionary.

        @return The record containing the foreign key, or None if not found.
        """

        keyL = key.split('__')

        if value == 0:
            return (0,)

        if value.isdigit():
            queryD = {'value': int(value)}
        else:
            queryD = {'alias': value}

        parameter_L = ['value', 'alias']

        rec = self._Single_search(queryD, parameter_L, 'utility', keyL[0])

        return rec

    def _Check_get_foreign_id_key(self, key, value):
        """
        @brief Get foreign key based on ID.

        @param key The key in the query dictionary that ends with '__id'.
        @param value The value associated with the key in the query dictionary.

        @return The record containing the foreign key, or None if not found.
        """

        keyL = key.split('__')

        if value in [0, 'none', 'null', '']:
            return (0,)
 
        queryD = {'id_code': keyL[1]}

        parameter_L = ["id_code", "id_schema", "id_table", "name_column", "alt_column"]

        search_rec = self._Single_search(queryD, parameter_L, 'utility', 'id_code_schema_table')

        if not search_rec:
            error_msg = '   ❌ ERROR Can not find foreign key: %s with queryD\n   %s' % (keyL[1], queryD)
            self.log(error_msg)
            return search_rec

        parameter_L = ["id"]

        if value.isdigit():

            queryD = {'id': value}
            rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

        else:

            queryD = {search_rec[3]: value}
            rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

            if not rec:
                queryD = {search_rec[3]: value.lower()}
                rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

                if not rec and search_rec[3] != search_rec[4]:

                    queryD = {search_rec[4]: value}
                    rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

                    if not rec:
                        queryD = {search_rec[4]: value.lower()}
                        rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

        return rec
    
    def _Check_get_foreign_key(self, key, value):
        """
        @brief Get foreign key based on ID.

        @param key The key in the query dictionary that ends with '__id'.
        @param value The value associated with the key in the query dictionary.

        @return The record containing the foreign key, or None if not found.
        """

        # Look for foregin key using the name of the foregign key itself, if not found try the table unitity.foreign_key
        keyL = key.split('__')

        if value in [0, 'none', 'null', '']:
            return (0,)
        
        table_name = key.split('__')[0].rsplit('_', 1)[0]
        
        sql = "SELECT table_schema FROM information_schema.tables WHERE table_name = '%s';" %table_name

        schema_name = self._Execute_search_single_sql(sql)

        error_msg = ''

        if schema_name:
            # SELECT the unique columns from the matching table
            unique_columns = self._Get_unique_keys(schema_name[0], table_name)
            for column in unique_columns:

                # It is not possible to search an fk value in a column that ends with _id, because it is not possible to know the name of the column in the foreign table that should be searched for the value
                if column[0].endswith('_id'):
                    continue
                
                queryD = {column[0]: value}
                parameter_L = ['id']
                rec = self._Single_search(queryD, parameter_L, schema_name[0], table_name)

                if rec:

                    if verbose > 2:

                        msg = '    Found foreign key <%s> in schema.table <%s.%s> ' % (value, schema_name[0], table_name)

                        print (msg)

                    return rec
            
            error_msg += '     🔺 WARNING Cannot find foreign key <%s> in schema.table <%s.%s>\n ' %(value, schema_name[0], table_name)
            
            error_msg += '    🔑 Check spelling or add the value <%s> to the column <%s> in schema.table <%s.%s>' % (value, column[0], schema_name[0], table_name)

        if verbose > 2:
        
            msg = '   Searching utility.foreign_key with key %s' % (keyL[1])
        
            print(msg)
        #queryD = {'id_code': keyL[1]}

        queryD = {'foreign_key': keyL[0]}

        #parameter_L = ["id_code", "id_schema", "id_table", "name_column", "alt_column"]
        parameter_L = ["foreign_key", "dst_schema", "dst_table", "dst_search_column", "dst_alt_search_column"]

        search_rec = self._Single_search(queryD, parameter_L, 'utility', 'foreign_key')

        if not search_rec:
            if error_msg:
                error_msg += '\n     🔑 or enter schema, table and column(s) to search in utility.foreign_key.' 
            else:
                error_msg = '   ❌ ERROR Can not find foreign key: %s with queryD\n   %s' % (keyL[1], queryD)
            print (error_msg)
            #self.log(error_msg)
            return None

        parameter_L = ["id"]

        if value.isdigit():

            queryD = {'id': value}
            rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

        else:

            queryD = {search_rec[3]: value}
            rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

            if not rec:
                queryD = {search_rec[3]: value.lower()}
                rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

                if not rec and search_rec[3] != search_rec[4]:

                    queryD = {search_rec[4]: value}
                    rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

                    if not rec:
                        queryD = {search_rec[4]: value.lower()}
                        rec = self._Single_search(queryD, parameter_L, search_rec[1], search_rec[2])

        return rec

    def _Check_insert_single_record(self, queryD, schema, table, overwrite=False, delete=False):
        """
        @brief Check if a record exists based on primary keys, and insert, overwrite,
               or delete as specified.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for or insert.
        @param schema The schema of the table.
        @param table The name of the table.
        @param overwrite If True, overwrite the existing record if found.
        @param delete If True, delete the existing record if found.

        @return The existing record if found and deleted or overwritten, True if a new
                record was inserted, or None if no action was taken.
        """

        search_L = self._Create_search_sql_from_tab_keys(queryD, schema, table)

        if not search_L or search_L == 'fk_error':
            return search_L

        tab_unique, where_sql, where_params, updated_query_D = search_L  # tab_unique used below

        select_sql = pgsql.SQL('SELECT * FROM {}.{} WHERE {};').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        rec = self._Execute_search_single_sql(select_sql, where_params)

        if rec is not None and delete:

            delete_sql = pgsql.SQL('DELETE FROM {}.{} WHERE {};').format(
                pgsql.Identifier(schema),
                pgsql.Identifier(table),
                where_sql
            )
            self._Execute_commit_sql(delete_sql, where_params)
            return rec

        elif rec is not None and overwrite:

            delete_sql = pgsql.SQL('DELETE FROM {}.{} WHERE {};').format(
                pgsql.Identifier(schema),
                pgsql.Identifier(table),
                where_sql
            )
            self._Execute_commit_sql(delete_sql, where_params)
            return self._Insert_record(queryD, schema, table)

        elif rec is None and not delete:

            return self._Insert_record(updated_query_D, schema, table)

        else:

            return True

    def _Check_insert_return_single_record(self, query_D, schema, table):
        """
        @brief Check if a record exists based on primary keys; insert and return serial_nr.

        @param query_D A dictionary where keys are column names and values to insert.
        @param schema The schema of the table.
        @param table The name of the table.

        @return The serial_nr of the inserted row, or None on failure.
        """

        return self._Insert_return_record(query_D, schema, table)

    def _Check_update_single_record(self, queryD, schema, table):
        """
        @brief Update a single record in the specified schema and table based on primary keys.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to update.
        @param schema The schema of the table.
        @param table The name of the table.

        @return True on success, None on failure.
        """

        updated_query_D = self._Convert_id_code(queryD)

        if not updated_query_D:
            return None

        search_L = self._Create_search_sql_from_tab_keys(updated_query_D, schema, table)

        if not search_L:
            return None

        tab_keys, where_sql, where_params, updated_query_D = search_L

        select_sql = pgsql.SQL('SELECT * FROM {}.{} WHERE {};').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        if self.verbose >= 3:
            self.log('\n               %s' % select_sql)

        records = self._Execute_search_all_sql(select_sql, where_params)

        if len(records) != 1:
            return None

        # Columns to update are those not part of the primary key
        update_cols = [k for k in updated_query_D if k not in tab_keys]
        update_vals = [updated_query_D[k] for k in update_cols]

        set_clause = pgsql.SQL(', ').join(
            pgsql.SQL('{} = %s').format(pgsql.Identifier(col))
            for col in update_cols
        )

        update_sql = pgsql.SQL('UPDATE {}.{} SET {} WHERE {};').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            set_clause,
            where_sql
        )

        if self.verbose >= 3:
            self.log('\n               %s' % update_sql)

        self._Execute_commit_sql(update_sql, update_vals + where_params)

        return True

    def _Distinct_search(self, queryD, paramL, schema, table, distinctcolsL):
        """
        @brief Select distinct records from any schema.table based on specified columns.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.
        @param distinctcolsL A list of column names to apply the DISTINCT ON clause.

        @return A list of distinct records found.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))
        dcols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, distinctcolsL))

        sql = pgsql.SQL('SELECT DISTINCT ON ({}) {} FROM {}.{} WHERE {};').format(
            dcols_sql,
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        if self.verbose > 1:
            self.log(str(sql))

        return self._Execute_search_all_sql(sql, params)

    def _Count_search(self, queryD, schema, table):
        """
        @brief Count records in any schema.table based on specified conditions.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param schema The schema of the table.
        @param table The name of the table.

        @return The count of records found.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        sql = pgsql.SQL('SELECT COUNT(*) FROM {}.{} WHERE {};').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        rec = self._Execute_search_single_sql(sql, params)

        if rec is None:
            self.log(str(sql))

        return rec

    def _Single_join_search(self, queryD, paramL, joinTables):
        """
        @brief Select a single record from joined schema.table.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param joinTables A joined-tables expression string (e.g. 'a.table1 JOIN b.table2 ON ...').

        @return The record found or None if not found.

        @note joinTables is a pre-built SQL fragment from trusted internal code. It is
              embedded as a literal SQL string and must NOT contain user input.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        # joinTables is a trusted internal SQL fragment — not user input
        sql = pgsql.SQL('SELECT {} FROM {} WHERE {};').format(
            cols_sql,
            pgsql.SQL(joinTables),
            where_sql
        )

        rec = self._Execute_search_single_sql(sql, params)

        if rec is None:
            self.log(str(sql))

        return rec

    def _Multi_search(self, queryD, paramL, schema, table):
        """
        @brief Select multiple records from any schema.table.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.

        @return A list of records found.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        return self._Execute_search_all_sql(sql, params)

    def _Multi_search_list(self, queryD, anyD, paramL, schema, table):
        """
        @brief Select multiple records from any schema.table with = ANY(%s) for list columns.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for (equality conditions).
        @param anyD A dictionary where keys are column names and values are lists of
               values for ANY() conditions.
        @param paramL A list of column names to retrieve.
        @param schema The schema of the table.
        @param table The name of the table.

        @return A list of records found.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_conditions, params = self._Dict_to_select(selectQuery)

        # Append = ANY(%s) conditions for anyD — values passed as list parameters
        any_conditions = []

        for key in anyD:
            any_conditions.append(
                pgsql.SQL('{} = ANY(%s)').format(pgsql.Identifier(key))
            )
            params.append(list(anyD[key]))

        all_conditions = [where_conditions] + any_conditions if any_conditions else [where_conditions]

        combined_where = pgsql.SQL(' AND ').join(all_conditions)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {}.{} WHERE {};').format(
            cols_sql,
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            combined_where
        )

        if self.verbose > 1:
            self.log(str(sql))

        return self._Execute_search_all_sql(sql, params)

    def _Multi_join_search(self, queryD, paramL, joinTables):
        """
        @brief Select multiple records from joined schema.table.

        @param queryD A dictionary where keys are column names and values are the
               corresponding values to search for.
        @param paramL A list of column names to retrieve.
        @param joinTables A joined-tables expression string from trusted internal code.

        @return A list of records found.

        @note joinTables must NOT contain user input — it is embedded as literal SQL.
        """

        selectQuery = {}

        for item in queryD:
            if isinstance(queryD[item], dict):
                selectQuery[item] = queryD[item]
            else:
                selectQuery[item] = {'op': '=', 'val': queryD[item]}

        where_sql, params = self._Dict_to_select(selectQuery)

        cols_sql = pgsql.SQL(', ').join(map(pgsql.Identifier, paramL))

        sql = pgsql.SQL('SELECT {} FROM {} WHERE {};').format(
            cols_sql,
            pgsql.SQL(joinTables),
            where_sql
        )

        records = self._Execute_search_all_sql(sql, params)

        if len(records) == 0:
            print ('   ⚠️ WARNING - No records found for multi-join search with queryD: %s' % (queryD))
            self.log(str(sql))

        return records

    def _Delete_records(self, schema, table, where_D):
        """
        @brief Deletes records from the specified schema and table.

        @param schema The schema of the table.
        @param table The name of the table.
        @param where_D Dict of {column_name: value} specifying which records to delete.
               IMPORTANT: must be a dict, NOT a raw SQL string.

        @return None
        """

        selectQuery = {col: {'op': '=', 'val': val} for col, val in where_D.items()}

        where_sql, params = self._Dict_to_select(selectQuery)

        sql = pgsql.SQL('DELETE FROM {}.{} WHERE {};').format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_sql
        )

        self._Execute_commit_sql(sql, params)