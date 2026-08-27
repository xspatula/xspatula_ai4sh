"""
 @file setup_db.py

 @brief Main module for creating and seeding PostgreSQL databases.

 @details Defines setup routines for database creation, roles, schemas, tables,
 and JSON-driven seed operations used during installation workflows.

 *Version History*:
 - Created: 2022-06-01
 - Updated: 2024-01-10
 - Updated: 2025-09-02 (Doxygen-style documentation)
 - Updated: 2026-03-11 (Parameterized queries via psycopg2.sql)

 @author Thomas Gumbricht

 @date Created: 2022-06-01
 @date Updated: 2024-01-10
 @date Updated: 2025-09-02 (Doxygen-style documentation)
 @date Updated: 2026-03-11 (Parameterized queries via psycopg2.sql)
"""

# Standard library imports
from hashlib import md5
from os import path

# Third-party imports
from psycopg2 import sql as pgsql

# Package application imports

from src.utils import Pprint_parameter, Read_json, Update_dict, Struct

from src.postgres import User_netrc_credentials, User_login_pswd, PG_psycopg2_connect

INITIAL_DATABASE_NAME = 'postgres'

_LIB_SETUP_DIR = path.dirname(path.abspath(__file__))

REVOKE_D = Read_json(path.join(_LIB_SETUP_DIR, 'revoke_privileges.json'))

ROLES_D = Read_json(path.join(_LIB_SETUP_DIR, 'roles_grants.json'))

class parameter():
    """
    @brief Parameter manager for PostgreSQL setup workflows.

    @details Merges default and user-provided setup parameters, then exposes the
    merged configuration as structured attributes.
    """

    def __init__(self, default_parameter_D):
        """
        @brief Initialize the parameter manager with default values.

        @details Initializes the parameter object with a dictionary of default parameters.

        @param default_parameter_D Dictionary containing default parameter values.
        """

        self.default_parameter_D = default_parameter_D

    def _User_parameter(self, user_parameter_D):
        """
        @brief Updates user parameters with defaults and converts them to structured attributes.

        @details This method updates the provided user parameter dictionary by filling in any missing variables
        from the default parameter values. It also updates each process in the user parameters by
        filling in missing variables from the default process parameters. Finally, it converts the
        updated dictionary into structured class attributes.

        @param user_parameter_D Dictionary containing user-specified parameter values.

        @return None
        """
        self.user_parameter_D = user_parameter_D

        # update user_parameter_D by filling in missing variables from the default parameter values
        Update_dict(self.user_parameter_D, self.default_parameter_D)

        # update processes, by filling in missing variables from the default parameter values
        if 'process' in self.user_parameter_D and 'process' in self.default_parameter_D:

            for p in self.user_parameter_D['process']:

                Update_dict(p, self.default_parameter_D['process'][0])

        # Convert the updated dict to structured class attributes
        self.process_parameter_S = Struct(self.user_parameter_D)

class PG_session:
    """
    @brief Simplified PostgreSQL session class for initial database setup.

    @details Creates a lightweight connection wrapper used by setup routines before
    the full process-oriented session classes are needed.
    """

    def __init__(self, query_D):
        """
        @brief Initialize a simplified PostgreSQL setup session.

        @details Initializes a PostgreSQL session using the provided query dictionary.
        Decodes the base64-encoded password, constructs the connection string,
        and establishes a connection to the database. Also creates a cursor for
        executing SQL commands.

        @param query Dictionary containing connection parameters:
        - db: Database name
        - user: Username
        - pswd: Base64-encoded password

        @return None
        """

        self.conn, self.cursor = PG_psycopg2_connect(query_D)

    def _execute_db_command(self, scheme_params_D, user_parameter):
        """
        @brief Executes a sequence of database processes as specified by user parameters.

        @details This method initializes the process parameters using the provided default and user-specific
        parameter dictionaries. It then iterates through each process defined in the parameters,
        setting verbosity, delete, and overwrite flags. Depending on the process ID, it dispatches
        the appropriate database operation, such as creating schemas/tables, inserting/updating records,
        granting permissions, or deleting database objects.

        @param scheme_params_D Dictionary containing default parameter values for the database processes.
        @param user_parameter Dictionary containing user-specified parameter values for the database processes.

        Supported process IDs:
        - create_schema: Creates a new schema.
        - create_table: Creates a new table in a schema.
        - table_insert: Inserts records into a table.
        - table_update: Updates records in a table.
        - execute_sql: Executes one or more raw SQL statements verbatim (e.g. CREATE FUNCTION).
        - create_trigger: Creates a row-level trigger on a table, wired to an existing function.
        - grant: Grants user rights.
        - delete_database: Deletes the database.
        - delete_schema: Deletes a schema.
        - delete_table: Deletes a table.

        @return None
        """

        self.parameter = parameter(scheme_params_D)

        self.parameter._User_parameter(user_parameter)

        self.postgresDB_D = scheme_params_D['postgresdb']

        self.process_parameter_S = self.parameter.process_parameter_S

        # Get the processes as listed in the json object
        for process in self.process_parameter_S.process:

            self.verbose = process.verbose

            self.delete = process.delete

            self.overwrite = process.overwrite

            if self.verbose > 1:

                print ('#===== process parameters:')

                Pprint_parameter (vars(process))

                print ('#===== ')

            if process.process_id == 'create_schema':

                self._Create_schema(process.parameters.schema)

            elif process.process_id == 'create_table':

                self._Create_table(process.parameters.schema,process.parameters.table,process.parameters.command)

            elif process.process_id == 'table_insert':

                self._Table_insert(process.parameters.schema,process.parameters.table,process.parameters.command.columns,process.parameters.command.values)

            elif process.process_id == 'table_update':

                self._Table_update(process.parameters.schema,process.parameters.table,
                                  process.parameters.command.where,process.parameters.command.columns,process.parameters.command.values)

            elif process.process_id == 'execute_sql':

                self._Execute_sql(process.parameters.sql)

            elif process.process_id == 'create_trigger':

                self._Create_trigger(process.parameters.schema, process.parameters.table, process.parameters.trigger,
                                      process.parameters.timing, process.parameters.events, process.parameters.function)

            elif process.process_id == 'grant':

                self._Grant(process.parameters.command)

            elif process.process_id == 'delete_database':

                self._Delete_database_content()

                self._Deactivate_pg_users(scheme_params_D)

            elif process.process_id == 'delete_schema':

                self._Delete_schema(process.schema)

            elif process.process_id == 'delete_table':

                self._Delete_table(process.schema, process.table)

            else:

                msg =  'Initial command not found in initiate.initialize', process.process_id

                print (msg)

    def _Create_schema(self, schema):
        """
        @brief Creates a PostgreSQL schema, with options to overwrite or delete if it exists.

        This method checks if the specified schema exists in the database. If it exists and the
        overwrite or delete flags are set, the schema is dropped and optionally recreated.
        If the delete flag is set, the method returns after deletion. If the schema does not exist,
        it is created. Verbose output is provided based on the verbosity level.

        @param schema Name of the schema to create.

        @details
        - If the schema exists:
            - If overwrite or delete is True, drops the schema.
            - If delete is True, returns after dropping.
            - Otherwise, prints a message and returns.
        - If the schema does not exist, creates it.
        - Uses self.cursor for database operations and self.conn for committing changes.
        - Verbosity is controlled by self.verbose.

        @return None
        """

        if self.verbose:

            print ('.   Creating schema:',schema)

        # schema name is passed as a query parameter (value), not interpolated
        self.cursor.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s;",
            (schema,)
        )

        record = self.cursor.fetchone()

        if record is not None:

            if self.overwrite or self.delete:

                if self.verbose > 2:
                    print ('            DROP SCHEMA', schema)

                self.cursor.execute(
                    pgsql.SQL("DROP SCHEMA {}").format(pgsql.Identifier(schema))
                )

                self.conn.commit()

                if self.delete:

                    return
            else:

                if self.verbose > 1:

                    print ('            schema already exists')

                return

        if self.verbose > 2:

            print ('            CREATE SCHEMA', schema)

        self.cursor.execute(
            pgsql.SQL("CREATE SCHEMA {};").format(pgsql.Identifier(schema))
        )

        self.conn.commit()

    def _Create_table(self,schema,table,cmd):
        """
        @brief Creates a PostgreSQL table in the specified schema, with options to overwrite or delete if it exists.

        This method checks if the specified table exists in the given schema. If the table exists and the
        overwrite or delete flags are set, the table is dropped and optionally recreated. If the delete flag
        is set, the method returns after deletion. If the table does not exist, it is created using the provided
        column definitions. Verbose output is provided based on the verbosity level.

        @param schema Name of the schema where the table will be created.
        @param table Name of the table to create.
        @param cmd List of column definitions and constraints for the table.

        @details
        - Checks for the existence of the table in the specified schema.
        - If the table exists:
            - If overwrite or delete is True, drops the table.
            - If delete is True, returns after dropping.
            - Otherwise, prints a message and returns.
        - If the table does not exist, creates it using the provided column definitions.
        - Uses self.cursor for database operations and self.conn for committing changes.
        - Verbosity is controlled by self.verbose.
        - Column definitions in cmd are joined and embedded as SQL literals; they must
          come from trusted configuration files only.

        @return None
        """

        cmd_str = ",".join(cmd)

        # schema and table names are passed as query parameters (values)
        self.cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
            (schema, table)
        )

        record = self.cursor.fetchone()

        if record[0]:

            if self.overwrite or self.delete:

                if self.verbose > 1:

                    print ('            DROP TABLE', schema, table)

                self.cursor.execute(
                    pgsql.SQL("DROP TABLE {}.{} CASCADE;").format(
                        pgsql.Identifier(schema), pgsql.Identifier(table)
                    )
                )

                self.conn.commit()

                if self.delete:

                    return
            else:

                if self.verbose > 0:

                    print ('.   table %s.%s already exists' %(schema, table))

                return

        elif self.verbose:

            print ('.   Creating schema.table: %s.%s' %(schema, table))

        if self.verbose > 2:

            print ('            CREATE TABLE', schema, table)

        # schema and table use Identifier; cmd_str is column definitions from trusted config
        self.cursor.execute(
            pgsql.SQL("CREATE TABLE {}.{} ({});").format(
                pgsql.Identifier(schema),
                pgsql.Identifier(table),
                pgsql.SQL(cmd_str)
            )
        )

        self.conn.commit()

    def _Execute_sql(self, sql):
        """
        @brief Executes one or more raw SQL statements exactly as given.

        @details Used for DDL that create_table cannot express, such as CREATE FUNCTION
        bodies. Statements are trusted configuration content, not user input, and are
        executed verbatim, in order, each in its own commit. overwrite/delete are not
        interpreted here — idempotency (IF EXISTS / OR REPLACE) must be written into
        the SQL itself.

        @param sql A single SQL statement (str) or a list of SQL statements.

        @return None
        """

        stmt_L = sql if isinstance(sql, list) else [sql]

        for stmt in stmt_L:

            if self.verbose > 2:

                print ('            EXECUTE SQL', stmt.strip()[:80].replace('\n', ' '), '...')

            self.cursor.execute(stmt)

            self.conn.commit()

    def _Create_trigger(self, schema, table, trigger, timing, events, function):
        """
        @brief Creates a row-level trigger on a table, with options to overwrite or delete if it exists.

        @details Checks pg_trigger for an existing trigger with the given name on the given
        table. If it exists and overwrite or delete is set, drops it first. If delete is set,
        returns after dropping. Otherwise creates the trigger to call the given function for
        each row on the given events. The function itself must already exist (created via
        execute_sql) before this process runs.

        @param schema Schema of the table the trigger is attached to.
        @param table Table the trigger is attached to.
        @param trigger Name of the trigger.
        @param timing Trigger timing, "BEFORE" or "AFTER".
        @param events List of trigger events, e.g. ["INSERT","UPDATE","DELETE"].
        @param function Fully qualified trigger function name, e.g. "audit.if_modified_func".

        @details
        - timing, events and function are embedded as SQL keywords/identifiers from trusted
          configuration files only, following the same trust model as create_table's cmd.

        @return None
        """

        self.cursor.execute(
            "SELECT 1 FROM pg_trigger WHERE tgname = %s AND tgrelid = %s::regclass;",
            (trigger, '%s.%s' % (schema, table))
        )

        record = self.cursor.fetchone()

        if record is not None:

            if self.overwrite or self.delete:

                if self.verbose > 1:

                    print ('            DROP TRIGGER', trigger, 'ON', schema, table)

                self.cursor.execute(
                    pgsql.SQL("DROP TRIGGER {} ON {}.{};").format(
                        pgsql.Identifier(trigger), pgsql.Identifier(schema), pgsql.Identifier(table)
                    )
                )

                self.conn.commit()

                if self.delete:

                    return
            else:

                if self.verbose > 0:

                    print ('.   trigger %s on %s.%s already exists' % (trigger, schema, table))

                return

        elif self.verbose:

            print ('.   Creating trigger %s on %s.%s' % (trigger, schema, table))

        events_str = " OR ".join(events)

        if self.verbose > 2:

            print ('            CREATE TRIGGER', trigger, timing, events_str, 'ON', schema, table)

        self.cursor.execute(
            pgsql.SQL("CREATE TRIGGER {trigger} " + timing + " " + events_str +
                      " ON {schema}.{table} FOR EACH ROW EXECUTE FUNCTION " + function + "();").format(
                trigger=pgsql.Identifier(trigger),
                schema=pgsql.Identifier(schema),
                table=pgsql.Identifier(table)
            )
        )

        self.conn.commit()

    def _Get_table_keys(self,table):
        """
        @brief Retrieves the primary key columns and their data types for a specified table.

        This method queries the PostgreSQL information schema to obtain the names and data types
        of columns that are part of the primary key for the given table.

        @param table Name of the table for which primary key columns are to be retrieved.

        @details
        - Constructs and executes a SQL query joining table constraints, constraint column usage,
          and columns metadata to identify primary key columns.
        - Only columns with a 'PRIMARY KEY' constraint are returned.
        - Returns a list of tuples, each containing the column name and its data type.

        @return List of tuples [(column_name, data_type), ...] representing the primary key columns of the table.
        """

        sql = """
            SELECT c.column_name, c.data_type
            FROM information_schema.table_constraints tc
            JOIN information_schema.constraint_column_usage AS ccu
                USING (constraint_schema, constraint_name)
            JOIN information_schema.columns AS c
                ON c.table_schema = tc.constraint_schema
                AND tc.table_name = c.table_name
                AND ccu.column_name = c.column_name
            WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
        """

        self.cursor.execute(sql, (table,))

        return self.cursor.fetchall()
    
    def _Get_table_uniques(self,table):
        """
        @brief Retrieves the unique constraint columns for a specified table.

        This method queries the PostgreSQL information schema to obtain the names of columns that are part of
        unique constraints for the given table.

        @param table Name of the table for which unique constraint columns are to be retrieved.

        @details
        - Constructs and executes a SQL query joining table constraints and key column usage to identify
          unique constraint columns.
        - Only columns with a 'UNIQUE' constraint are returned.
        - Returns a list of column names that have unique constraints.

        @return List of column names that are part of unique constraints in the specified table.
        """

        sql = """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'UNIQUE' AND tc.table_name = %s;
        """

        self.cursor.execute(sql, (table,))

        return self.cursor.fetchall()

    def _Get_column_types(self, schema, table):
        """
        @brief Retrieves column names and their data types for a specified table.

        @param schema Name of the schema containing the table.
        @param table Name of the table to inspect.

        @return Dict mapping lowercase column name to data_type string.
        """

        sql = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s;
        """

        self.cursor.execute(sql, (schema, table))

        return {row[0].lower(): row[1] for row in self.cursor.fetchall()}

    def _Resolve_foreign_key_columns(self, columns, valueL):
        """
        @brief Resolves '<col>__<lookup>' columns in a table_insert command to plain
        foreign-key id columns/values, via the utility.foreign_key catalog.

        @details Same catalog table and lookup convention as
        src.postgres.pg_common.PG_common._Check_get_foreign_key (used by the
        process-driven manage_<table> insert pipeline): a value that is all digits is
        treated as an id already; otherwise it's looked up by dst_search_column (falling
        back to dst_alt_search_column) in dst_schema.dst_table. Reimplemented minimally
        here rather than reused, since this bootstrap-time PG_session predates the
        process-registration system and does not inherit PG_common - only the single-value
        lookup case is supported (no '__code'/'__array' variants), which is all a static
        table_insert seed row needs.

        @param columns List of column name strings, as given in the table_insert command.
        @param valueL List of value lists, one per row, positionally matching columns.

        @return (columns, valueL) with any '<col>__<lookup>' columns replaced by their
        resolved '<col>' id columns and values. Unresolvable lookups are left as-is (with
        a warning) rather than aborting the whole insert.
        """

        fk_indices = [i for i, col in enumerate(columns) if '__' in col]

        if not fk_indices:

            return columns, valueL

        resolved_columns = list(columns)

        fk_targets = {}

        for i in fk_indices:

            fk_key = columns[i].split('__')[0]

            self.cursor.execute(
                "SELECT dst_schema, dst_table, dst_search_column, dst_alt_search_column "
                "FROM utility.foreign_key WHERE foreign_key = %s;",
                (fk_key,)
            )

            rec = self.cursor.fetchone()

            if not rec:

                print('❌ ERROR - no utility.foreign_key entry for %s, leaving column as-is' % fk_key)

                continue

            fk_targets[i] = (fk_key,) + rec

            resolved_columns[i] = fk_key

        resolved_valueL = []

        for values in valueL:

            row = list(values)

            for i, (fk_key, dst_schema, dst_table, dst_search_column, dst_alt_search_column) in fk_targets.items():

                value = row[i]

                if isinstance(value, str) and value.isdigit():

                    row[i] = value

                    continue

                # Case-insensitive: unique text columns are lowercased on write elsewhere
                # in this same pipeline (see text_unique_cols in _Table_insert), so a seed
                # value typed in natural case (e.g. "Stockholm University") must still match.
                self.cursor.execute(
                    pgsql.SQL("SELECT id FROM {}.{} WHERE LOWER({}) = LOWER(%s) OR LOWER({}) = LOWER(%s);").format(
                        pgsql.Identifier(dst_schema), pgsql.Identifier(dst_table),
                        pgsql.Identifier(dst_search_column), pgsql.Identifier(dst_alt_search_column)
                    ),
                    (value, value)
                )

                rec = self.cursor.fetchone()

                if not rec:

                    print('❌ ERROR - could not resolve %s=%s via %s.%s' % (fk_key, value, dst_schema, dst_table))

                    row[i] = None

                else:

                    row[i] = rec[0]

            resolved_valueL.append(row)

        return resolved_columns, resolved_valueL

    def _Table_insert(self,schema,table,columns,valueL):
        """
        @brief Inserts, replaces, or deletes records in a specified schema table.

        This method handles the insertion of new records into a PostgreSQL table. If a record with the same
        primary key already exists, it can either be replaced or deleted based on the flags set in the
        class instance. Verbose output is provided based on the verbosity level.

        @param schema Name of the schema where the table is located.
        @param table Name of the table to insert records into.
        @param columns List of column name strings. A column may use the '<col>__<lookup>'
               convention (e.g. 'territory_id__territory_name') to resolve a foreign key by
               name via utility.foreign_key instead of requiring the raw id - see
               _Resolve_foreign_key_columns.
        @param valueL List of lists containing Python-native values to insert (str, int, float, bool).
               IMPORTANT: values must be plain Python types, NOT SQL-formatted strings.
               JSON example: [["Alice", 30, true]] not [["'Alice'", "30", "TRUE"]]

        @details
        - Checks if the specified table exists in the schema.
        - If the table exists:
            - If overwrite or delete is True, deletes the existing record.
            - If delete is True, continues to the next record.
        - If the table does not exist, prints an error message.
        - Uses parameterized queries throughout to prevent SQL injection.

        @return None
        """

        columns, valueL = self._Resolve_foreign_key_columns(columns, valueL)

        # schema and table names as query parameters
        self.cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
            (schema, table)
        )

        record = self.cursor.fetchone()

        if not record[0]:

            print ('❌ ERROR Schema.table %s.%s does not exist' %(schema,table))

            return None

        if self.verbose:

            print ('.   Inserting records to schema.table: %s.%s' %(schema, table))

        #table_keys = self._Get_table_keys(table)

        table_uniques = self._Get_table_uniques(table)

        #key_cols = [str(item[0].lower()) for item in table_keys]
        unique_cols = [str(item[0].lower()) for item in table_uniques]

        if not unique_cols:

            table_keys = self._Get_table_keys(table)

            unique_cols = [str(item[0].lower()) for item in table_keys]

        # Identify text-type unique columns that should be lowercased
        col_types = self._Get_column_types(schema, table)

        TEXT_TYPES = {'text', 'character varying', 'character'}

        text_unique_cols = {col for col in unique_cols if col_types.get(col) in TEXT_TYPES}

        for values in valueL:

            # Build WHERE clause for primary key lookup using identifiers + params
            key_conditions = []
            key_params = []

            for x, col in enumerate(columns):

                if col.lower().strip() in unique_cols:

                    key_conditions.append(
                        pgsql.SQL("{} = %s").format(pgsql.Identifier(col))
                    )

                    val = values[x]

                    if col.lower().strip() in text_unique_cols and isinstance(val, str):

                        val = val.lower()

                    key_params.append(val)

            if not key_conditions:

                print ('❌ ERROR - No unique columns found for table %s.%s' %(schema,table))

                return None

            where_clause = pgsql.SQL(" AND ").join(key_conditions)

            select_sql = pgsql.SQL("SELECT * FROM {}.{} WHERE {};").format(
                pgsql.Identifier(schema),
                pgsql.Identifier(table),
                where_clause
            )

            if self.verbose > 2:

                print ('            ', select_sql.as_string(self.cursor))

            self.cursor.execute(select_sql, key_params)

            record = self.cursor.fetchone()

            if record is not None:

                if self.overwrite or self.delete:

                    delete_sql = pgsql.SQL("DELETE FROM {}.{} WHERE {};").format(
                        pgsql.Identifier(schema),
                        pgsql.Identifier(table),
                        where_clause
                    )

                    if self.verbose > 2:

                        print ('            ', delete_sql.as_string(self.cursor))

                    self.cursor.execute(delete_sql, key_params)

                    self.conn.commit()

                    if self.delete:

                        continue
                else:

                    if self.verbose > 1:

                        print ('            Record already exists\n')

                    continue

            if len(columns) != len(values):

                print ('❌  ERROR number of columns and values do not match\n  columns: %s\n  values: %s\n' %(columns, values))

                return

            # Lowercase string values for text-type unique columns before insert
            insert_values = [
                v.lower() if (isinstance(v, str) and columns[i].lower().strip() in text_unique_cols) else v
                for i, v in enumerate(values)
            ]

            if self.verbose > 1:

                print ('.    values', insert_values)

            insert_sql = pgsql.SQL("INSERT INTO {}.{} ({}) VALUES ({});").format(
                pgsql.Identifier(schema),
                pgsql.Identifier(table),
                pgsql.SQL(", ").join(map(pgsql.Identifier, columns)),
                pgsql.SQL(", ").join([pgsql.Placeholder()] * len(insert_values))
            )

            if self.verbose > 2:

                print ('            ', insert_sql.as_string(self.cursor))

            self.cursor.execute(insert_sql, insert_values)

            self.conn.commit()

    def _Table_update(self, schema, table, where, columns, values):
        """
        @brief Updates records in a specified PostgreSQL schema table.

        This method updates existing records in a table within a given schema. It first checks if the table exists,
        then verifies that the record to be updated exists based on the provided WHERE clause. If the record exists,
        it updates the specified columns with the provided values.

        @param schema Name of the schema containing the table.
        @param table Name of the table to update.
        @param where Dict of {column_name: value} identifying the record(s) to update.
               IMPORTANT: must be a dict, NOT a raw SQL string.
               JSON example: {"id": 5} not "id = 5"
        @param columns List of column name strings to update.
        @param values List of Python-native values to assign to the specified columns.

        @details
        - Checks if the table exists in the specified schema.
        - Checks if the record(s) to update exist using the WHERE clause.
        - If the record does not exist, raises ValueError.
        - Executes the SQL UPDATE statement and commits the transaction.
        - Uses parameterized queries throughout to prevent SQL injection.

        @return None
        """

        # check if table exists
        self.cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
            (schema, table)
        )

        record = self.cursor.fetchone()

        if not record[0]:

            print ('WARNING: Can not update; schema %s table %s does not exist' %(schema, table))

            return

        # Build WHERE clause from dict
        where_conditions = pgsql.SQL(" AND ").join(
            pgsql.SQL("{} = %s").format(pgsql.Identifier(col))
            for col in where
        )

        where_params = list(where.values())

        select_sql = pgsql.SQL("SELECT * FROM {}.{} WHERE {};").format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            where_conditions
        )

        self.cursor.execute(select_sql, where_params)

        record = self.cursor.fetchone()

        if record is None:

            raise ValueError('Can not update non-existing record in %s.%s' %(schema, table))

        set_clause = pgsql.SQL(", ").join(
            pgsql.SQL("{} = %s").format(pgsql.Identifier(col))
            for col in columns
        )

        update_sql = pgsql.SQL("UPDATE {}.{} SET {} WHERE {};").format(
            pgsql.Identifier(schema),
            pgsql.Identifier(table),
            set_clause,
            where_conditions
        )

        if self.verbose > 2:

            print ('    ', update_sql.as_string(self.cursor))

        self.cursor.execute(update_sql, list(values) + where_params)

        self.conn.commit()

    def _Grant(self, command):
        """
        @brief Grants user rights using pre-validated GRANT/REVOKE statements.

        @param command List of SQL GRANT or REVOKE statements.

        @details
        - Each command must start with GRANT or REVOKE (case-insensitive).
        - Commands containing semicolons beyond the final character are rejected
          to prevent statement stacking.
        - Raises ValueError if any command fails validation.
        """

        for cmd in command:

            cmd_stripped = cmd.strip()

            if not cmd_stripped.upper().startswith(('GRANT ', 'REVOKE ')):

                raise ValueError('_Grant: only GRANT/REVOKE statements are permitted, got: %s' % cmd_stripped[:80])

            # Reject stacked statements (multiple semicolons)
            if cmd_stripped.rstrip(';').count(';') > 0:

                raise ValueError('_Grant: statement stacking not permitted in: %s' % cmd_stripped[:80])

            self.cursor.execute(cmd_stripped)

        self.conn.commit()

    def _Deactivate_pg_users(self, scheme_params_D):
        """
        @brief Terminates active PostgreSQL connections to the target database.

        @details Executes `pg_terminate_backend` for all sessions connected to the
        configured database except the current backend, allowing destructive
        operations such as database drops to proceed.

        @param scheme_params_D Dictionary containing scheme parameters for the
        active setup workflow.

        @return None
        """

        sql = "SELECT pg_terminate_backend(pg_stat_activity.pid)\
        FROM pg_stat_activity\
        WHERE datname = '%s'\
        AND pid <> pg_backend_pid();" % self.postgresDB_D['db']

        self.cursor.execute(sql)

    def _Drop_database(self, scheme_params_D):
        """
        @brief Drops the configured production database if it exists.

        @details Opens an initial-database session, then issues a `DROP DATABASE`
        statement for the configured production database after active connections
        have been terminated.

        @param scheme_params_D Dictionary containing scheme parameters, including
        the production database name.

        @return None
        """

        # Login to the database ownner
        session = Initiate_session(INITIAL_DATABASE_NAME, scheme_params_D)

        # DROP the entire database; this will fail if there are any active connections, which is why we terminate them first
        self.cursor.execute(
                pgsql.SQL("DROP DATABASE IF EXISTS {}").format(pgsql.Identifier(scheme_params_D['postgresdb']['db']))
            )
        
        session._Close()

    def _Delete_database_content(self):
        """
        @brief Deletes all user-defined schemas and their tables from the PostgreSQL database.

        This method retrieves all schemas in the database except system schemas (such as 'information_schema', 'pg_catalog', and 'pg_toast').
        For each schema found, it calls the internal method to delete the schema, which in turn deletes all tables within that schema.
        This effectively removes all user data from the database, but does not drop the database itself.

        @details
        - Prints a message indicating the start of the database deletion process.
        - Calls self._Select_all_schema() to get a list of all user-defined schemas.
        - Iterates over each schema and calls self._Delete_schema(schema) to remove the schema and its tables.

        @return None
        """
        print ("deleting database content")

        schema_L = self._Select_all_schema()

        for schema in schema_L:

            self._Delete_schema(schema)

        # Delete all users except system users (usesysid > 100000) to reset permissions; this is done after deleting database objects to avoid permission issues
        pg_user_L = self._Select_all_pg_user()

        for pg_user in pg_user_L:

            if self.verbose > 1:

                print ('    deleting user', pg_user)

            try:
                self.cursor.execute(
                    pgsql.SQL("DROP USER {}").format(pgsql.Identifier(pg_user))
                )
            except Exception as e:

                print ('        could not delete user %s: %s' % (pg_user, e))

    def _Delete_schema(self, schema):
        """
        @brief Deletes a specified schema and all its tables from the PostgreSQL database.

        @param schema Name of the schema to delete.

        @return None
        """

        schemaL = self._Select_all_schema()

        if schema not in schemaL:

            print ('    Schema %s does not exist' %schema)

            return

        table_L = self._Select_all_schema_tables(schema)

        for table in table_L:

            self._Delete_table(schema, table)

        print ("    deleting schema", schema)

        self.cursor.execute(
            pgsql.SQL("DROP SCHEMA {} CASCADE").format(pgsql.Identifier(schema))
        )

        self.conn.commit()

    def _Delete_table(self, schema, table):
        """
        @brief Deletes a specified table from a schema in the PostgreSQL database.

        @param schema Name of the schema containing the table.
        @param table Name of the table to delete.

        @return None
        """

        table_L = self._Select_all_schema_tables(schema)

        if table not in table_L:

            print ('    Table %s does not exist' %table)

            return

        print ("        deleting table", schema, table)

        try:
            self.cursor.execute(
                pgsql.SQL("DELETE FROM {}.{}").format(
                    pgsql.Identifier(schema), pgsql.Identifier(table)
                )
            )

            self.conn.commit()

        except Exception as e:

            print ('            could not delete records from %s.%s: %s' % (schema, table, e))

            print ('            attempting to drop table using cascade')
        
        self.cursor.execute(
            pgsql.SQL("DROP TABLE {}.{} CASCADE").format(
                pgsql.Identifier(schema), pgsql.Identifier(table)
            )
        )

        self.conn.commit()

    def _Select_all_pg_user(self):
        """
        @brief Selects all PostgreSQL users from the database.

        @return List of PostgreSQL user names.
        """

        self.cursor.execute("SELECT usename FROM pg_user WHERE usesysid > 100000;;")

        records = self.cursor.fetchall()

        user_L = [rec[0] for rec in records]

        return user_L

    def _Select_all_schema(self):
        """
        @brief Selects all user-defined schemas from the PostgreSQL database.

        @return List of user-defined schema names.
        """

        sql = """SELECT schema_name FROM information_schema.schemata
                 WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast');"""

        self.cursor.execute(sql)

        records = self.cursor.fetchall()

        schema_L = []

        for rec in records:

            if rec[0][0:2] == 'pg' or rec[0] == 'information_schema':

                continue

            schema_L.append(rec[0])

        return schema_L

    def _Select_all_schema_tables(self,schema):
        """
        @brief Selects all tables in a given schema.

        @param schema Name of the schema to search.

        @return List of table names in the specified schema.
        """

        self.cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (schema,)
        )

        return [row[0] for row in self.cursor]

    def _Close(self):
        """
        @brief Closes the database connection and cursor.
        """

        self.cursor.close()

        self.conn.close()

def Initiate_session(db, scheme_params_D):
    """
    @brief Initiates a PostgreSQL session using user default parameters.

    @param scheme_params_D Dictionary containing user default parameters.

    @return PG_session object if connection is successful, None otherwise.
    """
    if 'host_netrc_id' in scheme_params_D['postgresdb']:
        env_query_D = User_netrc_credentials(scheme_params_D['postgresdb']['host_netrc_id'])

        if not env_query_D:

            print ('❌ ERROR - Could not retrieve user credentials from .netrc file, exiting')

            return
        
    elif 'user_name' in scheme_params_D['postgresdb'] and 'password' in scheme_params_D['postgresdb']:
        # user login credentials explicitly given in scheme_file
        env_query_D = User_login_pswd(scheme_params_D['postgresdb']['user_name'], scheme_params_D['postgresdb']['password'])

    else:

        print ('❌ ERROR - No valid user credentials provided for Postgres connection, exiting')

        return None

    # When creating a new production db in postgres, the db must be 'postgres'
    env_query_D['db'] = db

    for item in ['port','host']:

        if item not in scheme_params_D['postgresdb']:

            print ('❌ ERROR - missing object <%s> must be a child to <postgresdb> in the <scheme_file>' %(item))

            return None

    env_query_D['port'] = scheme_params_D['postgresdb']['port']

    env_query_D['host'] = scheme_params_D['postgresdb']['host']

    try:

        # Connect to the Postgres Server
        session = PG_session(env_query_D)

        return session

    except Exception as e:

        print ('❌ ERROR - Could not connect to Postgres server')
        print ('    Please check if the Postgres server is running and that you have access rights')
        print ('    db:', env_query_D['db'])
        print ('    port:', env_query_D['port'])
        print ('    host:', env_query_D['host'])
        print ('    error:', e)

        return None
    
def Revoke_db_roles(session, postgresDB_D, role_key, user_id):
    
    db = postgresDB_D['db']

    revoke_sql = REVOKE_D.get(role_key)

    if revoke_sql:

        revoke_cmd = revoke_sql.format(db=db, user=user_id)

        for statement in [s.strip() for s in revoke_cmd.split(';') if s.strip()]:

            try:

                session.cursor.execute(statement)

            except Exception as e:

                print (f'    WARNING: REVOKE failed for {user_id} using role {role_key}: {e}')

                session.conn.rollback()

        session.conn.commit()

def Create_db_roles(session, postgresDB_D):
    """
    @brief Create or update PostgreSQL roles and grants from configuration.

    @details Iterates through configured database users, builds role SQL from
    `ROLES_D`, compares a stored grants hash on existing roles, and only re-applies
    grants when the SQL definition has changed.

    @param session Active PostgreSQL setup session with cursor and connection.
    @param postgresDB_D Dictionary containing database name and `db_users`
    definitions with `user_id`, `role`, and optional `password`.

    @return None
    """

    db = postgresDB_D['db']

    # Loop over the users defined in the database configuration and create corresponding roles
    for item in postgresDB_D['db_users']:

        user_id = item['user_id']

        role = item['role']

        password = item.get('password', '').replace("'", "''")

        sql = ROLES_D.get(role)

        if not sql:

            print (f'❌ ERROR - Role {role} not found in ROLES_D, skipping user {user_id}')

            continue

        cmd = sql.format(db=db, user=user_id, password=password)

        sql_hash = md5(cmd.encode()).hexdigest()[:12]

        # Check if this PostgreSQL role already exists
        session.cursor.execute(
            "SELECT oid FROM pg_roles WHERE rolname = %s;",
            (user_id,)
        )

        row = session.cursor.fetchone()

        if row:

            # Read the hash stored as a role comment from the previous run
            session.cursor.execute(
                "SELECT pg_catalog.obj_description(%s::oid, 'pg_authid');",
                (row[0],)
            )

            desc = session.cursor.fetchone()[0] or ''

            stored_hash = desc.split('grants_hash:')[1][:12] if 'grants_hash:' in desc else None

            if stored_hash == sql_hash:

                print (f'.   Role grants for {user_id} already up to date, skipping')

                continue

            # SQL differs — revoke existing grants before re-applying
            Revoke_db_roles(session, postgresDB_D, role, user_id)
            
        for statement in [s.strip() for s in cmd.split(';') if s.strip()]:

            # Skip CREATE USER if the role already exists (re-apply grants path)
            if row and statement.upper().startswith('CREATE USER'):

                continue

            session.cursor.execute(statement)

        session.conn.commit()

        # Store the hash as a comment on the role so future runs can detect changes
        session.cursor.execute(
            pgsql.SQL("COMMENT ON ROLE {} IS %s;").format(pgsql.Identifier(user_id)),
            (f'grants_hash:{sql_hash}',)
        )

        session.conn.commit()


    print(f'\n. Database roles defined: {[item["user_id"] for item in postgresDB_D["db_users"]]}')

def Setup_prod_DB(scheme_params_D):
    """
    @brief Sets up the production database.

    @param scheme_params_D Dictionary containing user default parameters.

    @return True if the production database is set up successfully, None otherwise.
    """

    session = Initiate_session(INITIAL_DATABASE_NAME, scheme_params_D)

    if not session:

        return None

    verbose = scheme_params_D['process'][0]['verbose']

    # Set the name of your production database( db)
    production_db_D = {'dbname':scheme_params_D['postgresdb']['db']}

    # Select the current (cluster) db
    session.cursor.execute("SELECT current_database()")

    # Get the results from the SELECT statement
    record = session.cursor.fetchone()

    if verbose:

        print ('.   Current database:', record[0])

    # Select the logged in user
    session.cursor.execute("SELECT user")

    # Get the results from the SELECT statement
    record = session.cursor.fetchone()

    if verbose:
        # Print Current user
        print ('.   User:', record[0])

    # Select all databases in the cluster db
    session.cursor.execute("SELECT datname FROM pg_database;")

    # Get the results from the SELECT statement
    records = session.cursor.fetchall()

    # Convert the retrieved records to a simple list
    db_L = [item[0] for item in records]

    if verbose:
        # Print the list of all databases in the cluster
        print ('.   Databases:', db_L)

        print ('.   Databases to create:', production_db_D['dbname'])

    # Check if your required production db is in the list
    if production_db_D['dbname'] not in db_L:

        # The requested production db does not exist

        msg = '.    Creating database: %s' %( production_db_D['dbname'])

        # Import the psycopg extension that allows you to create a new db
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # Invoke the connection with the extension
        session.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Create your production db using Identifier to prevent injection
        session.cursor.execute(
            pgsql.SQL("CREATE DATABASE {};").format(
                pgsql.Identifier(production_db_D['dbname'])
            )
        )

    else:

        msg = '.   Database %s already exists' %( production_db_D['dbname'])

    if verbose:

        print (msg)

    # Close the db connection
    session._Close()

    return True

def Setup_schemas_tables(scheme_params_D, pilot_L):
    """
    @brief Sets up schemas and tables in the PostgreSQL database.

    @param scheme_params_D Dictionary containing scheme parameters.
    @param pilot_L List of pilot JSON file paths.

    @return None
    """

    session = Initiate_session(scheme_params_D['postgresdb']['db'], scheme_params_D)

    if not session:

        return None

    verbose = scheme_params_D['process'][0]['verbose']

    # Loop over all json files and create Schemas and Tables
    for json_file_obj in pilot_L:

        if verbose:

            print ('\n. Reading json file:',json_file_obj)

        if not path.exists(json_file_obj):

            print ('❌ ERROR - json file does not exist, skipping')

            continue

        user_parameter = Read_json(json_file_obj)

        if not user_parameter:

            print ('❌ ERROR - no parameters retrieved from json file, skipping\n  ',json_file_obj)

            continue

        session._execute_db_command(scheme_params_D, user_parameter)

    session._Close()

    # To delete the complete database, open a session with the database source and delete

    print (user_parameter)

    if user_parameter['process'][0]['process_id'] == 'delete_database' and user_parameter['process'][0]['delete']:

        session = Initiate_session(INITIAL_DATABASE_NAME, scheme_params_D)

        if not session:

            return None

        #session._Deactivate_pg_users(scheme_params_D)

        session._Drop_database(scheme_params_D)

        session._Close()

def Setup_prod_roles(scheme_params_D):
    """
    @brief Sets up the production database roles(users).

    @param scheme_params_D Dictionary containing user default parameters.

    @return True if the production database roles are set up successfully, None otherwise.
    """

    # Create the default roles and permissions for the production db
    if not 'db_users' in scheme_params_D['postgresdb']:

        return None

    session = Initiate_session(scheme_params_D['postgresdb']['db'], scheme_params_D)

    if not session:

        return None

    # To setup and grant roles, login to the newly created production db and run the role creation and grant commands defined in the config file
    Create_db_roles(session, scheme_params_D['postgresdb'])

    session._Close()

    return True