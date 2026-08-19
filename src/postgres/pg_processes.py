"""
 @file pg_processes.py

 @brief PostgreSQL process definition management module.

 @details Manages root processes, sub-processes, parameters, and dependent metadata
 records stored in the PostgreSQL process schema.

 *Version History*:
 - Created: 2018-02-21
 - Updated: 2024-01-04
 - Updated: 2025-09-09 (Generalized process handling and added documentation)
 - Updated: 2026-03-11 (Parameterized query fixes)

 @author Thomas Gumbricht

 @date Created: 2018-02-21
 @date Updated: 2024-01-04
 @date Updated: 2025-09-09 (Generalized process handling and added documentation)
 @date Updated: 2026-03-11 (Parameterized query fixes)
"""

# Package application imports
from src.postgres import PG_session

class Pg_manage_process(PG_session):
    """
    @brief PostgreSQL manager for root and sub-process definitions.

    @details Coordinates CRUD operations for process metadata stored in the
    process schema, including dependent parameter and schemata records.
    """

    def __init__(self, process_S):
        """
        @brief Initialize the process manager and connect to PostgreSQL.

        @param process_S Structured process object containing user status and
        process configuration.

        @return None
        """

        #Connect to the Postgres Server
        environment_dot_file = 'user_cat_%s' %process_S.user_status.stratum_code
        PG_session.__init__(self, environment_dot_file, process_S.process.verbose, 'manage_process')

    def _Manage_root_process(self, queryD, overwrite, delete):
        """
        @brief Insert, update or delete root process in the database.

        @param queryD Dictionary containing root process fields:
            - root_process: Unique identifier for the root process
            - title: Title of the root process
            - label: Label for the root process
            - creator: Creator of the root process
        @type queryD: dict

        @param overwrite If True, updates an existing root process record.
        @type overwrite: bool

        @param delete If True, attempts to delete the root process.
        @type delete: bool

        @return Status message indicating the operation result.
        @rtype: str
        """

        record = self._Execute_search_single_sql(
            "SELECT * FROM process.root_process WHERE root_process = %s;",
            (queryD['root_process'],)
        )

        msg = '    ✅ Root process %s already exists' % (queryD['root_process'])

        if record is None and not delete:

            self._Execute_commit_sql(
                "INSERT INTO process.root_process (root_process, title, label, creator) "
                "VALUES (%s, %s, %s, %s);",
                (queryD['root_process'], queryD['title'], queryD['label'], queryD['creator'])
            )

            msg = '    ✅ Root process %s inserted' % (queryD['root_process'])

        elif overwrite:

            self._Execute_commit_sql(
                "UPDATE process.root_process SET (title, label) = (%s, %s) "
                "WHERE root_process = %s;",
                (queryD['title'], queryD['label'], queryD['root_process'])
            )

            msg = '    ✅ Root process %s updated' % (queryD['root_process'])

        elif delete:

            record = self._Execute_search_single_sql(
                "SELECT * FROM process.process WHERE root_process = %s;",
                (queryD['root_process'],)
            )

            if record is None:

                self._Execute_commit_sql(
                    "DELETE FROM process.root_process WHERE root_process = %s;",
                    (queryD['root_process'],)
                )

                msg = '    ✅ Root process %s deleted' % (queryD['root_process'])

            else:

                msg = '    ❌ ERROR Can not delete Root process %s as it contains sub processes' % (
                    queryD['root_process']
                )
                self.log(msg)

        return msg

    def _Manage_process(self, process_S, queryD, overwrite, delete):
        """
        @brief Insert, update or delete sub process in the database.

        @param process_S Process structure containing process configuration and parameters.
        @param queryD Dictionary containing sub process fields.
        @param overwrite If True, updates an existing sub process record.
        @param delete If True, deletes the sub process and all related data.

        @return Status message indicating the operation result.
        @rtype: str
        """

        validation_msg = self._Validate_root_process_exists(process_S)

        if validation_msg:
            return validation_msg

        if delete or overwrite:

            self._Delete_process_dependencies(queryD)

        msg = self._Manage_process_record(queryD, overwrite, delete)

        if delete:
            return msg

        param_msg = self._Process_parameters(process_S, queryD)

        if param_msg:
            return param_msg

        schema_msg = self._Process_schemata(process_S, queryD)

        if schema_msg:
            return schema_msg

        return msg

    def _Validate_root_process_exists(self, process_S):
        """
        @brief Validate that the parent root process exists.

        @param process_S Process structure containing root_process.

        @return Error message if root process doesn't exist, None otherwise.
        @rtype: str or None
        """
        root_queryD = {'root_process': process_S.process.parameters.root_process}
        rec = self._Single_search(root_queryD, ['root_process'], 'process', 'root_process')

        if not rec:
            error_msg = '    ❌ ERROR root_process <%s> missing, can not insert process <%s>)' % (
                process_S.process.parameters.root_process,
                process_S.process.parameters.process
            )
            self.log(error_msg)
            return error_msg

        return None

    def _Manage_process_record(self, queryD, overwrite, delete):
        """
        @brief Insert, update, or delete the sub-process record.

        @param queryD Dictionary containing sub process fields.
        @param overwrite If True, updates existing record.
        @param delete If True, deletes the record.

        @return Status message.
        @rtype: str
        """

        record = self._Execute_search_single_sql(
            "SELECT * FROM process.process WHERE process = %s;",
            (queryD['process'],)
        )

        msg = '    ✅ Process <%s> already exists' % (queryD['process'])

        if record:

            record = self._Execute_search_single_sql(
                "SELECT root_process FROM process.process WHERE process = %s;",
                (queryD['process'],)
            )

            if record[0] != queryD['root_process']:

                return ('    ❌ WARNING the process %s is already defined, '
                        'but under root process %s (not %s)') % (
                    queryD['process'], record[0], queryD['root_process']
                )

        if record is None and not delete:
            msg = self._Insert_process(queryD)

        elif record and overwrite:
            msg = self._Update_process(queryD)

        elif record and delete:
            msg = self._Delete_process(queryD)

        return msg

    def _Insert_process(self, queryD):
        """
        @brief Insert a new sub-process record.

        @param queryD Dictionary containing sub process fields.

        @return Success or error message.
        @rtype: str
        """

        self._Execute_commit_sql(
            "INSERT INTO process.process "
            "(root_process, process, min_user_stratum, title, label, creator) "
            "VALUES (%s, %s, %s, %s, %s, %s);",
            (
                queryD['root_process'], queryD['process'],
                queryD['min_user_stratum'], queryD['title'], queryD['label'],
                queryD['creator']
            )
        )

        return ' ✅ Sub process %s inserted' % (queryD['process'])

    def _Update_process(self, queryD):
        """
        @brief Update an existing sub-process record.

        @param queryD Dictionary containing sub process fields.

        @return Success message.
        @rtype: str
        """

        self._Execute_commit_sql(
            "UPDATE process.process SET (min_user_stratum, title, label) = (%s, %s, %s) "
            "WHERE process = %s;",
            (
                queryD['min_user_stratum'], queryD['title'], queryD['label'],
                queryD['process']
            )
        )

        return '    ✅ Process <%s> updated' % (queryD['process'])

    def _Delete_process(self, queryD):
        """
        @brief Delete a sub-process record.

        @param queryD Dictionary containing sub process identifiers.

        @return Success message.
        @rtype: str
        """

        params = (queryD['process'],)

        self._Execute_commit_sql(
            "DELETE FROM process.process WHERE process = %s;",
            params
        )

        return '    ✅ Process <%s> deleted' % (queryD['process'])

    def _Delete_process_dependencies(self, queryD):
        """
        @brief Delete all dependent records for a sub-process.

        @param queryD Dictionary containing process.
        """

        params = (queryD['process'],)

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_set_value WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_minmax WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_schema_table WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_permission WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_inherit WHERE process = %s;",
            params
        )

        self._Execute_commit_sql(
            "DELETE FROM process.process_parameter_auto_name WHERE process = %s;",
            params
        )

        self._Delete_schemata_if_exists(queryD)

    def _Delete_schemata_if_exists(self, queryD):
        """
        @brief Delete schemata records if the schemata table exists.

        @param queryD Dictionary containing process.
        """

        rec = self._Execute_search_single_sql(
            "SELECT EXISTS (SELECT FROM information_schema.tables "
            "WHERE table_schema = 'process' AND table_name = 'schemata');"
        )

        if rec and rec[0]:
            self._Execute_commit_sql(
                "DELETE FROM process.schemata WHERE process = %s;",
                (queryD['process'],)
            )

    def _Process_parameters(self, process_S, queryD):
        """
        @brief Process all parameters for the sub-process.

        @param process_S Process structure containing parameter nodes.
        @param queryD Dictionary containing process.

        @return Error message if processing fails, None otherwise.
        @rtype: str or None
        """

        for node in process_S.process.nodes:

            for param in node.parameter:

                error_msg = self._Process_single_parameter(node, param, queryD)

                if error_msg:
                    return error_msg

        return None

    def _Process_single_parameter(self, node, param, queryD):
        """
        @brief Process a single parameter definition.

        @param node Node containing the parameter.
        @param param Parameter object.
        @param queryD Dictionary containing process.

        @return Error message if processing fails, None otherwise.
        @rtype: str or None
        """

        set_value_L = getattr(param, 'set_value', [])
        minmax_D = self._Extract_minmax(param)
        schema_table_D = self._Extract_schema_table(param)
        column_permission_D = self._Extract_permission(param)
        column_inherit_D = self._Extract_inherit(param)
        column_auto_name_D = self._Extract_auto_name(param)

        if not hasattr(node, 'element'):
            error_msg = ('    ❌ ERROR element missing for parameter %s in process %s'
                         '\n(file: %s;  process nr %s)') % (
                param, queryD['process'], self.json_file_FN, self.p_str
            )
            self.log(error_msg)
            return error_msg

        qpD = self._Build_parameter_dict(node, param, queryD)

        error_msg = self._Insert_parameter(qpD)
        if error_msg:
            return error_msg

        self._Process_parameter_set_values(qpD, set_value_L)
        self._Process_parameter_minmax(qpD, minmax_D)
        self._Process_parameter_schema_table(qpD, schema_table_D)
        self._Process_parameter_permission(qpD, column_permission_D)
        self._Process_parameter_inherit(qpD, column_inherit_D)
        self._Process_parameter_auto_name(qpD, column_auto_name_D)

        return None

    def _Extract_minmax(self, param):
        """
        @brief Extract min/max constraints from parameter.
        """
        if hasattr(param, 'minmax'):
            return {'min': param.minmax.min, 'max': param.minmax.max}
        return {}

    def _Extract_schema_table(self, param):
        """
        @brief Extract schema/table reference from parameter.
        """
        if hasattr(param, 'schema_table'):
            return {
                'schema': param.schema_table.schema.lower(),
                'table': param.schema_table.table.lower(),
                'write': param.schema_table.write,
                'array_column': getattr(param.schema_table, 'array_column', None),
                'array_constant_column': getattr(param.schema_table, 'array_constant_column', None),
                'array_constant_value': getattr(param.schema_table, 'array_constant_value', None)
            }
        return {}

    def _Extract_permission(self, param):
        """
        @brief Extract permission reference from parameter.
        """
        if hasattr(param, 'permission'):
            return {
                'column_update': param.permission.update,
                'column_delete': param.permission.delete
            }
        return {}

    def _Extract_inherit(self, param):
        """
        @brief Extract get inherit value reference from parameter.
        """
        if hasattr(param, 'inherit'):
            return {
                'process_parameter': param.inherit.process_parameter.lower(),
                'src_schema': param.inherit.src_schema.lower(),
                'src_table': param.inherit.src_table.lower(),
                'src_column': param.inherit.src_column.lower(),
                'search_column': param.inherit.search_column.lower(),
                'search_object': param.inherit.search_object.lower(),
                'filter_column': param.inherit.filter_column.lower() if hasattr(param.inherit, 'filter_column') else None,
                'filter_value': param.inherit.filter_value.lower() if hasattr(param.inherit, 'filter_value') else None
            }
        return {}
    
    def _Extract_auto_name(self, param):
        """
        @brief Extract get auto name value reference from parameter.
        """
        if hasattr(param, 'auto_name'):
            return {
                'concat': param.auto_name.concat
            }
        return {}

    def _Build_parameter_dict(self, node, param, queryD):
        """
        @brief Build parameter dictionary for database insertion.
        """

        hint = param.hint if hasattr(param, 'hint') else "To be completed"
        default_val = param.default_value if hasattr(param, 'default_value') else ''

        if isinstance(default_val, str):
            default_val = default_val.lower()

        keys = ['process', 'parent', 'element', 'parameter',
                'parameter_type', 'required', 'default_value', 'hint']

        values = [
            queryD['process'],
            node.parent.lower(),
            node.element.lower(),
            param.parameter.lower(),
            param.parameter_type.lower(),
            param.required.lower() if isinstance(param.required, str) else param.required,
            default_val,
            hint
        ]

        return dict(zip(keys, values))

    def _Insert_parameter(self, qpD):
        """
        @brief Insert parameter definition into database.

        @param qpD Parameter dictionary.

        @return Error message if insert fails, None otherwise.
        @rtype: str or None
        """

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter "
            "WHERE process = %s AND parameter = %s "
            "AND parent = %s AND element = %s;",
            (qpD['process'], qpD['parameter'],
             qpD['parent'], qpD['element'])
        )

        if rec is None:

            try:
                self._Execute_commit_sql(
                    "INSERT INTO process.process_parameter "
                    "(process, parent, element, parameter, parameter_type, "
                    "required, default_value, hint) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        qpD['process'], qpD['parent'], qpD['element'],
                        qpD['parameter'], qpD['parameter_type'], qpD['required'],
                        qpD['default_value'], qpD['hint']
                    )
                )

            except Exception as e:
                error_msg = '    ❌ ERROR INSERT error for process.process_parameter: %s' % e
                self.log(error_msg)
                return error_msg

        return None

    def _Process_parameter_set_values(self, qpD, set_value_L):
        """
        @brief Process predefined value alternatives for a parameter.

        @param qpD Parameter dictionary.
        @param set_value_L List of set value objects.
        """

        for set_value in set_value_L:

            sv_value = set_value.value.lower() if isinstance(set_value.value, str) else set_value.value
            sv_label = set_value.label.lower() if isinstance(set_value.label, str) else set_value.label

            rec = self._Execute_search_single_sql(
                "SELECT * FROM process.process_parameter_set_value "
                "WHERE process = %s AND parameter = %s AND parent = %s "
                "AND element = %s AND value = %s;",
                (qpD['process'], qpD['parameter'], qpD['parent'],
                 qpD['element'], sv_value)
            )

            if rec is None:

                self._Execute_commit_sql(
                    "INSERT INTO process.process_parameter_set_value "
                    "(process, parameter, parent, element, value, label) "
                    "VALUES (%s, %s, %s, %s, %s, %s);",
                    (qpD['process'], qpD['parameter'],
                     qpD['parent'], qpD['element'], sv_value, sv_label)
                )

    def _Process_parameter_minmax(self, qpD, minmax_D):
        """
        @brief Process min/max constraints for a parameter.

        @param qpD Parameter dictionary.
        @param minmax_D Dictionary with min and max values.
        """

        if not minmax_D:
            return

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter_minmax "
            "WHERE process = %s AND parameter = %s AND parent = %s "
            "AND element = %s;",
            (qpD['process'], qpD['parameter'], qpD['parent'],
             qpD['element'])
        )

        if rec is None:

            self._Execute_commit_sql(
                "INSERT INTO process.process_parameter_minmax "
                "(process, parameter, parent, element, minval, maxval) "
                "VALUES (%s, %s, %s, %s, %s, %s);",
                (qpD['process'], qpD['parameter'],
                 qpD['parent'], qpD['element'], minmax_D['min'], minmax_D['max'])
            )

    def _Process_parameter_schema_table(self, qpD, schema_table_D):
        """
        @brief Process schema/table reference for a parameter.

        @param qpD Parameter dictionary.
        @param schema_table_D Dictionary with schema, table, and write flag.
        """

        if not schema_table_D:
            return

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter_schema_table "
            "WHERE process = %s AND parameter = %s;",
            (qpD['process'], qpD['parameter'])
        )

        if rec is None:

            self._Execute_commit_sql(
                "INSERT INTO process.process_parameter_schema_table "
                "(process, parameter, in_schema, in_table, write, array_column, array_constant_column, array_constant_value) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                (qpD['process'], qpD['parameter'],
                 schema_table_D['schema'], schema_table_D['table'], schema_table_D['write'],
                 schema_table_D.get('array_column'), schema_table_D.get('array_constant_column'),
                 schema_table_D.get('array_constant_value'))
            )

    def _Process_parameter_permission(self, qpD, column_permission_D):
        """
        @brief Process column permissions for a parameter.

        @param qpD Parameter dictionary.
        @param column_permission_D Dictionary with column update and delete permissions.
        """

        if not column_permission_D:
            return

        if '__' in qpD['parameter']:
            param_id = qpD['parameter'].split('__')[0]
        else:
            param_id = qpD['parameter']

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter_permission "
            "WHERE process = %s AND parameter = %s;",
            (qpD['process'], param_id)
        )

        if rec is None:

            self._Execute_commit_sql(
                "INSERT INTO process.process_parameter_permission "
                "(process, parameter, column_update, column_delete) "
                "VALUES (%s, %s, %s, %s);",
                (qpD['process'], param_id,
                 column_permission_D['column_update'], column_permission_D['column_delete'])
            )

    def _Process_parameter_inherit(self, qpD, column_inherit_D):
        """
        @brief Process get inherit value for a parameter.

        @param qpD Parameter dictionary.
        @param column_inherit_D Dictionary with get inherit value information.
        """

        if not column_inherit_D:
            return

        if '__' in qpD['parameter']:
            param_id = qpD['parameter'].split('__')[0]
        else:
            param_id = qpD['parameter']

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter_inherit "
            "WHERE process = %s AND parameter = %s;",
            (qpD['process'], param_id)
        )

        if rec is None:

            self._Execute_commit_sql(
                "INSERT INTO process.process_parameter_inherit "
                "(process, parameter, process_parameter, src_schema, src_table, src_column, "
                "search_column, search_object, filter_column, filter_value) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                (
                    qpD['process'], param_id,
                    column_inherit_D['process_parameter'], column_inherit_D['src_schema'], column_inherit_D['src_table'],
                    column_inherit_D['src_column'], column_inherit_D['search_column'],
                    column_inherit_D['search_object'], column_inherit_D['filter_column'],
                    column_inherit_D['filter_value']
                )
            )

    def _Process_parameter_auto_name(self, qpD, column_auto_name_D):
        """
        @brief Process get auto name value for a parameter.

        @param qpD Parameter dictionary.
        @param column_auto_name_D Dictionary with get auto name value information.
        """

        if not column_auto_name_D:
            return

        param_id = qpD['parameter']

        rec = self._Execute_search_single_sql(
            "SELECT * FROM process.process_parameter_auto_name "
            "WHERE process = %s AND parameter = %s;",
            (qpD['process'], param_id)
        )

        if rec is None:

            self._Execute_commit_sql(
                "INSERT INTO process.process_parameter_auto_name "
                "(process, parameter, concat) "
                "VALUES (%s, %s, %s);",
                (
                    qpD['process'], param_id,
                    column_auto_name_D['concat']
                )
            )

    def _Process_schemata(self, process_S, queryD):
        """
        @brief Process schemata definitions for the sub-process.

        @param process_S Process structure containing schema definitions.
        @param queryD Dictionary containing process.

        @return None
        """

        if not hasattr(process_S.process, 'schema'):
            return None

        for schemata in process_S.process.schema:
            self._Insert_schemata(schemata, queryD)

        return None

    def _Insert_schemata(self, schemata, queryD):
        """
        @brief Insert a single schemata definition.

        @param schemata Schemata object.
        @param queryD Dictionary containing process.
        """

        schemata_D = dict(list(schemata.__dict__.items()))
        schemata_D['process'] = queryD['process']

        schemata_rec = self._Single_search_tab_keys(
            schemata_D,
            ['process'],
            'process',
            'schemata'
        )

        if schemata_rec:
            return

        self._Execute_commit_sql(
            "INSERT INTO process.schemata "
            "(process, schemata, src_schemata, dst_schemata, "
            "src_division, dst_division, src_epsg, dst_epsg) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
            (
                schemata_D['process'], schemata_D['schemata'],
                schemata_D['src_schemata'], schemata_D['dst_schemata'],
                schemata_D['src_division'], schemata_D['dst_division'],
                schemata_D['src_epsg'], schemata_D['dst_epsg']
            )
        )