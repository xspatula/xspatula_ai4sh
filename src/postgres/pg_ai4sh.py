# Standard library imports
import re

# Application package imports



#from src.postgres.pg_session import PG_session


# Sample latitude/longitude: geolocations imported with only x/y coordinates in the geographic
# spatial reference have latitude/longitude left at the default 0 - fall back to y/x for those
GEOLOCATION_LAT_LON_SQL = (
    "CASE WHEN COALESCE(OGE.latitude, 0) = 0 AND COALESCE(OGE.longitude, 0) = 0 "
    "AND OGE.spatial_reference_id = (SELECT id FROM observation_utility.spatial_reference WHERE name = 'geographic') "
    "THEN OGE.y_coordinate ELSE OGE.latitude END, "
    "CASE WHEN COALESCE(OGE.latitude, 0) = 0 AND COALESCE(OGE.longitude, 0) = 0 "
    "AND OGE.spatial_reference_id = (SELECT id FROM observation_utility.spatial_reference WHERE name = 'geographic') "
    "THEN OGE.x_coordinate ELSE OGE.longitude END")


def pgsql_quote_col(identifier):
    '''Quote an SQL identifier (schema, table or column name).'''

    return '"%s"' % str(identifier).replace('"', '""')


class PG_manage_AI4SH:
    '''
    DB support for setting up process
    '''

    def __init__(self, process_S, pg_session_C = None):
        """ The constructor connects to the database"""

        #Connect to the Postgres Server
        #PG_session.__init__(self, process_S.postgresdb.db, process_S.process.verbose, 'manage_xspatula')

        # Cache of (src_unit_name, dst_unit_name) -> unit_translate row, resolved lazily
        self._unit_translation_cache = {}

        # Cache of provision_name -> {indicator_name_or_alias: (indicator_id, analysis_method_id)},
        # for quantity 'class' (nominal classification) indicators, resolved lazily
        self._classification_indicator_cache = {}

        # Cache of (indicator_id, analysis_method_id) -> {class_name: nominal_classification_id}, resolved lazily
        self._nominal_classification_cache = {}

    #===== Custom project specific postgres functions =====
    def _Custom_function_SKIPPA(self, query_D):
        '''
        '''

        def _Get_id(key):
            ''' See _Convert_id_code
            '''
            pass

        # Step 1: copy the query dictionary
        updated_query_D = query_D.copy()

        for key in query_D:
                
            if key.endswith('_code'):

                rec = _Get_code(key)

                updated_query_D[key] = rec[1]
        
        return updated_query_D
    
    def _Retrieve_dataset_alias(self,query_D, pg_session_C):

        sql = "SELECT alias FROM observation.dataset WHERE name = '%(dataset_id__dataset_name)s';" %query_D

        rec = pg_session_C._Execute_search_single_sql(sql)

        if not rec:

            sql = "SELECT alias FROM observation.dataset WHERE alias = '%(dataset_id__dataset_name)s';" %query_D

            rec = pg_session_C._Execute_search_single_sql(sql)

        if not rec:

            return None
        
        else:

            return ('dataset_id__dataset_name','dataset_id__dataset_name', rec[0])
    
    def _Retrieve_sample_id_from_observation_log(self, query_D, pg_session_C):
        '''
        '''

        # Resolve the observation log as any foreign key: by name, tag, or a composed name given with
        # the alias or tag of its parts (e.g. sampling_log_name@provision_alias, see _Resolve_composed_name)
        log_rec = pg_session_C._Check_get_foreign_key('observation_log_id__observation_log_name', query_D['observation_log_id__observation_log_name'])

        rec = None

        if log_rec:

            sql = "SELECT OS.id from observation.sample as OS\
                    INNER JOIN observation.observation_log AS OOL\
                    ON OOL.sampling_log_id = OS.sampling_log_id\
                    WHERE OS.name = %s AND OOL.id = %s;"

            rec = pg_session_C._Execute_search_single_sql(sql, (query_D['sample_id__sample_name'], log_rec[0]))

        if not rec:

            # Check if the sample name exists at all, if not, print a warning message and return None
            sql = "SELECT OS.name from observation.sample AS OS\
            WHERE OS.name = '%s';" %query_D['sample_id__sample_name']

            recs_sample_name = pg_session_C._Execute_search_all_sql(sql)

            if not recs_sample_name:

                msg = '⚠️ Sample name %s not found via observation_log %s' %(query_D['sample_id__sample_name'], query_D['observation_log_id__observation_log_name'])
                print(msg)

                return None
            
            # Check the sampling log associated with the sample name, if not, print a warning message and return None
            sql = "SELECT OSL.name from observation.sampling_log AS OSL\
            INNER JOIN observation.sample as OS\
            ON OS.sampling_log_id = OSL.id\
            WHERE OS.name = '%s';" %query_D['sample_id__sample_name']

            recs_sampling_log = pg_session_C._Execute_search_all_sql(sql)

            if not recs_sampling_log:

                msg = '⚠️ No sampling log found for sample name: %s' %(query_D['sample_id__sample_name'])
                print(msg)

            else: # sample and sampling log are found, but the observation log is wrong
                sql = "SELECT OSL.name, OOL.name from observation.observation_log AS OOL\
                INNER JOIN observation.sampling_log AS OSL\
                ON OSL.id = OOL.sampling_log_id\
                INNER JOIN observation.sample as OS\
                ON OS.sampling_log_id = OSL.id\
                WHERE OS.name = '%s';" %query_D['sample_id__sample_name']

                recs_logs_sample = pg_session_C._Execute_search_all_sql(sql)

                if recs_logs_sample:

                    msg = '⚠️ Sample %s exists for following observation log(s): %s' %(query_D['sample_id__sample_name'], [rec[1] for rec in recs_logs_sample])
                    msg += '\n   but the json command states observation log %s' %(query_D['observation_log_id__observation_log_name'])
                    print(msg)   

            return None

            # Check which observation log names are associated with the sample name, if not, print a warning message and return None
            sql = "SELECT OOL.name from observation.observation_log AS OOL\
            INNER JOIN observation.sample as OS\
            ON OS.observation_log_id = OOL.id\
            WHERE OS.name = '%s';" %query_D['sample_id__sample_name']

            recs_observation_log = pg_session_C._Execute_search_all_sql(sql)

            if not recs_observation_log:

                msg = '⚠️ No observation log found for sample name: %s ' %(query_D['sample_id__sample_name'])
                print(msg)
        
            elif len(recs_observation_log) == 1:

                msg = '⚠️ The sample name: %s is associated with the observation log: %s\n   but the json command states that it should be %s' %(query_D['sample_id__sample_name'], recs_observation_log[0][0], query_D['observation_log_id__observation_log_name'])

            else:

                msg = '⚠️ The sample name: %s is associated with multiple observation logs: %s\n   but the json command states that it should be %s' %(query_D['sample_id__sample_name'], [rec[0] for rec in recs_observation_log], query_D['observation_log_id__observation_log_name'])
                print(msg)

            return None
            
        #sql = "SELECT OOL.name from observation.observation_log AS OOL\
        #    INNER JOIN observation.sample as OS\
        #    ON OS.sampling_log_id = OOL.id\
        #    WHERE OS.name = '%s' AND OOL.name = '%s';" %(query_D['sample_id__sample_name'], query_D['observation_log_id__observation_log_name'])
        
        return ('sample_id__sample_name','sample_id', rec[0])
    
    def _Retrieve_quantity_indicator_id_from_provision(self, query_D, at_columns_D, pg_session_C):
        '''
        '''

        # SQL that retrieves the indicator id, name and alias for the indicators that are provisioned for the observation log of the observation with the given id (observation_id) in the query_D
        # The query also checks that quantity and unit are OK
        sql = "select OUI.id, OUI.name, OUI.alias, OUAM.name, OUU.name from observation.observation as OO \
            inner join observation_utility.provision_indicator as OUPI \
            on OUPI.provision_id = OO.provision_id \
            inner join observation_utility.indicator as OUI \
            on OUI.id = OUPI.indicator_id \
            inner join observation_utility.analysis_method  as OUAM \
            on OUAM.id = OUPI.analysis_method_id \
            inner join observation_utility.unit  as OUU \
            on OUU.id = OUPI.unit_id \
            WHERE OO.id = %(observation_id)d;" %query_D
        
        recs = pg_session_C._Execute_search_all_sql(sql)

        if not recs:

            raise Exception('No record found for query: %s' %sql)
        
            return None
        
        # Convert the recs tuples to dicts with rec[0] as key and rec[1] and rec[2] as value array
        indicator_D = {rec[1]: rec[0] for rec in recs}
        
        # extend indicator_D with aliases
        indicator_D.update({rec[2]: rec[0] for rec in recs})
        # Dict comprehension to create a new dict with only the keys that start with '@' and strip the '@' from the key
        at_columns_D = {key.strip('@'): value for key, value in at_columns_D.items() if key.startswith('@')}
        # Create a return dict with the indicator id as key and the value from at_columns_D as value, but only for the keys that are in indicator_D
        return_D = {}
        # Just index the return_D with an index number
        index_nr = 0
        # Loop over the at_columns_D and check if the key is in indicator_D, if so add it to the return_D with the indicator id as key and the value from at_columns_D as value
        for col in at_columns_D:

            if not col in indicator_D:

                raise Exception('No indicator found for name/alias: %s' %col)
            
            else:

                # check if the values is a string, if so, strip it
                if isinstance(at_columns_D[col], str):

                    at_columns_D[col] = at_columns_D[col].strip()

                    # check if the values starts with '<', if so strip the '<', conver to float set half the value
                    if at_columns_D[col][0] == '<':

                        at_columns_D[col] = float(at_columns_D[col].strip('<'))/2.0

                    try:

                        at_columns_D[col] = float(at_columns_D[col])

                    except ValueError:

                        msg = '⚠️ Value for indicator %s is not a number: %s' %(col, at_columns_D[col])
                        print(msg)
                        at_columns_D[col] = None

                        pass

                # check if the value is not None, if it is None, skip it
                if at_columns_D[col] is None:

                    continue

                return_D[index_nr] = {'indicator_id':indicator_D[col], 'value': at_columns_D[col]}
                
                index_nr += 1

        # Check if all expected indicators are found
        if len(return_D) != len(recs):

            msg = '⚠️ Not all expected measured indicators are found. Expected: %s, found: %s' %(len(recs), len(return_D))
        
        return return_D

    def _Retrieve_classification_indicators_for_provision(self, provision_name, pg_session_C):
        '''Which quantity='class' (nominal classification) indicators are registered,
        via provision_indicator, for the given provision. Returns
        {indicator_name: (indicator_id, analysis_method_id)}, cached per provision_name.
        Used to gate derived nominal classifications (see nominal_classification.py):
        a classifier's result is only stored if its indicator name shows up here.
        '''

        if provision_name in self._classification_indicator_cache:

            return self._classification_indicator_cache[provision_name]

        resolved = self._Retrieve_name_from_name_alias(
            {'schema_table': 'observation_utility.provision', 'name': provision_name},
            pg_session_C
        )

        if not resolved:

            self._classification_indicator_cache[provision_name] = {}

            return {}

        sql = "SELECT OUI.name, OUI.alias, OUI.id, OUPI.analysis_method_id \
            FROM observation_utility.provision AS OUP \
            INNER JOIN observation_utility.provision_indicator AS OUPI ON OUPI.provision_id = OUP.id \
            INNER JOIN observation_utility.indicator AS OUI ON OUI.id = OUPI.indicator_id \
            INNER JOIN observation_utility.quantity AS OUQ ON OUQ.id = OUI.quantity_id \
            WHERE OUP.name = '%s' AND OUQ.name = 'class';" % resolved

        recs = pg_session_C._Execute_search_all_sql(sql)

        indicator_D = {}

        for name, alias, indicator_id, analysis_method_id in (recs or []):

            indicator_D[name] = (indicator_id, analysis_method_id)
            indicator_D[alias] = (indicator_id, analysis_method_id)

        self._classification_indicator_cache[provision_name] = indicator_D

        return indicator_D

    def _Retrieve_nominal_classification_id(self, indicator_id, analysis_method_id, class_name, pg_session_C):
        '''observation_utility.nominal_classification.id for (indicator_id,
        analysis_method_id, name=class_name) - the value_id to store in
        observation.observation_classification. Caches the full class-name map per
        (indicator_id, analysis_method_id) pair on first use. Returns None if the
        class name isn't a registered nominal_classification row (i.e. the Python
        classifier and the database's registered class list have drifted apart).
        '''

        cache_key = (indicator_id, analysis_method_id)

        if cache_key not in self._nominal_classification_cache:

            sql = "SELECT name, id FROM observation_utility.nominal_classification \
                WHERE indicator_id = %d AND analysis_method_id = %d;" % cache_key

            recs = pg_session_C._Execute_search_all_sql(sql)

            self._nominal_classification_cache[cache_key] = {name: id_ for name, id_ in (recs or [])}

        return self._nominal_classification_cache[cache_key].get(class_name)

    def _Retrieve_wavelength_cardinality_from_provision(self, provision_name, pg_session_C):

        resolved = self._Retrieve_name_from_name_alias(
            {'schema_table': 'observation_utility.provision', 'name': provision_name},
            pg_session_C
        )
        if not resolved:
            print('⚠️ Provision not found: %s' % provision_name)
            return None

        sql = "SELECT cardinality(wavelength_array) FROM observation_utility.spectrometer AS OUS \
                INNER JOIN observation_utility.provision AS OUP ON OUS.provision_id = OUP.id \
                WHERE OUP.name = '%s';" % resolved

        rec = pg_session_C._Execute_search_single_sql(sql)

        if not rec:
            raise Exception('No record found for query: %s' % sql)

        return rec[0]
    
    def _Retrieve_name_from_name_alias(self, query_D, pg_session_C):

        sql = "SELECT name FROM %(schema_table)s WHERE name = '%(name)s';" %query_D

        rec = pg_session_C._Execute_search_single_sql(sql)

        if not rec:

            sql = "SELECT name FROM %(schema_table)s WHERE alias = '%(name)s';" %query_D

            rec = pg_session_C._Execute_search_single_sql(sql)

        if not rec:

            return None
        
        else:

            return rec[0]
    
    def _Retrieve_unit_translation(self, query_D, pg_session_C):
        '''query_D: {src_unit_name, dst_unit_name}. Returns (factor, addon, exponent, process) or None.'''

        sql = "SELECT UT.factor, UT.addon, UT.exponent, UT.process \
            FROM observation_utility.unit_translate AS UT \
            INNER JOIN observation_utility.unit AS SRC ON UT.src_unit_id = SRC.id \
            INNER JOIN observation_utility.unit AS DST ON UT.dst_unit_id = DST.id \
            WHERE SRC.name = '%(src_unit_name)s' AND DST.name = '%(dst_unit_name)s';" % query_D

        return pg_session_C._Execute_search_single_sql(sql)

    def _Translate_unit_value(self, value, src_unit_name, dst_unit_name, pg_session_C):
        '''Translate value(s) recorded in src_unit_name into dst_unit_name, using
        observation_utility.unit_translate. value may be a scalar or a pandas Series.

        Formula: dst_value = (src_value * factor + addon) ** exponent.

        Raises Exception with actionable fix instructions if no translation row exists for the
        (src_unit_name, dst_unit_name) pair, or NotImplementedError if the row specifies a
        non-formulaic "process" (not yet supported).
        '''

        if not src_unit_name or src_unit_name == dst_unit_name:

            return value

        cache_key = (src_unit_name, dst_unit_name)

        if cache_key not in self._unit_translation_cache:

            self._unit_translation_cache[cache_key] = self._Retrieve_unit_translation(
                {'src_unit_name': src_unit_name, 'dst_unit_name': dst_unit_name}, pg_session_C)

        trans = self._unit_translation_cache[cache_key]

        if trans is None:

            raise Exception(
                'Missing unit translation: "%s" -> "%s".\n'
                '    Add a row to ai4sh/import_data/utility/observation/excel/unit_translate.xlsx:\n'
                '      src_unit_id__unit_name = %s\n'
                '      dst_unit_id__unit_name = %s\n'
                '    You must determine factor, addon, exponent, process yourself.\n'
                '    Then re-run ai4sh/import_data/insert_ai4sh_utility_data.ipynb to load it into the database.'
                % (src_unit_name, dst_unit_name, src_unit_name, dst_unit_name))

        factor, addon, exponent, process = trans

        if process and process != 'None':

            raise NotImplementedError(
                'Unit translation "%s" -> "%s" uses process "%s", which is not yet implemented.'
                % (src_unit_name, dst_unit_name, process))

        return (value * factor + addon) ** exponent

    def _Retrieve_dataset_data_points(self, query_D, pg_session_C):
        '''
        '''

        if 'dataset_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation.dataset', 'name': query_D['dataset_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['dataset_name'] = resolved
            else:
                print('⚠️ Dataset not found: %s' % query_D['dataset_name'])
                return []

        if 'indicator_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation_utility.indicator', 'name': query_D['indicator_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['indicator_name'] = resolved
            else:
                print('⚠️ Indicator not found: %s' % query_D['indicator_name'])
                return []

        #sql = "SELECT OUI.name, OUI.alias, OOD.value FROM observation.dataset_data_point as OOD \
        #        inner join observation_utility.indicator as OUI \
        #        on OOD.indicator_id = OUI.id \
        #        inner join observation.dataset as OD \
        #        on OOD.dataset_id = OD.id \
        #        where OD.name = '%(dataset_name)s' and OUI.name = '%(indicator_name)s';" %query_D

        sql = "select OD.name, OC.name, OS.name, OUI.display_name, OOM.value, OUU.display_name from observation.observation_measurement as OOM \
        inner join observation.observation as OO on OOM.observation_id = OO.id \
        inner join observation.sample as OS on OO.sample_id = OS.id \
        inner join observation.observation_log as OOL on OO.observation_log_id = OOL.id \
        inner join observation.sampling_log as OSL on OOL.sampling_log_id = OSL.id \
        inner join observation.campaign as OC on OSL.campaign_id = OC.id \
        inner join observation.dataset as OD on OC.dataset_id = OD.id \
        inner join observation_utility.provision_indicator as OUPI on OOL.provision_id = OUPI.provision_id AND OUPI.indicator_id = OOM.indicator_id \
        inner join observation_utility.indicator as OUI on OUPI.indicator_id = OUI.id \
        inner join observation_utility.unit as OUU on OUPI.unit_id = OUU.id \
        where OD.name = '%(dataset_name)s' and OUI.name = '%(indicator_name)s';" %query_D

        recs = pg_session_C._Execute_search_all_sql(sql)

        if not recs:

            raise Exception('No record found for query: %s' %sql)
        
            return None
        
        return recs

    def _Retrieve_spectrometer_wavelengths(self, query_D, pg_session_C):
        '''Return the wavelength_array for the spectrometer identified by provision_name.'''

        if 'provision_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation_utility.provision', 'name': query_D['provision_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['provision_name'] = resolved
            else:
                print('⚠️ Provision not found: %s' % query_D['provision_name'])
                return None

        sql = "SELECT OUS.wavelength_array \
            FROM observation_utility.spectrometer AS OUS \
            INNER JOIN observation_utility.provision AS OUP ON OUS.provision_id = OUP.id \
            WHERE OUP.name = '%s';" % query_D['provision_name']

        rec = pg_session_C._Execute_search_single_sql(sql)

        return list(rec[0]) if rec else None

    def _Retrieve_spectra_for_dataset(self, query_D, pg_session_C):
        '''Return spectral arrays with sample metadata for a dataset.

        query_D keys:
          dataset_name     - name or alias
          provision_name   - provision name or alias (identifies the spectrometer)
          campaign_name    - optional campaign filter (empty string = no filter)
          preparation_name - optional preparation filter (empty string = no filter)

        Returns list of tuples:
        (sample_name, campaign_name, latitude, longitude, profile_min, profile_max, signal_array)

        Spectral values are stored in observation.observation_measurement_array.value (REAL[]).
        '''
        if 'dataset_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation.dataset', 'name': query_D['dataset_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['dataset_name'] = resolved
            else:
                print('⚠️ Dataset not found: %s' % query_D['dataset_name'])
                return []

        if 'provision_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation_utility.provision', 'name': query_D['provision_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['provision_name'] = resolved
            else:
                print('⚠️ Provision not found: %s' % query_D['provision_name'])
                return []

        campaign_name = query_D.get('campaign_name', '').strip()
        preparation_name = query_D.get('preparation_name', '').strip()

        def _Build_sql():

            where = "OD.name = '%s' AND OUP.name = '%s'" % (
                query_D['dataset_name'],
                query_D['provision_name'],
            )

            if campaign_name:
                where += " AND OC.name = '%s'" % campaign_name

            prep_joins = ""
            if preparation_name:
                prep_joins = (
                    "LEFT JOIN observation.observation_log_meta AS OOLM "
                    "ON OOLM.observation_log_id = OOL.id "
                    "LEFT JOIN observation_utility.preparation AS OUPRE "
                    "ON OUPRE.id = OOLM.preparation_id "
                )
                where += " AND (OUPRE.name = '%s' OR OUPRE.alias = '%s')" % (
                    preparation_name, preparation_name)

            return (
                "SELECT OS.name, OC.name, " + GEOLOCATION_LAT_LON_SQL + ", "
                "OSP.profile_min, OSP.profile_max, OOMA.value "
                "FROM observation.observation_measurement_array AS OOMA "
                "INNER JOIN observation.observation AS OO ON OOMA.observation_id = OO.id "
                "INNER JOIN observation.sample AS OS ON OO.sample_id = OS.id "
                "INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id "
                "INNER JOIN observation_utility.provision AS OUP ON OOL.provision_id = OUP.id "
                "INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id "
                "INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id "
                "INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id "
                "LEFT JOIN observation.sample_profile AS OSP ON OSP.sample_id = OS.id "
                "LEFT JOIN observation.sample_geotag AS OSGT ON OSGT.sample_id = OS.id "
                "LEFT JOIN observation.geolocation AS OGE ON OGE.id = OSGT.geolocation_id "
                "%sWHERE %s;" % (prep_joins, where)
            )

        sql = _Build_sql()

        recs = pg_session_C._Execute_search_all_sql(sql)

        return recs if recs else []

    def _Retrieve_available_provisions_for_dataset(self, query_D, pg_session_C):
        '''Return distinct provision names that have spectral data for a dataset/campaign.

        query_D must contain:
          dataset_name  - dataset name or alias
          campaign_name - optional; pass empty string to skip filter
        '''
        if 'dataset_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation.dataset', 'name': query_D['dataset_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['dataset_name'] = resolved
            else:
                print('⚠️ Dataset not found: %s' % query_D['dataset_name'])
                return []

        campaign_filter = ""
        if query_D.get('campaign_name'):
            campaign_filter = " AND OC.name = '%s'" % query_D['campaign_name']

        sql = (
            "SELECT DISTINCT OUP.name "
            "FROM observation.observation_measurement_array AS OOMA "
            "INNER JOIN observation.observation AS OO ON OOMA.observation_id = OO.id "
            "INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id "
            "INNER JOIN observation_utility.provision AS OUP ON OOL.provision_id = OUP.id "
            "INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id "
            "INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id "
            "INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id "
            "WHERE OD.name = '%s'%s ORDER BY OUP.name;" % (
                query_D['dataset_name'], campaign_filter)
        )

        recs = pg_session_C._Execute_search_all_sql(sql)

        return [r[0] for r in recs] if recs else []

    def _Retrieve_available_indicators_for_dataset(self, query_D, pg_session_C):
        '''Return distinct indicator names that have lab measurements for a dataset/campaign.

        query_D must contain:
          dataset_name  - dataset name or alias
          campaign_name - optional; pass empty string to skip filter

        Returns list of tuples: (name, alias, unit_name). unit_name is the unit the
        provision behind the measurement records that indicator in
        (observation_utility.provision_indicator.unit_id), or None if no
        provision/unit link exists for that observation_log/indicator.
        '''
        if 'dataset_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation.dataset', 'name': query_D['dataset_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['dataset_name'] = resolved
            else:
                print('⚠️ Dataset not found: %s' % query_D['dataset_name'])
                return []

        campaign_filter = ""
        if query_D.get('campaign_name'):
            campaign_filter = " AND OC.name = '%s'" % query_D['campaign_name']

        sql = (
            "SELECT DISTINCT OUI.name, OUI.alias, OUU.name "
            "FROM observation.observation_measurement AS OOM "
            "INNER JOIN observation.observation AS OO ON OOM.observation_id = OO.id "
            "INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id "
            "INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id "
            "INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id "
            "INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id "
            "INNER JOIN observation_utility.indicator AS OUI ON OOM.indicator_id = OUI.id "
            "LEFT JOIN observation_utility.provision_indicator AS OUPI "
            "ON OUPI.provision_id = OOL.provision_id AND OUPI.indicator_id = OOM.indicator_id "
            "LEFT JOIN observation_utility.unit AS OUU ON OUU.id = OUPI.unit_id "
            "WHERE OD.name = '%s'%s ORDER BY OUI.name;" % (
                query_D['dataset_name'], campaign_filter)
        )

        recs = pg_session_C._Execute_search_all_sql(sql)

        return [(r[0], r[1], r[2]) for r in recs] if recs else []

    def _Retrieve_lab_measurements_for_dataset(self, query_D, pg_session_C):
        '''Return lab measurements for a dataset and list of indicator names.

        query_D must contain:
          dataset_name  - dataset name or alias
          lab_indicators - list of indicator names

        Returns list of tuples: (sample_name, indicator_name, value, unit_name)
        unit_name is the unit the provision behind the measurement records that indicator in
        (observation_utility.provision_indicator.unit_id), or None if no provision/unit link
        exists for that observation_log/indicator (e.g. legacy data with no provision_id set).
        '''
        if 'dataset_name' in query_D:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation.dataset', 'name': query_D['dataset_name']},
                pg_session_C
            )
            if resolved:
                query_D = dict(query_D)
                query_D['dataset_name'] = resolved
            else:
                print('⚠️ Dataset not found: %s' % query_D['dataset_name'])
                return []

        indicator_list = query_D.get('lab_indicators', [])

        if not indicator_list:
            return []

        # Map resolved canonical name → original user-specified name/alias
        name_map = {}
        for ind in indicator_list:
            resolved = self._Retrieve_name_from_name_alias(
                {'schema_table': 'observation_utility.indicator', 'name': ind},
                pg_session_C
            )
            if resolved:
                name_map[resolved] = ind
            else:
                print('⚠️ Indicator not found: %s' % ind)

        if not name_map:
            return []

        indicators_sql = ', '.join("'%s'" % name for name in name_map)

        sql = "SELECT OS.name, OUI.name, OOM.value, OUU.name \
            FROM observation.observation_measurement AS OOM \
            INNER JOIN observation.observation AS OO ON OOM.observation_id = OO.id \
            INNER JOIN observation.sample AS OS ON OO.sample_id = OS.id \
            INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id \
            INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id \
            INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id \
            INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id \
            INNER JOIN observation_utility.indicator AS OUI ON OOM.indicator_id = OUI.id \
            LEFT JOIN observation_utility.provision_indicator AS OUPI \
                ON OUPI.provision_id = OOL.provision_id AND OUPI.indicator_id = OOM.indicator_id \
            LEFT JOIN observation_utility.unit AS OUU ON OUU.id = OUPI.unit_id \
            WHERE OD.name = '%(dataset_name)s' \
            AND OUI.name IN (%(indicators)s);" % {
                'dataset_name': query_D['dataset_name'],
                'indicators': indicators_sql
            }

        recs = pg_session_C._Execute_search_all_sql(sql)

        if not recs:
            return []

        # Return tuples with the original user-specified name/alias so callers
        # can match results back to the indicator labels from the process file
        return [(r[0], name_map.get(r[1], r[1]), r[2], r[3]) for r in recs]

    # ===== Scope based retrieval for the select_data processes (see src/ai4sh/select.py) =====
    #
    # A scope is a dict {'dataset_id': int, 'campaign_id': int|None, 'sampling_log_id': int|None,
    # 'sample_names': list} defining the samples to select. All queries return the sample id as
    # key, as sample names are only unique within a sampling_log (e.g. LUCAS point ids in 2009 and 2015).

    def _Retrieve_record_for_select(self, schema, table, value, parent_D, pg_session_C):
        '''Find records in schema.table whose name, tag or alias equals value.

        parent_D: optional {column: id} restricting the search (e.g. {'dataset_id': 1}).

        Returns list of (id, name, tag) tuples. Tables without a tag column (e.g.
        observation_utility.preparation) return the alias as tag.
        '''
        cols = [r[0] for r in pg_session_C._Execute_search_all_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;",
            (schema, table))]

        search_L = [c for c in ('name', 'tag', 'alias') if c in cols]

        tag_col = 'tag' if 'tag' in cols else ('alias' if 'alias' in cols else 'name')

        where = '(%s)' % ' OR '.join(pgsql_quote_col(c) + ' = %s' for c in search_L)

        params = [value] * len(search_L)

        for col, id_ in (parent_D or {}).items():

            where += ' AND %s = %%s' % pgsql_quote_col(col)

            params.append(id_)

        sql = 'SELECT id, name, %s FROM %s.%s WHERE %s ORDER BY id;' % (
            pgsql_quote_col(tag_col), pgsql_quote_col(schema), pgsql_quote_col(table), where)

        return pg_session_C._Execute_search_all_sql(sql, params) or []

    def _Scope_sql(self, scope_D):
        '''Return (sql, params) for a subquery selecting the sample ids in scope.'''

        sql = 'SELECT OS.id FROM observation.sample AS OS \
            INNER JOIN observation.sampling_log AS OSL ON OS.sampling_log_id = OSL.id \
            INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id \
            WHERE OC.dataset_id = %s'

        params = [scope_D['dataset_id']]

        if scope_D.get('campaign_id'):

            sql += ' AND OC.id = %s'

            params.append(scope_D['campaign_id'])

        if scope_D.get('sampling_log_id'):

            sql += ' AND OSL.id = %s'

            params.append(scope_D['sampling_log_id'])

        if scope_D.get('sample_names'):

            sql += ' AND OS.name = ANY(%s)'

            params.append(list(scope_D['sample_names']))

        return sql, params

    def _Retrieve_sample_keys_for_scope(self, scope_D, pg_session_C):
        '''Return the key columns of all samples in scope:
        (sample_id, sample_name, sampling_log_tag, latitude, longitude, profile_min, profile_max).'''

        scope_sql, params = self._Scope_sql(scope_D)

        sql = 'SELECT OS.id, OS.name, OSL.tag, ' + GEOLOCATION_LAT_LON_SQL + ', OSP.profile_min, OSP.profile_max \
            FROM observation.sample AS OS \
            INNER JOIN observation.sampling_log AS OSL ON OS.sampling_log_id = OSL.id \
            LEFT JOIN observation.sample_profile AS OSP ON OSP.sample_id = OS.id \
            LEFT JOIN observation.sample_geotag AS OSGT ON OSGT.sample_id = OS.id \
            LEFT JOIN observation.geolocation AS OGE ON OGE.id = OSGT.geolocation_id \
            WHERE OS.id IN (%s) ORDER BY OS.id;' % scope_sql

        return pg_session_C._Execute_search_all_sql(sql, params) or []

    def _Retrieve_spectra_for_scope(self, scope_D, provision_id, preparation_id, pg_session_C):
        '''Return (sample_id, signal_array) for all spectra of the provision for samples in scope,
        optionally restricted to a sample preparation.'''

        scope_sql, params = self._Scope_sql(scope_D)

        prep_sql = ''

        if preparation_id:

            prep_sql = 'INNER JOIN observation.observation_log_meta AS OOLM \
                ON OOLM.observation_log_id = OOL.id AND OOLM.preparation_id = %s ' % int(preparation_id)

        sql = 'SELECT OO.sample_id, OOMA.value \
            FROM observation.observation_measurement_array AS OOMA \
            INNER JOIN observation.observation AS OO ON OOMA.observation_id = OO.id \
            INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id \
            %s\
            WHERE OOL.provision_id = %%s AND OO.sample_id IN (%s) ORDER BY OO.sample_id;' % (prep_sql, scope_sql)

        return pg_session_C._Execute_search_all_sql(sql, [provision_id] + params) or []

    def _Retrieve_indicator_for_select(self, value, pg_session_C):
        '''Return (id, name, alias, quantity_name) of the indicator with name or alias value, or None.'''

        sql = 'SELECT OUI.id, OUI.name, OUI.alias, OUQ.name FROM observation_utility.indicator AS OUI \
            LEFT JOIN observation_utility.quantity AS OUQ ON OUQ.id = OUI.quantity_id \
            WHERE OUI.name = %s OR OUI.alias = %s;'

        return pg_session_C._Execute_search_single_sql(sql, (value, value))

    def _Retrieve_measurements_for_scope(self, scope_D, indicator_ids, pg_session_C):
        '''Return (sample_id, indicator_id, value, unit_name) for numeric indicators of samples in scope.

        unit_name is the unit the provision behind the measurement records the indicator in
        (observation_utility.provision_indicator.unit_id), or None if no provision/unit link exists.
        '''
        scope_sql, params = self._Scope_sql(scope_D)

        sql = 'SELECT OO.sample_id, OOM.indicator_id, OOM.value, OUU.name \
            FROM observation.observation_measurement AS OOM \
            INNER JOIN observation.observation AS OO ON OOM.observation_id = OO.id \
            INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id \
            LEFT JOIN observation_utility.provision_indicator AS OUPI \
                ON OUPI.provision_id = OOL.provision_id AND OUPI.indicator_id = OOM.indicator_id \
            LEFT JOIN observation_utility.unit AS OUU ON OUU.id = OUPI.unit_id \
            WHERE OOM.indicator_id = ANY(%%s) AND OO.sample_id IN (%s);' % scope_sql

        return pg_session_C._Execute_search_all_sql(sql, [list(indicator_ids)] + params) or []

    def _Retrieve_classifications_for_scope(self, scope_D, indicator_ids, pg_session_C):
        '''Return (sample_id, indicator_id, class_name) for nominal (class) indicators of samples in scope.'''

        scope_sql, params = self._Scope_sql(scope_D)

        sql = 'SELECT OO.sample_id, OOC.indicator_id, OUNC.name \
            FROM observation.observation_classification AS OOC \
            INNER JOIN observation.observation AS OO ON OOC.observation_id = OO.id \
            INNER JOIN observation_utility.nominal_classification AS OUNC ON OUNC.id = OOC.value_id \
            WHERE OOC.indicator_id = ANY(%%s) AND OO.sample_id IN (%s);' % scope_sql

        return pg_session_C._Execute_search_all_sql(sql, [list(indicator_ids)] + params) or []

    def _Retrieve_landscape_table(self, name, pg_session_C):
        '''For a landscape covariate name (e.g. landcover, landuse, biogeo16) return a dict
        {value_column, lookup_table, hierarchical, label_columns}, where landscape.<name>_observation.<value_column>
        references landscape_utility.<lookup_table>. hierarchical is True when the lookup table holds all
        ranks of a classification (parent_<lookup>_id + class_rank_id, as land cover and land use).
        Returns None if there is no such landscape table.'''

        if not re.fullmatch(r'[a-z0-9_]+', name or ''):

            return None

        cols = [r[0] for r in pg_session_C._Execute_search_all_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'landscape' AND table_name = %s;",
            ('%s_observation' % name,))]

        value_cols = [c for c in cols if c.endswith('_id') and c not in ('id', 'observation_log_id', 'geolocation_id')]

        if len(value_cols) != 1:

            return None

        lookup = value_cols[0][:-3]

        lookup_cols = {r[0] for r in pg_session_C._Execute_search_all_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'landscape_utility' AND table_name = %s;",
            (lookup,))}

        return {'value_column': value_cols[0], 'lookup_table': lookup,
                'hierarchical': {'parent_%s_id' % lookup, 'class_rank_id'} <= lookup_cols,
                'label_columns': [c for c in ('tag', 'display_name', 'alias') if c in lookup_cols]}

    def _Retrieve_class_rank_id(self, rank, pg_session_C):
        '''Return the id of a class rank (order, family, genus) by name or alias, or None.'''

        rec = pg_session_C._Execute_search_single_sql(
            'SELECT id FROM landscape_utility.class_rank WHERE name = %s OR alias = %s;', (rank, rank))

        return rec[0] if rec else None

    def _Retrieve_landscape_for_scope(self, scope_D, name, landscape_D, rank_id, pg_session_C):
        '''Return (sample_id, value, name, display_name, alias, rank) for a landscape covariate, joined to the
        samples in scope by position (sample geolocation). value is the class tag (name if the class has no tag).

        If a geolocation has several observations, the one from the sample's own sampling_log is preferred,
        then the most recent. For a hierarchical classification the recorded class is rolled up through its
        parents to rank_id (1 = order, 2 = family, 3 = genus); a class recorded at a coarser rank than rank_id
        is kept as recorded. rank_id None keeps the rank recorded (lowest).'''

        scope_sql, params = self._Scope_sql(scope_D)

        lookup = landscape_D['lookup_table']

        label_cols = landscape_D['label_columns']

        sql_D = {'table': pgsql_quote_col('%s_observation' % name), 'lookup': pgsql_quote_col(lookup),
                 'col': pgsql_quote_col(landscape_D['value_column']), 'scope': scope_sql,
                 'parent': pgsql_quote_col('parent_%s_id' % lookup),
                 'value': 'COALESCE(LU.tag, LU.name)' if 'tag' in label_cols else 'LU.name',
                 'display': 'LU.display_name' if 'display_name' in label_cols else 'NULL',
                 'alias': 'LU.alias' if 'alias' in label_cols else 'NULL'}

        observed = 'SELECT DISTINCT ON (OS.id) OS.id AS sample_id, LO.%(col)s AS class_id \
            FROM observation.sample AS OS \
            INNER JOIN observation.sample_geotag AS OSGT ON OSGT.sample_id = OS.id \
            INNER JOIN landscape.%(table)s AS LO ON LO.geolocation_id = OSGT.geolocation_id \
            INNER JOIN observation.observation_log AS OOL ON OOL.id = LO.observation_log_id \
            WHERE OS.id IN (%(scope)s) \
            ORDER BY OS.id, (OOL.sampling_log_id = OS.sampling_log_id) DESC, LO.observed_at DESC' % sql_D

        sql_D['observed'] = observed

        if not landscape_D['hierarchical']:

            sql = 'WITH obs AS (%(observed)s) \
                SELECT obs.sample_id, %(value)s, LU.name, %(display)s, %(alias)s, NULL \
                FROM obs INNER JOIN landscape_utility.%(lookup)s AS LU ON LU.id = obs.class_id;' % sql_D

            return pg_session_C._Execute_search_all_sql(sql, params) or []

        # Climb from the recorded class while it is finer than the requested rank; the coarsest node
        # reached per sample is the class at (or above, if not recorded that fine) the requested rank
        sql = 'WITH RECURSIVE obs AS (%(observed)s), \
            chain AS ( \
                SELECT obs.sample_id, obs.class_id AS node_id FROM obs \
                UNION ALL \
                SELECT C.sample_id, LU.%(parent)s FROM chain AS C \
                INNER JOIN landscape_utility.%(lookup)s AS LU ON LU.id = C.node_id \
                WHERE LU.%(parent)s IS NOT NULL AND LU.class_rank_id > %%s) \
            SELECT DISTINCT ON (C.sample_id) C.sample_id, %(value)s, LU.name, %(display)s, %(alias)s, LCR.name \
            FROM chain AS C \
            INNER JOIN landscape_utility.%(lookup)s AS LU ON LU.id = C.node_id \
            LEFT JOIN landscape_utility.class_rank AS LCR ON LCR.id = LU.class_rank_id \
            ORDER BY C.sample_id, LU.class_rank_id;' % sql_D

        return pg_session_C._Execute_search_all_sql(sql, params + [rank_id if rank_id else 99]) or []

    def _Retrieve_observation_id_from_observationOLD(self, query_D):

        sql = "SELECT OO.id FROM observation.observation as OO \
                INNER JOIN observation.sample  as OS\
                ON OO.sample_id = OS.id\
                INNER JOIN observation.observation_log as OOL \
                ON OO.observation_log_id = OOL.id\
                WHERE OS.name ='%(sample_id__sample_name)s' AND OOL.name = '%(observation_log_id__observation_log_name)s'\
                AND OO.subsample = '%(subsample)s' AND OO.replicate = %(replicate)s;" %query_D
        
        rec = self._Execute_search_single_sql(sql)

        if not rec:

            raise Exception('No record found for query: %s' %sql)
        
            return None
        
        return (rec[0])        
    
    
    def _Retrieve_observation_log_indicators_from_log_nameOLD(self, observation_log_name):
        ''' Retrieve observation log indicators based on the query dictionary
        '''
        
        sql = "SELECT OUI.alias, OUI.id FROM observation_utility.indicator AS OUI \
            INNER JOIN observation_utility.provision_indicator AS OUPI \
            ON OUPI.indicator_id = OUI.id \
            INNER JOIN observation.observation_log_provision AS OOLP \
            ON OOLP.provision_id = OUPI.provision_id \
            INNER JOIN observation.observation_log AS OOL \
            ON OOLP.observation_log_id = OOL.id \
            WHERE OOL.name = '%s';" %observation_log_name

        recs = self._Execute_search_all_sql(sql)
        
        return recs
    
    def _Retrieve_sample_id_from_sample_nameOLD(self, sample_name):
        ''' Retrieve sample id based on the query dictionary
        '''
        
        sql = "SELECT id FROM observation.sample WHERE name = '%s';" %sample_name

        rec = self._Execute_search_single_sql(sql)
        
        return ('sample_name','sample_id', rec[0])