# Application package imports



#from src.postgres.pg_session import PG_session


class PG_manage_AI4SH:
    '''
    DB support for setting up process
    '''

    def __init__(self, process_S, pg_session_C = None):
        """ The constructor connects to the database"""

        #Connect to the Postgres Server
        #PG_session.__init__(self, process_S.postgresdb.db, process_S.process.verbose, 'manage_xspatula')

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

        sql = "SELECT OS.id from observation.sample as OS\
                INNER JOIN observation.sampling_log as OSL\
                ON OS.sampling_log_id = OSL.id\
                INNER JOIN observation.observation_log AS OOL\
                ON OOL.sampling_log_id = OSL.id\
                WHERE OS.name ='%(sample_id__sample_name)s' AND OOL.name  = '%(observation_log_id__observation_log_name)s';" %query_D

        rec = pg_session_C._Execute_search_single_sql(sql)

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
                "SELECT OS.name, OC.name, OGE.latitude, OGE.longitude, "
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
            "SELECT DISTINCT OUI.name, OUI.alias "
            "FROM observation.observation_measurement AS OOM "
            "INNER JOIN observation.observation AS OO ON OOM.observation_id = OO.id "
            "INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id "
            "INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id "
            "INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id "
            "INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id "
            "INNER JOIN observation_utility.indicator AS OUI ON OOM.indicator_id = OUI.id "
            "WHERE OD.name = '%s'%s ORDER BY OUI.name;" % (
                query_D['dataset_name'], campaign_filter)
        )

        recs = pg_session_C._Execute_search_all_sql(sql)

        return [(r[0], r[1]) for r in recs] if recs else []

    def _Retrieve_lab_measurements_for_dataset(self, query_D, pg_session_C):
        '''Return lab measurements for a dataset and list of indicator names.

        query_D must contain:
          dataset_name  - dataset name or alias
          lab_indicators - list of indicator names

        Returns list of tuples: (sample_name, indicator_name, value)
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

        sql = "SELECT OS.name, OUI.name, OOM.value \
            FROM observation.observation_measurement AS OOM \
            INNER JOIN observation.observation AS OO ON OOM.observation_id = OO.id \
            INNER JOIN observation.sample AS OS ON OO.sample_id = OS.id \
            INNER JOIN observation.observation_log AS OOL ON OO.observation_log_id = OOL.id \
            INNER JOIN observation.sampling_log AS OSL ON OOL.sampling_log_id = OSL.id \
            INNER JOIN observation.campaign AS OC ON OSL.campaign_id = OC.id \
            INNER JOIN observation.dataset AS OD ON OC.dataset_id = OD.id \
            INNER JOIN observation_utility.indicator AS OUI ON OOM.indicator_id = OUI.id \
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
        return [(r[0], name_map.get(r[1], r[1]), r[2]) for r in recs]

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