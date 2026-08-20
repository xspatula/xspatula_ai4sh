'''
Created on 28 Jan 2026

@author: thomas gumbricht
'''

# Standard library imports

import os

from copy import deepcopy

from src.utils.pretty_print import Pprint_parameter as Pprint


INVERSE_PREPCODE_D = {'soil-undisturbed-in-situ':'no-prep',
              'mixed-untreated-soil-in-lab':'mix-wet',
              'dried-sieved-soil-in-lab':'dried sieved<2mm',
              'dried sieved<2mm':'ds2mm',
              'robert-minarik-cu':'rm-cu',
              'xspectre-d10':'d10',
              'xspectre-d20':'d20',
              'soaked':'post-infiltration',
              'dried-aggregate-select+soaked':'dry-pick-soak',
              'eo-data':'eo-data',
              'none':'none'}


class Structure_data():
    '''class for exporting OSSL data to AI4SH/XSPECTRE JSON format'''

    def __init__(self, record_D, geo_extent_D):
        
        '''
        '''

        self.record_D = record_D
 
        self.analysis_instrument = record_D['analysis_instrument_model__name']

        self.analysis_instrument_method_D = {self.analysis_instrument: []}

        # ===== Set the DB output objects =====

        # Set the dataset parameters
        self._Set_dataset()

        print(' Dataset:')

        Pprint(self.dataset)

        self._Set_dataset_location(geo_extent_D)

        print(' Dataset location:')
        Pprint(self.dataset_location)

        self._Set_campaign()

        print(' Campaign:')
        Pprint(self.campaign)

        self._Set_campaign_location(geo_extent_D)

        print(' Campaign location:')
        Pprint(self.campaign_location)

        self._Set_campaign_analysis_instrument()

        print(' Campaign analysis instrument:')
        Pprint(self.campaign_analysis_instrument)

        self._Set_campaign_meta()

        print(' Campaign meta:')
        Pprint(self.campaign_meta)

        # Set the sampling log id
        self._Set_sampling_log_id()

        print(' Sampling log id: %s' % self.record_D['sampling_log_id'])

        # Set the sampling log parameters
        self._Set_sampling_log()
        print(' Sampling log:')
        Pprint(self.sampling_log)

        # Set sample id
        self._Set_sample_id()

        print(' Sample id: %s' % self.record_D['sample_id'])

        # Set the sample parameters
        self._Set_sample()
        print(' Sample:')
        Pprint(self.sample)

        # Set the locus
        self._Set_locus()
        print(' Locus:')
        Pprint(self.locus)

        self._Set_observation_log()
        print(' Observation log:')
        Pprint(self.observation_log)
        
        self._Set_observation()
        print(' Observation:')
        Pprint(self.observation)

        # Attach the observation measurements to the equipment_method_D dictionary 
        self._Set_measurement()
        
        print(' Observation measurements:')
        Pprint(self.observation)

        # Set the db object point
        #self._Set_point()

        # Set the db object site
        #self._Set_site()

        # Set the db object data_soruce (= pilot for AI4SH)
        #self._Set_data_source()

        

        # Reset n_repeats to 3 (standard used for all neospectra measurements in AI4SH)
        #self.record_D['n_repetitions'] = 1

        # Set the observation metadata
        #self._Set_observation_metadata()

        

        # Assemble the complete sample event to a final dictionary containing all parameters
        #sample_event_ai4sh = self._Assemble_sample_event_AI4SH_xspectre()
        
        #if sample_event_ai4sh:

        #    # Dump the complete sample event to a JSON file
        #    self._Dump_sample_json(sample_event_ai4sh, 'ai4sh')

        #else:

        #    print(' ❌ Error creating AI4SH JSON post')
        
        # Assemble the complete sample event to a final dictionary containing all parameters
        #sample_event_xspectre = self._Assemble_sample_event_xspectre_xspectre()

        #if sample_event_xspectre:

        #    # Dump the complete sample event to a JSON file
        #    self._Dump_sample_json(sample_event_xspectre, 'xspectre')

        #else:

        #   print(' ❌ Error creating xspectre JSON post')


    def _Set_dataset(self):

        self.dataset = {"name": self.record_D['dataset_id__dataset_name'].lower(),
                    "source_id__source_name": self.record_D['dataset_source_id__source__name'],
                    "substance_array": ['soil'],
                    "keyword_array": ['agriculture','soil health','spectroscopy','soil properties'],
                    "abstract": "This campaign contains soil spectral and associated laboratory data collected as part of the EU-wide LUCAS soil sampling frame.",
                    "url": self.record_D['dataset_url'],
                    "doi": self.record_D['dataset_doi'],
                    "contact": self.record_D['dataset_contact__contact_name'],
                    "license": self.record_D['dataset_license_id__license_name'].lower(),
                    "license_url": self.record_D['dataset_license_url'],
                    "geographic": True,
                    "profile_type_code": "depth",
                    "in_lab_probing": True                            
                    }

    def _Set_dataset_location(self, geo_extent_D):

        self.dataset_location = geo_extent_D

    def _Set_campaign(self):

        self.campaign = {
                        "dataset_id__dataset_name": self.dataset['name'],
                        "supervisor": self.record_D['supervisor'],
                        "supervisor_name__email": self.record_D['supervisor_name__email'],
                        "name": '%s-%s_%s' % (self.record_D['campaign'].lower(),self.record_D['sample_year'],self.record_D['site_id'].lower()),
                        
                        "begin_date": self.record_D['begin_sample_date'],
                        "end_date": self.record_D['end_sample_date'],
                        "status_code": 10
                }
        
    def _Set_campaign_location(self, geo_extent_D):

        self.campaign_location = geo_extent_D

        add_D = {
                        "territory_code": self.record_D['territory'],
                        "site_code": self.record_D['site'],
                        "location_method_code": self.record_D['location_method_code'],
                        "location_error_m": self.record_D['location_error_m']
                }
        
        self.campaign_location = {**self.campaign_location, **add_D}

    def _Set_campaign_analysis_instrument(self):

        self.campaign_analysis_instrument = {
                        "campaign_id__campaign_name": self.campaign['name'],
                        "analysis_instrument_id__instrument_name": self.record_D['analysis_instrument_brand__name']
                }
        
    def _Set_campaign_meta(self):

        self.campaign_meta = {
                        "campaign_id__campaign_name": self.campaign['name'],
                        "substance_array": ['soil'],
                        "keyword_array": ['agriculture','soil health','spectroscopy','soil properties'],
                        "abstract": "This campaign contains soil spectral and associated laboratory data collected as part of the EU-wide LUCAS soil sampling frame.",

                }
        

        
    def _Set_sampling_log_id(self):
        """
        @brief Constructs a unique sampling log identifier and assigns it to the record dictionary.

        @details
        This method generates a unique sampling log identifier by concatenating the pilot country, pilot site, point ID, and sample date.
        The resulting identifier is stored in the record_D dictionary under the key 'sampling_log_id'.

        @return None. The sampling_log_id is stored in self.record_D for later use.
        """

        self.record_D['campaign_analysis_instrument_id'] = '%s-%s' %(self.record_D['campaign'].lower(), self.record_D['sample_year'])
    
        if self.record_D['campaign'].lower() == 'lucas':

            self.record_D['sampling_log_id'] =  '%s-%s-%s-%s' %(self.record_D['campaign'].lower(),
                    self.record_D['territory_id'].lower(),
                    self.record_D['site_id'].lower(),
                    self.record_D['sample_year'])
            
        else:
        
            self.record_D['sampling_log_id'] =  '%s-%s-%s-%s' %(self.record_D['campaign'].lower(),
                    self.record_D['territory_id'].lower(),
                    self.record_D['site_id'].lower(),
                    self.record_D['sample_date']) 

    def _Set_sampling_log(self):

        self.sampling_log = {"camapaign_id__campaign_name": self.campaign['name'],
            "name": self.record_D['sampling_log_id'].lower(),
            "supervisor_name__email": self.record_D['supervisor_name__email'],                          
            "sampling_equipment_id__equipment_name": self.record_D['sampling_equipment_name'],
            "begin_date": self.record_D['begin_sample_date'],
            "end_date": self.record_D['end_sample_date'],
            "contact__email": self.record_D['user_sampling__email']}
        
    def _Set_sample_id(self):
        """
        @brief Constructs a unique sample identifier and assigns it to the record dictionary.

        @details
        This method generates a unique sample identifier by concatenating the sampling log ID, point ID, minimum depth, and maximum depth.
        The resulting identifier is stored in the record_D dictionary under the key 'sample_id'.

        @return None. The sample_id is stored in self.record_D for later use.
        """
        
        self.record_D['sample_id'] =  self.record_D['sampling_log_id'].lower()+\
                    '_'+self.record_D['point_id'].lower()+\
                    '_'+self.record_D['min_depth']+\
                    '-'+self.record_D['max_depth']
        
        
    def _Set_sample(self):
        """
        @brief Creates sample metadata dictionary with name and depth range.

        @details
        This method constructs a sample metadata dictionary containing:
        - name: Lowercase sample identifier from record_D
        - min_depth: Minimum depth converted to integer (cm)
        - max_depth: Maximum depth converted to integer (cm)

        @return None. The data sample dictionary is stored in self.sample for later use.
        
        The sample dictionary is stored in self.sample and used later in hierarchical JSON assembly.
        This method assumes sample_id, min_depth, and max_depth are already set in self.record_D.
        """

        self.sample = {"sampling_log_id__sampling_log_name": self.record_D['sampling_log_id'].lower(),
            "name": self.record_D['sample_id'].lower()}
        
    def _Set_locus(self):

        self.locus =  {"sample_id__sample_name": self.record_D['sample_id'],
                "setting": self.record_D['setting'],
                "latitude_dd_wgs84": self.record_D['latitude_dd_wgs84'],
                "longitude_dd_wgs84": self.record_D['longitude_dd_wgs84'],
                "min_profile_cm": self.record_D["min_depth"], 
                "max_profile_cm": self.record_D['max_depth']}    
        

    def _Set_observation_log(self):
        """
        @brief Placeholder for setting observation log parameters.

        @details
        This method is intended to set observation log parameters but is currently a placeholder.
        Future implementation may include populating observation-related attributes.

        @return None.
        """

        if self.record_D['analysis_instrument_id'] == 'unknown':

            self.record_D['analysis_instrument_id'] =  '%s-%s' %(self.record_D['campaign'].lower(), self.record_D['begin_observation_date'])

        self.observation_log_name = '%s-%s' %(self.record_D['sampling_log_id'].lower(),self.record_D['analysis_instrument_id'] )

        self.observation_log = {"camapaign_id__campaign_name": self.campaign['name'],
            "sampling_log_id__sampling_log_name": self.record_D['sampling_log_id'].lower(),
            "supervisor_name__email": self.record_D['supervisor_analysis__email'],            
            "name": self.observation_log_name,             
            "instrument_model_id__instrument_name": self.record_D['analysis_instrument_model__name'],
            "instrument_id__instrument_name": self.record_D['analysis_instrument_id'],
            "begin_date": self.record_D['begin_observation_date'],
            "end_date": self.record_D['end_observation_date'],
            "doi": self.record_D['analysis_doi'],
            "license": self.record_D['analysis_license:'],
            "license_url": self.record_D['analysis_license_url']
        }

    def _Set_observation(self):
        """
        @brief Placeholder for setting observation parameters.

        @details
        This method is intended to set observation parameters but is currently a placeholder.
        Future implementation may include populating observation-related attributes.

        @return None.
        """

        self.observation = {"sample_id__sample_name": self.sample['name'],
            "observation_log_id__observation_log_name": self.observation_log_name,
            "sample_id__sample_name": self.sample['name'],
            "subsample": self.record_D['subsample'],
            "replicate": self.record_D['replicate'],
            "sample_prep_code": INVERSE_PREPCODE_D.get(self.record_D['sample_preparation__name'],'none'),
            "date_stamp": self.record_D['sample_analysis_date'],
        }

    def _Set_measurement(self):
        """
        @brief Extracts observation measurements from a data row and appends them to the equipment-method mapping.

        @details
        This function iterates over the columns in the data row, identifies measurement columns based on the method dictionary,
        and processes their values. It handles missing values, values with '<' (interpreted as half the threshold), and optionally
        includes standard deviation if provided. The processed observation is appended to the corresponding equipment-method list.

        @param data_row List of values representing a single row of measurement data.
        @param sd_column (Optional) Index of the column containing standard deviation values. If provided, standard deviation is included in the observation dictionary.

        @return None
        """

        for indicator_key in self.record_D['indicator__name']:
            '''
            '''
            observation_D =  {'value': self.record_D['value'][indicator_key]}
                              
            if 'standard_deviation' in self.record_D:
 
                observation_D['standard_deviation'] = self.record_D['standard_deviation'][indicator_key]

            #if 'wavelength' in self.record_D:
            #    observation_D['wavelength_nm'] = self.record_D['wavelength_nm']
                            
            meta_observation_D =  {'unit__name': self.record_D['unit__name'][indicator_key],
                            'indicator__name':  self.record_D['indicator__name'][indicator_key],
                            'procedure': self.record_D['procedure'],
                            'analysis_method_id__method_name': self.record_D['analysis_method__name'], 
                            'analysis_instrument_id': self.record_D['analysis_instrument_id']}
            
            observation_D = {**observation_D, **meta_observation_D}

            #self.analysis_instrument_method_D[self.analysis_instrument].append(observation_D)

            #self.xspectre_method_D = {self.analysis_instrument: []}

            #self.xspectre_method_D[self.analysis_instrument].append(observation_D)

    def _Set_observation_metadata(self):
        """
        @brief Creates observation metadata dictionary with sample preparation, analysis, and logistic information.

        @details
        This method constructs a comprehensive observation metadata dictionary containing:
        - Sample preparation details (preparation method name)
        - Analysis information (analyst email, subsample, replicate, number of repeats, analysis date)
        - Logistic details nested in a sub-dictionary (preservation, transport, storage methods and durations, logistic person email)
        
        The observation metadata dictionary is stored in self.observation_metadata and used later in hierarchical JSON assembly
        in the data_source → site → point → sampling_log → sample → observation → analysis_method hierarchy.
        
        All data is retrieved from self.record_D which must be populated before calling this method.

        @return bool True if observation metadata is successfully created and stored in self.observation_metadata.
        """

        self.observation_metadata = {
                        "sample_preparation__name": self.record_D['sample_preparation__name'],
                        "person__email": self.record_D['user_analysis__email'],
                        "subsample": self.record_D['subsample'],
                        "replicate": self.record_D['replicate'],
                        "n_repeats": self.record_D['n_repetitions'],
                        "date_stamp": self.record_D['sample_analysis_date'],
                        "logistic": {
                          "sample_preservation__name": self.record_D['sample_preservation__name'],
                          "sample_transport__name": self.record_D['sample_transport__name'],
                          "transport_duration_h": self.record_D['transport_duration_h'],
                          "sample_storage__name": self.record_D['sample_storage__name'],
                          "storage_duration_h":  self.record_D['storage_duration_h'],
                          "person__email": self.record_D['user_logistic__email']}
                        }
        
        return True
    

        
    def _Assemble_sample_event_AI4SH_xspectre(self):
        """
        @brief Assembles a hierarchical sample event dictionary for JSON export.

        @details
        This function constructs a nested dictionary representing a sample event, including data source, site, point, sampling log, sample, observation, and analysis method. 
        The structure is built from previously set metadata and measurement dictionaries within the class instance.

        @return Dictionary containing the complete sample event structure, or None if an error occurs during assembly.
        """

        try:

            self.analysis_instrument_method_D[self.analysis_instrument][0]['spectra_scan_tuning'].pop('value_standard_deviation')

            self.analysis_instrument_method_D[self.analysis_instrument][0]['spectra_scan_tuning'].pop('dark_value')

            self.analysis_instrument_method_D[self.analysis_instrument][0]['spectra_scan_tuning'].pop('dark_value_standard_deviation')

        except:

            pass

        self.equipment_method_copy_D = deepcopy(self.analysis_instrument_method_D)
    
        try:

            analysis_method = {self.record_D['procedure']: self.equipment_method_copy_D[self.analysis_instrument]}

            observation = [{**self.observation_metadata, "analysis_method": analysis_method}]

            sample = [{**self.sample, "observation": observation}]

            sampling_log = [{**self.sampling_log, "sample": sample}]

            point = [{**self.point, "sampling_log": sampling_log}]

            site = [{**self.site, "point": point}]

            campaign = [{**self.data_source, "site": site}]

            dataset = [{**self.dataset, "campaign": campaign}]

            sample_event = {"dataset": dataset}

            return sample_event

        except:

            return None
                
               
    def _Assemble_sample_event_xspectre_xspectre(self):
        """
        @brief Assembles a hierarchical sample event dictionary for JSON export.

         @details
        This function constructs a nested dictionary representing a sample event, including data source, site, point, sampling log, sample, observation, and analysis method. 
        The structure is built from previously set metadata and measurement dictionaries within the class instance.

        @return Dictionary containing the complete sample event structure, or None if an error occurs during assembly.
        """
        
        analysis = {} 

        # Re-organise the analysis method dictionary to have indicators as keys
        for item in self.analysis_instrument_method_D[self.analysis_instrument]: 
            
            indicator = item['indicator__name']

            analysis[indicator] = item  

        if hasattr(self, 'xspectre_spectra_meta_D') and self.process.parameters.extended_metadata:

            self.observation_metadata['metadata'] = self.xspectre_spectra_meta_D

            self.observation_metadata['scan_dn'] = self.original_scan_dn_D

            self.observation_metadata['white_reference'] = self.white_reference_D

        observation = {**self.observation_metadata, "analysis": analysis}

        sample = {"sample": self.sample['name']}

        sampling_log = {"sampling_log": self.sampling_log}

        locus = {"locus": locus}

        sample_event = {"dataset":self.dataset, "campaign":self.data_source['name'], **sampling_log, **sample, **locus, "observation": observation}

        return sample_event 