'''
Created on 29 May 2025

@author: thomasgumbricht
'''

# Standard library imports
from os import path

from src.ai4sh.import_data import Process_import_JSON

from src.ai4sh.plot import Process_plot

from src.ai4sh.select import Process_select

from src.ai4sh.machine_learning_preprocess import Process_ml_preprocess
from src.ai4sh.machine_learning_model import Process_regression_model

from src.lib.login import Get_set_database_session

from src.utils import Log, Today_as_str_YYYYMMDD

def Run_process(structured_process_D, scheme_params_D):
    '''
    '''
    
    result_print_L = []
    insert_result_L = []

    # Get user status and postgres session for process execution
    user_status_D, pg_session_C = Get_set_database_session(scheme_params_D)

    if not user_status_D:

        return None
    
    print ('\n########### STARTING PROCESSES ########### \n')
    
    for key in structured_process_D:

        json_file_name = path.split(key)[1]

        if scheme_params_D['process'][0]['verbose'] > 0:
        
            msg = '. Command file: %s\n.   (%s ready processes to run)' %(key, len(structured_process_D[key]))

            print (msg)

        for p_nr, process_S in structured_process_D[key].items():

            # Check the process stratum requirements against the user status and skip if not met
            rec = pg_session_C._Single_search(
                {'process': process_S.process.process},
                ['min_user_stratum'],
                'process',
                'process'
            )

            process_stratum = rec[0] if rec else 0
   
            if process_stratum > user_status_D['stratum_code']:

                print ('    ⚠️  Skipping process nr: %s %s (stratum requirement not met)' %(p_nr, 
                    process_S.process.process))

                continue
        
            root_process = process_S.process.root_process

            if process_S.process.verbose > 0:

                if process_S.process.overwrite: 

                    msg = '.   Running process nr: %s %s (overwriting)' %(p_nr, 
                        process_S.process.process)

                elif process_S.process.delete:

                    msg = '.   Running process nr: %s %s (deleting)' %(p_nr, 
                        process_S.process.process)
                    
                else:

                    msg = '.   Running process nr: %s %s' %(p_nr,
                        process_S.process.process)
                        
                print (msg)

            if root_process == 'translate_data':

                import_C = Process_import_JSON(process_S,pg_session_C,scheme_params_D['project_root_FP'],scheme_params_D)

                result = import_C._Sub_process(key)

                if result:

                    if process_S.process.process.startswith('insert'):

                        insert_result_L.append(result)

                    else:

                        result_print_L.append(result,)

            elif root_process == 'manage_table_data':

                import_C = Process_import_JSON(process_S,pg_session_C,scheme_params_D['project_root_FP'],scheme_params_D)

                result = import_C._Sub_process(key)

                if result:

                    insert_result_L.append(result)

            elif root_process == 'select_data':

                select_C = Process_select(process_S, pg_session_C)

                select_C._Sub_process(key)

            elif root_process == 'plot':
                # Iinitate the plot class
                plot_C = Process_plot(process_S,pg_session_C)

                plot_C._Sub_process(key)

            elif root_process == 'machine_learning':

                if process_S.process.process == 'regression_modeling':
                    model_C = Process_regression_model(process_S, pg_session_C)
                    model_C._Sub_process(key)
                else:
                    ml_C = Process_ml_preprocess(process_S, pg_session_C)
                    ml_C._Sub_process(key)

            else:
                
                error_msg = '    ❌ WARNING root_process <%s> not available\n \
                    (file: %s;  process nr %s)' %(root_process,
                                                json_file_name,
                                                p_nr)

                Log(error_msg)

                return

    if pg_session_C is not None:

        try:

            pg_session_C._Close()

        except Exception as e:

            Log('⚠️  Warning - could not close database session: %s' % e)

    print ('\n########### ALL PROCESSES FINISHED ########### \n') 

    if result_print_L:

        print ('\nData successfully translated to JSON formated process files. \n')
        print ('⚠️ You have 2 alternatives for inserting the data to the database:')
        print ('   1. Copy paste the results below into a txt file and use as pilot file.')
        print ('   2. Create a python code block and set each JSON file as a single process file.\n')
          
        print ('##############################################') 
        print ('### PROCESSES CREATED BY XSPATULA %s ###' % Today_as_str_YYYYMMDD())
        print ('##############################################')   
        for row in result_print_L:

            print (row)

    if insert_result_L:

        print ('\nData applied to the database.\n')

        failed_process_count = getattr(pg_session_C, 'failed_process_count', 0)

        if failed_process_count == 0:

            print ('✅ All processes completed successfully.\n')

        else:

            print ('❌ %s process(es) failed.\n' % failed_process_count)

        if structured_process_D[key][0].process.verbose:

            print ('   The following JSON process files were applied:')

            for row in dict.fromkeys(insert_result_L):

                print (row)