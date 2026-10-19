'''
Created on 19 October 2026

@author: thomasgumbricht

History of the Parquet dataframes saved in a directory, kept in the hidden file .previous_dataframe.

Every process that saves a Parquet file (select_*, compile_dataset, the machine learning preprocessing
steps) appends a record to .previous_dataframe in the directory it saved to:

    {"history": [
        {"process": "select_indicator_only", "created_at": "2026-10-19T08:12:03",
         "note": "indicator: 199 samples", "entries": [{"dataframe": "clay_0-20.parquet", ...}]},
        ...
    ]}

The last record is the previous dataframe: a process whose parquet_file/dataframe parameter is empty reads
it, so a notebook can run select → inspect → plot → preprocess without repeating file names. The earlier
records are the history of all dataframes created in the directory, oldest first.

Entries may carry the keys indicator, regressor and selector (None when the dataframe is shared by all),
used by the array processes of machine_learning_preprocess to find one dataframe per combination.

Files written before the history format (a single record, no "history" key) are read as a one-record history.
'''

# Standard library imports
import os
import json

from datetime import datetime

PREVIOUS_DF_FN = '.previous_dataframe'


def Previous_df_path(directory):
    '''Return the path of .previous_dataframe in directory.'''

    return os.path.join(directory, PREVIOUS_DF_FN)


def Read_previous_history(directory):
    '''Return the list of records in .previous_dataframe of directory (oldest first); [] if none.'''

    fpn = Previous_df_path(directory)

    if not os.path.exists(fpn):
        return []

    try:

        with open(fpn) as fh:
            data = json.load(fh)

    except (OSError, json.JSONDecodeError):

        return []

    if isinstance(data, dict) and 'history' in data:
        return data['history']

    return [data] if isinstance(data, dict) else []


def Read_previous_df(directory):
    '''Return the latest record in .previous_dataframe of directory, or None.'''

    history = Read_previous_history(directory)

    return history[-1] if history else None


def Write_previous_df(directory, process_name, entries, note=''):
    '''Append a record {process, created_at, note, entries} to .previous_dataframe of directory.

    entries: list of dicts with at least 'dataframe' (file name relative to directory).
    '''
    history = Read_previous_history(directory)

    history.append({'process': process_name, 'created_at': datetime.now().isoformat(timespec='seconds'),
                    'note': note, 'entries': entries})

    with open(Previous_df_path(directory), 'w') as fh:
        json.dump({'history': history}, fh, indent=2)


def Register_parquet(parquet_fpn, process_name, note=''):
    '''Register a saved Parquet file as the previous dataframe of its directory.'''

    Write_previous_df(os.path.dirname(parquet_fpn), process_name,
                      [{'dataframe': os.path.basename(parquet_fpn), 'indicator': None, 'regressor': None,
                        'selector': None}], note)


def Previous_parquet(directory, verbose=True):
    '''Return the path of the latest single dataframe registered in directory, or None (error printed).'''

    prev = Read_previous_df(directory)

    if prev is None:

        print('    ERROR: no %s found in %s' % (PREVIOUS_DF_FN, directory))

        return None

    entries = prev.get('entries', [])

    if not entries:

        print('    ERROR: %s has no entries.' % PREVIOUS_DF_FN)

        return None

    parquet_name = entries[0].get('dataframe', '')

    parquet_fpn = os.path.join(directory, parquet_name)

    if not os.path.exists(parquet_fpn):

        print('    ERROR: previous dataframe not found: %s' % parquet_fpn)

        return None

    if verbose:
        print('    Previous: %s — %s (%s)' % (prev.get('process', '?'), parquet_name, prev.get('created_at', '?')))

    return parquet_fpn
