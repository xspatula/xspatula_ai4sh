"""
 @file json_read_write.py

 @brief Utility module for safe JSON file reading and writing.

 @details Wraps JSON input and output operations with existence checks, error
 handling, and optional verbose feedback for configuration files.

 *Version History*:
 - Created: 2024-01-04

 @author Thomas Gumbricht

 @date Created: 2024-01-04
"""

# Standard library imports
from os import path

import json

def Read_json(FPN,verbose=0):
    """
    @brief Reads a JSON file and returns its contents as a Python object.

    @details
    This function safely reads and parses JSON files with comprehensive error handling. It performs
    file existence checking before attempting to read, and catches JSON parsing errors to prevent
    application crashes. The function supports optional verbose output for debugging and logging purposes.
    
    @param FPN String containing the full path name to the JSON file. Can be relative or absolute path.
               Must point to a valid JSON file with .json extension (though extension not enforced).
    
    @param verbose Integer flag for verbose output. Default is 0 (silent mode).
                   Set to 1 to print status messages during execution.
                   Any non-zero value enables verbose mode.

    @return Python object (typically dict or list) containing the parsed JSON data if successful.
            Returns None if the file doesn't exist or if JSON parsing fails.
            The return type matches the top-level JSON structure:
            - dict for JSON objects {}
            - list for JSON arrays []
            - str, int, float, bool, None for JSON primitives (rare as top-level)
    
    @note The function uses Python's json.load() which assumes UTF-8 encoding by default. For files
          with different encodings, this function would need modification to specify encoding parameter.
    
    @warning The bare except clause catches all exceptions during JSON parsing, which may hide specific
             error details. Consider logging the actual exception for debugging: except Exception as e.
    
    @see json.load() documentation: https://docs.python.org/3/library/json.html#json.load
    """
        
    if verbose:
        
        print ('    Reading json file: %s' %(FPN)) 

    if not path.exists(FPN):
        
        msg = ' ❌ ERROR  - json file not found: %s' %(FPN)

        print (msg)
        
        return None

    with open(FPN) as f:

        # returns JSON object
        try: 
            
            json_D = json.load(f)
        
        except Exception as e:

            msg = 'Error reading json file: %s\n  %s' %(FPN, e)

            print(msg)

            return None
        
    return json_D
    
def Dump_json(FPN, data, indent=2, verbose=0):
    """
    @brief Writes a Python object to a JSON file with formatted output.

    @details
    This function serializes Python objects to JSON format and writes them to a file with proper
    formatting and error handling. It uses json.dump() to convert Python data structures to JSON,
    supporting configurable indentation for human-readable output. The function catches serialization
    errors and provides status feedback for successful or failed operations.
    
    @param FPN String containing the full path name of the JSON file to write.
               Parent directories must exist or FileNotFoundError will be raised.
               Existing files will be overwritten without warning.
    
    @param data Python object to serialize and write to JSON file. Must be JSON-serializable
                (dict, list, str, int, float, bool, None and nested combinations).
                NumPy arrays should be converted to lists first using .tolist().
    
    @param indent Integer specifying number of spaces for JSON indentation. Default is 2.
                  Use None for compact single-line output; 0 adds newlines but no indentation.
                  Positive integers create human-readable formatted output.
    
    @param verbose Integer flag for verbose output. Default is 0 (silent mode).
                   Set to 1 to print status messages showing the file being written.
                   Any non-zero value enables verbose mode.

    @return Boolean True if JSON file was written successfully.
            Returns None if serialization fails (e.g., unsupported data type, encoding error).
    
    @warning The bare except clause catches all exceptions during JSON writing, which may hide specific
             error details. Consider catching specific exceptions for better error diagnostics:
             except (TypeError, ValueError, OverflowError) as e.
    
    @see json.dump() documentation: https://docs.python.org/3/library/json.html#json.dump
    @see JSON specification: https://www.json.org/
    """
    
    if verbose:
        
        print ('    Writing json file:\n     %s' %(FPN)) 

    with open(FPN, 'w') as outfile:
        
        try:

            json.dump(data, outfile, indent=indent)

        except Exception as e:

            msg = ' ❌ Error writing json file: %s\n  %s' %(FPN, e)

            print(msg)

            return None
        
    return True