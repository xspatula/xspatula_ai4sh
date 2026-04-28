"""!
@file csv_read_write.py

@brief Utility module for reading and writing CSV and text files with proper formatting and error handling.

@details
This module provides a comprehensive set of functions for CSV and text file I/O operations. It wraps
Python's built-in csv module to provide convenient interfaces for common file operations with added
safety features like file existence checking and automatic CSV formatting.

The module includes four main functions:
- **Read_csv()**: Safe CSV reading with file existence validation
- **Read_excel()**: Excel-dialect CSV reading with existence validation
- **Write_txt_L()**: Simple text file writing from string lists
- **Write_csv_header_data()**: Structured CSV writing with headers

@author thomasgumbricht

@date Created on 4 Jan 2024
"""

# Standard library imports
import csv  

from os import path

# Third parthy imports
import pandas as pd

def Read_csv(FPN, mode = 'r'):
    """!
    @brief Reads a CSV file and returns its header and data rows.

    @details
    This function safely reads CSV files by first checking for file existence before attempting to open.
    If the file exists, it uses Python's csv.reader to parse the file, extracting the first row as headers
    and subsequent rows as data. 
    
    The function uses the default CSV dialect which handles:
    - Comma (,) as field delimiter
    - Standard quoting and escaping conventions
    - Automatic handling of quoted fields containing delimiters
    
    @param FPN String containing the full path name to the CSV file. Can be relative or absolute path.
    
    @param mode String specifying the file open mode. Default is 'r' (read text mode).
                Other options include 'rb' for binary mode, though 'r' is standard for CSV files.

    @return Tuple (column_L, data_L_L) where:
            - column_L: List of strings representing column headers from the first row
            - data_L_L: List of lists, where each inner list is a row of string values
            
            Returns None if the file does not exist, allowing calling code to handle missing files.
        
    @see csv.reader documentation: https://docs.python.org/3/library/csv.html#csv.reader
    """
    if not path.exists(FPN):
        
        msg = ' ❌ ERROR - csv file not found:\n     %s' %(FPN)

        print (msg)
        
        return None

    with open(FPN, mode) as csv_file:
   
        csvreader = csv.reader(csv_file)

        column_L = next(csvreader)

        data_L_L = [row for row in csvreader]

    return (column_L, data_L_L)

def Read_excel(FPN):
    """!
    @brief Reads a CSV file using the 'excel' dialect and returns its header and data rows.

    @details
    This function reads CSV files formatted according to the Excel dialect specification, which handles
    Excel-specific CSV formatting conventions such as comma delimiters, double-quote text qualifiers,
    and carriage return/line feed line terminators. The function extracts the header row separately
    from the data rows for convenient processing.
    
    The Excel dialect is defined by Python's csv module and corresponds to the CSV format produced
    by Microsoft Excel. It uses:
    - Comma (,) as field delimiter
    - Double quotes (") for text qualification
    - CRLF (\\r\\n) as line terminator
    - Double quotes escaped by doubling ("")
    
    @param FPN String containing the full path name to the CSV file. Must be a valid path to an
               existing file in Excel CSV format.

    @return Tuple (column_L, data_L_L) where:
            - column_L: List of strings representing column headers from the first row
            - data_L_L: List of lists, where each inner list is a row of string values

            Returns None if the file does not exist, allowing calling code to handle missing files.
    
    @see csv.reader documentation: https://docs.python.org/3/library/csv.html#csv.reader
    """
    if not path.exists(FPN):
        
        msg = ' ❌ ERROR - excel file not found:\n     %s' %(FPN)

        print (msg)
        
        return None
    try:
        result = pd.read_excel(FPN)

        column_L =list(result.columns.values)

        #data_L_L = result.values.tolist()

        data_L_L = [[y if pd.notna(y) else 'null' for y in x] for x in result.values.tolist()]
        

    except:
        
        msg = ' ❌ ERROR - failed to read excel file:\n     %s' %(FPN)

        print (msg)

        return None

    return (column_L, data_L_L)
    
def Write_txt_L(FPN, data_L):
    """!
    @brief Writes a list of strings to a text file, one per line.

    @details
    This function writes a list of string values to a text file, writing each string exactly as provided
    in the list. Unlike Write_csv_header_data() which formats data as CSV with headers, this function
    performs simple line-by-line text file writing without any special formatting.
    
    @param FPN String containing the full path name to the output text file. If the file exists,
               it will be overwritten. Parent directories must exist or FileNotFoundError will be raised.
    
    @param data_L List of strings to write to the file. Each element is written sequentially without
                  modification. Include '\n' in strings if line breaks are needed.

    @return None. The function writes directly to the file system.
    
    @warning The file is opened in write mode ('w'), which will completely overwrite any existing file
             at the specified path without warning. Use caution to avoid data loss.
    
    @see Write_csv_header_data() for structured CSV file writing with headers
    """
    with open(FPN, 'w') as txt_file:
        for line in data_L:
            txt_file.write(line)    

def Write_csv_header_data(FPN, column_L, data_L_L, mode = 'w', lineterminator='\n'):
    """!
    @brief Writes a CSV file with a header row and data rows using Python's csv.writer.

    @details
    This function creates properly formatted CSV files with a header row followed by data rows.
    It uses Python's csv.writer which automatically handles CSV formatting conventions including:
    - Comma (,) as field delimiter
    - Automatic quoting of fields containing special characters (commas, quotes, newlines)
    - Proper escaping of quote characters within fields
    - Consistent line termination

    @param FPN String containing the full path name to the output CSV file. If the file exists and
               mode is 'w', it will be overwritten. Parent directories must exist.
    
    @param column_L List of strings representing column headers. These will be written as the first
                    row of the CSV file. Order determines column order in the output.
    
    @param data_L_L List of lists containing the data rows. Each inner list represents one row and
                    should have the same length as column_L. Values are automatically converted to
                    strings and properly escaped for CSV format.
    
    @param mode String specifying file open mode. Default is 'w' (write, overwrites existing file).
                Other options include 'a' (append), 'x' (exclusive creation). Must be a valid
                file mode for Python's open() function.
    
    @param lineterminator String specifying the line terminator character(s). Default is '\\n' (Unix
                          line ending). Can be set to '\\r\\n' for Windows-style CRLF line endings
                          or other values as needed.

    @return None. The function writes directly to the file system.

    @see csv.writer documentation: https://docs.python.org/3/library/csv.html#csv.writer  
    """

    with open(FPN, mode, newline='') as csv_file:

        csvwriter = csv.writer(csv_file, lineterminator=lineterminator)

        csvwriter.writerow(column_L)

        csvwriter.writerows(data_L_L)