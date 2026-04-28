"""
 @file pretty_print.py

 @brief Utility module for formatted parameter printing.

 @details Provides a thin wrapper around Python's pprint utilities for readable
 inspection of nested parameters and configuration dictionaries.

 *Version History*:
 - Created: 2022-11-03

 @author Thomas Gumbricht

 @date Created: 2022-11-03
"""

# Standard library imports
from pprint import pprint

def Pprint_parameter(parameter):
        """
        @brief Pretty-prints a given parameter using Python's pprint module for better readability.

        @param parameter: The parameter to be pretty-printed.

        @return None
        """
        pprint (parameter)