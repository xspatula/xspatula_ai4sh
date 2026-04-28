"""
 @file update_dict.py

 @brief Dictionary merging utility with default value injection.

 @details Adds missing keys from a default dictionary into a target dictionary,
 with optional recursive handling for nested configuration structures.

 *Version History*:
 - Created: 2023-01-13

 @author Thomas Gumbricht

 @date Created: 2023-01-13
"""


def Update_dict(main_D, default_D, recursive=True):
    """
    @brief Update a dictionary with default values for missing keys.

    @details
    This function updates the dictionary `main_D` by adding any keys from `default_D`
    that are not present in `main_D`. If a key exists in both dictionaries, the value 
    in `main_D` is preserved. 
    
    When recursive mode is enabled (default), nested dictionaries are merged recursively,
    combining keys at all levels while preserving main_D values at each level.

    @param main_D dict
        The main dictionary to be updated. Values in this dictionary take precedence.
        Modified in place.
    
    @param default_D dict
        The dictionary containing default values. Only missing keys are added from this.
    
    @param recursive bool, optional (default=True)
        If True, recursively merge nested dictionaries at all levels.
        If False, only merge top-level keys without recursion.
    
    @return None
        The function modifies main_D in place and does not return a value.
    """

    for key in default_D:
        
        if key not in main_D:
            # Key missing in main_D, add it from default_D
            main_D[key] = default_D[key]
            
        elif recursive and isinstance(main_D[key], dict) and isinstance(default_D[key], dict):
            # Both values are dictionaries and recursive mode is enabled
            # Recursively merge the nested dictionaries
            Update_dict(main_D[key], default_D[key], recursive=True)