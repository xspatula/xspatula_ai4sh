"""
 @file struct.py

 @brief Recursive dictionary-to-object conversion utility for configuration management.

 @details Converts nested dictionaries and lists into attribute-accessible Python
 objects so configuration data can be consumed with simpler dot-notation syntax.

 *Version History*:
 - Created: 2023-09-03

 @author Thomas Gumbricht

 @date Created: 2023-09-03
"""

class Struct(object):
    """
    @brief Attribute-access wrapper for nested configuration data.

    @details Recursively converts dictionaries and supported iterables into nested
    Struct instances so configuration values can be accessed with dot notation.
    """

    def __init__(self, data):
        """
        @brief Initialize a Struct object from a dictionary.

        @details
        This constructor takes a dictionary and sets each key-value pair as an attribute of the Struct instance.
        If a value is itself a dictionary, it is recursively wrapped as a Struct object. Lists, tuples, sets, and frozensets
        are also recursively wrapped.

        @param data Dictionary containing the data to initialize the Struct object.
        """

        for name, value in data.items():

            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        """
        @brief Recursively wraps values for Struct initialization.

        @details
        This method takes a value and recursively wraps it as a Struct object if it is a dictionary.
        For iterable types (tuple, list, set, frozenset), it applies itself to each element and returns
        the same type containing the wrapped elements. For other types, it returns the value unchanged.

        @param value The value to wrap. Can be a dict, tuple, list, set, frozenset, or any other type.
        @return The wrapped value: Struct if dict, recursively wrapped iterable if tuple/list/set/frozenset, or the value itself otherwise.
        """

        if isinstance(value, (tuple, list, set, frozenset)):

            return type(value)([self._wrap(v) for v in value])

        else:

            return Struct(value) if isinstance(value, dict) else value