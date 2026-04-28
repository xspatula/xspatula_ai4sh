"""
 @file version.py

 @brief Version metadata module for shared utility components.

 @details Stores version constants and package metadata for the utility helper
 collection used across the repository.

 *Version History*:
 - Created: 2018-02-21

 @author Thomas Gumbricht

 @date Created: 2018-02-21
"""
__version__ = '1.0'
VERSION = tuple( int(x) for x in __version__.split('.') )
metadataD = { 'name':'support', 'author':'Thomas Gumbricht', 'author_email':'thomas.gumbricht@gmail.com',
             'title':'support', 'label':'Various support processes.',
             'image':'avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A'}
