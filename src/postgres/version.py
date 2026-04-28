"""
 @file version.py

 @brief Version metadata module for PostgreSQL support components.

 @details Stores version constants and package metadata for the PostgreSQL helper
 layer exported by the package.

 *Version History*:
 - Created: 2018-02-21
 - Updated: 2021-01-02
 - Updated: 2024-01-04

 @author Thomas Gumbricht

 @date Created: 2018-02-21
 @date Updated: 2021-01-02
 @date Updated: 2024-01-04
"""
__version__ = '1.0'

VERSION = tuple( int(x) for x in __version__.split('.') )

metadataD = { 'name':'postgresdb', 'author':'Thomas Gumbricht', 'author_email':'thomas.gumbricht@gmail.com',
             'title':'postgresdb', 'label':'PosgreSQL interfacing.',
             'image':'avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A'}