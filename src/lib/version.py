"""
 @file version.py

 @brief Version metadata module for the core process library package.

 @details Stores package version constants and metadata used when exposing the
 library identity to notebooks, scripts, and downstream integrations.

 *Version History*:
 - Created: 2024-01-04
 - Updated: 2026-03-15

 @author Thomas Gumbricht

 @date Created: 2024-01-04
 @date Updated: 2026-03-15
"""
__version__ = '1.0.0'

VERSION = tuple( int(x) for x in __version__.split('.') )

metadataD = { 'name':'device_models', 'author':'Thomas Gumbricht', 'author_email':'thomas.gumbricht@gmail.com',
             'title':'postgresdb', 'label':'PosgreSQL interfacing.',
             'image':'avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A'}