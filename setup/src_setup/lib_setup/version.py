"""
 @file version.py

 @brief Version metadata module for setup workflow components.

 @details Stores version constants and package metadata for the database setup
 and seeding helpers.

 *Version History*:
 - Created: 2020-12-31
 - Updated: 2025-01-10
 - Updated: 2026-03-15

 @author Thomas Gumbricht

 @date Created: 2020-12-31
 @date Updated: 2025-01-10
 @date Updated: 2026-03-15
"""
__version__ = '1.0.0'

VERSION = tuple( int(x) for x in __version__.split('.') )

metadataD = { 'name':'setup_db', 'author':'Thomas Gumbricht', 'author_email':'thomas.gumbricht@gmail.com',
             'title':'setup database', 'label':'Setup postgreSQL database using json files and python'}