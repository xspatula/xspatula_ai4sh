"""
 @file __init__.py

 @brief Public package interface for database setup workflows.

 @details Exports the main setup, schema creation, and process registration
 helpers used by the setup notebooks.

 *Version History*:
 - Updated: 2026-03-14

 @author Thomas Gumbricht

 @date Updated: 2026-03-14
"""

from .version import __version__, VERSION, metadataD

from .setup_db import Setup_prod_DB, Setup_schemas_tables

from .setup_db_audit import Assemble_audit_config

from .manage_process import Run_process

from .setup_db_initiate import Initiate_database, Initiate_audit

__all__ = ['PGsession', 'parameter']