"""
 @file __init__.py

 @brief Public package interface for process initialization utilities.

 @details Exports the main helpers for locating configuration files, logging in,
 structuring process definitions, and starting package workflows from notebooks
 or scripts.

 *Version History*:
 - Created: 2025-01-05

 @author Thomas Gumbricht

 @date Created: 2025-01-05
"""
from .version import __version__, VERSION, metadataD

from .pilot import Get_project_path, Full_path_locate, Get_scheme_project_path_setup

from .login import Project_login

from .structure import Structure_processes

from .initiate import Initiate_process