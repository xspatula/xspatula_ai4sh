"""
 @file __init__.py

 @brief Public package interface for community.user account provisioning.

 @details Re-exports password generation/hashing and transactional email helpers.
 Reusable by any xspatula project that authenticates against a community.user-shaped
 table - nothing here is AI4SH-specific.

 @author Thomas Gumbricht

 @date Created: 2026-08-14
"""

from .password import Generate_password, Hash_password, Verify_password

from .email import Send_email

from .registration import Provision_user_passwords
