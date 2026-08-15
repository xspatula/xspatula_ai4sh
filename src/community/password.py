"""
 @file password.py

 @brief Password generation and hashing for community.user accounts.

 @details Generates one-time random passwords for newly registered users and
 hashes/verifies passwords with bcrypt. Nothing here is AI4SH-specific - it's
 meant to be reusable by any xspatula project that authenticates against a
 community.user-shaped table.

 @author Thomas Gumbricht

 @date Created: 2026-08-14
"""

import secrets

import bcrypt


def Generate_password(n_bytes=16):
    """
    @brief Generate a random, URL-safe password for a newly registered user.

    @param n_bytes Number of random bytes to draw before base64 encoding (default 16,
        ~128 bits of entropy).

    @details Uses secrets.token_urlsafe rather than the diceware-style passwords used
        elsewhere in this repo's own seed data, since those are meant to be memorized/typed
        by admins - this one is generated once, emailed, and copy-pasted by the recipient,
        so readability doesn't matter and entropy does.

    @return A random password string.
    """
    return secrets.token_urlsafe(n_bytes)


def Hash_password(plaintext_password):
    """
    @brief Hash a plaintext password for storage in community.user.password.

    @param plaintext_password Plaintext password to hash.

    @return The bcrypt hash, as a str (safe to store in a VARCHAR column).
    """
    return bcrypt.hashpw(plaintext_password.encode('utf-8'), bcrypt.gensalt()).decode('ascii')


def Verify_password(plaintext_password, hashed_password):
    """
    @brief Check a plaintext password against a stored bcrypt hash.

    @param plaintext_password Plaintext password supplied at login.
    @param hashed_password Bcrypt hash previously stored via Hash_password.

    @return True if the password matches, False otherwise (including on malformed hashes).
    """
    try:
        return bcrypt.checkpw(plaintext_password.encode('utf-8'), hashed_password.encode('utf-8'))

    except (ValueError, TypeError):

        return False
