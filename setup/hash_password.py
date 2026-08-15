#!/usr/bin/env python3
"""
 @file hash_password.py

 @brief Standalone CLI to bcrypt-hash a password for manual seeding.

 @details Prints a bcrypt hash for copy-pasting into a community.user "password" value -
 either directly into setup/zzz/ai4sh/setup_db/json_ai4sh/community/user_records_v10_sql.json
 (for a fresh setup_db.ipynb run) or into a live UPDATE community.user SET password = '<hash>'
 WHERE ... statement. Needed to bootstrap the very first user(s): the Excel-based
 registration notebook (ai4sh/user_management/register_users.ipynb) inserts through a
 registered process, which requires an already-logged-in community.user - so the first
 user(s) have to be seeded by hand, and since community.user.password is now a bcrypt hash
 (not plaintext), that hash has to be computed somewhere outside the database.

 Usage:
     python3 setup/hash_password.py                  # prompts for the password, hidden input
     python3 setup/hash_password.py 'some-password'   # hash a password given directly (visible
                                                        # in shell history/process list - prefer
                                                        # the no-argument prompt on a shared machine)

 @author Thomas Gumbricht

 @date Created: 2026-08-15
"""

from os import path

import sys

import getpass

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))

from src.community.password import Hash_password


def main():

    if len(sys.argv) > 1:

        plaintext_password = sys.argv[1]

    else:

        plaintext_password = getpass.getpass('Password to hash: ')

        confirm_password = getpass.getpass('Repeat password: ')

        if plaintext_password != confirm_password:

            print('❌ ERROR - passwords do not match')

            sys.exit(1)

    print(Hash_password(plaintext_password))


if __name__ == '__main__':

    main()
