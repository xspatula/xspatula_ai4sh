"""
 @file registration.py

 @brief Glue between the generic translate_tabular_data/manage_process pipeline and
 password provisioning for new community.user accounts.

 @details The generic pipeline (translate_tabular_data -> manage_<table>) has no notion
 of generating a value that isn't in the source Excel, or of side effects like sending an
 email - by design, it's a pure column-to-parameter mapper. Registration excel sheets
 intentionally have no password column (see ai4sh/user_management/user/excel/user.xlsx),
 so a generated manage_user.json has no password parameter yet. Provision_user_passwords
 fills that gap: for each row, generate a password, hash it into the parameters (so the
 normal manage_user process can insert it like any other field), and email the plaintext
 password once to the new user. Nothing here is AI4SH-specific.

 @author Thomas Gumbricht

 @date Created: 2026-08-14
"""

from src.utils import Read_json, Dump_json

from .password import Generate_password, Hash_password

from .email import Send_email

WELCOME_SUBJECT = 'Your xspatula account'

WELCOME_BODY = """Hello %s,

An account has been created for you.

  user name: %s
  password:  %s

Please keep this password somewhere safe - there is currently no self-service way to
change or recover it, so contact the administrator if you need it reset.
"""


def Provision_user_passwords(manage_user_json_path):
    """
    @brief Generate, hash and email a password for each user in a generated manage_user.json.

    @param manage_user_json_path Path to a manage_user.json produced by translate_tabular_data
        (a {"process": [{"process": "manage_user", "parameters": {...}}, ...]} file).

    @details For each process block: generates a random password, stores its bcrypt hash in
        parameters['password'], emails the plaintext password to parameters['email'], then
        rewrites the file in place with the hashes filled in. A row whose email fails to
        send still gets its password hash written (so the account isn't blocked), but is
        flagged in the returned summary so it can be handled manually.

    @return List of dicts, one per processed row: {'user_name', 'email', 'emailed': bool}.
    """

    data_D = Read_json(manage_user_json_path)

    summary_L = []

    for block_D in data_D.get('process', []):

        parameters_D = block_D.get('parameters', {})

        plaintext_password = Generate_password()

        parameters_D['password'] = Hash_password(plaintext_password)

        user_name = parameters_D.get('user_name', '')

        email = parameters_D.get('email', '')

        first_name = parameters_D.get('first_name', user_name)

        emailed = False

        if email:

            emailed = Send_email(
                email,
                WELCOME_SUBJECT,
                WELCOME_BODY % (first_name, user_name, plaintext_password)
            )

        summary_L.append({'user_name': user_name, 'email': email, 'emailed': emailed})

    Dump_json(manage_user_json_path, data_D)

    return summary_L
