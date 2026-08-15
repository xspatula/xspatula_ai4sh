"""
 @file email.py

 @brief Transactional email sending for community.user account provisioning.

 @details Sends one-off emails (e.g. a newly generated password) via SMTP. Credentials
 are loaded from src/postgres/environment/.<env_name>.env, the same gitignored,
 dotenv-based convention used for database credentials in src/postgres/pg_session.py.

 @author Thomas Gumbricht

 @date Created: 2026-08-14
"""

from os import path, getenv

import smtplib

import ssl

from email.message import EmailMessage

from dotenv import load_dotenv


def Get_smtp_env_var(env_name='xspatula_email'):
    """
    @brief Load SMTP credentials from src/postgres/environment/.<env_name>.env.

    @param env_name Name of the environment file, without the leading dot or .env suffix
        (default 'xspatula_email').

    @return Dictionary with keys 'host', 'port', 'user', 'password', 'use_tls', or None if
        the environment file is missing or incomplete.
    """

    BASEDIR = path.abspath(path.join(path.dirname(__file__), '..', 'postgres'))

    env_FN = '.%s.env' % (env_name)

    env_FP = path.join(BASEDIR, 'environment', env_FN)

    if not path.exists(env_FP):

        print('❌ ERROR: the email environment file <%s> does not exist' % (env_FN))

        return None

    load_dotenv(env_FP, override=True)

    SMTP_HOST = getenv('SMTP_HOST')
    SMTP_PORT = getenv('SMTP_PORT')
    SMTP_USER = getenv('SMTP_USER')
    SMTP_PASSWORD = getenv('SMTP_PASSWORD')
    SMTP_USE_TLS = getenv('SMTP_USE_TLS', 'true').lower() in ('1', 'true', 'yes')

    missing = [k for k, v in (('SMTP_HOST', SMTP_HOST), ('SMTP_PORT', SMTP_PORT),
                               ('SMTP_USER', SMTP_USER), ('SMTP_PASSWORD', SMTP_PASSWORD)) if v is None]
    if missing:

        print('❌ ERROR: missing environment variable(s) in <%s>: %s' % (env_FN, ', '.join(missing)))

        return None

    return {'host': SMTP_HOST, 'port': int(SMTP_PORT), 'user': SMTP_USER,
            'password': SMTP_PASSWORD, 'use_tls': SMTP_USE_TLS}


def Send_email(to_addr, subject, body_text, env_name='xspatula_email'):
    """
    @brief Send a single plaintext email via SMTP.

    @param to_addr Recipient email address.
    @param subject Email subject line.
    @param body_text Plaintext email body.
    @param env_name Name of the SMTP environment file to use (default 'xspatula_email').

    @return True if the message was handed off to the SMTP server, False otherwise.
    """

    smtp_D = Get_smtp_env_var(env_name)

    if smtp_D is None:

        return False

    msg = EmailMessage()

    msg['Subject'] = subject

    msg['From'] = smtp_D['user']

    msg['To'] = to_addr

    msg.set_content(body_text)

    try:
        if smtp_D['use_tls']:

            with smtplib.SMTP(smtp_D['host'], smtp_D['port']) as server:

                server.starttls(context=ssl.create_default_context())

                server.login(smtp_D['user'], smtp_D['password'])

                server.send_message(msg)

        else:

            with smtplib.SMTP_SSL(smtp_D['host'], smtp_D['port'], context=ssl.create_default_context()) as server:

                server.login(smtp_D['user'], smtp_D['password'])

                server.send_message(msg)

        return True

    except Exception as e:

        print('❌ ERROR - could not send email to %s: %s' % (to_addr, e))

        return False
