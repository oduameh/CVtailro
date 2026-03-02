"""Email service — verification and password-reset tokens, SMTP delivery."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import current_app, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token generation / validation
# ---------------------------------------------------------------------------

_VERIFY_SALT = "email-verify-v1"
_RESET_SALT = "password-reset-v1"


def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def generate_verification_token(email: str) -> str:
    """Create a URL-safe token for email verification (valid 24 h)."""
    return _get_serializer().dumps(email, salt=_VERIFY_SALT)


def confirm_verification_token(token: str, max_age: int = 86400) -> str | None:
    """Return the email if the token is valid, else None."""
    try:
        return _get_serializer().loads(token, salt=_VERIFY_SALT, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None


def generate_reset_token(email: str) -> str:
    """Create a URL-safe token for password reset (valid 1 h)."""
    return _get_serializer().dumps(email, salt=_RESET_SALT)


def confirm_reset_token(token: str, max_age: int = 3600) -> str | None:
    """Return the email if the reset token is valid, else None."""
    try:
        return _get_serializer().loads(token, salt=_RESET_SALT, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None


# ---------------------------------------------------------------------------
# SMTP helpers
# ---------------------------------------------------------------------------

_SMTP_HOST = os.environ.get("SMTP_HOST", "")
_SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
_SMTP_USER = os.environ.get("SMTP_USER", "")
_SMTP_PASS = os.environ.get("SMTP_PASS", "")
_MAIL_FROM = os.environ.get("MAIL_FROM", "")


def _smtp_configured() -> bool:
    return bool(_SMTP_HOST and _SMTP_USER and _SMTP_PASS and _MAIL_FROM)


def send_email(to: str, subject: str, html_body: str, text_body: str) -> bool:
    """Send an email via SMTP.  Returns True on success, False on failure."""
    if not _smtp_configured():
        logger.warning("SMTP not configured — email to %s suppressed: %s", to, subject)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = _MAIL_FROM
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(_SMTP_USER, _SMTP_PASS)
            server.sendmail(_MAIL_FROM, [to], msg.as_string())
        logger.info("Email sent to %s: %s", to, subject)
        return True
    except Exception:
        logger.exception("Failed to send email to %s: %s", to, subject)
        return False


# ---------------------------------------------------------------------------
# High-level email senders
# ---------------------------------------------------------------------------

_BASE_STYLE = (
    "font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, sans-serif; max-width: 520px; margin: 0 auto; padding: 32px;"
)
_BUTTON_STYLE = (
    "display: inline-block; background: #2563eb; color: #ffffff; "
    "padding: 12px 28px; border-radius: 8px; text-decoration: none; "
    "font-weight: 600; font-size: 15px; margin: 20px 0;"
)


def send_verification_email(email: str, name: str) -> bool:
    """Send the email-verification link."""
    token = generate_verification_token(email)
    verify_url = url_for("auth.verify_email", token=token, _external=True)
    display_name = name or email.split("@")[0]

    html = f"""\
<div style="{_BASE_STYLE}">
    <h2 style="color:#0f172a;margin-bottom:8px;">Welcome to CVtailro</h2>
    <p style="color:#64748b;font-size:15px;">Hi {display_name},</p>
    <p style="color:#64748b;font-size:15px;">
        Please verify your email address to activate your account.
    </p>
    <a href="{verify_url}" style="{_BUTTON_STYLE}">Verify Email Address</a>
    <p style="color:#94a3b8;font-size:13px;margin-top:24px;">
        This link expires in 24 hours. If you didn&rsquo;t create an account,
        you can safely ignore this email.
    </p>
</div>"""

    text = (
        f"Hi {display_name},\n\n"
        f"Please verify your email by visiting:\n{verify_url}\n\n"
        f"This link expires in 24 hours.\n"
    )

    return send_email(email, "Verify your CVtailro account", html, text)


def send_reset_email(email: str, name: str) -> bool:
    """Send the password-reset link."""
    token = generate_reset_token(email)
    reset_url = url_for("auth.reset_password_page", token=token, _external=True)
    display_name = name or email.split("@")[0]

    html = f"""\
<div style="{_BASE_STYLE}">
    <h2 style="color:#0f172a;margin-bottom:8px;">Reset your password</h2>
    <p style="color:#64748b;font-size:15px;">Hi {display_name},</p>
    <p style="color:#64748b;font-size:15px;">
        We received a request to reset your password. Click the button below
        to choose a new password.
    </p>
    <a href="{reset_url}" style="{_BUTTON_STYLE}">Reset Password</a>
    <p style="color:#94a3b8;font-size:13px;margin-top:24px;">
        This link expires in 1 hour. If you didn&rsquo;t request a reset,
        you can safely ignore this email.
    </p>
</div>"""

    text = (
        f"Hi {display_name},\n\n"
        f"Reset your password by visiting:\n{reset_url}\n\n"
        f"This link expires in 1 hour.\n"
    )

    return send_email(email, "Reset your CVtailro password", html, text)
