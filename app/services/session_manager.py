"""Session manager — server-side session lifecycle, GeoIP, and UA parsing."""

from __future__ import annotations

import logging
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import requests as http_requests
from flask import session as flask_session

from app.extensions import db

logger = logging.getLogger(__name__)

_ACTIVITY_THROTTLE_SECONDS = 60
_GEOIP_TIMEOUT = 3
_GEOIP_CACHE_SIZE = 256


# ---------------------------------------------------------------------------
# Admin-configurable defaults
# ---------------------------------------------------------------------------

def _get_config(key: str, default: int) -> int:
    """Read a session-policy setting from the admin config (DB-backed)."""
    try:
        from app.services.admin_config import AdminConfigManager

        val = AdminConfigManager.get(key)
        if val is not None:
            return int(val)
    except Exception:
        pass
    return default


def _max_sessions() -> int:
    return _get_config("max_concurrent_sessions", 3)


def _idle_timeout_minutes() -> int:
    return _get_config("idle_timeout_minutes", 60)


def _session_max_age_hours() -> int:
    return _get_config("session_max_age_hours", 168)


# ---------------------------------------------------------------------------
# User-Agent parsing
# ---------------------------------------------------------------------------

def _parse_user_agent(ua_string: str) -> dict:
    """Extract device type, browser name, and OS from a User-Agent string."""
    result = {"device_type": "unknown", "browser_name": "unknown", "os_name": "unknown"}
    if not ua_string:
        return result

    try:
        from user_agents import parse

        ua = parse(ua_string)
        if ua.is_mobile:
            result["device_type"] = "mobile"
        elif ua.is_tablet:
            result["device_type"] = "tablet"
        elif ua.is_pc:
            result["device_type"] = "desktop"
        elif ua.is_bot:
            result["device_type"] = "bot"

        result["browser_name"] = ua.browser.family or "unknown"
        result["os_name"] = ua.os.family or "unknown"
    except ImportError:
        ua_lower = ua_string.lower()
        if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
            result["device_type"] = "mobile"
        elif "tablet" in ua_lower or "ipad" in ua_lower:
            result["device_type"] = "tablet"
        else:
            result["device_type"] = "desktop"

        for browser in ("Chrome", "Firefox", "Safari", "Edge", "Opera"):
            if browser.lower() in ua_lower:
                result["browser_name"] = browser
                break

        for os_name in ("Windows", "Mac OS", "Linux", "Android", "iOS"):
            if os_name.lower().replace(" ", "") in ua_lower.replace(" ", ""):
                result["os_name"] = os_name
                break
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# GeoIP lookup (ip-api.com — free for server-side, no key needed)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=_GEOIP_CACHE_SIZE)
def _geoip_lookup_cached(ip: str) -> dict:
    """Cached GeoIP lookup. Returns {"country": ..., "city": ...}."""
    result = {"country": None, "city": None}
    if not ip or ip in ("unknown", "127.0.0.1", "::1"):
        return result
    try:
        resp = http_requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,country,city",
            timeout=_GEOIP_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                result["country"] = data.get("country")
                result["city"] = data.get("city")
    except Exception:
        logger.debug(f"GeoIP lookup failed for {ip}")
    return result


def _geoip_lookup_async(ip: str, session_id: str, app) -> None:
    """Run GeoIP lookup in a background thread and update the session row."""
    def _do_lookup():
        geo = _geoip_lookup_cached(ip)
        if geo["country"] or geo["city"]:
            try:
                with app.app_context():
                    from app.models.user_session import UserSession

                    sess = db.session.get(UserSession, session_id)
                    if sess:
                        sess.country = geo["country"]
                        sess.city = geo["city"]
                        db.session.commit()
            except Exception:
                logger.debug(f"Failed to update GeoIP for session {session_id}")

    thread = threading.Thread(target=_do_lookup, daemon=True)
    thread.start()


def _geoip_lookup_async_event(ip: str, event_id: str, app) -> None:
    """Run GeoIP lookup in a background thread and update the login_event row."""
    def _do_lookup():
        geo = _geoip_lookup_cached(ip)
        if geo["country"] or geo["city"]:
            try:
                with app.app_context():
                    from app.models.login_event import LoginEvent

                    evt = db.session.get(LoginEvent, event_id)
                    if evt:
                        evt.country = geo["country"]
                        evt.city = geo["city"]
                        db.session.commit()
            except Exception:
                logger.debug(f"Failed to update GeoIP for event {event_id}")

    thread = threading.Thread(target=_do_lookup, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def create_session(user, request_obj) -> str:
    """Create a tracked server-side session. Returns the session token."""
    from flask import current_app

    from app.models.user_session import UserSession

    token = secrets.token_hex(32)
    ip = request_obj.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request_obj.remote_addr or "unknown"
    ua_string = request_obj.headers.get("User-Agent", "")
    ua_info = _parse_user_agent(ua_string)

    now = datetime.now(timezone.utc)
    max_age = _session_max_age_hours()
    expires = now + timedelta(hours=max_age)

    session_id = uuid.uuid4().hex
    new_session = UserSession(
        id=session_id,
        user_id=user.id,
        session_token=token,
        ip_address=ip,
        user_agent=ua_string[:512] if ua_string else None,
        device_type=ua_info["device_type"],
        browser_name=ua_info["browser_name"],
        os_name=ua_info["os_name"],
        created_at=now,
        last_activity_at=now,
        expires_at=expires,
        is_active=True,
    )
    db.session.add(new_session)

    _enforce_max_sessions(user.id)

    db.session.commit()
    flask_session["session_token"] = token

    _geoip_lookup_async(ip, session_id, current_app._get_current_object())

    log_login_event(
        user_id=user.id,
        email=user.email,
        event_type="login",
        ip_address=ip,
        user_agent=ua_string,
        success=True,
    )

    return token


def _enforce_max_sessions(user_id: str) -> None:
    """Revoke oldest sessions if user exceeds the max concurrent limit."""
    from app.models.user_session import UserSession

    max_s = _max_sessions()
    active = (
        UserSession.query
        .filter_by(user_id=user_id, is_active=True)
        .order_by(UserSession.created_at.asc())
        .all()
    )

    # +1 because we just added the new session but haven't committed yet
    while len(active) > max_s:
        oldest = active.pop(0)
        oldest.is_active = False
        oldest.revoked_at = datetime.now(timezone.utc)
        oldest.revoked_reason = "max_sessions"
        logger.info(f"Revoked oldest session {oldest.id[:8]} for user {user_id[:8]} (max_sessions)")


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC). SQLite stores naive datetimes."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def validate_session(session_token: str) -> bool:
    """Check if a session token is still valid (active, not expired, not idle)."""
    from app.models.user_session import UserSession

    sess = UserSession.query.filter_by(session_token=session_token, is_active=True).first()
    if not sess:
        return False

    now = datetime.now(timezone.utc)
    expires = _ensure_utc(sess.expires_at)

    if expires and now > expires:
        sess.is_active = False
        sess.revoked_at = now
        sess.revoked_reason = "expired"
        db.session.commit()
        return False

    idle_minutes = _idle_timeout_minutes()
    last_activity = _ensure_utc(sess.last_activity_at)
    if idle_minutes > 0 and last_activity:
        idle_cutoff = now - timedelta(minutes=idle_minutes)
        if last_activity < idle_cutoff:
            sess.is_active = False
            sess.revoked_at = now
            sess.revoked_reason = "idle_timeout"
            db.session.commit()
            log_login_event(
                user_id=sess.user_id,
                event_type="session_expired",
                ip_address=sess.ip_address,
                user_agent=sess.user_agent,
            )
            return False

    return True


def update_activity(session_token: str, force: bool = False) -> None:
    """Update last_activity_at, throttled to avoid DB spam."""
    from app.models.user_session import UserSession

    sess = UserSession.query.filter_by(session_token=session_token, is_active=True).first()
    if not sess:
        return

    now = datetime.now(timezone.utc)
    last = _ensure_utc(sess.last_activity_at)
    if not force and last:
        elapsed = (now - last).total_seconds()
        if elapsed < _ACTIVITY_THROTTLE_SECONDS:
            return

    sess.last_activity_at = now
    db.session.commit()


def get_session_info(session_token: str):
    """Get a UserSession by token, or None."""
    from app.models.user_session import UserSession

    return UserSession.query.filter_by(session_token=session_token).first()


def get_active_sessions(user_id: str) -> list:
    """List all active sessions for a user, newest first."""
    from app.models.user_session import UserSession

    return (
        UserSession.query
        .filter_by(user_id=user_id, is_active=True)
        .order_by(UserSession.last_activity_at.desc())
        .all()
    )


def revoke_session(session_id: str, reason: str = "manual") -> bool:
    """Revoke a specific session by its ID."""
    from app.models.user_session import UserSession

    sess = db.session.get(UserSession, session_id)
    if not sess or not sess.is_active:
        return False

    sess.is_active = False
    sess.revoked_at = datetime.now(timezone.utc)
    sess.revoked_reason = reason
    db.session.commit()

    log_login_event(
        user_id=sess.user_id,
        event_type="session_revoked",
        ip_address=sess.ip_address,
        user_agent=sess.user_agent,
    )
    return True


def revoke_session_by_token(session_token: str, reason: str = "logout") -> bool:
    """Revoke a session by its token."""
    from app.models.user_session import UserSession

    sess = UserSession.query.filter_by(session_token=session_token, is_active=True).first()
    if not sess:
        return False

    sess.is_active = False
    sess.revoked_at = datetime.now(timezone.utc)
    sess.revoked_reason = reason
    db.session.commit()
    return True


def revoke_all_other_sessions(user_id: str, current_token: str | None) -> int:
    """Revoke all sessions except the one matching current_token. Returns count revoked."""
    from app.models.user_session import UserSession

    now = datetime.now(timezone.utc)
    sessions = UserSession.query.filter_by(user_id=user_id, is_active=True).all()
    count = 0
    for s in sessions:
        if s.session_token != current_token:
            s.is_active = False
            s.revoked_at = now
            s.revoked_reason = "manual"
            count += 1
    if count:
        db.session.commit()
    return count


def revoke_all_user_sessions(user_id: str) -> int:
    """Admin: force-logout a user from all devices. Returns count revoked."""
    from app.models.user_session import UserSession

    now = datetime.now(timezone.utc)
    sessions = UserSession.query.filter_by(user_id=user_id, is_active=True).all()
    count = 0
    for s in sessions:
        s.is_active = False
        s.revoked_at = now
        s.revoked_reason = "admin"
        count += 1
    if count:
        db.session.commit()
    return count


def cleanup_expired_sessions() -> int:
    """Mark expired/idle sessions as inactive. Called periodically."""
    from app.models.user_session import UserSession

    now = datetime.now(timezone.utc)
    idle_cutoff = now - timedelta(minutes=_idle_timeout_minutes())

    expired = (
        UserSession.query
        .filter_by(is_active=True)
        .filter(
            db.or_(
                UserSession.expires_at < now,
                UserSession.last_activity_at < idle_cutoff,
            )
        )
        .all()
    )

    for s in expired:
        s.is_active = False
        s.revoked_at = now
        exp = _ensure_utc(s.expires_at)
        s.revoked_reason = "expired" if (exp and now > exp) else "idle_timeout"

    if expired:
        db.session.commit()
    return len(expired)


# ---------------------------------------------------------------------------
# Login event logging
# ---------------------------------------------------------------------------

def log_login_event(
    user_id: str | None = None,
    email: str | None = None,
    event_type: str = "login",
    ip_address: str | None = None,
    user_agent: str | None = None,
    success: bool = True,
    failure_reason: str | None = None,
) -> None:
    """Write a LoginEvent row."""
    try:
        from flask import current_app

        from app.models.login_event import LoginEvent

        ua_info = _parse_user_agent(user_agent or "")
        event_id = uuid.uuid4().hex

        event = LoginEvent(
            id=event_id,
            user_id=user_id,
            email=email,
            event_type=event_type,
            ip_address=ip_address,
            user_agent=(user_agent or "")[:512],
            device_type=ua_info["device_type"],
            browser_name=ua_info["browser_name"],
            os_name=ua_info["os_name"],
            success=success,
            failure_reason=failure_reason,
            created_at=datetime.now(timezone.utc),
        )
        db.session.add(event)
        db.session.commit()

        if ip_address and ip_address not in ("unknown", "127.0.0.1", "::1"):
            _geoip_lookup_async_event(ip_address, event_id, current_app._get_current_object())

    except Exception:
        logger.exception("Failed to log login event")
        db.session.rollback()


# ---------------------------------------------------------------------------
# Admin queries
# ---------------------------------------------------------------------------

def get_all_active_sessions(limit: int = 100) -> list:
    """Admin: get all active sessions across all users."""
    from app.models.user_session import UserSession

    return (
        UserSession.query
        .filter_by(is_active=True)
        .order_by(UserSession.last_activity_at.desc())
        .limit(limit)
        .all()
    )


def get_login_history(
    user_id: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    """Admin: get login events with optional filters."""
    from app.models.login_event import LoginEvent

    q = LoginEvent.query
    if user_id:
        q = q.filter_by(user_id=user_id)
    if event_type:
        q = q.filter_by(event_type=event_type)
    return q.order_by(LoginEvent.created_at.desc()).offset(offset).limit(limit).all()


def get_login_history_count(user_id: str | None = None, event_type: str | None = None) -> int:
    """Admin: count login events with optional filters."""
    from app.models.login_event import LoginEvent

    q = LoginEvent.query
    if user_id:
        q = q.filter_by(user_id=user_id)
    if event_type:
        q = q.filter_by(event_type=event_type)
    return q.count()
