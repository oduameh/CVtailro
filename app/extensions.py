"""Shared Flask extensions — initialized without an app, bound later in create_app()."""

import os

from authlib.integrations.flask_client import OAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.session_protection = "strong"
oauth = OAuth()
csrf = CSRFProtect()
sess = Session()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per minute"],
    storage_uri=os.environ.get("REDIS_URL") or "memory://",
)
