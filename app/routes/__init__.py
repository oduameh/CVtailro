"""Route blueprints package — registers all blueprints with the Flask app."""

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from app.routes.admin import admin_bp
    from app.routes.api import api_bp
    from app.routes.auth import auth_bp
    from app.routes.blog import blog_bp
    from app.routes.history import history_bp
    from app.routes.main import main_bp
    from app.routes.saved_resumes import saved_resumes_bp
    from app.routes.tracker import tracker_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(blog_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(saved_resumes_bp)
    app.register_blueprint(tracker_bp)
