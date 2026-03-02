"""WSGI entry point — used by Gunicorn in production and for local development."""

import logging
import os

from app import create_app
from utils import setup_logging

setup_logging(verbose=False)
application = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logging.getLogger("cvtailro").info(f"Starting CVtailro on port {port}")
    print(f"\n  CVtailro Web UI\n  {'─' * 15}\n  Open http://localhost:{port} in your browser\n")
    application.run(host="0.0.0.0", port=port, debug=False, threaded=True)
