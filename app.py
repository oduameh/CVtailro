"""Legacy entry point — delegates to wsgi.py / create_app().

Kept for backward compatibility with existing deployment configs.
Use `python wsgi.py` or `gunicorn wsgi:application` for new deployments.
"""

import os
import logging

from utils import setup_logging
from app import create_app

setup_logging(verbose=False)
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logging.getLogger("cvtailro").info(f"Starting CVtailro on port {port}")
    print(f"\n  CVtailro Web UI\n  {'─' * 15}\n  Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
