"""WSGI entry point — used by Gunicorn in production.

Logging is configured by create_app() via init_structured_logging().
"""

import logging
import os

from app import create_app

application = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logging.getLogger("cvtailro").info("Starting CVtailro on port %s", port)
    print(f"\n  CVtailro Web UI\n  {'─' * 15}\n  Open http://localhost:{port} in your browser\n")
    application.run(host="0.0.0.0", port=port, debug=False, threaded=True)
