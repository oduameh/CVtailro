"""WSGI entry point — used by Gunicorn in production.

Wraps create_app() with error handling so startup failures are visible
in Railway deployment logs instead of silently crashing the worker.
"""

import logging
import os
import sys
import traceback

try:
    from app import create_app

    application = create_app()
    print("[wsgi] Application created successfully", flush=True)
except Exception:
    traceback.print_exc()
    print("[wsgi] FATAL: Application failed to start", file=sys.stderr, flush=True)
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logging.getLogger("cvtailro").info("Starting CVtailro on port %s", port)
    print(f"\n  CVtailro Web UI\n  {'─' * 15}\n  Open http://localhost:{port} in your browser\n")
    application.run(host="0.0.0.0", port=port, debug=False, threaded=True)
