"""Development entry point.

Use `python app.py` for local dev, or `gunicorn wsgi:application` for production.
Logging is configured by create_app() via init_structured_logging().
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    import logging

    port = int(os.environ.get("PORT", 5050))
    logging.getLogger("cvtailro").info("Starting CVtailro on port %s", port)
    print(f"\n  CVtailro Web UI\n  {'─' * 15}\n  Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
