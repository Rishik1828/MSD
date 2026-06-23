"""
app.py — MUSICBEATS Flask Application Entry Point
──────────────────────────────────────────────────
Creates the Flask app, registers blueprints, and loads the emotion
detection model once at startup before serving any requests.

Run:
    cd c:\\Users\\saiha\\Downloads\\MSD
    .\\venv\\Scripts\\activate
    pip install flask
    python web_app/app.py

Then open: http://localhost:5000
"""

import sys
from pathlib import Path

from flask import Flask

# ── Ensure this file can be run from anywhere inside the project ───────────────
# Add project root to sys.path so `from web_app.X import Y` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ────────────────────────────────────────────────────────────────────
from web_app.blueprints.main import main_bp
from web_app.blueprints.api  import api_bp
from web_app.services.emotion_service import init_emotion_service


def create_app() -> Flask:
    """
    Flask application factory.

    1. Creates the Flask instance with correct template/static dirs.
    2. Registers blueprints.
    3. Loads the PyTorch emotion model into memory (once).
    """
    app = Flask(
        __name__,
        template_folder = str(Path(__file__).parent / "templates"),
        static_folder   = str(Path(__file__).parent / "static"),
    )

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    # ── Load emotion model at startup (singleton in emotion_service.py) ───────
    # This takes ~3 seconds but only happens once — not on every request.
    print("⏳ Loading emotion detection model…")
    init_emotion_service()
    print("🚀 MUSICBEATS ready at http://localhost:5000")

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.run(
        host  = "127.0.0.1",
        port  = 5000,
        debug = True,   # set True during development for auto-reload
    )
