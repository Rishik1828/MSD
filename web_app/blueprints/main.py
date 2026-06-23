"""
blueprints/main.py
──────────────────
Serves the single-page app shell.

Routes
──────
    GET /  → renders templates/index.html
"""

from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Render the MUSICBEATS single-page app shell."""
    return render_template("index.html")
