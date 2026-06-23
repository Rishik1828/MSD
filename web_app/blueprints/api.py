"""
blueprints/api.py
─────────────────
REST API routes for the MUSICBEATS web app.

Routes
──────
    POST /api/detect
        Body : JSON { "image": "<base64-encoded JPEG>" }
        Returns: { emotion, confidence, all_probs, emotion_tag, message }

    GET /api/songs
        Query params: emotion, language, era, genre
        Returns: { songs: [...], count: N }

    GET /api/filters
        Returns available filter values for dropdowns.

    GET /api/audio/<int:track_id>
        Streams the MP3 file with full Range-request support so the
        HTML5 <audio> element can seek within the track.
"""

import base64
import os
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file, abort

from web_app.services.emotion_service import detect_emotion
from web_app.services.song_service    import get_songs, get_song_by_id, get_filter_options

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Project root — needed to resolve absolute audio file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── POST /api/detect ──────────────────────────────────────────────────────────

@api_bp.route("/detect", methods=["POST"])
def detect():
    """
    Receive a base64-encoded JPEG frame, run face detection + emotion
    classification, and return the result as JSON.

    Expected body (JSON):
        { "image": "data:image/jpeg;base64,<data>" }
    or just the raw base64 string without the data-URI prefix.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    raw = data["image"]

    # Strip the data-URI prefix if present (e.g. "data:image/jpeg;base64,")
    if "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw)
    except Exception:
        return jsonify({"error": "Invalid base64 image data."}), 400

    try:
        result = detect_emotion(image_bytes)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


# ── GET /api/songs ────────────────────────────────────────────────────────────

@api_bp.route("/songs", methods=["GET"])
def songs():
    """
    Return songs matching the given filter parameters.

    Query params (all optional):
        emotion   — DB emotion_tag value (e.g. "happy")
        language  — "Telugu" | "Hindi" | "English"
        era       — "old" | "new"
        genre     — partial match string
    """
    emotion  = request.args.get("emotion")  or None
    language = request.args.get("language") or None
    era      = request.args.get("era")      or None
    genre    = request.args.get("genre")    or None

    try:
        results = get_songs(
            emotion_tag=emotion,
            language=language,
            era=era,
            genre=genre,
        )
        return jsonify({"songs": results, "count": len(results)})
    except FileNotFoundError as exc:
        return jsonify({
            "error": str(exc),
            "hint": "Run `python setup_db.py` from the project root to create the database."
        }), 503
    except Exception as exc:
        return jsonify({"error": f"Database error: {exc}"}), 500


# ── GET /api/filters ──────────────────────────────────────────────────────────

@api_bp.route("/filters", methods=["GET"])
def filters():
    """Return available filter options (languages, eras, genres, emotions)."""
    return jsonify(get_filter_options())


# ── GET /api/audio/<track_id> ─────────────────────────────────────────────────

@api_bp.route("/audio/<int:track_id>", methods=["GET"])
def audio(track_id: int):
    """
    Stream the MP3 file for the given track_id.

    Uses Flask's send_file with conditional=True which automatically
    handles Range request headers — required for the HTML5 <audio>
    element to seek within a track without re-downloading.
    """
    song = get_song_by_id(track_id)
    if song is None:
        abort(404, description=f"Track {track_id} not found in database.")

    fp = song.get("file_path", "")
    if not fp:
        abort(404, description=f"Track {track_id} has no file path in database.")

    # Resolve absolute path (file_path is relative to project root)
    abs_path = PROJECT_ROOT / fp
    if not abs_path.exists():
        abort(404, description=f"Audio file not found on disk: {abs_path}")

    return send_file(
        abs_path,
        mimetype    = "audio/mpeg",
        conditional = True,   # enables Range-request / partial-content (206) support
        as_attachment = False,
    )
