"""
services/song_service.py
────────────────────────
SQLite query helpers for the MUSICBEATS song database.

Schema (from songs_schema.sql):
    songs(song_id, title, artist, language, genre, era,
          emotion_tag, mood, file_path)

    emotion_tag ∈ {'happy','sad','angry','neutral','fear','surprise','disgust'}
    mood        ∈ {'happy','sad','angry','calm','romantic','energetic','anxious','surprised'}
    language    ∈ {'Telugu','Hindi','English'}
    era         ∈ {'old','new'}

Public API
──────────
    get_songs(emotion_tag, language, era, genre, limit) → list[dict]
    get_song_by_id(song_id)                             → dict | None
    get_filter_options()                                → dict of lists
"""

import sqlite3
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # MSD/
DB_PATH      = PROJECT_ROOT / "data" / "songs.db"


def _connect() -> sqlite3.Connection:
    """Open a thread-local DB connection with Row factory."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run:  python setup_db.py  to create it."
        )
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row, project_root: Path) -> dict:
    """
    Convert a DB row to a plain dict and enrich it with:
      - has_file : bool — whether the MP3 actually exists on disk
      - display_path : str — just the filename (not full path)
    """
    d = dict(row)
    fp = d.get("file_path", "")
    full_path = project_root / fp if fp else None
    d["has_file"]     = bool(fp and full_path and full_path.exists())
    d["display_path"] = Path(fp).name if fp else ""
    return d


def get_songs(
    emotion_tag: Optional[str] = None,
    language:    Optional[str] = None,
    era:         Optional[str] = None,
    genre:       Optional[str] = None,
    limit:       int           = 60,
) -> list[dict]:
    """
    Query songs matching the given filters.

    Parameters
    ----------
    emotion_tag : str, optional
        One of the DB emotion_tag values (e.g. "happy").
    language : str, optional
        "Telugu", "Hindi", or "English".
    era : str, optional
        "old" or "new".
    genre : str, optional
        Genre string — uses LIKE match (partial OK).
    limit : int
        Max rows to return (default 60).

    Returns
    -------
    list of song dicts. Each dict includes all DB columns plus
    `has_file` (bool) and `display_path` (str).
    """
    conditions: list[str] = []
    params:     list      = []

    if emotion_tag:
        conditions.append("emotion_tag = ?")
        params.append(emotion_tag)
    if language:
        conditions.append("language = ?")
        params.append(language)
    if era:
        conditions.append("era = ?")
        params.append(era)
    if genre:
        conditions.append("genre LIKE ?")
        params.append(f"%{genre}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql   = f"""
        SELECT DISTINCT song_id, title, artist, language, genre, era,
               emotion_tag, mood, file_path
        FROM   songs
        {where}
        ORDER  BY title
        LIMIT  ?
    """
    params.append(limit)

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r, PROJECT_ROOT) for r in rows]
    finally:
        conn.close()


def get_song_by_id(song_id: int) -> Optional[dict]:
    """
    Fetch a single song by primary key.

    Returns None if not found.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM songs WHERE song_id = ?", (song_id,)
        ).fetchone()
        return _row_to_dict(row, PROJECT_ROOT) if row else None
    finally:
        conn.close()


def get_filter_options() -> dict:
    """
    Return all distinct values for each filterable column.
    Used to populate the frontend filter dropdowns.

    Returns
    -------
    {
        "languages": [...],
        "eras":      [...],
        "genres":    [...],
        "emotions":  [...],
    }
    """
    conn = _connect()
    try:
        def distinct(col: str) -> list:
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM songs WHERE {col} IS NOT NULL ORDER BY {col}"
            ).fetchall()
            return [r[0] for r in rows if r[0]]

        return {
            "languages": distinct("language"),
            "eras":      distinct("era"),
            "genres":    distinct("genre"),
            "emotions":  distinct("emotion_tag"),
        }
    finally:
        conn.close()
