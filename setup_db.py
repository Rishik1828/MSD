"""
setup_db.py
───────────
Creates data/songs.db and populates it from two sources:

1.  songs.csv.txt  — curated song metadata (title, artist, language,
                     genre, era, emotion_tag, mood)
2.  songs/<mood>/<language>/*.mp3 — real audio files, used to set
                     file_path on matched rows

Matching strategy:
    For each MP3 file, try to fuzzy-match it against a DB row with the
    same mood + language by comparing normalised title tokens in the filename.

Run:
    cd c:\\Users\\saiha\\Downloads\\MSD
    .\\venv\\Scripts\\activate
    python setup_db.py
"""

import csv
import re
import sqlite3
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR       = PROJECT_ROOT / "data"
DB_PATH      = DB_DIR / "songs.db"
CSV_PATH     = PROJECT_ROOT / "songs.csv.txt"
SONGS_DIR    = PROJECT_ROOT / "songs"

DB_DIR.mkdir(exist_ok=True)

# ── Mood → emotion_tag mapping (from schema CHECK constraint) ─────────────────
# Not all moods map 1:1 to emotion_tags; this is used only as fallback.
MOOD_TO_EMOTION: dict[str, str] = {
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "calm":      "neutral",
    "romantic":  "sad",
    "energetic": "happy",
    "anxious":   "fear",
    "surprised": "surprise",
}

# Valid DB CHECK values
VALID_EMOTIONS = {"happy", "sad", "angry", "neutral", "fear", "surprise", "disgust"}
VALID_MOODS    = {"happy", "sad", "angry", "calm", "romantic", "energetic", "anxious", "surprised"}
VALID_LANGS    = {"Telugu", "Hindi", "English"}
VALID_ERAS     = {"old", "new"}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Create schema
# ═════════════════════════════════════════════════════════════════════════════
def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS songs (
            song_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    NOT NULL,
            artist      TEXT    NOT NULL,
            language    TEXT    NOT NULL,
            genre       TEXT,
            era         TEXT    NOT NULL DEFAULT 'new',
            emotion_tag TEXT    NOT NULL DEFAULT 'neutral',
            mood        TEXT    NOT NULL DEFAULT 'calm',
            file_path   TEXT    DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_emotion  ON songs(emotion_tag);
        CREATE INDEX IF NOT EXISTS idx_mood     ON songs(mood);
        CREATE INDEX IF NOT EXISTS idx_language ON songs(language);
        CREATE INDEX IF NOT EXISTS idx_combined ON songs(emotion_tag, language, era);
    """)
    conn.commit()
    print("✅ Schema created.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Import CSV rows
# ═════════════════════════════════════════════════════════════════════════════
def import_csv(conn: sqlite3.Connection) -> int:
    """Import songs from songs.csv.txt. Returns number of rows inserted."""
    if not CSV_PATH.exists():
        print(f"⚠️  CSV not found at {CSV_PATH} — skipping CSV import.")
        return 0

    inserted = 0
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)   # song_title,artist,language,genre,era,emotion_tag,,mood

        for row in reader:
            if len(row) < 6:
                continue

            title       = row[0].strip()
            artist      = row[1].strip()
            language    = row[2].strip()
            genre       = row[3].strip()
            era         = row[4].strip() or "new"
            emotion_tag = row[5].strip().lower()
            # column 6 is blank (double comma), column 7 is mood
            mood        = row[7].strip().lower() if len(row) > 7 else ""

            # Sanitise against CHECK constraints
            if language    not in VALID_LANGS:    continue
            if era         not in VALID_ERAS:     era         = "new"
            if emotion_tag not in VALID_EMOTIONS: emotion_tag = "neutral"
            if mood        not in VALID_MOODS:    mood        = MOOD_TO_EMOTION.get(emotion_tag, "calm")
            if not title or not artist:           continue

            conn.execute(
                """INSERT INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, '')""",
                (title, artist, language, genre, era, emotion_tag, mood),
            )
            inserted += 1

    conn.commit()
    print(f"✅ Imported {inserted} rows from CSV.")
    return inserted


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Scan songs/ folder and insert/update file paths
# ═════════════════════════════════════════════════════════════════════════════
def _normalise(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into tokens for fuzzy matching."""
    text  = text.lower()
    text  = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 2)


def scan_and_match(conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Walk songs/<mood>/<language>/*.mp3 and either:
      - UPDATE an existing row's file_path if a title match is found
      - INSERT a new row derived from the filename if no match is found

    Returns (matched_count, inserted_count).
    """
    if not SONGS_DIR.exists():
        print(f"⚠️  songs/ directory not found at {SONGS_DIR}")
        return 0, 0

    matched  = 0
    inserted = 0

    for mp3 in sorted(SONGS_DIR.rglob("*.mp3")):
        # Expected path: songs/<mood>/<language>/filename.mp3
        parts = mp3.relative_to(SONGS_DIR).parts
        if len(parts) < 3:
            continue

        mood_folder = parts[0]   # e.g. "happy"
        lang_folder = parts[1]   # e.g. "Telugu"
        rel_path    = mp3.relative_to(PROJECT_ROOT).as_posix()  # songs/happy/Telugu/...

        if lang_folder not in VALID_LANGS:
            continue
        if mood_folder not in VALID_MOODS:
            continue

        emotion_tag = MOOD_TO_EMOTION.get(mood_folder, "neutral")
        filename_tokens = _normalise(mp3.stem)

        # ── Try to match against existing DB rows ──────────────────────────
        candidates = conn.execute(
            "SELECT song_id, title, artist FROM songs WHERE mood = ? AND language = ?",
            (mood_folder, lang_folder),
        ).fetchall()

        best_id    = None
        best_score = 0

        for song_id, title, artist in candidates:
            title_tokens = _normalise(title)
            if not title_tokens:
                continue
            overlap = len(filename_tokens & title_tokens)
            score   = overlap / max(len(title_tokens), 1)
            if score > best_score:
                best_score = score
                best_id    = song_id

        if best_id and best_score >= 0.4:
            # Good match — update file_path on the existing row
            conn.execute(
                "UPDATE songs SET file_path = ? WHERE song_id = ?",
                (rel_path, best_id),
            )
            matched += 1
        else:
            # No CSV match — derive a clean title from the filename
            raw_stem = mp3.stem
            # Remove common suffixes like "official_audio", "Telugu_song", etc.
            clean = re.sub(
                r"[_](official|audio|song|video|full|hd|lyrics|[a-z]{2,8}$)",
                "", raw_stem, flags=re.IGNORECASE,
            )
            clean_title = clean.replace("_", " ").strip()

            conn.execute(
                """INSERT INTO songs (title, artist, language, genre, era,
                                     emotion_tag, mood, file_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (clean_title, "Unknown", lang_folder, "", "new",
                 emotion_tag, mood_folder, rel_path),
            )
            inserted += 1

    conn.commit()
    print(f"✅ Matched {matched} files to existing rows, inserted {inserted} new rows from files.")
    return matched, inserted


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"📂 Creating database at {DB_PATH} …\n")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    create_schema(conn)
    csv_count = import_csv(conn)
    matched, new_from_files = scan_and_match(conn)

    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    with_file = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE file_path != ''"
    ).fetchone()[0]

    conn.close()

    print(f"\n{'─'*50}")
    print(f"  Total songs in DB : {total}")
    print(f"  With audio file   : {with_file}")
    print(f"  Without audio     : {total - with_file}")
    print(f"{'─'*50}")
    print(f"\n✅ Database ready at: {DB_PATH}")
    print("   Now start the web app:  python web_app/app.py")
