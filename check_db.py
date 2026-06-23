import sqlite3
from pathlib import Path

db_path = Path("data/songs.db")
if not db_path.exists():
    print("Database does not exist!")
else:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    total = cursor.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    matched = cursor.execute("SELECT COUNT(*) FROM songs WHERE file_path != ''").fetchone()[0]
    print(f"Total songs in DB: {total}")
    print(f"Songs with matched file_path: {matched}")
    
    print("\nFirst 10 songs:")
    rows = cursor.execute("SELECT title, artist, language, mood, file_path FROM songs LIMIT 10").fetchall()
    for row in rows:
        print(dict(row))
        
    print("\nFirst 10 unmatched songs:")
    unmatched_rows = cursor.execute("SELECT title, artist, language, mood, file_path FROM songs WHERE file_path = '' LIMIT 10").fetchall()
    for row in unmatched_rows:
        print(dict(row))
        
    conn.close()
