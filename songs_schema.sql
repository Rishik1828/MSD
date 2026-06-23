-- songs_schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Schema for the Emotion-Based Music Player song database.
-- Run this via:   python src/setup_db.py
-- Or manually:   sqlite3 data/songs.db < songs_schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Main table ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS songs (
    song_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    artist      TEXT    NOT NULL,
    language    TEXT    NOT NULL CHECK(language    IN ('Telugu','Hindi','English')),
    genre       TEXT,
    era         TEXT    NOT NULL CHECK(era         IN ('old','new')),
    emotion_tag TEXT    NOT NULL CHECK(emotion_tag IN (
                            'happy','sad','angry','neutral',
                            'fear','surprise','disgust')),
    mood        TEXT    NOT NULL CHECK(mood        IN (
                            'happy','sad','angry','calm',
                            'romantic','energetic','anxious','surprised')),
    file_path   TEXT    DEFAULT ''     -- local path to .mp3 / .wav
);

-- ── Indices for fast filtered queries ─────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_emotion  ON songs(emotion_tag);
CREATE INDEX IF NOT EXISTS idx_mood     ON songs(mood);
CREATE INDEX IF NOT EXISTS idx_language ON songs(language);
CREATE INDEX IF NOT EXISTS idx_era      ON songs(era);
CREATE INDEX IF NOT EXISTS idx_combined ON songs(emotion_tag, language, era);

-- ─────────────────────────────────────────────────────────────────────────────
-- Sample INSERT statements (representative subset — full DB built via setup_db.py)
-- ─────────────────────────────────────────────────────────────────────────────

-- Telugu / Happy / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Butta Bomma',              'Armaan Malik',       'Telugu', 'Pop',    'new', 'happy', 'happy', ''),
  ('Samajavaragamana',         'Sid Sriram',          'Telugu', 'Melody', 'new', 'happy', 'happy', ''),
  ('Ramuloo Ramulaa',          'Anurag Kulkarni',     'Telugu', 'Folk',   'new', 'happy', 'happy', ''),
  ('Naatu Naatu',              'Rahul Sipligunj',     'Telugu', 'Folk',   'new', 'happy', 'happy', ''),
  ('Oo Antava',                'Indravathi Chauhan',  'Telugu', 'Mass',   'new', 'happy', 'happy', '');

-- Telugu / Happy / Old
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Bangaru Kodipetta',        'SP Balasubrahmanyam', 'Telugu', 'Folk',   'old', 'happy', 'happy', ''),
  ('Aaresukoboyi Paresukunnanu','SP Balasubrahmanyam','Telugu', 'Folk',   'old', 'happy', 'happy', '');

-- Hindi / Sad / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Tum Hi Ho',                'Arijit Singh',        'Hindi',  'Romantic','new', 'sad',  'sad', ''),
  ('Channa Mereya',            'Arijit Singh',        'Hindi',  'Romantic','new', 'sad',  'sad', ''),
  ('Agar Tum Saath Ho',        'Alka Yagnik',         'Hindi',  'Melody', 'new', 'sad',  'sad', '');

-- English / Angry / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Believer',                 'Imagine Dragons',     'English','Rock',   'new', 'angry','angry',''),
  ('Enemy',                    'Imagine Dragons',     'English','Rock',   'new', 'angry','angry',''),
  ('Warriors',                 'Imagine Dragons',     'English','Rock',   'new', 'angry','angry','');

-- English / Angry / Old
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('In The End',               'Linkin Park',         'English','Rock',   'old', 'angry','angry',''),
  ('Numb',                     'Linkin Park',         'English','Rock',   'old', 'angry','angry',''),
  ('Lose Yourself',            'Eminem',              'English','Hip-Hop','old', 'angry','angry','');

-- Telugu / Neutral / Old (classical)
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Jagadananda Karaka',       'SP Balasubrahmanyam', 'Telugu', 'Classical','old','neutral','calm',''),
  ('Brahmam Okate',            'SP Balasubrahmanyam', 'Telugu', 'Devotional','old','neutral','calm',''),
  ('Endaro Mahanubhavulu',     'MS Subbulakshmi',     'Telugu', 'Classical','old','neutral','calm','');

-- English / Neutral / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Perfect',                  'Ed Sheeran',          'English','Pop',    'new', 'neutral','calm',''),
  ('Photograph',               'Ed Sheeran',          'English','Pop',    'new', 'neutral','calm',''),
  ('Yellow',                   'Coldplay',             'English','Rock',   'old', 'neutral','calm','');

-- Telugu / Fear / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Baahubali Theme',          'MM Keeravani',        'Telugu', 'Orchestral','new','fear','anxious',''),
  ('RRR Theme',                'MM Keeravani',        'Telugu', 'Orchestral','new','fear','anxious',''),
  ('Komuram Bheemudo',         'Kaala Bhairava',      'Telugu', 'Folk',   'new', 'fear','anxious','');

-- Hindi / Surprise / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Kesariya',                 'Arijit Singh',        'Hindi',  'Romantic','new','surprise','surprised',''),
  ('Hawayein',                 'Arijit Singh',        'Hindi',  'Romantic','new','surprise','surprised',''),
  ('Love You Zindagi',         'Jasleen Royal',       'Hindi',  'Pop',    'new', 'surprise','surprised','');

-- English / Disgust / New (break-up / heartbreak mood)
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Drivers License',          'Olivia Rodrigo',      'English','Pop',    'new', 'disgust','sad',''),
  ('Easy On Me',               'Adele',               'English','Pop',    'new', 'disgust','sad',''),
  ('Someone You Loved',        'Lewis Capaldi',       'English','Pop',    'new', 'disgust','sad','');

-- Telugu / Sad / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Ninnila Ninnila',          'Sid Sriram',          'Telugu', 'Melody', 'new', 'sad',  'sad', ''),
  ('Undiporaadhey',            'Sid Sriram',          'Telugu', 'Melody', 'new', 'sad',  'sad', ''),
  ('Yenti Yenti',              'Chinmayi',            'Telugu', 'Romantic','new','sad',  'sad', '');

-- English / Happy / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Happy',                    'Pharrell Williams',   'English','Pop',    'new', 'happy','happy',''),
  ('Uptown Funk',              'Mark Ronson',         'English','Funk',   'new', 'happy','happy',''),
  ('Can''t Stop The Feeling',  'Justin Timberlake',   'English','Pop',    'new', 'happy','happy','');

-- English / Fear / Old
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Thriller',                 'Michael Jackson',     'English','Pop',    'old', 'fear', 'anxious', ''),
  ('Zombie',                   'The Cranberries',     'English','Rock',   'old', 'fear', 'anxious', ''),
  ('My Immortal',              'Evanescence',         'English','Ballad', 'old', 'fear', 'anxious', '');

-- English / Surprise / New
INSERT OR IGNORE INTO songs (title, artist, language, genre, era, emotion_tag, mood, file_path) VALUES
  ('Viva La Vida',             'Coldplay',             'English','Rock',   'new', 'surprise','surprised',''),
  ('Wake Me Up',               'Avicii',               'English','EDM',    'new', 'surprise','surprised',''),
  ('Titanium',                 'David Guetta',         'English','EDM',    'new', 'surprise','surprised','');
