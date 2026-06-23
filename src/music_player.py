"""
music_player.py
───────────────
Emotion-to-music mapping and playback module.

Reads MP3 files directly from the local  songs/<mood>/<language>/  directory
tree (no database needed) and plays them with pygame.mixer.

Folder structure expected:
    songs/
      angry/     English/  Hindi/  Telugu/
      anxious/   English/  Hindi/  Telugu/
      calm/      English/  Hindi/  Telugu/
      energetic/ English/  Hindi/  Telugu/
      happy/     English/  Hindi/  Telugu/
      romantic/  English/  Hindi/  Telugu/
      sad/       English/  Hindi/  Telugu/
      surprised/ English/  Hindi/  Telugu/

Emotion → Mood mapping (matches the 8 model classes):
    Happy    → happy
    Sad      → sad
    Anger    → angry
    Neutral  → calm
    Fear     → anxious
    Contempt → sad
    Disgust  → sad
    Surprise → surprised
"""

import os
import random
import threading
import time
from pathlib import Path


# ── pygame import (optional — gracefully disabled if not installed) ──────────
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False
    print("⚠️  pygame not found. Run: pip install pygame>=2.5.0")


# ── Emotion → Mood folder mapping ────────────────────────────────────────────
EMOTION_TO_MOOD: dict[str, str] = {
    "Happy":    "happy",
    "Sad":      "sad",
    "Anger":    "angry",
    "Neutral":  "calm",
    "Fear":     "anxious",
    "Contempt": "sad",
    "Disgust":  "sad",
    "Surprise": "surprised",
}

# Valid language sub-folders
VALID_LANGUAGES = {"Telugu", "Hindi", "English"}


# ── MusicPlayer ───────────────────────────────────────────────────────────────

class MusicPlayer:
    """
    Thread-safe, non-blocking music player for emotion-based playback.

    Parameters
    ----------
    songs_root : str | Path
        Path to the root `songs/` directory.
    language : str
        One of 'Telugu', 'Hindi', 'English', or 'all' (picks any language).
    emotion_hold_secs : float
        Seconds the emotion must be stable before the song changes.
        Prevents rapid switching on every frame.
    fade_ms : int
        Fade-out duration in milliseconds when switching songs.
    volume : float
        Playback volume in [0.0, 1.0].
    """

    def __init__(
        self,
        songs_root: str | Path = "songs",
        language: str = "all",
        emotion_hold_secs: float = 2.0,
        fade_ms: int = 800,
        volume: float = 0.85,
    ):
        self.songs_root = Path(songs_root)
        self.language = language
        self.emotion_hold_secs = emotion_hold_secs
        self.fade_ms = fade_ms
        self.volume = volume

        self._lock = threading.Lock()
        self._current_mood: str | None = None
        self._current_file: str | None = None
        self._last_seen_emotion: str | None = None
        self._emotion_since: float = 0.0
        self._enabled = False

        # Build the in-memory song index on startup
        self._index: dict[str, list[Path]] = {}
        self._build_index()

        if _PYGAME_OK:
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self._enabled = True
            print(f"🎵 MusicPlayer ready | songs root: {self.songs_root.resolve()} | language: {language}")
        else:
            print("🎵 MusicPlayer disabled (pygame not available)")

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Scan songs directory and index all MP3 files by mood (+ language)."""
        if not self.songs_root.exists():
            print(f"⚠️  Songs directory not found: {self.songs_root.resolve()}")
            return

        for mood_dir in sorted(self.songs_root.iterdir()):
            if not mood_dir.is_dir():
                continue
            mood = mood_dir.name  # e.g. "happy", "sad"
            files: list[Path] = []

            for lang_dir in mood_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                lang = lang_dir.name  # e.g. "Telugu"
                if self.language != "all" and lang != self.language:
                    continue
                mp3s = list(lang_dir.glob("*.mp3"))
                files.extend(mp3s)

            self._index[mood] = files

        total = sum(len(v) for v in self._index.values())
        print(f"🎵 Indexed {total} songs across {len(self._index)} moods")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, emotion: str):
        """
        Call this every webcam frame with the current detected emotion.
        Internally debounces; only switches song when emotion is stable.
        """
        if not self._enabled:
            return

        now = time.time()

        with self._lock:
            if emotion != self._last_seen_emotion:
                # Emotion just changed — reset the stability timer
                self._last_seen_emotion = emotion
                self._emotion_since = now
                return

            # Same emotion — check if it's been stable long enough
            elapsed = now - self._emotion_since
            if elapsed < self.emotion_hold_secs:
                return

            # Stable long enough — resolve mood
            mood = EMOTION_TO_MOOD.get(emotion)
            if mood is None:
                return

            # Already playing the right mood → keep going
            if mood == self._current_mood and pygame.mixer.music.get_busy():
                return

            # Switch song
            self._play_mood(mood)

    def play_for_emotion(self, emotion: str):
        """
        Immediately play a song for the given emotion (no debounce).
        Useful for single-image mode.
        """
        if not self._enabled:
            return
        mood = EMOTION_TO_MOOD.get(emotion)
        if mood:
            with self._lock:
                self._play_mood(mood)

    def stop(self):
        """Stop playback and fade out."""
        if self._enabled and pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(self.fade_ms)
        with self._lock:
            self._current_mood = None
            self._current_file = None

    def pause(self):
        if self._enabled:
            pygame.mixer.music.pause()

    def resume(self):
        if self._enabled:
            pygame.mixer.music.unpause()

    @property
    def now_playing(self) -> str | None:
        """Human-readable name of the currently loaded song file, or None."""
        if self._current_file is None:
            return None
        stem = Path(self._current_file).stem
        # Convert underscores → spaces for display; strip trailing junk
        parts = stem.split("_")
        # Drop trailing technical tokens (last 1-2 words that look like tags)
        display_parts = []
        skip_tokens = {"official", "audio", "song", "version", "hd", "lyrics"}
        for p in parts:
            if p.lower() in skip_tokens:
                break
            display_parts.append(p)
        return " ".join(display_parts) if display_parts else stem

    # ── Internal ──────────────────────────────────────────────────────────────

    def _play_mood(self, mood: str):
        """Pick a random song for `mood` and start playing. Must hold _lock."""
        songs = self._index.get(mood, [])
        if not songs:
            print(f"⚠️  No songs found for mood '{mood}' (language={self.language})")
            return

        # Pick a different song than what's currently playing
        candidates = [s for s in songs if str(s) != self._current_file]
        if not candidates:
            candidates = songs
        chosen = random.choice(candidates)

        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(self.fade_ms)
                time.sleep(self.fade_ms / 1000.0)

            pygame.mixer.music.load(str(chosen))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()

            self._current_mood = mood
            self._current_file = str(chosen)
            print(f"🎵 Now playing [{mood}]: {chosen.name}")
        except Exception as exc:
            print(f"⚠️  Could not play '{chosen}': {exc}")
