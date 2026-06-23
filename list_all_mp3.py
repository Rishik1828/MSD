from pathlib import Path

songs_dir = Path("songs")
mp3_files = list(songs_dir.rglob("*.mp3"))

print(f"Total MP3 files found: {len(mp3_files)}")

with open("all_mp3_files.txt", "w", encoding="utf-8") as f:
    f.write(f"Total MP3 files: {len(mp3_files)}\n\n")
    for file in sorted(mp3_files):
        f.write(f"{file.as_posix()}\n")

print("Wrote all MP3 file paths to all_mp3_files.txt")
