"""
services/emotion_service.py
────────────────────────────
Singleton wrapper around the existing inference pipeline.

The model, face detector, and config are loaded ONCE at app startup
(via init_emotion_service()) and reused across all requests — avoiding
the ~3-second cold-start cost on every detection call.

Public API
──────────
    init_emotion_service(config_path, checkpoint_path)
        Call this from the Flask app factory before serving requests.

    detect_emotion(image_bytes: bytes) -> dict
        Accepts raw image bytes (JPEG/PNG), returns:
        {
            "emotion":     str | None,   # e.g. "Happy"
            "confidence":  float,        # 0.0–1.0
            "all_probs":   list[float],  # one per class, sums to 1.0
            "message":     str,          # human-readable status
        }
"""

import sys
import io
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch

# ── Resolve project root so we can import from src/ ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # MSD/
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from inference import load_model, get_face_detector, detect_faces, predict_face  # noqa: E402


# ── Emotion → DB emotion_tag mapping ─────────────────────────────────────────
# The model outputs 8 classes; the DB emotion_tag column has 7 values.
# Contempt is not in the DB so we map it to 'sad'.
EMOTION_TO_TAG: dict[str, str] = {
    "Happy":    "happy",
    "Sad":      "sad",
    "Anger":    "angry",
    "Neutral":  "neutral",
    "Fear":     "fear",
    "Contempt": "sad",
    "Disgust":  "disgust",
    "Surprise": "surprise",
}

# ── Singleton state ───────────────────────────────────────────────────────────
_model    = None
_detector = None
_device   = None
_cfg      = None


def init_emotion_service(
    config_path: str | Path     = None,
    checkpoint_path: str | Path = None,
) -> None:
    """
    Load model + detector into module-level singletons.
    Must be called once before any detect_emotion() calls.
    """
    global _model, _detector, _device, _cfg

    # Default paths relative to project root
    cfg_path  = Path(config_path)     if config_path     else PROJECT_ROOT / "configs" / "config.yaml"
    ckpt_path = Path(checkpoint_path) if checkpoint_path else PROJECT_ROOT / "checkpoints" / "best_emotion_model.pth"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(cfg_path) as f:
        _cfg = yaml.safe_load(f)

    _device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model    = load_model(_cfg, str(ckpt_path), _device)
    _detector = get_face_detector()

    print(f"✅ EmotionService ready (device={_device})")


def detect_emotion(image_bytes: bytes) -> dict:
    """
    Run face detection + emotion classification on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw JPEG/PNG image data (e.g. from a canvas snapshot).

    Returns
    -------
    dict with keys: emotion, confidence, all_probs, emotion_tag, message
    """
    if _model is None:
        raise RuntimeError("EmotionService not initialised — call init_emotion_service() first.")

    # ── Decode bytes → numpy BGR frame ───────────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "emotion":     None,
            "confidence":  0.0,
            "all_probs":   [],
            "emotion_tag": None,
            "message":     "Could not decode image.",
        }

    # ── Face detection ───────────────────────────────────────────────────────
    faces = detect_faces(frame, _detector)
    if not faces:
        return {
            "emotion":     None,
            "confidence":  0.0,
            "all_probs":   [],
            "emotion_tag": None,
            "message":     "No face detected. Move closer or improve lighting.",
        }

    # Use the largest detected face (by area) for a cleaner result
    x, y, w, h  = max(faces, key=lambda r: r[2] * r[3])
    face_crop   = frame[y : y + h, x : x + w]
    frame_h, frame_w = frame.shape[:2]

    # ── Emotion prediction ───────────────────────────────────────────────────
    emotion, confidence, all_probs = predict_face(face_crop, _model, _device)
    emotion_tag = EMOTION_TO_TAG.get(emotion, "neutral")

    return {
        "emotion":     emotion,
        "confidence":  round(confidence, 4),
        "all_probs":   [round(p, 4) for p in all_probs],
        "emotion_tag": emotion_tag,
        "face_box": {
            "x": round(x / frame_w, 4),
            "y": round(y / frame_h, 4),
            "w": round(w / frame_w, 4),
            "h": round(h / frame_h, 4),
        },
        "message": "ok",
    }
