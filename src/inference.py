"""
inference.py
────────────
Inference module for the trained facial emotion classifier.

Supports two modes:
  1. Single image  → python src/inference.py --image path/to/face.jpg
  2. Webcam (live) → python src/inference.py --webcam

In both modes:
  • OpenCV Haar Cascade detects the face(s) in each frame
  • Each detected face is cropped, preprocessed, and passed to the model
  • Predicted emotion + confidence bar overlay is rendered
  • (Optional) A matching song plays automatically from songs/<mood>/<lang>/

Usage:
    cd c:\\Users\\saiha\\Downloads\\MSD

    # Single image
    python src/inference.py --image test_face.jpg

    # Webcam (press Q to quit)
    python src/inference.py --webcam

    # Webcam + emotion-based music (all languages)
    python src/inference.py --webcam --music

    # Webcam + music in Telugu only
    python src/inference.py --webcam --music --language Telugu

    # Custom checkpoint
    python src/inference.py --webcam --checkpoint checkpoints/best_emotion_model.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from torchvision import transforms
from PIL import Image

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import EMOTION_CLASSES, pad_to_square
from model import EmotionClassifier
from music_player import MusicPlayer


# ── Emotion colour palette ────────────────────────────────────────────────────
# BGR colours for each emotion class
EMOTION_COLORS = {
    "Anger"   : (0,   0,   220),   # Red
    "Contempt": (0,   128, 200),   # Orange-ish
    "Disgust" : (0,   200, 100),   # Yellow-green
    "Fear"    : (180, 0,   180),   # Purple
    "Happy"   : (0,   200, 0),     # Green
    "Neutral" : (180, 180, 180),   # Gray
    "Sad"     : (200, 100, 0),     # Blue
    "Surprise": (0,   220, 220),   # Cyan
}


# ── Preprocessing transform (val/test style, no augmentation) ─────────────────
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Face detector (Haar Cascade — CPU, fast, no extra deps) ───────────────────

def get_face_detector():
    """Load OpenCV's pre-trained Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(
            "Could not load Haar Cascade. "
            "Ensure opencv-python is properly installed."
        )
    return detector


def detect_faces(frame_bgr: np.ndarray, detector) -> list:
    """
    Detect faces in a BGR frame.
    Returns list of (x, y, w, h) rectangles.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (48, 48),
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return faces.tolist()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> EmotionClassifier:
    """Load trained EmotionClassifier from a checkpoint."""
    model = EmotionClassifier(
        num_classes  = cfg["data"]["num_classes"],
        backbone     = cfg["model"]["backbone"],
        pretrained   = False,
        dropout_rate = cfg["model"]["dropout_rate"],
        hidden_dim   = cfg["model"]["hidden_dim"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded from '{checkpoint_path}' (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model


# ── Single-face prediction ────────────────────────────────────────────────────

@torch.no_grad()
def predict_face(
    face_bgr: np.ndarray,
    model: EmotionClassifier,
    device: torch.device,
) -> tuple:
    """
    Predict emotion for a single face crop (BGR numpy array).
    Returns (emotion_label: str, confidence: float, all_probs: list)
    """
    # BGR → RGB → PIL → pad → transform
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_sq  = pad_to_square(face_rgb)
    pil_img  = Image.fromarray(face_sq)
    tensor   = _INFER_TRANSFORM(pil_img).unsqueeze(0).to(device)  # (1,3,224,224)

    probs = model.predict(tensor)[0].cpu().numpy()   # (8,)
    pred_idx = int(np.argmax(probs))
    return EMOTION_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()


# ── Frame overlay ─────────────────────────────────────────────────────────────

def draw_emotion_overlay(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    emotion: str,
    confidence: float,
    all_probs: list,
):
    """
    Draw bounding box, emotion label, confidence bar, and probability bars
    onto `frame` (in-place).
    """
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ── Emotion label + confidence ────────────────────────────────────────────
    label = f"{emotion}  {confidence*100:.1f}%"
    label_y = y - 12 if y > 30 else y + h + 25
    cv2.putText(frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # ── Mini probability bar chart (top-right corner of face box) ─────────────
    bar_x = x + w + 10
    bar_y = y
    bar_w = 120
    bar_h = 14

    if bar_x + bar_w + 60 < frame.shape[1]:   # only draw if it fits
        for i, (cls, prob) in enumerate(zip(EMOTION_CLASSES, all_probs)):
            cx = EMOTION_COLORS.get(cls, (200, 200, 200))
            filled = int(bar_w * prob)
            row_y  = bar_y + i * (bar_h + 2)
            # Background
            cv2.rectangle(frame, (bar_x, row_y), (bar_x + bar_w, row_y + bar_h),
                          (50, 50, 50), -1)
            # Filled
            cv2.rectangle(frame, (bar_x, row_y), (bar_x + filled, row_y + bar_h),
                          cx, -1)
            # Label
            cv2.putText(frame, f"{cls[:3]} {prob*100:4.1f}%",
                        (bar_x + bar_w + 4, row_y + bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


# ── Single image mode ─────────────────────────────────────────────────────────

def run_image(image_path: str, model, detector, device, player: MusicPlayer | None = None):
    """Run emotion detection on a single image file and show the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        return

    faces = detect_faces(frame, detector)
    if not faces:
        print("⚠️  No faces detected. Try with a clearer frontal face image.")
    else:
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            emotion, conf, probs = predict_face(face_crop, model, device)
            print(f"  Detected: {emotion} ({conf*100:.1f}%)")
            draw_emotion_overlay(frame, x, y, w, h, emotion, conf, probs)
            if player is not None:
                player.play_for_emotion(emotion)
                if player.now_playing:
                    print(f"  🎵 Playing: {player.now_playing}")

    out_path = str(Path(image_path).stem) + "_emotion_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"  Result saved → {out_path}")

    cv2.imshow("Emotion Detection", frame)
    print("  Press any key to close…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if player is not None:
        player.stop()


# ── Webcam mode ───────────────────────────────────────────────────────────────

def _draw_now_playing(frame: np.ndarray, text: str):
    """
    Draw a semi-transparent 'Now Playing' bar at the bottom of the frame.
    """
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    note = "\u266b"
    label = f"{note}  {text}"
    cv2.putText(frame, label, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 2, cv2.LINE_AA)


def run_webcam(model, detector, device, player: MusicPlayer | None = None):
    """Real-time emotion detection from webcam. Press Q to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam (index 0). Try --camera 1 or --camera 2.")
        return

    print("\n🎥 Webcam started. Press Q to quit.\n")
    fps_history = []
    dominant_emotion: str | None = None

    import time
    while True:
        t0  = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            emotion, conf, probs = predict_face(face_crop, model, device)
            draw_emotion_overlay(frame, x, y, w, h, emotion, conf, probs)
            dominant_emotion = emotion   # last face wins (single-face typical)

        # ── Music update ──────────────────────────────────────────────────────
        if player is not None and dominant_emotion is not None:
            player.update(dominant_emotion)

        # ── Now-playing overlay ────────────────────────────────────────────────
        if player is not None and player.now_playing:
            _draw_now_playing(frame, player.now_playing)

        # ── FPS counter ────────────────────────────────────────────────────────
        dt = time.time() - t0
        fps_history.append(1.0 / dt if dt > 0 else 0)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = np.mean(fps_history)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detector (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if player is not None:
        player.stop()
    print("\n\U0001f44b Webcam closed.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion detector inference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Path to a single image file")
    mode.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--config",       type=str,   default="configs/config.yaml")
    parser.add_argument("--checkpoint",   type=str,   default="checkpoints/best_emotion_model.pth")
    parser.add_argument("--root",         type=str,   default=".")
    parser.add_argument("--camera",       type=int,   default=0,    help="Webcam device index")
    # ── Music options ──────────────────────────────────────────────────────────
    parser.add_argument("--music",        action="store_true",       help="Enable emotion-based music playback")
    parser.add_argument("--language",     type=str,   default="all",
                        choices=["Telugu", "Hindi", "English", "all"],
                        help="Song language filter (default: all)")
    parser.add_argument("--emotion-hold", type=float, default=2.0,
                        help="Seconds emotion must be stable before song switches (default: 2.0)")
    parser.add_argument("--volume",       type=float, default=0.85,
                        help="Music volume 0.0-1.0 (default: 0.85)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = Path(args.root) / args.config
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent.parent / args.config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.root) / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = Path(__file__).parent.parent / args.checkpoint

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_model(cfg, str(ckpt_path), device)
    detector = get_face_detector()

    # ── Music player (optional) ────────────────────────────────────────────────
    player: MusicPlayer | None = None
    if args.music:
        # Resolve songs/ relative to project root (one level up from src/)
        project_root = Path(__file__).parent.parent
        songs_root   = project_root / "songs"
        player = MusicPlayer(
            songs_root        = songs_root,
            language          = args.language,
            emotion_hold_secs = args.emotion_hold,
            volume            = args.volume,
        )

    if args.image:
        run_image(args.image, model, detector, device, player)
    elif args.webcam:
        run_webcam(model, detector, device, player)
