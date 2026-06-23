"""
dataset.py
──────────
Custom PyTorch Dataset for AffectNet in YOLO format.

Label format (per line in .txt):
    <class_id> <cx> <cy> <w> <h>   (all values normalised 0–1)

This Dataset:
  1. Pairs each .png image with its matching .txt label file
  2. Reads the YOLO bounding box → crops the face region
  3. Pads the crop to a square (preserves aspect ratio)
  4. Applies train/val augmentation transforms
  5. Returns (face_tensor, class_label)
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Class names (must match data.yaml order) ──────────────────────────────────
EMOTION_CLASSES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]
NUM_CLASSES = len(EMOTION_CLASSES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert normalised YOLO bbox → pixel (x1, y1, x2, y2) coordinates.
    Clamps to image boundaries.
    """
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad a cropped face image to square with black padding.
    Preserves the face's aspect ratio before resizing.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    AffectNet YOLO-format dataset for facial emotion classification.

    Args:
        img_dir  : Path to directory containing .png images
        label_dir: Path to directory containing YOLO .txt labels
        transform: torchvision transform pipeline (train vs. val)
        img_size : Target square size after cropping (default 224)
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
    ):
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size  = img_size

        # ── Pair images with labels ────────────────────────────────────────────
        self.samples: List[Tuple[Path, Path]] = []
        self.labels:  List[int] = []

        img_extensions = {".png", ".jpg", ".jpeg"}
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue
            lbl_path = self.label_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue  # skip images missing a label
            # Parse label: take first (and usually only) annotation
            with open(lbl_path, "r") as f:
                line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id >= NUM_CLASSES:
                continue
            self.samples.append((img_path, lbl_path))
            self.labels.append(cls_id)

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in {img_dir}. "
                "Check that label files exist alongside images."
            )

        print(f"  Loaded {len(self.samples)} samples from {img_dir}")

    # ── Class weight computation ───────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted CrossEntropyLoss.
        Returns a (NUM_CLASSES,) float tensor on CPU.
        """
        counts = Counter(self.labels)
        total  = len(self.labels)
        weights = []
        for i in range(NUM_CLASSES):
            cnt = counts.get(i, 1)   # avoid div-by-zero for missing classes
            weights.append(total / (NUM_CLASSES * cnt))
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum() * NUM_CLASSES   # normalise so mean weight ≈ 1
        return w

    def get_class_distribution(self) -> dict:
        """Returns {class_name: count} for diagnostics."""
        counts = Counter(self.labels)
        return {EMOTION_CLASSES[i]: counts.get(i, 0) for i in range(NUM_CLASSES)}

    # ── Core item loading ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, lbl_path = self.samples[idx]
        cls_id = self.labels[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Fallback: try PIL
            pil_img = Image.open(img_path).convert("RGB")
            img_rgb = np.array(pil_img)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_h, img_w = img_rgb.shape[:2]

        # ── Parse YOLO bounding box ───────────────────────────────────────────
        with open(lbl_path, "r") as f:
            line = f.readline().strip()
        parts   = line.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)

        # ── Crop & pad face region ────────────────────────────────────────────
        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            # Fallback: use entire image if bbox is degenerate
            face_crop = img_rgb

        face_sq = pad_to_square(face_crop)

        # ── Apply transforms ──────────────────────────────────────────────────
        face_pil = Image.fromarray(face_sq)
        if self.transform:
            face_tensor = self.transform(face_pil)
        else:
            face_tensor = transforms.ToTensor()(face_pil)

        return face_tensor, cls_id


# ── Transform factories ───────────────────────────────────────────────────────

# ImageNet normalisation statistics
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Strong augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout-style
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Minimal transforms for validation / test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: dict,
    project_root: str = "."
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders from config dict.

    Returns:
        train_loader, val_loader, test_loader, class_weights (Tensor)
    """
    root     = Path(project_root)
    img_size = cfg["training"]["img_size"]
    bs       = cfg["training"]["batch_size"]
    workers  = cfg["training"]["num_workers"]

    train_ds = EmotionDataset(
        img_dir   = root / cfg["data"]["train_images"],
        label_dir = root / cfg["data"]["train_labels"],
        transform = get_train_transforms(img_size),
        img_size  = img_size,
    )
    val_ds = EmotionDataset(
        img_dir   = root / cfg["data"]["val_images"],
        label_dir = root / cfg["data"]["val_labels"],
        transform = get_val_transforms(img_size),
        img_size  = img_size,
    )
    test_ds = EmotionDataset(
        img_dir   = root / cfg["data"]["test_images"],
        label_dir = root / cfg["data"]["test_labels"],
        transform = get_val_transforms(img_size),
        img_size  = img_size,
    )

    # ── Compute class weights from training set only ───────────────────────────
    class_weights = train_ds.get_class_weights()

    train_loader = DataLoader(
        train_ds,
        batch_size  = bs,
        shuffle     = True,
        num_workers = workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, sys

    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(__file__).parent.parent

    print("Building DataLoaders…")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        cfg, project_root=str(project_root)
    )

    print(f"\nClass weights: {class_weights.tolist()}")

    # Fetch one batch
    imgs, labels = next(iter(train_loader))
    print(f"\nSample batch → images: {imgs.shape}, labels: {labels.shape}")
    print(f"Label distribution in batch: {Counter(labels.tolist())}")
    print("\n✅ Dataset sanity check passed.")
