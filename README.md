# Emotion Detector — Facial Expression Recognition with PyTorch

A GPU-accelerated 8-class facial emotion detector trained on the **AffectNet** dataset using **EfficientNet-B2** transfer learning.

## 🎭 Emotion Classes

| ID | Emotion   | ID | Emotion  |
|----|-----------|-----|----------|
| 0  | Anger     | 4  | Happy    |
| 1  | Contempt  | 5  | Neutral  |
| 2  | Disgust   | 6  | Sad      |
| 3  | Fear      | 7  | Surprise |

---

## 📁 Project Structure

```
MSD/
├── YOLO_format/           # Dataset (images + YOLO labels)
│   ├── train/
│   ├── valid/
│   └── test/
├── src/
│   ├── dataset.py         # Custom DataLoader with face crop + augmentation
│   ├── model.py           # EfficientNet-B2 + custom head
│   ├── train.py           # GPU training loop
│   ├── evaluate.py        # Metrics, confusion matrix, report
│   └── inference.py       # Single image / real-time webcam
├── configs/
│   └── config.yaml        # All hyperparameters
├── checkpoints/           # Saved model weights (auto-created)
├── logs/                  # TensorBoard logs (auto-created)
├── outputs/               # Evaluation charts (auto-created)
└── requirements.txt
```

---

## ⚡ Setup

### 1. Install PyTorch with CUDA (recommended)

```powershell
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (slow for training)
pip install torch torchvision torchaudio
```

### 2. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify GPU

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 🚀 Training

```powershell
cd c:\Users\saiha\Downloads\MSD

# Start training (uses configs/config.yaml)
python src/train.py

# Custom config location
python src/train.py --config configs/config.yaml --root .
```

**Monitor with TensorBoard:**
```powershell
tensorboard --logdir logs
# Open http://localhost:6006 in browser
```

### Key training settings (in `configs/config.yaml`)

| Setting | Default | Notes |
|---------|---------|-------|
| `batch_size` | 64 | Reduce to 32 if CUDA OOM |
| `num_epochs` | 50 | Early stopping at patience=7 |
| `learning_rate` | 1e-4 | AdamW |
| `freeze_backbone_epochs` | 5 | Warmup: head-only |
| `use_amp` | true | FP16 mixed precision |

### Expected training time (50 epochs)

| GPU | VRAM | Time |
|-----|------|------|
| RTX 3080 | 10 GB | ~20 min |
| RTX 3060 | 8 GB | ~30 min |
| GTX 1660 | 6 GB | ~45 min |

---

## 📊 Evaluation

```powershell
python src/evaluate.py

# Custom checkpoint
python src/evaluate.py --checkpoint checkpoints/best_emotion_model.pth
```

Outputs saved to `outputs/`:
- `confusion_matrix_normalised.png`
- `confusion_matrix_counts.png`
- `per_class_accuracy.png`
- `classification_report.txt`

---

## 🔍 Inference

### Single image

```powershell
python src/inference.py --image path/to/face_photo.jpg
```

### Webcam (real-time)

```powershell
python src/inference.py --webcam

# If camera index 0 doesn't work
python src/inference.py --webcam --camera 1
```

Press **Q** or **Esc** to quit webcam mode.

---

## 🧪 Dataset Sanity Check

```powershell
python src/dataset.py
```

Verifies label parsing, bounding box cropping, and DataLoader shape.

---

## 🏗 Model Architecture

```
Input (3×224×224)
    ↓
EfficientNet-B2 (pretrained ImageNet)   ~9.2M params
    ↓ Global Average Pool
    ↓ Dropout(0.3)
    ↓ Linear(1408→512) + GELU + BN
    ↓ Dropout(0.2)
    ↓ Linear(512→8)
    ↓
Logits (8 emotions)
```

---

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| Test accuracy | 65–75% |
| Happy / Neutral F1 | ≥ 0.80 |
| Anger / Disgust F1 | ≥ 0.60 |
| Contempt / Fear F1 | ≥ 0.50 |

> **Note:** Contempt and Fear have fewer training samples in AffectNet — lower F1 is expected for these classes.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` to 32 or 16 in config |
| `num_workers` error on Windows | Set `num_workers: 0` in config |
| No faces detected in webcam | Improve lighting; ensure frontal face view |
| `timm` model not found | `pip install timm --upgrade` |
| Slow training without GPU | Check `torch.cuda.is_available()` returns `True` |
