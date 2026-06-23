"""
train.py
────────
Full GPU-accelerated training loop for the facial emotion classifier.

Features:
  ✓ Automatic CUDA device selection with CPU fallback
  ✓ Mixed-precision training (FP16) via torch.amp
  ✓ 2-phase training: backbone frozen warmup → full fine-tune
  ✓ AdamW optimizer + CosineAnnealingLR scheduler
  ✓ Weighted CrossEntropyLoss for class imbalance
  ✓ Label smoothing to reduce overconfidence
  ✓ Gradient clipping to prevent exploding gradients
  ✓ Early stopping on validation loss
  ✓ Best model checkpoint saved automatically
  ✓ TensorBoard logging (loss, accuracy, LR)
  ✓ Live tqdm progress bars

Usage:
    cd c:\\Users\\saiha\\Downloads\\MSD
    python src/train.py
    python src/train.py --config configs/config.yaml

Monitor training:
    tensorboard --logdir logs
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders, EMOTION_CLASSES
from model import EmotionClassifier, save_checkpoint


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (slightly slower — comment out for max speed)
    # torch.backends.cudnn.deterministic = True


# ── Metrics helpers ───────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, clip_norm, use_amp
):
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward (with optional FP16 autocast) ─────────────────────────────
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        # ── Backward ──────────────────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # ── Running metrics ────────────────────────────────────────────────────
        acc = accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, split_name="Val"):
    """Evaluate on val or test set. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"  {split_name}", leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        acc = accuracy(logits, labels)
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg: dict, project_root: str = "."):
    root = Path(project_root)

    # ── Setup ──────────────────────────────────────────────────────────────────
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  🚀 Emotion Detector — Training")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model  : {cfg['model']['backbone']}")
    print(f"  Classes: {cfg['data']['num_classes']}")
    print(f"  Epochs : {cfg['training']['num_epochs']}")
    print(f"  Batch  : {cfg['training']['batch_size']}")
    print(f"  AMP    : {cfg['training']['use_amp']}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    print("📂 Loading datasets…")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        cfg, project_root=project_root
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n🏗  Building model…")
    model = EmotionClassifier(
        num_classes  = cfg["data"]["num_classes"],
        backbone     = cfg["model"]["backbone"],
        pretrained   = cfg["model"]["pretrained"],
        dropout_rate = cfg["model"]["dropout_rate"],
        hidden_dim   = cfg["model"]["hidden_dim"],
    ).to(device)

    params = model.count_parameters()
    print(f"   Parameters — total: {params['total']:,} | trainable: {params['trainable']:,}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    if cfg["training"]["use_class_weights"]:
        cw = class_weights.to(device)
        print(f"   Class weights: {[f'{w:.3f}' for w in cw.cpu().tolist()]}")
    else:
        cw = None

    criterion = nn.CrossEntropyLoss(
        weight          = cw,
        label_smoothing = cfg["training"]["label_smoothing"],
    )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # ── Scheduler ──────────────────────────────────────────────────────────────
    n_epochs = cfg["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = n_epochs,
        eta_min = cfg["training"]["eta_min"],
    )

    # ── Mixed precision scaler (PyTorch 2.x compatible) ──────────────────────
    use_amp = cfg["training"]["use_amp"] and (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_dir = root / cfg["output"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(log_dir))

    ckpt_dir = root / cfg["output"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / cfg["output"]["best_model_name"]

    # ── Early stopping state ───────────────────────────────────────────────────
    patience        = cfg["training"]["early_stopping_patience"]
    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    no_improve_cnt  = 0
    freeze_epochs   = cfg["training"]["freeze_backbone_epochs"]

    # ── Warmup: freeze backbone initially ────────────────────────────────────
    model.freeze_backbone()

    print(f"\n{'─'*60}")
    print(f"  Starting training for {n_epochs} epochs…")
    print(f"  Phase 1 (epochs 1–{freeze_epochs}): Head-only warmup")
    print(f"  Phase 2 (epochs {freeze_epochs+1}+): Full fine-tuning")
    print(f"{'─'*60}\n")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss"  : [], "val_acc"  : [],
    }

    total_start = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # ── Unfreeze backbone at the right epoch ──────────────────────────────
        if epoch == freeze_epochs + 1:
            print(f"\n  ⟶  Epoch {epoch}: Unfreezing backbone for full fine-tuning…")
            model.unfreeze_backbone()
            # Reset optimizer with a lower LR for the backbone
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["training"]["learning_rate"] * 0.1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:3d}/{n_epochs}]  LR={current_lr:.2e}")

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, cfg["training"]["grad_clip_norm"], use_amp
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_amp, split_name="Val"
        )

        scheduler.step()

        # ── Log metrics ───────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"[{epoch_time:.1f}s]"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            no_improve_cnt = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, str(best_ckpt))
            print(f"  💾 Best model saved! val_acc={best_val_acc:.4f}")
        else:
            no_improve_cnt += 1

        # ── Periodic checkpoint ────────────────────────────────────────────────
        save_every = cfg["output"].get("save_every_n_epochs", 5)
        if epoch % save_every == 0:
            periodic_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, str(periodic_path))

        # ── Early stopping ─────────────────────────────────────────────────────
        if no_improve_cnt >= patience:
            print(f"\n  🛑 Early stopping triggered (no improvement for {patience} epochs)")
            break

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete in {total_time/60:.1f} min")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Best val loss     : {best_val_loss:.4f}")
    print(f"  Best checkpoint   : {best_ckpt}")
    print(f"{'='*60}")

    writer.close()

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n📊 Loading best model for test evaluation…")
    best_state = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, use_amp, split_name="Test"
    )
    print(f"  Test loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    return model, history


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train facial emotion classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = Path(args.root) / args.config
    if not cfg_path.exists():
        # Try relative to script location
        cfg_path = Path(__file__).parent.parent / args.config

    print(f"Loading config from: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    project_root = str(Path(args.root).resolve())
    train(cfg, project_root=project_root)
