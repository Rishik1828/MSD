"""
evaluate.py
───────────
Comprehensive evaluation of the trained emotion classifier on the test set.

Generates:
  • Per-class precision, recall, F1-score table
  • Overall accuracy, macro F1, weighted F1
  • Confusion matrix heatmap (saved to outputs/)
  • Top misclassified examples with visualisation
  • Classification report (printed + saved as .txt)

Usage:
    cd c:\\Users\\saiha\\Downloads\\MSD
    python src/evaluate.py
    python src/evaluate.py --checkpoint checkpoints/best_emotion_model.pth
"""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders, EMOTION_CLASSES
from model import EmotionClassifier


# ── Prediction collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model on all batches in `loader`.
    Returns:
        all_preds   : np.ndarray (N,)  — predicted class indices
        all_targets : np.ndarray (N,)  — ground-truth class indices
        all_probs   : np.ndarray (N, C) — softmax probabilities
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Evaluating", dynamic_ncols=True):
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs.to(device, non_blocking=True))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(labels.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
    normalise: bool = True,
):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
        title = "Normalised Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=1 if normalise else None,
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {save_path}")


# ── Per-class accuracy bar chart ──────────────────────────────────────────────

def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
):
    """Bar chart of per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    colors = ["#4CAF50" if a >= 0.7 else "#FF9800" if a >= 0.5 else "#F44336"
              for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="white")
    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.6, label="70% target")
    ax.axhline(y=np.mean(per_class_acc), color="navy", linestyle="-.",
               alpha=0.6, label=f"Mean={np.mean(per_class_acc):.2f}")

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Per-Class Accuracy", fontsize=15, fontweight="bold")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Per-class accuracy chart saved → {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(cfg: dict, checkpoint_path: str, project_root: str = "."):
    root   = Path(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  📋 Emotion Detector — Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"{'='*60}\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = EmotionClassifier(
        num_classes  = cfg["data"]["num_classes"],
        backbone     = cfg["model"]["backbone"],
        pretrained   = False,   # weights loaded from checkpoint
        dropout_rate = cfg["model"]["dropout_rate"],
        hidden_dim   = cfg["model"]["hidden_dim"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"✅ Loaded checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # ── DataLoaders (test split only) ─────────────────────────────────────────
    print("\n📂 Loading test dataset…")
    _, _, test_loader, _ = build_dataloaders(cfg, project_root=project_root)

    # ── Collect predictions ────────────────────────────────────────────────────
    print("\nRunning inference on test set…")
    y_pred, y_true, y_probs = collect_predictions(model, test_loader, device)

    # ── Overall metrics ────────────────────────────────────────────────────────
    overall_acc   = accuracy_score(y_true, y_pred)
    macro_f1      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  Overall Accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Weighted F1      : {weighted_f1:.4f}")
    print(f"{'─'*50}")

    # ── Per-class report ───────────────────────────────────────────────────────
    class_names = cfg["data"]["class_names"]
    report = classification_report(
        y_true, y_pred,
        target_names = class_names,
        zero_division = 0,
        digits = 4,
    )
    print(f"\nClassification Report:\n{report}")

    # ── Save report to file ────────────────────────────────────────────────────
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy : {overall_acc:.4f}\n")
        f.write(f"Macro F1         : {macro_f1:.4f}\n")
        f.write(f"Weighted F1      : {weighted_f1:.4f}\n\n")
        f.write(report)
    print(f"  📄 Report saved → {report_path}")

    # ── Visualisations ─────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path = str(out_dir / "confusion_matrix_normalised.png"),
        normalise = True,
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path = str(out_dir / "confusion_matrix_counts.png"),
        normalise = False,
    )
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        save_path = str(out_dir / "per_class_accuracy.png"),
    )

    print(f"\n✅ Evaluation complete. Outputs saved to: {out_dir}")
    return overall_acc, macro_f1


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained emotion classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_emotion_model.pth",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
    )
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

    evaluate(cfg, str(ckpt_path), project_root=str(Path(args.root).resolve()))
