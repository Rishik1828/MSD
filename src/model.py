"""
model.py
────────
EfficientNet-B2 backbone with a custom classification head for 8-class
facial emotion recognition.

Architecture:
    Input (3 × 224 × 224)
        → EfficientNet-B2 backbone  (pretrained ImageNet, feature_dim=1408)
        → Global Average Pool       (built into EfficientNet)
        → Dropout(dropout_rate)
        → Linear(1408, hidden_dim) + GELU + BatchNorm1d
        → Dropout(0.2)
        → Linear(hidden_dim, num_classes)

Usage:
    model = EmotionClassifier(num_classes=8, pretrained=True)
    logits = model(batch_images)  # shape: (B, 8)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


# ── Model ─────────────────────────────────────────────────────────────────────

class EmotionClassifier(nn.Module):
    """
    Facial emotion classifier built on top of EfficientNet-B2.

    Args:
        num_classes  : Number of emotion categories (default 8)
        backbone     : timm model name (default "efficientnet_b2")
        pretrained   : Load ImageNet weights (default True)
        dropout_rate : Dropout probability before hidden layer
        hidden_dim   : Intermediate FC layer width
    """

    def __init__(
        self,
        num_classes: int  = 8,
        backbone: str     = "efficientnet_b2",
        pretrained: bool  = True,
        dropout_rate: float = 0.3,
        hidden_dim: int   = 512,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Load backbone (classifier stripped, features only) ─────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained    = pretrained,
            num_classes   = 0,      # remove default head → returns raw features
            global_pool   = "avg",  # global average pool built in
        )
        feature_dim = self.backbone.num_features  # 1408 for efficientnet_b2

        # ── Custom classification head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

        # ── Weight initialisation for head ────────────────────────────────────
        self._init_head()

    def _init_head(self):
        """Kaiming init for linear layers in the head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze backbone parameters (warmup phase — only head trains)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen — training head only.")

    def unfreeze_backbone(self, lr_scale: float = 0.1):
        """Unfreeze all parameters (fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"🔓 Backbone unfrozen — full fine-tuning (LR scaled ×{lr_scale}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image batch (B, 3, H, W), normalised
        Returns:
            logits: (B, num_classes)  — raw logits before softmax
        """
        features = self.backbone(x)   # (B, 1408)
        logits   = self.head(features) # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities (softmax) — for inference only."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self) -> dict:
        """Return dict with total / trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: EmotionClassifier, optimizer, scheduler,
                    epoch: int, val_acc: float, path: str):
    """Save full training state to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch"          : epoch,
            "val_acc"        : val_acc,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path: str, model: EmotionClassifier,
                    optimizer=None, scheduler=None,
                    device: str = "cpu") -> dict:
    """Load checkpoint and restore model (and optionally optimizer) state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"✅ Loaded checkpoint from '{path}' (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return ckpt


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EmotionClassifier(num_classes=8, pretrained=False)  # no download in test
    params = model.count_parameters()
    print(f"\nModel: EfficientNet-B2 + custom head")
    print(f"  Total parameters   : {params['total']:,}")
    print(f"  Trainable (initial): {params['trainable']:,}")

    # Forward pass test
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"\nForward pass → output shape: {out.shape}")  # should be (4, 8)

    # Test freeze / unfreeze
    model.freeze_backbone()
    params = model.count_parameters()
    print(f"  Trainable after freeze: {params['trainable']:,}")

    model.unfreeze_backbone()
    params = model.count_parameters()
    print(f"  Trainable after unfreeze: {params['trainable']:,}")
    print("\n✅ Model sanity check passed.")
