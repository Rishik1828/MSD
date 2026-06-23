"""
gpu_check.py — Run this to diagnose your GPU/CUDA setup.
Usage: python gpu_check.py
"""
import subprocess
import sys

print("=" * 55)
print("  GPU / CUDA Diagnostic")
print("=" * 55)

# ── PyTorch info ───────────────────────────────────────────
try:
    import torch
    print(f"\n✅ PyTorch version   : {torch.__version__}")
    print(f"   CUDA available    : {torch.cuda.is_available()}")
    print(f"   CUDA version      : {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"\n🎮 GPU Name          : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM (total)      : {total:.1f} GB")
        # Quick tensor test
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        print(f"\n✅ GPU tensor test   : PASSED (shape={y.shape})")
    else:
        print("\n❌ CUDA NOT available.")
        print("   Most likely cause: PyTorch was installed WITHOUT CUDA.")
        print("   Fix → see below.")
except ImportError:
    print("❌ PyTorch is not installed.")
    sys.exit(1)

# ── NVIDIA driver check ────────────────────────────────────
print("\n" + "─" * 55)
print("  nvidia-smi output:")
print("─" * 55)
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print(f"  GPU found: {result.stdout.strip()}")
    else:
        print("  nvidia-smi failed — no NVIDIA GPU or driver not installed.")
except FileNotFoundError:
    print("  nvidia-smi not found — NVIDIA driver may not be installed.")

# ── Fix instructions ───────────────────────────────────────
if not torch.cuda.is_available():
    print("\n" + "=" * 55)
    print("  🔧 FIX: Reinstall PyTorch WITH CUDA support")
    print("=" * 55)
    print("""
  Run ONE of these (pick based on your CUDA version):

  # CUDA 11.8  (most common, works on GTX/RTX 10xx–40xx)
  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  # CUDA 12.1  (newer cards / drivers)
  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  After reinstalling, run this script again to confirm CUDA=True.
""")
else:
    print("\n✅ Everything looks good! GPU training should be fast.")
    print(f"   Expected: ~20–45 min for 50 epochs depending on dataset size.")

print("=" * 55)
