---
title: Installation Guide
description: Install AutoTimm via pip or from source. Supports Python 3.10-3.13, PyTorch 2.0+, and optional GPU acceleration with CUDA.
---

# Installation

## Requirements

- Python 3.10-3.13
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Install from PyPI

### Core Installation

```bash
pip install autotimm
```

This installs the core package with **all** vision task dependencies:

**Core Dependencies:**

- `torch>=2.0`
- `torchvision>=0.15`
- `timm>=1.0`
- `pytorch-lightning>=2.0`
- `torchmetrics>=1.0`
- `numpy>=1.23`
- `albumentations>=1.3` *(included by default)*
- `opencv-python-headless>=4.8` *(included by default)*
- `pycocotools>=2.0` *(included by default)*
- `huggingface_hub>=0.20`
- `transformers>=4.30` *(included by default)*
- `matplotlib>=3.7` *(included by default)*
- `watermark>=2.3`
- `loguru>=0.7`
- `plotly>=5.0`

!!! note "All tasks supported out of the box"
    The core installation now includes everything needed for classification, detection, segmentation, and instance segmentation tasks. No additional extras needed!

### Optional Dependencies

#### Logger Backends

```bash
# TensorBoard
pip install autotimm[tensorboard]

# Weights & Biases
pip install autotimm[wandb]

# MLflow
pip install autotimm[mlflow]
```

#### Everything

```bash
pip install autotimm[all]
```

Includes all optional dependencies: tensorboard, mlflow, and wandb.

## Development Installation

For contributing or developing locally:

```bash
git clone https://github.com/Noctillio-Ai/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
```

This installs:

- All optional dependencies
- Development tools: pytest, pytest-cov, ruff, black

## Verify Installation

```python
import autotimm as at  # recommended alias

# Check available backbones
print(len(at.list_backbones()))  # 1000+ models

# Check version
print(at.__version__)
```

## Update Notifications

When you `import autotimm`, AutoTimm checks whether a newer release is on PyPI. If yours is out of date, it logs a warning:

```
WARNING | A new version of autotimm is available: 0.7.34 -> 0.7.37. Upgrade with: pip install -U autotimm
```

Upgrade with:

```bash
pip install -U autotimm
```

How the check works:

- **One anonymous request:** a single `GET` to `https://pypi.org/pypi/autotimm/json`. No information about you, your code, or your data is sent.
- **At most once a day:** the latest version is cached in `~/.cache/autotimm/update_check.json` (or `$XDG_CACHE_HOME/autotimm/`) for 24 hours. While the cache is fresh, no request is made.
- **Never blocks or fails:** the lookup runs in a background thread with a short timeout. If you're offline or behind a firewall, it's skipped silently.
- **Shown once per process:** DataLoader workers and non-zero distributed ranks (`RANK` / `LOCAL_RANK`) don't repeat it.
- **Notify only:** AutoTimm never upgrades itself. Pre-releases on PyPI are ignored.

The check is skipped automatically in CI (when `CI` is set), in offline mode (`AUTOTIMM_OFFLINE=1` or `HF_HUB_OFFLINE=1`), and for development installs from source.

To turn it off:

```bash
export AUTOTIMM_DISABLE_UPDATE_CHECK=1
```

To check on demand (waits for the PyPI lookup):

```python
import autotimm as at

at.check_for_updates(blocking=True)
```

## GPU Support

AutoTimm automatically uses GPU when available. Ensure you have CUDA-compatible PyTorch:

```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

## Troubleshooting

For common installation issues, see the [Troubleshooting - Installation](../troubleshooting/environment/installation.md) which covers:

- ImportError: No module named 'timm'
- CUDA out of memory
- Albumentations not found
- And more installation-related issues
