"""Push a trained model checkpoint to Hugging Face Hub.

Reads configuration from the ``HF_PUSH_CFG`` environment variable (JSON)
or from CLI arguments. Strips optimizer state and sensitive paths before
uploading, and generates a model card.

Usage::

    python -m autotimm.flow.push_to_hub --config '{"repo_id": "...", ...}'
    # or via env var (used by NightFlow):
    HF_PUSH_CFG='{"repo_id": "...", ...}' python -m autotimm.flow.push_to_hub
    # or (token read from HF_TOKEN env var to avoid leaking it into shell history):
    HF_TOKEN=hf_... autotimm-flow push-to-hub --repo-id user/model --ckpt-dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def push(cfg: dict) -> dict:
    """Push model to HuggingFace Hub. Returns ``{"success": True, "url": ...}`` or ``{"error": ...}``."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return {
            "error": "huggingface_hub is not installed. Run: pip install huggingface_hub"
        }

    repo_id = cfg["repo_id"]
    token = cfg["token"]
    ckpt_dir = cfg["ckpt_dir"]
    hparams_path = cfg.get("hparams_path", "")
    task_type = cfg.get("task_type", "Classification")
    backbone = cfg.get("backbone", "unknown")
    num_classes = cfg.get("num_classes", 10)
    image_size = cfg.get("image_size", 224)
    acc_str = cfg.get("acc_str", "N/A")
    test_acc_str = cfg.get("test_acc_str", "N/A")
    is_private = cfg.get("is_private", False)
    model_name = cfg.get("model_name", f"{backbone} — {task_type}")
    description = cfg.get("description", "")
    license_id = cfg.get("license", "apache-2.0")
    user_tags = [t.strip() for t in cfg.get("tags", "").split(",") if t.strip()]

    # Find checkpoint
    ckpt_file = None
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".ckpt"):
                ckpt_file = os.path.join(ckpt_dir, f)

    if not ckpt_file:
        return {
            "error": "No checkpoint file found. Training may not have saved a model yet."
        }

    api = HfApi(token=token)

    # Create repo (ok if exists)
    try:
        create_repo(repo_id, token=token, exist_ok=True, private=is_private)
    except Exception as e:
        return {"error": f"Failed to create repository: {e}"}

    # Strip optimizer state from checkpoint to reduce size
    import torch

    from autotimm.core.utils import safe_torch_load

    clean_ckpt = None
    try:
        raw = safe_torch_load(ckpt_file, map_location="cpu")
        if isinstance(raw, dict):
            keep_keys = {
                "state_dict",
                "hyper_parameters",
                "pytorch-lightning_version",
                "epoch",
                "global_step",
            }
            cleaned = {k: v for k, v in raw.items() if k in keep_keys}
            if cleaned.get("hyper_parameters") and isinstance(
                cleaned["hyper_parameters"], dict
            ):
                sensitive = {
                    "data_dir",
                    "data_directory",
                    "project_path",
                    "log_dir",
                    "default_root_dir",
                    "output_dir",
                    "root_dir",
                    "save_dir",
                    "resume_from_checkpoint",
                    "ssh_command",
                    "ssh_key",
                }
                cleaned["hyper_parameters"] = {
                    k: v
                    for k, v in cleaned["hyper_parameters"].items()
                    if k not in sensitive
                    and not (isinstance(v, str) and ("/" in v or "\\" in v))
                }
            with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as _tmp:
                clean_ckpt = _tmp.name
            torch.save(cleaned, clean_ckpt)
    except Exception:
        clean_ckpt = None

    upload_ckpt = clean_ckpt or ckpt_file

    # Generate model card
    all_tags = list(dict.fromkeys(["autotimm"] + user_tags))
    tags_yaml = "\n".join(f"- {t}" for t in all_tags)
    desc_line = f"\n\n{description}\n" if description else "\n"
    model_card = f"""---
library_name: pytorch
license: {license_id}
tags:
{tags_yaml}
datasets: []
metrics:
- accuracy
---

# {model_name}
{desc_line}
Trained using the **AutoTimm** framework.

## Model Details

| Property | Value |
|----------|-------|
| **Backbone** | {backbone} |
| **Task** | {task_type} |
| **Image Size** | {image_size}x{image_size} |
| **Num Classes** | {num_classes} |
| **Val Accuracy** | {acc_str} |
| **Test Accuracy** | {test_acc_str} |

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load("pytorch_model.ckpt")
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize(({image_size}, {image_size})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
```

## Training

Trained with PyTorch Lightning + timm via AutoTimm.
"""

    # Sanitize hparams — strip local paths and sensitive keys before uploading
    clean_hparams_path = None
    if hparams_path and os.path.isfile(hparams_path):
        try:
            import yaml

            with open(hparams_path) as f:
                hp = yaml.safe_load(f) or {}
            if isinstance(hp, dict):
                sensitive = {
                    "data_dir",
                    "data_directory",
                    "project_path",
                    "log_dir",
                    "default_root_dir",
                    "output_dir",
                    "root_dir",
                    "save_dir",
                    "resume_from_checkpoint",
                    "ssh_command",
                    "ssh_key",
                    "callbacks",
                    "logger",
                    "profiler",
                }
                hp = {
                    k: v
                    for k, v in hp.items()
                    if k not in sensitive
                    and not (isinstance(v, str) and (os.sep in v or "/" in v))
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as _htmp:
                    yaml.dump(hp, _htmp, default_flow_style=False)
                clean_hparams_path = _htmp.name
        except Exception:
            pass

    # Upload files
    try:
        api.upload_file(
            path_or_fileobj=upload_ckpt,
            path_in_repo="pytorch_model.ckpt",
            repo_id=repo_id,
            token=token,
        )

        if clean_hparams_path:
            api.upload_file(
                path_or_fileobj=clean_hparams_path,
                path_in_repo="hparams.yaml",
                repo_id=repo_id,
                token=token,
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(model_card)
        api.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        os.unlink(tmp.name)

        url = f"https://huggingface.co/{repo_id}"
        return {"success": True, "url": url}
    except Exception as e:
        return {"error": f"Upload failed: {e}"}
    finally:
        if clean_ckpt and os.path.isfile(clean_ckpt):
            os.unlink(clean_ckpt)
        if clean_hparams_path and os.path.isfile(clean_hparams_path):
            os.unlink(clean_hparams_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--config",
        default=None,
        help="JSON config string (alternative to HF_PUSH_CFG env var)",
    )
    parser.add_argument("--repo-id", default=None, help="HuggingFace repo ID")
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "HuggingFace token. Prefer setting the HF_TOKEN env var instead — "
            "a token passed on the command line is visible in shell history "
            "and process listings."
        ),
    )
    parser.add_argument("--ckpt-dir", default=None, help="Checkpoint directory")
    parser.add_argument("--hparams-path", default=None, help="Path to hparams.yaml")
    parser.add_argument("--task-type", default="Classification")
    parser.add_argument("--backbone", default="unknown")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--acc-str", default="N/A")
    parser.add_argument("--test-acc-str", default="N/A")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--description", default="")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--tags", default="")
    args = parser.parse_args()

    # Priority: --config flag > HF_PUSH_CFG env var > individual CLI args
    if args.config:
        cfg = json.loads(args.config)
    elif "HF_PUSH_CFG" in os.environ:
        cfg = json.loads(os.environ["HF_PUSH_CFG"])
    elif args.repo_id and (args.token or os.environ.get("HF_TOKEN")) and args.ckpt_dir:
        cfg = {
            "repo_id": args.repo_id,
            "token": args.token or os.environ.get("HF_TOKEN"),
            "ckpt_dir": args.ckpt_dir,
            "hparams_path": args.hparams_path or "",
            "task_type": args.task_type,
            "backbone": args.backbone,
            "num_classes": args.num_classes,
            "image_size": args.image_size,
            "acc_str": args.acc_str,
            "test_acc_str": args.test_acc_str,
            "is_private": args.private,
            "model_name": args.model_name or f"{args.backbone} — {args.task_type}",
            "description": args.description,
            "license": args.license,
            "tags": args.tags,
        }
    else:
        print(
            json.dumps(
                {
                    "error": "Provide --config, set HF_PUSH_CFG env var, or pass --repo-id --token --ckpt-dir"
                }
            )
        )
        sys.exit(1)

    result = push(cfg)
    print(json.dumps(result))
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
