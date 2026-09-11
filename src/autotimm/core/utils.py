"""Utility helpers for autotimm."""

from __future__ import annotations

import os
import pickle
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Set random seeds for reproducibility across all libraries.

    Seeds Python's random, NumPy, PyTorch (CPU and CUDA), and sets environment
    variables for deterministic behavior.

    Parameters:
        seed: Random seed value. Default is 42.
        deterministic: If ``True``, enables deterministic algorithms in PyTorch.
            This may impact performance but ensures fully reproducible results.
            Default is ``False``.

    Returns:
        The seed value that was set.

    Example:
        >>> seed_everything(42)
        42
        >>> # For fully deterministic training (slower but reproducible)
        >>> seed_everything(42, deterministic=True)
        42

    Note:
        Setting ``deterministic=True`` may reduce performance. Use it only when
        full reproducibility is required (e.g., for research or debugging).
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch backends
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Fallback for older PyTorch versions
            torch.set_deterministic(True)
    else:
        # Enable cuDNN benchmark for faster training (default)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def safe_torch_load(checkpoint_path: str, map_location: Any = "cpu") -> dict:
    """Load a checkpoint, preferring PyTorch's restricted unpickler.

    Tries ``torch.load(..., weights_only=True)`` first, which refuses to
    unpickle anything beyond tensors and a safe list of primitives. Lightning
    checkpoints can contain extra pickled objects (e.g. in ``loops``) that
    this restricted mode rejects, so this falls back to
    ``weights_only=False`` only when the safe load fails — and logs a
    warning when it does, since that path can execute arbitrary code
    embedded in the checkpoint file.

    Parameters:
        checkpoint_path: Path to the ``.ckpt``/``.pt`` file.
        map_location: Passed through to ``torch.load``.

    Returns:
        The deserialized checkpoint dictionary.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        from loguru import logger

        logger.warning(
            f"Loading '{checkpoint_path}' with weights_only=False because the "
            "restricted (safe) unpickler rejected it. Only do this for "
            "checkpoints you trust — this path can execute arbitrary code "
            "embedded in the file."
        )
        return torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of parameters in a model.

    Parameters:
        model: A ``torch.nn.Module``.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def list_optimizers(include_timm: bool = True) -> dict[str, list[str]]:
    """List available optimizers from torch and optionally timm.

    Parameters:
        include_timm: If ``True``, include timm optimizers (requires timm).

    Returns:
        Dictionary with keys ``"torch"`` and optionally ``"timm"``, each containing
        a list of optimizer names.

    Example:
        >>> optimizers = list_optimizers()
        >>> print(optimizers["torch"])
        ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd', ...]
        >>> print(optimizers.get("timm", []))
        ['adabelief', 'adafactor', 'adahessian', 'adamp', ...]
    """
    import inspect
    import torch.optim as torch_optim

    # Dynamically discover PyTorch optimizers
    torch_optimizers = []
    for name, obj in inspect.getmembers(torch_optim):
        if (
            inspect.isclass(obj)
            and issubclass(obj, torch_optim.Optimizer)
            and obj is not torch_optim.Optimizer
        ):
            torch_optimizers.append(name.lower())

    torch_optimizers.sort()
    result = {"torch": torch_optimizers}

    if include_timm:
        try:
            import timm.optim as timm_optim

            # Dynamically discover timm optimizers
            timm_optimizers = []
            for name, obj in inspect.getmembers(timm_optim):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, torch_optim.Optimizer)
                    and obj.__module__.startswith("timm.optim")
                ):
                    # Use lowercase name without 'Optimizer' suffix
                    clean_name = name.replace("Optimizer", "").lower()
                    if clean_name and clean_name not in timm_optimizers:
                        timm_optimizers.append(clean_name)

            timm_optimizers.sort()
            result["timm"] = timm_optimizers
        except ImportError:
            result["timm"] = []

    return result


def list_schedulers(include_timm: bool = True) -> dict[str, list[str]]:
    """List available learning rate schedulers from torch and optionally timm.

    Parameters:
        include_timm: If ``True``, include timm schedulers (requires timm).

    Returns:
        Dictionary with keys ``"torch"`` and optionally ``"timm"``, each containing
        a list of scheduler names.

    Example:
        >>> schedulers = list_schedulers()
        >>> print(schedulers["torch"])
        ['chainedscheduler', 'constantlr', 'cosineannealinglr', ...]
        >>> print(schedulers.get("timm", []))
        ['cosinelrscheduler', 'multisteplrscheduler', ...]
    """
    import inspect
    import torch.optim.lr_scheduler as torch_scheduler

    # Classes to exclude (not actual schedulers)
    exclude_classes = {
        "LRScheduler",
        "_LRScheduler",
        "Optimizer",
        "Counter",
        "Tensor",
        "Any",
        "SupportsFloat",
        "partial",
        "ref",
    }

    # Dynamically discover PyTorch schedulers
    torch_schedulers = []
    for name, obj in inspect.getmembers(torch_scheduler):
        if (
            inspect.isclass(obj)
            and not name.startswith("_")
            and name not in exclude_classes
            and obj.__module__ == "torch.optim.lr_scheduler"
            and (
                # Match common scheduler patterns
                "LR" in name
                or "Scheduler" in name
                or "Cyclic" in name
                or "Annealing" in name
                or "Warm" in name
            )
        ):
            # Convert to lowercase for consistency
            torch_schedulers.append(name.lower())

    torch_schedulers.sort()
    result = {"torch": torch_schedulers}

    if include_timm:
        try:
            import timm.scheduler as timm_scheduler

            # Dynamically discover timm schedulers
            timm_schedulers = []
            for name, obj in inspect.getmembers(timm_scheduler):
                if (
                    inspect.isclass(obj)
                    and hasattr(obj, "step")
                    and obj.__module__.startswith("timm.scheduler")
                    and not name.startswith("_")
                ):
                    # Use lowercase name
                    clean_name = name.lower()
                    if clean_name and clean_name not in timm_schedulers:
                        timm_schedulers.append(clean_name)

            timm_schedulers.sort()
            result["timm"] = timm_schedulers
        except ImportError:
            result["timm"] = []

    return result
