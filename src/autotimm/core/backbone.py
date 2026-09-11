"""Backbone creation and discovery via timm."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import timm
from torch import nn


class ModelSource(str, Enum):
    """Source tag for backbone models.

    Attributes:
        TIMM: Model from the built-in timm library.
        HF_HUB: Model from Hugging Face Hub.
    """

    TIMM = "timm"
    HF_HUB = "hf_hub"


try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


@dataclass
class BackboneConfig:
    """Configuration for a timm backbone.

    Attributes:
        model_name: Name recognized by ``timm.create_model`` (e.g. ``"resnet50"``).
        pretrained: Whether to load pretrained weights.
        num_classes: Set to 0 to get the feature extractor without the
            classification head (the head is provided separately by the task).
        drop_rate: Dropout rate applied inside the model.
        drop_path_rate: Stochastic depth rate.
        extra_kwargs: Forwarded verbatim to ``timm.create_model``.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _is_hf_hub_model(model_name: str) -> bool:
    """Check if a model name refers to a Hugging Face Hub model.

    HF Hub models start with 'hf-hub:', 'hf_hub:', or 'timm/' prefix.
    """
    return model_name.startswith(("hf-hub:", "hf_hub:", "timm/"))


def get_model_source(model_name: str) -> ModelSource:
    """Get the source tag for a model name.

    Determines whether a model comes from the built-in timm library
    or from Hugging Face Hub based on its name/prefix.

    Parameters:
        model_name: The model identifier string.

    Returns:
        ``ModelSource.HF_HUB`` if the model uses a HF Hub prefix
        (``hf-hub:``, ``hf_hub:``, or ``timm/``),
        otherwise ``ModelSource.TIMM``.

    Examples::

        get_model_source("resnet50")  # ModelSource.TIMM
        get_model_source("hf-hub:timm/resnet50.a1_in1k")  # ModelSource.HF_HUB
        get_model_source("timm/vit_base_patch16_224")  # ModelSource.HF_HUB
    """
    if _is_hf_hub_model(model_name):
        return ModelSource.HF_HUB
    return ModelSource.TIMM


def _validate_timm_model_name(model_name: str) -> None:
    """Raise ValueError if *model_name* is not a known timm model.

    HF Hub names are skipped (they are not in ``timm.list_models()``).
    Pretrained-tag suffixes (e.g. ``"resnet50.a1_in1k"``) are stripped
    before checking, since ``timm.list_models()`` returns untagged names.
    """
    if _is_hf_hub_model(model_name):
        return
    base_name = model_name.split(".")[0]
    if base_name not in timm.list_models():
        raise ValueError(
            f"Model '{model_name}' not found in timm. "
            f"Use autotimm.list_backbones('*pattern*') to search, "
            f"or use 'hf-hub:' prefix for Hugging Face Hub models."
        )


def create_backbone(cfg: BackboneConfig | str) -> nn.Module:
    """Create a timm backbone from a config or model name string.

    When *cfg* is a plain string it is treated as a model name with
    ``pretrained=True`` and ``num_classes=0`` (headless).

    Supports both timm models and Hugging Face Hub models:
        - Timm models: Use the model name directly (e.g., ``"resnet50"``).
        - HF Hub models: Use the ``hf-hub:`` prefix (e.g., ``"hf-hub:timm/resnet50.a1_in1k"``).

    Examples::

        create_backbone("resnet50")  # Standard timm model
        create_backbone("hf-hub:timm/resnet50.a1_in1k")  # HF Hub model

    Raises ``ValueError`` if the model name is not found.
    """
    if isinstance(cfg, str):
        cfg = BackboneConfig(model_name=cfg)

    # Skip validation for HF Hub models as they're not in timm.list_models()
    _validate_timm_model_name(cfg.model_name)

    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        **cfg.extra_kwargs,
    )


def get_backbone_out_features(backbone: nn.Module) -> int:
    """Return the number of output features from a headless timm backbone."""
    return backbone.num_features


def list_backbones(
    pattern: str = "",
    pretrained_only: bool = False,
    with_source: bool = False,
) -> list[str] | list[tuple[str, ModelSource]]:
    """List available timm backbones, optionally filtered by glob *pattern*.

    This lists models available in the timm library. To search Hugging Face Hub
    models, use ``list_hf_hub_backbones()``.

    Parameters:
        pattern: Glob pattern to filter model names (e.g., ``"*resnet*"``).
        pretrained_only: If True, only list models with pretrained weights.
        with_source: If True, return tuples of (model_name, ModelSource.TIMM).

    Returns:
        List of model names, or list of (model_name, source) tuples if
        ``with_source=True``.

    Examples::

        list_backbones("*resnet*")
        list_backbones("*efficientnet*", pretrained_only=True)
        list_backbones("*vit*", with_source=True)  # [(name, ModelSource.TIMM), ...]
    """
    models = timm.list_models(pattern or "*", pretrained=pretrained_only)
    if with_source:
        return [(name, ModelSource.TIMM) for name in models]
    return models


def list_hf_hub_backbones(
    author: str = "timm",
    model_name: str | None = None,
    limit: int = 50,
    with_source: bool = False,
) -> list[str] | list[tuple[str, ModelSource]]:
    """List timm-compatible models available on Hugging Face Hub.

    Parameters:
        author: Filter by author/organization (default: ``"timm"`` for official timm models).
        model_name: Optional search query for model names.
        limit: Maximum number of results to return (default: 50).
        with_source: If True, return tuples of (model_name, ModelSource.HF_HUB).

    Returns:
        List of model identifiers in ``hf-hub:`` format that can be used directly
        with ``create_backbone()`` or ``create_feature_backbone()``.
        If ``with_source=True``, returns list of (model_name, source) tuples.

    Examples::

        # List official timm models on HF Hub
        list_hf_hub_backbones()

        # Search for ResNet models
        list_hf_hub_backbones(model_name="resnet")

        # Search models from a specific author
        list_hf_hub_backbones(author="facebook", model_name="convnext")

        # Get models with source tags
        list_hf_hub_backbones(model_name="vit", with_source=True)

    Note:
        Requires ``huggingface_hub`` to be installed.
    """
    if HfApi is None:
        raise ImportError(
            "huggingface_hub is required to list HF Hub models. "
            "Install it with: pip install huggingface_hub"
        )

    api = HfApi()

    # Build search query - combine author and model name
    search_query = None
    if author or model_name:
        parts = []
        if author:
            parts.append(author)
        if model_name:
            parts.append(model_name)
        search_query = " ".join(parts)

    try:
        # Search for models with timm filter
        models = api.list_models(
            filter="timm",
            author=author if author else None,
            search=search_query,
            limit=limit,
            sort="downloads",
        )

        # Format as hf-hub: strings
        result = []
        for model in models:
            model_id = model.id
            model_str = f"hf-hub:{model_id}"
            if with_source:
                result.append((model_str, ModelSource.HF_HUB))
            else:
                result.append(model_str)

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to fetch models from Hugging Face Hub: {e}")


@dataclass
class FeatureBackboneConfig:
    """Configuration for a timm backbone with multi-scale feature extraction.

    Used for tasks like object detection that require feature maps at multiple
    scales (e.g., C2, C3, C4, C5 from ResNet).

    Attributes:
        model_name: Name recognized by ``timm.create_model`` (e.g. ``"resnet50"``).
        pretrained: Whether to load pretrained weights.
        out_indices: Tuple of stage indices to extract features from.
            Default (1, 2, 3, 4) extracts C2, C3, C4, C5 for most architectures.
        drop_rate: Dropout rate applied inside the model.
        drop_path_rate: Stochastic depth rate.
        extra_kwargs: Forwarded verbatim to ``timm.create_model``.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    out_indices: tuple[int, ...] = (1, 2, 3, 4)
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def create_feature_backbone(cfg: FeatureBackboneConfig | str) -> nn.Module:
    """Create a timm backbone for multi-scale feature extraction.

    Uses timm's ``features_only=True`` mode to extract intermediate feature maps,
    typically used for object detection, segmentation, etc.

    When *cfg* is a plain string it is treated as a model name with
    ``pretrained=True`` and default ``out_indices=(1, 2, 3, 4)``.

    Supports both timm models and Hugging Face Hub models:
        - Timm models: Use the model name directly (e.g., ``"resnet50"``).
        - HF Hub models: Use the ``hf-hub:`` prefix (e.g., ``"hf-hub:timm/resnet50.a1_in1k"``).

    Returns a model that outputs a list of feature tensors, one per stage.

    Examples::

        create_feature_backbone("resnet50")  # Standard timm model
        create_feature_backbone("hf-hub:timm/convnext_base.fb_in22k_ft_in1k")  # HF Hub model

    Raises ``ValueError`` if the model name is not found.
    """
    if isinstance(cfg, str):
        cfg = FeatureBackboneConfig(model_name=cfg)

    # Skip validation for HF Hub models as they're not in timm.list_models()
    _validate_timm_model_name(cfg.model_name)

    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        features_only=True,
        out_indices=cfg.out_indices,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        **cfg.extra_kwargs,
    )


def get_feature_info(backbone: nn.Module) -> list[dict[str, Any]]:
    """Return feature info for a backbone created with features_only=True.

    Each dict contains:
        - ``num_chs``: Number of output channels
        - ``reduction``: Spatial reduction factor (stride) relative to input
        - ``module``: Name of the module producing this feature

    Only returns info for features that are actually output (respecting out_indices).

    Raises ``AttributeError`` if the backbone was not created with features_only=True.
    """
    if not hasattr(backbone, "feature_info"):
        raise AttributeError(
            "Backbone does not have 'feature_info'. "
            "Ensure it was created with features_only=True."
        )
    # Use get_dicts() to get info only for the output indices
    return backbone.feature_info.get_dicts()


def get_feature_channels(backbone: nn.Module) -> list[int]:
    """Return the number of channels for each feature level.

    Convenience function that extracts just the channel counts from feature_info.
    Only returns channels for features that are actually output (respecting out_indices).
    """
    if not hasattr(backbone, "feature_info"):
        raise AttributeError(
            "Backbone does not have 'feature_info'. "
            "Ensure it was created with features_only=True."
        )
    return backbone.feature_info.channels()


def get_feature_strides(backbone: nn.Module) -> list[int]:
    """Return the stride (spatial reduction) for each feature level.

    Convenience function that extracts just the strides from feature_info.
    Only returns strides for features that are actually output (respecting out_indices).
    """
    if not hasattr(backbone, "feature_info"):
        raise AttributeError(
            "Backbone does not have 'feature_info'. "
            "Ensure it was created with features_only=True."
        )
    return backbone.feature_info.reduction()
