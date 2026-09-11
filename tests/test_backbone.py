"""Tests for backbone creation and discovery."""

import pytest
from torch import nn

from autotimm.core.backbone import (
    BackboneConfig,
    ModelSource,
    create_backbone,
    get_backbone_out_features,
    get_model_source,
    list_backbones,
)


def test_create_backbone_from_string():
    model = create_backbone("resnet18")
    assert isinstance(model, nn.Module)
    assert get_backbone_out_features(model) == 512


def test_create_backbone_from_config():
    cfg = BackboneConfig(model_name="resnet18", pretrained=False, num_classes=0)
    model = create_backbone(cfg)
    assert isinstance(model, nn.Module)
    assert get_backbone_out_features(model) == 512


def test_create_backbone_invalid_name():
    with pytest.raises(ValueError, match="not found in timm"):
        create_backbone("totally_fake_model_xyz")


def test_list_backbones_returns_list():
    models = list_backbones("resnet*")
    assert isinstance(models, list)
    assert len(models) > 0
    assert any("resnet" in m for m in models)


def test_list_backbones_empty_pattern():
    models = list_backbones()
    assert len(models) > 0


def test_model_source_enum_values():
    assert ModelSource.TIMM.value == "timm"
    assert ModelSource.HF_HUB.value == "hf_hub"


def test_get_model_source_timm():
    assert get_model_source("resnet50") == ModelSource.TIMM
    assert get_model_source("efficientnet_b0") == ModelSource.TIMM
    assert get_model_source("vit_base_patch16_224") == ModelSource.TIMM


def test_get_model_source_hf_hub():
    assert get_model_source("hf-hub:timm/resnet50.a1_in1k") == ModelSource.HF_HUB
    assert get_model_source("hf_hub:timm/resnet50.a1_in1k") == ModelSource.HF_HUB
    assert get_model_source("timm/resnet50.a1_in1k") == ModelSource.HF_HUB


def test_list_backbones_with_source():
    models = list_backbones("resnet18", with_source=True)
    assert isinstance(models, list)
    assert len(models) > 0
    for name, source in models:
        assert isinstance(name, str)
        assert source == ModelSource.TIMM
