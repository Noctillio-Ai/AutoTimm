"""Tests for loading official-format YOLOX checkpoints."""

import pytest
import torch

from autotimm.models.csp_darknet import build_csp_darknet
from autotimm.models.yolox_pafpn import build_yolox_pafpn
from autotimm.models.yolox_utils import load_yolox_pretrained


def _official_ckpt(model_name: str, path) -> dict:
    """Write a checkpoint in the official Megvii layout and return its tensors.

    Official ``YOLOX.backbone`` is ``YOLOPAFPN`` whose ``.backbone`` is
    ``CSPDarknet``; the head lives under ``head.``.
    """
    torch.manual_seed(0)
    backbone, neck = build_csp_darknet(model_name), build_yolox_pafpn(model_name)
    state = {}
    for k, v in backbone.state_dict().items():
        state[f"backbone.backbone.{k}"] = (
            v.clone().float().uniform_() if v.is_floating_point() else v
        )
    for k, v in neck.state_dict().items():
        if k.startswith(("out_proj_n3.", "out_proj_n4.")):
            continue  # AutoTimm-only layers, absent from official checkpoints
        state[f"backbone.{k}"] = (
            v.clone().float().uniform_() if v.is_floating_point() else v
        )
    state["head.cls_preds.0.weight"] = torch.zeros(80, 128, 1, 1)
    state["head.obj_preds.0.weight"] = torch.zeros(1, 128, 1, 1)
    torch.save({"start_epoch": 300, "model": state}, path)
    return state


def test_loads_backbone_and_neck(tmp_path):
    path = tmp_path / "yolox_tiny.pth"
    state = _official_ckpt("yolox-tiny", path)
    backbone, neck = build_csp_darknet("yolox-tiny"), build_yolox_pafpn("yolox-tiny")

    counts = load_yolox_pretrained(backbone, neck, "yolox-tiny", str(path))

    assert counts["backbone"] == len(backbone.state_dict())
    assert torch.equal(
        backbone.stem.conv.conv.weight, state["backbone.backbone.stem.conv.conv.weight"]
    )
    assert torch.equal(
        neck.C3_p4.conv1.conv.weight, state["backbone.C3_p4.conv1.conv.weight"]
    )


def test_shape_mismatch_raises(tmp_path):
    path = tmp_path / "yolox_s.pth"
    _official_ckpt("yolox-s", path)
    backbone, neck = build_csp_darknet("yolox-tiny"), build_yolox_pafpn("yolox-tiny")
    with pytest.raises(ValueError, match="does not match"):
        load_yolox_pretrained(backbone, neck, "yolox-tiny", str(path))


def test_nano_has_no_official_weights():
    backbone, neck = build_csp_darknet("yolox-nano"), build_yolox_pafpn("yolox-nano")
    with pytest.raises(ValueError, match="depthwise"):
        load_yolox_pretrained(backbone, neck, "yolox-nano", True)


def test_detector_pretrained_path(tmp_path):
    from autotimm import YOLOXDetector

    path = tmp_path / "yolox_tiny.pth"
    state = _official_ckpt("yolox-tiny", path)
    model = YOLOXDetector(
        model_name="yolox-tiny",
        num_classes=3,
        pretrained=str(path),
        compile_model=False,
    )
    assert torch.equal(
        model.backbone.dark5[0].conv.weight,
        state["backbone.backbone.dark5.0.conv.weight"],
    )
    assert model.head is not None
