"""Tests for backbone freezing (BackboneFreezeMixin) across task models."""

import pytest
import torch

from autotimm.core.backbone import BackboneConfig
from autotimm.tasks.classification import ImageClassifier
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.tasks.semantic_segmentation import SemanticSegmentor


def _vit_cfg():
    return BackboneConfig(
        model_name="vit_tiny_patch16_224", pretrained=False, drop_path_rate=0.1
    )


def _resnet_cfg():
    return BackboneConfig(model_name="resnet18", pretrained=False)


def _bn_layers(module):
    return [m for m in module.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def test_vit_classifier_frozen_backbone_not_updated():
    model = ImageClassifier(
        backbone=_vit_cfg(),
        num_classes=3,
        freeze_backbone=True,
        compile_model=False,
        metrics=None,
        scheduler=None,
        lr=1.0,
    )
    assert model.freeze_backbone is True
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())

    model.train()
    # Frozen backbone stays in eval mode (no drop_path / dropout).
    assert not model.backbone.training
    assert model.head.training

    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    opt = model.configure_optimizers()["optimizer"]
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    for group in opt.param_groups:
        assert not any(id(p) in backbone_ids for p in group["params"])

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 3, (2,))
    loss = model.training_step((x, y), batch_idx=0)
    loss.backward()
    opt.step()

    for k, v in model.backbone.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_frozen_backbone_batchnorm_stats_unchanged():
    model = ImageClassifier(
        backbone=_resnet_cfg(),
        num_classes=3,
        freeze_backbone=True,
        compile_model=False,
        metrics=None,
    )
    model.train()
    bn = _bn_layers(model.backbone)[0]
    mean_before = bn.running_mean.clone()
    model(torch.randn(2, 3, 64, 64) * 5 + 3)
    assert torch.equal(bn.running_mean, mean_before)


def test_unfreeze_and_refreeze():
    model = ImageClassifier(
        backbone=_resnet_cfg(),
        num_classes=3,
        freeze_backbone=True,
        compile_model=False,
        metrics=None,
    )
    model.train()
    model.unfreeze_backbone()
    assert model.freeze_backbone is False
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert model.backbone.training

    model.freeze_backbone = True
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert not model.backbone.training
    model.train()
    assert not model.backbone.training


def test_unfrozen_by_default():
    model = ImageClassifier(
        backbone=_resnet_cfg(), num_classes=3, compile_model=False, metrics=None
    )
    assert model.freeze_backbone is False
    model.train()
    assert model.backbone.training
    assert all(p.requires_grad for p in model.backbone.parameters())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ObjectDetector(
            backbone="resnet18",
            num_classes=3,
            freeze_backbone=True,
            compile_model=False,
            metrics=None,
        ),
        lambda: SemanticSegmentor(
            backbone="resnet18",
            num_classes=3,
            head_type="fcn",
            freeze_backbone=True,
            compile_model=False,
            metrics=None,
        ),
    ],
    ids=["detector", "segmentor"],
)
def test_other_tasks_keep_frozen_backbone_in_eval(factory):
    model = factory()
    model.train()
    assert model.freeze_backbone is True
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
