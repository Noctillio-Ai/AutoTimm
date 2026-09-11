"""Regression tests for bugs fixed in the 0.8.0rc1 quality pass.

Each test pins a bug that previously had no coverage. See CHANGELOG.md
("rc1 quality pass") for the full list.
"""

from __future__ import annotations

import csv
import warnings

import numpy as np
import pytest
import torch
from PIL import Image

import autotimm as at
from autotimm.core.backbone import BackboneConfig, FeatureBackboneConfig

# ---------------------------------------------------------------------------
# Transforms / albumentations 2.x compatibility
# ---------------------------------------------------------------------------


def test_all_presets_construct_without_invalid_arg_warnings():
    """albumentations 2.x silently ignores removed args via UserWarning."""
    from autotimm.data.detection_transforms import get_detection_transforms
    from autotimm.data.segmentation_transforms import (
        get_segmentation_preset,
        instance_segmentation_transforms,
    )
    from autotimm.data.transforms import get_train_transforms

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for preset in ["default", "strong", "light"]:
            get_segmentation_preset(preset, image_size=64)
        instance_segmentation_transforms(64, train=True)
        for preset in ["default", "strong"]:
            get_detection_transforms(preset, 64)
        get_detection_transforms("default", 64, is_train=False)
        for preset in [
            "default",
            "autoaugment",
            "randaugment",
            "trivialaugment",
            "light",
        ]:
            get_train_transforms(preset)
        for preset in ["default", "strong", "light"]:
            get_train_transforms(preset, backend="albumentations")


def test_light_preset_matches_listing():
    """list_transform_presets advertises 'light'; it must be constructible."""
    from autotimm.data.transforms import get_train_transforms

    for backend in ["torchvision", "albumentations"]:
        for preset in at.list_transform_presets(backend=backend):
            get_train_transforms(preset, backend=backend)


def test_segmentation_mask_padding_uses_ignore_index():
    """PadIfNeeded must fill padded mask regions with 255, not class 0."""
    from autotimm.data.segmentation_transforms import segmentation_train_transforms

    t = segmentation_train_transforms(image_size=64)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)  # every real pixel is class 1
    out = t(image=img, mask=mask)
    mask_out = np.asarray(out["mask"])
    # After padding 32->64 there must be ignore_index pixels, and no
    # padded pixel may have silently become class 0.
    assert (mask_out == 255).any()
    assert set(np.unique(mask_out)) <= {1, 255}


# ---------------------------------------------------------------------------
# CSV detection bbox format
# ---------------------------------------------------------------------------


@pytest.fixture()
def det_csv(tmp_path):
    img_dir = tmp_path / "det"
    img_dir.mkdir()
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8)).save(
        img_dir / "img0.png"
    )
    csv_path = img_dir / "train.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "x_min", "y_min", "x_max", "y_max", "label"])
        w.writerow(["img0.png", 10, 10, 40, 40, "cat"])
    return img_dir, csv_path


def test_csv_detection_flip_produces_correct_xyxy(det_csv):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from autotimm.data.detection_dataset import CSVDetectionDataset

    img_dir, csv_path = det_csv
    t = A.Compose(
        [A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )
    ds = CSVDetectionDataset(csv_path=csv_path, image_dir=img_dir, transform=t)
    boxes = ds[0]["boxes"]
    expected = torch.tensor([[60.0, 10.0, 90.0, 40.0]])
    assert torch.allclose(boxes, expected, atol=1e-3)


def test_csv_detection_edge_box_does_not_crash(tmp_path):
    """xyxy boxes near the right edge crashed COCO-format validation."""
    img_dir = tmp_path / "det"
    img_dir.mkdir()
    Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8)).save(
        img_dir / "img0.png"
    )
    csv_path = img_dir / "train.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "x_min", "y_min", "x_max", "y_max", "label"])
        w.writerow(["img0.png", 60, 60, 90, 90, "cat"])

    dm = at.DetectionDataModule(
        train_csv=csv_path, image_dir=img_dir, image_size=100, num_workers=0
    )
    dm.setup("fit")
    sample = dm.train_dataset[0]
    boxes = sample["boxes"]
    assert boxes.shape == (1, 4)
    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3] > boxes[:, 1]).all()


# ---------------------------------------------------------------------------
# ImageDataModule split correctness
# ---------------------------------------------------------------------------


@pytest.fixture()
def folder_dataset(tmp_path):
    root = tmp_path / "cls"
    for cls in ["a", "b"]:
        d = root / "train" / cls
        d.mkdir(parents=True)
        for i in range(6):
            Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8)).save(
                d / f"{i}.png"
            )
    return root


def test_balanced_sampling_with_auto_val_split(folder_dataset):
    dm = at.ImageDataModule(
        data_dir=folder_dataset,
        image_size=32,
        batch_size=4,
        num_workers=0,
        val_split=0.5,
        balanced_sampling=True,
    )
    dm.setup("fit")
    # weights must align with the subset, not the full dataset
    assert len(dm._train_targets) == len(dm.train_dataset)
    for _ in dm.train_dataloader():
        pass


def test_auto_val_split_uses_eval_transforms(folder_dataset):
    dm = at.ImageDataModule(
        data_dir=folder_dataset, image_size=32, num_workers=0, val_split=0.5
    )
    dm.setup("fit")
    val_ds = dm.val_dataset
    inner = val_ds.dataset if hasattr(val_ds, "dataset") else val_ds
    assert inner.transform is dm.eval_transforms
    assert inner.transform is not dm.train_transforms


# ---------------------------------------------------------------------------
# Backbone config handling
# ---------------------------------------------------------------------------


def test_tagged_timm_names_accepted():
    model = at.create_backbone(
        BackboneConfig(model_name="resnet18.a1_in1k", pretrained=False)
    )
    assert model is not None
    with pytest.raises(ValueError):
        at.create_backbone("definitely_not_a_model_name")


def test_image_classifier_honors_backbone_config():
    m = at.ImageClassifier(
        backbone=BackboneConfig(model_name="resnet18", pretrained=False, drop_rate=0.3),
        num_classes=3,
        seed=None,
        deterministic=False,
        compile_model=False,
    )
    assert m.backbone.drop_rate == 0.3
    assert m.hparams["backbone"] == "resnet18"


def test_object_detector_honors_feature_backbone_config():
    m = at.ObjectDetector(
        backbone=FeatureBackboneConfig(model_name="resnet18", pretrained=False),
        num_classes=3,
        seed=None,
        deterministic=False,
        compile_model=False,
    )
    # architecture requires exactly 3 feature levels
    assert len(m.backbone.feature_info.channels()) == 3


# ---------------------------------------------------------------------------
# timm data-config resolution
# ---------------------------------------------------------------------------


def test_string_backbone_resolves_model_specific_stats():
    from autotimm.data.timm_transforms import resolve_backbone_data_config

    cfg = resolve_backbone_data_config("vit_base_patch16_224")
    assert cfg["mean"] == (0.5, 0.5, 0.5)
    assert cfg["std"] == (0.5, 0.5, 0.5)
    assert cfg["crop_pct"] == 0.9

    # tagged and hf-hub ids resolve too
    assert resolve_backbone_data_config("resnet50.a1_in1k")["crop_pct"] == 0.95
    assert (
        resolve_backbone_data_config("hf-hub:timm/resnet50.a1_in1k")["crop_pct"] == 0.95
    )

    # unknown names fall back to ImageNet defaults without raising
    assert resolve_backbone_data_config("not_a_model")["mean"] == (
        0.485,
        0.456,
        0.406,
    )


def test_transform_config_explicit_values_still_win():
    from autotimm.data.timm_transforms import get_transforms_from_backbone

    t = get_transforms_from_backbone(
        "vit_base_patch16_224",
        at.TransformConfig(crop_pct=0.5, image_size=224),
        is_train=False,
    )
    # resize size = image_size / crop_pct = 448 when the override is honored
    assert t.transforms[0].size == 448


# ---------------------------------------------------------------------------
# Optimizers / schedulers
# ---------------------------------------------------------------------------


def _tiny_classifier(**kwargs):
    return at.ImageClassifier(
        backbone="resnet18",
        num_classes=2,
        seed=None,
        deterministic=False,
        compile_model=False,
        **kwargs,
    )


@pytest.mark.parametrize("name", ["adamp", "sgdp", "lamb", "madgrad", "novograd"])
def test_timm_optimizers_create(name):
    m = _tiny_classifier(optimizer=name)
    opt = m._create_optimizer(m.parameters())
    assert isinstance(opt, torch.optim.Optimizer)


def test_plateau_scheduler_accepts_monitor_kwarg():
    m = _tiny_classifier(
        scheduler="plateau", scheduler_kwargs={"monitor": "val/accuracy"}
    )
    opt = m._create_optimizer(m.parameters())
    cfg = m._create_scheduler(opt)
    assert cfg["monitor"] == "val/accuracy"


def test_lr_scheduler_step_supports_timm_schedulers():
    import timm.scheduler as timm_scheduler

    m = _tiny_classifier()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = timm_scheduler.CosineLRScheduler(opt, t_initial=5)
    # Lightning calls lr_scheduler_step(scheduler, metric) — this must not
    # raise even though timm schedulers require step(epoch).
    m.lr_scheduler_step(sched, None)


# ---------------------------------------------------------------------------
# Detection losses
# ---------------------------------------------------------------------------


def test_yolox_iou_loss_zero_for_identical_ltrb():
    m = at.YOLOXDetector(
        model_name="yolox-nano",
        num_classes=3,
        seed=None,
        deterministic=False,
        compile_model=False,
    )
    ltrb = torch.tensor([[5.0, 5.0, 5.0, 5.0], [2.0, 3.0, 4.0, 5.0]])
    assert m._compute_iou_loss(ltrb, ltrb).item() == pytest.approx(0.0, abs=1e-5)


def test_yolox_background_batch_has_cls_gradient():
    m = at.YOLOXDetector(
        model_name="yolox-nano",
        num_classes=3,
        seed=None,
        deterministic=False,
        compile_model=False,
    )
    images = torch.randn(2, 3, 64, 64)
    batch = {
        "image": images,
        "boxes": [torch.zeros(0, 4), torch.zeros(0, 4)],
        "labels": [torch.zeros(0, dtype=torch.long)] * 2,
    }
    loss = m.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.detach().item() > 0.0


def test_object_detector_uses_custom_reg_loss():
    m = at.ObjectDetector(
        backbone=FeatureBackboneConfig(model_name="resnet18", pretrained=False),
        num_classes=2,
        seed=None,
        deterministic=False,
        compile_model=False,
        reg_loss_fn="giou",
    )
    assert m._custom_reg_loss is True
    batch = {
        "image": torch.randn(1, 3, 64, 64),
        "boxes": [torch.tensor([[8.0, 8.0, 40.0, 40.0]])],
        "labels": [torch.tensor([1])],
    }
    loss = m.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_ltrb_to_relative_boxes_roundtrip():
    from autotimm.tasks._fcos_targets import _ltrb_to_relative_boxes

    ltrb = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    boxes = _ltrb_to_relative_boxes(ltrb)
    assert torch.equal(boxes, torch.tensor([[-3.0, -4.0, 5.0, 6.0]]))


# ---------------------------------------------------------------------------
# Semantic segmentation metrics
# ---------------------------------------------------------------------------


def test_semantic_segmentor_metrics_shape_match():
    m = at.SemanticSegmentor(
        backbone=FeatureBackboneConfig(model_name="resnet18", pretrained=False),
        num_classes=3,
        seed=None,
        deterministic=False,
        compile_model=False,
        metrics=[
            at.MetricConfig(
                name="iou",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 3},
                stages=["train"],
            )
        ],
    )
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "mask": torch.randint(0, 3, (2, 64, 64)),
    }
    loss = m.training_step(batch, 0)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# InstanceSegmentationDataModule + transform_config
# ---------------------------------------------------------------------------


def test_instance_transform_config_has_bbox_params():
    from autotimm.data.timm_transforms import get_transforms_from_backbone

    t = get_transforms_from_backbone(
        "resnet18",
        at.TransformConfig(backend="albumentations", image_size=64),
        is_train=True,
        task="instance_segmentation",
    )
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = t(
        image=img,
        masks=[np.ones((64, 64), np.uint8)],
        bboxes=[[5, 5, 30, 30]],
        labels=[0],
    )
    assert out["image"].shape[0] == 3


# ---------------------------------------------------------------------------
# YOLOXDetector dict configs
# ---------------------------------------------------------------------------


def test_yolox_detector_dict_optimizer_and_scheduler():
    m = at.YOLOXDetector(
        model_name="yolox-nano",
        num_classes=2,
        seed=None,
        deterministic=False,
        compile_model=False,
        optimizer={"class": "torch.optim.AdamW", "params": {"lr": 1e-3}},
        scheduler={
            "class": "torch.optim.lr_scheduler.StepLR",
            "params": {"step_size": 5},
        },
    )
    cfg = m.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    assert isinstance(cfg["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
