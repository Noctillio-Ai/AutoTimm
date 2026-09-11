"""Tests for TransformConfig, timm_transforms, and model preprocessing."""

import numpy as np
import pytest
import torch
from PIL import Image

from autotimm.core.metrics import MetricConfig
from autotimm.data.timm_transforms import (
    create_inference_transform,
    get_transforms_from_backbone,
    resolve_backbone_data_config,
)
from autotimm.data.transform_config import TransformConfig
from autotimm.tasks.classification import ImageClassifier


class TestTransformConfig:
    """Tests for TransformConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransformConfig()
        assert config.preset == "default"
        assert config.backend == "torchvision"
        assert config.image_size == 224
        assert config.use_timm_config is True
        assert config.mean is None
        assert config.std is None
        # None means "use the model's pretrained value" (falls back to
        # bicubic / 0.875 when no model config is available)
        assert config.interpolation is None
        assert config.crop_pct is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransformConfig(
            preset="randaugment",
            image_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        assert config.preset == "randaugment"
        assert config.image_size == 384
        assert config.mean == (0.5, 0.5, 0.5)
        assert config.std == (0.5, 0.5, 0.5)

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            TransformConfig(backend="invalid")

    def test_invalid_image_size(self):
        """Test that non-positive image size raises ValueError."""
        with pytest.raises(ValueError, match="image_size must be positive"):
            TransformConfig(image_size=0)

    def test_invalid_crop_pct(self):
        """Test that invalid crop_pct raises ValueError."""
        with pytest.raises(ValueError, match="crop_pct must be in"):
            TransformConfig(crop_pct=0.0)
        with pytest.raises(ValueError, match="crop_pct must be in"):
            TransformConfig(crop_pct=1.5)

    def test_invalid_mean_length(self):
        """Test that mean with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="mean must have 3 values"):
            TransformConfig(mean=(0.5, 0.5))

    def test_invalid_std_length(self):
        """Test that std with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="std must have 3 values"):
            TransformConfig(std=(0.5, 0.5, 0.5, 0.5))

    def test_with_overrides(self):
        """Test creating a new config with overrides."""
        config = TransformConfig(image_size=224)
        new_config = config.with_overrides(image_size=384, preset="strong")
        assert new_config.image_size == 384
        assert new_config.preset == "strong"
        # Original unchanged
        assert config.image_size == 224
        assert config.preset == "default"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TransformConfig(preset="randaugment", image_size=384)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["preset"] == "randaugment"
        assert d["image_size"] == 384


class TestResolveBackboneDataConfig:
    """Tests for resolve_backbone_data_config function."""

    def test_resolve_from_model_name(self):
        """Test resolving config from a model name."""
        config = resolve_backbone_data_config("resnet50")
        assert "mean" in config
        assert "std" in config
        assert "input_size" in config
        assert len(config["mean"]) == 3
        assert len(config["std"]) == 3

    def test_resolve_with_overrides(self):
        """Test resolving config with overrides."""
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.25, 0.25, 0.25)
        config = resolve_backbone_data_config(
            "resnet50",
            override_mean=custom_mean,
            override_std=custom_std,
        )
        assert config["mean"] == custom_mean
        assert config["std"] == custom_std

    def test_resolve_unknown_model_fallback(self):
        """Test that unknown model falls back to ImageNet defaults."""
        config = resolve_backbone_data_config("unknown_model_xyz")
        # Should fall back to ImageNet defaults
        assert config["mean"] == (0.485, 0.456, 0.406)
        assert config["std"] == (0.229, 0.224, 0.225)


class TestGetTransformsFromBackbone:
    """Tests for get_transforms_from_backbone function."""

    def test_create_train_transforms(self):
        """Test creating training transforms."""
        config = TransformConfig(preset="default", image_size=224)
        transform = get_transforms_from_backbone(
            backbone="resnet50",
            transform_config=config,
            is_train=True,
        )
        assert transform is not None
        # Test that transform works
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_create_eval_transforms(self):
        """Test creating evaluation transforms."""
        config = TransformConfig(image_size=224)
        transform = get_transforms_from_backbone(
            backbone="resnet50",
            transform_config=config,
            is_train=False,
        )
        assert transform is not None
        # Test that transform works
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_different_presets(self):
        """Test different augmentation presets."""
        presets = ["default", "autoaugment", "randaugment", "trivialaugment", "light"]
        for preset in presets:
            config = TransformConfig(preset=preset, image_size=224)
            transform = get_transforms_from_backbone(
                backbone="resnet50",
                transform_config=config,
                is_train=True,
            )
            img = Image.new("RGB", (256, 256))
            tensor = transform(img)
            assert tensor.shape == (3, 224, 224), f"Failed for preset: {preset}"

    def test_custom_image_size(self):
        """Test transforms with custom image size."""
        config = TransformConfig(image_size=384)
        transform = get_transforms_from_backbone(
            backbone="resnet50",
            transform_config=config,
            is_train=False,
        )
        img = Image.new("RGB", (512, 512))
        tensor = transform(img)
        assert tensor.shape == (3, 384, 384)

    def test_custom_normalization(self):
        """Test transforms with custom normalization values."""
        config = TransformConfig(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            use_timm_config=False,
            image_size=224,
        )
        transform = get_transforms_from_backbone(
            backbone="resnet50",
            transform_config=config,
            is_train=False,
        )
        # Create a white image
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        tensor = transform(img)
        # For white image with mean=0.5, std=0.5: (1.0 - 0.5) / 0.5 = 1.0
        assert torch.allclose(tensor, torch.ones_like(tensor), atol=0.1)


class TestCreateInferenceTransform:
    """Tests for create_inference_transform function."""

    def test_create_basic_transform(self):
        """Test creating basic inference transform."""
        transform = create_inference_transform("resnet50")
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_create_with_config(self):
        """Test creating inference transform with config."""
        config = TransformConfig(image_size=384)
        transform = create_inference_transform("resnet50", transform_config=config)
        img = Image.new("RGB", (512, 512))
        tensor = transform(img)
        assert tensor.shape == (3, 384, 384)


class TestModelPreprocessing:
    """Tests for model preprocessing via PreprocessingMixin."""

    @pytest.fixture
    def basic_metrics(self):
        return [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val", "test"],
                prog_bar=True,
            ),
        ]

    def test_model_without_transform_config(self, basic_metrics):
        """Test that model works without transform_config."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
        )
        # Forward should still work
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_model_with_transform_config(self, basic_metrics):
        """Test model with transform_config enables preprocessing."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(),
        )
        # Create a test image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        tensor = model.preprocess(img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_single_image(self, basic_metrics):
        """Test preprocessing a single PIL image."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(image_size=224),
        )
        img = Image.new("RGB", (256, 256))
        tensor = model.preprocess(img)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_image_list(self, basic_metrics):
        """Test preprocessing a list of PIL images."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(image_size=224),
        )
        images = [Image.new("RGB", (256, 256)) for _ in range(4)]
        tensor = model.preprocess(images)
        assert tensor.shape == (4, 3, 224, 224)

    def test_preprocess_numpy_array(self, basic_metrics):
        """Test preprocessing a numpy array."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(image_size=224),
        )
        # Single image as numpy array
        img_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tensor = model.preprocess(img_np)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_tensor_passthrough(self, basic_metrics):
        """Test that tensor input passes through unchanged."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(),
        )
        input_tensor = torch.randn(2, 3, 224, 224)
        output_tensor = model.preprocess(input_tensor)
        assert torch.equal(input_tensor, output_tensor)

    def test_preprocess_without_config_raises(self, basic_metrics):
        """Test that preprocess raises when no transform_config provided."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
        )
        img = Image.new("RGB", (256, 256))
        with pytest.raises(RuntimeError, match="Transforms not set up"):
            model.preprocess(img)

    def test_get_data_config(self, basic_metrics):
        """Test getting data config from model."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(),
        )
        config = model.get_data_config()
        assert "mean" in config
        assert "std" in config
        assert "input_size" in config
        assert len(config["mean"]) == 3
        assert len(config["std"]) == 3

    def test_get_data_config_without_config_raises(self, basic_metrics):
        """Test that get_data_config raises when no transform_config."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
        )
        with pytest.raises(RuntimeError, match="Data config not available"):
            model.get_data_config()

    def test_get_transform(self, basic_metrics):
        """Test getting transforms from model."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(preset="randaugment"),
        )
        train_transform = model.get_transform(is_train=True)
        eval_transform = model.get_transform(is_train=False)
        assert train_transform is not None
        assert eval_transform is not None
        # They should be different objects
        assert train_transform is not eval_transform

    def test_preprocess_with_forward(self, basic_metrics):
        """Test full pipeline: preprocess -> forward."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(image_size=224),
        )
        model.eval()
        img = Image.new("RGB", (256, 256), color=(128, 64, 192))
        tensor = model.preprocess(img)
        with torch.no_grad():
            output = model(tensor)
        assert output.shape == (1, 5)

    def test_preprocess_grayscale_image(self, basic_metrics):
        """Test preprocessing grayscale images (auto-convert to RGB)."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(),
        )
        # Grayscale image
        img = Image.new("L", (256, 256))
        tensor = model.preprocess(img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_custom_image_size_in_config(self, basic_metrics):
        """Test that custom image_size is respected."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
            transform_config=TransformConfig(image_size=384),
        )
        img = Image.new("RGB", (512, 512))
        tensor = model.preprocess(img)
        assert tensor.shape == (1, 3, 384, 384)


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    @pytest.fixture
    def basic_metrics(self):
        return [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val", "test"],
            ),
        ]

    def test_model_without_transform_config_still_works(self, basic_metrics):
        """Ensure models without transform_config work as before."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_training_step_unchanged(self, basic_metrics):
        """Ensure training_step still works without transform_config."""
        model = ImageClassifier(
            backbone="resnet18",
            num_classes=5,
            metrics=basic_metrics,
        )
        model.train()
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 5, (4,))
        loss = model.training_step((x, y), batch_idx=0)
        assert loss.ndim == 0
        assert loss.requires_grad


class TestListTransformPresets:
    """Tests for list_transform_presets function."""

    def test_list_torchvision_presets(self):
        """Test listing torchvision presets."""
        from autotimm import list_transform_presets

        presets = list_transform_presets(backend="torchvision")
        assert isinstance(presets, list)
        assert "default" in presets
        assert "autoaugment" in presets
        assert "randaugment" in presets
        assert "trivialaugment" in presets
        assert "light" in presets

    def test_list_albumentations_presets(self):
        """Test listing albumentations presets."""
        from autotimm import list_transform_presets

        presets = list_transform_presets(backend="albumentations")
        assert isinstance(presets, list)
        assert "default" in presets
        assert "strong" in presets
        assert "light" in presets

    def test_list_presets_default_backend(self):
        """Test that default backend is torchvision."""
        from autotimm import list_transform_presets

        presets = list_transform_presets()
        assert "randaugment" in presets  # torchvision-specific preset

    def test_list_presets_verbose_mode(self):
        """Test verbose mode returns tuples with descriptions."""
        from autotimm import list_transform_presets

        presets = list_transform_presets(backend="torchvision", verbose=True)
        assert isinstance(presets, list)
        assert len(presets) > 0
        # Each item should be a tuple of (name, description)
        for item in presets:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, description = item
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_list_presets_verbose_albumentations(self):
        """Test verbose mode for albumentations backend."""
        from autotimm import list_transform_presets

        presets = list_transform_presets(backend="albumentations", verbose=True)
        names = [name for name, _ in presets]
        assert "default" in names
        assert "strong" in names
        assert "light" in names

    def test_list_presets_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        from autotimm import list_transform_presets

        with pytest.raises(ValueError, match="Unknown backend"):
            list_transform_presets(backend="invalid_backend")

    def test_list_presets_import_from_data(self):
        """Test that function can be imported from autotimm.data."""
        from autotimm.data import list_transform_presets

        presets = list_transform_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_list_presets_import_from_transform_config(self):
        """Test that function can be imported from transform_config module."""
        from autotimm.data.transform_config import list_transform_presets

        presets = list_transform_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0
