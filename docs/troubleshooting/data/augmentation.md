# Data Augmentation Issues

Transform and augmentation configuration problems.

## Augmentation Too Strong

**Symptoms:** Training accuracy remains low, loss doesn't converge

```python
# Use weaker augmentation preset
data = ImageDataModule(
    data_dir="./data",
    augmentation_preset="light",  # Instead of "strong"
)

# Or disable augmentation temporarily
data = ImageDataModule(
    data_dir="./data",
    augmentation_preset=None,
)
```

## Custom Transform Errors

```python
import autotimm as at  # recommended alias
from autotimm import TransformConfig

# Debug transforms
transform_config = TransformConfig(
    preset="light",
)

# Test transform on single image
from PIL import Image
img = Image.open("test_image.jpg")
transforms = transform_config.get_train_transforms(image_size=224)

try:
    transformed = transforms(img)
    print(f"Transform successful: {transformed.shape}")
except Exception as e:
    print(f"Transform failed: {e}")
```

## Bbox Transforms for Detection

```python
# Ensure bbox transforms are compatible
from autotimm import DetectionDataModule

data = DetectionDataModule(
    data_dir="./data",
    image_size=640,
    # Geometric transforms automatically handle bboxes
    augmentation_preset="strong",  # or "default"
)
```

## Wrong Predictions After Training

**Problem:** Inference doesn't match training performance

**Solution:** Match normalization between training and inference:

```python
# Use model's preprocess method
tensor = model.preprocess(image)

# Or manually match:
config = model.get_data_config()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std']),
])
```

## Related Issues

- [Data Loading](data-loading.md) - Dataset loading issues
- [Convergence](../training/convergence.md) - Training doesn't improve
