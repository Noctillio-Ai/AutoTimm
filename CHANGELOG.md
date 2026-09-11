# Changelog

All notable changes to AutoTimm are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). For the full
per-release history, see [GitHub Releases](https://github.com/theja-vanka/AutoTimm/releases),
which are auto-generated from commits on each tagged publish.

## [0.8.0rc1] — 2026-09-11

First release candidate. Beyond the changes below, this release includes a
full-codebase quality pass over every argument combination exposed by the
public API.

### Fixed — rc1 quality pass
- **CSV detection datasets corrupted bounding boxes.** `CSVDetectionDataset`
  passed xyxy boxes into transform pipelines declared with COCO
  (`[x, y, w, h]`) bbox format and used the output as xyxy. Any geometric
  transform (flip, pad) silently produced wrong boxes, and boxes near the
  right/bottom image edge crashed albumentations validation. Boxes are now
  converted xyxy → coco before the transform and back after.
- **albumentations 2.x silently ignored renamed arguments.**
  `PadIfNeeded(value=..., mask_value=...)` and `GaussNoise(var_limit=...)`
  are no-ops in albumentations 2.x — segmentation mask padding was filled
  with class 0 instead of `ignore_index` 255, and detection padding lost its
  gray fill. Migrated to `fill=`/`fill_mask=`/`std_range=`.
- **`balanced_sampling` with an automatic val split crashed or mis-weighted.**
  Sampler weights were computed over the full pre-split dataset, so
  `WeightedRandomSampler` indexed past the training subset (IndexError).
  Built-in datasets with tensor targets (MNIST) additionally hashed targets
  by identity, silently making balanced sampling a no-op.
- **Automatic validation splits were augmented with training transforms.**
  `random_split` shared the training dataset instance; the val subset now
  wraps a separate dataset built with eval transforms (`ImageDataModule`,
  `MultiLabelImageDataModule`, built-in dataset mode).
- **Model-specific normalization was silently wrong for string backbones.**
  `resolve_backbone_data_config("vit_...")` fell back to generic ImageNet
  stats (ViTs should use mean/std 0.5 and crop_pct 0.9). String backbones now
  resolve through `timm.get_pretrained_cfg`, including tagged
  (`resnet50.a1_in1k`) and `hf-hub:`/`timm/` names.
  `TransformConfig.interpolation`/`crop_pct` now default to `None` so the
  model's pretrained values win unless explicitly overridden.
- **Every timm optimizer name raised `AttributeError`.** The optimizer maps
  referenced `timm.optim.NovGrad`, which doesn't exist in timm 1.x, and the
  map was built eagerly — `optimizer="adamp"` (or any timm optimizer) crashed.
  `"novograd"` now maps to `NvNovoGrad` and classes are looked up lazily.
- **`scheduler="cosine_with_restarts"` crashed at the first scheduler step**
  (timm schedulers need `step(epoch)`); `ImageClassifier` now overrides
  `lr_scheduler_step`. `scheduler="plateau"` with `monitor` in
  `scheduler_kwargs` no longer raises `TypeError`.
- **Tasks silently discarded `BackboneConfig`/`FeatureBackboneConfig` fields**
  (`pretrained`, `drop_rate`, `drop_path_rate`, `extra_kwargs`); the config
  is now honored, with hparams still storing the serializable name.
- **YOLOXDetector's regression loss was mathematically wrong** — raw LTRB
  distances were fed to `GIoULoss` as if they were xyxy boxes (identical
  predictions scored loss 1.0). Its classification loss also ignored all
  background locations. Both now match the shared FCOS implementation.
- **`reg_loss_fn` was accepted but never used** by
  `ObjectDetector`/`InstanceSegmentor`; a custom regression loss is now
  applied (on point-relative xyxy boxes). Default behavior unchanged.
- **InstanceSegmentor mask training/inference was spatially misaligned.**
  Training targets resized the full-image mask to 28×28 while the head
  predicts within each ROI box (targets are now `roi_align`-cropped), and
  `predict()` stretched the ROI mask over the entire image (now pasted into
  the box region).
- **`SemanticSegmentor` with metrics crashed on shape mismatch** — argmax
  predictions were at head resolution while masks are full-resolution;
  predictions now share the loss path's interpolation.
- **`InstanceSegmentationDataModule` + `transform_config` crashed** on the
  first sample (transforms built without `bbox_params`); a new
  `instance_segmentation` task mode attaches pascal_voc bbox params.
- `get_train_transforms("light")` was rejected although
  `list_transform_presets()` advertises it; added for both backends.
- `DetectionDataModule` CSV mode without `val_csv` crashed in
  `val_dataloader()`; validation is now skipped.
- `YOLOXDetector` documented dict optimizer/scheduler configs but crashed on
  them (`.lower()` on a dict); dict configs are now supported.
- Valid tagged timm names (`resnet50.a1_in1k`) were wrongly rejected by
  `create_backbone`/`create_feature_backbone`.
- `autotimm-flow augmentation-preview` rendered normalized tensors (dark or
  garbage previews); pixels are now denormalized. `tensorrt-convert` no
  longer writes a `None` engine on build failure.
- Examples and docs: fixed 30+ code snippets that passed parameters which do
  not exist (`head_channels`, `mask_head_channels`, `train_split`,
  `weighted_sampling`, `bbox_format`, `channels_last`, `loss_kwargs`,
  `LoggingConfig(log_dir=...)`, string `logger=` values, a fictional
  `PresetManager` class, wrong `FCOSLoss` weight names, and more).

### Fixed — InstanceSegmentor detection path was non-functional (breaking)
`InstanceSegmentor`'s detection head previously received **zero training
signal** — `_compute_detection_loss` was a hardcoded placeholder
(`sum(...) * 0.0` for every term) regardless of the head's actual
predictions — and `predict()` unconditionally returned empty
boxes/scores/labels/masks for every input. Only the mask head (trained on
ground-truth boxes) ever learned anything; the detection head stayed at
random initialization and inference always reported zero detections.

Both are now real implementations, sharing the same validated FCOS
target-assignment/loss/decode logic as `ObjectDetector` (factored out into
`autotimm.tasks._fcos_targets`). Also fixed as part of this:
`validation_step`/`test_step` now cast target masks to `bool` before handing
them to torchmetrics' `MeanAveragePrecision(iou_type="segm")`, which
previously raised on the dataset's native float masks the first time a batch
with real masks reached it (i.e. always, once `predict()` was actually
returning results).

**Breaking:** `InstanceSegmentor`'s backbone now uses 3 feature levels
(`out_indices=(2, 3, 4)`) instead of 4, to match the 5-level (P3–P7)
`strides`/`regress_ranges` convention — the previous 4-level default was
itself inconsistent with those defaults (part of why this went unnoticed).
Any checkpoint saved from a previous `InstanceSegmentor` will not load into
this version due to the resulting shape mismatch. Given the model never
produced working detections before this fix, no working checkpoint should
exist to migrate.

### Fixed
- `ImageDataModule`/`DetectionDataModule`: model-specific eval-time normalization
  (`transform_config` + `backbone`) was silently overwritten by generic default
  transforms because eval-transform resolution wasn't guarded the same way as
  train-transform resolution.
- `SmoothGrad.explain` and `ExplanationMetrics.sensitivity_n`: noisy samples were
  clamped to `[0, 1]` even when the underlying tensor was ImageNet-normalized
  (roughly `[-2, 2.6]`), which destroyed most of the signal before noise-based
  attribution methods ran.
- `ExplanationMetrics.deletion`/`insertion`: AUC computation could divide by a
  near-zero prediction score with no epsilon guard.
- `ExplanationMetrics.model_parameter_randomization_test`: model weights are now
  restored in a `finally` block, so an exception during `explain()` no longer
  leaves the model's parameters permanently randomized.
- `COCOInstanceDataset.__getitem__`: added the missing `None` check after
  `cv2.imread` (present in sibling dataset classes) so a missing/corrupt image
  raises a clear `RuntimeError` instead of an opaque OpenCV assertion.
- Checkpoint loading (`cli/interpret_cli.py`, `flow/push_to_hub.py`,
  `export/export_jit.py`, `export/export_onnx.py`) now tries
  `torch.load(..., weights_only=True)` first via a shared `safe_torch_load`
  helper, only falling back to the unsafe unpickler (with a logged warning) if
  the restricted load fails.

### Changed
- Detection batches now use the `"image"` key (singular) instead of `"images"`,
  matching the key already used by segmentation/instance-segmentation batches
  and the documented contract in `docs/api/detection_data.md`.
- `AutoTrainer` no longer mutates the caller-supplied `callbacks` list in place.
- `autotimm.flow.push_to_hub` now prefers an `HF_TOKEN` environment variable
  over `--token` on the command line, to avoid leaking tokens into shell
  history/process listings.
- `autotimm.core.logging` no longer reconfigures the process-wide loguru
  sink when `AUTOTIMM_NO_LOG_CONFIG=1` is set, so embedding applications with
  their own loguru setup aren't silently overridden by `import autotimm`.
- CI lint/format checks (`ruff`, `black`) now actually fail the build instead
  of being masked with `|| true`; same for the HF Hub smoke test in the
  publish pipeline.
- PyPI/TestPyPI publishing now uses OIDC trusted publishing instead of
  long-lived API token secrets (requires trusted publisher configuration on
  PyPI's project settings — see `CONTRIBUTING.md`).

### Performance
- `ExplanationMetrics.deletion`/`insertion` vectorized the per-pixel baseline
  replacement (previously a Python-level loop over up to H×W pixels per step).

### Removed
- `FCOSLoss` no longer constructs an unused `GIoULoss` instance; its docstring
  now accurately describes the simplified IoU regression loss actually used.
