# Changelog

All notable changes to AutoTimm are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). For the full
per-release history, see [GitHub Releases](https://github.com/theja-vanka/AutoTimm/releases),
which are auto-generated from commits on each tagged publish.

## [Unreleased]

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
