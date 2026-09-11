"""LightningCLI integration for AutoTimm.

Provides a command-line interface for training, validation, testing,
and prediction using AutoTimm task classes and data modules.

Usage:
    autotimm fit --config config.yaml
    python -m autotimm fit --config config.yaml
    autotimm fit --config config.yaml --model.lr 0.001 --trainer.max_epochs 20
"""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

# Import data module classes so LightningCLI can discover them as subclasses
from autotimm.data.datamodule import ImageDataModule  # noqa: F401
from autotimm.data.detection_datamodule import DetectionDataModule  # noqa: F401
from autotimm.data.instance_datamodule import (  # noqa: F401
    InstanceSegmentationDataModule,
)
from autotimm.data.multilabel_datamodule import MultiLabelImageDataModule  # noqa: F401
from autotimm.data.segmentation_datamodule import SegmentationDataModule  # noqa: F401

# Import task classes so LightningCLI can discover them as subclasses
from autotimm.tasks.classification import ImageClassifier  # noqa: F401
from autotimm.tasks.instance_segmentation import InstanceSegmentor  # noqa: F401
from autotimm.tasks.object_detection import ObjectDetector  # noqa: F401
from autotimm.tasks.semantic_segmentation import SemanticSegmentor  # noqa: F401
from autotimm.tasks.yolox_detector import YOLOXDetector  # noqa: F401
from autotimm.training.trainer import AutoTrainer


class AutoTimmCLI(LightningCLI):
    """AutoTimm command-line interface built on LightningCLI.

    Supports subcommands: ``fit``, ``validate``, ``test``, ``predict``.
    Uses ``AutoTrainer`` as the default trainer class and discovers all
    AutoTimm task and data module classes automatically.

    Example:
        >>> # From the command line:
        >>> # autotimm fit --config config.yaml
        >>> # autotimm fit --config config.yaml --trainer.max_epochs 20
    """


def main() -> None:
    """Entry point for the AutoTimm CLI."""
    AutoTimmCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        trainer_class=AutoTrainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=False,
    )


if __name__ == "__main__":
    main()
