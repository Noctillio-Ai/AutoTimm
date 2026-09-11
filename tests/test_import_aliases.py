"""Tests for import aliases."""


class TestImportAliases:
    """Test submodule import aliases."""

    def test_loss_alias(self):
        """Test autotimm.loss alias."""
        import autotimm.loss

        assert autotimm.loss is not None

        # Check that key losses are available
        assert hasattr(autotimm.loss, "DiceLoss")
        assert hasattr(autotimm.loss, "MaskLoss")
        assert hasattr(autotimm.loss, "CombinedSegmentationLoss")
        assert hasattr(autotimm.loss, "FocalLoss")

    def test_metric_alias(self):
        """Test autotimm.metric alias."""
        import autotimm.metric

        assert autotimm.metric is not None

        # Check that key metrics are available
        assert hasattr(autotimm.metric, "MetricConfig")
        assert hasattr(autotimm.metric, "MetricManager")
        assert hasattr(autotimm.metric, "LoggingConfig")

    def test_head_alias(self):
        """Test autotimm.head alias."""
        import autotimm.head

        assert autotimm.head is not None

        # Check that key heads are available
        assert hasattr(autotimm.head, "DeepLabV3PlusHead")
        assert hasattr(autotimm.head, "FCNHead")
        assert hasattr(autotimm.head, "MaskHead")
        assert hasattr(autotimm.head, "ClassificationHead")
        assert hasattr(autotimm.head, "DetectionHead")

    def test_task_alias(self):
        """Test autotimm.task alias."""
        import autotimm.task

        assert autotimm.task is not None

        # Check that key tasks are available
        assert hasattr(autotimm.task, "SemanticSegmentor")
        assert hasattr(autotimm.task, "InstanceSegmentor")
        assert hasattr(autotimm.task, "ImageClassifier")
        assert hasattr(autotimm.task, "ObjectDetector")

    def test_import_from_aliases(self):
        """Test importing from aliases."""
        from autotimm.head import DeepLabV3PlusHead
        from autotimm.loss import DiceLoss
        from autotimm.metric import MetricConfig
        from autotimm.task import SemanticSegmentor

        assert DiceLoss is not None
        assert MetricConfig is not None
        assert DeepLabV3PlusHead is not None
        assert SemanticSegmentor is not None

    def test_original_imports_still_work(self):
        """Test that original imports still work."""
        from autotimm.loss import DiceLoss as DiceLoss2

        from autotimm.losses import DiceLoss as DiceLoss1

        # Should be the same class
        assert DiceLoss1 is DiceLoss2

    def test_namespace_access(self):
        """Test accessing via autotimm namespace."""
        import autotimm

        # Check aliases are accessible
        assert hasattr(autotimm, "loss")
        assert hasattr(autotimm, "metric")
        assert hasattr(autotimm, "head")
        assert hasattr(autotimm, "task")

        # Check can access classes via namespace
        assert hasattr(autotimm.loss, "DiceLoss")
        assert hasattr(autotimm.metric, "MetricConfig")
        assert hasattr(autotimm.head, "DeepLabV3PlusHead")
        assert hasattr(autotimm.task, "SemanticSegmentor")

    def test_all_loss_classes(self):
        """Test all loss classes are accessible via alias."""
        from autotimm.loss import (
            CombinedSegmentationLoss,
            DiceLoss,
            FocalLoss,
            FocalLossPixelwise,
            GIoULoss,
            MaskLoss,
            TverskyLoss,
        )

        losses = [
            DiceLoss,
            FocalLoss,
            FocalLossPixelwise,
            MaskLoss,
            TverskyLoss,
            CombinedSegmentationLoss,
            GIoULoss,
        ]

        for loss_cls in losses:
            assert loss_cls is not None

    def test_all_head_classes(self):
        """Test all head classes are accessible via alias."""
        from autotimm.head import (
            ASPP,
            FPN,
            ClassificationHead,
            DeepLabV3PlusHead,
            DetectionHead,
            FCNHead,
            MaskHead,
        )

        heads = [
            ASPP,
            ClassificationHead,
            DeepLabV3PlusHead,
            DetectionHead,
            FCNHead,
            FPN,
            MaskHead,
        ]

        for head_cls in heads:
            assert head_cls is not None

    def test_all_task_classes(self):
        """Test all task classes are accessible via alias."""
        from autotimm.task import (
            ImageClassifier,
            InstanceSegmentor,
            ObjectDetector,
            SemanticSegmentor,
        )

        tasks = [
            ImageClassifier,
            InstanceSegmentor,
            ObjectDetector,
            SemanticSegmentor,
        ]

        for task_cls in tasks:
            assert task_cls is not None

    def test_create_instances_via_aliases(self):
        """Test creating instances using aliases."""
        from autotimm.loss import DiceLoss
        from autotimm.task import SemanticSegmentor

        # Create loss instance
        loss = DiceLoss(num_classes=10)
        assert loss is not None

        # Create model instance
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            head_type="fcn",
            metrics=None,
        )
        assert model is not None
