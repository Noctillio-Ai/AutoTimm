"""Mixin that adds backbone freezing / unfreezing to task models."""

from __future__ import annotations


class BackboneFreezeMixin:
    """Freeze or unfreeze ``self.backbone`` so pretrained weights stay intact.

    Freezing a pretrained backbone (especially a large transformer such as
    ViT, DeiT or Swin) and training only the task head avoids catastrophic
    forgetting of the pretrained representation.

    A frozen backbone:

    * has ``requires_grad=False`` on every parameter, so the optimizer never
      updates it (``configure_optimizers`` only collects trainable params);
    * is kept in ``eval()`` mode even while the rest of the model trains, so
      BatchNorm running statistics are not overwritten and dropout /
      stochastic depth are disabled. Without this, ``requires_grad=False``
      alone still lets BatchNorm statistics drift on the new data.

    Usage::

        model = ImageClassifier(backbone="vit_base_patch16_224",
                                num_classes=10, freeze_backbone=True)
        model.freeze_backbone          # True
        model.unfreeze_backbone()      # or: model.freeze_backbone = False

    Note:
        Changing the frozen state takes effect on the optimizer the next time
        ``configure_optimizers`` runs, i.e. at the next ``trainer.fit()`` call.
    """

    _backbone_frozen: bool = False

    @property
    def freeze_backbone(self) -> bool:
        """Whether the backbone is currently frozen. Assign to change it."""
        return self._backbone_frozen

    @freeze_backbone.setter
    def freeze_backbone(self, frozen: bool) -> None:
        self.set_backbone_frozen(frozen)

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Freeze (``True``) or unfreeze (``False``) all backbone parameters."""
        frozen = bool(frozen)
        object.__setattr__(self, "_backbone_frozen", frozen)
        backbone = getattr(self, "backbone", None)
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = not frozen
        if frozen:
            backbone.eval()
        elif getattr(self, "training", False):
            backbone.train()

    def unfreeze_backbone(self) -> None:
        """Make all backbone parameters trainable again."""
        self.set_backbone_frozen(False)

    def train(self, mode: bool = True):
        """Set train/eval mode, keeping a frozen backbone in eval mode."""
        result = super().train(mode)
        if mode and self._backbone_frozen:
            backbone = getattr(self, "backbone", None)
            if backbone is not None:
                backbone.eval()
        return result
