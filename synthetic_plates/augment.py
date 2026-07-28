"""Online augmentation: inject synthetic plates during YOLO training.

Provides an albumentations-compatible transform that generates fresh synthetic
plates on-the-fly and overlays them onto training images. This is used as custom
augmentation in Ultralytics ``model.train(augmentations=[...])``.

Usage in training scripts::

    from synthetic_plates.augment import synthetic_plate_augmentation
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")
    model.train(
        data="data.yaml",
        augmentations=[synthetic_plate_augmentation()],
        ...
    )

Note: this module requires ``albumentations``. Install it with:
    pip install albumentations
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from synthetic_plates.overlay import (
    overlay_on_background,
    OverlayParams,
)

logger = logging.getLogger(__name__)


def _require_albumentations():
    """Lazy-import albumentations, raising a clear error if missing."""
    try:
        import albumentations  # noqa: F401
        return albumentations
    except ImportError:
        raise ImportError(
            "albumentations is required for synthetic_plates.augment. "
            "Install it with: pip install albumentations"
        )


class SyntheticPlateAugment:
    """Albumentations transform that overlays synthetic plates onto training images.

    This is a *destructive* augmentation: it modifies the image pixels directly.
    It does NOT modify the bounding boxes (the original labels are preserved).

    Args:
        prob: Probability of applying the augmentation to an image.
        plates_per_image: How many plates to overlay when triggered.
        overlay_params: Parameters for the overlay transform.
        always_apply: Albumentations flag.
        p: Albumentations alias for ``prob``.
    """

    def __init__(
        self,
        prob: float = 0.5,
        plates_per_image: int = 1,
        overlay_params: OverlayParams | None = None,
        always_apply: bool = False,
        p: float | None = None,
    ):
        A = _require_albumentations()
        # Dynamically rebase to A.ImageOnlyTransform so ultralytics
        # recognises this as a valid transform class.
        SyntheticPlateAugment.__bases__ = (A.ImageOnlyTransform,)
        A.ImageOnlyTransform.__init__(
            self, always_apply=always_apply, p=p if p is not None else prob
        )
        self.plates_per_image = plates_per_image
        self.overlay_params = overlay_params or OverlayParams()

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Overlay synthetic plates onto the image."""
        result = img.copy()
        for _ in range(self.plates_per_image):
            result, _ = overlay_on_background(
                background=result,
                params=self.overlay_params,
            )
        return result

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("plates_per_image", "overlay_params")


def synthetic_plate_augmentation(
    prob: float = 0.5,
    plates_per_image: int = 1,
):
    """Create a synthetic plate augmentation transform for Ultralytics training.

    Args:
        prob: Probability of overlying plates on any given image.
        plates_per_image: Number of plates to overlay per triggered image.

    Returns:
        An ``A.ImageOnlyTransform`` ready for ``model.train(augmentations=[...])``.

    Example::

        from ultralytics import YOLO
        from synthetic_plates.augment import synthetic_plate_augmentation

        model = YOLO("yolo26s.pt")
        model.train(
            data="data.yaml",
            augmentations=[synthetic_plate_augmentation(prob=0.3, plates_per_image=2)],
            epochs=100,
        )

    Note:
        This transform only modifies image pixels — it does NOT add new labels.
        Existing YOLO labels are preserved. For the model, plates will appear
        as unlabeled distractors, which helps it focus on real plates.
        To add new labels, pre-generate a synthetic dataset with
        :func:`synthetic_plates.dataset_builder.build_dataset` and merge it
        into your ``data.yaml``.
    """
    return SyntheticPlateAugment(
        prob=prob,
        plates_per_image=plates_per_image,
    )