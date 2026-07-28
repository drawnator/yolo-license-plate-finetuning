"""Overlay synthetic license plates onto background images with realistic transformations.

Applies:
- Random perspective (homography warp) to simulate camera angles
- Random scaling within a plausible range
- Random position on the background
- Gaussian blur (motion simulation)
- Brightness/contrast variation
- Alpha blending at plate edges for seamless compositing

Yields both the composited image and the YOLO-format bounding box label.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from synthetic_plates.plate_generator import render_plate
from synthetic_plates.plate_types import PlateType, random_plate_type

logger = logging.getLogger(__name__)

# ── Overlay parameters ───────────────────────────────────────────────


@dataclass
class OverlayParams:
    """Parameters controlling how a plate is overlaid onto a background.

    All angles in degrees. All fractions in [0, 1].
    """

    # Scale: fraction of background width the plate will occupy
    # (before perspective distortion). Plates on cars are typically
    # 5-15% of the vehicle width; on a full image they may be 2-8%.
    scale_min: float = 0.03
    scale_max: float = 0.12

    # Perspective rotation (simulates off-axis camera angles)
    pitch_min: float = -30.0   # tilt up/down
    pitch_max: float = 30.0
    yaw_min: float = -40.0     # left/right
    yaw_max: float = 40.0
    roll_min: float = -10.0    # rotation
    roll_max: float = 10.0

    # Position jitter: fraction of bg dimensions where plate center can land
    pos_x_min: float = 0.10
    pos_x_max: float = 0.90
    pos_y_min: float = 0.10
    pos_y_max: float = 0.90

    # Blur (motion simulation)
    blur_prob: float = 0.4
    blur_kernel_min: int = 1
    blur_kernel_max: int = 5

    # Brightness/contrast jitter
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)

    # Edge feathering (blend the plate border into the background)
    edge_feather: int = 2  # px of alpha falloff at edges

    # Noise
    noise_prob: float = 0.3
    noise_stddev: float = 5.0


def random_overlay_params(**overrides) -> OverlayParams:
    """Create OverlayParams with optional overrides."""
    kwargs = {}
    for field_name in OverlayParams.__dataclass_fields__:
        kwargs[field_name] = overrides.get(field_name, getattr(OverlayParams, field_name))
    return OverlayParams(**kwargs)


# ── Core overlay logic ───────────────────────────────────────────────


def overlay_on_background(
    background: np.ndarray,
    plate_img: np.ndarray | None = None,
    plate_type: PlateType | None = None,
    params: OverlayParams | None = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Overlay a synthetic plate onto a background image.

    Args:
        background: Background image as numpy array (H, W, 3) in BGR or RGB.
        plate_img: Pre-rendered plate image. If None, one is generated.
        plate_type: Plate type for generation (used only if plate_img is None).
        params: Overlay parameters. If None, defaults are used.

    Returns:
        A tuple of ``(composited_image, yolo_label)``.
        ``composited_image`` is a numpy array in the same color space as the input.
        ``yolo_label`` is ``(class_id, x_center, y_center, width, height)``
        in normalized YOLO format.
    """
    if params is None:
        params = OverlayParams()

    bg_h, bg_w = background.shape[:2]

    # 1. Get or generate plate image
    if plate_img is None:
        pil_plate = render_plate(plate_type=plate_type)
        plate_img = np.array(pil_plate)
        # PIL gives RGB, convert to BGR to match OpenCV convention
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
    else:
        # Ensure uint8
        if plate_img.dtype != np.uint8:
            plate_img = (plate_img * 255).astype(np.uint8)

    ph, pw = plate_img.shape[:2]

    # 2. Random scale: plate width as fraction of background width
    scale = random.uniform(params.scale_min, params.scale_max)
    target_w = int(bg_w * scale)
    aspect = ph / pw
    target_h = int(target_w * aspect)

    # Clamp to reasonable minimum
    if target_w < 20 or target_h < 8:
        target_w = max(20, target_w)
        target_h = max(8, target_h)

    # 3. Perspective transform
    # Source: 4 corners of the plate
    src_pts = np.float32([
        [0, 0],
        [pw - 1, 0],
        [pw - 1, ph - 1],
        [0, ph - 1],
    ])

    # Destination: apply pitch/yaw/roll to the target rectangle
    dst_pts = _compute_perspective_corners(
        target_w, target_h,
        pitch=random.uniform(params.pitch_min, params.pitch_max),
        yaw=random.uniform(params.yaw_min, params.yaw_max),
        roll=random.uniform(params.roll_min, params.roll_max),
    )

    # Translate to random position within the background
    cx = random.uniform(params.pos_x_min, params.pos_x_max) * bg_w
    cy = random.uniform(params.pos_y_min, params.pos_y_max) * bg_h
    dst_pts[:, 0] += cx
    dst_pts[:, 1] += cy

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the plate
    warped = cv2.warpPerspective(plate_img, M, (bg_w, bg_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

    # Create an alpha mask (all opaque) and warp it the same way
    alpha = np.ones((ph, pw), dtype=np.float32) * 255
    alpha_warped = cv2.warpPerspective(alpha, M, (bg_w, bg_h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

    # 4. Feather the alpha edges
    if params.edge_feather > 0:
        kernel = np.ones((params.edge_feather, params.edge_feather), np.float32)
        alpha_warped = cv2.erode(alpha_warped, kernel, iterations=1)
        alpha_warped = cv2.GaussianBlur(alpha_warped, (params.edge_feather * 2 + 1,) * 2, 0)

    alpha_warped = alpha_warped / 255.0
    alpha_3ch = np.stack([alpha_warped] * 3, axis=-1)

    # 5. Apply blur to the warped plate (motion)
    if random.random() < params.blur_prob:
        ksize = random.randrange(params.blur_kernel_min, params.blur_kernel_max + 1)
        if ksize > 1:
            if ksize % 2 == 0:
                ksize += 1  # must be odd
            warped = cv2.GaussianBlur(warped, (ksize, ksize), 0)

    # 6. Brightness/contrast jitter on the plate
    warped = _adjust_brightness_contrast(
        warped,
        alpha=random.uniform(*params.contrast_range),
        beta=int(random.uniform(
            (params.brightness_range[0] - 1) * 50,
            (params.brightness_range[1] - 1) * 50,
        )),
    )

    # 7. Noise on the plate
    if random.random() < params.noise_prob:
        noise = np.random.randn(*warped.shape) * params.noise_stddev
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. Composite: plate * alpha + background * (1 - alpha)
    bg_f = background.astype(np.float32)
    warped_f = warped.astype(np.float32)
    composited = (warped_f * alpha_3ch + bg_f * (1.0 - alpha_3ch)).astype(np.uint8)

    # 9. Compute YOLO label from the 4 destination corners
    #    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    x_min = dst_pts[:, 0].min()
    x_max = dst_pts[:, 0].max()
    y_min = dst_pts[:, 1].min()
    y_max = dst_pts[:, 1].max()

    # Clamp to image bounds
    x_min = max(0, x_min)
    x_max = min(bg_w - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(bg_h - 1, y_max)

    box_w = (x_max - x_min) / bg_w
    box_h = (y_max - y_min) / bg_h
    box_xc = (x_min + x_max) / 2 / bg_w
    box_yc = (y_min + y_max) / 2 / bg_h

    # Class 0 = plate
    yolo_label = (0, box_xc, box_yc, box_w, box_h)

    return composited, yolo_label


def _compute_perspective_corners(
    w: int, h: int,
    pitch: float, yaw: float, roll: float,
) -> np.ndarray:
    """Compute the 4 corners of a plate after perspective distortion.

    The plate is placed at the center of the image and then rotated
    in 3D (pitch, yaw, roll), projected orthographically.

    Returns 4 corners as (N, 2) float32 array.
    """
    # Start with a rectangle centered at origin
    hw, hh = w / 2, h / 2
    corners = np.float32([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ])

    # Convert angles to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    # Build 3D rotation matrix (approximate — we only need the visual effect)
    # Rotation order: roll → pitch → yaw
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    # Roll matrix
    R_roll = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

    # Apply pitch + yaw as a simple scaling on x/y (simplified projection)
    scale_x = cos_y * (1.0 - abs(sin_p) * 0.3)
    scale_y = cos_p

    # Apply transformations
    transformed = (corners @ R_roll.T).astype(np.float32)
    transformed[:, 0] *= float(scale_x)
    transformed[:, 1] *= float(scale_y)

    # Translate to a random position within the image (done by caller)
    # We return centered; the caller will offset
    return transformed


def _overlay_plate_at_position(
    background: np.ndarray,
    plate_img: np.ndarray,
    dst_corners: np.ndarray,
    params: OverlayParams,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Low-level: warp and blend the plate at specific corner positions."""
    bg_h, bg_w = background.shape[:2]
    ph, pw = plate_img.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [pw - 1, 0],
        [pw - 1, ph - 1],
        [0, ph - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_corners)

    warped = cv2.warpPerspective(plate_img, M, (bg_w, bg_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

    alpha = np.ones((ph, pw), dtype=np.float32) * 255
    alpha_warped = cv2.warpPerspective(alpha, M, (bg_w, bg_h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

    if params.edge_feather > 0:
        kernel = np.ones((params.edge_feather, params.edge_feather), np.float32)
        alpha_warped = cv2.erode(alpha_warped, kernel, iterations=1)
        alpha_warped = cv2.GaussianBlur(alpha_warped,
                                        (params.edge_feather * 2 + 1,) * 2, 0)

    alpha_warped = alpha_warped / 255.0
    alpha_3ch = np.stack([alpha_warped] * 3, axis=-1)

    bg_f = background.astype(np.float32)
    warped_f = warped.astype(np.float32)
    composited = (warped_f * alpha_3ch + bg_f * (1.0 - alpha_3ch)).astype(np.uint8)

    # YOLO label
    x_min = dst_corners[:, 0].min()
    x_max = dst_corners[:, 0].max()
    y_min = dst_corners[:, 1].min()
    y_max = dst_corners[:, 1].max()

    x_min = max(0.0, x_min)
    x_max = min(float(bg_w - 1), x_max)
    y_min = max(0.0, y_min)
    y_max = min(float(bg_h - 1), y_max)

    box_w = (x_max - x_min) / bg_w
    box_h = (y_max - y_min) / bg_h
    box_xc = (x_min + x_max) / 2 / bg_w
    box_yc = (y_min + y_max) / 2 / bg_h

    return composited, (0, box_xc, box_yc, box_w, box_h)


def _adjust_brightness_contrast(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: int = 0,
) -> np.ndarray:
    """Adjust brightness (beta) and contrast (alpha) of an image.

    output = img * alpha + beta
    """
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return result


# ── Utility: generate plate plate_img directly as numpy (for callers that want BGR) ──

def generate_plate_numpy(
    plate_type: PlateType | None = None,
) -> np.ndarray:
    """Generate a plate as a BGR numpy array (for OpenCV compositing)."""
    pil_img = render_plate(plate_type=plate_type)
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)