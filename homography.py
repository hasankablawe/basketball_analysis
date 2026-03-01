"""
Homography module — court keypoint detection and perspective transform.

Functions:
    load_court_pose_model   — load the YOLO pose model for court keypoints
    detect_court_keypoints  — predict 33 keypoints from a video frame
    compute_homography      — build homography matrix from detected ↔ reference keypoints
    transform_point         — project a single image point to 2D court coords
    transform_points        — batch projection
"""

import numpy as np
import cv2
from ultralytics import YOLO

from constants import (
    COURT_POSE_MODEL,
    COURT_KP_CONFIDENCE,
    COURT_REFERENCE_KEYPOINTS,
    DEVICE,
)


def load_court_pose_model(model_path: str | None = None, device: str = DEVICE) -> YOLO:
    """Load the court keypoint pose model.

    Args:
        model_path: Override path. If *None*, uses ``COURT_POSE_MODEL`` from constants.
        device:     Inference device.

    Returns:
        Loaded YOLO model.
    """
    path = model_path or str(COURT_POSE_MODEL)
    print(f"[homography] Loading court pose model: {path}")
    return YOLO(path)


def detect_court_keypoints(
    model: YOLO,
    frame: np.ndarray,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect court keypoints in a frame.

    Args:
        model:  Loaded court pose model.
        frame:  BGR image.
        device: Inference device.

    Returns:
        A tuple ``(keypoints, confidences)`` where *keypoints* is shape (N, 2)
        with (x, y) pixel coordinates, and *confidences* is shape (N,).
        N should be 33 when the model detects the court.
    """
    results = model(frame, device=device, verbose=False)

    if not results or results[0].keypoints is None:
        return np.empty((0, 2)), np.empty(0)

    kps_data = results[0].keypoints
    # Shape: (num_objects, num_keypoints, 3) — last dim is (x, y, conf)
    if kps_data.data is None or len(kps_data.data) == 0:
        return np.empty((0, 2)), np.empty(0)

    # Take the first detected "court" object
    kps = kps_data.data[0].cpu().numpy()  # (33, 3)
    xy = kps[:, :2]       # (33, 2)
    conf = kps[:, 2]      # (33,)

    return xy, conf


def compute_homography(
    detected_kps: np.ndarray,
    confidences: np.ndarray,
    min_conf: float = COURT_KP_CONFIDENCE,
) -> np.ndarray | None:
    """Compute homography from detected keypoints to reference court coords.

    Only keypoints with confidence ≥ ``min_conf`` are used.  Requires at
    least 4 valid correspondences.

    Args:
        detected_kps: (N, 2) detected pixel coordinates.
        confidences:  (N,) per-keypoint confidence.
        min_conf:     Minimum confidence threshold.

    Returns:
        3×3 homography matrix, or *None* if not enough valid keypoints.
    """
    ref_kps = np.array(COURT_REFERENCE_KEYPOINTS, dtype=np.float64)

    # Filter by confidence
    n = min(len(detected_kps), len(ref_kps), len(confidences))
    valid = confidences[:n] >= min_conf
    if valid.sum() < 4:
        return None

    src_pts = detected_kps[:n][valid].astype(np.float64)
    dst_pts = ref_kps[:n][valid].astype(np.float64)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def transform_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Project a single image-space point to court coordinates.

    Args:
        H:     3×3 homography matrix.
        point: (x, y) pixel coordinate.

    Returns:
        (x_ft, y_ft) court coordinate.
    """
    p = np.array([point[0], point[1], 1.0], dtype=np.float64)
    tp = H @ p
    if abs(tp[2]) < 1e-8:
        return (0.0, 0.0)
    return (tp[0] / tp[2], tp[1] / tp[2])


def transform_points(
    H: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Batch-project image-space points to court coordinates.

    Args:
        H:      3×3 homography matrix.
        points: (N, 2) pixel coordinates.

    Returns:
        (N, 2) court coordinates in feet.
    """
    if len(points) == 0:
        return np.empty((0, 2))

    n = len(points)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([points.astype(np.float64), ones])  # (N, 3)
    transformed = (H @ pts_h.T).T  # (N, 3)

    # Normalize by w
    w = transformed[:, 2:3]
    w[np.abs(w) < 1e-8] = 1e-8
    result = transformed[:, :2] / w

    return result
