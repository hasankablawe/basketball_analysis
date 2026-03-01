"""
Detection module — RT-DETR model loading and tracking.

Functions:
    load_detector      — load an RTDETR model from disk
    detect_and_track   — run detection + BoT-SORT tracking on a single frame
"""

import numpy as np
from ultralytics import RTDETR

from constants import DETECTOR_MODEL, DETECTOR_FALLBACK, CONFIDENCE_THRESHOLD, DEVICE


def load_detector(
    model_path: str | None = None,
    device: str = DEVICE,
) -> RTDETR:
    """Load the RT-DETR detection model.

    Tries the TensorRT engine first; falls back to the .pt checkpoint.

    Args:
        model_path: Explicit path override. If *None*, uses the default
                    from ``constants.py``.
        device:     Inference device (``"0"`` for GPU, ``"cpu"`` for CPU).

    Returns:
        A loaded ``RTDETR`` model instance.
    """
    if model_path is None:
        path = DETECTOR_MODEL if DETECTOR_MODEL.exists() else DETECTOR_FALLBACK
    else:
        from pathlib import Path
        path = Path(model_path)

    print(f"[detection] Loading model: {path.name}")
    return RTDETR(str(path))


def detect_and_track(
    model: RTDETR,
    frame: np.ndarray,
    conf: float = CONFIDENCE_THRESHOLD,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run detection + BoT-SORT tracking on a single frame.

    Args:
        model:   Loaded RTDETR model.
        frame:   BGR image (H, W, 3).
        conf:    Minimum confidence threshold.
        device:  Inference device.

    Returns:
        A 4-tuple ``(boxes, confs, class_ids, track_ids)`` where each is a
        numpy array.  If nothing is detected, all arrays are empty.
    """
    results = model.track(
        source=frame,
        persist=True,
        conf=conf,
        device=device,
        verbose=False,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        empty = np.empty((0, 4))
        return empty, np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = (
        result.boxes.id.cpu().numpy().astype(int)
        if result.boxes.id is not None
        else np.arange(len(boxes))
    )

    return boxes, confs, class_ids, track_ids
