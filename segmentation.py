"""
Segmentation module — SAM2 real-time tracking.

Approach (from Roboflow reference):
    Frame 0:   Prompt SAM2 with detection boxes → get initial masks + track IDs
    Frame 1+:  propagate() → SAM2 tracks masks forward (no re-detection needed)

Falls back to per-frame ultralytics SAM2 if real-time fork is unavailable.

Classes:
    SAM2Tracker — prompt-once, track-forward SAM2 wrapper
"""

import numpy as np
import torch
import cv2

from constants import SAM_MODEL, DEVICE


# ─── SAM2 Real-time Tracker ───────────────────────────────────────────────────

class SAM2Tracker:
    """SAM2 real-time tracking: prompt first frame, propagate forward.

    This is dramatically faster than per-frame segmentation because SAM2
    only encodes the image once and propagates masks using temporal memory.
    """

    def __init__(self, device: str = DEVICE):
        """Initialize SAM2 tracker.

        Args:
            device: Inference device (e.g. "0" for GPU).
        """
        self.device = device
        self.predictor = None
        self._prompted = False
        self._use_realtime = False
        self._ultralytics_model = None

        # Try loading SAM2 real-time fork first
        try:
            from sam2.build_sam import build_sam2_camera_predictor
            import os

            # Find checkpoint and config
            sam2_dir = self._find_sam2_dir()
            if sam2_dir:
                checkpoint = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_tiny.pt")
                config = "configs/sam2.1/sam2.1_hiera_t.yaml"

                if os.path.exists(checkpoint):
                    print(f"[segmentation] Loading SAM2 real-time (tiny): {checkpoint}")
                    self.predictor = build_sam2_camera_predictor(config, checkpoint)
                    self._use_realtime = True
                    return

            print("[segmentation] SAM2 real-time checkpoint not found, falling back to ultralytics")
        except ImportError:
            print("[segmentation] SAM2 real-time not installed, falling back to ultralytics")

        # Fallback: ultralytics SAM2
        self._load_ultralytics_sam()

    def _find_sam2_dir(self) -> str | None:
        """Find the segment-anything-2-real-time directory."""
        import os
        candidates = [
            os.path.expanduser("~/segment-anything-2-real-time"),
            os.path.expanduser("~/my_new_project/segment-anything-2-real-time"),
            os.path.join(os.path.dirname(__file__), "segment-anything-2-real-time"),
        ]
        for d in candidates:
            if os.path.isdir(d):
                return d
        return None

    def _load_ultralytics_sam(self):
        """Load ultralytics SAM2 as fallback."""
        from ultralytics import SAM
        model_name = str(SAM_MODEL)
        print(f"[segmentation] Loading SAM2 (ultralytics): {model_name}")
        self._ultralytics_model = SAM(model_name)

    def prompt_first_frame(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        track_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prompt SAM2 with detection boxes on the first frame.

        Args:
            frame:      BGR image.
            boxes:      (N, 4) xyxy bounding boxes.
            track_ids:  (N,) track IDs. Auto-generated if None.

        Returns:
            (masks, tracker_ids) — initial masks and their IDs.
        """
        if track_ids is None:
            track_ids = np.arange(1, len(boxes) + 1)

        if self._use_realtime and self.predictor is not None:
            return self._prompt_realtime(frame, boxes, track_ids)
        else:
            return self._prompt_ultralytics(frame, boxes, track_ids)

    def _prompt_realtime(self, frame, boxes, track_ids):
        """Prompt using SAM2 real-time fork."""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(boxes, track_ids):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=bbox,
                )

        self._prompted = True
        # Get initial masks by propagating frame 0
        return self.propagate(frame)

    def _prompt_ultralytics(self, frame, boxes, track_ids):
        """Prompt using ultralytics SAM2 (per-frame fallback)."""
        self._prompted = True
        self._last_track_ids = track_ids
        masks = self._segment_ultralytics(frame, boxes)
        return masks, track_ids

    def propagate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate masks to the next frame.

        Args:
            frame: BGR image.

        Returns:
            (masks, tracker_ids) — propagated masks and their track IDs.
        """
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        if self._use_realtime and self.predictor is not None:
            return self._propagate_realtime(frame)
        else:
            # Ultralytics fallback: no real tracking, just return empty
            # (detection + ultralytics SAM handles this in main loop)
            return np.empty((0,) + frame.shape[:2], dtype=bool), np.empty(0, dtype=int)

    def _propagate_realtime(self, frame):
        """Propagate using SAM2 real-time fork."""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        if masks.ndim == 2:
            masks = masks[None, ...]

        return masks, tracker_ids

    def _segment_ultralytics(self, frame, boxes):
        """Per-frame segmentation using ultralytics SAM2."""
        if self._ultralytics_model is None or len(boxes) == 0:
            return []

        h, w = frame.shape[:2]
        results = self._ultralytics_model(
            frame,
            bboxes=boxes,
            device=self.device,
            verbose=False,
        )

        masks = []
        if results and results[0].masks is not None:
            for m in results[0].masks.data:
                mask = m.cpu().numpy().astype(np.uint8)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(mask.astype(bool))

        return masks

    @property
    def uses_realtime(self) -> bool:
        """Whether the SAM2 real-time fork is being used."""
        return self._use_realtime

    def reset(self):
        """Reset the tracker state."""
        self._prompted = False


# ─── Legacy API (for backward compatibility) ─────────────────────────────────

def load_segmentor(model_name: str | None = None, device: str = DEVICE):
    """Load SAM2 model (ultralytics fallback).

    Returns a SAM2Tracker instance.
    """
    return SAM2Tracker(device=device)


def segment_objects(model, frame, boxes, device=DEVICE):
    """Per-frame segmentation using ultralytics SAM2 (legacy API).

    Args:
        model:  SAM2Tracker instance or ultralytics SAM model.
        frame:  BGR image.
        boxes:  (N, 4) xyxy bounding boxes.
        device: Inference device.

    Returns:
        List of boolean masks.
    """
    if isinstance(model, SAM2Tracker):
        if model._ultralytics_model is not None:
            return model._segment_ultralytics(frame, boxes)
        return []

    # Direct ultralytics model
    from ultralytics import SAM
    if not isinstance(model, SAM):
        return []

    h, w = frame.shape[:2]
    results = model(frame, bboxes=boxes, device=device, verbose=False)
    masks = []
    if results and results[0].masks is not None:
        for m in results[0].masks.data:
            mask = m.cpu().numpy().astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask.astype(bool))
    return masks
