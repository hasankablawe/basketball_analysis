"""
Game analytics — SigLIP+UMAP+K-means team clustering, goal detection, game state.

Team classification uses a FIT/PREDICT pattern:
    1. fit()     — collect player crops over first N frames, train UMAP+K-means
    2. predict() — classify new crops using the fitted model (no re-fitting)

Classes:
    TeamClassifier — SigLIP embeddings → UMAP → K-means (k=2)
    GameState      — tracks score, shots, per-player movement history

Functions:
    get_player_foot_pos  — estimate foot position from bounding box
    detect_shot          — check if ball is near a hoop
"""

from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from constants import (
    HOOP_LEFT,
    HOOP_RIGHT,
    SHOT_PROXIMITY_FT,
    SHOT_COOLDOWN_FRAMES,
    TRAIL_LENGTH,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    UNKNOWN_TEAM_COLOR,
)


# ─── Helper: center crop (like sv.scale_boxes) ──────────────────────────────

def _center_crop(frame: np.ndarray, box: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Crop the center portion of a bounding box (jersey area).

    Uses the same logic as sv.scale_boxes(factor=0.4) from the reference.
    This isolates the jersey while excluding head, legs, and background.

    Args:
        frame:  BGR image.
        box:    (4,) xyxy bounding box.
        factor: Fraction of the box to keep (0.4 = center 40%).

    Returns:
        Cropped BGR image, or a 32x32 black fallback if empty.
    """
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor

    cx1 = max(0, int(cx - w / 2))
    cy1 = max(0, int(cy - h / 2))
    cx2 = min(frame.shape[1], int(cx + w / 2))
    cy2 = min(frame.shape[0], int(cy + h / 2))

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return crop


# ─── Hue/Saturation Histogram Team Classifier ──────────────────────────────────

class HueTeamClassifier:
    """Cluster players into 2 teams using HSV (Hue & Saturation) color histograms.
    
    This replaces abstract embeddings (like SigLIP) with direct measurement of
    actual jersey colors, yielding significantly more robust team assignment.
    """

    def __init__(self):
        self._fitted = False
        self._kmeans = None
        self._collected_crops = []
        self._collected_tids = []
        self.team_map = {}
        
    def collect(self, frame: np.ndarray, boxes: np.ndarray, tids: np.ndarray) -> None:
        """Collect raw crops for future fitting."""
        for box, tid in zip(boxes, tids):
            # Same 40% center crop to grab just the torso/jersey
            crop = _center_crop(frame, box, factor=0.4)
            if crop is not None and crop.size > 0:
                self._collected_crops.append(crop)
                self._collected_tids.append(int(tid))

    def _extract_features(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Convert BGR crops into 26D HSV histogram features and calculate saturation."""
        crop_feats = []
        crop_sats = []
        valid_indices = []

        for i, crop in enumerate(crops):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Fore-ground: pixels that are not near-black
            fg = hsv[crop.max(axis=2) > 30]
            if len(fg) > 50:
                # 18-bin Hue (0-180) + 8-bin Saturation (0-256)
                h_hist = np.histogram(fg[:, 0], bins=18, range=(0, 180), density=True)[0]
                s_hist = np.histogram(fg[:, 1], bins=8,  range=(0, 256), density=True)[0]
                feat = np.concatenate([h_hist, s_hist])
                crop_feats.append(feat)
                crop_sats.append(fg[:, 1].mean())
                valid_indices.append(i)

        if not crop_feats:
            return np.array([]), np.array([]), []

        return np.array(crop_feats, dtype=np.float32), np.array(crop_sats, dtype=np.float32), valid_indices

    def fit(self, swap_teams: bool = False) -> dict[int, int]:
        """Fit the K-means model on collected crops and map each tracking ID to a team.
        
        Args:
            swap_teams: Force-flip team label assignment if colors appear swapped.
            
        Returns:
            Dictionary mapping tracking ID to team label (0 or 1).
        """
        from collections import Counter

        if len(self._collected_crops) < 10:
            print(f"[analytics] Not enough crops to fit ({len(self._collected_crops)}). Skipping classification.")
            return {}

        print(f"[analytics] Fitting HueTeamClassifier on {len(self._collected_crops)} crops...")
        
        features, saturations, valid_indices = self._extract_features(self._collected_crops)
        if len(features) < 10:
             print("[analytics] Not enough valid foreground crops to fit. Skipping classification.")
             return {}

        valid_tids = [self._collected_tids[i] for i in valid_indices]

        # Cluster all crops into 2 teams based on color distribution
        self._kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        crop_labels = self._kmeans.fit_predict(features)
        
        # Anchor teams: White/light jerseys = Team A (0), Colored = Team B (1)
        # by checking average saturation of each cluster.
        avg_sat = {
            0: saturations[crop_labels == 0].mean() if np.any(crop_labels == 0) else 128.0,
            1: saturations[crop_labels == 1].mean() if np.any(crop_labels == 1) else 128.0
        }
        
        print(f"[analytics] Cluster saturation: 0={avg_sat[0]:.3f}, 1={avg_sat[1]:.3f}")
        
        needs_flip = avg_sat[0] > avg_sat[1]
        if swap_teams:
            needs_flip = not needs_flip
            
        if needs_flip:
            crop_labels = 1 - crop_labels
            print("[analytics] Flipped clusters (assigning less saturated to Team A).")

        # Majority vote per track -> determines the player's true team
        tid_label_pairs = list(zip(valid_tids, crop_labels))
        
        # Get unique valid TIDs
        unique_tids = set(valid_tids)
        
        for tid in unique_tids:
            labels_for_tid = [l for t, l in tid_label_pairs if t == tid]
            if labels_for_tid:
                # Assign the most common label
                self.team_map[tid] = Counter(labels_for_tid).most_common(1)[0][0]

        n_a = sum(1 for v in self.team_map.values() if v == 0)
        n_b = sum(1 for v in self.team_map.values() if v == 1)
        print(f"[analytics] FINAL: Team A = {n_a} players, Team B = {n_b} players.")

        self._fitted = True
        return self.team_map

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ─── Game State ──────────────────────────────────────────────────────────────

class GameState:
    """Tracks score, shot attempts, and per-player movement history."""

    def __init__(self):
        self.score = {"A": 0, "B": 0}
        self.shots = {"A": {"made": 0, "missed": 0}, "B": {"made": 0, "missed": 0}}
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
        self.team_map: dict[int, str] = {}          # track_id → "A" | "B"
        self.team_colors: dict[int, tuple] = {}     # track_id → BGR
        self._shot_cooldown = 0
        self._ball_near_hoop = False
        self._last_near_hoop_side: str | None = None

    def update_trail(self, track_id: int, court_pos: tuple[float, float]):
        """Append a court position to the player's trail."""
        self.trails[track_id].append(court_pos)

    def get_trails_dict(self) -> dict[int, list[tuple[float, float]]]:
        """Return trails as plain dict of lists (for rendering)."""
        return {tid: list(dq) for tid, dq in self.trails.items()}

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        """Return team colour for a player."""
        return self.team_colors.get(track_id, UNKNOWN_TEAM_COLOR)


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_player_foot_pos(box: np.ndarray) -> tuple[float, float]:
    """Estimate foot position as the bottom-center of the bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def detect_shot(
    ball_court_pos: tuple[float, float] | None,
    game_state: GameState,
) -> str | None:
    """Check if the ball is near a hoop and update game state.

    Returns:
        ``"left_goal"``, ``"right_goal"``, or *None* if no event.
    """
    if game_state._shot_cooldown > 0:
        game_state._shot_cooldown -= 1
        return None

    if ball_court_pos is None:
        if game_state._ball_near_hoop and game_state._last_near_hoop_side:
            side = game_state._last_near_hoop_side
            game_state._ball_near_hoop = False
            game_state._shot_cooldown = SHOT_COOLDOWN_FRAMES
            return f"{side}_goal"
        return None

    bx, by = ball_court_pos
    dist_left = np.sqrt((bx - HOOP_LEFT[0]) ** 2 + (by - HOOP_LEFT[1]) ** 2)
    dist_right = np.sqrt((bx - HOOP_RIGHT[0]) ** 2 + (by - HOOP_RIGHT[1]) ** 2)

    near = False
    side = None
    if dist_left < SHOT_PROXIMITY_FT:
        near = True
        side = "left"
    elif dist_right < SHOT_PROXIMITY_FT:
        near = True
        side = "right"

    if near:
        game_state._ball_near_hoop = True
        game_state._last_near_hoop_side = side
    elif game_state._ball_near_hoop:
        event_side = game_state._last_near_hoop_side
        game_state._ball_near_hoop = False
        game_state._last_near_hoop_side = None
        game_state._shot_cooldown = SHOT_COOLDOWN_FRAMES
        return f"{event_side}_goal"

    return None


# ─── Path Smoothing (from reference: Savitzky-Golay filter) ──────────────────

def smooth_trail(
    trail: list[tuple[float, float]],
    window: int = 9,
    poly: int = 2,
) -> list[tuple[float, float]]:
    """Smooth a player's movement trail using Savitzky-Golay filter.

    Removes jitter and jump artifacts from court position trails.
    Falls back to simple moving average if scipy is not available.

    Args:
        trail:  List of (x, y) court positions.
        window: Smoothing window size (must be odd).
        poly:   Polynomial order for Savitzky-Golay.

    Returns:
        Smoothed trail as list of (x, y).
    """
    if len(trail) < window:
        return trail

    pts = np.array(trail)

    try:
        from scipy.signal import savgol_filter
        x_smooth = savgol_filter(pts[:, 0], window, poly)
        y_smooth = savgol_filter(pts[:, 1], window, poly)
    except ImportError:
        # Fallback: simple moving average
        kernel = np.ones(window) / window
        x_smooth = np.convolve(pts[:, 0], kernel, mode='same')
        y_smooth = np.convolve(pts[:, 1], kernel, mode='same')

    return list(zip(x_smooth.tolist(), y_smooth.tolist()))

