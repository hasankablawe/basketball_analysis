"""
Constants & configuration for Basketball AI Pipeline.
"""

import cv2
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DETECTOR_MODEL = PROJECT_DIR / "best.engine"
DETECTOR_FALLBACK = PROJECT_DIR / "RTdeatrL_best.pt"
SAM_MODEL = "sam2_b.pt"
COURT_POSE_MODEL = PROJECT_DIR / "pose_yolo26m.pt"

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "0"                        # "0" for GPU, "cpu" for CPU

# ─── Detection ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
COURT_KP_CONFIDENCE = 0.3

# ─── Court Dimensions (feet) ─────────────────────────────────────────────────
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0

# ─── 33 Reference Court Keypoints (feet, origin = bottom-left corner) ────────
COURT_REFERENCE_KEYPOINTS = [
    (0.0,  50.0),   #  0  Top-left corner
    (0.0,  47.0),   #  1  Left baseline, 3pt corner (top)
    (0.0,  33.0),   #  2  Left baseline, paint top
    (0.0,  17.0),   #  3  Left baseline, paint bottom
    (0.0,   3.0),   #  4  Left baseline, 3pt corner (bottom)
    (0.0,   0.0),   #  5  Bottom-left corner
    (0.0,  25.0),   #  6  Left baseline center
    (14.0, 47.0),   #  7  Left 3pt corner line end (top)
    (14.0,  3.0),   #  8  Left 3pt corner line end (bottom)
    (19.0, 33.0),   #  9  Left FT line, paint top
    (19.0, 25.0),   # 10  Left FT line center
    (19.0, 17.0),   # 11  Left FT line, paint bottom
    (19.0, 50.0),   # 12  Left FT line extended top
    (29.0, 25.0),   # 13  Left 3pt arc apex
    (19.0,  0.0),   # 14  Left FT line extended bottom
    (47.0, 50.0),   # 15  Half-court top
    (47.0, 25.0),   # 16  Center circle center
    (47.0,  0.0),   # 17  Half-court bottom
    (75.0, 50.0),   # 18  Right FT line extended top
    (65.0, 25.0),   # 19  Right 3pt arc apex
    (75.0,  0.0),   # 20  Right FT line extended bottom
    (75.0, 33.0),   # 21  Right FT line, paint top
    (75.0, 25.0),   # 22  Right FT line center
    (75.0, 17.0),   # 23  Right FT line, paint bottom
    (80.0, 47.0),   # 24  Right 3pt corner line end (top)
    (80.0,  3.0),   # 25  Right 3pt corner line end (bottom)
    (94.0, 25.0),   # 26  Right baseline center
    (94.0, 50.0),   # 27  Top-right corner
    (94.0, 47.0),   # 28  Right baseline, 3pt corner (top)
    (94.0, 33.0),   # 29  Right baseline, paint top
    (94.0, 17.0),   # 30  Right baseline, paint bottom
    (94.0,  3.0),   # 31  Right baseline, 3pt corner (bottom)
    (94.0,  0.0),   # 32  Bottom-right corner
]

# ─── Hoop Positions (feet) ───────────────────────────────────────────────────
HOOP_LEFT = (4.0, 25.0)             # left basket center
HOOP_RIGHT = (90.0, 25.0)           # right basket center
THREE_POINT_RADIUS_FT = 23.75       # 3-point arc distance from hoop

# ─── Shot Detection ─────────────────────────────────────────────────────────
SHOT_PROXIMITY_FT = 8.0             # ball must be within this of hoop
SHOT_COOLDOWN_FRAMES = 45           # ignore shots for N frames after one

# ─── Tracking & Classification ───────────────────────────────────────────────
TEAM_CLASSIFICATION_STRIDE = 50     # sample frames for clustering
JUMP_RISE_THRESHOLD = 25            # px jump rise threshold

# ─── Team Colours (BGR for OpenCV) ───────────────────────────────────────────
TEAM_A_COLOR = (60, 220, 0)         # vivid lime-green  (BGR)
TEAM_B_COLOR = (255, 100, 0)        # electric blue      (BGR)
UNKNOWN_TEAM_COLOR = (180, 180, 180)

# ─── Visualization ──────────────────────────────────────────────────────────
COLOR_PALETTE = [
    (80, 80, 255), (255, 80, 80), (80, 255, 80),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
    (200, 128, 80), (128, 80, 200),
]
MASK_ALPHA = 0.3
FONT = cv2.FONT_HERSHEY_SIMPLEX
MINIMAP_MARGIN = 10

# ─── Minimap ─────────────────────────────────────────────────────────────────
MINIMAP_WIDTH = 600
MINIMAP_HEIGHT = 320
MINIMAP_COURT_COLOR = (0, 0, 0)       # pure black court fill
MINIMAP_LINE_COLOR = (255, 255, 255)  # pure white court lines

# ─── Trails ──────────────────────────────────────────────────────────────────
TRAIL_LENGTH = 90
TRAIL_FADE = True

# ─── Output ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "Basketball AI"
OUTPUT_FILENAME = "output.mp4"
OUTPUT_CODEC = "mp4v"
