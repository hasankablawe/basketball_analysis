"""
Visualization utilities for drawing annotations on video frames.

Uses the `supervision` library when available for professional-quality annotations.
Falls back to manual OpenCV drawing if supervision is not installed.

Functions:
    overlay_mask       — blend a binary segmentation mask onto a frame
    overlay_minimap    — composite 2D court minimap onto bottom-center of frame
    draw_score_overlay — draw scoreboard with team scores and shot stats
    annotate_frame_sv  — full-frame annotation using supervision (masks + labels)
"""

import cv2
import numpy as np

from constants import (
    COLOR_PALETTE,
    MASK_ALPHA,
    FONT,
    MINIMAP_MARGIN,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
)

# ─── Try loading supervision ─────────────────────────────────────────────────

try:
    import supervision as sv
    _HAS_SV = True
except ImportError:
    _HAS_SV = False
    print("[visualization] supervision library not found, using OpenCV fallback")


# ─── Supervision-based annotation ────────────────────────────────────────────

def annotate_frame_sv(
    frame: np.ndarray,
    player_masks: list[np.ndarray],
    player_colors: list[tuple],
    player_labels: list[str],
    ball_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Annotate a frame using supervision library (masks + labels).

    Args:
        frame:         BGR image.
        player_masks:  List of boolean masks for players.
        player_colors: BGR color per player (team color).
        player_labels: Label string per player (e.g. "#7 Brown").
        ball_masks:    Optional list of ball masks (orange).

    Returns:
        Annotated frame.
    """
    if not _HAS_SV or not player_masks:
        return frame

    annotated = frame.copy()

    # Build supervision Detections for players
    masks_np = np.array(player_masks, dtype=bool)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    xyxy = sv.mask_to_xyxy(masks=masks_np)

    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks_np,
        tracker_id=np.arange(len(player_masks)),
    )

    # Team color palette
    hex_colors = []
    for bgr in player_colors:
        r, g, b = bgr[2], bgr[1], bgr[0]
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    if hex_colors:
        palette = sv.ColorPalette.from_hex(hex_colors)
        mask_annotator = sv.MaskAnnotator(
            color=palette,
            opacity=MASK_ALPHA,
            color_lookup=sv.ColorLookup.INDEX,
        )
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    # Ball masks (orange)
    if ball_masks:
        ball_masks_np = np.array(ball_masks, dtype=bool)
        if ball_masks_np.ndim == 2:
            ball_masks_np = ball_masks_np[None, ...]
        ball_xyxy = sv.mask_to_xyxy(masks=ball_masks_np)
        ball_det = sv.Detections(xyxy=ball_xyxy, mask=ball_masks_np)
        ball_palette = sv.ColorPalette.from_hex(["#FF8C00"])
        ball_annotator = sv.MaskAnnotator(
            color=ball_palette,
            opacity=MASK_ALPHA,
            color_lookup=sv.ColorLookup.INDEX,
        )
        annotated = ball_annotator.annotate(scene=annotated, detections=ball_det)

    return annotated


# ─── OpenCV fallback functions ────────────────────────────────────────────────

def get_color(track_id: int) -> tuple:
    """Return a deterministic colour for the given track ID."""
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """Blend a binary mask onto *frame* with *color* at *alpha* transparency."""
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask > 0] = color
    return cv2.addWeighted(colored, alpha, frame, 1.0, 0)


def overlay_minimap(
    frame: np.ndarray,
    court_img: np.ndarray,
) -> None:
    """Composite a 2D court minimap at the bottom-center of the frame (in-place).

    Args:
        frame:     BGR video frame.
        court_img: Rendered court image.
    """
    fh, fw = frame.shape[:2]
    mh, mw = court_img.shape[:2]
    m = MINIMAP_MARGIN

    # Bottom-center
    x1 = (fw - mw) // 2
    y1 = fh - mh - m

    x2, y2 = x1 + mw, y1 + mh

    # Semi-transparent background
    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(court_img, 0.85, roi, 0.15, 0)
    frame[y1:y2, x1:x2] = blended

    # Border
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2, y2), (255, 255, 255), 1)


def draw_score_overlay(
    frame: np.ndarray,
    game_state,
) -> None:
    """Draw a scoreboard overlay at the top-center of the frame (in-place).

    Shows team scores and shot stats (made/attempted).
    """
    fh, fw = frame.shape[:2]

    # Scoreboard background
    sb_w, sb_h = 320, 60
    sb_x = (fw - sb_w) // 2
    sb_y = 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Team A
    score_a = game_state.score["A"]
    shots_a = game_state.shots["A"]
    a_text = f"Team A: {score_a}"
    cv2.putText(frame, a_text, (sb_x + 15, sb_y + 25),
                FONT, 0.7, TEAM_A_COLOR, 2, cv2.LINE_AA)
    a_detail = f"{shots_a['made']}/{shots_a['made'] + shots_a['missed']} shots"
    cv2.putText(frame, a_detail, (sb_x + 15, sb_y + 48),
                FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Separator
    cv2.line(frame, (sb_x + sb_w // 2, sb_y + 5),
             (sb_x + sb_w // 2, sb_y + sb_h - 5), (100, 100, 100), 1)

    # Team B
    score_b = game_state.score["B"]
    shots_b = game_state.shots["B"]
    b_text = f"Team B: {score_b}"
    cv2.putText(frame, b_text, (sb_x + sb_w // 2 + 15, sb_y + 25),
                FONT, 0.7, TEAM_B_COLOR, 2, cv2.LINE_AA)
    b_detail = f"{shots_b['made']}/{shots_b['made'] + shots_b['missed']} shots"
    cv2.putText(frame, b_detail, (sb_x + sb_w // 2 + 15, sb_y + 48),
                FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
