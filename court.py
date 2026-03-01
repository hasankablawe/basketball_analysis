"""
2D Court reference map — drawing and player plotting.

Functions:
    draw_court_2d       — render a clean 2D NBA court diagram
    draw_players_on_court — plot tracked player dots + IDs
    draw_ball_on_court  — plot ball position
    draw_trails         — draw fading movement trails
"""

import cv2
import numpy as np

from constants import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    MINIMAP_WIDTH,
    MINIMAP_HEIGHT,
    MINIMAP_COURT_COLOR,
    MINIMAP_LINE_COLOR,
    TRAIL_LENGTH,
    TRAIL_FADE,
)


def _ft_to_px(x_ft: float, y_ft: float, w: int = MINIMAP_WIDTH, h: int = MINIMAP_HEIGHT) -> tuple[int, int]:
    """Convert court feet → minimap pixel coordinates."""
    px = int(x_ft / COURT_LENGTH_FT * w)
    py = int((COURT_WIDTH_FT - y_ft) / COURT_WIDTH_FT * h)  # flip Y
    return px, py


def _draw_arc(img, center_ft, radius_ft, start_angle, end_angle, w, h, color, thickness=1):
    """Draw an arc on the court image in feet-space."""
    cx, cy = _ft_to_px(center_ft[0], center_ft[1], w, h)
    rx = int(radius_ft / COURT_LENGTH_FT * w)
    ry = int(radius_ft / COURT_WIDTH_FT * h)
    axes = (rx, ry)
    # OpenCV angles: 0=right, counter-clockwise
    cv2.ellipse(img, (cx, cy), axes, 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


def draw_court_2d(w: int = MINIMAP_WIDTH, h: int = MINIMAP_HEIGHT) -> np.ndarray:
    """Render a clean top-down 2D NBA court diagram.

    Returns:
        BGR image of size (h, w, 3).
    """
    court = np.full((h, w, 3), MINIMAP_COURT_COLOR, dtype=np.uint8)
    c = MINIMAP_LINE_COLOR
    t = 3  # line thickness — thicker at HD resolution

    # Court boundary
    cv2.rectangle(court, (0, 0), (w - 1, h - 1), c, t + 1)

    # Half-court line
    mid_x = w // 2
    cv2.line(court, (mid_x, 0), (mid_x, h), c, t)

    # Center circle (radius 6ft)
    center = _ft_to_px(47.0, 25.0, w, h)
    r_px = int(6.0 / COURT_LENGTH_FT * w)
    cv2.circle(court, center, r_px, c, t, cv2.LINE_AA)

    # ── Left side ────────────────────────────────────────────────────────
    # Paint (key)
    p1 = _ft_to_px(0.0, 17.0, w, h)
    p2 = _ft_to_px(19.0, 33.0, w, h)
    cv2.rectangle(court, p1, p2, c, t)

    # Free-throw circle (top half — inside key)
    ft_center_l = _ft_to_px(19.0, 25.0, w, h)
    ft_r = int(6.0 / COURT_WIDTH_FT * h)
    cv2.ellipse(court, ft_center_l, (ft_r, ft_r), 0, -90, 90, c, t, cv2.LINE_AA)

    # Restricted area arc (4ft radius)
    hoop_l = _ft_to_px(4.0, 25.0, w, h)
    ra_r = int(4.0 / COURT_WIDTH_FT * h)
    cv2.ellipse(court, hoop_l, (ra_r, ra_r), 0, -90, 90, c, t, cv2.LINE_AA)

    # Hoop
    cv2.circle(court, hoop_l, max(3, int(0.75 / COURT_LENGTH_FT * w)), c, -1, cv2.LINE_AA)

    # Backboard
    bb_l_top = _ft_to_px(0.0, 22.0, w, h)
    bb_l_bot = _ft_to_px(0.0, 28.0, w, h)
    cv2.line(court, bb_l_top, bb_l_bot, c, t + 1)

    # Three-point line (simplified as lines + arc)
    corner_len_ft = 14.0
    # Corner lines
    cv2.line(court, _ft_to_px(0, 3, w, h), _ft_to_px(corner_len_ft, 3, w, h), c, t)
    cv2.line(court, _ft_to_px(0, 47, w, h), _ft_to_px(corner_len_ft, 47, w, h), c, t)
    # Arc
    _draw_arc(court, (4.0, 25.0), 23.75, -68, 68, w, h, c, t)

    # ── Right side (mirror) ──────────────────────────────────────────────
    # Paint
    p1r = _ft_to_px(75.0, 17.0, w, h)
    p2r = _ft_to_px(94.0, 33.0, w, h)
    cv2.rectangle(court, p1r, p2r, c, t)

    # Free-throw circle
    ft_center_r = _ft_to_px(75.0, 25.0, w, h)
    cv2.ellipse(court, ft_center_r, (ft_r, ft_r), 0, 90, 270, c, t, cv2.LINE_AA)

    # Restricted area arc
    hoop_r = _ft_to_px(90.0, 25.0, w, h)
    cv2.ellipse(court, hoop_r, (ra_r, ra_r), 0, 90, 270, c, t, cv2.LINE_AA)

    # Hoop
    cv2.circle(court, hoop_r, max(3, int(0.75 / COURT_LENGTH_FT * w)), c, -1, cv2.LINE_AA)

    # Backboard
    bb_r_top = _ft_to_px(94.0, 22.0, w, h)
    bb_r_bot = _ft_to_px(94.0, 28.0, w, h)
    cv2.line(court, bb_r_top, bb_r_bot, c, t + 1)

    # Three-point line
    cv2.line(court, _ft_to_px(94, 3, w, h), _ft_to_px(94 - corner_len_ft, 3, w, h), c, t)
    cv2.line(court, _ft_to_px(94, 47, w, h), _ft_to_px(94 - corner_len_ft, 47, w, h), c, t)
    _draw_arc(court, (90.0, 25.0), 23.75, 112, 248, w, h, c, t)

    return court


def draw_players_on_court(
    court_img: np.ndarray,
    positions: list[tuple[float, float]],
    colors: list[tuple[int, int, int]],
    track_ids: list[int],
    show_ids: bool = False,
) -> None:
    """Plot player dots + optional IDs on a court image (in-place).

    Args:
        court_img:  Court image (will be modified in-place).
        positions:  List of (x_ft, y_ft) court coordinates.
        colors:     BGR colour per player.
        track_ids:  Track ID per player.
        show_ids:   Whether to draw track ID text labels.
    """
    w, h = court_img.shape[1], court_img.shape[0]

    for pos, color, tid in zip(positions, colors, track_ids):
        px, py = _ft_to_px(pos[0], pos[1], w, h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        cv2.circle(court_img, (px, py), 14, color, -1, cv2.LINE_AA)
        cv2.circle(court_img, (px, py), 14, (255, 255, 255), 2, cv2.LINE_AA)
        if show_ids:
            cv2.putText(court_img, str(tid), (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_ball_on_court(court_img: np.ndarray, pos_ft: tuple[float, float]) -> None:
    """Plot ball position as an orange dot on court image (in-place)."""
    w, h = court_img.shape[1], court_img.shape[0]
    px, py = _ft_to_px(pos_ft[0], pos_ft[1], w, h)
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    cv2.circle(court_img, (px, py), 5, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.circle(court_img, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_shot_marker(
    court_img: np.ndarray,
    pos_ft: tuple[float, float],
    is_goal: bool,
) -> None:
    """Draw a GOAL or MISS marker on the court at the shot position (in-place).

    Args:
        court_img: Court image (will be modified in-place).
        pos_ft:    (x_ft, y_ft) shot position on court.
        is_goal:   True for goal, False for miss.
    """
    w, h = court_img.shape[1], court_img.shape[0]
    px, py = _ft_to_px(pos_ft[0], pos_ft[1], w, h)
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))

    if is_goal:
        color = (0, 255, 0)  # green
        label = "GOAL"
    else:
        color = (0, 0, 255)  # red
        label = "MISS"

    cv2.circle(court_img, (px, py), 10, color, 2, cv2.LINE_AA)
    cv2.putText(court_img, label, (px - 15, py - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_trails(
    court_img: np.ndarray,
    trail_history: dict[int, list[tuple[float, float]]],
    colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """Draw fading movement trails on court image (in-place).

    Args:
        court_img:     Court image (will be modified).
        trail_history: ``{track_id: [(x_ft, y_ft), ...]}`` — recent positions.
        colors:        ``{track_id: BGR}`` colour per player.

    Returns:
        Same image reference.
    """
    w, h = court_img.shape[1], court_img.shape[0]

    for tid, trail in trail_history.items():
        if len(trail) < 2:
            continue
        color = colors.get(tid, (200, 200, 200))
        n = len(trail)
        for i in range(1, n):
            if TRAIL_FADE:
                alpha = i / n  # older = more transparent
                c = tuple(int(v * alpha) for v in color)
            else:
                c = color
            p1 = _ft_to_px(trail[i - 1][0], trail[i - 1][1], w, h)
            p2 = _ft_to_px(trail[i][0], trail[i][1], w, h)
            cv2.line(court_img, p1, p2, c, 2, cv2.LINE_AA)

    return court_img
