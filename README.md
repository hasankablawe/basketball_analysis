# basketball_analysis
=======
# Basketball AI Analytics Pipeline

This repository contains a modular, state-of-the-art computer vision pipeline for basketball video analysis. It detects and tracks players in real-time, projects their positions onto a 2D court representation, classifies teams based on jersey colors, and detects shot events (made or missed).

## Features

- **Robust Object Detection**: Uses RT-DETR for fast and accurate player, action (jump shot/layup), and basketball detection.
- **Pixel-Perfect Tracking**: Integrates Meta's Segment Anything Model 2 (SAM 2). Objects are detected once and their pixel masks are propagated temporally via SAM 2 for highly robust tracking.
- **2D Court Mapping (Homography)**: Employs a custom YOLO pose model to detect 33 court keypoints, computing a homography matrix to project 3D camera coordinates onto a top-down 2D minimap.
- **True-Color Team Classification**: Extracts the exact SAM 2 silhouette of a player, eliminating background noise, and uses K-Means clustering in HSV color space to reliably assign players to their respective teams.
- **Shot Event State Machine**: A robust frame-to-frame state machine (`ShotStateMachine`) tracks the lifecycle of a shot (from jump to "ball in basket") to correctly identify points scored and update the scoreboard.
- **Smart Trail Smoothing**: Applies mathematical smoothing and jump suppression to remove tracking jitter, producing clean visualization paths on the minimap.

## Architecture & Code Structure

The project has been refactored from a monolithic notebook script into a clean, maintainable software architecture:

- `main.py`: The central orchestrator containing the video processing loop, logic sync, and state machines.
- `constants.py`: The single source of truth for all "magic numbers"—model paths, confidence thresholds, API config, court dimensions, and team colors (editable here).
- `detection.py`: RT-DETR object detection and bounding-box assignment.
- `segmentation.py`: SAM 2 mask generation and real-time temporal propagation.
- `homography.py`: YOLO keypoint detection, homography matrix computation, and perspective transformation math.
- `analytics.py`: Team classification clustering mathematically.
- `court.py` & `visualization.py`: 2D minimap drawing, trail rendering, and on-screen overlays.

## Requirements

Ensure you have a modern GPU with CUDA. Key dependencies (see `requirements.txt` for full list) include:

- `torch` & `torchvision`
- `ultralytics`
- `supervision`
- `opencv-python`
- Meta's Segment Anything 2 (SAM 2) real-time fork

## Usage

Run the analysis by passing a source video to the main script. Output files are saved locally.

```bash
# Process a video and save the dual output
python main.py NBA.mp4 --save

# Start processing at 10 seconds and run for 30 seconds
python main.py NBA.mp4 --start 10 --duration 30 --save

# If team colors appear mapped backwards on the minimap, flip them:
python main.py NBA.mp4 --save --swap-teams
```

The script generates two high-definition MP4 files simultaneously:
1. `output_tracking.mp4`: The broadcast video rendered with SAM 2 silhouette overlays.
2. `output_map.mp4`: A mathematically smoothed 2D animated top-down view showing player trails and an automated scoreboard.
>>>>>>> c071802 (Initial commit for Basketball Analysis project)
