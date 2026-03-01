"""
Basketball AI — Detection, Tracking, Team ID, Court Mapping,
                Shot Detection, Shot Chart, Path Smoothing.

Uses SAM2 real-time video predictor for tracking (prompt-once → propagate).
RT-DETR runs periodically to find new players and re-prompt SAM2.

Usage:
    python main.py NBA.mp4 --start 10 --duration 30 --save
    python main.py NBA.mp4 --start 10 --duration 30 --save --no-sam
"""

import argparse, sys, os
import cv2, numpy as np, torch, tempfile, shutil
import supervision as sv
from collections import defaultdict, Counter
from sklearn.cluster import KMeans as _KMeans
from ultralytics import RTDETR, YOLO



from constants import (
    PROJECT_DIR, DETECTOR_MODEL, DETECTOR_FALLBACK,
    COURT_POSE_MODEL, CONFIDENCE_THRESHOLD, COURT_KP_CONFIDENCE,
    COURT_REFERENCE_KEYPOINTS, MINIMAP_WIDTH, MINIMAP_HEIGHT,
    OUTPUT_CODEC, TRAIL_LENGTH,
    HOOP_LEFT, HOOP_RIGHT, THREE_POINT_RADIUS_FT,
    TEAM_A_COLOR, TEAM_B_COLOR, TEAM_CLASSIFICATION_STRIDE, JUMP_RISE_THRESHOLD
)
from court import draw_court_2d, draw_players_on_court, draw_trails, draw_shot_marker

# ── Class IDs ────────────────────────────────────────────────────────────────
BALL_IN_BASKET_ID = 1
PLAYER_IDS = [3, 4, 5, 6, 7]
JUMP_SHOT_ID = 5
LAYUP_DUNK_ID = 6

# ── SAM2 Tracker Config ──────────────────────────────────────────────────────
SAM2_DIR = PROJECT_DIR / "segment-anything-2"
SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"



def parse_args():
    p = argparse.ArgumentParser(description="Basketball AI")
    p.add_argument("source", nargs="?", default="0")
    p.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--save", action="store_true")
    # Ignored legacy flags to avoid breaking existing bash commands
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--no-homography", action="store_true")
    p.add_argument("--swap-teams", action="store_true",
                   help="Force-flip team label assignment if colors appear swapped")
    return p.parse_args()


class ShotStateMachine:
    """
    Exactly implements the shot detection state diagram:

    IDLE ──[jump_frames>=3 OR layup_frames>=3]──▶ SHOT IN PROGRESS
                                                 emit START, set deadline
    
    SHOT IN PROGRESS ──[ball_in_basket_frames>=3]──▶ IDLE
                                                    emit MADE
    
    SHOT IN PROGRESS ──[jump_frames>=3 OR layup_frames>=3]──▶ SHOT IN PROGRESS
                                                              emit MISSED, emit START, set deadline

    SHOT IN PROGRESS ──[frame_index >= deadline]──▶ IDLE
                                                   emit MISSED

    Also: ball_in_basket from IDLE (direct basket) instantly emits MADE.
    """

    JUMP_THRESHOLD  = 3
    LAYUP_THRESHOLD = 3
    BIB_THRESHOLD   = 3

    def __init__(self, deadline_frames: int, cooldown_frames: int):
        self.deadline_frames  = deadline_frames  # how many frames a shot event waits
        self.cooldown_frames  = cooldown_frames  # frames to suppress new shots after MADE

        self._in_progress     = False
        self._deadline        = None
        self._last_made_frame = None

        self._jump_ct  = 0
        self._layup_ct = 0
        self._bib_ct   = 0

    def update(self, frame_idx: int, has_jump: bool, has_layup: bool, has_bib: bool):
        events = []

        # Increment / reset consecutive counters
        self._jump_ct  = self._jump_ct  + 1 if has_jump  else 0
        self._layup_ct = self._layup_ct + 1 if has_layup else 0
        self._bib_ct   = self._bib_ct   + 1 if has_bib   else 0

        shot_trigger = (self._jump_ct  >= self.JUMP_THRESHOLD or
                        self._layup_ct >= self.LAYUP_THRESHOLD)
        bib_trigger  =  self._bib_ct   >= self.BIB_THRESHOLD

        # ── Post-MADE cooldown blocks new shots ──────────────────────────────
        in_cooldown = (self._last_made_frame is not None and
                       frame_idx - self._last_made_frame < self.cooldown_frames)

        if not self._in_progress:
            # ━━ IDLE STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if bib_trigger and not in_cooldown:
                # Direct basket — MADE without a prior shot start
                events.append({"event": "MADE", "frame": frame_idx})
                self._last_made_frame = frame_idx
                self._reset_counters()

            elif shot_trigger and not in_cooldown:
                # Transition IDLE → SHOT IN PROGRESS
                self._in_progress = True
                self._deadline    = frame_idx + self.deadline_frames
                self._reset_counters()
                events.append({"event": "START", "frame": frame_idx})

        else:
            # ━━ SHOT IN PROGRESS STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if bib_trigger:
                # → MADE: ball confirmed in basket
                events.append({"event": "MADE", "frame": frame_idx})
                self._last_made_frame = frame_idx
                self._in_progress     = False
                self._deadline        = None
                self._reset_counters()

            elif shot_trigger and not in_cooldown:
                # New jump/layup while already in progress → MISSED + new START
                events.append({"event": "MISSED", "frame": frame_idx})
                self._deadline = frame_idx + self.deadline_frames
                self._reset_counters()
                events.append({"event": "START", "frame": frame_idx})

            elif frame_idx >= self._deadline:
                # Deadline exceeded → MISSED, back to IDLE
                events.append({"event": "MISSED", "frame": frame_idx})
                self._in_progress = False
                self._deadline    = None
                self._reset_counters()

        return events

    def _reset_counters(self):
        self._jump_ct  = 0
        self._layup_ct = 0
        self._bib_ct   = 0

    @property
    def in_progress(self):
        return self._in_progress


def _distance_from_hoop(court_xy):
    dl = np.sqrt((court_xy[0] - HOOP_LEFT[0])**2 + (court_xy[1] - HOOP_LEFT[1])**2)
    dr = np.sqrt((court_xy[0] - HOOP_RIGHT[0])**2 + (court_xy[1] - HOOP_RIGHT[1])**2)
    if dl < dr:
        return dl, "left"
    return dr, "right"


def main():
    args = parse_args()
    print("[main] Initializing Batch Pipeline...")

    # 1. Models
    det_path = DETECTOR_MODEL if DETECTOR_MODEL.exists() else DETECTOR_FALLBACK
    detector = RTDETR(str(det_path))
    court_model = YOLO(str(COURT_POSE_MODEL))
    
    sys.path.insert(0, str(SAM2_DIR))

    from analytics import HueTeamClassifier, smooth_trail
    from sports import clean_paths
    base_court = draw_court_2d(MINIMAP_WIDTH, MINIMAP_HEIGHT)

    # 2. Extract Frames
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    max_frames = int(args.duration * fps) if args.duration > 0 else 0

    # ── RAM optimization: downscale to 720p before writing to disk ──────────
    # SAM2 resizes input to 1024px internally anyway — no quality loss.
    # Smaller temp frames = smaller SAM2 inference_state["images"] in RAM.
    # Native resolution — swap space available, process at full quality
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = tempfile.mkdtemp()
        frame_count = 0
        frame_0 = None
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and frame_count >= max_frames): break
            if frame_count == 0:
                frame_0 = frame.copy()
            cv2.imwrite(os.path.join(temp_dir, f"{frame_count:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_count += 1
        cap.release()
        if frame_count == 0: sys.exit("[main] No frames read")
        print(f"[main] Extracted {frame_count} frames at {w}x{h} to disk.")

        # 3. Phase 1: Initialize Frame 0 (Detect & Classify Teams ONCE)
        print("[main] Phase 1/3: Initializing tracks & teams from Frame 0...")
        res0 = detector.predict(frame_0, conf=args.conf, iou=0.9, verbose=False)[0]
        dets0 = sv.Detections.from_ultralytics(res0)
        dets0 = dets0[np.isin(dets0.class_id, PLAYER_IDS)]
        
        # Exclude refs / noisy boxes, take top players by confidence
        if len(dets0) > 10:
            dets0 = dets0[np.argsort(dets0.confidence)[-10:]]
        dets0.tracker_id = np.arange(1, len(dets0) + 1)
        # FREE UP VRAM BEFORE SAM2! (We will classify teams in Phase 3 now using robust Masks)
        del detector
        del court_model
        torch.cuda.empty_cache()

        # 4. Phase 2: Propagate SAM2 tracks natively & collect Court XY & Crops
        print(f"[main] Phase 2/3: Tracking {frame_count} frames with SAM2 Official API...")
        
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda" if args.device == "0" else "cpu")

        masks_dir = os.path.join(temp_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
            
        print("[main] Loading fallback detector to CPU for periodic tracker recovery...")
        recovery_detector = RTDETR(str(DETECTOR_FALLBACK))
        next_tracker_id = int(dets0.tracker_id.max()) + 1 if len(dets0) > 0 else 1
        player_crops = defaultdict(list)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            predictor.reset_state(inference_state)
            
            valid_box_count = 0
            for xyxy, obj_id in zip(dets0.xyxy, dets0.tracker_id):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=int(obj_id),
                    box=np.asarray([xyxy], dtype=np.float32)[0]
                )
                valid_box_count += 1
                
            detections_history = [sv.Detections.empty()] * frame_count
            
            if valid_box_count > 0:
                start_frame = 0
                while start_frame < frame_count:
                    broke_early = False
                    for out_idx, out_ids, out_masks in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame):
                        masks = (out_masks > 0.0).cpu().numpy().squeeze(1).astype(bool)
                        if masks.ndim == 2:
                            masks = masks[None, ...]
                        
                        masks = np.array([
                            sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
                            for m in masks
                        ])
                        xyxy = sv.mask_to_xyxy(masks=masks)
                        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
                        
                        final_masks = masks if valid.all() else masks[valid]
                        final_xyxy = xyxy if valid.all() else xyxy[valid]
                        final_ids = np.asarray(out_ids, dtype=np.int32) if valid.all() else np.asarray(out_ids, dtype=np.int32)[valid]
                        
                        np.savez_compressed(os.path.join(masks_dir, f"{out_idx:05d}.npz"), masks=final_masks)
                        detections_history[out_idx] = sv.Detections(xyxy=final_xyxy, tracker_id=final_ids)
                        
                        # 1. Mask-Aware Crop Sampling (Every 15 frames, stored small for RAM)
                        if out_idx % 15 == 0:
                            frame_img = cv2.imread(os.path.join(temp_dir, f"{out_idx:05d}.jpg"))
                            if frame_img is not None:
                                for m, hid in zip(final_masks, final_ids):
                                    y_ind, x_ind = np.where(m)
                                    if len(x_ind) > 0:
                                        x1, x2 = x_ind.min(), x_ind.max()
                                        y1, y2 = y_ind.min(), y_ind.max()
                                        crop = frame_img[y1:y2+1, x1:x2+1].copy()
                                        crop[~m[y1:y2+1, x1:x2+1]] = 0
                                        if len(player_crops[hid]) < 40:  # more samples = better SigLIP clustering
                                            # Resize to 96x96 — enough for SigLIP, saves ~80% RAM
                                            crop_small = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
                                            player_crops[hid].append(crop_small)
                                            
                        if out_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            # Clear SAM2's internal frame feature cache to free CPU RAM
                            # (keeps only a rolling window of recent features needed for tracking)
                            if "cached_features" in inference_state:
                                # Keep last 10 frames of features; drop older ones
                                cached = inference_state["cached_features"]
                                keys_to_drop = sorted(cached.keys())[:-30]  # keep last 30 frames for stronger continuity
                                for k in keys_to_drop:
                                    del cached[k]
                            
                        # 2. Periodic Tracker Recovery (Every 60 frames)
                        if out_idx > start_frame and out_idx % 60 == 0:
                            frame_img = cv2.imread(os.path.join(temp_dir, f"{out_idx:05d}.jpg"))
                            if frame_img is not None:
                                rec_res = recovery_detector.predict(frame_img, conf=args.conf, iou=0.9, verbose=False, device="cpu")[0]
                                rec_dets = sv.Detections.from_ultralytics(rec_res)
                                rec_dets = rec_dets[np.isin(rec_dets.class_id, PLAYER_IDS)]
                                
                                injected = False
                                for r_xyxy in rec_dets.xyxy:
                                    is_new = True
                                    if len(final_xyxy) > 0:
                                        ious = sv.box_iou_batch(np.array([r_xyxy]), final_xyxy)[0]
                                        if ious.max() > 0.3:
                                            is_new = False
                                            
                                    if is_new:
                                        predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=out_idx,
                                            obj_id=next_tracker_id,
                                            box=np.asarray([r_xyxy], dtype=np.float32)[0]
                                        )
                                        next_tracker_id += 1
                                        injected = True
                                        
                                if injected:
                                    print(f"[main] Recovered lost players at frame {out_idx}. Restarting tracker generator...")
                                    start_frame = out_idx
                                    broke_early = True
                                    break
                                    
                    if not broke_early:
                        break
        
        print("[main] SAM2 Propagation Complete.")
        
        # ── Phase 2.5: Maximum-Accuracy Team Classification ─────────────────────
        # Triple-anchor system:
        #   Signal 1 — SigLIP semantic embedding KMeans (reference notebook approach)
        #   Signal 2 — HSV Hue histogram KMeans  (measures actual jersey color directly)
        #   Signal 3 — Court-side position at frame 0 (left half = Team A, right = Team B)
        # Final flip decision is a MAJORITY VOTE across all three signals.
        # --swap-teams inverts the final majority decision.
        # ─────────────────────────────────────────────────────────────────────────
        STRIDE = 30
        print(f"[main] Phase 2.5: Collecting jersey crops at stride={STRIDE}...")

        # ── Phase 2.5: High-Accuracy Team Classification (Histogram KMeans) ──────
        # Replaced earlier approaches: SigLIP abstracts too much; 5v5 heuristics fail 
        # when subs/refs/occlusions happen. The purest signal is ACTUAL COLOR. 
        # We cluster 26D vectors of (18-bin Hue + 8-bin Saturation) directly.
        print(f"[main] Phase 2.5: Collecting jersey crops at stride={TEAM_CLASSIFICATION_STRIDE}...")

        # ── Referee / outlier filter ──────────────────────────────────────────────
        # Reject tracks whose median box area is surprisingly small (refs/ball-boys)
        track_areas = defaultdict(list)
        for fi in range(0, frame_count, TEAM_CLASSIFICATION_STRIDE):
            sam_dets_fi = detections_history[fi]
            for tid, xyxy in zip(sam_dets_fi.tracker_id, sam_dets_fi.xyxy):
                w_box = xyxy[2] - xyxy[0]
                h_box = xyxy[3] - xyxy[1]
                track_areas[int(tid)].append(float(w_box * h_box))

        median_areas = {tid: float(np.median(v)) for tid, v in track_areas.items() if v}
        if median_areas:
            area_threshold = np.percentile(list(median_areas.values()), 10)
        else:
            area_threshold = 0.0

        team_classifier = HueTeamClassifier()
        
        # Collect valid crops
        for fi in range(0, frame_count, TEAM_CLASSIFICATION_STRIDE):
            sam_dets_fi = detections_history[fi]
            if len(sam_dets_fi) == 0:
                continue
            
            # Filter detections for valid tracks
            valid_mask = np.array([median_areas.get(int(tid), 0) >= area_threshold for tid in sam_dets_fi.tracker_id])
            if not np.any(valid_mask):
                continue
                
            valid_dets = sam_dets_fi[valid_mask]
            frame_fi = cv2.imread(os.path.join(temp_dir, f"{fi:05d}.jpg"))
            if frame_fi is not None:
                team_classifier.collect(frame_fi, valid_dets.xyxy, valid_dets.tracker_id)

        TEAMS_DICT = team_classifier.fit(swap_teams=getattr(args, 'swap_teams', False))

        MAX_TRACKER_ID = next_tracker_id - 1

        # FREE UP VRAM BEFORE YOLOs!
        del recovery_detector
        del predictor
        inference_state = None
        torch.cuda.empty_cache()
        
        # RE-LOAD DETECTION MODELS
        detector = RTDETR(str(det_path))
        court_model = YOLO(str(COURT_POSE_MODEL))
        
        shot_tracker = ShotStateMachine(
            deadline_frames=int(fps * 1.7),
            cooldown_frames=int(fps * 0.5),
        )
        
        video_xy_raw = np.full((frame_count, MAX_TRACKER_ID, 2), np.nan)
        H_history = []
        shot_markers = []
        score = {"A": 0, "B": 0}
        
        REF_KP = np.array(COURT_REFERENCE_KEYPOINTS, dtype=np.float64)
        out_H = None
        
        from tqdm import tqdm
        print("[main] Computing homography & shots...")
        with torch.inference_mode():
            for i in tqdm(range(frame_count)):
                frame = cv2.imread(os.path.join(temp_dir, f"{i:05d}.jpg"))
                sam_dets = detections_history[i]
                
                # b) Find homography
                kr = court_model(frame, device=args.device, verbose=False)
                if kr and kr[0].keypoints is not None:
                    kd = kr[0].keypoints.data
                    if kd is not None and len(kd) > 0:
                        kps = kd[0].cpu().numpy()
                        xy, conf = kps[:, :2], kps[:, 2]
                        m = min(len(xy), len(REF_KP), len(conf))
                        ok_mask = conf[:m] >= COURT_KP_CONFIDENCE
                        if ok_mask.sum() >= 6:
                            src, dst = xy[:m][ok_mask].astype(np.float64), REF_KP[:m][ok_mask]
                            new_H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                            if new_H is not None:
                                out_H = new_H
        
                H_history.append(out_H)
                
                # c) Project SAM bottoms to court
                if out_H is not None and len(sam_dets) > 0:
                    foot = sam_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    pts_h = np.hstack([foot.astype(np.float64), np.ones((len(foot), 1))])
                    proj = (out_H @ pts_h.T).T
                    denom = proj[:, 2:3]; denom[np.abs(denom) < 1e-8] = 1e-8
                    ct_xy = proj[:, :2] / denom
                    
                    for j, tid in enumerate(sam_dets.tracker_id):
                        idx = tid - 1
                        if 0 <= idx < MAX_TRACKER_ID:
                            video_xy_raw[i, idx] = ct_xy[j]
                
                # d) Shot detection
                res = detector.predict(frame, conf=args.conf, iou=0.9, verbose=False, device=args.device)[0]
                rt_dets = sv.Detections.from_ultralytics(res)
                has_js = (rt_dets.class_id == JUMP_SHOT_ID).any()
                has_ld = (rt_dets.class_id == LAYUP_DUNK_ID).any()
                has_bib = (rt_dets.class_id == BALL_IN_BASKET_ID).any()
                
                # Match shooter location to SAM bottom center to guess team
                shooter_team = "A"
                shooter_xy_ct = None
                if has_js or has_ld:
                    shooters = rt_dets[np.isin(rt_dets.class_id, [JUMP_SHOT_ID, LAYUP_DUNK_ID])]
                    if len(shooters) > 0:
                        s_xy = shooters.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                        if out_H is not None:
                            s_pt = np.array([[s_xy]], dtype=np.float64)
                            shooter_xy_ct = cv2.perspectiveTransform(s_pt, out_H)[0][0]
                            if len(sam_dets) > 0:
                                dists = np.linalg.norm((foot - s_xy), axis=1)
                                best_j = np.argmin(dists)
                                if dists[best_j] < 100:
                                    tid = sam_dets.tracker_id[best_j]
                                    shooter_team = "A" if TEAMS_DICT.get(int(tid), 0) == 0 else "B"
                
                # Fix: ball-in-basket directly implies a made shot - use closest player as shooter
                if has_bib and shooter_xy_ct is None and out_H is not None and len(sam_dets) > 0:
                    bibs = rt_dets[rt_dets.class_id == BALL_IN_BASKET_ID]
                    if len(bibs) > 0:
                        b_xy = bibs.get_anchors_coordinates(sv.Position.CENTER)[0]
                        b_pt = np.array([[b_xy]], dtype=np.float64)
                        bib_ct = cv2.perspectiveTransform(b_pt, out_H)[0][0]
                        # Nearest hoop determines which team scored (away team defends nearest hoop)
                        dl = np.linalg.norm(bib_ct - np.array(HOOP_LEFT))
                        dr = np.linalg.norm(bib_ct - np.array(HOOP_RIGHT))
                        # Pick approximate shot court position (just in front of closest hoop)
                        shooter_xy_ct = bib_ct.copy()
                        # The closest player in front of the hoop is likely the scorer
                        foot = sam_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        dists = np.linalg.norm(foot - b_xy, axis=1)
                        best_j = np.argmin(dists)
                        tid = sam_dets.tracker_id[best_j]
                        shooter_team = "A" if TEAMS_DICT.get(int(tid), 0) == 0 else "B"
        
                for ev in shot_tracker.update(i, has_js, has_ld, has_bib):
                    is_three = False
                    if shooter_xy_ct is not None:
                        dist, _ = _distance_from_hoop(shooter_xy_ct)
                        if dist > THREE_POINT_RADIUS_FT: is_three = True
                    val = 3 if is_three else 2
        
                    if ev["event"] == "MADE":
                        score[shooter_team] += val
                        if shooter_xy_ct is not None:
                            shot_markers.append((shooter_xy_ct.copy(), True))
                    elif ev["event"] == "MISSED":
                        if shooter_xy_ct is not None:
                            shot_markers.append((shooter_xy_ct.copy(), False))
                
                del frame, res
                try:
                    del kr  # may be empty list if court model found nothing
                except Exception:
                    pass
                if i % 10 == 0:
                    torch.cuda.empty_cache()


        # 5. Phase 3: Mathematical cleanup of positions
        print("[main] Phase 3/3: Cleaning tracking paths mathematically...")
        cleaned_xy, _ = clean_paths(
            video_xy_raw,
            jump_sigma=2.8,      # Tighter — remove smaller jerks
            min_jump_dist=0.5,
            max_jump_run=20,
            pad_around_runs=3,
            smooth_window=30,    # Maximum smoothness — less jitter
            smooth_poly=2,
        )
        
        # -- Jump Suppression: mask out positions where a player's foot suddenly RISES fast
        # (in image space Y decreases upward, i.e. foot Y going DOWN = jumping up)
        # We freeze their court dot at the last stable position when they airball
        foot_y_history = np.full((frame_count, MAX_TRACKER_ID), np.nan)
        for _i, _dets in enumerate(detections_history):
            if len(_dets) == 0: continue
            for _j, _tid in enumerate(_dets.tracker_id):
                _idx = _tid - 1
                if 0 <= _idx < MAX_TRACKER_ID:
                    foot_y_history[_i, _idx] = _dets.xyxy[_j, 3]  # bottom-Y in image space
        
        jump_mask = np.zeros((frame_count, MAX_TRACKER_ID), dtype=bool)
        # JUMP_RISE_THRESHOLD imported from constants.py
        for _p in range(MAX_TRACKER_ID):
            ys = foot_y_history[:, _p]
            for _i in range(1, frame_count):
                if not np.isnan(ys[_i]) and not np.isnan(ys[_i-1]):
                    delta = ys[_i] - ys[_i-1]  # negative = foot moving up = jump
                    if delta < -JUMP_RISE_THRESHOLD:
                        jump_mask[_i, _p] = True  # suppress this frame
        
        # Dual output: Tracking (segmentation only) + Court Map (HD)
        TRACKING_FILENAME = "output_tracking.mp4"
        MAP_FILENAME = "output_map.mp4"
        MAP_W, MAP_H = 1440, 1080  # Ultra HD court map
        
        print(f"[main] Rendering dual output videos...")
        # Pre-render both court base images ONCE — then .copy() per frame (critical for performance)
        base_court_hd = draw_court_2d(MAP_W, MAP_H)
        writer_track = cv2.VideoWriter(TRACKING_FILENAME, cv2.VideoWriter_fourcc(*OUTPUT_CODEC), fps, (w, h))
        writer_map   = cv2.VideoWriter(MAP_FILENAME,      cv2.VideoWriter_fourcc(*OUTPUT_CODEC), fps, (MAP_W, MAP_H))
        
        palette = sv.ColorPalette.from_hex([
            f"#{TEAM_A_COLOR[2]:02x}{TEAM_A_COLOR[1]:02x}{TEAM_A_COLOR[0]:02x}",
            f"#{TEAM_B_COLOR[2]:02x}{TEAM_B_COLOR[1]:02x}{TEAM_B_COLOR[0]:02x}"
        ])
        mask_ann = sv.MaskAnnotator(color=palette, opacity=0.3, color_lookup=sv.ColorLookup.CLASS)

        for i in tqdm(range(frame_count)):
            frame = cv2.imread(os.path.join(temp_dir, f"{i:05d}.jpg"))
            dets = detections_history[i]
            
            # Load mask from disk — use context manager to avoid file handle leak
            mask_path = os.path.join(masks_dir, f"{i:05d}.npz")
            if os.path.exists(mask_path):
                with np.load(mask_path) as npz:
                    dets.mask = npz["masks"]
                
            out_frame = frame.copy()
            
            if len(dets) > 0:
                my_teams = np.zeros(len(dets), dtype=int)
                for j, tid in enumerate(dets.tracker_id):
                    my_teams[j] = TEAMS_DICT.get(int(tid), 0)
        
                if dets.mask is not None:
                    dets.class_id = my_teams
                    out_frame = mask_ann.annotate(out_frame, dets)
                
            # ── Court map frame (HD — copy pre-drawn base, never re-render court lines) ──
            court_full = base_court_hd.copy()
            c_xy = cleaned_xy[i]
            
            # Build trails — vectorised numpy slice instead of per-frame loop
            trail_dict = {}
            colors_dict = {}
            s = max(0, i - TRAIL_LENGTH)
            for tid, t in TEAMS_DICT.items():
                idx = tid - 1
                if idx < 0 or idx >= MAX_TRACKER_ID: continue
                # Slice the smoothed path and drop jump/NaN frames
                xy_slice  = cleaned_xy[s:i+1, idx, :]          # (L, 2)
                jump_slice = jump_mask[s:i+1, idx]             # (L,)
                valid = ~np.isnan(xy_slice[:, 0]) & ~jump_slice
                pts = [(float(x), float(y)) for x, y in xy_slice[valid]]
                if len(pts) > 3:
                    trail_dict[tid] = pts
                    colors_dict[tid] = TEAM_A_COLOR if t == 0 else TEAM_B_COLOR
                    
            if trail_dict:
                draw_trails(court_full, trail_dict, colors_dict)

            # Draw dots (suppressed during jump frames)
            positions, colors = [], []
            for tid, t in TEAMS_DICT.items():
                idx = tid - 1
                if idx < 0 or idx >= MAX_TRACKER_ID: continue
                if jump_mask[i, idx]: continue
                px, py = c_xy[idx]
                if not np.isnan(px):
                    positions.append((px, py))
                    colors.append(TEAM_A_COLOR if t == 0 else TEAM_B_COLOR)
            if positions:
                draw_players_on_court(court_full, positions, colors, list(range(1, len(positions)+1)))

            # Shot markers removed from map — clean court view only

            # ── Draw Scoreboard UI ────────────────────────────────────────────────
            # Add a dark transparent bar at the top
            overlay = court_full.copy()
            cv2.rectangle(overlay, (0, 0), (MAP_W, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, court_full, 0.4, 0, court_full)
            
            # Draw score text
            score_text = f"TEAM A: {score['A']}   |   TEAM B: {score['B']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(score_text, font, 1.2, 3)[0]
            text_x = (MAP_W - text_size[0]) // 2
            cv2.putText(court_full, score_text, (text_x, 42), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            # ── Write tracking video (pure segmentation, NO minimap overlay) ──
            writer_track.write(out_frame)
            
            # ── Write standalone HD court-map video (NO scoreboard) ──
            writer_map.write(court_full)

        writer_track.release()
        writer_map.release()
        print(f"[main] Saved: {TRACKING_FILENAME}")
        print(f"[main] Saved: {MAP_FILENAME}")
        print("[main] Done.")

if __name__ == "__main__":
    main()
