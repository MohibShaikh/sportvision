"""
SportVision Demo — end-to-end detection + tracking + team classification on video.

Usage:
    python examples/demo.py --source match.mp4 --output analyzed.mp4
    python examples/demo.py --source match.mp4 --output analyzed.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from sportvision.analytics.distance import DistanceCalculator
from sportvision.analytics.possession import PossessionTracker
from sportvision.analytics.speed import SpeedEstimator
from sportvision.annotators import StatsOverlayAnnotator, TeamColorAnnotator, TrailAnnotator
from sportvision.detection import SportsDetector
from sportvision.teams import TeamClassifier
from sportvision.tracking import SportsTracker


def main():
    parser = argparse.ArgumentParser(description="SportVision Demo")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument(
        "--model", default="rfdetr-base", help="Detection model (rfdetr-base/rfdetr-large)"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: cannot open {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.1f} FPS, {total} frames")
    print(f"Model: {args.model}, threshold: {args.threshold}")

    # Init modules
    detector = SportsDetector(model=args.model, confidence_threshold=args.threshold)
    tracker = SportsTracker()
    team_clf = TeamClassifier(n_teams=2)
    possession = PossessionTracker()
    speed_est = SpeedEstimator(fps=fps)
    distance_calc = DistanceCalculator()

    team_ann = TeamColorAnnotator(home_color=(0, 0, 255), away_color=(255, 165, 0))
    stats_ann = StatsOverlayAnnotator()
    trail_ann = TrailAnnotator(trail_length=30)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Collect crops from first 100 frames for team classification
    print("Phase 1: Collecting crops for team classification...")
    all_crops = []
    warmup_frames = []
    frame_idx = 0

    while frame_idx < min(100, total if total > 0 else 100):
        ret, frame = cap.read()
        if not ret:
            break
        warmup_frames.append(frame)
        dets = detector.detect(frame)
        for box in dets["xyxy"]:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                all_crops.append(crop)
        frame_idx += 1

    if len(all_crops) >= 2:
        print(f"  Fitting team classifier on {len(all_crops)} crops...")
        team_clf.fit(all_crops)
    else:
        print("  Not enough crops for team classification, skipping.")

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Phase 2: Full processing
    print("Phase 2: Processing video...")
    positions_history: dict[int, list[tuple[float, float]]] = {}
    frame_history: dict[int, list[int]] = {}
    trail_history: dict[int, list[tuple[float, float]]] = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        dets = detector.detect(frame)
        tracked = tracker.update(dets)

        xyxy = tracked["xyxy"]
        tracker_ids = tracked.get("tracker_id", np.arange(len(xyxy)))

        # Team classification
        team_ids = np.zeros(len(xyxy), dtype=int)
        if team_clf._model is not None and len(xyxy) > 0:
            crops = []
            for box in xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    crops.append(crop)
                else:
                    crops.append(np.zeros((20, 20, 3), dtype=np.uint8))
            team_ids = np.array(team_clf.predict(crops))

        # Track positions (use box center)
        centers = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
        ]) if len(xyxy) > 0 else np.empty((0, 2))

        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            pos = (float(centers[i, 0]), float(centers[i, 1]))
            positions_history.setdefault(tid, []).append(pos)
            frame_history.setdefault(tid, []).append(frame_idx)
            trail_history.setdefault(tid, []).append(pos)

        # Ball possession (use first detected "ball-like" object or smallest bbox)
        if len(centers) > 0:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            ball_idx = np.argmin(areas)
            ball_pos = centers[ball_idx]
            player_mask = np.arange(len(centers)) != ball_idx
            if player_mask.sum() > 0:
                possession.update(
                    centers[player_mask], team_ids[player_mask], ball_pos
                )

        # Annotate
        annotated = team_ann.annotate(frame.copy(), xyxy, team_ids)
        annotated = trail_ann.annotate(annotated, trail_history)
        annotated = stats_ann.annotate(
            annotated, possession=possession.get_stats()
        )

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            proc_fps = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{total} ({proc_fps:.1f} FPS)")

    elapsed = time.time() - t0
    cap.release()
    writer.release()

    # Final stats
    speeds = speed_est.calculate(positions_history, frame_history)
    distances = distance_calc.calculate(positions_history)
    poss = possession.get_stats()

    stats = {
        "frames_processed": frame_idx,
        "processing_time_s": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 1) if elapsed > 0 else 0,
        "possession": {str(k): round(v, 3) for k, v in poss.items()},
        "players_tracked": len(positions_history),
        "avg_speed_kmh": {
            str(k): round(v, 1) for k, v in speeds.items()
        },
        "total_distance_px": {
            str(k): round(v, 1) for k, v in distances.items()
        },
    }

    stats_path = args.output.rsplit(".", 1)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Output: {args.output}")
    print(f"Stats: {stats_path}")
    print(f"Possession: Team 0={poss.get(0, 0):.1%}, Team 1={poss.get(1, 0):.1%}")
    print(f"Players tracked: {len(positions_history)}")
    print(f"Processing: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
