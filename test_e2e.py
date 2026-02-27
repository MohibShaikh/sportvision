"""End-to-end test: run full SportVision pipeline on a real soccer clip."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from sportvision.detection import SportsDetector
from sportvision.tracking import SportsTracker
from sportvision.teams import TeamClassifier
from sportvision.homography import FieldHomography
from sportvision.analytics.possession import PossessionTracker
from sportvision.analytics.speed import SpeedEstimator
from sportvision.analytics.distance import DistanceCalculator
from sportvision.analytics.heatmap import HeatmapGenerator
from sportvision.annotators import TeamColorAnnotator, StatsOverlayAnnotator, TrailAnnotator

VIDEO = Path(__file__).parent / "test_data" / "soccer_broadcast.mp4"
OUTPUT = Path(__file__).parent / "test_data" / "output.mp4"


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened(), f"Cannot open {VIDEO}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps}fps, {total} frames")

    # Init modules
    detector = SportsDetector(confidence_threshold=0.10)
    tracker = SportsTracker()
    possession = PossessionTracker()
    speed_est = SpeedEstimator(fps=fps)
    distance_calc = DistanceCalculator()
    heatmap = HeatmapGenerator(resolution=(105, 68))
    team_ann = TeamColorAnnotator(home_color=(0, 0, 255), away_color=(255, 0, 0))
    stats_ann = StatsOverlayAnnotator()
    trail_ann = TrailAnnotator(trail_length=20)

    # Track positions across frames for speed/distance
    positions: dict[int, list[tuple[float, float]]] = {}
    frame_indices: dict[int, list[int]] = {}
    trails: dict[int, list[tuple[float, float]]] = {}

    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_num = 0
    det_counts = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= 150:
            break

        # Detect
        dets = detector.detect(frame)
        n_dets = len(dets["xyxy"])
        det_counts.append(n_dets)

        # Track
        tracked = tracker.update(dets)
        n_tracked = len(tracked["xyxy"])

        # Compute centers for analytics
        if n_tracked > 0:
            centers = (tracked["xyxy"][:, :2] + tracked["xyxy"][:, 2:]) / 2

            # Store positions per tracker_id
            for i, tid in enumerate(tracked.get("tracker_id", range(n_tracked))):
                tid = int(tid)
                cx, cy = float(centers[i][0]), float(centers[i][1])
                positions.setdefault(tid, []).append((cx, cy))
                frame_indices.setdefault(tid, []).append(frame_num)
                trails.setdefault(tid, []).append((cx, cy))

            # Ball = class_id 1, players = class_id 0
            class_ids = tracked.get("class_id", np.zeros(n_tracked, dtype=int))
            ball_mask = class_ids == 1
            player_mask = class_ids == 0

            if ball_mask.any() and player_mask.any():
                ball_pos = centers[ball_mask][0]
                player_pos = centers[player_mask]
                # Use class_id as fake team_id for now (no team classifier fit yet)
                team_ids = np.zeros(len(player_pos), dtype=int)
                if len(player_pos) > 1:
                    team_ids[len(player_pos)//2:] = 1
                possession.update(player_pos, team_ids, ball_pos)
                heatmap.update(player_pos / np.array([[w/105, h/68]]), team_ids)

        # Annotate
        annotated = frame.copy()
        if n_tracked > 0:
            team_ids_ann = tracked.get("class_id", np.zeros(n_tracked, dtype=int))
            annotated = team_ann.annotate(annotated, tracked["xyxy"], team_ids_ann)

        poss_stats = possession.get_stats()
        annotated = stats_ann.annotate(annotated, possession=poss_stats)
        annotated = trail_ann.annotate(annotated, trails)

        writer.write(annotated)
        frame_num += 1

        if frame_num % 25 == 0:
            print(f"  Frame {frame_num}/{total}: {n_dets} detections, {n_tracked} tracked")

    cap.release()
    writer.release()

    # Stats summary
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Frames processed: {frame_num}")
    print(f"Avg detections/frame: {np.mean(det_counts):.1f}")
    print(f"Max detections/frame: {max(det_counts)}")
    print(f"Min detections/frame: {min(det_counts)}")
    print(f"Unique tracker IDs: {len(positions)}")

    # Speed
    speeds = speed_est.calculate(positions, frame_indices)
    if speeds:
        print(f"Speed range: {min(speeds.values()):.1f} - {max(speeds.values()):.1f} km/h")

    # Distance
    distances = distance_calc.calculate(positions)
    if distances:
        print(f"Distance range: {min(distances.values()):.1f} - {max(distances.values()):.1f} px")

    # Possession
    print(f"Possession: {poss_stats}")

    print(f"\nOutput video: {OUTPUT}")
    print(f"Output size: {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
