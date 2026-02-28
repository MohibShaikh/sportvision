"""Integration test: run workflow blocks on real video frames.

Downloads the Roboflow supervision basketball sample video (cached in
test_data/) and runs YOLOv8 detection, then pipes real detections through:
  SportsDetectionFilter → TeamClassifier → PossessionTracker → DistanceCalculator

Saves annotated frames to test_data/integration_output/ for visual review.

Run with::

    CUDA_VISIBLE_DEVICES="" pytest tests/test_workflow_integration.py -vs
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import supervision as sv

VIDEO_DIR = Path(__file__).resolve().parent.parent / "test_data"
VIDEO_PATH = VIDEO_DIR / "basketball-1.mp4"
START_FRAME = 100  # players appear around frame 100
NUM_FRAMES = 30


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def video_path() -> Path:
    """Ensure the sample video is present (download once)."""
    if not VIDEO_PATH.exists():
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        from supervision.assets import VideoAssets, download_assets

        download_assets(VideoAssets.BASKETBALL)
        downloaded = Path("basketball-1.mp4")
        downloaded.rename(VIDEO_PATH)
    return VIDEO_PATH


@pytest.fixture(scope="module")
def frames(video_path: Path) -> list[np.ndarray]:
    """Read NUM_FRAMES starting from START_FRAME."""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open {video_path}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    imgs: list[np.ndarray] = []
    for _ in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        imgs.append(frame)
    cap.release()
    assert len(imgs) == NUM_FRAMES
    return imgs


@pytest.fixture(scope="module")
def yolo_model():
    """Load YOLOv8n once for the whole module."""
    from ultralytics import YOLO

    return YOLO("yolov8n.pt")


def _make_workflow_image(img: np.ndarray):
    mock = MagicMock()
    mock.numpy_image = img
    return mock


def _detect_with_yolo(
    model, frame: np.ndarray, tracker_id_offset: int = 0
) -> sv.Detections:
    """Run YOLO on a frame and return sv.Detections."""
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return sv.Detections.empty()

    xyxy = boxes.xyxy.cpu().numpy()
    class_id = boxes.cls.cpu().numpy().astype(int)
    confidence = boxes.conf.cpu().numpy().astype(np.float32)
    dets = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)
    dets.tracker_id = np.arange(len(dets)) + tracker_id_offset
    return dets


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------


class TestWorkflowBlocksIntegration:
    """Chain all four workflow blocks on real video frames."""

    @pytest.fixture(autouse=True)
    def _import_blocks(self):
        from sportvision.workflows.distance_calculator.v1 import (
            DistanceCalculatorBlockV1,
        )
        from sportvision.workflows.possession_tracker.v1 import (
            PossessionTrackerBlockV1,
        )
        from sportvision.workflows.sports_detection_filter.v1 import (
            SportsDetectionFilterBlockV1,
        )
        from sportvision.workflows.team_classifier.v1 import (
            TeamClassifierBlockV1,
        )

        self.filter_block = SportsDetectionFilterBlockV1()
        self.team_block = TeamClassifierBlockV1()
        self.possession_block = PossessionTrackerBlockV1()
        self.distance_block = DistanceCalculatorBlockV1()

    def test_full_chain_real_frames(self, frames, yolo_model):
        """Run YOLO → filter → classify → possession → distance.

        Saves annotated frames to test_data/integration_output/.
        """
        from sportvision.annotators import (
            StatsOverlayAnnotator,
            TeamColorAnnotator,
        )

        out_dir = VIDEO_DIR / "integration_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        team_ann = TeamColorAnnotator(
            home_color=(0, 120, 255),
            away_color=(255, 50, 50),
            thickness=3,
        )
        stats_ann = StatsOverlayAnnotator(font_scale=1.0)

        h, w = frames[0].shape[:2]
        frames_with_players = 0

        for i, frame in enumerate(frames):
            # --- Real YOLO detection ---
            raw_dets = _detect_with_yolo(yolo_model, frame)

            if len(raw_dets) == 0:
                cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), frame)
                continue

            # 1. Filter: COCO IDs → sports IDs
            filtered = self.filter_block.run(detections=raw_dets)["detections"]

            if len(filtered) == 0:
                cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), frame)
                continue

            # Ensure tracker_id survives filtering
            if filtered.tracker_id is None:
                filtered.tracker_id = np.arange(len(filtered))

            # 2. Team classify players (class_id == 0)
            player_mask = filtered.class_id == 0
            player_dets = filtered[player_mask]

            if len(player_dets) >= 2:
                image = _make_workflow_image(frame)
                classified = self.team_block.run(
                    image=image,
                    detections=player_dets,
                    n_teams=2,
                    refit_every=10,
                )["detections"]
                assert "team_id" in classified.data
            else:
                classified = player_dets
                classified.data["team_id"] = np.zeros(len(player_dets), dtype=int)

            # 3. Recombine for possession
            ball_dets = filtered[~player_mask]
            if len(ball_dets) > 0 and len(classified) > 0:
                combined_xyxy = np.vstack([classified.xyxy, ball_dets.xyxy])
                combined_class = np.concatenate(
                    [classified.class_id, ball_dets.class_id]
                )
                combined_conf = np.concatenate(
                    [classified.confidence, ball_dets.confidence]
                )
                combined = sv.Detections(
                    xyxy=combined_xyxy,
                    class_id=combined_class,
                    confidence=combined_conf,
                )
                team_ids = np.concatenate(
                    [
                        classified.data["team_id"],
                        [-1] * len(ball_dets),
                    ]
                )
                combined.data["team_id"] = team_ids
                combined.tracker_id = np.arange(len(combined))
            else:
                combined = filtered
                if "team_id" not in combined.data:
                    combined.data["team_id"] = np.zeros(len(combined), dtype=int)

            # 4. Possession tracking
            poss = self.possession_block.run(
                detections=combined,
                ball_class_id=1,
                ball_proximity_threshold=9999.0,
            )
            assert "possession_stats" in poss
            assert "possessing_team" in poss

            # 5. Distance calculation
            dist = self.distance_block.run(detections=combined)
            dist_dets = dist["detections"]
            assert "distance" in dist_dets.data

            frames_with_players += 1

            # --- Annotate and save ---
            vis = frame.copy()

            # Team-colored boxes on players
            if len(classified) > 0:
                vis = team_ann.annotate(
                    vis,
                    classified.xyxy,
                    classified.data["team_id"],
                )

            # Ball box in green
            for box in ball_dets.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    vis,
                    "BALL",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Distance + team labels on players
            for j in range(len(classified)):
                x1, y1 = classified.xyxy[j][:2].astype(int)
                d = dist_dets.data["distance"][j]
                tid = int(classified.data["team_id"][j])
                label = f"T{tid} {d:.0f}px"
                cv2.putText(
                    vis,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Possession overlay
            vis = stats_ann.annotate(vis, possession=poss["possession_stats"])

            # Frame counter
            cv2.putText(
                vis,
                f"Frame {START_FRAME + i}",
                (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), vis)

        assert frames_with_players > 0, "No players detected"

        # Save summary on last annotated frame
        summary = vis.copy()
        final_poss = poss["possession_stats"]
        y = h - 150
        cv2.putText(
            summary,
            "=== FINAL SUMMARY ===",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        for team, pct in final_poss.items():
            y += 35
            cv2.putText(
                summary,
                f"Team {team}: {pct:.1%} possession",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        y += 35
        cv2.putText(
            summary,
            f"Possessing team: {poss['possessing_team']}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(out_dir / "summary.jpg"), summary)

        print(f"\n  Frames with detections: {frames_with_players}")
        print(f"  Possession: {final_poss}")
        print(f"  Output: {out_dir}")

    def test_refit_changes_model(self, frames, yolo_model):
        """Verify refit_every actually refits on real image data."""
        models_seen = []
        for frame in frames[:12]:
            raw = _detect_with_yolo(yolo_model, frame)
            filtered = self.filter_block.run(detections=raw)["detections"]
            players = filtered[filtered.class_id == 0]
            if len(players) < 2:
                continue
            image = _make_workflow_image(frame)
            self.team_block.run(
                image=image,
                detections=players,
                n_teams=2,
                refit_every=5,
            )
            models_seen.append(id(self.team_block._model))

        assert len(set(models_seen)) > 1

    def test_possession_warns_without_team_id(self, frames, yolo_model):
        """Possession block warns when team_id is absent."""
        raw = _detect_with_yolo(yolo_model, frames[15])
        filtered = self.filter_block.run(detections=raw)["detections"]
        if len(filtered) == 0:
            pytest.skip("No detections in frame")
        result = self.possession_block.run(
            detections=filtered,
            ball_class_id=1,
            ball_proximity_threshold=9999.0,
        )
        # No team_id was set → should warn (if ball+player present)
        players = filtered.class_id == 0
        balls = filtered.class_id == 1
        if players.any() and balls.any():
            assert result["warning"] != ""
            assert "team_id" in result["warning"]

    def test_distance_with_homography(self, frames, yolo_model):
        """Distance with homography uses transformed coords."""
        h_mat = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]

        for frame in frames[:10]:
            raw = _detect_with_yolo(yolo_model, frame)
            if len(raw) == 0:
                continue
            filtered = self.filter_block.run(detections=raw)["detections"]
            if filtered.tracker_id is None:
                filtered.tracker_id = np.arange(len(filtered))
            self.distance_block.run(detections=filtered, homography_matrix=h_mat)

        # Distances with 0.01 scale should be small
        result = self.distance_block.run(detections=filtered, homography_matrix=h_mat)
        distances = result["detections"].data["distance"]
        assert all(d < 200 for d in distances)
