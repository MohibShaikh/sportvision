"""Integration test: run workflow blocks on real video.

Downloads the Roboflow supervision basketball sample video (cached in
test_data/) and runs YOLOv8 detection, then pipes real detections through:
  SportsDetectionFilter → TeamClassifier → PossessionTracker → DistanceCalculator

Outputs an annotated video to test_data/integration_output.mp4.

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
OUTPUT_VIDEO = VIDEO_DIR / "integration_output.mp4"
START_FRAME = 100  # players appear around frame 100
NUM_FRAMES = 120  # ~2 seconds of action at 60fps


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


def _annotate_frame(
    frame, classified, ball_dets, dist_dets, poss, frame_num, team_ann, stats_ann
):
    """Draw all annotations on a frame and return it."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Team-colored boxes on players
    if len(classified) > 0:
        vis = team_ann.annotate(vis, classified.xyxy, classified.data["team_id"])

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
        cv2.putText(
            vis,
            f"T{tid} {d:.0f}px",
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
        f"Frame {frame_num}",
        (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return vis


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestWorkflowBlocksIntegration:
    """Chain all four workflow blocks on real video."""

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

    def test_full_pipeline_video(self, video_path, yolo_model):
        """Process video through full pipeline, output annotated mp4.

        Reads directly from video file and writes annotated output.
        """
        from sportvision.annotators import (
            StatsOverlayAnnotator,
            TeamColorAnnotator,
        )

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Cannot open {video_path}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))
        assert writer.isOpened(), "Cannot create output video"

        team_ann = TeamColorAnnotator(
            home_color=(0, 120, 255),
            away_color=(255, 50, 50),
            thickness=3,
        )
        stats_ann = StatsOverlayAnnotator(font_scale=1.0)

        frames_processed = 0
        frames_with_players = 0
        poss = None

        for _ in range(NUM_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break

            frame_num = START_FRAME + frames_processed
            frames_processed += 1

            # --- Real YOLO detection ---
            raw_dets = _detect_with_yolo(yolo_model, frame)

            if len(raw_dets) == 0:
                writer.write(frame)
                continue

            # 1. Filter: COCO IDs → sports IDs
            filtered = self.filter_block.run(detections=raw_dets)["detections"]

            if len(filtered) == 0:
                writer.write(frame)
                continue

            if filtered.tracker_id is None:
                filtered.tracker_id = np.arange(len(filtered))

            # 2. Team classify players
            player_mask = filtered.class_id == 0
            player_dets = filtered[player_mask]

            if len(player_dets) >= 2:
                image = _make_workflow_image(frame)
                classified = self.team_block.run(
                    image=image,
                    detections=player_dets,
                    n_teams=2,
                    refit_every=30,
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

            # 5. Distance calculation
            dist = self.distance_block.run(detections=combined)
            dist_dets = dist["detections"]
            assert "distance" in dist_dets.data

            frames_with_players += 1

            # --- Annotate and write ---
            vis = _annotate_frame(
                frame,
                classified,
                ball_dets,
                dist_dets,
                poss,
                frame_num,
                team_ann,
                stats_ann,
            )
            writer.write(vis)

        cap.release()
        writer.release()

        assert frames_processed == NUM_FRAMES
        assert frames_with_players > 0, "No players detected"
        assert OUTPUT_VIDEO.exists()

        size_kb = OUTPUT_VIDEO.stat().st_size / 1024
        print(f"\n  Video: {frames_processed} frames processed")
        print(f"  Frames with detections: {frames_with_players}")
        if poss:
            print(f"  Possession: {poss['possession_stats']}")
        print(f"  Output: {OUTPUT_VIDEO} ({size_kb:.0f} KB)")

    def test_refit_changes_model(self, video_path, yolo_model):
        """Verify refit_every refits by reading frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        models_seen = []
        for _ in range(20):
            ok, frame = cap.read()
            if not ok:
                break
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

        cap.release()
        assert len(set(models_seen)) > 1

    def test_possession_warns_without_team_id(self, video_path, yolo_model):
        """Possession block warns when team_id is absent."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME + 15)
        ok, frame = cap.read()
        cap.release()
        assert ok

        raw = _detect_with_yolo(yolo_model, frame)
        filtered = self.filter_block.run(detections=raw)["detections"]
        if len(filtered) == 0:
            pytest.skip("No detections in frame")
        result = self.possession_block.run(
            detections=filtered,
            ball_class_id=1,
            ball_proximity_threshold=9999.0,
        )
        players = filtered.class_id == 0
        balls = filtered.class_id == 1
        if players.any() and balls.any():
            assert result["warning"] != ""
            assert "team_id" in result["warning"]

    def test_distance_with_homography(self, video_path, yolo_model):
        """Distance with homography from video frames."""
        h_mat = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        last_filtered = None
        for _ in range(15):
            ok, frame = cap.read()
            if not ok:
                break
            raw = _detect_with_yolo(yolo_model, frame)
            if len(raw) == 0:
                continue
            filtered = self.filter_block.run(detections=raw)["detections"]
            if filtered.tracker_id is None:
                filtered.tracker_id = np.arange(len(filtered))
            self.distance_block.run(detections=filtered, homography_matrix=h_mat)
            last_filtered = filtered

        cap.release()
        assert last_filtered is not None

        result = self.distance_block.run(
            detections=last_filtered, homography_matrix=h_mat
        )
        distances = result["detections"].data["distance"]
        assert all(d < 200 for d in distances)
