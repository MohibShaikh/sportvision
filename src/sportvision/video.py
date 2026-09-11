"""Analyze a clip: python examples/demo.py --source match.mp4 --model yolov8n.pt."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2

from sportvision.pipeline import SportVisionPipeline
from sportvision.review import frame_record, review_summary


def analyze_video(
    source,
    output="output.mp4",
    model="yolov8n.pt",
    max_frames=0,
    *,
    team_colors=None,
    analysis_mode="quick",
    progress=None,
):
    """Write annotated video and a JSON summary of observed ball proximity."""
    if analysis_mode not in ("quick", "detailed"):
        raise ValueError("analysis_mode must be quick or detailed")
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Output must differ from the source video")
    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative")
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        # `or 30.0` only catches an exact zero. NaN and negatives are both
        # truthy and reach the writer, which then fails with a misleading
        # "cannot write video" instead of naming the real problem.
        reported = cap.get(cv2.CAP_PROP_FPS)
        fps = reported if math.isfinite(reported) and reported > 0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise ValueError(f"Cannot write video: {output}")
        pipeline = SportVisionPipeline(
            model=model,
            possession_method="body" if analysis_mode == "detailed" else "center",
            infer_size=1280 if analysis_mode == "detailed" else 640,
            ball_confidence_threshold=0.15 if analysis_mode == "detailed" else 0.25,
        )
        if team_colors is not None:
            pipeline.set_team_colors(team_colors)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected = min(total, max_frames) if max_frames and total else total
        observations = []
        frames = ball_frames = 0
        player_tracks = set()
        start = time.perf_counter()
        while max_frames == 0 or frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame)
            tracked = result["detections"]
            observations.append(
                frame_record(frames, tracked, method=pipeline.possession_method)
            )
            ball_frames += int((tracked["class_id"] == 1).any())
            player_tracks.update(
                int(tid)
                for tid in tracked["tracker_id"][tracked["class_id"] == 0]
                if tid >= 0
            )
            writer.write(result["annotated_frame"])
            frames += 1
            if progress is not None and (frames == 1 or frames % 10 == 0):
                progress(frames, expected, result["annotated_frame"])
        if frames and progress is not None:
            progress(frames, frames, result["annotated_frame"])
        if not frames:
            raise ValueError("Video contains no readable frames")
        elapsed = time.perf_counter() - start
        stats = {
            "model": model,
            "analysis_mode": analysis_mode,
            "ball_confidence_threshold": pipeline._detector.ball_confidence_threshold,
            "tracking_minimum_iou": pipeline._tracker.match_thresh,
            "infer_size": pipeline.infer_size,
            "confidence_threshold": pipeline._detector.confidence_threshold,
            "team_colors_bgr": team_colors,
            "frames_processed": frames,
            "processing_time_s": round(elapsed, 2),
            "processing_fps": round(frames / elapsed, 2) if elapsed else 0.0,
            "frames_with_tracked_ball": ball_frames,
            "unique_player_track_ids": len(player_tracks),
            "attributed_frames": sum(pipeline._possession._counts.values()),
            "team_mode": "chosen colors" if team_colors else "automatic clusters",
            "possession": pipeline.get_stats()["possession"],
            "possession_method": pipeline.possession_method,
            "note": "Track IDs are not player identities. Possession is a proximity "
            "estimate on observed frames, not official match possession.",
        }
        digest = hashlib.sha256()
        with open(source, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        stats["review"] = {
            "clip_sha256": digest.hexdigest(),
            "schema_version": 1,
            "fps": fps,
            "width": width,
            "height": height,
            "threshold_px": 50.0,
            "possession_method": pipeline.possession_method,
            "frames": observations,
        }
        stats.update(review_summary(stats["review"]))
        stats_path = Path(output).with_suffix(".json")
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        return stats
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def read_frame(source, frame_index):
    """Read a zero-based frame for review; never substitute another frame on failure."""
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open source video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_index}")
        return frame
    finally:
        cap.release()


def export_review(source, output, review, progress=None):
    """Re-render corrected observations without loading or invoking a detector."""
    import numpy as np

    from sportvision.annotators import StatsOverlayAnnotator, TeamColorAnnotator

    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Output must differ from source")
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open source video")
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            review["fps"],
            (review["width"], review["height"]),
        )
        if not writer.isOpened():
            raise ValueError("Cannot create corrected video")
        counts = {}
        for index, row in enumerate(review["frames"]):
            if row["frame_index"] != index:
                raise ValueError("Review frames must be consecutive, starting at zero")
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Source ended before reviewed frame {index}")
            evidence = row["evidence"]
            if evidence["team_id"] is not None:
                team = evidence["team_id"]
                counts[team] = counts.get(team, 0) + 1
            total = sum(counts.values())
            shares = {i: counts.get(i, 0) / total for i in range(2)} if total else {}
            frame = TeamColorAnnotator().annotate(
                frame,
                np.asarray(row["xyxy"]).reshape(-1, 4),
                np.asarray(row["team_id"]),
            )
            frame = StatsOverlayAnnotator().annotate(frame, possession=shares)
            for box, tid, cls in zip(row["xyxy"], row["tracker_id"], row["class_id"]):
                if cls == 0:
                    cv2.putText(
                        frame,
                        f"Track {tid}",
                        (int(box[0]), max(20, int(box[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
            if "manual_ball_xy" in row:
                point = tuple(int(v) for v in row["manual_ball_xy"])
                cv2.circle(frame, point, 8, (0, 215, 255), 2)
                cv2.putText(
                    frame,
                    "Manual ball",
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 215, 255),
                    1,
                )
            if "manual_possession_team" in row:
                team = row["manual_possession_team"]
                label = "No possession" if team is None else f"Team {team + 1}"
                cv2.putText(
                    frame,
                    f"Manual: {label}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 215, 255),
                    2,
                )
            writer.write(frame)
            if progress and (index % 10 == 0 or index == len(review["frames"]) - 1):
                progress(index + 1, len(review["frames"]))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
