"""Inspectable frame evidence and reversible track-level team corrections."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

REASONS = {
    "attributed": "Ball close to an assigned player",
    "no_ball": "The model did not find the ball; it may still be visible",
    "manual_possession": "Possession confirmed manually for this frame",
    "manual_no_possession": "No possession confirmed manually for this frame",
    "multiple_balls": "Multiple balls detected; ambiguous",
    "no_players": "No tracked players",
    "ambiguous_players": "Ball close to multiple players; possession uncertain",
    "too_far": "Ball too far from the nearest player",
    "untracked_player": "Nearest player does not yet have a confirmed track",
    "unknown_team": "Nearest player has no team",
    "ignored_player": "Nearest player is marked referee / ignore",
}


def possession_evidence(
    detections: dict, threshold: float = 50.0, *, method: str = "center"
) -> dict:
    """Return the observation used for attribution, including rejected evidence."""
    if method not in ("center", "body"):
        raise ValueError("Possession method must be center or body")
    boxes = np.asarray(detections["xyxy"], dtype=float).reshape(-1, 4)
    classes = np.asarray(detections["class_id"])
    balls = np.flatnonzero(classes == 1)
    players = np.flatnonzero(classes == 0)
    result = {
        "method": method,
        "team_id": None,
        "tracker_id": None,
        "distance_px": None,
        "ball_xy": None,
        "reason": "no_ball",
        "ball_source": "manual" if "manual_ball_xy" in detections else "detector",
    }
    manual_ball = detections.get("manual_ball_xy")
    if manual_ball is None and len(balls) == 0:
        return result
    if manual_ball is None and len(balls) > 1:
        return {**result, "reason": "multiple_balls"}
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    ball_xy = np.asarray(manual_ball) if manual_ball is not None else centers[balls[0]]
    result["ball_xy"] = ball_xy.tolist()
    if not len(players):
        return {**result, "reason": "no_players"}
    distances = np.linalg.norm(centers[players] - ball_xy, axis=1)
    if method == "body":
        player_boxes = boxes[players]
        closest = np.clip(ball_xy, player_boxes[:, :2], player_boxes[:, 2:])
        distances = np.linalg.norm(closest - ball_xy, axis=1)
        # Normalize for camera resolution and apparent player size.
        scores = distances / np.maximum(player_boxes[:, 3] - player_boxes[:, 1], 1)
        result["body_distance_limit"] = 0.15
        if np.count_nonzero(scores <= 0.15) > 1:
            return {**result, "reason": "ambiguous_players"}
    else:
        scores = distances
    nearest = int(np.argmin(scores))
    idx = players[nearest]
    result["tracker_id"] = int(detections["tracker_id"][idx])
    result["distance_px"] = float(distances[nearest])
    team = int(detections["team_id"][idx])
    if scores[nearest] > (0.15 if method == "body" else threshold):
        result["reason"] = "too_far"
    elif result["tracker_id"] < 0:
        result["reason"] = "untracked_player"
    elif team == -2:
        result["reason"] = "ignored_player"
    elif team < 0:
        result["reason"] = "unknown_team"
    else:
        result.update(team_id=team, reason="attributed")
    return result


def frame_record(
    frame_index: int, tracked: dict, threshold: float = 50.0, *, method: str = "center"
) -> dict:
    record = {
        key: np.asarray(tracked[key]).tolist()
        for key in ("xyxy", "class_id", "confidence", "tracker_id", "team_id")
    }
    return {
        **record,
        "frame_index": frame_index,
        "evidence": possession_evidence(record, threshold, method=method),
    }


def correct_review(
    review: dict,
    overrides: dict[int, int],
    *,
    ball_overrides: dict | None = None,
    possession_overrides: dict | None = None,
) -> dict:
    """Apply team choices to every appearance of the same track in this clip.

    -2 excludes a track; removing its override restores the measured assignments.
    This intentionally does not guess identity across lost/reassigned track IDs.
    """
    known = {
        int(tid)
        for row in review["frames"]
        for tid, cls in zip(row["tracker_id"], row["class_id"])
        if cls == 0 and int(tid) >= 0
    }
    if any(
        tid not in known or team not in (-2, 0, 1) for tid, team in overrides.items()
    ):
        raise ValueError(
            "Corrections require a player track and Team 1, Team 2, or ignore"
        )
    ball_overrides = ball_overrides or {}
    possession_overrides = possession_overrides or {}
    frame_ids = {row["frame_index"] for row in review["frames"]}
    for idx, point in ball_overrides.items():
        point = np.asarray(point, dtype=float)
        if (
            idx not in frame_ids
            or point.shape != (2,)
            or not np.isfinite(point).all()
            or not (
                0 <= point[0] < review["width"] and 0 <= point[1] < review["height"]
            )
        ):
            raise ValueError("Ball correction must be inside a reviewed frame")
    if any(
        idx not in frame_ids or team not in (None, 0, 1)
        for idx, team in possession_overrides.items()
    ):
        raise ValueError(
            "Possession correction requires a reviewed frame and valid team"
        )
    corrected = deepcopy(review)
    for row in corrected["frames"]:
        row["measured_team_id"] = row.get("measured_team_id", row["team_id"]).copy()
        row["team_id"] = row["measured_team_id"].copy()
        row.pop("manual_ball_xy", None)
        row.pop("manual_possession_team", None)
        for i, (tid, cls) in enumerate(zip(row["tracker_id"], row["class_id"])):
            if cls == 0 and tid in overrides:
                row["team_id"][i] = overrides[tid]
        idx = row["frame_index"]
        if idx in ball_overrides:
            row["manual_ball_xy"] = list(ball_overrides[idx])
        row["evidence"] = possession_evidence(
            row,
            review["threshold_px"],
            method=review.get("possession_method", "center"),
        )
        if idx in possession_overrides:
            row["manual_possession_team"] = possession_overrides[idx]
            row["evidence"]["automatic_reason"] = row["evidence"]["reason"]
            row["evidence"].update(
                team_id=possession_overrides[idx],
                reason="manual_possession"
                if possession_overrides[idx] is not None
                else "manual_no_possession",
            )
    corrected["corrections"] = {str(tid): team for tid, team in overrides.items()}
    corrected["ball_corrections"] = {str(k): list(v) for k, v in ball_overrides.items()}
    corrected["possession_corrections"] = {
        str(k): v for k, v in possession_overrides.items()
    }
    return corrected


def review_summary(review: dict) -> dict:
    counts = Counter(
        row["evidence"]["team_id"]
        for row in review["frames"]
        if row["evidence"]["team_id"] is not None
    )
    total = sum(counts.values())
    return {
        "possession": {i: counts[i] / total for i in range(2)} if total else {},
        "attributed_frames": total,
        "manually_confirmed_frames": sum(
            "manual_possession_team" in row for row in review["frames"]
        ),
        "manually_marked_ball_frames": sum(
            "manual_ball_xy" in row for row in review["frames"]
        ),
        "excluded_frames": dict(
            Counter(
                row["evidence"]["reason"]
                for row in review["frames"]
                if row["evidence"]["team_id"] is None
            )
        ),
    }


def timeline_segments(review: dict) -> list[dict]:
    segments = []
    for row in review["frames"]:
        evidence = row["evidence"]
        key = (evidence["reason"], evidence["team_id"], evidence["tracker_id"])
        if not segments or segments[-1]["key"] != key:
            segments.append(
                {
                    "key": key,
                    "start_frame": row["frame_index"],
                    "end_frame": row["frame_index"],
                }
            )
        else:
            segments[-1]["end_frame"] = row["frame_index"]
    return [
        {
            **segment,
            "start_s": segment["start_frame"] / review["fps"],
            "end_s": (segment["end_frame"] + 1) / review["fps"],
        }
        for segment in segments
    ]
