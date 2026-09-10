"""Evaluate saved observations against independent, partially labeled frames.

Run: python -m sportvision.evaluation report.json labels.json
Unlabeled fields are excluded, never treated as negative observations.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


def evaluate(review: dict, labels: dict, iou_threshold: float = 0.5) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if not labels.get("clip_sha256") or labels["clip_sha256"] != review.get(
        "clip_sha256"
    ):
        raise ValueError("Labels must identify the same source clip by SHA-256")
    rows = {r["frame_index"]: r for r in review["frames"]}
    seen = set()
    ball_tp = ball_fp = ball_fn = ball_labeled = 0
    team_correct = team_matched = player_labeled = 0
    possession_correct = possession_labeled = possession_predicted = 0
    identity_switches = identity_comparisons = 0
    last_track = {}
    for truth in sorted(labels["frames"], key=lambda r: r["frame_index"]):
        idx = truth["frame_index"]
        if idx not in rows or idx in seen:
            raise ValueError(
                "Each labeled frame must occur once and exist in the report"
            )
        seen.add(idx)
        row = rows[idx]
        visible = truth.get("ball_visible")
        if visible is not None:
            if not isinstance(visible, bool):
                raise ValueError("ball_visible must be true, false, or null")
            predicted = 1 in row["class_id"]
            ball_labeled += 1
            ball_tp += int(visible and predicted)
            ball_fp += int(not visible and predicted)
            ball_fn += int(visible and not predicted)
        # null = explicitly no possession; omitted = not labeled.
        if "possession_team" in truth:
            team = truth["possession_team"]
            if team not in (None, 0, 1):
                raise ValueError("possession_team must be null, 0, or 1")
            prediction = row["evidence"]["team_id"]
            possession_labeled += 1
            possession_predicted += int(prediction is not None)
            possession_correct += int(prediction == team)
        players = truth.get("players", [])
        ids = [p["player_id"] for p in players]
        if len(ids) != len(set(ids)):
            raise ValueError("Human player IDs must be unique within a labeled frame")
        for p in players:
            box = np.asarray(p["xyxy"], dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])
            ):
                raise ValueError(
                    "Player boxes must have positive area and finite coordinates"
                )
            if p["team_id"] not in (0, 1):
                raise ValueError("Labeled player teams must be 0 or 1")
        player_labeled += len(players)
        candidates = sorted(
            (
                (_iou(p["xyxy"], box), j, i)
                for j, p in enumerate(players)
                for i, (box, cls) in enumerate(zip(row["xyxy"], row["class_id"]))
                if cls == 0
            ),
            reverse=True,
        )
        used_truth, used_predictions = set(), set()
        for overlap, j, i in candidates:
            if overlap < iou_threshold or j in used_truth or i in used_predictions:
                continue
            used_truth.add(j)
            used_predictions.add(i)
            team_matched += 1
            team_correct += int(players[j]["team_id"] == row["team_id"][i])
            identity = players[j]["player_id"]
            track = row["tracker_id"][i]
            if track < 0:
                continue
            if identity in last_track:
                identity_comparisons += 1
                identity_switches += int(last_track[identity] != track)
            last_track[identity] = track
    return {
        "contains_manual_corrections": bool(
            review.get("corrections")
            or review.get("ball_corrections")
            or review.get("possession_corrections")
        ),
        "annotation_status": labels.get("annotation_status", "unspecified"),
        "labeled_frames": len(seen),
        "ball_labeled_frames": ball_labeled,
        "ball_visible_frame_recall": _ratio(ball_tp, ball_tp + ball_fn),
        "ball_visible_frame_precision": _ratio(ball_tp, ball_tp + ball_fp),
        "labeled_player_boxes": player_labeled,
        "matched_player_boxes": team_matched,
        "player_box_recall": _ratio(team_matched, player_labeled),
        "team_accuracy_on_matched_players": _ratio(team_correct, team_matched),
        "identity_switches_between_labeled_matches": identity_switches,
        "identity_comparisons": identity_comparisons,
        "possession_labeled_frames": possession_labeled,
        "possession_agreement": _ratio(possession_correct, possession_labeled),
        "possession_prediction_coverage": _ratio(
            possession_predicted, possession_labeled
        ),
        "iou_threshold": iou_threshold,
        "note": "Partial-label diagnostic, not a full tracking benchmark. "
        "Null metrics mean no eligible observations.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("labels")
    args = parser.parse_args()
    with open(args.report) as f:
        report = json.load(f)
    with open(args.labels) as f:
        labels = json.load(f)
    print(json.dumps(evaluate(report["review"], labels), indent=2))


if __name__ == "__main__":
    main()
