from copy import deepcopy

import numpy as np
import pytest

from sportvision.evaluation import evaluate
from sportvision.review import (
    correct_review,
    frame_record,
    possession_evidence,
    review_summary,
    timeline_segments,
)


def make_review():
    tracked = {
        "xyxy": [[0, 0, 20, 20], [100, 0, 120, 20], [8, 8, 12, 12]],
        "class_id": [0, 0, 1],
        "tracker_id": [7, 9, 11],
        "team_id": [0, 1, -1],
        "confidence": [0.9, 0.9, 0.9],
    }
    rows = [frame_record(i, tracked) for i in range(3)]
    rows[2]["class_id"][-1] = 5
    rows[2]["evidence"] = possession_evidence(rows[2])
    return {
        "fps": 30,
        "width": 150,
        "height": 100,
        "threshold_px": 50.0,
        "clip_sha256": "fixture",
        "frames": rows,
    }


def test_correction_applies_to_track_and_recomputes_history_without_mutation():
    original = make_review()
    saved = deepcopy(original)
    corrected = correct_review(original, {7: 1})
    assert review_summary(corrected)["possession"] == {0: 0, 1: 1}
    assert corrected["frames"][0]["measured_team_id"] == [0, 1, -1]
    assert corrected["corrections"] == {"7": 1}
    assert original == saved
    assert review_summary(correct_review(original, {}))["possession"] == {0: 1, 1: 0}


def test_ignored_player_does_not_give_possession_to_someone_else():
    corrected = correct_review(make_review(), {7: -2})
    assert corrected["frames"][0]["evidence"]["reason"] == "ignored_player"
    assert review_summary(corrected)["attributed_frames"] == 0
    assert review_summary(corrected)["excluded_frames"] == {
        "ignored_player": 2,
        "no_ball": 1,
    }


def test_timeline_covers_each_frame_with_exact_boundaries():
    segments = timeline_segments(make_review())
    assert [(s["start_frame"], s["end_frame"]) for s in segments] == [(0, 1), (2, 2)]
    assert segments[0]["end_s"] == segments[1]["start_s"] == 2 / 30
    assert segments[-1]["end_s"] == 0.1


def test_multiple_balls_are_ambiguous():
    row = make_review()["frames"][0]
    row["class_id"][1] = 1
    assert possession_evidence(row)["reason"] == "multiple_balls"


@pytest.mark.parametrize("overrides", [{999: 0}, {7: 10}, {11: 0}])
def test_invalid_corrections_rejected(overrides):
    with pytest.raises(ValueError):
        correct_review(make_review(), overrides)


def test_evaluation_counts_misses_and_identity_changes():
    review = make_review()
    review["frames"][1]["tracker_id"][0] = 15
    labels = {
        "clip_sha256": "fixture",
        "frames": [
            {
                "frame_index": i,
                "ball_visible": True,
                "possession_team": 0,
                "players": [
                    {"player_id": "white", "xyxy": [0, 0, 20, 20], "team_id": 0}
                ],
            }
            for i in range(3)
        ],
    }
    result = evaluate(review, labels)
    assert result["ball_visible_frame_recall"] == 2 / 3
    assert result["possession_agreement"] == 2 / 3
    assert result["team_accuracy_on_matched_players"] == 1
    assert result["identity_switches_between_labeled_matches"] == 2
    assert result["identity_comparisons"] == 2


def test_unlabeled_is_not_a_negative_and_wrong_clip_is_rejected():
    labels = {"clip_sha256": "fixture", "frames": [{"frame_index": 0}]}
    result = evaluate(make_review(), labels)
    assert result["ball_visible_frame_recall"] is None
    assert result["possession_agreement"] is None
    labels["clip_sha256"] = "wrong"
    with pytest.raises(ValueError, match="same source clip"):
        evaluate(make_review(), labels)


def test_corrected_export_uses_cached_observations(tmp_path):
    import cv2

    from sportvision.video import export_review, read_frame

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (150, 100)
    )
    for _ in range(3):
        writer.write(np.zeros((100, 150, 3), np.uint8))
    writer.release()
    out = tmp_path / "corrected.mp4"
    review = correct_review(make_review(), {7: 1})
    progress = []
    export_review(source, out, review, lambda n, total: progress.append((n, total)))
    assert progress[-1] == (3, 3)
    frame = read_frame(out, 2)
    assert frame.shape == (100, 150, 3)
    with pytest.raises(ValueError, match="Source ended"):
        review["frames"].append({**review["frames"][-1], "frame_index": 3})
        export_review(source, out, review)


def test_unconfirmed_ids_cannot_link_people_or_receive_possession():
    review = make_review()
    review["frames"][0]["tracker_id"][:2] = [-1, -1]
    assert possession_evidence(review["frames"][0])["reason"] == "untracked_player"
    with pytest.raises(ValueError):
        correct_review(review, {-1: 1})


def test_missed_ball_correction_is_local_and_preserves_model_output():
    original = make_review()
    corrected = correct_review(original, {}, ball_overrides={2: [110.0, 10.0]})
    assert corrected["frames"][2]["evidence"]["team_id"] == 1
    assert corrected["frames"][2]["evidence"]["ball_source"] == "manual"
    assert corrected["frames"][2]["class_id"] == original["frames"][2]["class_id"]
    assert "manual_ball_xy" not in corrected["frames"][1]
    assert review_summary(corrected)["manually_marked_ball_frames"] == 1
    restored = correct_review(corrected, {})
    assert restored["frames"][2]["evidence"]["reason"] == "no_ball"


def test_manual_possession_does_not_pretend_detector_found_ball():
    original = make_review()
    corrected = correct_review(original, {}, possession_overrides={2: 1})
    evidence = corrected["frames"][2]["evidence"]
    assert evidence["reason"] == "manual_possession"
    assert evidence["automatic_reason"] == "no_ball"
    assert evidence["ball_xy"] is None
    assert review_summary(corrected)["possession"] == {0: 2 / 3, 1: 1 / 3}
    assert review_summary(corrected)["manually_confirmed_frames"] == 1
    none = correct_review(original, {}, possession_overrides={0: None})
    assert none["frames"][0]["evidence"]["reason"] == "manual_no_possession"
    assert review_summary(none)["attributed_frames"] == 1


@pytest.mark.parametrize("point", [[-1, 10], [150, 10], [10, 100], [float("nan"), 0]])
def test_bad_manual_ball_coordinates_are_rejected(point):
    with pytest.raises(ValueError, match="inside a reviewed frame"):
        correct_review(make_review(), {}, ball_overrides={0: point})


def test_body_proximity_is_scale_invariant_and_abstains_on_overlap():
    from sportvision.review import possession_evidence

    row = {
        "xyxy": [[0, 0, 100, 200], [300, 0, 400, 200], [105, 25, 115, 35]],
        "class_id": [0, 0, 1],
        "team_id": [0, 1, -1],
        "tracker_id": [7, 9, -1],
    }
    assert possession_evidence(row)["reason"] == "too_far"
    assert possession_evidence(row, method="body")["team_id"] == 0
    row["xyxy"] = (np.asarray(row["xyxy"]) * 3).tolist()
    assert possession_evidence(row, method="body")["team_id"] == 0
    row["xyxy"][1] = row["xyxy"][0]
    assert possession_evidence(row, method="body")["reason"] == "ambiguous_players"


def test_corrections_preserve_body_possession_method():
    review = make_review()
    review["possession_method"] = "body"
    corrected = correct_review(review, {})
    assert all(row["evidence"]["method"] == "body" for row in corrected["frames"])
