import numpy as np

from sportvision.tracking import SportsTracker


class TestSportsTracker:
    def test_init_default(self):
        tracker = SportsTracker()
        assert tracker.tracker_type == "bytetrack"

    def test_update_adds_tracker_ids(self, sample_detections):
        tracker = SportsTracker()
        result = tracker.update(sample_detections)
        assert "tracker_id" in result
        assert len(result["tracker_id"]) == len(result["xyxy"])

    def test_update_consistent_ids_across_frames(self, sample_detections):
        tracker = SportsTracker()
        tracker.update(sample_detections)
        r2 = tracker.update(sample_detections)
        r3 = tracker.update(sample_detections)
        assert len(r2["tracker_id"]) > 0
        np.testing.assert_array_equal(r2["tracker_id"], r3["tracker_id"])

    def test_reset_clears_state(self, sample_detections):
        tracker = SportsTracker()
        tracker.update(sample_detections)
        tracker.reset()
        result = tracker.update(sample_detections)
        assert "tracker_id" in result


def test_ball_survives_first_observation_and_does_not_get_person_id():
    tracker = SportsTracker()
    detections = {
        "xyxy": np.array([[0, 0, 100, 200], [20, 20, 30, 30]], float),
        "class_id": np.array([0, 1]),
        "confidence": np.array([0.9, 0.16]),
    }
    for _ in range(3):
        result = tracker.update(detections)
        ball = result["class_id"] == 1
        assert ball.sum() == 1
        assert result["tracker_id"][ball].tolist() == [-1]
        np.testing.assert_array_equal(result["xyxy"][ball], [[20, 20, 30, 30]])
    assert (result["tracker_id"][result["class_id"] == 0] >= 0).all()


def test_motion_keeps_confirmed_player_identity():
    tracker = SportsTracker()
    ids = []
    for x in (0, 0, 20, 40, 60):
        result = tracker.update(
            {
                "xyxy": np.array([[x, 0, x + 100, 200]], float),
                "class_id": np.array([0]),
                "confidence": np.array([0.95]),
            }
        )
        ids.extend(tid for tid in result["tracker_id"] if tid >= 0)
    assert len(ids) >= 4
    assert len(set(ids)) == 1


def test_unsupported_tracker_and_invalid_thresholds_fail_explicitly():
    import pytest

    for kwargs in (
        {"tracker_type": "typo"},
        {"match_thresh": 0},
        {"track_thresh": float("nan")},
    ):
        with pytest.raises(ValueError):
            SportsTracker(**kwargs)
