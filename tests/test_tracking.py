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
        assert len(r2["tracker_id"]) > 0

    def test_reset_clears_state(self, sample_detections):
        tracker = SportsTracker()
        tracker.update(sample_detections)
        tracker.reset()
        result = tracker.update(sample_detections)
        assert "tracker_id" in result
