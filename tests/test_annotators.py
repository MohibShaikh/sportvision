import numpy as np
from sportvision.annotators import TeamColorAnnotator, StatsOverlayAnnotator, TrailAnnotator


class TestTeamColorAnnotator:
    def test_annotate_draws_boxes(self, sample_frame, sample_detections):
        ann = TeamColorAnnotator(home_color=(255, 0, 0), away_color=(0, 0, 255))
        team_ids = np.array([0, 1, 0, 1, 0, 1, -1])
        result = ann.annotate(sample_frame.copy(), sample_detections["xyxy"], team_ids)
        assert result.shape == sample_frame.shape
        assert not np.array_equal(result, sample_frame)


class TestStatsOverlayAnnotator:
    def test_annotate_adds_text(self, sample_frame):
        ann = StatsOverlayAnnotator()
        result = ann.annotate(sample_frame.copy(), possession={0: 0.55, 1: 0.45})
        assert result.shape == sample_frame.shape
        assert not np.array_equal(result, sample_frame)


class TestTrailAnnotator:
    def test_annotate_draws_trails(self, sample_frame):
        ann = TrailAnnotator(trail_length=5)
        tracks = {0: [(100, 100), (110, 105), (120, 110)]}
        result = ann.annotate(sample_frame.copy(), tracks)
        assert result.shape == sample_frame.shape
