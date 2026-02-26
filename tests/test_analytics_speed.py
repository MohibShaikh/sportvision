import numpy as np

from sportvision.analytics.distance import DistanceCalculator
from sportvision.analytics.speed import SpeedEstimator


class TestSpeedEstimator:
    def test_init(self):
        se = SpeedEstimator(fps=30)
        assert se.fps == 30

    def test_calculate_speed(self):
        se = SpeedEstimator(fps=30)
        positions = {0: [(0, 0), (10, 0)]}
        frame_indices = {0: [0, 30]}
        speeds = se.calculate(positions, frame_indices)
        np.testing.assert_allclose(speeds[0], 36.0, atol=1.0)

    def test_stationary_player(self):
        se = SpeedEstimator(fps=30)
        positions = {0: [(5, 5), (5, 5)]}
        frame_indices = {0: [0, 30]}
        speeds = se.calculate(positions, frame_indices)
        assert speeds[0] == 0.0


class TestDistanceCalculator:
    def test_total_distance(self):
        dc = DistanceCalculator()
        positions = {0: [(0, 0), (3, 4), (6, 8)]}
        dist = dc.calculate(positions)
        np.testing.assert_allclose(dist[0], 10.0, atol=0.01)

    def test_no_movement(self):
        dc = DistanceCalculator()
        positions = {0: [(5, 5), (5, 5)]}
        dist = dc.calculate(positions)
        assert dist[0] == 0.0
