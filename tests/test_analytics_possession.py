import numpy as np

from sportvision.analytics.possession import PossessionTracker


class TestPossessionTracker:
    def test_init(self):
        pt = PossessionTracker(ball_proximity_thresh=50)
        assert pt.ball_proximity_thresh == 50

    def test_update_and_stats(self):
        pt = PossessionTracker()
        player_positions = np.array([[100, 100], [500, 500]])
        team_ids = np.array([0, 1])
        ball_position = np.array([110, 105])
        pt.update(player_positions, team_ids, ball_position)
        stats = pt.get_stats()
        assert stats[0] > stats[1]
        assert abs(sum(stats.values()) - 1.0) < 1e-6

    def test_no_updates_returns_even_split(self):
        pt = PossessionTracker(n_teams=2)
        stats = pt.get_stats()
        assert stats[0] == 0.5
        assert stats[1] == 0.5

    def test_multiple_updates(self):
        pt = PossessionTracker()
        for _ in range(10):
            pt.update(np.array([[100, 100]]), np.array([0]), np.array([105, 102]))
        for _ in range(5):
            pt.update(np.array([[200, 200]]), np.array([1]), np.array([205, 202]))
        stats = pt.get_stats()
        assert stats[0] > stats[1]

    def test_reset(self):
        pt = PossessionTracker()
        pt.update(np.array([[100, 100]]), np.array([0]), np.array([105, 100]))
        pt.reset()
        stats = pt.get_stats()
        assert stats[0] == 0.5
