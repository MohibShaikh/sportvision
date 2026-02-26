import numpy as np
from sportvision.analytics.heatmap import HeatmapGenerator


class TestHeatmapGenerator:
    def test_init(self):
        hg = HeatmapGenerator(resolution=(105, 68))
        assert hg.resolution == (105, 68)

    def test_update_and_render(self):
        hg = HeatmapGenerator(resolution=(105, 68))
        positions = np.array([[50.0, 30.0], [52.0, 34.0]])
        team_ids = np.array([0, 0])
        hg.update(positions, team_ids)
        img = hg.render(team=0)
        assert img.shape == (68, 105, 3)
        assert img.dtype == np.uint8

    def test_render_nonzero_after_update(self):
        hg = HeatmapGenerator(resolution=(105, 68))
        for _ in range(50):
            hg.update(np.array([[52.0, 34.0]]), np.array([0]))
        img = hg.render(team=0)
        assert img.sum() > 0

    def test_reset(self):
        hg = HeatmapGenerator(resolution=(105, 68))
        hg.update(np.array([[50.0, 30.0]]), np.array([0]))
        hg.reset()
        img = hg.render(team=0)
        assert img.sum() == 0
