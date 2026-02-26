import numpy as np
import pytest
from sportvision.teams import TeamClassifier


class TestTeamClassifier:
    def _make_crops(self, n: int, color: tuple[int, int, int]) -> list[np.ndarray]:
        crops = []
        for _ in range(n):
            crop = np.full((160, 80, 3), color, dtype=np.uint8)
            crop += np.random.randint(0, 20, crop.shape, dtype=np.uint8)
            crops.append(crop)
        return crops

    def test_init(self):
        clf = TeamClassifier(n_teams=2)
        assert clf.n_teams == 2

    def test_fit_and_predict(self):
        clf = TeamClassifier(n_teams=2)
        red_crops = self._make_crops(10, (200, 50, 50))
        blue_crops = self._make_crops(10, (50, 50, 200))
        clf.fit(red_crops + blue_crops)
        red_ids = clf.predict(red_crops)
        blue_ids = clf.predict(blue_crops)
        assert len(set(red_ids)) == 1
        assert len(set(blue_ids)) == 1
        assert red_ids[0] != blue_ids[0]

    def test_predict_before_fit_raises(self):
        clf = TeamClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([np.zeros((160, 80, 3), dtype=np.uint8)])

    def test_extract_features_shape(self):
        clf = TeamClassifier()
        crop = np.full((160, 80, 3), 128, dtype=np.uint8)
        feat = clf._extract_features(crop)
        assert feat.ndim == 1
