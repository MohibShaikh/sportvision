import numpy as np
import pytest

from sportvision.homography import FieldHomography


class TestFieldHomography:
    def test_soccer_factory(self):
        h = FieldHomography.soccer()
        assert h.field_length == 105
        assert h.field_width == 68

    def test_set_keypoints_and_transform(self):
        h = FieldHomography.soccer()
        src = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)
        dst = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
        h.set_keypoints(src, dst)
        pts = np.array([[640, 360]], dtype=np.float32)
        result = h.transform(pts)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [52.5, 34.0], atol=1.0)

    def test_transform_without_keypoints_raises(self):
        h = FieldHomography.soccer()
        with pytest.raises(RuntimeError):
            h.transform(np.array([[0, 0]], dtype=np.float32))

    def test_set_keypoints_minimum_four(self):
        h = FieldHomography.soccer()
        with pytest.raises(ValueError):
            h.set_keypoints(
                np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
                np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
            )
