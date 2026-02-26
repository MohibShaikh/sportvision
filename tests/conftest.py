import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """720p BGR frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Fake detections: 6 players, 1 ball."""
    rng = np.random.default_rng(42)
    n = 7
    x1 = rng.integers(0, 1000, n).astype(float)
    y1 = rng.integers(0, 500, n).astype(float)
    return {
        "xyxy": np.column_stack([x1, y1, x1 + 80, y1 + 160]),
        "class_id": np.array([0, 0, 0, 0, 0, 0, 1]),
        "confidence": rng.uniform(0.7, 1.0, n),
        "tracker_id": np.arange(n),
    }
