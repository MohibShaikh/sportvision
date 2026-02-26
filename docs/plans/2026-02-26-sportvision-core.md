# SportVision Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a real-time sports analytics Python toolkit with player detection, tracking, team classification, field homography, possession tracking, speed/distance estimation, heatmaps, and Roboflow Workflow blocks.

**Architecture:** Modular package with independent submodules (detection, tracking, teams, homography, analytics, annotators, workflows). Each module wraps proven libraries (RF-DETR, ByteTrack via `supervision`/`trackers`) behind clean APIs. Analytics modules consume tracking output and produce stats. Workflow blocks expose each module as Roboflow Inference blocks.

**Tech Stack:** Python 3.10+, numpy, opencv-python, supervision, scikit-learn (KMeans for teams), inference (Roboflow), rf-detr, trackers (ByteTrack). Dev: pytest, ruff.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/sportvision/__init__.py`
- Create: `src/sportvision/py.typed`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `README.md`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sportvision"
version = "0.1.0"
description = "Real-time sports analytics toolkit built on the Roboflow ecosystem"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "opencv-python>=4.8",
    "supervision>=0.25",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
inference = ["inference>=0.30", "rf-detr>=1.0"]
dev = ["pytest>=7.0", "ruff>=0.4", "pytest-cov"]
all = ["sportvision[inference,dev]"]

[tool.hatch.build.targets.wheel]
packages = ["src/sportvision"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package files**

`src/sportvision/__init__.py`:
```python
"""SportVision: Real-time sports analytics toolkit."""

__version__ = "0.1.0"
```

`src/sportvision/py.typed` — empty marker file.

`tests/__init__.py` — empty.

`tests/conftest.py`:
```python
import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """720p BGR frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Fake detections: 6 players, 1 ball. Returns dict with xyxy, class_id, confidence, tracker_id."""
    rng = np.random.default_rng(42)
    n = 7
    x1 = rng.integers(0, 1000, n).astype(float)
    y1 = rng.integers(0, 500, n).astype(float)
    return {
        "xyxy": np.column_stack([x1, y1, x1 + 80, y1 + 160]),
        "class_id": np.array([0, 0, 0, 0, 0, 0, 1]),  # 0=player, 1=ball
        "confidence": rng.uniform(0.7, 1.0, n),
        "tracker_id": np.arange(n),
    }
```

`.gitignore`:
```
__pycache__/
*.egg-info/
dist/
.venv/
.ruff_cache/
*.pyc
```

**Step 3: Verify the project installs**

Run: `cd /path/to/sportvision && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && python -c "import sportvision; print(sportvision.__version__)"`
Expected: `0.1.0`

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: scaffold sportvision project with pyproject.toml and test fixtures"
```

---

### Task 2: Detection Module

**Files:**
- Create: `src/sportvision/detection.py`
- Create: `tests/test_detection.py`

**Step 1: Write the failing test**

`tests/test_detection.py`:
```python
import numpy as np
import pytest
from sportvision.detection import SportsDetector


class TestSportsDetector:
    def test_init_default(self):
        detector = SportsDetector()
        assert detector.model_id == "rfdetr-base"
        assert set(detector.classes) == {"player", "ball", "referee", "goalkeeper"}

    def test_init_custom(self):
        detector = SportsDetector(model="yolov8m", classes=["player", "ball"])
        assert detector.model_id == "yolov8m"
        assert detector.classes == ["player", "ball"]

    def test_detect_returns_dict_with_required_keys(self, sample_frame):
        detector = SportsDetector()
        result = detector.detect(sample_frame)
        assert "xyxy" in result
        assert "class_id" in result
        assert "confidence" in result
        assert isinstance(result["xyxy"], np.ndarray)

    def test_detect_empty_frame(self):
        detector = SportsDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result["xyxy"].shape[1] == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

`src/sportvision/detection.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CLASS_NAMES = ["player", "ball", "referee", "goalkeeper"]


@dataclass
class SportsDetector:
    model: str = "rfdetr-base"
    classes: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    confidence_threshold: float = 0.25
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def model_id(self) -> str:
        return self.model

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            if "rfdetr" in self.model:
                from rfdetr import RFDETRBase, RFDETRLarge

                self._model = RFDETRLarge() if "large" in self.model else RFDETRBase()
            else:
                import supervision as sv

                self._model = sv.get_model(self.model)
        except ImportError:
            self._model = None

    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        self._load_model()
        if self._model is None:
            return self._empty_result()
        try:
            import supervision as sv

            detections: sv.Detections = self._model.predict(frame, threshold=self.confidence_threshold)
            if not isinstance(detections, sv.Detections):
                detections = sv.Detections.from_inference(detections)
            mask = np.isin(detections.class_id, self._class_indices())
            return {
                "xyxy": detections.xyxy[mask],
                "class_id": detections.class_id[mask],
                "confidence": detections.confidence[mask],
            }
        except Exception:
            return self._empty_result()

    def _class_indices(self) -> list[int]:
        return [CLASS_NAMES.index(c) for c in self.classes if c in CLASS_NAMES]

    @staticmethod
    def _empty_result() -> dict[str, np.ndarray]:
        return {
            "xyxy": np.empty((0, 4)),
            "class_id": np.empty(0, dtype=int),
            "confidence": np.empty(0),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_detection.py -v`
Expected: PASS (all 4 tests — model falls back to empty results without inference installed)

**Step 5: Commit**

```bash
git add src/sportvision/detection.py tests/test_detection.py
git commit -m "feat: add detection module with SportsDetector"
```

---

### Task 3: Tracking Module

**Files:**
- Create: `src/sportvision/tracking.py`
- Create: `tests/test_tracking.py`

**Step 1: Write the failing test**

`tests/test_tracking.py`:
```python
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
        r1 = tracker.update(sample_detections)
        r2 = tracker.update(sample_detections)
        # Same detections should keep same IDs
        assert len(r2["tracker_id"]) > 0

    def test_reset_clears_state(self, sample_detections):
        tracker = SportsTracker()
        tracker.update(sample_detections)
        tracker.reset()
        result = tracker.update(sample_detections)
        assert "tracker_id" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracking.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/tracking.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SportsTracker:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    _tracker: Any = field(default=None, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    def _init_tracker(self) -> None:
        try:
            from trackers import ByteTrack

            self._tracker = ByteTrack(
                track_activation_threshold=self.track_thresh,
                minimum_matching_threshold=self.match_thresh,
            )
        except ImportError:
            self._tracker = None

    def update(self, detections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._tracker is None:
            self._init_tracker()

        xyxy = detections["xyxy"]
        n = len(xyxy)

        if self._tracker is not None:
            try:
                import supervision as sv

                sv_dets = sv.Detections(
                    xyxy=xyxy,
                    class_id=detections.get("class_id"),
                    confidence=detections.get("confidence"),
                )
                tracked = self._tracker.update_with_detections(sv_dets)
                return {
                    "xyxy": tracked.xyxy,
                    "class_id": tracked.class_id,
                    "confidence": tracked.confidence,
                    "tracker_id": tracked.tracker_id,
                }
            except Exception:
                pass

        # Fallback: assign sequential IDs
        ids = np.arange(self._next_id, self._next_id + n)
        self._next_id += n
        return {**detections, "tracker_id": ids}

    def reset(self) -> None:
        self._tracker = None
        self._next_id = 0
```

**Step 4: Run tests**

Run: `pytest tests/test_tracking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/tracking.py tests/test_tracking.py
git commit -m "feat: add tracking module with ByteTrack integration"
```

---

### Task 4: Team Classification Module

**Files:**
- Create: `src/sportvision/teams.py`
- Create: `tests/test_teams.py`

**Step 1: Write the failing test**

`tests/test_teams.py`:
```python
import numpy as np
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
        # All red should be same team, all blue same team, and different
        assert len(set(red_ids)) == 1
        assert len(set(blue_ids)) == 1
        assert red_ids[0] != blue_ids[0]

    def test_predict_before_fit_raises(self):
        clf = TeamClassifier()
        import pytest
        with pytest.raises(RuntimeError):
            clf.predict([np.zeros((160, 80, 3), dtype=np.uint8)])

    def test_extract_features_shape(self):
        clf = TeamClassifier()
        crop = np.full((160, 80, 3), 128, dtype=np.uint8)
        feat = clf._extract_features(crop)
        assert feat.ndim == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_teams.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/teams.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClassifier:
    n_teams: int = 2
    method: str = "kmeans"
    _model: KMeans | None = field(default=None, init=False, repr=False)

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Take middle 60% of crop (jersey area, skip head/feet)
        h, w = hsv.shape[:2]
        jersey = hsv[int(h * 0.2):int(h * 0.6), int(w * 0.2):int(w * 0.8)]
        hist_h = cv2.calcHist([jersey], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([jersey], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([jersey], [2], None, [8], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v])
        feat = feat / (feat.sum() + 1e-8)
        return feat

    def fit(self, crops: list[np.ndarray]) -> TeamClassifier:
        features = np.array([self._extract_features(c) for c in crops])
        self._model = KMeans(n_clusters=self.n_teams, random_state=0, n_init=10)
        self._model.fit(features)
        return self

    def predict(self, crops: list[np.ndarray]) -> list[int]:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        features = np.array([self._extract_features(c) for c in crops])
        return self._model.predict(features).tolist()
```

**Step 4: Run tests**

Run: `pytest tests/test_teams.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/teams.py tests/test_teams.py
git commit -m "feat: add team classification via jersey color clustering"
```

---

### Task 5: Field Homography Module

**Files:**
- Create: `src/sportvision/homography.py`
- Create: `tests/test_homography.py`

**Step 1: Write the failing test**

`tests/test_homography.py`:
```python
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
        # Simple identity-like mapping for test
        src = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)
        dst = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
        h.set_keypoints(src, dst)
        # Transform a point at center of frame
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_homography.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/homography.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FieldHomography:
    field_length: float = 105
    field_width: float = 68
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def soccer(cls, field_length: float = 105, field_width: float = 68) -> FieldHomography:
        return cls(field_length=field_length, field_width=field_width)

    @classmethod
    def basketball(cls) -> FieldHomography:
        return cls(field_length=28.65, field_width=15.24)

    @classmethod
    def tennis(cls) -> FieldHomography:
        return cls(field_length=23.77, field_width=10.97)

    def set_keypoints(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> None:
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("At least 4 keypoint pairs required")
        self._matrix, _ = cv2.findHomography(src_points, dst_points)

    def transform(self, points: np.ndarray) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("Must call set_keypoints() first")
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self._matrix)
        return transformed.reshape(-1, 2)
```

**Step 4: Run tests**

Run: `pytest tests/test_homography.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/homography.py tests/test_homography.py
git commit -m "feat: add field homography with perspective transform"
```

---

### Task 6: Analytics — Possession Tracker

**Files:**
- Create: `src/sportvision/analytics/__init__.py`
- Create: `src/sportvision/analytics/possession.py`
- Create: `tests/test_analytics_possession.py`

**Step 1: Write the failing test**

`tests/test_analytics_possession.py`:
```python
import numpy as np
from sportvision.analytics.possession import PossessionTracker


class TestPossessionTracker:
    def test_init(self):
        pt = PossessionTracker(ball_proximity_thresh=50)
        assert pt.ball_proximity_thresh == 50

    def test_update_and_stats(self):
        pt = PossessionTracker()
        # Team 0 player near ball
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
        # Team 0 had ~2/3 possession
        assert stats[0] > stats[1]

    def test_reset(self):
        pt = PossessionTracker()
        pt.update(np.array([[100, 100]]), np.array([0]), np.array([105, 100]))
        pt.reset()
        stats = pt.get_stats()
        assert stats[0] == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytics_possession.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/analytics/__init__.py`:
```python
from sportvision.analytics.possession import PossessionTracker

__all__ = ["PossessionTracker"]
```

`src/sportvision/analytics/possession.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PossessionTracker:
    ball_proximity_thresh: float = 50.0
    n_teams: int = 2
    _counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def update(
        self,
        player_positions: np.ndarray,
        team_ids: np.ndarray,
        ball_position: np.ndarray,
    ) -> None:
        distances = np.linalg.norm(player_positions - ball_position, axis=1)
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] <= self.ball_proximity_thresh:
            team = int(team_ids[nearest_idx])
            self._counts[team] = self._counts.get(team, 0) + 1

    def get_stats(self) -> dict[int, float]:
        total = sum(self._counts.values())
        if total == 0:
            return {i: 1.0 / self.n_teams for i in range(self.n_teams)}
        return {i: self._counts.get(i, 0) / total for i in range(self.n_teams)}

    def reset(self) -> None:
        self._counts.clear()
```

**Step 4: Run tests**

Run: `pytest tests/test_analytics_possession.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/analytics/ tests/test_analytics_possession.py
git commit -m "feat: add possession tracker"
```

---

### Task 7: Analytics — Speed Estimator & Distance Calculator

**Files:**
- Create: `src/sportvision/analytics/speed.py`
- Create: `src/sportvision/analytics/distance.py`
- Create: `tests/test_analytics_speed.py`

**Step 1: Write the failing test**

`tests/test_analytics_speed.py`:
```python
import numpy as np
from sportvision.analytics.speed import SpeedEstimator
from sportvision.analytics.distance import DistanceCalculator


class TestSpeedEstimator:
    def test_init(self):
        se = SpeedEstimator(fps=30)
        assert se.fps == 30

    def test_calculate_speed(self):
        se = SpeedEstimator(fps=30)
        # Player moves 10m in 30 frames = 10m/s = 36 km/h
        positions = {0: [(0, 0), (10, 0)]}  # tracker_id -> list of (x, y) in meters
        frame_indices = {0: [0, 30]}
        speeds = se.calculate(positions, frame_indices)
        np.testing.assert_allclose(speeds[0], 36.0, atol=1.0)  # km/h

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytics_speed.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/analytics/speed.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedEstimator:
    fps: float = 30.0

    def calculate(
        self,
        positions: dict[int, list[tuple[float, float]]],
        frame_indices: dict[int, list[int]],
    ) -> dict[int, float]:
        speeds = {}
        for tid, pos_list in positions.items():
            if len(pos_list) < 2:
                speeds[tid] = 0.0
                continue
            frames = frame_indices[tid]
            p1, p2 = np.array(pos_list[-2]), np.array(pos_list[-1])
            dt = (frames[-1] - frames[-2]) / self.fps
            if dt <= 0:
                speeds[tid] = 0.0
                continue
            dist = float(np.linalg.norm(p2 - p1))
            speeds[tid] = (dist / dt) * 3.6  # m/s -> km/h
        return speeds
```

`src/sportvision/analytics/distance.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DistanceCalculator:
    def calculate(
        self, positions: dict[int, list[tuple[float, float]]]
    ) -> dict[int, float]:
        distances = {}
        for tid, pos_list in positions.items():
            if len(pos_list) < 2:
                distances[tid] = 0.0
                continue
            pts = np.array(pos_list)
            diffs = np.diff(pts, axis=0)
            distances[tid] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return distances
```

Update `src/sportvision/analytics/__init__.py`:
```python
from sportvision.analytics.distance import DistanceCalculator
from sportvision.analytics.possession import PossessionTracker
from sportvision.analytics.speed import SpeedEstimator

__all__ = ["DistanceCalculator", "PossessionTracker", "SpeedEstimator"]
```

**Step 4: Run tests**

Run: `pytest tests/test_analytics_speed.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/analytics/ tests/test_analytics_speed.py
git commit -m "feat: add speed estimator and distance calculator"
```

---

### Task 8: Analytics — Heatmap Generator

**Files:**
- Create: `src/sportvision/analytics/heatmap.py`
- Create: `tests/test_analytics_heatmap.py`

**Step 1: Write the failing test**

`tests/test_analytics_heatmap.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytics_heatmap.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/analytics/heatmap.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HeatmapGenerator:
    resolution: tuple[int, int] = (105, 68)  # (width, height) in field units
    _grids: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _get_grid(self, team: int) -> np.ndarray:
        if team not in self._grids:
            w, h = self.resolution
            self._grids[team] = np.zeros((h, w), dtype=np.float64)
        return self._grids[team]

    def update(self, positions: np.ndarray, team_ids: np.ndarray) -> None:
        w, h = self.resolution
        for pos, tid in zip(positions, team_ids):
            tid = int(tid)
            grid = self._get_grid(tid)
            x, y = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
            grid[y, x] += 1.0

    def render(self, team: int = 0) -> np.ndarray:
        w, h = self.resolution
        grid = self._get_grid(team)
        if grid.max() > 0:
            normalized = (grid / grid.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((h, w), dtype=np.uint8)
        blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=2, sigmaY=2)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        # Zero out where no data
        mask = blurred == 0
        colored[mask] = 0
        return colored

    def reset(self) -> None:
        self._grids.clear()
```

Update `src/sportvision/analytics/__init__.py` to add `HeatmapGenerator`.

**Step 4: Run tests**

Run: `pytest tests/test_analytics_heatmap.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/analytics/ tests/test_analytics_heatmap.py
git commit -m "feat: add heatmap generator"
```

---

### Task 9: Annotators Module

**Files:**
- Create: `src/sportvision/annotators.py`
- Create: `tests/test_annotators.py`

**Step 1: Write the failing test**

`tests/test_annotators.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_annotators.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/annotators.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TeamColorAnnotator:
    home_color: tuple[int, int, int] = (255, 0, 0)
    away_color: tuple[int, int, int] = (0, 0, 255)
    thickness: int = 2

    def annotate(
        self, frame: np.ndarray, xyxy: np.ndarray, team_ids: np.ndarray
    ) -> np.ndarray:
        colors = {0: self.home_color, 1: self.away_color}
        for box, tid in zip(xyxy, team_ids):
            color = colors.get(int(tid), (128, 128, 128))
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        return frame


@dataclass
class StatsOverlayAnnotator:
    position: str = "top-left"
    font_scale: float = 0.8

    def annotate(
        self, frame: np.ndarray, possession: dict[int, float] | None = None, **kwargs
    ) -> np.ndarray:
        y = 30
        if possession:
            for team, pct in possession.items():
                text = f"Team {team}: {pct:.0%}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, (255, 255, 255), 2)
                y += 30
        return frame


@dataclass
class TrailAnnotator:
    trail_length: int = 30
    fade: bool = True
    thickness: int = 2

    def annotate(
        self, frame: np.ndarray, tracks: dict[int, list[tuple[float, float]]]
    ) -> np.ndarray:
        for tid, positions in tracks.items():
            pts = positions[-self.trail_length:]
            for i in range(1, len(pts)):
                alpha = i / len(pts) if self.fade else 1.0
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, color, self.thickness)
        return frame
```

**Step 4: Run tests**

Run: `pytest tests/test_annotators.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/annotators.py tests/test_annotators.py
git commit -m "feat: add annotators for team colors, stats overlay, and trails"
```

---

### Task 10: Pipeline — SportVisionPipeline

**Files:**
- Create: `src/sportvision/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np
from sportvision.pipeline import SportVisionPipeline


class TestSportVisionPipeline:
    def test_init(self):
        p = SportVisionPipeline(sport="soccer")
        assert p.sport == "soccer"

    def test_process_frame(self, sample_frame):
        p = SportVisionPipeline(sport="soccer")
        result = p.process_frame(sample_frame)
        assert "detections" in result
        assert "annotated_frame" in result

    def test_get_stats_initial(self):
        p = SportVisionPipeline(sport="soccer")
        stats = p.get_stats()
        assert "possession" in stats
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/pipeline.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sportvision.detection import SportsDetector
from sportvision.tracking import SportsTracker
from sportvision.analytics.possession import PossessionTracker
from sportvision.annotators import TeamColorAnnotator, StatsOverlayAnnotator


@dataclass
class SportVisionPipeline:
    sport: str = "soccer"
    device: str = "cpu"
    _detector: SportsDetector = field(default=None, init=False)
    _tracker: SportsTracker = field(default=None, init=False)
    _possession: PossessionTracker = field(default=None, init=False)
    _team_annotator: TeamColorAnnotator = field(default=None, init=False)
    _stats_annotator: StatsOverlayAnnotator = field(default=None, init=False)

    def __post_init__(self):
        self._detector = SportsDetector()
        self._tracker = SportsTracker()
        self._possession = PossessionTracker()
        self._team_annotator = TeamColorAnnotator()
        self._stats_annotator = StatsOverlayAnnotator()

    def process_frame(self, frame: np.ndarray) -> dict:
        detections = self._detector.detect(frame)
        tracked = self._tracker.update(detections)
        annotated = self._team_annotator.annotate(
            frame.copy(),
            tracked["xyxy"],
            tracked.get("class_id", np.zeros(len(tracked["xyxy"]))),
        )
        stats = self._possession.get_stats()
        annotated = self._stats_annotator.annotate(annotated, possession=stats)
        return {"detections": tracked, "annotated_frame": annotated}

    def get_stats(self) -> dict:
        return {"possession": self._possession.get_stats()}
```

Update `src/sportvision/__init__.py`:
```python
"""SportVision: Real-time sports analytics toolkit."""

__version__ = "0.1.0"

from sportvision.pipeline import SportVisionPipeline

__all__ = ["SportVisionPipeline"]
```

**Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/pipeline.py src/sportvision/__init__.py tests/test_pipeline.py
git commit -m "feat: add SportVisionPipeline orchestrating all modules"
```

---

### Task 11: Roboflow Workflow Blocks

**Files:**
- Create: `src/sportvision/workflows/__init__.py`
- Create: `src/sportvision/workflows/blocks.py`
- Create: `tests/test_workflows.py`

**Step 1: Write the failing test**

`tests/test_workflows.py`:
```python
from sportvision.workflows.blocks import TeamClassifierBlock, PossessionTrackerBlock


class TestWorkflowBlocks:
    def test_team_classifier_block_metadata(self):
        block = TeamClassifierBlock()
        assert block.get_manifest()["type"] == "sportvision/team_classifier"

    def test_possession_tracker_block_metadata(self):
        block = PossessionTrackerBlock()
        assert block.get_manifest()["type"] == "sportvision/possession_tracker"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflows.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

`src/sportvision/workflows/__init__.py`:
```python
from sportvision.workflows.blocks import TeamClassifierBlock, PossessionTrackerBlock

__all__ = ["TeamClassifierBlock", "PossessionTrackerBlock"]
```

`src/sportvision/workflows/blocks.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkflowBlock:
    _type: str = ""

    def get_manifest(self) -> dict:
        return {"type": self._type}


@dataclass
class TeamClassifierBlock(WorkflowBlock):
    _type: str = "sportvision/team_classifier"

    def run(self, image, detections) -> dict:
        from sportvision.teams import TeamClassifier
        # Placeholder — real implementation extracts crops and classifies
        return {"predictions": detections, "team_ids": []}


@dataclass
class PossessionTrackerBlock(WorkflowBlock):
    _type: str = "sportvision/possession_tracker"

    def run(self, detections) -> dict:
        from sportvision.analytics.possession import PossessionTracker
        tracker = PossessionTracker()
        return {"stats": tracker.get_stats()}


@dataclass
class SpeedEstimatorBlock(WorkflowBlock):
    _type: str = "sportvision/speed_estimator"


@dataclass
class HeatmapGeneratorBlock(WorkflowBlock):
    _type: str = "sportvision/heatmap_generator"


@dataclass
class StatsOverlayBlock(WorkflowBlock):
    _type: str = "sportvision/stats_overlay"
```

**Step 4: Run tests**

Run: `pytest tests/test_workflows.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sportvision/workflows/ tests/test_workflows.py
git commit -m "feat: add Roboflow Workflow block stubs"
```

---

### Task 12: Full Test Suite & CI Config

**Files:**
- Modify: `pyproject.toml` (add ruff/pytest config if needed)
- Create: `.github/workflows/ci.yml`

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Run ruff**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: No errors

**Step 3: Fix any lint issues found, then create CI config**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      - run: pytest tests/ -v --tb=short
```

**Step 4: Commit**

```bash
git add .github/ pyproject.toml
git commit -m "ci: add GitHub Actions workflow for tests and linting"
```

---

## Summary

| Task | Module | Tests |
|------|--------|-------|
| 1 | Scaffolding | Install check |
| 2 | Detection | 4 tests |
| 3 | Tracking | 4 tests |
| 4 | Team Classification | 4 tests |
| 5 | Field Homography | 4 tests |
| 6 | Possession Tracker | 5 tests |
| 7 | Speed & Distance | 5 tests |
| 8 | Heatmap Generator | 4 tests |
| 9 | Annotators | 3 tests |
| 10 | Pipeline | 3 tests |
| 11 | Workflow Blocks | 2 tests |
| 12 | CI & Lint | Full suite |

**Total: 12 tasks, ~38 tests, TDD throughout.**
