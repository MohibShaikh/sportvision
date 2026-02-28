from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(
    n: int = 6,
    class_ids: list[int] | None = None,
    with_tracker: bool = True,
    with_team: bool = False,
) -> sv.Detections:
    """Create synthetic sv.Detections for testing."""
    rng = np.random.default_rng(42)
    x1 = rng.integers(50, 900, n).astype(float)
    y1 = rng.integers(50, 400, n).astype(float)
    xyxy = np.column_stack([x1, y1, x1 + 80, y1 + 160])
    if class_ids is None:
        class_ids = [0] * n
    dets = sv.Detections(
        xyxy=xyxy,
        class_id=np.array(class_ids, dtype=int),
        confidence=rng.uniform(0.7, 1.0, n).astype(np.float32),
    )
    if with_tracker:
        dets.tracker_id = np.arange(n)
    if with_team:
        dets.data["team_id"] = np.array([i % 2 for i in range(n)])
    return dets


def _make_image(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_workflow_image(img: np.ndarray | None = None):
    """Return a mock WorkflowImageData whose numpy_image returns img."""
    if img is None:
        img = _make_image()
    mock = MagicMock()
    mock.numpy_image = img
    return mock


# ---------------------------------------------------------------------------
# Team Classifier
# ---------------------------------------------------------------------------


class TestTeamClassifierBlock:
    @pytest.fixture(autouse=True)
    def _import(self):
        from sportvision.workflows.team_classifier.v1 import (
            TeamClassifierBlockV1,
            TeamClassifierManifest,
        )

        self.BlockClass = TeamClassifierBlockV1
        self.Manifest = TeamClassifierManifest

    def test_manifest_type(self):
        name = self.Manifest.model_config["json_schema_extra"]["name"]
        assert name == "Team Classifier"

    def test_get_manifest(self):
        assert self.BlockClass.get_manifest() is self.Manifest

    def test_run_assigns_team_ids(self):
        block = self.BlockClass()
        dets = _make_detections(n=6)
        image = _make_workflow_image()
        result = block.run(image=image, detections=dets, n_teams=2)
        out_dets = result["detections"]
        assert "team_id" in out_dets.data
        assert len(out_dets.data["team_id"]) == 6
        assert set(out_dets.data["team_id"]).issubset({0, 1})

    def test_run_empty_detections(self):
        block = self.BlockClass()
        dets = sv.Detections.empty()
        image = _make_workflow_image()
        result = block.run(image=image, detections=dets, n_teams=2)
        assert len(result["detections"].data["team_id"]) == 0

    def test_stateful_reuses_model(self):
        block = self.BlockClass()
        image = _make_workflow_image()
        dets = _make_detections(n=6)
        block.run(image=image, detections=dets, n_teams=2)
        model_ref = block._model
        # Second call should reuse the same model
        dets2 = _make_detections(n=6)
        block.run(image=image, detections=dets2, n_teams=2)
        assert block._model is model_ref

    def test_refit_every_triggers_refit(self):
        block = self.BlockClass()
        image = _make_workflow_image()
        # Run 3 frames with refit_every=2; model should change on frame 2
        dets = _make_detections(n=6)
        block.run(image=image, detections=dets, n_teams=2, refit_every=2)
        model_after_1 = block._model
        dets2 = _make_detections(n=6)
        block.run(image=image, detections=dets2, n_teams=2, refit_every=2)
        model_after_2 = block._model
        # Frame 2 triggers refit (frame_count=2, 2%2==0)
        assert model_after_2 is not model_after_1

    def test_refit_every_zero_no_refit(self):
        block = self.BlockClass()
        image = _make_workflow_image()
        dets = _make_detections(n=6)
        block.run(image=image, detections=dets, n_teams=2, refit_every=0)
        model_ref = block._model
        for _ in range(5):
            dets2 = _make_detections(n=6)
            block.run(image=image, detections=dets2, n_teams=2, refit_every=0)
        assert block._model is model_ref


# ---------------------------------------------------------------------------
# Possession Tracker
# ---------------------------------------------------------------------------


class TestPossessionTrackerBlock:
    @pytest.fixture(autouse=True)
    def _import(self):
        from sportvision.workflows.possession_tracker.v1 import (
            PossessionTrackerBlockV1,
            PossessionTrackerManifest,
        )

        self.BlockClass = PossessionTrackerBlockV1
        self.Manifest = PossessionTrackerManifest

    def test_manifest_type(self):
        name = self.Manifest.model_config["json_schema_extra"]["name"]
        assert name == "Possession Tracker"

    def test_run_with_ball_and_players(self):
        block = self.BlockClass()
        # 4 players (class 0) + 1 ball (class 1)
        class_ids = [0, 0, 0, 0, 1]
        dets = _make_detections(n=5, class_ids=class_ids, with_team=True)
        result = block.run(
            detections=dets,
            ball_class_id=1,
            ball_proximity_threshold=9999.0,
        )
        assert "possession_stats" in result
        assert "possessing_team" in result
        assert result["possessing_team"] in (0, 1)

    def test_run_empty_detections(self):
        block = self.BlockClass()
        dets = sv.Detections.empty()
        result = block.run(detections=dets)
        assert result["possessing_team"] == -1

    def test_run_no_ball(self):
        block = self.BlockClass()
        dets = _make_detections(n=4, class_ids=[0, 0, 0, 0])
        result = block.run(detections=dets)
        assert result["possessing_team"] == -1

    def test_possession_warns_no_team_id(self):
        block = self.BlockClass()
        # Players + ball, but no team_id assigned
        class_ids = [0, 0, 0, 0, 1]
        dets = _make_detections(n=5, class_ids=class_ids, with_team=False)
        result = block.run(
            detections=dets,
            ball_class_id=1,
            ball_proximity_threshold=9999.0,
        )
        assert result["warning"] != ""
        assert "team_id" in result["warning"]

    def test_accumulates_over_frames(self):
        block = self.BlockClass()
        class_ids = [0, 0, 1]
        for _ in range(10):
            dets = _make_detections(n=3, class_ids=class_ids, with_team=True)
            result = block.run(
                detections=dets,
                ball_proximity_threshold=9999.0,
            )
        total = sum(result["possession_stats"].values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Distance Calculator
# ---------------------------------------------------------------------------


class TestDistanceCalculatorBlock:
    @pytest.fixture(autouse=True)
    def _import(self):
        from sportvision.workflows.distance_calculator.v1 import (
            DistanceCalculatorBlockV1,
            DistanceCalculatorManifest,
        )

        self.BlockClass = DistanceCalculatorBlockV1
        self.Manifest = DistanceCalculatorManifest

    def test_manifest_type(self):
        name = self.Manifest.model_config["json_schema_extra"]["name"]
        assert name == "Distance Calculator"

    def test_run_computes_distance(self):
        block = self.BlockClass()
        # Frame 1
        dets1 = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            class_id=np.array([0]),
        )
        dets1.tracker_id = np.array([0])
        result1 = block.run(detections=dets1)
        assert result1["detections"].data["distance"][0] == 0.0

        # Frame 2: moved 100px to the right
        dets2 = sv.Detections(
            xyxy=np.array([[100, 0, 110, 10]], dtype=float),
            class_id=np.array([0]),
        )
        dets2.tracker_id = np.array([0])
        result2 = block.run(detections=dets2)
        assert result2["detections"].data["distance"][0] == pytest.approx(100.0)

    def test_run_empty(self):
        block = self.BlockClass()
        dets = sv.Detections.empty()
        result = block.run(detections=dets)
        assert len(result["detections"].data["distance"]) == 0

    def test_distance_with_homography(self):
        block = self.BlockClass()
        # Identity-like homography scaled by 0.1 (pixels -> ~meters)
        h_mat = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]
        dets1 = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            class_id=np.array([0]),
        )
        dets1.tracker_id = np.array([0])
        block.run(detections=dets1, homography_matrix=h_mat)

        # Move 100px right -> 10 field-units with 0.1 scale
        dets2 = sv.Detections(
            xyxy=np.array([[100, 0, 110, 10]], dtype=float),
            class_id=np.array([0]),
        )
        dets2.tracker_id = np.array([0])
        result = block.run(detections=dets2, homography_matrix=h_mat)
        assert result["detections"].data["distance"][0] == pytest.approx(10.0)

    def test_no_tracker_id(self):
        block = self.BlockClass()
        dets = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            class_id=np.array([0]),
        )
        result = block.run(detections=dets)
        assert result["detections"].data["distance"][0] == 0.0


# ---------------------------------------------------------------------------
# Sports Detection Filter
# ---------------------------------------------------------------------------


class TestSportsDetectionFilterBlock:
    @pytest.fixture(autouse=True)
    def _import(self):
        from sportvision.workflows.sports_detection_filter.v1 import (
            SportsDetectionFilterBlockV1,
            SportsDetectionFilterManifest,
        )

        self.BlockClass = SportsDetectionFilterBlockV1
        self.Manifest = SportsDetectionFilterManifest

    def test_manifest_type(self):
        name = self.Manifest.model_config["json_schema_extra"]["name"]
        assert name == "Sports Detection Filter"

    def test_filters_and_remaps(self):
        block = self.BlockClass()
        # person=0, car=2, sports_ball=32
        dets = sv.Detections(
            xyxy=np.array(
                [
                    [0, 0, 10, 10],
                    [20, 20, 30, 30],
                    [40, 40, 50, 50],
                ],
                dtype=float,
            ),
            class_id=np.array([0, 2, 32]),
            confidence=np.array([0.9, 0.8, 0.7]),
        )
        result = block.run(detections=dets)
        out = result["detections"]
        assert len(out) == 2
        assert list(out.class_id) == [0, 1]  # person->0, ball->1

    def test_custom_mapping(self):
        block = self.BlockClass()
        dets = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=float),
            class_id=np.array([5]),
            confidence=np.array([0.9]),
        )
        result = block.run(detections=dets, class_mapping={5: 99})
        assert result["detections"].class_id[0] == 99

    def test_empty_detections(self):
        block = self.BlockClass()
        dets = sv.Detections.empty()
        result = block.run(detections=dets)
        assert len(result["detections"]) == 0


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_load_blocks_returns_four(self):
        from sportvision.workflows import load_blocks

        blocks = load_blocks()
        assert len(blocks) == 4

    def test_load_blocks_types(self):
        from sportvision.workflows import load_blocks
        from sportvision.workflows.distance_calculator.v1 import (
            DistanceCalculatorBlockV1,
        )
        from sportvision.workflows.possession_tracker.v1 import (
            PossessionTrackerBlockV1,
        )
        from sportvision.workflows.sports_detection_filter.v1 import (
            SportsDetectionFilterBlockV1,
        )
        from sportvision.workflows.team_classifier.v1 import TeamClassifierBlockV1

        blocks = load_blocks()
        assert TeamClassifierBlockV1 in blocks
        assert PossessionTrackerBlockV1 in blocks
        assert DistanceCalculatorBlockV1 in blocks
        assert SportsDetectionFilterBlockV1 in blocks
