"""Regression coverage for misleading analytics and detector failures."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from sportvision.analytics.possession import PossessionTracker
from sportvision.detection import SportsDetector
from sportvision.pipeline import SportVisionPipeline
from sportvision.tracking import SportsTracker
from sportvision.workflows.possession_tracker.v1 import PossessionTrackerBlockV1
from sportvision.workflows.team_classifier.v1 import TeamClassifierBlockV1


def test_rfdetr_uses_rgb_and_sparse_coco_ids():
    detector = SportsDetector(infer_size=None)
    detector._model = MagicMock()
    detector._backend = "rfdetr"
    detector._model.predict.return_value = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * 3, dtype=float),
        class_id=np.array([1, 37, 32]),
        confidence=np.array([0.9, 0.8, 0.7]),
    )
    result = detector.detect(np.full((20, 20, 3), (10, 20, 30), np.uint8))
    assert result["class_id"].tolist() == [0, 1]
    np.testing.assert_array_equal(
        detector._model.predict.call_args.args[0][0, 0], [30, 20, 10]
    )


def test_detection_failure_is_not_an_empty_success():
    detector = SportsDetector()
    detector._model = MagicMock()
    detector._backend = "yolo"
    detector._model.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        detector.detect(np.zeros((20, 20, 3), np.uint8))


def test_missing_rfdetr_does_not_silently_switch_models():
    with patch.dict("sys.modules", {"rfdetr": None}):
        with pytest.raises(RuntimeError, match="RF-DETR is unavailable"):
            SportsDetector()._load_model()


def test_tracker_failure_does_not_fabricate_ids():
    tracker = SportsTracker()
    tracker._tracker = MagicMock()
    tracker._tracker.update.side_effect = RuntimeError("broken tracking")
    with pytest.raises(RuntimeError, match="broken tracking"):
        tracker.update(SportsDetector._empty_result())


def test_unknown_team_does_not_pollute_possession():
    tracker = PossessionTracker()
    tracker.update(np.array([[1, 1]]), np.array([-1]), np.array([1, 1]))
    assert tracker.get_stats() == {}


def test_pipeline_handles_players_disappearing_and_skipped_frames():
    # fit_min_samples=2 restores the old fit-immediately behaviour, which this
    # test needs so that teams are assigned on the very first frame.
    pipeline = SportVisionPipeline(detect_every=2, fit_min_samples=2)
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :100] = (0, 0, 255)
    frame[:, 100:] = (255, 0, 0)
    tracked = {
        "xyxy": np.array([[0, 0, 80, 80], [110, 0, 190, 80], [35, 35, 45, 45.0]]),
        "class_id": np.array([0, 0, 1]),
        "confidence": np.ones(3),
        "tracker_id": np.arange(3),
    }
    empty = {**SportsDetector._empty_result(), "tracker_id": np.array([], int)}
    pipeline._detector.detect = MagicMock(side_effect=[tracked, empty])
    pipeline._tracker.update = lambda detections: detections
    first = pipeline.process_frame(frame)
    assert set(first["detections"]["team_id"]) == {-1, 0, 1}
    assert sum(pipeline._possession._counts.values()) == 1
    assert not pipeline.process_frame(frame)["fresh_detections"]
    assert sum(pipeline._possession._counts.values()) == 1
    assert pipeline.process_frame(frame)["fresh_detections"]


def test_referee_is_not_a_ball():
    p = SportVisionPipeline()
    p._team_classifier._model = MagicMock()
    p._update_possession(
        {
            "xyxy": np.array([[0, 0, 20, 20], [5, 5, 15, 15]]),
            "class_id": np.array([0, 2]),
        },
        np.array([0, -1]),
        np.array([True, False]),
    )
    assert p.get_stats()["possession"] == {}


@pytest.mark.parametrize("kwargs", [{"detect_every": 0}, {"n_teams": 0}])
def test_invalid_pipeline_configuration(kwargs):
    with pytest.raises(ValueError):
        SportVisionPipeline(**kwargs)


def test_workflow_does_not_classify_ball_or_crash_on_one_player():
    block = TeamClassifierBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 20, 20], [30, 30, 40, 40]], float),
        class_id=np.array([0, 1]),
    )
    result = block.run(
        MagicMock(numpy_image=np.zeros((50, 50, 3), np.uint8)), detections
    )
    assert result["detections"].data["team_id"].tolist() == [-1, -1]


def test_workflow_current_possession_is_not_historical_leader():
    block = PossessionTrackerBlockV1()

    def observation(team):
        dets = sv.Detections(
            xyxy=np.array([[0, 0, 20, 20], [5, 5, 10, 10]], float),
            class_id=np.array([0, 1]),
        )
        dets.data["team_id"] = np.array([team, -1])
        return block.run(dets)

    observation(0)
    observation(0)
    assert observation(1)["possessing_team"] == 1
    absent = block.run(sv.Detections.empty())
    assert absent["possessing_team"] == -1
    assert absent["possession_stats"] == {0: 2 / 3, 1: 1 / 3}


def test_distance_requires_tracking_instead_of_fabricating_zero():
    from sportvision.workflows.distance_calculator.v1 import DistanceCalculatorBlockV1

    dets = sv.Detections(xyxy=np.array([[0, 0, 20, 20]], float))
    with pytest.raises(ValueError, match="persistent tracker_id"):
        DistanceCalculatorBlockV1().run(dets)


def _two_team_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :100] = (0, 0, 255)
    frame[:, 100:] = (255, 0, 0)
    tracked = {
        "xyxy": np.array([[0, 0, 80, 80], [110, 0, 190, 80]]),
        "class_id": np.array([0, 0]),
        "confidence": np.ones(2),
        "tracker_id": np.arange(2),
    }
    return frame, tracked


def test_teams_are_unassigned_until_the_sample_is_broad_enough():
    """Fitting on the first frame that holds two players scored 56% against
    explicit jersey colours on the sample clip. Waiting for a wider sample
    scored 92%. Unassigned is the honest answer until then."""
    frame, tracked = _two_team_frame()
    pipeline = SportVisionPipeline(fit_min_samples=8, fit_max_frames=1000)
    pipeline._detector.detect = MagicMock(return_value=tracked)
    pipeline._tracker.update = lambda detections: detections

    for _ in range(3):
        result = pipeline.process_frame(frame)
        assert set(result["detections"]["team_id"]) == {-1}
    assert not pipeline._team_classifier.is_fitted

    # The fourth frame brings the buffer to eight crops.
    result = pipeline.process_frame(frame)
    assert pipeline._team_classifier.is_fitted
    assert set(result["detections"]["team_id"]) == {0, 1}


def test_the_frame_cap_fits_even_when_players_are_scarce():
    frame, tracked = _two_team_frame()
    pipeline = SportVisionPipeline(fit_min_samples=1000, fit_max_frames=3)
    pipeline._detector.detect = MagicMock(return_value=tracked)
    pipeline._tracker.update = lambda detections: detections
    for _ in range(3):
        pipeline.process_frame(frame)
    assert not pipeline._team_classifier.is_fitted
    pipeline.process_frame(frame)
    assert pipeline._team_classifier.is_fitted


def test_explicit_jersey_colors_skip_the_warm_up_entirely():
    frame, tracked = _two_team_frame()
    pipeline = SportVisionPipeline()
    pipeline.set_team_colors([(0, 0, 255), (255, 0, 0)])
    pipeline._detector.detect = MagicMock(return_value=tracked)
    pipeline._tracker.update = lambda detections: detections
    result = pipeline.process_frame(frame)
    assert set(result["detections"]["team_id"]) == {0, 1}


def test_fit_settings_are_validated():
    with pytest.raises(ValueError, match="fit_min_samples"):
        SportVisionPipeline(fit_min_samples=1, n_teams=2)
    with pytest.raises(ValueError, match="fit_max_frames"):
        SportVisionPipeline(fit_max_frames=0)
