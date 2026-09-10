from unittest.mock import MagicMock

import numpy as np

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
        detector._model = MagicMock()
        detector._backend = "yolo"
        detector._detect_yolo = MagicMock(return_value=detector._empty_result())
        result = detector.detect(sample_frame)
        assert "xyxy" in result
        assert "class_id" in result
        assert "confidence" in result
        assert isinstance(result["xyxy"], np.ndarray)

    def test_detect_empty_frame(self):
        detector = SportsDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector._model = MagicMock()
        detector._backend = "yolo"
        detector._detect_yolo = MagicMock(return_value=detector._empty_result())
        result = detector.detect(frame)
        assert result["xyxy"].shape[1] == 4

    def test_filter_coco_ball_mapping(self):
        xyxy = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float
        )
        class_id = np.array([0, 32, 37])
        conf = np.array([0.9, 0.8, 0.7])
        result = SportsDetector._filter_coco(xyxy, class_id, conf)
        assert result["class_id"].tolist() == [0, 1]

    def test_filter_classes_honors_classes_param(self):
        detector = SportsDetector(classes=["ball"])
        result = {
            "xyxy": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            "class_id": np.array([0, 1]),
            "confidence": np.array([0.9, 0.8]),
        }
        filtered = detector._filter_classes(result)
        assert filtered["class_id"].tolist() == [1]


def test_native_yolo_resolution_and_independent_ball_threshold():
    detector = SportsDetector(
        model="yolov8n.pt", infer_size=1280, ball_confidence_threshold=0.15
    )
    detector._backend = "yolo"
    detector._model = MagicMock()
    boxes = MagicMock()
    boxes.__len__.return_value = 3
    boxes.xyxy.cpu().numpy.return_value = np.array([[100, 200, 300, 400]] * 3, float)
    boxes.cls.cpu().numpy.return_value = np.array([0, 32, 0])
    boxes.conf.cpu().numpy.return_value = np.array([0.9, 0.2, 0.2])
    detector._model.return_value = [MagicMock(boxes=boxes)]
    frame = np.zeros((1080, 1920, 3), np.uint8)
    result = detector.detect(frame)
    assert detector._model.call_args.args[0] is frame
    assert detector._model.call_args.kwargs["imgsz"] == 1280
    assert detector._model.call_args.kwargs["conf"] == 0.15
    assert result["class_id"].tolist() == [0, 1]
    np.testing.assert_array_equal(result["xyxy"], [[100, 200, 300, 400]] * 2)


def test_invalid_frame_fails_before_loading_model():
    import pytest

    detector = SportsDetector()
    detector._load_model = MagicMock()
    for frame in (
        None,
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((10, 10)),
        np.zeros((10, 10, 3)),
    ):
        with pytest.raises(ValueError, match="BGR"):
            detector.detect(frame)
    detector._load_model.assert_not_called()


def test_invalid_detection_thresholds():
    import pytest

    for value in (0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SportsDetector(ball_confidence_threshold=value)
