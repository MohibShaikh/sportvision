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
