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
