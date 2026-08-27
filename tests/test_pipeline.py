from unittest.mock import MagicMock

import numpy as np

from sportvision.pipeline import SportVisionPipeline


class TestSportVisionPipeline:
    def test_init(self):
        p = SportVisionPipeline()
        assert p.device == "auto"
        assert p.n_teams == 2

    def test_device_wired_to_detector(self):
        p = SportVisionPipeline(device="cuda", infer_size=960)
        assert p._detector.device == "cuda"
        assert p._detector.infer_size == 960

    def test_process_frame(self, sample_frame):
        p = SportVisionPipeline()
        result = p.process_frame(sample_frame)
        assert "detections" in result
        assert "annotated_frame" in result

    def test_get_stats_initial(self):
        p = SportVisionPipeline()
        stats = p.get_stats()
        assert "possession" in stats
        assert stats["possession"] == {0: 0.5, 1: 0.5}

    def test_process_frame_runs_possession(self):
        from sportvision.teams import TeamClassifier

        red = [np.full((80, 60, 3), (50, 50, 210), np.uint8) for _ in range(3)]
        blue = [np.full((80, 60, 3), (210, 60, 40), np.uint8) for _ in range(3)]
        p = SportVisionPipeline()
        p._team_classifier = TeamClassifier(n_teams=2)
        p._team_classifier.fit(red + blue)

        xyxy = np.array(
            [
                [100, 100, 160, 180],  # red player (center 130,140)
                [200, 100, 260, 180],  # red player
                [600, 100, 660, 180],  # blue player
                [700, 100, 760, 180],  # blue player
                [140, 140, 150, 150],  # ball (center 145,145)
            ],
            dtype=float,
        )
        dets = {
            "xyxy": xyxy,
            "class_id": np.array([0, 0, 0, 0, 1]),
            "confidence": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        }
        p._detector.detect = MagicMock(return_value=dets)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for box, color in zip(xyxy, [(50, 50, 210)] * 4 + [(255, 255, 255)]):
            x1, y1, x2, y2 = box.astype(int)
            frame[y1:y2, x1:x2] = color

        p.process_frame(frame)
        p.process_frame(frame)

        stats = p.get_stats()["possession"]
        assert stats != {0: 0.5, 1: 0.5}
        assert abs(sum(stats.values()) - 1.0) < 1e-6

    def test_detect_every_stride_reuses_detections(self):
        p = SportVisionPipeline(detect_every=2)
        dets = {
            "xyxy": np.empty((0, 4)),
            "class_id": np.empty(0, dtype=int),
            "confidence": np.empty(0),
        }
        p._detector.detect = MagicMock(return_value=dets)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        p.process_frame(frame)
        p.process_frame(frame)

        assert p._detector.detect.call_count == 1
