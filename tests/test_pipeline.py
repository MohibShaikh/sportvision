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
