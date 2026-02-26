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
        except (ImportError, AttributeError):
            self._model = None

    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        self._load_model()
        if self._model is None:
            return self._empty_result()
        try:
            import supervision as sv

            detections: sv.Detections = self._model.predict(
                frame, threshold=self.confidence_threshold
            )
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
