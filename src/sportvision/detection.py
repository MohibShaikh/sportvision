from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CLASS_NAMES = ["player", "ball", "referee", "goalkeeper"]

# COCO class IDs that map to sports classes
# person(0) -> player(0), sports ball(37) -> ball(1)
COCO_TO_SPORTS = {0: 0, 37: 1}
COCO_SPORTS_IDS = set(COCO_TO_SPORTS.keys())


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
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    if device == "cuda":
                        torch.zeros(1, device="cuda")
                except Exception:
                    device = "cpu"

                self._model = (
                    RFDETRLarge(device=device)
                    if "large" in self.model
                    else RFDETRBase(device=device)
                )
            else:
                self._model = None
        except ImportError:
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

            # Filter to sports-relevant COCO classes and remap IDs
            coco_ids = (
                detections.class_id
                if detections.class_id is not None
                else np.zeros(len(detections.xyxy), dtype=int)
            )
            mask = np.isin(coco_ids, list(COCO_SPORTS_IDS))
            sports_ids = np.array(
                [COCO_TO_SPORTS.get(int(c), 0) for c in coco_ids[mask]],
                dtype=int,
            )
            conf = (
                detections.confidence[mask]
                if detections.confidence is not None
                else np.ones(mask.sum())
            )
            return {
                "xyxy": detections.xyxy[mask],
                "class_id": sports_ids,
                "confidence": conf,
            }
        except Exception:
            return self._empty_result()

    @staticmethod
    def _empty_result() -> dict[str, np.ndarray]:
        return {
            "xyxy": np.empty((0, 4)),
            "class_id": np.empty(0, dtype=int),
            "confidence": np.empty(0),
        }
