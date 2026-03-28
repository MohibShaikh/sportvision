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
    _backend: str | None = field(default=None, init=False, repr=False)

    @property
    def model_id(self) -> str:
        return self.model

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if "rfdetr" in self.model:
            try:
                import torch
                from rfdetr import RFDETRBase, RFDETRLarge

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
                self._backend = "rfdetr"
                return
            except ImportError:
                pass
        # YOLOv8 / ultralytics
        try:
            from ultralytics import YOLO

            model_path = self.model if self.model.endswith(".pt") else "yolov8n.pt"
            self._model = YOLO(model_path)
            self._backend = "yolo"
        except ImportError:
            self._model = None
            self._backend = None

    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        self._load_model()
        if self._model is None:
            return self._empty_result()
        try:
            if self._backend == "yolo":
                return self._detect_yolo(frame)
            return self._detect_rfdetr(frame)
        except Exception:
            return self._empty_result()

    def _detect_rfdetr(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        import supervision as sv

        detections: sv.Detections = self._model.predict(
            frame, threshold=self.confidence_threshold
        )
        if not isinstance(detections, sv.Detections):
            detections = sv.Detections.from_inference(detections)
        return self._filter_coco(
            detections.xyxy, detections.class_id, detections.confidence
        )

    def _detect_yolo(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return self._empty_result()
        xyxy = boxes.xyxy.cpu().numpy()
        class_id = boxes.cls.cpu().numpy().astype(int)
        confidence = boxes.conf.cpu().numpy().astype(np.float32)
        return self._filter_coco(xyxy, class_id, confidence)

    @staticmethod
    def _filter_coco(
        xyxy: np.ndarray,
        class_id: np.ndarray | None,
        confidence: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        coco_ids = class_id if class_id is not None else np.zeros(len(xyxy), dtype=int)
        mask = np.isin(coco_ids, list(COCO_SPORTS_IDS))
        sports_ids = np.array(
            [COCO_TO_SPORTS.get(int(c), 0) for c in coco_ids[mask]],
            dtype=int,
        )
        conf = confidence[mask] if confidence is not None else np.ones(mask.sum())
        return {
            "xyxy": xyxy[mask],
            "class_id": sports_ids,
            "confidence": conf,
        }

    @staticmethod
    def _empty_result() -> dict[str, np.ndarray]:
        return {
            "xyxy": np.empty((0, 4)),
            "class_id": np.empty(0, dtype=int),
            "confidence": np.empty(0),
        }
