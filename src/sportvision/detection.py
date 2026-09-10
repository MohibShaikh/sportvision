from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

CLASS_NAMES = ["player", "ball", "referee", "goalkeeper"]

# COCO class IDs that map to sports classes
# person(0) -> player(0), sports ball(32) -> ball(1)
COCO_TO_SPORTS = {0: 0, 32: 1}
COCO_SPORTS_IDS = set(COCO_TO_SPORTS.keys())


@dataclass
class SportsDetector:
    model: str = "rfdetr-base"
    classes: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    confidence_threshold: float = 0.25
    ball_confidence_threshold: float | None = None
    device: str = "auto"
    infer_size: int | None = 640
    _model: Any = field(default=None, init=False, repr=False)
    _backend: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for value in (self.confidence_threshold, self.ball_confidence_threshold):
            if value is not None and (not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError("Confidence thresholds must be in (0, 1]")
        if self.infer_size is not None and (
            not isinstance(self.infer_size, int) or self.infer_size < 1
        ):
            raise ValueError("infer_size must be positive or None")

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

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                self._model = (
                    RFDETRLarge(device=device)
                    if "large" in self.model
                    else RFDETRBase(device=device)
                )
                self._backend = "rfdetr"
                return
            except ImportError as exc:
                raise RuntimeError(
                    "RF-DETR is unavailable; install 'sportvision[inference]' "
                    "or explicitly select a YOLO model with ultralytics installed"
                ) from exc
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO requires ultralytics; pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model)
        self._backend = "yolo"

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if self.infer_size is None:
            return frame, 1.0
        scale = float(self.infer_size) / max(h, w)
        if scale >= 1.0:
            return frame, 1.0
        resized = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            raise ValueError("Expected a nonempty uint8 BGR image with three channels")
        self._load_model()
        if self._backend == "yolo":
            # Ultralytics handles letterboxing and returns source coordinates.
            result = self._detect_yolo(frame)
        else:
            scaled_frame, scale = self._prepare(frame)
            result = self._detect_rfdetr(scaled_frame)
            if scale != 1.0:
                result["xyxy"] = result["xyxy"] / scale
        return self._filter_classes(result)

    def _detect_rfdetr(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        import supervision as sv

        detections: sv.Detections = self._model.predict(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self._inference_threshold
        )
        if not isinstance(detections, sv.Detections):
            detections = sv.Detections.from_inference(detections)
        return self._filter_coco(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
            class_mapping={1: 0, 37: 1},
        )

    @property
    def _inference_threshold(self) -> float:
        return min(
            self.confidence_threshold,
            self.ball_confidence_threshold or self.confidence_threshold,
        )

    def _detect_yolo(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        kwargs: dict[str, Any] = {
            "conf": self._inference_threshold,
            "verbose": False,
            "classes": [0, 32],
        }
        if self.infer_size is not None:
            kwargs["imgsz"] = self.infer_size
        if self.device != "auto":
            kwargs["device"] = self.device
        results = self._model(frame, **kwargs)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return self._empty_result()
        xyxy = boxes.xyxy.cpu().numpy()
        class_id = boxes.cls.cpu().numpy().astype(int)
        confidence = boxes.conf.cpu().numpy().astype(np.float32)
        return self._filter_coco(xyxy, class_id, confidence)

    def _filter_classes(self, result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        keep = {CLASS_NAMES.index(c) for c in self.classes if c in CLASS_NAMES}
        thresholds = np.where(
            result["class_id"] == 1,
            self.ball_confidence_threshold or self.confidence_threshold,
            self.confidence_threshold,
        )
        mask = np.isin(result["class_id"], list(keep)) & (
            result["confidence"] >= thresholds
        )
        return {
            "xyxy": result["xyxy"][mask],
            "class_id": result["class_id"][mask],
            "confidence": result["confidence"][mask],
        }

    @staticmethod
    def _filter_coco(
        xyxy: np.ndarray,
        class_id: np.ndarray | None,
        confidence: np.ndarray | None,
        class_mapping: dict[int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        mapping = COCO_TO_SPORTS if class_mapping is None else class_mapping
        coco_ids = class_id if class_id is not None else np.zeros(len(xyxy), dtype=int)
        mask = np.isin(coco_ids, list(mapping))
        sports_ids = np.array(
            [mapping[int(c)] for c in coco_ids[mask]],
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
