from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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
    device: str = "auto"
    infer_size: int | None = 640
    _model: Any = field(default=None, init=False, repr=False)
    _backend: str | None = field(default=None, init=False, repr=False)
    _warned: bool = field(default=False, init=False, repr=False)

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
        self._load_model()
        if self._model is None:
            if not self._warned:
                logger.warning(
                    "No detection backend available (install 'sportvision[inference]' "
                    "or ultralytics); returning empty detections"
                )
                self._warned = True
            return self._empty_result()
        scaled_frame, scale = self._prepare(frame)
        try:
            if self._backend == "yolo":
                result = self._detect_yolo(scaled_frame)
            else:
                result = self._detect_rfdetr(scaled_frame)
        except Exception:
            if not self._warned:
                logger.exception("Detection failed; returning empty detections")
                self._warned = True
            return self._empty_result()
        if scale != 1.0:
            result["xyxy"] = result["xyxy"] / scale
        return self._filter_classes(result)

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
        kwargs: dict[str, Any] = {"conf": self.confidence_threshold, "verbose": False}
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
        if keep == set(range(len(CLASS_NAMES))):
            return result
        mask = np.isin(result["class_id"], list(keep))
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
