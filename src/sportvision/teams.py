from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClassifier:
    n_teams: int = 2
    method: str = "kmeans"
    _model: KMeans | None = field(default=None, init=False, repr=False)

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        jersey = hsv[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
        hist_h = cv2.calcHist([jersey], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([jersey], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([jersey], [2], None, [8], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v])
        feat = feat / (feat.sum() + 1e-8)
        return feat

    def fit(self, crops: list[np.ndarray]) -> TeamClassifier:
        features = np.array([self._extract_features(c) for c in crops])
        self._model = KMeans(n_clusters=self.n_teams, random_state=0, n_init=10)
        self._model.fit(features)
        return self

    def predict(self, crops: list[np.ndarray]) -> list[int]:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        features = np.array([self._extract_features(c) for c in crops])
        return self._model.predict(features).tolist()
