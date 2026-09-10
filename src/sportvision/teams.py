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

    _references: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None or self._references is not None

    def set_team_samples(self, samples: list[np.ndarray]) -> None:
        """Set one representative jersey crop per team, in team-ID order."""
        if len(samples) != self.n_teams:
            raise ValueError("Provide one jersey sample per team")
        self._references = np.array([self._jersey_color(c) for c in samples])

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        jersey = crop[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v])
        feat = feat / (feat.sum() + 1e-8)
        return feat

    def fit(self, crops: list[np.ndarray]) -> TeamClassifier:
        features = np.array([self._extract_features(c) for c in crops])
        self._model = KMeans(n_clusters=self.n_teams, random_state=0, n_init=2)
        self._model.fit(features)
        return self

    def predict(self, crops: list[np.ndarray]) -> list[int]:
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        if not crops:
            return []
        features = np.array([self._extract_features(c) for c in crops])
        if self._references is not None:
            colors = np.array([self._jersey_color(c) for c in crops])
            distances = np.linalg.norm(
                colors[:, None, :] - self._references[None, :, :], axis=2
            )
            return distances.argmin(axis=1).tolist()
        return self._model.predict(features).tolist()

    @staticmethod
    def _jersey_color(crop: np.ndarray) -> np.ndarray:
        """Compare chosen colors in Lab, allowing continuous shade differences."""
        h, w = crop.shape[:2]
        jersey = crop[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
        if jersey.size == 0:
            raise ValueError("Jersey sample is too small")
        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        color = np.median(lab, axis=0)
        # Brighter jersey pixels reduce the effect of folds and shadows.
        color[0] = np.percentile(lab[:, 0], 65)
        return color
