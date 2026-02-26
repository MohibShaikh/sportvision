from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PossessionTracker:
    ball_proximity_thresh: float = 50.0
    n_teams: int = 2
    _counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def update(
        self,
        player_positions: np.ndarray,
        team_ids: np.ndarray,
        ball_position: np.ndarray,
    ) -> None:
        distances = np.linalg.norm(player_positions - ball_position, axis=1)
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] <= self.ball_proximity_thresh:
            team = int(team_ids[nearest_idx])
            self._counts[team] = self._counts.get(team, 0) + 1

    def get_stats(self) -> dict[int, float]:
        total = sum(self._counts.values())
        if total == 0:
            return {i: 1.0 / self.n_teams for i in range(self.n_teams)}
        return {i: self._counts.get(i, 0) / total for i in range(self.n_teams)}

    def reset(self) -> None:
        self._counts.clear()
