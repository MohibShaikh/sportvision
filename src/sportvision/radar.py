"""Project tracked players onto a top-down court schematic.

Calibration needs the four court corners clicked on one frame. Everything else
reuses :class:`~sportvision.homography.FieldHomography`, so the projection is
the same maths the analytics modules already use.

A fixed homography assumes a fixed camera. If the camera pans or zooms, the
projection drifts and the court must be recalibrated on a later frame.

A homography maps one plane, and the plane here is the floor. Only things
touching the floor can be placed. Players' feet qualify; a ball being held or
in flight does not, and projecting one at chest height lands it roughly 16 m
from the player holding it.

So the ball gets no position of its own. What the model does establish is
which player is nearest it, and that player is ringed in the ball color. The
radar says "this player has the ball", which is the claim the evidence
supports, rather than "the ball is at this point", which it does not.
"""

from __future__ import annotations

import cv2
import numpy as np

from sportvision.homography import FieldHomography

COURTS = {
    "basketball": FieldHomography.basketball,
    "soccer": FieldHomography.soccer,
    "tennis": FieldHomography.tennis,
}

# BGR, matching the slate palette in theme.py.
_BG = (39, 32, 29)
_LINE = (107, 100, 94)
_LINE_FAINT = (73, 65, 61)
_TEAM = {0: (210, 162, 103), 1: (94, 109, 214)}
_OTHER = (107, 100, 94)
_BALL = (104, 181, 217)

CORNER_ORDER = (
    "far left",
    "far right",
    "near right",
    "near left",
)


def _court(sport: str) -> FieldHomography:
    try:
        return COURTS[sport]()
    except KeyError:
        raise ValueError(
            f"Unknown sport {sport!r}. Choose one of {', '.join(sorted(COURTS))}."
        ) from None


def court_size(sport: str) -> tuple[float, float]:
    field = _court(sport)
    return field.field_length, field.field_width


def court_corners(sport: str) -> np.ndarray:
    """The four court corners in meters, in ``CORNER_ORDER``."""
    length, width = court_size(sport)
    return np.array(
        [[0, 0], [length, 0], [length, width], [0, width]], dtype=np.float32
    )


def calibrate(sport: str, image_points) -> FieldHomography:
    """Map four clicked image points onto the court, in ``CORNER_ORDER``.

    Raises ValueError if the points cannot describe a court. A degenerate set,
    four points in a line for instance, otherwise yields a homography that
    projects every player to a plausible looking but wrong position, which is
    worse than refusing.
    """
    try:
        points = np.asarray(image_points, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("Corner points must be numbers") from None
    if points.shape != (4, 2):
        raise ValueError("Need exactly four corner points")
    if not np.isfinite(points).all():
        raise ValueError("Corner points must be finite numbers")
    if len(np.unique(points, axis=0)) < 4:
        raise ValueError("The four corners must be distinct points")
    # Shoelace area. Collinear points enclose nothing.
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    span = max(np.ptp(x), np.ptp(y))
    if area < max(1.0, 0.01 * span * span):
        raise ValueError(
            "Those four points are nearly in a line. Click the corners of the "
            "playing area so they enclose it."
        )
    field = _court(sport)
    corners = court_corners(sport)
    field.set_keypoints(points, corners)
    # Round-trip the inputs. cv2 returns None for an unsolvable system and a
    # near-singular matrix for a marginal one; both fail this.
    try:
        check = field.transform(points)
    except RuntimeError:
        raise ValueError(
            "Those four points do not define a court. Click the corners of "
            "the playing area."
        ) from None
    if not np.isfinite(check).all() or np.abs(check - corners).max() > 0.5:
        raise ValueError(
            "Those four points do not map onto a court cleanly. Click the "
            "corners of the playing area, in order."
        )
    return field


def feet(xyxy) -> np.ndarray:
    """Bottom-center of each box, which is where a player meets the ground."""
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return np.empty((0, 2), dtype=np.float32)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def project(field: FieldHomography, xyxy) -> np.ndarray:
    """Project boxes onto court coordinates in meters."""
    points = feet(xyxy)
    if not len(points):
        return np.empty((0, 2), dtype=np.float32)
    return field.transform(points)


def on_court(positions, sport: str, tolerance: float = 3.0) -> np.ndarray:
    """Mask of positions inside the court, within a tolerance in meters.

    Benches, officials and spectators project well outside the lines. Drawing
    them would imply the model knows they are not players, which it does not.
    """
    points = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return np.zeros(0, dtype=bool)
    length, width = court_size(sport)
    return (
        (points[:, 0] >= -tolerance)
        & (points[:, 0] <= length + tolerance)
        & (points[:, 1] >= -tolerance)
        & (points[:, 1] <= width + tolerance)
    )


def _markings(canvas, sport, to_px, length, width):
    cv2.rectangle(canvas, to_px(0, 0), to_px(length, width), _LINE, 2)
    cv2.line(canvas, to_px(length / 2, 0), to_px(length / 2, width), _LINE_FAINT, 1)
    radius = {"soccer": 9.15, "basketball": 1.8, "tennis": 0.0}[sport]
    if radius:
        edge = to_px(length / 2 + radius, width / 2)
        center = to_px(length / 2, width / 2)
        cv2.circle(canvas, center, abs(edge[0] - center[0]), _LINE_FAINT, 1)
    if sport == "soccer":
        for x0, x1 in ((0, 16.5), (length - 16.5, length)):
            cv2.rectangle(
                canvas,
                to_px(x0, width / 2 - 20.16),
                to_px(x1, width / 2 + 20.16),
                _LINE_FAINT,
                1,
            )


def draw(
    sport: str,
    positions=(),
    teams=(),
    possession=(),
    width_px: int = 720,
    margin: int = 26,
) -> np.ndarray:
    """Draw the court and put a dot at each projected position.

    ``positions`` are court meters and ``teams`` are team ids aligned to them.
    ``possession`` is an optional boolean per position; a true entry rings that
    player in the ball color. Returns a BGR image.
    """
    length, width = court_size(sport)
    margin = max(0, int(margin))
    usable = int(width_px) - 2 * margin
    if usable < 10:
        raise ValueError("width_px must leave at least 10 pixels inside the margins")
    scale = usable / length
    height_px = int(round(width * scale)) + 2 * margin
    canvas = np.full((height_px, width_px, 3), _BG, dtype=np.uint8)

    def to_px(x, y):
        return (int(round(margin + x * scale)), int(round(margin + y * scale)))

    _markings(canvas, sport, to_px, length, width)
    points = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    ids = list(teams)
    keep = on_court(points, sport)
    # Sized to the canvas rather than to meters. A real 0.5m player on a
    # 105m pitch comes out three pixels across, which nobody can read.
    dot = max(4, int(round(width_px / 110)))
    holding = list(possession)
    holding += [False] * (len(points) - len(holding))
    for (x, y), team, has_ball in zip(
        points[keep],
        [t for t, k in zip(ids, keep) if k],
        [h for h, k in zip(holding, keep) if k],
    ):
        center = to_px(x, y)
        cv2.circle(canvas, center, dot, _TEAM.get(team, _OTHER), -1)
        if has_ball:
            cv2.circle(canvas, center, dot + 4, _BALL, 2)
    return canvas


def draw_frame(sport: str, field: FieldHomography, row: dict, **kwargs) -> np.ndarray:
    """Radar for one reviewed frame record.

    The player holding the ball is ringed. The ball itself gets no dot, see
    the module docstring.
    """
    boxes = np.asarray(row["xyxy"], dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(row["class_id"])
    teams = np.asarray(row["team_id"])
    tracks = np.asarray(row.get("tracker_id", [-1] * len(boxes)))
    players = classes == 0
    positions = project(field, boxes[players]) if players.any() else ()
    evidence = row.get("evidence") or {}
    holder = evidence.get("tracker_id") if evidence.get("team_id") is not None else None
    possession = (
        [tid == holder for tid in tracks[players]] if holder is not None else []
    )
    return draw(sport, positions, teams[players], possession=possession, **kwargs)
