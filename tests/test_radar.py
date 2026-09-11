import numpy as np
import pytest

from sportvision import radar


def square_calibration(sport="basketball"):
    """A synthetic camera that maps the court onto a 1000x600 image."""
    length, width = radar.court_size(sport)
    image = np.array([[100, 100], [900, 100], [900, 500], [100, 500]], dtype=np.float32)
    return radar.calibrate(sport, image), image, length, width


def test_court_corners_match_presets():
    length, width = radar.court_size("basketball")
    assert (length, width) == (28.65, 15.24)
    corners = radar.court_corners("basketball")
    assert corners.shape == (4, 2)
    assert np.allclose(corners[0], [0, 0])
    assert np.allclose(corners[2], [length, width])


def test_calibrate_maps_corners_onto_the_court():
    field, image, length, width = square_calibration()
    projected = field.transform(image)
    expected = radar.court_corners("basketball")
    assert np.allclose(projected, expected, atol=1e-3)


@pytest.mark.parametrize(
    "points", [np.zeros((3, 2)), np.zeros((5, 2)), np.zeros((4, 3))]
)
def test_calibrate_rejects_wrong_shapes(points):
    with pytest.raises(ValueError):
        radar.calibrate("basketball", points)


def test_calibrate_rejects_duplicate_corners():
    points = [[0, 0], [0, 0], [10, 10], [0, 10]]
    with pytest.raises(ValueError):
        radar.calibrate("basketball", points)


def test_feet_is_bottom_center_of_the_box():
    assert radar.feet([[10, 20, 30, 60]]).tolist() == [[20.0, 60.0]]


def test_feet_handles_no_boxes():
    assert radar.feet([]).shape == (0, 2)


def test_project_puts_a_center_box_at_the_center_of_the_court():
    field, _, length, width = square_calibration()
    # A box whose feet sit at the image center of the calibrated quad.
    box = [[480, 280, 520, 300]]
    position = radar.project(field, box)[0]
    assert np.allclose(position, [length / 2, width / 2], atol=0.2)


def test_on_court_excludes_the_bench():
    positions = [[5, 5], [-9, 5], [5, 40]]
    assert radar.on_court(positions, "basketball").tolist() == [True, False, False]


def test_on_court_handles_empty():
    assert radar.on_court([], "basketball").shape == (0,)


def test_draw_returns_an_image_with_the_court_aspect_ratio():
    image = radar.draw("basketball", width_px=400, margin=20)
    length, width = radar.court_size("basketball")
    usable = 400 - 40
    assert image.shape[1] == 400
    assert image.shape[0] == round(width * usable / length) + 40
    assert image.dtype == np.uint8


def test_draw_puts_a_dot_where_the_player_is():
    blank = radar.draw("basketball", width_px=400)
    length, width = radar.court_size("basketball")
    drawn = radar.draw("basketball", [[length / 2, width / 2]], [0], width_px=400)
    assert not np.array_equal(blank, drawn)


def test_draw_ignores_positions_off_the_court():
    blank = radar.draw("basketball", width_px=400)
    drawn = radar.draw("basketball", [[-50, -50]], [0], width_px=400)
    assert np.array_equal(blank, drawn)


def test_draw_frame_uses_players_only():
    field, _, _, _ = square_calibration()
    row = {
        "xyxy": [[480, 280, 520, 300], [200, 200, 240, 260]],
        "class_id": [0, 1],
        "team_id": [0, -1],
        "tracker_id": [7, -1],
    }
    image = radar.draw_frame("basketball", field, row, width_px=400)
    assert image.shape[1] == 400


def test_the_ball_never_gets_a_position_of_its_own():
    """A floor homography cannot place a ball that is not on the floor.

    Projecting one held at chest height puts it about 16 m from the player, so
    an unattributed frame with a ball must render exactly like one without.
    """
    field, _, _, _ = square_calibration()
    player = [480, 280, 520, 300]
    base = {"xyxy": [player], "class_id": [0], "team_id": [0], "tracker_id": [7]}
    without = radar.draw_frame("basketball", field, base, width_px=400)
    with_ball = radar.draw_frame(
        "basketball",
        field,
        {
            "xyxy": [player, [700, 150, 720, 170]],
            "class_id": [0, 1],
            "team_id": [0, -1],
            "tracker_id": [7, -1],
            "evidence": {"team_id": None, "tracker_id": None},
        },
        width_px=400,
    )
    assert np.array_equal(without, with_ball)


def test_attributed_possession_rings_the_player_holding_the_ball():
    field, _, _, _ = square_calibration()
    row = {
        "xyxy": [[480, 280, 520, 300]],
        "class_id": [0],
        "team_id": [0],
        "tracker_id": [7],
    }
    plain = radar.draw_frame("basketball", field, row, width_px=400)
    ringed = radar.draw_frame(
        "basketball",
        field,
        {**row, "evidence": {"team_id": 0, "tracker_id": 7}},
        width_px=400,
    )
    assert not np.array_equal(plain, ringed)


def test_unattributed_evidence_rings_nobody():
    field, _, _, _ = square_calibration()
    row = {
        "xyxy": [[480, 280, 520, 300]],
        "class_id": [0],
        "team_id": [0],
        "tracker_id": [7],
    }
    plain = radar.draw_frame("basketball", field, row, width_px=400)
    # Nearest player known, but attribution failed, so team_id stays None.
    rejected = radar.draw_frame(
        "basketball",
        field,
        {**row, "evidence": {"team_id": None, "tracker_id": 7}},
        width_px=400,
    )
    assert np.array_equal(plain, rejected)


def test_the_ring_follows_the_track_id_not_the_row_order():
    field, _, _, _ = square_calibration()
    row = {
        "xyxy": [[300, 280, 340, 300], [480, 280, 520, 300]],
        "class_id": [0, 0],
        "team_id": [0, 1],
        "tracker_id": [3, 9],
    }
    first = radar.draw_frame(
        "basketball",
        field,
        {**row, "evidence": {"team_id": 0, "tracker_id": 3}},
        width_px=400,
    )
    second = radar.draw_frame(
        "basketball",
        field,
        {**row, "evidence": {"team_id": 1, "tracker_id": 9}},
        width_px=400,
    )
    assert not np.array_equal(first, second)


def test_every_sport_renders():
    for sport in radar.COURTS:
        assert radar.draw(sport, width_px=300).shape[2] == 3


class TestRejectsBadCalibration:
    """A degenerate homography draws plausible looking, wrong positions.

    Refusing is better than that, so each of these must raise rather than
    return a field.
    """

    @pytest.mark.parametrize(
        "points",
        [
            pytest.param([[0, 0], [10, 0], [20, 0], [30, 0]], id="collinear"),
            pytest.param([[0, 0], [0, 1], [0.0001, 0], [5, 5]], id="near-degenerate"),
            pytest.param([[0, 0], [1, 1], [2, 2], [3, 3]], id="diagonal-line"),
        ],
    )
    def test_points_that_enclose_nothing(self, points):
        with pytest.raises(ValueError, match="line"):
            radar.calibrate("basketball", points)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_corners(self, bad):
        with pytest.raises(ValueError, match="finite"):
            radar.calibrate("basketball", [[0, 0], [bad, 0], [1, 1], [0, 1]])

    def test_non_numeric_corners(self):
        with pytest.raises(ValueError, match="numbers"):
            radar.calibrate("basketball", [["a", "b"], [1, 0], [1, 1], [0, 1]])

    def test_unknown_sport_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="basketball"):
            radar.calibrate("hockey", [[0, 0], [9, 0], [9, 5], [0, 5]])


class TestAcceptsRealCalibration:
    @pytest.mark.parametrize(
        "points",
        [
            pytest.param([[100, 100], [900, 100], [900, 500], [100, 500]], id="square"),
            pytest.param(
                [[205, 362], [1193, 362], [1226, 597], [173, 597]], id="perspective"
            ),
            pytest.param([[10, 10], [60, 12], [62, 40], [8, 38]], id="small-and-skew"),
        ],
    )
    def test_quads_that_enclose_the_court(self, points):
        field = radar.calibrate("basketball", points)
        recovered = field.transform(np.asarray(points, dtype=np.float32))
        assert np.allclose(recovered, radar.court_corners("basketball"), atol=0.5)


class TestDrawBounds:
    @pytest.mark.parametrize("width_px", [0, -100, 10])
    def test_canvas_too_small_to_draw(self, width_px):
        with pytest.raises(ValueError, match="width_px"):
            radar.draw("basketball", width_px=width_px)

    def test_unknown_sport(self):
        with pytest.raises(ValueError, match="basketball"):
            radar.draw("hockey")

    def test_non_finite_positions_are_dropped_not_drawn(self):
        blank = radar.draw("basketball", width_px=300)
        for bad in ([[np.nan, np.nan]], [[np.inf, 1]], [[1, -np.inf]]):
            assert np.array_equal(
                radar.draw("basketball", bad, [0], width_px=300), blank
            )

    def test_more_teams_than_positions_and_the_reverse(self):
        assert radar.draw("basketball", [[5, 5]], [0, 1, 1], width_px=300) is not None
        assert radar.draw("basketball", [[5, 5], [6, 6]], [0], width_px=300) is not None

    def test_unknown_team_id_still_draws(self):
        assert radar.draw("basketball", [[5, 5]], [None], width_px=300) is not None


class TestDrawFrameTolerates:
    def test_row_without_tracker_ids(self):
        field, _, _, _ = square_calibration()
        row = {"xyxy": [[480, 280, 520, 300]], "class_id": [0], "team_id": [0]}
        assert radar.draw_frame("basketball", field, row, width_px=300) is not None

    def test_completely_empty_row(self):
        field, _, _, _ = square_calibration()
        row = {"xyxy": [], "class_id": [], "team_id": [], "tracker_id": []}
        assert radar.draw_frame("basketball", field, row, width_px=300) is not None

    def test_row_with_only_a_ball(self):
        field, _, _, _ = square_calibration()
        row = {
            "xyxy": [[1, 2, 3, 4]],
            "class_id": [1],
            "team_id": [-1],
            "tracker_id": [-1],
        }
        assert radar.draw_frame("basketball", field, row, width_px=300) is not None

    def test_evidence_naming_a_track_that_is_not_present(self):
        field, _, _, _ = square_calibration()
        row = {
            "xyxy": [[480, 280, 520, 300]],
            "class_id": [0],
            "team_id": [0],
            "tracker_id": [7],
            "evidence": {"team_id": 0, "tracker_id": 999},
        }
        assert radar.draw_frame("basketball", field, row, width_px=300) is not None
