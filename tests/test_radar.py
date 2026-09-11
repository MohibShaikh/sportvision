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
    }
    image = radar.draw_frame("basketball", field, row, width_px=400)
    assert image.shape[1] == 400


def test_every_sport_renders():
    for sport in radar.COURTS:
        assert radar.draw(sport, width_px=300).shape[2] == 3
