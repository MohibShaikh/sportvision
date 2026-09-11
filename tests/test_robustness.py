"""Edge cases found by auditing, kept here so they cannot come back."""

import math

import pytest

from sportvision import review

FRAME = {
    "frame_index": 0,
    "xyxy": [[10, 10, 20, 40]],
    "class_id": [0],
    "team_id": [0],
    "tracker_id": [1],
    "evidence": {
        "reason": "attributed",
        "team_id": 0,
        "tracker_id": 1,
        "distance_px": 5.0,
        "ball_xy": [15, 15],
        "method": "center",
        "ball_source": "detector",
    },
}
REVIEW = {
    "width": 100,
    "height": 100,
    "fps": 30.0,
    "threshold_px": 50.0,
    "frames": [FRAME],
}


@pytest.mark.parametrize("fps", [0, 0.0, -5, float("nan"), float("inf"), None, "abc"])
def test_timeline_survives_a_broken_frame_rate(fps):
    """The timeline is cosmetic, so a bad fps must not take the page down."""
    segments = review.timeline_segments({**REVIEW, "fps": fps})
    assert segments
    assert math.isfinite(segments[0]["start_s"])
    assert math.isfinite(segments[0]["end_s"])


def test_timeline_of_an_empty_review():
    assert review.timeline_segments({**REVIEW, "frames": []}) == []


def test_summary_of_an_empty_review():
    summary = review.review_summary({**REVIEW, "frames": []})
    assert summary["possession"] == {}
    assert summary["attributed_frames"] == 0


class TestCorrectionsValidateInput:
    """Bad corrections must raise a message worth showing, not corrupt state."""

    def test_unknown_track(self):
        with pytest.raises(ValueError, match="track"):
            review.correct_review(REVIEW, {999: 1})

    def test_team_outside_the_allowed_set(self):
        with pytest.raises(ValueError, match="Team"):
            review.correct_review(REVIEW, {1: 42})

    @pytest.mark.parametrize("point", [[9999, 9999], [-5, -5], None, [float("nan"), 1]])
    def test_ball_outside_the_frame(self, point):
        with pytest.raises(ValueError, match="[Bb]all"):
            review.correct_review(REVIEW, {}, ball_overrides={0: point})

    def test_ball_on_a_frame_that_was_not_reviewed(self):
        with pytest.raises(ValueError, match="[Bb]all"):
            review.correct_review(REVIEW, {}, ball_overrides={99: [5, 5]})

    def test_possession_override_of_junk(self):
        with pytest.raises(ValueError, match="[Pp]ossession"):
            review.correct_review(REVIEW, {}, possession_overrides={0: "x"})

    @pytest.mark.parametrize("team", [0, 1, None])
    def test_valid_possession_overrides_are_accepted(self, team):
        corrected = review.correct_review(REVIEW, {}, possession_overrides={0: team})
        assert corrected["frames"]
