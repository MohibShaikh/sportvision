from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src/sportvision/app.py")


def test_guided_upload_download_and_settings_invalidation():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert not app.button
    app.file_uploader[0].set_value(("clip.mp4", b"video", "video/mp4")).run()
    assert app.button[0].label == "Analyze clip"

    def analyze(source, output, **kwargs):
        assert source.read_bytes() == b"video"
        assert kwargs["team_colors"] == [(255, 255, 255), (21, 21, 21)]
        assert kwargs["analysis_mode"] == "detailed"
        output.write_bytes(b"annotated video")
        kwargs["progress"](10, 10, np.zeros((80, 120, 3), np.uint8))
        return {
            "frames_processed": 10,
            "frames_with_tracked_ball": 0,
            "attributed_frames": 0,
            "possession": {},
        }

    with patch("sportvision.video.analyze_video", side_effect=analyze):
        app.button[0].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    assert any("Not enough evidence" in w.value for w in app.warning)
    assert app.session_state.result["video"] == b"annotated video"
    app.color_picker[0].set_value("#FF0000").run()
    assert not app.get("download_button")


def test_processing_error_is_recoverable():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.file_uploader[0].set_value(("bad.mp4", b"bad", "video/mp4")).run()
    with patch("sportvision.video.analyze_video", side_effect=ValueError("bad video")):
        app.button[0].click().run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.get("download_button")


def test_identical_jerseys_disable_analysis():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.file_uploader[0].set_value(("clip.mp4", b"video", "video/mp4")).run()
    app.color_picker[1].set_value("#FFFFFF").run()
    assert app.button[0].disabled


def test_click_player_correct_restore_and_seek():
    from tests.test_review import make_review

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.file_uploader[0].set_value(("clip.mp4", b"video", "video/mp4")).run()

    def analyze(source, output, **kwargs):
        output.write_bytes(b"video")
        kwargs["progress"](3, 3, np.zeros((100, 150, 3), np.uint8))
        return {
            "frames_processed": 3,
            "frames_with_tracked_ball": 2,
            "attributed_frames": 2,
            "possession": {0: 1.0, 1: 0.0},
            "review": make_review(),
        }

    with (
        patch("sportvision.video.analyze_video", side_effect=analyze),
        patch(
            "sportvision.review_ui.read_frame",
            return_value=np.zeros((100, 150, 3), np.uint8),
        ),
        patch("sportvision.review_ui.player_view") as component,
    ):
        component.return_value = {
            "kind": "player",
            "track_id": 7,
            "frame_index": 0,
            "event_id": 1,
        }
        app.button[0].click().run()
        assert not app.exception
        assert app.selectbox(key="selected_track").value == 7
        app.slider(key="review_frame").set_value(1).run()
        app.radio[1].set_value(1).run()
        next(b for b in app.button if b.label == "Apply to this track").click().run()
        assert not app.exception
        result = app.session_state.result
        assert result["display_stats"]["possession"] == {0: 0, 1: 1}
        assert result["video"] is None
        assert app.slider(key="review_frame").value == 1
        assert len(app.get("download_button")) == 1
        component.return_value = {"kind": "seek", "frame_index": 2, "event_id": 2}
        app.run()
        assert not app.exception
        assert app.slider(key="review_frame").value == 2
        next(
            b for b in app.button if b.label == "Restore automatic assignment"
        ).click().run()
        assert not app.exception
        assert app.session_state.result["display_stats"]["possession"] == {0: 1, 1: 0}


def test_missed_ball_and_possession_corrections_are_reversible():
    from tests.test_review import make_review

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.file_uploader[0].set_value(("clip.mp4", b"video", "video/mp4")).run()

    def analyze(source, output, **kwargs):
        output.write_bytes(b"video")
        kwargs["progress"](3, 3, np.zeros((100, 150, 3), np.uint8))
        return {
            "frames_processed": 3,
            "frames_with_tracked_ball": 2,
            "attributed_frames": 2,
            "possession": {0: 1.0, 1: 0.0},
            "review": make_review(),
        }

    with (
        patch("sportvision.video.analyze_video", side_effect=analyze),
        patch(
            "sportvision.review_ui.read_frame",
            return_value=np.zeros((100, 150, 3), np.uint8),
        ),
        patch("sportvision.review_ui.player_view") as component,
    ):
        component.return_value = None
        app.button[0].click().run()
        app.slider(key="review_frame").set_value(2).run()
        app.toggle[0].set_value(True).run()
        component.return_value = {
            "kind": "ball",
            "frame_index": 2,
            "x": 110.0,
            "y": 10.0,
            "event_id": 42,
        }
        app.run()
        assert not app.exception
        result = app.session_state.result
        assert result["ball_overrides"] == {2: [110.0, 10.0]}
        assert result["display_stats"]["manually_marked_ball_frames"] == 1
        assert result["video"] is None
        assert app.slider(key="review_frame").value == 2
        app.selectbox(key="possession_choice_2").set_value(1).run()
        next(b for b in app.button if b.label == "Save frame possession").click().run()
        assert not app.exception
        assert (
            app.session_state.result["display_stats"]["manually_confirmed_frames"] == 1
        )
        next(
            b for b in app.button if b.label == "Restore model ball position"
        ).click().run()
        assert not app.exception
        assert (
            app.session_state.result["display_stats"]["review"]["frames"][2][
                "evidence"
            ]["reason"]
            == "manual_possession"
        )
        app.selectbox(key="possession_choice_2").set_value("model").run()
        next(b for b in app.button if b.label == "Save frame possession").click().run()
        assert not app.exception
        assert (
            app.session_state.result["display_stats"]["review"]["frames"][2][
                "evidence"
            ]["reason"]
            == "no_ball"
        )
