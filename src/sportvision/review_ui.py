"""Review a cached analysis and regenerate corrected exports."""

import base64
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from streamlit.components.v1 import declare_component

from sportvision.review import (
    REASONS,
    correct_review,
    review_summary,
    timeline_segments,
)
from sportvision.video import export_review, read_frame

player_view = declare_component(
    "player_review", path=str(Path(__file__).parent / "components" / "review")
)


def render_review(result, data, filename):
    original = result["stats"].get("review")
    if not original:
        return
    st.subheader("Review and correct")
    st.caption(
        "Corrections apply to every appearance of this track in this clip. "
        "If tracking loses a player and assigns a new ID, correct that track too."
    )
    overrides = result.setdefault("overrides", {})
    ball_overrides = result.setdefault("ball_overrides", {})
    possession_overrides = result.setdefault("possession_overrides", {})
    review = correct_review(
        original,
        overrides,
        ball_overrides=ball_overrides,
        possession_overrides=possession_overrides,
    )
    result["corrected_review"] = review
    rows = review["frames"]
    if not rows:
        return
    if "pending_review_frame" in st.session_state:
        st.session_state["review_frame"] = st.session_state.pop("pending_review_frame")
    if len(rows) > 1:
        index = st.slider("Review frame", 0, len(rows) - 1, key="review_frame")
    else:
        index = 0
    row = rows[index]
    with tempfile.TemporaryDirectory(prefix="sportvision-frame-") as tmp:
        source = Path(tmp) / ("clip" + Path(filename).suffix.lower())
        source.write_bytes(data)
        try:
            frame = read_frame(source, index)
        except ValueError as exc:
            st.error(str(exc))
            return
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        st.error("Could not display this frame")
        return
    players = [
        {"track_id": tid, "box": box, "team_id": team}
        for box, tid, team, cls in zip(
            row["xyxy"], row["tracker_id"], row["team_id"], row["class_id"]
        )
        if cls == 0
    ]
    mark_ball = st.toggle("Mark a missed ball on this frame", key="mark_ball")
    if mark_ball:
        st.info(
            "Click the center of the ball in the image. This edits only this frame."
        )
    event = player_view(
        image=base64.b64encode(jpeg).decode(),
        players=players,
        mark_ball=mark_ball,
        manual_ball=row.get("manual_ball_xy"),
        balls=[box for box, cls in zip(row["xyxy"], row["class_id"]) if cls == 1],
        width=review["width"],
        height=review["height"],
        frame_index=index,
        fps=review["fps"],
        total=len(rows),
        segments=timeline_segments(review),
        key="player_view",
        default=None,
    )
    if event and event.get("event_id") != result.get("last_event"):
        result["last_event"] = event.get("event_id")
        target = event.get("frame_index")
        if (
            event.get("kind") == "seek"
            and isinstance(target, int)
            and 0 <= target < len(rows)
        ):
            # A slider cannot be mutated after instantiation; apply on the next run.
            st.session_state["pending_review_frame"] = target
            st.rerun()
        if event.get("kind") == "ball" and target == index and mark_ball:
            point = [event.get("x"), event.get("y")]
            try:
                correct_review(
                    original, overrides, ball_overrides={**ball_overrides, index: point}
                )
            except (ValueError, TypeError):
                st.error("Choose a point inside the video frame.")
            else:
                ball_overrides[index] = point
                result["video"] = None
                st.session_state["pending_review_frame"] = index
                st.rerun()
        if event.get("kind") == "player" and target == index:
            tid = event.get("track_id")
            if tid in [p["track_id"] for p in players]:
                st.session_state["selected_track"] = tid
    evidence = row["evidence"]
    st.write(f"Evidence: {REASONS[evidence['reason']]}")
    if evidence["distance_px"] is not None:
        if evidence.get("method") == "body":
            st.caption(
                f"Nearest track: {evidence['tracker_id']} · "
                f"Ball distance from body: {evidence['distance_px']:.1f} px · "
                "Within 15% of player height; overlapping candidates are uncertain."
            )
        else:
            st.caption(
                f"Nearest track: {evidence['tracker_id']} · "
                f"Ball distance: {evidence['distance_px']:.1f} px · "
                f"Attribution threshold: {review['threshold_px']:.0f} px"
            )
    if index in ball_overrides:
        st.caption("Ball position: manually marked on this frame.")
        if st.button("Restore model ball position"):
            ball_overrides.pop(index)
            result["video"] = None
            st.session_state["pending_review_frame"] = index
            st.rerun()
    with st.expander("Correct possession for this frame"):
        st.caption(
            "If the player visibly has the ball, confirm the team here. "
            "This is a manual observation, not a successful model detection."
        )
        choices = {
            "model": "Use model estimate",
            0: "Team 1 has the ball",
            1: "Team 2 has the ball",
            None: "No team has possession",
        }
        confirmed = st.selectbox(
            "Possession on this frame",
            list(choices),
            index=list(choices).index(possession_overrides.get(index, "model")),
            format_func=choices.get,
            key=f"possession_choice_{index}",
        )
        if st.button("Save frame possession"):
            if confirmed == "model":
                possession_overrides.pop(index, None)
            else:
                possession_overrides[index] = confirmed
            result["video"] = None
            st.session_state["pending_review_frame"] = index
            st.rerun()
    tracks = sorted(
        {
            tid
            for r in rows
            for tid, cls in zip(r["tracker_id"], r["class_id"])
            if cls == 0 and tid >= 0
        }
    )
    if tracks:
        if st.session_state.get("selected_track") not in tracks:
            st.session_state["selected_track"] = tracks[0]
        tid = st.selectbox("Selected player track", tracks, key="selected_track")
        options = {0: "Team 1", 1: "Team 2", -2: "Referee / ignore"}
        selection = st.radio(
            "Assign selected player to",
            list(options),
            format_func=options.get,
            horizontal=True,
        )
        apply, restore = st.columns(2)
        if apply.button("Apply to this track"):
            overrides[tid] = selection
            result["video"] = None
            st.session_state["pending_review_frame"] = index
            st.rerun()
        if restore.button(
            "Restore automatic assignment", disabled=tid not in overrides
        ):
            overrides.pop(tid, None)
            result["video"] = None
            st.session_state["pending_review_frame"] = index
            st.rerun()
    result["display_stats"] = {
        **result["stats"],
        **review_summary(review),
        "review": review,
    }
    if overrides:
        st.caption(
            f"{len(overrides)} track correction(s) applied. "
            "The statistics and timeline reflect these changes."
        )
    if result["video"] is None:
        st.info(
            "Your report is updated. Build the corrected video to download "
            "matching annotations."
        )
        if st.button("Build corrected video", type="primary"):
            bar = st.progress(0)
            try:
                with tempfile.TemporaryDirectory(prefix="sportvision-export-") as tmp:
                    source = Path(tmp) / ("clip" + Path(filename).suffix.lower())
                    source.write_bytes(data)
                    output = Path(tmp) / "corrected.mp4"
                    export_review(
                        source,
                        output,
                        review,
                        progress=lambda done, total: bar.progress(done / total),
                    )
                    result["video"] = output.read_bytes()
                st.session_state["pending_review_frame"] = index
                st.rerun()
            except (ValueError, OSError, cv2.error) as exc:
                st.error(f"Could not export the corrected video: {exc}")
            finally:
                bar.empty()
