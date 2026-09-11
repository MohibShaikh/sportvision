"""Local clip review UI. Launch with `sportvision app`."""

import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from sportvision import theme
from sportvision.review_ui import render_review
from sportvision.video import analyze_video

ISSUES = "https://github.com/MohibShaikh/sportvision/issues"


def bgr(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (5, 3, 1))


def main():
    st.set_page_config(
        page_title="SportVision · Clip review", page_icon="⚽", layout="wide"
    )
    theme.apply()
    theme.masthead(
        "Local edition · nothing leaves this machine",
        "See your clip.<br><em>Check the play.</em>",
        "Track the players, settle the team colors, and leave with an "
        "annotated video. <b>Every frame stays on your computer.</b>",
    )
    theme.section(1, "Add your clip", "step one of three")
    upload = st.file_uploader(
        "Choose a clip",
        type=["mp4", "mov", "avi", "mkv"],
        help="Up to 200 MB. Start with a short clip.",
    )
    if upload is None:
        theme.guide(
            [
                "Ten seconds is plenty for a first run. Processing takes "
                "longer than the clip runs.",
                "A steady, wide view works best. Keep both teams and the ball in shot.",
                "No account, no API key, no upload. The first run downloads "
                "model weights and then works offline.",
            ]
        )
        theme.note(
            "This is an experimental review tool. Read the annotations before "
            "you trust the estimates."
        )
        theme.colophon(
            [
                "SportVision",
                f'<a href="{ISSUES}">Help &amp; feedback</a>',
                "Apache-2.0",
            ]
        )
        st.session_state.pop("result", None)
        return
    data = upload.getvalue()
    identity = hashlib.sha256(data).hexdigest()
    if st.session_state.get("clip_id") != identity:
        st.session_state.pop("result", None)
        st.session_state["clip_id"] = identity
    with st.expander("Preview the source clip"):
        st.video(data)
    theme.section(2, "Set the teams", "step two of three")
    wide, narrow = st.columns([3, 2])
    with wide:
        mode = st.radio(
            "Team colors",
            ["Choose jersey colors", "Group automatically"],
            help=(
                "Chosen colors keep team labels consistent. Lighting and"
                " similar kits can still confuse the model."
            ),
        )
        left, right = st.columns(2)
        chosen = mode == "Choose jersey colors"
        color1 = left.color_picker("Team 1 jersey", "#FFFFFF", disabled=not chosen)
        color2 = right.color_picker("Team 2 jersey", "#151515", disabled=not chosen)
        theme.note(
            "Pick the dominant shirt color. If the grouping comes out wrong, "
            "change it and run again."
        )
    with narrow:
        limit = st.selectbox(
            "Review length",
            [300, 900, 1800],
            format_func=lambda n: (
                f"First {n} frames (about {n // 30} seconds at 30 FPS)"
            ),
        )
        analysis_mode = st.selectbox(
            "Analysis detail",
            ["quick", "detailed"],
            index=1,
            format_func=lambda value: (
                "Quick · faster preview"
                if value == "quick"
                else "Detailed · smaller ball search"
            ),
            help=(
                "Detailed uses higher resolution and a lower ball confidence "
                "threshold. It is slower and may also find false balls. "
                "Review the results."
            ),
        )
        theme.note(
            "Processing can take longer than the clip itself. The first run "
            "downloads model weights."
        )
    settings = (identity, mode, color1, color2, limit, analysis_mode)
    if st.session_state.get("result_settings") != settings:
        st.session_state.pop("result", None)
    invalid_colors = chosen and color1 == color2
    if invalid_colors:
        st.warning("Both teams are set to the same color. Change one of them.")
    if st.button("Analyze clip", type="primary", disabled=invalid_colors):
        st.session_state.pop("result", None)
        bar = st.progress(
            0, text="Loading the detector… the first run takes a little longer."
        )
        preview = st.empty()
        last_frame = []

        def progress(done, total, frame):
            last_frame[:] = [frame]
            bar.progress(
                min(done / total, 1.0) if total else 0, text=f"Reviewed {done} frames"
            )
            preview.image(
                frame,
                channels="BGR",
                caption="Live preview · watch the team assignments",
                use_container_width=True,
            )

        try:
            with tempfile.TemporaryDirectory(prefix="sportvision-") as tmp:
                source = Path(tmp) / ("input" + Path(upload.name).suffix.lower())
                output = Path(tmp) / "review.mp4"
                source.write_bytes(data)
                stats = analyze_video(
                    source,
                    output,
                    max_frames=limit,
                    analysis_mode=analysis_mode,
                    team_colors=[bgr(color1), bgr(color2)] if chosen else None,
                    progress=progress,
                )
                st.session_state.result = {
                    "video": output.read_bytes(),
                    "stats": stats,
                    "preview": last_frame[0],
                }
                st.session_state.result_settings = settings
                for key in ("review_frame", "pending_review_frame", "selected_track"):
                    st.session_state.pop(key, None)
                if "review" in stats:
                    st.session_state["review_frame"] = next(
                        (
                            r["frame_index"]
                            for r in stats["review"]["frames"]
                            if any(
                                cls == 0 and tid >= 0
                                for cls, tid in zip(r["class_id"], r["tracker_id"])
                            )
                        ),
                        0,
                    )
            bar.empty()
            preview.empty()
        except Exception as exc:
            bar.empty()
            preview.empty()
            st.error(
                "We couldn't finish this clip. Try a shorter MP4, and check your "
                "connection if the model weights haven't downloaded yet."
            )
            with st.expander("Error details"):
                st.code(str(exc))
    result = st.session_state.get("result")
    if result:
        theme.section(3, "Read the report", "step three of three")
        render_review(result, data, upload.name)
        stats = result.get("display_stats", result["stats"])
        if "review" not in stats:
            st.image(
                result["preview"],
                channels="BGR",
                caption="Last reviewed frame · check the team colors",
                use_container_width=True,
            )
        if stats["possession"]:
            inks = (theme.PALETTE["team_1"], theme.PALETTE["team_2"])
            theme.shares(
                [
                    (
                        f"Team {int(team) + 1}",
                        share,
                        inks[int(team) % 2],
                    )
                    for team, share in stats["possession"].items()
                ]
            )
            theme.note(
                "Estimated from attributed frames only. This is not official "
                "match possession."
            )
        else:
            st.warning(
                "Not enough evidence for possession, so no percentage has been "
                "invented. Look for missed balls or wrong team colors in the "
                "review above."
            )
        ball_share = stats["frames_with_tracked_ball"] / stats["frames_processed"]
        theme.figures(
            [
                (f"{ball_share:.0%}", "of frames had a ball the model could track"),
                (stats["frames_processed"], "frames reviewed"),
                (stats["attributed_frames"], "frames used for possession"),
            ]
        )
        theme.note(
            "Ball coverage is the share of frames with a detected ball. It is "
            "not a detector accuracy score."
        )
        manual_count = stats.get("manually_confirmed_frames", 0)
        marked_count = stats.get("manually_marked_ball_frames", 0)
        if manual_count or marked_count:
            theme.note(
                f"Manual evidence: {manual_count} possession confirmations and "
                f"{marked_count} marked ball positions are counted in this "
                "report. Model detection coverage is unchanged."
            )
        if stats.get("excluded_frames"):
            with st.expander("Why frames were excluded"):
                from sportvision.review import REASONS

                for reason, count in stats["excluded_frames"].items():
                    st.write(f"{REASONS[reason]}: {count} frames")
        take, report = st.columns([2, 3])
        if result["video"] is not None:
            take.download_button(
                "Download annotated video",
                result["video"],
                file_name="sportvision-review.mp4",
                mime="video/mp4",
                type="primary",
            )
        report.download_button(
            "Download report (JSON)",
            json.dumps(stats, indent=2),
            file_name="sportvision-report.json",
            mime="application/json",
        )
        theme.note(
            "Open the MP4 in any video player. Frame and track corrections "
            "update the report without rerunning detection."
        )
        with st.expander("Help improve SportVision"):
            st.write(
                "Did this save you time? Tell us your sport, your camera view, "
                "what failed, and whether you'd run it on another clip. Attach "
                "the report if you like; no need to share private footage."
            )
            st.link_button("Share feedback", f"{ISSUES}/new")
        theme.colophon(
            [
                "SportVision",
                f'<a href="{ISSUES}">Help &amp; feedback</a>',
                "Apache-2.0",
            ]
        )


if __name__ == "__main__":
    main()
