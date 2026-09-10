"""Local clip review UI. Launch with `sportvision app`."""

import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from sportvision.review_ui import render_review
from sportvision.video import analyze_video


def bgr(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (5, 3, 1))


def main():
    st.set_page_config(
        page_title="SportVision · Clip review", page_icon="⚽", layout="wide"
    )
    st.caption("SPORTVISION / CLIP REVIEW")
    st.title("See your clip. Check the play.")
    st.write("Track players, review team colors, and take an annotated video with you.")
    st.caption(
        "Runs on your computer. Your video is not sent to a video-analysis service."
    )
    with st.sidebar:
        st.subheader("A good first clip")
        st.write(
            "Start with 10 seconds of a steady, wide view. Keep both teams and the "
            "ball visible. "
        )
        st.info(
            "This is an experimental review tool. Check the annotations before "
            "using the estimates. "
        )
        st.link_button(
            "Help & feedback", "https://github.com/MohibShaikh/sportvision/issues"
        )
    st.subheader("1. Add your video")
    upload = st.file_uploader(
        "Choose a clip",
        type=["mp4", "mov", "avi", "mkv"],
        help="Up to 200 MB. Start with a short clip.",
    )
    if upload is None:
        st.info("Drop a video above to get started. No account or API key needed.")
        st.session_state.pop("result", None)
        return
    data = upload.getvalue()
    identity = hashlib.sha256(data).hexdigest()
    if st.session_state.get("clip_id") != identity:
        st.session_state.pop("result", None)
        st.session_state["clip_id"] = identity
    with st.expander("Preview your source clip"):
        st.video(data)
    st.subheader("2. Set up your review")
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio(
            "Team colors",
            ["Choose jersey colors", "Group automatically"],
            help=(
                "Chosen colors keep team labels consistent. Lighting and"
                " similar kits can still confuse the model."
            ),
        )
        left, right = st.columns(2)
        color1 = left.color_picker(
            "Team 1 jersey", "#FFFFFF", disabled=mode != "Choose jersey colors"
        )
        color2 = right.color_picker(
            "Team 2 jersey", "#151515", disabled=mode != "Choose jersey colors"
        )
        st.caption(
            "Pick the main shirt color. If the grouping looks wrong, adjust it and "
            "run again. "
        )
    with col2:
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
        st.caption(
            "Processing can take longer than the clip. The first run downloads "
            "model weights. "
        )
    settings = (identity, mode, color1, color2, limit, analysis_mode)
    if st.session_state.get("result_settings") != settings:
        st.session_state.pop("result", None)
    invalid_colors = mode == "Choose jersey colors" and color1 == color2
    if invalid_colors:
        st.warning("Choose different colors for the two teams.")
    if st.button("Analyze clip", type="primary", disabled=invalid_colors):
        st.session_state.pop("result", None)
        bar = st.progress(
            0, text="Loading the detector… first run may take a little longer."
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
                caption="Live preview · check the team assignments",
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
                    team_colors=[bgr(color1), bgr(color2)]
                    if mode == "Choose jersey colors"
                    else None,
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
                "connection if model weights haven't downloaded yet. "
            )
            with st.expander("Error details"):
                st.code(str(exc))
    result = st.session_state.get("result")
    if result:
        st.subheader("3. Take your review")
        st.caption(
            "Overlay colors: Team 1 is blue; Team 2 is red. Gray means unassigned."
        )
        render_review(result, data, upload.name)
        stats = result.get("display_stats", result["stats"])
        if "review" not in stats:
            st.image(
                result["preview"],
                channels="BGR",
                caption="Last reviewed frame · check the team colors",
                use_container_width=True,
            )
        a, b, c = st.columns(3)
        a.metric("Frames reviewed", stats["frames_processed"])
        b.metric(
            "Frames where model found a ball",
            f"{stats['frames_with_tracked_ball'] / stats['frames_processed']:.0%}",
        )
        c.metric("Frames used for possession", stats["attributed_frames"])
        manual_count = stats.get("manually_confirmed_frames", 0)
        marked_count = stats.get("manually_marked_ball_frames", 0)
        if manual_count or marked_count:
            st.caption(
                f"Manual evidence: {manual_count} possession confirmations, "
                f"{marked_count} marked ball positions. Included in this report; "
                "model detection coverage remains unchanged."
            )
        if stats.get("excluded_frames"):
            with st.expander("Why frames were excluded"):
                from sportvision.review import REASONS

                for reason, count in stats["excluded_frames"].items():
                    st.write(f"{REASONS[reason]}: {count} frames")
        st.caption(
            "Ball coverage is the share of frames with a detected ball, not a "
            "detector accuracy score. "
        )
        if stats["possession"]:
            for team, share in stats["possession"].items():
                st.write(f"Team {int(team) + 1} · estimated possession {share:.0%}")
            st.caption(
                "Shares cover attributed frames only. They are not official match "
                "possession. "
            )
        else:
            st.warning(
                "Not enough evidence for possession. No percentage has been invented. "
                "Review the video for missed balls or incorrect team colors. "
            )
        if result["video"] is not None:
            st.download_button(
                "Download annotated video",
                result["video"],
                file_name="sportvision-review.mp4",
                mime="video/mp4",
                type="primary",
            )
        st.download_button(
            "Download report",
            json.dumps(stats, indent=2),
            file_name="sportvision-report.json",
            mime="application/json",
        )
        st.caption(
            "Open the downloaded MP4 in your video player. Frame and track "
            "corrections update the report without rerunning detection."
        )
        with st.expander("Help improve SportVision"):
            st.write(
                "Did this save you time? Tell us your sport, camera view, what failed, "
                "and whether you'd use it on another clip. Include the report if you "
                "want; no need to share private footage. "
            )
            st.link_button(
                "Share feedback",
                "https://github.com/MohibShaikh/sportvision/issues/new",
            )


if __name__ == "__main__":
    main()
