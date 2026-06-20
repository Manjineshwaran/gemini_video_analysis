import tempfile
from pathlib import Path

import streamlit as st

from geimini_feedback_with_customer_query import (
    DEFAULT_FPS,
    DEFAULT_MEDIA_RESOLUTION,
    analyze_video_pipeline,
)


st.set_page_config(
    page_title="Gemini Fitness Analyzer",
    page_icon="🏋️",
    layout="centered",
)

st.title("Gemini Fitness Video Analyzer")
st.caption(
    "Upload an exercise video. Gemini analyzes reps, form errors, and gives "
    "timestamped feedback."
)

with st.sidebar:
    st.header("Analysis settings")
    fps = st.slider(
        "Frames per second",
        min_value=1.0,
        max_value=10.0,
        value=float(DEFAULT_FPS),
        step=0.5,
        help=(
            "Higher = better rep counting and form detail, but slower and "
            "uses more tokens. Try 8–10 for fast movements."
        ),
    )
    media_resolution = st.selectbox(
        "Frame detail",
        options=["low", "medium", "high"],
        index=["low", "medium", "high"].index(DEFAULT_MEDIA_RESOLUTION),
        help="High keeps more joint-angle detail for form analysis.",
    )
    body_side = st.selectbox(
        "Visible body side (rule-based)",
        options=["left", "right"],
        index=0,
        help="Which side of the body faces the camera for pose analysis.",
    )

uploaded_file = st.file_uploader(
    "Upload exercise video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded_file:
    st.video(uploaded_file)

customer_query = st.text_area(
    "Customer query (optional)",
    placeholder="Example: Is my squat depth good enough?",
)

analyze_clicked = st.button("Analyze Video", type="primary")

if analyze_clicked:
    if not uploaded_file:
        st.error("Please upload a video before running analysis.")
        st.stop()

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_video_path = None
    skeleton_video_path = None
    annotated_video_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_video_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix="_skeleton.mp4") as skel_tmp:
            skeleton_video_path = skel_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4") as out_tmp:
            annotated_video_path = out_tmp.name

        with st.status("Running analysis pipeline...", expanded=True) as status:
            st.write("Step 1: Rule-based pose analysis (MediaPipe skeleton)...")
            st.write("Step 2: Gemini coaching report...")
            st.write("Step 3: Burning mistake overlays onto video...")
            pipeline_result = analyze_video_pipeline(
                temp_video_path,
                customer_query=customer_query,
                fps=fps,
                media_resolution=media_resolution,
                side=body_side,
                skeleton_video_path=skeleton_video_path,
                out_video_path=annotated_video_path,
            )
            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False,
            )

        rule_result = pipeline_result["rule_based"]
        timing = pipeline_result.get("timing", {})
        gemini_error = pipeline_result.get("gemini_error")

        if gemini_error:
            st.warning(
                "Gemini video analysis was unavailable. "
                "Rule-based analysis, skeleton video, and mistake overlays are "
                "still available below. Try again in a minute for full video coaching."
            )
            with st.expander("Gemini error details"):
                st.code(gemini_error)
        elif pipeline_result.get("gemini_report"):
            st.success("Analysis complete.")
        else:
            st.success("Rule-based analysis complete.")

        if timing:
            st.subheader("Processing time")
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Rule-based", f"{timing.get('rule_based_sec', 0)}s")
            t2.metric("Gemini upload", f"{timing.get('gemini_upload_sec', 0)}s")
            t3.metric("Gemini response", f"{timing.get('gemini_response_sec', 0)}s")
            t4.metric("Video overlay", f"{timing.get('overlay_sec', 0)}s")
            t5.metric("Total", f"{timing.get('total_sec', 0)}s")

        skeleton_path = pipeline_result.get("skeleton_video_path")
        if skeleton_path and Path(skeleton_path).is_file():
            st.subheader("Skeleton overlay (rule-based)")
            st.caption("MediaPipe pose landmarks drawn during rule-based analysis.")
            skeleton_bytes = Path(skeleton_path).read_bytes()
            st.video(skeleton_bytes)
            st.download_button(
                label="Download skeleton video",
                data=skeleton_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_skeleton.mp4",
                mime="video/mp4",
                key="download_skeleton",
            )

        annotated_path = pipeline_result.get("annotated_video_path")
        if annotated_path and Path(annotated_path).is_file():
            st.subheader("Annotated video (coaching notes on skeleton)")
            st.caption(
                "Friendly coaching notes appear at each mistake timestamp "
                "on top of the skeleton overlay."
            )
            annotated_bytes = Path(annotated_path).read_bytes()
            st.video(annotated_bytes)
            st.download_button(
                label="Download annotated video",
                data=annotated_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_mistakes.mp4",
                mime="video/mp4",
                key="download_annotated",
            )

        st.subheader("Rule-Based Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total reps", rule_result.get("total_reps", 0))
        col2.metric("Correct reps", rule_result.get("correct_reps", 0))
        col3.metric("Accuracy", f"{rule_result.get('accuracy_percent', 0)}%")

        st.subheader("Rule-based JSON")
        st.json(rule_result)

        mistakes = rule_result.get("timestamped_mistakes") or []
        st.subheader("Coaching feedback")
        if mistakes:
            for m in mistakes:
                st.markdown(
                    f"- **Rep {m['rep_number']} at {m['timestamp']}** — {m['label']}"
                )
            with st.expander("Mistakes (structured JSON)"):
                st.json(mistakes)
        else:
            st.info("No form mistakes detected.")

        st.subheader("Gemini Fitness Report")
        gemini_report = pipeline_result.get("gemini_report")
        if gemini_report:
            if pipeline_result.get("gemini_fallback_used"):
                st.caption(
                    "Generated from rule-based data only — Gemini video API was "
                    "temporarily unavailable."
                )
            st.markdown(gemini_report)
        else:
            st.info(
                "No Gemini report for this run. Coaching feedback from "
                "rule-based analysis is shown above."
            )
    except Exception as exc:
        st.exception(exc)
    finally:
        if temp_video_path:
            Path(temp_video_path).unlink(missing_ok=True)
        if skeleton_video_path:
            Path(skeleton_video_path).unlink(missing_ok=True)
        if annotated_video_path:
            Path(annotated_video_path).unlink(missing_ok=True)
