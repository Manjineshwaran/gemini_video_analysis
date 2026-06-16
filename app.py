import tempfile
from pathlib import Path

import streamlit as st

from geimini_feedback_with_customer_query import analyze_video, upload_video


st.set_page_config(
    page_title="Gemini Fitness Analyzer",
    page_icon="🏋️",
    layout="centered",
)

st.title("Gemini Fitness Video Analyzer")
st.caption("Upload an exercise video and optionally include a customer query.")

uploaded_file = st.file_uploader(
    "Upload exercise video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

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

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_video_path = tmp.name

        with st.spinner("Uploading and processing video in Gemini..."):
            video_file = upload_video(temp_video_path)

        with st.spinner("Analyzing movement and generating feedback..."):
            result = analyze_video(
                video_file,
                customer_query=customer_query,
            )

        st.success("Analysis complete.")
        st.subheader("Gemini Fitness Report")
        st.markdown(result)
    except Exception as exc:
        st.exception(exc)
    finally:
        if temp_video_path:
            Path(temp_video_path).unlink(missing_ok=True)
