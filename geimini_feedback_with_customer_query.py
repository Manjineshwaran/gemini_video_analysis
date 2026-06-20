import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

import rule_based


# -----------------------------
# ENV
# -----------------------------

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY missing"
    )


client = genai.Client(
    api_key=API_KEY
)

MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
]

# Frames sampled per second of video.
# Gemini defaults to 1 FPS, which is far too low to count reps or catch
# form errors. 5 FPS captures the bottom/top of most exercise movements.
DEFAULT_FPS = 5.0

# HIGH keeps enough per-frame detail to read joint angles (knees, hips, back).
DEFAULT_MEDIA_RESOLUTION = "high"



# -----------------------------
# UPLOAD VIDEO
# -----------------------------


def upload_video(video_path):

    print("Uploading video...")

    video_file = client.files.upload(
        file=video_path
    )


    print(
        "Uploaded:",
        video_file.name
    )


    # wait until Gemini processes video

    while video_file.state.name == "PROCESSING":

        print("Processing video...")

        time.sleep(5)

        video_file = client.files.get(
            name=video_file.name
        )


    if video_file.state.name == "FAILED":

        raise RuntimeError(
            "Video processing failed"
        )


    print(
        "Video ready"
    )

    return video_file




# -----------------------------
# PROMPT
# -----------------------------


BASE_PROMPT = """
    You are an expert strength & conditioning coach and computer vision exercise analyst.

    Your ONLY priority is HIGH ACCURACY exercise analysis from the actual video frames.

    Do NOT guess.
    Do NOT predict.
    Do NOT follow a rhythm.
    Do NOT assume repeated movements.
    Do NOT create evenly spaced repetitions.

    You must analyze the video like a frame-by-frame human reviewer.

    ==============================
    FRAME ANALYSIS RULES
    ==============================

    1. First inspect the complete video:
    - Determine exact FPS and duration if available.
    - Review the entire video from beginning to end.
    - Analyze the movement continuously across frames.
    - Do not jump directly to counting reps.

    2. For every possible repetition:
    You MUST verify:

    START POSITION:
    - Identify the initial position of the exercise.

    MOVEMENT PHASE:
    - Track the body landmark responsible for the exercise:
    Push-up:
        - chest/shoulder vertical movement
        - elbow angle
        - body alignment

    Squat:
        - hip height
        - knee angle

    Curl:
        - elbow angle / forearm movement

    BOTTOM OR TURNING POINT:
    - Identify the exact lowest/highest point.

    RETURN:
    - Confirm the body returned to the original start position.

    ONLY AFTER RETURNING TO START POSITION:
    Count it as ONE REP.

    If the person only completes half movement:
    DO NOT COUNT IT.

    ==============================
    ANTI-HALLUCINATION RULES
    ==============================

    NEVER:
    - count based on time intervals
    - count based on average rep duration
    - assume a rep because the body moved
    - copy the previous rep description
    - invent missing timestamps
    - force the count to look normal

    If you are unsure between two counts:
    Choose the LOWER count.

    Accuracy is more important than completeness.

    ==============================
    REP VALIDATION TABLE
    ==============================

    Before giving final output, internally create:

    Rep number:
    Start timestamp:
    Bottom/peak timestamp:
    Return timestamp:
    Completed? YES/NO

    Only completed YES reps appear in final output.

    ==============================
    MOTION TRACE
    ==============================

    Describe the actual observed movement:

    Example:

    00:00 - standing plank position
    00:02 - body lowering
    00:03 - lowest chest position reached
    00:05 - body returned to plank

    Do this chronologically.

    ==============================
    COACHING LANGUAGE (REQUIRED)
    ==============================

    Write like a supportive personal trainer talking to a real person.
    Use plain, everyday English — not biomechanics jargon.

    For every mistake, use this pattern:
    1. Say what went wrong in simple words ("Your hips are sagging")
    2. Say what to do instead ("keep your body straight and brace your core")

    DO:
    - Speak directly to the athlete using "you" and "your"
    - Keep each point to 1-2 short, clear sentences
    - Sound encouraging, not clinical

    DO NOT:
    - Use technical labels like "hip sag", "incomplete lockout", or "eccentric phase"
    - Use dash-separated jargon like "Hip sag — body line dropped, core not braced"
    - List error codes or angles without explaining them in plain English

    BAD:  "Hip sag — body line dropped, core not braced"
    GOOD: "Your hips dropped below your shoulders. Tighten your core and keep your body in one straight line from head to heels."

    BAD:  "Wrist not under shoulder — poor stacking"
    GOOD: "Your hands are too far forward. Place your wrists directly under your shoulders."

    ==============================
    COACHING LANGUAGE (REQUIRED)
    ==============================

    Write for a real person at the gym — like a coach talking face to face.

    DO:
    - Use plain, everyday English anyone can understand
    - Say what is wrong AND what to fix
      Example: "Your hips are sagging — straighten your body and brace your core."
    - Keep each mistake to one or two short, clear sentences
    - Use "you" and "your" when giving feedback

    DO NOT:
    - Use technical labels or jargon (e.g. "hip sag", "ROM", "eccentric phase")
    - Use dash-separated diagnosis strings (e.g. "Hip sag — body line dropped")
    - Copy raw error codes or internal field names from the JSON
    - List timestamps with cluttered ranges like "(from 0:00.00 to 0:01.37)"

    ==============================
    FINAL OUTPUT FORMAT
    ==============================


    Video duration:
    (actual duration)

    Exercise:
    (name)

    Main landmark used:
    (the body point used for counting)

    Motion trace:
    (real timeline observation)

    Validated repetitions:

    Rep 1:
    Start:
    Bottom:
    Return:
    Status:
    Completed

    Rep 2:
    Start:
    Bottom:
    Return:
    Status:
    Completed


    Counted turnarounds:
    - Rep 1 bottom at MM:SS
    - Rep 2 bottom at MM:SS


    Total reps:
    (number of validated completed repetitions only)


    Rep-by-rep coaching:

    Rep 1:
    - What went well: (plain English)
    - What to fix: (plain English — say what you saw and what to change)
    - Mistakes: (one friendly sentence per mistake, with timestamp)

    Rep 2:
    - What went well:
    - What to fix:
    - Mistakes:


    Speed analysis:
    Describe in plain English:
    - How fast they move on the way down
    - How fast they push back up
    - Whether the pace stays consistent across reps

    Do NOT say "good speed" without explaining why in simple terms.


    Overall form rating:
    Excellent / Good / Needs work / Poor
    (Add one short sentence explaining why in everyday language.)


    Mistakes detected:
    List every flagged mistake as a clear coaching line:
    - Rep number and exact timestamp (from the data — do not invent times)
    - One plain-English sentence: what is wrong and what to do differently
    Example format:
    "Rep 1 at 0:02 — Your hips are sagging. Straighten your body and brace your core."


    Improvement suggestions:
    Give 3-6 simple coaching cues a beginner would understand.


    Confidence:
    High / Medium / Low

    Explain limitations:
    - camera angle
    - visibility
    - occlusion
    - blur
    - body parts hidden


    FINAL CHECK BEFORE ANSWERING:

    Verify:
    1. Total reps == number of validated completed reps
    2. Every counted rep has start + bottom + return
    3. No timestamp exceeds video duration
    4. No rep was counted only because of repeated timing
    5. If uncertain, lower the count

    Never sacrifice accuracy to make the answer look complete.
"""


CUSTOMER_QUERY_PROMPT = """

    The customer has asked the following question:

    "{customer_query}"

    Important:
    - Still provide the full exercise analysis described above.
    - You MUST directly answer the customer's question using what you observe in the video.
    - Base your answer on the video evidence; do not guess.
    - Answer in plain, friendly English — like a coach talking to the athlete.
    - If the question is about form, reps, depth, speed, or technique, focus your analysis on that area.
    - Add a dedicated section titled "Customer question answer" with a clear, direct response.

"""

RULE_BASED_CONTEXT_PROMPT = """

    ==============================
    RULE-BASED POSE ANALYSIS (GROUND TRUTH)
    ==============================

    The JSON below was computed from MediaPipe pose estimation on every frame.
    Treat these numbers as authoritative for:
    - rep count and per-rep timing
    - measurable form errors (hip sag, depth, lockout, wrist alignment, etc.)
    - fatigue trends across reps

    Your job:
    - Do NOT contradict rep counts, error flags, or timing in this JSON.
    - Use the video to explain WHY each flagged error happened and give coaching cues.
    - For "timestamped_mistakes", use the exact timestamps provided — do NOT invent new ones.
    - Each item in timestamped_mistakes has a "label" field with a plain-English coaching
      message. Use that wording (or rephrase in the same friendly style) in your report.
    - In "Mistakes detected", list every item from timestamped_mistakes as:
      "Rep N at MM:SS — [plain coaching sentence]"
    - Do NOT repeat technical error codes (e.g. hip_sag, incomplete_lockout).
    - Do NOT claim to measure items listed under "not_measurable".
    - If the video visibly conflicts with the JSON, note the discrepancy and trust the JSON.

    ```json
    {rule_based_json}
    ```

"""


def build_prompt(customer_query=None, rule_based_data=None):

    query = (customer_query or "").strip()
    prompt = BASE_PROMPT

    if rule_based_data is not None:
        coaching_json = rule_based.to_coaching_payload(rule_based_data)
        prompt += RULE_BASED_CONTEXT_PROMPT.format(
            rule_based_json=json.dumps(coaching_json, indent=2),
        )

    if query:
        prompt += CUSTOMER_QUERY_PROMPT.format(
            customer_query=query
        )

    return prompt


def _normalize_model_name(model_name):

    return (
        (model_name or "")
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def _build_model_candidates():

    env_model = _normalize_model_name(
        os.getenv("GEMINI_MODEL", "")
    )

    candidates = [
        m for m in [env_model, *MODEL_CANDIDATES] if m
    ]

    # preserve order and remove duplicates
    seen = set()
    unique = []
    for model in candidates:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _is_retryable_error(exc):

    text = str(exc).upper()
    return (
        "503" in text
        or "500" in text
        or "UNAVAILABLE" in text
        or "DEADLINE_EXCEEDED" in text
        or "RATE_LIMIT_EXCEEDED" in text
        or "RESOURCE_EXHAUSTED" in text
        or "INTERNAL" in text
        or "429" in text
        or "OVERLOADED" in text
    )


def _model_uses_thinking(model_name):

    return "pro" in (model_name or "").lower()


_MEDIA_RESOLUTION_MAP = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


def _resolve_media_resolution(name):

    key = (name or DEFAULT_MEDIA_RESOLUTION).strip().lower()
    return _MEDIA_RESOLUTION_MAP.get(
        key, types.MediaResolution.MEDIA_RESOLUTION_HIGH
    )


def _build_video_part(video_file, fps):
    """Wrap the uploaded file in a Part that forces a custom sampling FPS.

    Without video_metadata Gemini samples only 1 frame/second, which is too
    sparse to count reps or detect form errors.
    """

    return types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
        ),
        video_metadata=types.VideoMetadata(fps=fps),
    )


def _build_config(media_resolution, use_thinking=False):

    kwargs = {
        "temperature": 0.0,
        "media_resolution": _resolve_media_resolution(media_resolution),
    }
    if use_thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=-1,
        )
    return types.GenerateContentConfig(**kwargs)



# -----------------------------
# GEMINI VIDEO ANALYSIS
# -----------------------------


def run_rule_based_analysis(video_path, side="left", out_video_path=None):
    """Run MediaPipe pose analysis locally and return structured JSON."""
    return rule_based.analyze(video_path, out_video_path=out_video_path, side=side)


def analyze_video(
    video_file,
    customer_query=None,
    fps=DEFAULT_FPS,
    media_resolution=DEFAULT_MEDIA_RESOLUTION,
    max_retries_per_model=5,
    rule_based_data=None,
):


    prompt = build_prompt(
        customer_query,
        rule_based_data=rule_based_data,
    )

    video_part = _build_video_part(video_file, fps)

    last_error = None

    for model_name in _build_model_candidates():
        config = _build_config(
            media_resolution,
            use_thinking=_model_uses_thinking(model_name),
        )
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        video_part,
                        prompt,
                    ],
                    config=config,
                )
                print(f"Gemini response received from {model_name}")
                return response.text
            except Exception as exc:
                last_error = exc
                print(f"Gemini error ({model_name}, attempt {attempt}): {exc}")
                if not _is_retryable_error(exc):
                    raise

                if attempt < max_retries_per_model:
                    wait_seconds = min(2 ** attempt, 30)
                    print(
                        f"Retrying {model_name} in {wait_seconds}s "
                        f"(attempt {attempt}/{max_retries_per_model})..."
                    )
                    time.sleep(wait_seconds)
                else:
                    print(
                        f"Model {model_name} unavailable after "
                        f"{max_retries_per_model} attempts. "
                        "Trying next fallback model..."
                    )

    detail = str(last_error) if last_error else "unknown error"
    raise RuntimeError(
        "Gemini service is temporarily unavailable across fallback models. "
        f"Last error: {detail}. Please retry in a minute."
    ) from last_error


def analyze_text_fallback(
    customer_query=None,
    media_resolution=DEFAULT_MEDIA_RESOLUTION,
    max_retries_per_model=3,
    rule_based_data=None,
):
    """Last-resort coaching report from rule-based JSON when video API is down."""
    prompt = build_prompt(customer_query, rule_based_data=rule_based_data)
    prompt += """

    IMPORTANT:
    Video analysis is temporarily unavailable.
    Base this entire report on the rule-based JSON above.
    Do not claim to have watched the video frames.
    Still write the full report in plain, friendly coaching English.
    """

    last_error = None
    for model_name in _build_model_candidates():
        config = _build_config(media_resolution, use_thinking=False)
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=config,
                )
                print(f"Gemini text fallback received from {model_name}")
                return response.text
            except Exception as exc:
                last_error = exc
                print(f"Gemini text fallback error ({model_name}): {exc}")
                if not _is_retryable_error(exc):
                    raise
                if attempt < max_retries_per_model:
                    time.sleep(min(2 ** attempt, 15))

    raise RuntimeError(
        "Gemini text fallback also failed. "
        f"Last error: {last_error}"
    ) from last_error


def overlay_mistakes_video(video_path, mistakes, out_video_path=None):
    """Write a copy of the video with timestamped mistake labels burned in."""
    if out_video_path is None:
        stem = Path(video_path).stem
        out_video_path = str(
            Path(video_path).with_name(f"{stem}_mistakes_overlay.mp4")
        )
    rule_based.overlay_mistakes_on_video(video_path, mistakes, out_video_path)
    return out_video_path


def analyze_video_pipeline(
    video_path,
    customer_query=None,
    fps=DEFAULT_FPS,
    media_resolution=DEFAULT_MEDIA_RESOLUTION,
    side="left",
    out_video_path=None,
    skeleton_video_path=None,
):
    """Rule-based analysis first, then Gemini coaching on video + JSON."""
    pipeline_start = time.perf_counter()

    if skeleton_video_path is None:
        stem = Path(video_path).stem
        skeleton_video_path = str(
            Path(video_path).with_name(f"{stem}_skeleton.mp4")
        )

    print("Running rule-based pose analysis...")
    rule_start = time.perf_counter()
    rule_result = run_rule_based_analysis(
        video_path,
        side=side,
        out_video_path=skeleton_video_path,
    )
    rule_based_sec = round(time.perf_counter() - rule_start, 2)
    print(f"Skeleton overlay video saved to: {skeleton_video_path}")

    print(
        f"Rule-based: {rule_result.get('total_reps', 0)} reps, "
        f"{rule_result.get('accuracy_percent', 0)}% accuracy "
        f"({rule_based_sec}s)"
    )
    print("\n----- RULE-BASED JSON -----\n")
    print(json.dumps(rule_result, indent=2))

    mistakes = rule_result.get("timestamped_mistakes") or []
    if mistakes:
        print("\n----- TIMESTAMPED MISTAKES -----\n")
        print(rule_based.format_mistakes_report(rule_result))
    else:
        print("No form mistakes detected by rule-based analysis.")

    gemini_start = time.perf_counter()
    gemini_report = None
    gemini_error = None
    gemini_fallback_used = False
    gemini_upload_sec = 0.0
    gemini_response_sec = 0.0

    try:
        upload_start = time.perf_counter()
        video_file = upload_video(video_path)
        gemini_upload_sec = round(time.perf_counter() - upload_start, 2)

        response_start = time.perf_counter()
        gemini_report = analyze_video(
            video_file,
            customer_query=customer_query,
            fps=fps,
            media_resolution=media_resolution,
            rule_based_data=rule_result,
        )
        gemini_response_sec = round(time.perf_counter() - response_start, 2)
    except Exception as exc:
        gemini_error = str(exc)
        print(f"\nGemini video analysis failed — trying text-only fallback.\n{exc}")
        try:
            fallback_start = time.perf_counter()
            gemini_report = analyze_text_fallback(
                customer_query=customer_query,
                media_resolution=media_resolution,
                rule_based_data=rule_result,
            )
            gemini_response_sec = round(time.perf_counter() - fallback_start, 2)
            gemini_error = None
            gemini_fallback_used = True
            print("Gemini text-only fallback succeeded.")
        except Exception as fallback_exc:
            gemini_error = (
                f"{gemini_error}\n\nText-only fallback also failed: {fallback_exc}"
            )
            print(f"\nGemini analysis failed — continuing with rule-based results.\n{gemini_error}")

    gemini_total_sec = round(time.perf_counter() - gemini_start, 2)

    overlay_start = time.perf_counter()
    overlay_source = (
        skeleton_video_path
        if skeleton_video_path and Path(skeleton_video_path).is_file()
        else video_path
    )
    annotated_video_path = overlay_mistakes_video(
        overlay_source,
        mistakes,
        out_video_path=out_video_path,
    )
    overlay_sec = round(time.perf_counter() - overlay_start, 2)
    total_sec = round(time.perf_counter() - pipeline_start, 2)

    timing = {
        "rule_based_sec": rule_based_sec,
        "gemini_upload_sec": gemini_upload_sec,
        "gemini_response_sec": gemini_response_sec,
        "gemini_total_sec": gemini_total_sec,
        "overlay_sec": overlay_sec,
        "total_sec": total_sec,
    }

    print(
        f"\nTiming — rule-based: {rule_based_sec}s | "
        f"Gemini upload: {gemini_upload_sec}s | "
        f"Gemini response: {gemini_response_sec}s | "
        f"overlay: {overlay_sec}s | "
        f"total: {total_sec}s"
    )
    print(f"\nAnnotated video saved to: {annotated_video_path}")

    return {
        "rule_based": rule_result,
        "gemini_report": gemini_report,
        "gemini_error": gemini_error,
        "gemini_fallback_used": gemini_fallback_used,
        "skeleton_video_path": skeleton_video_path,
        "annotated_video_path": annotated_video_path,
        "timing": timing,
    }


# -----------------------------
# MAIN
# -----------------------------


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Gemini fitness video analysis with optional customer query"
    )

    parser.add_argument(
        "--video",
        default=(
            r"D:\AIDS\cv\fitness_assist"
            r"\inputs\squat\VID_20260316_132404.mp4"
        ),
        help="Path to the exercise video",
    )

    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Optional customer question to include in the analysis prompt",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=(
            "Frames per second to sample from the video (default 5). "
            "Higher = more detail / better rep counting but slower & more "
            "tokens. Try 8-10 for fast movements."
        ),
    )

    parser.add_argument(
        "--media-resolution",
        choices=["low", "medium", "high"],
        default=DEFAULT_MEDIA_RESOLUTION,
        help="Per-frame detail sent to Gemini (default high).",
    )

    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="Which body side faces the camera for rule-based pose analysis.",
    )

    parser.add_argument(
        "--out-video",
        default=None,
        help="Path to save the annotated video with mistake overlays.",
    )

    parser.add_argument(
        "--skeleton-video",
        default=None,
        help="Path to save the MediaPipe skeleton overlay video after rule-based analysis.",
    )

    args = parser.parse_args()


    if args.query and args.query.strip():
        print(
            f"\nCustomer query: {args.query.strip()}\n"
        )

    print(
        f"\nAnalyzing at {args.fps} fps, "
        f"{args.media_resolution} resolution...\n"
    )


    pipeline_result = analyze_video_pipeline(
        args.video,
        customer_query=args.query,
        fps=args.fps,
        media_resolution=args.media_resolution,
        side=args.side,
        out_video_path=args.out_video,
        skeleton_video_path=args.skeleton_video,
    )


    print(
        "\n----- RULE-BASED ANALYSIS -----\n"
    )
    print(json.dumps(pipeline_result["rule_based"], indent=2))

    print(
        "\n----- GEMINI FITNESS REPORT -----\n"
    )
    print(pipeline_result["gemini_report"])

    timing = pipeline_result.get("timing", {})
    if timing:
        print(
            "\n----- TIMING -----\n"
            f"Rule-based analysis: {timing.get('rule_based_sec', 0)}s\n"
            f"Gemini upload:        {timing.get('gemini_upload_sec', 0)}s\n"
            f"Gemini response:      {timing.get('gemini_response_sec', 0)}s\n"
            f"Video overlay:        {timing.get('overlay_sec', 0)}s\n"
            f"Gemini total:         {timing.get('gemini_total_sec', 0)}s\n"
            f"Pipeline total:       {timing.get('total_sec', 0)}s"
        )

    skeleton = pipeline_result.get("skeleton_video_path")
    if skeleton:
        print(f"\n----- SKELETON VIDEO -----\n{skeleton}")

    annotated = pipeline_result.get("annotated_video_path")
    if annotated:
        print(f"\n----- ANNOTATED VIDEO -----\n{annotated}")