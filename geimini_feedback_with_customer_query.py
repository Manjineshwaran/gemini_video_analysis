import argparse
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types


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
]



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

You are a professional AI fitness trainer.

Analyze this complete exercise video.

Important:
- Understand the full movement sequence.
- Do not analyze single frames separately.
- Track the person's movement from start to end.
- Count only completed repetitions.
- Ignore partial movements.

Return:

1. Exercise name

2. Total completed repetitions

3. Rep-by-rep analysis

4. Posture correctness

5. Mistakes detected

6. Improvement suggestions


Example:

Exercise:
Squat

Reps:
12

Form:
Good

Problems:
Knees moving inward in some reps

Suggestions:
Keep knees aligned with toes.


"""


CUSTOMER_QUERY_PROMPT = """

The customer has asked the following question:

"{customer_query}"

Important:
- Still provide the full exercise analysis described above.
- You MUST directly answer the customer's question using what you observe in the video.
- Base your answer on the video evidence; do not guess.
- If the question is about form, reps, depth, speed, or technique, focus your analysis on that area.
- Add a dedicated section titled "Customer question answer" with a clear, direct response.


"""


def build_prompt(customer_query=None):

    query = (customer_query or "").strip()

    if not query:
        return BASE_PROMPT

    return (
        BASE_PROMPT
        + CUSTOMER_QUERY_PROMPT.format(
            customer_query=query
        )
    )


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
        or "UNAVAILABLE" in text
        or "DEADLINE_EXCEEDED" in text
        or "RATE_LIMIT_EXCEEDED" in text
        or "429" in text
    )



# -----------------------------
# GEMINI VIDEO ANALYSIS
# -----------------------------


def analyze_video(video_file, customer_query=None, max_retries_per_model=3):


    prompt = build_prompt(
        customer_query
    )

    last_error = None

    for model_name in _build_model_candidates():
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        video_file,
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2
                    )
                )
                return response.text
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    raise

                if attempt < max_retries_per_model:
                    wait_seconds = min(2 ** attempt, 10)
                    print(
                        f"Retrying {model_name} in {wait_seconds}s "
                        f"(attempt {attempt}/{max_retries_per_model})..."
                    )
                    time.sleep(wait_seconds)
                else:
                    print(
                        f"Model {model_name} is busy. "
                        "Trying next fallback model..."
                    )

    raise RuntimeError(
        "Gemini service is temporarily unavailable across fallback models. "
        "Please retry in a minute."
    ) from last_error





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

    args = parser.parse_args()


    video = upload_video(
        args.video
    )


    if args.query and args.query.strip():
        print(
            f"\nCustomer query: {args.query.strip()}\n"
        )

    print("\nAnalyzing...\n")


    result = analyze_video(
        video,
        customer_query=args.query,
    )


    print(
        "\n----- GEMINI FITNESS REPORT -----\n"
    )


    print(result)