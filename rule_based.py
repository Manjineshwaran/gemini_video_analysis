"""
Side-View Pushup Analyzer — full breakdown
--------------------------------------------
Everything here is computed from real keypoint geometry & timing — nothing is
guessed by an LLM. Items that are NOT physically measurable from a single
side-view 2D camera (left/right imbalance, hand width, elbow flare, shoulder
rotation, core twisting) are explicitly listed under "not_measurable" instead
of being faked. See the message accompanying this file for why.

Usage:
    python side_pushup_analyzer.py --video 1.mp4 --out_video annotated.mp4
"""

import asyncio
import concurrent.futures
import cv2
import json
import math
import argparse
import os
import shutil
import subprocess
import tempfile
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
LM = mp_pose.PoseLandmark

FFMPEG_BIN_DIRS = [
    os.getenv("FFMPEG_BIN", "").strip(),
    r"C:\ffmpeg\bin",   # local Windows install
    "/usr/bin",         # Streamlit Cloud / Debian after packages.txt
]


def _ensure_ffmpeg_on_path():
    """Expose ffmpeg/ffprobe to pydub and subprocess (venv PATH often lacks it)."""
    for bin_dir in FFMPEG_BIN_DIRS:
        if not bin_dir or not os.path.isdir(bin_dir):
            continue
        path = os.environ.get("PATH", "")
        if bin_dir.lower() not in path.lower():
            os.environ["PATH"] = bin_dir + os.pathsep + path
        return bin_dir
    return None


_ensure_ffmpeg_on_path()

# ---------------------------------------------------------------------
# Tunable thresholds — these are reasonable starting points, NOT measured
# from your specific camera/body. Calibrate against a few known-good and
# known-bad reps from your own footage and adjust.
# ---------------------------------------------------------------------
THRESH = {
    "top_angle": 160,          # elbow angle considered "fully locked out"
    "bottom_angle": 90,        # elbow angle considered "full depth"
    "lockout_min": 155,        # below this at top = incomplete lockout
    "excessive_depth_max": 50, # below this at bottom = too deep / shoulder strain risk
    "hip_sag_angle": 160,      # shoulder-hip-ankle angle below this = sagging
    "hip_high_angle": 195,     # above this = hips piked too high (rare but real)
    "head_drop_angle": 140,    # nose-shoulder-hip angle below this = head dropping
    "wrist_offset_ratio": 0.15,# |wrist_x - shoulder_x| / torso_length beyond this = wrist not under shoulder
    "min_error_fraction": 0.35,# fraction of rep's frames an issue must persist in to count (anti-jitter)
    "fast_descent_sec": 0.4,   # descent faster than this = "too fast"
    "slow_descent_sec": 2.5,   # descent slower than this = "very controlled / slow"
    "pause_velocity_deg_s": 15,# |angular velocity| below this = considered a pause
    "smoothing_window": 3,     # moving-average window (frames) to denoise angle signal
}

NOT_MEASURABLE_FROM_SIDE = [
    "left_right_imbalance — only one side of the body is visible; the far side is occluded",
    "hand_width / stance_width — needs front or top-down view to see lateral spread",
    "elbow_flare (outward) — lateral/depth movement invisible to a side camera",
    "shoulder_rotation — needs front view",
    "core_twisting — needs front or top-down view",
    "hips_rotating_backward (transverse plane) — needs top-down view",
]

ERROR_LABELS = {
    "incomplete_depth": (
        "You did not go low enough. Bend your elbows more and bring your chest closer to the floor."
    ),
    "incomplete_lockout": (
        "You did not fully straighten your arms at the top. Push all the way up and lock your elbows."
    ),
    "excessive_depth": (
        "You went too deep. Your elbows bent farther than they should — stop a little higher to protect your shoulders."
    ),
    "hip_sag": (
        "Your hips are sagging. Keep your body in one straight line — head, hips, "
        "and ankles parallel — and brace your core."
    ),
    "hip_too_high": (
        "Your hips are too high. Lower them so your body stays in one straight line, not a pike."
    ),
    "head_drop": (
        "Your head is dropping. Keep your neck in line with your spine and look slightly ahead."
    ),
    "wrist_not_under_shoulder": (
        "Your hands are not under your shoulders. Stack your wrists directly below your shoulders."
    ),
    "descent_too_fast": (
        "You are dropping too fast on the way down. Slow down and control the descent."
    ),
    "descent_very_slow": (
        "You are moving very slowly on the way down. Try a steadier, controlled tempo."
    ),
}


def format_timestamp(sec):
    """Format seconds as M:SS.ss (e.g. 0:12.45, 1:03.20)."""
    sec = max(0.0, float(sec))
    minutes = int(sec // 60)
    seconds = sec % 60
    if minutes:
        return f"{minutes}:{seconds:05.2f}"
    return f"0:{seconds:05.2f}"


def _mistake_window(seg_t, seg_frames, key, condition, worst_idx_fn):
    """Return start/end/worst timestamps for frames matching a condition."""
    indices = [i for i, f in enumerate(seg_frames) if condition(f[key])]
    if not indices:
        return None
    worst_i = worst_idx_fn(indices, seg_frames)
    return {
        "timestamp_sec": round(float(seg_t[worst_i]), 2),
        "start_time_sec": round(float(seg_t[indices[0]]), 2),
        "end_time_sec": round(float(seg_t[indices[-1]]), 2),
    }


def build_rep_mistakes(rep_n, seg_t, seg_frames, min_idx_local, errors):
    """Attach exact timestamps to each detected mistake in a rep."""
    if not errors:
        return []

    bottom_t = float(seg_t[min_idx_local])
    top_t = float(seg_t[-1])
    descent_mid_t = (float(seg_t[0]) + bottom_t) / 2

    mistake_builders = {
        "incomplete_depth": lambda: {
            "timestamp_sec": round(bottom_t, 2),
            "phase": "bottom",
        },
        "excessive_depth": lambda: {
            "timestamp_sec": round(bottom_t, 2),
            "phase": "bottom",
        },
        "incomplete_lockout": lambda: {
            "timestamp_sec": round(top_t, 2),
            "phase": "top",
        },
        "descent_too_fast": lambda: {
            "timestamp_sec": round(descent_mid_t, 2),
            "phase": "descent",
        },
        "descent_very_slow": lambda: {
            "timestamp_sec": round(descent_mid_t, 2),
            "phase": "descent",
        },
        "hip_sag": lambda: _mistake_window(
            seg_t, seg_frames, "body_line_angle",
            lambda v: v < THRESH["hip_sag_angle"],
            lambda idxs, frames: min(
                idxs, key=lambda i: frames[i]["body_line_angle"]
            ),
        ),
        "hip_too_high": lambda: _mistake_window(
            seg_t, seg_frames, "body_line_angle",
            lambda v: v > THRESH["hip_high_angle"],
            lambda idxs, frames: max(
                idxs, key=lambda i: frames[i]["body_line_angle"]
            ),
        ),
        "head_drop": lambda: _mistake_window(
            seg_t, seg_frames, "neck_angle",
            lambda v: v < THRESH["head_drop_angle"],
            lambda idxs, frames: min(
                idxs, key=lambda i: frames[i]["neck_angle"]
            ),
        ),
        "wrist_not_under_shoulder": lambda: _mistake_window(
            seg_t, seg_frames, "wrist_offset_ratio",
            lambda v: v > THRESH["wrist_offset_ratio"],
            lambda idxs, frames: max(
                idxs, key=lambda i: frames[i]["wrist_offset_ratio"]
            ),
        ),
    }

    mistakes = []
    for err in errors:
        builder = mistake_builders.get(err)
        if not builder:
            continue
        timing = builder()
        if not timing:
            continue
        entry = {
            "error": err,
            "label": ERROR_LABELS.get(err, err.replace("_", " ")),
            "rep_number": rep_n,
            "timestamp_sec": timing["timestamp_sec"],
            "timestamp": format_timestamp(timing["timestamp_sec"]),
            "phase": timing.get("phase"),
        }
        entry["coaching_note"] = (
            f"Rep {rep_n} at {entry['timestamp']}: {entry['label']}"
        )
        entry["coaching_note"] = (
            f"Rep {rep_n} at {entry['timestamp']}: {entry['label']}"
        )
        if "start_time_sec" in timing:
            entry["start_time_sec"] = timing["start_time_sec"]
            entry["end_time_sec"] = timing["end_time_sec"]
            entry["start_time"] = format_timestamp(timing["start_time_sec"])
            entry["end_time"] = format_timestamp(timing["end_time_sec"])
            if timing["start_time_sec"] != timing["end_time_sec"]:
                entry["duration_sec"] = round(
                    timing["end_time_sec"] - timing["start_time_sec"], 2
                )
        mistakes.append(entry)

    return mistakes


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def xy(lms, idx, w, h):
    lm = lms[idx]
    return (lm.x * w, lm.y * h)


def smooth(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ---------------------------------------------------------------------
# 1. Extract per-frame signals
# ---------------------------------------------------------------------

def extract_frames(video_path, out_video_path=None, side="left"):
    """side: which body side is facing/visible to the camera ('left' or 'right')."""
    S = {
        "shoulder": LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER,
        "elbow": LM.LEFT_ELBOW if side == "left" else LM.RIGHT_ELBOW,
        "wrist": LM.LEFT_WRIST if side == "left" else LM.RIGHT_WRIST,
        "hip": LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP,
        "knee": LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE,
        "ankle": LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE,
    }

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video_path:
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frames = []  # list of dicts, one per detected frame
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            if writer:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            shoulder = xy(lms, S["shoulder"], w, h)
            elbow = xy(lms, S["elbow"], w, h)
            wrist = xy(lms, S["wrist"], w, h)
            hip = xy(lms, S["hip"], w, h)
            knee = xy(lms, S["knee"], w, h)
            ankle = xy(lms, S["ankle"], w, h)
            nose = xy(lms, LM.NOSE, w, h)

            elbow_angle = calc_angle(shoulder, elbow, wrist)
            body_line_angle = calc_angle(shoulder, hip, ankle)
            neck_angle = calc_angle(nose, shoulder, hip)
            knee_angle = calc_angle(hip, knee, ankle)  # plank leg straightness, side-measurable

            torso_len = math.dist(shoulder, hip) + 1e-6
            wrist_offset_ratio = abs(wrist[0] - shoulder[0]) / torso_len

            frames.append({
                "t": idx / fps,
                "elbow_angle": elbow_angle,
                "body_line_angle": body_line_angle,
                "neck_angle": neck_angle,
                "knee_angle": knee_angle,
                "wrist_offset_ratio": wrist_offset_ratio,
            })

        if writer:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer:
        writer.release()
    pose.close()
    return frames, fps


# ---------------------------------------------------------------------
# 2. Rep segmentation with phase detection (descent/bottom-pause/ascent/top-pause)
# ---------------------------------------------------------------------

def segment_reps(frames):
    if not frames:
        return []

    t = np.array([f["t"] for f in frames])
    raw_angle = np.array([f["elbow_angle"] for f in frames])
    angle = smooth(raw_angle, THRESH["smoothing_window"])

    top, bottom = THRESH["top_angle"], THRESH["bottom_angle"]
    mid = (top + bottom) / 2

    state = "top"
    reps_idx = []
    start_i = 0
    for i, a in enumerate(angle):
        if state == "top" and a < mid:
            state = "going_down"
        elif state == "going_down" and a < bottom:
            state = "bottom"
        elif state == "bottom" and a > mid:
            state = "going_up"
        elif state == "going_up" and a > top:
            reps_idx.append((start_i, i))
            start_i = i
            state = "top"

    reps = []
    for rep_n, (s, e) in enumerate(reps_idx, start=1):
        seg_t = t[s:e + 1]
        seg_angle = angle[s:e + 1]
        seg_frames = frames[s:e + 1]

        min_idx_local = int(np.argmin(seg_angle))
        bottom_t = seg_t[min_idx_local]

        descent_sec = round(float(bottom_t - seg_t[0]), 2)
        ascent_sec = round(float(seg_t[-1] - bottom_t), 2)

        # angular velocity (deg/sec) for pause + speed analysis
        if len(seg_t) > 1:
            velocity = np.gradient(seg_angle, seg_t)
        else:
            velocity = np.array([0.0])
        pause_mask = np.abs(velocity) < THRESH["pause_velocity_deg_s"]
        # only count pauses near the bottom (mid-rep), not the natural near-zero velocity at start/end
        window = max(1, len(seg_t) // 6)
        bottom_zone = slice(max(0, min_idx_local - window), min(len(seg_t), min_idx_local + window))
        pause_sec = round(float(np.sum(pause_mask[bottom_zone]) * np.mean(np.diff(seg_t))) if len(seg_t) > 1 else 0.0, 2)

        min_angle = float(np.min(seg_angle))
        max_angle = float(np.max(seg_angle))

        def persistent(key, condition, min_fraction=THRESH["min_error_fraction"]):
            flags = [condition(f[key]) for f in seg_frames]
            return (sum(flags) / len(flags)) >= min_fraction if flags else False

        hip_sag = persistent("body_line_angle", lambda v: v < THRESH["hip_sag_angle"])
        hip_high = persistent("body_line_angle", lambda v: v > THRESH["hip_high_angle"])
        head_drop = persistent("neck_angle", lambda v: v < THRESH["head_drop_angle"])
        wrist_misaligned = persistent("wrist_offset_ratio", lambda v: v > THRESH["wrist_offset_ratio"])

        depth_achieved = min_angle <= THRESH["bottom_angle"] + 5
        full_lockout = max_angle >= THRESH["lockout_min"]
        excessive_depth = min_angle < THRESH["excessive_depth_max"]

        errors = []
        if not depth_achieved:
            errors.append("incomplete_depth")
        if not full_lockout:
            errors.append("incomplete_lockout")
        if excessive_depth:
            errors.append("excessive_depth")
        if hip_sag:
            errors.append("hip_sag")
        if hip_high:
            errors.append("hip_too_high")
        if head_drop:
            errors.append("head_drop")
        if wrist_misaligned:
            errors.append("wrist_not_under_shoulder")
        if descent_sec < THRESH["fast_descent_sec"]:
            errors.append("descent_too_fast")
        if descent_sec > THRESH["slow_descent_sec"]:
            errors.append("descent_very_slow")  # informational, not necessarily bad

        is_complete_rep = depth_achieved and full_lockout
        is_good_rep = is_complete_rep and not any(
            e in errors for e in ["hip_sag", "hip_too_high", "head_drop", "wrist_not_under_shoulder",
                                   "excessive_depth", "descent_too_fast"]
        )

        mistakes = build_rep_mistakes(rep_n, seg_t, seg_frames, min_idx_local, errors)

        reps.append({
            "rep_number": rep_n,
            "start_time": round(float(seg_t[0]), 2),
            "end_time": round(float(seg_t[-1]), 2),
            "start_timestamp": format_timestamp(seg_t[0]),
            "end_timestamp": format_timestamp(seg_t[-1]),
            "bottom_timestamp": format_timestamp(bottom_t),
            "rep_duration_sec": round(float(seg_t[-1] - seg_t[0]), 2),
            "phases": {
                "descent_sec": descent_sec,
                "ascent_sec": ascent_sec,
                "bottom_pause_sec": pause_sec,
            },
            "range_of_motion": {
                "min_elbow_angle": round(min_angle, 1),
                "max_elbow_angle": round(max_angle, 1),
                "depth_achieved": depth_achieved,
                "full_lockout": full_lockout,
            },
            "completeness": "full" if is_complete_rep else "half/incomplete",
            "quality": "good" if is_good_rep else "needs_work",
            "errors": errors,
            "mistakes": mistakes,
        })

    return reps


# ---------------------------------------------------------------------
# 3. Aggregate performance metrics + fatigue detection
# ---------------------------------------------------------------------

def build_summary(reps, total_duration):
    if not reps:
        return {
            "exercise": "pushup", "camera_view": "side",
            "total_reps": 0, "note": "No reps detected / no pose found.",
            "not_measurable": NOT_MEASURABLE_FROM_SIDE,
        }

    total = len(reps)
    good = sum(1 for r in reps if r["quality"] == "good")
    bad = total - good
    accuracy_pct = round(100 * good / total, 1)

    durations = [r["rep_duration_sec"] for r in reps]
    avg_duration = round(float(np.mean(durations)), 2)

    best_rep = min(
        reps,
        key=lambda r: (len(r["errors"]), r["range_of_motion"]["min_elbow_angle"])
    )

    # fatigue: compare error rate / depth of first half vs second half of reps
    half = max(1, total // 2)
    first_half, second_half = reps[:half], reps[half:]
    fatigue_detected = False
    fatigue_note = ""
    if second_half:
        first_err_rate = np.mean([len(r["errors"]) for r in first_half])
        second_err_rate = np.mean([len(r["errors"]) for r in second_half])
        first_depth = np.mean([r["range_of_motion"]["min_elbow_angle"] for r in first_half])
        second_depth = np.mean([r["range_of_motion"]["min_elbow_angle"] for r in second_half])
        if second_err_rate > first_err_rate * 1.3 or second_depth > first_depth + 8:
            fatigue_detected = True
            fatigue_note = (
                f"Form degraded in later reps (avg errors/rep {first_err_rate:.1f} -> "
                f"{second_err_rate:.1f}, depth angle {first_depth:.0f}° -> {second_depth:.0f}°)."
            )

    error_counts = {}
    timestamped_mistakes = []
    for r in reps:
        for e in r["errors"]:
            error_counts[e] = error_counts.get(e, 0) + 1
        timestamped_mistakes.extend(r.get("mistakes", []))
    timestamped_mistakes.sort(key=lambda m: m["timestamp_sec"])

    return {
        "exercise": "pushup",
        "camera_view": "side",
        "total_reps": total,
        "complete_reps": sum(1 for r in reps if r["completeness"] == "full"),
        "incomplete_reps": sum(1 for r in reps if r["completeness"] != "full"),
        "correct_reps": good,
        "bad_reps": bad,
        "accuracy_percent": accuracy_pct,
        "duration_sec": round(total_duration, 2),
        "avg_rep_duration_sec": avg_duration,
        "best_rep_number": best_rep["rep_number"],
        "fatigue_detected": fatigue_detected,
        "fatigue_note": fatigue_note,
        "error_counts": error_counts,
        "timestamped_mistakes": timestamped_mistakes,
        "reps": reps,
        "not_measurable": NOT_MEASURABLE_FROM_SIDE,
    }


def analyze(video_path, out_video_path=None, side="left"):
    frames, fps = extract_frames(video_path, out_video_path, side=side)
    reps = segment_reps(frames)
    total_duration = frames[-1]["t"] if frames else 0.0
    return build_summary(reps, total_duration)


def _active_mistakes_at_time(mistakes, t_sec, point_window=2.5):
    """Return mistakes visible at video time t_sec."""
    active = []
    for m in mistakes:
        start = m.get("start_time_sec")
        end = m.get("end_time_sec")
        if start is not None and end is not None and start != end:
            if start <= t_sec <= end:
                active.append(m)
            continue
        ts = float(m.get("timestamp_sec", 0))
        if ts - 0.5 <= t_sec <= ts + point_window:
            active.append(m)
    return active


def _wrap_text(text, max_chars=42):
    lines = []
    words = text.split()
    current = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def _sanitize_overlay_text(text):
    """Make text safe for OpenCV putText (ASCII-friendly)."""
    replacements = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _mistake_coaching_lines(mistake, max_chars):
    """Build overlay lines for one friendly coaching note."""
    label = mistake.get("label") or mistake.get("error", "Form issue")
    rep = mistake.get("rep_number", "?")
    ts = mistake.get("timestamp", "")
    label = _sanitize_overlay_text(str(label))

    lines = [_sanitize_overlay_text(f"Rep {rep} at {ts}:")]
    lines.extend(_wrap_text(label, max_chars))
    return lines


TTS_VOICE = "en-US-GuyNeural"
TTS_PAUSE_PADDING_SEC = 0.25


def _resolve_ffmpeg_tool(tool_name):
    """Return full path to ffmpeg/ffprobe, preferring a local install."""
    exe_name = f"{tool_name}.exe" if os.name == "nt" else tool_name

    for bin_dir in FFMPEG_BIN_DIRS:
        if not bin_dir:
            continue
        candidate = os.path.join(bin_dir, exe_name)
        if os.path.isfile(candidate):
            return candidate

    found = shutil.which(tool_name)
    if found:
        return found

    if tool_name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    raise RuntimeError(
        f"{tool_name} is required for voice overlays. "
        "Local Windows: set FFMPEG_BIN=C:\\ffmpeg\\bin. "
        "Streamlit Cloud: add ffmpeg to packages.txt in the repo root."
    )


def _configure_pydub_ffmpeg():
    _ensure_ffmpeg_on_path()
    from pydub import AudioSegment

    ffmpeg = _resolve_ffmpeg_tool("ffmpeg")
    AudioSegment.converter = ffmpeg
    try:
        AudioSegment.ffprobe = _resolve_ffmpeg_tool("ffprobe")
    except RuntimeError:
        pass
    return AudioSegment


def _get_ffmpeg_exe():
    return _resolve_ffmpeg_tool("ffmpeg")


def _mistake_speech_text(mistake):
    """Plain text read aloud during a mistake pause."""
    label = mistake.get("label") or mistake.get("error", "Form issue")
    rep = mistake.get("rep_number", "?")
    ts = mistake.get("timestamp", "")
    label = _sanitize_overlay_text(str(label))
    return f"Rep {rep} at {ts}. {label}"


async def _tts_to_file_async(text, out_path):
    import edge_tts
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(out_path)


def _generate_tts_audio(text, out_path):
    def _run():
        asyncio.run(_tts_to_file_async(text, out_path))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _run()
        return

    # Streamlit already runs an event loop — asyncio.run() would fail silently
    # inside our except block and produce a voice-less video.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_run).result()


def _load_source_audio(video_path):
    AudioSegment = _configure_pydub_ffmpeg()
    try:
        return AudioSegment.from_file(video_path)
    except Exception:
        return None


def _audio_duration_sec(audio_path):
    AudioSegment = _configure_pydub_ffmpeg()
    clip = AudioSegment.from_file(audio_path)
    return len(clip) / 1000.0


def _estimate_speech_duration_sec(text):
    words = max(1, len(text.split()))
    return max(2.0, words / 2.5 + 0.5)


def _prepare_pause_audio(mistake, tmp_dir, pause_idx):
    """Return (audio_path, duration_sec) for one mistake pause."""
    speech = _mistake_speech_text(mistake)
    audio_path = os.path.join(tmp_dir, f"pause_{pause_idx:03d}.mp3")
    try:
        _generate_tts_audio(speech, audio_path)
        if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 128:
            raise RuntimeError("TTS produced an empty audio file")
        duration = _audio_duration_sec(audio_path) + TTS_PAUSE_PADDING_SEC
        print(f"    Voice ready ({duration:.1f}s): {speech[:70]}...")
        return audio_path, duration
    except Exception as exc:
        print(f"    Voice generation failed: {exc}")
        duration = _estimate_speech_duration_sec(speech)
        return None, duration


def _overlay_style(w, h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 1000)
    line_height = int(24 * font_scale / 0.5)
    pad = 12
    max_chars = max(32, int(w / 22))
    return font, font_scale, line_height, pad, max_chars


def _draw_mistake_overlay(frame, mistake, w, h):
    """Draw one coaching note on a frozen frame."""
    font, font_scale, line_height, pad, max_chars = _overlay_style(w, h)
    annotated = frame.copy()

    lines = ["Coaching note"]
    lines.extend(_mistake_coaching_lines(mistake, max_chars))

    box_h = pad * 2 + line_height * len(lines)
    box_top = max(0, h - box_h)
    tint = annotated.copy()
    cv2.rectangle(tint, (0, box_top), (w, h), (16, 16, 16), -1)
    cv2.addWeighted(tint, 0.78, annotated, 0.22, 0, annotated)

    y = box_top + pad + line_height
    for line in lines:
        if not line:
            y += line_height // 2
            continue
        if line == "Coaching note":
            color = (120, 220, 120)
            thickness = 2
        elif line.startswith("Rep "):
            color = (100, 220, 255)
            thickness = 1
        else:
            color = (240, 240, 240)
            thickness = 1
        cv2.putText(
            annotated,
            line,
            (pad, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return annotated


def _read_frame_at(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame


def _build_overlay_timeline(mistakes, fps, total_frames):
    """Play until each mistake second, pause there, then resume."""
    mistakes_sorted = sorted(
        mistakes,
        key=lambda m: float(m.get("timestamp_sec", 0)),
    )
    segments = []
    cursor = 0

    for mistake in mistakes_sorted:
        ts = float(mistake.get("timestamp_sec", 0))
        frame_idx = min(max(0, int(round(ts * fps))), max(0, total_frames - 1))

        if frame_idx >= cursor:
            if frame_idx > cursor:
                segments.append({
                    "type": "play",
                    "start_frame": cursor,
                    "end_frame": frame_idx,
                })
            segments.append({
                "type": "pause",
                "frame_idx": frame_idx,
                "mistake": mistake,
            })
            cursor = frame_idx + 1

    if cursor < total_frames:
        segments.append({
            "type": "play",
            "start_frame": cursor,
            "end_frame": total_frames,
        })
    return segments


def _build_output_audio(segments, pause_audio, source_audio, fps):
    AudioSegment = _configure_pydub_ffmpeg()

    timeline = AudioSegment.silent(duration=0)
    for segment in segments:
        if segment["type"] == "play":
            start_ms = int(round(segment["start_frame"] / fps * 1000))
            end_ms = int(round(segment["end_frame"] / fps * 1000))
            if source_audio is not None and end_ms > start_ms:
                timeline += source_audio[start_ms:end_ms]
            else:
                duration_ms = max(0, end_ms - start_ms)
                timeline += AudioSegment.silent(duration=duration_ms)
        else:
            audio_path, duration_sec = pause_audio[id(segment["mistake"])]
            duration_ms = int(round(duration_sec * 1000))
            if audio_path and os.path.isfile(audio_path):
                voice = AudioSegment.from_file(audio_path)
                if len(voice) < duration_ms:
                    voice += AudioSegment.silent(duration=duration_ms - len(voice))
                elif len(voice) > duration_ms:
                    voice = voice[:duration_ms]
                timeline += voice
            else:
                timeline += AudioSegment.silent(duration=duration_ms)
    return timeline


def _mux_video_audio(video_path, audio_path, out_video_path):
    ffmpeg = _get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        out_video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to mux coaching audio into video.\n"
            f"{result.stderr or result.stdout}"
        )


def overlay_mistakes_on_video(
    video_path,
    mistakes,
    out_video_path,
    point_window=3.0,
    max_visible=2,
):
    """Pause at each mistake second, show text, read it aloud, then resume."""
    del point_window, max_visible  # kept for backward-compatible call sites

    mistakes = mistakes or []
    if not mistakes:
        shutil.copy2(video_path, out_video_path)
        return out_video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    segments = _build_overlay_timeline(mistakes, fps, total_frames)
    pause_segments = [s for s in segments if s["type"] == "pause"]
    print(
        f"Building annotated video with voice overlays "
        f"({len(mistakes)} mistake(s), {len(pause_segments)} pause(s))..."
    )

    print("  Loading source audio...")
    source_audio = _load_source_audio(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="mistake_overlay_")
    silent_video_path = os.path.join(tmp_dir, "silent.mp4")
    mixed_audio_path = os.path.join(tmp_dir, "voice.mp3")
    writer = cv2.VideoWriter(
        silent_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Cannot write video: {out_video_path}")

    pause_audio = {}
    try:
        for pause_idx, segment in enumerate(pause_segments, start=1):
            mistake = segment["mistake"]
            print(
                f"  Generating voice {pause_idx}/{len(pause_segments)}: "
                f"Rep {mistake.get('rep_number')} at {mistake.get('timestamp')}"
            )
            pause_audio[id(mistake)] = _prepare_pause_audio(
                mistake, tmp_dir, pause_idx
            )

        voice_count = sum(
            1 for path, _ in pause_audio.values() if path and os.path.isfile(path)
        )
        if voice_count == 0:
            raise RuntimeError(
                "No coaching voice audio was generated. "
                "Check internet access for edge-tts and ffmpeg at C:\\ffmpeg\\bin."
            )
        print(f"  Generated {voice_count}/{len(pause_segments)} voice clip(s)")

        for segment in segments:
            if segment["type"] == "play":
                for frame_idx in range(segment["start_frame"], segment["end_frame"]):
                    frame = _read_frame_at(cap, frame_idx)
                    writer.write(frame)
                continue

            frame_idx = segment["frame_idx"]
            mistake = segment["mistake"]
            frame = _read_frame_at(cap, frame_idx)
            annotated = _draw_mistake_overlay(frame, mistake, w, h)
            _, pause_duration_sec = pause_audio[id(mistake)]
            freeze_frames = max(1, int(round(pause_duration_sec * fps)))
            for _ in range(freeze_frames):
                writer.write(annotated)

        writer.release()
        cap.release()

        print("  Mixing coaching audio track...")
        output_audio = _build_output_audio(segments, pause_audio, source_audio, fps)
        if len(output_audio) < 500:
            raise RuntimeError("Mixed coaching audio track is empty")
        output_audio.export(mixed_audio_path, format="mp3", bitrate="192k")

        print("  Muxing final annotated video with audio...")
        _mux_video_audio(silent_video_path, mixed_audio_path, out_video_path)
        print(f"Annotated video ready with audio: {out_video_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_video_path


def format_mistakes_report(summary):
    """Human-readable list of every mistake with exact timestamps."""
    mistakes = summary.get("timestamped_mistakes") or []
    if not mistakes:
        return "No form mistakes detected."

    lines = []
    for i, m in enumerate(mistakes, start=1):
        note = m.get("coaching_note") or (
            f"Rep {m['rep_number']} at {m['timestamp']}: {m['label']}"
        )
        line = f"{i}. {note}"
        lines.append(line)
    return "\n".join(lines)


def to_coaching_payload(summary):
    """JSON for Gemini — plain coaching language, no internal error codes."""
    payload = to_llm_payload(summary)

    error_counts = {}
    for code, count in (summary.get("error_counts") or {}).items():
        error_counts[ERROR_LABELS.get(code, code.replace("_", " "))] = count
    payload["error_counts"] = error_counts

    payload["timestamped_mistakes"] = [
        {k: v for k, v in m.items() if k != "error"}
        for m in summary.get("timestamped_mistakes") or []
    ]

    payload["reps"] = []
    for r in summary.get("reps") or []:
        payload["reps"].append({
            "rep_number": r["rep_number"],
            "start_timestamp": r.get("start_timestamp"),
            "end_timestamp": r.get("end_timestamp"),
            "bottom_timestamp": r.get("bottom_timestamp"),
            "quality": r.get("quality"),
            "completeness": r.get("completeness"),
            "coaching_issues": [
                ERROR_LABELS.get(e, e.replace("_", " "))
                for e in r.get("errors", [])
            ],
            "mistakes": r.get("mistakes"),
        })
    return payload


def to_llm_payload(summary):
    """Small payload for Gemini — numbers only, no raw frames/video."""
    return {
        "exercise": summary.get("exercise"),
        "total_reps": summary.get("total_reps"),
        "correct_reps": summary.get("correct_reps"),
        "bad_reps": summary.get("bad_reps"),
        "accuracy_percent": summary.get("accuracy_percent"),
        "avg_rep_duration_sec": summary.get("avg_rep_duration_sec"),
        "fatigue_detected": summary.get("fatigue_detected"),
        "fatigue_note": summary.get("fatigue_note"),
        "error_counts": summary.get("error_counts"),
        "timestamped_mistakes": summary.get("timestamped_mistakes", []),
        "not_measurable": summary.get("not_measurable", []),
    }


if __name__ == "__main__":
    input_video = r"D:\AIDS\cv\fitness_assist\inputs\pushup\WhatsApp Video 2026-06-12 at 18.11.18.mp4"
    out_video = None
    side = "left"

    result = analyze(input_video, out_video, side=side)

    print("\n=== Full Analysis ===")
    print(json.dumps(result, indent=2))

    print("\n=== Timestamped mistakes ===")
    print(format_mistakes_report(result))

    print("\n=== Compact payload for Gemini ===")
    print(json.dumps(to_llm_payload(result), indent=2))