import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr

# -----------------------------
# Safe cascade load (no !empty() errors)
# -----------------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade: " + CASCADE_PATH)


# -----------------------------
# Helpers
# -----------------------------
def detect_speaker_segments(
    video_path: str,
    frame_skip: int = 2,
    min_segment_dur: float = 5.0,
    pad: float = 0.15,
    resize_width: int = 960,
) -> List[Tuple[float, float]]:
    """Return (start, end) regions where a face is present for >= min_segment_dur."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    segments = []
    active_start = None
    frame_idx = 0

    while True:
        for _ in range(max(frame_skip - 1, 0)):
            if not cap.grab():
                break

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
        )

        t = frame_idx / fps
        if len(faces) > 0:
            if active_start is None:
                active_start = max(t - pad, 0)
        else:
            if active_start is not None:
                end_t = t + pad
                if end_t - active_start >= min_segment_dur:
                    segments.append((active_start, end_t))
                active_start = None

        frame_idx += frame_skip

    if active_start is not None:
        end_t = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        if end_t - active_start >= min_segment_dur:
            segments.append((active_start, end_t))

    cap.release()

    # merge tiny gaps
    merged = []
    merge_gap = 1.0
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        ps, pe = merged[-1]
        s, e = seg
        if s - pe <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append(seg)
    return merged


def detect_speech_intervals(
    audio_path: str,
    silence_thresh_db: int = -45,
    min_silence_len_ms: int = 800,
    keep_silence_ms: int = 100,
) -> List[Tuple[float, float]]:
    """Return (start, end) speech intervals (in seconds) using pydub."""
    audio = AudioSegment.from_file(audio_path)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )
    return [
        (
            max(0, s - keep_silence_ms) / 1000.0,
            min(len(audio), e + keep_silence_ms) / 1000.0,
        )
        for s, e in ranges
    ]


def intersect_intervals(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    res = []
    i = j = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s, e = max(s1, s2), min(e1, e2)
        if e > s:
            res.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return res


def enforce_target_lengths(
    segs: List[Tuple[float, float]],
    target_min: float,
    target_max: float,
    max_merge_gap: float = 5.0,
) -> List[Tuple[float, float]]:
    """Merge/split to ensure every final clip is within [target_min, target_max]."""
    if not segs:
        return []

    # 1) merge close-by first
    merged = []
    cs, ce = segs[0]
    for s, e in segs[1:]:
        if s - ce <= max_merge_gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    # 2) pack to [min, max]
    packed = []
    for s, e in merged:
        dur = e - s
        if dur < target_min:
            # too short, skip
            continue
        # split long ones into chunks of size <= target_max
        while dur > target_max:
            packed.append((s, s + target_max))
            s += target_max
            dur = e - s
        if dur >= target_min:
            packed.append((s, e))
    return packed


def extract_speaker_clips(
    video_path: str,
    output_folder: str = "clips",
    frame_skip: int = 2,
    min_segment_dur: float = 5.0,
    silence_thresh_db: int = -45,
    min_silence_len_ms: int = 800,
    keep_silence_ms: int = 100,
    target_min_len: float = 20.0,
    target_max_len: float = 30.0,
):
    os.makedirs(output_folder, exist_ok=True)

    # 1) Vision (face)
    face_segments = detect_speaker_segments(
        video_path,
        frame_skip=frame_skip,
        min_segment_dur=min_segment_dur,
    )
    if not face_segments:
        return None, "No face segments found."

    # 2) Audio (speech)
    clip = VideoFileClip(video_path)
    temp_audio = os.path.join(output_folder, "temp_audio.wav")
    clip.audio.write_audiofile(temp_audio, logger=None)
    speech_segments = detect_speech_intervals(
        temp_audio,
        silence_thresh_db=silence_thresh_db,
        min_silence_len_ms=min_silence_len_ms,
        keep_silence_ms=keep_silence_ms,
    )

    # 3) Intersection
    merged = intersect_intervals(face_segments, speech_segments)

    # 4) Enforce 20–30s
    merged = enforce_target_lengths(
        merged, target_min=target_min_len, target_max=target_max_len, max_merge_gap=5.0
    )
    if not merged:
        clip.close()
        return None, "No overlapping face+voice segments within the target length."

    # 5) Export clips with denoised audio
    sr = 16000
    output_paths = []
    for i, (start, end) in enumerate(merged, 1):
        subclip = clip.subclip(start, end)

        audio_np = subclip.audio.to_soundarray(fps=sr)
        if audio_np.ndim == 2:
            audio_mono = audio_np.mean(axis=1)
        else:
            audio_mono = audio_np

        reduced = nr.reduce_noise(y=audio_mono, sr=sr)

        # back to stereo for aac
        stereo = np.stack([reduced, reduced], axis=1)
        new_audio = AudioArrayClip(stereo, fps=sr)
        subclip = subclip.set_audio(new_audio)

        out_path = os.path.join(output_folder, f"clip_{i:03d}.mp4")
        subclip.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        output_paths.append(out_path)
        subclip.close()

    clip.close()

    if not output_paths:
        return None, "No valid segments after filtering."

    zip_path = os.path.join(output_folder, "speaker_segments.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in output_paths:
            zf.write(p, arcname=os.path.basename(p))

    return zip_path, f"✅ Extracted {len(output_paths)} clips between {target_min_len}-{target_max_len}s."


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎤 Speaker Extractor (Face + Voice + Denoise)")

uploaded_video = st.file_uploader(
    "Upload a video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
)

with st.expander("Advanced settings", expanded=False):
    frame_skip = st.slider("Frame skip (higher=faster, less accurate)", 1, 20, 2, 1)
    min_seg = st.slider("Minimum raw segment dur (sec, before shaping)", 1, 15, 5)
    silence_thresh_db = st.slider("Silence threshold (dBFS)", -80, -10, -45, 1)
    min_silence_len_ms = st.slider("Min silence length (ms)", 100, 2000, 800, 50)
    keep_silence_ms = st.slider("Keep silence padding (ms)", 0, 1000, 100, 25)
    target_min_len = st.slider("Target min clip length (sec)", 5, 60, 20)
    target_max_len = st.slider("Target max clip length (sec)", 10, 120, 30)

if uploaded_video:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.success("Video uploaded. Processing...")

    try:
        zip_path, status = extract_speaker_clips(
            "input_video.mp4",
            output_folder="clips",
            frame_skip=frame_skip,
            min_segment_dur=min_seg,
            silence_thresh_db=silence_thresh_db,
            min_silence_len_ms=min_silence_len_ms,
            keep_silence_ms=keep_silence_ms,
            target_min_len=target_min_len,
            target_max_len=target_max_len,
        )
    except Exception as e:
        st.error(f"Error: {e}")
    else:
        if not zip_path:
            st.error(status)
        else:
            st.success(status)
            with open(zip_path, "rb") as f:
                st.download_button(
                    "📦 Download Speaker Segments (ZIP)",
                    f.read(),
                    file_name="speaker_segments.zip",
                )
