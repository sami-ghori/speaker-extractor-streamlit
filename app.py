import cv2
import os
import zipfile
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr

CASCADE_PATH = "haarcascade_frontalface_default.xml"

def detect_speaker_segments(video_path, frame_skip=2, min_segment_dur=20, pad=0.15, resize_width=640):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Face cascade failed to load.")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    segments, active_start, frame_idx = [], None, 0
    while True:
        for _ in range(frame_skip - 1):
            if not cap.grab(): break
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        if w > resize_width:
            frame = cv2.resize(frame, (resize_width, int(h * resize_width / w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
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
    return segments

def detect_speech_intervals(audio_path, silence_thresh_db=-45, min_silence_len_ms=800, keep_silence_ms=150):
    audio = AudioSegment.from_file(audio_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
    result = []
    for s, e in nonsilent:
        result.append((max(0, (s - keep_silence_ms)/1000), (e + keep_silence_ms)/1000))
    return result

def intersect_intervals(a, b):
    res, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s, e = max(s1, s2), min(e1, e2)
        if e > s: res.append((s, e))
        if e1 < e2: i += 1
        else: j += 1
    return res

def extract_speaker_clips(video_path, output_folder="clips"):
    os.makedirs(output_folder, exist_ok=True)
    face_segments = detect_speaker_segments(video_path)
    if not face_segments:
        return None, "No face+voice segments long enough."
    clip = VideoFileClip(video_path)
    audio_temp = os.path.join(output_folder, "audio_full.wav")
    clip.audio.write_audiofile(audio_temp, logger=None)
    speech_intervals = detect_speech_intervals(audio_temp)
    candidates = intersect_intervals(face_segments, speech_intervals)
    output_paths = []
    for idx, (start, end) in enumerate(candidates):
        duration = end - start
        if duration < 20:
            continue
        sub = clip.subclip(start, end)
        arr = sub.audio.to_soundarray(fps=22050)[:,0]
        den = nr.reduce_noise(y=arr, sr=22050)
        out = os.path.join(output_folder, f"clip_{idx+1:02d}.mp4")
        sub.write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
        output_paths.append(out)
    clip.close()
    if not output_paths:
        return None, "No valid clips extracted."
    zipf = os.path.join(output_folder, "speaker_clips.zip")
    with zipfile.ZipFile(zipf, "w") as z:
        for p in output_paths:
            z.write(p, arcname=os.path.basename(p))
    return zipf, f"✅ Extracted {len(output_paths)} clip(s)."

st.title("🎤 Speaker Extractor (Face + Voice + Denoise)")
vid = st.file_uploader("Upload a TED‑talk style video", type=["mp4","mov","avi"])
if vid is not None:
    with open("input_video.mp4","wb") as f:
        f.write(vid.read())
    st.info("Processing… this can take ~1 min per 5 min video.")
    zipfile_path, msg = extract_speaker_clips("input_video.mp4")
    if zipfile_path:
        st.success(msg)
        st.download_button("Download ZIP", open(zipfile_path, "rb"), file_name="speaker_clips.zip")
    else:
        st.warning(msg)
