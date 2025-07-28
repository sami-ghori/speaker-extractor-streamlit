import cv2
import os
import zipfile
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr
import soundfile as sf
from pathlib import Path

CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")

def detect_speaker_segments(video_path, frame_skip=2, min_segment_dur=5, pad=0.15, resize_width=960):
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
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
            frame = cv2.resize(frame, (resize_width, int(h * (resize_width / w))))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        t = frame_idx / fps
        if len(faces) > 0:
            if active_start is None:
                active_start = max(t - pad, 0)
        elif active_start is not None:
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

def detect_speech_intervals(audio_path, silence_thresh_db=-45, min_silence_len_ms=800, keep_silence_ms=100):
    audio = AudioSegment.from_file(audio_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
    return [(max(0, s-keep_silence_ms)/1000.0, min(len(audio), e+keep_silence_ms)/1000.0) for s, e in nonsilent]

def intersect_intervals(a, b):
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

def extract_speaker_clips(video_path, output_folder="clips"):
    os.makedirs(output_folder, exist_ok=True)
    face_segments = detect_speaker_segments(video_path)
    clip = VideoFileClip(video_path)
    temp_audio = os.path.join(output_folder, "temp_audio.wav")
    clip.audio.write_audiofile(temp_audio, logger=None)
    speech_segments = detect_speech_intervals(temp_audio)
    merged = intersect_intervals(face_segments, speech_segments)

    output_paths = []
    for i, (start, end) in enumerate(merged):
        if end - start < 5:
            continue
        subclip = clip.subclip(start, end)
        audio_array, sr = sf.read(subclip.audio.to_soundarray(fps=44100), dtype='float32'), 44100
        reduced = nr.reduce_noise(y=audio_array, sr=sr)
        clip_path = os.path.join(output_folder, f"clip_{i+1}.mp4")
        subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
        output_paths.append(clip_path)

    zip_path = os.path.join(output_folder, "speaker_segments.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in output_paths:
            zf.write(p, arcname=os.path.basename(p))
    clip.close()
    return zip_path

st.title("🎤 Speaker Extractor (Face + Voice + Denoise)")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if uploaded_video:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
    st.success("Video uploaded. Processing...")
    zip_path = extract_speaker_clips("input_video.mp4")
    with open(zip_path, "rb") as f:
        st.download_button("📦 Download Speaker Segments (ZIP)", f, file_name="speaker_segments.zip")
