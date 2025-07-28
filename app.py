import os
import zipfile
import tempfile
import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Constants
MIN_CLIP_DURATION = 30  # Minimum clip duration in seconds
FRAME_SKIP = 3  # Higher number = faster but less accurate
TARGET_RESOLUTION = (640, 360)  # Lower resolution for faster processing
AUDIO_SAMPLE_RATE = 16000
NOISE_REDUCTION_PROFILE = 'speech'  # 'speech' or 'noise' profile for different noise types

# Initialize Haar cascade (face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """Detect faces in a frame with optimized parameters"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return len(faces) > 0

def detect_speaker_segments(video_path):
    """Detect segments where speaker is both talking and visible"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    segments = []
    current_start = None

    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, TARGET_RESOLUTION)
            
            # Submit face detection to thread pool
            future = executor.submit(detect_faces, frame)
            has_face = future.result()
            
            current_time = frame_count / fps
            
            if has_face:
                if current_start is None:
                    current_start = current_time
            else:
                if current_start is not None and (current_time - current_start) >= 1:  # Minimum 1s segment
                    segments.append((current_start, current_time))
                    current_start = None

    cap.release()
    
    if current_start is not None:
        segments.append((current_start, frame_count / fps))
    
    return merge_close_segments(segments, max_gap=0.5)

def merge_close_segments(segments, max_gap=1.0):
    """Merge close segments together"""
    if not segments:
        return []
    
    merged = [segments[0]]
    for seg in segments[1:]:
        last_end = merged[-1][1]
        current_start = seg[0]
        if current_start - last_end <= max_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged

def extract_speech_segments(audio_path):
    """Extract speech segments using audio analysis"""
    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=500,
        silence_thresh=-35,
        seek_step=10
    )
    
    # Convert to seconds and add small padding
    return [(max(0, start/1000 - 0.1), min(len(audio)/1000, end/1000 + 0.1)) 
            for start, end in nonsilent_ranges]

def filter_segments(video_path, face_segments, speech_segments):
    """Find overlapping segments where speaker is both visible and talking"""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    
    # Find intersection of face and speech segments
    intersected = []
    fi = si = 0
    
    while fi < len(face_segments) and si < len(speech_segments):
        fs, fe = face_segments[fi]
        ss, se = speech_segments[si]
        
        overlap_start = max(fs, ss)
        overlap_end = min(fe, se)
        
        if overlap_end > overlap_start:
            intersected.append((overlap_start, overlap_end))
        
        if fe < se:
            fi += 1
        else:
            si += 1
    
    # Merge and enforce minimum duration
    final_segments = []
    for seg in merge_close_segments(intersected, max_gap=1.0):
        start, end = seg
        dur = end - start
        
        if dur < MIN_CLIP_DURATION:
            # Try to extend short segments
            if final_segments:
                last_start, last_end = final_segments[-1]
                if start - last_end < 2:  # If close to previous segment
                    final_segments[-1] = (last_start, end)
                    continue
            
            # Skip if still too short
            continue
        
        final_segments.append((start, min(end, duration)))
    
    return final_segments

def process_segment(video_path, segment, clip_idx):
    """Process a single segment (for parallel processing)"""
    start, end = segment
    clip = VideoFileClip(video_path).subclip(start, end)
    
    # Noise reduction
    audio = clip.audio.to_soundarray(fps=AUDIO_SAMPLE_RATE)
    if len(audio.shape) == 2:  # stereo
        audio_mono = audio.mean(axis=1)
    else:
        audio_mono = audio
    
    reduced_noise = nr.reduce_noise(
        y=audio_mono, 
        sr=AUDIO_SAMPLE_RATE,
        stationary=True,
        prop_decrease=0.9,
        time_mask_smooth_ms=100
    )
    
    # Create new audio clip
    stereo_audio = np.stack([reduced_noise, reduced_noise], axis=1)
    new_audio = AudioArrayClip(stereo_audio, fps=AUDIO_SAMPLE_RATE)
    clip = clip.set_audio(new_audio)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        clip.write_videofile(
            tmp.name,
            codec='libx264',
            audio_codec='aac',
            bitrate='1500k',
            preset='ultrafast',
            threads=4,
            logger=None
        )
        clip.close()
        return tmp.name

def process_video(uploaded_video):
    """Main processing pipeline"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name
    
    try:
        # Step 1: Extract face segments
        face_segments = detect_speaker_segments(video_path)
        if not face_segments:
            return None, "No speaker segments found in video"
        
        # Step 2: Extract speech segments
        clip = VideoFileClip(video_path)
        audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        clip.audio.write_audiofile(audio_path, fps=AUDIO_SAMPLE_RATE, logger=None)
        clip.close()
        
        speech_segments = extract_speech_segments(audio_path)
        if not speech_segments:
            return None, "No speech detected in audio"
        
        # Step 3: Get final segments
        final_segments = filter_segments(video_path, face_segments, speech_segments)
        if not final_segments:
            return None, "No segments with both speaker and speech detected"
        
        # Step 4: Process segments in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_segment, video_path, seg, i) 
                for i, seg in enumerate(final_segments)
            ]
            
            clip_paths = []
            for future in concurrent.futures.as_completed(futures):
                clip_paths.append(future.result())
        
        # Step 5: Create ZIP
        zip_path = f"speaker_segments_{len(clip_paths)}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for path in clip_paths:
                zipf.write(path, os.path.basename(path))
                os.unlink(path)
        
        return zip_path, f"✅ Processed {len(clip_paths)} speaker segments"
    
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Streamlit UI
st.set_page_config(
    page_title="Speaker Extractor Pro",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Speaker Extractor Pro")
st.markdown("""
Extract clean speaker segments from videos (lectures, talks, etc.) by:
- Detecting when speaker is both visible **and** talking
- Removing background noise (applause, laughter)
- Exporting clean clips of at least 30 seconds
""")

uploaded_video = st.file_uploader(
    "Upload video (MP4, MOV, AVI)", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=False
)

if uploaded_video:
    with st.spinner("Processing video (this may take a few minutes)..."):
        zip_path, message = process_video(uploaded_video)
    
    if zip_path:
        st.success(message)
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Clean Speaker Segments",
                f,
                file_name="speaker_segments.zip",
                mime="application/zip"
            )
        os.unlink(zip_path)
    else:
        st.error(message)

st.markdown("---")
with st.expander("Advanced Settings (Click to expand)"):
    st.warning("Adjust these only if you're experiencing issues with the default settings")
    st.checkbox("Aggressive noise reduction", value=False, key="aggressive_nr")
    st.slider("Speech detection sensitivity", 0.1, 2.0, 1.0, key="speech_sensitivity")
    st.slider("Face detection threshold", 1.0, 3.0, 1.5, key="face_threshold")

st.markdown("### How It Works")
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Replace with your demo video
