from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import cv2
import numpy as np
import os
from typing import Optional, List

# Audio dependencies
import librosa
import scipy.signal as signal

app = FastAPI()

class Payload(BaseModel):
    video_dir: Optional[str] = None            # directory containing all hourly video files
    audio_dir: Optional[str] = None            # directory containing hourly audio files
    ad_video_path: Optional[str] = None
    ad_audio_path: Optional[str] = None
    threshold: float = 0.8                     # visual/template threshold
    audio_threshold: float = 0.7               # audio similarity threshold (normalized cross-correlation)
    start_datetime: str        # e.g., "2025-11-11T08:00:00"
    end_datetime: str          # e.g., "2025-11-11T20:00:00"


def extract_template(ad_path):
    cap = cv2.VideoCapture(ad_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read ad video frame")
    template = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return template, template.shape[::-1]


def detect_in_video(video_path, template, w, h, threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    occurrences = []
    frame_no = 0
    while frame_no < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            sec = frame_no / fps
            occurrences.append(sec)
        frame_no += 1

    cap.release()

    # group into blocks
    if occurrences:
        starts = [occurrences[0]]
        ends = []
        for i in range(1, len(occurrences)):
            if occurrences[i] - occurrences[i-1] > 1:
                ends.append(occurrences[i-1])
                starts.append(occurrences[i])
        ends.append(occurrences[-1])
        blocks = [{"start_sec": s, "end_sec": e} for s, e in zip(starts, ends)]
        total_seconds = sum(e-s for s, e in zip(starts, ends))
    else:
        blocks = []
        total_seconds = 0

    return blocks, total_seconds


# ---------------------- New audio helper functions ----------------------
def load_audio(path, sr=None):
    # librosa will handle many formats (wav, mp3, etc.) if soundfile/ffmpeg installed
    y, sr_actual = librosa.load(path, sr=sr, mono=True)
    return y, sr_actual


def extract_audio_template(ad_audio_path):
    y, sr = load_audio(ad_audio_path, sr=None)
    if y is None or len(y) == 0:
        raise ValueError("Failed to read ad audio")
    # normalize energy
    y = y / (np.sqrt(np.sum(y**2)) + 1e-8)
    return y, sr


def normalized_cross_correlation(signal_full, template):
    # compute normalized cross-correlation using FFT convolution for speed
    # signal_full and template are 1D numpy arrays
    template_rev = template[::-1]
    corr = signal.fftconvolve(signal_full, template_rev, mode='valid')
    # normalization: divide by sqrt(sum_sq_window * sum_sq_template)
    template_energy = np.sum(template**2)
    if template_energy == 0:
        return np.zeros_like(corr)
    # sliding window energy for the full signal
    window = np.ones(len(template))
    sig_sq = signal.fftconvolve(signal_full**2, window, mode='valid')
    denom = np.sqrt(sig_sq * template_energy) + 1e-8
    ncc = corr / denom
    return ncc


def detect_in_audio(audio_path, ad_signal, ad_sr, threshold):
    y, sr = load_audio(audio_path, sr=None)
    if y is None or len(y) == 0:
        return [], 0

    # resample ad if sample rates differ
    if sr != ad_sr:
        ad_resampled = librosa.resample(ad_signal, orig_sr=ad_sr, target_sr=sr)
    else:
        ad_resampled = ad_signal

    # normalize both
    if np.sum(ad_resampled**2) == 0 or np.sum(y**2) == 0:
        return [], 0
    y_norm = y / (np.sqrt(np.sum(y**2)) + 1e-8)
    ad_norm = ad_resampled / (np.sqrt(np.sum(ad_resampled**2)) + 1e-8)

    # compute normalized cross-correlation
    ncc = normalized_cross_correlation(y_norm, ad_norm)

    # find indices where similarity >= threshold
    indices = np.where(ncc >= threshold)[0]

    occurrences_seconds = []
    if len(indices) > 0:
        # convert index positions to times (index is the start sample of match in valid mode)
        times = indices / sr
        occurrences_seconds = list(times)

    # group into blocks (merge contiguous detections within 1 second)
    if occurrences_seconds:
        starts = [occurrences_seconds[0]]
        ends = []
        for i in range(1, len(occurrences_seconds)):
            if occurrences_seconds[i] - occurrences_seconds[i-1] > 1:
                ends.append(occurrences_seconds[i-1])
                starts.append(occurrences_seconds[i])
        ends.append(occurrences_seconds[-1])
        blocks = [{"start_sec": s, "end_sec": e} for s, e in zip(starts, ends)]
        total_seconds = sum(e - s for s, e in zip(starts, ends))
    else:
        blocks = []
        total_seconds = 0

    return blocks, total_seconds


# ---------------------- Endpoint updated to support audio ----------------------
@app.post("/detect_ad")
def detect_ad(payload: Payload):
    # parse datetime
    start_dt = datetime.fromisoformat(payload.start_datetime)
    end_dt = datetime.fromisoformat(payload.end_datetime)
    if start_dt >= end_dt:
        return {"error": "start_datetime must be before end_datetime"}

    total_hours = int((end_dt - start_dt).total_seconds() // 3600)

    total_seconds = 0
    all_blocks: List[dict] = []

    # Video processing (unchanged) if ad_video_path and video_dir provided
    if payload.ad_video_path and payload.video_dir:
        all_videos = sorted([os.path.join(payload.video_dir, f)
                             for f in os.listdir(payload.video_dir)
                             if f.endswith(('.mp4', '.avi'))])
        videos_to_process = all_videos[:total_hours]
        template, (w, h) = extract_template(payload.ad_video_path)
        for vid_path in videos_to_process:
            blocks, seconds = detect_in_video(vid_path, template, w, h, payload.threshold)
            total_seconds += seconds
            all_blocks.extend([{"media": os.path.basename(vid_path), "type": "video", **b} for b in blocks])

    # Audio processing if ad_audio_path and audio_dir provided
    if payload.ad_audio_path and payload.audio_dir:
        all_audios = sorted([os.path.join(payload.audio_dir, f)
                             for f in os.listdir(payload.audio_dir)
                             if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac'))])
        audios_to_process = all_audios[:total_hours]
        ad_signal, ad_sr = extract_audio_template(payload.ad_audio_path)
        for aud_path in audios_to_process:
            blocks, seconds = detect_in_audio(aud_path, ad_signal, ad_sr, payload.audio_threshold)
            total_seconds += seconds
            all_blocks.extend([{"media": os.path.basename(aud_path), "type": "audio", **b} for b in blocks])

    return {
        "total_appearances": len(all_blocks),
        "total_ad_seconds": total_seconds,
        "occurrences": all_blocks,
        "total_hours_requested": total_hours
    }
