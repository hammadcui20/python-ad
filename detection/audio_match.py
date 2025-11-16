# detection/audio_match.py
"""
Lightweight audio ad matching by sliding-window log-mel fingerprint similarity.

This module provides a function `find_audio_occurrences` which loads a short sample
audio file and a longer target audio file and returns occurrences where the sample
is likely present in the target.

Method (fast heuristic):
 - Load both files and resample to a common sampling rate
 - Compute log-mel spectrograms with a fixed hop length
 - Reduce each window to a single fingerprint vector by averaging mel bands across time
 - Compute running mean fingerprint of the target for windows equal to the sample length
 - Compute cosine similarity between the sample fingerprint and each target window fingerprint
 - Threshold and merge contiguous windows to form detection intervals

This is intentionally simple and fast (no DTW), and works well when sample and target
are recorded with similar quality and no dramatic transformations. It's a good starting
point and can be extended later with more advanced fingerprinting.
"""
from typing import List, Dict
import numpy as np
import librosa


def _ms_to_timestamp(ms: int) -> str:
    if ms < 0:
        ms = 0
    s, ms_rem = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_audio_occurrences(
    sample_audio_path: str,
    target_audio_path: str,
    sr: int = 22050,
    hop_ms: int = 50,
    n_mels: int = 64,
    match_threshold: float = 0.72,
    min_match_duration_seconds: float = 1.0,
) -> List[Dict]:
    """Find occurrences of sample audio inside target audio.

    Returns list of dicts with start_ms, end_ms, duration_ms, confidence, start, end, interval.
    """
    # Load audio
    y_s, sr_s = librosa.load(sample_audio_path, sr=sr, mono=True)
    y_t, sr_t = librosa.load(target_audio_path, sr=sr, mono=True)

    if y_s.size == 0 or y_t.size == 0:
        return []

    # Frame/hop settings
    hop_length = max(1, int(sr * (hop_ms / 1000.0)))
    # Compute log-mel spectrograms
    S_s = librosa.feature.melspectrogram(y_s, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048)
    S_t = librosa.feature.melspectrogram(y_t, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048)
    log_s = librosa.power_to_db(S_s, ref=np.max)
    log_t = librosa.power_to_db(S_t, ref=np.max)

    # Each column is a time frame. Compute the number of frames in sample
    frames_sample = log_s.shape[1]
    frames_target = log_t.shape[1]

    if frames_sample == 0 or frames_target == 0 or frames_sample > frames_target:
        return []

    # Compute sample fingerprint (mean across time -> n_mels vector)
    sample_fp = np.mean(log_s, axis=1)
    # Normalize
    sample_fp = sample_fp - np.mean(sample_fp)
    if np.linalg.norm(sample_fp) > 0:
        sample_fp = sample_fp / np.linalg.norm(sample_fp)

    # For target, compute running mean fingerprints for windows of length frames_sample
    # Use cumulative sum for fast sliding mean
    cumsum = np.cumsum(log_t, axis=1)
    # pad cumsum with zero column on left for easier calculation
    csum = np.concatenate((np.zeros((log_t.shape[0], 1)), cumsum), axis=1)

    window_sums = csum[:, frames_sample:] - csum[:, :frames_target - frames_sample + 1]
    # window_sums shape: (n_mels, num_windows)
    window_means = window_sums / float(frames_sample)

    # Normalize target window fingerprints
    # Subtract mean and normalize each column
    window_means = window_means - np.mean(window_means, axis=0, keepdims=True)
    norms = np.linalg.norm(window_means, axis=0)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    window_normed = window_means / norms

    # Compute cosine similarity between sample_fp and each window (dot product)
    scores = np.dot(sample_fp, window_normed)
    # Clip to [0,1]
    scores = np.clip(scores, -1.0, 1.0)
    # Positive similarities only (we care about similarity)
    scores = (scores + 1.0) / 2.0  # map from [-1,1] to [0,1]

    # Map window indices to times
    window_hop_sec = float(hop_length) / sr
    window_times = np.arange(0, len(scores)) * window_hop_sec

    # Thresholding
    mask = scores >= match_threshold

    # Convert mask to segments (in window indices), require minimum duration
    min_windows = max(1, int(np.ceil(min_match_duration_seconds / window_hop_sec)))

    segments = []
    start_idx = None
    for i, m in enumerate(mask):
        if m and start_idx is None:
            start_idx = i
        if (not m or i == len(mask) - 1) and start_idx is not None:
            end_idx = i if not m else i + 1
            length = end_idx - start_idx
            if length >= min_windows:
                start_time = start_idx * window_hop_sec
                end_time = (end_idx - 1) * window_hop_sec + (frames_sample * window_hop_sec)
                # end_time approx start_time + sample_duration
                segments.append({
                    "start_sec": float(start_time),
                    "end_sec": float(end_time),
                    "avg_score": float(np.mean(scores[start_idx:end_idx]))
                })
            start_idx = None

    # Merge close segments (gap < 0.5s)
    merged = []
    for seg in segments:
        if not merged or seg["start_sec"] - merged[-1]["end_sec"] > 0.5:
            merged.append(seg)
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], seg["end_sec"])
            merged[-1]["avg_score"] = max(merged[-1]["avg_score"], seg["avg_score"])

    # Prepare output format in ms and human timestamps
    out = []
    sample_duration_sec = float(len(y_s)) / sr
    for seg in merged:
        s_ms = int(round(seg["start_sec"] * 1000))
        e_ms = int(round(seg["end_sec"] * 1000))
        # clamp end to start + sample duration if shorter
        if e_ms - s_ms < int(round(sample_duration_sec * 1000)):
            e_ms = s_ms + int(round(sample_duration_sec * 1000))
        duration_ms = e_ms - s_ms
        occ = {
            "start_ms": s_ms,
            "end_ms": e_ms,
            "duration_ms": duration_ms,
            "confidence": float(seg["avg_score"]),
            "start": _ms_to_timestamp(s_ms),
            "end": _ms_to_timestamp(e_ms),
            "interval": f"{_ms_to_timestamp(s_ms)} - {_ms_to_timestamp(e_ms)}",
        }
        out.append(occ)

    return out
