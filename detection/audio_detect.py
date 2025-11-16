# detection/audio_detect.py
from typing import List, Dict, Tuple
import subprocess
import re
import numpy as np
import librosa


def _silence_regions_ffmpeg(
    path: str, silence_db: float = -35.0, min_silence_ms: int = 300
) -> List[Tuple[int, int]]:
    """
    Use ffmpeg silencedetect to get silence spans with sub-second precision.
    Returns list of (start_ms, end_ms).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={silence_db}dB:d={min_silence_ms/1000.0}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    out = proc.stderr

    # Use clearer float patterns to match ffmpeg's timestamps (e.g. 12.345)
    re_start = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
    re_end = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*silence_duration:\s*([0-9]+(?:\.[0-9]+)?)")

    starts = []
    ends = []
    for line in out.splitlines():
        m1 = re_start.search(line)
        if m1:
            starts.append(float(m1.group(1)))
        m2 = re_end.search(line)
        if m2:
            ends.append(float(m2.group(1)))

    # Pair up best-effort
    regions = []
    si = 0
    ei = 0
    while si < len(starts) and ei < len(ends):
        s = int(starts[si] * 1000)
        e = int(ends[ei] * 1000)
        if e >= s:
            regions.append((s, e))
            si += 1
            ei += 1
        else:
            ei += 1
    return regions


def _rule_based_musiciness(y: np.ndarray, sr: int, window_ms: int, hop_ms: int) -> np.ndarray:
    """
    Simple rule-based window classifier for music/advert-like texture:
      - higher spectral flatness
      - higher zero-crossing rate
      - higher RMS (loudness)
    Returns boolean mask per window (True -> likely ad).
    """
    win = int(window_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    if win < 1:
        win = 1
    if hop < 1:
        hop = 1

    # Features
    # Ensure FFT size is at least the window length. librosa.stft requires n_fft >= win_length.
    # Use n_fft = max(2048, win) to avoid extremely large FFT sizes for large windows.
    n_fft = max(2048, win)
    # Compute STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)) + 1e-9
    # Compute RMS from time-domain signal so frame_length matches 'win'
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).flatten()           # 0..1+
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop).flatten()
    flatness = librosa.feature.spectral_flatness(S=S).flatten()

    # Robust thresholds via percentiles
    r_t = np.percentile(rms, 60)
    z_t = np.percentile(zcr, 60)
    f_t = np.percentile(flatness, 60)


    score = (
        (rms > r_t).astype(np.int32)
        + (zcr > z_t).astype(np.int32)
        + (flatness > f_t).astype(np.int32)
    )
    # 2-of-3 rule
    return (score >= 2)


def _mask_to_segments(mask: np.ndarray, hop_ms: int, min_len_ms: int = 2000) -> List[Dict]:
    """
    Convert window mask to merged segments in milliseconds.
    """
    segments = []
    if len(mask) == 0:
        return segments

    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end_idx = i if not v else i + 1
            s_ms = start * hop_ms
            e_ms = end_idx * hop_ms
            if (e_ms - s_ms) >= min_len_ms:
                segments.append({"start_ms": int(s_ms), "end_ms": int(e_ms)})
            start = None
    # Merge close segments (< 500 ms gap)
    merged = []
    for seg in segments:
        if not merged or seg["start_ms"] - merged[-1]["end_ms"] > 500:
            merged.append(seg)
        else:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], seg["end_ms"])
    for seg in merged:
        seg["duration_ms"] = seg["end_ms"] - seg["start_ms"]
    return merged


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    if ms < 0:
        ms = 0
    s, ms_rem = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def detect_audio_ads(
    audio_path: str,
    silence_db: float = -35.0,
    min_silence_ms: int = 300,
    window_ms: int = 500,
    hop_ms: int = 100,
) -> List[Dict]:
    """
    Two-stage heuristic for audio ad detection:
      1) Rule-based "musiciness" classifier over sliding windows (ms resolution).
      2) Align and trim with detected silences to get clean boundaries.

    Returns list of {start_ms, end_ms, duration_ms}.
    """
    # Load mono, preserve precision
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if y.size == 0:
        return []

    # Stage 1: window classification
    music_mask = _rule_based_musiciness(y, sr, window_ms=window_ms, hop_ms=hop_ms)
    rough_segments = _mask_to_segments(music_mask, hop_ms=hop_ms, min_len_ms=2000)

    # Stage 2: silence trimming to improve boundaries
    silences = _silence_regions_ffmpeg(audio_path, silence_db=silence_db, min_silence_ms=min_silence_ms)

    def trim_to_silence(seg):
        s = seg["start_ms"]
        e = seg["end_ms"]
        # Find nearest preceding and following silence boundaries
        left_candidates = [ss for (ss, se) in silences if ss <= s] + [se for (ss, se) in silences if se <= s]
        right_candidates = [ss for (ss, se) in silences if ss >= e] + [se for (ss, se) in silences if se >= e]
        if left_candidates:
            s = max(s, max(left_candidates))
        if right_candidates:
            e = min(e, min(right_candidates))
        if e <= s:
            return None
        return {"start_ms": s, "end_ms": e, "duration_ms": e - s}

    final_segments = []
    for seg in rough_segments:
        trimmed = trim_to_silence(seg)
        if trimmed and trimmed["duration_ms"] >= 1500:
            final_segments.append(trimmed)

    # Deduplicate and sort
    final_segments.sort(key=lambda x: x["start_ms"])
    merged = []
    for seg in final_segments:
        if not merged or seg["start_ms"] > merged[-1]["end_ms"]:
            merged.append(seg)
        else:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], seg["end_ms"])
            merged[-1]["duration_ms"] = merged[-1]["end_ms"] - merged[-1]["start_ms"]

    # Attach ms-accurate human-readable timestamps for each segment
    for seg in merged:
        seg_start = int(seg["start_ms"]) if isinstance(seg["start_ms"], (int, float)) else int(seg["start_ms"])
        seg_end = int(seg["end_ms"]) if isinstance(seg["end_ms"], (int, float)) else int(seg["end_ms"])
        seg["start"] = _ms_to_timestamp(seg_start)
        seg["end"] = _ms_to_timestamp(seg_end)
        seg["interval"] = f"{seg['start']} - {seg['end']}"

    return merged

