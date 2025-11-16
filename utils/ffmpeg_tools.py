# utils/ffmpeg_tools.py
import json
import os
import shutil
import subprocess
from typing import Optional


def ensure_ffmpeg_installed():
    for bin_name in ["ffmpeg", "ffprobe"]:
        if shutil.which(bin_name) is None:
            raise RuntimeError(f"{bin_name} not found in PATH. Please install ffmpeg.")


def probe_media(path: str) -> Optional[dict]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def has_audio_stream(media_info: dict) -> bool:
    """Check if media info contains at least one audio stream."""
    streams = media_info.get("streams", [])
    return any(s.get("codec_type") == "audio" for s in streams)


def extract_audio(
    in_path: str,
    out_path: str,
    sample_rate: int = 48000,
    mono: bool = True,
):
    """
    Extract audio from video file to WAV format.
    Raises RuntimeError if extraction fails or if no audio stream exists.
    """
    # First check if the file has an audio stream
    media_info = probe_media(in_path)
    if not media_info:
        raise RuntimeError(f"Unable to probe media file: {in_path}")
    
    if not has_audio_stream(media_info):
        raise RuntimeError(f"Video file has no audio stream: {in_path}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1" if mono else "2",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr}")