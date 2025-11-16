# app.py
import os
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from utils.ffmpeg_tools import probe_media, extract_audio, ensure_ffmpeg_installed, has_audio_stream
from detection.video_detect import detect_video_ad_candidates
from detection.audio_detect import detect_audio_ads
from detection.ad_match_v2 import AdMatcher, AdMatchConfig
from detection.audio_match import find_audio_occurrences

from config import AppConfig

from datetime import datetime, timezone

app = Flask(__name__)
app.config.from_object(AppConfig)


def _parse_time_to_epoch_ms(val: str):
    """Parse either integer milliseconds or an ISO-8601 datetime string into epoch ms (UTC).
    Returns None on parse failure.
    """
    if val is None:
        return None
    val = str(val).strip()
    if val == "":
        return None
    # Try integer milliseconds
    try:
        return int(val)
    except Exception:
        pass
    # Try ISO format; allow Z
    try:
        s = val
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _epoch_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST multipart/form-data:
      - file: video/audio file
      - optional params:
          silence_db: float (default -35.0)
          min_silence_ms: int (default 300)
          window_ms: int (default 500)
          hop_ms: int (default 100)
          recording_start: ISO8601 or epoch-ms (absolute start time of THIS uploaded file)
          series_start: ISO8601 or epoch-ms (absolute start time of the whole multi-part recording)
          part_duration_ms: int (duration of each part in ms; default 3600000 = 1 hour)
          total_parts: int (optional - expected number of parts in the whole series)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)

    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="ingest_", dir=app.config["UPLOAD_DIR"])
    in_path = os.path.join(tmp_dir, filename)
    file.save(in_path)

    try:
        ensure_ffmpeg_installed()

        media_info = probe_media(in_path)
        if not media_info:
            return jsonify({"error": "Unable to probe media"}), 400

        # Parameters (with practical defaults)
        silence_db = float(request.form.get("silence_db", -35.0))
        min_silence_ms = int(request.form.get("min_silence_ms", 300))
        window_ms = int(request.form.get("window_ms", 500))
        hop_ms = int(request.form.get("hop_ms", 100))

        # Optional recording/series metadata
        recording_start_raw = request.form.get("recording_start")
        series_start_raw = request.form.get("series_start")
        part_duration_ms = int(request.form.get("part_duration_ms", 3600000))
        total_parts_raw = request.form.get("total_parts")

        is_video = any(s.get("codec_type") == "video" for s in media_info.get("streams", []))
        has_audio = has_audio_stream(media_info)
        duration_ms = int(float(media_info.get("format", {}).get("duration", 0.0)) * 1000)

        result = {
            "media": {
                "filename": filename,
                "format": media_info.get("format", {}).get("format_name"),
                "duration_ms": duration_ms,
                "is_video": is_video,
                "has_audio": has_audio,
            },
            "video_candidates": [],
            "audio_ads": [],
        }

        # Parse recording/series timestamps if provided
        recording_start_ms = _parse_time_to_epoch_ms(recording_start_raw)
        series_start_ms = _parse_time_to_epoch_ms(series_start_raw) if series_start_raw else recording_start_ms
        total_parts = int(total_parts_raw) if (total_parts_raw is not None and total_parts_raw != "") else None

        # If recording start provided, add absolute recording start/end to result
        if recording_start_ms is not None:
            result["recording"] = {
                "start_epoch_ms": recording_start_ms,
                "start_iso": _epoch_ms_to_iso(recording_start_ms),
                "end_epoch_ms": recording_start_ms + duration_ms,
                "end_iso": _epoch_ms_to_iso(recording_start_ms + duration_ms),
                "duration_ms": duration_ms,
                "part_duration_ms": part_duration_ms,
                "series_start_epoch_ms": series_start_ms,
                "series_start_iso": _epoch_ms_to_iso(series_start_ms) if series_start_ms is not None else None,
            }

        # If video: first detect ad-like candidates by cut density
        audio_path = None
        if is_video:
            result["video_candidates"] = detect_video_ad_candidates(
                in_path,
                min_block_ms=5000,          # group dense cuts into >=5s blocks
                window_ms=5000,             # rolling window size for cut density
                min_cuts_in_window=3        # tuneable
            )

            # Extract audio as WAV 48k mono for stable analysis (only if audio stream exists)
            if has_audio:
                audio_path = os.path.join(tmp_dir, "audio.wav")
                extract_audio(in_path, audio_path, sample_rate=48000, mono=True)
        else:
            audio_path = in_path

        # Audio ad detection (ms resolution) - only if we have audio
        if has_audio and audio_path:
            audio_ads = detect_audio_ads(
                audio_path,
                silence_db=silence_db,
                min_silence_ms=min_silence_ms,
                window_ms=window_ms,
                hop_ms=hop_ms,
            )
            result["audio_ads"] = audio_ads
        else:
            result["audio_ads"] = []

        # If recording start provided, compute absolute timestamps for each detection and parts covered
        if recording_start_ms is not None:
            covered_parts = set()

            def _segment_add_abs_and_parts(seg):
                # seg expected to have start_ms/end_ms fields
                s_offset = int(seg.get("start_ms", 0))
                e_offset = int(seg.get("end_ms", 0))
                abs_s = recording_start_ms + s_offset
                abs_e = recording_start_ms + e_offset
                seg["abs_start_epoch_ms"] = abs_s
                seg["abs_end_epoch_ms"] = abs_e
                seg["abs_start_iso"] = _epoch_ms_to_iso(abs_s)
                seg["abs_end_iso"] = _epoch_ms_to_iso(abs_e)
                # compute parts spanned relative to series_start_ms
                if series_start_ms is not None and part_duration_ms > 0:
                    # Treat end as exclusive so a segment that ends exactly on a boundary
                    # does not include the next part. Use abs_e - 1 when computing the end index.
                    part_idx_start = (abs_s - series_start_ms) // part_duration_ms
                    part_idx_end = (max(abs_e - 1, abs_s) - series_start_ms) // part_duration_ms
                    # ensure integer and clamp to valid range
                    part_idx_start = int(part_idx_start)
                    part_idx_end = int(part_idx_end)
                    # If total_parts is provided, clamp to [0, total_parts-1]
                    if total_parts is not None:
                        max_idx = max(0, int(total_parts) - 1)
                        part_idx_start = max(0, min(part_idx_start, max_idx))
                        part_idx_end = max(0, min(part_idx_end, max_idx))
                    else:
                        part_idx_start = max(0, part_idx_start)
                        part_idx_end = max(0, part_idx_end)

                    if part_idx_end >= part_idx_start:
                        parts = list(range(part_idx_start, part_idx_end + 1))
                    else:
                        parts = []
                    seg["parts_spanned"] = parts
                    for p in parts:
                        covered_parts.add(p)
                else:
                    seg["parts_spanned"] = []

            for a in result.get("audio_ads", []):
                _segment_add_abs_and_parts(a)
            for v in result.get("video_candidates", []):
                _segment_add_abs_and_parts(v)

            # If total_parts provided, build expected parts list and compute missing
            parts_info = None
            if total_parts is not None and series_start_ms is not None:
                expected = []
                for i in range(total_parts):
                    p_start = series_start_ms + i * part_duration_ms
                    p_end = p_start + part_duration_ms
                    expected.append(
                        {
                            "index": i,
                            "start_epoch_ms": p_start,
                            "start_iso": _epoch_ms_to_iso(p_start),
                            "end_epoch_ms": p_end,
                            "end_iso": _epoch_ms_to_iso(p_end),
                        }
                    )
                expected_indices = set(range(total_parts))
                missing = sorted(list(expected_indices - covered_parts))

                parts_info = {
                    "expected_count": total_parts,
                    "part_duration_ms": part_duration_ms,
                    "expected": expected,
                    "covered": sorted(list(covered_parts)),
                    "missing": missing,
                }
            else:
                parts_info = {"covered": sorted(list(covered_parts))}

            result["parts"] = parts_info

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/match", methods=["POST"])
def match_ad():
    """
    POST multipart/form-data:
      - sample_video: advertisement sample video file
      - target_video: video file in which to search for the advertisement
      - optional params from AdMatchConfig, e.g.:
          frame_sampling_fps: float
          fallback_static_threshold: float
          enable_dynamic_threshold: bool
          debug_mode: bool
    """
    if "sample_video" not in request.files:
        return jsonify({"error": "No sample_video field provided"}), 400
    
    if "target_video" not in request.files:
        return jsonify({"error": "No target_video field provided"}), 400
    
    sample_file = request.files["sample_video"]
    target_file = request.files["target_video"]
    
    if sample_file.filename == "":
        return jsonify({"error": "Empty sample_video filename"}), 400
    
    if target_file.filename == "":
        return jsonify({"error": "Empty target_video filename"}), 400
    
    sample_filename = secure_filename(sample_file.filename)
    target_filename = secure_filename(target_file.filename)
    
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="match_", dir=app.config["UPLOAD_DIR"])
    sample_path = os.path.join(tmp_dir, f"sample_{sample_filename}")
    target_path = os.path.join(tmp_dir, f"target_{target_filename}")
    
    sample_file.save(sample_path)
    target_file.save(target_path)
    
    try:
        ensure_ffmpeg_installed()
        
        sample_info = probe_media(sample_path)
        target_info = probe_media(target_path)
        
        if not sample_info or not any(s.get("codec_type") == "video" for s in sample_info.get("streams", [])):
            return jsonify({"error": "Sample file is not a valid video"}), 400
        
        if not target_info or not any(s.get("codec_type") == "video" for s in target_info.get("streams", [])):
            return jsonify({"error": "Target file is not a valid video"}), 400

        # Build config from request form, using AdMatchConfig defaults
        config = AdMatchConfig()
        for key, default_value in AdMatchConfig.__dataclass_fields__.items():
            if key in request.form:
                # Coerce type based on default value
                value_type = type(default_value.default)
                try:
                    form_val = request.form[key]
                    if value_type == bool:
                        coerced_value = form_val.lower() in ['true', '1', 'yes']
                    else:
                        coerced_value = value_type(form_val)
                    setattr(config, key, coerced_value)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid type for parameter '{key}'. Expected {value_type.__name__}."}), 400

        if config.debug_mode:
            config.debug_frame_output_dir = os.path.join(tmp_dir, "debug_frames")

        # Initialize and run the matcher
        matcher = AdMatcher(config)
        occurrences = matcher.find_ad_occurrences(
            sample_video_path=sample_path,
            target_video_path=target_path,
        )
        
        # Calculate summary statistics
        total_occurrences = len(occurrences)
        total_duration_ms = sum(occ["duration_ms"] for occ in occurrences)
        avg_confidence = sum(occ["confidence"] for occ in occurrences) / total_occurrences if total_occurrences > 0 else 0.0
        
        result = {
            "sample_video": {
                "filename": sample_filename,
                "duration_ms": int(float(sample_info.get("format", {}).get("duration", 0.0)) * 1000),
            },
            "target_video": {
                "filename": target_filename,
                "duration_ms": int(float(target_info.get("format", {}).get("duration", 0.0)) * 1000),
            },
            "matches": {
                "total_occurrences": total_occurrences,
                "total_duration_ms": total_duration_ms,
                "total_duration_seconds": round(total_duration_ms / 1000.0, 2),
                "average_confidence": round(avg_confidence, 3),
            },
            "occurrences": occurrences,
        }
        
        # If in debug mode, include path to debug frames
        if config.debug_mode and config.debug_frame_output_dir:
            result["debug_info"] = {
                "message": "Debug frames and logs are stored on the server.",
                "debug_frame_output_dir": config.debug_frame_output_dir
            }

        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/match_audio", methods=["POST"])
def match_audio():
    """
    POST multipart/form-data:
      - sample_audio: audio file (wav/mp3) containing the advertisement sample
      - target: audio or video file to search in
      - optional form parameters: sr, hop_ms, n_mels, match_threshold, min_match_duration_seconds
    """
    if "sample_audio" not in request.files:
        return jsonify({"error": "No sample_audio field provided"}), 400
    if "target" not in request.files:
        return jsonify({"error": "No target field provided"}), 400

    sample_file = request.files["sample_audio"]
    target_file = request.files["target"]

    if sample_file.filename == "":
        return jsonify({"error": "Empty sample_audio filename"}), 400
    if target_file.filename == "":
        return jsonify({"error": "Empty target filename"}), 400

    sample_filename = secure_filename(sample_file.filename)
    target_filename = secure_filename(target_file.filename)

    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="match_audio_", dir=app.config["UPLOAD_DIR"])
    sample_path = os.path.join(tmp_dir, f"sample_{sample_filename}")
    target_path = os.path.join(tmp_dir, f"target_{target_filename}")

    sample_file.save(sample_path)
    target_file.save(target_path)

    try:
        ensure_ffmpeg_installed()
        # If target is video, extract audio
        media_info = probe_media(target_path)
        has_audio = False
        if media_info:
            has_audio = has_audio_stream(media_info)
        if media_info and any(s.get("codec_type") == "video" for s in media_info.get("streams", [])):
            if not has_audio:
                return jsonify({"error": "Target video has no audio stream"}), 400
            audio_target_path = os.path.join(tmp_dir, "target_audio.wav")
            extract_audio(target_path, audio_target_path, sample_rate=22050, mono=True)
        else:
            # target is an audio file already
            audio_target_path = target_path

        # Read optional params
        sr = int(request.form.get("sr", 22050))
        hop_ms = int(request.form.get("hop_ms", 50))
        n_mels = int(request.form.get("n_mels", 64))
        match_threshold = float(request.form.get("match_threshold", 0.72))
        min_match_duration_seconds = float(request.form.get("min_match_duration_seconds", 1.0))

        occurrences = find_audio_occurrences(
            sample_audio_path=sample_path,
            target_audio_path=audio_target_path,
            sr=sr,
            hop_ms=hop_ms,
            n_mels=n_mels,
            match_threshold=match_threshold,
            min_match_duration_seconds=min_match_duration_seconds,
        )

        # Summarize
        total_occurrences = len(occurrences)
        total_duration_ms = sum(o["duration_ms"] for o in occurrences)
        avg_confidence = sum(o["confidence"] for o in occurrences) / total_occurrences if total_occurrences > 0 else 0.0

        result = {
            "sample_audio": {"filename": sample_filename},
            "target": {"filename": target_filename},
            "matches": {"total_occurrences": total_occurrences, "total_duration_ms": total_duration_ms, "average_confidence": round(avg_confidence, 3)},
            "occurrences": occurrences,
        }

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
