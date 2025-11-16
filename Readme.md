# README.md
## Flask Ad Duration Detector (Video + Audio)

### Features
- Upload video or audio.
- If video: detect ad-like segments from rapid scene cuts, then extract audio and run audio analysis.
- Millisecond-precision timestamps via ffprobe/ffmpeg + numpy timebase.
- JSON output listing segments.

### Prerequisites
- ffmpeg/ffprobe installed (`ffmpeg -version`).
- Python 3.10+ (or use Docker).

### Install (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_RUN_PORT=8000
python app.py
```

### Run with Docker
```bash
docker build -t ad-detector .
docker run --rm -p 8000:8000 -e UPLOAD_DIR=/tmp/ad_uploads ad-detector
```

### API
- GET /health
- POST /detect (multipart/form-data)
  - file: video/audio
  - optional form fields:
    - silence_db (float, default -35.0)
    - min_silence_ms (int, default 300)
    - window_ms (int, default 500)
    - hop_ms (int, default 100)

### Example
```bash
curl -F "file=@/absolute/path/to/sample.mp4" \
     -F "silence_db=-32" \
     -F "min_silence_ms=250" \
     http://localhost:8000/detect | jq
```

### Output (example)
```json
{
  "media": {
    "filename": "sample.mp4",
    "format": "mov,mp4,m4a,3gp,3g2,mj2",
    "duration_ms": 60231,
    "is_video": true
  },
  "video_candidates": [
    {"start_ms": 10250, "end_ms": 17200, "duration_ms": 6950, "cuts": 5}
  ],
  "audio_ads": [
    {"start_ms": 10500, "end_ms": 16850, "duration_ms": 6350}
  ]
}
```

### Notes on Accuracy
- Millisecond timestamps are produced, but absolute “ad vs. content” classification uses heuristics:
  - video: dense hard cuts
  - audio: musical texture + silence boundary alignment
- For broadcast/OTT-grade accuracy, replace the rule-based classifier with a trained model or template matching of known jingles, or integrate logo detection. This project exposes clean hooks to do so:
  - swap `_rule_based_musiciness` with a learned classifier (e.g., VGGish embeddings + classifier).
  - align to silences via `_silence_regions_ffmpeg` (already in place).