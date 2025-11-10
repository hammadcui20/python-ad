from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import cv2
import numpy as np
import os

app = FastAPI()

class Payload(BaseModel):
    video_dir: str            # directory containing all hourly video files
    ad_video_path: str
    threshold: float = 0.8
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


@app.post("/detect_ad")
def detect_ad(payload: Payload):
    # parse datetime
    start_dt = datetime.fromisoformat(payload.start_datetime)
    end_dt = datetime.fromisoformat(payload.end_datetime)
    if start_dt >= end_dt:
        return {"error": "start_datetime must be before end_datetime"}

    total_hours = int((end_dt - start_dt).total_seconds() // 3600)

    # fetch videos from directory (assuming sorted by timestamp in filename)
    all_videos = sorted([os.path.join(payload.video_dir, f)
                         for f in os.listdir(payload.video_dir)
                         if f.endswith(('.mp4','.avi'))])

    videos_to_process = all_videos[:total_hours]  # pick first N videos matching total hours

    template, (w, h) = extract_template(payload.ad_video_path)

    total_seconds = 0
    all_blocks = []
    for vid_path in videos_to_process:
        blocks, seconds = detect_in_video(vid_path, template, w, h, payload.threshold)
        total_seconds += seconds
        all_blocks.extend([{"video": os.path.basename(vid_path), **b} for b in blocks])

    return {
        "total_appearances": len(all_blocks),
        "total_ad_seconds": total_seconds,
        "occurrences": all_blocks,
        "total_hours_requested": total_hours
    }
