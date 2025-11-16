from typing import List, Dict
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_manager import VideoManager
from scenedetect.stats_manager import StatsManager


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    if ms < 0:
        ms = 0
    s, ms_rem = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def detect_video_ad_candidates(
    video_path: str,
    min_block_ms: int = 5000,
    window_ms: int = 5000,
    min_cuts_in_window: int = 3,
) -> List[Dict]:
    """
    Heuristic: commercials often have rapid cuts. We:
      - detect hard cuts with PySceneDetect
      - slide a window collecting cut timestamps
      - group windows with >= min_cuts_in_window into contiguous candidate blocks

    Returns list of {start_ms, end_ms, duration_ms, cuts:int}
    """
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=27.0))  # tuneable

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Cut list expressed as timestamps in milliseconds
        scene_list = scene_manager.get_scene_list()
        cut_ms = []
        for i, (start, end) in enumerate(scene_list):
            if i == 0:
                # first scene "cut" at 0
                cut_ms.append(int(start.get_seconds() * 1000))
            cut_ms.append(int(end.get_seconds() * 1000))

        if not cut_ms:
            return []

        cut_ms = sorted(set(cut_ms))
        # Build rolling windows
        candidates = []
        i = 0
        n = len(cut_ms)
        w = window_ms
        while i < n:
            start = cut_ms[i]
            j = i
            while j < n and cut_ms[j] - start <= w:
                j += 1
            count = j - i
            if count >= min_cuts_in_window:
                # Extend contiguous windows
                block_start = start
                block_end = cut_ms[j - 1]
                k = j
                while k < n:
                    # Try to extend if next window still dense
                    next_start = cut_ms[k]
                    m = k
                    while m < n and cut_ms[m] - next_start <= w:
                        m += 1
                    if (m - k) >= min_cuts_in_window and (next_start - block_end) <= w:
                        block_end = cut_ms[m - 1]
                        k = m
                    else:
                        break
                if (block_end - block_start) >= min_block_ms:
                    candidates.append(
                        {
                            "start_ms": block_start,
                            "end_ms": block_end,
                            "duration_ms": block_end - block_start,
                            "cuts": count,
                        }
                    )
                i = k
            else:
                i += 1

        # Merge overlapping candidates
        if not candidates:
            return []

        candidates.sort(key=lambda x: x["start_ms"])
        merged = [candidates[0]]
        for c in candidates[1:]:
            last = merged[-1]
            if c["start_ms"] <= last["end_ms"]:
                last["end_ms"] = max(last["end_ms"], c["end_ms"])
                last["duration_ms"] = last["end_ms"] - last["start_ms"]
                last["cuts"] += c.get("cuts", 0)
            else:
                merged.append(c)

        # Attach human-readable timestamps
        for seg in merged:
            seg_start = int(seg["start_ms"]) if isinstance(seg["start_ms"], (int, float)) else int(seg["start_ms"])
            seg_end = int(seg["end_ms"]) if isinstance(seg["end_ms"], (int, float)) else int(seg["end_ms"])
            seg["start"] = _ms_to_timestamp(seg_start)
            seg["end"] = _ms_to_timestamp(seg_end)
            seg["interval"] = f"{seg['start']} - {seg['end']}"

        return merged
    finally:
        video_manager.release()