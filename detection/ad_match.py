# detection/ad_match.py
import cv2
import numpy as np
from typing import List, Dict, Tuple
import os


def extract_frames(video_path: str, fps: float = 1.0) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from video at specified FPS.
    Returns list of (timestamp_seconds, frame) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frames = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # Default fallback
    
    frame_skip = max(1, int(video_fps / fps)) if fps > 0 else 1
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            current_time = frame_count / video_fps
            frames.append((current_time, frame))
        
        frame_count += 1
    
    cap.release()
    return frames


def compute_frame_features(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ORB features and descriptors for a frame.
    Returns (keypoints, descriptors).
    """
    orb = cv2.ORB_create(nfeatures=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_frames(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.75) -> float:
    """
    Match two frame descriptors using BFMatcher.
    Returns match score (0-1).
    """
    if desc1 is None or desc2 is None:
        return 0.0
    
    if len(desc1) == 0 or len(desc2) == 0:
        return 0.0
    
    # Use BFMatcher for ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    try:
        # Need at least 2 descriptors for knnMatch
        if len(desc1) < 2 or len(desc2) < 2:
            # Use simple matching if not enough descriptors
            matches = bf.match(desc1, desc2)
            if not matches:
                return 0.0
            # Use average distance (inverted and normalized)
            avg_distance = np.mean([m.distance for m in matches])
            # Normalize: ORB distance is typically 0-256, convert to 0-1 score
            score = max(0.0, 1.0 - (avg_distance / 64.0))  # 64 is a reasonable threshold
            return min(1.0, score)
        
        matches = bf.knnMatch(desc1, desc2, k=2)
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # Calculate match score based on number of good matches
        max_possible = min(len(desc1), len(desc2))
        if max_possible == 0:
            return 0.0
        
        # Normalize by the smaller descriptor set
        score = len(good_matches) / max_possible
        
        # Also consider match quality (lower distance = better)
        if good_matches:
            avg_distance = np.mean([m.distance for m in good_matches])
            quality_factor = max(0.0, 1.0 - (avg_distance / 64.0))
            score = score * (0.7 + 0.3 * quality_factor)  # Weight by quality
        
        return min(1.0, score)
    except Exception as e:
        print(f"Error in match_frames: {e}")
        return 0.0


def compute_template_match_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute template matching score between two frames.
    Returns normalized score (0-1).
    """
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Resize to same size for comparison - use a standard size to avoid issues
        target_size = (320, 240)  # Standard size for comparison
        gray1 = cv2.resize(gray1, target_size)
        gray2 = cv2.resize(gray2, target_size)
        
        # Use normalized cross-correlation
        # Note: matchTemplate requires template to be smaller than image
        # Since we resized to same size, we'll use a different approach
        # Calculate structural similarity or use histogram comparison as fallback
        
        # Try direct comparison if sizes allow
        if gray1.shape == gray2.shape:
            # Use normalized cross-correlation on the full image
            # We can use cv2.matchTemplate with a slightly smaller template
            template_h, template_w = gray1.shape
            if template_h > 50 and template_w > 50:
                # Use a central region as template
                margin_h, margin_w = template_h // 10, template_w // 10
                template = gray1[margin_h:template_h-margin_h, margin_w:template_w-margin_w]
                result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
                score = float(np.max(result))
            else:
                # For small images, use direct correlation
                gray1_norm = gray1.astype(np.float32) / 255.0
                gray2_norm = gray2.astype(np.float32) / 255.0
                correlation = np.corrcoef(gray1_norm.flatten(), gray2_norm.flatten())[0, 1]
                score = float(max(0.0, correlation)) if not np.isnan(correlation) else 0.0
        else:
            # Fallback: histogram comparison
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            score = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Error in compute_template_match_score: {e}")
        return 0.0


def find_ad_occurrences(
    sample_video_path: str,
    target_video_path: str,
    match_threshold: float = 0.15,  # Lower default threshold for better detection
    frame_sampling_fps: float = 2.0,
    min_match_duration_seconds: float = 1.0,
    min_consecutive_matches: int = 2,  # Lower requirement for consecutive matches
) -> List[Dict]:
    """
    Find occurrences of sample advertisement in target video.
    
    Args:
        sample_video_path: Path to advertisement sample video
        target_video_path: Path to target video to search in
        match_threshold: Minimum similarity score to consider a match (0-1)
        frame_sampling_fps: FPS for frame extraction (lower = faster but less accurate)
        min_match_duration_seconds: Minimum duration for a valid occurrence
        min_consecutive_matches: Minimum consecutive frame matches to start tracking
    
    Returns:
        List of occurrences with start_ms, end_ms, duration_ms, and confidence
    """
    # Extract frames from both videos
    print(f"Extracting frames from sample video: {sample_video_path}")
    sample_frames = extract_frames(sample_video_path, fps=frame_sampling_fps)
    if not sample_frames:
        raise RuntimeError("No frames extracted from sample video")
    
    print(f"Extracting frames from target video: {target_video_path}")
    target_frames = extract_frames(target_video_path, fps=frame_sampling_fps)
    if not target_frames:
        raise RuntimeError("No frames extracted from target video")
    
    print(f"Sample frames: {len(sample_frames)}, Target frames: {len(target_frames)}")
    
    # Compute features for sample frames (use representative frames)
    # Use more frames for better matching - sample evenly throughout the video
    num_key_frames = min(10, len(sample_frames))
    if num_key_frames < len(sample_frames):
        step = max(1, len(sample_frames) // num_key_frames)
        sample_key_indices = list(range(0, len(sample_frames), step))[:num_key_frames]
    else:
        sample_key_indices = list(range(len(sample_frames)))
    
    sample_key_frames = [sample_frames[i] for i in sample_key_indices if i < len(sample_frames)]
    
    print(f"Computing features for {len(sample_key_frames)} key sample frames...")
    sample_features = []
    for _, frame in sample_key_frames:
        kp, desc = compute_frame_features(frame)
        sample_features.append((kp, desc, frame))
    
    # Match target frames against sample frames
    print("Matching frames...")
    match_scores = []
    max_score = 0.0
    min_score = 1.0
    scores_above_threshold = 0
    
    for idx, (target_time, target_frame) in enumerate(target_frames):
        target_kp, target_desc = compute_frame_features(target_frame)
        
        # Match against all sample key frames and take best score
        best_score = 0.0
        best_feature_score = 0.0
        best_template_score = 0.0
        
        for sample_kp, sample_desc, sample_frame in sample_features:
            # Feature matching
            feature_score = 0.0
            if sample_desc is not None and target_desc is not None:
                feature_score = match_frames(sample_desc, target_desc)
            
            # Template matching - compare with each sample frame
            template_score = compute_template_match_score(sample_frame, target_frame)
            
            # Combine scores (weighted average)
            combined_score = 0.6 * feature_score + 0.4 * template_score
            if combined_score > best_score:
                best_score = combined_score
                best_feature_score = feature_score
                best_template_score = template_score
        
        match_scores.append((target_time, best_score))
        max_score = max(max_score, best_score)
        min_score = min(min_score, best_score)
        if best_score >= match_threshold:
            scores_above_threshold += 1
        
        # Log first few matches and some statistics
        if idx < 10 or (idx % 100 == 0):
            print(f"  Frame {idx} (t={target_time:.2f}s): score={best_score:.3f} "
                  f"(feature={best_feature_score:.3f}, template={best_template_score:.3f})")
    
    print(f"Match score statistics: min={min_score:.3f}, max={max_score:.3f}, "
          f"threshold={match_threshold:.3f}, above_threshold={scores_above_threshold}/{len(match_scores)}")
    
    # If no scores above threshold, suggest a lower threshold
    if scores_above_threshold == 0 and len(match_scores) > 0:
        suggested_threshold = max(0.1, max_score * 0.8)
        print(f"WARNING: No matches found above threshold {match_threshold}. "
              f"Highest score was {max_score:.3f}. Consider using threshold <= {suggested_threshold:.3f}")
    
    # Find continuous sequences of matches
    occurrences = []
    in_match = False
    match_start_time = None
    match_start_index = None
    consecutive_matches = 0
    
    for i, (time, score) in enumerate(match_scores):
        if score >= match_threshold:
            if not in_match:
                consecutive_matches = 1
                match_start_time = time
                match_start_index = i
                in_match = True
            else:
                consecutive_matches += 1
        else:
            if in_match:
                # Check if we have enough consecutive matches
                if consecutive_matches >= min_consecutive_matches:
                    # Find end time (use next frame time or estimate)
                    if i < len(match_scores):
                        end_time = match_scores[i][0]
                    else:
                        # Last frame, estimate end
                        end_time = time + (1.0 / frame_sampling_fps)
                    
                    duration = end_time - match_start_time
                    if duration >= min_match_duration_seconds:
                        # Calculate average confidence
                        match_scores_in_range = [s for _, s in match_scores[match_start_index:i]]
                        avg_confidence = np.mean(match_scores_in_range) if match_scores_in_range else match_threshold
                        
                        occurrences.append({
                            "start_ms": int(match_start_time * 1000),
                            "end_ms": int(end_time * 1000),
                            "duration_ms": int(duration * 1000),
                            "confidence": float(avg_confidence),
                        })
                
                in_match = False
                consecutive_matches = 0
    
    # Handle case where match extends to end of video
    if in_match and consecutive_matches >= min_consecutive_matches:
        end_time = match_scores[-1][0] + (1.0 / frame_sampling_fps)
        duration = end_time - match_start_time
        if duration >= min_match_duration_seconds:
            match_scores_in_range = [s for _, s in match_scores[match_start_index:]]
            avg_confidence = np.mean(match_scores_in_range) if match_scores_in_range else match_threshold
            
            occurrences.append({
                "start_ms": int(match_start_time * 1000),
                "end_ms": int(end_time * 1000),
                "duration_ms": int(duration * 1000),
                "confidence": float(avg_confidence),
            })
    
    # Merge overlapping or very close occurrences
    if occurrences:
        occurrences.sort(key=lambda x: x["start_ms"])
        merged = [occurrences[0]]
        for occ in occurrences[1:]:
            last = merged[-1]
            # Merge if within 2 seconds of each other
            if occ["start_ms"] <= last["end_ms"] + 2000:
                last["end_ms"] = max(last["end_ms"], occ["end_ms"])
                last["duration_ms"] = last["end_ms"] - last["start_ms"]
                last["confidence"] = max(last["confidence"], occ["confidence"])
            else:
                merged.append(occ)
        occurrences = merged
    
    # Add human-readable timestamps
    for occ in occurrences:
        occ["start"] = _ms_to_timestamp(occ["start_ms"])
        occ["end"] = _ms_to_timestamp(occ["end_ms"])
        occ["interval"] = f"{occ['start']} - {occ['end']}"
    
    return occurrences


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    if ms < 0:
        ms = 0
    s, ms_rem = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"

