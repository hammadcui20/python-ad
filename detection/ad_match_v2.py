# detection/ad_match_v2.py
from dataclasses import dataclass, field
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


@dataclass
class AdMatchConfig:
    """Configuration for the ad matching process."""
    frame_sampling_fps: float = 2.0
    orb_features: int = 1000
    matcher_ratio_threshold: float = 0.75
    # Match score combination weights
    feature_match_weight: float = 0.6
    template_match_weight: float = 0.4
    # Match sequence detection
    min_consecutive_matches: int = 2
    min_match_duration_seconds: float = 1.0
    # Dynamic thresholding parameters
    enable_dynamic_threshold: bool = True
    dynamic_threshold_percentile: float = 95.0  # Use 95th percentile of scores
    fallback_static_threshold: float = 0.25  # Used if dynamic fails or is disabled
    # Post-processing
    merge_occurrences_gap_seconds: float = 2.0
    # Debugging
    debug_mode: bool = False
    debug_frame_output_dir: Optional[str] = None


@dataclass
class MatchResult:
    """Detailed result of a frame match."""
    target_time: float
    target_frame_index: int
    best_score: float
    feature_score: float
    template_score: float
    sample_frame_index: int


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    if ms < 0:
        ms = 0
    s, ms_rem = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def _save_debug_frame(
    frame: np.ndarray,
    output_dir: str,
    filename: str,
    text: Optional[str] = None,
):
    """Save a frame with optional text for debugging."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if text:
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    cv2.imwrite(os.path.join(output_dir, filename), frame)


class AdMatcher:
    """
    A class to find occurrences of a sample video within a target video.
    """
    def __init__(self, config: AdMatchConfig = AdMatchConfig()):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_ad_occurrences(
        self, sample_video_path: str, target_video_path: str
    ) -> List[Dict]:
        """
        Main entry point to find ad occurrences.
        """
        # 1. Extract frames
        sample_frames = self._extract_frames(sample_video_path)
        target_frames = self._extract_frames(target_video_path)

        # 2. Compute features for sample frames
        sample_features = self._compute_features_for_frames(sample_frames)

        # 3. Match target frames against sample frames
        match_results = self._match_all_frames(sample_features, target_frames)

        # 4. Determine match threshold
        match_threshold = self._determine_match_threshold(match_results)

        # 5. Find continuous sequences of matches
        occurrences = self._find_match_sequences(match_results, match_threshold)

        # 6. Post-process (merge) occurrences
        merged_occurrences = self._merge_occurrences(occurrences)

        # 7. Format final output
        return self._format_output(merged_occurrences)

    def _extract_frames(
        self, video_path: str
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at specified FPS."""
        logging.info(f"Extracting frames from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        frames = []
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        
        frame_skip = max(1, int(video_fps / self.config.frame_sampling_fps))
        
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
        logging.info(f"Extracted {len(frames)} frames.")
        return frames

    def _compute_features_for_frames(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> List[Dict]:
        """Compute ORB features for a list of frames."""
        logging.info(f"Computing features for {len(frames)} frames...")
        features = []
        for i, (timestamp, frame) in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            features.append({
                "frame_index": i,
                "timestamp": timestamp,
                "frame": frame,
                "gray": gray,
                "keypoints": keypoints,
                "descriptors": descriptors,
            })
        return features

    def _match_all_frames(
        self,
        sample_features: List[Dict],
        target_frames: List[Tuple[float, np.ndarray]],
    ) -> List[MatchResult]:
        """Match each target frame against all sample frames."""
        logging.info("Matching target frames against sample frames...")
        results = []
        
        for i, (target_time, target_frame) in enumerate(target_frames):
            target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
            _, target_desc = self.orb.detectAndCompute(target_gray, None)
            
            best_score = -1.0
            best_feature_score = -1.0
            best_template_score = -1.0
            best_sample_index = -1

            for sample in sample_features:
                feature_score = self._compute_feature_match(
                    sample["descriptors"], target_desc
                )
                template_score = self._compute_template_match(
                    sample["gray"], target_gray
                )
                
                combined_score = (
                    self.config.feature_match_weight * feature_score +
                    self.config.template_match_weight * template_score
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_feature_score = feature_score
                    best_template_score = template_score
                    best_sample_index = sample["frame_index"]
            
            results.append(
                MatchResult(
                    target_time=target_time,
                    target_frame_index=i,
                    best_score=best_score,
                    feature_score=best_feature_score,
                    template_score=best_template_score,
                    sample_frame_index=best_sample_index,
                )
            )
            if i % 100 == 0:
                logging.info(f"  Matched frame {i}/{len(target_frames)} - score: {best_score:.3f}")
        
        return results

    def _compute_feature_match(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> float:
        """Match two frame descriptors using BFMatcher."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0.0
        
        try:
            matches = self.bf.knnMatch(desc1, desc2, k=2)
            good_matches = [
                m for m, n in matches if m.distance < self.config.matcher_ratio_threshold * n.distance
            ]
            
            score = len(good_matches) / min(len(desc1), len(desc2))
            return min(1.0, score)
        except Exception:
            return 0.0

    def _compute_template_match(
        self, gray1: np.ndarray, gray2: np.ndarray
    ) -> float:
        """Compute normalized cross-correlation template matching score."""
        try:
            # Resize to a standard small size for fast and consistent matching
            h, w = 120, 160
            template = cv2.resize(gray1, (w, h))
            target = cv2.resize(gray2, (w, h))
            
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            score = float(np.max(result))
            return max(0.0, score)
        except Exception:
            return 0.0

    def _determine_match_threshold(self, results: List[MatchResult]) -> float:
        """Determine the match threshold, dynamically if enabled."""
        if not self.config.enable_dynamic_threshold:
            logging.info(f"Using fixed match threshold: {self.config.fallback_static_threshold}")
            return self.config.fallback_static_threshold

        if not results:
            return self.config.fallback_static_threshold

        scores = [r.best_score for r in results]
        if not scores:
            return self.config.fallback_static_threshold

        # Use percentile to find a good threshold
        threshold = np.percentile(scores, self.config.dynamic_threshold_percentile)
        
        # Sanity check: ensure threshold is not too low or too high
        threshold = max(threshold, 0.15)  # Minimum sensible threshold
        threshold = min(threshold, 0.9)   # Maximum sensible threshold

        logging.info(f"Determined dynamic match threshold: {threshold:.3f}")
        return threshold

    def _find_match_sequences(
        self, results: List[MatchResult], threshold: float
    ) -> List[Dict]:
        """Find continuous sequences of high-score matches."""
        logging.info("Finding match sequences...")
        occurrences = []
        in_match = False
        match_start_time = 0.0
        match_start_index = -1

        for i, res in enumerate(results):
            is_match = res.best_score >= threshold
            
            if is_match and not in_match:
                in_match = True
                match_start_time = res.target_time
                match_start_index = i
            elif not is_match and in_match:
                # End of a potential match sequence
                duration = results[i-1].target_time - match_start_time
                num_matches = i - match_start_index
                
                if (
                    duration >= self.config.min_match_duration_seconds and
                    num_matches >= self.config.min_consecutive_matches
                ):
                    match_scores = [r.best_score for r in results[match_start_index:i]]
                    avg_confidence = np.mean(match_scores)
                    
                    occurrences.append({
                        "start_ms": int(match_start_time * 1000),
                        "end_ms": int(results[i-1].target_time * 1000),
                        "confidence": float(avg_confidence),
                    })
                in_match = False

        # Handle case where match extends to the end of the video
        if in_match:
            duration = results[-1].target_time - match_start_time
            num_matches = len(results) - match_start_index
            if (
                duration >= self.config.min_match_duration_seconds and
                num_matches >= self.config.min_consecutive_matches
            ):
                match_scores = [r.best_score for r in results[match_start_index:]]
                avg_confidence = np.mean(match_scores)
                occurrences.append({
                    "start_ms": int(match_start_time * 1000),
                    "end_ms": int(results[-1].target_time * 1000),
                    "confidence": float(avg_confidence),
                })
        
        logging.info(f"Found {len(occurrences)} potential occurrences.")
        return occurrences

    def _merge_occurrences(self, occurrences: List[Dict]) -> List[Dict]:
        """Merge overlapping or very close occurrences."""
        if not occurrences:
            return []

        occurrences.sort(key=lambda x: x["start_ms"])
        merged = [occurrences[0]]
        
        for occ in occurrences[1:]:
            last = merged[-1]
            gap = occ["start_ms"] - last["end_ms"]
            
            if gap <= self.config.merge_occurrences_gap_seconds * 1000:
                # Merge
                last["end_ms"] = max(last["end_ms"], occ["end_ms"])
                # Take the higher confidence of the merged segments
                last["confidence"] = max(last["confidence"], occ["confidence"])
            else:
                merged.append(occ)
        
        logging.info(f"Merged occurrences: {len(occurrences)} -> {len(merged)}")
        return merged

    def _format_output(self, occurrences: List[Dict]) -> List[Dict]:
        """Add human-readable fields to the final output."""
        for occ in occurrences:
            occ["duration_ms"] = occ["end_ms"] - occ["start_ms"]
            occ["start"] = _ms_to_timestamp(occ["start_ms"])
            occ["end"] = _ms_to_timestamp(occ["end_ms"])
            occ["interval"] = f"{occ['start']} - {occ['end']}"
        return occurrences
