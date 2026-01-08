"""
ELEANOR V8 — ML-Based Anomaly Detection
----------------------------------------

Lightweight anomaly detection for unusual patterns in sanitized data.
Uses statistical methods and pattern analysis to detect anomalies without heavy ML dependencies.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Anomaly detection score and metadata."""
    score: float  # 0.0 (normal) to 1.0 (highly anomalous)
    confidence: float  # 0.0 to 1.0
    reasons: List[str] = field(default_factory=list)
    pattern_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternAnomalyDetector:
    """
    Lightweight anomaly detector using statistical pattern analysis.
    
    Detects anomalies based on:
    - Unusual character distributions
    - Suspicious pattern combinations
    - Frequency-based analysis
    - Entropy analysis
    - Pattern deviation from baseline
    """
    
    def __init__(
        self,
        enable_ml: bool = True,
        baseline_window: int = 1000,
        anomaly_threshold: float = 0.7,
    ):
        """
        Initialize anomaly detector.
        
        Args:
            enable_ml: Enable ML-based anomaly detection
            baseline_window: Number of samples to use for baseline
            anomaly_threshold: Score threshold for flagging anomalies (0.0-1.0)
        """
        self.enable_ml = enable_ml
        self.baseline_window = baseline_window
        self.anomaly_threshold = anomaly_threshold
        
        # Baseline statistics
        self._baseline_patterns: deque = deque(maxlen=baseline_window)
        self._baseline_entropy: deque = deque(maxlen=baseline_window)
        self._baseline_char_dist: Dict[str, float] = {}
        self._pattern_frequencies: Dict[str, int] = defaultdict(int)
        self._sample_count = 0
        
        # Suspicious pattern combinations
        self._suspicious_combinations = [
            (r"eval\s*\(", r"exec\s*\("),  # eval + exec
            (r"base64", r"decode"),  # base64 encoding
            (r"<script", r"javascript:"),  # XSS patterns
            (r"union", r"select"),  # SQL injection
            (r"\.\./", r"\.\.\\"),  # Path traversal
        ]
        
        logger.info(
            "anomaly_detector_initialized",
            extra={
                "enable_ml": enable_ml,
                "baseline_window": baseline_window,
                "anomaly_threshold": anomaly_threshold,
            },
        )
    
    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> AnomalyScore:
        """
        Analyze text for anomalies.
        
        Args:
            text: Text to analyze
            context: Optional context information
        
        Returns:
            AnomalyScore with detection results
        """
        if not self.enable_ml or not text:
            return AnomalyScore(score=0.0, confidence=0.0)
        
        reasons = []
        scores = []
        
        # 1. Character distribution analysis
        char_dist_score = self._analyze_character_distribution(text)
        if char_dist_score > 0.5:
            reasons.append(f"Unusual character distribution (score: {char_dist_score:.2f})")
            scores.append(char_dist_score)
        
        # 2. Entropy analysis
        entropy_score = self._analyze_entropy(text)
        if entropy_score > 0.5:
            reasons.append(f"Unusual entropy pattern (score: {entropy_score:.2f})")
            scores.append(entropy_score)
        
        # 3. Suspicious pattern combinations
        combo_score = self._detect_suspicious_combinations(text)
        if combo_score > 0.5:
            reasons.append(f"Suspicious pattern combination detected (score: {combo_score:.2f})")
            scores.append(combo_score)
        
        # 4. Pattern frequency deviation
        freq_score = self._analyze_pattern_frequency(text)
        if freq_score > 0.5:
            reasons.append(f"Unusual pattern frequency (score: {freq_score:.2f})")
            scores.append(freq_score)
        
        # 5. Encoding anomalies
        encoding_score = self._detect_encoding_anomalies(text)
        if encoding_score > 0.5:
            reasons.append(f"Encoding anomaly detected (score: {encoding_score:.2f})")
            scores.append(encoding_score)
        
        # Calculate overall score
        if scores:
            overall_score = max(scores)  # Use maximum for conservative detection
            confidence = min(1.0, len(scores) * 0.2)  # Higher confidence with more indicators
        else:
            overall_score = 0.0
            confidence = 0.0
        
        # Update baseline
        self._update_baseline(text)
        
        return AnomalyScore(
            score=overall_score,
            confidence=confidence,
            reasons=reasons,
            pattern_type=self._classify_pattern_type(text),
            metadata={
                "char_dist_score": char_dist_score,
                "entropy_score": entropy_score,
                "combo_score": combo_score,
                "freq_score": freq_score,
                "encoding_score": encoding_score,
            },
        )
    
    def _analyze_character_distribution(self, text: str) -> float:
        """Analyze character distribution for anomalies."""
        if len(text) < 10:
            return 0.0
        
        # Calculate character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        total_chars = len(text)
        char_probs = {char: count / total_chars for char, count in char_counts.items()}
        
        # Compare to baseline
        if not self._baseline_char_dist:
            return 0.0
        
        # Calculate KL divergence or simple difference
        score = 0.0
        for char, prob in char_probs.items():
            baseline_prob = self._baseline_char_dist.get(char, 0.0)
            if baseline_prob > 0:
                # Calculate deviation
                deviation = abs(prob - baseline_prob) / baseline_prob
                score += deviation
        
        # Normalize
        normalized_score = min(1.0, score / len(char_probs))
        return normalized_score
    
    def _analyze_entropy(self, text: str) -> float:
        """Analyze text entropy for anomalies."""
        if len(text) < 10:
            return 0.0
        
        # Calculate Shannon entropy
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        entropy = 0.0
        length = len(text)
        for count in char_counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Normalize entropy (max entropy for ASCII is ~7)
        normalized_entropy = entropy / 7.0
        
        # Compare to baseline
        if len(self._baseline_entropy) < 10:
            return 0.0
        
        baseline_entropy = sum(self._baseline_entropy) / len(self._baseline_entropy)
        entropy_deviation = abs(normalized_entropy - baseline_entropy)
        
        # High deviation indicates anomaly
        return min(1.0, entropy_deviation * 2.0)
    
    def _detect_suspicious_combinations(self, text: str) -> float:
        """Detect suspicious pattern combinations."""
        text_lower = text.lower()
        max_score = 0.0
        
        for pattern1, pattern2 in self._suspicious_combinations:
            match1 = bool(re.search(pattern1, text_lower, re.IGNORECASE))
            match2 = bool(re.search(pattern2, text_lower, re.IGNORECASE))
            
            if match1 and match2:
                # Both patterns present - high anomaly score
                max_score = max(max_score, 0.9)
            elif match1 or match2:
                # One pattern - moderate score
                max_score = max(max_score, 0.5)
        
        return max_score
    
    def _analyze_pattern_frequency(self, text: str) -> float:
        """Analyze pattern frequency against baseline."""
        if len(self._baseline_patterns) < 10:
            return 0.0
        
        # Extract key patterns from text
        patterns = self._extract_patterns(text)
        
        # Compare to baseline frequencies
        score = 0.0
        for pattern in patterns:
            baseline_freq = self._pattern_frequencies.get(pattern, 0)
            current_freq = 1  # Pattern found once in current text
            
            if baseline_freq > 0:
                # Calculate frequency deviation
                freq_ratio = current_freq / (baseline_freq / len(self._baseline_patterns))
                if freq_ratio > 2.0:  # More than 2x baseline frequency
                    score += 0.3
        
        return min(1.0, score)
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract key patterns from text."""
        patterns = []
        
        # Extract common suspicious patterns
        suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"base64",
            r"<script",
            r"javascript:",
            r"union\s+select",
            r"\.\./",
            r"\.\.\\",
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns.append(pattern)
        
        return patterns
    
    def _detect_encoding_anomalies(self, text: str) -> float:
        """Detect encoding anomalies (mixed encodings, unusual bytes)."""
        score = 0.0
        
        # Check for mixed encodings
        try:
            text.encode('ascii')
            # Pure ASCII - check for unusual control characters
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
            if control_chars > len(text) * 0.1:  # More than 10% control chars
                score += 0.5
        except UnicodeEncodeError:
            # Non-ASCII - check for unusual Unicode patterns
            try:
                text.encode('utf-8')
                # Check for unusual Unicode ranges
                unusual_unicode = sum(1 for c in text if ord(c) > 0xFFFF)
                if unusual_unicode > len(text) * 0.05:  # More than 5% unusual Unicode
                    score += 0.4
            except:
                score += 0.6  # Encoding issues
        
        # Check for URL encoding anomalies
        url_encoded = text.count('%')
        if url_encoded > len(text) * 0.1:  # More than 10% URL encoded
            score += 0.3
        
        return min(1.0, score)
    
    def _classify_pattern_type(self, text: str) -> str:
        """Classify the type of pattern detected."""
        text_lower = text.lower()
        
        if re.search(r"eval|exec|base64", text_lower):
            return "code_injection"
        elif re.search(r"<script|javascript:", text_lower):
            return "xss"
        elif re.search(r"union|select|insert|delete", text_lower):
            return "sql_injection"
        elif re.search(r"\.\./|\.\.\\", text_lower):
            return "path_traversal"
        elif re.search(r"%[0-9a-fA-F]{2}", text_lower):
            return "encoding_anomaly"
        else:
            return "statistical_anomaly"
    
    def _update_baseline(self, text: str) -> None:
        """Update baseline statistics with new sample."""
        if not text:
            return
        
        self._sample_count += 1
        
        # Update pattern frequencies
        patterns = self._extract_patterns(text)
        for pattern in patterns:
            self._pattern_frequencies[pattern] += 1
        
        # Update character distribution baseline
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        total_chars = len(text)
        for char, count in char_counts.items():
            current_prob = self._baseline_char_dist.get(char, 0.0)
            new_prob = count / total_chars
            # Exponential moving average
            alpha = 0.1
            self._baseline_char_dist[char] = alpha * new_prob + (1 - alpha) * current_prob
        
        # Update entropy baseline
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        entropy = 0.0
        length = len(text)
        for count in char_counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        normalized_entropy = entropy / 7.0
        self._baseline_entropy.append(normalized_entropy)
        
        # Store pattern for frequency analysis
        self._baseline_patterns.append(text[:100])  # Store first 100 chars
    
    def is_anomalous(self, text: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if text is anomalous (convenience method).
        
        Args:
            text: Text to check
            context: Optional context information
        
        Returns:
            True if anomalous, False otherwise
        """
        score = self.analyze(text, context)
        return score.score >= self.anomaly_threshold
    
    def get_baseline_stats(self) -> Dict[str, Any]:
        """Get baseline statistics for monitoring."""
        return {
            "sample_count": self._sample_count,
            "baseline_patterns_count": len(self._baseline_patterns),
            "baseline_entropy_avg": (
                sum(self._baseline_entropy) / len(self._baseline_entropy)
                if self._baseline_entropy
                else 0.0
            ),
            "pattern_frequencies": dict(self._pattern_frequencies),
            "char_distribution_size": len(self._baseline_char_dist),
        }


# Global detector instance
_global_detector: Optional[PatternAnomalyDetector] = None


def get_anomaly_detector(
    enable_ml: bool = True,
    baseline_window: int = 1000,
    anomaly_threshold: float = 0.7,
) -> PatternAnomalyDetector:
    """
    Get or create global anomaly detector instance.
    
    Args:
        enable_ml: Enable ML-based anomaly detection
        baseline_window: Number of samples for baseline
        anomaly_threshold: Score threshold for flagging anomalies
    
    Returns:
        PatternAnomalyDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = PatternAnomalyDetector(
            enable_ml=enable_ml,
            baseline_window=baseline_window,
            anomaly_threshold=anomaly_threshold,
        )
    return _global_detector


__all__ = [
    "AnomalyScore",
    "PatternAnomalyDetector",
    "get_anomaly_detector",
]
