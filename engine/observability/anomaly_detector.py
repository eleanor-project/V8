"""
ELEANOR V8 — Anomaly Detection
-------------------------------

Detect anomalies in system behavior using ML-based detection.
Enables proactive problem detection before issues escalate.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    logger.warning("scikit-learn not available. Anomaly detection will use statistical methods.")


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    metric_name: str
    anomaly_score: float  # 0.0 to 1.0 (higher = more anomalous)
    severity: AnomalySeverity
    current_value: float
    expected_range: Tuple[float, float]
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detector (fallback when ML not available).
    
    Uses z-score and percentile-based detection.
    """
    
    def __init__(self, z_threshold: float = 3.0):
        """Initialize statistical detector."""
        self.z_threshold = z_threshold
        self.historical_data: Dict[str, List[float]] = {}
    
    def fit(self, metric_name: str, values: List[float]) -> None:
        """Fit detector with historical data."""
        self.historical_data[metric_name] = values
    
    def detect(self, metric_name: str, current_value: float) -> Optional[Anomaly]:
        """Detect anomaly using statistical methods."""
        if metric_name not in self.historical_data:
            return None
        
        historical = self.historical_data[metric_name]
        if len(historical) < 10:
            return None  # Need at least 10 samples
        
        # Calculate statistics
        mean = statistics.mean(historical)
        std_dev = statistics.stdev(historical) if len(historical) > 1 else 0.0
        
        if std_dev == 0:
            return None
        
        # Calculate z-score
        z_score = abs((current_value - mean) / std_dev)
        
        # Calculate percentile
        percentile = (sum(1 for v in historical if v <= current_value) / len(historical)) * 100
        
        # Determine if anomaly
        is_anomaly = z_score > self.z_threshold
        
        if not is_anomaly:
            return None
        
        # Calculate anomaly score (0.0 to 1.0)
        anomaly_score = min(1.0, z_score / (self.z_threshold * 2))
        
        # Determine severity
        if z_score >= self.z_threshold * 2:
            severity = AnomalySeverity.CRITICAL
        elif z_score >= self.z_threshold * 1.5:
            severity = AnomalySeverity.HIGH
        elif z_score >= self.z_threshold * 1.2:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
        
        # Expected range (mean ± 2 std devs)
        expected_range = (mean - 2 * std_dev, mean + 2 * std_dev)
        
        description = (
            f"Value {current_value:.2f} is {z_score:.2f} standard deviations from mean "
            f"({mean:.2f}). Percentile: {percentile:.1f}%"
        )
        
        recommendations = self._generate_recommendations(metric_name, z_score, percentile)
        
        return Anomaly(
            metric_name=metric_name,
            anomaly_score=anomaly_score,
            severity=severity,
            current_value=current_value,
            expected_range=expected_range,
            description=description,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        metric_name: str,
        z_score: float,
        percentile: float
    ) -> List[str]:
        """Generate recommendations based on anomaly."""
        recommendations = []
        
        if "latency" in metric_name.lower() or "duration" in metric_name.lower():
            if percentile > 95:
                recommendations.append("Investigate performance degradation")
                recommendations.append("Check for resource constraints")
            else:
                recommendations.append("Check for network issues")
                recommendations.append("Review recent code changes")
        
        elif "error" in metric_name.lower() or "failure" in metric_name.lower():
            recommendations.append("Review error logs for patterns")
            recommendations.append("Check dependent services")
            recommendations.append("Consider scaling resources")
        
        elif "cost" in metric_name.lower():
            recommendations.append("Review model selection strategy")
            recommendations.append("Check for inefficient queries")
        
        else:
            recommendations.append("Review system metrics")
            recommendations.append("Check for related anomalies")
        
        return recommendations


class MLAnomalyDetector:
    """
    ML-based anomaly detector using Isolation Forest.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize ML anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MLAnomalyDetector")
        
        self.contamination = contamination
        self.model: Optional[Any] = None
        self.metric_features: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.is_trained = False
    
    def fit(self, historical_metrics: List[Dict[str, float]]) -> None:
        """
        Train anomaly detection model.
        
        Args:
            historical_metrics: List of metric dictionaries (one per time period)
        """
        if not historical_metrics:
            logger.warning("No historical data provided for training")
            return
        
        # Extract feature names
        all_keys = set()
        for metrics in historical_metrics:
            all_keys.update(metrics.keys())
        
        self.feature_names = sorted(all_keys)
        self.metric_features = {name: i for i, name in enumerate(self.feature_names)}
        
        # Build feature matrix
        features = []
        for metrics in historical_metrics:
            feature_vector = [metrics.get(name, 0.0) for name in self.feature_names]
            features.append(feature_vector)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(features)
        self.is_trained = True
        
        logger.info(
            f"Trained ML anomaly detector with {len(features)} samples and "
            f"{len(self.feature_names)} features"
        )
    
    def detect_anomalies(
        self,
        current_metrics: Dict[str, float]
    ) -> List[Anomaly]:
        """
        Detect anomalies in current metrics.
        
        Args:
            current_metrics: Current metric values
            
        Returns:
            List of detected anomalies
        """
        if not self.is_trained or not self.model:
            logger.warning("Model not trained, cannot detect anomalies")
            return []
        
        # Build feature vector (use 0.0 for missing metrics)
        feature_vector = [
            current_metrics.get(name, 0.0) for name in self.feature_names
        ]
        
        # Predict
        prediction = self.model.predict([feature_vector])[0]
        score = self.model.score_samples([feature_vector])[0]
        
        # Convert score to anomaly score (lower score = more anomalous)
        # Isolation Forest scores are negative for anomalies
        anomaly_score = abs(min(0, score)) if score < 0 else 0.0
        
        if prediction == -1:  # Anomaly detected
            # Determine which metrics are contributing most
            anomalies = []
            
            # For each metric, check if it's significantly different from typical values
            # This is a simplified approach - could use SHAP values for better attribution
            for metric_name, value in current_metrics.items():
                if metric_name not in self.metric_features:
                    continue
                
                # Simple heuristic: if value is very different from 0 (typical after normalization)
                # and score indicates anomaly, flag it
                if abs(value) > 2.0:  # Arbitrary threshold
                    severity = (
                        AnomalySeverity.CRITICAL if abs(value) > 4.0 else
                        AnomalySeverity.HIGH if abs(value) > 3.0 else
                        AnomalySeverity.MEDIUM if abs(value) > 2.5 else
                        AnomalySeverity.LOW
                    )
                    
                    anomalies.append(Anomaly(
                        metric_name=metric_name,
                        anomaly_score=anomaly_score,
                        severity=severity,
                        current_value=value,
                        expected_range=(-2.0, 2.0),  # Approximate range after normalization
                        description=f"ML model detected anomaly in {metric_name}",
                        recommendations=self._generate_recommendations(metric_name)
                    ))
            
            return anomalies if anomalies else [
                Anomaly(
                    metric_name="system",
                    anomaly_score=anomaly_score,
                    severity=AnomalySeverity.MEDIUM,
                    current_value=score,
                    expected_range=(-1.0, 0.0),
                    description="ML model detected system-level anomaly",
                    recommendations=["Review all system metrics", "Check logs for errors"]
                )
            ]
        
        return []
    
    def _generate_recommendations(self, metric_name: str) -> List[str]:
        """Generate recommendations for metric."""
        recommendations = []
        
        if "latency" in metric_name.lower():
            recommendations.extend([
                "Check for performance bottlenecks",
                "Review resource utilization",
                "Investigate recent deployments"
            ])
        elif "error" in metric_name.lower():
            recommendations.extend([
                "Review error logs",
                "Check dependent services",
                "Verify recent code changes"
            ])
        else:
            recommendations.append("Investigate metric deviation")
        
        return recommendations


class AnomalyDetector:
    """
    Unified anomaly detector with ML and statistical fallback.
    """
    
    def __init__(
        self,
        use_ml: bool = True,
        contamination: float = 0.1,
        z_threshold: float = 3.0
    ):
        """
        Initialize anomaly detector.
        
        Args:
            use_ml: Whether to use ML-based detection (if available)
            contamination: Expected anomaly proportion for ML model
            z_threshold: Z-score threshold for statistical detection
        """
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        self.ml_detector: Optional[MLAnomalyDetector] = None
        self.statistical_detector = StatisticalAnomalyDetector(z_threshold=z_threshold)
        
        if self.use_ml:
            try:
                self.ml_detector = MLAnomalyDetector(contamination=contamination)
            except Exception as e:
                logger.warning(f"Failed to initialize ML detector: {e}, using statistical fallback")
                self.use_ml = False
        
        logger.info(
            f"Initialized AnomalyDetector (ML: {self.use_ml}, Statistical: True)"
        )
    
    def train(self, historical_metrics: List[Dict[str, float]]) -> None:
        """Train detector with historical data."""
        if self.use_ml and self.ml_detector:
            self.ml_detector.fit(historical_metrics)
        
        # Also train statistical detector for individual metrics
        if historical_metrics:
            all_metrics = set()
            for metrics in historical_metrics:
                all_metrics.update(metrics.keys())
            
            for metric_name in all_metrics:
                values = [m.get(metric_name, 0.0) for m in historical_metrics if metric_name in m]
                if len(values) >= 10:
                    self.statistical_detector.fit(metric_name, values)
    
    def detect_anomalies(
        self,
        current_metrics: Dict[str, float]
    ) -> List[Anomaly]:
        """Detect anomalies in current metrics."""
        anomalies = []
        
        # Try ML detection first
        if self.use_ml and self.ml_detector and self.ml_detector.is_trained:
            try:
                ml_anomalies = self.ml_detector.detect_anomalies(current_metrics)
                anomalies.extend(ml_anomalies)
            except Exception as e:
                logger.error(f"ML anomaly detection failed: {e}", exc_info=True)
        
        # Also check individual metrics with statistical detector
        for metric_name, value in current_metrics.items():
            stat_anomaly = self.statistical_detector.detect(metric_name, value)
            if stat_anomaly:
                # Avoid duplicates (check if ML already detected this)
                if not any(a.metric_name == metric_name for a in anomalies):
                    anomalies.append(stat_anomaly)
        
        # Sort by severity (critical first)
        severity_order = {
            AnomalySeverity.CRITICAL: 0,
            AnomalySeverity.HIGH: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 3
        }
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))
        
        return anomalies


__all__ = [
    "AnomalyDetector",
    "MLAnomalyDetector",
    "StatisticalAnomalyDetector",
    "Anomaly",
    "AnomalySeverity",
]
