"""Anomaly Detection module for AMR Intelligence System using Isolation Forest.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "preprocessors", "isolation_forest.joblib")


def fit_and_save_anomaly_detector(X_train, save_path=ANOMALY_MODEL_PATH, random_state=42):
    """Fit Isolation Forest on baseline training feature distribution."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    detector = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=random_state,
        n_jobs=-1
    )
    detector.fit(X_train)
    joblib.dump(detector, save_path)
    return detector


def load_anomaly_detector(model_path=ANOMALY_MODEL_PATH):
    """Load the persisted Isolation Forest detector."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Anomaly detector not found at {model_path}. Run training first.")
    return joblib.load(model_path)


def detect_anomaly(X_sample, detector=None):
    """Evaluate whether an input sample is normal or unusual relative to baseline reference data.
    
    Returns:
        dict: {
            'is_anomaly': bool,
            'status': str ('Normal relative to reference data' or 'Unusual relative to reference data'),
            'anomaly_score': float, # raw decision score, negative indicates anomaly
            'normality_index': float # 0 to 1 normalized normality percentage
        }
    """
    if detector is None:
        detector = load_anomaly_detector()

    if isinstance(X_sample, (list, tuple)):
        X_sample = np.array(X_sample)
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    raw_score = float(detector.decision_function(X_sample)[0])
    prediction = int(detector.predict(X_sample)[0])  # +1 = normal, -1 = anomaly
    
    # Sigmoid normalization for human-readable normality percentage
    normality_index = float(1.0 / (1.0 + np.exp(-5.0 * raw_score)))

    is_anomaly = (prediction == -1)
    status = "Unusual relative to reference data" if is_anomaly else "Normal relative to reference data"

    return {
        "is_anomaly": is_anomaly,
        "status": status,
        "anomaly_score": round(raw_score, 4),
        "normality_index": round(normality_index, 4)
    }
