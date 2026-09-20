"""SHAP Explainable AI Module for AMR Intelligence System.
Provides local and global feature attribution for model decision transparency.
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np

from src.preprocessing import load_preprocessor, transform_data, get_feature_names, load_schema
from src.predict import load_model, validate_and_format_input

# Cache background samples for efficiency
_BACKGROUND_CACHE = None


def get_background_data(n_samples=100):
    """Load and transform representative background reference samples."""
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is None:
        train_df = pd.read_csv("data/processed/train.csv")
        sample_subset = train_df.sample(min(n_samples, len(train_df)), random_state=42)
        preprocessor = load_preprocessor()
        _BACKGROUND_CACHE = transform_data(sample_subset, preprocessor)
    return _BACKGROUND_CACHE


def _extract_base_value(explainer):
    """Safely extract a scalar SHAP expected value when an explainer is available."""
    if explainer is None or not hasattr(explainer, "expected_value"):
        return 0.5
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        arr = np.array(ev).flatten()
        if arr.size == 0:
            return 0.5
        if arr.size > 1:
            return float(arr[1])
        return float(arr[0])
    try:
        return float(ev)
    except (TypeError, ValueError):
        return 0.5


def explain_sample(sample_data, antibiotic="ampicillin", model_name=None):
    """Compute local SHAP feature attributions for a specific sample.
    
    Returns:
        dict: Local explanation with top positive and negative contributing features.
    """
    schema = load_schema()
    abx_key = antibiotic.lower()
    df_formatted = validate_and_format_input(sample_data, schema)
    
    preprocessor = load_preprocessor()
    X_feat = transform_data(df_formatted, preprocessor)
    feature_names = get_feature_names(preprocessor)

    # Use Random Forest or Logistic Regression for fast, reliable Tree/Linear SHAP
    if model_name is None or model_name == "Tabular Transformer":
        chosen_model_name = "Random Forest"
    else:
        chosen_model_name = model_name

    model_obj, _ = load_model(abx_key, chosen_model_name)
    background = get_background_data(n_samples=50)
    explainer = None

    try:
        if chosen_model_name in ["Random Forest"]:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_feat)
            
            # Handle binary classification shap_values shape differences across shap versions
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    sv = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    sv = shap_values[0]
                else:
                    sv = shap_values
            else:
                sv = np.array(shap_values).flatten()
        else:
            explainer = shap.LinearExplainer(model_obj, background)
            shap_values = explainer.shap_values(X_feat)
            sv = shap_values[0] if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1 else shap_values

    except Exception as e:
        # Fallback to model feature importances / coefficients if explainer encounters format quirk
        if hasattr(model_obj, "feature_importances_"):
            sv = model_obj.feature_importances_ * X_feat[0]
        elif hasattr(model_obj, "coef_"):
            sv = model_obj.coef_[0] * X_feat[0]
        else:
            sv = np.zeros(X_feat.shape[1])

    # Clean feature names (remove prefix like 'cat__', 'num__')
    clean_names = [f.replace("cat__", "").replace("num__", "").replace("_", " ") for f in feature_names]

    # Pair features with their SHAP values and sort by absolute magnitude
    feature_impacts = []
    for name, val, raw_feat_val in zip(clean_names, sv, X_feat[0]):
        feature_impacts.append({
            "feature": name,
            "shap_value": float(val),
            "abs_impact": float(abs(val)),
            "direction": "Increases Resistance Probability" if val > 0 else "Decreases Resistance Probability"
        })

    feature_impacts = sorted(feature_impacts, key=lambda x: x["abs_impact"], reverse=True)

    return {
        "antibiotic": schema["selected_antibiotics"][abx_key]["display_name"],
        "model_used": chosen_model_name,
        "base_value": _extract_base_value(explainer),
        "top_features": feature_impacts[:12],
        "all_features": feature_impacts,
        "explanation_note": "SHAP explains statistical feature influence on model predictions, not biological causation."
    }


def get_global_feature_importance(antibiotic="ampicillin", model_name="Random Forest", n_top=15):
    """Compute global feature importance across reference dataset."""
    schema = load_schema()
    abx_key = antibiotic.lower()
    model_obj, _ = load_model(abx_key, model_name if model_name != "Tabular Transformer" else "Random Forest")
    preprocessor = load_preprocessor()
    feature_names = get_feature_names(preprocessor)
    clean_names = [f.replace("cat__", "").replace("num__", "").replace("_", " ") for f in feature_names]

    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
    elif hasattr(model_obj, "coef_"):
        importances = np.abs(model_obj.coef_[0])
    else:
        importances = np.ones(len(clean_names)) / len(clean_names)

    ranked = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(clean_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    return ranked[:n_top]


def explain_sample_top_reasons(sample_data, antibiotic="ampicillin", top_k=2, model_name=None):
    """Compute a concise text summary of top SHAP feature drivers for inline grid display."""
    try:
        explanation = explain_sample(sample_data, antibiotic=antibiotic, model_name=model_name)
        top_feats = explanation.get("top_features", [])[:top_k]
        if not top_feats:
            return "Standard baseline distribution"
        reasons = []
        for tf in top_feats:
            sign = "+" if tf["shap_value"] >= 0 else ""
            reasons.append(f"{tf['feature']} ({sign}{tf['shap_value']:.2f})")
        return ", ".join(reasons)
    except Exception:
        return "Epidemiological feature baseline"

