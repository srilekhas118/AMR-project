"""Prediction Engine for AMR Intelligence System.
Transforms sample input, loads persisted models, and returns structured predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import load_preprocessor, transform_data, load_schema
from src.transformer_model import TabularTransformer
from src.evaluate import load_transformer_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "models")
METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics.json")


def get_best_model_name(abx_key):
    """Retrieve the empirical best model name from saved metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                return metrics.get("best_models", {}).get(abx_key, {}).get("model_name", "Random Forest")
        except Exception:
            pass
    return "Random Forest"


def load_model(abx_key, model_name=None):
    """Load a specific or best trained model for a given antibiotic target."""
    if model_name is None:
        model_name = get_best_model_name(abx_key)

    if model_name == "Tabular Transformer":
        return load_transformer_model(abx_key), model_name

    model_bundle_path = os.path.join(MODELS_DIR, f"{abx_key}_models.joblib")
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}. Run training first.")
    
    bundle = joblib.load(model_bundle_path)
    if model_name not in bundle:
        # fallback to first available
        model_name = list(bundle.keys())[0]
    return bundle[model_name], model_name


def validate_and_format_input(sample_data, schema=None):
    """Format input dict into standardized DataFrame conforming to schema."""
    if schema is None:
        schema = load_schema()

    if isinstance(sample_data, dict):
        df_input = pd.DataFrame([sample_data])
    elif isinstance(sample_data, pd.DataFrame):
        df_input = sample_data.copy()
    else:
        raise ValueError("sample_data must be a dict or pandas DataFrame.")

    # Ensure all required features are present
    cat_cols = schema["features"]["categorical"]
    num_cols = schema["features"]["numerical"]
    
    for col in cat_cols:
        if col not in df_input.columns:
            df_input[col] = "Unknown"
        else:
            df_input[col] = df_input[col].fillna("Unknown").astype(str)

    for col in num_cols:
        if col not in df_input.columns:
            df_input[col] = schema["numerical_ranges"].get(col, {}).get("default", 2015)
        else:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(2015).astype(int)

    return df_input[cat_cols + num_cols]


def predict_sample(sample_data, antibiotic="ampicillin", model_name=None):
    """Generate standardized resistance prediction and probability for a single sample.
    
    Args:
        sample_data (dict or DataFrame): Sample features.
        antibiotic (str): Antibiotic key e.g. 'ampicillin', 'tetracycline', 'ciprofloxacin', 'streptomycin'.
        model_name (str, optional): Name of model architecture to use. Defaults to empirical best.
        
    Returns:
        dict: Standardized prediction result.
    """
    schema = load_schema()
    abx_key = antibiotic.lower()
    if abx_key not in schema["selected_antibiotics"]:
        raise ValueError(f"Unknown antibiotic '{antibiotic}'. Must be one of {list(schema['selected_antibiotics'].keys())}")

    abx_info = schema["selected_antibiotics"][abx_key]
    df_formatted = validate_and_format_input(sample_data, schema)
    
    preprocessor = load_preprocessor()
    X_feat = transform_data(df_formatted, preprocessor)

    model_obj, resolved_model_name = load_model(abx_key, model_name)

    if resolved_model_name == "Tabular Transformer":
        prob_resistant = float(model_obj.predict_proba(X_feat)[0])
    else:
        prob_resistant = float(model_obj.predict_proba(X_feat)[0, 1])

    is_resistant = bool(prob_resistant >= 0.5)
    prediction_label = "Resistant" if is_resistant else "Susceptible"

    return {
        "antibiotic": abx_info["display_name"],
        "antibiotic_key": abx_key,
        "drug_class": abx_info["drug_class"],
        "model_used": resolved_model_name,
        "prediction": prediction_label,
        "prediction_code": 1 if is_resistant else 0,
        "is_resistant": is_resistant,
        "resistance_probability": round(prob_resistant, 4),
        "susceptibility_probability": round(1.0 - prob_resistant, 4),
        "formatted_input": df_formatted.to_dict(orient="records")[0]
    }
