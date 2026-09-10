"""Classical ML Training Script for AMR Prediction.
Trains Logistic Regression, Random Forest, and Gradient Boosting per antibiotic.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import (
    fit_and_save_preprocessor,
    transform_data,
    load_schema,
    validate_dataset,
)
from src.models import get_classical_models
from src.anomaly_detection import fit_and_save_anomaly_detector

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "models")


def train_classical_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    schema = load_schema()
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    cleaned_path = os.path.join("data", "processed", "narms_cleaned.csv")
    if os.path.exists(cleaned_path):
        cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
        validation = validate_dataset(cleaned_df, schema=schema)
        print(f"Dataset validation passed: {validation['n_records']} usable records.")
    print(f"Loaded Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # 1. Fit & save the unified preprocessing pipeline on training data
    print("Fitting unified feature preprocessor...")
    preprocessor = fit_and_save_preprocessor(train_df, schema=schema)
    
    # Transform full training features for anomaly detector fitting
    X_train_full = transform_data(train_df, preprocessor)
    print(f"Transformed feature matrix shape: {X_train_full.shape}")

    # 2. Fit Anomaly Detector
    print("Fitting baseline Isolation Forest anomaly detector...")
    fit_and_save_anomaly_detector(X_train_full)

    # 3. Train models for each selected antibiotic
    antibiotics = schema["selected_antibiotics"]

    for abx_key, abx_info in antibiotics.items():
        display_name = abx_info["display_name"]
        target_col = abx_info["target_column"]
        print(f"\n==========================================")
        print(f"Training models for {display_name} ({target_col})...")
        print(f"==========================================")

        # Filter valid observations (0=S, 1=R)
        train_mask = train_df[target_col].notnull()
        X_train_subset = train_df[train_mask]
        y_train = train_df.loc[train_mask, target_col].astype(int).values

        val_mask = val_df[target_col].notnull()
        X_val_subset = val_df[val_mask]
        y_val = val_df.loc[val_mask, target_col].astype(int).values

        X_train_feat = transform_data(X_train_subset, preprocessor)
        X_val_feat = transform_data(X_val_subset, preprocessor)

        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        print(f"Train samples: {len(y_train)} (Resistant: {pos_count}, Susceptible: {neg_count})")

        trained_abx_models = {}
        model_factories = get_classical_models(random_state=42)

        for model_name, model_obj in model_factories.items():
            print(f"  Training {model_name}...")
            model_obj.fit(X_train_feat, y_train)
            
            # Validation check
            val_score = model_obj.score(X_val_feat, y_val)
            print(f"    Val Accuracy: {val_score:.4f}")
            trained_abx_models[model_name] = model_obj

        # Save model bundle
        save_path = os.path.join(MODELS_DIR, f"{abx_key}_models.joblib")
        joblib.dump(trained_abx_models, save_path)
        print(f"Saved {display_name} models to {save_path}")

    print("\nClassical ML training completed successfully.")


if __name__ == "__main__":
    train_classical_models()
