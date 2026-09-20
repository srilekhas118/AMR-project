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


def train_classical_models(only_new=True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    schema = load_schema()
    
    cleaned_path = os.path.join("data", "processed", "narms_cleaned.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    
    cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
    validation = validate_dataset(cleaned_df, schema=schema)
    print(f"Dataset validation passed: {validation['n_records']} usable records.")

    train_df = pd.read_csv("data/processed/train.csv")
    # 1. Fit & save the unified preprocessing pipeline on training data
    print("Fitting unified feature preprocessor...")
    preprocessor = fit_and_save_preprocessor(train_df, schema=schema)
    
    # Transform full training features for anomaly detector fitting
    X_train_full = transform_data(train_df, preprocessor)
    print(f"Transformed feature matrix shape: {X_train_full.shape}")

    # 2. Fit Anomaly Detector
    print("Fitting baseline Isolation Forest anomaly detector...")
    fit_and_save_anomaly_detector(X_train_full)

    # 3. Train models for selected antibiotics
    antibiotics = schema["selected_antibiotics"]
    existing_keys = {"ampicillin", "tetracycline", "ciprofloxacin", "streptomycin"}

    for abx_key, abx_info in antibiotics.items():
        save_path = os.path.join(MODELS_DIR, f"{abx_key}_models.joblib")
        if only_new and abx_key in existing_keys and os.path.exists(save_path):
            print(f"\nSkipping existing model for {abx_info['display_name']} ({abx_key}) as instructed.")
            continue

        display_name = abx_info["display_name"]
        target_col = abx_info["target_column"]
        print(f"\n==========================================")
        print(f"Training models for {display_name} ({target_col})...")
        print(f"==========================================")

        # Filter valid records for this antibiotic
        valid_subset = cleaned_df[cleaned_df[target_col].notnull()].copy()
        n_valid = len(valid_subset)
        print(f"Total valid observations for {display_name}: {n_valid}")

        # Sample N=50,000 if valid records > 50,000 before splitting
        if n_valid > 50000:
            print(f"Sampling N=50,000 (random_state=42) from {n_valid} valid records...")
            valid_subset = valid_subset.sample(n=50000, random_state=42)

        # 70/15/15 train/val/test split with random_state=42
        from sklearn.model_selection import train_test_split
        train_sub, temp_sub = train_test_split(valid_subset, test_size=0.30, random_state=42, shuffle=True)
        val_sub, test_sub = train_test_split(temp_sub, test_size=0.50, random_state=42, shuffle=True)

        y_train = train_sub[target_col].astype(int).values
        y_val = val_sub[target_col].astype(int).values

        X_train_feat = transform_data(train_sub, preprocessor)
        X_val_feat = transform_data(val_sub, preprocessor)

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
        joblib.dump(trained_abx_models, save_path)
        print(f"Saved {display_name} models to {save_path}")

    print("\nClassical ML training completed successfully.")


if __name__ == "__main__":
    train_classical_models(only_new=True)
