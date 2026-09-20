"""Preprocessing pipeline module for AMR Intelligence System.
Uses config/feature_schema.json as the Single Source of Truth.
"""

import os
import sys
import types
import json
import joblib
import pandas as pd
import numpy as np
import sklearn.compose._column_transformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Backward compatibility shims for unpickling artifacts across scikit-learn versions
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    sklearn.compose._column_transformer._RemainderColsList = list

if "_loss" not in sys.modules:
    try:
        import sklearn._loss.loss as _sk_loss
        loss_cls = getattr(_sk_loss, "HalfBinomialLoss", None)
    except Exception:
        loss_cls = None
    _loss_mod = types.ModuleType("_loss")
    _loss_mod.CyHalfBinomialLoss = loss_cls
    sys.modules["_loss"] = _loss_mod

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "feature_schema.json")
PREPROCESSOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "preprocessors")
PIPELINE_PATH = os.path.join(PREPROCESSOR_DIR, "feature_pipeline.joblib")


def load_schema(schema_path=SCHEMA_PATH):
    """Load the single source of truth feature schema."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_dataset(df, schema=None, min_records=10000):
    """Validate usable record count, features, antibiotics, and both target classes."""
    if schema is None:
        schema = load_schema()

    n_records = len(df)
    if n_records < min_records:
        raise ValueError(
            f"Usable dataset has {n_records} records, below the required minimum of {min_records}."
        )

    cat_cols = schema["features"]["categorical"]
    num_cols = schema["features"]["numerical"]
    missing_features = [c for c in cat_cols + num_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Required feature columns missing: {missing_features}")

    leakage_tokens = ("concl", "mic", "geno", "target_")
    feature_like = [c for c in df.columns if c in cat_cols + num_cols]
    leaked = [c for c in feature_like if any(tok in c.lower() and c.startswith("target_") for tok in leakage_tokens)]
    if leaked:
        raise ValueError(f"Potential target leakage in feature columns: {leaked}")

    for abx_key, abx_info in schema["selected_antibiotics"].items():
        target_col = abx_info["target_column"]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' for {abx_key} is missing.")
        valid = df[target_col].dropna()
        classes = set(valid.unique().tolist())
        if 0 not in classes or 1 not in classes:
            raise ValueError(f"Antibiotic {abx_key} does not contain both classes 0 and 1.")

    return {
        "n_records": n_records,
        "n_features": len(cat_cols) + len(num_cols),
        "antibiotics": list(schema["selected_antibiotics"].keys()),
    }


def build_preprocessor(schema=None):
    """Build a Scikit-learn ColumnTransformer for categorical and numerical features."""
    if schema is None:
        schema = load_schema()

    cat_cols = schema["features"]["categorical"]
    num_cols = schema["features"]["numerical"]

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def fit_and_save_preprocessor(train_df, schema=None, save_path=PIPELINE_PATH):
    """Fit the preprocessor on the training dataframe only and persist it."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if schema is None:
        schema = load_schema()

    preprocessor = build_preprocessor(schema)
    preprocessor.fit(train_df)

    joblib.dump(preprocessor, save_path)
    return preprocessor


def load_preprocessor(pipeline_path=PIPELINE_PATH):
    """Load the fitted preprocessor pipeline."""
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Preprocessor not found at {pipeline_path}. Please train or fit first.")
    return joblib.load(pipeline_path)


def transform_data(df, preprocessor=None):
    """Transform a dataframe using the fitted preprocessor and return dense numpy array."""
    if preprocessor is None:
        preprocessor = load_preprocessor()
    return preprocessor.transform(df)


def get_feature_names(preprocessor=None):
    """Extract expanded one-hot and scaled feature names from preprocessor."""
    if preprocessor is None:
        preprocessor = load_preprocessor()
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        schema = load_schema()
        return schema["features"]["categorical"] + schema["features"]["numerical"]
