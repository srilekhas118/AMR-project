"""Unit tests for classical ML models and Tabular Transformer architecture.
"""

import os
import pytest
import numpy as np
import torch

from src.models import get_classical_models
from src.transformer_model import TabularTransformer
from src.predict import load_model


def test_classical_model_instantiation():
    models = get_classical_models(random_state=42)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "Gradient Boosting" in models


def test_transformer_forward_pass():
    input_dim = 59
    batch_size = 4
    model = TabularTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    
    dummy_x = np.random.randn(batch_size, input_dim).astype(np.float32)
    probs = model.predict_proba(dummy_x)
    
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (batch_size,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_load_classical_model_bundle():
    for abx in ["ampicillin", "tetracycline", "ciprofloxacin", "streptomycin"]:
        model_obj, resolved_name = load_model(abx, "Random Forest")
        assert model_obj is not None
        assert hasattr(model_obj, "predict_proba")


def test_load_transformer_checkpoints():
    from src.evaluate import load_transformer_model
    for abx in ["ampicillin", "tetracycline", "ciprofloxacin", "streptomycin"]:
        model_obj = load_transformer_model(abx)
        assert model_obj is not None
        dummy = np.random.randn(2, model_obj.input_dim).astype(np.float32)
        probs = model_obj.predict_proba(dummy)
        assert probs.shape == (2,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
