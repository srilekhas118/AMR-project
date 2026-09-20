"""PyTorch Tabular Transformer Training Script for AMR Prediction.
Trains lightweight CPU-friendly Transformer models per antibiotic.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from src.preprocessing import load_preprocessor, transform_data, load_schema
from src.transformer_model import TabularTransformer

TRANSFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "transformer")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_transformer(abx_key, X_train, y_train, X_val, y_val, epochs=10, batch_size=512, lr=0.002):
    input_dim = X_train.shape[1]
    
    pos_weight = float((len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train)))
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    model = TabularTransformer(input_dim=input_dim, n_tokens=8, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
        val_loss = val_loss / len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


def train_all_transformers(only_new=True):
    set_seed(42)
    os.makedirs(TRANSFORMER_DIR, exist_ok=True)
    schema = load_schema()

    cleaned_path = os.path.join("data", "processed", "narms_cleaned.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    cleaned_df = pd.read_csv(cleaned_path, low_memory=False)

    antibiotics = schema["selected_antibiotics"]
    existing_keys = {"ampicillin", "tetracycline", "ciprofloxacin", "streptomycin"}

    for abx_key, abx_info in antibiotics.items():
        checkpoint_path = os.path.join(TRANSFORMER_DIR, f"{abx_key}_transformer.pt")
        if only_new and abx_key in existing_keys and os.path.exists(checkpoint_path):
            print(f"\nSkipping existing transformer checkpoint for {abx_info['display_name']} ({abx_key}) as instructed.")
            continue

        display_name = abx_info["display_name"]
        target_col = abx_info["target_column"]
        print(f"\n==========================================", flush=True)
        print(f"Training Tabular Transformer for {display_name}...", flush=True)
        print(f"==========================================", flush=True)

        valid_subset = cleaned_df[cleaned_df[target_col].notnull()].copy()
        n_valid = len(valid_subset)
        if n_valid > 50000:
            print(f"Sampling N=50,000 (random_state=42) from {n_valid} valid records...", flush=True)
            valid_subset = valid_subset.sample(n=50000, random_state=42)

        from sklearn.model_selection import train_test_split
        train_sub, temp_sub = train_test_split(valid_subset, test_size=0.30, random_state=42, shuffle=True)
        val_sub, test_sub = train_test_split(temp_sub, test_size=0.50, random_state=42, shuffle=True)

        preprocessor = load_preprocessor()

        X_train = transform_data(train_sub, preprocessor)
        y_train = train_sub[target_col].astype(int).values

        X_val = transform_data(val_sub, preprocessor)
        y_val = val_sub[target_col].astype(int).values

        model = train_single_transformer(abx_key, X_train, y_train, X_val, y_val, epochs=10, batch_size=512, lr=0.002)

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "n_tokens": 8,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 64
        }, checkpoint_path)
        print(f"Saved Transformer checkpoint to {checkpoint_path}", flush=True)

    print("\nTransformer training completed successfully.", flush=True)


if __name__ == "__main__":
    train_all_transformers(only_new=True)
