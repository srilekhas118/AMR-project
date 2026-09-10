"""Lightweight Tabular Transformer in PyTorch for fast, CPU-friendly AMR resistance prediction.
"""

import torch
import torch.nn as nn
import numpy as np


class TabularTransformer(nn.Module):
    """Lightweight Tabular Transformer architecture for fast CPU-friendly tabular classification."""
    def __init__(self, input_dim, n_tokens=8, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super(TabularTransformer, self).__init__()
        self.input_dim = input_dim
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Project tabular input vector to n_tokens latent feature tokens
        self.tokenizer = nn.Sequential(
            nn.Linear(input_dim, n_tokens * d_model),
            nn.LayerNorm(n_tokens * d_model),
            nn.ReLU()
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]
        
        # Tokenize tabular features into (batch_size, n_tokens, d_model)
        tokens = self.tokenizer(x).view(batch_size, self.n_tokens, self.d_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embedding[:, :self.n_tokens + 1, :]
        
        # Pass through Transformer encoder
        encoded = self.transformer_encoder(x_emb)
        encoded = self.layer_norm(encoded)
        
        # CLS token representation
        cls_out = encoded[:, 0, :]
        
        logits = self.head(cls_out).squeeze(-1)
        return logits

    def predict_proba(self, x_np):
        """Compute resistance probabilities for numpy input array."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            logits = self.forward(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
