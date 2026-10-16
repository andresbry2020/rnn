"""
Definición de la arquitectura de Red Neuronal Recurrente Bidireccional (BiLSTM)
para clasificación de texto secuencial en PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SpamBiLSTM(nn.Module):
    """
    Red Neuronal Recurrente Bidireccional (BiLSTM) con capa de Embeddings,
    mecanismo de Mean + Max Pooling global y cabezal de clasificación denso.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dense_dim: int = 64,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super(SpamBiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dense_dim = dense_dim
        self.dropout_rate = dropout
        self.num_directions = 2 if bidirectional else 1
        
        # 1. Capa de Embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self.embedding_dropout = nn.Dropout(p=dropout)
        
        # 2. Capa Recurrente LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Dimensión de salida de la LSTM
        lstm_out_dim = hidden_dim * self.num_directions
        
        # Usamos concatenación de Mean-Pooling y Max-Pooling a lo largo de la dimensión temporal:
        # Esto captura tanto palabras clave de alto impacto (Max) como el contexto global (Mean).
        pooled_dim = lstm_out_dim * 2
        
        # 4. Capas densas de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dense_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del modelo.
        Entrada:
            x: Tensor de enteros de dimensiones (batch_size, seq_len)
        Salida:
            logits: Tensor de dimensiones (batch_size, 1) sin aplicar Sigmoid
        """
        # (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(embedded)
        
        # Global Average Pooling y Global Max Pooling
        # mean_pool: (batch_size, hidden_dim * num_directions)
        # max_pool: (batch_size, hidden_dim * num_directions)
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Concatenación: (batch_size, hidden_dim * num_directions * 2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        
        # Clasificador
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades calibradas mediante activación Sigmoid [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def get_config(self) -> Dict[str, Any]:
        """Retorna el diccionario de hiperparámetros para serialización."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout_rate
        }


def count_parameters(model: nn.Module) -> int:
    """Calcula el número total de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
