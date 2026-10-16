"""
Módulo de preprocesamiento, tokenización y preparación de secuencias para Deep Learning (BiLSTM).
Maneja limpieza de texto, construcción del vocabulario, padding/truncating y DataLoaders de PyTorch.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def clean_text(text: str) -> str:
    """
    Limpia y normaliza un texto para clasificación secuencial:
    - Conversión a minúsculas
    - Reemplazo de URLs por token especial
    - Reemplazo de símbolos de moneda ($, £, €) por token
    - Reemplazo de números y dígitos
    - Eliminación de caracteres especiales conservando palabras
    - Normalización de espacios en blanco
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Reemplazo de URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' <url> ', text)
    
    # Reemplazo de números de teléfono o códigos numéricos largos
    text = re.sub(r'\b\d{5,}\b', ' <phone> ', text)
    
    # Reemplazo de símbolos monetarios
    text = re.sub(r'[£$€¥₹]', ' <money> ', text)
    
    # Reemplazo de números aislados
    text = re.sub(r'\b\d+\b', ' <number> ', text)
    
    # Eliminar puntuación irrelevante manteniendo caracteres alfanuméricos y tokens especiales
    text = re.sub(r'[^\w\s<>]', ' ', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokeniza un texto limpio en lista de tokens/palabras."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Tokeniza preservando los tokens especiales como <url>, <money>, <number>, etc.
    tokens = re.findall(r'<[a-z]+>|\b\w+\b', cleaned)
    return tokens


class Vocabulary:
    """
    Gestiona el mapeo bidireccional entre palabras e índices enteros,
    con soporte para tokens especiales <pad> y <unk>.
    """
    def __init__(self, max_size: int = 6000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_counts: Counter = Counter()

    def __len__(self) -> int:
        return len(self.word2idx)

    def build_vocab(self, texts: List[str]):
        """Construye el vocabulario a partir de una lista de textos de entrenamiento."""
        self.word_counts = Counter()
        for text in texts:
            tokens = tokenize(text)
            self.word_counts.update(tokens)
        
        # Filtrar por frecuencia mínima y ordenar por ocurrencias
        sorted_words = [word for word, count in self.word_counts.most_common() if count >= self.min_freq]
        
        # Limitar al tamaño máximo reservando espacio para PAD y UNK
        available_slots = self.max_size - 2
        selected_words = sorted_words[:available_slots]
        
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        
        for idx, word in enumerate(selected_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_indices(self, text: str, max_len: int = 80) -> List[int]:
        """Convierte un texto en una lista de índices con padding o truncamiento."""
        tokens = tokenize(text)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        
        if len(indices) < max_len:
            # Padding al final (post-padding)
            indices = indices + [PAD_IDX] * (max_len - len(indices))
        else:
            # Truncar al límite
            indices = indices[:max_len]
        return indices

    def save(self, filepath: str):
        """Serializa el vocabulario a formato JSON."""
        data = {
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "vocab_size": len(self.word2idx)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Carga un vocabulario serializado desde un archivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        return vocab


class TextSequenceDataset(Dataset):
    """Dataset de PyTorch para secuencias numéricas y etiquetas binarias."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def prepare_datasets_and_loaders(
    csv_path: str,
    batch_size: int = 32,
    max_len: int = 80,
    max_vocab_size: int = 6000,
    min_freq: int = 2,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Carga el dataset, realiza limpieza, partición estratificada en 3 subconjuntos:
    - Entrenamiento (~70%)
    - Validación (~15%)
    - Prueba (~15%)
    Construye el vocabulario exclusivamente sobre el conjunto de entrenamiento
    y genera los DataLoaders correspondientes.
    """
    df = pd.read_csv(csv_path)
    df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})
    
    # 1. División Train+Val y Test (Test = 15%)
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_bin"]
    )
    
    # 2. División Train y Val (Val = 15% del total -> ~17.65% de train_val)
    adjusted_val_size = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=df_train_val["label_bin"]
    )
    
    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Construcción de vocabulario sobre train exclusivamente para prevenir data leakage
    vocab = Vocabulary(max_size=max_vocab_size, min_freq=min_freq)
    vocab.build_vocab(df_train["message"].tolist())
    
    # Conversión de textos a secuencias de índices numéricos
    X_train = np.array([vocab.text_to_indices(t, max_len=max_len) for t in df_train["message"]])
    y_train = df_train["label_bin"].values
    
    X_val = np.array([vocab.text_to_indices(t, max_len=max_len) for t in df_val["message"]])
    y_val = df_val["label_bin"].values
    
    X_test = np.array([vocab.text_to_indices(t, max_len=max_len) for t in df_test["message"]])
    y_test = df_test["label_bin"].values
    
    # Creación de Datasets y DataLoaders
    train_dataset = TextSequenceDataset(X_train, y_train)
    val_dataset = TextSequenceDataset(X_val, y_val)
    test_dataset = TextSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    dfs = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }
    
    return train_loader, val_loader, test_loader, vocab, dfs
