"""
Script principal de entrenamiento para la red neuronal BiLSTM.
Realiza partición de datos, entrenamiento con validación, early stopping,
cálculo de métricas completas (Accuracy, F1, Precision, Recall, AUC, Confusion Matrix),
generación de gráficas de entrenamiento y persistencia de artefactos.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim

# Importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import prepare_datasets_and_loaders, clean_text, PAD_IDX
from src.model import SpamBiLSTM, count_parameters


def set_seed(seed: int = 42):
    """Garantiza reproducibilidad determinística en PyTorch y NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Ejecuta una época de entrenamiento."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping para estabilidad en RNN/LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += sequences.size(0)
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo en un conjunto de validación o prueba."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            total_loss += loss.item() * sequences.size(0)
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
            
    epoch_loss = total_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0
        
    metrics = {
        "loss": epoch_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities": all_probs
    }
    return metrics


def plot_training_history(history, save_path):
    """Genera y guarda la gráfica de evolución de Loss y Accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', color='#1f77b4', label='Entrenamiento')
    plt.plot(epochs, history["val_loss"], 's-', color='#ff7f0e', label='Validación')
    plt.title('Evolución de la Pérdida (Loss) por Época', fontsize=12, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('BCE Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'o-', color='#2ca02c', label='Entrenamiento')
    plt.plot(epochs, history["val_acc"], 's-', color='#d62728', label='Validación')
    plt.title('Evolución de la Exactitud (Accuracy) por Época', fontsize=12, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfica de curvas de entrenamiento guardada en: {save_path}")


def plot_confusion_matrix_heatmap(cm, save_path):
    """Genera y guarda la matriz de confusión formateada."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham (Legítimo)', 'Spam (No deseado)'],
        yticklabels=['Ham (Legítimo)', 'Spam (No deseado)'],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title('Matriz de Confusión en Conjunto de Prueba', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicción del Modelo', fontsize=11)
    plt.ylabel('Etiqueta Real', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def plot_class_distribution(dfs, save_path):
    """Genera gráfica con la distribución de clases en Train, Val y Test."""
    plt.figure(figsize=(8, 4.5))
    data_summary = []
    for split_name, df_split in dfs.items():
        counts = df_split["label"].value_counts()
        for label, count in counts.items():
            data_summary.append({"Split": split_name.capitalize(), "Clase": label.upper(), "Conteo": count})
    df_counts = pd.DataFrame(data_summary)
    
    ax = sns.barplot(data=df_counts, x="Split", y="Conteo", hue="Clase", palette=["#2ecc71", "#e74c3c"])
    plt.title("Distribución de Clases por Subconjunto (Train / Val / Test)", fontsize=12, fontweight="bold")
    plt.xlabel("Subconjunto")
    plt.ylabel("Cantidad de Mensajes")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h / 2),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfica de distribución guardada en: {save_path}")


def run_training(args):
    """Flujo completo de ejecución y entrenamiento."""
    set_seed(args.seed)
    
    # Rutas base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "spam.csv")
    models_dir = os.path.join(base_dir, "saved_models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Dispositivo de Entrenamiento: {device} ===")
    
    # 1. Preparación de Datasets, Vocabulario y DataLoaders
    print("\n--- 1. Preparando Datos y Vocabulario ---")
    train_loader, val_loader, test_loader, vocab, dfs = prepare_datasets_and_loaders(
        csv_path=data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        test_size=0.15,
        val_size=0.15,
        random_state=args.seed
    )
    
    print(f"Dataset total: {len(dfs['train']) + len(dfs['val']) + len(dfs['test'])} muestras")
    print(f"  Train: {len(dfs['train'])} muestras ({len(dfs['train'])/55.72:.1f}%)")
    print(f"  Val:   {len(dfs['val'])} muestras ({len(dfs['val'])/55.72:.1f}%)")
    print(f"  Test:  {len(dfs['test'])} muestras ({len(dfs['test'])/55.72:.1f}%)")
    print(f"Vocabulario construido: {len(vocab)} tokens únicos")
    
    # Guardar vocabulario
    vocab_file = os.path.join(models_dir, "vocab.json")
    vocab.save(vocab_file)
    print(f"Vocabulario guardado en: {vocab_file}")
    
    # Gráfica de distribución de clases
    plot_class_distribution(dfs, os.path.join(plots_dir, "class_distribution.png"))
    
    # 2. Inicialización del Modelo BiLSTM
    print("\n--- 2. Construcción de la Red Neuronal BiLSTM ---")
    model = SpamBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=True,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
        padding_idx=PAD_IDX
    ).to(device)
    
    print(f"Parámetros entrenables totales: {count_parameters(model):,}")
    
    # Ponderación para clase minoritaria (spam = 1)
    num_neg = (dfs['train']["label_bin"] == 0).sum()
    num_pos = (dfs['train']["label_bin"] == 1).sum()
    pos_weight = torch.tensor([num_neg / (num_pos * 1.5)]).to(device) # factor balanceado
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 3. Bucle de Entrenamiento con Early Stopping
    print("\n--- 3. Bucle de Entrenamiento y Validación ---")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": []
    }
    
    best_val_f1 = -1.0
    best_model_path = os.path.join(models_dir, "best_bilstm_model.pt")
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        
        print(f"Época {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['accuracy']*100:.2f}% - "
              f"F1: {val_metrics['f1']:.4f} - Rec: {val_metrics['recall']:.4f}")
        
        # Guardar mejor modelo según F1-Score en Validación
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            best_indicator = " [* MEJOR CHECKPOINT GUARDADO *]"
        else:
            patience_counter += 1
            best_indicator = ""
            
        if best_indicator:
            print(f"   --> {best_indicator}")
            
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No se observó mejora en {args.patience} épocas consecutivas. Deteniendo.")
            break
            
    # Guardar historial
    with open(os.path.join(models_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    # Gráficas de curvas de entrenamiento
    plot_training_history(history, os.path.join(plots_dir, "training_curves.png"))
    
    # 4. Evaluación en Conjunto de Prueba (Test)
    print("\n--- 4. Evaluación Final en Conjunto de Prueba (Test Set) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    cm = confusion_matrix(test_metrics["targets"], test_metrics["predictions"])
    plot_confusion_matrix_heatmap(cm, os.path.join(plots_dir, "confusion_matrix.png"))
    
    print("\nReporte de Clasificación en Prueba:")
    print(classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        target_names=["Ham (Legítimo)", "Spam (No deseado)"],
        digits=4
    ))
    
    print("Métricas Clave en Prueba:")
    print(f"  Exactitud (Accuracy):   {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precisión (Precision):  {test_metrics['precision']*100:.2f}%")
    print(f"  Sensibilidad (Recall):  {test_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:               {test_metrics['f1']*100:.2f}%")
    print(f"  Área bajo Curva (AUC):  {test_metrics['auc']:.4f}")
    
    # Guardar métricas numéricas
    saved_metrics = {
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1_score": float(test_metrics["f1"]),
        "roc_auc": float(test_metrics["auc"]),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(test_metrics["targets"])
    }
    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(saved_metrics, f, indent=2)
        
    # Guardar configuración del modelo
    model_config = model.get_config()
    model_config["max_len"] = args.max_len
    with open(os.path.join(models_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
        
    # 5. Mínimo 10 Predicciones de Prueba Detalladas
    print("\n--- 5. Generando Predicciones de Prueba Detalladas (Mínimo 10) ---")
    df_test = dfs["test"].copy()
    df_test["predicted_label"] = np.where(test_metrics["predictions"] == 1, "spam", "ham")
    df_test["spam_probability"] = test_metrics["probabilities"]
    df_test["is_correct"] = df_test["label"] == df_test["predicted_label"]
    
    # Seleccionamos una muestra balanceada y representativa de 15 ejemplos
    sample_spams = df_test[df_test["label"] == "spam"].head(8)
    sample_hams = df_test[df_test["label"] == "ham"].head(8)
    sample_preds = pd.concat([sample_spams, sample_hams]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    predictions_list = []
    print("\n" + "="*85)
    print(f"{'#':<3} | {'Texto (Mensaje SMS)':<45} | {'Real':<6} | {'Pred':<6} | {'Prob Spam':<10} | {'Estado'}")
    print("="*85)
    
    for i, row in sample_preds.iterrows():
        short_text = (row['message'][:42] + '...') if len(row['message']) > 42 else row['message']
        status = "ACIERTO" if row['is_correct'] else "FALLO"
        prob_str = f"{row['spam_probability']*100:.2f}%"
        print(f"{i+1:<3} | {short_text:<45} | {row['label']:<6} | {row['predicted_label']:<6} | {prob_str:<10} | {status}")
        
        predictions_list.append({
            "id": int(i + 1),
            "message": row["message"],
            "true_label": row["label"],
            "predicted_label": row["predicted_label"],
            "spam_probability": float(row["spam_probability"]),
            "is_correct": bool(row["is_correct"])
        })
    print("="*85)
    
    with open(os.path.join(models_dir, "sample_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, ensure_ascii=False, indent=2)
    print(f"Predicciones de prueba guardadas en: {os.path.join(models_dir, 'sample_predictions.json')}")
    print("\n=== ¡Entrenamiento y Evaluación Finalizados con Éxito! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo BiLSTM para detección de SMS Spam")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimensión de embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimensión oculta de LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas LSTM apiladas")
    parser.add_argument("--dense_dim", type=int, default=64, help="Dimensión de la capa densa")
    parser.add_argument("--dropout", type=float, default=0.3, help="Tasa de dropout")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--max_len", type=int, default=80, help="Longitud máxima de secuencia")
    parser.add_argument("--max_vocab_size", type=int, default=6000, help="Tamaño máximo de vocabulario")
    parser.add_argument("--min_freq", type=int, default=2, help="Frecuencia mínima de palabra")
    parser.add_argument("--patience", type=int, default=4, help="Paciencia para early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    args = parser.parse_args()
    run_training(args)
