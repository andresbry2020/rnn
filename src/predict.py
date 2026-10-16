"""
Módulo de inferencia y predicción para nuevos mensajes utilizando el modelo BiLSTM entrenado.
Ofrece tanto API programática como interfaz de línea de comandos (CLI).
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import torch

# Importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text, tokenize, Vocabulary, PAD_IDX
from src.model import SpamBiLSTM


class SpamPredictor:
    """Clase encargada de cargar el modelo BiLSTM y realizar inferencias sobre textos nuevos."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_dir = os.path.join(base_dir, "saved_models")
        
        self.model_path = model_path or os.path.join(saved_dir, "best_bilstm_model.pt")
        self.vocab_path = vocab_path or os.path.join(saved_dir, "vocab.json")
        self.config_path = config_path or os.path.join(saved_dir, "model_config.json")
        
        # Selección de dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self._load_components()

    def _load_components(self):
        """Carga el vocabulario, la configuración y los pesos del modelo."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"No se encontró el vocabulario en: {self.vocab_path}. Debe entrenar el modelo primero.")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"No se encontró la configuración en: {self.config_path}. Debe entrenar el modelo primero.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el checkpoint en: {self.model_path}. Debe entrenar el modelo primero.")
            
        # Cargar vocabulario
        self.vocab = Vocabulary.load(self.vocab_path)
        
        # Cargar configuración
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.max_len = self.config.get("max_len", 80)
        
        # Construir e inicializar modelo
        self.model = SpamBiLSTM(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            bidirectional=self.config.get("bidirectional", True),
            dense_dim=self.config["dense_dim"],
            dropout=self.config.get("dropout", 0.3),
            padding_idx=PAD_IDX
        ).to(self.device)
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predice si un mensaje individual es Spam o Ham.
        Retorna etiqueta, probabilidad, nivel de confianza y detalles de tokenización.
        """
        cleaned = clean_text(text)
        tokens = tokenize(text)
        
        if not tokens:
            return {
                "label": "ham",
                "is_spam": False,
                "spam_probability": 0.0,
                "ham_probability": 1.0,
                "confidence": 1.0,
                "tokens": [],
                "token_count": 0,
                "cleaned_text": "",
                "status_message": "Texto vacío o sin tokens reconocibles"
            }
            
        # Transformación a secuencia de índices numéricos
        indices = self.vocab.text_to_indices(text, max_len=self.max_len)
        input_tensor = torch.tensor([indices], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            prob = self.model.predict_proba(input_tensor).item()
            
        is_spam = bool(prob >= 0.5)
        label = "spam" if is_spam else "ham"
        label_es = "SPAM (Fraudulento / No deseado)" if is_spam else "LEGÍTIMO (Seguro)"
        confidence = prob if is_spam else (1.0 - prob)
        
        return {
            "label": label,
            "label_es": label_es,
            "is_spam": is_spam,
            "spam_probability": float(prob),
            "ham_probability": float(1.0 - prob),
            "confidence": float(confidence),
            "tokens": tokens,
            "token_count": len(tokens),
            "cleaned_text": cleaned
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Realiza predicciones sobre una lista o lote de textos."""
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="Inferencia con modelo BiLSTM para detección de Spam")
    parser.add_argument("--text", type=str, help="Texto del mensaje a clasificar")
    args = parser.parse_args()
    
    sample_text = args.text or "¡URGENTE! Su cuenta bancaria ha sido suspendida temporalmente. Verifique su identidad ahora en http://seguridad-banca-alerta.com"
    print(f"\n--- Probando Inferencia con Texto de Entrada ---")
    print(f"Texto: '{sample_text}'\n")
    
    predictor = SpamPredictor()
    result = predictor.predict(sample_text)
    
    print(f"Resultado de Clasificación:")
    print(f"  Etiqueta Predicha:       {result['label_es']}")
    print(f"  Probabilidad de Spam:    {result['spam_probability']*100:.2f}%")
    print(f"  Probabilidad Legítimo:   {result['ham_probability']*100:.2f}%")
    print(f"  Nivel de Confianza:      {result['confidence']*100:.2f}%")
    print(f"  Tokens extraídos ({result['token_count']}): {result['tokens']}")


if __name__ == "__main__":
    main()
