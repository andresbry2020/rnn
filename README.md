# 🛡️ Detección de Mensajes Spam y Phishing con Deep Learning (BiLSTM)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/deep_learning_nlp_lstm/blob/main/notebooks/deep_learning_spam_lstm.ipynb)

Proyecto integral de **Deep Learning** que implementa una **Red Neuronal Recurrente Bidireccional (BiLSTM)** en **PyTorch** para la clasificación de datos secuenciales de texto, resolviendo el problema de detección en tiempo real de mensajes **Spam** y ataques de **Phishing** (SMS/mensajería) con **soporte bilingüe en Español 🇪🇸 e Inglés 🇺🇸**.

---

## 📋 Tabla de Contenidos
1. [Descripción del Problema y Dataset](#-1-descripción-del-problema-y-dataset)
2. [Análisis Exploratorio y Preprocesamiento](#-2-análisis-exploratorio-y-preprocesamiento)
3. [División de Datos (Train / Val / Test)](#-3-división-de-datos-train--val--test)
4. [Arquitectura de la Red Neuronal (BiLSTM)](#-4-arquitectura-de-la-red-neuronal-bilstm)
5. [Métricas y Evaluación en Conjunto de Prueba](#-5-métricas-y-evaluación-en-conjunto-de-prueba)
6. [Gráficas del Entrenamiento](#-6-gráficas-del-entrenamiento)
7. [Muestra de 16 Predicciones de Prueba](#-7-muestra-de-16-predicciones-de-prueba)
8. [Estructura del Repositorio](#-8-estructura-del-repositorio)
9. [Instrucciones de Instalación y Ejecución](#-9-instrucciones-de-instalación-y-ejecución)
10. [Aplicación Web con Streamlit](#-10-aplicación-web-con-streamlit)

---

## 📌 1. Descripción del Problema y Dataset

El filtrado de mensajes no deseados (spam) y fraudes por mensajería (smishing/phishing) requiere procesar secuencias de texto breves donde el orden y el contexto de las palabras determinan la intención del mensaje.

- **Dataset seleccionado:** *SMS Spam Collection Dataset* (UCI Machine Learning Repository) enriquecido con un corpus de mensajes en **Español** (alertas bancarias falsas, sorteos fraudulentos, paquetes retenidos, frente a conversaciones cotidianas de trabajo, familia y estudio).
- **Total de registros:** `6,772` mensajes etiquetados (Español + Inglés).
- **Clases:**
  - `ham` (Mensajes legítimos / seguros): **5,675** muestras (`83.80%`).
  - `spam` (Mensajes fraudulentos / no deseados): **1,097** muestras (`16.20%`).

> Documentación ampliada del dataset disponible en [`data/dataset_description.md`](data/dataset_description.md).

---

## 🧹 2. Análisis Exploratorio y Preprocesamiento

El pipeline de limpieza y normalización secuencial implementado en [`src/preprocess.py`](src/preprocess.py) realiza:
1. Conversión a minúsculas.
2. Sustitución semántica de patrones críticos:
   - Enlaces web (`http://...`, `www...`) $\rightarrow$ `<url>`
   - Símbolos monetarios (`£`, `$`, `€`) $\rightarrow$ `<money>`
   - Números telefónicos y códigos de 5+ dígitos $\rightarrow$ `<phone>`
   - Cifras numéricas $\rightarrow$ `<number>`
3. Tokenización por expresiones regulares conservando tokens semánticos.
4. Construcción del vocabulario sobre el conjunto de entrenamiento (`3,544` palabras únicas en español e inglés con frecuencia mínima $\ge 2$), incluyendo tokens especiales `<pad>` (índice `0`) y `<unk>` (índice `1`).
5. Relleno (*post-padding*) y truncado de secuencias a longitud fija `max_len = 80`.

---

## ✂️ 3. División de Datos (Train / Val / Test)

Se aplicó una **partición estratificada** para conservar la proporción exacta de clases en los tres subconjuntos:

| Subconjunto | Porcentaje | Total Muestras | Ham (Legítimo) | Spam (No deseado) | Propósito |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento (Train)** | `70%` | 4,740 | 3,972 | 768 | Ajuste de pesos y construcción del vocabulario |
| **Validación (Validation)** | `15%` | 1,016 | 852 | 164 | Monitoreo por época, *Early Stopping* y selección del mejor modelo |
| **Prueba (Test)** | `15%` | 1,016 | 851 | 165 | Evaluación final totalmente independiente (*Held-Out*) |

---

## 🧠 4. Arquitectura de la Red Neuronal (BiLSTM)

La red neuronal implementada en [`src/model.py`](src/model.py) (`SpamBiLSTM`) combina representaciones densas (*Word Embeddings*), procesamiento recurrente bidireccional de doble capa y agregación global dual (*Mean + Max Pooling*):

```mermaid
flowchart TD
    A["Secuencia de Entrada (Batch, 80)"] --> B["Embedding Layer (Vocab: 3544, Dim: 128)"]
    B --> C["Spatial Embedding Dropout (p = 0.3)"]
    C --> D["Bidirectional LSTM (2 Capas, Hidden: 128, Dropout: 0.3)"]
    D --> E["Global Average Pooling (Dim: 256)"]
    D --> F["Global Max Pooling (Dim: 256)"]
    E --> G["Concatenación Temporal (Dim: 512)"]
    F --> G
    G --> H["Linear (512 -> 64) + BatchNorm1d + ReLU + Dropout (0.3)"]
    H --> I["Linear Output (64 -> 1) + Sigmoid"]
    I --> J["Probabilidad de Spam [0.0, 1.0]"]
```

- **Total de parámetros entrenables:** `1,146,113`
- **Función de pérdida:** `BCEWithLogitsLoss` con `pos_weight` ponderado para mitigar el desbalance de clases.
- **Optimizador:** `AdamW` (`lr = 0.001`, `weight_decay = 1e-4`) + Scheduler `ReduceLROnPlateau`.

---

## 🎯 5. Métricas y Evaluación en Conjunto de Prueba

Resultados obtenidos al evaluar el mejor checkpoint (`saved_models/best_bilstm_model.pt`) sobre las **1,016 muestras del conjunto de prueba**:

| Métrica | Valor |
| :--- | :---: |
| **Exactitud Global (Accuracy)** | **99.21%** |
| **Precisión (Precision - Spam)** | **98.76%** |
| **Sensibilidad (Recall - Spam)** | **96.36%** |
| **F1-Score (Spam)** | **97.55%** |
| **F1-Score Macro Promedio** | **98.54%** |
| **Área Bajo la Curva ROC (ROC-AUC)** | **0.9952** |

### Reporte por Clase (Test Set)
```text
                   precision    recall  f1-score   support

   Ham (Legítimo)     0.9930    0.9976    0.9953       851
Spam (No deseado)     0.9876    0.9636    0.9755       165

         accuracy                         0.9921      1016
        macro avg     0.9903    0.9806    0.9854      1016
     weighted avg     0.9921    0.9921    0.9921      1016
```

---

## 📈 6. Gráficas del Entrenamiento

Las gráficas generadas durante el proceso de entrenamiento se encuentran almacenadas en el directorio [`plots/`](plots/):

1. **Curvas de Aprendizaje (Pérdida y Exactitud por Época):** [`plots/training_curves.png`](plots/training_curves.png)
2. **Matriz de Confusión en Prueba:** [`plots/confusion_matrix.png`](plots/confusion_matrix.png)
3. **Distribución Estratificada de Clases:** [`plots/class_distribution.png`](plots/class_distribution.png)

---

## 📋 7. Muestra de 16 Predicciones de Prueba

A continuación se presentan **16 predicciones reales verificadas** sobre el subconjunto de prueba en Español e Inglés (almacenadas en [`saved_models/sample_predictions.json`](saved_models/sample_predictions.json)):

| # | Mensaje SMS (Conjunto de Prueba) | Etiqueta Real | Predicción BiLSTM | Prob. Spam | Resultado |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | *Servicio Postal: No pudimos entregar su envío el día de hoy. Actualice su dirección de domicilio y pague $1,000 de reenvío en http://bono-regalo-reclamar.info* | `spam` | `spam` | `100.00%` | ✅ Acierto |
| 2 | *ALERTA: Tienes una factura vencida de $500. Evita el reporte negativo y suspensión del servicio pagando de inmediato en https://verifica-tu-cuenta-hoy.net* | `spam` | `spam` | `99.98%` | ✅ Acierto |
| 3 | *Hi, the SEXYCHAT girls are waiting for you to text them...* | `spam` | `spam` | `99.55%` | ✅ Acierto |
| 4 | *Jay wants to work out first, how's 4 sound?* | `ham` | `ham` | `0.29%` | ✅ Acierto |
| 5 | *Hey... are you going to quit soon? Xuhui and I are working till late...* | `ham` | `ham` | `0.05%` | ✅ Acierto |
| 6 | *I'll be late...* | `ham` | `ham` | `0.03%` | ✅ Acierto |
| 7 | *K give me a sec, breaking a <#>* | `ham` | `ham` | `0.03%` | ✅ Acierto |
| 8 | *Ummma.will call after check in.our life will begin from qatar...* | `ham` | `ham` | `0.42%` | ✅ Acierto |
| 9 | *Has recibido una transferencia pendiente por $500. Para liberar los fondos en tu cuenta confirma tu clave dinámica en http://seguridad-banca-alerta.com* | `spam` | `spam` | `99.97%` | ✅ Acierto |
| 10 | *Ela kano.,il download, come wen ur free..* | `ham` | `ham` | `0.27%` | ✅ Acierto |
| 11 | *Fancy a shag? I do.Interested? sextextuk.com txt XXUK SUZY to 69876...* | `spam` | `spam` | `99.94%` | ✅ Acierto |
| 12 | *URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010* | `spam` | `spam` | `100.00%` | ✅ Acierto |
| 13 | *Ok...* | `ham` | `ham` | `0.11%` | ✅ Acierto |
| 14 | *Dear, take care. I am just reaching home.love u a lot.* | `ham` | `ham` | `0.07%` | ✅ Acierto |
| 15 | *Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed £1000 cash or £5000 prize!* | `spam` | `spam` | `99.95%` | ✅ Acierto |
| 16 | *Notificación de Entrega: Su paquete #90500 está retenido en aduana por falta de pago de impuestos de $350. Pague aquí: https://rastreo-paquete-express.com* | `spam` | `spam` | `99.99%` | ✅ Acierto |

---

## 📂 8. Estructura del Repositorio

```text
deep_learning_nlp_lstm/
│
├── data/
│   ├── spam.csv                        # Dataset bilingüe completo (6,772 mensajes en Español e Inglés)
│   ├── sms.tsv                         # Archivo original UCI tabulado
│   └── dataset_description.md          # Documentación y diccionario de datos
│
├── notebooks/
│   └── deep_learning_spam_lstm.ipynb   # Notebook completo de experimentación (Google Colab / Jupyter)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                   # Limpieza, tokenización, Vocabulary y DataLoaders
│   ├── model.py                        # Definición de la arquitectura neuronal SpamBiLSTM
│   ├── train.py                        # Script de entrenamiento, validación, evaluación y gráficas
│   └── predict.py                      # Módulo de inferencia individual, por lote y CLI
│
├── saved_models/
│   ├── best_bilstm_model.pt            # Pesos del modelo BiLSTM entrenado
│   ├── vocab.json                      # Vocabulario serializado (3,544 tokens en ES/EN)
│   ├── model_config.json               # Hiperparámetros del modelo
│   ├── metrics.json                    # Métricas de evaluación en el conjunto de prueba
│   ├── training_history.json           # Historial época por época (Loss, Accuracy, F1)
│   └── sample_predictions.json         # Predicciones detalladas de prueba
│
├── plots/
│   ├── training_curves.png             # Gráfica de Loss y Accuracy (Train vs Val)
│   ├── confusion_matrix.png            # Matriz de confusión en Test Set
│   └── class_distribution.png          # Distribución de clases en Train, Val y Test
│
├── app.py                              # Aplicación web interactiva en Streamlit (100% en Español)
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore                          # Archivos excluidos del control de versiones
└── README.md                           # Documentación principal e instrucciones
```

---

## 🚀 9. Instrucciones de Instalación y Ejecución

### Paso 1: Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd deep_learning_nlp_lstm
```

### Paso 2: Crear un entorno virtual e instalar dependencias
```bash
python -m venv .venv

# Activar en Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activar en Linux / macOS:
source .venv/bin/activate

# Instalar dependencias:
pip install -r requirements.txt
```

### Paso 3: Entrenar el modelo desde cero (Opcional, el modelo entrenado ya está incluido)
```bash
python src/train.py --epochs 6 --batch_size 32 --lr 0.001
```

### Paso 4: Realizar predicciones desde la terminal (CLI) en Español o Inglés
```bash
python src/predict.py --text "¡URGENTE! Su cuenta bancaria ha sido suspendida temporalmente. Verifique su identidad ahora en http://seguridad-banca-alerta.com"
```

---

## 🌐 10. Aplicación Web con Streamlit

Para iniciar la interfaz gráfica interactiva que carga el modelo entrenado y permite clasificar mensajes en tiempo real o por lotes (CSV):

```bash
streamlit run app.py
```

### Funcionalidades de la Aplicación Streamlit:
1. **🔍 Analizador en Tiempo Real:** Ingrese cualquier texto en **español o inglés**, o utilice los **botones de ejemplo rápido en español** para obtener la etiqueta predicha (`SPAM / FRAUDULENTO` vs `LEGÍTIMO / SEGURO`), probabilidad calibrada, nivel de confianza y desglose de tokens identificados.
2. **📊 Métricas & Arquitectura:** Visualice las curvas de entrenamiento, la matriz de confusión, el balance de subconjuntos y el diagrama de la red neuronal BiLSTM.
3. **📋 Predicciones de Prueba (10+):** Explore de forma interactiva las predicciones sobre el conjunto de prueba con filtros dinámicos.
4. **📁 Predicción por Lote (CSV):** Suba un archivo CSV con múltiples mensajes para clasificarlos masivamente y descargar los resultados enriquecidos.
