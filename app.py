"""
Aplicación Web Interactiva con Streamlit para Clasificación y Detección de Mensajes Spam
utilizando una Red Neuronal Recurrente Bidireccional (BiLSTM) en PyTorch.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página
st.set_page_config(
    page_title="Detección de Spam con BiLSTM | Deep Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Añadir raíz al path para importar src
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.predict import SpamPredictor
from src.preprocess import clean_text, tokenize


# Inyectar estilos CSS modernos
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .spam-badge {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.3);
    }
    .ham-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    .token-chip {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-family: monospace;
        margin: 3px;
        border: 1px solid #c7d2fe;
    }
    .token-chip-special {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-family: monospace;
        font-weight: bold;
        margin: 3px;
        border: 1px solid #fecaca;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Carga y cachea la instancia del modelo predictor."""
    try:
        predictor = SpamPredictor()
        return predictor, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_metrics():
    """Carga las métricas de prueba guardadas."""
    path = os.path.join(SAVED_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_sample_predictions():
    """Carga las predicciones de prueba generadas."""
    path = os.path.join(SAVED_DIR, "sample_predictions.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_model_config():
    """Carga la configuración de hiperparámetros del modelo."""
    path = os.path.join(SAVED_DIR, "model_config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # Encabezado principal
    st.markdown("<div class='main-title'>🛡️ Detección de Mensajes Spam y Phishing</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Proyecto de Deep Learning con Redes Neuronales Recurrentes Bidireccionales (BiLSTM) en PyTorch</div>", unsafe_allow_html=True)

    # Cargar predictor y métricas
    predictor, error_msg = load_predictor()
    metrics = load_metrics()
    config = load_model_config()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Especificaciones Técnicas")
        st.markdown("**Arquitectura:** BiLSTM (Bidirectional LSTM)")
        st.markdown("**Framework:** PyTorch 2.x")
        
        if config:
            st.markdown(f"- **Vocabulario:** `{config.get('vocab_size', 'N/A'):,}` palabras")
            st.markdown(f"- **Dimensión Embeddings:** `{config.get('embed_dim', 'N/A')}`")
            st.markdown(f"- **Dimensión Oculta LSTM:** `{config.get('hidden_dim', 'N/A')}`")
            st.markdown(f"- **Capas LSTM:** `{config.get('num_layers', 'N/A')}` (Bidireccional)")
            st.markdown(f"- **Dropout:** `{config.get('dropout', 'N/A')}`")
            st.markdown(f"- **Longitud Máx. Secuencia:** `{config.get('max_len', 80)}` tokens")
            st.markdown(f"- **Pooling:** Concat Mean + Max")
        
        st.divider()
        st.subheader("🎯 Desempeño en Test Set")
        if metrics:
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            col_m2.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
            col_m3, col_m4 = st.columns(2)
            col_m3.metric("Precision", f"{metrics['precision']*100:.2f}%")
            col_m4.metric("Recall", f"{metrics['recall']*100:.2f}%")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        else:
            st.info("Entrenando modelo para mostrar métricas...")
            
        st.divider()
        st.markdown("🌎 **Idiomas soportados:** Español 🇪🇸 & Inglés 🇺🇸")
        st.markdown("👨‍💻 **Dataset:** [UCI SMS Spam + Corpus Español](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)")
        st.markdown("📦 **Muestras:** 6,772 mensajes")

    if error_msg:
        st.error(f"⚠️ El modelo aún no ha sido entrenado o no se encontraron los archivos en `saved_models/`.\nError: {error_msg}")
        st.info("Por favor ejecute `python src/train.py` para entrenar la red neuronal y generar los checkpoints.")
        return

    # --- PESTAÑAS PRINCIPALES ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Analizador en Tiempo Real",
        "📊 Métricas & Arquitectura",
        "📋 Predicciones de Prueba (10+)",
        "📁 Predicción por Lote (CSV)"
    ])

    # ==========================================
    # PESTAÑA 1: ANALIZADOR EN TIEMPO REAL
    # ==========================================
    with tab1:
        st.subheader("Ingresar Mensaje para Clasificación (Español o Inglés)")
        st.write("Escriba cualquier mensaje de texto SMS, correo o notificación en **español** o **inglés** para evaluar su probabilidad de Spam o Phishing en tiempo real.")

        # Botones de ejemplos rápidos en Español
        st.markdown("**Ejemplos predefinidos para prueba rápida (Español e Inglés):**")
        ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
        
        ejemplo_seleccionado = ""
        if ex_col1.button("🚨 Premio $1,000 (Spam ES)"):
            ejemplo_seleccionado = "¡FELICIDADES! Has sido seleccionado como ganador de un premio en efectivo de $1,000. Reclama tu código 87121 llamando al 09061701461 hoy mismo."
        if ex_col2.button("🚨 Alerta Bancaria (Spam ES)"):
            ejemplo_seleccionado = "¡URGENTE! Su cuenta bancaria Bancolombia ha sido suspendida temporalmente por actividad sospechosa. Verifique su identidad ahora en http://seguridad-banca-alerta.com"
        if ex_col3.button("🟢 Reunión Trabajo (Legítimo ES)"):
            ejemplo_seleccionado = "Hola Carlos, ¿cómo vas? Te escribo para confirmar nuestra reunión de trabajo mañana a las 10:00 AM en la oficina. Nos vemos."
        if ex_col4.button("🟢 Mensaje Familiar (Legítimo ES)"):
            ejemplo_seleccionado = "Mamá, ya salí de la universidad, voy en camino a casa. ¿Necesitas que compre algo en el supermercado para la cena?"

        # Área de texto
        default_val = ejemplo_seleccionado if ejemplo_seleccionado else "¡URGENTE! Su cuenta bancaria ha sido bloqueada temporalmente. Ingrese inmediatamente a https://verifica-tu-cuenta-hoy.net para evitar la suspensión definitiva."
        user_input = st.text_area(
            "Mensaje de texto a evaluar:",
            value=default_val,
            height=110,
            placeholder="Escriba aquí su mensaje en español o inglés..."
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            analizar_btn = st.button("🚀 Clasificar Mensaje", type="primary", use_container_width=True)

        if user_input.strip() and (analizar_btn or ejemplo_seleccionado):
            with st.spinner("Procesando secuencia a través de la red neuronal BiLSTM..."):
                res = predictor.predict(user_input)

            st.divider()
            
            # Badge de resultado
            col_res1, col_res2 = st.columns([1.2, 1])
            with col_res1:
                if res["is_spam"]:
                    st.markdown("""
                    <div class='spam-badge'>
                        🚨 ALERTA: MENSAJE SPAM / FRAUDULENTO DETECTADO
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='ham-badge'>
                        🛡️ MENSAJE LEGÍTIMO (SEGURO) - NO SE DETECTA AMENAZA
                    </div>
                    """, unsafe_allow_html=True)

                # Tarjetas de probabilidades
                c_p1, c_p2, c_p3 = st.columns(3)
                c_p1.metric("Probabilidad de Spam", f"{res['spam_probability']*100:.2f}%")
                c_p2.metric("Probabilidad Legítimo", f"{res['ham_probability']*100:.2f}%")
                c_p3.metric("Confianza del Modelo", f"{res['confidence']*100:.2f}%")

                # Barra de probabilidad de Spam
                st.write("**Probabilidad Estimada de Spam:**")
                st.progress(float(res["spam_probability"]))

            with col_res2:
                st.markdown("#### 🔬 Desglose Secuencial de Tokens")
                st.write(f"- **Longitud en caracteres:** {len(user_input)}")
                st.write(f"- **Tokens reconocidos:** {res['token_count']}")
                st.write(f"- **Texto normalizado:** `{res['cleaned_text']}`")

                st.write("**Secuencia de Tokens Identificados:**")
                chips_html = ""
                for t in res["tokens"]:
                    if t.startswith("<") and t.endswith(">"):
                        chips_html += f"<span class='token-chip-special'>{t}</span>"
                    else:
                        chips_html += f"<span class='token-chip'>{t}</span>"
                st.markdown(chips_html, unsafe_allow_html=True)

    # ==========================================
    # PESTAÑA 2: MÉTRICAS Y ARQUITECTURA
    # ==========================================
    with tab2:
        st.subheader("Evaluación Global del Modelo y Curvas de Entrenamiento")
        
        col_g1, col_g2 = st.columns(2)
        
        # Gráfica de curvas de entrenamiento
        training_plot_path = os.path.join(PLOTS_DIR, "training_curves.png")
        if os.path.exists(training_plot_path):
            with col_g1:
                st.markdown("#### 📈 Curvas de Aprendizaje (Loss y Accuracy)")
                st.image(training_plot_path, use_container_width=True)
        else:
            with col_g1:
                st.info("La gráfica de curvas de entrenamiento estará disponible tras ejecutar el entrenamiento.")

        # Gráfica de matriz de confusión
        cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_plot_path):
            with col_g2:
                st.markdown("#### 🗂️ Matriz de Confusión en Conjunto de Prueba")
                st.image(cm_plot_path, use_container_width=True)
        else:
            with col_g2:
                st.info("La matriz de confusión se generará al finalizar el entrenamiento.")

        st.divider()

        # Distribución de clases y arquitectura
        col_arch1, col_arch2 = st.columns(2)
        dist_plot_path = os.path.join(PLOTS_DIR, "class_distribution.png")
        if os.path.exists(dist_plot_path):
            with col_arch1:
                st.markdown("#### ⚖️ Balance de Clases en Subconjuntos")
                st.image(dist_plot_path, use_container_width=True)
                
        with col_arch2:
            st.markdown("#### 🧠 Diagrama de Arquitectura de la Red")
            st.markdown("""
            ```mermaid
            flowchart TD
                A["Texto de Entrada SMS"] --> B["Limpieza y Tokenización"]
                B --> C["Padding / Truncado a 80 tokens"]
                C --> D["Capa Embedding (Dim: 128)"]
                D --> E["Dropout (0.3)"]
                E --> F["BiLSTM Capa 1 (128 unidades)"]
                F --> G["BiLSTM Capa 2 (128 unidades)"]
                G --> H["Global Mean Pooling"]
                G --> I["Global Max Pooling"]
                H --> J["Concatenación (Dim: 512)"]
                I --> J
                J --> K["Capa Densa (64) + BatchNorm + ReLU + Dropout"]
                K --> L["Proyección Lineal (1) + Sigmoid"]
                L --> M["Probabilidad Spam [0.0 - 1.0]"]
            ```
            """)

    # ==========================================
    # PESTAÑA 3: PREDICCIONES DE PRUEBA (10+)
    # ==========================================
    with tab3:
        st.subheader("Muestra de Predicciones sobre el Conjunto de Prueba Independiente")
        st.write("A continuación se presentan predicciones detalladas sobre mensajes nunca vistos durante el entrenamiento (Held-Out Test Set).")

        samples = load_sample_predictions()
        if samples:
            df_samples = pd.DataFrame(samples)
            
            # Filtros interactivos
            filtro_clase = st.selectbox(
                "Filtrar por etiqueta:",
                ["Todos los ejemplos", "Solo Spam (No deseado)", "Solo Ham (Legítimo)", "Solo Aciertos", "Solo Fallos"]
            )
            
            df_filtrado = df_samples.copy()
            if filtro_clase == "Solo Spam (No deseado)":
                df_filtrado = df_filtrado[df_filtrado["true_label"] == "spam"]
            elif filtro_clase == "Solo Ham (Legítimo)":
                df_filtrado = df_filtrado[df_filtrado["true_label"] == "ham"]
            elif filtro_clase == "Solo Aciertos":
                df_filtrado = df_filtrado[df_filtrado["is_correct"] == True]
            elif filtro_clase == "Solo Fallos":
                df_filtrado = df_filtrado[df_filtrado["is_correct"] == False]

            # Formatear tabla para presentación en español
            display_df = df_filtrado.copy()
            display_df["spam_probability"] = display_df["spam_probability"].apply(lambda p: f"{p*100:.2f}%")
            display_df["Estado"] = display_df["is_correct"].apply(lambda ok: "✅ Acierto" if ok else "❌ Fallo")
            label_map_es = {"ham": "🟢 LEGÍTIMO", "spam": "🚨 SPAM"}
            display_df["true_label"] = display_df["true_label"].map(lambda x: label_map_es.get(x, x.upper()))
            display_df["predicted_label"] = display_df["predicted_label"].map(lambda x: label_map_es.get(x, x.upper()))
            
            display_df = display_df.rename(columns={
                "id": "ID",
                "message": "Mensaje SMS",
                "true_label": "Etiqueta Real",
                "predicted_label": "Predicción BiLSTM",
                "spam_probability": "Prob. Spam",
                "Estado": "Resultado"
            })
            
            cols_to_show = ["ID", "Mensaje SMS", "Etiqueta Real", "Predicción BiLSTM", "Prob. Spam", "Resultado"]
            st.dataframe(
                display_df[cols_to_show],
                use_container_width=True,
                height=450
            )
            
            total_m = len(df_filtrado)
            aciertos_m = df_filtrado["is_correct"].sum()
            st.caption(f"Mostrando {total_m} ejemplos seleccionados. Tasa de acierto en este subconjunto: {aciertos_m}/{total_m} ({aciertos_m/total_m*100:.1f}%)")
        else:
            st.info("Ejecute el entrenamiento para generar el conjunto de predicciones de prueba.")

    # ==========================================
    # PESTAÑA 4: PREDICCIÓN POR LOTE (CSV)
    # ==========================================
    with tab4:
        st.subheader("Clasificación Masiva de Mensajes (Batch Inference)")
        st.write("Suba un archivo CSV que contenga una columna de texto con los mensajes a analizar (nombre de columna sugerido: `message` o `texto`).")

        uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Archivo cargado exitosamente. Total de filas: **{len(batch_df):,}**")
                
                # Identificar columna de texto
                text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
                col_sel = st.selectbox("Seleccione la columna que contiene el texto de los mensajes:", text_cols)
                
                if st.button("⚡ Procesar Todo el Lote", type="primary"):
                    with st.spinner("Procesando mensajes con el modelo BiLSTM..."):
                        preds_res = predictor.predict_batch(batch_df[col_sel].astype(str).tolist())
                        
                    batch_df["Prediccion"] = ["SPAM" if r["is_spam"] else "LEGÍTIMO" for r in preds_res]
                    batch_df["Prob_Spam"] = [round(r["spam_probability"] * 100, 2) for r in preds_res]
                    batch_df["Confianza"] = [round(r["confidence"] * 100, 2) for r in preds_res]
                    
                    st.success("¡Procesamiento completado con éxito!")
                    
                    # Gráfica de conteo de predicciones
                    spam_count = (batch_df["Prediccion"] == "SPAM").sum()
                    ham_count = (batch_df["Prediccion"] == "LEGÍTIMO").sum()
                    
                    col_b1, col_b2 = st.columns([1, 2])
                    with col_b1:
                        st.metric("Spam Detectados", f"{spam_count} ({spam_count/len(batch_df)*100:.1f}%)")
                        st.metric("Mensajes Legítimos", f"{ham_count} ({ham_count/len(batch_df)*100:.1f}%)")
                    with col_b2:
                        st.dataframe(batch_df.head(20), use_container_width=True)
                        
                    # Descargar CSV
                    csv_export = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar Resultados en CSV",
                        data=csv_export,
                        file_name="predicciones_bilstm_spam.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error al procesar el archivo CSV: {str(e)}")


if __name__ == "__main__":
    main()
