# Descripción del Dataset: SMS Spam Collection Dataset

## 1. Contexto y Justificación del Problema
El filtrado de mensajes spam y detección de phishing en comunicaciones móviles y mensajería instantánea es un problema de alta criticidad en ciberseguridad y procesamiento de lenguaje natural (NLP). A diferencia de los correos electrónicos tradicionales, los mensajes SMS presentan características secuenciales particulares:
- Longitud limitada (secuencias breves).
- Uso frecuente de abreviaturas, jerga, faltas ortográficas deliberadas y técnicas de ofuscación (por ejemplo: "WINNER!!", "txt", "call now", "urgent", "£900").
- Variabilidad léxica y mezcla de patrones semánticos.

Las Redes Neuronales Recurrentes (RNN), específicamente las arquitecturas **LSTM (Long Short-Term Memory)** y **BiLSTM (Bidirectional LSTM)**, son idóneas para este problema porque procesan la información de manera secuencial conservando dependencias temporales y contextuales en ambas direcciones (hacia adelante y hacia atrás), superando las limitaciones de los modelos tradicionales tipo Bag-of-Words o Naive Bayes.

## 2. Origen del Dataset
- **Fuente original:** UCI Machine Learning Repository (Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. 2011).
- **Referencia bibliográfica:** *Contributions to the Study of SMS Spam Filtering: New Collection and Results*, ACM Symposium on Document Engineering (DocEng'11).
- **Archivo en el proyecto:** `data/spam.csv` (y `data/sms.tsv`).

## 3. Estructura y Dimensiones
El dataset contiene **5,572 mensajes SMS** en idioma inglés etiquetados manualmente:

| Columna | Tipo de Dato | Descripción | Valores Posibles |
| :--- | :--- | :--- | :--- |
| `label` | Categórico / Cadena | Etiqueta de clasificación del mensaje | `ham` (legítimo), `spam` (no deseado / fraudulento) |
| `message` | Texto libre | Contenido textual del mensaje SMS | Cadena de caracteres alfanuméricos y símbolos |

### Distribución de Clases
- **Ham (mensajes legítimos):** 4,825 ejemplos (86.6%)
- **Spam (mensajes fraudulentos):** 747 ejemplos (13.4%)
- **Total:** 5,572 ejemplos

> **Nota sobre Desbalance de Clases:**  
> Como es característico en escenarios reales de detección de fraude y anomalías, existe un desbalance inherente (~6.5 a 1). Este desbalance se aborda en el proyecto mediante:
> 1. Estratificación en la división de datos (Train 70%, Validation 15%, Test 15%).
> 2. Ponderación de pérdida positiva (`pos_weight`) en la función `BCEWithLogitsLoss`.
> 3. Evaluación rigurosa basada en F1-Score, Precisión, Recall y Matriz de Confusión, más allá de la simple exactitud (Accuracy).

## 4. Ejemplos Representativos

### Mensajes Legítimos (`ham`):
1. *"I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."*
2. *"I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted..."*
3. *"Ok lar... Joking wif u oni..."*

### Mensajes Spam (`spam`):
1. *"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."*
2. *"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question..."*
3. *"URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010..."*
