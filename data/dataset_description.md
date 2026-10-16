# Descripción del Dataset: SMS Spam Collection + Corpus en Español (Bilingüe)

## 1. Contexto y Justificación del Problema
El filtrado de mensajes spam y detección de phishing en comunicaciones móviles y mensajería instantánea es un problema de alta criticidad en ciberseguridad y procesamiento de lenguaje natural (NLP). A diferencia de los correos electrónicos tradicionales, los mensajes SMS presentan características secuenciales particulares:
- Longitud limitada (secuencias breves).
- Uso frecuente de abreviaturas, jerga, faltas ortográficas deliberadas y técnicas de ofuscación (por ejemplo: "¡URGENTE!", "WINNER!!", "reclama ahora", "call now", "$1,000", "£900").
- Variabilidad léxica y mezcla de patrones semánticos tanto en **Español** como en **Inglés**.

Las Redes Neuronales Recurrentes (RNN), específicamente las arquitecturas **LSTM (Long Short-Term Memory)** y **BiLSTM (Bidirectional LSTM)**, son idóneas para este problema porque procesan la información de manera secuencial conservando dependencias temporales y contextuales en ambas direcciones (hacia adelante y hacia atrás), superando las limitaciones de los modelos tradicionales tipo Bag-of-Words o Naive Bayes.

## 2. Origen y Composición del Dataset
- **Fuente base en inglés:** UCI Machine Learning Repository (Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. 2011 - *SMS Spam Collection Dataset*, 5,572 registros).
- **Extensión en español:** Corpus complementario de `1,200` mensajes SMS en español (alertas bancarias falsas, sorteos fraudulentos, paquetes retenidos, préstamos engañosos, frente a mensajes legítimos cotidianos de trabajo, familia y universidad).
- **Archivo en el proyecto:** `data/spam.csv`.

## 3. Estructura y Dimensiones
El dataset bilingüe contiene **6,772 mensajes SMS** en Español e Inglés etiquetados:

| Columna | Tipo de Dato | Descripción | Valores Posibles |
| :--- | :--- | :--- | :--- |
| `label` | Categórico / Cadena | Etiqueta de clasificación del mensaje | `ham` (legítimo / seguro), `spam` (no deseado / fraudulento) |
| `message` | Texto libre | Contenido textual del mensaje SMS (ES / EN) | Cadena de caracteres alfanuméricos y símbolos |

### Distribución de Clases
- **Ham (mensajes legítimos):** 5,675 ejemplos (83.8%)
- **Spam (mensajes fraudulentos):** 1,097 ejemplos (16.2%)
- **Total:** 6,772 ejemplos

## 4. Ejemplos Representativos

### Mensajes Legítimos (`ham` - Español e Inglés):
1. *"Hola Carlos, ¿cómo vas? Te escribo para confirmar nuestra reunión de trabajo mañana a las 10:00 AM en la oficina."*
2. *"Mamá, ya salí de la universidad, voy en camino a casa. ¿Necesitas que compre algo en el supermercado para la cena?"*
3. *"I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?"*

### Mensajes Spam (`spam` - Español e Inglés):
1. *"¡URGENTE! Su cuenta bancaria Bancolombia ha sido suspendida temporalmente por actividad sospechosa. Verifique su identidad ahora en http://seguridad-banca-alerta.com"*
2. *"¡FELICIDADES! Has sido seleccionado como ganador de un premio en efectivo de $1,000. Reclama tu código 87121 llamando al 09061701461 hoy mismo."*
3. *"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461."*
