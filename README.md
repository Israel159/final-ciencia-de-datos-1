# 🔌 Predicción de Consumo Eléctrico Residencial con Deep Learning y Big Data

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-orange.svg)](https://spark.apache.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io)

> **Proyecto Final — Ciencia de Datos I**  
> Predicción de series de tiempo multivariadas con redes neuronales recurrentes (RNN/LSTM/GRU) en entorno Big Data con PySpark.

---

## 📊 Dataset

**Individual Household Electric Power Consumption**  
🔗 [Kaggle — UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

| Característica | Descripción |
|----------------|-------------|
| **Período** | Diciembre 2006 — Noviembre 2010 (47 meses) |
| **Registros** | 2,075,259 mediciones minutales |
| **Tamaño** | ~130 MB (comprimido) |
| **Tipo** | Serie temporal multivariada |
| **Frecuencia** | 1 minuto |

### Variables del dataset

| # | Variable | Descripción | Unidad |
|---|----------|-------------|--------|
| 1 | `Date` | Fecha | dd/mm/aaaa |
| 2 | `Time` | Hora | hh:mm:ss |
| 3 | `Global_active_power` | Potencia activa promedio | kilovatios (kW) |
| 4 | `Global_reactive_power` | Potencia reactiva promedio | kilovatios (kW) |
| 5 | `Voltage` | Voltaje promedio | voltios (V) |
| 6 | `Global_intensity` | Intensidad de corriente | amperios (A) |
| 7 | `Sub_metering_1` | Cocina (lavavajillas, horno, microondas) | vatios-hora (Wh) |
| 8 | `Sub_metering_2` | Lavandería (lavadora, secadora, refrigerador) | vatios-hora (Wh) |
| 9 | `Sub_metering_3` | Calefacción/AC (calentador de agua, aire acondicionado) | vatios-hora (Wh) |

### Notas del dataset

- **Valores faltantes:** ~1.25% de las filas, representados por `?` entre separadores `;`
- **Energía no medida:** `(global_active_power × 1000 / 60) - sub_metering_1 - sub_metering_2 - sub_metering_3` representa el consumo de equipos no monitoreados individualmente (~45% del total)
- **Ejemplo de registro:**
Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
plain

---

## 🏗️ Pipeline del Proyecto
┌─────────────────────────────────────────────────────────────────────────┐
│  01. EXPLORACIÓN (PySpark)                                              │
│     • Lectura de 2M+ registros con esquema explícito                   │
│     • Análisis de nulos, outliers, distribuciones                      │
│     • Visualización de patrones estacionales y cíclicos                │
├─────────────────────────────────────────────────────────────────────────┤
│  02. PREPROCESAMIENTO (PySpark + pandas híbrido)                        │
│     • Imputación temporal (ffill/bfill)                                │
│     • Agregación horaria (factor 60× de reducción)                     │
│     • Ingeniería de features: lags, ventanas móviles, diferencias    │
│     • Codificación cíclica (hora, día, mes)                            │
│     • Escalado Min-Max [0,1] sin fuga de datos                         │
├─────────────────────────────────────────────────────────────────────────┤
│  03-05. MODELADO (TensorFlow/Keras)                                     │
│     • RNN:  2 capas SimpleRNN (64+32) — baseline                        │
│     • LSTM: 2 capas LSTM (64+32) — memoria de largo plazo              │
│     • GRU:  2 capas GRU (64+32) — eficiencia paramétrica               │
│     • Tuning: Early stopping, ReduceLROnPlateau, dropout 20%           │
├─────────────────────────────────────────────────────────────────────────┤
│  06. COMPARACIÓN                                                        │
│     • Métricas: MAE, RMSE, R² en test set temporal (2010)             │
│     • Selección del modelo ganador por R²                               │
├─────────────────────────────────────────────────────────────────────────┤
│  DASHBOARD (Streamlit)                                                  │
│     • Visualización interactiva de datos históricos                    │
│     • Predicciones futuras 24h y 168h                                  │
│     • Comparación de modelos con métricas y justificación              │
└─────────────────────────────────────────────────────────────────────────┘
plain

---

## 📁 Estructura del Repositorio
├── 01_exploracion_pyspark.ipynb      # EDA con PySpark
├── 02_preprocesamiento.ipynb         # Pipeline de limpieza y features
├── 03_modelo_rnn.py                  # Baseline SimpleRNN
├── 04_modelo_lstm.py                 # LSTM con memoria de largo plazo
├── 05_modelo_gru.py                  # GRU (ganador)
├── 06_comparacion_modelos.ipynb      # Tabla comparativa y selección
├── app_streamlit/                    # Dashboard interactivo
│   ├── app.py                        # Aplicación Streamlit
│   ├── requirements.txt              # Dependencias del dashboard
│   └── modelo_final/                 # Artefactos del modelo ganador
│       ├── min_max_scaler.json       # Parámetros de escalado
│       ├── pred_gru_24h.npy          # Predicciones pre-calculadas 24h
│       ├── pred_gru_168h.npy         # Predicciones pre-calculadas 168h
│       ├── metrics_gru.json          # Métricas de evaluación
│       └── resumen_comparativa.json  # Justificación del ganador
├── requirements.txt                  # Dependencias globales del proyecto
├── .gitignore                        # Exclusión de datos y modelos grandes
└── README.md                         # Este archivo
plain

> **Nota:** Los archivos de datos (`household_power_consumption.txt`, `preprocessed_data.zip`) y modelos entrenados (`*.keras`, `*.h5`) **no se incluyen en el repositorio** por tamaño. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) y ejecuta los notebooks en orden.

