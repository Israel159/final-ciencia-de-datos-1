# 🔌 Predicción de Consumo Eléctrico Residencial con Deep Learning y Big Data

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
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

---

## 🏗️ Pipeline del Proyecto

| Etapa | Archivo | Descripcion |
|-------|---------|-------------|
| 01 | `01_exploracion_pyspark.ipynb` | EDA con PySpark sobre 2M+ registros |
| 02 | `02_preprocesamiento.ipynb` | Limpieza, features, escalado |
| 03 | `03_modelo_rnn.py` | Baseline SimpleRNN |
| 04 | `04_modelo_lstm.py` | LSTM con memoria de largo plazo |
| 05 | `05_modelo_gru.py` | GRU (ganador) |
| 06 | `06_comparacion_modelos.ipynb` | Tabla comparativa y seleccion |
| App | `app_streamlit/app.py` | Dashboard interactivo Streamlit |

**Flujo de datos:**


## 🏆 Resultados
| Modelo    | MAE (kW)   | RMSE (kW)  | R²         | Épocas |
| --------- | ---------- | ---------- | ---------- | ------ |
| **RNN**   | 0.3508     | 0.5130     | 0.5638     | 26     |
| **LSTM**  | 0.3360     | 0.4924     | 0.5981     | 25     |
| **GRU** ⭐ | **0.3392** | **0.4918** | **0.5991** | **23** |

Modelo ganador: GRU
- Mayor R² (59.9% de variabilidad explicada)
- Convergencia más rápida (23 épocas)
- Eficiencia paramétrica superior a LSTM (~25-30% menos parámetros)

## 🛠️ Tecnologías Utilizadas
| Capa                 | Herramientas                         |
| -------------------- | ------------------------------------ |
| **Big Data**         | PySpark, Spark SQL, Window functions |
| **Preprocesamiento** | pandas, numpy, Parquet               |
| **Deep Learning**    | TensorFlow, Keras                    |
| **Visualización**    | matplotlib, seaborn, plotly          |
| **Dashboard**        | Streamlit                            |
| **Evaluación**       | scikit-learn (MAE, RMSE, R²)         |

## 📚 Referencias
- Hebrail, G. & Berard, A. (2012). Individual Household Electric Power Consumption.
