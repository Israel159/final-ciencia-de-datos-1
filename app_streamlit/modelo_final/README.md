# Modelo Final — Predicción de Consumo Eléctrico

## Modelo seleccionado: GRU (Gated Recurrent Unit)

### Justificación
- **R² = 0.599** (mayor de los 3 modelos)
- **MAE = 0.339 kW** (equivalente a LSTM, 0.003 kW mayor)
- **Convergencia en 23 épocas** (más rápido que RNN y LSTM)
- **Eficiencia paramétrica**: ~25-30% menos parámetros que LSTM

### Métricas comparativas (Test Set 2010)

| Modelo | MAE (kW) | RMSE (kW) | R²   | Épocas |
|--------|----------|-----------|------|--------|
| RNN    | 0.351    | 0.513     | 0.564 | 26     |
| LSTM   | 0.336    | 0.492     | 0.598 | 25     |
| **GRU**| **0.339**| **0.492** | **0.599** | **23** |

### Archivos incluidos

| Archivo | Descripción |
|---------|-------------|
| `modelo_gru_final.keras` | Modelo entrenado (Keras) |
| `min_max_scaler.json` | Parámetros Min-Max para desescalado |
| `pred_gru_24h.npy` | Predicciones futuras 24 horas |
| `pred_gru_168h.npy` | Predicciones futuras 168 horas (1 semana) |
| `metrics_gru.json` | Métricas de evaluación |
| `resumen_comparativa.json` | Resultado completo de la comparación |

### Uso en Streamlit

```python
import tensorflow as tf
model = tf.keras.models.load_model('modelo_gru_final.keras')