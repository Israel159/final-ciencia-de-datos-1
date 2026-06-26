# -*- coding: utf-8 -*-
"""
Dashboard de Predicción de Consumo Eléctrico — Streamlit
======================================================

Aplicación interactiva para visualizar datos, comparar modelos (RNN/LSTM/GRU),
ejecutar predicciones futuras y mostrar métricas del modelo ganador (GRU).

Desplegado en: Streamlit Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predicción de Consumo Eléctrico",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# RUTAS DE ARCHIVOS (relativas al directorio de la app)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "modelo_gru_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "min_max_scaler.json")
RESUMEN_PATH = os.path.join(BASE_DIR, "resumen_comparativa.json")

# ============================================================
# CARGA DE RECURSOS (con cache para no recargar en cada interacción)
# ============================================================
@st.cache_resource
def load_model():
    """Carga el modelo GRU entrenado."""
    return keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_scaler():
    """Carga los parámetros de escalado Min-Max."""
    with open(SCALER_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_resumen():
    """Carga el resumen comparativo de modelos."""
    with open(RESUMEN_PATH, "r") as f:
        return json.load(f)

# ============================================================
# FEATURE COLUMNS (reconstruidas desde el código de los notebooks)
# ============================================================
# Orden exacto en que se crearon en create_sequences()
# scaled_cols se ordenan alfabéticamente + cyclic_cols ordenadas + IsWeekend

SCALED_COLS = sorted([
    'Global_active_power_scaled',
    'Global_active_power_lag12h_scaled',
    'Global_active_power_lag168h_scaled',
    'Global_active_power_lag1h_scaled',
    'Global_active_power_lag24h_scaled',
    'Global_active_power_lag2h_scaled',
    'Global_active_power_lag3h_scaled',
    'Global_active_power_lag6h_scaled',
    'Global_intensity_scaled',
    'Global_reactive_power_log_scaled',
    'Global_reactive_power_scaled',
    'Sub_metering_1_log_scaled',
    'Sub_metering_1_scaled',
    'Sub_metering_2_log_scaled',
    'Sub_metering_2_scaled',
    'Sub_metering_3_log_scaled',
    'Sub_metering_3_scaled',
    'Voltage_scaled',
    'Unmetered_energy_scaled',
    'GAP_ma24h_scaled',
    'GAP_ma168h_scaled',
    'GAP_std24h_scaled',
    'SM3_ma24h_scaled',
    'GAP_diff1h_scaled',
    'GAP_diff24h_scaled',
    'GAP_diff168h_scaled',
    'Sub_metering_1_lag1h_scaled',
    'Sub_metering_1_lag24h_scaled',
    'Sub_metering_1_lag168h_scaled',
    'Sub_metering_2_lag1h_scaled',
    'Sub_metering_2_lag24h_scaled',
    'Sub_metering_2_lag168h_scaled',
    'Sub_metering_3_lag1h_scaled',
    'Sub_metering_3_lag2h_scaled',
    'Sub_metering_3_lag3h_scaled',
    'Sub_metering_3_lag6h_scaled',
    'Sub_metering_3_lag12h_scaled',
    'Sub_metering_3_lag24h_scaled',
    'Sub_metering_3_lag168h_scaled',
    'Voltage_lag1h_scaled',
    'Voltage_lag2h_scaled',
    'Voltage_lag3h_scaled',
    'Voltage_lag6h_scaled',
    'Voltage_lag12h_scaled',
    'Voltage_lag24h_scaled',
    'Voltage_lag168h_scaled',
    'Global_reactive_power_lag1h_scaled',
    'Global_reactive_power_lag2h_scaled',
    'Global_reactive_power_lag3h_scaled',
    'Global_reactive_power_lag6h_scaled',
    'Global_reactive_power_lag12h_scaled',
    'Global_reactive_power_lag24h_scaled',
    'Global_reactive_power_lag168h_scaled',
])

CYCLIC_COLS = [
    'DayOfWeek_cos', 'DayOfWeek_sin',
    'Hour_cos', 'Hour_sin',
    'Month_cos', 'Month_sin'
]

FEATURE_COLS = SCALED_COLS + CYCLIC_COLS + ['IsWeekend']

LOOKBACK = 24  # horas de historia

# Índices de variables clave para actualizar en forecasting
TARGET_IDX = FEATURE_COLS.index('Global_active_power_scaled')
HOUR_SIN_IDX = FEATURE_COLS.index('Hour_sin')
HOUR_COS_IDX = FEATURE_COLS.index('Hour_cos')
DOW_SIN_IDX = FEATURE_COLS.index('DayOfWeek_sin')
DOW_COS_IDX = FEATURE_COLS.index('DayOfWeek_cos')
MONTH_SIN_IDX = FEATURE_COLS.index('Month_sin')
MONTH_COS_IDX = FEATURE_COLS.index('Month_cos')
LAG1_IDX = FEATURE_COLS.index('Global_active_power_lag1h_scaled') if 'Global_active_power_lag1h_scaled' in FEATURE_COLS else None

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def apply_minmax_scaling(value, col_name, scaler_dict):
    """Aplica escalado Min-Max a un valor individual."""
    params = scaler_dict[col_name.replace('_scaled', '')]
    min_val = params['min']
    max_val = params['max']
    denom = max_val - min_val if (max_val - min_val) != 0 else 1e-8
    return (value - min_val) / denom

def inverse_minmax_scaling(scaled_value, col_name, scaler_dict):
    """Invierte el escalado Min-Max."""
    params = scaler_dict[col_name.replace('_scaled', '')]
    min_val = params['min']
    max_val = params['max']
    return scaled_value * (max_val - min_val) + min_val

def predict_future(model, last_sequence, n_steps, scaler_dict, start_dt):
    """
    Genera predicciones futuras iterativas (autoregresivas).
    Actualiza variables cíclicas (hora, día de semana, mes) en cada paso.
    """
    predictions_scaled = []
    current_seq = last_sequence.copy().astype(np.float32)

    gap_min = scaler_dict['Global_active_power']['min']
    gap_max = scaler_dict['Global_active_power']['max']

    current_dt = pd.to_datetime(start_dt)

    for step in range(n_steps):
        # Predicción del siguiente paso
        pred = model.predict(
            current_seq.reshape(1, LOOKBACK, len(FEATURE_COLS)),
            verbose=0
        )
        pred_val = pred[0, 0]
        predictions_scaled.append(pred_val)

        # Avanzar 1 hora
        current_dt += pd.Timedelta(hours=1)
        h = current_dt.hour
        dow = (current_dt.weekday() + 2) % 7
        if dow == 0:
            dow = 7
        m = current_dt.month

        # Crear nuevo timestep
        new_step = current_seq[-1].copy()
        new_step[TARGET_IDX] = pred_val

        # Actualizar variables cíclicas
        new_step[HOUR_SIN_IDX] = np.sin(2 * np.pi * h / 24)
        new_step[HOUR_COS_IDX] = np.cos(2 * np.pi * h / 24)
        new_step[DOW_SIN_IDX] = np.sin(2 * np.pi * (dow - 1) / 7)
        new_step[DOW_COS_IDX] = np.cos(2 * np.pi * (dow - 1) / 7)
        new_step[MONTH_SIN_IDX] = np.sin(2 * np.pi * (m - 1) / 12)
        new_step[MONTH_COS_IDX] = np.cos(2 * np.pi * (m - 1) / 12)

        # Actualizar lag1h
        if LAG1_IDX is not None:
            new_step[LAG1_IDX] = pred_val

        # Desplazar ventana
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = new_step

    # Desescalar a kW
    return np.array(predictions_scaled) * (gap_max - gap_min) + gap_min


def generate_synthetic_last_window(scaler_dict):
    """
    Genera una ventana sintética de 24h para demostración.
    Simula un patrón realista de consumo eléctrico residencial.
    """
    np.random.seed(42)
    window = np.zeros((LOOKBACK, len(FEATURE_COLS)), dtype=np.float32)

    # Simular consumo con patrón diario (bajo de madrugada, pico mañana/noche)
    hours = np.arange(LOOKBACK)
    base_consumption = 0.5 + 0.8 * np.sin(2 * np.pi * (hours - 6) / 24) ** 2
    base_consumption += np.random.normal(0, 0.1, LOOKBACK)
    base_consumption = np.clip(base_consumption, 0.1, 3.0)

    # Fecha de inicio (último día disponible del dataset: finales de noviembre 2010)
    start_dt = pd.to_datetime("2010-11-26 00:00:00")

    for i in range(LOOKBACK):
        dt = start_dt + pd.Timedelta(hours=i)
        h = dt.hour
        dow = (dt.weekday() + 2) % 7
        if dow == 0:
            dow = 7
        m = dt.month

        # Variables escaladas (valores aproximados del dataset)
        row = np.zeros(len(FEATURE_COLS))

        # Global_active_power_scaled
        gap_scaled = apply_minmax_scaling(base_consumption[i], 'Global_active_power_scaled', scaler_dict)
        row[TARGET_IDX] = gap_scaled

        # Lags de GAP (simulados con valores cercanos)
        for lag_col in SCALED_COLS:
            if 'lag' in lag_col and 'Global_active_power' in lag_col:
                lag_hours = int(''.join(filter(str.isdigit, lag_col.split('lag')[1])))
                idx = max(0, i - lag_hours)
                lag_val = apply_minmax_scaling(base_consumption[idx], lag_col, scaler_dict)
                row[FEATURE_COLS.index(lag_col)] = lag_val

        # Sub_metering_3 (calefacción) — mayor en invierno
        sm3 = 5.0 + 3.0 * np.sin(2 * np.pi * (h - 6) / 24) ** 2
        row[FEATURE_COLS.index('Sub_metering_3_scaled')] = apply_minmax_scaling(sm3, 'Sub_metering_3_scaled', scaler_dict)

        # Sub_metering_1 y 2 (valores bajos)
        sm1 = 0.5 if 7 <= h <= 9 or 18 <= h <= 21 else 0.0
        row[FEATURE_COLS.index('Sub_metering_1_scaled')] = apply_minmax_scaling(sm1, 'Sub_metering_1_scaled', scaler_dict)

        sm2 = 1.0 + 0.5 * np.random.rand()
        row[FEATURE_COLS.index('Sub_metering_2_scaled')] = apply_minmax_scaling(sm2, 'Sub_metering_2_scaled', scaler_dict)

        # Voltage (casi constante ~240V)
        voltage = 240 + np.random.normal(0, 2)
        row[FEATURE_COLS.index('Voltage_scaled')] = apply_minmax_scaling(voltage, 'Voltage_scaled', scaler_dict)

        # Global_reactive_power
        grp = 0.1 + 0.05 * base_consumption[i]
        row[FEATURE_COLS.index('Global_reactive_power_scaled')] = apply_minmax_scaling(grp, 'Global_reactive_power_scaled', scaler_dict)

        # Global_intensity
        gi = base_consumption[i] * 4.2  # aproximadamente
        row[FEATURE_COLS.index('Global_intensity_scaled')] = apply_minmax_scaling(gi, 'Global_intensity_scaled', scaler_dict)

        # Unmetered_energy
        unm = base_consumption[i] * 1000 - sm1 - sm2 - sm3
        row[FEATURE_COLS.index('Unmetered_energy_scaled')] = apply_minmax_scaling(max(0, unm), 'Unmetered_energy_scaled', scaler_dict)

        # Medias móviles y diferencias (aproximadas)
        row[FEATURE_COLS.index('GAP_ma24h_scaled')] = gap_scaled
        row[FEATURE_COLS.index('GAP_ma168h_scaled')] = gap_scaled
        row[FEATURE_COLS.index('GAP_std24h_scaled')] = apply_minmax_scaling(0.3, 'GAP_std24h_scaled', scaler_dict)
        row[FEATURE_COLS.index('SM3_ma24h_scaled')] = row[FEATURE_COLS.index('Sub_metering_3_scaled')]
        row[FEATURE_COLS.index('GAP_diff1h_scaled')] = 0.0
        row[FEATURE_COLS.index('GAP_diff24h_scaled')] = 0.0
        row[FEATURE_COLS.index('GAP_diff168h_scaled')] = 0.0

        # Variables cíclicas
        row[HOUR_SIN_IDX] = np.sin(2 * np.pi * h / 24)
        row[HOUR_COS_IDX] = np.cos(2 * np.pi * h / 24)
        row[DOW_SIN_IDX] = np.sin(2 * np.pi * (dow - 1) / 7)
        row[DOW_COS_IDX] = np.cos(2 * np.pi * (dow - 1) / 7)
        row[MONTH_SIN_IDX] = np.sin(2 * np.pi * (m - 1) / 12)
        row[MONTH_COS_IDX] = np.cos(2 * np.pi * (m - 1) / 12)

        # IsWeekend
        row[FEATURE_COLS.index('IsWeekend')] = 1.0 if dow in [1, 7] else 0.0

        window[i] = row

    return window, start_dt + pd.Timedelta(hours=LOOKBACK)


# ============================================================
# CARGA DE RECURSOS
# ============================================================
try:
    model = load_model()
    scaler_dict = load_scaler()
    resumen = load_resumen()
    RESOURCES_LOADED = True
except Exception as e:
    st.error(f"❌ Error cargando recursos: {e}")
    RESOURCES_LOADED = False

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚡ Panel de Control")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "📊 Comparar Modelos", "🔮 Predicciones", "📈 Métricas Detalladas"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset:** UCI Individual Household Electric Power Consumption  
**Frecuencia:** Horaria (agregada desde minutal)  
**Período:** Dic 2006 – Nov 2010  
**Variable objetivo:** Global Active Power (kW)
""")

# ============================================================
# PÁGINA: INICIO
# ============================================================
if page == "🏠 Inicio":
    st.title("⚡ Predicción de Consumo Eléctrico Residencial")
    st.markdown("""
    Bienvenido al dashboard interactivo de predicción de series temporales multivariadas.
    
    Este proyecto utiliza **redes neuronales recurrentes** (RNN, LSTM, GRU) para predecir
    el consumo eléctrico de un hogar con base en mediciones históricas de potencia activa,
    voltaje, submediciones y características temporales.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros procesados", "2,075,259")
    with col2:
        st.metric("Frecuencia original", "1 minuto")
    with col3:
        st.metric("Agregación final", "Horaria")

    st.markdown("---")
    st.subheader("🏆 Modelo Seleccionado: GRU")

    if RESOURCES_LOADED:
        ganador = resumen.get('modelo_ganador', 'GRU')
        metricas = resumen.get('metricas_ganador', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metricas.get('mae', 0.339):.4f} kW")
        with col2:
            st.metric("RMSE", f"{metricas.get('rmse', 0.481):.4f} kW")
        with col3:
            st.metric("R²", f"{metricas.get('r2', 0.599):.4f}")

        st.markdown(f"""
        **Justificación:** {resumen.get('recomendacion', 'GRU ofrece el mejor balance entre precisión y eficiencia paramétrica.')}
        """)

    st.markdown("---")
    st.subheader("📋 Pipeline del Proyecto")
    st.markdown("""
    1. **Exploración** → PySpark para lectura y análisis de 2M+ registros
    2. **Preprocesamiento** → Imputación, agregación horaria, lags, codificación cíclica
    3. **Modelos** → RNN (baseline) → LSTM → GRU (ganador)
    4. **Comparación** → MAE, RMSE, R² en test set (2010)
    5. **Despliegue** → Dashboard interactivo con predicciones futuras
    """)

# ============================================================
# PÁGINA: COMPARAR MODELOS
# ============================================================
elif page == "📊 Comparar Modelos":
    st.title("📊 Comparación de Modelos Recurrentes")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Recursos no cargados. Verifica los archivos en modelo_final/")
        st.stop()

    # Tabla comparativa
    st.subheader("Tabla de Métricas — Test Set (2010)")
    comparativa = resumen.get('comparativa_completa', [
        {'modelo': 'RNN', 'mae': 0.351, 'rmse': 0.502, 'r2': 0.564, 'epochs_trained': 26, 'best_val_loss': 0.045},
        {'modelo': 'LSTM', 'mae': 0.336, 'rmse': 0.478, 'r2': 0.598, 'epochs_trained': 25, 'best_val_loss': 0.042},
        {'modelo': 'GRU', 'mae': 0.339, 'rmse': 0.481, 'r2': 0.599, 'epochs_trained': 23, 'best_val_loss': 0.041},
    ])

    df_comp = pd.DataFrame(comparativa)
    df_comp = df_comp[['modelo', 'mae', 'rmse', 'r2', 'epochs_trained', 'best_val_loss']]

    # Destacar ganador
    def highlight_winner(row):
        if row['modelo'] == resumen.get('modelo_ganador', 'GRU'):
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)

    st.dataframe(df_comp.style.apply(highlight_winner, axis=1), use_container_width=True)

    # Gráficos comparativos
    st.subheader("Visualización Comparativa")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        colors = {'RNN': '#FF6B6B', 'LSTM': '#4ECDC4', 'GRU': '#45B7D1'}
        for _, row in df_comp.iterrows():
            fig.add_trace(go.Bar(
                name=row['modelo'],
                x=['MAE', 'RMSE'],
                y=[row['mae'], row['rmse']],
                marker_color=colors.get(row['modelo'], '#888'),
                text=[f"{row['mae']:.3f}", f"{row['rmse']:.3f}"],
                textposition='outside'
            ))
        fig.update_layout(
            title="MAE y RMSE por Modelo (menor es mejor)",
            barmode='group',
            yaxis_title="kW",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for _, row in df_comp.iterrows():
            fig.add_trace(go.Bar(
                name=row['modelo'],
                x=['R²'],
                y=[row['r2']],
                marker_color=colors.get(row['modelo'], '#888'),
                text=[f"{row['r2']:.3f}"],
                textposition='outside'
            ))
        fig.update_layout(
            title="R² por Modelo (mayor es mejor)",
            yaxis=dict(range=[0.5, 0.65]),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Análisis de eficiencia
    st.subheader("Análisis de Eficiencia")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RNN épocas", f"{df_comp[df_comp['modelo']=='RNN']['epochs_trained'].values[0]}")
    with col2:
        st.metric("LSTM épocas", f"{df_comp[df_comp['modelo']=='LSTM']['epochs_trained'].values[0]}")
    with col3:
        st.metric("GRU épocas", f"{df_comp[df_comp['modelo']=='GRU']['epochs_trained'].values[0]}", delta="-3 vs RNN")

    st.markdown("""
    **Interpretación:**
    - **RNN** (baseline): Limitado por *vanishing gradients*, no captura bien dependencias de largo plazo.
    - **LSTM**: Mejor MAE individual pero requiere más parámetros (3 compuertas).
    - **GRU** 🏆: Mejor R² con ~25-30% menos parámetros que LSTM. Converge más rápido.
    """)

# ============================================================
# PÁGINA: PREDICCIONES
# ============================================================
elif page == "🔮 Predicciones":
    st.title("🔮 Predicciones Futuras de Consumo")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Modelo no cargado. Verifica modelo_gru_final.keras")
        st.stop()

    st.markdown("""
    Genera predicciones de consumo eléctrico para las próximas horas usando el modelo GRU.
    
    Puedes usar una **ventana sintética** (simulación) o cargar tus propios datos.
    """)

    # Opciones de entrada
    input_method = st.radio(
        "Método de entrada",
        ["🎲 Generar ventana sintética", "📁 Cargar datos propios (CSV)"],
        horizontal=True
    )

    last_window = None
    start_time = None

    if input_method == "🎲 Generar ventana sintética":
        st.info("Se genera una ventana de 24h con patrón realista de consumo (invierno, fin de semana).")
        if st.button("Generar ventana"):
            with st.spinner("Generando datos sintéticos..."):
                last_window, start_time = generate_synthetic_last_window(scaler_dict)
            st.success(f"✅ Ventana generada. Inicio de predicción: {start_time}")
            st.session_state['last_window'] = last_window
            st.session_state['start_time'] = start_time

    else:
        st.info("""
        Sube un CSV con **exactamente 24 filas** (1 por hora) y las siguientes columnas:
        `Datetime, Global_active_power, Global_reactive_power, Voltage, Global_intensity,
        Sub_metering_1, Sub_metering_2, Sub_metering_3` (y derivadas si las tienes).
        
        ⚠️ El preprocesamiento completo (lags, cíclicas, escalado) debe estar aplicado.
        """)
        uploaded_file = st.file_uploader("Subir CSV", type=['csv'])
        if uploaded_file is not None:
            st.warning("⚠️ Carga de CSV propio requiere preprocesamiento completo. Funcionalidad en desarrollo.")
            # Aquí iría la lógica de preprocesamiento si el usuario sube datos crudos

    # Generar predicciones si hay ventana disponible
    if 'last_window' in st.session_state:
        last_window = st.session_state['last_window']
        start_time = st.session_state['start_time']

        st.markdown("---")
        st.subheader("⚙️ Configuración de Predicción")

        horizon = st.slider("Horizonte de predicción (horas)", 1, 168, 24, 1)
        st.caption("Máximo recomendado: 24h (confiable). A 168h el error se acumula.")

        if st.button("🚀 Ejecutar Predicción", type="primary"):
            with st.spinner(f"Prediciendo {horizon} horas..."):
                predictions = predict_future(
                    model, last_window, horizon, scaler_dict, start_time
                )

            st.success("✅ Predicción completada")

            # Mostrar resultados
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media", f"{predictions.mean():.3f} kW")
            with col2:
                st.metric("Mínimo", f"{predictions.min():.3f} kW")
            with col3:
                st.metric("Máximo", f"{predictions.max():.3f} kW")
            with col4:
                st.metric("Desv. Est.", f"{predictions.std():.3f} kW")

            # Gráfico interactivo
            fig = go.Figure()
            future_hours = pd.date_range(start=start_time, periods=horizon, freq='H')

            fig.add_trace(go.Scatter(
                x=future_hours,
                y=predictions,
                mode='lines+markers',
                name='Predicción GRU',
                line=dict(color='#45B7D1', width=2),
                marker=dict(size=6)
            ))

            fig.add_hline(
                y=predictions.mean(),
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Media: {predictions.mean():.2f} kW"
            )

            fig.update_layout(
                title=f"Predicción de Consumo — Próximas {horizon} horas",
                xaxis_title="Fecha/Hora",
                yaxis_title="Global Active Power (kW)",
                hovermode='x unified',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tabla de predicciones
            st.subheader("📋 Tabla de Predicciones")
            df_pred = pd.DataFrame({
                'Fecha/Hora': future_hours,
                'Predicción (kW)': predictions,
                'Día de semana': [d.strftime('%A') for d in future_hours]
            })
            st.dataframe(df_pred, use_container_width=True)

            # Descargar predicciones
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar predicciones (CSV)",
                csv,
                f"prediccion_gru_{horizon}h.csv",
                "text/csv"
            )

# ============================================================
# PÁGINA: MÉTRICAS DETALLADAS
# ============================================================
elif page == "📈 Métricas Detalladas":
    st.title("📈 Métricas y Rendimiento del Modelo")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Recursos no cargados")
        st.stop()

    st.subheader("Métricas del Modelo Ganador (GRU)")

    metricas = resumen.get('metricas_ganador', {})
    analisis = resumen.get('analisis', {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "MAE",
            f"{metricas.get('mae', 0.339):.4f} kW",
            help="Error absoluto medio. En promedio, el modelo se equivoca en ±0.34 kW."
        )
    with col2:
        st.metric(
            "RMSE",
            f"{metricas.get('rmse', 0.481):.4f} kW",
            help="Raíz del error cuadrático medio. Penaliza errores grandes."
        )
    with col3:
        st.metric(
            "R²",
            f"{metricas.get('r2', 0.599):.4f}",
            help="Coeficiente de determinación. El modelo explica ~60% de la variabilidad del consumo."
        )

    st.markdown("---")
    st.subheader("Contexto del Error")

    consumo_medio = 1.092  # kW, del EDA
    mae = metricas.get('mae', 0.339)
    error_relativo = (mae / consumo_medio) * 100

    st.markdown(f"""
    | Métrica | Valor | Interpretación |
    |---|---|---|
    | Consumo medio del hogar | {consumo_medio:.3f} kW | Basado en datos históricos (2007-2010) |
    | Error absoluto medio (MAE) | {mae:.4f} kW | Equivale a ~{error_relativo:.1f}% del consumo medio |
    | Error en términos prácticos | ±{mae:.2f} kW | Similar a encender/apagar un microondas |

    **Aplicabilidad:** Este nivel de error es aceptable para:
    - Planificación energética residencial
    - Detección de anomalías (picos >2× MAE)
    - Optimización de horarios de consumo
    """)

    st.markdown("---")
    st.subheader("Comparativa de Convergencia")

    # Placeholder para gráficos de convergencia (se cargarían desde history si estuvieran disponibles)
    st.info("""
    Los gráficos de convergencia (train/validation loss por época) están disponibles
    en los notebooks originales (03, 04, 05). En el dashboard se muestran las métricas finales.
    
    **Resumen de convergencia:**
    - RNN: 26 épocas hasta early stopping
    - LSTM: 25 épocas hasta early stopping  
    - GRU: 23 épocas hasta early stopping (más rápido)
    """)

    st.markdown("---")
    st.subheader("Análisis de la Selección")

    st.markdown(f"""
    **{resumen.get('modelo_ganador', 'GRU')} fue seleccionado porque:**

    1. **Mayor R²** ({metricas.get('r2', 0.599):.3f}): Mejor capacidad explicativa del consumo
    2. **Eficiencia paramétrica**: ~25-30% menos parámetros que LSTM
    3. **Convergencia rápida**: {df_comp[df_comp['modelo']=='GRU']['epochs_trained'].values[0] if 'df_comp' in locals() else 23} épocas vs 26 de RNN
    4. **Balance precisión/velocidad**: Métricas equivalentes a LSTM con menor costo computacional

    **Mejora sobre baseline (RNN):** {analisis.get('mejora_sobre_rnn_pct', 6.1)}% en R²
    """)

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("""
📅 Proyecto Final — Ciencia de Datos I  
🎓 Predicción de Series de Tiempo Multivariadas  
🛠️ PySpark + TensorFlow + Streamlit
""")