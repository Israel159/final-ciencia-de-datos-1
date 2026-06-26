# -*- coding: utf-8 -*-
"""
Dashboard de Predicción de Consumo Eléctrico — Streamlit
======================================================

Aplicación interactiva para visualizar datos, comparar modelos (RNN/LSTM/GRU),
ejecutar predicciones futuras y mostrar métricas del modelo ganador (GRU).

SOLUCIÓN DE COMPATIBILIDAD: El modelo se reconstruye manualmente desde JSON
y los pesos se cargan desde .weights.h5, evitando problemas de deserialización
entre versiones de Keras/TensorFlow.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

import plotly.graph_objects as go

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
# RUTAS DE ARCHIVOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHITECTURE_PATH = os.path.join(BASE_DIR, "modelo_gru_architecture.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "modelo_gru_weights.weights.h5")
SCALER_PATH = os.path.join(BASE_DIR, "min_max_scaler.json")
RESUMEN_PATH = os.path.join(BASE_DIR, "resumen_comparativa.json")

# ============================================================
# FEATURE COLUMNS (cargadas desde JSON para garantizar coincidencia exacta)
# ============================================================
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_columns.json")

if os.path.exists(FEATURE_COLS_PATH):
    with open(FEATURE_COLS_PATH, 'r') as f:
        FEATURE_COLS = json.load(f)
else:
    raise FileNotFoundError(
        f"❌ No se encontró {FEATURE_COLS_PATH}. "
        f"Genera este archivo en Colab ejecutando:\n"
        f"  json.dump(feature_cols, open('feature_columns.json','w'))"
    )

LOOKBACK = 24

# Índices de variables clave para actualizar en forecasting
TARGET_IDX = FEATURE_COLS.index('Global_active_power_scaled')
HOUR_SIN_IDX = FEATURE_COLS.index('Hour_sin')
HOUR_COS_IDX = FEATURE_COLS.index('Hour_cos')
DOW_SIN_IDX = FEATURE_COLS.index('DayOfWeek_sin')
DOW_COS_IDX = FEATURE_COLS.index('DayOfWeek_cos')
MONTH_SIN_IDX = FEATURE_COLS.index('Month_sin')
MONTH_COS_IDX = FEATURE_COLS.index('Month_cos')
LAG1_IDX = FEATURE_COLS.index('Global_active_power_lag1h_scaled') if 'Global_active_power_lag1h_scaled' in FEATURE_COLS else None

# Para compatibilidad con generate_synthetic_last_window (necesita SCALED_COLS separados)
SCALED_COLS = [c for c in FEATURE_COLS if c.endswith('_scaled')]
CYCLIC_COLS = [c for c in FEATURE_COLS if c in ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']]

# ============================================================
# FUNCIÓN: RECONSTRUIR MODELO DESDE JSON + CARGAR PESOS
# ============================================================
@st.cache_resource
def build_and_load_model():
    """
    Reconstruye la arquitectura GRU manualmente y carga los pesos desde .weights.h5.
    Esto evita problemas de deserialización entre versiones de Keras/TensorFlow.
    """
    # Verificar que los archivos existen
    if not os.path.exists(ARCHITECTURE_PATH):
        raise FileNotFoundError(f"No se encontró: {ARCHITECTURE_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"No se encontró: {WEIGHTS_PATH}")

    # Cargar arquitectura
    with open(ARCHITECTURE_PATH, 'r') as f:
        arch = json.load(f)

    input_shape = tuple(arch['input_shape'])

    # Reconstruir modelo secuencial
    model = Sequential()

    for i, layer in enumerate(arch['layers']):
        if layer['type'] == 'GRU':
            kwargs = {
                'units': layer['units'],
                'activation': layer.get('activation', 'tanh'),
                'return_sequences': layer.get('return_sequences', False),
                'name': layer.get('name', f'gru_{i+1}')
            }
            if i == 0:
                kwargs['input_shape'] = input_shape
            model.add(GRU(**kwargs))

        elif layer['type'] == 'Dropout':
            model.add(Dropout(rate=layer['rate'], name=layer.get('name', f'dropout_{i+1}')))

        elif layer['type'] == 'Dense':
            model.add(Dense(
                units=layer['units'],
                activation=layer.get('activation', 'linear'),
                name=layer.get('name', f'dense_{i+1}')
            ))

    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Cargar pesos
    model.load_weights(WEIGHTS_PATH)

    return model

# ============================================================
# CARGA DE RECURSOS
# ============================================================
@st.cache_data
def load_scaler():
    with open(SCALER_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_resumen():
    with open(RESUMEN_PATH, "r") as f:
        return json.load(f)

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
    
    INCLUYE VALIDACIONES DEFENSIVAS para diagnóstico de errores.
    """
    predictions_scaled = []
    current_seq = last_sequence.copy().astype(np.float32)

    gap_min = scaler_dict['Global_active_power']['min']
    gap_max = scaler_dict['Global_active_power']['max']

    # === VALIDACIONES DEFENSIVAS ===
    if np.isnan(current_seq).any():
        raise ValueError(f"❌ current_seq contiene NaN. Posiciones: {np.argwhere(np.isnan(current_seq))}")
    if np.isinf(current_seq).any():
        raise ValueError(f"❌ current_seq contiene infinitos. Posiciones: {np.argwhere(np.isinf(current_seq))}")
    
    expected_shape = (LOOKBACK, len(FEATURE_COLS))
    if current_seq.shape != expected_shape:
        raise ValueError(
            f"❌ Shape incorrecto: {current_seq.shape}, esperado: {expected_shape}. "
            f"LOOKBACK={LOOKBACK}, len(FEATURE_COLS)={len(FEATURE_COLS)}"
        )

    # Log para debugging (solo en desarrollo)
    st.write(f"🔍 DEBUG - Model input shape: {model.input_shape}")
    st.write(f"🔍 DEBUG - current_seq shape: {current_seq.shape}")
    st.write(f"🔍 DEBUG - FEATURE_COLS count: {len(FEATURE_COLS)}")
    st.write(f"🔍 DEBUG - current_seq dtype: {current_seq.dtype}")
    st.write(f"🔍 DEBUG - current_seq min/max: {current_seq.min():.4f} / {current_seq.max():.4f}")

    current_dt = pd.to_datetime(start_dt)

    for step in range(n_steps):
        # Reshape explícito
        input_data = current_seq.reshape(1, LOOKBACK, len(FEATURE_COLS))
        
        # Validación del input antes de predict
        if np.isnan(input_data).any():
            raise ValueError(f"❌ NaN en input_data en paso {step}")
        
        try:
            pred = model.predict(input_data, verbose=0)
        except Exception as e:
            st.error(f"❌ Error en model.predict() paso {step}: {str(e)}")
            st.error(f"   Input shape: {input_data.shape}")
            st.error(f"   Input dtype: {input_data.dtype}")
            st.error(f"   Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
            raise

        pred_val = float(pred[0, 0])
        predictions_scaled.append(pred_val)

        # Avanzar 1 hora en el calendario
        current_dt += pd.Timedelta(hours=1)
        h = current_dt.hour
        dow = (current_dt.weekday() + 2) % 7
        if dow == 0:
            dow = 7
        m = current_dt.month

        # Crear nuevo timestep actualizando cíclicas y objetivo
        new_step = current_seq[-1].copy()
        new_step[TARGET_IDX] = pred_val

        # Actualizar variables cíclicas con la hora/día/mes FUTURO
        new_step[HOUR_SIN_IDX] = np.sin(2 * np.pi * h / 24)
        new_step[HOUR_COS_IDX] = np.cos(2 * np.pi * h / 24)
        new_step[DOW_SIN_IDX] = np.sin(2 * np.pi * (dow - 1) / 7)
        new_step[DOW_COS_IDX] = np.cos(2 * np.pi * (dow - 1) / 7)
        new_step[MONTH_SIN_IDX] = np.sin(2 * np.pi * (m - 1) / 12)
        new_step[MONTH_COS_IDX] = np.cos(2 * np.pi * (m - 1) / 12)

        # Actualizar lag1h con la predicción reciente
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
    
    UNA "VENTANA" son los últimos 24 datos históricos que el modelo necesita
    para predecir la siguiente hora. Es como la "memoria" del modelo.
    """
    np.random.seed(42)
    window = np.zeros((LOOKBACK, len(FEATURE_COLS)), dtype=np.float32)

    # Simular consumo con patrón diario realista:
    # - Madrugada (0-5h): bajo (~0.5 kW)
    # - Mañana (6-9h): pico al despertar (~1.7 kW)  
    # - Tarde (10-16h): medio (~1.0 kW)
    # - Noche (17-21h): pico al regresar (~1.8 kW)
    # - Noche tarde (22-23h): bajando (~0.8 kW)
    hours = np.arange(LOOKBACK)
    base_consumption = 0.5 + 0.8 * np.sin(2 * np.pi * (hours - 6) / 24) ** 2
    base_consumption += np.random.normal(0, 0.1, LOOKBACK)
    base_consumption = np.clip(base_consumption, 0.1, 3.0)

    # Fecha de inicio: último día disponible del dataset (finales de noviembre 2010)
    start_dt = pd.to_datetime("2010-11-26 00:00:00")

    for i in range(LOOKBACK):
        dt = start_dt + pd.Timedelta(hours=i)
        h = dt.hour
        dow = (dt.weekday() + 2) % 7
        if dow == 0:
            dow = 7
        m = dt.month

        # Inicializar fila de features
        row = np.zeros(len(FEATURE_COLS))

        # Global_active_power (variable objetivo)
        gap_scaled = apply_minmax_scaling(base_consumption[i], 'Global_active_power_scaled', scaler_dict)
        row[TARGET_IDX] = gap_scaled

        # Lags de GAP (valores históricos simulados)
        for lag_col in SCALED_COLS:
            if 'lag' in lag_col and 'Global_active_power' in lag_col:
                lag_hours = int(''.join(filter(str.isdigit, lag_col.split('lag')[1])))
                idx = max(0, i - lag_hours)
                lag_val = apply_minmax_scaling(base_consumption[idx], lag_col, scaler_dict)
                row[FEATURE_COLS.index(lag_col)] = lag_val

        # Sub_metering_3 (calefacción/AC) — mayor en invierno, patrón diario
        sm3 = 5.0 + 3.0 * np.sin(2 * np.pi * (h - 6) / 24) ** 2
        row[FEATURE_COLS.index('Sub_metering_3_scaled')] = apply_minmax_scaling(sm3, 'Sub_metering_3_scaled', scaler_dict)

        # Sub_metering_1 (cocina) — uso esporádico en horas de comida
        sm1 = 0.5 if 7 <= h <= 9 or 18 <= h <= 21 else 0.0
        row[FEATURE_COLS.index('Sub_metering_1_scaled')] = apply_minmax_scaling(sm1, 'Sub_metering_1_scaled', scaler_dict)

        # Sub_metering_2 (lavandería) — base continua del refrigerador
        sm2 = 1.0 + 0.5 * np.random.rand()
        row[FEATURE_COLS.index('Sub_metering_2_scaled')] = apply_minmax_scaling(sm2, 'Sub_metering_2_scaled', scaler_dict)

        # Voltage (casi constante ~240V)
        voltage = 240 + np.random.normal(0, 2)
        row[FEATURE_COLS.index('Voltage_scaled')] = apply_minmax_scaling(voltage, 'Voltage_scaled', scaler_dict)

        # Global_reactive_power
        grp = 0.1 + 0.05 * base_consumption[i]
        row[FEATURE_COLS.index('Global_reactive_power_scaled')] = apply_minmax_scaling(grp, 'Global_reactive_power_scaled', scaler_dict)

        # Global_intensity (proporcional a potencia activa)
        gi = base_consumption[i] * 4.2
        row[FEATURE_COLS.index('Global_intensity_scaled')] = apply_minmax_scaling(gi, 'Global_intensity_scaled', scaler_dict)

        # Energía no medida
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

        # Variables cíclicas (codificación seno/coseno)
        row[HOUR_SIN_IDX] = np.sin(2 * np.pi * h / 24)
        row[HOUR_COS_IDX] = np.cos(2 * np.pi * h / 24)
        row[DOW_SIN_IDX] = np.sin(2 * np.pi * (dow - 1) / 7)
        row[DOW_COS_IDX] = np.cos(2 * np.pi * (dow - 1) / 7)
        row[MONTH_SIN_IDX] = np.sin(2 * np.pi * (m - 1) / 12)
        row[MONTH_COS_IDX] = np.cos(2 * np.pi * (m - 1) / 12)

        # IsWeekend (1 = fin de semana, 0 = día de semana)
        row[FEATURE_COLS.index('IsWeekend')] = 1.0 if dow in [1, 7] else 0.0

        window[i] = row

    # === VALIDACIÓN FINAL ===
    if np.isnan(window).any():
        nan_positions = np.argwhere(np.isnan(window))
        raise ValueError(f"❌ Ventana sintética contiene NaN en posiciones: {nan_positions}")
    
    if window.shape != (LOOKBACK, len(FEATURE_COLS)):
        raise ValueError(
            f"❌ Shape final incorrecto: {window.shape}, esperado: {(LOOKBACK, len(FEATURE_COLS))}"
        )

    # Fecha de inicio de la predicción (después de las 24h de ventana)
    prediction_start = start_dt + pd.Timedelta(hours=LOOKBACK)
    
    return window, prediction_start

# ============================================================
# CARGA DE RECURSOS CON MANEJO DE ERRORES
# ============================================================
try:
    model = build_and_load_model()
    scaler_dict = load_scaler()
    resumen = load_resumen()
    RESOURCES_LOADED = True
    st.sidebar.success("✅ Modelo GRU cargado correctamente")
except Exception as e:
    RESOURCES_LOADED = False
    st.sidebar.error(f"❌ Error cargando recursos: {str(e)}")
    st.sidebar.info("Verifica que los archivos estén en la carpeta de la app.")

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

    st.markdown("### 📊 Estadísticas del Dataset")
    st.write("- **Registros procesados:** 2,075,259")
    st.write("- **Frecuencia original:** 1 minuto")
    st.write("- **Agregación final:** Horaria")

    st.markdown("---")
    st.subheader("🏆 Modelo Seleccionado: GRU")

    if RESOURCES_LOADED:
        ganador = resumen.get('modelo_ganador', 'GRU')
        metricas = resumen.get('metricas_ganador', {})

        # Tabla simple (evita st.metric que causa removeChild)
        data = {
            'Métrica': ['MAE (kW)', 'RMSE (kW)', 'R²'],
            'Valor': [
                f"{metricas.get('mae', 0.339):.4f}",
                f"{metricas.get('rmse', 0.481):.4f}",
                f"{metricas.get('r2', 0.599):.4f}"
            ]
        }
        st.table(pd.DataFrame(data))

        st.markdown(f"**Justificación:** {resumen.get('recomendacion', 'GRU ofrece el mejor balance entre precisión y eficiencia paramétrica.')}")

    else:
        st.warning("⚠️ Modelo no cargado. Verifica los archivos.")

    st.markdown("---")
    st.subheader("📋 Pipeline del Proyecto")
    st.markdown("""
    1. **Exploración** → PySpark para lectura y análisis de 2M+ registros
    2. **Preprocesamiento** → Imputación, agregación horaria, lags, codificación cíclica
    3. **Modelos** → RNN (baseline) → LSTM → GRU (ganador)
    4. **Comparación** → MAE, RMSE, R² en test set (2010)
    5. **Despliegue** → Dashboard interactivo con predicciones futuras
    """)

    st.markdown("---")
    st.info("""
    💡 **¿Qué es una "ventana"?**  
    El modelo necesita ver las últimas **24 horas** de datos para predecir la siguiente.
    Como el dataset termina en 2010, la app puede generar una **ventana sintética** 
    (datos simulados realistas) para demostrar las predicciones.
    """)

# ============================================================
# PÁGINA: COMPARAR MODELOS
# ============================================================
elif page == "📊 Comparar Modelos":
    st.title("📊 Comparación de Modelos Recurrentes")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Recursos no cargados. Verifica los archivos.")
        st.stop()

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

    st.markdown("**Mejor modelo por métrica:**")
    st.write(f"- MAE (menor): **{df_comp.loc[df_comp['mae'].idxmin(), 'modelo']}**")
    st.write(f"- RMSE (menor): **{df_comp.loc[df_comp['rmse'].idxmin(), 'modelo']}**")
    st.write(f"- R² (mayor): **{df_comp.loc[df_comp['r2'].idxmax(), 'modelo']}**")

    # Gráficos con Plotly
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
            title="MAE y RMSE (menor es mejor)",
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
            title="R² (mayor es mejor)",
            yaxis=dict(range=[0.5, 0.65]),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interpretación:**")
    st.write("- **RNN** (baseline): Limitado por vanishing gradients")
    st.write("- **LSTM**: Mejor MAE pero más parámetros")
    st.write("- **GRU** 🏆: Mejor R² con ~25-30% menos parámetros, converge más rápido")

# ============================================================
# PÁGINA: PREDICCIONES
# ============================================================
elif page == "🔮 Predicciones":
    st.title("🔮 Predicciones Futuras de Consumo")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Modelo no cargado. Verifica los archivos.")
        st.stop()

    st.markdown("""
    Genera predicciones de consumo eléctrico con el modelo GRU.
    
    💡 **¿Cómo funciona?** El modelo necesita una **ventana de 24 horas** (datos históricos)
    para predecir la siguiente. Como el dataset termina en 2010, puedes generar una
    **ventana sintética** con patrones realistas de consumo.
    """)

    input_method = st.radio(
        "Método de entrada",
        ["🎲 Generar ventana sintética", "📁 Cargar datos propios (CSV)"],
        horizontal=True
    )

    last_window = None
    start_time = None

    if input_method == "🎲 Generar ventana sintética":
        st.info("""
        **¿Qué es una ventana sintética?**  
        Son 24 horas de datos simulados que imitan el comportamiento real de un hogar:
        - 🔴 Madrugada: consumo bajo (dormidos)
        - 🟡 Mañana: pico al despertar (café, ducha, desayuno)
        - 🟢 Tarde: consumo medio (trabajo/estudio fuera)
        - 🔵 Noche: pico al regresar (cena, TV, calefacción)
        """)
        
        if st.button("Generar ventana", type="primary"):
            with st.spinner("Generando 24 horas de datos simulados..."):
                try:
                    last_window, start_time = generate_synthetic_last_window(scaler_dict)
                    st.success(f"✅ Ventana generada. Inicio de predicción: {start_time}")
                    st.session_state['last_window'] = last_window
                    st.session_state['start_time'] = start_time
                    
                    # Mostrar preview de la ventana generada
                    st.markdown("### 👁️ Preview de la ventana generada")
                    preview_df = pd.DataFrame({
                        'Hora': range(24),
                        'Consumo_simulado_kW': [
                            inverse_minmax_scaling(
                                last_window[i, TARGET_IDX], 
                                'Global_active_power_scaled', 
                                scaler_dict
                            ) for i in range(24)
                        ]
                    })
                    st.line_chart(preview_df.set_index('Hora'))
                    
                except Exception as e:
                    st.error(f"❌ Error generando ventana: {str(e)}")
                    st.stop()

    else:
        st.info("""
        Sube un CSV con **exactamente 24 filas** (1 por hora) con las columnas preprocesadas.
        
        ⚠️ El preprocesamiento completo (lags, cíclicas, escalado) debe estar aplicado.
        """)
        uploaded_file = st.file_uploader("Subir CSV", type=['csv'])
        if uploaded_file is not None:
            st.warning("⚠️ Funcionalidad en desarrollo. Usa 'Generar ventana sintética'.")

    # Generar predicciones si hay ventana disponible
    if 'last_window' in st.session_state:
        last_window = st.session_state['last_window']
        start_time = st.session_state['start_time']

        st.markdown("---")
        st.subheader("⚙️ Configuración de Predicción")

        horizon = st.slider("Horizonte de predicción (horas)", 1, 168, 24, 1)
        st.caption("Máximo recomendado: 24h (confiable). A 168h el error se acumula.")

        if st.button("🚀 Ejecutar Predicción", type="primary"):
            with st.spinner(f"Prediciendo {horizon} horas con el modelo GRU..."):
                try:
                    predictions = predict_future(
                        model, last_window, horizon, scaler_dict, start_time
                    )
                    
                    st.success("✅ Predicción completada")

                    # Estadísticas
                    st.markdown("### 📊 Estadísticas de la Predicción")
                    stats = {
                        'Métrica': ['Media (kW)', 'Mínimo (kW)', 'Máximo (kW)', 'Desv. Est. (kW)'],
                        'Valor': [
                            f"{predictions.mean():.3f}",
                            f"{predictions.min():.3f}",
                            f"{predictions.max():.3f}",
                            f"{predictions.std():.3f}"
                        ]
                    }
                    st.table(pd.DataFrame(stats))

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

                    # Tabla descargable
                    st.subheader("📋 Tabla de Predicciones")
                    df_pred = pd.DataFrame({
                        'Fecha/Hora': future_hours,
                        'Predicción (kW)': predictions,
                        'Día de semana': [d.strftime('%A') for d in future_hours]
                    })
                    st.dataframe(df_pred, use_container_width=True)

                    csv = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Descargar predicciones (CSV)",
                        csv,
                        f"prediccion_gru_{horizon}h.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error en predicción: {str(e)}")
                    st.info("Revisa los logs de debug arriba para más detalles.")

# ============================================================
# PÁGINA: MÉTRICAS DETALLADAS
# ============================================================
elif page == "📈 Métricas Detalladas":
    st.title("📈 Métricas y Rendimiento del Modelo")

    if not RESOURCES_LOADED:
        st.warning("⚠️ Recursos no cargados.")
        st.stop()

    st.subheader("Métricas del Modelo Ganador (GRU)")
    metricas = resumen.get('metricas_ganador', {})

    # Tabla en lugar de st.metric()
    data = {
        'Métrica': ['MAE', 'RMSE', 'R²'],
        'Valor': [
            f"{metricas.get('mae', 0.339):.4f} kW",
            f"{metricas.get('rmse', 0.481):.4f} kW",
            f"{metricas.get('r2', 0.599):.4f}"
        ],
        'Descripción': [
            'Error absoluto medio',
            'Raíz error cuadrático medio',
            'Coeficiente de determinación'
        ]
    }
    st.table(pd.DataFrame(data))

    st.markdown("---")
    st.subheader("Contexto del Error")

    consumo_medio = 1.092
    mae = metricas.get('mae', 0.339)
    error_relativo = (mae / consumo_medio) * 100

    st.markdown(f"""
    | Métrica | Valor | Interpretación |
    |---|---|---|
    | Consumo medio del hogar | {consumo_medio:.3f} kW | Basado en datos históricos (2007-2010) |
    | MAE | {mae:.4f} kW | ~{error_relativo:.1f}% del consumo medio |
    | Error práctico | ±{mae:.2f} kW | Similar a encender/apagar un microondas |

    **Aplicabilidad:** Este nivel de error es aceptable para:
    - Planificación energética residencial
    - Detección de anomalías (picos >2× MAE)
    - Optimización de horarios de consumo
    """)

    st.markdown("---")
    st.subheader("Análisis de la Selección")

    st.markdown(f"""
    **GRU fue seleccionado porque:**

    1. **Mayor R²** ({metricas.get('r2', 0.599):.3f}): Mejor capacidad explicativa del consumo
    2. **Eficiencia paramétrica**: ~25-30% menos parámetros que LSTM
    3. **Convergencia rápida**: 23 épocas vs 26 de RNN
    4. **Balance precisión/velocidad**: Métricas equivalentes a LSTM con menor costo

    **Mejora sobre baseline (RNN):** ~6.1% en R²
    """)

st.sidebar.markdown("---")
st.sidebar.caption("📅 Proyecto Final — Ciencia de Datos I | PySpark + TensorFlow + Streamlit")