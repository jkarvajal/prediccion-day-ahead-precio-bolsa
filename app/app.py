"""
Aplicación web de predicción day-ahead del precio de bolsa de energía
del mercado eléctrico colombiano (SIN).

Presenta la persistencia diaria como modelo de referencia seleccionado en el
proyecto (predice cada hora del día siguiente con el precio de la misma hora
del día anterior) y un modelo de aprendizaje automático (Lasso, 24 regresiones
independientes por hora) como comparación. Permite consultar el perfil horario
del día siguiente a partir de un día base del histórico, con las métricas de
ambos modelos y el desempeño por hora.

Ejecutar:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Rutas del proyecto ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

MODEL_PATH = MODELS_DIR / "modelo_final_dayahead.joblib"
METADATA_PATH = MODELS_DIR / "modelo_final_metadata.json"
FEATURES_PATH = PROCESSED_DIR / "features_day_ahead.csv"
METRICS_BY_HOUR_PATH = OUTPUTS_DIR / "hourly_dayahead_metrics_by_hour.csv"
BASELINE_RESULTS_PATH = OUTPUTS_DIR / "baseline_global_results.csv"

PEAK_HOURS = [18, 19, 20, 21, 22]
TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]
ML_MODEL_LABEL = "Lasso_horario_con_exogenas"

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
    "Sunday": "Domingo",
}
MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre",
    11: "noviembre", 12: "diciembre",
}


def fecha_larga_es(fecha):
    """Formatea una fecha en español: 'Sábado 28 de marzo de 2026'."""
    dia = DIAS_ES.get(fecha.strftime("%A"), fecha.strftime("%A"))
    return f"{dia} {fecha.day} de {MESES_ES[fecha.month]} de {fecha.year}"


st.set_page_config(
    page_title="Predicción day-ahead: precio de bolsa Colombia",
    page_icon="⚡",
    layout="wide",
)


# --- Carga cacheada de artefactos ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_PATH)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"], errors="coerce")
    return df.sort_values("FECHA").reset_index(drop=True)


@st.cache_data
def load_metrics_by_hour():
    if not METRICS_BY_HOUR_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_BY_HOUR_PATH)
    return df[df["modelo"] == ML_MODEL_LABEL].reset_index(drop=True)


@st.cache_data
def load_persistencia_metrics():
    """Métricas de la persistencia diaria en el conjunto de prueba."""
    if BASELINE_RESULTS_PATH.exists():
        df = pd.read_csv(BASELINE_RESULTS_PATH)
        fila = df[df["modelo"] == "Baseline_hoy"]
        if not fila.empty:
            r = fila.iloc[0]
            return {"MAE": float(r["MAE"]), "RMSE": float(r["RMSE"]), "R2": float(r["R2"])}
    return {"MAE": 68.95, "RMSE": 157.89, "R2": 0.828}


def predict_ml(models, row):
    """Predicción del modelo Lasso para las 24 horas de un día base."""
    preds = []
    for h in range(24):
        m = models[h]
        features = [f for f in m["features"] if f in row.index]
        X = row[features].to_frame().T
        preds.append(float(m["pipeline"].predict(X)[0]))
    return preds


# --- Verificación de artefactos ---
if not MODEL_PATH.exists():
    st.error(
        "No se encontró el modelo entrenado. Ejecute primero: "
        "python -m src.train_final_model"
    )
    st.stop()

models = load_model()
metadata = load_metadata()
df = load_features()
metrics_by_hour = load_metrics_by_hour()
persist_metrics = load_persistencia_metrics()
lasso_metrics = metadata["metricas_test_nb07"]

# --- Encabezado ---
st.title("Predicción day-ahead del precio de bolsa de energía")
st.markdown(
    "Mercado eléctrico colombiano (SIN). Pronóstico de las **24 horas del día "
    "siguiente** a partir de la información disponible hasta el día base."
)

# --- Barra lateral ---
with st.sidebar:
    st.header("Modelos comparados")
    st.caption(
        "Desempeño en el conjunto de prueba (dic-2022 a mar-2026, partición "
        "temporal 80/20)."
    )
    comp = pd.DataFrame([
        {"Modelo": "Persistencia diaria (referencia)",
         "MAE": round(persist_metrics["MAE"], 1),
         "RMSE": round(persist_metrics["RMSE"], 1),
         "R2": round(persist_metrics["R2"], 3)},
        {"Modelo": "Lasso (comparación)",
         "MAE": round(lasso_metrics["MAE"], 1),
         "RMSE": round(lasso_metrics["RMSE"], 1),
         "R2": round(lasso_metrics["R2"], 3)},
    ])
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption(
        "La persistencia diaria es el modelo de referencia seleccionado por su "
        "menor MAE. Al recalibrar el modelo periódicamente (notebook NB08), este "
        "iguala la persistencia en MAE y la supera de forma significativa en RMSE "
        "y R2, reduciendo los errores grandes en periodos de alta volatilidad."
    )

    st.divider()
    st.subheader("Día base")
    st.caption("Se predice el precio horario del día siguiente al seleccionado.")

    fechas_validas = df.dropna(subset=["fecha_objetivo"]).copy()
    fecha_min = fechas_validas["FECHA"].min().date()
    fecha_max = fechas_validas["FECHA"].max().date()

    fecha_sel = st.date_input(
        "Fecha base", value=fecha_max, min_value=fecha_min, max_value=fecha_max,
    )

# --- Selección de la fila del día base ---
fila_match = df[df["FECHA"].dt.date == fecha_sel]
if fila_match.empty:
    st.warning(
        f"No hay datos para el día base {fecha_sel}. "
        "Seleccione una fecha dentro del rango disponible."
    )
    st.stop()

row = fila_match.iloc[0]
fecha_objetivo = row["fecha_objetivo"]
horas = list(range(24))

# --- Pronósticos ---
pred_persist = [float(row[f"h{h:02d}"]) for h in range(24)]   # persistencia: hoy -> mañana
pred_ml = predict_ml(models, row)

tiene_real = row[TARGET_COLS].notna().any()
reales = [float(row[f"target_h{h:02d}"]) if tiene_real else np.nan for h in range(24)]

# --- Métricas resumen del pronóstico (persistencia = referencia) ---
promedio = np.mean(pred_persist)
pico = np.mean([pred_persist[h] for h in PEAK_HOURS])
hora_max = int(np.argmax(pred_persist))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Día objetivo", fecha_objetivo.strftime("%Y-%m-%d"))
col2.metric("Precio promedio (referencia)", f"${promedio:.1f}")
col3.metric("Promedio pico (18-22 h)", f"${pico:.1f}")
col4.metric("Hora más cara", f"{hora_max:02d}:00 (${pred_persist[hora_max]:.0f})")

# --- Gráfico principal: perfil horario ---
st.subheader(f"Perfil horario del día siguiente: {fecha_larga_es(fecha_objetivo)}")

fig = go.Figure()
fig.add_vrect(
    x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08, line_width=0,
    annotation_text="Franja pico", annotation_position="top left",
)
fig.add_trace(go.Scatter(
    x=horas, y=pred_persist, mode="lines+markers",
    name="Persistencia diaria (referencia)", line=dict(color="#2563eb", width=3),
    marker=dict(size=7),
))
fig.add_trace(go.Scatter(
    x=horas, y=pred_ml, mode="lines+markers",
    name="Modelo Lasso (comparación)", line=dict(color="#16a34a", width=2),
    marker=dict(size=6),
))
if tiene_real:
    fig.add_trace(go.Scatter(
        x=horas, y=reales, mode="lines+markers",
        name="Precio real", line=dict(color="#111827", width=2, dash="dot"),
        marker=dict(size=6),
    ))
fig.update_layout(
    xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
    yaxis=dict(title="Precio de bolsa ($/kWh)"),
    hovermode="x unified", height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
)
st.plotly_chart(fig, use_container_width=True)

# --- Error del día (ambos modelos) cuando hay valores reales ---
if tiene_real:
    mae_persist = np.mean([abs(reales[h] - pred_persist[h]) for h in range(24)])
    mae_ml = np.mean([abs(reales[h] - pred_ml[h]) for h in range(24)])
    c1, c2 = st.columns(2)
    c1.metric("MAE del día (persistencia)", f"${mae_persist:.1f}")
    c2.metric("MAE del día (Lasso)", f"${mae_ml:.1f}")
    st.caption("La línea punteada muestra el precio realmente observado ese día.")

# --- Tabla de predicción ---
with st.expander("Ver tabla por hora"):
    tabla = pd.DataFrame({
        "Hora": [f"{h:02d}:00" for h in horas],
        "Persistencia ($/kWh)": [round(p, 2) for p in pred_persist],
        "Lasso ($/kWh)": [round(p, 2) for p in pred_ml],
        "Franja pico": ["Sí" if h in PEAK_HOURS else "" for h in horas],
    })
    if tiene_real:
        tabla["Real ($/kWh)"] = [round(reales[h], 2) for h in horas]
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar pronóstico (CSV)", data=csv,
        file_name=f"pronostico_dayahead_{fecha_objetivo.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# --- Desempeño por hora del modelo Lasso ---
if not metrics_by_hour.empty:
    st.subheader("Desempeño del modelo Lasso por hora (conjunto de prueba)")
    st.caption(
        "Error medio absoluto (MAE) y R2 por hora del día. Las horas pico (18-22) "
        "suelen ser más difíciles de predecir."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            x=metrics_by_hour["hora"], y=metrics_by_hour["MAE"],
            marker_color=["crimson" if h in PEAK_HOURS else "#2563eb"
                          for h in metrics_by_hour["hora"]],
        ))
        fig_mae.update_layout(title="MAE por hora", xaxis_title="Hora",
                              yaxis_title="MAE", height=340, margin=dict(t=40))
        st.plotly_chart(fig_mae, use_container_width=True)
    with c2:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=metrics_by_hour["hora"], y=metrics_by_hour["R2"],
            marker_color=["crimson" if h in PEAK_HOURS else "#16a34a"
                          for h in metrics_by_hour["hora"]],
        ))
        fig_r2.update_layout(title="R2 por hora", xaxis_title="Hora",
                             yaxis_title="R2", height=340, margin=dict(t=40))
        st.plotly_chart(fig_r2, use_container_width=True)

# --- Pie ---
st.divider()
st.caption(
    "Proyecto aplicado, Maestría en Ciencia de Datos y Analítica, UNAD. "
    "Modelo de referencia: persistencia diaria. Modelo de comparación: 24 "
    "regresiones Lasso por hora con variables exógenas (ONI, embalses). "
    "Fuente de datos: XM (SIN) y NOAA CPC."
)
