"""
Aplicación web de predicción del precio horario de bolsa de energía para el día
siguiente en el mercado eléctrico colombiano (SIN).

Presenta la persistencia diaria como modelo de referencia seleccionado en el
proyecto y un modelo de aprendizaje automático (Lasso, 24 regresiones por hora)
como comparación. La interfaz se organiza en cuatro vistas: histórico del precio,
predicción del día siguiente, exploración de patrones y relación con el clima.

Ejecutar:
    streamlit run app/app.py
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# --- Rutas del proyecto ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

MODEL_PATH = MODELS_DIR / "modelo_final_dayahead.joblib"
MODEL_RESIDUAL_PATH = MODELS_DIR / "modelo_residual_dayahead.joblib"
METADATA_PATH = MODELS_DIR / "modelo_final_metadata.json"
FEATURES_PATH = PROCESSED_DIR / "features_day_ahead.csv"
METRICS_BY_HOUR_PATH = OUTPUTS_DIR / "hourly_dayahead_metrics_by_hour.csv"
BASELINE_RESULTS_PATH = OUTPUTS_DIR / "baseline_global_results.csv"
BASELINE_HOURLY_PATH = OUTPUTS_DIR / "baseline_hourly_results.csv"
REGIMEN_PATH = OUTPUTS_DIR / "consolidado_por_regimen.csv"
INTERVALOS_PATH = OUTPUTS_DIR / "intervalos_conformales.csv"
PREDICCION_AUTO_PATH = OUTPUTS_DIR / "prediccion_automatica.csv"

PEAK_HOURS = [18, 19, 20, 21, 22]
TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]
HOUR_COLS = [f"h{h:02d}" for h in range(24)]
ML_MODEL_LABEL = "Lasso_horario_con_exogenas"

AZUL = "#2563eb"
VERDE = "#16a34a"
NEGRO = "#111827"
NARANJA = "#ea580c"

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
TIPO_DIA_ES = {"habil": "Hábil", "sabado": "Sábado", "domingo": "Domingo",
               "festivo": "Festivo"}


def fecha_larga_es(fecha):
    """Formatea una fecha en español: 'Sábado 28 de marzo de 2026'."""
    dia = DIAS_ES.get(fecha.strftime("%A"), fecha.strftime("%A"))
    return f"{dia} {fecha.day} de {MESES_ES[fecha.month]} de {fecha.year}"


st.set_page_config(
    page_title="Predicción del precio de bolsa: mercado eléctrico colombiano",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Carga cacheada de artefactos ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_model_residual():
    return joblib.load(MODEL_RESIDUAL_PATH) if MODEL_RESIDUAL_PATH.exists() else None


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
def serie_historica(_df):
    """Serie diaria del precio observado: promedio, mínimo y máximo de cada día."""
    s = _df.dropna(subset=["FECHA"]).copy()
    s["promedio"] = s[HOUR_COLS].mean(axis=1)
    s["minimo"] = s[HOUR_COLS].min(axis=1)
    s["maximo"] = s[HOUR_COLS].max(axis=1)
    return s[["FECHA", "promedio", "minimo", "maximo"]].reset_index(drop=True)


@st.cache_data
def serie_horaria_larga(_df):
    """Precio en formato largo: una fila por día y hora, con el tipo de día."""
    id_vars = ["FECHA"] + (["base_tipo_dia"] if "base_tipo_dia" in _df.columns else [])
    m = _df.dropna(subset=["FECHA"]).melt(
        id_vars=id_vars, value_vars=HOUR_COLS, var_name="hora_col", value_name="precio")
    m["hora"] = m["hora_col"].str[1:].astype(int)
    return m.dropna(subset=["precio"]).reset_index(drop=True)


@st.cache_data
def serie_clima(_df):
    """Serie diaria del precio promedio junto a las variables exógenas."""
    s = _df.dropna(subset=["FECHA"]).copy()
    s["precio_promedio"] = s[HOUR_COLS].mean(axis=1)
    cols = ["FECHA", "precio_promedio"]
    if "oni" in s.columns:
        cols.append("oni")
    if "embalse_porcentaje_diario" in s.columns:
        cols.append("embalse_porcentaje_diario")
    return s[cols].reset_index(drop=True)


@st.cache_data
def load_metrics_by_hour():
    if not METRICS_BY_HOUR_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(METRICS_BY_HOUR_PATH)
    return df[df["modelo"] == ML_MODEL_LABEL].reset_index(drop=True)


@st.cache_data
def load_persistencia_metrics():
    if BASELINE_RESULTS_PATH.exists():
        df = pd.read_csv(BASELINE_RESULTS_PATH)
        fila = df[df["modelo"] == "Baseline_hoy"]
        if not fila.empty:
            r = fila.iloc[0]
            return {"MAE": float(r["MAE"]), "RMSE": float(r["RMSE"]), "R2": float(r["R2"])}
    return {"MAE": 68.95, "RMSE": 157.89, "R2": 0.828}


@st.cache_data
def load_error_por_hora():
    """MAE por hora del día: Lasso con exógenas frente a la persistencia diaria."""
    if not (METRICS_BY_HOUR_PATH.exists() and BASELINE_HOURLY_PATH.exists()):
        return pd.DataFrame()
    ml = pd.read_csv(METRICS_BY_HOUR_PATH)
    ml = ml.loc[ml["modelo"] == ML_MODEL_LABEL, ["hora", "MAE"]].copy()
    ml["Modelo"] = "Lasso"
    bl = pd.read_csv(BASELINE_HOURLY_PATH)
    bl = bl.loc[bl["modelo"] == "Baseline_hoy", ["hora", "MAE"]].copy()
    bl["Modelo"] = "Persistencia"
    return pd.concat([ml, bl], ignore_index=True).sort_values(["hora", "Modelo"])


@st.cache_data
def load_intervalos():
    """Intervalos conformales por hora (q_low, q_high) para Lasso y Persistencia."""
    if not INTERVALOS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(INTERVALOS_PATH)


@st.cache_data(ttl=3600)
def load_prediccion_auto():
    """
    Carga la prediccion automatica generada por el pipeline diario.
    Retorna None si el archivo no existe o la prediccion tiene mas de 2 dias
    de antiguedad (posible fallo del pipeline).
    """
    if not PREDICCION_AUTO_PATH.exists():
        return None
    df = pd.read_csv(PREDICCION_AUTO_PATH)
    if df.empty or "fecha_objetivo" not in df.columns:
        return None
    fecha_obj = pd.to_datetime(df["fecha_objetivo"].iloc[0]).date()
    from datetime import date as _date  # noqa: PLC0415
    # Mostrar si la prediccion tiene como maximo 5 dias de diferencia con hoy
    # (el pipeline de XM tiene un rezago de hasta 3 dias en la disponibilidad de datos)
    if abs((fecha_obj - _date.today()).days) > 5:
        return None
    return df


@st.cache_data
def load_regimen():
    """Error (MAE) por estrategia según el régimen del precio (calmo / volátil)."""
    if not REGIMEN_PATH.exists():
        return pd.DataFrame()
    d = pd.read_csv(REGIMEN_PATH)
    d["Régimen"] = d["regimen"].map(
        {"calmo (<300)": "Precio calmo<br>(< 300 $/kWh)",
         "volatil (>=300)": "Precio volátil<br>(≥ 300 $/kWh)"}
    ).fillna(d["regimen"])
    d["Modelo"] = d["estrategia"].map(
        {"Persistencia diaria": "Persistencia (referencia)",
         "Lasso horario": "Lasso",
         "Aprendizaje residual": "Aprendizaje residual"}
    ).fillna(d["estrategia"])
    return d


def predict_ml(models, row):
    """Predicción del modelo Lasso para las 24 horas de un día base."""
    preds = []
    for h in range(24):
        m = models[h]
        features = [f for f in m["features"] if f in row.index]
        X = row[features].to_frame().T
        preds.append(float(m["pipeline"].predict(X)[0]))
    return preds


def predict_residual(models_res, row):
    """Aprendizaje residual: persistencia (precio del día base) + residuo estimado."""
    preds = []
    for h in range(24):
        m = models_res[h]
        features = [f for f in m["features"] if f in row.index]
        X = row[features].to_frame().T
        resid = float(m["pipeline"].predict(X)[0])
        preds.append(float(row[f"h{h:02d}"]) + resid)
    return preds


def estilo_grafico(fig, plantilla, alto=440):
    fig.update_layout(template=plantilla, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=alto, margin=dict(t=40))
    return fig


# --- Verificación de artefactos ---
if not MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecute primero: "
             "python -m src.train_final_model")
    st.stop()

models = load_model()
models_res = load_model_residual()
metadata = load_metadata()
df = load_features()
historico = serie_historica(df)
metrics_by_hour = load_metrics_by_hour()
persist_metrics = load_persistencia_metrics()
lasso_metrics = metadata["metricas_test_nb07"]

# --- Selector de tema claro / oscuro ---
with st.sidebar:
    oscuro = st.toggle("🌙 Modo oscuro", value=False, help="Cambia entre tema claro y oscuro")
    st.divider()

PLANTILLA = "plotly_dark" if oscuro else "plotly_white"
COLOR_REAL = "#e5e7eb" if oscuro else NEGRO

# --- Estilo de las pestañas: las inactivas en color neutro (no azul de enlace),
# la activa en negrita con fondo e indicador visibles, para que se reconozcan
# como pestañas y no como texto plano. ---
_TAB_INACTIVO = "#9ca3af" if oscuro else "#6b7280"
_TAB_BORDE = "#2a2e37" if oscuro else "#e5e7eb"
_TAB_FONDO_ACTIVO = "rgba(37, 99, 235, 0.18)" if oscuro else "rgba(37, 99, 235, 0.10)"
_TAB_FONDO_HOVER = "rgba(37, 99, 235, 0.10)" if oscuro else "rgba(37, 99, 235, 0.06)"

# --- Estilo de los botones secundarios (navegación de días, descarga): contraste
# explícito en ambos modos, para que nunca queden en blanco sobre blanco. ---
_BTN_FONDO = "#1a1d24" if oscuro else "#ffffff"
_BTN_TEXTO = "#fafafa" if oscuro else "#111827"
_BTN_BORDE = "#3f3f46" if oscuro else "#d1d5db"
_BTN_HOVER_FONDO = "#262a33" if oscuro else "#f1f5f9"

# CSS del modo oscuro. Se interpola dentro del mismo bloque de estilos de abajo (no
# como un st.markdown aparte) para que la cantidad de elementos antes de las pestañas
# no cambie al alternar el tema: así st.tabs no se re-monta y conserva la pestaña activa.
_dark_css = """
    .stApp { background-color: #1c212c !important; }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    h1, h2, h3, h4, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: #fafafa !important; }
    [data-testid="stSidebar"] { background-color: #242a38 !important; }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    [data-testid="stHeader"] { background: rgba(0, 0, 0, 0) !important; }
""" if oscuro else ""

st.markdown(
    f"""
    <style>
    /* Sube el título: reduce el espacio superior por debajo de la barra fija. */
    [data-testid="stMainBlockContainer"] {{
        padding-top: 3.75rem !important;
    }}
    button[data-testid^="stBaseButton"] {{
        background-color: {_BTN_FONDO} !important;
        border: 1px solid {_BTN_BORDE} !important;
    }}
    button[data-testid^="stBaseButton"] p {{
        color: {_BTN_TEXTO} !important;
    }}
    button[data-testid^="stBaseButton"]:hover {{
        background-color: {_BTN_HOVER_FONDO} !important;
        border-color: {AZUL} !important;
    }}
    button[data-testid^="stBaseButton"]:hover p {{
        color: {AZUL} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 2px solid {_TAB_BORDE};
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"] {{
        height: auto;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        background-color: transparent;
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"] p {{
        color: {_TAB_INACTIVO} !important;
        font-weight: 500 !important;
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        background-color: {_TAB_FONDO_HOVER};
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"]:hover p {{
        color: {AZUL} !important;
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {_TAB_FONDO_ACTIVO};
    }}
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {{
        color: {AZUL} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: {AZUL} !important;
        height: 3px;
    }}
    {_dark_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Encabezado ---
st.title("Precio de bolsa de energía: histórico, predicción y exploración")
st.markdown(
    "Mercado eléctrico colombiano (SIN). Las pestañas siguen el orden del análisis: el "
    "**histórico** del precio, la **exploración** de patrones y la relación con el "
    "**clima**, luego la **predicción** del día siguiente y el **desempeño** del modelo."
)

# --- Barra lateral: contexto de los modelos ---
with st.sidebar:
    st.header("¿Qué modelos se comparan?")
    st.markdown(
        "**Persistencia diaria** (referencia): asume que mañana, a cada hora, el "
        "precio será el de hoy a esa misma hora. Es simple y difícil de superar.\n\n"
        "**Modelo Lasso** (comparación): un modelo de aprendizaje automático, uno por "
        "cada hora del día, con variables de clima y embalses.\n\n"
        "**Aprendizaje residual** (comparación): la persistencia más una corrección "
        "aprendida; destaca en el régimen volátil."
    )
    comp_filas = [
        {"Modelo": "Persistencia diaria", "MAE": round(persist_metrics["MAE"], 1),
         "RMSE": round(persist_metrics["RMSE"], 1), "R2": round(persist_metrics["R2"], 3)},
        {"Modelo": "Lasso", "MAE": round(lasso_metrics["MAE"], 1),
         "RMSE": round(lasso_metrics["RMSE"], 1), "R2": round(lasso_metrics["R2"], 3)},
    ]
    if models_res is not None:
        comp_filas.append({"Modelo": "Aprendizaje residual", "MAE": 70.5,
                           "RMSE": 148.7, "R2": 0.847})
    st.dataframe(pd.DataFrame(comp_filas), hide_index=True, use_container_width=True)
    st.caption(
        "MAE y RMSE son el error promedio del pronóstico, en pesos por kilovatio hora: "
        "entre más bajos, mejor. R2 indica qué tanto explica el modelo el "
        "comportamiento del precio. Conjunto de prueba: diciembre de 2022 a marzo de 2026."
    )

with st.expander("¿Cómo usar esta aplicación?"):
    st.markdown(
        "La aplicación tiene **seis pestañas** que siguen el orden del análisis:\n\n"
        "- **Histórico del precio**: evolución del precio de bolsa desde 2012, con "
        "filtros de periodo y agregaciones diaria, semanal y mensual.\n"
        "- **Explorar patrones**: distribución del precio por hora del día y por tipo "
        "de día (hábil, sábado, domingo, festivo), usando diagramas de caja.\n"
        "- **Clima y precio**: relación del precio con el índice de El Niño (ONI) y "
        "el nivel de embalses, con gráficas de serie temporal y dispersión.\n"
        "- **Predicción del día siguiente**: elige un día base con los botones o el "
        "calendario, selecciona qué modelos mostrar y consulta el pronóstico horario. "
        "Si el día ya ocurrió, aparece también el precio real observado y las métricas "
        "comparativas (MAE, RMSE, sMAPE).\n"
        "- **Desempeño del modelo**: error medio por hora del día y comparación según "
        "el régimen del precio (calmo o volátil). Es el resultado central del proyecto.\n"
        "- **Pronóstico con datos nuevos**: predicción automática del pipeline diario "
        "(actualizada cada mañana desde XM) y carga manual de un CSV con días base "
        "propios para generar pronósticos con los mismos modelos."
    )

# Orden del análisis: primero entender los datos (histórico, patrones, clima) y luego
# el modelo (predicción y desempeño).
tab_hist, tab_expl, tab_clima, tab_pred, tab_desemp, tab_carga = st.tabs(
    ["Histórico del precio", "Explorar patrones", "Clima y precio",
     "Predicción del día siguiente", "Desempeño del modelo", "Pronóstico con datos nuevos"])


# ======================================================================== #
# Vista 1: histórico del precio (rango y agregación)
# ======================================================================== #
def filtrar_periodo(serie, key):
    """Devuelve la serie filtrada según un selector de periodo con dos modos."""
    modo = st.radio("Cómo elegir el periodo", ["Rangos rápidos", "Rango personalizado"],
                    horizontal=True, key=f"modo_{key}")
    fmin = serie["FECHA"].min().date()
    fmax = serie["FECHA"].max().date()
    if modo == "Rangos rápidos":
        opciones = {"Último mes": 30, "Últimos 3 meses": 90, "Último año": 365,
                    "Todo el histórico": None}
        etiqueta = st.radio("Periodo", list(opciones.keys()), index=2, horizontal=True,
                            key=f"preset_{key}")
        dias = opciones[etiqueta]
        s = serie if dias is None else serie[serie["FECHA"] >= serie["FECHA"].max() - pd.Timedelta(days=dias)]
    else:
        rango = st.slider("Rango de fechas", min_value=fmin, max_value=fmax,
                          value=(max(fmin, fmax.replace(year=fmax.year - 1)), fmax),
                          key=f"slider_{key}")
        s = serie[(serie["FECHA"].dt.date >= rango[0]) & (serie["FECHA"].dt.date <= rango[1])]
    return s


with tab_hist:
    st.subheader("¿Cómo se ha comportado el precio de bolsa?")
    st.caption("Precio promedio de cada día. La franja sombreada es el rango entre el "
               "mínimo y el máximo del día.")

    hist = filtrar_periodo(historico, "hist")
    agr = st.radio("Agregación", ["Diaria", "Semanal", "Mensual"], horizontal=True,
                   key="agr_hist")
    if agr != "Diaria" and not hist.empty:
        regla = {"Semanal": "W", "Mensual": "MS"}[agr]
        hist = (hist.set_index("FECHA")
                    .resample(regla)
                    .agg(promedio=("promedio", "mean"), minimo=("minimo", "min"),
                         maximo=("maximo", "max"))
                    .dropna().reset_index())

    if hist.empty:
        st.info("No hay datos en el periodo seleccionado.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio promedio del periodo", f"${hist['promedio'].mean():.1f}")
        m2.metric("Máximo", f"${hist['maximo'].max():.1f}")
        m3.metric("Mínimo", f"${hist['minimo'].min():.1f}")
        m4.metric("Puntos mostrados", f"{len(hist)}")

        # Franja del rango min-max en naranja semitransparente, legible en claro y oscuro.
        color_rango = "rgba(251,146,60,0.30)" if oscuro else "rgba(234,88,12,0.18)"
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist["FECHA"], y=hist["maximo"], mode="lines",
                                   line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_h.add_trace(go.Scatter(x=hist["FECHA"], y=hist["minimo"], mode="lines", fill="tonexty",
                                   fillcolor=color_rango, line=dict(width=0),
                                   name="Rango del día (mín-máx)", hoverinfo="skip"))
        fig_h.add_trace(go.Scatter(x=hist["FECHA"], y=hist["promedio"], mode="lines",
                                   name="Precio promedio", line=dict(color=AZUL, width=2)))
        fig_h.update_layout(xaxis_title="Fecha", yaxis_title="Precio de bolsa ($/kWh)",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1))
        st.plotly_chart(estilo_grafico(fig_h, PLANTILLA, 460), use_container_width=True)


# ======================================================================== #
# Vista 2: predicción del día siguiente (navegación y modelos)
# ======================================================================== #
with tab_pred:
    st.subheader("Elige un día y consulta la predicción del día siguiente")
    fechas_validas = df.dropna(subset=["fecha_objetivo"])
    fecha_min = fechas_validas["FECHA"].min().date()
    fecha_max = fechas_validas["FECHA"].max().date()
    st.session_state.setdefault("fecha_pred", fecha_max)

    def _mover(delta):
        nueva = st.session_state.fecha_pred + timedelta(days=delta)
        st.session_state.fecha_pred = max(fecha_min, min(fecha_max, nueva))

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("◀ Día anterior", on_click=_mover, args=(-1,), use_container_width=True)
    with nav3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("Día siguiente ▶", on_click=_mover, args=(1,), use_container_width=True)
    with nav2:
        st.date_input("Día base", min_value=fecha_min, max_value=fecha_max, key="fecha_pred")
    fecha_sel = st.session_state.fecha_pred

    st.markdown("**Mostrar en la gráfica:**")
    m1, m2, m3, m4 = st.columns(4)
    ver_real = m1.checkbox("Precio real observado", value=True)
    ver_lasso = m2.checkbox("Modelo Lasso", value=True)
    ver_residual = m3.checkbox("Aprendizaje residual", value=models_res is not None,
                               disabled=models_res is None)
    ver_persist = m4.checkbox("Persistencia diaria", value=True)

    intervalos = load_intervalos()
    b1, b2 = st.columns([1.1, 1])
    ver_banda = b1.checkbox("Mostrar intervalo de incertidumbre (90 %)", value=False,
                            disabled=intervalos.empty,
                            help="Banda conformal: el rango donde se espera que caiga el "
                                 "precio real con un 90 % de probabilidad.")
    opciones_banda = ["Lasso", "Persistencia"]
    if models_res is not None:
        opciones_banda.insert(1, "Aprendizaje residual")
    modelo_banda = b2.selectbox("Intervalo del modelo", opciones_banda,
                                disabled=(intervalos.empty or not ver_banda),
                                label_visibility="collapsed")

    fila_match = df[df["FECHA"].dt.date == fecha_sel]
    if fila_match.empty:
        st.warning(f"No hay datos para el día base {fecha_sel}. Elige otra fecha.")
    else:
        row = fila_match.iloc[0]
        fecha_objetivo = row["fecha_objetivo"]
        horas = list(range(24))
        pred_persist = [float(row[f"h{h:02d}"]) for h in range(24)]
        pred_ml = predict_ml(models, row)
        pred_residual = predict_residual(models_res, row) if models_res is not None else None
        tiene_real = row[TARGET_COLS].notna().any()
        reales = [float(row[f"target_h{h:02d}"]) if tiene_real else np.nan for h in range(24)]

        promedio = np.mean(pred_persist)
        pico = np.mean([pred_persist[h] for h in PEAK_HOURS])
        hora_max = int(np.argmax(pred_persist))

        st.markdown(f"**Día pronosticado:** {fecha_larga_es(fecha_objetivo)}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio promedio", f"${promedio:.1f}")
        c2.metric("Promedio en horas pico (18-22 h)", f"${pico:.1f}")
        c3.metric("Hora más cara", f"{hora_max:02d}:00 (${pred_persist[hora_max]:.0f})")

        fig = go.Figure()
        fig.add_vrect(x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08, line_width=0,
                      annotation_text="Franja pico", annotation_position="top left")
        # Banda de incertidumbre conformal del modelo seleccionado (detrás de las líneas).
        if ver_banda and not intervalos.empty:
            base_por_modelo = {"Lasso": pred_ml, "Persistencia": pred_persist,
                               "Aprendizaje residual": pred_residual}
            base_pred = base_por_modelo.get(modelo_banda) or pred_ml
            iv = intervalos[intervalos["modelo"] == modelo_banda].set_index("hora")
            lower = [base_pred[h] + float(iv.loc[h, "q_low"]) for h in horas]
            upper = [base_pred[h] + float(iv.loc[h, "q_high"]) for h in horas]
            fill_por_modelo = {"Lasso": "rgba(22,163,74,0.15)",
                               "Persistencia": "rgba(37,99,235,0.15)",
                               "Aprendizaje residual": "rgba(234,88,12,0.15)"}
            fill_banda = fill_por_modelo.get(modelo_banda, "rgba(22,163,74,0.15)")
            fig.add_trace(go.Scatter(x=horas, y=upper, mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=horas, y=lower, mode="lines", fill="tonexty",
                                     fillcolor=fill_banda, line=dict(width=0),
                                     name=f"Intervalo 90 % ({modelo_banda})",
                                     hoverinfo="skip"))
        if ver_persist:
            fig.add_trace(go.Scatter(x=horas, y=pred_persist, mode="lines+markers",
                                     name="Persistencia diaria (referencia)",
                                     line=dict(color=AZUL, width=3), marker=dict(size=7)))
        if ver_lasso:
            fig.add_trace(go.Scatter(x=horas, y=pred_ml, mode="lines+markers",
                                     name="Modelo Lasso (comparación)",
                                     line=dict(color=VERDE, width=2), marker=dict(size=6)))
        if ver_residual and pred_residual is not None:
            fig.add_trace(go.Scatter(x=horas, y=pred_residual, mode="lines+markers",
                                     name="Aprendizaje residual (comparación)",
                                     line=dict(color=NARANJA, width=2), marker=dict(size=6)))
        if tiene_real and ver_real:
            fig.add_trace(go.Scatter(x=horas, y=reales, mode="lines+markers",
                                     name="Precio real observado",
                                     line=dict(color=COLOR_REAL, width=2, dash="dot"),
                                     marker=dict(size=6)))
        fig.update_layout(xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
                          yaxis=dict(title="Precio de bolsa ($/kWh)"), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="top", y=-0.2,
                                      xanchor="center", x=0.5))
        fig = estilo_grafico(fig, PLANTILLA, 460)
        fig.update_layout(margin=dict(t=40, b=90))
        st.plotly_chart(fig, use_container_width=True)

        if ver_banda and not intervalos.empty:
            st.caption(
                f"La **banda sombreada** es el intervalo de incertidumbre del **90 %** del "
                f"modelo {modelo_banda}: con esa probabilidad el precio real cae dentro de "
                f"ella. Se obtiene por predicción conformal sobre el conjunto de prueba y es "
                f"más ancha en las horas de mayor volatilidad."
            )

        st.caption(
            "\\* La línea **azul, persistencia diaria**, repite el precio real del día "
            "base como pronóstico del día siguiente: es la referencia simple a superar. "
            "La **verde** es el pronóstico del modelo Lasso, la **naranja** el del "
            "aprendizaje residual y la **punteada** es el precio real observado, que "
            "aparece solo cuando ya hay datos de esa fecha."
        )

        if tiene_real:
            reales_arr = np.array(reales, dtype=float)

            def _mae(pred):
                return float(np.mean(np.abs(reales_arr - np.array(pred, dtype=float))))

            def _rmse(pred):
                return float(np.sqrt(np.mean((reales_arr - np.array(pred, dtype=float)) ** 2)))

            def _smape(pred):
                pred = np.array(pred, dtype=float)
                denom = np.abs(reales_arr) + np.abs(pred)
                return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(reales_arr - pred) / denom)) * 100)

            modelos_met = {"Lasso": pred_ml}
            if pred_residual is not None:
                modelos_met["Aprendizaje residual"] = pred_residual
            modelos_met["Persistencia"] = pred_persist

            met = pd.DataFrame({"Métrica": ["MAE ($/kWh)", "RMSE ($/kWh)", "sMAPE (%)"]})
            for nombre, pred in modelos_met.items():
                met[nombre] = [round(_mae(pred), 2), round(_rmse(pred), 2), round(_smape(pred), 2)]
            cols_modelo = list(modelos_met.keys())
            met["Mejor"] = [min(cols_modelo, key=lambda c: met.loc[i, c]) for i in met.index]
            st.markdown("**Métricas comparativas del día**")
            st.dataframe(met, use_container_width=True, hide_index=True)
            st.caption("Error del pronóstico del día siguiente frente al precio real observado; "
                       "entre más bajos, mejor. La columna *Mejor* indica el modelo con menor error "
                       "en cada métrica.")
        else:
            st.info("Este día base aún no tiene precio observado del día siguiente, por lo que no "
                    "se pueden calcular las métricas comparativas.")

        st.markdown("**Detalle por hora del día siguiente**")
        tabla = pd.DataFrame({
            "Hora": [f"{h:02d}:00" for h in horas],
            "Persistencia ($/kWh)": [round(p, 2) for p in pred_persist],
            "Lasso ($/kWh)": [round(p, 2) for p in pred_ml],
        })
        if pred_residual is not None:
            tabla["Aprendizaje residual ($/kWh)"] = [round(p, 2) for p in pred_residual]
        tabla["Franja pico"] = ["Sí" if h in PEAK_HOURS else "" for h in horas]
        if tiene_real:
            tabla["Real ($/kWh)"] = [round(reales[h], 2) for h in horas]
        st.dataframe(tabla, use_container_width=True, hide_index=True, height=300)
        csv = tabla.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Descargar pronóstico (CSV)", data=csv,
                           file_name=f"pronostico_{fecha_objetivo.strftime('%Y%m%d')}.csv",
                           mime="text/csv")


# ======================================================================== #
# Vista 3: explorar patrones (distribución)
# ======================================================================== #
with tab_expl:
    st.subheader("Patrones del precio")
    st.caption("Cada caja resume la distribución del precio: la línea central es la "
               "mediana y la caja abarca la mitad de los valores. Ayuda a ver en qué "
               "horas y en qué días el precio es más alto o más disperso.")

    larga = serie_horaria_larga(df)
    if len(larga) > 60000:
        larga = larga.sample(60000, random_state=42)

    st.markdown("**Distribución del precio por hora del día**")
    # points=False oculta los valores atípicos: así las cajas se agrandan y se comparan
    # mejor las medianas entre horas.
    fig_hora = px.box(larga, x="hora", y="precio", points=False,
                      labels={"hora": "Hora del día", "precio": "Precio de bolsa ($/kWh)"},
                      color_discrete_sequence=[AZUL])
    fig_hora.update_traces(whiskerwidth=0.5)
    st.plotly_chart(estilo_grafico(fig_hora, PLANTILLA, 400), use_container_width=True)
    st.caption("Los bigotes llegan hasta 1.5 veces el rango intercuartílico. "
               "Los valores individuales más extremos se omiten para no distorsionar la escala.")

    if "base_tipo_dia" in larga.columns:
        st.markdown("**Distribución del precio por tipo de día**")
        larga["Tipo de día"] = larga["base_tipo_dia"].map(TIPO_DIA_ES).fillna(larga["base_tipo_dia"])
        orden = ["Hábil", "Sábado", "Domingo", "Festivo"]
        fig_tipo = px.box(larga, x="Tipo de día", y="precio", points=False,
                          category_orders={"Tipo de día": orden},
                          labels={"precio": "Precio de bolsa ($/kWh)"},
                          color_discrete_sequence=[VERDE])
        fig_tipo.update_traces(whiskerwidth=0.5)
        st.plotly_chart(estilo_grafico(fig_tipo, PLANTILLA, 400), use_container_width=True)


# ======================================================================== #
# Vista 4: desempeño del modelo (error por hora y por régimen)
# ======================================================================== #
with tab_desemp:
    st.subheader("¿Qué tan bien predice el modelo, y cuándo?")
    st.caption("Evaluación sobre el conjunto de prueba (diciembre de 2022 a marzo de "
               "2026), desde dos ángulos: el error a lo largo del día y según el régimen "
               "del precio. Es el resultado central del proyecto.")

    st.markdown("**Error medio por hora del día (MAE)**")
    eph = load_error_por_hora()
    if eph.empty:
        st.info("No hay datos de error por hora disponibles.")
    else:
        fig_eph = px.line(
            eph, x="hora", y="MAE", color="Modelo", markers=True,
            labels={"hora": "Hora del día", "MAE": "MAE ($/kWh)"},
            color_discrete_map={"Lasso": VERDE, "Persistencia": AZUL})
        fig_eph.add_vrect(x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08, line_width=0,
                          annotation_text="Franja pico", annotation_position="top left")
        fig_eph.update_xaxes(tickmode="linear", dtick=2)
        fig_eph.update_layout(hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="right", x=1))
        st.plotly_chart(estilo_grafico(fig_eph, PLANTILLA, 400), use_container_width=True)
        st.caption("Entre más bajo el MAE, mejor. El error de ambos modelos crece en las "
                   "horas de la tarde y la noche, que son las de mayor precio y "
                   "volatilidad; es ahí donde el pronóstico es más difícil.")

    st.markdown("**Desempeño según el régimen del precio**")
    reg = load_regimen()
    if reg.empty:
        st.info("No hay datos de régimen disponibles.")
    else:
        destacadas = ["Persistencia diaria", "Lasso horario", "Aprendizaje residual"]
        reg_f = reg[reg["estrategia"].isin(destacadas)].copy()
        orden_reg = ["Precio calmo<br>(< 300 $/kWh)", "Precio volátil<br>(≥ 300 $/kWh)"]
        orden_mod = ["Persistencia (referencia)", "Lasso", "Aprendizaje residual"]
        fig_reg = px.bar(
            reg_f, x="Régimen", y="MAE", color="Modelo", barmode="group",
            text_auto=".0f",
            category_orders={"Régimen": orden_reg, "Modelo": orden_mod},
            labels={"MAE": "Error medio MAE ($/kWh)", "Régimen": ""},
            color_discrete_map={"Persistencia (referencia)": AZUL, "Lasso": VERDE,
                                "Aprendizaje residual": NARANJA})
        fig_reg.update_traces(textfont_size=13, textposition="outside", cliponaxis=False)
        fig_reg.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="center", x=0.5, title_text=""))
        st.plotly_chart(estilo_grafico(fig_reg, PLANTILLA, 440), use_container_width=True)
        st.caption("El precio se separa en régimen calmo (menor a 300 $/kWh) y volátil "
                   "(300 o más). En el régimen calmo la persistencia es imbatible, pero en "
                   "el régimen volátil el Lasso y el aprendizaje residual la superan: ahí "
                   "es donde el modelo aporta, y es justamente donde un error cuesta más "
                   "caro. Este matiz es el hallazgo principal del trabajo.")


# ======================================================================== #
# Vista 5: clima y precio (ONI y embalses)
# ======================================================================== #
with tab_clima:
    st.subheader("Relación entre el precio y las condiciones climáticas")
    st.caption("El precio de bolsa depende de una generación mayoritariamente hidráulica. "
               "El índice de El Niño (ONI) y el nivel de embalses ayudan a explicar sus "
               "variaciones de mediano plazo.")

    clima = serie_clima(df)
    clima_f = filtrar_periodo(clima, "clima")

    if clima_f.empty:
        st.info("No hay datos en el periodo seleccionado.")
    else:
        leyenda = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

        def grafico_doble_eje(y2_col, y2_nombre, y2_titulo, color2, dash=None):
            """Serie temporal con el precio (eje izquierdo) y una exógena (eje derecho)."""
            f = make_subplots(specs=[[{"secondary_y": True}]])
            f.add_trace(go.Scatter(x=clima_f["FECHA"], y=clima_f["precio_promedio"],
                                   name="Precio promedio ($/kWh)",
                                   line=dict(color=AZUL, width=2)), secondary_y=False)
            f.add_trace(go.Scatter(x=clima_f["FECHA"], y=clima_f[y2_col], name=y2_nombre,
                                   line=dict(color=color2, width=2, dash=dash)), secondary_y=True)
            f.update_yaxes(title_text="Precio de bolsa ($/kWh)", secondary_y=False)
            f.update_yaxes(title_text=y2_titulo, secondary_y=True)
            f.update_layout(xaxis_title="Fecha", hovermode="x unified", legend=leyenda)
            return estilo_grafico(f, PLANTILLA, 380)

        if "oni" in clima_f.columns:
            st.markdown("**Precio de bolsa frente al Índice de El Niño (ONI)**")
            st.plotly_chart(grafico_doble_eje("oni", "Índice ONI (El Niño)", "ONI", NARANJA),
                            use_container_width=True)
            st.caption("Los periodos de ONI positivo (fase de El Niño) suelen coincidir "
                       "con precios de bolsa más altos.")

        if "embalse_porcentaje_diario" in clima_f.columns:
            st.markdown("**Precio de bolsa frente al nivel de embalses**")
            st.plotly_chart(grafico_doble_eje("embalse_porcentaje_diario",
                                              "Nivel de embalses (%)", "Nivel de embalses (%)",
                                              VERDE, dash="dot"),
                            use_container_width=True)
            st.caption("Un nivel de embalses más bajo (sequía) tiende a coincidir con "
                       "precios más altos. El dato de embalses está disponible desde 2013.")

        def grafico_dispersion(x_col, x_titulo, color):
            """Dispersión de la exógena frente al precio sobre todo el histórico, con línea
            de tendencia (mínimos cuadrados) y el coeficiente de correlación."""
            d = clima[[x_col, "precio_promedio"]].dropna()  # histórico completo, no filtrado
            if len(d) < 3:
                return None, None
            x, y = d[x_col].to_numpy(float), d["precio_promedio"].to_numpy(float)
            m, b = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            r = float(np.corrcoef(x, y)[0, 1])
            f = go.Figure()
            f.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Días",
                                   marker=dict(color=color, size=5, opacity=0.35)))
            f.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Tendencia",
                                   line=dict(color=COLOR_REAL, width=2, dash="dash")))
            f.update_layout(xaxis_title=x_titulo, yaxis_title="Precio de bolsa ($/kWh)",
                            legend=leyenda)
            return estilo_grafico(f, PLANTILLA, 360), r

        st.markdown("**Asociación directa (todo el histórico)**")
        st.caption("Cada punto es un día (serie completa, sin el filtro de periodo). La "
                   "línea de tendencia resume la relación: su pendiente indica si el precio "
                   "sube o baja con la variable.")
        d1, d2 = st.columns(2)
        if "oni" in clima.columns:
            fig_s1, r1 = grafico_dispersion("oni", "Índice ONI (El Niño)", NARANJA)
            if fig_s1 is not None:
                d1.plotly_chart(fig_s1, use_container_width=True)
                rel1 = "a mayor El Niño, mayor precio" if r1 >= 0 else "relación inversa"
                d1.caption(f"Precio vs. ONI. Correlación r = {r1:.2f} ({rel1}).")
        if "embalse_porcentaje_diario" in clima.columns:
            fig_s2, r2 = grafico_dispersion("embalse_porcentaje_diario",
                                            "Nivel de embalses (%)", VERDE)
            if fig_s2 is not None:
                d2.plotly_chart(fig_s2, use_container_width=True)
                rel2 = "a menor nivel, mayor precio" if r2 <= 0 else "relación directa"
                d2.caption(f"Precio vs. embalses. Correlación r = {r2:.2f} ({rel2}).")

# ======================================================================== #
# Vista 6: pronóstico con datos cargados por el usuario
# ======================================================================== #
with tab_carga:
    # --- Banner de prediccion automatica (pipeline diario GitHub Actions) ---
    pred_auto = load_prediccion_auto()
    if pred_auto is not None:
        from datetime import date as _hoy_date  # noqa: PLC0415
        fecha_obj_auto = pd.to_datetime(pred_auto["fecha_objetivo"].iloc[0]).date()
        fecha_base_auto = pd.to_datetime(pred_auto["fecha_base"].iloc[0]).date()
        label_auto = "hoy" if fecha_obj_auto == _hoy_date.today() else fecha_larga_es(
            pd.Timestamp(fecha_obj_auto))
        st.subheader("Prediccion automatizada (pipeline diario)")
        st.caption(
            f"Generada automaticamente a partir de los precios del "
            f"{fecha_base_auto.strftime('%d %b %Y')} via GitHub Actions + API de XM."
        )
        with st.container(border=True):
            st.markdown(
                f"**Pronostico para {label_auto} "
                f"({fecha_obj_auto.strftime('%d %b %Y')})**"
            )
            prom_persist = float(pred_auto["persistencia"].mean())
            prom_lasso = float(pred_auto["lasso"].mean())
            prom_res = float(pred_auto["residual"].mean())
            ca, cb, cc = st.columns(3)
            ca.metric("Persistencia (prom. diario)", f"${prom_persist:.1f}/kWh")
            cb.metric("Lasso (prom. diario)", f"${prom_lasso:.1f}/kWh")
            cc.metric("Aprendizaje residual (prom.)", f"${prom_res:.1f}/kWh")
            _mae_lasso = round(lasso_metrics["MAE"], 1)
            _mae_persist = round(persist_metrics["MAE"], 1)
            st.caption(
                f"Margen de error historico del modelo (conjunto de prueba, dic 2022 - mar 2026): "
                f"Lasso MAE = **{_mae_lasso} \\$/kWh** | Persistencia MAE = **{_mae_persist} \\$/kWh**. "
                f"El precio real puede diferir del pronostico en ese margen aproximado."
            )
            fig_auto = go.Figure()
            fig_auto.add_vrect(x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08,
                               line_width=0, annotation_text="Franja pico",
                               annotation_position="top left")
            horas_auto = list(range(24))
            fig_auto.add_trace(go.Scatter(
                x=horas_auto, y=pred_auto["persistencia"].tolist(),
                mode="lines+markers", name="Persistencia",
                line=dict(color=AZUL, width=3), marker=dict(size=6)))
            fig_auto.add_trace(go.Scatter(
                x=horas_auto, y=pred_auto["lasso"].tolist(),
                mode="lines+markers", name="Lasso",
                line=dict(color=VERDE, width=2), marker=dict(size=5)))
            fig_auto.add_trace(go.Scatter(
                x=horas_auto, y=pred_auto["residual"].tolist(),
                mode="lines+markers", name="Aprendizaje residual",
                line=dict(color=NARANJA, width=2), marker=dict(size=5)))
            fig_auto.update_layout(
                xaxis_title="Hora del dia", yaxis_title="Precio ($/kWh)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300, margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_auto, use_container_width=True)
        st.divider()

    st.subheader("Genera un pronóstico con tus propios datos")
    st.caption(
        "Sube un archivo CSV con uno o más días base en el formato de la base del "
        "proyecto (las 24 columnas horarias h00…h23 y las variables predictoras). La "
        "aplicación ejecuta los modelos y entrega el pronóstico del día siguiente. Para "
        "una conexión automática en tiempo real desde XM, ver los trabajos futuros."
    )

    # Se quitan solo los precios objetivo (futuro); las variables de calendario del día
    # objetivo (target_anio, target_dia, …) se conocen de antemano y el modelo las usa.
    cols_precio_futuro = [f"target_h{h:02d}" for h in range(24)]
    plantilla = df.drop(columns=cols_precio_futuro, errors="ignore").head(3)
    st.download_button(
        "Descargar plantilla de ejemplo (CSV)",
        data=plantilla.to_csv(index=False).encode("utf-8-sig"),
        file_name="plantilla_dia_base.csv", mime="text/csv",
        help="Tres días base de ejemplo con todas las columnas que el modelo necesita.")

    archivo = st.file_uploader("Archivo CSV con días base", type=["csv"])
    if archivo is None:
        st.info("Sube un CSV para generar el pronóstico. Puedes usar la plantilla de "
                "ejemplo como referencia del formato.")
    else:
        try:
            nuevo = pd.read_csv(archivo)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            nuevo = None

        req_feats = sorted({f for m in models.values() for f in m["features"]})
        if nuevo is not None and nuevo.empty:
            st.error("El archivo no tiene filas.")
        elif nuevo is not None:
            faltan_horas = [c for c in HOUR_COLS if c not in nuevo.columns]
            faltan_feats = [f for f in req_feats if f not in nuevo.columns]
            if faltan_horas:
                st.error("Faltan columnas de las horas del día base (h00…h23): "
                         + ", ".join(faltan_horas[:6]) + ("…" if len(faltan_horas) > 6 else ""))
            elif faltan_feats:
                st.error(f"Faltan {len(faltan_feats)} variables que el modelo necesita. "
                         f"Usa la plantilla como referencia. Ejemplos: "
                         + ", ".join(faltan_feats[:6]) + ".")
            else:
                etiquetas = ([str(v) for v in nuevo["FECHA"]] if "FECHA" in nuevo.columns
                             else [f"Fila {i + 1}" for i in range(len(nuevo))])
                pos = st.selectbox("Día base a pronosticar", range(len(nuevo)),
                                   format_func=lambda i: etiquetas[i])
                row = nuevo.iloc[pos]

                pred_persist_c = [float(row[f"h{h:02d}"]) for h in range(24)]
                pred_ml_c = predict_ml(models, row)
                pred_res_c = predict_residual(models_res, row) if models_res is not None else None
                tcols = [f"target_h{h:02d}" for h in range(24)]
                tiene_real_c = all(c in nuevo.columns for c in tcols) and row[tcols].notna().any()
                reales_c = ([float(row[c]) for c in tcols] if tiene_real_c
                            else [np.nan] * 24)

                figc = go.Figure()
                figc.add_vrect(x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08,
                               line_width=0, annotation_text="Franja pico",
                               annotation_position="top left")
                figc.add_trace(go.Scatter(x=list(range(24)), y=pred_persist_c,
                                          mode="lines+markers", name="Persistencia diaria",
                                          line=dict(color=AZUL, width=3), marker=dict(size=7)))
                figc.add_trace(go.Scatter(x=list(range(24)), y=pred_ml_c, mode="lines+markers",
                                          name="Modelo Lasso",
                                          line=dict(color=VERDE, width=2), marker=dict(size=6)))
                if pred_res_c is not None:
                    figc.add_trace(go.Scatter(x=list(range(24)), y=pred_res_c,
                                              mode="lines+markers", name="Aprendizaje residual",
                                              line=dict(color=NARANJA, width=2), marker=dict(size=6)))
                if tiene_real_c:
                    figc.add_trace(go.Scatter(x=list(range(24)), y=reales_c, mode="lines+markers",
                                              name="Precio real observado",
                                              line=dict(color=COLOR_REAL, width=2, dash="dot"),
                                              marker=dict(size=6)))
                figc.update_layout(xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
                                   yaxis=dict(title="Precio de bolsa ($/kWh)"),
                                   hovermode="x unified",
                                   legend=dict(orientation="h", yanchor="top", y=-0.2,
                                               xanchor="center", x=0.5))
                figc = estilo_grafico(figc, PLANTILLA, 460)
                figc.update_layout(margin=dict(t=40, b=90))
                st.plotly_chart(figc, use_container_width=True)

                salida = pd.DataFrame({
                    "Hora": [f"{h:02d}:00" for h in range(24)],
                    "Persistencia ($/kWh)": [round(p, 2) for p in pred_persist_c],
                    "Lasso ($/kWh)": [round(p, 2) for p in pred_ml_c],
                })
                if pred_res_c is not None:
                    salida["Aprendizaje residual ($/kWh)"] = [round(p, 2) for p in pred_res_c]
                st.dataframe(salida, use_container_width=True, hide_index=True, height=300)
                st.download_button(
                    "Descargar pronóstico (CSV)",
                    data=salida.to_csv(index=False).encode("utf-8-sig"),
                    file_name="pronostico_datos_cargados.csv", mime="text/csv")
                st.caption("El pronóstico usa los mismos modelos del proyecto. La calidad "
                           "depende de que las variables del día base estén bien calculadas, "
                           "como en la plantilla.")


# --- Pie ---
st.divider()
st.caption(
    "Proyecto aplicado, Maestría en Ciencia de Datos y Analítica, UNAD. "
    "Modelo de referencia: persistencia diaria. Modelo de comparación: 24 "
    "regresiones Lasso por hora con variables exógenas (ONI, embalses). "
    "Fuente de datos: XM (SIN) y NOAA CPC."
)
