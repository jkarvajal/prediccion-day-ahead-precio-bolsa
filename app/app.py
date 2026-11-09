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
PREDICCION_VERIF_PATH = OUTPUTS_DIR / "prediccion_verificacion.csv"

PEAK_HOURS = [18, 19, 20, 21, 22]
TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]
HOUR_COLS = [f"h{h:02d}" for h in range(24)]
ML_MODEL_LABEL = "Lasso_horario_con_exogenas"

AZUL = "#2563eb"
VERDE = "#16a34a"
NEGRO = "#111827"
NARANJA = "#ea580c"

# --- Paleta del modo oscuro ---
# Superficie en tono pizarra (no casi-negro) con un panel elevado para que las
# graficas resalten del fondo, y series re-escaladas a la banda oscura para que
# conserven contraste sobre el panel. Validadas con la guia de visualizacion:
# sobre la superficie del panel las tres series pasan contraste >= 3:1 y
# separacion CVD (azul/verde/naranja).
DARK_PAGE = "#202838"      # fondo de la pagina
DARK_PANEL = "#283243"     # superficie elevada de las graficas
DARK_SIDEBAR = "#1a2230"   # barra lateral, un paso mas oscura para separarla
DARK_GRID = "#3b4759"      # lineas de rejilla (visibles pero recesivas)
DARK_AXIS = "#55617a"      # lineas de eje y de cero
AZUL_OSCURO = "#3987e5"
VERDE_OSCURO = "#199e70"
NARANJA_OSCURO = "#d95926"

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


@st.cache_data(ttl=3600)
def load_verificacion():
    """
    Carga la verificacion del pipeline: pronostico frente al precio real observado
    del ultimo dia con datos disponibles. Retorna None si el archivo no existe, no
    trae la columna 'real', o el dia objetivo tiene mas de 15 dias de antiguedad.
    """
    if not PREDICCION_VERIF_PATH.exists():
        return None
    df = pd.read_csv(PREDICCION_VERIF_PATH)
    if df.empty or "real" not in df.columns or "fecha_objetivo" not in df.columns:
        return None
    if df["real"].isna().all():
        return None
    fecha_obj = pd.to_datetime(df["fecha_objetivo"].iloc[0]).date()
    from datetime import date as _date  # noqa: PLC0415
    if abs((fecha_obj - _date.today()).days) > 15:
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
    es_oscuro = plantilla == "plotly_dark"
    # En oscuro el area de trazado usa el panel elevado (no transparente) para
    # separarse del fondo; el area exterior queda transparente y hereda la pagina.
    fig.update_layout(template=plantilla, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=DARK_PANEL if es_oscuro else "rgba(0,0,0,0)",
                      height=alto, margin=dict(t=40))
    if es_oscuro:
        fig.update_xaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_AXIS, linecolor=DARK_AXIS)
        fig.update_yaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_AXIS, linecolor=DARK_AXIS)
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

# En modo oscuro se usan las series re-escaladas a la banda oscura, para que
# resalten sobre el panel elevado. Se reasignan aqui (antes de los estilos y las
# pestañas) para que todas las graficas y el resaltado de la pestaña activa las usen.
if oscuro:
    AZUL, VERDE, NARANJA = AZUL_OSCURO, VERDE_OSCURO, NARANJA_OSCURO

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
_dark_css = f"""
    .stApp {{ background-color: {DARK_PAGE} !important; }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    h1, h2, h3, h4, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color: #fafafa !important; }}
    [data-testid="stSidebar"] {{ background-color: {DARK_SIDEBAR} !important; }}
    [data-testid="stSidebar"] * {{ color: #e5e7eb !important; }}
    [data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0) !important; }}
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

    larga_full = serie_horaria_larga(df)
    larga_full["año"] = pd.to_datetime(larga_full["FECHA"]).dt.year
    años_disp = sorted(larga_full["año"].unique().tolist())

    # --- Filtro de período (compartido por todas las sub-pestañas) ---
    _fc1, _fc2 = st.columns([1, 2])
    with _fc1:
        modo_periodo = st.radio(
            "Período de análisis:",
            ["Todo el período", "Por año", "Rango de fechas"],
            horizontal=False, key="modo_periodo_expl",
        )
    with _fc2:
        if modo_periodo == "Por año":
            años_sel = st.multiselect(
                "Selecciona uno o más años:",
                años_disp, default=años_disp, key="años_expl",
            )
            if not años_sel:
                años_sel = años_disp
            larga = larga_full[larga_full["año"].isin(años_sel)].copy()
        elif modo_periodo == "Rango de fechas":
            fecha_min = pd.to_datetime(larga_full["FECHA"]).min().date()
            fecha_max = pd.to_datetime(larga_full["FECHA"]).max().date()
            rango = st.date_input(
                "Desde / Hasta:",
                value=(fecha_min, fecha_max),
                min_value=fecha_min, max_value=fecha_max,
                key="rango_fechas_expl",
            )
            if isinstance(rango, (list, tuple)) and len(rango) == 2:
                f_ini, f_fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
            else:
                f_ini, f_fin = pd.Timestamp(fecha_min), pd.Timestamp(fecha_max)
            larga = larga_full[
                (pd.to_datetime(larga_full["FECHA"]) >= f_ini) &
                (pd.to_datetime(larga_full["FECHA"]) <= f_fin)
            ].copy()
        else:
            larga = larga_full.copy()

    n_dias = larga["FECHA"].nunique()
    st.caption(f"{n_dias:,} días en el período seleccionado.")

    # larga_agg: sin sampleo, para todas las agregaciones
    larga_agg = larga.copy()
    larga_agg["mes"] = pd.to_datetime(larga_agg["FECHA"]).dt.month
    if len(larga) > 60000:
        larga = larga.sample(60000, random_state=42)

    _azul_r = int(AZUL[1:3], 16)
    _azul_g = int(AZUL[3:5], 16)
    _azul_b = int(AZUL[5:7], 16)
    _verd_r = int(VERDE[1:3], 16)
    _verd_g = int(VERDE[3:5], 16)
    _verd_b = int(VERDE[5:7], 16)
    _narj_r = int(NARANJA[1:3], 16)
    _narj_g = int(NARANJA[3:5], 16)
    _narj_b = int(NARANJA[5:7], 16)

    # Sub-pestañas de análisis
    _stab_hora, _stab_tipo, _stab_mes, _stab_año = st.tabs(
        ["Horario", "Tipo de día", "Mes", "Año"]
    )

    # ------------------------------------------------------------------ #
    # Sub-pestaña: Horario
    # ------------------------------------------------------------------ #
    with _stab_hora:
        _stats_h = (
            larga_agg.groupby("hora")["precio"]
            .agg(
                q1=lambda x: float(x.quantile(0.25)),
                mediana="median",
                q3=lambda x: float(x.quantile(0.75)),
                minimo="min",
                maximo="max",
                media="mean",
                std="std",
            )
            .reset_index()
        )
        _horas = _stats_h["hora"].tolist()
        _q1  = _stats_h["q1"].tolist()
        _q3  = _stats_h["q3"].tolist()
        _med = _stats_h["mediana"].tolist()

        # --- 1. Banda IQR ---
        st.markdown("**1. Rango intercuartílico (Q1, mediana, Q3)**")
        st.caption("Muestra el 50 % central del precio por hora. La banda sombreada va de Q1 a Q3.")
        fig_iqr = go.Figure()
        fig_iqr.add_trace(go.Scatter(
            x=_horas + _horas[::-1], y=_q3 + _q1[::-1],
            fill="toself", fillcolor=f"rgba({_azul_r},{_azul_g},{_azul_b},0.12)",
            line=dict(width=0), name="Banda Q1-Q3", hoverinfo="skip",
        ))
        fig_iqr.add_trace(go.Scatter(
            x=_horas, y=_q3, mode="lines",
            line=dict(color=NARANJA, width=1.8, dash="dot"), name="Q3 (p75)",
            hovertemplate="Hora %{x}h — Q3: $%{y:.1f}<extra></extra>",
        ))
        fig_iqr.add_trace(go.Scatter(
            x=_horas, y=_med, mode="lines+markers",
            line=dict(color=AZUL, width=2.5), marker=dict(size=5), name="Mediana",
            hovertemplate="Hora %{x}h — Mediana: $%{y:.1f}<extra></extra>",
        ))
        fig_iqr.add_trace(go.Scatter(
            x=_horas, y=_q1, mode="lines",
            line=dict(color=VERDE, width=1.8, dash="dot"), name="Q1 (p25)",
            hovertemplate="Hora %{x}h — Q1: $%{y:.1f}<extra></extra>",
        ))
        fig_iqr.update_layout(
            xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_iqr, PLANTILLA, 420), use_container_width=True)

        # --- 2. Rango completo (mín/máx reales) ---
        st.markdown("**2. Rango completo: mínimo, máximo y cuartiles**")
        st.caption("Bigotes = mínimo y máximo histórico real (sin recorte al 1.5×IQR).")
        fig_full = go.Figure()
        for _, _r in _stats_h.iterrows():
            fig_full.add_trace(go.Box(
                x=[int(_r["hora"])], q1=[_r["q1"]], median=[_r["mediana"]],
                q3=[_r["q3"]], lowerfence=[_r["minimo"]], upperfence=[_r["maximo"]],
                name="", showlegend=False, whiskerwidth=0.6,
                marker_color=AZUL, line_color=AZUL,
                fillcolor=f"rgba({_azul_r},{_azul_g},{_azul_b},0.25)",
            ))
        fig_full.update_layout(
            xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
            yaxis_title="Precio de bolsa ($/kWh)",
        )
        st.plotly_chart(estilo_grafico(fig_full, PLANTILLA, 420), use_container_width=True)

        # --- 3. Violín ---
        st.markdown("**3. Distribución por densidad (violín)**")
        st.caption("La forma del violín revela cómo se concentra el precio en cada hora; más ancho = más frecuente.")
        fig_vio = px.violin(
            larga, x="hora", y="precio", points=False,
            labels={"hora": "Hora del día", "precio": "Precio de bolsa ($/kWh)"},
            color_discrete_sequence=[AZUL],
        )
        fig_vio.update_traces(meanline_visible=True)
        st.plotly_chart(estilo_grafico(fig_vio, PLANTILLA, 420), use_container_width=True)

        # --- 4. Percentiles múltiples (P10 a P90) ---
        st.markdown("**4. Percentiles múltiples (P10, P25, P50, P75, P90)**")
        st.caption("Cinco umbrales de precio: el área entre P10 y P90 contiene el 80 % de las observaciones.")
        _pcts = (
            larga_agg.groupby("hora")["precio"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .unstack(level=1)
        )
        _pcts.columns = ["P10", "P25", "P50", "P75", "P90"]
        _pcts = _pcts.reset_index()
        _pct_horas = _pcts["hora"].tolist()
        _colores_pct = [AZUL, VERDE, NARANJA, VERDE, AZUL]
        _dash_pct    = ["dash", "dot", "solid", "dot", "dash"]
        _width_pct   = [1.2, 1.5, 2.5, 1.5, 1.2]
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=_pct_horas + _pct_horas[::-1],
            y=_pcts["P90"].tolist() + _pcts["P10"].tolist()[::-1],
            fill="toself", fillcolor=f"rgba({_azul_r},{_azul_g},{_azul_b},0.08)",
            line=dict(width=0), name="Banda P10-P90", hoverinfo="skip",
        ))
        for _col, _c, _d, _w in zip(
            ["P10", "P25", "P50", "P75", "P90"],
            _colores_pct, _dash_pct, _width_pct
        ):
            fig_pct.add_trace(go.Scatter(
                x=_pct_horas, y=_pcts[_col].tolist(),
                mode="lines", line=dict(color=_c, width=_w, dash=_d),
                name=_col,
                hovertemplate=f"Hora %{{x}}h — {_col}: $%{{y:.1f}}<extra></extra>",
            ))
        fig_pct.update_layout(
            xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="v", yanchor="bottom", y=0, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_pct, PLANTILLA, 420), use_container_width=True)

        # --- 5. Mapa de calor hora × mes ---
        st.markdown("**5. Mapa de calor: precio promedio por hora y mes**")
        st.caption("Cada celda es el precio medio para esa combinación de hora del día y mes del año.")
        _heat = (
            larga_agg.groupby(["hora", "mes"])["precio"]
            .mean()
            .reset_index()
        )
        _pivot_hm = _heat.pivot(index="hora", columns="mes", values="precio")
        _meses_abrev = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                        7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
        _pivot_hm.columns = [_meses_abrev.get(c, str(c)) for c in _pivot_hm.columns]
        fig_heat = px.imshow(
            _pivot_hm,
            labels=dict(x="Mes", y="Hora del día", color="$/kWh"),
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_heat.update_yaxes(tickmode="linear", dtick=2)
        st.plotly_chart(estilo_grafico(fig_heat, PLANTILLA, 420), use_container_width=True)

        # --- 6. Media ± desviación estándar ---
        st.markdown("**6. Media y desviación estándar**")
        st.caption("La línea central es el precio medio; la banda sombreada abarca ±1 desviación estándar (~68 % de los datos).")
        _media = _stats_h["media"].tolist()
        _std   = _stats_h["std"].tolist()
        _sup   = [m + s for m, s in zip(_media, _std)]
        _inf   = [max(0, m - s) for m, s in zip(_media, _std)]
        fig_std = go.Figure()
        fig_std.add_trace(go.Scatter(
            x=_horas + _horas[::-1], y=_sup + _inf[::-1],
            fill="toself", fillcolor=f"rgba({_narj_r},{_narj_g},{_narj_b},0.15)",
            line=dict(width=0), name="±1 Desv. est.", hoverinfo="skip",
        ))
        fig_std.add_trace(go.Scatter(
            x=_horas, y=_media, mode="lines+markers",
            line=dict(color=NARANJA, width=2.5), marker=dict(size=5),
            name="Media",
            hovertemplate="Hora %{x}h — Media: $%{y:.1f}<extra></extra>",
        ))
        fig_std.update_layout(
            xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="v", yanchor="bottom", y=0, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_std, PLANTILLA, 420), use_container_width=True)

        # --- 7. Histograma de precio por hora ---
        st.markdown("**7. Histograma de frecuencias por hora**")
        st.caption("Distribución de precios para cada hora del día. Las barras muestran con qué frecuencia el precio cayó en cada rango.")
        _hora_hist = st.select_slider(
            "Selecciona una hora para ver su histograma:",
            options=sorted(larga_agg["hora"].unique().tolist()),
            value=12,
            key="hora_hist_sel",
        )
        _datos_hora = larga_agg[larga_agg["hora"] == _hora_hist]["precio"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=_datos_hora,
            nbinsx=40,
            marker_color=AZUL,
            marker_line=dict(color="white", width=0.5),
            opacity=0.85,
            name=f"Hora {_hora_hist}h",
            hovertemplate="$%{x:.0f}/kWh — %{y} días<extra></extra>",
        ))
        _p25 = float(_datos_hora.quantile(0.25))
        _p50 = float(_datos_hora.quantile(0.50))
        _p75 = float(_datos_hora.quantile(0.75))
        for _val, _lbl, _col in [(_p25, "Q1", VERDE), (_p50, "Mediana", AZUL), (_p75, "Q3", NARANJA)]:
            fig_hist.add_vline(
                x=_val, line_dash="dash", line_color=_col, line_width=1.8,
                annotation_text=_lbl, annotation_position="top",
                annotation_font_color=_col,
            )
        fig_hist.update_layout(
            xaxis_title="Precio de bolsa ($/kWh)",
            yaxis_title="Número de días",
            showlegend=False,
            bargap=0.05,
        )
        st.plotly_chart(estilo_grafico(fig_hist, PLANTILLA, 400), use_container_width=True)

    # ------------------------------------------------------------------ #
    # Sub-pestaña: Tipo de día
    # ------------------------------------------------------------------ #
    with _stab_tipo:
        if "base_tipo_dia" not in larga_agg.columns:
            st.info("No hay datos de tipo de día disponibles.")
        else:
            larga_agg["Tipo de día"] = larga_agg["base_tipo_dia"].map(TIPO_DIA_ES).fillna(larga_agg["base_tipo_dia"])
            _orden_tipo = ["Hábil", "Sábado", "Domingo", "Festivo"]
            _tipos_pres = [t for t in _orden_tipo if t in larga_agg["Tipo de día"].unique()]
            _stats_tipo = (
                larga_agg.groupby("Tipo de día")["precio"]
                .agg(
                    q1=lambda x: float(x.quantile(0.25)),
                    mediana="median",
                    q3=lambda x: float(x.quantile(0.75)),
                    minimo="min",
                    maximo="max",
                    media="mean",
                    std="std",
                )
                .reindex(_tipos_pres)
                .reset_index()
            )
            _colores_tipo = {
                "Hábil": AZUL, "Sábado": VERDE, "Domingo": NARANJA, "Festivo": "#8b5cf6"
            }

            # --- 1. Rango completo (boxplot) ---
            st.markdown("**1. Rango completo por tipo de día**")
            st.caption("Bigotes = mínimo y máximo histórico real. La caja abarca Q1–Q3 y la línea interior es la mediana.")
            fig_td_box = go.Figure()
            for _, _row in _stats_tipo.iterrows():
                _c = _colores_tipo.get(_row["Tipo de día"], AZUL)
                _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
                fig_td_box.add_trace(go.Box(
                    x=[_row["Tipo de día"]],
                    q1=[_row["q1"]], median=[_row["mediana"]], q3=[_row["q3"]],
                    lowerfence=[_row["minimo"]], upperfence=[_row["maximo"]],
                    name=_row["Tipo de día"], showlegend=False, whiskerwidth=0.6,
                    marker_color=_c, line_color=_c,
                    fillcolor=f"rgba({_r},{_g},{_b},0.25)",
                ))
            fig_td_box.update_layout(xaxis_title="Tipo de día", yaxis_title="Precio de bolsa ($/kWh)")
            st.plotly_chart(estilo_grafico(fig_td_box, PLANTILLA, 420), use_container_width=True)

            # --- 2. Violín por tipo de día ---
            st.markdown("**2. Distribución por densidad (violín)**")
            st.caption("Muestra la forma completa de la distribución del precio para cada tipo de día.")
            _larga_tipo = larga[larga["FECHA"].isin(larga_agg["FECHA"])].copy() if "FECHA" in larga.columns else larga.copy()
            _larga_tipo["Tipo de día"] = _larga_tipo["base_tipo_dia"].map(TIPO_DIA_ES).fillna(_larga_tipo["base_tipo_dia"]) if "base_tipo_dia" in _larga_tipo.columns else None
            if _larga_tipo["Tipo de día"].notna().any():
                _larga_tipo = _larga_tipo[_larga_tipo["Tipo de día"].isin(_tipos_pres)].copy()
                _larga_tipo["Tipo de día"] = pd.Categorical(_larga_tipo["Tipo de día"], categories=_tipos_pres, ordered=True)
                fig_td_vio = px.violin(
                    _larga_tipo.sort_values("Tipo de día"), x="Tipo de día", y="precio", points=False,
                    color="Tipo de día", color_discrete_map=_colores_tipo,
                    labels={"precio": "Precio de bolsa ($/kWh)"},
                    category_orders={"Tipo de día": _tipos_pres},
                )
                fig_td_vio.update_traces(meanline_visible=True)
                fig_td_vio.update_layout(showlegend=False)
                st.plotly_chart(estilo_grafico(fig_td_vio, PLANTILLA, 420), use_container_width=True)

            # --- 3. Media y mediana por tipo de día (barras) ---
            st.markdown("**3. Media y mediana por tipo de día**")
            st.caption("Comparación directa entre la media aritmética (sensible a extremos) y la mediana (valor central).")
            fig_td_bar = go.Figure()
            fig_td_bar.add_trace(go.Bar(
                x=_stats_tipo["Tipo de día"], y=_stats_tipo["media"],
                name="Media", marker_color=AZUL, opacity=0.85,
                hovertemplate="%{x} — Media: $%{y:.1f}<extra></extra>",
            ))
            fig_td_bar.add_trace(go.Bar(
                x=_stats_tipo["Tipo de día"], y=_stats_tipo["mediana"],
                name="Mediana", marker_color=VERDE, opacity=0.85,
                hovertemplate="%{x} — Mediana: $%{y:.1f}<extra></extra>",
            ))
            fig_td_bar.update_layout(
                barmode="group",
                xaxis_title="Tipo de día", yaxis_title="Precio de bolsa ($/kWh)",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            )
            st.plotly_chart(estilo_grafico(fig_td_bar, PLANTILLA, 400), use_container_width=True)

            # --- 4. Perfil horario por tipo de día ---
            st.markdown("**4. Perfil horario por tipo de día (mediana)**")
            st.caption("Mediana del precio a lo largo del día para cada tipo de jornada.")
            _perfil_td = (
                larga_agg.groupby(["Tipo de día", "hora"])["precio"]
                .median().reset_index()
            )
            fig_td_perf = go.Figure()
            for _td in _tipos_pres:
                _sub = _perfil_td[_perfil_td["Tipo de día"] == _td]
                _c = _colores_tipo.get(_td, AZUL)
                fig_td_perf.add_trace(go.Scatter(
                    x=_sub["hora"], y=_sub["precio"],
                    mode="lines+markers", name=_td,
                    line=dict(color=_c, width=2), marker=dict(size=4),
                    hovertemplate=f"{_td} — Hora %{{x}}h: $%{{y:.1f}}<extra></extra>",
                ))
            fig_td_perf.update_layout(
                xaxis=dict(title="Hora del día", tickmode="linear", dtick=2),
                yaxis_title="Mediana del precio ($/kWh)",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                hovermode="x unified",
            )
            st.plotly_chart(estilo_grafico(fig_td_perf, PLANTILLA, 420), use_container_width=True)

    # ------------------------------------------------------------------ #
    # Sub-pestaña: Mes
    # ------------------------------------------------------------------ #
    with _stab_mes:
        _meses_nombre = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                         7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
        _stats_mes = (
            larga_agg.groupby("mes")["precio"]
            .agg(
                q1=lambda x: float(x.quantile(0.25)),
                mediana="median",
                q3=lambda x: float(x.quantile(0.75)),
                minimo="min",
                maximo="max",
                media="mean",
                std="std",
            )
            .reset_index()
        )
        _stats_mes["mes_nombre"] = _stats_mes["mes"].map(_meses_nombre)

        # --- 1. Rango completo por mes (boxplot) ---
        st.markdown("**1. Rango completo por mes**")
        st.caption("Distribución del precio para cada mes. Bigotes = mínimo y máximo histórico real.")
        fig_mes_box = go.Figure()
        for _, _row in _stats_mes.iterrows():
            fig_mes_box.add_trace(go.Box(
                x=[_row["mes_nombre"]],
                q1=[_row["q1"]], median=[_row["mediana"]], q3=[_row["q3"]],
                lowerfence=[_row["minimo"]], upperfence=[_row["maximo"]],
                name="", showlegend=False, whiskerwidth=0.6,
                marker_color=AZUL, line_color=AZUL,
                fillcolor=f"rgba({_azul_r},{_azul_g},{_azul_b},0.25)",
            ))
        fig_mes_box.update_layout(
            xaxis=dict(title="Mes", categoryorder="array",
                       categoryarray=list(_meses_nombre.values())),
            yaxis_title="Precio de bolsa ($/kWh)",
        )
        st.plotly_chart(estilo_grafico(fig_mes_box, PLANTILLA, 420), use_container_width=True)

        # --- 2. Banda IQR mensual ---
        st.markdown("**2. Rango intercuartílico mensual (Q1, mediana, Q3)**")
        st.caption("Vista del 50 % central del precio por mes del año.")
        _meses_ord = list(_meses_nombre.values())
        _sm = _stats_mes.set_index("mes_nombre").reindex(_meses_ord).reset_index()
        _sm_q1  = _sm["q1"].tolist()
        _sm_q3  = _sm["q3"].tolist()
        _sm_med = _sm["mediana"].tolist()
        _idx_mes = list(range(len(_meses_ord)))
        fig_mes_iqr = go.Figure()
        fig_mes_iqr.add_trace(go.Scatter(
            x=_meses_ord + _meses_ord[::-1], y=_sm_q3 + _sm_q1[::-1],
            fill="toself", fillcolor=f"rgba({_azul_r},{_azul_g},{_azul_b},0.12)",
            line=dict(width=0), name="Banda Q1-Q3", hoverinfo="skip",
        ))
        fig_mes_iqr.add_trace(go.Scatter(
            x=_meses_ord, y=_sm_q3, mode="lines+markers",
            line=dict(color=NARANJA, width=1.8, dash="dot"), marker=dict(size=4),
            name="Q3 (p75)", hovertemplate="Mes %{x} — Q3: $%{y:.1f}<extra></extra>",
        ))
        fig_mes_iqr.add_trace(go.Scatter(
            x=_meses_ord, y=_sm_med, mode="lines+markers",
            line=dict(color=AZUL, width=2.5), marker=dict(size=5),
            name="Mediana", hovertemplate="Mes %{x} — Mediana: $%{y:.1f}<extra></extra>",
        ))
        fig_mes_iqr.add_trace(go.Scatter(
            x=_meses_ord, y=_sm_q1, mode="lines+markers",
            line=dict(color=VERDE, width=1.8, dash="dot"), marker=dict(size=4),
            name="Q1 (p25)", hovertemplate="Mes %{x} — Q1: $%{y:.1f}<extra></extra>",
        ))
        fig_mes_iqr.update_layout(
            xaxis=dict(title="Mes", categoryorder="array", categoryarray=_meses_ord),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_mes_iqr, PLANTILLA, 420), use_container_width=True)

        # --- 3. Mapa de calor mes × hora ---
        st.markdown("**3. Mapa de calor: precio promedio por mes y hora**")
        st.caption("Cada celda es el precio medio para esa combinación de mes y hora del día.")
        _heat_mh = (
            larga_agg.groupby(["mes", "hora"])["precio"].mean().reset_index()
        )
        _pivot_mh = _heat_mh.pivot(index="mes", columns="hora", values="precio")
        _pivot_mh.index = [_meses_nombre.get(i, str(i)) for i in _pivot_mh.index]
        fig_heat_mh = px.imshow(
            _pivot_mh,
            labels=dict(x="Hora del día", y="Mes", color="$/kWh"),
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_heat_mh.update_xaxes(tickmode="linear", dtick=2)
        st.plotly_chart(estilo_grafico(fig_heat_mh, PLANTILLA, 420), use_container_width=True)

        # --- 4. Histograma por mes ---
        st.markdown("**4. Histograma por mes**")
        st.caption("Distribución del precio para el mes seleccionado, con líneas de referencia Q1, mediana y Q3.")
        _mes_hist = st.select_slider(
            "Selecciona un mes:",
            options=list(range(1, 13)),
            format_func=lambda m: _meses_nombre[m],
            value=6,
            key="mes_hist_sel",
        )
        _datos_mes = larga_agg[larga_agg["mes"] == _mes_hist]["precio"]
        fig_hist_mes = go.Figure()
        fig_hist_mes.add_trace(go.Histogram(
            x=_datos_mes, nbinsx=40,
            marker_color=VERDE, marker_line=dict(color="white", width=0.5),
            opacity=0.85, showlegend=False,
            hovertemplate="$%{x:.0f}/kWh — %{y} días<extra></extra>",
        ))
        for _val, _lbl, _col in [
            (float(_datos_mes.quantile(0.25)), "Q1", VERDE),
            (float(_datos_mes.quantile(0.50)), "Mediana", AZUL),
            (float(_datos_mes.quantile(0.75)), "Q3", NARANJA),
        ]:
            fig_hist_mes.add_vline(
                x=_val, line_dash="dash", line_color=_col, line_width=1.8,
                annotation_text=_lbl, annotation_position="top",
                annotation_font_color=_col,
            )
        fig_hist_mes.update_layout(
            xaxis_title="Precio de bolsa ($/kWh)",
            yaxis_title="Número de días",
            bargap=0.05,
        )
        st.plotly_chart(estilo_grafico(fig_hist_mes, PLANTILLA, 400), use_container_width=True)

    # ------------------------------------------------------------------ #
    # Sub-pestaña: Año
    # ------------------------------------------------------------------ #
    with _stab_año:
        _stats_año = (
            larga_agg.groupby("año")["precio"]
            .agg(
                q1=lambda x: float(x.quantile(0.25)),
                mediana="median",
                q3=lambda x: float(x.quantile(0.75)),
                minimo="min",
                maximo="max",
                media="mean",
                std="std",
            )
            .reset_index()
        )
        _años_ord = _stats_año["año"].tolist()

        # --- 1. Rango completo por año (boxplot) ---
        st.markdown("**1. Rango completo por año**")
        st.caption("Distribución anual del precio. Bigotes = mínimo y máximo histórico real.")
        fig_año_box = go.Figure()
        for _, _row in _stats_año.iterrows():
            fig_año_box.add_trace(go.Box(
                x=[int(_row["año"])],
                q1=[_row["q1"]], median=[_row["mediana"]], q3=[_row["q3"]],
                lowerfence=[_row["minimo"]], upperfence=[_row["maximo"]],
                name="", showlegend=False, whiskerwidth=0.6,
                marker_color=NARANJA, line_color=NARANJA,
                fillcolor=f"rgba({_narj_r},{_narj_g},{_narj_b},0.25)",
            ))
        fig_año_box.update_layout(
            xaxis=dict(title="Año", tickmode="linear", dtick=1),
            yaxis_title="Precio de bolsa ($/kWh)",
        )
        st.plotly_chart(estilo_grafico(fig_año_box, PLANTILLA, 420), use_container_width=True)

        # --- 2. Evolución anual: media y mediana ---
        st.markdown("**2. Evolución anual de media y mediana**")
        st.caption("Tendencia del nivel de precios año a año. La diferencia entre media y mediana indica la presencia de valores extremos.")
        fig_año_line = go.Figure()
        fig_año_line.add_trace(go.Scatter(
            x=_años_ord, y=_stats_año["media"].tolist(),
            mode="lines+markers", name="Media",
            line=dict(color=AZUL, width=2), marker=dict(size=6),
            hovertemplate="Año %{x} — Media: $%{y:.1f}<extra></extra>",
        ))
        fig_año_line.add_trace(go.Scatter(
            x=_años_ord, y=_stats_año["mediana"].tolist(),
            mode="lines+markers", name="Mediana",
            line=dict(color=VERDE, width=2, dash="dot"), marker=dict(size=6),
            hovertemplate="Año %{x} — Mediana: $%{y:.1f}<extra></extra>",
        ))
        fig_año_line.update_layout(
            xaxis=dict(title="Año", tickmode="linear", dtick=1),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_año_line, PLANTILLA, 400), use_container_width=True)

        # --- 3. Banda IQR anual ---
        st.markdown("**3. Rango intercuartílico anual (Q1, mediana, Q3)**")
        st.caption("Evolución del 50 % central del precio a lo largo de los años.")
        _sa_q1  = _stats_año["q1"].tolist()
        _sa_q3  = _stats_año["q3"].tolist()
        _sa_med = _stats_año["mediana"].tolist()
        fig_año_iqr = go.Figure()
        fig_año_iqr.add_trace(go.Scatter(
            x=_años_ord + _años_ord[::-1], y=_sa_q3 + _sa_q1[::-1],
            fill="toself", fillcolor=f"rgba({_narj_r},{_narj_g},{_narj_b},0.12)",
            line=dict(width=0), name="Banda Q1-Q3", hoverinfo="skip",
        ))
        fig_año_iqr.add_trace(go.Scatter(
            x=_años_ord, y=_sa_q3, mode="lines+markers",
            line=dict(color=NARANJA, width=1.8, dash="dot"), marker=dict(size=5),
            name="Q3 (p75)", hovertemplate="Año %{x} — Q3: $%{y:.1f}<extra></extra>",
        ))
        fig_año_iqr.add_trace(go.Scatter(
            x=_años_ord, y=_sa_med, mode="lines+markers",
            line=dict(color=AZUL, width=2.5), marker=dict(size=5),
            name="Mediana", hovertemplate="Año %{x} — Mediana: $%{y:.1f}<extra></extra>",
        ))
        fig_año_iqr.add_trace(go.Scatter(
            x=_años_ord, y=_sa_q1, mode="lines+markers",
            line=dict(color=VERDE, width=1.8, dash="dot"), marker=dict(size=5),
            name="Q1 (p25)", hovertemplate="Año %{x} — Q1: $%{y:.1f}<extra></extra>",
        ))
        fig_año_iqr.update_layout(
            xaxis=dict(title="Año", tickmode="linear", dtick=1),
            yaxis_title="Precio de bolsa ($/kWh)",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        st.plotly_chart(estilo_grafico(fig_año_iqr, PLANTILLA, 420), use_container_width=True)

        # --- 4. Perfil horario por año (mapa de calor) ---
        st.markdown("**4. Mapa de calor: precio promedio por año y hora**")
        st.caption("Cada celda muestra el precio medio para esa combinación de año y hora del día.")
        _heat_ah = (
            larga_agg.groupby(["año", "hora"])["precio"].mean().reset_index()
        )
        _pivot_ah = _heat_ah.pivot(index="año", columns="hora", values="precio")
        fig_heat_ah = px.imshow(
            _pivot_ah,
            labels=dict(x="Hora del día", y="Año", color="$/kWh"),
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_heat_ah.update_xaxes(tickmode="linear", dtick=2)
        fig_heat_ah.update_yaxes(tickmode="linear", dtick=1)
        st.plotly_chart(estilo_grafico(fig_heat_ah, PLANTILLA, 420), use_container_width=True)


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

        if "oni" in clima_f.columns and "embalse_porcentaje_diario" in clima_f.columns:
            st.markdown("**Nivel de embalses frente al Índice de El Niño (ONI)**")
            f_emb_oni = make_subplots(specs=[[{"secondary_y": True}]])
            f_emb_oni.add_trace(
                go.Scatter(x=clima_f["FECHA"], y=clima_f["embalse_porcentaje_diario"],
                           name="Nivel de embalses",
                           line=dict(color=VERDE, width=2, dash="dot")), secondary_y=False)
            f_emb_oni.add_trace(
                go.Scatter(x=clima_f["FECHA"], y=clima_f["oni"],
                           name="Índice ONI (El Niño)",
                           line=dict(color=NARANJA, width=2)), secondary_y=True)
            # Linea de referencia ONI=0 (separa El Nino de La Nina)
            f_emb_oni.add_trace(
                go.Scatter(x=[clima_f["FECHA"].min(), clima_f["FECHA"].max()],
                           y=[0, 0], mode="lines", showlegend=False,
                           line=dict(color=COLOR_REAL, width=1, dash="dot")),
                secondary_y=True)
            f_emb_oni.update_yaxes(title_text="Nivel de embalses", secondary_y=False)
            f_emb_oni.update_yaxes(title_text="ONI", secondary_y=True)
            f_emb_oni.update_layout(xaxis_title="Fecha", hovermode="x unified", legend=leyenda)
            st.plotly_chart(estilo_grafico(f_emb_oni, PLANTILLA, 380), use_container_width=True)
            st.caption("La relación inversa entre embalses y ONI refleja el efecto de "
                       "El Niño sobre la hidrología del SIN: episodios de ONI positivo "
                       "suelen reducir las precipitaciones y con ello el nivel de embalses. "
                       "La línea punteada marca ONI = 0 (frontera El Niño / La Niña).")

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
            )
            fig_auto = estilo_grafico(fig_auto, PLANTILLA, 300)
            fig_auto.update_layout(margin=dict(t=30, b=40))
            st.plotly_chart(fig_auto, use_container_width=True)
        st.divider()

    # --- Verificacion: pronostico frente al precio real observado ---
    # El banner de arriba es day-ahead (su dia objetivo aun no tiene precio real).
    # Este bloque muestra el ultimo dia que SI tiene precio real, para comparar el
    # pronostico del modelo contra lo que realmente ocurrio.
    verif = load_verificacion()
    if verif is not None:
        v_obj = pd.to_datetime(verif["fecha_objetivo"].iloc[0]).date()
        st.subheader("Verificación: pronóstico frente al precio real")
        st.caption(
            f"Último día con precio real disponible: {fecha_larga_es(v_obj)}. Compara lo "
            "que el modelo habría pronosticado (con los datos del día anterior) contra lo "
            "que efectivamente ocurrió."
        )
        with st.container(border=True):
            horas_v = verif["hora"].tolist()
            reales_v = verif["real"].to_numpy(dtype=float)

            def _mae_v(col):
                p = verif[col].to_numpy(dtype=float)
                m = ~(np.isnan(p) | np.isnan(reales_v))
                return float(np.mean(np.abs(p[m] - reales_v[m]))) if m.any() else float("nan")

            mv1, mv2, mv3 = st.columns(3)
            mv1.metric("Error Persistencia (MAE)", f"${_mae_v('persistencia'):.1f}/kWh")
            mv2.metric("Error Lasso (MAE)", f"${_mae_v('lasso'):.1f}/kWh")
            mv3.metric("Error Ap. residual (MAE)", f"${_mae_v('residual'):.1f}/kWh")
            fig_v = go.Figure()
            fig_v.add_vrect(x0=17.5, x1=22.5, fillcolor="crimson", opacity=0.08,
                            line_width=0, annotation_text="Franja pico",
                            annotation_position="top left")
            fig_v.add_trace(go.Scatter(x=horas_v, y=verif["persistencia"].tolist(),
                                       mode="lines+markers", name="Persistencia",
                                       line=dict(color=AZUL, width=3), marker=dict(size=6)))
            fig_v.add_trace(go.Scatter(x=horas_v, y=verif["lasso"].tolist(),
                                       mode="lines+markers", name="Lasso",
                                       line=dict(color=VERDE, width=2), marker=dict(size=5)))
            fig_v.add_trace(go.Scatter(x=horas_v, y=verif["residual"].tolist(),
                                       mode="lines+markers", name="Aprendizaje residual",
                                       line=dict(color=NARANJA, width=2), marker=dict(size=5)))
            fig_v.add_trace(go.Scatter(x=horas_v, y=verif["real"].tolist(),
                                       mode="lines+markers", name="Precio real observado",
                                       line=dict(color=COLOR_REAL, width=2, dash="dot"),
                                       marker=dict(size=6)))
            fig_v.update_layout(
                xaxis_title="Hora del día", yaxis_title="Precio ($/kWh)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_v = estilo_grafico(fig_v, PLANTILLA, 320)
            fig_v.update_layout(margin=dict(t=30, b=40))
            st.plotly_chart(fig_v, use_container_width=True)
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
