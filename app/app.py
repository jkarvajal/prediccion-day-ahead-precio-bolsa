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
METADATA_PATH = MODELS_DIR / "modelo_final_metadata.json"
FEATURES_PATH = PROCESSED_DIR / "features_day_ahead.csv"
METRICS_BY_HOUR_PATH = OUTPUTS_DIR / "hourly_dayahead_metrics_by_hour.csv"
BASELINE_RESULTS_PATH = OUTPUTS_DIR / "baseline_global_results.csv"

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


def predict_ml(models, row):
    """Predicción del modelo Lasso para las 24 horas de un día base."""
    preds = []
    for h in range(24):
        m = models[h]
        features = [f for f in m["features"] if f in row.index]
        X = row[features].to_frame().T
        preds.append(float(m["pipeline"].predict(X)[0]))
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
    </style>
    """,
    unsafe_allow_html=True,
)

if oscuro:
    st.markdown(
        """
        <style>
        .stApp { background-color: #1c212c !important; }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
        h1, h2, h3, h4, [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: #fafafa !important; }
        [data-testid="stSidebar"] { background-color: #242a38 !important; }
        [data-testid="stSidebar"] * { color: #e5e7eb !important; }
        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Encabezado ---
st.title("Precio de bolsa de energía: histórico, predicción y exploración")
st.markdown(
    "Mercado eléctrico colombiano (SIN). Elige una pestaña: el **histórico** del "
    "precio, la **predicción** del día siguiente, la **exploración** de patrones o la "
    "relación con el **clima**."
)

# --- Barra lateral: contexto de los modelos ---
with st.sidebar:
    st.header("¿Qué modelos se comparan?")
    st.markdown(
        "**Persistencia diaria** (referencia): asume que mañana, a cada hora, el "
        "precio será el de hoy a esa misma hora. Es simple y difícil de superar.\n\n"
        "**Modelo Lasso** (comparación): un modelo de aprendizaje automático, uno por "
        "cada hora del día, con variables de clima y embalses."
    )
    comp = pd.DataFrame([
        {"Modelo": "Persistencia diaria", "MAE": round(persist_metrics["MAE"], 1),
         "RMSE": round(persist_metrics["RMSE"], 1), "R2": round(persist_metrics["R2"], 3)},
        {"Modelo": "Lasso", "MAE": round(lasso_metrics["MAE"], 1),
         "RMSE": round(lasso_metrics["RMSE"], 1), "R2": round(lasso_metrics["R2"], 3)},
    ])
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption(
        "MAE y RMSE son el error promedio del pronóstico, en pesos por kilovatio hora: "
        "entre más bajos, mejor. R2 indica qué tanto explica el modelo el "
        "comportamiento del precio. Conjunto de prueba: diciembre de 2022 a marzo de 2026."
    )

tab_hist, tab_pred, tab_expl, tab_clima = st.tabs(
    ["Histórico del precio", "Predicción del día siguiente",
     "Explorar patrones", "Clima y precio"])


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

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist["FECHA"], y=hist["maximo"], mode="lines",
                                   line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_h.add_trace(go.Scatter(x=hist["FECHA"], y=hist["minimo"], mode="lines", fill="tonexty",
                                   fillcolor="rgba(37,99,235,0.12)", line=dict(width=0),
                                   name="Rango del periodo", hoverinfo="skip"))
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
    m1, m2, m3 = st.columns(3)
    ver_real = m1.checkbox("Precio real (precio observado día siguiente)", value=True)
    ver_lasso = m2.checkbox("Modelo Lasso (día siguiente predicho)", value=True)
    ver_persist = m3.checkbox("Persistencia diaria (día base)", value=True)

    fila_match = df[df["FECHA"].dt.date == fecha_sel]
    if fila_match.empty:
        st.warning(f"No hay datos para el día base {fecha_sel}. Elige otra fecha.")
    else:
        row = fila_match.iloc[0]
        fecha_objetivo = row["fecha_objetivo"]
        horas = list(range(24))
        pred_persist = [float(row[f"h{h:02d}"]) for h in range(24)]
        pred_ml = predict_ml(models, row)
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
        if ver_persist:
            fig.add_trace(go.Scatter(x=horas, y=pred_persist, mode="lines+markers",
                                     name="Persistencia diaria (referencia)",
                                     line=dict(color=AZUL, width=3), marker=dict(size=7)))
        if ver_lasso:
            fig.add_trace(go.Scatter(x=horas, y=pred_ml, mode="lines+markers",
                                     name="Modelo Lasso (comparación)",
                                     line=dict(color=VERDE, width=2), marker=dict(size=6)))
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

        st.markdown(
            "| Línea | Qué representa |\n"
            "|---|---|\n"
            "| **Persistencia diaria** (azul) | Precio real del **día base**, repetido "
            "como pronóstico del día siguiente. |\n"
            "| **Modelo Lasso** (verde) | Pronóstico del modelo para el **día siguiente**. |\n"
            "| **Precio real observado** (punteada) | Precio que **realmente ocurrió** el "
            "día siguiente; aparece solo cuando ya hay datos de esa fecha. |"
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

            met = pd.DataFrame({
                "Métrica": ["MAE ($/kWh)", "RMSE ($/kWh)", "sMAPE (%)"],
                "Lasso (día siguiente)": [round(_mae(pred_ml), 2), round(_rmse(pred_ml), 2),
                                          round(_smape(pred_ml), 2)],
                "Persistencia (día base)": [round(_mae(pred_persist), 2), round(_rmse(pred_persist), 2),
                                            round(_smape(pred_persist), 2)],
            })
            met["Mejor"] = [
                "Lasso" if l < p else ("Persistencia" if p < l else "Empate")
                for l, p in zip(met["Lasso (día siguiente)"], met["Persistencia (día base)"])
            ]
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
            "Franja pico": ["Sí" if h in PEAK_HOURS else "" for h in horas],
        })
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
    fig_hora = px.box(larga, x="hora", y="precio",
                      labels={"hora": "Hora del día", "precio": "Precio de bolsa ($/kWh)"},
                      color_discrete_sequence=[AZUL])
    st.plotly_chart(estilo_grafico(fig_hora, PLANTILLA, 400), use_container_width=True)

    if "base_tipo_dia" in larga.columns:
        st.markdown("**Distribución del precio por tipo de día**")
        larga["Tipo de día"] = larga["base_tipo_dia"].map(TIPO_DIA_ES).fillna(larga["base_tipo_dia"])
        orden = ["Hábil", "Sábado", "Domingo", "Festivo"]
        fig_tipo = px.box(larga, x="Tipo de día", y="precio",
                          category_orders={"Tipo de día": orden},
                          labels={"precio": "Precio de bolsa ($/kWh)"},
                          color_discrete_sequence=[VERDE])
        st.plotly_chart(estilo_grafico(fig_tipo, PLANTILLA, 400), use_container_width=True)


# ======================================================================== #
# Vista 4: clima y precio (ONI y embalses)
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
        fig_c = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c.add_trace(go.Scatter(x=clima_f["FECHA"], y=clima_f["precio_promedio"],
                                   name="Precio promedio ($/kWh)", line=dict(color=AZUL, width=2)),
                        secondary_y=False)
        if "oni" in clima_f.columns:
            fig_c.add_trace(go.Scatter(x=clima_f["FECHA"], y=clima_f["oni"],
                                       name="Índice de El Niño (ONI)",
                                       line=dict(color=NARANJA, width=2)), secondary_y=True)
        if "embalse_porcentaje_diario" in clima_f.columns:
            fig_c.add_trace(go.Scatter(x=clima_f["FECHA"], y=clima_f["embalse_porcentaje_diario"],
                                       name="Nivel de embalses (%)",
                                       line=dict(color=VERDE, width=2, dash="dot")),
                            secondary_y=True)
        fig_c.update_yaxes(title_text="Precio de bolsa ($/kWh)", secondary_y=False)
        fig_c.update_yaxes(title_text="ONI y nivel de embalses", secondary_y=True)
        fig_c.update_layout(xaxis_title="Fecha", hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1))
        st.plotly_chart(estilo_grafico(fig_c, PLANTILLA, 480), use_container_width=True)
        st.caption("Un ONI positivo (fase de El Niño) suele coincidir con embalses más "
                   "bajos y precios más altos; la fase de La Niña, con lo contrario.")

# --- Pie ---
st.divider()
st.caption(
    "Proyecto aplicado, Maestría en Ciencia de Datos y Analítica, UNAD. "
    "Modelo de referencia: persistencia diaria. Modelo de comparación: 24 "
    "regresiones Lasso por hora con variables exógenas (ONI, embalses). "
    "Fuente de datos: XM (SIN) y NOAA CPC."
)
