"""
Consolida bajo un protocolo de evaluacion homogeneo las estrategias avanzadas de
prediccion day-ahead que se exploraron durante la fase diagnostica: aprendizaje
residual sobre la persistencia, modelos de dos etapas, segmentacion por tipo de
dia y por horas pico, Prophet y redes LSTM.

Protocolo unico (identico al de NB07 y NB08):
  - Insumo: data/processed/features_day_ahead.csv
  - Particion temporal 80/20 sin solapamiento
  - Prediccion simultanea de las 24 horas del dia siguiente
  - Metricas MAE, RMSE, sMAPE y R2 agrupando las 24 horas del conjunto de prueba

Todas las estrategias reciben exactamente la misma informacion disponible al
cierre del dia anterior, de modo que las diferencias observadas son atribuibles
al enfoque de modelado y no al protocolo de evaluacion.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR
from src.evaluation import mae, rmse, r2_score, smape

warnings.filterwarnings("ignore")

TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]
HOUR_COLS = [f"h{h:02d}" for h in range(24)]
PEAK_HOURS = [18, 19, 20, 21, 22]

LEAKAGE_PREFIXES = [
    "target_h", "delta_target_", "shape_target_",
    "target_promedio_diario", "target_max_diario",
    "target_min_diario", "target_std_diario",
    "evento_pico_p90_target", "evento_pico_p95_target",
]
NON_FEATURE_COLS = ["FECHA", "fecha_objetivo", "base_tipo_dia", "transicion_tipo_dia"]
EXOG_KEYWORDS = ["oni", "embalse"]


# --------------------------------------------------------------------------- #
# Preparacion de datos y de conjuntos de variables
# --------------------------------------------------------------------------- #

def is_leakage(col: str) -> bool:
    """Indica si una columna filtra informacion del dia objetivo."""
    return any(col.startswith(p) or col == p for p in LEAKAGE_PREFIXES)


def cargar_datos(train_ratio: float = 0.80):
    """
    Carga la base supervisada day-ahead y aplica la particion temporal 80/20.

    Retorna
    -------
    tuple
        (df_completo, train_df, test_df)
    """
    df = pd.read_csv(PROCESSED_DIR / "features_day_ahead.csv")
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"], errors="coerce")
    df = df.sort_values("FECHA").reset_index(drop=True)

    df_model = df.dropna(subset=TARGET_COLS, how="all").reset_index(drop=True)
    corte = int(len(df_model) * train_ratio)

    train_df = df_model.iloc[:corte].reset_index(drop=True)
    test_df = df_model.iloc[corte:].reset_index(drop=True)

    assert train_df["FECHA"].max() < test_df["FECHA"].min(), "Solapamiento temporal"
    return df_model, train_df, test_df


def features_globales(df: pd.DataFrame, incluir_exogenas: bool) -> list[str]:
    """Variables no especificas de una hora concreta."""
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not is_leakage(c)
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(c.startswith(f"h{h:02d}") for h in range(24))
        and not any(c.startswith(f"shape_base_h{h:02d}") for h in range(24))
    ]
    if incluir_exogenas:
        return cols
    return [c for c in cols if not any(c.startswith(kw) for kw in EXOG_KEYWORDS)]


def features_por_hora(df: pd.DataFrame, hora: int, incluir_exogenas: bool = True) -> list[str]:
    """Variables para predecir una hora: su propio precio, rezagos, forma y globales."""
    h = f"h{hora:02d}"
    especificas = [
        c for c in df.columns
        if c in (h, f"{h}_lag_1d", f"{h}_lag_2d", f"{h}_lag_7d",
                 f"{h}_lag_14d", f"{h}_diff_vs_7d", f"shape_base_{h}")
        and not is_leakage(c)
    ]
    return sorted(set(especificas + features_globales(df, incluir_exogenas)))


def _pipeline_lasso(alpha: float = 1.0) -> Pipeline:
    """Pipeline estandar: imputacion por mediana, escalado y Lasso."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=5000)),
    ])


# --------------------------------------------------------------------------- #
# Evaluacion comun
# --------------------------------------------------------------------------- #

def evaluar(y_true: np.ndarray, y_pred: np.ndarray, nombre: str, familia: str) -> dict:
    """Calcula las metricas globales agrupando las 24 horas del dia siguiente."""
    valido = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valido], y_pred[valido]
    return {
        "estrategia": nombre,
        "familia": familia,
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "sMAPE": smape(yt, yp),
        "R2": r2_score(yt, yp),
        "n_obs": int(len(yt)),
    }


def apilar(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convierte 24 columnas horarias en un vector unico ordenado por dia y hora."""
    return df[cols].to_numpy(dtype=float).ravel()


# --------------------------------------------------------------------------- #
# Estrategias
# --------------------------------------------------------------------------- #

def estrategia_persistencia(train_df, test_df) -> np.ndarray:
    """Persistencia diaria: el precio de manana a la hora h es el de hoy a la hora h."""
    return apilar(test_df, HOUR_COLS)


def estrategia_lasso_directo(train_df, test_df, incluir_exogenas=True) -> np.ndarray:
    """Lasso independiente por hora que predice el nivel del precio."""
    preds = np.empty((len(test_df), 24), dtype=float)
    for h in range(24):
        col = f"target_h{h:02d}"
        feats = [f for f in features_por_hora(train_df, h, incluir_exogenas) if f in train_df.columns]
        y = train_df[col]
        ok = ~y.isna()
        pipe = _pipeline_lasso()
        pipe.fit(train_df.loc[ok, feats], y[ok])
        preds[:, h] = pipe.predict(test_df[feats])
    return preds.ravel()


def estrategia_residual(train_df, test_df, incluir_exogenas=True) -> np.ndarray:
    """
    Aprendizaje residual: el modelo aprende la correccion sobre la persistencia
    diaria en lugar del nivel del precio. La prediccion final es el precio de hoy
    mas el residuo estimado.
    """
    preds = np.empty((len(test_df), 24), dtype=float)
    for h in range(24):
        col, hcol = f"target_h{h:02d}", f"h{h:02d}"
        feats = [f for f in features_por_hora(train_df, h, incluir_exogenas) if f in train_df.columns]
        residuo = train_df[col] - train_df[hcol]
        ok = ~residuo.isna()
        pipe = _pipeline_lasso()
        pipe.fit(train_df.loc[ok, feats], residuo[ok])
        preds[:, h] = test_df[hcol].to_numpy(dtype=float) + pipe.predict(test_df[feats])
    return preds.ravel()


def estrategia_dos_etapas(train_df, test_df, incluir_exogenas=True) -> np.ndarray:
    """
    Modelo de dos etapas. La primera etapa predice el nivel medio diario del dia
    siguiente; la segunda predice la forma del perfil horario, es decir la razon
    entre el precio de cada hora y la media del dia. El producto de ambas
    estimaciones reconstruye las 24 horas.
    """
    feats_g = [f for f in features_globales(train_df, incluir_exogenas) if f in train_df.columns]

    # Etapa 1: nivel medio del dia objetivo
    media_train = train_df[TARGET_COLS].mean(axis=1)
    ok = ~media_train.isna()
    pipe_nivel = _pipeline_lasso()
    pipe_nivel.fit(train_df.loc[ok, feats_g], media_train[ok])
    nivel_pred = pipe_nivel.predict(test_df[feats_g])

    # Etapa 2: forma del perfil horario
    preds = np.empty((len(test_df), 24), dtype=float)
    for h in range(24):
        forma = train_df[f"target_h{h:02d}"] / media_train.replace(0, np.nan)
        feats = [f for f in features_por_hora(train_df, h, incluir_exogenas) if f in train_df.columns]
        ok_f = ~forma.isna()
        pipe_forma = _pipeline_lasso(alpha=0.01)
        pipe_forma.fit(train_df.loc[ok_f, feats], forma[ok_f])
        preds[:, h] = nivel_pred * pipe_forma.predict(test_df[feats])
    return preds.ravel()


def estrategia_segmentada_tipo_dia(train_df, test_df, incluir_exogenas=True) -> np.ndarray:
    """
    Segmentacion temporal: se entrena un modelo independiente por cada
    combinacion de hora y tipo de dia objetivo (habil, sabado, domingo, festivo),
    bajo el supuesto de que cada tipo de dia tiene una dinamica propia.
    """
    seg_col = "target_tipo_dia"
    preds = np.empty((len(test_df), 24), dtype=float)
    segmentos = train_df[seg_col].dropna().unique()

    for h in range(24):
        col = f"target_h{h:02d}"
        feats = [f for f in features_por_hora(train_df, h, incluir_exogenas) if f in train_df.columns]
        # Modelo de respaldo entrenado con todos los dias
        y_all = train_df[col]
        ok_all = ~y_all.isna()
        respaldo = _pipeline_lasso()
        respaldo.fit(train_df.loc[ok_all, feats], y_all[ok_all])
        preds[:, h] = respaldo.predict(test_df[feats])

        for seg in segmentos:
            m_tr = (train_df[seg_col] == seg) & ok_all
            m_te = (test_df[seg_col] == seg).to_numpy()
            # Solo se especializa el segmento si tiene tamano suficiente
            if m_tr.sum() < 100 or m_te.sum() == 0:
                continue
            pipe = _pipeline_lasso()
            pipe.fit(train_df.loc[m_tr, feats], y_all[m_tr])
            preds[m_te, h] = pipe.predict(test_df.loc[m_te, feats])
    return preds.ravel()


def estrategia_hibrida_horas_pico(train_df, test_df, incluir_exogenas=True) -> np.ndarray:
    """
    Enfoque hibrido: la persistencia diaria se conserva en las horas de baja
    variabilidad y solo se corrige mediante aprendizaje residual en las horas
    pico, donde se concentra el mayor error del pronostico.
    """
    preds = apilar(test_df, HOUR_COLS).reshape(len(test_df), 24).copy()
    for h in PEAK_HOURS:
        col, hcol = f"target_h{h:02d}", f"h{h:02d}"
        feats = [f for f in features_por_hora(train_df, h, incluir_exogenas) if f in train_df.columns]
        residuo = train_df[col] - train_df[hcol]
        ok = ~residuo.isna()
        pipe = _pipeline_lasso()
        pipe.fit(train_df.loc[ok, feats], residuo[ok])
        preds[:, h] = test_df[hcol].to_numpy(dtype=float) + pipe.predict(test_df[feats])
    return preds.ravel()


# --------------------------------------------------------------------------- #
# Modelos especializados de series temporales
# --------------------------------------------------------------------------- #

def _serie_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la serie horaria continua a partir de las columnas del dia base."""
    m = df[["FECHA"] + HOUR_COLS].melt(id_vars="FECHA", var_name="hora", value_name="y")
    m["ds"] = m["FECHA"] + pd.to_timedelta(m["hora"].str[1:].astype(int), unit="h")
    return m[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)


def estrategia_prophet(train_df, test_df, ventana_anios: int = 1,
                       verbose: bool = False) -> np.ndarray:
    """
    Prophet, modelo aditivo de descomposicion en tendencia y estacionalidades
    multiples (diaria, semanal y anual).

    El modelo carece de componentes autorregresivas explicitas, de modo que sin
    reajuste periodico quedaria en desventaja frente a las estrategias que
    reciben el precio del dia anterior como entrada. Para equiparar las
    condiciones se recalibra semanalmente sobre una ventana movil, de manera que
    su horizonte de pronostico nunca supera los siete dias y siempre incorpora la
    informacion mas reciente disponible.
    """
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)

    historia = pd.concat([_serie_horaria(train_df), _serie_horaria(test_df)],
                         ignore_index=True).sort_values("ds")

    fechas_obj = test_df["fecha_objetivo"]
    semanas = fechas_obj.dt.to_period("W")
    periodos = semanas.unique()
    preds = np.full((len(test_df), 24), np.nan, dtype=float)

    for i, periodo in enumerate(periodos):
        mask = (semanas == periodo).to_numpy()
        inicio = fechas_obj[mask].min()

        # Solo informacion estrictamente anterior a la semana pronosticada
        hist = historia[historia["ds"] < inicio]
        hist = hist[hist["ds"] >= hist["ds"].max() - pd.DateOffset(years=ventana_anios)]

        modelo = Prophet(daily_seasonality=True, weekly_seasonality=True,
                         yearly_seasonality=True)
        modelo.fit(hist)

        objetivo = pd.DataFrame({
            "ds": [d + pd.Timedelta(hours=h)
                   for d in fechas_obj[mask] for h in range(24)]
        })
        preds[mask] = modelo.predict(objetivo)["yhat"].to_numpy().reshape(-1, 24)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Prophet: {i + 1}/{len(periodos)} semanas", flush=True)

    return preds.ravel()


def estrategia_lstm(train_df, test_df, dias_ventana: int = 7, epocas: int = 80,
                    semilla: int = 42, verbose: bool = False) -> np.ndarray:
    """
    Red neuronal recurrente LSTM que recibe como entrada la secuencia de precios
    de los ultimos dias, organizada en pasos de 24 horas, y produce de forma
    simultanea las 24 horas del dia siguiente.

    Para equiparar la informacion disponible con la de las estrategias tabulares,
    cada paso temporal incorpora ademas el contexto de calendario del dia
    correspondiente, codificado de forma ciclica, junto con los indicadores de
    fin de semana y de festivo.

    Las series se estandarizan con los estadisticos del conjunto de
    entrenamiento para evitar filtracion de informacion del periodo de prueba.
    """
    import os
    import random

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    np.random.seed(semilla)
    random.seed(semilla)
    tf.random.set_seed(semilla)

    cal_cols = [c for c in ("base_dia_semana_sin", "base_dia_semana_cos",
                            "base_mes_sin", "base_mes_cos",
                            "base_es_fin_semana", "base_es_festivo")
                if c in train_df.columns]

    base_tr = train_df[HOUR_COLS].to_numpy(dtype=float)
    base_te = test_df[HOUR_COLS].to_numpy(dtype=float)
    cal_tr = train_df[cal_cols].to_numpy(dtype=float)
    cal_te = test_df[cal_cols].to_numpy(dtype=float)
    y_tr = train_df[TARGET_COLS].to_numpy(dtype=float)

    mu, sigma = np.nanmean(base_tr), np.nanstd(base_tr)
    n_var = 24 + len(cal_cols)

    def secuencias(base: np.ndarray, cal: np.ndarray) -> np.ndarray:
        """Apila los ultimos `dias_ventana` dias como pasos temporales."""
        rasgos = np.hstack([(base - mu) / sigma, cal])
        n = len(rasgos)
        X = np.full((n, dias_ventana, n_var), np.nan, dtype=float)
        for d in range(dias_ventana):
            desplazado = np.roll(rasgos, d, axis=0)
            desplazado[:d] = np.nan
            X[:, dias_ventana - 1 - d] = desplazado
        return X

    X_tr = secuencias(base_tr, cal_tr)
    X_te = secuencias(base_te, cal_te)
    Y_tr = (y_tr - mu) / sigma

    completos = ~(np.isnan(X_tr).any(axis=(1, 2)) | np.isnan(Y_tr).any(axis=1))
    X_tr, Y_tr = X_tr[completos], Y_tr[completos]
    X_te = np.nan_to_num(X_te, nan=0.0)

    red = models.Sequential([
        layers.Input(shape=(dias_ventana, n_var)),
        layers.LSTM(96, return_sequences=True),
        layers.LSTM(48),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dense(24),
    ])
    red.compile(optimizer="adam", loss="mae")
    red.fit(X_tr, Y_tr, epochs=epocas, batch_size=32, validation_split=0.1,
            shuffle=False, verbose=1 if verbose else 0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True)])

    return (red.predict(X_te, verbose=0) * sigma + mu).ravel()
