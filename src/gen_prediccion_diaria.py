# -*- coding: utf-8 -*-
"""
Genera la prediccion automatica del precio de bolsa para el dia siguiente.

Flujo:
  1. Descarga los ultimos 35 dias de datos desde XM (precios y embalse) y
     el historico completo del ONI desde NOAA.
  2. Construye la fila de features para ayer (ultimo dia completo disponible).
  3. Aplica los tres modelos: persistencia diaria, Lasso horario y aprendizaje
     residual.
  4. Guarda data/outputs/prediccion_automatica.csv (24 filas, una por hora).

Este script esta disenado para ejecutarse en GitHub Actions de forma programada.
El archivo de salida es el unico artefacto que se compromete al repositorio;
los datos de XM se descargan en tiempo de ejecucion y no se persisten.

Uso:
    python src/gen_prediccion_diaria.py
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "data" / "outputs"

MODEL_LASSO_PATH = MODELS_DIR / "modelo_final_dayahead.joblib"
MODEL_RESIDUAL_PATH = MODELS_DIR / "modelo_residual_dayahead.joblib"
SALIDA = OUTPUTS_DIR / "prediccion_automatica.csv"
SALIDA_VERIF = OUTPUTS_DIR / "prediccion_verificacion.csv"

DIAS_HISTORIAL = 35  # dias hacia atras para calcular rezagos y EWMA


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

def _predict_lasso(modelos: dict, row: dict) -> list:
    preds = []
    for h in range(24):
        info = modelos[h]
        pipe = info["pipeline"]
        feats = info["features"]
        X = pd.DataFrame([{f: row.get(f, float("nan")) for f in feats}])
        preds.append(float(pipe.predict(X)[0]))
    return preds


def _predict_residual(modelos: dict, row: dict) -> list:
    preds = []
    for h in range(24):
        info = modelos[h]
        pipe = info["pipeline"]
        feats = info["features"]
        base_h = float(row.get(f"h{h:02d}", float("nan")))
        X = pd.DataFrame([{f: row.get(f, float("nan")) for f in feats}])
        residuo = float(pipe.predict(X)[0])
        preds.append(base_h + residuo)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from src.ingest_xm import descargar_embalse, descargar_oni, descargar_precios
    from src.build_features_incremental import construir_fila

    hoy = date.today()
    inicio = hoy - timedelta(days=DIAS_HISTORIAL)

    log.info("Descargando datos de XM: %s a %s", inicio, hoy)

    try:
        precios_df = descargar_precios(inicio, hoy)
    except Exception as exc:
        log.error("Error al descargar precios de XM: %s", exc)
        sys.exit(1)

    # Usar el ultimo dia con datos completos disponibles en la API
    fecha_base = precios_df["FECHA"].max().date()
    fecha_objetivo = fecha_base + timedelta(days=1)
    log.info("Fecha base para prediccion: %s (predice %s)", fecha_base, fecha_objetivo)

    try:
        embalse_df = descargar_embalse(inicio, hoy)
        embalse_serie = embalse_df.set_index("FECHA")["embalse_porcentaje_diario"].sort_index()
    except Exception as exc:
        log.warning("No se pudo descargar embalse de XM (%s); se usaran NaN.", exc)
        embalse_serie = pd.Series(dtype=float)

    try:
        oni_serie = descargar_oni()
    except Exception as exc:
        log.warning("No se pudo descargar ONI de NOAA (%s); se usaran NaN.", exc)
        oni_serie = pd.Series(dtype=float)

    # Construir fila de features para fecha_base
    try:
        fila = construir_fila(fecha_base, precios_df, embalse_serie, oni_serie)
    except ValueError as exc:
        log.error("Error al construir features: %s", exc)
        log.info("Fechas disponibles en precios_df: %s",
                 precios_df["FECHA"].dt.date.tolist()[-5:])
        sys.exit(1)

    # Persistencia: precios del dia base
    pred_persist = [float(fila.get(f"h{h:02d}", float("nan"))) for h in range(24)]

    # Lasso
    if MODEL_LASSO_PATH.exists():
        modelos_lasso = joblib.load(MODEL_LASSO_PATH)
        pred_lasso = _predict_lasso(modelos_lasso, fila)
        log.info("Lasso: promedio %.1f $/kWh", float(np.nanmean(pred_lasso)))
    else:
        log.warning("Modelo Lasso no encontrado en %s", MODEL_LASSO_PATH)
        pred_lasso = [float("nan")] * 24

    # Aprendizaje residual
    if MODEL_RESIDUAL_PATH.exists():
        modelos_res = joblib.load(MODEL_RESIDUAL_PATH)
        pred_residual = _predict_residual(modelos_res, fila)
        log.info("Residual: promedio %.1f $/kWh", float(np.nanmean(pred_residual)))
    else:
        log.warning("Modelo residual no encontrado en %s", MODEL_RESIDUAL_PATH)
        pred_residual = [float("nan")] * 24

    log.info("Persistencia: promedio %.1f $/kWh", float(np.nanmean(pred_persist)))

    # Guardar
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    salida = pd.DataFrame({
        "fecha_base": str(fecha_base),
        "fecha_objetivo": str(fecha_objetivo),
        "hora": list(range(24)),
        "persistencia": [round(p, 2) for p in pred_persist],
        "lasso": [round(p, 2) for p in pred_lasso],
        "residual": [round(p, 2) for p in pred_residual],
    })
    salida.to_csv(SALIDA, index=False, encoding="utf-8")
    log.info("Prediccion guardada: %s", SALIDA)

    print(f"\nPrediccion para {fecha_objetivo} (base: {fecha_base}):")
    print(salida[["hora", "persistencia", "lasso", "residual"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # Verificacion: pronostico frente al precio real observado.
    # El pronostico automatico es day-ahead, asi que su dia objetivo aun no tiene
    # precio real. Para poder comparar contra la realidad se genera, en paralelo,
    # un pronostico usando como base el penultimo dia disponible: su objetivo es el
    # ultimo dia disponible (fecha_base), que SI tiene precio real. Este archivo es
    # el que alimenta la grafica de "pronostico vs. real" del dashboard.
    # -----------------------------------------------------------------------
    try:
        verif_base = fecha_base - timedelta(days=1)
        fila_v = construir_fila(verif_base, precios_df, embalse_serie, oni_serie)
        v_persist = [float(fila_v.get(f"h{h:02d}", float("nan"))) for h in range(24)]
        v_lasso = (_predict_lasso(joblib.load(MODEL_LASSO_PATH), fila_v)
                   if MODEL_LASSO_PATH.exists() else [float("nan")] * 24)
        v_residual = (_predict_residual(joblib.load(MODEL_RESIDUAL_PATH), fila_v)
                      if MODEL_RESIDUAL_PATH.exists() else [float("nan")] * 24)
        real_row = precios_df[precios_df["FECHA"].dt.date == fecha_base].iloc[0]
        v_real = [float(real_row.get(f"h{h:02d}", float("nan"))) for h in range(24)]
        verif = pd.DataFrame({
            "fecha_base": str(verif_base),
            "fecha_objetivo": str(fecha_base),
            "hora": list(range(24)),
            "persistencia": [round(p, 2) for p in v_persist],
            "lasso": [round(p, 2) for p in v_lasso],
            "residual": [round(p, 2) for p in v_residual],
            "real": [round(p, 2) for p in v_real],
        })
        verif.to_csv(SALIDA_VERIF, index=False, encoding="utf-8")
        log.info("Verificacion guardada: %s (objetivo %s con precio real)", SALIDA_VERIF, fecha_base)
    except Exception as exc:
        log.warning("No se pudo generar la verificacion: %s", exc)


if __name__ == "__main__":
    main()
