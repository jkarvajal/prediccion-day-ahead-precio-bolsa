# -*- coding: utf-8 -*-
"""
Precalcula intervalos conformales por hora para el modelo Lasso y para la persistencia
diaria, a partir de los residuos del conjunto de prueba (día base entre dic-2022 y
mar-2026). Guarda data/outputs/intervalos_conformales.csv, que consume la aplicación web
para sombrear la banda de incertidumbre alrededor de la predicción.

Método: conformal split por cuantiles. Para una cobertura del 90 %, por cada hora se toma
el cuantil 5 y 95 de los residuos (real - predicción). La banda de un pronóstico es
    [pred + q_low(h), pred + q_high(h)].

Uso:
    python src/gen_intervalos_conformales.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
PROC = ROOT / "data" / "processed"

COBERTURA = 0.90
ALPHA = 1 - COBERTURA
HORAS = range(24)


def cuantiles_residuos(real, pred):
    """Cuantiles bajo y alto de los residuos (real - pred) para la cobertura fijada."""
    r = (np.asarray(real, float) - np.asarray(pred, float))
    r = r[~np.isnan(r)]
    return float(np.quantile(r, ALPHA / 2)), float(np.quantile(r, 1 - ALPHA / 2))


def main():
    filas = []

    # --- Lasso: residuos del conjunto de prueba (archivo de predicciones horarias) ---
    lasso = pd.read_csv(OUT / "hourly_dayahead_predictions.csv")
    for h in HORAS:
        qlo, qhi = cuantiles_residuos(lasso[f"real_h{h:02d}"], lasso[f"pred_h{h:02d}"])
        filas.append({"modelo": "Lasso", "hora": h, "q_low": qlo, "q_high": qhi})

    # --- Persistencia: pred = valor del día base; real = objetivo del día siguiente ---
    feats = pd.read_csv(PROC / "features_day_ahead.csv")
    feats["FECHA"] = pd.to_datetime(feats["FECHA"], errors="coerce")
    fechas_test = pd.to_datetime(lasso["FECHA"])
    mask = feats["FECHA"].isin(fechas_test)
    ft = feats[mask]
    for h in HORAS:
        pred = ft[f"h{h:02d}"]          # día base, misma hora (persistencia)
        real = ft[f"target_h{h:02d}"]   # día siguiente observado
        qlo, qhi = cuantiles_residuos(real, pred)
        filas.append({"modelo": "Persistencia", "hora": h, "q_low": qlo, "q_high": qhi})

    # --- Aprendizaje residual: predicciones fuera de muestra sobre la partición 80/20 ---
    from src.exploratory_models import cargar_datos, estrategia_residual
    _, train_df, test_df = cargar_datos()
    pred_res = estrategia_residual(train_df, test_df).reshape(len(test_df), 24)
    for h in HORAS:
        real = test_df[f"target_h{h:02d}"]
        qlo, qhi = cuantiles_residuos(real, pred_res[:, h])
        filas.append({"modelo": "Aprendizaje residual", "hora": h,
                      "q_low": qlo, "q_high": qhi})

    df = pd.DataFrame(filas)
    df["cobertura"] = COBERTURA
    salida = OUT / "intervalos_conformales.csv"
    df.to_csv(salida, index=False, encoding="utf-8")
    print("Guardado:", salida, "|", len(df), "filas (2 modelos x 24 horas)")
    print(df.groupby("modelo")[["q_low", "q_high"]].mean().round(1))


if __name__ == "__main__":
    main()
