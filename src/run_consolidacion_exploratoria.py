"""
Ejecuta todas las estrategias de prediccion day-ahead bajo el protocolo de
evaluacion homogeneo y consolida los resultados en una tabla comparativa unica.

Genera en data/outputs/:
  - consolidado_estrategias.csv        metricas globales por estrategia
  - consolidado_predicciones.npz       predicciones para pruebas de significancia
  - consolidado_diebold_mariano.csv    contraste de cada estrategia contra la persistencia
  - consolidado_por_regimen.csv        desempeno en regimen calmo y volatil

Uso:
    python -m src.run_consolidacion_exploratoria
"""

import time

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.evaluation import diebold_mariano, mae
from src.exploratory_models import (
    TARGET_COLS,
    apilar,
    cargar_datos,
    estrategia_dos_etapas,
    estrategia_hibrida_horas_pico,
    estrategia_lasso_directo,
    estrategia_lstm,
    estrategia_persistencia,
    estrategia_prophet,
    estrategia_residual,
    estrategia_segmentada_tipo_dia,
    evaluar,
)

UMBRAL_REGIMEN = 300.0

ESTRATEGIAS = [
    ("Persistencia diaria", "Linea base", estrategia_persistencia),
    ("Lasso horario", "Aprendizaje automatico", estrategia_lasso_directo),
    ("Aprendizaje residual", "Estrategia avanzada", estrategia_residual),
    ("Modelo de dos etapas", "Estrategia avanzada", estrategia_dos_etapas),
    ("Segmentacion por tipo de dia", "Estrategia avanzada", estrategia_segmentada_tipo_dia),
    ("Hibrido con correccion en horas pico", "Estrategia avanzada", estrategia_hibrida_horas_pico),
    ("Prophet", "Serie temporal especializada", estrategia_prophet),
    ("LSTM", "Aprendizaje profundo", estrategia_lstm),
]


def main() -> None:
    df, train_df, test_df = cargar_datos()
    print(f"Entrenamiento: {len(train_df)} dias "
          f"({train_df.FECHA.min().date()} a {train_df.FECHA.max().date()})")
    print(f"Prueba:        {len(test_df)} dias "
          f"({test_df.FECHA.min().date()} a {test_df.FECHA.max().date()})")

    y_true = apilar(test_df, TARGET_COLS)
    print(f"Observaciones horarias de prueba: {len(y_true)}\n")

    resultados, predicciones = [], {}

    for nombre, familia, funcion in ESTRATEGIAS:
        print(f"Ejecutando: {nombre} ...", flush=True)
        inicio = time.time()
        y_pred = funcion(train_df, test_df)
        transcurrido = time.time() - inicio

        metricas = evaluar(y_true, y_pred, nombre, familia)
        metricas["segundos"] = round(transcurrido, 1)
        resultados.append(metricas)
        predicciones[nombre] = y_pred

        print(f"  MAE={metricas['MAE']:.2f}  RMSE={metricas['RMSE']:.2f}  "
              f"sMAPE={metricas['sMAPE']:.2f}  R2={metricas['R2']:.4f}  "
              f"({transcurrido:.1f}s)\n", flush=True)

    tabla = pd.DataFrame(resultados).sort_values("MAE").reset_index(drop=True)
    tabla.to_csv(OUTPUTS_DIR / "consolidado_estrategias.csv", index=False)

    np.savez_compressed(OUTPUTS_DIR / "consolidado_predicciones.npz",
                        y_true=y_true, **predicciones)

    # Significancia estadistica frente a la persistencia diaria
    referencia = predicciones["Persistencia diaria"]
    err_ref = y_true - referencia
    filas_dm = []
    for nombre, y_pred in predicciones.items():
        if nombre == "Persistencia diaria":
            continue
        for criterio, perdida in [("MAE", "abs"), ("RMSE", "squared")]:
            prueba = diebold_mariano(y_true - y_pred, err_ref, loss=perdida)
            filas_dm.append({
                "estrategia": nombre,
                "criterio": criterio,
                "DM_stat": round(prueba["dm_stat"], 3),
                "p_value": round(prueba["p_value"], 4),
                "significativo_5pct": prueba["p_value"] < 0.05,
                "favorece": "estrategia" if prueba["dm_stat"] < 0 else "persistencia",
            })
    pd.DataFrame(filas_dm).to_csv(
        OUTPUTS_DIR / "consolidado_diebold_mariano.csv", index=False)

    # Desempeno por regimen de precio
    calmo = y_true < UMBRAL_REGIMEN
    filas_reg = []
    for nombre, y_pred in predicciones.items():
        for etiqueta, mascara in [("calmo (<300)", calmo), ("volatil (>=300)", ~calmo)]:
            filas_reg.append({
                "estrategia": nombre,
                "regimen": etiqueta,
                "n_obs": int(mascara.sum()),
                "MAE": round(mae(y_true[mascara], y_pred[mascara]), 3),
            })
    pd.DataFrame(filas_reg).to_csv(
        OUTPUTS_DIR / "consolidado_por_regimen.csv", index=False)

    print("=" * 78)
    print(tabla.to_string(index=False))
    print("\nArchivos generados en", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
