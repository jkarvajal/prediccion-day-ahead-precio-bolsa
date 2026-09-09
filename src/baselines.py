"""
Implementa modelos de referencia o lineas base de pronostico, utilizados para
comparar de forma objetiva el desempeno de los modelos de aprendizaje automatico
en el horizonte day-ahead.

Todas las funciones operan sobre la base day-ahead (una fila por dia base, con
columnas h00..h23 para el perfil del dia actual y target_h00..target_h23 para el
dia siguiente) y devuelven un DataFrame de predicciones con columnas
target_h00..target_h23, alineado por indice con la entrada.
"""

import numpy as np
import pandas as pd

HORA_COLS = [f"h{h:02d}" for h in range(24)]
TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]


def persistencia_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Persistencia diaria: predice cada hora del dia siguiente con el precio de la
    misma hora del dia actual. Es la referencia principal del proyecto.
    """
    pred = df[HORA_COLS].copy()
    pred.columns = TARGET_COLS
    return pred.reset_index(drop=True)


def persistencia_semanal(df: pd.DataFrame, fecha_col: str = "fecha_objetivo") -> pd.DataFrame:
    """
    Persistencia semanal: predice el perfil horario del dia siguiente con el
    perfil observado siete dias antes (misma hora, misma posicion semanal).
    Requiere la columna de fecha objetivo para el emparejamiento temporal.
    """
    base = df.copy()
    lookup = base.set_index(fecha_col)[HORA_COLS]
    fechas_lag7 = pd.to_datetime(base[fecha_col]) - pd.Timedelta(days=7)
    pred = lookup.reindex(fechas_lag7.values).reset_index(drop=True)
    pred.columns = TARGET_COLS
    return pred


def promedio_historico_hora(
    df: pd.DataFrame, train_size: int
) -> pd.DataFrame:
    """
    Promedio historico por hora: predice cada hora con el promedio de esa hora
    calculado unicamente con el tramo de entrenamiento, evitando fuga de datos.
    """
    medias = df[TARGET_COLS].iloc[:train_size].mean()
    pred = pd.DataFrame(
        np.tile(medias.values, (len(df), 1)), columns=TARGET_COLS
    )
    return pred


def evaluar_baseline(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> dict:
    """
    Metricas globales (MAE, RMSE) sobre las 24 horas, ignorando filas con
    valores faltantes en el pronostico. Reutiliza src.evaluation si esta
    disponible.
    """
    from src.evaluation import mae, rmse

    mask = ~y_pred_df.isna().any(axis=1).values
    yt = y_true_df.iloc[mask].values.flatten()
    yp = y_pred_df.iloc[mask].values.flatten()
    return {"n_dias": int(mask.sum()), "MAE": mae(yt, yp), "RMSE": rmse(yt, yp)}
