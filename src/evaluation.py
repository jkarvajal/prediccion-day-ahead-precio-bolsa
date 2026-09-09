"""
Agrupa funciones para evaluar y comparar el desempeño predictivo de líneas base
y modelos, mediante métricas como MAE, RMSE, sMAPE y R², incluyendo análisis
global y por franja horaria.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """
    Calcula el error absoluto medio (MAE).

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Retorna
    -------
    float
        Valor del MAE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred) -> float:
    """
    Calcula la raíz del error cuadrático medio (RMSE).

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Retorna
    -------
    float
        Valor del RMSE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred) -> float:
    """
    Calcula el error porcentual absoluto medio simétrico (sMAPE).

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Retorna
    -------
    float
        Valor del sMAPE en porcentaje.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid = denominator != 0

    result = np.zeros_like(y_true, dtype=float)
    result[valid] = np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]

    return np.mean(result) * 100


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Calcula un conjunto básico de métricas de evaluación para predicciones.

    Parámetros
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Retorna
    -------
    dict
        Diccionario con MAE, RMSE y sMAPE.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def diebold_mariano(errors_a, errors_b, loss: str = "abs", h: int = 1) -> dict:
    """
    Prueba de Diebold-Mariano para comparar la precisión de dos pronósticos.

    Hipótesis nula: ambos pronósticos tienen la misma precisión. Un estadístico
    negativo con p-valor pequeño indica que el modelo A es más preciso que el B.

    Parámetros
    ----------
    errors_a, errors_b : array-like
        Errores de pronóstico (real - predicho) de los modelos A y B.
    loss : str
        "abs" (orientada a MAE) o "squared" (orientada a RMSE).
    h : int
        Horizonte de pronóstico; corrige la varianza por autocorrelación
        (Newey-West) para h > 1.

    Retorna
    -------
    dict con dm_stat, p_value, media_diferencial y n.
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    mask = ~(np.isnan(ea) | np.isnan(eb))
    ea, eb = ea[mask], eb[mask]

    if loss == "squared":
        d = ea ** 2 - eb ** 2
    else:
        d = np.abs(ea) - np.abs(eb)

    n = len(d)
    if n < 2:
        return {"dm_stat": np.nan, "p_value": np.nan, "media_diferencial": np.nan, "n": n}

    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        var += 2.0 * (1.0 - k / h) * gk

    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "media_diferencial": d_bar, "n": n}

    dm_stat = d_bar / np.sqrt(var / n)
    # p-valor de dos colas usando la aproximación normal (erfc)
    from math import erfc, sqrt
    p_value = erfc(abs(dm_stat) / sqrt(2.0))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "media_diferencial": float(d_bar),
        "n": int(n),
    }


def evaluate_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    Evalúa predicciones por grupos, por ejemplo por hora del día.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con valores reales y predichos.
    y_true_col : str
        Nombre de la columna con valores reales.
    y_pred_col : str
        Nombre de la columna con valores predichos.
    group_col : str
        Nombre de la columna por la cual se agrupará la evaluación.

    Retorna
    -------
    pd.DataFrame
        DataFrame con métricas por grupo.
    """
    results = []

    temp = df.dropna(subset=[y_true_col, y_pred_col, group_col]).copy()

    for group_value in sorted(temp[group_col].unique()):
        group_df = temp[temp[group_col] == group_value]

        metrics = evaluate_predictions(
            group_df[y_true_col],
            group_df[y_pred_col]
        )

        results.append({
            group_col: group_value,
            "n_obs": len(group_df),
            **metrics
        })

    return pd.DataFrame(results)