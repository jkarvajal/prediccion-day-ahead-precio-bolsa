"""
Contiene funciones para la construcción de variables derivadas del problema de series
temporales, incluyendo rezagos, estadísticas rodantes, variables de calendario y
transformaciones cíclicas para apoyar el modelado day-ahead del precio de bolsa.
"""

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "precio_bolsa",
    lags: list[int] = [1, 24, 168]
) -> pd.DataFrame:
    """
    Agrega variables rezagadas de la serie objetivo.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    target_col : str, default="precio_bolsa"
        Nombre de la variable objetivo.
    lags : list[int], default=[1, 24, 168]
        Lista de rezagos a construir.

    Retorna
    -------
    pd.DataFrame
        DataFrame con nuevas columnas de rezagos.
    """
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "precio_bolsa",
    windows: list[int] = [3, 24, 168]
) -> pd.DataFrame:
    """
    Agrega estadísticas rodantes de la serie objetivo.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    target_col : str, default="precio_bolsa"
        Nombre de la variable objetivo.
    windows : list[int], default=[3, 24, 168]
        Tamaños de ventana para estadísticas móviles.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas de media y desviación estándar rodante.
    """
    df = df.copy()

    for window in windows:
        df[f"rolling_mean_{window}"] = df[target_col].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df[target_col].rolling(window=window).std()

    return df


def add_calendar_features(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora"
) -> pd.DataFrame:
    """
    Agrega variables de calendario derivadas de una columna datetime.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    datetime_col : str, default="fecha_hora"
        Nombre de la columna temporal.

    Retorna
    -------
    pd.DataFrame
        DataFrame con variables de calendario.
    """
    df = df.copy()

    dt = pd.to_datetime(df[datetime_col], errors="coerce")

    df["hora"] = dt.dt.hour
    df["dia_semana"] = dt.dt.dayofweek
    df["dia_mes"] = dt.dt.day
    df["mes"] = dt.dt.month
    df["anio"] = dt.dt.year
    df["semana_anio"] = dt.dt.isocalendar().week.astype(int)

    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega transformaciones cíclicas para variables temporales periódicas.

    Requiere que ya existan las columnas:
    - hora
    - dia_semana
    - mes

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame con variables seno y coseno.
    """
    df = df.copy()

    if "hora" in df.columns:
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

    if "dia_semana" in df.columns:
        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    if "mes" in df.columns:
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    return df


def build_feature_dataframe(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora",
    target_col: str = "precio_bolsa",
    lags: list[int] = [1, 24, 168],
    windows: list[int] = [3, 24, 168],
    dropna: bool = True
) -> pd.DataFrame:
    """
    Construye un dataframe de features combinando variables de calendario,
    variables cíclicas, rezagos y estadísticas rodantes.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    datetime_col : str, default="fecha_hora"
        Nombre de la columna temporal.
    target_col : str, default="precio_bolsa"
        Nombre de la variable objetivo.
    lags : list[int], default=[1, 24, 168]
        Rezagos a construir.
    windows : list[int], default=[3, 24, 168]
        Ventanas para estadísticas rodantes.
    dropna : bool, default=True
        Si es True, elimina filas con valores faltantes generados por rezagos
        o ventanas móviles.

    Retorna
    -------
    pd.DataFrame
        DataFrame listo para análisis o modelado.
    """
    df = df.copy()

    df = add_calendar_features(df, datetime_col=datetime_col)
    df = add_cyclic_time_features(df)
    df = add_lag_features(df, target_col=target_col, lags=lags)
    df = add_rolling_features(df, target_col=target_col, windows=windows)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df