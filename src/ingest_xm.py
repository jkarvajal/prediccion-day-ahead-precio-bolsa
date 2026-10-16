# -*- coding: utf-8 -*-
"""
Descarga datos recientes desde la API publica de XM (precio de bolsa y nivel de
embalse) y desde NOAA (indice climatico ONI).

Variables utilizadas de la API XM:
  - PrecBolsNaci / Sistema : Precio Bolsa Nacional horario (COP/kWh)
  - PorcVoluUtilDiar / Sistema: Volumen Util diario porcentaje del SIN agregado

Uso:
    from src.ingest_xm import descargar_precios, descargar_embalse, descargar_oni
"""

import logging
from datetime import date
from io import StringIO

import pandas as pd
import requests

log = logging.getLogger(__name__)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
TIMEOUT = 60

# Variables correctas en la API de XM
COLECCION_PRECIO = "PrecBolsNaci"
ENTIDAD_PRECIO = "Sistema"

COLECCION_EMBALSE = "PorcVoluUtilDiar"
ENTIDAD_EMBALSE = "Sistema"


def _reader():
    try:
        from pydataxm.pydataxm import ReadDB  # noqa: PLC0415
        return ReadDB()
    except ImportError as exc:
        raise ImportError(
            "pydataxm no esta instalado. Ejecuta: pip install pydataxm"
        ) from exc


def _xm_hourly_to_dayrows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el resultado horario de XM (columnas Values_Hour01..24, columna Date)
    a filas diarias con columnas FECHA y h00..h23.
    """
    df = df_raw.copy()
    if "Date" in df.columns:
        df["FECHA"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("No se encontro columna Date en el resultado de XM.")

    for h in range(24):
        old = f"Values_Hour{h + 1:02d}"
        new = f"h{h:02d}"
        if old in df.columns:
            df[new] = pd.to_numeric(df[old], errors="coerce")

    hora_cols = [f"h{h:02d}" for h in range(24)]
    cols = ["FECHA"] + [c for c in hora_cols if c in df.columns]
    df = df[cols].dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)

    # Agrupar por fecha por si hubiera filas duplicadas
    df = df.groupby("FECHA")[hora_cols].mean().reset_index()
    return df


def descargar_precios(inicio: date, fin: date) -> pd.DataFrame:
    """
    Descarga el precio de bolsa horario desde la API de XM.

    Retorna un DataFrame con columnas FECHA (Timestamp) y h00..h23 (COP/kWh).
    """
    log.info("XM %s/%s: %s a %s", COLECCION_PRECIO, ENTIDAD_PRECIO, inicio, fin)
    obj = _reader()
    df_raw = obj.request_data(
        COLECCION_PRECIO, ENTIDAD_PRECIO,
        str(inicio), str(fin)
    )
    if df_raw is None or df_raw.empty:
        raise ValueError(
            f"XM no retorno datos de {COLECCION_PRECIO} para {inicio} a {fin}."
        )
    df = _xm_hourly_to_dayrows(df_raw)
    log.info("Precios descargados: %d dias", len(df))
    return df


def descargar_embalse(inicio: date, fin: date) -> pd.DataFrame:
    """
    Descarga el porcentaje de volumen util diario agregado del SIN desde XM.

    Retorna un DataFrame con columnas FECHA (Timestamp) y embalse_porcentaje_diario
    en rango [0, 1].
    """
    log.info("XM %s/%s: %s a %s", COLECCION_EMBALSE, ENTIDAD_EMBALSE, inicio, fin)
    obj = _reader()
    df_raw = obj.request_data(
        COLECCION_EMBALSE, ENTIDAD_EMBALSE,
        str(inicio), str(fin)
    )
    if df_raw is None or df_raw.empty:
        raise ValueError(
            f"XM no retorno datos de {COLECCION_EMBALSE} para {inicio} a {fin}."
        )

    df = df_raw.copy()
    if "Date" in df.columns:
        df["FECHA"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("No se encontro columna Date en el resultado de XM (embalse).")

    # PorcVoluUtilDiar retorna un solo valor diario en Values_Hour01 o en Value
    for col_candidata in ["Values_Hour01", "Value"]:
        if col_candidata in df.columns:
            df["embalse_porcentaje_diario"] = pd.to_numeric(df[col_candidata], errors="coerce")
            break
    else:
        raise ValueError(
            f"No se encontro columna de valor en {COLECCION_EMBALSE}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = (
        df[["FECHA", "embalse_porcentaje_diario"]]
        .dropna(subset=["FECHA"])
        .groupby("FECHA")["embalse_porcentaje_diario"].mean()
        .reset_index()
        .sort_values("FECHA")
        .reset_index(drop=True)
    )

    # Normalizar a [0,1] si los valores estan en porcentaje [0,100]
    if df["embalse_porcentaje_diario"].max() > 1.5:
        df["embalse_porcentaje_diario"] /= 100.0

    log.info("Embalse descargado: %d dias", len(df))
    return df


# ---------------------------------------------------------------------------
# NOAA ONI
# ---------------------------------------------------------------------------

def descargar_oni() -> pd.Series:
    """
    Descarga el indice ONI desde NOAA.

    Retorna una Series indexada por el primer dia del mes, valores ANOM (anomalia).
    """
    log.info("NOAA ONI: descargando desde %s", ONI_URL)
    resp = requests.get(ONI_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep=r"\s+", engine="python")
    season_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df["MES"] = df["SEAS"].map(season_map)
    df["FECHA"] = pd.to_datetime(dict(year=df["YR"], month=df["MES"], day=1))
    df = df.rename(columns={"ANOM": "oni"})[["FECHA", "oni"]].sort_values("FECHA")
    s = df.set_index("FECHA")["oni"]
    log.info("ONI descargado: %d meses (ultimo: %s)", len(s), s.index.max().date())
    return s
