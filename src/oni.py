"""
Contiene funciones para cargar índices climáticos asociados al fenómeno ENSO,
como ONI, RONI y SOI, con el fin de analizarlos como posibles variables
exógenas en el modelamiento del precio de bolsa.
"""

import pandas as pd
import requests
from io import StringIO

# =========================================================
# FUENTES
# =========================================================
url_oni = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
url_roni = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
url_soi = "https://www.cpc.ncep.noaa.gov/data/indices/soi"


def cargar_indice_trimestral(url: str) -> pd.DataFrame:
    """
    Carga un índice trimestral tipo ONI o RONI y lo convierte a una serie
    temporal con una fecha mensual representativa.

    Parámetros
    ----------
    url : str
        URL del archivo fuente.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas originales más:
        - MES
        - FECHA
    """
    df = pd.read_csv(url, sep=r"\s+", engine="python")

    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }

    df["MES"] = df["SEAS"].map(season_to_month)
    df["FECHA"] = pd.to_datetime(
        dict(year=df["YR"], month=df["MES"], day=1)
    )

    df = df.sort_values("FECHA").reset_index(drop=True)
    return df


def cargar_soi(url: str) -> pd.DataFrame:
    """
    Carga el índice SOI desde la fuente NOAA y lo transforma a formato largo
    con una columna temporal FECHA.

    Parámetros
    ----------
    url : str
        URL del archivo fuente.

    Retorna
    -------
    pd.DataFrame
        DataFrame en formato largo con:
        - YEAR
        - MONTH
        - SOI
        - MES
        - FECHA
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text

    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("No se encontró la fila YEAR en el archivo SOI.")

    table_text = "\n".join(lines[start_idx:])
    df = pd.read_fwf(StringIO(table_text))

    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={df.columns[0]: "YEAR"})
    df = df[df["YEAR"].notna()]
    df = df[df["YEAR"].astype(str) != "YEAR"]

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=["YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    month_cols = [col for col in month_map.keys() if col in df.columns]

    if not month_cols:
        raise ValueError(
            "No se encontraron columnas mensuales esperadas en el archivo SOI."
        )

    df_long = df.melt(
        id_vars="YEAR",
        value_vars=month_cols,
        var_name="MONTH",
        value_name="SOI"
    )

    df_long["SOI"] = pd.to_numeric(df_long["SOI"], errors="coerce")
    df_long["MES"] = df_long["MONTH"].map(month_map)
    df_long["FECHA"] = pd.to_datetime(
        dict(year=df_long["YEAR"], month=df_long["MES"], day=1)
    )

    df_long = df_long.sort_values("FECHA").reset_index(drop=True)
    return df_long


# -----------------------------------------------------------------------------------------------------------------------------
def cargar_indice_trimestral_MDh(
    url: str,
    frecuencia: str = "M",
    hora_separada: str = "si"
) -> pd.DataFrame:
    """
    Carga un índice tipo ONI/RONI y lo transforma
    a frecuencia mensual, diaria u horaria.

    Parámetros
    ----------
    url : str
        Ruta o URL del archivo fuente.

    frecuencia : str, opcional

        - "M" : Mensual
        - "D" : Diaria
        - "h" : Horaria

    hora_separada : str, opcional

        Solo aplica para frecuencia = "h"

        - "si" : separa FECHA y hora
        - "no" : deja fecha y hora juntas
                   en columna HORA

    Retorna
    -------
    pd.DataFrame
    """

    # ==========================
    # CARGA
    # ==========================
    df = pd.read_csv(
        url,
        sep=r"\s+",
        engine="python"
    )

    # ==========================
    # VALIDACIONES
    # ==========================
    if frecuencia not in ["M", "D", "h"]:
        raise ValueError(
            "Frecuencia no válida. Use: 'M', 'D' o 'h'."
        )

    if hora_separada not in ["si", "no"]:
        raise ValueError(
            "hora_separada debe ser 'si' o 'no'"
        )

    # ==========================
    # MAPEO TEMPORADAS
    # ==========================
    season_to_month = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12
    }

    df["MES"] = df["SEAS"].map(
        season_to_month
    )

    # ==========================
    # FECHA BASE
    # ==========================
    df["FECHA"] = pd.to_datetime(
        dict(
            year=df["YR"],
            month=df["MES"],
            day=1
        )
    )

    # ==========================
    # SOLO COLUMNAS NECESARIAS
    # ==========================
    df = (
        df[["FECHA", "ANOM"]]
        .rename(columns={
            "ANOM": "oni"
        })
        .sort_values("FECHA")
        .reset_index(drop=True)
    )

    # ==========================
    # MENSUAL
    # ==========================
    if frecuencia == "M":

        df["FECHA"] = (
            df["FECHA"]
            .dt.strftime("%Y-%m-%d")
        )

        return df

    # ==========================
    # DIARIA
    # ==========================
    elif frecuencia == "D":

        registros = []

        for _, row in df.iterrows():

            fecha_inicio = row["FECHA"]

            fecha_fin = (
                fecha_inicio
                + pd.offsets.MonthEnd(0)
            )

            fechas = pd.date_range(
                start=fecha_inicio,
                end=fecha_fin,
                freq="D"
            )

            temp = pd.DataFrame({
                "FECHA": fechas,
                "oni": row["oni"]
            })

            registros.append(temp)

        df_daily = pd.concat(
            registros,
            ignore_index=True
        )

        df_daily["FECHA"] = (
            df_daily["FECHA"]
            .dt.strftime("%Y-%m-%d")
        )

        return df_daily

    # ==========================
    # HORARIA
    # ==========================
    elif frecuencia == "h":

        registros = []

        for _, row in df.iterrows():

            fecha_inicio = row["FECHA"]

            fecha_fin = (
                fecha_inicio
                + pd.offsets.MonthEnd(0)
                + pd.Timedelta(hours=23)
            )

            fechas = pd.date_range(
                start=fecha_inicio,
                end=fecha_fin,
                freq="h"
            )

            temp = pd.DataFrame({
                "datetime": fechas,
                "oni": row["oni"]
            })

            registros.append(temp)

        df_hourly = pd.concat(
            registros,
            ignore_index=True
        )

        # ==================================
        # HORA SEPARADA
        # ==================================
        if hora_separada == "si":

            df_hourly["FECHA"] = (
                df_hourly["datetime"]
                .dt.strftime("%Y-%m-%d")
            )

            df_hourly["hora"] = (
                df_hourly["datetime"]
                .dt.hour
            )

            df_hourly = df_hourly[
                ["FECHA", "hora", "oni"]
            ]

            return df_hourly

        # ==================================
        # FECHA Y HORA JUNTAS
        # ==================================
        else:

            df_hourly["FECHA"] = (
                df_hourly["datetime"]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

            df_hourly = df_hourly[
                ["FECHA", "oni"]
            ]

            return df_hourly