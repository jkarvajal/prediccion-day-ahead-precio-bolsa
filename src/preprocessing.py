"""
Agrupa funciones de limpieza, validación y preparación inicial de los datos,
incluyendo la transformación del archivo PB_Nal.csv desde formato ancho diario
a formato largo horario, listo para análisis y modelado.
"""

import pandas as pd


def ensure_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convierte una columna a tipo datetime.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col : str
        Nombre de la columna a convertir.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna convertida a datetime.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_numeric_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convierte una columna a tipo numérico.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col : str
        Nombre de la columna a convertir.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna convertida a numérica.
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_price_dataframe(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora",
    price_col: str = "precio_bolsa",
) -> pd.DataFrame:
    """
    Limpia un dataframe que ya se encuentra en formato largo, con columnas de fecha-hora
    y precio, eliminando nulos y ordenando cronológicamente.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada en formato largo.
    datetime_col : str, default="fecha_hora"
        Nombre de la columna temporal.
    price_col : str, default="precio_bolsa"
        Nombre de la columna de precio.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio y ordenado.
    """
    df = df.copy()

    df = ensure_datetime_column(df, datetime_col)
    df = ensure_numeric_column(df, price_col)

    df = df.dropna(subset=[datetime_col, price_col]).copy()
    df = df.sort_values(datetime_col).reset_index(drop=True)

    return df


def add_basic_time_columns(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora",
) -> pd.DataFrame:
    """
    Agrega columnas temporales básicas de apoyo para análisis exploratorio y modelado.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    datetime_col : str, default="fecha_hora"
        Nombre de la columna temporal.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas de fecha, hora, día de semana, mes y año.
    """
    df = df.copy()

    df["fecha"] = df[datetime_col].dt.date
    df["hora"] = df[datetime_col].dt.hour
    df["dia_semana"] = df[datetime_col].dt.dayofweek
    df["mes"] = df[datetime_col].dt.month
    df["anio"] = df[datetime_col].dt.year

    return df


def transform_pb_nal_wide_to_long(
    df: pd.DataFrame,
    date_col: str = "Date",
    entity_col: str = "Id",
    value_prefix: str = "Values_Hour"
) -> pd.DataFrame:
    """
    Transforma el archivo PB_Nal.csv desde formato ancho diario a formato largo horario.

    El archivo original contiene una fila por fecha y 24 columnas horarias
    (Values_Hour01 a Values_Hour24). Esta función convierte esa estructura
    en un dataframe con una fila por hora, generando las columnas:
    - fecha_hora
    - precio_bolsa

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original en formato ancho.
    date_col : str, default="Date"
        Nombre de la columna de fecha.
    entity_col : str, default="Id"
        Nombre de la columna identificadora de la entidad.
    value_prefix : str, default="Values_Hour"
        Prefijo de las columnas horarias.

    Retorna
    -------
    pd.DataFrame
        DataFrame en formato largo con columnas:
        fecha_hora, precio_bolsa, fecha, hora, Id
    """
    df = df.copy()

    hour_cols = [col for col in df.columns if col.startswith(value_prefix)]

    if date_col not in df.columns:
        raise KeyError(f"No se encontró la columna de fecha: {date_col}")

    if not hour_cols:
        raise ValueError("No se encontraron columnas horarias tipo Values_Hour01...Values_Hour24")

    id_vars = [col for col in [entity_col, date_col] if col in df.columns]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=hour_cols,
        var_name="hora_col",
        value_name="precio_bolsa"
    )

    df_long[date_col] = pd.to_datetime(df_long[date_col], errors="coerce")
    df_long["precio_bolsa"] = pd.to_numeric(df_long["precio_bolsa"], errors="coerce")

    # Extraer el número de hora desde columnas tipo Values_Hour01
    df_long["hora_origen"] = (
        df_long["hora_col"]
        .str.extract(r"(\d{2})$")
        .astype(float)
        .astype("Int64")
    )

    # Ajustar a rango 0-23 para construir fecha_hora válida
    df_long["hora"] = df_long["hora_origen"] - 1

    df_long["fecha_hora"] = (
        df_long[date_col] +
        pd.to_timedelta(df_long["hora"], unit="h")
    )

    df_long = df_long.drop(columns=["hora_col", "hora_origen"])

    df_long = df_long.dropna(subset=["fecha_hora", "precio_bolsa"]).copy()
    df_long = df_long.sort_values("fecha_hora").reset_index(drop=True)

    df_long["fecha"] = df_long["fecha_hora"].dt.date

    return df_long


def prepare_pb_nal_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el flujo completo de preparación del archivo .csv:
        Transformación a formato largo
        Limpieza
        Ordenamiento temporal.

    Parámetros
    ----------
    df_raw : pd.DataFrame
        DataFrame original leído desde la fuente .csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame final listo para EDA, feature engineering y modelado.
    """

    df = transform_pb_nal_wide_to_long(df_raw)

    df = clean_price_dataframe(
        df,
        datetime_col="fecha_hora",
        price_col="precio_bolsa"
    )

    df = add_basic_time_columns(
        df,
        datetime_col="fecha_hora"
    )

    return df

#----------------------------------------------------
'''
def agregar_serie_temporal(
    df,
    fecha_col="fecha_hora",
    valor_col="precio_bolsa",
    frecuencia="M",
    agregacion="mean",
    nombre_salida=None
):
    """
    Agrega una serie temporal a una frecuencia específica.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame original.

    fecha_col : str
        Nombre de la columna de fecha.

    valor_col : str
        Nombre de la columna numérica a agregar.

    frecuencia : str
        Frecuencia temporal:
            - "h" : Horaria
            - "D" : Diaria
            - "M" : Mensual

    agregacion : str
        Tipo de agregación:
            - "mean"
            - "sum"
            - "max"
            - "min"
            - etc.

    nombre_salida : str, opcional
        Nombre personalizado para la columna resultado.
        Si no se proporciona, se genera automáticamente.

    Retorna
    -------
    pandas.DataFrame
        Serie temporal agregada.
    """

    # Copia para evitar modificar original
    df_out = df.copy()

    # Convertir fecha
    df_out[fecha_col] = pd.to_datetime(
        df_out[fecha_col],
        errors="coerce"
    )

    # Convertir valores numéricos
    df_out[valor_col] = pd.to_numeric(
        df_out[valor_col],
        errors="coerce"
    )

    # Eliminar nulos
    df_out = df_out.dropna(
        subset=[fecha_col, valor_col]
    )

    # Validar frecuencia
    if frecuencia not in ["h", "D", "M"]:
        raise ValueError(
            "Frecuencia no válida. Use: 'h', 'D' o 'M'"
        )

    # Crear columna temporal agregada
    df_out["FECHA"] = (
        df_out[fecha_col]
        .dt.to_period(frecuencia)
        .dt.to_timestamp()
    )

    # Definir nombre de salida
    if nombre_salida is None:
        nombre_resultado = (
            f"{valor_col}_{agregacion}_{frecuencia}"
        )
    else:
        nombre_resultado = nombre_salida

    # Agrupación
    df_agg = (
        df_out
        .groupby("FECHA")[valor_col]
        .agg(**{nombre_resultado: agregacion})
        .reset_index()
    )

    return df_agg
'''

#---------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------
#'''
def agregar_serie_temporal(
    df,
    fecha_col="fecha_hora",
    valor_col="precio_bolsa",
    frecuencia="M",
    agregacion="mean",
    nombre_salida=None,
    hora_separada="si"
):

    df_out = df.copy()

    # ==========================
    # CONVERSIÓN TIPOS
    # ==========================
    df_out[fecha_col] = pd.to_datetime(
        df_out[fecha_col],
        errors="coerce"
    )

    df_out[valor_col] = pd.to_numeric(
        df_out[valor_col],
        errors="coerce"
    )

    df_out = df_out.dropna(
        subset=[fecha_col, valor_col]
    )

    # ==========================
    # VALIDACIONES
    # ==========================
    if frecuencia not in ["h", "D", "M"]:
        raise ValueError(
            "Frecuencia no válida. Use: 'h', 'D' o 'M'"
        )

    if hora_separada not in ["si", "no"]:
        raise ValueError(
            "hora_separada debe ser 'si' o 'no'"
        )

    # Nombre salida
    nombre_resultado = (
        nombre_salida
        if nombre_salida is not None
        else f"{valor_col}_{agregacion}_{frecuencia}"
    )

    # ==========================
    # HORARIA
    # ==========================
    if frecuencia == "h":

        df_out["datetime"] = df_out[fecha_col].dt.floor("h")

        # ==========================
        # HORA SEPARADA
        # ==========================
        if hora_separada == "si":

            df_out["FECHA"] = df_out["datetime"].dt.strftime("%Y-%m-%d")
            df_out["hora"] = df_out["datetime"].dt.hour

            df_agg = (
                df_out
                .groupby(["FECHA", "hora"])[valor_col]
                .agg(**{nombre_resultado: agregacion})
                .reset_index()
            )

            return df_agg

        # ==========================
        # FECHA Y HORA JUNTAS
        # ==========================
        else:

            df_out["FECHA"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

            df_agg = (
                df_out
                .groupby("FECHA")[valor_col]
                .agg(**{nombre_resultado: agregacion})
                .reset_index()
            )

            return df_agg

    # ==========================
    # DIARIA / MENSUAL
    # ==========================
    else:

        df_out["FECHA"] = (
            df_out[fecha_col]
            .dt.to_period(frecuencia)
            .dt.to_timestamp()
        )

        df_agg = (
            df_out
            .groupby("FECHA")[valor_col]
            .agg(**{nombre_resultado: agregacion})
            .reset_index()
        )

        df_agg["FECHA"] = df_agg["FECHA"].dt.strftime("%Y-%m-%d")

        return df_agg

#'''
#-----------------------------------------------------------------------------------------------------------------------------

def cargar_embalses_MDh(
    path: str,
    frecuencia: str = "M",
    hora_separada: str = "si"
) -> pd.DataFrame:
    """
    Carga y transforma la serie de embalses
    a frecuencia mensual, diaria u horaria.

    Parámetros
    ----------
    path : str
        Ruta del archivo .csv.

    frecuencia : str, opcional
        Frecuencia deseada:

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
    df = pd.read_csv(path)

    # ==========================
    # LIMPIEZA
    # ==========================
    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce"
    )

    df["VolumenPorcentaje"] = pd.to_numeric(
        df["VolumenPorcentaje"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Fecha", "VolumenPorcentaje"]
    ).copy()

    df = (
        df
        .sort_values("Fecha")
        .reset_index(drop=True)
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
    # AGREGACIÓN MENSUAL
    # ==========================
    df["FECHA"] = (
        df["Fecha"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df_monthly = (
        df
        .groupby("FECHA")
        .agg(
            embalse_promedio_mensual=(
                "VolumenPorcentaje",
                "mean"
            ),
            embalse_fin_mes=(
                "VolumenPorcentaje",
                "last"
            )
        )
        .reset_index()
    )

    # ==========================
    # MENSUAL
    # ==========================
    if frecuencia == "M":

        df_monthly["FECHA"] = (
            df_monthly["FECHA"]
            .dt.strftime("%Y-%m-%d")
        )

        return df_monthly

    # ==========================
    # DIARIA
    # ==========================
    elif frecuencia == "D":

        registros = []

        for _, row in df_monthly.iterrows():

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
                "embalse_promedio_mensual":
                    row["embalse_promedio_mensual"],
                "embalse_fin_mes":
                    row["embalse_fin_mes"]
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

        for _, row in df_monthly.iterrows():

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
                "embalse_promedio_mensual":
                    row["embalse_promedio_mensual"],
                "embalse_fin_mes":
                    row["embalse_fin_mes"]
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
                [
                    "FECHA",
                    "hora",
                    "embalse_promedio_mensual",
                    "embalse_fin_mes"
                ]
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
                [
                    "FECHA",
                    "embalse_promedio_mensual",
                    "embalse_fin_mes"
                ]
            ]

            return df_hourly