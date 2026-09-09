"""
Incluye funciones para el análisis exploratorio de datos, como resúmenes estadísticos, perfiles temporales,
resamples y visualizaciones orientadas a caracterizar el comportamiento del precio de bolsa.
"""
import pandas as pd
import matplotlib.pyplot as plt


def get_basic_summary(df: pd.DataFrame, price_col: str = "precio_bolsa") -> pd.DataFrame:
    """
    Resumen estadístico básico de la variable precio.
    """
    return df[price_col].describe().to_frame(name="valor")


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reporte de nulos por columna.
    """
    return pd.DataFrame({
        "nulos": df.isnull().sum(),
        "porcentaje_nulos": (df.isnull().mean() * 100).round(2),
    })


def resample_summary(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora",
    price_col: str = "precio_bolsa",
    freq: str = "D",
) -> pd.DataFrame:
    """
    Genera resumen por frecuencia temporal.
    Ejemplos de freq:
    - D  : diario
    - W  : semanal
    - ME : mensual fin de mes
    - Y / YE : anual (fin de año)
    - YS  : anual (inicio de año)
    """
    serie = df.set_index(datetime_col)[price_col].sort_index()
    return serie.resample(freq).agg(["mean", "median", "min", "max", "std"])


def hourly_profile(
    df: pd.DataFrame,
    hour_col: str = "hora",
    price_col: str = "precio_bolsa",
) -> pd.DataFrame:
    """
    Perfil horario agregado.
    """
    return (
        df.groupby(hour_col)[price_col]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )


def plot_time_series(
    df: pd.DataFrame,
    datetime_col: str = "fecha_hora",
    price_col: str = "precio_bolsa",
    title: str = "Serie temporal del precio de bolsa",
    leyendax: str = "Fecha",
    leyenday: str = "Precio",
    figsize=(14, 4),
):
    """
    Grafica la serie temporal.
    """
    plt.figure(figsize=figsize)
    plt.plot(df[datetime_col], df[price_col])
    plt.title(title)
    plt.xlabel(leyendax)
    plt.ylabel(leyenday)
    plt.tight_layout()
    plt.show()


def plot_daily_mean(
    resumen_diario: pd.DataFrame,
    title: str = "Promedio diario del precio de bolsa",
    figsize=(14, 4),
):
    """
    Grafica el promedio diario a partir de un resumen diario.
    """
    plt.figure(figsize=figsize)
    plt.plot(resumen_diario.index, resumen_diario["mean"])
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Precio promedio diario")
    plt.tight_layout()
    plt.show()

def plot_monthly_mean_median(
    resumen_mensual: pd.DataFrame,
    title: str = "Comparación mensual del precio de bolsa",
    xleyenda: str = "Mes",
    yleyenda: str = "Precio (COP/MWh)",
    figsize=(14, 4),
):
    """
    Grafica media y mediana mensual.
    """
    plt.figure(figsize=figsize)
    plt.plot(resumen_mensual.index, resumen_mensual["mean"], marker="o", label="Media mensual")
    plt.plot(resumen_mensual.index, resumen_mensual["median"], marker="o", label="Mediana mensual")
    plt.title(title)
    plt.xlabel(xleyenda)
    plt.ylabel(yleyenda)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_yearly_mean_median(
    resumen_anual: pd.DataFrame,
    title: str = "Comparación anual del precio de bolsa",
    xleyenda: str = "Mes",
    yleyenda: str = "Precio (COP/MWh)",
    figsize=(10, 5),
):
    """
    Grafica media y mediana anual.
    """
    plt.figure(figsize=figsize)
    plt.plot(resumen_anual.index, resumen_anual["mean"], marker="o", label="Media anual")
    plt.plot(resumen_anual.index, resumen_anual["median"], marker="o", label="Mediana anual")
    plt.title(title)
    plt.xlabel(xleyenda)
    plt.ylabel(yleyenda)
    plt.legend()
    plt.tight_layout()
    plt.show()

