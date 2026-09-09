"""
Contiene funciones para la carga de datos desde archivos fuente,
como CSV, y posteriormente podrá ampliarse para incorporar la
adquisición desde la API de XM sin afectar el resto del pipeline.
"""
from pathlib import Path
import pandas as pd


def load_csv(path_csv, **kwargs):
    """
    Carga un archivo CSV y retorna un DataFrame.

    Parámetros
    ----------
    path_csv : str o Path
        Ruta del CSV.
    kwargs :
        Argumentos adicionales para pd.read_csv().

    Retorna
    -------
    pd.DataFrame
    """
    path_csv = Path(path_csv)

    if not path_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")

    return pd.read_csv(path_csv, **kwargs)