# -*- coding: utf-8 -*-
"""
Compromiso MAE-RMSE entre las estrategias (sin Prophet), figura del capitulo 3.
Lee el consolidado de estrategias y solo grafica; no recalcula.

Datos: data/outputs/consolidado_estrategias.csv.
Paleta Okabe-Ito (config.py): la referencia (persistencia) se destaca en bermellon.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (OUTPUTS_DIR, FIGURES_DIR, MEDIA_LATEX_DIR,
                    COLOR_PRIMARIO, COLOR_RESALTADO, COLOR_NEUTRO)

# Etiqueta y desplazamiento del texto por estrategia (para evitar solapes).
ETIQUETAS = {
    "Persistencia diaria": ("Persistencia diaria (ref.)", (0, 12), "center"),
    "Hibrido con correccion en horas pico": ("Híbrido (picos)", (-8, -16), "left"),
    "Aprendizaje residual": ("Aprendizaje residual", (-8, 12), "left"),
    "Segmentacion por tipo de dia": ("Segmentación tipo de día", (0, -16), "center"),
    "Lasso horario": ("Lasso horario", (0, 12), "center"),
    "Modelo de dos etapas": ("Modelo de dos etapas", (10, -6), "left"),
    "LSTM": ("LSTM", (10, 2), "left"),
}


def generar():
    df = pd.read_csv(OUTPUTS_DIR / "consolidado_estrategias.csv")
    df = df[df["estrategia"] != "Prophet"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, fila in df.iterrows():
        ref = fila["estrategia"] == "Persistencia diaria"
        ax.scatter(fila["MAE"], fila["RMSE"],
                   color=COLOR_RESALTADO if ref else COLOR_PRIMARIO,
                   s=160 if ref else 90, zorder=3)
        etq, (dx, dy), ha = ETIQUETAS.get(
            fila["estrategia"], (fila["estrategia"], (0, 12), "center"))
        ax.annotate(etq, (fila["MAE"], fila["RMSE"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9,
                    color=COLOR_RESALTADO if ref else COLOR_NEUTRO,
                    fontweight="bold" if ref else "normal")

    ax.set_xlabel("MAE ($/kWh) - menor es mejor")
    ax.set_ylabel("RMSE ($/kWh) - menor es mejor")
    ax.set_title("Compromiso MAE-RMSE entre estrategias (sin Prophet)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "cap3_comparacion_estrategias.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX_DIR.exists():
        fig.savefig(MEDIA_LATEX_DIR / "cap3_comparacion_estrategias.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


if __name__ == "__main__":
    generar()
