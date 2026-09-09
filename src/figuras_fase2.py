# -*- coding: utf-8 -*-
"""
Genera las figuras que acompanan el capitulo de resultados por objetivo
especifico de la Fase 2. Las figuras se guardan en reports/figures/ con el
mismo estilo grafico de las ya incorporadas al documento.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUTS_DIR

FIGURAS_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

AZUL = "#1f77b4"
NARANJA = "#ff7f0e"
GRIS = "#7f7f7f"
PEAK_HOURS = [18, 19, 20]


def _guardar(fig, nombre: str) -> Path:
    ruta = FIGURAS_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


def figura_comparacion() -> Path:
    """Comparacion consolidada de estrategias por MAE y RMSE."""
    df = pd.read_csv(OUTPUTS_DIR / "consolidado_estrategias.csv").sort_values("MAE")
    etiquetas = [e.replace("Hibrido con correccion en horas pico", "Híbrido en horas pico")
                  .replace("Segmentacion por tipo de dia", "Segmentación por tipo de día")
                  .replace("Modelo de dos etapas", "Dos etapas")
                  .replace("Aprendizaje residual", "Aprendizaje residual")
                  .replace("Persistencia diaria", "Persistencia diaria")
                 for e in df["estrategia"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metrica, titulo in [(axes[0], "MAE", "Error absoluto medio"),
                                (axes[1], "RMSE", "Raíz del error cuadrático medio")]:
        orden = df.sort_values(metrica)
        etq = [etiquetas[list(df["estrategia"]).index(e)] for e in orden["estrategia"]]
        colores = [NARANJA if e == "Persistencia diaria" else AZUL
                   for e in orden["estrategia"]]
        ax.barh(range(len(orden)), orden[metrica], color=colores)
        ax.set_yticks(range(len(orden)))
        ax.set_yticklabels(etq, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(titulo)
        ax.set_title(titulo)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Comparación consolidada de estrategias bajo protocolo homogéneo")
    fig.tight_layout()
    return _guardar(fig, "fase2_comparacion_estrategias.png")


def figura_por_hora() -> Path:
    """Error absoluto medio por hora del dia, destacando las horas pico."""
    df = pd.read_csv(OUTPUTS_DIR / "persistencia_metricas_por_hora.csv")
    colores = [NARANJA if h in PEAK_HOURS else AZUL for h in df["hora"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["hora"], df["MAE"], color=colores)
    ax.axhline(df["MAE"].mean(), color=GRIS, linestyle="--", linewidth=1.2,
               label=f"Promedio del día ({df['MAE'].mean():.1f})")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Error absoluto medio")
    ax.set_title("Error absoluto medio por hora del día")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Identificacion de las horas pico
    handles = [plt.Rectangle((0, 0), 1, 1, color=NARANJA)]
    ax.legend(handles + [plt.Line2D([0], [0], color=GRIS, linestyle="--")],
              ["Horas pico (18, 19 y 20)",
               f"Promedio del día ({df['MAE'].mean():.1f})"])
    fig.tight_layout()
    return _guardar(fig, "fase2_mae_por_hora.png")


def figura_regimen() -> Path:
    """Desempeno comparado en regimen de precio calmo y volatil."""
    df = pd.read_csv(OUTPUTS_DIR / "consolidado_por_regimen.csv")
    pivote = df.pivot(index="estrategia", columns="regimen", values="MAE")
    pivote = pivote[pivote.index != "Prophet"]  # escala no comparable
    pivote = pivote.sort_values("volatil (>=300)")

    etiquetas = [e.replace("Hibrido con correccion en horas pico", "Híbrido en horas pico")
                  .replace("Segmentacion por tipo de dia", "Segmentación por tipo de día")
                  .replace("Modelo de dos etapas", "Dos etapas")
                 for e in pivote.index]

    x = range(len(pivote))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - ancho / 2 for i in x], pivote["calmo (<300)"], ancho,
           label="Régimen calmo (menor a 300)", color=AZUL)
    ax.bar([i + ancho / 2 for i in x], pivote["volatil (>=300)"], ancho,
           label="Régimen volátil (300 o más)", color=NARANJA)
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Error absoluto medio")
    ax.set_title("Desempeño comparado según régimen de precio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, "fase2_regimen.png")


def main() -> None:
    figura_comparacion()
    figura_por_hora()
    figura_regimen()


if __name__ == "__main__":
    main()
