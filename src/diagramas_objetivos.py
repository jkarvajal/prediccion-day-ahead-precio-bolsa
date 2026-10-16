# -*- coding: utf-8 -*-
"""
Genera dos diagramas para la presentacion de los objetivos:
  1. Sintesis de los objetivos, de ocho a cuatro.
  2. Relacion del objetivo general con los cuatro especificos (escalera de Bloom).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AZUL = "#1f4e79"
COLORES = {"G1": "#2E75B6", "G2": "#E8871E", "G3": "#3C8C4A", "G4": "#8E5AA8"}
GRIS_TXT = "#222222"


def caja(ax, x, y, w, h, texto, color, tcolor="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fs, color=tcolor, weight="bold" if bold else "normal", wrap=True)


def conector(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.8, color=color, alpha=0.85,
        connectionstyle="arc3,rad=0.0"))


# --------------------------------------------------------------------------- #
# Diagrama 1: sintesis de ocho a cuatro
# --------------------------------------------------------------------------- #

def diagrama_sintesis():
    viejos = [
        ("1. Construir y depurar la serie horaria", "G1"),
        ("2. Caracterizar el comportamiento (EDA)", "G2"),
        ("3. Diseñar variables predictoras", "G1"),
        ("4. Integrar variables exógenas", "G1"),
        ("5. Implementar y comparar modelos", "G3"),
        ("6. Evaluar el desempeño (métricas)", "G3"),
        ("7. Definir criterios de selección", "G3"),
        ("8. Preparar la visualización en línea", "G4"),
    ]
    nuevos = [
        ("OE1. Construir\n(aplicar)", "G1", 0.80),
        ("OE2. Caracterizar\n(analizar)", "G2", 0.58),
        ("OE3. Evaluar\n(evaluar)", "G3", 0.36),
        ("OE4. Comunicar\n(crear)", "G4", 0.14),
    ]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.18, 0.955, "ANTES: ocho objetivos", ha="center", fontsize=13,
            weight="bold", color=GRIS_TXT)
    ax.text(0.82, 0.955, "DESPUÉS: cuatro objetivos", ha="center", fontsize=13,
            weight="bold", color=GRIS_TXT)

    # cajas izquierda (8), en orden, coloreadas por su grupo destino
    n = len(viejos)
    y0, y1, h = 0.06, 0.90, 0.072
    ys = [y1 - i * (y1 - y0) / (n - 1) - h / 2 for i in range(n)]
    centros_izq = {}
    for (txt, g), y in zip(viejos, ys):
        caja(ax, 0.03, y, 0.30, h, txt, COLORES[g], fs=10)
        centros_izq.setdefault(g, []).append((0.33, y + h / 2))

    # cajas derecha (4)
    centros_der = {}
    for txt, g, yc in nuevos:
        caja(ax, 0.67, yc - 0.06, 0.30, 0.12, txt, COLORES[g], fs=12)
        centros_der[g] = (0.67, yc)

    # conectores por grupo
    for g, puntos in centros_izq.items():
        xd, yd = centros_der[g]
        for (xi, yi) in puntos:
            conector(ax, xi, yi, xd, yd, COLORES[g])

    ax.text(0.5, 0.005,
            "Cada objetivo nuevo agrupa objetivos anteriores relacionados, con un solo verbo y nivel creciente de Bloom.",
            ha="center", fontsize=10, style="italic", color=GRIS_TXT)
    ax.set_title("Síntesis de los objetivos específicos: de ocho a cuatro",
                 fontsize=15, weight="bold", color=AZUL, pad=16)
    fig.tight_layout()
    ruta = FIG_DIR / "objetivos_sintesis_8a4.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)


# --------------------------------------------------------------------------- #
# Diagrama 2: relacion objetivo general con los especificos (escalera Bloom)
# --------------------------------------------------------------------------- #

def diagrama_flujo():
    pasos = [
        ("OE1. Construir", "Serie horaria depurada\ny variables predictoras", "Aplicar", "G1"),
        ("OE2. Caracterizar", "Comportamiento del precio\ny aporte de exógenas", "Analizar", "G2"),
        ("OE3. Evaluar", "Desempeño de modelos\ny selección del enfoque", "Evaluar", "G3"),
        ("OE4. Comunicar", "Resultados y desempeño,\nincluido el entorno web", "Crear", "G4"),
    ]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # objetivo general (arriba, ancho)
    caja(ax, 0.10, 0.80, 0.80, 0.13,
         "OBJETIVO GENERAL\nDesarrollar un modelo de predicción del precio horario de bolsa de energía\n"
         "para el día siguiente en el mercado eléctrico colombiano, mediante ciencia de datos",
         AZUL, fs=11)

    # escalera de cuatro pasos ascendentes
    xs = [0.06, 0.29, 0.52, 0.75]
    ys = [0.14, 0.28, 0.42, 0.56]
    w, h = 0.19, 0.15
    centros = []
    for (titulo, desc, nivel, g), x, y in zip(pasos, xs, ys):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
            linewidth=1.4, edgecolor=COLORES[g], facecolor=COLORES[g], alpha=0.92))
        ax.text(x + w / 2, y + h - 0.032, titulo, ha="center", va="center",
                fontsize=11.5, color="white", weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.012, desc, ha="center", va="center",
                fontsize=8.8, color="white")
        ax.text(x + w / 2, y - 0.03, nivel, ha="center", va="center",
                fontsize=10, color=COLORES[g], weight="bold", style="italic")
        centros.append((x + w / 2, y + h))

    # flechas de progresion entre pasos
    for i in range(len(pasos) - 1):
        x0 = xs[i] + w; y0 = ys[i] + h / 2
        x1 = xs[i + 1]; y1 = ys[i + 1] + h / 2
        conector(ax, x0, y0, x1, y1, "#7f7f7f")

    # flechas de cada paso hacia el objetivo general
    for (cx, cy) in centros:
        ax.add_patch(FancyArrowPatch(
            (cx, cy), (cx, 0.80), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.3, color="#9aa0a6", alpha=0.7, linestyle=(0, (4, 3))))

    # eje de nivel cognitivo
    ax.annotate("", xy=(0.028, 0.72), xytext=(0.028, 0.12),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6))
    ax.text(0.018, 0.42, "Nivel cognitivo creciente (taxonomía de Bloom)",
            rotation=90, va="center", ha="center", fontsize=9.5, color="#555")

    ax.set_title("Relación del objetivo general con los objetivos específicos",
                 fontsize=15, weight="bold", color=AZUL, pad=16)
    ax.text(0.5, 0.02,
            "Los cuatro objetivos específicos ascienden en nivel cognitivo y convergen en el objetivo general.",
            ha="center", fontsize=10, style="italic", color=GRIS_TXT)
    fig.tight_layout()
    ruta = FIG_DIR / "objetivos_flujo_general.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)


if __name__ == "__main__":
    diagrama_sintesis()
    diagrama_flujo()
