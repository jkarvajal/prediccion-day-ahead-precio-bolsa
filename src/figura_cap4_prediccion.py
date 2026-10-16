# -*- coding: utf-8 -*-
"""
Genera la figura de la vista de prediccion del dia siguiente del entorno web
(capitulo 4), fiel al dashboard: las tres estrategias (persistencia, Lasso y
aprendizaje residual) frente al precio real observado, con la banda del
intervalo conformal al 90 % del modelo Lasso y la franja pico resaltada.

Usa datos reales de un dia de prueba (dia base 2026-03-27, dia pronosticado
2026-03-28), los mismos outputs que consume la aplicacion:
  - hourly_dayahead_predictions.csv   (real y prediccion Lasso)
  - recalibracion_predicciones_residual.csv (persistencia y aprendizaje residual)
  - intervalos_conformales.csv        (banda conformal q_low, q_high por hora)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = RAIZ / "data" / "outputs"
FIGURAS_DIR = RAIZ / "reports" / "figures"
# Carpeta de imagenes del documento LaTeX (segunda copia sincronizada).
MEDIA_LATEX = Path(r"D:\Claude\Paper Latex\workspace\media")

# Colores del dashboard (app/app.py).
AZUL = "#2563eb"      # Persistencia diaria (referencia)
VERDE = "#16a34a"     # Modelo Lasso (comparacion)
NARANJA = "#ea580c"   # Aprendizaje residual (comparacion)
NEGRO = "#111111"     # Precio real observado
BANDA = (22 / 255, 163 / 255, 74 / 255, 0.15)  # rgba(22,163,74,0.15)

FECHA_OBJETIVO = "2026-03-28"
PICO_INI, PICO_FIN = 17.5, 22.5


def _serie_horaria(fila, prefijo):
    return [float(fila[f"{prefijo}_h{h:02d}"]) for h in range(24)]


def generar() -> Path:
    horas = list(range(24))

    # Lasso y precio real.
    hourly = pd.read_csv(OUTPUTS_DIR / "hourly_dayahead_predictions.csv")
    hourly.columns = [c.lstrip("\ufeff") for c in hourly.columns]
    fila = hourly[hourly["fecha_objetivo"] == FECHA_OBJETIVO].iloc[0]
    real = _serie_horaria(fila, "real")
    lasso = _serie_horaria(fila, "pred")

    # Persistencia (base) y aprendizaje residual (pred).
    res = pd.read_csv(OUTPUTS_DIR / "recalibracion_predicciones_residual.csv")
    res.columns = [c.lstrip("\ufeff") for c in res.columns]
    res = res[res["fecha_objetivo"] == FECHA_OBJETIVO].sort_values("hora")
    persistencia = res["base"].tolist()
    residual = res["pred"].tolist()

    # Banda conformal al 90 % del modelo Lasso: Lasso + (q_low, q_high) por hora.
    iv = pd.read_csv(OUTPUTS_DIR / "intervalos_conformales.csv")
    iv = iv[iv["modelo"] == "Lasso"].set_index("hora")
    lower = [lasso[h] + float(iv.loc[h, "q_low"]) for h in horas]
    upper = [lasso[h] + float(iv.loc[h, "q_high"]) for h in horas]

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Franja pico (18 a 21 h) y banda conformal detras de las lineas.
    ax.axvspan(PICO_INI, PICO_FIN, color="crimson", alpha=0.07, zorder=0)
    ax.fill_between(horas, lower, upper, color=BANDA, zorder=1,
                    label="Intervalo 90 % (Lasso)")

    ax.plot(horas, persistencia, color=AZUL, lw=2.4, marker="o", ms=5,
            label="Persistencia diaria (referencia)", zorder=3)
    ax.plot(horas, lasso, color=VERDE, lw=2, marker="o", ms=4,
            label="Modelo Lasso (comparación)", zorder=4)
    ax.plot(horas, residual, color=NARANJA, lw=2, marker="o", ms=4,
            label="Aprendizaje residual (comparación)", zorder=5)
    ax.plot(horas, real, color=NEGRO, lw=1.8, ls=":", marker="^", ms=4,
            label="Precio real observado", zorder=6)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Precio de bolsa ($/kWh)")
    ax.set_title("Predicción del día siguiente frente al precio real "
                 "(sábado 28 de marzo de 2026)")
    ax.grid(alpha=0.3)

    # Leyenda con la franja pico incluida.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="crimson", alpha=0.12))
    labels.append("Franja pico (18 a 21 h)")
    ax.legend(handles, labels, fontsize=8.5, ncol=2, loc="upper left")

    fig.tight_layout()

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS_DIR / "cap4_prediccion_dashboard.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX.exists():
        fig.savefig(MEDIA_LATEX / "cap4_prediccion_dashboard.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta)
    return ruta


if __name__ == "__main__":
    generar()
