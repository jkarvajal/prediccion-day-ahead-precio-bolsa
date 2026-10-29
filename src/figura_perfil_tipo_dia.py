# -*- coding: utf-8 -*-
"""
Genera la figura del perfil horario promedio del precio de bolsa segun el tipo
de dia (habil, sabado, domingo, festivo), capitulo 2. Sustenta la afirmacion de
que los dias habiles muestran el perfil mas pronunciado y los domingos y
festivos uno mas bajo y aplanado.

Datos: data/processed/features_day_ahead.csv (columnas h00..h23 y base_tipo_dia),
la misma fuente que consume el dashboard.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = RAIZ / "data" / "processed" / "features_day_ahead.csv"
FIGURAS_DIR = RAIZ / "reports" / "figures"
MEDIA_LATEX = Path(r"D:\Claude\Paper Latex\workspace\media")

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
PICO_INI, PICO_FIN = 17.5, 21.5

# Un color por tipo de dia; el habil se resalta (mas grueso).
ESTILOS = {
    "habil":   ("Hábil",   "#6b3fa0", "-",   2.6),
    "sabado":  ("Sábado",  "#2a9d8f", "-",   2.0),
    "domingo": ("Domingo", "#e76f51", "--",  2.0),
    "festivo": ("Festivo", "#577590", "-.",  2.0),
}


def generar() -> Path:
    df = pd.read_csv(FEATURES, usecols=["base_tipo_dia"] + HOUR_COLS)
    perfil = df.groupby("base_tipo_dia")[HOUR_COLS].mean()
    horas = list(range(24))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(PICO_INI, PICO_FIN, color="crimson", alpha=0.07, zorder=0)

    for tipo, (etiqueta, color, estilo, grosor) in ESTILOS.items():
        if tipo in perfil.index:
            y = [perfil.loc[tipo, c] for c in HOUR_COLS]
            ax.plot(horas, y, color=color, lw=grosor, ls=estilo, marker="o",
                    ms=3.5, label=etiqueta, zorder=3)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Precio de bolsa ($/kWh)")
    ax.set_title("Perfil horario promedio del precio por tipo de día")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="crimson", alpha=0.12))
    labels.append("Franja pico (18 a 21 h)")
    ax.legend(handles, labels, fontsize=8.5, ncol=3, loc="upper left")

    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS_DIR / "cap2_perfil_tipo_dia.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX.exists():
        fig.savefig(MEDIA_LATEX / "cap2_perfil_tipo_dia.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta)
    return ruta


if __name__ == "__main__":
    generar()
