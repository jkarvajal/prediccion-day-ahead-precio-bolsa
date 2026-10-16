# -*- coding: utf-8 -*-
"""
Validador de coherencia: compara las cifras afirmadas en el documento de grado contra
los resultados reales que produce el código (archivos de data/outputs y data/processed
y los metadatos del modelo). Imprime un informe por objetivo con el estado de cada
afirmación y guarda un JSON (data/outputs/validacion_coherencia.json) que consume la
herramienta visual del ciclo del proyecto.

Uso:
    python src/validar_coherencia.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "outputs"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

resultados = []


def check(objetivo, afirmacion, valor_doc, valor_real, ok, fuente):
    resultados.append({
        "objetivo": objetivo, "afirmacion": afirmacion,
        "valor_documento": str(valor_doc), "valor_real": str(valor_real),
        "coherente": bool(ok), "fuente": fuente,
    })


def aprox(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ----------------------------- Objetivo 1 --------------------------------- #
precio = pd.read_csv(PROC / "precio_horario_procesado.csv")
check("1. Construir", "Serie horaria depurada: 142.344 registros",
      "142.344", f"{len(precio):,}".replace(",", "."),
      len(precio) == 142344, "precio_horario_procesado.csv")

feats = pd.read_csv(PROC / "features_day_ahead.csv")
feats["FECHA"] = pd.to_datetime(feats["FECHA"], errors="coerce")
check("1. Construir", "Base supervisada: 5.930 días",
      "5.930", f"{feats.shape[0]:,}".replace(",", "."),
      feats.shape[0] == 5930, "features_day_ahead.csv")
check("1. Construir", "Base supervisada: 333 columnas",
      "333", feats.shape[1], feats.shape[1] == 333, "features_day_ahead.csv")
rango = f"{feats['FECHA'].min().year}-{feats['FECHA'].max().year}"
check("1. Construir", "Periodo de la serie: 2010-2026",
      "2010-2026", rango, rango == "2010-2026", "features_day_ahead.csv")


# ----------------------------- Objetivo 2 --------------------------------- #
perfil = pd.read_csv(OUT / "eda_perfil_horario.csv")
hora_max = int(perfil.loc[perfil["precio_promedio"].idxmax(), "hora"])
check("2. Caracterizar", "Hora de precio máximo: 19 h",
      "19", hora_max, hora_max == 19, "eda_perfil_horario.csv")

pers = pd.read_csv(OUT / "eda_persistencia_hoy_vs_manana.csv")
cmin = round(pers["correlacion_hoy_vs_manana"].min(), 2)
cmax = round(pers["correlacion_hoy_vs_manana"].max(), 2)
check("2. Caracterizar", "Autocorrelación diaria entre 0,92 y 0,96",
      "0,92 - 0,96", f"{cmin} - {cmax}",
      cmin == 0.92 and cmax == 0.96, "eda_persistencia_hoy_vs_manana.csv")

ev = pd.read_csv(OUT / "eda_resumen_eventos_extremos.csv").set_index("umbral")
check("2. Caracterizar", "Umbral p95 cercano a 797 ($/kWh)",
      "~797", round(ev.loc["p95", "valor_umbral"]),
      aprox(ev.loc["p95", "valor_umbral"], 797, 2), "eda_resumen_eventos_extremos.csv")
check("2. Caracterizar", "Eventos sobre p95: 7.120 (5,0%)",
      "7.120", f"{int(ev.loc['p95', 'n_eventos']):,}".replace(",", "."),
      int(ev.loc["p95", "n_eventos"]) == 7120, "eda_resumen_eventos_extremos.csv")
check("2. Caracterizar", "Eventos sobre p99: 1.433 (1,0%)",
      "1.433", f"{int(ev.loc['p99', 'n_eventos']):,}".replace(",", "."),
      int(ev.loc["p99", "n_eventos"]) == 1433, "eda_resumen_eventos_extremos.csv")

corr = pd.read_csv(OUT / "eda_correlacion_oni_rezagado_embalses.csv").set_index("rezago_dias")
c0 = round(corr.loc[0, "correlacion_oni_embalses"], 2)
c180 = round(corr.loc[180, "correlacion_oni_embalses"], 2)
check("2. Caracterizar", "Corr. ONI-embalses: -0,20 (contemporánea) a -0,49 (180 días)",
      "-0,20 -> -0,49", f"{c0} -> {c180}",
      c0 == -0.20 and c180 == -0.49, "eda_correlacion_oni_rezagado_embalses.csv")

adf = pd.read_csv(OUT / "eda_test_estacionariedad_adf.csv")
diaria = adf[adf["serie"] == "Precio promedio diario"].iloc[0]
check("2. Caracterizar", "Precio diario estacionario en nivel (ADF)",
      "Estacionaria", diaria["estacionaria_5%"],
      str(diaria["estacionaria_5%"]).strip().lower().startswith("s"),
      "eda_test_estacionariedad_adf.csv")

# Correlación del precio con exógenas (ONI y embalses)
hcols = [f"h{h:02d}" for h in range(24)]
cf = feats.dropna(subset=["FECHA"]).copy()
cf["precio_prom"] = cf[hcols].mean(axis=1)
if "oni" in cf.columns:
    r_oni = round(cf[["precio_prom", "oni"]].dropna().corr().iloc[0, 1], 2)
    check("2. Caracterizar", "Correlación precio-ONI cercana a 0,45",
          "~0,45", r_oni, aprox(r_oni, 0.45, 0.07), "features_day_ahead.csv")
if "embalse_porcentaje_diario" in cf.columns:
    r_emb = round(cf[["precio_prom", "embalse_porcentaje_diario"]].dropna().corr().iloc[0, 1], 2)
    check("2. Caracterizar", "Correlación precio-embalses cercana a -0,17",
          "~-0,17", r_emb, aprox(r_emb, -0.17, 0.07), "features_day_ahead.csv")


# ----------------------------- Objetivo 3 --------------------------------- #
cons = pd.read_csv(OUT / "consolidado_estrategias.csv")
check("3. Evaluar", "Ocho estrategias evaluadas",
      "8", len(cons), len(cons) == 8, "consolidado_estrategias.csv")

pers_row = cons[cons["estrategia"].str.contains("Persistencia", case=False)].iloc[0]
check("3. Evaluar", "Persistencia MAE 68,9",
      "68,9", round(pers_row["MAE"], 1), aprox(pers_row["MAE"], 68.9, 0.1),
      "consolidado_estrategias.csv")
check("3. Evaluar", "Persistencia RMSE 157,9",
      "157,9", round(pers_row["RMSE"], 1), aprox(pers_row["RMSE"], 157.9, 0.1),
      "consolidado_estrategias.csv")
mae_min = cons["MAE"].min()
check("3. Evaluar", "Ninguna estrategia supera a la persistencia en MAE",
      "Persistencia = MAE mínimo",
      cons.loc[cons["MAE"].idxmin(), "estrategia"],
      aprox(mae_min, pers_row["MAE"], 1e-6), "consolidado_estrategias.csv")

recal = pd.read_csv(OUT / "recalibracion_resultados.csv")
rdir = recal[recal["modelo"].str.contains("directo", case=False)].iloc[0]
check("3. Evaluar", "Lasso recalibrado (directo) RMSE 141,4",
      "141,4", round(rdir["RMSE"], 1), aprox(rdir["RMSE"], 141.4, 0.2),
      "recalibracion_resultados.csv")
check("3. Evaluar", "Lasso recalibrado (directo) MAE ~69,4 (iguala a persistencia)",
      "69,4", round(rdir["MAE"], 1), aprox(rdir["MAE"], 69.4, 0.3),
      "recalibracion_resultados.csv")

reg = pd.read_csv(OUT / "consolidado_por_regimen.csv")
reg_v = reg[reg["regimen"].str.contains("volatil", case=False)].set_index("estrategia")["MAE"]
lasso_v = round(reg_v.get("Lasso horario", np.nan), 1)
pers_v = round(reg_v.get("Persistencia diaria", np.nan), 1)
check("3. Evaluar", "Régimen volátil: Lasso (88,2) supera a persistencia (90,4)",
      "Lasso 88,2 < Persist. 90,4", f"Lasso {lasso_v} < Persist. {pers_v}",
      lasso_v < pers_v, "consolidado_por_regimen.csv")


# ----------------------------- Objetivo 4 --------------------------------- #
with open(MODELS / "modelo_final_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
mt = meta["metricas_test_nb07"]
check("4. Comunicar", "Modelo final: 24 Lasso por hora",
      "24", len(meta["n_features_por_hora"]),
      len(meta["n_features_por_hora"]) == 24, "modelo_final_metadata.json")
check("4. Comunicar", "Lasso final MAE 77,2",
      "77,2", mt["MAE"], aprox(mt["MAE"], 77.2, 0.1), "modelo_final_metadata.json")
check("4. Comunicar", "Lasso final RMSE 154,7",
      "154,7", round(mt["RMSE"], 1), aprox(mt["RMSE"], 154.7, 0.1),
      "modelo_final_metadata.json")
check("4. Comunicar", "Lasso final R2 0,835",
      "0,835", round(mt["R2"], 3), aprox(mt["R2"], 0.835, 0.001),
      "modelo_final_metadata.json")


# ------------------------------- Informe ---------------------------------- #
def main():
    n_ok = sum(r["coherente"] for r in resultados)
    n_tot = len(resultados)
    print("=" * 78)
    print("VALIDACIÓN DE COHERENCIA DOCUMENTO <-> CÓDIGO")
    print("=" * 78)
    obj_actual = None
    for r in resultados:
        if r["objetivo"] != obj_actual:
            obj_actual = r["objetivo"]
            print(f"\n## Objetivo {obj_actual}")
        marca = "[OK] " if r["coherente"] else "[XX] "
        print(f"  {marca}{r['afirmacion']}")
        print(f"        doc: {r['valor_documento']}  |  codigo: {r['valor_real']}  "
              f"({r['fuente']})")
    print("\n" + "=" * 78)
    print(f"RESULTADO: {n_ok}/{n_tot} afirmaciones coherentes "
          f"({'TODO COHERENTE' if n_ok == n_tot else 'HAY DISCREPANCIAS'})")
    print("=" * 78)

    salida = OUT / "validacion_coherencia.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump({"total": n_tot, "coherentes": n_ok, "items": resultados},
                  f, ensure_ascii=False, indent=2)
    print("JSON guardado en:", salida)


if __name__ == "__main__":
    main()
