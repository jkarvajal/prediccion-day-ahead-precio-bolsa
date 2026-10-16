# -*- coding: utf-8 -*-
"""
Entrena y serializa el modelo de APRENDIZAJE RESIDUAL para el día siguiente: por cada
hora, un Lasso aprende la corrección (residuo = precio objetivo - precio del día base)
sobre la persistencia diaria. La predicción final de una hora es:

    pred(h) = precio_del_dia_base(h) + residuo_estimado(h)

Replica la estrategia `estrategia_residual` de src/exploratory_models.py, pero entrenando
sobre todos los datos disponibles para su uso en la aplicación web.

Salida: models/modelo_residual_dayahead.joblib  -> dict {hora: {pipeline, features}}

Uso:
    python src/train_residual_model.py
"""

from pathlib import Path

import joblib
import pandas as pd

from src.config import BASE_DIR, PROCESSED_DIR
from src.exploratory_models import features_por_hora, _pipeline_lasso

MODELS_DIR = BASE_DIR / "models"
SALIDA = MODELS_DIR / "modelo_residual_dayahead.joblib"


def main():
    df = pd.read_csv(PROCESSED_DIR / "features_day_ahead.csv")
    modelos = {}
    for h in range(24):
        col, hcol = f"target_h{h:02d}", f"h{h:02d}"
        feats = [f for f in features_por_hora(df, h, incluir_exogenas=True)
                 if f in df.columns]
        residuo = df[col] - df[hcol]
        ok = residuo.notna() & df[hcol].notna()
        pipe = _pipeline_lasso()
        pipe.fit(df.loc[ok, feats], residuo[ok])
        modelos[h] = {"pipeline": pipe, "features": feats}
        print(f"  Hora {h:02d}: {int(ok.sum())} obs, {len(feats)} variables")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelos, SALIDA)
    print("Guardado:", SALIDA)


if __name__ == "__main__":
    main()
