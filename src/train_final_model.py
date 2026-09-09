"""
Entrena y persiste el modelo final de predicción day-ahead (24 horas)
del precio de bolsa, replicando la configuración ganadora del notebook
07_modelos_horarios_dayahead (Lasso con variables exógenas, un modelo
por hora con selección de variables específica por hora).

Genera en la carpeta models/:
- modelo_final_dayahead.joblib : dict {hora: (pipeline, lista_features)}
- modelo_final_metadata.json   : metadatos (fecha, métricas, features)

Uso:
    python -m src.train_final_model
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR, PROCESSED_DIR

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [f"target_h{h:02d}" for h in range(24)]

LEAKAGE_PREFIXES = [
    "target_h", "delta_target_", "shape_target_",
    "target_promedio_diario", "target_max_diario",
    "target_min_diario", "target_std_diario",
    "evento_pico_p90_target", "evento_pico_p95_target",
]

NON_FEATURE_COLS = ["FECHA", "fecha_objetivo", "base_tipo_dia", "transicion_tipo_dia"]


def is_leakage(col: str) -> bool:
    return any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES)


def get_global_features(df: pd.DataFrame) -> list:
    """Variables globales (no específicas de una hora), incluyendo exógenas."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not is_leakage(c)
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(c.startswith(f"h{h:02d}") for h in range(24))
        and not any(c.startswith(f"shape_base_h{h:02d}") for h in range(24))
    ]


def get_hour_features(df: pd.DataFrame, hour: int, global_features: list) -> list:
    """Variables para predecir la hora `hour`: precio de esa hora, rezagos y globales."""
    h_str = f"h{hour:02d}"
    hour_specific = [
        c for c in df.columns
        if c in (
            h_str,
            f"{h_str}_lag_1d",
            f"{h_str}_lag_2d",
            f"{h_str}_lag_7d",
            f"{h_str}_lag_14d",
            f"{h_str}_diff_vs_7d",
            f"shape_base_{h_str}",
        )
        and not is_leakage(c)
    ]
    return sorted(set(hour_specific + global_features))


def train_final_model() -> None:
    features_path = PROCESSED_DIR / "features_day_ahead.csv"
    print(f"Cargando {features_path} ...")

    df = pd.read_csv(features_path)
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df = df.sort_values("FECHA").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS, how="all").reset_index(drop=True)

    print(f"Observaciones: {len(df)} | Rango: {df['FECHA'].min()} -> {df['FECHA'].max()}")

    global_features = get_global_features(df)
    models = {}

    for h in range(24):
        col = f"target_h{h:02d}"
        features = get_hour_features(df, h, global_features)

        y = df[col]
        valid = ~y.isna()

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1.0, max_iter=5000)),
        ])
        pipe.fit(df.loc[valid, features], y[valid])
        models[h] = {"pipeline": pipe, "features": features}
        print(f"  Hora {h:02d}: entrenado con {valid.sum()} obs, {len(features)} variables")

    model_path = MODELS_DIR / "modelo_final_dayahead.joblib"
    joblib.dump(models, model_path)

    metadata = {
        "modelo": "Lasso_horario_con_exogenas",
        "descripcion": "24 modelos Lasso independientes (uno por hora), seleccionado en NB07",
        "fecha_entrenamiento": datetime.now().isoformat(),
        "datos_hasta": str(df["FECHA"].max().date()),
        "n_observaciones": int(len(df)),
        "metricas_test_nb07": {
            "MAE": 77.20,
            "RMSE": 154.71,
            "sMAPE": 18.91,
            "R2": 0.8349,
            "nota": "Evaluado en NB07 con particion temporal 80/20 (test: 2022-12 a 2026-03)",
        },
        "n_features_por_hora": {str(h): len(m["features"]) for h, m in models.items()},
    }

    metadata_path = MODELS_DIR / "modelo_final_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en:   {model_path}")
    print(f"Metadatos guardados:  {metadata_path}")


if __name__ == "__main__":
    train_final_model()
