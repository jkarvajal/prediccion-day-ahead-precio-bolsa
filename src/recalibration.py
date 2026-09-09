"""
Pronóstico day-ahead con recalibración de ventana expansiva.

Para cada bloque temporal del periodo de prueba (mensual o semanal), entrena
24 modelos Lasso , uno por hora,  sobre todo el histórico disponible hasta el
inicio del bloque y predice sus 24 valores horarios. Esto reproduce la práctica
estándar de la literatura de pronóstico de precios eléctricos (recalibración
periódica), que permite al modelo adaptarse a los cambios de régimen del precio.

Dos modos:
- "direct":   el modelo predice directamente el precio horario del día siguiente.
- "residual": el modelo predice la desviación frente a la persistencia
              (target - precio_de_hoy) y el pronóstico se reconstruye sumando
              esa desviación al precio de hoy.

Uso:
    from src.recalibration import rolling_recalibrated_forecast
    pred = rolling_recalibrated_forecast(df, test_start, freq="M", mode="residual")
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train_final_model import get_global_features, get_hour_features, TARGET_COLS


def rolling_recalibrated_forecast(
    df: pd.DataFrame,
    test_start,
    freq: str = "M",
    mode: str = "residual",
    alpha: float = 1.0,
    min_train: int = 200,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Genera pronósticos day-ahead recalibrando el modelo por bloques temporales.

    Parámetros
    ----------
    df : DataFrame
        Base day-ahead ordenada por FECHA, con columnas h00..h23, target_h00..h23,
        FECHA y fecha_objetivo.
    test_start : fecha
        Primera fecha del periodo de prueba (todo lo anterior es histórico inicial).
    freq : str
        Frecuencia de recalibración: "M" (mensual) o "W" (semanal).
    mode : str
        "direct" o "residual".
    alpha : float
        Parámetro de regularización del Lasso.
    min_train : int
        Mínimo de observaciones de entrenamiento para ajustar un bloque.
    verbose : bool
        Imprime avance por bloque.

    Retorna
    -------
    DataFrame largo con columnas: FECHA, fecha_objetivo, hora, real, pred, base.
    """
    df = df.sort_values("FECHA").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLS, how="all").reset_index(drop=True)

    global_features = get_global_features(df)
    feat_by_hour = {
        h: [f for f in get_hour_features(df, h, global_features) if f in df.columns]
        for h in range(24)
    }

    test = df[df["FECHA"] >= pd.Timestamp(test_start)].copy()
    test["_blk"] = test["FECHA"].dt.to_period(freq)
    blocks = sorted(test["_blk"].unique())

    records = []
    for blk in blocks:
        block = test[test["_blk"] == blk]
        block_start = block["FECHA"].min()
        train = df[df["FECHA"] < block_start]
        if len(train) < min_train:
            continue

        for h in range(24):
            tcol = f"target_h{h:02d}"
            bcol = f"h{h:02d}"
            feats = feat_by_hour[h]

            if mode == "residual":
                y = train[tcol] - train[bcol]
            else:
                y = train[tcol]

            valid = ~y.isna()
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, max_iter=5000)),
            ])
            pipe.fit(train[feats].values[valid.values], y.values[valid.values])
            raw = pipe.predict(block[feats].values)
            pred = block[bcol].values + raw if mode == "residual" else raw

            records.append(pd.DataFrame({
                "FECHA": block["FECHA"].values,
                "fecha_objetivo": block["fecha_objetivo"].values,
                "hora": h,
                "real": block[tcol].values,
                "pred": pred,
                "base": block[bcol].values,
            }))

        if verbose:
            print(f"  bloque {blk} entrenado ({len(train)} obs)")

    return pd.concat(records, ignore_index=True).dropna(subset=["real", "pred", "base"])
