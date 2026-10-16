# -*- coding: utf-8 -*-
"""
Construye la fila de features para un dia base dado un historial de precios,
embalse y ONI. La fila resultante es compatible con features_day_ahead.csv y
puede usarse directamente con los modelos Lasso y residual entrenados.

Las columnas de leakage (target_h00..h23, delta_target_*, etc.) se dejan en NaN;
los modelos usan SimpleImputer y las ignoraran automaticamente.

Uso:
    from src.build_features_incremental import construir_fila
    row = construir_fila(fecha_base, precios_df, embalse_serie, oni_serie)
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

PEAK_HOURS = [18, 19, 20, 21, 22]
MADRUGADA_HOURS = [0, 1, 2, 3, 4, 5]
LAGS_DIAS = [1, 2, 7, 14]

TIPO_DIA_CATS = ["domingo", "festivo", "habil", "sabado"]
TRANSICION_CATS = [
    "domingo_a_festivo", "domingo_a_habil", "festivo_a_domingo",
    "festivo_a_festivo", "festivo_a_habil", "festivo_a_sabado",
    "habil_a_festivo", "habil_a_habil", "habil_a_sabado",
    "sabado_a_domingo", "sabado_a_festivo",
]


def _co_holidays():
    try:
        import holidays  # noqa: PLC0415
        return holidays.Colombia(years=range(2010, 2035))
    except ImportError:
        return set()


def _tipo_dia(d: date, co_hols) -> str:
    if d in co_hols:
        return "festivo"
    if d.weekday() == 5:
        return "sabado"
    if d.weekday() == 6:
        return "domingo"
    return "habil"


def _calendario(d: date, prefix: str, co_hols) -> dict:
    dia_anio = d.timetuple().tm_yday
    dia_semana = d.weekday()
    tipo = _tipo_dia(d, co_hols)
    return {
        f"{prefix}_anio": d.year,
        f"{prefix}_mes": d.month,
        f"{prefix}_dia": d.day,
        f"{prefix}_dia_semana": dia_semana,
        f"{prefix}_semana_anio": d.isocalendar()[1],
        f"{prefix}_trimestre": (d.month - 1) // 3 + 1,
        f"{prefix}_dia_anio": dia_anio,
        f"{prefix}_es_fin_semana": int(dia_semana >= 5),
        f"{prefix}_es_festivo": int(d in co_hols),
        f"{prefix}_tipo_dia": tipo,
        f"{prefix}_mes_sin": float(np.sin(2 * np.pi * d.month / 12)),
        f"{prefix}_mes_cos": float(np.cos(2 * np.pi * d.month / 12)),
        f"{prefix}_dia_semana_sin": float(np.sin(2 * np.pi * dia_semana / 7)),
        f"{prefix}_dia_semana_cos": float(np.cos(2 * np.pi * dia_semana / 7)),
        f"{prefix}_dia_anio_sin": float(np.sin(2 * np.pi * dia_anio / 365)),
        f"{prefix}_dia_anio_cos": float(np.cos(2 * np.pi * dia_anio / 365)),
    }


def _ewma_ultimo(serie: "pd.Series", span: int) -> float:
    if len(serie) == 0:
        return float("nan")
    return float(serie.ewm(span=span, min_periods=1).mean().iloc[-1])


def _oni_en(oni_serie: "pd.Series", d: date) -> float:
    ts = pd.Timestamp(d.replace(day=1))
    if ts in oni_serie.index:
        return float(oni_serie[ts])
    pasados = oni_serie.index[oni_serie.index <= ts]
    if len(pasados) == 0:
        return float("nan")
    return float(oni_serie[pasados[-1]])


def _embalse_en(embalse_serie: "pd.Series", d: date) -> float:
    ts = pd.Timestamp(d)
    if ts in embalse_serie.index:
        return float(embalse_serie[ts])
    pasados = embalse_serie.index[embalse_serie.index <= ts]
    if len(pasados) == 0:
        return float("nan")
    return float(embalse_serie[pasados[-1]])


def construir_fila(
    fecha_base: date,
    precios_df: "pd.DataFrame",
    embalse_serie: "pd.Series",
    oni_serie: "pd.Series",
    co_hols=None,
) -> dict:
    """
    Construye la fila de features para fecha_base.

    Parametros
    ----------
    fecha_base : date
        Dia cuyos precios se usan como entrada (h00-h23).
    precios_df : pd.DataFrame
        Columnas FECHA (datetime), h00..h23. Debe incluir al menos los
        14 dias anteriores a fecha_base para calcular lags.
    embalse_serie : pd.Series
        Indexada por Timestamp, valores de embalse_porcentaje_diario.
        Debe cubrir al menos 30 dias antes de fecha_base.
    oni_serie : pd.Series
        Indexada por Timestamp (primer dia del mes), valores de ONI.
    co_hols : set-like o None
        Festivos colombianos. Si None se carga la libreria holidays.

    Retorna
    -------
    dict con todas las columnas de features_day_ahead.csv. Las columnas de
    leakage (target_h*) tienen valor NaN.
    """
    if co_hols is None:
        co_hols = _co_holidays()

    hcols = [f"h{h:02d}" for h in range(24)]
    ts_base = pd.Timestamp(fecha_base)
    manana = fecha_base + timedelta(days=1)

    # Fila del dia base
    mask_hoy = precios_df["FECHA"] == ts_base
    if not mask_hoy.any():
        raise ValueError(f"No se encontraron precios para {fecha_base} en precios_df.")
    hoy_row = precios_df[mask_hoy].iloc[0]

    row: dict = {"FECHA": fecha_base}

    # h00-h23
    prices = np.array([float(hoy_row.get(c, float("nan"))) for c in hcols])
    for h, c in enumerate(hcols):
        row[c] = prices[h]

    # fecha_objetivo y target (desconocidos al momento de predecir)
    row["fecha_objetivo"] = manana
    for h in range(24):
        row[f"target_h{h:02d}"] = float("nan")

    # Estadisticas de precio diario
    row["precio_promedio_diario"] = float(np.nanmean(prices))
    row["precio_mediana_diario"] = float(np.nanmedian(prices))
    row["precio_min_diario"] = float(np.nanmin(prices))
    row["precio_max_diario"] = float(np.nanmax(prices))
    row["precio_std_diario"] = float(np.nanstd(prices))

    # ONI y embalse del dia
    row["oni"] = _oni_en(oni_serie, fecha_base)
    row["embalse_porcentaje_diario"] = _embalse_en(embalse_serie, fecha_base)

    # Variables de calendario
    row.update(_calendario(fecha_base, "base", co_hols))
    row.update(_calendario(manana, "target", co_hols))

    base_tipo = row["base_tipo_dia"]
    target_tipo = row["target_tipo_dia"]
    row["transicion_tipo_dia"] = f"{base_tipo}_a_{target_tipo}"

    # Perfil horario
    peak_p = [prices[h] for h in PEAK_HOURS if not np.isnan(prices[h])]
    mad_p = [prices[h] for h in MADRUGADA_HOURS if not np.isnan(prices[h])]
    prom_pico = float(np.mean(peak_p)) if peak_p else float("nan")
    prom_mad = float(np.mean(mad_p)) if mad_p else float("nan")

    row["perfil_promedio_hoy"] = float(np.nanmean(prices))
    row["perfil_mediana_hoy"] = float(np.nanmedian(prices))
    row["perfil_min_hoy"] = float(np.nanmin(prices))
    row["perfil_max_hoy"] = float(np.nanmax(prices))
    row["perfil_std_hoy"] = float(np.nanstd(prices))
    row["perfil_rango_hoy"] = float(np.nanmax(prices) - np.nanmin(prices))
    row["pico_promedio_hoy"] = prom_pico
    row["pico_max_hoy"] = float(np.max(peak_p)) if peak_p else float("nan")
    row["pico_min_hoy"] = float(np.min(peak_p)) if peak_p else float("nan")
    row["pico_std_hoy"] = float(np.std(peak_p)) if peak_p else float("nan")
    row["pico_h20_hoy"] = float(prices[20]) if not np.isnan(prices[20]) else float("nan")
    row["madrugada_promedio_hoy"] = prom_mad
    row["madrugada_std_hoy"] = float(np.std(mad_p)) if mad_p else float("nan")
    row["ratio_pico_valle"] = (prom_pico / prom_mad) if (prom_mad and not np.isnan(prom_mad)) else float("nan")

    # ONI rezagos (datos mensuales)
    for lag_dias, clave in [(7, "oni_lag_7d"), (30, "oni_lag_30d"), (90, "oni_lag_90d")]:
        row[clave] = _oni_en(oni_serie, fecha_base - timedelta(days=lag_dias))

    # Embalse rezagos
    for lag_dias, clave in [
        (1, "embalse_lag_1d"), (7, "embalse_lag_7d"),
        (14, "embalse_lag_14d"), (30, "embalse_lag_30d"),
    ]:
        row[clave] = _embalse_en(embalse_serie, fecha_base - timedelta(days=lag_dias))

    # Embalse EWMA 7d
    emb_hist = embalse_serie[embalse_serie.index <= ts_base].tail(30)
    row["embalse_ewma_7d"] = _ewma_ultimo(emb_hist, 7)

    # Precio EWMA (sobre promedio diario)
    hist_precios = precios_df[precios_df["FECHA"] <= ts_base].copy()
    hist_precios = hist_precios.sort_values("FECHA")
    promedio_series = hist_precios[hcols].mean(axis=1)
    promedio_series.index = hist_precios["FECHA"].values
    promedio_series = pd.Series(promedio_series.values, index=hist_precios["FECHA"])
    row["precio_ewma_7d"] = _ewma_ultimo(promedio_series, 7)
    row["precio_ewma_14d"] = _ewma_ultimo(promedio_series, 14)
    row["precio_ewma_30d"] = _ewma_ultimo(promedio_series, 30)

    # Shape del perfil base
    prom = row["precio_promedio_diario"]
    row["perfil_promedio_base"] = prom
    for h in range(24):
        row[f"shape_base_h{h:02d}"] = (prices[h] / prom) if (prom and not np.isnan(prom)) else float("nan")

    # Rezagos horarios (1d, 2d, 7d, 14d)
    precios_idx = precios_df.set_index("FECHA")
    for lag_dias in LAGS_DIAS:
        ts_lag = pd.Timestamp(fecha_base - timedelta(days=lag_dias))
        if ts_lag in precios_idx.index:
            lag_row = precios_idx.loc[ts_lag]
            for h in range(24):
                row[f"h{h:02d}_lag_{lag_dias}d"] = float(lag_row.get(f"h{h:02d}", float("nan")))
        else:
            for h in range(24):
                row[f"h{h:02d}_lag_{lag_dias}d"] = float("nan")

    # Diferencia vs 7 dias atras
    for h in range(24):
        v_hoy = row.get(f"h{h:02d}", float("nan"))
        v_7d = row.get(f"h{h:02d}_lag_7d", float("nan"))
        if not (np.isnan(v_hoy) or np.isnan(v_7d)):
            row[f"h{h:02d}_diff_vs_7d"] = v_hoy - v_7d
        else:
            row[f"h{h:02d}_diff_vs_7d"] = float("nan")

    # Columnas de leakage: NaN
    row["target_promedio_diario"] = float("nan")
    row["target_max_diario"] = float("nan")
    row["target_min_diario"] = float("nan")
    row["target_std_diario"] = float("nan")
    for h in range(24):
        row[f"delta_target_h{h:02d}"] = float("nan")
        row[f"shape_target_h{h:02d}"] = float("nan")
    row["evento_pico_p90_target"] = float("nan")
    row["evento_pico_p95_target"] = float("nan")

    # One-hot tipo_dia
    for td in TIPO_DIA_CATS:
        row[f"base_tipo_dia_{td}"] = int(row["base_tipo_dia"] == td)
        row[f"target_tipo_dia_{td}"] = int(row["target_tipo_dia"] == td)

    # One-hot transicion
    trans = row["transicion_tipo_dia"]
    for cat in TRANSICION_CATS:
        row[f"transicion_tipo_dia_{cat}"] = int(trans == cat)

    return row
