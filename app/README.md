# App de despliegue, Predicción day-ahead

Aplicación web (Streamlit) que consume el modelo final entrenado y permite consultar
la predicción de las 24 horas del día siguiente del precio de bolsa de energía.

## Qué muestra

- Perfil horario (24 h) del día objetivo con dos modelos: la persistencia diaria (modelo de referencia seleccionado) y el modelo Lasso (comparación), con la franja pico resaltada.
- Comparación con el precio real cuando el día base es histórico (línea punteada), con el MAE del día para ambos modelos.
- Panel con las métricas de ambos modelos en el conjunto de prueba (MAE, RMSE, R²) y nota sobre la mejora por recalibración (NB08).
- Desempeño por hora del día del modelo Lasso (MAE y R²).
- Tabla descargable del pronóstico en CSV.

## Requisitos previos

El modelo debe estar entrenado y persistido. Si no existe `models/modelo_final_dayahead.joblib`:

```bash
python -m src.train_final_model
```

## Ejecución local

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

La app abre en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud (gratuito)

1. Subir el repositorio a GitHub (incluyendo `models/modelo_final_dayahead.joblib`, `models/modelo_final_metadata.json`, `data/processed/features_day_ahead.csv` y
   `data/outputs/hourly_dayahead_metrics_by_hour.csv`).
2. En [share.streamlit.io](https://share.streamlit.io) crear una nueva app apuntando a:
   - **Repository:** el repositorio del proyecto.
   - **Main file path:** `app/app.py`.
3. Streamlit instala automáticamente `requirements.txt` y publica una URL pública.

> Los artefactos son livianos (modelo ~153 KB). El único archivo grande es
> `features_day_ahead.csv` (~23 MB), dentro de los límites de GitHub y Streamlit Cloud.

## Archivos que consume la app

| Archivo | Uso |
|---------|-----|
| `models/modelo_final_dayahead.joblib` | 24 modelos Lasso (uno por hora) |
| `models/modelo_final_metadata.json` | Métricas y metadatos del modelo |
| `data/processed/features_day_ahead.csv` | Variables por día base para predecir |
| `data/outputs/hourly_dayahead_metrics_by_hour.csv` | Desempeño histórico por hora |
