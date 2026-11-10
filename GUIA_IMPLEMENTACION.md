# Guía de implementación

Esta guía explica, paso a paso, cómo poner en marcha el proyecto de predicción
day-ahead del precio de bolsa de energía del mercado eléctrico colombiano: desde la
instalación local de la aplicación web hasta la reproducción del flujo analítico
completo, la automatización de la predicción diaria y el despliegue en la nube.

Está pensada para una persona con conocimientos básicos de Python y línea de comandos.
El resumen del proyecto, los resultados y la estructura general están en el
[README](README.md); aquí el foco es la puesta en operación.

---

## 1. Requisitos previos

| Herramienta | Versión recomendada | Nota |
|---|---|---|
| Python | 3.11 | El bot de GitHub Actions usa 3.11; evitar 3.13+ por compatibilidad de paquetes. |
| pip | Reciente | `python -m pip install --upgrade pip` |
| Git | Cualquiera | Para clonar y para el flujo automático. |

No se necesitan claves ni credenciales: las fuentes de datos (XM y NOAA) son públicas.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/jkarvajal/prediccion-day-ahead-precio-bolsa.git
cd prediccion-day-ahead-precio-bolsa
```

Se recomienda trabajar en un entorno virtual aislado:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

---

## 3. Dependencias: dos niveles

El proyecto separa las dependencias en dos archivos, según lo que se quiera hacer.

**a) Solo ejecutar la aplicación web** (lo mínimo):

```bash
pip install -r requirements.txt
```

Instala `streamlit`, `pandas`, `numpy`, `scikit-learn==1.7.2`, `joblib` y `plotly`.
La versión de `scikit-learn` está fijada a la misma con la que se serializó el modelo,
para que cargue sin advertencias de compatibilidad.

**b) Ejecutar el flujo analítico completo** (procesar datos, entrenar, comparar y
regenerar figuras):

```bash
pip install -r requirements-pipeline.txt
```

Añade, sobre lo anterior, `matplotlib`, `statsmodels`, `holidays`, `openpyxl`,
`requests`, `python-docx`, `pydataxm`, `prophet`, `tensorflow`, `jupyter` e
`ipykernel`.

---

## 4. Estructura del proyecto

```
├── app/
│   └── app.py                 Aplicación web (Streamlit), 6 vistas
├── src/                       Módulos reutilizables del flujo analítico
│   ├── config.py              Rutas del proyecto
│   ├── ingest_xm.py           Descarga de datos de XM (API pydataxm)
│   ├── oni.py                 Descarga del índice ONI (NOAA CPC)
│   ├── data_loader.py         Carga de datos
│   ├── preprocessing.py       Limpieza y estructuración horaria
│   ├── features.py            Ingeniería de variables predictoras
│   ├── build_features_incremental.py  Construcción incremental de features
│   ├── baselines.py           Líneas base de persistencia
│   ├── exploratory_models.py  Estrategias avanzadas (protocolo homogéneo)
│   ├── recalibration.py       Recalibración periódica
│   ├── evaluation.py          Métricas y prueba de Diebold-Mariano
│   ├── train_final_model.py   Entrena y persiste el modelo final
│   ├── run_consolidacion_exploratoria.py  Comparación consolidada
│   └── gen_prediccion_diaria.py           Bot de predicción diaria
├── models/                    Modelo final entrenado (.joblib) y metadatos
├── data/outputs/              Artefactos (CSV) que consume la aplicación
├── notebooks/                 Cuadernos NB00 a NB08 (pipeline completo)
└── .github/workflows/
    └── update_data.yml        Automatización diaria de la predicción
```

Los datos crudos e intermedios no se versionan (pesan cientos de MB y se regeneran).
El repositorio conserva solo los artefactos que la aplicación necesita para operar.

---

## 5. Ejecutar la aplicación localmente

```bash
streamlit run app/app.py
```

Abre el navegador en `http://localhost:8501`. La aplicación lee los CSV de
`data/outputs/` y el modelo de `models/`, de modo que funciona sin conexión a las
fuentes de datos.

---

## 6. Reproducir el flujo analítico completo

El pipeline está documentado en los cuadernos de `notebooks/`, pensados para
ejecutarse en orden. Requieren `requirements-pipeline.txt`.

| Cuaderno | Propósito |
|---|---|
| `00_Analisis_consolidado_por_objetivos` | Vista consolidada de resultados por objetivo. |
| `01_procesamiento_datos` | Descarga y depuración de precio, embalse y ONI. |
| `02_eda` | Análisis exploratorio (perfiles, estacionalidad, extremos). |
| `03_ingenieria_variables` | Construcción de la base supervisada day-ahead. |
| `04_modelos_baseline` | Líneas base de persistencia y promedios. |
| `05_modelos_mensuales` | Modelado a resolución mensual. |
| `06_modelos_diarios` | Modelado a resolución diaria. |
| `07_modelos_horarios_dayahead` | 24 modelos horarios para el día siguiente. |
| `08_recalibracion_y_significancia` | Recalibración y prueba de Diebold-Mariano. |

Para lanzar los cuadernos:

```bash
jupyter notebook
```

Algunas tareas tienen un punto de entrada directo como módulo, sin abrir cuadernos:

```bash
# Comparación consolidada de todas las estrategias
python -m src.run_consolidacion_exploratoria

# Entrenar y serializar el modelo final a models/
python -m src.train_final_model
```

### Fuentes de datos

| Fuente | Variable | Rango | Frecuencia | Acceso |
|---|---|---|---|---|
| XM (SIN) | Precio de bolsa nacional | 2010 a 2026 | Horaria | API `pydataxm` |
| XM (SIN) | Nivel agregado de embalses | 2013 a 2026 | Diaria | API `pydataxm` |
| NOAA CPC | Índice oceánico de El Niño (ONI) | 2010 a 2026 | Mensual | Descarga web |

XM publica los datos de un día con un rezago de 3 a 4 días en los canales públicos;
por eso la aplicación es retrospectiva y metodológica, no una herramienta operativa
en tiempo real.

---

## 7. Predicción diaria automática (bot)

El script `src/gen_prediccion_diaria.py` genera la predicción de las 24 horas del día
siguiente de forma autónoma:

1. Descarga los últimos 35 días de XM (precio y embalse) y el histórico del ONI.
2. Construye la fila de variables del último día completo disponible.
3. Aplica tres modelos: persistencia diaria, Lasso horario y aprendizaje residual.
4. Guarda `data/outputs/prediccion_automatica.csv` (24 filas, una por hora).

Ejecución manual:

```bash
python src/gen_prediccion_diaria.py
```

### Automatización con GitHub Actions

El flujo `.github/workflows/update_data.yml` ejecuta el bot de forma programada
(`cron` diario) y también permite lanzarlo a mano desde la pestaña **Actions**
(`workflow_dispatch`). El job instala `requirements.txt` más `pydataxm`, `holidays`
y `requests`, corre el script y confirma el CSV de salida al repositorio con el
usuario `github-actions[bot]`. Requiere el permiso `contents: write`, ya activado en
el workflow. No hace falta configurar secretos porque las fuentes son públicas.

---

## 8. Despliegue en Streamlit Community Cloud

La aplicación está desplegada en Streamlit Community Cloud:
`https://prediccion-precio-bolsa-co.streamlit.app`

Pasos para desplegar desde cero:

1. Entrar a [share.streamlit.io](https://share.streamlit.io) e iniciar sesión con la
   cuenta de GitHub que aloja el repositorio.
2. **New app** y seleccionar el repositorio, la rama `main` y el archivo principal
   `app/app.py`.
3. Streamlit Cloud detecta `requirements.txt` e instala las dependencias de la
   aplicación automáticamente.
4. Desplegar. Cada `git push origin main` vuelve a desplegar la aplicación.

> Nota: el bloque YAML al inicio del `README.md` (con `sdk: docker`, `app_port` y
> emoji) corresponde a un despliegue previo en Hugging Face Spaces y Streamlit Cloud
> lo ignora. Puede conservarse o eliminarse sin afectar el despliegue actual.

---

## 9. Solución de problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| Advertencias al cargar el modelo `.joblib` | Versión de `scikit-learn` distinta a 1.7.2 | Instalar exactamente `scikit-learn==1.7.2`. |
| Falla la instalación por `numpy` | Python muy reciente sin binario precompilado | Usar Python 3.11; `numpy` se deja como rango en `requirements.txt`. |
| `pydataxm` no devuelve datos | Rezago de publicación de XM o corte de la API | Reintentar más tarde; el bot usa una ventana de 35 días para tolerarlo. |
| La app no muestra la predicción automática | Falta `data/outputs/prediccion_automatica.csv` | Ejecutar el bot o esperar la corrida programada de GitHub Actions. |

---

## 10. Licencia, cita y contacto

- **Autor:** Jonathan Carvajal García. Maestría en Ciencia de Datos y Analítica, UNAD.
- **Cómo citar:** Carvajal García, J. (2026). *Predicción del precio horario de bolsa
  de energía para el día siguiente en el mercado eléctrico colombiano mediante
  técnicas de ciencia de datos* [Trabajo de grado, Universidad Nacional Abierta y a
  Distancia UNAD].
- **Dashboard:** https://prediccion-precio-bolsa-co.streamlit.app
