# Predicción day-ahead del precio de bolsa de energía

Proyecto aplicado de la Maestría en Ciencia de Datos y Analítica (UNAD). Implementa
y evalúa modelos de predicción day-ahead, es decir de las 24 horas del día
siguiente, del precio de bolsa horario del Sistema Interconectado Nacional
colombiano, incorporando variables exógenas climáticas (índice ONI) e hidrológicas
(nivel agregado de embalses).

## Aplicación web

La aplicación permite consultar la predicción de las 24 horas del día siguiente a
partir de cualquier día base del histórico, comparando el modelo de referencia con
el modelo de aprendizaje automático y con el precio realmente observado.

Ejecución local:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Resultados principales

Todas las estrategias se evaluaron bajo un protocolo único: partición temporal
80/20, conjunto de prueba de 1.186 días consecutivos entre diciembre de 2022 y
marzo de 2026, equivalente a 28.464 observaciones horarias.

| Estrategia | MAE | RMSE | R2 |
|---|---|---|---|
| **Persistencia diaria (referencia)** | **68,95** | 157,89 | 0,828 |
| Híbrido con corrección en horas pico | 68,99 | 155,83 | 0,832 |
| Aprendizaje residual | 70,49 | 148,74 | 0,847 |
| Segmentación por tipo de día | 74,43 | **144,93** | **0,855** |
| Lasso horario | 75,70 | 148,82 | 0,847 |
| Modelo de dos etapas | 77,87 | 149,03 | 0,847 |
| Red LSTM | 79,92 | 165,22 | 0,812 |
| Modelo aditivo Prophet | 189,41 | 270,46 | 0,495 |

El hallazgo central es que ninguna estrategia supera a la persistencia diaria en
error absoluto medio, pero varias la superan de forma estadísticamente
significativa en error cuadrático y en capacidad explicativa (prueba de Diebold y
Mariano, p < 0,001). El aporte del modelado aprendido reside en el control de los
errores de mayor magnitud durante los periodos de alta volatilidad, no en la
precisión típica. Este patrón es consistente con la literatura del área, que
advierte sobre las mejoras que no resisten el contraste frente a una línea base
ingenua bien construida (Weron, 2014; Lago et al., 2021).

## Estructura del repositorio

```
├── app/                 Aplicación web (Streamlit)
├── src/                 Módulos reutilizables del flujo analítico
│   ├── config.py                Rutas del proyecto
│   ├── data_loader.py           Carga de datos
│   ├── preprocessing.py         Limpieza y preparación
│   ├── features.py              Ingeniería de variables
│   ├── oni.py                   Descarga de índices ENSO (NOAA)
│   ├── baselines.py             Líneas base de persistencia
│   ├── evaluation.py            Métricas y prueba de Diebold-Mariano
│   ├── recalibration.py         Recalibración periódica
│   ├── exploratory_models.py    Estrategias avanzadas bajo protocolo homogéneo
│   └── train_final_model.py     Entrena y persiste el modelo final
├── models/              Modelo final entrenado y sus metadatos
└── data/                Artefactos que la aplicación consume
```

Los datos crudos e intermedios, así como los cuadernos de trabajo, no forman parte
del repositorio: pesan varios cientos de megabytes y se regeneran ejecutando el
flujo. El repositorio conserva únicamente los artefactos que la aplicación
necesita para operar.

## Datos

| Fuente | Variable | Rango | Frecuencia |
|---|---|---|---|
| XM (SIN) | Precio de bolsa nacional | 2010 a 2026 | Horaria |
| XM (SIN) | Nivel agregado de embalses | 2013 a 2026 | Diaria |
| NOAA CPC | Índice oceánico de El Niño (ONI) | 2010 a 2026 | Mensual |

## Metodología

1. **Líneas base:** persistencia diaria, persistencia semanal, promedio histórico
   por hora y promedio móvil de siete días.
2. **Modelos tabulares:** 24 modelos independientes, uno por hora, con selección
   de variables específica. Algoritmos de cresta, Lasso, bosques aleatorios,
   árboles extremadamente aleatorizados y potenciación del gradiente, con y sin
   variables exógenas.
3. **Estrategias avanzadas:** aprendizaje residual sobre la persistencia, modelo
   de dos etapas, segmentación por tipo de día y enfoque híbrido en horas pico.
4. **Modelos especializados:** descomposición aditiva con recalibración semanal y
   red neuronal recurrente de memoria de largo y corto plazo.
5. **Evaluación:** partición temporal estricta, métricas MAE, RMSE, sMAPE y R2,
   con análisis global, por hora del día, por franja pico y por régimen de precio,
   más contraste de significancia estadística.

Para reproducir la comparación consolidada:

```bash
python -m src.run_consolidacion_exploratoria
```

## Autor

Jonathan Carvajal García. Maestría en Ciencia de Datos y Analítica, UNAD.
