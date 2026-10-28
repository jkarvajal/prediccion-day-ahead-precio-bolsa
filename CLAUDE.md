# CLAUDE.md - Proyecto MCDA

## Proyecto

**Titulo:** Implementacion de un modelo de prediccion day-ahead (24 h) del precio de bolsa de energia en el mercado electrico colombiano  
**Programa:** Maestria en Ciencia de Datos, UNAD (ECBTI)  
**Curso:** Seminario de Investigacion Aplicada III  
**Autor:** Jonathan Carvajal (jcarvajalga@outlook.com)  
**Repositorio:** `D:\Claude\MCDA` -- rama `main`, desplegado en Streamlit Community Cloud

---

## Sesiones y sus roles

Cada sesion trabaja sobre archivos especificos. No mezclar.

| Sesion | Archivos que edita | Lo que NO toca |
|---|---|---|
| **Tesis** | `borrador.tex` (ver ruta abajo) | No toca `gen_borrador.py` |
| **Paper** | `reports/paper/paper.tex` | No toca el borrador |
| **Despliegue** | `app/app.py` | No toca documentos |
| **Fase 3 - Entrega** | `reports/entregables_fase3/` | Solo valida e integra |
| **Desarrollo inicial** | Solo consulta, no edita | Todo |

---

## Mapa de archivos clave

```
D:\Claude\MCDA\                          <- repo principal
├── app/app.py                           <- dashboard Streamlit
├── src/gen_borrador.py                  <- genera .docx de la tesis (parametrizado por fase)
├── src/gen_prediccion_diaria.py         <- bot de prediccion automatica diaria
├── models/modelo_final_dayahead.joblib  <- modelo principal entrenado
├── data/outputs/                        <- CSVs con resultados de notebooks
├── reports/figures/                     <- figuras (paleta morado #6b3fa0 / teal #2a9d8f)
├── reports/paper/paper.tex             <- articulo cientifico
├── reports/entregables_fase1/          <- entregables Fase 1 (referencia)
├── reports/entregables_fase3/          <- entregables Fase 3 (crear aqui)
├── guias/                               <- guias de cada fase del semestre
└── notebooks/                          <- NB00 a NB08 (pipeline completo)

D:\Claude\Paper Latex\workspace\         <- directorio EXTERNO (editor LaTeX local)
├── borrador.tex                         <- fuente principal de la tesis (editar aqui)
├── build/                               <- salida de pdflatex
└── media/                              <- copiar figuras desde reports/figures/ aqui
```

---

## Documento de tesis

**Dos fuentes paralelas -- flujo desde 2026-10-20:**
- Editar siempre `borrador.tex` (externo). `gen_borrador.py` se sincroniza SOLO cuando el usuario lo pide.
- Compilar: `pdflatex -output-directory=build borrador.tex` (MiKTeX; el "exit 1" de la primera corrida es normal).
- Generar .docx: `python src/gen_borrador.py 3` (Fase 3 incluye Cap. 5).

**Estructura completa (todos los capitulos completos al 30 sep 2026):**
- Introduccion, Planteamiento + PICOC + Justificacion, Objetivos, Marco, Metodologia (CRISP-DM)
- Cap. 1 "De la serie historica a la base supervisada" (obj 1, preparacion datos)
- Cap. 2 "El comportamiento del precio y la senal de las exogenas" (obj 2, EDA)
- Cap. 3 "La prueba del pronostico: evaluacion y seleccion" (obj 3, modelado)
- Cap. 4 "De los resultados al producto: comunicacion y despliegue" (obj 4, dashboard)
- Cap. 5 "Discusion" (Fase 3 -- ya en ambas fuentes)
- Conclusiones, Recomendaciones, Limitaciones
- Referencias: 33 entradas (bibliografía manual con `\bibitem`; APA 7 estricto)

---

## Marcos metodologicos

| Marco | Proposito |
|---|---|
| **PICOC** | Fundamenta la pregunta de investigacion (cierre del Planteamiento) |
| **PRISMA 2020** | Revision bibliografica sistematica (Scopus + WoS, 2021-2026) |
| **CRISP-DM** | Marco de desarrollo/experimentacion (6 fases mapeadas a objetivos) |

---

## Fases del semestre y estado

| Fase | Titulo | Cierre | Estado |
|---|---|---|---|
| 1 | Coherencia entre resultados preliminares | 30 ago 2026 | Completa (`reports/entregables_fase1/`) |
| 2 | Diseno y Construccion de Resultados | 1 oct 2026 | Completa |
| 3 | Discusion de Resultados y Aportes | 29 oct 2026 | **En curso** -- Cap. 5 escrito, pendiente entregable PDF |
| 4 | Entrega Documento del Proyecto | 22 nov 2026 | Pendiente |
| 5 | Sustentacion y Divulgacion | 6 dic 2026 | Pendiente |

**Entregable Fase 3:** capitulo de discusion (Cap. 5) + documento consolidado en PDF segun normas UNAD. Guardar en `reports/entregables_fase3/`.

---

## Dashboard desplegado

- URL: https://prediccion-precio-bolsa-co.streamlit.app
- Redeploy: `git push origin main`
- Fuente: `app/app.py`

---

## Compromisos pendientes (reunion 4 nov 2026 con Rafael Gaitan)

**Dashboard (`app/app.py`):** todos implementados en reunion del 21 oct 2026.

**Documento (`borrador.tex`):**
- Agregar instrucciones de uso del dashboard como Anexo o subseccion final del Cap. 4
- Documentar rezago de XM como limitacion formal (3-4 dias en datos publicos)
- Evaluar conferencia en Barranquilla (8 dic) para divulgacion

---

## Convenciones que aplican a TODAS las sesiones

- **Sin em-dash:** nunca usar el caracter "--" tipografico; usar coma, punto y coma o reescribir
- **Alcance:** el modelo es retrospectivo/metodologico, NO operativo en tiempo real; declararlo en Alcance y en Limitaciones
- **Citas APA 7:** 3+ autores con `et al.` desde la primera cita; 2 autores con ambos apellidos unidos por `y`
- **Precios:** en COP/kWh ($/kWh)
- **Figuras:** paleta morado `#6b3fa0` / teal `#2a9d8f`
- **No recompilar PDF** sin que el usuario lo pida explicitamente

---

## Edición conjunta del LaTeX (Paper Latex)

El documento se edita en la app Paper Latex mientras trabajamos. Antes de
editar, lee `D:\Claude\Paper Latex\.paper-latex\editor-state.json` para saber
qué archivo, línea y texto seleccionado tiene el usuario ("esto", "aquí" =
esa selección); el archivo abierto es el campo `path` (puede estar en un
proyecto: `workspace\<Proyecto>\archivo.tex`).

Mis observaciones son comentarios anclados que NO están en el .tex. Léelos con
`node "D:\Claude\Paper Latex\tools\paper-latex.mjs" comentarios` y responde a cada
uno con `sugerir --comentario <id> --reemplazo "..."`: la sugerencia aparece en
la app y yo la acepto o rechazo (aceptar cierra el comentario). No edites el
.tex para atender un comentario ni lo resuelvas tú. Si te pido un cambio directo
en el chat, edita el .tex con cambios mínimos y localizados (vuelve a leer el
archivo antes de editar; figuras en el `media\` del proyecto). Antes de una
tanda de cambios, `punto "antes de ..."`; después `diff` para ver solo lo
modificado. Para comprobar que compila: `compilar` (errores con archivo:línea).
Guía completa: `D:\Claude\Paper Latex\GUIA-CLAUDE-CODE.md`.
