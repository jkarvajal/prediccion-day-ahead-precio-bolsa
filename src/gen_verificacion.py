# -*- coding: utf-8 -*-
"""
Genera un documento de VERIFICACIÓN APARTE (no es parte del trabajo de grado):
  1. Lista de chequeo de cierre (Fase 4).
  2. Matriz de trazabilidad: objetivo -> capítulo -> resultado principal -> conclusión.

Sirve como herramienta de control para el estudiante y para mostrar al director que se
verificó cada requisito. Si se decide incluir la matriz de trazabilidad en el documento
final, se puede copiar la tabla correspondiente.

Reutiliza el estilo y los ayudantes de gen_borrador.py.
"""

import gen_borrador as gb
from gen_borrador import nuevo_documento, titulo1, titulo2, parrafo, tabla_n, OBJETIVOS_ESPECIFICOS

SALIDA = gb.DOCUMENTOS / "Verificacion_y_Trazabilidad_Fase4.docx"

# --------------------------------------------------------------------------- #
# 1. Lista de chequeo de cierre (Fase 4)
# --------------------------------------------------------------------------- #
# Estado: "Cumplido" (verificado en el documento) / "Externo/pendiente" (gestión o
# revisión que no depende del documento en sí).
CHECKLIST = [
    ("El documento contiene todos los capítulos (según la lista de la universidad o del "
     "director)", "Cumplido", "Resumen, Abstract, Introducción, Planteamiento, Objetivos, "
     "Marco, Metodología, Cap. 1-5, Conclusiones y Referencias."),
    ("Introducción incluida", "Cumplido", "Sección inicial del documento."),
    ("Resumen y Abstract incluidos", "Cumplido", "Con palabras clave y keywords."),
    ("Capítulo de conclusiones, recomendaciones y limitaciones", "Cumplido",
     "Incluye conclusiones por objetivo, respuesta a la pregunta, limitaciones y "
     "recomendaciones."),
    ("Cada objetivo específico respondido en resultados y conclusiones", "Cumplido",
     "Ver la matriz de trazabilidad de este documento."),
    ("Correspondencia cita -> referencia (toda cita tiene su referencia)", "Cumplido",
     "Verificado automáticamente: 0 citas sin referencia."),
    ("Correspondencia referencia -> cita (toda referencia se cita)", "Cumplido",
     "Verificado automáticamente: 0 referencias sin citar."),
    ("Normas de citación y referencias APA 7", "Cumplido",
     "Autor-año; et al. para 3+ autores; sangría francesa en la lista."),
    ("Formato APA: títulos, tablas y figuras numeradas y referenciadas", "Cumplido",
     "Tablas y figuras con número y título; referidas por número en el texto."),
    ("Formato APA: márgenes 2,54 cm, interlineado doble, paginación, fuente TNR 12",
     "Cumplido", "Aplicado en el documento Word."),
    ("Revisión con Turnitin", "Externo/pendiente",
     "Fase 2 dio 7 por ciento; repetir sobre la versión de Fase 3/4."),
    ("Verificación contra la plantilla/norma exacta de la biblioteca UNAD",
     "Externo/pendiente", "Contrastar portada, orden de secciones y estilo."),
    ("Realimentación con el director y el tutor", "Externo/pendiente",
     "Antes de la entrega final."),
    ("Documento posteado en el foro de la fase", "Externo/pendiente",
     "Paso administrativo de la plataforma."),
    ("PDF consolidado subido a la tarea de la fase", "Externo/pendiente",
     "Paso administrativo de la plataforma."),
]

# --------------------------------------------------------------------------- #
# 2. Matriz de trazabilidad (objetivo -> capítulo -> resultado -> conclusión)
# --------------------------------------------------------------------------- #
TRAZABILIDAD = [
    ("Objetivo 1: construir la serie horaria depurada y las variables predictoras para "
     "el día siguiente.",
     "Capítulo 1. De la serie histórica a la base supervisada de pronóstico.",
     "Serie horaria depurada de 142.344 registros (2010-2026) y base supervisada de "
     "5.930 días con 333 columnas (calendario, rezagos, estadísticas móviles y exógenas).",
     "Se obtuvo una base trazable y reproducible, insumo común de todo el trabajo "
     "posterior."),
    ("Objetivo 2: caracterizar el comportamiento del precio y el aporte de las variables "
     "exógenas.",
     "Capítulo 2. El comportamiento del precio y la señal de las exógenas.",
     "Estacionalidad horaria (pico 18-21 h), distribución asimétrica con extremos en "
     "días hábiles, estacionariedad en nivel, autocorrelación diaria de 0,93 a 0,96; "
     "exógenas de aporte moderado y anticipado (ONI).",
     "La capacidad de pronóstico descansa sobre todo en la memoria de corto plazo; el "
     "aporte de las exógenas es moderado y de naturaleza anticipada."),
    ("Objetivo 3: evaluar líneas base, modelos y estrategias bajo un protocolo homogéneo "
     "y seleccionar un enfoque defendible.",
     "Capítulo 3. La prueba del pronóstico: evaluación y selección del enfoque.",
     "Ocho estrategias con partición temporal, métricas múltiples y prueba de "
     "Diebold-Mariano: ninguna supera a la persistencia en MAE, pero varias la superan "
     "en RMSE y en el régimen volátil; el Lasso recalibrado mejora el RMSE.",
     "La mejora verificable existe, pero es acotada; se adopta la persistencia como "
     "referencia y el Lasso recalibrado como complemento defendible."),
    ("Objetivo 4: comunicar los resultados mediante productos de visualización, incluido "
     "un entorno web.",
     "Capítulo 4. De los resultados al producto: comunicación y despliegue.",
     "Entorno web público (Streamlit) con el histórico, la predicción del día siguiente "
     "frente al precio real, la exploración de patrones y la relación con el clima.",
     "Los resultados se traducen en una herramienta accesible; se cierra el ciclo "
     "CRISP-DM con el despliegue."),
]


def main():
    gb._CONTADOR_TABLA[0] = 0
    gb._CONTADOR_FIGURA[0] = 0
    doc = nuevo_documento()

    titulo1(doc, "Lista de verificación y matriz de trazabilidad (Fase 4)")
    parrafo(doc,
            "Este es un documento de control, auxiliar y separado del trabajo de grado. "
            "Reúne la lista de chequeo de cierre de la Fase 4 y la matriz de trazabilidad "
            "entre los objetivos específicos, los capítulos, los resultados y las "
            "conclusiones. No forma parte del documento final; la matriz de trazabilidad "
            "puede incorporarse al documento si el director o la norma lo requieren.")

    titulo1(doc, "1. Lista de chequeo de cierre")
    tabla_n(doc, CHECKLIST, ["Ítem de verificación", "Estado", "Observación"],
            "Lista de chequeo de cierre del documento (Fase 4)", negrita_primera=False)

    titulo1(doc, "2. Matriz de trazabilidad")
    parrafo(doc,
            "La tabla relaciona cada objetivo específico con el capítulo que lo "
            "desarrolla, su resultado principal y la conclusión correspondiente, de modo "
            "que se verifique de un vistazo que cada objetivo quedó respondido en los "
            "resultados y en las conclusiones.")
    tabla_n(doc, TRAZABILIDAD,
            ["Objetivo específico", "Capítulo", "Resultado principal", "Conclusión"],
            "Trazabilidad entre objetivos, capítulos, resultados y conclusiones",
            negrita_primera=False)

    doc.core_properties.author = "Jonathan Carvajal García"
    doc.core_properties.title = "Verificación y trazabilidad (Fase 4)"
    doc.save(SALIDA)
    print("Guardado:", SALIDA)


if __name__ == "__main__":
    main()
