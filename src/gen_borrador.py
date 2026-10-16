# -*- coding: utf-8 -*-
"""
Construye el borrador del trabajo de grado, sección por sección, en orden de
lectura. Es la fuente única del documento: cada paso acordado se agrega aquí y se
regenera el Word completo. Luego se ejecuta src.borrador_a_html para actualizar la
versión de consulta.

    python -m src.gen_borrador
    python -m src.borrador_a_html
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, Inches, RGBColor

FUENTE = "Times New Roman"
RAIZ = Path(__file__).resolve().parents[1]
DOCUMENTOS = RAIZ / "notebooks" / "Documentos"
SALIDA = DOCUMENTOS / "Borrador_Trabajo_de_Grado_Jonathan_Carvajal.docx"
SALIDA_FASE3 = DOCUMENTOS / "Borrador_Trabajo_de_Grado_Jonathan_Carvajal_Fase3.docx"
FIGURAS = RAIZ / "reports" / "figures"

TITULO = ("Predicción del precio horario de bolsa de energía para el día siguiente "
          "en el mercado eléctrico colombiano mediante técnicas de ciencia de datos")

OBJETIVO_GENERAL = (
    "Desarrollar un modelo de predicción del precio horario de bolsa de energía para "
    "el día siguiente en el mercado eléctrico colombiano, mediante un flujo "
    "reproducible de ciencia de datos, con validación temporal y comparación frente a "
    "líneas base."
)

OBJETIVOS_ESPECIFICOS = [
    "Construir la serie horaria depurada del precio de bolsa de energía del mercado "
    "eléctrico colombiano y las variables predictoras para el día siguiente, con "
    "información de calendario, rezagos temporales y variables exógenas climáticas e "
    "hidrológicas.",
    "Caracterizar el comportamiento histórico del precio horario de bolsa de energía "
    "del mercado eléctrico colombiano y el aporte predictivo de las variables "
    "exógenas.",
    "Evaluar el desempeño de líneas base, modelos y estrategias de pronóstico bajo un "
    "protocolo homogéneo con métricas múltiples, orientado a la selección de un "
    "enfoque defendible.",
    "Comunicar los resultados y el desempeño del enfoque seleccionado mediante "
    "productos de visualización analítica, incluido un entorno web.",
]

INTRODUCCION = [
    "En el mercado eléctrico colombiano, el precio de bolsa de energía es el precio de "
    "corto plazo del mercado mayorista y varía hora a hora según la oferta de "
    "generación y la demanda del sistema. Al depender de una generación mayoritariamente "
    "hidráulica, su comportamiento está condicionado por el nivel de los embalses, "
    "ligado a las condiciones hidrológicas y climáticas. Cuando el recurso hídrico "
    "escasea, parte de la demanda se atiende con generación térmica basada en "
    "combustibles "
    "fósiles, cuyo mayor costo de operación, asociado al precio de los combustibles, "
    "eleva el precio y ocasiona una alta volatilidad y episodios extremos. A esto se "
    "suma la variación de la "
    "demanda, determinada por las rutinas de consumo de los distintos sectores, que "
    "genera patrones horarios y diferencias por tipo de día. Por esta razón, "
    "su pronóstico constituye "
    "un reto relevante para la analítica energética aplicada y para la construcción de "
    "herramientas que apoyen la lectura de corto plazo del mercado.",

    "En un escenario de alta variabilidad, la construcción de pronósticos confiables "
    "depende de decisiones metodológicas clave: la resolución temporal, el horizonte "
    "de predicción, la selección de variables, el tratamiento de periodos atípicos, "
    "las métricas de evaluación y el uso de esquemas de validación temporal coherentes "
    "con la naturaleza de las series de tiempo. Una evaluación que divida los datos de "
    "forma aleatoria, sin respetar el orden cronológico de la serie, o que no los "
    "compare contra líneas base competitivas, puede conducir a conclusiones optimistas "
    "y poco defendibles.",

    "En este contexto, el proyecto aplicado desarrolla y evalúa un modelo de "
    "predicción del precio horario de bolsa de energía para el día siguiente en el "
    "mercado eléctrico colombiano. El pronóstico se entiende como la estimación del "
    "perfil horario completo del día siguiente; en consecuencia, el objetivo no se "
    "limita a predecir un único valor diario, sino a generar una curva de veinticuatro "
    "valores horarios y evaluar su desempeño tanto de forma global como por hora. La "
    "serie del precio de bolsa se obtiene de XM, operador del mercado eléctrico "
    "colombiano, y se complementa con variables exógenas del índice climático de El "
    "Niño de la NOAA y del nivel de embalses de centrales de generación reportado por "
    "XM, durante el periodo 2010 a "
    "2026.",

    "El desarrollo se organiza como un flujo reproducible de ciencia de datos que "
    "integra la preparación de la serie histórica, el análisis exploratorio, la "
    "ingeniería de variables, la implementación de líneas base, la evaluación de "
    "modelos de aprendizaje automático, la incorporación de variables exógenas y el "
    "análisis de estrategias avanzadas como los modelos especializados de series "
    "temporales, el aprendizaje residual, la segmentación temporal y los modelos de "
    "dos etapas. El flujo se complementa con un entorno web de visualización que "
    "comunica los históricos, las predicciones y las métricas del enfoque "
    "seleccionado.",
]


def _config_encabezado(estilo, centrado):
    """Da formato APA 7 a un estilo de encabezado (negrita, TNR 12, negro)."""
    estilo.font.name = FUENTE
    estilo.font.size = Pt(12)
    estilo.font.bold = True
    estilo.font.italic = False
    estilo.font.color.rgb = RGBColor(0, 0, 0)
    pf = estilo.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER if centrado else WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 2.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.keep_with_next = True


def _numero_de_pagina(doc):
    """Agrega el número de página (APA 7) arriba a la derecha en todas las páginas."""
    seccion = doc.sections[0]
    encabezado = seccion.header
    encabezado.is_linked_to_previous = False
    p = encabezado.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run()
    run.font.name = FUENTE
    run.font.size = Pt(12)
    ini = OxmlElement("w:fldChar")
    ini.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fin = OxmlElement("w:fldChar")
    fin.set(qn("w:fldCharType"), "end")
    run._r.append(ini)
    run._r.append(instr)
    run._r.append(fin)


def nuevo_documento():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FUENTE
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    # Márgenes APA 7: 2,54 cm (1 pulgada) en los cuatro lados.
    for s in doc.sections:
        s.left_margin = Cm(2.54)
        s.right_margin = Cm(2.54)
        s.top_margin = Cm(2.54)
        s.bottom_margin = Cm(2.54)
    _numero_de_pagina(doc)
    # Encabezados en formato APA 7: nivel 1 centrado, nivel 2 a la izquierda; ambos
    # en negrita, Times New Roman 12 y color negro.
    _config_encabezado(doc.styles["Heading 1"], centrado=True)
    _config_encabezado(doc.styles["Heading 2"], centrado=False)
    return doc


def centrado(doc, texto, negrita=False, size=12):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(texto)
    r.font.name = FUENTE
    r.font.size = Pt(size)
    r.bold = negrita
    return p


def vacio(doc, n=1):
    for _ in range(n):
        doc.add_paragraph()


def _fijar_fuente_titulo(p):
    """Fuerza Times New Roman en el texto del título, por encima de la fuente de tema
    (Calibri) que heredan los estilos de encabezado de Word."""
    for r in p.runs:
        r.font.name = FUENTE
        rpr = r._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.insert(0, rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            rfonts.set(qn(attr), FUENTE)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme"):
            if rfonts.get(qn(attr)) is not None:
                del rfonts.attrib[qn(attr)]
    return p


def titulo1(doc, texto):
    p = doc.add_paragraph(texto, style="Heading 1")
    return _fijar_fuente_titulo(p)


def titulo2(doc, texto):
    p = doc.add_paragraph(texto, style="Heading 2")
    return _fijar_fuente_titulo(p)


def parrafo(doc, texto):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(1.27)
    r = p.add_run(texto)
    r.font.name = FUENTE
    r.font.size = Pt(12)
    return p


def salto_pagina(doc):
    from docx.enum.text import WD_BREAK
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def enlace(parrafo_obj, url, texto):
    """Agrega un hipervínculo clicable (azul, subrayado, TNR 12) al final del párrafo."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    r_id = parrafo_obj.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hlink = OxmlElement("w:hyperlink")
    hlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), FUENTE)
    rfonts.set(qn("w:hAnsi"), FUENTE)
    rpr.append(rfonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    rpr.append(color)
    subr = OxmlElement("w:u")
    subr.set(qn("w:val"), "single")
    rpr.append(subr)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "24")
    rpr.append(sz)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = texto
    run.append(t)
    hlink.append(run)
    parrafo_obj._p.append(hlink)
    return hlink


URL_DASHBOARD = "https://prediccion-precio-bolsa-co.streamlit.app"


def _celda(celda, texto, negrita=False):
    celda.paragraphs[0].text = ""
    r = celda.paragraphs[0].add_run(texto)
    r.font.name = FUENTE
    r.font.size = Pt(11)
    r.bold = negrita


def referencia_parrafo(doc, texto):
    """Párrafo de referencia con sangría francesa."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(texto)
    r.font.name = FUENTE
    r.font.size = Pt(12)
    return p


_CONTADOR_FIGURA = [0]


def figura(doc, nombre_archivo, titulo, nota=None, ancho_cm=14.0):
    """Inserta una figura en formato APA 7: número (negrita) y título (cursiva) encima
    de la imagen centrada, y una Nota opcional debajo."""
    _CONTADOR_FIGURA[0] += 1
    pnum = doc.add_paragraph()
    rnum = pnum.add_run("Figura %d" % _CONTADOR_FIGURA[0])
    rnum.bold = True
    rnum.font.name = FUENTE
    rnum.font.size = Pt(12)
    ptit = doc.add_paragraph()
    rtit = ptit.add_run(titulo)
    rtit.italic = True
    rtit.font.name = FUENTE
    rtit.font.size = Pt(12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIGURAS / nombre_archivo), width=Cm(ancho_cm))
    if nota:
        pn = doc.add_paragraph()
        r1 = pn.add_run("Nota. ")
        r1.italic = True
        r1.font.name = FUENTE
        r1.font.size = Pt(10)
        r2 = pn.add_run(nota)
        r2.font.name = FUENTE
        r2.font.size = Pt(10)
    return p


_CONTADOR_TABLA = [0]


def _borde_xml(tipo, val="single", sz="4"):
    el = OxmlElement("w:" + tipo)
    el.set(qn("w:val"), val)
    if val != "none":
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
    return el


def _bordes_apa(tabla):
    """Deja solo líneas horizontales superior e inferior; sin verticales ni internas."""
    tblPr = tabla._element.tblPr
    viejo = tblPr.find(qn("w:tblBorders"))
    if viejo is not None:
        tblPr.remove(viejo)
    bordes = OxmlElement("w:tblBorders")
    bordes.append(_borde_xml("top"))
    bordes.append(_borde_xml("bottom"))
    for t in ("left", "right", "insideH", "insideV"):
        bordes.append(_borde_xml(t, val="none"))
    tblPr.append(bordes)


def _linea_bajo_encabezado(fila):
    """Agrega una línea horizontal bajo la fila de encabezado."""
    for celda in fila.cells:
        tcPr = celda._element.get_or_add_tcPr()
        tcBorders = tcPr.find(qn("w:tcBorders"))
        if tcBorders is None:
            tcBorders = OxmlElement("w:tcBorders")
            tcPr.append(tcBorders)
        antiguo = tcBorders.find(qn("w:bottom"))
        if antiguo is not None:
            tcBorders.remove(antiguo)
        tcBorders.append(_borde_xml("bottom"))


def titulo_tabla(doc, titulo):
    """Número de tabla (negrita) y título (cursiva) en formato APA 7, encima de la tabla."""
    _CONTADOR_TABLA[0] += 1
    p1 = doc.add_paragraph()
    r1 = p1.add_run("Tabla %d" % _CONTADOR_TABLA[0])
    r1.bold = True
    r1.font.name = FUENTE
    r1.font.size = Pt(12)
    p2 = doc.add_paragraph()
    r2 = p2.add_run(titulo)
    r2.italic = True
    r2.font.name = FUENTE
    r2.font.size = Pt(12)


def tabla_dos_columnas(doc, filas, encabezados, titulo):
    """Tabla APA de dos columnas: número y título encima, solo líneas horizontales."""
    titulo_tabla(doc, titulo)
    tabla = doc.add_table(rows=0, cols=2)
    fila_enc = tabla.add_row()
    _celda(fila_enc.cells[0], encabezados[0], negrita=True)
    _celda(fila_enc.cells[1], encabezados[1], negrita=True)
    for izq, der in filas:
        celdas = tabla.add_row().cells
        _celda(celdas[0], izq, negrita=True)
        _celda(celdas[1], der)
    _bordes_apa(tabla)
    _linea_bajo_encabezado(fila_enc)
    return tabla


def tabla_n(doc, filas, encabezados, titulo, negrita_primera=True):
    """Tabla APA de N columnas: número y título encima, solo líneas horizontales."""
    titulo_tabla(doc, titulo)
    n = len(encabezados)
    tabla = doc.add_table(rows=0, cols=n)
    fila_enc = tabla.add_row()
    for i, enc in enumerate(encabezados):
        _celda(fila_enc.cells[i], enc, negrita=True)
    for fila in filas:
        celdas = tabla.add_row().cells
        for i, val in enumerate(fila):
            _celda(celdas[i], str(val), negrita=(negrita_primera and i == 0))
    _bordes_apa(tabla)
    _linea_bajo_encabezado(fila_enc)
    return tabla


PICOC = [
    ("Población / Problema (P)",
     "La serie horaria del precio de bolsa de energía del mercado eléctrico "
     "colombiano."),
    ("Intervención (I)",
     "Un modelo de predicción del precio para el día siguiente construido "
     "mediante un flujo reproducible de ciencia de datos."),
    ("Comparación (C)",
     "Una línea base competitiva, la persistencia diaria, que toma como "
     "pronóstico el precio de la misma hora del día anterior."),
    ("Resultado (O)",
     "El desempeño del pronóstico medido con métricas de error bajo validación "
     "temporal."),
    ("Contexto (C)",
     "El mercado mayorista de corto plazo, con una generación mayoritariamente "
     "hidráulica que condiciona la formación del precio."),
]


# --------------------------------------------------------------------------- #
# Secciones del documento, en orden de lectura
# --------------------------------------------------------------------------- #

def portada(doc):
    vacio(doc, 3)
    centrado(doc, TITULO, negrita=True)
    vacio(doc, 2)
    centrado(doc, "Jonathan Carvajal García")
    vacio(doc, 2)
    centrado(doc, "Trabajo de grado presentado como requisito para optar al título de")
    centrado(doc, "Magíster en Ciencia de Datos y Analítica")
    vacio(doc, 2)
    centrado(doc, "Director")
    centrado(doc, "Rafael Gaitán Ospina")
    vacio(doc, 3)
    centrado(doc, "Universidad Nacional Abierta y a Distancia (UNAD)")
    centrado(doc, "Escuela de Ciencias Básicas, Tecnología e Ingeniería (ECBTI)")
    centrado(doc, "Maestría en Ciencia de Datos y Analítica")
    centrado(doc, "2026")


def objetivo_numerado(doc, n, texto):
    """Objetivo específico numerado, con sangría francesa (sin sangría de primera línea)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    r = p.add_run(f"{n}. {texto}")
    r.font.name = FUENTE
    r.font.size = Pt(12)
    return p


PLANTEAMIENTO = [
    "El precio de bolsa de energía es el precio de corto plazo del mercado mayorista "
    "colombiano. Los aspectos comerciales de este mercado están reglamentados por la "
    "Resolución CREG 024 de 1995 (Comisión de Regulación de Energía y Gas [CREG], 1995). "
    "Este precio presenta un comportamiento complejo, "
    "caracterizado por "
    "estacionalidad horaria, variaciones asociadas al tipo de día, alta volatilidad y "
    "ocurrencia de episodios extremos. Estas características dificultan la identificación "
    "de patrones estables y hacen que su pronóstico para el día siguiente sea "
    "metodológicamente exigente.",

    "Contar con pronósticos confiables del precio para el día siguiente es relevante "
    "para los agentes que participan en el mercado y para el análisis de corto plazo de "
    "su comportamiento. Sin embargo, una implementación de pronóstico no puede "
    "considerarse adecuada por el solo hecho de utilizar técnicas de aprendizaje "
    "automático. Para que su resultado sea defendible, debe evaluarse frente a modelos "
    "de referencia, respetar el orden cronológico de los datos, documentar las "
    "variables utilizadas y analizar el desempeño por hora, por tipo de día y durante "
    "los episodios de alta volatilidad.",

    "La literatura de pronóstico de precios eléctricos advierte que una parte "
    "considerable de las mejoras reportadas no resiste el contraste contra una línea "
    "base competitiva bien construida (Weron, 2014; Lago et al., 2021). En este "
    "mercado, la persistencia diaria, que toma como pronóstico el precio de la misma "
    "hora del día anterior, constituye una línea base fuerte, de modo que cualquier "
    "modelo más complejo debe demostrar una mejora comprobable frente a ella y no solo "
    "frente a alternativas menos exigentes.",

    "El vacío que aborda el proyecto no es la ausencia de un modelo de pronóstico, que "
    "puede construirse con técnicas ya conocidas, sino la falta de una comparación "
    "reproducible que establezca si un enfoque más complejo supera de manera "
    "verificable a una línea base competitiva, junto con la necesidad de comunicar las "
    "predicciones de forma accesible para su lectura de corto plazo.",

    "Por lo anterior, el problema de investigación se concreta en la necesidad de "
    "desarrollar y evaluar, mediante validación temporal y métricas de error, un modelo "
    "de predicción del precio horario de bolsa de energía para el día siguiente en el "
    "mercado eléctrico colombiano, estableciendo comparaciones con líneas base "
    "competitivas, documentando un proceso reproducible y comunicando los resultados "
    "mediante visualización analítica.",

    "En consecuencia, la pregunta que orienta la investigación es la siguiente: ¿en qué "
    "medida un modelo de predicción del precio horario de bolsa de energía para el día "
    "siguiente en el mercado eléctrico colombiano, desarrollado mediante un flujo "
    "reproducible de ciencia de datos, mejora de forma verificable el desempeño de una "
    "línea base competitiva bajo validación temporal y métricas de error?",
]


def introduccion(doc):
    titulo1(doc, "Introducción")
    for parr in INTRODUCCION:
        parrafo(doc, parr)


JUSTIFICACION = [
    "El precio de bolsa de energía es una variable central para las decisiones de "
    "corto plazo de los agentes del mercado mayorista. Un pronóstico confiable de su "
    "perfil horario para el día siguiente apoya la elaboración de ofertas, la gestión "
    "de contratos y la administración del riesgo en un mercado caracterizado por una "
    "alta volatilidad y una fuerte dependencia de las condiciones hidrológicas. Contar "
    "con estimaciones ordenadas y evaluadas de forma rigurosa contribuye, por tanto, a "
    "una lectura más informada del comportamiento del precio.",

    "Desde el punto de vista metodológico, la literatura advierte que una parte "
    "considerable de las mejoras reportadas en el pronóstico de precios eléctricos no "
    "resiste el contraste frente a una línea base competitiva (Weron, 2014; Lago et "
    "al., 2021). Este proyecto se justifica en la necesidad de aportar una evaluación "
    "reproducible, con validación temporal, métricas múltiples y análisis desagregado "
    "por hora, por tipo de día y durante los episodios de alta volatilidad, de modo "
    "que las conclusiones se sostengan en evidencia verificable y no en comparaciones "
    "optimistas.",

    "La pertinencia del estudio se refuerza por las particularidades del mercado "
    "eléctrico colombiano. Frente a la amplia literatura internacional sobre el "
    "pronóstico de precios eléctricos (Weron, 2014; Lago et al., 2021), la producción "
    "específica sobre el mercado colombiano es comparativamente reducida, como se "
    "evidencia en la revisión de literatura de este trabajo. Además, el caso colombiano "
    "combina una generación mayoritariamente hidráulica (XM, 2026) con una sensibilidad "
    "marcada a fenómenos climáticos como El Niño y a los niveles de los embalses. "
    "Abordarlo aporta valor tanto para los agentes locales como para la comprensión "
    "analítica de un mercado de estas características.",

    "Finalmente, el proyecto no se limita a la construcción de un modelo. Entrega "
    "productos de visualización analítica, incluido un entorno web, que acercan las "
    "predicciones a una lectura de corto plazo accesible, y aplica un flujo "
    "reproducible de ciencia de datos coherente con la formación de la maestría. De "
    "esta forma, los resultados pueden auditarse, comunicarse y reutilizarse, lo que "
    "amplía su utilidad más allá del ejercicio académico.",
]


ALCANCE = [
    "El alcance de este trabajo es la construcción, la evaluación y la comunicación de un "
    "enfoque de predicción del precio de bolsa para el día siguiente sobre datos "
    "históricos. Comprende la preparación de la serie horaria y de las variables "
    "predictoras, la caracterización del comportamiento del precio, la comparación de "
    "estrategias de pronóstico bajo un protocolo homogéneo con un conjunto de prueba "
    "comprendido entre diciembre de 2022 y marzo de 2026, y la comunicación de los "
    "resultados mediante un entorno web de visualización analítica.",

    "Queda fuera del alcance la puesta en operación de un sistema que reciba datos en "
    "tiempo real y genere el pronóstico de una fecha futura real. Aunque el modelo es de "
    "tipo day-ahead y puede predecir cualquier día siguiente a partir de la información "
    "del día base, hacerlo para una fecha posterior al periodo estudiado exigiría una "
    "ingesta y un reprocesamiento continuos de los datos observados, lo que constituye "
    "una extensión operativa que se plantea como trabajo futuro y no como parte de los "
    "productos entregados.",
]


RESUMEN = (
    "El precio de bolsa de energía del mercado eléctrico colombiano se caracteriza por "
    "una alta volatilidad y una fuerte dependencia de las condiciones hidrológicas, lo "
    "que hace de su pronóstico un problema exigente. Este trabajo desarrolla y evalúa un "
    "modelo de predicción del precio horario para el día siguiente, siguiendo un flujo "
    "reproducible de ciencia de datos estructurado con el marco CRISP-DM. A partir de "
    "una serie horaria depurada del periodo 2010 a 2026 se construyó una base supervisada "
    "de 5.930 días con variables de calendario, rezagos, estadísticas móviles y variables "
    "exógenas climáticas (índice ONI) e hidrológicas (nivel de embalses). Se compararon "
    "ocho estrategias de pronóstico, desde la persistencia diaria hasta modelos de "
    "aprendizaje automático y profundo, bajo un protocolo homogéneo con partición "
    "temporal, métricas múltiples (MAE, RMSE, sMAPE y R2), análisis por régimen de precio "
    "y la prueba de significancia de Diebold-Mariano. Los resultados muestran que ninguna "
    "estrategia supera a la persistencia diaria en error absoluto medio, pero varias la "
    "superan de forma significativa en error cuadrático y en el régimen volátil; en "
    "particular, un modelo Lasso con recalibración periódica mejora el RMSE sin degradar "
    "el MAE. Los resultados se comunican mediante un entorno web interactivo. La "
    "contribución principal no es un modelo inédito, sino una evaluación reproducible y "
    "verificable de los límites de la mejora alcanzable frente a una línea base exigente."
)

PALABRAS_CLAVE = (
    "pronóstico de precios de electricidad, mercado eléctrico colombiano, predicción "
    "para el día siguiente, validación temporal, persistencia diaria, CRISP-DM."
)

ABSTRACT = (
    "The day-ahead electricity spot price of the Colombian market is highly volatile and "
    "strongly dependent on hydrological conditions, which makes its forecasting a "
    "demanding problem. This work develops and evaluates an hourly day-ahead price "
    "forecasting model, following a reproducible data science workflow structured with "
    "the CRISP-DM framework. From a cleaned hourly series covering 2010 to 2026, a "
    "supervised dataset of 5,930 days was built with calendar features, lags, rolling "
    "statistics and climatic (ONI index) and hydrological (reservoir level) exogenous "
    "variables. Eight forecasting strategies were compared, from daily persistence to "
    "machine and deep learning models, under a homogeneous protocol with temporal "
    "splitting, multiple metrics (MAE, RMSE, sMAPE and R2), price-regime analysis and the "
    "Diebold-Mariano significance test. Results show that no strategy outperforms daily "
    "persistence in mean absolute error, but several significantly outperform it in "
    "squared error and in the volatile regime; in particular, a Lasso model with periodic "
    "recalibration improves RMSE without degrading MAE. Results are communicated through "
    "an interactive web application. The main contribution is not a novel model but a "
    "reproducible, verifiable evaluation of the limits of achievable improvement against "
    "a strong baseline."
)

KEYWORDS = (
    "electricity price forecasting, Colombian electricity market, day-ahead prediction, "
    "temporal validation, daily persistence, CRISP-DM."
)


def _clave(doc, etiqueta, texto):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(etiqueta)
    r.bold = True
    r.font.name = FUENTE
    r.font.size = Pt(12)
    r2 = p.add_run(texto)
    r2.font.name = FUENTE
    r2.font.size = Pt(12)
    return p


def resumen(doc):
    titulo1(doc, "Resumen")
    parrafo(doc, RESUMEN)
    _clave(doc, "Palabras clave: ", PALABRAS_CLAVE)
    titulo1(doc, "Abstract")
    parrafo(doc, ABSTRACT)
    _clave(doc, "Keywords: ", KEYWORDS)


def planteamiento(doc):
    titulo1(doc, "Planteamiento del problema y Justificación")
    titulo2(doc, "Planteamiento del problema")
    for parr in PLANTEAMIENTO:
        parrafo(doc, parr)
    parrafo(doc,
            "La pregunta se estructura con el marco PICOC, adaptación reconocida en "
            "ingeniería y computación para preguntas comparativas de carácter empírico "
            "(Kitchenham y Charters, 2007). Sus componentes se resumen en la Tabla 1.")
    tabla_dos_columnas(doc, PICOC, ["Componente", "En la investigación"],
                       "Componentes de la pregunta de investigación según el marco PICOC")
    titulo2(doc, "Justificación")
    for parr in JUSTIFICACION:
        parrafo(doc, parr)
    titulo2(doc, "Alcance")
    for parr in ALCANCE:
        parrafo(doc, parr)


def objetivos(doc):
    titulo1(doc, "Objetivos")
    titulo2(doc, "Objetivo general")
    parrafo(doc, OBJETIVO_GENERAL)
    titulo2(doc, "Objetivos específicos")
    for i, texto in enumerate(OBJETIVOS_ESPECIFICOS, start=1):
        objetivo_numerado(doc, i, texto)


MARCO_FORMACION = [
    "El precio de bolsa de energía se forma en el mercado mayorista de electricidad de "
    "Colombia, administrado por XM como operador del sistema y del mercado, bajo la "
    "regulación de la Comisión de Regulación de Energía y Gas (CREG). En este mercado, "
    "los "
    "generadores presentan diariamente una oferta de precio por cada hora del día "
    "siguiente, y a partir de esas ofertas se determina el precio horario.",

    "El cálculo del precio se establece en la Resolución CREG 024 de 1995 (CREG, 1995), "
    "modificada por la Resolución CREG 051 de 2009 (CREG, 2009). De acuerdo con esta "
    "reglamentación, el precio "
    "horario en la bolsa corresponde al precio de oferta de la planta marginal, es "
    "decir, la última unidad no inflexible requerida para atender la demanda en el "
    "despacho ideal, "
    "más un valor adicional asociado a los costos de arranque y parada de las plantas "
    "térmicas. El despacho ideal se calcula para los veinticuatro periodos horarios del "
    "día, con la demanda y la disponibilidad reales del sistema.",

    "Este mecanismo marginalista, unido a una matriz de generación mayoritariamente "
    "hidráulica, explica buena parte del comportamiento del precio: cuando la hidrología "
    "es favorable, la oferta hidráulica de bajo costo fija precios reducidos, mientras "
    "que en periodos secos la entrada de unidades térmicas más costosas eleva el precio "
    "y da lugar a episodios extremos. Por ello, la formación del precio no puede "
    "entenderse sin tener en cuenta las condiciones hidrológicas y climáticas del "
    "país.",
]


MARCO_DETERMINANTES = [
    "Dado el peso de la generación hidráulica, el nivel de los embalses es uno de los "
    "principales determinantes del precio de bolsa. Cuando los embalses están altos, el "
    "sistema despacha energía hidráulica de bajo costo y el precio tiende a ser bajo y "
    "estable; cuando descienden, se recurre a generación térmica más costosa y el precio "
    "sube, con mayor probabilidad de episodios extremos. El nivel de embalses resume, en "
    "buena medida, la disponibilidad del recurso hídrico en el corto y mediano plazo.",

    "El comportamiento hidrológico está ligado a la variabilidad climática, en "
    "particular al fenómeno El Niño-Oscilación del Sur, cuya fase cálida se asocia con "
    "menores aportes hídricos y presiones al alza sobre el precio, y su fase fría con lo "
    "contrario. El Índice Oceánico de El Niño, publicado por la NOAA, sintetiza esta "
    "señal climática y aporta información anticipada sobre las condiciones de mediano "
    "plazo.",

    "Por el lado de la demanda, el consumo del sistema sigue patrones regulares "
    "asociados a las rutinas de los distintos sectores: variaciones a lo largo del día, "
    "con horas de mayor consumo, y diferencias entre días hábiles, sábados, domingos y "
    "festivos. Estos patrones de calendario, junto con la estacionalidad horaria, "
    "estructuran el perfil típico del precio y justifican su tratamiento como una serie "
    "con componentes horarios y de tipo de día.",
]


MARCO_ESTADO_ARTE = [
    "El pronóstico del precio de electricidad es un campo consolidado a escala "
    "internacional, con revisiones que sistematizan métodos, métricas y buenas "
    "prácticas (Weron, 2014; Lago et al., 2021; Piotrowski, 2026; Dudek, 2026). En "
    "contraste, la producción específica sobre el mercado colombiano es reducida: en la "
    "revisión sistemática de este trabajo, realizada en Scopus y Web of Science para el "
    "periodo 2021 a 2026, la búsqueda general identificó 190 registros, mientras que la "
    "búsqueda acotada al caso colombiano arrojó apenas 16, lo que confirma la escasez de "
    "literatura local. A esta búsqueda sistemática se sumó una búsqueda complementaria de "
    "tipo manual, por revisión de referencias y por autor, para incorporar literatura "
    "fundacional y específica del caso colombiano publicada fuera de la ventana temporal: "
    "tanto los trabajos pioneros de pronóstico del precio de bolsa en Colombia como la "
    "evidencia clásica sobre la relación entre el fenómeno ENSO y la hidrología nacional.",

    "En cuanto a los enfoques, la línea estadística y econométrica modela el precio con "
    "métodos de series de tiempo que capturan la estacionalidad y la memoria larga "
    "(Balagula et al., 2026), así como con formulaciones funcionales aplicadas al caso "
    "colombiano (Gallón y Barrientos, 2021). Estos métodos aportan interpretabilidad, pero "
    "pueden quedar limitados ante la fuerte no linealidad y volatilidad del precio.",

    "La línea de aprendizaje automático y aprendizaje profundo ha ganado protagonismo en "
    "los años recientes. Se han propuesto arquitecturas basadas en transformadores "
    "(Fan et al., 2026a, 2026b), modelos híbridos y redes neuronales (Li et al., 2026; "
    "Radziukynienė, 2026), "
    "enfoques en dos etapas (Aksu et al., 2026), redes sobre grafos (Aslam et al., 2026), esquemas de "
    "conjunto (Wei, 2026), aprendizaje en línea con recalibración (Salihoglu et al., 2026) y "
    "métodos probabilísticos que cuantifican la incertidumbre (Ni et al., 2026), junto con un "
    "interés creciente por la interpretabilidad mediante inteligencia artificial "
    "explicable (Biró et al., 2026). Para el mercado colombiano, se han explorado modelos de "
    "aprendizaje profundo sobre el precio spot (Trespalacios y Franco Ceballos, 2025), la selección de "
    "variables predictoras (Gonzalez-Sierra, 2024), arquitecturas de tipo transformador "
    "probabilístico (Cantillo-Luna et al., 2023) y modelos Time2Vec-LSTM para el intradía "
    "(Cantillo-Luna et al., 2024), y el pronóstico de ofertas para el día siguiente "
    "(Herrera-Mejía et al., 2026).",

    "Un aspecto transversal, y decisivo para este proyecto, es la evaluación. La "
    "literatura insiste en comparar los modelos contra líneas base competitivas y en "
    "emplear métricas y criterios de valoración estadística y económica adecuados "
    "(Weron, 2014; Lago et al., 2021; Maciejowska et al., 2026), y advierte que muchas mejoras "
    "reportadas no se sostienen bajo un contraste riguroso. En consecuencia, la brecha "
    "que aborda este trabajo no es proponer un modelo inédito, sino evaluar de forma "
    "reproducible, bajo un mismo protocolo, si un enfoque más complejo supera de manera "
    "verificable a una línea base fuerte en el mercado colombiano.",
]


MARCO_EVALUACION = [
    "La evaluación de un pronóstico requiere una referencia frente a la cual medir su "
    "desempeño. En el pronóstico de precios eléctricos, la línea base más exigente suele "
    "ser la persistencia, que toma como pronóstico un valor observado reciente. En este "
    "trabajo se adopta la persistencia diaria, que asigna a cada hora del día siguiente "
    "el precio observado a esa misma hora del día anterior. Pese a su sencillez, es una "
    "referencia difícil de superar y, por ello, constituye el patrón de comparación "
    "central (Lago et al., 2021).",

    "La forma de dividir los datos condiciona la validez de las conclusiones. Al tratarse "
    "de una serie de tiempo, la evaluación debe respetar el orden cronológico: el modelo "
    "se entrena con el pasado y se evalúa con el futuro, sin mezclar aleatoriamente las "
    "observaciones. En este proyecto se emplea una partición temporal, en la que un tramo "
    "final de la serie se reserva para la prueba, de modo que el desempeño reportado "
    "refleje una situación realista de pronóstico.",

    "El desempeño se mide con métricas de error de uso extendido, como el error absoluto "
    "medio y la raíz del error cuadrático medio, complementadas con un análisis "
    "desagregado por hora, por tipo de día y durante los episodios de alta volatilidad. "
    "Este análisis desagregado es relevante porque un modelo puede comportarse bien en "
    "promedio y mal en las horas o los periodos más críticos.",

    "Para determinar si las diferencias de desempeño entre dos enfoques son "
    "estadísticamente significativas, y no producto del azar, se utiliza la prueba de "
    "Diebold-Mariano (Diebold y Mariano, 1995). En conjunto, estos elementos, una línea "
    "base competitiva, una partición temporal, métricas múltiples con análisis "
    "desagregado y pruebas de significancia, conforman el protocolo homogéneo de "
    "evaluación bajo el cual se comparan todas las estrategias, lo que garantiza que las "
    "comparaciones sean justas y reproducibles.",
]


def marco(doc):
    titulo1(doc, "Marco de referencia")
    titulo2(doc, "El precio de bolsa de energía y su formación")
    for parr in MARCO_FORMACION:
        parrafo(doc, parr)
    titulo2(doc, "Determinantes del precio: hidrología, clima y demanda")
    for parr in MARCO_DETERMINANTES:
        parrafo(doc, parr)
    titulo2(doc, "El pronóstico del precio de electricidad: enfoques y estado del arte")
    for parr in MARCO_ESTADO_ARTE:
        parrafo(doc, parr)
    titulo2(doc, "Evaluación y líneas base")
    for parr in MARCO_EVALUACION:
        parrafo(doc, parr)


METODOLOGIA_ENFOQUE = [
    "El desarrollo del proyecto, planteado como un flujo reproducible de ciencia de "
    "datos, se estructura sobre CRISP-DM (Cross-Industry Standard Process for Data "
    "Mining), un marco de proceso que organiza el trabajo en fases "
    "iterativas: comprensión del negocio, comprensión de los datos, preparación de los "
    "datos, modelado, evaluación y despliegue (Chapman et al., 2000). Se adopta por ser "
    "el estándar de facto todavía más utilizado en proyectos de minería y ciencia de "
    "datos (Martínez-Plumed et al., 2021). Dado que los proyectos de ciencia de datos "
    "son más exploratorios que la minería de datos clásica, la literatura reciente "
    "plantea flexibilizar su secuencia rígida hacia trayectorias adaptadas al problema "
    "(Martínez-Plumed et al., 2021); en línea con ello, este trabajo adapta el marco al "
    "pronóstico de series temporales incorporando validación cronológica en todas las "
    "etapas, de modo que ninguna decisión utilice información futura respecto al "
    "horizonte predicho (Weron, 2014; Lago et al., 2021).",

    "Las fases de CRISP-DM se instancian en los objetivos específicos del proyecto, de "
    "manera que el proceso metodológico y la estructura del documento comparten el mismo "
    "orden. La comprensión del negocio corresponde al planteamiento del problema de "
    "predicción del precio de bolsa para el día siguiente en el mercado eléctrico "
    "colombiano. La preparación de los datos consolida y depura la serie horaria del "
    "precio y genera las variables predictoras y los objetivos para el día siguiente "
    "(objetivo 1). La comprensión de los datos caracteriza el comportamiento histórico "
    "del precio y el aporte de las variables exógenas mediante análisis exploratorio "
    "(objetivo 2). El modelado y la evaluación valoran, bajo un protocolo homogéneo, el "
    "desempeño de líneas base, modelos y estrategias de pronóstico, y seleccionan un "
    "enfoque defendible (objetivo 3). El despliegue comunica los resultados a través de "
    "productos de visualización analítica, incluido un entorno web (objetivo 4).",

    "Coherente con el ciclo iterativo de CRISP-DM, todo el flujo se implementa de forma "
    "reproducible: cada etapa se encapsula en módulos de código versionados, los "
    "resultados intermedios se almacenan en archivos estandarizados y las decisiones "
    "metodológicas quedan documentadas. Esta trazabilidad permite auditar y repetir el "
    "análisis, y es coherente con el criterio de evaluación adoptado, que prioriza "
    "conclusiones verificables sobre comparaciones optimistas.",
]


def metodologia(doc):
    titulo1(doc, "Metodología")
    titulo2(doc, "Enfoque metodológico y flujo de trabajo")
    for parr in METODOLOGIA_ENFOQUE:
        parrafo(doc, parr)


# --------------------------------------------------------------------------- #
# Capítulo 1. Preparación de los datos (objetivo 1)
# --------------------------------------------------------------------------- #

CAP1_ENFOQUE = [
    "Este capítulo desarrolla el primer objetivo específico: construir la serie horaria "
    "depurada del precio de bolsa y las variables predictoras para el día siguiente. "
    "Corresponde a la fase de preparación de los datos de CRISP-DM y produce el insumo "
    "sobre el que se apoyan los tres capítulos siguientes: una base supervisada, "
    "trazable y lista para el modelado del precio del día siguiente. El desarrollo se "
    "organiza en tres momentos: la consolidación y depuración de las fuentes, la "
    "construcción de las variables predictoras y la definición de las variables "
    "objetivo que representan las veinticuatro horas del día siguiente.",
]

CAP1_FUENTES = [
    "El análisis integra tres fuentes de datos. La serie del precio de bolsa nacional, "
    "publicada por XM, constituye la variable central, expresada en pesos colombianos por "
    "kilovatio hora (COP/kWh), y se recibe en formato ancho, con una fila por día y "
    "veinticuatro columnas horarias. La serie de nivel de embalses, "
    "como indicador de la disponibilidad del recurso hídrico, aporta la principal "
    "variable exógena hidrológica. El Índice Oceánico de El Niño (ONI), publicado por la "
    "NOAA, incorpora la señal climática de mediano plazo asociada al fenómeno El "
    "Niño-Oscilación del Sur. La elección de estas variables exógenas responde a los "
    "determinantes del precio descritos en el marco de referencia: la formación del "
    "precio en un mercado mayoritariamente hidráulico no puede entenderse sin las "
    "condiciones hidrológicas y climáticas del país.",
]

CAP1_FUENTES_TABLA = [
    ("Precio de bolsa (XM)",
     "Serie horaria del precio de bolsa nacional (PB_Nal). Variable central del "
     "problema y base de los objetivos de pronóstico."),
    ("Nivel de embalses",
     "Volumen útil embalsado, en porcentaje. Variable exógena hidrológica que resume "
     "la disponibilidad del recurso hídrico."),
    ("Índice Oceánico de El Niño (NOAA)",
     "Índice ONI. Variable exógena climática con información anticipada de mediano "
     "plazo sobre las condiciones hidrológicas."),
]

CAP1_DEPURACION = [
    "El precio se transforma de su formato ancho diario a un formato largo horario, en "
    "el que cada registro corresponde a una hora específica, reconstruyendo la marca "
    "temporal a partir de la fecha y del número de hora. Sobre esa estructura se aplica "
    "un protocolo homogéneo de depuración: conversión de tipos a fecha y a valor "
    "numérico, eliminación de los registros sin fecha o sin precio válido, y "
    "ordenamiento cronológico estricto.",

    "Las series exógenas se alinean a la misma rejilla temporal. El nivel de embalses, "
    "de frecuencia mensual, se expande a la resolución diaria conservando el valor del "
    "periodo, y el ONI se incorpora con la misma lógica de alineación. El resultado de "
    "esta etapa es una serie horaria continua y ordenada de 142.344 registros desde "
    "2010, que constituye la base común para el análisis exploratorio y la ingeniería de "
    "variables.",
]

CAP1_VARIABLES = [
    "A partir de la serie depurada se construyen las variables predictoras, agrupadas "
    "por su naturaleza. Las variables de calendario capturan la estacionalidad anual, "
    "semanal y el tipo de día; sus versiones cíclicas, mediante transformaciones de "
    "seno y coseno, preservan la continuidad de las magnitudes periódicas, de modo que "
    "horas o meses contiguos queden próximos en el espacio de variables.",

    "Los rezagos y las estadísticas móviles del precio codifican la memoria de corto "
    "plazo y la tendencia reciente de la serie, y las medias móviles exponenciales "
    "ponderan con mayor peso la información más próxima. Las variables exógenas "
    "incorporan el ONI y el nivel de embalses junto con sus rezagos. Por último, un "
    "conjunto de descriptores derivados del perfil horario, de las horas pico y valle y "
    "de la volatilidad resume la forma de la curva de precios del día base. La Tabla 3 "
    "sintetiza los grupos de variables construidos.",
]

CAP1_VARIABLES_TABLA = [
    ("Temporales de calendario",
     "Año, mes, día, día de la semana, semana del año y trimestre, entre otras."),
    ("Cíclicas",
     "Codificación seno-coseno de la hora, el día de la semana y el mes."),
    ("Rezagos del precio",
     "Valores pasados del precio a distintos horizontes, incluidos rezagos por hora."),
    ("Estadísticas móviles del precio",
     "Medias y desviaciones móviles sobre ventanas de distinta amplitud."),
    ("Medias exponenciales (EWMA)",
     "Ponderación exponencial del precio y del nivel de embalses."),
    ("Exógenas",
     "ONI y nivel de embalses diario, con sus rezagos."),
    ("Perfil horario",
     "Descriptores del perfil promedio, máximo y de dispersión del día base."),
    ("Horas pico y valle",
     "Indicadores de los niveles en las horas de mayor y menor precio."),
    ("Volatilidad",
     "Razón de rango, desviación diaria y dispersión del perfil."),
    ("Objetivos (day-ahead)",
     "Precio de cada una de las 24 horas del día siguiente y precio promedio diario "
     "del día siguiente."),
]

CAP1_OBJETIVOS = [
    "Para formular el problema como pronóstico del día siguiente, la base se organiza de "
    "modo que cada fila representa un día base y reúne toda la información disponible "
    "hasta ese día, mientras que las variables objetivo describen el día siguiente. Se "
    "definen veinticuatro objetivos horarios, uno por cada hora del día siguiente, y un "
    "objetivo agregado correspondiente al precio promedio diario del día siguiente.",

    "Esta organización garantiza, por diseño, que ninguna variable predictora utilice "
    "información posterior al día base, requisito indispensable para una evaluación "
    "temporal honesta y coherente con el protocolo del capítulo 3. De este modo, la "
    "estructura de los datos incorpora desde el inicio la restricción cronológica que "
    "exige el pronóstico operativo.",
]

CAP1_SINTESIS = [
    "El proceso produce una base supervisada de 5.930 días y 333 columnas, entre "
    "variables predictoras y objetivos, con cobertura del 1 de enero de 2010 al 27 de "
    "marzo de 2026. Cada etapa queda encapsulada en módulos de código versionados y sus "
    "salidas se almacenan en archivos estandarizados, lo que asegura la reproducibilidad "
    "y la trazabilidad del insumo. Con esta base disponible, el capítulo siguiente "
    "caracteriza el comportamiento histórico del precio y el aporte predictivo de las "
    "variables exógenas.",
]


def capitulo1(doc):
    titulo1(doc, "Capítulo 1. De la serie histórica a la base supervisada de pronóstico")
    titulo2(doc, "Enfoque del capítulo")
    for parr in CAP1_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Fuentes de datos")
    for parr in CAP1_FUENTES:
        parrafo(doc, parr)
    tabla_dos_columnas(doc, CAP1_FUENTES_TABLA, ["Fuente", "Contenido y uso"],
                       "Fuentes de datos integradas en la construcción de la base")
    titulo2(doc, "Depuración y estructuración de la serie horaria")
    for parr in CAP1_DEPURACION:
        parrafo(doc, parr)
    titulo2(doc, "Ingeniería de variables predictoras")
    for parr in CAP1_VARIABLES:
        parrafo(doc, parr)
    tabla_dos_columnas(doc, CAP1_VARIABLES_TABLA, ["Grupo de variables", "Contenido"],
                       "Grupos de variables construidos en la ingeniería de variables")
    titulo2(doc, "Variables objetivo y estructura para el día siguiente")
    for parr in CAP1_OBJETIVOS:
        parrafo(doc, parr)
    titulo2(doc, "Base supervisada resultante y síntesis")
    for parr in CAP1_SINTESIS:
        parrafo(doc, parr)


# --------------------------------------------------------------------------- #
# Capítulo 2. Comprensión de los datos (objetivo 2)
# --------------------------------------------------------------------------- #

CAP2_ENFOQUE = [
    "Este capítulo desarrolla el segundo objetivo específico: caracterizar el "
    "comportamiento histórico del precio horario de bolsa y el aporte predictivo de las "
    "variables exógenas. Corresponde a la fase de comprensión de los datos de CRISP-DM y "
    "su propósito es doble: describir los patrones que estructuran el precio y justificar, "
    "con evidencia, las decisiones de modelado y de evaluación de los capítulos "
    "siguientes. El análisis se organiza en torno al perfil horario, la asimetría y los "
    "eventos extremos, la estructura temporal de la serie y la relación del precio con "
    "las variables exógenas.",
]

CAP2_PERFIL = [
    "El precio presenta un perfil horario regular, que sintetiza la Figura 1. Alcanza "
    "su nivel más bajo en la "
    "madrugada, con un valle alrededor de las tres de la mañana (precio promedio cercano "
    "a 204 COP/kWh y mediana cercana a 132 COP/kWh), asciende de forma sostenida durante "
    "el día y registra su punto más alto en las horas de la noche. Las horas de mayor "
    "precio se concentran entre las dieciocho y las veintiuna, con un máximo a las "
    "diecinueve horas (precio promedio cercano a 305 COP/kWh y mediana cercana a 202 "
    "COP/kWh). Este patrón "
    "justifica el tratamiento de la serie con componentes horarios y la evaluación "
    "desagregada por hora.",

    "El perfil también varía según el tipo de día. Los días hábiles muestran el perfil "
    "más pronunciado, mientras que los domingos y festivos presentan niveles más bajos y "
    "una curva más aplanada, en coherencia con los patrones de consumo del sistema. "
    "Esta diferencia respalda la inclusión de variables de calendario y de tipo de día "
    "entre los predictores.",
]

CAP2_EXTREMOS = [
    "La distribución del precio es marcadamente asimétrica. En todas las horas el precio "
    "promedio supera a la mediana, lo que indica un sesgo a la derecha y la presencia de "
    "colas pesadas: episodios poco frecuentes pero de magnitud muy elevada que elevan el "
    "promedio por encima del valor típico. La dispersión, medida por la desviación "
    "estándar, es mayor en las horas de la tarde y la noche, que son también las de "
    "mayor precio.",

    "Para dimensionar los episodios extremos se definieron umbrales sobre los percentiles "
    "altos de la serie. El umbral del percentil 95 se sitúa en torno a 797 y es superado "
    "por el 5,0 por ciento de las observaciones (7.120 registros), mientras que el umbral "
    "del percentil 99, cercano a 1.072, es superado por el 1,0 por ciento (1.433 "
    "registros). Estos eventos no se reparten de manera uniforme: se concentran en los "
    "días hábiles y son menos frecuentes en domingos y festivos, como resume la "
    "Tabla 4.",
]

CAP2_EXTREMOS_TABLA = [
    ("Días hábiles",
     "5,3 por ciento de las observaciones sobre el percentil 95 y 1,1 por ciento sobre "
     "el percentil 99."),
    ("Sábados",
     "5,0 por ciento sobre el percentil 95 y 0,8 por ciento sobre el percentil 99."),
    ("Domingos",
     "4,0 por ciento sobre el percentil 95 y 0,6 por ciento sobre el percentil 99."),
    ("Festivos",
     "3,7 por ciento sobre el percentil 95 y 0,9 por ciento sobre el percentil 99."),
]

CAP2_TEMPORAL = [
    "La estructura temporal de la serie se examinó con pruebas de estacionariedad y con "
    "el análisis de la dependencia entre días consecutivos. La prueba de Dickey-Fuller "
    "aumentada indica que el precio promedio diario es estacionario al cinco por ciento "
    "(estadístico cercano a -4,94, con valor p inferior a 0,001), al igual que el precio "
    "promedio mensual. Esta estacionariedad favorece el uso de modelos que operan sobre "
    "el nivel de la serie sin requerir una diferenciación previa.",

    "Al mismo tiempo, la serie exhibe una fuerte memoria de corto plazo. La correlación "
    "entre el precio de una hora y el de esa misma hora del día anterior es muy alta, "
    "entre 0,92 y 0,96 según la hora. Esta elevada autocorrelación diaria anticipa un "
    "resultado central de este trabajo: la persistencia diaria es una línea base "
    "especialmente difícil de superar, lo que refuerza su elección como patrón de "
    "comparación en el capítulo de evaluación.",
]

CAP2_EXOGENAS = [
    "El aporte de las variables exógenas se valoró mediante su relación con el precio "
    "promedio diario. El Índice Oceánico de El Niño presenta una correlación positiva "
    "moderada (cercana a 0,45): su fase cálida, asociada a menores aportes hídricos, "
    "coincide con precios más altos. El nivel de embalses muestra una correlación "
    "negativa débil en términos contemporáneos (cercana a -0,17), coherente con el "
    "efecto esperado de una mayor disponibilidad hídrica sobre el precio.",

    "El valor predictivo del clima se aprecia mejor en su carácter anticipado. La "
    "correlación entre el ONI y el nivel de embalses, que ilustra la Figura 2, se vuelve "
    "más negativa a medida que "
    "aumenta el rezago, pasando de -0,20 de manera contemporánea a -0,49 con un rezago "
    "de ciento ochenta días. En otras palabras, la señal climática anticipa en varios "
    "meses la evolución del recurso hídrico, que a su vez condiciona el precio. Esto "
    "justifica incorporar el ONI y el nivel de embalses, con sus rezagos, como "
    "predictores de mediano plazo. La Tabla 5 resume estas relaciones.",
]

CAP2_EXOGENAS_TABLA = [
    ("Índice ONI (El Niño)",
     "Correlación positiva moderada con el precio (cercana a 0,45); la fase cálida se "
     "asocia a precios más altos."),
    ("Nivel de embalses",
     "Correlación negativa débil y contemporánea con el precio (cercana a -0,17); mayor "
     "disponibilidad hídrica, menor precio."),
    ("ONI anticipado",
     "Anticipa el nivel de embalses: la correlación ONI-embalses pasa de -0,20 a -0,49 "
     "al aumentar el rezago hasta ciento ochenta días."),
]

CAP2_SINTESIS = [
    "La caracterización confirma que el precio es una serie con estacionalidad horaria y "
    "de tipo de día, distribución asimétrica con eventos extremos concentrados en días "
    "hábiles, estacionariedad en nivel y fuerte memoria de corto plazo, y una relación "
    "con las variables exógenas climáticas e hidrológicas que opera principalmente en el "
    "mediano plazo. Estos hallazgos fundamentan las decisiones del capítulo siguiente, "
    "donde se evalúan líneas base, modelos y estrategias de pronóstico bajo un protocolo "
    "homogéneo.",
]


def capitulo2(doc):
    titulo1(doc, "Capítulo 2. El comportamiento del precio y la señal de las exógenas")
    titulo2(doc, "Enfoque del capítulo")
    for parr in CAP2_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Perfil horario y estacionalidad")
    for parr in CAP2_PERFIL:
        parrafo(doc, parr)
    figura(doc, "eda_perfil_horario.png",
           "Perfil horario del precio de bolsa",
           nota="Promedio, mediana y percentil 95 por hora del día. La franja sombreada "
                "marca las horas pico (18 a 21).")
    titulo2(doc, "Asimetría, volatilidad y eventos extremos")
    for parr in CAP2_EXTREMOS:
        parrafo(doc, parr)
    tabla_dos_columnas(doc, CAP2_EXTREMOS_TABLA,
                       ["Tipo de día", "Frecuencia de eventos extremos"],
                       "Frecuencia de eventos extremos del precio según el tipo de día")
    titulo2(doc, "Estructura temporal: estacionariedad y persistencia")
    for parr in CAP2_TEMPORAL:
        parrafo(doc, parr)
    titulo2(doc, "Aporte predictivo de las variables exógenas")
    for parr in CAP2_EXOGENAS:
        parrafo(doc, parr)
    tabla_dos_columnas(doc, CAP2_EXOGENAS_TABLA,
                       ["Variable exógena", "Relación con el precio"],
                       "Relación de las variables exógenas con el precio promedio diario")
    figura(doc, "eda_corr_oni_embalses.png",
           "Correlación entre el ONI y el nivel de embalses según el rezago",
           nota="La relación se hace más negativa a mayor rezago en días, lo que "
                "evidencia la capacidad anticipatoria del Índice Oceánico de El Niño "
                "sobre el nivel de embalses.")
    titulo2(doc, "Síntesis y enlace")
    for parr in CAP2_SINTESIS:
        parrafo(doc, parr)


# --------------------------------------------------------------------------- #
# Capítulo 3. Modelado y evaluación (objetivo 3)
# --------------------------------------------------------------------------- #

CAP3_ENFOQUE = [
    "Este capítulo desarrolla el tercer objetivo específico: evaluar el desempeño de "
    "líneas base, modelos y estrategias de pronóstico bajo un protocolo homogéneo con "
    "métricas múltiples, orientado a la selección de un enfoque defendible. Corresponde "
    "a las fases de modelado y evaluación de CRISP-DM. A diferencia de los capítulos "
    "anteriores, el eje no es proponer un modelo inédito, sino determinar, con un "
    "contraste justo y reproducible, si un enfoque más complejo supera de manera "
    "verificable a una línea base fuerte en el mercado colombiano.",
]

CAP3_PROTOCOLO = [
    "Todas las estrategias se evalúan bajo el mismo protocolo, condición necesaria para "
    "que las comparaciones sean legítimas. Los datos se dividen mediante una partición "
    "temporal 80/20, en la que el tramo final de la serie se reserva para prueba: el "
    "conjunto de prueba abarca 1.186 días (28.464 observaciones horarias), del período "
    "de diciembre de 2022 a marzo de 2026. El entrenamiento emplea solo información "
    "anterior, de modo que ninguna predicción utilice datos futuros.",

    "El desempeño se mide con métricas de error complementarias: el error absoluto medio "
    "(MAE), la raíz del error cuadrático medio (RMSE), el error porcentual absoluto medio "
    "simétrico (sMAPE) y el coeficiente de determinación (R2). El MAE penaliza por igual "
    "todos los errores, mientras que el RMSE castiga con más fuerza los errores grandes, "
    "por lo que ambos ofrecen lecturas distintas y relevantes en una serie con eventos "
    "extremos. La comparación se complementa con un análisis desagregado por régimen de "
    "precio y con pruebas de significancia estadística.",

    "La línea base es la persistencia diaria, que asigna a cada hora del día siguiente el "
    "precio observado a esa misma hora del día anterior. Como se mostró en el capítulo 2, "
    "la fuerte autocorrelación diaria del precio hace de esta referencia un rival "
    "exigente. Para decidir si una diferencia de desempeño es real y no fruto del azar se "
    "aplica la prueba de Diebold-Mariano (Diebold y Mariano, 1995).",
]

CAP3_ESTRATEGIAS = [
    "Se evaluaron ocho estrategias de distinta complejidad y familia, desde la línea "
    "base hasta el aprendizaje profundo, lo que permite situar el aporte de cada nivel "
    "de sofisticación frente a la referencia.",
]

CAP3_ESTRATEGIAS_TABLA = [
    ["Persistencia diaria", "Línea base"],
    ["Híbrido con corrección en horas pico", "Estrategia avanzada"],
    ["Aprendizaje residual", "Estrategia avanzada"],
    ["Segmentación por tipo de día", "Estrategia avanzada"],
    ["Modelo de dos etapas", "Estrategia avanzada"],
    ["Lasso horario", "Aprendizaje automático"],
    ["LSTM", "Aprendizaje profundo"],
    ["Prophet", "Serie temporal especializada"],
]

CAP3_RESULTADOS = [
    "La Tabla 7 resume el desempeño de cada estrategia sobre el conjunto de "
    "prueba. El hallazgo más relevante es que ninguna estrategia mejora el MAE de la "
    "persistencia diaria: la referencia obtiene el menor error absoluto medio. Sin "
    "embargo, varias estrategias logran un RMSE y un R2 claramente mejores, lo que "
    "indica que, sin reducir el error promedio, sí atenúan los errores más grandes. La "
    "Figura 3 ilustra este compromiso entre MAE y RMSE.",
]

CAP3_RESULTADOS_TABLA = [
    ["Persistencia diaria", "68,9", "157,9", "0,828"],
    ["Híbrido con corrección en picos", "69,0", "155,8", "0,832"],
    ["Aprendizaje residual", "70,5", "148,7", "0,847"],
    ["Segmentación por tipo de día", "74,4", "144,9", "0,855"],
    ["Lasso horario", "75,7", "148,8", "0,847"],
    ["Modelo de dos etapas", "77,9", "149,0", "0,847"],
    ["LSTM", "79,9", "165,2", "0,812"],
    ["Prophet", "189,4", "270,5", "0,495"],
]

CAP3_SIGNIFICANCIA = [
    "La prueba de Diebold-Mariano confirma la lectura anterior y la dota de respaldo "
    "estadístico. Frente a la persistencia, las estrategias avanzadas y el Lasso "
    "resultan significativamente peores en MAE, pero significativamente mejores en RMSE "
    "(valores p inferiores a 0,001 en ambos sentidos). La única excepción en MAE es el "
    "híbrido con corrección en horas pico, cuya diferencia no es significativa (valor p "
    "de 0,76), es decir, iguala a la persistencia en error absoluto y la mejora en RMSE. "
    "Prophet y la LSTM, en cambio, no son competitivos.",
]

CAP3_SIGNIFICANCIA_TABLA = [
    ["Híbrido con corrección en picos", "Empata (no significativo)", "Mejor (significativo)"],
    ["Aprendizaje residual", "Peor (significativo)", "Mejor (significativo)"],
    ["Segmentación por tipo de día", "Peor (significativo)", "Mejor (significativo)"],
    ["Modelo de dos etapas", "Peor (significativo)", "Mejor (significativo)"],
    ["Lasso horario", "Peor (significativo)", "Mejor (significativo)"],
]

CAP3_REGIMEN = [
    "El desempeño cambia según el régimen de precio. En el régimen calmo (precios por "
    "debajo de 300), la persistencia y el híbrido en horas pico son los más precisos, "
    "mientras que en el régimen volátil (precios iguales o superiores a 300) el Lasso "
    "horario y el aprendizaje residual logran el menor MAE, por debajo de la "
    "persistencia. Es decir, los modelos aportan precisamente donde la referencia es más "
    "débil: en los episodios de alto precio. La Figura 4 muestra esta diferencia por "
    "régimen.",

    "La recalibración periódica refuerza este resultado. Un Lasso reajustado cada mes "
    "con las observaciones más recientes prácticamente iguala a la persistencia en MAE "
    "(69,4 frente a "
    "68,9) y la supera de forma apreciable en RMSE (141,4 frente a 157,9) y en R2 (0,862 "
    "frente a 0,828). La recalibración, por tanto, convierte al modelo lineal en una "
    "alternativa equivalente en error promedio y superior en el control de los errores "
    "grandes.",
]

CAP3_SELECCION = [
    "La evidencia respalda una lectura equilibrada de los resultados. La persistencia "
    "diaria sigue siendo insuperable en el error absoluto medio, por lo que se adopta "
    "como enfoque "
    "principal y referencia del producto de comunicación del capítulo 4. No obstante, "
    "para el control de los errores grandes y para el régimen volátil, el Lasso horario "
    "con recalibración mensual ofrece una mejora verificable, estadísticamente "
    "significativa, sin degradar el error promedio, por lo que se presenta como "
    "complemento defendible de la referencia.",

    "Este resultado es coherente con la literatura, que advierte que muchas mejoras "
    "reportadas no resisten un contraste riguroso frente a una línea base competitiva "
    "(Weron, 2014; Lago et al., 2021; Maciejowska et al., 2026). El aporte del capítulo no es "
    "declarar un modelo vencedor, sino precisar en qué métrica y bajo qué condiciones "
    "conviene cada enfoque, con un protocolo que hace reproducibles las "
    "comparaciones.",
]


def capitulo3(doc):
    titulo1(doc, "Capítulo 3. La prueba del pronóstico: evaluación y selección del enfoque")
    titulo2(doc, "Enfoque del capítulo")
    for parr in CAP3_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Protocolo homogéneo de evaluación")
    for parr in CAP3_PROTOCOLO:
        parrafo(doc, parr)
    titulo2(doc, "Estrategias y modelos considerados")
    for parr in CAP3_ESTRATEGIAS:
        parrafo(doc, parr)
    tabla_n(doc, CAP3_ESTRATEGIAS_TABLA, ["Estrategia", "Familia"],
            "Estrategias de pronóstico evaluadas, por familia")
    titulo2(doc, "Resultados comparativos")
    for parr in CAP3_RESULTADOS:
        parrafo(doc, parr)
    tabla_n(doc, CAP3_RESULTADOS_TABLA,
            ["Estrategia", "MAE", "RMSE", "R²"],
            "Desempeño de las estrategias sobre el conjunto de prueba (MAE y RMSE en "
            "COP/kWh)")
    figura(doc, "cap3_comparacion_estrategias.png",
           "Compromiso entre MAE y RMSE de las estrategias",
           nota="Se excluye Prophet por quedar fuera de rango. La persistencia logra el "
                "menor MAE; varias estrategias reducen el RMSE a cambio de un MAE algo "
                "mayor.")
    titulo2(doc, "Significancia estadística")
    for parr in CAP3_SIGNIFICANCIA:
        parrafo(doc, parr)
    tabla_n(doc, CAP3_SIGNIFICANCIA_TABLA,
            ["Estrategia frente a la persistencia", "En MAE", "En RMSE"],
            "Resultado de la prueba de Diebold-Mariano frente a la persistencia diaria")
    titulo2(doc, "Desempeño por régimen y efecto de la recalibración")
    for parr in CAP3_REGIMEN:
        parrafo(doc, parr)
    figura(doc, "cap3_regimen.png",
           "Error absoluto medio por régimen de precio",
           nota="En el régimen volátil, el Lasso horario y el aprendizaje residual "
                "superan a la persistencia.")
    titulo2(doc, "Selección del enfoque defendible y síntesis")
    for parr in CAP3_SELECCION:
        parrafo(doc, parr)


# --------------------------------------------------------------------------- #
# Capítulo 4. Despliegue y comunicación (objetivo 4)
# --------------------------------------------------------------------------- #

CAP4_ENFOQUE = [
    "Este capítulo desarrolla el cuarto objetivo específico: comunicar los resultados y "
    "el desempeño del enfoque seleccionado mediante productos de visualización "
    "analítica, incluido un entorno web. Corresponde a la fase de despliegue de "
    "CRISP-DM, la que cierra el ciclo al poner los hallazgos a disposición de los grupos "
    "de interés. El capítulo describe cómo el modelo seleccionado se convierte en un "
    "artefacto reutilizable y cómo una aplicación web traduce las predicciones y su "
    "evaluación a una lectura accesible.",
]

CAP4_MODELO = [
    "El punto de partida del despliegue es la serialización del modelo. A partir de la "
    "base supervisada del capítulo 1 se entrena el enfoque de aprendizaje automático "
    "seleccionado, compuesto por veinticuatro modelos Lasso independientes, uno por cada "
    "hora del día siguiente, cada uno con su propia selección de variables y con las "
    "variables exógenas de clima y embalses. Cada modelo se encapsula en una tubería que "
    "imputa valores faltantes, estandariza las variables y ajusta la regresión, con "
    "controles explícitos para evitar el uso de información del día objetivo.",

    "El resultado se guarda como dos artefactos versionados: el modelo entrenado, que "
    "reúne las veinticuatro tuberías con sus listas de variables, y un archivo de "
    "metadatos con la fecha de entrenamiento, el rango de datos, el número de "
    "observaciones y las métricas de referencia (MAE de 77,2, RMSE de 154,7 y R2 de "
    "0,835 sobre el conjunto de prueba). Estos artefactos desacoplan el entrenamiento de "
    "la aplicación: la interfaz no reentrena nada, solo carga el modelo y lo consume.",
]

# Fase 3: segundo modelo serializado (aprendizaje residual) que la aplicación consume.
CAP4_MODELO_RESIDUAL_F3 = (
    "Junto al modelo Lasso se serializa también el modelo de aprendizaje residual, que "
    "corrige la persistencia diaria con un residuo estimado y que, bajo el protocolo de "
    "evaluación homogéneo del capítulo 3, alcanza un MAE de 70,5, un RMSE de 148,7 y un "
    "R2 de 0,847, con una mejora apreciable sobre la persistencia en el régimen volátil. "
    "La aplicación carga ambos modelos y los presenta como alternativas de comparación "
    "frente a la persistencia.")

CAP4_WEB = [
    "La comunicación se materializa en una aplicación web interactiva, construida con "
    "Streamlit, que consume los artefactos del modelo y presenta la información en {n} "
    "vistas complementarias. El diseño prioriza la claridad para un público no "
    "especializado: lenguaje llano, un modelo de referencia explícito y comparaciones "
    "visuales directas, con modo claro y oscuro.",
]

# Vista adicional incorporada en la Fase 3 (evaluación del modelo).
CAP4_WEB_VISTA_DESEMPENO = (
    "Desempeño del modelo",
    "Evaluación del pronóstico sobre el conjunto de prueba: error medio por hora del día "
    "y desempeño por régimen de precio (calmo y volátil), que evidencia dónde el modelo "
    "supera a la persistencia.")

# Versión ampliada (Fase 3) de la vista de predicción: suma el aprendizaje residual y el
# intervalo de incertidumbre conformal.
CAP4_WEB_VISTA_PREDICCION_F3 = (
    "Predicción del día siguiente",
    "Pronóstico horario del día siguiente con el modelo Lasso, el aprendizaje residual y "
    "la persistencia, frente al precio real observado, con un intervalo de incertidumbre "
    "(predicción conformal) y métricas comparativas por día.")

# Vista incorporada en la Fase 3 para dar funcionalidad operativa al prototipo.
CAP4_WEB_VISTA_CARGA = (
    "Pronóstico con datos nuevos",
    "Permite cargar un archivo con uno o más días base en el formato de la base del "
    "proyecto y genera el pronóstico del día siguiente con los modelos, como paso hacia "
    "el uso operativo de la herramienta.")

CAP4_WEB_TABLA = [
    ("Histórico del precio",
     "Serie histórica del precio con filtros de período y de agregación (diaria, semanal "
     "o mensual) y métricas de resumen."),
    ("Explorar patrones",
     "Distribución del precio por hora del día y por tipo de día, para reconocer los "
     "patrones descritos en el capítulo 2."),
    ("Clima y precio",
     "Relación del precio con las condiciones climáticas e hidrológicas (Índice ONI y "
     "nivel de embalses)."),
    ("Predicción del día siguiente",
     "Pronóstico horario del día siguiente con el modelo Lasso y la persistencia, frente "
     "al precio real observado, con métricas comparativas por día."),
]

CAP4_COHERENCIA = [
    "La vista de predicción del día siguiente concentra el mensaje central del trabajo y "
    "es coherente con los hallazgos del capítulo 3. El usuario elige un día base y la "
    "aplicación muestra, para cada hora del día siguiente, la predicción del modelo "
    "Lasso, la de la persistencia diaria y, cuando ya existe, el precio real observado, "
    "junto con las métricas de error del día. Así, la persistencia se presenta como "
    "referencia principal, reflejando que es insuperable en error absoluto medio, "
    "mientras que el modelo Lasso aparece como comparación.",

    "La Figura 5 reproduce esta vista para un día de ejemplo. Las tres curvas permiten "
    "apreciar de un vistazo qué tan bien se ajustan los pronósticos al precio que "
    "realmente ocurrió, y las métricas del día cuantifican esa distancia. Al mostrar las "
    "tres curvas juntas, la aplicación no disimula la fortaleza de la línea base, en "
    "coherencia con el criterio de evaluación que guía todo el proyecto.",
]

# Fase 3: versión de los párrafos de coherencia que suma el aprendizaje residual y la
# banda del intervalo conformal a la descripción de la vista de predicción.
CAP4_COHERENCIA_F3 = [
    "La vista de predicción del día siguiente concentra el mensaje central del trabajo y "
    "es coherente con los hallazgos del capítulo 3. El usuario elige un día base y la "
    "aplicación muestra, para cada hora del día siguiente, la predicción del modelo "
    "Lasso, la del aprendizaje residual, la de la persistencia diaria y, cuando ya "
    "existe, el precio real observado, junto con las métricas de error del día. Así, la "
    "persistencia se presenta como referencia principal, reflejando que es insuperable en "
    "error absoluto medio, mientras que el modelo Lasso y el aprendizaje residual aparecen "
    "como comparación.",

    "La Figura 5 reproduce esta vista para un día de ejemplo. Las curvas de los tres "
    "enfoques, junto con el precio real, permiten apreciar de un vistazo qué tan bien se "
    "ajustan los pronósticos al precio que realmente ocurrió, y las métricas del día "
    "cuantifican esa distancia. Al mostrar las tres estrategias juntas, la aplicación no "
    "disimula la fortaleza de la línea base, en coherencia con el criterio de evaluación "
    "que guía todo el proyecto.",

    "Sobre esa misma vista, la aplicación ofrece de forma opcional una banda de "
    "incertidumbre alrededor del pronóstico. La banda corresponde a un intervalo de "
    "predicción conformal al 90 %: con esa probabilidad el precio real de cada hora cae "
    "dentro de ella. El intervalo se construye a partir de los errores del modelo en el "
    "conjunto de prueba, por lo que no asume una distribución particular del error, y se "
    "ensancha en las horas de mayor volatilidad, donde la incertidumbre del pronóstico es "
    "mayor. El usuario puede elegir para cuál de los modelos se muestra la banda. De este "
    "modo, la vista comunica no solo el valor puntual esperado, sino también el rango "
    "plausible del precio, una información útil para la lectura de riesgo de corto plazo "
    "del mercado.",
]

# Párrafo incorporado en la Fase 3: conecta la vista de desempeño con el hallazgo central.
CAP4_COHERENCIA_DESEMPENO = (
    "La vista de desempeño del modelo lleva este criterio un paso más allá. Al presentar "
    "el error por hora del día y, sobre todo, el desempeño por régimen de precio, hace "
    "visible el hallazgo central del capítulo 3: en el régimen calmo la persistencia es "
    "imbatible, pero en el régimen volátil el modelo Lasso y el aprendizaje residual la "
    "superan, que es justamente donde un error cuesta más caro. De este modo, la "
    "aplicación comunica no solo qué pronostica cada enfoque, sino en qué condiciones "
    "conviene cada uno, que es la conclusión más defendible del trabajo.")

CAP4_DESPLIEGUE = [
    "Toda la solución mantiene la trazabilidad exigida por el enfoque metodológico. El "
    "flujo analítico y la aplicación se organizan en módulos de código versionados, con "
    "los artefactos del modelo y los resultados intermedios almacenados en archivos "
    "estandarizados. La aplicación se publica en un servicio de alojamiento gratuito "
    "(Streamlit Community Cloud), conectado directamente al repositorio de código, de "
    "modo que cada actualización del repositorio se refleja en el entorno web sin pasos "
    "manuales adicionales. El entorno web está disponible públicamente en",

    "Con la publicación del entorno web se cierra el ciclo de CRISP-DM: el problema "
    "planteado en la comprensión del negocio recorre la preparación de los datos, su "
    "comprensión, el modelado y la evaluación, y culmina en un despliegue que devuelve "
    "los resultados a los grupos de interés. De este modo, el trabajo entrega un flujo "
    "completo, reproducible y verificable, en lugar de la promesa de un modelo "
    "infalible.",
]


def capitulo4(doc, fase=2):
    titulo1(doc, "Capítulo 4. De los resultados al producto: comunicación y despliegue")
    titulo2(doc, "Enfoque del capítulo")
    for parr in CAP4_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Del análisis al producto: el modelo final serializado")
    for parr in CAP4_MODELO:
        parrafo(doc, parr)
    if fase >= 3:
        parrafo(doc, CAP4_MODELO_RESIDUAL_F3)
    titulo2(doc, "El entorno web: vistas y arquitectura")
    n_vistas = "seis" if fase >= 3 else "cuatro"
    parrafo(doc, CAP4_WEB[0].format(n=n_vistas))
    vistas = list(CAP4_WEB_TABLA)
    if fase >= 3:
        # Fase 3: la predicción suma el aprendizaje residual y el intervalo conformal,
        # y se agregan las vistas de desempeño y de carga de datos al final.
        vistas[3] = CAP4_WEB_VISTA_PREDICCION_F3
        vistas.append(CAP4_WEB_VISTA_DESEMPENO)
        vistas.append(CAP4_WEB_VISTA_CARGA)
    tabla_dos_columnas(doc, vistas, ["Vista", "Qué ofrece"],
                       "Vistas del entorno web de visualización analítica")
    titulo2(doc, "Coherencia con los hallazgos")
    for parr in (CAP4_COHERENCIA_F3 if fase >= 3 else CAP4_COHERENCIA):
        parrafo(doc, parr)
    if fase >= 3:
        nota_fig = ("Pronóstico del modelo Lasso, del aprendizaje residual y de la "
                    "persistencia frente al precio real observado, con la banda del "
                    "intervalo de predicción conformal al 90 % del modelo Lasso y la "
                    "franja pico resaltada.")
    else:
        nota_fig = ("Pronóstico del modelo Lasso y de la persistencia frente al precio "
                    "real observado, con la franja pico resaltada.")
    figura(doc, "cap4_prediccion_dashboard.png",
           "Vista de predicción del día siguiente del entorno web",
           nota=nota_fig)
    if fase >= 3:
        parrafo(doc, CAP4_COHERENCIA_DESEMPENO)
    titulo2(doc, "Reproducibilidad y despliegue")
    for i, parr in enumerate(CAP4_DESPLIEGUE):
        p = parrafo(doc, parr)
        if i == 0:
            sep = p.add_run(" ")
            sep.font.name = FUENTE
            sep.font.size = Pt(12)
            enlace(p, URL_DASHBOARD, URL_DASHBOARD)
            fin = p.add_run(".")
            fin.font.name = FUENTE
            fin.font.size = Pt(12)


# --------------------------------------------------------------------------- #
# Referencias
# --------------------------------------------------------------------------- #

REFERENCIAS = [
    "Aksu, I. Ö., Ghaemi, S. y Anvari-Moghaddam, A. (2026). A two-stage deep learning "
    "approach for accurate day-ahead electricity price forecasting. Engineering "
    "Applications of Artificial Intelligence. https://doi.org/10.1016/j.engappai.2025.112721",

    "Aslam, U. y cols. (2026). Graph-X: Graph-structured deep learning for price "
    "forecasting. Mathematics, 14(16), 2974. https://doi.org/10.3390/math14162974",

    "Balagula, Y., Baimel, D. y Aharon, I. (2026). Comparing time series and neural "
    "network models of long memory for electricity price forecasting. Results in "
    "Engineering. https://doi.org/10.1016/j.rineng.2025.108465",

    "Barrientos, J., Rodas, E., Velilla, E., Lopera, M. y Villada, F. (2012). Modelo para "
    "el pronóstico del precio de la energía eléctrica en Colombia. Lecturas de Economía, "
    "(77), 91-127.",

    "Biró, B., Kiss, C., Molontay, R. y Aszódi, A. (2026). Industry-adaptable explainable "
    "AI based methodology for forecasting electricity prices. Energy Conversion and "
    "Management: X. https://doi.org/10.1016/j.ecmx.2026.101583",

    "Cantillo-Luna, S., Moreno-Chuquen, R., Lopez-Sotelo, J. y Celeita, D. (2023). An "
    "intra-day electricity price forecasting based on a probabilistic transformer neural "
    "network architecture. Energies, 16(19), 6767. https://doi.org/10.3390/en16196767",

    "Cantillo-Luna, S., Moreno-Chuquen, R. y Lopez-Sotelo, J. (2024). Intra-day "
    "electricity price forecasting based on a Time2Vec-LSTM model. En Applied Informatics "
    "(Springer). https://doi.org/10.1007/978-3-031-48415-5_8",

    "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C. y "
    "Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.",

    "Comisión de Regulación de Energía y Gas. (1995). Resolución CREG 024 de 1995. "
    "https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0024_1995.htm",

    "Comisión de Regulación de Energía y Gas. (2009). Resolución CREG 051 de 2009. "
    "https://gestornormativo.creg.gov.co/gestor/entorno/docs/resolucion_creg_0051_2009.htm",

    "Diebold, F. X. y Mariano, R. S. (1995). Comparing predictive accuracy. Journal of "
    "Business & Economic Statistics, 13(3), 253-263. "
    "https://doi.org/10.1080/07350015.1995.10524599",

    "Dudek, G. (2026). Recent advances in probabilistic electricity price forecasting: A "
    "review of methods and evaluation metrics. Energies, 19(15), 3552. "
    "https://doi.org/10.3390/en19153552",

    "Fan, H., Liu, W. y cols. (2026a). EPformer: Unlocking day-ahead electricity price "
    "forecasting accuracy using the time-frequency domain feature learning strategy "
    "considering renewable energy. Renewable Energy. "
    "https://doi.org/10.1016/j.renene.2026.125296",

    "Fan, H., Yang, Y., Liu, W., Lu, Y., Liu, D. y Ma, T. (2026b). EP-Net: End-to-end "
    "probabilistic day-ahead electricity price forecasting with multi-channel fusion. "
    "Sustainable Energy, Grids and Networks. https://doi.org/10.1016/j.segan.2026.102463",

    "Gallón, S. y Barrientos, J. (2021). Forecasting the Colombian electricity spot price "
    "under a functional approach. International Journal of Energy Economics and Policy, "
    "11(2), 67-74. https://doi.org/10.32479/ijeep.10607",

    "Gonzalez-Sierra, M. A. (2024). Feature selection for forecasting of energy spot "
    "price in the Colombian market. En 2024 IEEE ANDESCON. "
    "https://doi.org/10.1109/ANDESCON61840.2024.10755699",

    "Herrera-Mejía, L., Hoyos, C. D., La Rotta, D., Mejía, Y. y Castillo, J. (2026). "
    "Forecasting day-ahead hydropower bids in the Colombian electricity market: A "
    "two-stage machine learning framework. Smart Energy. "
    "https://doi.org/10.1016/j.segy.2025.100221",

    "Kitchenham, B. y Charters, S. (2007). Guidelines for performing Systematic Literature "
    "Reviews in Software Engineering. Keele University and Durham University.",

    "Lago, J., Marcjasz, G., De Schutter, B. y Weron, R. (2021). Forecasting day-ahead "
    "electricity prices: A review of state-of-the-art algorithms, best practices and an "
    "open-access benchmark. Applied Energy, 293, 116983. "
    "https://doi.org/10.1016/j.apenergy.2021.116983",

    "Li, T., Wang, Y. y Zhang, Y. (2026). Hybrid deep learning models for hourly "
    "electricity price forecasting. IET Conference Proceedings. "
    "https://doi.org/10.1049/icp.2026.1597",

    "Maciejowska, K., Lipiecki, A. y Uniejewski, B. (2026). Statistical and economic "
    "evaluation of forecasts in electricity markets: Beyond RMSE and MAE. arXiv preprint "
    "arXiv:2511.13616. https://arxiv.org/abs/2511.13616",

    "Marín, J. B., Orozco, E. T. y Velilla, E. (2018). Forecasting electricity price in "
    "Colombia: A comparison between neural network, ARMA process and hybrid models. "
    "International Journal of Energy Economics and Policy, 8(3), 97-106.",

    "Martínez-Plumed, F., Contreras-Ochando, L., Ferri, C., Hernández-Orallo, J., Kull, "
    "M., Lachiche, N., Ramírez-Quintana, M. J. y Flach, P. (2021). CRISP-DM twenty years "
    "later: From data mining processes to data science trajectories. IEEE Transactions on "
    "Knowledge and Data Engineering, 33(8), 3048-3061. "
    "https://doi.org/10.1109/TKDE.2019.2962680",

    "Muñoz-Santiago, A., Urquijo-Vanstrahlengs, J., Castro-Otero, A. y Lombana, J. (2017). "
    "Pronóstico del precio de la energía en Colombia utilizando modelos ARIMA con IGARCH. "
    "Revista de Economía del Rosario, 20(1), 127-161. "
    "https://doi.org/10.12804/revistas.urosario.edu.co/economia/a.6152",

    "Ni, T., Hou, H., Wu, X., Yang, J. y cols. (2026). AI-driven probabilistic day-ahead "
    "electricity price forecasting with tail risk mitigation via dynamic spike "
    "calibration. IEEE Transactions on Consumer Electronics, 72(3), 6417. "
    "https://doi.org/10.1109/TCE.2026.3679105",

    "Piotrowski, P. (2026). Short-term electricity price forecasting: A review of point "
    "forecasting methods, metrics, and empirical evaluation. Energies, 19(17), 4000. "
    "https://doi.org/10.3390/en19174000",

    "Poveda, G. y Mesa, O. J. (1996). Las fases extremas del fenómeno ENSO (El Niño y La "
    "Niña) y su influencia sobre la hidrología de Colombia. Ingeniería Hidráulica en "
    "México, 11(1), 21-37.",

    "Radziukynienė, N. (2026). A multi-parallel hybrid neural network model for "
    "short-term electricity price forecasting under high market volatility. Applied "
    "Sciences, 16(8), 3865. https://doi.org/10.3390/app16083865",

    "Salihoglu, S., Ahmed, I. y Asadi, A. (2026). Adaptive online learning with LSTM "
    "networks for energy price prediction. International Journal of Data Science and "
    "Analytics, 22, 271. https://doi.org/10.1007/s41060-026-01251-1",

    "Trespalacios, A. y Franco Ceballos, L. E. (2025). Aproximación tipo aprendizaje "
    "profundo al precio spot de la bolsa de energía. Revista Colombiana de Computación, "
    "26(2), 1-15. https://doi.org/10.29375/25392115.4587",

    "Wei, H. (2026). Research on day-ahead electricity price prediction algorithm based "
    "on ensemble learning. En Proceedings of the ACM International Conference. "
    "https://doi.org/10.1145/3808707.3808828",

    "Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art "
    "with a look into the future. International Journal of Forecasting, 30(4), 1030-1081. "
    "https://doi.org/10.1016/j.ijforecast.2014.08.008",

    "XM S.A. E.S.P. (2026). Precio de bolsa nacional [conjunto de datos]. SINERGOX. "
    "https://www.xm.com.co",
]


def referencias(doc):
    titulo1(doc, "Referencias")
    for ref in REFERENCIAS:
        referencia_parrafo(doc, ref)


# --------------------------------------------------------------------------- #
# Conclusiones, recomendaciones y limitaciones
# --------------------------------------------------------------------------- #

CONCLUSIONES_ENFOQUE = [
    "Este capítulo cierra el trabajo respondiendo a cada uno de los objetivos "
    "específicos y a la pregunta de investigación, reconoce las limitaciones del "
    "alcance entregado y plantea las líneas de trabajo futuro que de él se derivan.",
]

CONCLUSIONES_POR_OBJETIVO = [
    "Respecto al primer objetivo, se construyó una serie horaria depurada del precio "
    "de bolsa de 142.344 registros (2010-2026) y, a partir de ella, una base "
    "supervisada de 5.930 días y 333 columnas que organiza la información hasta el día "
    "base y las variables objetivo del día siguiente, con variables de calendario, "
    "rezagos, estadísticas móviles y exógenas climáticas e hidrológicas. Esta base, "
    "trazable y reproducible, es el insumo común de todo el trabajo posterior.",

    "Respecto al segundo objetivo, la caracterización mostró un precio con "
    "estacionalidad horaria marcada (valle de madrugada, pico entre las 18 y las 21 "
    "horas), distribución asimétrica con eventos extremos concentrados en días "
    "hábiles, estacionariedad en nivel y una autocorrelación diaria muy alta (entre "
    "0,92 y 0,96). El aporte de las exógenas resultó moderado y de naturaleza "
    "anticipada: el Índice ONI antecede en varios meses al nivel de embalses, más que "
    "explicar el precio de forma contemporánea.",

    "Respecto al tercer objetivo, la evaluación bajo un protocolo homogéneo (partición "
    "temporal, ocho estrategias, prueba de Diebold-Mariano) estableció que ninguna "
    "estrategia supera a la persistencia diaria en error absoluto medio, pero varias la "
    "superan de forma significativa en error cuadrático y, sobre todo, en el régimen "
    "volátil del precio. La recalibración periódica del modelo Lasso prácticamente "
    "iguala a la persistencia en MAE y la mejora en RMSE, lo que constituye la mejora más "
    "defendible identificada en el trabajo.",

    "Respecto al cuarto objetivo, los resultados y el modelo seleccionado se "
    "comunicaron mediante una aplicación web pública que presenta el histórico, la "
    "predicción del día siguiente frente al precio real observado, la exploración de "
    "patrones y la relación con el clima, publicada en un servicio de alojamiento "
    "gratuito conectado al repositorio de código.",
]

# Fase 3: versión del cuarto objetivo que suma la banda de incertidumbre conformal.
CONCLUSIONES_CUARTO_OBJETIVO_F3 = (
    "Respecto al cuarto objetivo, los resultados y el modelo seleccionado se comunicaron "
    "mediante una aplicación web pública que presenta el histórico, la predicción del día "
    "siguiente frente al precio real observado con su banda de incertidumbre conformal, "
    "la exploración de patrones y la relación con el clima, publicada en un servicio de "
    "alojamiento gratuito conectado al repositorio de código.")

CONCLUSIONES_PREGUNTA = [
    "En conjunto, estos resultados responden a la pregunta que orientó la "
    "investigación: en qué medida un modelo de predicción del precio horario de bolsa "
    "para el día siguiente mejora de forma verificable el desempeño de una línea base "
    "competitiva bajo validación temporal y métricas de error. La respuesta admite "
    "matices. El modelo de aprendizaje automático no logra superar a la persistencia "
    "diaria en el error absoluto medio, pero sí lo hace de forma estadísticamente "
    "significativa en el error cuadrático y en el régimen de precios altos, que es "
    "justamente donde un pronóstico impreciso tiene mayor costo. Más que proponer un "
    "modelo definitivo, el trabajo aporta una respuesta verificable, respaldada por "
    "evidencia estadística, sobre hasta dónde puede mejorarse el pronóstico frente a "
    "una referencia exigente.",
]

CONCLUSIONES_LIMITACIONES = [
    "El alcance entregado es metodológico y retrospectivo: el trabajo construye, "
    "evalúa y comunica el enfoque de predicción sobre un conjunto de datos histórico, "
    "con prueba entre diciembre de 2022 y marzo de 2026. No incluye la puesta en "
    "operación de un sistema que reciba datos en tiempo real y produzca el pronóstico "
    "de un día siguiente real; esa alimentación continua de datos queda fuera del "
    "alcance entregado y se retoma como trabajo futuro.",

    "La evaluación se apoya en una única partición temporal entrenamiento-prueba; sus "
    "implicaciones para la validez de los resultados se analizan en el capítulo 5, y su "
    "efecto se mitigaría con una validación de origen móvil.",

    "La cobertura de las variables exógenas es desigual en el tiempo, pues el nivel de "
    "embalses solo está disponible desde 2013 frente a la serie de precio que parte de "
    "2010, con las consecuencias para el uso de los rezagos largos que se detallan en el "
    "capítulo 5.",

    "El aporte predictivo de las exógenas climáticas e hidrológicas, aunque "
    "significativo, es moderado frente al peso de la estructura temporal propia del "
    "precio; buena parte de la capacidad de pronóstico descansa en la memoria de corto "
    "plazo de la serie, no en las condiciones externas.",

    "Finalmente, el hallazgo de que ningún enfoque supera a la persistencia diaria en "
    "error absoluto medio es, en sí mismo, una limitación del margen de mejora "
    "disponible con las estrategias exploradas, y no debe leerse como una debilidad "
    "del diseño de la evaluación, que es precisamente lo que permite documentarlo con "
    "rigor.",
]

CONCLUSIONES_RECOMENDACIONES = [
    "La línea de trabajo futuro más directa es la puesta en operación del pronóstico "
    "con datos en tiempo real: una ingesta periódica desde XM y la NOAA, un "
    "reprocesamiento automático de variables y la generación diaria del pronóstico del "
    "día siguiente real, lo que convertiría el prototipo actual en un sistema "
    "operativo.",

    "Una ruta de implementación viable, sin infraestructura adicional, aprovecharía la "
    "que ya usa el proyecto. Se propone una tarea programada (por ejemplo, con las "
    "acciones automatizadas del repositorio de código) que, de forma diaria, ejecute "
    "tres pasos: primero, la ingesta de los datos más recientes de precio y embalses "
    "desde XM y del Índice ONI desde la NOAA; segundo, el reprocesamiento de la base y "
    "la ingeniería de variables, con una recalibración periódica del modelo que "
    "mantenga su desempeño en el tiempo; y tercero, el registro del pronóstico generado "
    "en el repositorio, desde donde el entorno web lo consume y lo actualiza de forma "
    "automática. La aplicación ya admite datos nuevos mediante la carga de archivos, de "
    "modo que el único componente pendiente para la operación continua es la "
    "automatización de esta ingesta.",

    "En el plano metodológico, se recomienda complementar la partición temporal única "
    "con una validación de origen móvil, que reentrene y evalúe el modelo en varios "
    "cortes sucesivos y permita estimar la variabilidad del desempeño a través del "
    "tiempo. También conviene extender o sustituir la fuente del nivel de embalses "
    "para cubrir el periodo 2010-2012 y explorar variables de demanda del sistema como "
    "exógenas adicionales.",

    "Dado que la ventaja de los modelos se concentra en el régimen volátil, una línea "
    "prometedora es diseñar una estrategia combinada que use la persistencia en "
    "condiciones calmas y el modelo recalibrado en condiciones volátiles, detectadas "
    "con una regla simple de umbral sobre el precio reciente. Finalmente, se recomienda "
    "incorporar criterios de evaluación económica, no solo estadística, en línea con "
    "lo propuesto por Maciejowska et al. (2026), para valorar el pronóstico desde la "
    "perspectiva de las decisiones que apoya.",
]


def conclusiones(doc, fase=2):
    titulo1(doc, "Conclusiones, recomendaciones y limitaciones")
    for parr in CONCLUSIONES_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Conclusiones por objetivo")
    objetivos = list(CONCLUSIONES_POR_OBJETIVO)
    if fase >= 3:
        # Fase 3: el cuarto objetivo refleja la banda de incertidumbre conformal.
        objetivos[-1] = CONCLUSIONES_CUARTO_OBJETIVO_F3
    for parr in objetivos:
        parrafo(doc, parr)
    titulo2(doc, "Respuesta a la pregunta de investigación")
    for parr in CONCLUSIONES_PREGUNTA:
        parrafo(doc, parr)
    titulo2(doc, "Limitaciones")
    for parr in CONCLUSIONES_LIMITACIONES:
        parrafo(doc, parr)
    titulo2(doc, "Recomendaciones y trabajos futuros")
    for parr in CONCLUSIONES_RECOMENDACIONES:
        parrafo(doc, parr)


# --------------------------------------------------------------------------- #
# Capítulo de Discusión (Fase 3)
# --------------------------------------------------------------------------- #

DISCUSION_ENFOQUE = [
    "Este capítulo discute los hallazgos del trabajo a la luz de la literatura reciente. "
    "Primero contrasta los resultados con estudios previos para señalar coincidencias y "
    "diferencias; luego los sitúa frente a las perspectivas teóricas y metodológicas de "
    "los últimos años; después precisa la contribución del trabajo a la problemática y a "
    "la disciplina; y, por último, somete el estudio a una autocrítica de sus "
    "debilidades.",
]

DISCUSION_CONTRASTE = [
    "El resultado central del trabajo, que ninguna estrategia supera a la persistencia "
    "diaria en el error absoluto medio y que las mejoras verificables se concentran en "
    "el error cuadrático y en el régimen volátil, coincide de forma directa con la "
    "advertencia metodológica de la literatura de pronóstico de precios eléctricos. "
    "Weron (2014) y Lago et al. (2021) documentan que una parte considerable de las "
    "mejoras reportadas no resiste el contraste frente a una línea base competitiva, y "
    "Maciejowska et al. (2026) insiste en evaluar los modelos con criterios estadísticos y "
    "económicos exigentes. Nuestros resultados ofrecen evidencia empírica, para el "
    "mercado colombiano, que respalda esa postura: la persistencia diaria, sostenida por "
    "una autocorrelación diaria muy alta, es un rival difícil de batir.",

    "El uso de varias métricas (MAE, RMSE, sMAPE y R2) y el análisis desagregado por "
    "hora y por régimen de precio es coherente con las recomendaciones de las revisiones "
    "recientes sobre evaluación del pronóstico puntual (Piotrowski, 2026; Dudek, 2026), "
    "que advierten que una sola métrica puede ocultar diferencias relevantes de "
    "desempeño. En nuestro caso, esa desagregación fue decisiva: medido solo por el MAE "
    "global, el estudio habría concluido que ningún modelo aporta, mientras que el "
    "análisis por métrica y por régimen revela dónde sí existe una mejora verificable.",

    "Frente a los estudios que reportan mejoras de modelos complejos de aprendizaje "
    "automático y profundo (Fan et al., 2026a, 2026b; Aslam et al., 2026; Wei, 2026; Salihoglu et al., 2026), "
    "nuestros resultados introducen un matiz antes que una contradicción. La ventaja de "
    "los modelos existe, pero es específica: aparece en el control de los errores "
    "grandes y en los periodos de precios altos, no en el error promedio. En particular, "
    "la mejora que aporta la recalibración periódica del modelo Lasso es consistente con "
    "el valor que la literatura atribuye al aprendizaje en línea y a la actualización "
    "frecuente de los parámetros (Salihoglu et al., 2026).",

    "El pronóstico del precio de bolsa en Colombia tiene además una trayectoria previa. "
    "Se ha abordado tanto con modelos que identifican los determinantes del precio "
    "mediante corrección de errores vectorial y redes neuronales (Barrientos et al., "
    "2012), como con modelos econométricos de series de tiempo que capturan su "
    "volatilidad, como los esquemas ARIMA con IGARCH, adecuados para una serie con "
    "comportamiento abrupto y sin estacionariedad (Muñoz-Santiago et al., 2017). Más "
    "recientemente, los estudios locales se han orientado sobre todo a "
    "proponer arquitecturas: aprendizaje profundo sobre el precio spot (Trespalacios, "
    "2025), selección de variables predictoras (Gonzalez-Sierra, 2024), modelos de tipo "
    "transformador probabilístico y Time2Vec-LSTM (Cantillo-Luna et al., 2023, 2024) y "
    "pronóstico de ofertas para el día siguiente (Herrera-Mejía et al., 2026). Frente a esa "
    "línea, este trabajo aporta un énfasis distinto y complementario: no la búsqueda de "
    "una arquitectura más sofisticada, sino la comparación reproducible de varias "
    "estrategias bajo un mismo protocolo y contra una línea base fuerte, un contraste "
    "menos frecuente en la producción local.",

    "Dos excepciones cercanas a ese énfasis son Gallón y Barrientos (2021), que evalúan "
    "un enfoque funcional del precio spot colombiano frente a líneas base estándar, y "
    "Marín et al. (2018), quienes compararon redes neuronales con modelos ARMA mediante "
    "la prueba de Diebold-Mariano y no hallaron una ventaja estadísticamente "
    "significativa de las primeras. Este último resultado es especialmente relevante: "
    "coincide de forma directa con el hallazgo del presente trabajo, lo que sugiere que "
    "la dificultad para superar de manera significativa a referencias simples no es un "
    "rasgo aislado de nuestra evaluación, sino un patrón recurrente en el mercado "
    "colombiano.",
]

DISCUSION_PERSPECTIVAS = [
    "Situado frente a las perspectivas de los últimos cinco años, el trabajo se inscribe "
    "en la corriente que, sin desconocer el avance de las arquitecturas complejas "
    "(transformadores, redes sobre grafos, métodos probabilísticos e interpretabilidad "
    "mediante inteligencia artificial explicable), reclama evaluaciones reproducibles y "
    "líneas base exigentes como condición para que una mejora sea creíble (Lago et al., "
    "2021; Maciejowska et al., 2026). En ese marco, el valor del estudio no reside en el modelo "
    "empleado, sino en el rigor y la transparencia con que se compara.",

    "El hallazgo de que la capacidad de pronóstico descansa sobre todo en la memoria de "
    "corto plazo de la serie, con un aporte moderado de las variables exógenas "
    "climáticas e hidrológicas, dialoga con la discusión abierta sobre el peso relativo "
    "de las variables fundamentales frente a la estructura autorregresiva del precio. En "
    "un mercado de generación mayoritariamente hidráulica cabría esperar un papel más "
    "fuerte de los embalses y del clima; el estudio muestra que su efecto, aunque "
    "estadísticamente presente, es anticipado y gradual (el Índice ONI antecede en "
    "varios meses al nivel de embalses) más que un predictor contemporáneo dominante. "
    "Esta relación anticipatoria es coherente con la evidencia clásica sobre la fuerte "
    "influencia del fenómeno ENSO en la hidrología colombiana (Poveda y Mesa, 1996).",

    "Por último, que la ventaja de los modelos se concentre en el régimen volátil "
    "conecta con la idea, cada vez más presente en la literatura, de que el valor "
    "práctico de un pronóstico se juega en los momentos extremos, que son los que mayor "
    "costo imponen a las decisiones de los agentes. Esta lectura refuerza la pertinencia "
    "de evaluar el desempeño por régimen y no solo de forma agregada.",
]

DISCUSION_CONTRIBUCION = [
    "La contribución del trabajo a la solución de la problemática es, ante todo, "
    "metodológica: aporta un protocolo de evaluación reproducible para el pronóstico "
    "day-ahead del precio de bolsa en Colombia, que integra una partición temporal, una "
    "línea base competitiva, métricas múltiples, análisis por régimen y una prueba de "
    "significancia estadística. Este protocolo permite distinguir con claridad una "
    "mejora real de una aparente, y queda disponible para evaluaciones futuras de nuevos "
    "modelos sobre el mismo mercado.",

    "En el plano práctico, el trabajo identifica una estrategia defendible, el modelo "
    "Lasso con recalibración periódica, útil para el control de los errores grandes y "
    "del régimen volátil, y la pone a disposición de los grupos de interés mediante un "
    "entorno web que comunica el histórico, la predicción del día siguiente y su "
    "contraste con el precio real. De este modo, los resultados no quedan en el plano "
    "analítico, sino que se traducen en una herramienta accesible para la lectura de "
    "corto plazo del mercado.",

    "Dentro de ese plano práctico, un aporte de valor adicional es la incorporación de "
    "intervalos de predicción conformales a la comunicación del pronóstico. Más allá del "
    "valor puntual esperado para cada hora, la herramienta muestra una banda de "
    "incertidumbre al 90 %, construida a partir de los errores del modelo en el conjunto "
    "de prueba y, por lo tanto, sin supuestos sobre la distribución del error, que se "
    "ensancha en las horas de mayor volatilidad. Esta cuantificación de la incertidumbre "
    "conecta con la corriente de pronóstico probabilístico que la literatura reciente "
    "destaca (Lago et al., 2021; Maciejowska et al., 2026) y resulta especialmente "
    "pertinente en un mercado de episodios extremos, pues ofrece a los agentes no solo "
    "una estimación central, sino el rango plausible del precio, una base más informada "
    "para la lectura de riesgo de corto plazo.",

    "Para la disciplina, el aporte es una pieza de evidencia empírica, referida a un "
    "mercado hidro-dependiente poco representado en la literatura internacional, que "
    "respalda la tesis de que las líneas base simples son competitivas y que las mejoras "
    "deben demostrarse con rigor. Al documentar en qué métrica y bajo qué condiciones "
    "cada enfoque es preferible, el trabajo ofrece un punto de referencia reproducible "
    "para la investigación posterior sobre el precio de bolsa colombiano.",
]

DISCUSION_AUTOCRITICA = [
    "El trabajo tiene debilidades que conviene reconocer. La principal es de validez "
    "externa de la evaluación: se apoya en una única partición temporal "
    "entrenamiento-prueba que, aunque respeta el orden cronológico, no cuantifica la "
    "variabilidad del desempeño ante distintos puntos de corte. Una validación de origen "
    "móvil, con varios cortes sucesivos, habría dado una medida más robusta de la "
    "estabilidad de los resultados y es una deuda metodológica del estudio.",

    "Una segunda debilidad es el alcance de las variables y de los modelos. La cobertura "
    "del nivel de embalses comienza en 2013, mientras que la serie de precio parte de "
    "2010, lo que limita el uso de esa variable en los primeros años y en los rezagos "
    "largos. Además, el conjunto de estrategias, aunque amplio, no agota el espacio de "
    "métodos disponibles: no se exploraron ensambles más sofisticados ni esquemas "
    "híbridos que combinen de forma adaptativa la persistencia con los modelos según el "
    "régimen, pese a que los resultados sugieren que esa combinación sería prometedora.",

    "Una tercera debilidad es que la evaluación se limitó a criterios estadísticos. No "
    "se incorporó una valoración económica del pronóstico, es decir, su efecto sobre las "
    "decisiones y los costos de los agentes, que la literatura reciente señala como "
    "complemento necesario de las métricas de error (Maciejowska et al., 2026). Reconocer estas "
    "limitaciones no resta validez a los hallazgos, pero delimita con honestidad su "
    "alcance y orienta las líneas de mejora del trabajo.",
]


def discusion(doc):
    titulo1(doc, "Capítulo 5. Discusión: los hallazgos frente a la literatura y su "
                 "aporte")
    for parr in DISCUSION_ENFOQUE:
        parrafo(doc, parr)
    titulo2(doc, "Los hallazgos frente a estudios previos")
    for parr in DISCUSION_CONTRASTE:
        parrafo(doc, parr)
    titulo2(doc, "Lectura desde las perspectivas recientes")
    for parr in DISCUSION_PERSPECTIVAS:
        parrafo(doc, parr)
    titulo2(doc, "Contribución a la problemática y a la disciplina")
    for parr in DISCUSION_CONTRIBUCION:
        parrafo(doc, parr)
    titulo2(doc, "Debilidades del trabajo: una autocrítica")
    for parr in DISCUSION_AUTOCRITICA:
        parrafo(doc, parr)


def main(fase=2):
    _CONTADOR_TABLA[0] = 0
    _CONTADOR_FIGURA[0] = 0
    doc = nuevo_documento()
    portada(doc)
    salto_pagina(doc)
    resumen(doc)
    salto_pagina(doc)
    introduccion(doc)
    planteamiento(doc)
    objetivos(doc)
    marco(doc)
    metodologia(doc)
    salto_pagina(doc)
    capitulo1(doc)
    salto_pagina(doc)
    capitulo2(doc)
    salto_pagina(doc)
    capitulo3(doc)
    salto_pagina(doc)
    capitulo4(doc, fase)
    if fase >= 3:
        salto_pagina(doc)
        discusion(doc)
    salto_pagina(doc)
    conclusiones(doc, fase)
    salto_pagina(doc)
    referencias(doc)

    doc.core_properties.author = "Jonathan Carvajal García"
    doc.core_properties.title = TITULO
    doc.core_properties.keywords = PALABRAS_CLAVE.rstrip(".")
    salida = SALIDA_FASE3 if fase >= 3 else SALIDA
    doc.save(salida)
    print("Guardado:", salida)


if __name__ == "__main__":
    import sys
    fase = 2
    if len(sys.argv) > 1 and sys.argv[1] in ("2", "3"):
        fase = int(sys.argv[1])
    main(fase)
