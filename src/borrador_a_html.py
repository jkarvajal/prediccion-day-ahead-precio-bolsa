# -*- coding: utf-8 -*-
"""
Genera una versión HTML del borrador del trabajo de grado, para consultarla en el
navegador sin abrir el Word y sin generar conflictos de archivo.

Se ejecuta después de cada actualización del borrador:
    python -m src.borrador_a_html
"""
import html
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

RAIZ = Path(__file__).resolve().parents[1]
DOCX = RAIZ / "notebooks" / "Documentos" / "Borrador_Trabajo_de_Grado_Jonathan_Carvajal.docx"
HTML = RAIZ / "notebooks" / "Documentos" / "Borrador_Trabajo_de_Grado_Jonathan_Carvajal.html"

# Gestor de referencias (SQLite). Se leen las referencias de la colección MCDA
# para armar la lista final y para detectar citas que aún no están registradas.
GESTOR_DB = Path(r"D:\Claude\Gestor de referencias\data\library.db")
COLECCION = "MCDA"

CSS = """
:root { color-scheme: light; }
body { font-family: 'Times New Roman', Times, serif; max-width: 760px;
       margin: 0 auto; padding: 2rem 2.2rem 4rem; line-height: 2.0;
       font-size: 18px; color: #111; background: #fff; }
.barra { position: sticky; top: 0; background: #1f4e79; color: #fff;
         margin: -2rem -2.2rem 1.8rem; padding: .6rem 1.5rem; font-family: system-ui, sans-serif;
         font-size: .8rem; line-height: 1.4; display: flex; justify-content: space-between; }
h1 { font-size: 1.35rem; font-weight: bold; color: #111; text-align: center;
     margin: 2.2rem 0 .8rem; }
h2 { font-size: 1.15rem; font-weight: bold; color: #111; margin: 1.6rem 0 .4rem; }
h3 { font-size: 1.02rem; font-weight: bold; font-style: italic; color: #111; margin: 1.1rem 0 .3rem; }
p { margin: 0; text-align: justify; text-indent: 1.27cm; }
.center { text-align: center; text-indent: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem;
        line-height: 1.35; }
th, td { border: 1px solid #999; padding: .35rem .55rem; text-align: left;
         vertical-align: top; text-indent: 0; }
th { background: #1f4e79; color: #fff; }
tr:nth-child(even) td { background: #f2f4f8; }
hr.sep { border: none; border-top: 1px solid #c9c9c9; margin: 2.6rem 0 0; }
.cita-faltante { color: #c0392b; font-weight: 600; }
.cita { color: inherit; text-decoration: none; border-bottom: 1px dotted #1f4e79; }
.cita:hover { color: #1f4e79; }
.referencias .ref { text-indent: -1.4cm; margin-left: 1.4cm; margin-bottom: .55rem;
                    scroll-margin-top: 90px; }
.referencias .ref:target { background: #fff6d6; }
.nota-ref { text-indent: 0; font-style: italic; color: #777; font-size: .82rem;
            margin-bottom: .8rem; }
.toc { position: fixed; top: 84px; left: calc(50% - 605px); width: 195px;
       font-family: system-ui, sans-serif; font-size: .8rem; line-height: 1.45;
       max-height: 78vh; overflow: auto; border-left: 2px solid #1f4e79;
       padding-left: .8rem; }
.toc .titulo { font-weight: 700; color: #1f4e79; margin-bottom: .3rem; }
.toc a { display: block; color: #444; text-decoration: none; padding: .1rem 0; }
.toc a:hover { color: #1f4e79; }
.toc a.n2 { padding-left: .9rem; color: #777; font-size: .75rem; }
@media (max-width: 1180px) {
  .toc { position: static; width: auto; left: auto; max-height: none; margin: 0 0 1.6rem;
         border: 1px solid #ddd; border-left: 3px solid #1f4e79; padding: .6rem .9rem;
         background: #f7f7f5; }
}
"""


def _sin_tags(texto):
    """Quita las etiquetas HTML dejando el texto ya escapado."""
    return re.sub(r"<[^>]+>", "", texto)


def _runs_html(p):
    partes = []
    for r in p.runs:
        t = html.escape(r.text)
        if not t:
            continue
        if r.bold:
            t = f"<strong>{t}</strong>"
        if r.italic:
            t = f"<em>{t}</em>"
        partes.append(t)
    return "".join(partes) or html.escape(p.text)


def _parrafo_html(p):
    estilo = p.style.name
    contenido = _runs_html(p)
    if not contenido.strip():
        return ""
    if estilo.startswith("Heading 1"):
        return f"<h1>{contenido}</h1>"
    if estilo.startswith("Heading 2"):
        return f"<h2>{contenido}</h2>"
    if estilo.startswith("Heading 3"):
        return f"<h3>{contenido}</h3>"
    clase = ' class="center"' if p.alignment == WD_ALIGN_PARAGRAPH.CENTER else ""
    return f"<p{clase}>{contenido}</p>"


def _tabla_html(t):
    filas = []
    for i, row in enumerate(t.rows):
        celdas = []
        etiqueta = "th" if i == 0 else "td"
        for c in row.cells:
            texto = "<br>".join(html.escape(x) for x in c.text.split("\n"))
            celdas.append(f"<{etiqueta}>{texto}</{etiqueta}>")
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    return "<table>" + "".join(filas) + "</table>"


# --------------------------------------------------------------------------- #
# Integración con el gestor de referencias
# --------------------------------------------------------------------------- #

def cargar_referencias():
    """Lee los ítems de la colección MCDA con sus autores. Devuelve lista de dicts."""
    if not GESTOR_DB.exists():
        print("Aviso: no se encontró el gestor de referencias; se omiten las referencias.")
        return []
    try:
        con = sqlite3.connect(f"file:{GESTOR_DB}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        con = sqlite3.connect(str(GESTOR_DB))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    q = """SELECT i.* FROM items i
           JOIN collection_items ci ON ci.item_id = i.id
           JOIN collections c ON c.id = ci.collection_id
           WHERE c.name = ? AND (i.trashed IS NULL OR i.trashed = 0)"""
    items = []
    for fila in cur.execute(q, (COLECCION,)).fetchall():
        it = dict(fila)
        autores = cur.execute(
            "SELECT last_name, first_name, role, position FROM creators "
            "WHERE item_id = ? ORDER BY position", (it["id"],)).fetchall()
        it["autores"] = [dict(a) for a in autores]
        items.append(it)
    con.close()
    return items


def _iniciales(nombre):
    if not nombre:
        return ""
    partes = re.split(r"[\s.]+", nombre.strip())
    return " ".join(f"{p[0].upper()}." for p in partes if p)


def _apa_autores(autores):
    """Formatea la lista de autores en estilo APA 7."""
    principales = [a for a in autores if (a.get("role") or "author") in ("author", "editor")]
    if not principales:
        return ""
    nombres = []
    editores = all((a.get("role") == "editor") for a in principales)
    for a in principales:
        ini = _iniciales(a.get("first_name"))
        nombres.append(f"{a['last_name']}, {ini}".strip().rstrip(",") if ini else a["last_name"])
    if len(nombres) == 1:
        cadena = nombres[0]
    else:
        cadena = ", ".join(nombres[:-1]) + ", & " + nombres[-1]
    if editores:
        cadena += " (Ed.)" if len(nombres) == 1 else " (Eds.)"
    return cadena


def _primer_apellido(item):
    for a in sorted(item.get("autores", []), key=lambda x: x.get("position") or 0):
        if (a.get("role") or "author") in ("author", "editor"):
            return (a.get("last_name") or "").lower()
    return (item.get("title") or "").lower()


def _apa_entrada(item):
    """Devuelve una referencia formateada en HTML (APA 7) según el tipo."""
    tipo = (item.get("type") or "").lower()
    autores = _apa_autores(item.get("autores", []))
    anio = item.get("year") or "s. f."
    titulo = html.escape(item.get("title") or "")
    doi = item.get("doi")
    url = item.get("url")
    pub = html.escape(item.get("publication") or "")
    editorial = html.escape(item.get("publisher") or "")
    vol = html.escape(str(item.get("volume") or ""))
    num = html.escape(str(item.get("issue") or ""))
    pags = html.escape(str(item.get("pages") or ""))
    edicion = html.escape(str(item.get("edition") or ""))
    enlace = ""
    if doi:
        d = doi if str(doi).startswith("http") else f"https://doi.org/{doi}"
        enlace = f' <a href="{html.escape(d)}">{html.escape(d)}</a>'
    elif url:
        enlace = f' <a href="{html.escape(url)}">{html.escape(url)}</a>'

    inicio = f"{autores} ({anio}). " if autores else f"({anio}). "

    if tipo in ("journalarticle", "article", "journal-article", "journal_article"):
        vni = vol + (f"({num})" if num else "")
        cola = f"<em>{pub}</em>" + (f", <em>{vni}</em>" if vni else "")
        if pags:
            cola += f", {pags}"
        return f"{inicio}{titulo}. {cola}.{enlace}"
    if tipo in ("book",):
        if edicion:
            ed = f" ({edicion})" if re.search(r"ed\.?$", edicion, re.I) else f" ({edicion} ed.)"
        else:
            ed = ""
        cola = f" {editorial}." if editorial else ""
        return f"{inicio}<em>{titulo}</em>{ed}.{cola}{enlace}"
    if tipo in ("booksection", "chapter", "bookchapter"):
        cola = f" En <em>{pub}</em>" + (f" (pp. {pags})" if pags else "") + "."
        cola += f" {editorial}." if editorial else ""
        return f"{inicio}{titulo}.{cola}{enlace}"
    if tipo in ("conferencepaper", "paper-conference", "conference", "proceedings"):
        cola = f" <em>{pub}</em>." if pub else ""
        return f"{inicio}{titulo}.{cola}{enlace}"
    if tipo in ("report", "techreport"):
        cola = f" {editorial}." if editorial else ""
        return f"{inicio}<em>{titulo}</em>.{cola}{enlace}"
    if tipo in ("thesis",):
        cola = f" [Tesis, {editorial}]." if editorial else "."
        return f"{inicio}<em>{titulo}</em>{cola}{enlace}"
    if tipo in ("statute", "legislation", "legal", "bill", "regulation"):
        fuente = f" {pub}." if pub else ""
        return f"{inicio}<em>{titulo}</em>.{fuente}{enlace}"
    if tipo in ("webpage", "website", "web"):
        sitio = f" <em>{pub}</em>." if pub else ""
        return f"{inicio}{titulo}.{sitio}{enlace}"
    # Genérico
    cola = f" <em>{pub}</em>." if pub else (f" {editorial}." if editorial else "")
    return f"{inicio}<em>{titulo}</em>.{cola}{enlace}"


def _indices(refs):
    """Construye índices de búsqueda: (apellido, año) y (tipo_legal, número, año)."""
    ay = {}
    legal = {}
    for it in refs:
        anio = it.get("year")
        for a in it.get("autores", []):
            if (a.get("role") or "author") in ("author", "editor") and a.get("last_name"):
                ay[(a["last_name"].lower(), anio)] = it
                ay[(a["last_name"].split()[0].lower(), anio)] = it
        titulo = (it.get("title") or "").lower()
        for m in re.finditer(r"(creg|ley)\s+0*(\d+)\s+de\s+(\d{4})", titulo):
            legal[(m.group(1), m.group(2), m.group(3))] = it
    return ay, legal


RE_PAREN = re.compile(r"\(([^()]*?\b\d{4}\b[^()]*?)\)")
RE_SUB = re.compile(r"^(?P<auth>.+?),\s*(?P<year>\d{4})[a-z]?$")
RE_LEGAL = re.compile(r"(Resoluci[oó]n\s+CREG|Ley)\s+0*(\d+)\s+de\s+(\d{4})")


def _rojo(texto):
    return f'<span class="cita-faltante" title="Falta en el gestor">{texto}</span>'


def marcar_citas(cuerpo, refs):
    """Marca en rojo las citas ausentes del gestor y devuelve las citadas presentes."""
    ay, legal = _indices(refs)
    citadas = {}

    def sub_legal(m):
        clase = "creg" if "creg" in m.group(1).lower() else "ley"
        clave = (clase, m.group(2), m.group(3))
        it = legal.get(clave)
        if it:
            citadas[it["id"]] = it
            return f'<a class="cita" href="#ref-{it["id"]}">{m.group(0)}</a>'
        return _rojo(m.group(0))

    def sub_paren(m):
        partes = [p.strip() for p in m.group(1).split(";")]
        analizadas = [RE_SUB.match(p) for p in partes]
        if not all(analizadas):
            return m.group(0)  # no parece una cita
        salida = []
        for texto, mm in zip(partes, analizadas):
            auth = mm.group("auth").replace(" et al.", "")
            auth = re.split(r"\s+y\s+|\s+&\s+", auth)[0].strip()
            apellido = auth.split()[0].lower() if auth.split() else auth.lower()
            it = ay.get((apellido, int(mm.group("year"))))
            if it:
                citadas[it["id"]] = it
                salida.append(f'<a class="cita" href="#ref-{it["id"]}">{texto}</a>')
            else:
                salida.append(_rojo(texto))
        return "(" + "; ".join(salida) + ")"

    cuerpo = RE_LEGAL.sub(sub_legal, cuerpo)
    cuerpo = RE_PAREN.sub(sub_paren, cuerpo)
    return cuerpo, list(citadas.values())


def seccion_referencias(citadas):
    """Bloque HTML de la sección Referencias (APA, orden alfabético, sangría francesa)."""
    partes = ['<hr class="sep">', '<h1 id="referencias">Referencias</h1>',
              '<section class="referencias">']
    if not citadas:
        partes.append('<p class="nota-ref">Ninguna de las citas del documento está '
                      'todavía en el gestor (colección MCDA). Las citas pendientes '
                      'aparecen marcadas en rojo en el texto.</p>')
    else:
        partes.append('<p class="nota-ref">Las citas marcadas en rojo en el texto aún '
                      'no están en el gestor.</p>')
        for it in sorted(citadas, key=lambda x: (_primer_apellido(x), x.get("year") or 0)):
            partes.append(f'<p class="ref" id="ref-{it["id"]}">{_apa_entrada(it)}</p>')
    partes.append("</section>")
    return "\n".join(partes)


def main():
    doc = Document(DOCX)
    partes = []
    toc = []
    n = 0
    for el in doc.element.body:
        if el.tag.endswith("}tbl"):
            partes.append(_tabla_html(Table(el, doc)))
            continue
        if not el.tag.endswith("}p"):
            continue
        p = Paragraph(el, doc)
        estilo = p.style.name
        if estilo.startswith("Heading 1") or estilo.startswith("Heading 2"):
            if not _runs_html(p).strip():
                continue
            n += 1
            hid = f"sec{n}"
            nivel = 1 if estilo.startswith("Heading 1") else 2
            texto = _runs_html(p)
            if nivel == 1:
                partes.append('<hr class="sep">')
                partes.append(f'<h1 id="{hid}">{texto}</h1>')
            else:
                partes.append(f'<h2 id="{hid}">{texto}</h2>')
            toc.append((nivel, hid, texto))
        else:
            partes.append(_parrafo_html(p))

    # Referencias del gestor: marcar citas ausentes y armar la lista final
    refs = cargar_referencias()
    cuerpo_texto = "\n".join(p for p in partes if p)
    cuerpo_texto, citadas = marcar_citas(cuerpo_texto, refs)
    toc.append((1, "referencias", "Referencias"))

    enlaces = "".join(
        f'<a class="n{niv}" href="#{hid}">{_sin_tags(txt)}</a>'
        for niv, hid, txt in toc
    )
    tabla_contenido = (
        f'<nav class="toc"><div class="titulo">Contenido</div>{enlaces}</nav>'
        if toc else ""
    )
    cuerpo = tabla_contenido + cuerpo_texto + "\n" + seccion_referencias(citadas)

    sello = datetime.now().strftime("%d/%m/%Y %H:%M")
    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Borrador, trabajo de grado</title>
<style>{CSS}</style></head>
<body>
<div class="barra"><span>Borrador del trabajo de grado</span><span>Actualizado: {sello}</span></div>
{cuerpo}
</body></html>"""

    HTML.write_text(pagina, encoding="utf-8")
    print("Generado:", HTML)


if __name__ == "__main__":
    main()
