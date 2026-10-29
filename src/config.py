"""
Centraliza la configuración general del proyecto, especialmente las rutas de carpetas y
archivos necesarias para la carga, procesamiento, análisis y almacenamiento de resultados.
"""
from pathlib import Path

# Raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas principales
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DRAFTS_DIR = REPORTS_DIR / "drafts"

# Crear carpetas si no existen
for folder in [
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    DRAFTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Carpeta de figuras del documento LaTeX (segunda copia sincronizada).
MEDIA_LATEX_DIR = Path(r"D:\Claude\Paper Latex\workspace\media")

# --- Paleta Okabe-Ito (segura para daltonismo) para las figuras del documento ---
# Es el estandar de facto en publicaciones cientificas: ocho colores distinguibles
# bajo deuteranopia/protanopia y razonables en escala de grises. Se combinan con
# estilos de linea y marcadores para no depender solo del color.
OKABE_ITO = {
    "azul": "#0072B2",
    "naranja": "#E69F00",
    "verde": "#009E73",
    "bermellon": "#D55E00",
    "purpura": "#CC79A7",
    "celeste": "#56B4E9",
    "amarillo": "#F0E442",
    "negro": "#000000",
}
# Roles semanticos usados en el documento.
COLOR_PRIMARIO = OKABE_ITO["azul"]        # serie principal / promedio / dia habil
COLOR_SECUNDARIO = OKABE_ITO["verde"]     # segunda serie / mediana / sabado
COLOR_TERCIARIO = OKABE_ITO["naranja"]    # tercera serie / percentil 95
COLOR_RESALTADO = OKABE_ITO["bermellon"]  # elemento destacado / referencia / volatil
COLOR_CUATERNARIO = OKABE_ITO["purpura"]  # cuarta categoria / festivo
COLOR_NEUTRO = "#444444"                  # ejes, lineas de referencia, texto
BANDA_PICO = "#9a9a9a"                    # franja de horas pico (gris, con alpha)