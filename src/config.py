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