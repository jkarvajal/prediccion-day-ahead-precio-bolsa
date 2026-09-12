# Imagen para desplegar la aplicacion en un entorno de contenedor, por ejemplo
# Hugging Face Spaces. Fija la version de Python, de modo que el despliegue no
# depende de la version que elija el servidor, que fue la causa del fallo previo
# en Streamlit Community Cloud.
FROM python:3.13-slim

WORKDIR /app

# Dependencias del sistema necesarias para compilar algunas ruedas de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalacion de dependencias de la aplicacion
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la aplicacion y artefactos que consume en tiempo de ejecucion
COPY .streamlit/ ./.streamlit/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/features_day_ahead.csv ./data/processed/
COPY data/outputs/hourly_dayahead_metrics_by_hour.csv ./data/outputs/
COPY data/outputs/baseline_global_results.csv ./data/outputs/

# Directorio de escritura para la cache de Streamlit en entornos sin usuario root
ENV HOME=/tmp

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
