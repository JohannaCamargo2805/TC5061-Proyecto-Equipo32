# Servidor de MLflow (tracking + Model Registry) del sistema orquestado con docker-compose.yml.
# Misma imagen base que la API (Python 3.14.8 fijado por digest) y MLflow 3.17.2, la versión del
# proyecto. El backend (SQLite) y los artefactos viven en el volumen /mlflow, así que el registry
# sobrevive a "docker compose down" (solo "down -v" lo borra).
ARG IMAGEN_PYTHON=docker.io/library/python:3.14.8-slim-trixie@sha256:fbecd461fc72d21b2143c6eaba25143738079154a6d0ea3da3aec724f50eed4a

FROM ${IMAGEN_PYTHON} AS dependencias
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install pip==26.2.1
COPY requirements-mlflow.txt /tmp/requirements-mlflow.txt
RUN /opt/venv/bin/pip install --only-binary=:all: -r /tmp/requirements-mlflow.txt

FROM ${IMAGEN_PYTHON}
LABEL org.opencontainers.image.title="mibici-mlflow" \
      org.opencontainers.image.description="Servidor de MLflow 3.17.2 (tracking y Model Registry) del proyecto MiBici" \
      org.opencontainers.image.source="https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --create-home --home-dir /home/mlflow \
       --shell /usr/sbin/nologin mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R 10001:10001 /mlflow

COPY --from=dependencias /opt/venv /opt/venv
COPY registrar_modelo.py /app/registrar_modelo.py

USER 10001:10001
WORKDIR /mlflow
VOLUME ["/mlflow"]
EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status != 200)"]

# --allowed-hosts: MLflow 3 rechaza cabeceras Host que no conoce (protección contra DNS rebinding);
# dentro de la red de Compose el servidor se llama "mlflow".
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000", \
     "--workers", "1", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"]
