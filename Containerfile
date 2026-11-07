# Containerfile (estándar OCI) de la API REST de MiBici P1 (FastAPI + modelo del Model Registry).
# Sirve igual con Podman o con Docker:
#
#   podman build -f Containerfile -t mibici-p1-api:modelo-v1 .
#   podman run -d --name mibici-p1-api -p 8000:8000 mibici-p1-api:modelo-v1
#
#   docker build -f Containerfile -t mibici-p1-api:modelo-v1 .
#   docker run -d --name mibici-p1-api -p 8000:8000 mibici-p1-api:modelo-v1
#
# Antes de construir, models/servicio/ debe contener la versión del registry que se va a servir
# (python scripts/exportar_modelo_registry.py). En el repositorio ya está la v1 (@champion).
# La etiqueta de la imagen y MODELO_VERSION deben coincidir con models/servicio/registro.json
# (scripts/ejecutar_servicio.ps1 y "make imagen" los leen de ahí).

# Imagen base con versión exacta: Python 3.14.8, el mismo con el que se entrenó y registró el
# modelo (MLmodel -> python_version). Nombre completo (docker.io/...) para que Podman no
# tenga que preguntar de qué registro descargarla, y digest (el índice multiplataforma que
# resolvió la construcción de la evidencia) para que una etiqueta re-publicada no cambie la base.
ARG IMAGEN_PYTHON=docker.io/library/python:3.14.8-slim-trixie@sha256:fbecd461fc72d21b2143c6eaba25143738079154a6d0ea3da3aec724f50eed4a

# ---------- Etapa 1: dependencias en un entorno virtual ----------
FROM ${IMAGEN_PYTHON} AS dependencias
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install pip==26.2.1
COPY requirements-api.txt /tmp/requirements-api.txt
# Solo ruedas precompiladas: la imagen no necesita compiladores y la instalación es repetible.
RUN /opt/venv/bin/pip install --only-binary=:all: -r /tmp/requirements-api.txt

# ---------- Etapa 2: imagen final (sin pip cache ni archivos de construcción) ----------
FROM ${IMAGEN_PYTHON}
ARG MODELO_VERSION=1

LABEL org.opencontainers.image.title="mibici-p1-api" \
      org.opencontainers.image.description="API REST (FastAPI) del modelo P1 de MiBici registrado en MLflow" \
      org.opencontainers.image.source="https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32" \
      org.opencontainers.image.authors="Leydy Johanna Camargo Medina (Equipo 32, TC5061)" \
      org.opencontainers.image.licenses="MIT" \
      mx.tec.tc5061.modelo="mibici-equipo32-p1-demanda" \
      mx.tec.tc5061.modelo.version="${MODELO_VERSION}"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODELO_URI=/app/models/servicio \
    VERIFICAR_INTEGRIDAD=1

# Usuario sin privilegios: el proceso no corre como root dentro del contenedor.
RUN groupadd --system --gid 10001 mibici \
    && useradd --system --uid 10001 --gid mibici --create-home --home-dir /home/mibici \
       --shell /usr/sbin/nologin mibici

WORKDIR /app
COPY --from=dependencias /opt/venv /opt/venv
COPY api/ /app/api/
COPY models/servicio/ /app/models/servicio/

USER 10001:10001
EXPOSE 8000

# Docker lo usa para marcar el contenedor como healthy/unhealthy; Podman lo respeta al
# construir con --format docker (el formato OCI no guarda HEALTHCHECK).
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
