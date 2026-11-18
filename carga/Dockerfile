# Imagen de la prueba de carga (servicio "locust", perfil "carga" de docker-compose.yml).
# Locust 2.47 genera la carga; pandas y matplotlib arman las gráficas y el análisis FinOps con los
# CSV de Locust y las muestras de docker stats. Misma base que la API (Python 3.14.8 por digest).
ARG IMAGEN_PYTHON=docker.io/library/python:3.14.8-slim-trixie@sha256:fbecd461fc72d21b2143c6eaba25143738079154a6d0ea3da3aec724f50eed4a

FROM ${IMAGEN_PYTHON} AS dependencias
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install pip==26.2.1
COPY requirements-carga.txt /tmp/requirements-carga.txt
RUN /opt/venv/bin/pip install --only-binary=:all: -r /tmp/requirements-carga.txt

FROM ${IMAGEN_PYTHON}
LABEL org.opencontainers.image.title="mibici-carga" \
      org.opencontainers.image.description="Locust y análisis de la prueba de carga de la API MiBici P1" \
      org.opencontainers.image.source="https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32" \
      org.opencontainers.image.licenses="MIT"
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
RUN groupadd --system --gid 10001 carga \
    && useradd --system --uid 10001 --gid carga --create-home --home-dir /home/carga \
       --shell /usr/sbin/nologin carga
COPY --from=dependencias /opt/venv /opt/venv
COPY peticiones.py locustfile.py /carga/
WORKDIR /carga
USER 10001:10001
EXPOSE 8089
ENTRYPOINT ["locust", "-f", "/carga/locustfile.py"]
CMD ["--host", "http://api:8000"]
