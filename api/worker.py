"""Worker de RQ que procesa la cola ``predicciones`` (servicio ``worker`` de docker-compose.yml).

    python -m api.worker            # carga el modelo y atiende la cola hasta recibir SIGTERM
    python -m api.worker --salud    # healthcheck: Redis responde y este worker está registrado

Carga el modelo **antes** de aceptar trabajos (si no carga, el contenedor falla y Compose lo
reinicia) y usa ``SimpleWorker``: ejecuta cada trabajo en el mismo proceso, con el modelo ya en
memoria, en lugar de hacer ``fork`` por trabajo.
"""

import logging
import os
import socket
import sys

from redis import Redis
from rq import Queue, SimpleWorker, Worker

# Solo lo liviano a nivel de módulo: el healthcheck (--salud) corre cada 10 s y no debe importar
# MLflow ni el modelo (costaba ~1 s de CPU y ~100 MB por chequeo en la primera medición).
from api.config import COLA

PREFIJO = "mibici-worker"
log = logging.getLogger("mibici.worker")


def _conexion() -> Redis:
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        raise SystemExit("Falta REDIS_URL (por ejemplo redis://redis:6379/0)")
    return Redis.from_url(url, socket_connect_timeout=3)


def _nombre() -> str:
    return f"{PREFIJO}-{socket.gethostname()}"


def salud() -> int:
    conexion = _conexion()
    conexion.ping()
    nombres = {w.name for w in Worker.all(connection=conexion)}
    return 0 if _nombre() in nombres else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from api import tareas

    servicio = tareas.precargar()
    i = servicio.info
    log.info("Modelo %s v%s listo en el worker (origen: %s)", i.nombre, i.version, i.origen)
    conexion = _conexion()
    trabajador = SimpleWorker(
        [Queue(COLA, connection=conexion)], connection=conexion, name=_nombre()
    )
    trabajador.work(with_scheduler=False, logging_level="INFO")
    return 0


if __name__ == "__main__":
    sys.exit(salud() if "--salud" in sys.argv else main())
