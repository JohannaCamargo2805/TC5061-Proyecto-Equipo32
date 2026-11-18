"""API REST del proyecto MiBici (Equipo 32): sirve el modelo P1 registrado en MLflow.

El paquete es independiente de ``mibici`` (el código de entrenamiento) para que la
imagen del contenedor lleve solo lo necesario para predecir:

* ``config.py``: ajustes leídos de variables de entorno (origen del modelo, lote máximo).
* ``esquemas.py``: contratos de entrada y salida con Pydantic (validación).
* ``modelo.py``: carga del modelo (copia empaquetada del registry o el registry en vivo),
  verificación de integridad y predicción.
* ``main.py``: aplicación FastAPI y endpoints.
* ``asincrono.py``: ``POST /predict/asincrono`` y ``GET /tareas/{id}`` sobre Redis Queue (RQ),
  activos solo si hay ``REDIS_URL`` (entrega 6, ``docker-compose.yml``).
* ``worker.py``: worker de RQ que carga el modelo una vez y procesa los lotes en cola.
"""

__version__ = "1.1.0"
