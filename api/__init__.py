"""API REST del proyecto MiBici (Equipo 32): sirve el modelo P1 registrado en MLflow.

El paquete es independiente de ``mibici`` (el código de entrenamiento) para que la
imagen del contenedor lleve solo lo necesario para predecir:

* ``config.py``: ajustes leídos de variables de entorno (origen del modelo, lote máximo).
* ``esquemas.py``: contratos de entrada y salida con Pydantic (validación).
* ``modelo.py``: carga del modelo (copia empaquetada del registry o el registry en vivo),
  verificación de integridad y predicción.
* ``main.py``: aplicación FastAPI y endpoints.
"""

__version__ = "1.0.0"
