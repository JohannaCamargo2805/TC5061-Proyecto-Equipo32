"""Ajustes de la API, leídos de variables de entorno al arrancar.

| Variable | Valor por defecto | Uso |
|---|---|---|
| ``MODELO_URI`` | ``<raíz>/models/servicio`` | Carpeta del modelo empaquetado en la imagen, o una URI del registry (``models:/<nombre>/<versión>`` o ``models:/<nombre>@<alias>``) |
| ``MLFLOW_TRACKING_URI`` | ``http://127.0.0.1:5000`` | Servidor de MLflow; solo se usa cuando ``MODELO_URI`` es una URI ``models:/`` |
| ``VERIFICAR_INTEGRIDAD`` | ``1`` | Compara el SHA-256 de cada archivo del modelo con ``registro.json`` (empaquetado, o del registry si la versión lo trae) |
| ``ACELERAR_INFERENCIA`` | ``0`` | ``1``: predice con el motor en NumPy equivalente al Pipeline (``acelerado.py``), solo si da las mismas predicciones |
| ``REDIS_URL`` | (vacía) | Cola de Redis para ``POST /predict/asincrono`` (worker de RQ); sin ella esos endpoints no existen |

Dentro del contenedor la raíz es ``/app``, así que el valor por defecto apunta a
``/app/models/servicio``, que es donde el ``Containerfile`` copia el modelo.
"""

from dataclasses import dataclass
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
COLA = "predicciones"  # cola de Redis de las predicciones asíncronas (asincrono.py y worker.py)
MODELO_LOCAL_DEFECTO = RAIZ / "models" / "servicio"
TRACKING_DEFECTO = "http://127.0.0.1:5000"


def _booleano(valor: str) -> bool:
    return valor.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Ajustes:
    modelo_uri: str
    tracking_uri: str
    verificar_integridad: bool
    redis_url: str = ""
    acelerar: bool = False

    @property
    def desde_registry(self) -> bool:
        """True si el modelo se descarga del registry al arrancar (``models:/...``)."""
        return self.modelo_uri.startswith("models:/")


def cargar_ajustes() -> Ajustes:
    return Ajustes(
        modelo_uri=os.getenv("MODELO_URI", str(MODELO_LOCAL_DEFECTO)),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", TRACKING_DEFECTO),
        verificar_integridad=_booleano(os.getenv("VERIFICAR_INTEGRIDAD", "1")),
        redis_url=os.getenv("REDIS_URL", "").strip(),
        acelerar=_booleano(os.getenv("ACELERAR_INFERENCIA", "0")),
    )
