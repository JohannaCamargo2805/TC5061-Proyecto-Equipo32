"""Ajustes de la API, leídos de variables de entorno al arrancar.

| Variable | Valor por defecto | Uso |
|---|---|---|
| ``MODELO_URI`` | ``<raíz>/models/servicio`` | Carpeta del modelo empaquetado en la imagen, o una URI del registry (``models:/<nombre>/<versión>`` o ``models:/<nombre>@<alias>``) |
| ``MLFLOW_TRACKING_URI`` | ``http://127.0.0.1:5000`` | Servidor de MLflow; solo se usa cuando ``MODELO_URI`` es una URI ``models:/`` |
| ``VERIFICAR_INTEGRIDAD`` | ``1`` | Compara el SHA-256 de cada archivo del modelo empaquetado con ``registro.json`` |

Dentro del contenedor la raíz es ``/app``, así que el valor por defecto apunta a
``/app/models/servicio``, que es donde el ``Containerfile`` copia el modelo.
"""

from dataclasses import dataclass
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODELO_LOCAL_DEFECTO = RAIZ / "models" / "servicio"
TRACKING_DEFECTO = "http://127.0.0.1:5000"


def _booleano(valor: str) -> bool:
    return valor.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Ajustes:
    modelo_uri: str
    tracking_uri: str
    verificar_integridad: bool

    @property
    def desde_registry(self) -> bool:
        """True si el modelo se descarga del registry al arrancar (``models:/...``)."""
        return self.modelo_uri.startswith("models:/")


def cargar_ajustes() -> Ajustes:
    return Ajustes(
        modelo_uri=os.getenv("MODELO_URI", str(MODELO_LOCAL_DEFECTO)),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", TRACKING_DEFECTO),
        verificar_integridad=_booleano(os.getenv("VERIFICAR_INTEGRIDAD", "1")),
    )
