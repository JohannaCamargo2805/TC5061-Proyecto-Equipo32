"""Trabajos que ejecuta el worker de RQ (``api/worker.py``).

El modelo se carga una sola vez por proceso (``precargar``) con los mismos ajustes que la API
(``MODELO_URI``, ``MLFLOW_TRACKING_URI``, ``VERIFICAR_INTEGRIDAD``) y se reutiliza en cada trabajo:
el worker es un ``SimpleWorker`` (sin ``fork`` por trabajo), así que no paga la carga por lote.
"""

from api.config import cargar_ajustes
from api.esquemas import RespuestaLote, SolicitudLote
from api.modelo import ModeloServicio, cargar_modelo, predicciones

_servicio: ModeloServicio | None = None


def precargar() -> ModeloServicio:
    global _servicio
    if _servicio is None:
        _servicio = cargar_modelo(cargar_ajustes())
    return _servicio


def predecir_lote(instancias: list[dict]) -> dict:
    """Valida de nuevo el lote (el contrato no depende de quién encola) y predice en una llamada."""
    solicitud = SolicitudLote.model_validate({"instancias": instancias})
    servicio = precargar()
    items = predicciones(servicio, solicitud.instancias)
    respuesta = RespuestaLote(n=len(items), modelo=servicio.resumen, predicciones=items)
    return respuesta.model_dump(mode="json")
