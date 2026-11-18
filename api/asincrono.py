"""Predicciones asíncronas con Redis Queue (RQ) (entrega 6, opcional).

Para lotes grandes, o clientes que no deben esperar la respuesta (por ejemplo, el cálculo
nocturno de las ~470 estaciones para el día siguiente), la API encola el lote en Redis y responde
**202** de inmediato con un ``tarea_id``; el worker de RQ (``api/worker.py``), que ya tiene el
modelo en memoria, lo procesa y deja el resultado en Redis durante una hora.

* ``POST /predict/asincrono``: mismo cuerpo que ``/predict/lote`` (validado igual, antes de encolar).
* ``GET /tareas/{tarea_id}``: estado (``queued``, ``started``, ``finished``, ``failed``…) y, al
  terminar, las predicciones con el mismo formato que ``/predict/lote``.

Solo existen si la API arranca con ``REDIS_URL`` (``docker-compose.yml`` la define). ``redis`` y
``rq`` se importan aquí, no en ``main.py``, para que la API funcione sin ellos.
"""

from datetime import datetime
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from api.config import COLA
from api.esquemas import EstadoTarea, SolicitudLote, TareaCreada

FUNCION = "api.tareas.predecir_lote"
RESULTADO_TTL = 3600  # segundos que el resultado queda disponible en Redis
LIMITE_SEGUNDOS = 120  # tiempo máximo de un trabajo en el worker


def crear_cola(url: str) -> Queue:
    """La conexión es perezosa: si Redis no responde, cada petición devuelve 503."""
    conexion = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=5)
    return Queue(COLA, connection=conexion, default_timeout=LIMITE_SEGUNDOS)


def cola_actual(request: Request) -> Queue:
    cola = getattr(request.app.state, "cola", None)
    if cola is None:
        raise HTTPException(status_code=503, detail="No hay cola de predicciones configurada")
    return cola


ColaRQ = Annotated[Queue, Depends(cola_actual)]
router = APIRouter(tags=["predicción asíncrona"])


def _iso(valor: datetime | None) -> str | None:
    return valor.isoformat(timespec="seconds") if valor else None


@router.post(
    "/predict/asincrono",
    status_code=202,
    response_model=TareaCreada,
    responses={
        422: {"description": "Entrada inválida: tipo, rango o consistencia entre campos"},
        503: {"description": "Redis no disponible"},
    },
    summary="Encola un lote de predicciones (responde 202 con el id de la tarea)",
)
def encolar(solicitud: SolicitudLote, cola: ColaRQ) -> TareaCreada:
    instancias = [e.model_dump(mode="json") for e in solicitud.instancias]
    tarea_id = uuid.uuid4().hex
    try:
        trabajo = cola.enqueue_call(
            FUNCION,
            args=(instancias,),
            job_id=tarea_id,
            result_ttl=RESULTADO_TTL,
            failure_ttl=RESULTADO_TTL,
            description=f"P1: {len(instancias)} instancias",
        )
    except RedisError as error:
        raise HTTPException(status_code=503, detail=f"Redis no disponible: {error}") from error
    return TareaCreada(
        tarea_id=trabajo.id,
        estado=str(trabajo.get_status(refresh=False).value),
        n=len(instancias),
        consultar=f"/tareas/{trabajo.id}",
    )


@router.get(
    "/tareas/{tarea_id}",
    response_model=EstadoTarea,
    responses={404: {"description": "La tarea no existe o su resultado ya expiró"}},
    summary="Estado y resultado de una tarea asíncrona",
)
def consultar(tarea_id: str, cola: ColaRQ) -> EstadoTarea:
    try:
        trabajo = Job.fetch(tarea_id, connection=cola.connection)
        estado = trabajo.get_status().value
    except NoSuchJobError as error:
        raise HTTPException(status_code=404, detail="La tarea no existe o ya expiró") from error
    except RedisError as error:
        raise HTTPException(status_code=503, detail=f"Redis no disponible: {error}") from error
    resultado = trabajo.return_value() if estado == "finished" else None
    error = None
    if estado == "failed":
        ultimo = trabajo.latest_result()
        error = (ultimo.exc_string or "").strip().splitlines()[-1:] if ultimo else None
        error = error[0] if error else "falló sin detalle"
    return EstadoTarea(
        tarea_id=tarea_id,
        estado=estado,
        resultado=resultado,
        error=error,
        encolada=_iso(trabajo.enqueued_at),
        terminada=_iso(trabajo.ended_at),
    )
