"""API REST de MiBici P1: viajes esperados por estación para el día siguiente.

El modelo se carga una sola vez al arrancar (``lifespan``); si no se puede cargar o
no pasa la verificación de integridad y de firma, el proceso termina con error en
lugar de servir respuestas incorrectas. Ejecutar en local::

    uvicorn api.main:app --port 8000

Documentación interactiva (OpenAPI/Swagger) en ``/docs``.
"""

from contextlib import asynccontextmanager
import logging
import math
import time
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import __version__
from api.config import Ajustes, cargar_ajustes
from api.esquemas import (
    LOTE_MAXIMO,
    EntradaP1,
    EstadoSalud,
    InfoModelo,
    PrediccionItem,
    PrediccionP1,
    RespuestaLote,
    SolicitudLote,
)
from api.modelo import ErrorModelo, ModeloServicio, cargar_modelo

log = logging.getLogger("uvicorn.error")

DESCRIPCION = """
Sirve el modelo **P1** del proyecto MiBici (Equipo 32): regresión de los viajes que
se iniciarán en una estación el día siguiente, a partir de 12 variables conocidas al
cierre del día anterior. El modelo es la versión registrada en el **Model Registry de
MLflow** como `mibici-equipo32-p1-demanda` (alias `champion`); `GET /modelo` muestra la
versión exacta, su corrida de origen y sus métricas.

* `POST /predict`: una estación y un día.
* `POST /predict/lote`: hasta 1000 estaciones o días en una sola llamada.
* Las entradas se validan por tipo, rango y consistencia entre campos; un error
  devuelve **422** con el detalle de cada campo.
"""

ERROR_422 = {422: {"description": "Entrada inválida: tipo, rango o consistencia entre campos"}}


async def cabeceras_de_trazabilidad(request: Request, call_next):
    """Agrega el modelo que respondió y el tiempo de respuesta a cada respuesta."""
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    respuesta.headers["X-Tiempo-Respuesta-ms"] = f"{(time.perf_counter() - inicio) * 1000:.1f}"
    servicio = getattr(request.app.state, "servicio", None)
    if servicio is not None:
        respuesta.headers["X-Modelo"] = f"{servicio.info.nombre}/{servicio.info.version}"
    return respuesta


def _json_valido(valor):
    """NaN e infinito no son JSON válido: se devuelven como texto al describir el error."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {k: _json_valido(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [_json_valido(v) for v in valor]
    return valor


async def error_de_validacion(request: Request, error: RequestValidationError) -> JSONResponse:
    """Mismo formato 422 de FastAPI, pero sin fallar cuando la entrada trae NaN o infinito."""
    return JSONResponse(
        status_code=422, content={"detail": _json_valido(jsonable_encoder(error.errors()))}
    )


def servicio_actual(request: Request) -> ModeloServicio:
    servicio = getattr(request.app.state, "servicio", None)
    if servicio is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")
    return servicio


Servicio = Annotated[ModeloServicio, Depends(servicio_actual)]


def _redondear(valor: float) -> int:
    return math.floor(valor + 0.5)


def _predecir(servicio: ModeloServicio, entradas: list[EntradaP1]) -> list[PrediccionItem]:
    try:
        valores = servicio.predecir([e.fila() for e in entradas])
    except ErrorModelo as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return [
        PrediccionItem(
            estacion_id=e.estacion_id,
            fecha=e.fecha,
            viajes_esperados=round(float(v), 3),
            viajes_esperados_redondeado=_redondear(float(v)),
            advertencias=e.advertencias(),
        )
        for e, v in zip(entradas, valores)
    ]


router = APIRouter()


@router.get("/", tags=["servicio"], summary="Descripción del servicio")
def inicio(servicio: Servicio) -> dict:
    return {
        "servicio": "API MiBici P1 (Equipo 32)",
        "version_api": __version__,
        "modelo": f"{servicio.info.nombre} v{servicio.info.version}",
        "documentacion": "/docs",
        "endpoints": ["GET /health", "GET /modelo", "POST /predict", "POST /predict/lote"],
    }


@router.get(
    "/health",
    response_model=EstadoSalud,
    tags=["servicio"],
    summary="Verificación de salud (la usa el HEALTHCHECK del contenedor)",
)
def salud(servicio: Servicio) -> EstadoSalud:
    return EstadoSalud(
        estado="ok",
        modelo=servicio.info.nombre,
        version=servicio.info.version,
        origen=servicio.info.origen,
    )


@router.get(
    "/modelo",
    response_model=InfoModelo,
    tags=["servicio"],
    summary="Versión del registry que se sirve, linaje y métricas",
)
def informacion_modelo(servicio: Servicio) -> InfoModelo:
    return servicio.info


@router.post(
    "/predict",
    response_model=PrediccionP1,
    tags=["predicción"],
    responses=ERROR_422,
    summary="Viajes esperados de una estación para el día D",
)
def predecir(entrada: EntradaP1, servicio: Servicio) -> PrediccionP1:
    item = _predecir(servicio, [entrada])[0]
    return PrediccionP1(**item.model_dump(), modelo=servicio.resumen)


@router.post(
    "/predict/lote",
    response_model=RespuestaLote,
    tags=["predicción"],
    responses=ERROR_422,
    summary=f"Viajes esperados de varias estaciones o días (máximo {LOTE_MAXIMO})",
)
def predecir_lote(solicitud: SolicitudLote, servicio: Servicio) -> RespuestaLote:
    predicciones = _predecir(servicio, solicitud.instancias)
    return RespuestaLote(n=len(predicciones), modelo=servicio.resumen, predicciones=predicciones)


def crear_app(ajustes: Ajustes | None = None) -> FastAPI:
    """Crea la aplicación. Sin ``ajustes`` los lee de las variables de entorno al arrancar."""

    @asynccontextmanager
    async def ciclo_de_vida(aplicacion: FastAPI):
        servicio = cargar_modelo(ajustes or cargar_ajustes())
        aplicacion.state.servicio = servicio
        i = servicio.info
        log.info(
            "Modelo %s v%s listo (origen: %s, aliases: %s, integridad verificada: %s)",
            i.nombre,
            i.version,
            i.origen,
            ",".join(i.alias) or "-",
            i.integridad_verificada,
        )
        yield
        aplicacion.state.servicio = None

    aplicacion = FastAPI(
        title="API MiBici P1: demanda diaria por estación",
        version=__version__,
        description=DESCRIPCION,
        lifespan=ciclo_de_vida,
    )
    aplicacion.middleware("http")(cabeceras_de_trazabilidad)
    aplicacion.add_exception_handler(RequestValidationError, error_de_validacion)
    aplicacion.include_router(router)
    return aplicacion


app = crear_app()
