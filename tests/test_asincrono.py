"""Pruebas de las predicciones asíncronas con Redis Queue (entrega 6).

Usan ``fakeredis`` (Redis en memoria) y el modelo real empaquetado en ``models/servicio``: la API
encola el lote, un ``SimpleWorker`` de RQ lo procesa en modo *burst* (como el servicio ``worker``
de ``docker-compose.yml``) y el resultado debe ser idéntico al de ``POST /predict/lote``.
"""

import sys

import pytest

# RQ necesita señales POSIX (SIGALRM) y fork: en Windows el worker corre dentro del contenedor
# Linux de docker-compose.yml, así que estas pruebas solo corren en Linux/macOS.
if sys.platform == "win32":
    pytest.skip("RQ no corre de forma nativa en Windows", allow_module_level=True)

import fakeredis
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker

from api import asincrono, tareas
from api.config import Ajustes
from api.esquemas import EJEMPLO_ENTRADA
from api.main import crear_app
from mibici.config import SERVICIO_MODELO_DIR

pytestmark = pytest.mark.skipif(
    not (SERVICIO_MODELO_DIR / "MLmodel").is_file(), reason="falta models/servicio"
)

LOTE = {
    "instancias": [
        EJEMPLO_ENTRADA,
        {**EJEMPLO_ENTRADA, "estacion_id": 3, "lag_1": 20, "media_7d": 30.0},
    ]
}


@pytest.fixture
def redis_falso():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def cliente(redis_falso, monkeypatch):
    monkeypatch.setenv("MODELO_URI", str(SERVICIO_MODELO_DIR))
    monkeypatch.setattr(tareas, "_servicio", None)
    monkeypatch.setattr(
        asincrono,
        "crear_cola",
        lambda url: Queue(asincrono.COLA, connection=redis_falso, default_timeout=60),
    )
    ajustes = Ajustes(
        modelo_uri=str(SERVICIO_MODELO_DIR),
        tracking_uri="http://127.0.0.1:5000",
        verificar_integridad=True,
        redis_url="redis://redis:6379/0",
    )
    with TestClient(crear_app(ajustes)) as c:
        yield c


def _procesar_cola(conexion) -> None:
    trabajador = SimpleWorker([Queue(asincrono.COLA, connection=conexion)], connection=conexion)
    trabajador.work(burst=True)


def test_encola_procesa_y_devuelve_lo_mismo_que_el_lote(cliente, redis_falso):
    r = cliente.post("/predict/asincrono", json=LOTE)
    assert r.status_code == 202, r.text
    tarea = r.json()
    assert tarea["estado"] == "queued" and tarea["n"] == 2
    assert tarea["consultar"] == f"/tareas/{tarea['tarea_id']}"
    assert cliente.get(tarea["consultar"]).json()["estado"] == "queued"

    _procesar_cola(redis_falso)

    estado = cliente.get(tarea["consultar"]).json()
    assert estado["estado"] == "finished" and estado["error"] is None
    sincrono = cliente.post("/predict/lote", json=LOTE).json()
    assert estado["resultado"]["predicciones"] == sincrono["predicciones"]
    assert estado["resultado"]["modelo"] == sincrono["modelo"]


def test_lote_invalido_no_se_encola(cliente, redis_falso):
    malo = {"instancias": [{**EJEMPLO_ENTRADA, "fin_semana": 1}]}
    assert cliente.post("/predict/asincrono", json=malo).status_code == 422
    assert Queue(asincrono.COLA, connection=redis_falso).count == 0


def test_tarea_inexistente_es_404(cliente):
    assert cliente.get("/tareas/no-existe").status_code == 404


def test_la_raiz_anuncia_los_endpoints_asincronos(cliente):
    info = cliente.get("/").json()
    assert info["predicciones_asincronas"] is True
    assert "POST /predict/asincrono" in info["endpoints"]


def test_sin_redis_url_no_hay_endpoints_asincronos():
    ajustes = Ajustes(
        modelo_uri=str(SERVICIO_MODELO_DIR),
        tracking_uri="http://127.0.0.1:5000",
        verificar_integridad=True,
    )
    with TestClient(crear_app(ajustes)) as c:
        assert c.post("/predict/asincrono", json=LOTE).status_code == 404
        assert c.get("/").json()["predicciones_asincronas"] is False


def test_redis_caido_responde_503(monkeypatch):
    monkeypatch.setattr(
        asincrono, "crear_cola", lambda url: Queue(asincrono.COLA, connection=_redis_cerrado())
    )
    ajustes = Ajustes(
        modelo_uri=str(SERVICIO_MODELO_DIR),
        tracking_uri="http://127.0.0.1:5000",
        verificar_integridad=True,
        redis_url="redis://127.0.0.1:1/0",
    )
    with TestClient(crear_app(ajustes)) as c:
        assert c.post("/predict/asincrono", json=LOTE).status_code == 503


def _redis_cerrado():
    from redis import Redis

    return Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.5)


def test_el_trabajo_valida_su_entrada(monkeypatch):
    monkeypatch.setenv("MODELO_URI", str(SERVICIO_MODELO_DIR))
    monkeypatch.setattr(tareas, "_servicio", None)
    with pytest.raises(ValueError):
        tareas.predecir_lote([{**EJEMPLO_ENTRADA, "dia_semana": 9}])
    salida = tareas.predecir_lote([EJEMPLO_ENTRADA])
    assert salida["n"] == 1 and salida["predicciones"][0]["viajes_esperados"] > 0
