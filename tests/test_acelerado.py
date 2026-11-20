"""Motor de inferencia en NumPy (entrega 6): mismas predicciones que el Pipeline registrado.

Con el modelo real empaquetado en ``models/servicio``: el motor se compila de sus parámetros,
coincide con ``pyfunc`` en las 2,000 peticiones de la prueba de carga y en entradas extremas, la
API responde lo mismo con y sin ``ACELERAR_INFERENCIA`` y, si el modelo no tiene la estructura
conocida, la API sigue con ``pyfunc`` en lugar de fallar.
"""

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from api import acelerado
from api.config import Ajustes
from api.esquemas import VARIABLES_P1
from api.main import crear_app
from api.modelo import cargar_modelo
from mibici.config import PROJ_ROOT, SERVICIO_MODELO_DIR

pytestmark = pytest.mark.skipif(
    not (SERVICIO_MODELO_DIR / "MLmodel").is_file(), reason="falta models/servicio"
)
sys.path.insert(0, str(PROJ_ROOT / "carga"))
from peticiones import generar  # noqa: E402


def _ajustes(acelerar: bool) -> Ajustes:
    return Ajustes(modelo_uri=str(SERVICIO_MODELO_DIR), tracking_uri="http://127.0.0.1:5000",
                   verificar_integridad=True, acelerar=acelerar)


@pytest.fixture(scope="module")
def servicios():
    return cargar_modelo(_ajustes(False)), cargar_modelo(_ajustes(True))


def test_el_motor_se_activa_y_lo_declara(servicios):
    normal, rapido = servicios
    assert normal.rapido is None and normal.info.motor_inferencia == "mlflow.pyfunc"
    assert rapido.rapido is not None
    assert rapido.info.motor_inferencia.startswith("numpy (verificado")


def test_mismas_predicciones_en_las_peticiones_de_la_prueba_de_carga(servicios):
    normal, rapido = servicios
    filas = [{c: float(p[c]) for c in VARIABLES_P1} for p in generar(2000)]
    a, b = normal.predecir(filas), rapido.predecir(filas)
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)


def test_mismas_predicciones_en_entradas_extremas(servicios):
    normal, _ = servicios
    x = acelerado.entradas_sinteticas(500, semilla=1)
    motor = acelerado.compilar(normal.modelo)
    referencia = normal.modelo.predict(pd.DataFrame(x, columns=list(VARIABLES_P1)))
    np.testing.assert_allclose(motor(x), np.asarray(referencia), rtol=1e-9)
    assert acelerado.verificar(motor, normal.modelo) <= acelerado.TOLERANCIA


def test_la_api_responde_igual_con_y_sin_motor(servicios):
    lote = {"instancias": [{k: v for k, v in p.items()} for p in generar(50)]}
    with TestClient(crear_app(_ajustes(False))) as c1, TestClient(crear_app(_ajustes(True))) as c2:
        r1, r2 = c1.post("/predict/lote", json=lote).json(), c2.post("/predict/lote", json=lote).json()
        assert r1["predicciones"] == r2["predicciones"]
        assert c2.get("/modelo").json()["motor_inferencia"].startswith("numpy")


def test_modelo_desconocido_sigue_con_pyfunc(servicios):
    normal, _ = servicios
    assert acelerado.compilar(object()) is None

    class Otro:
        named_steps = {"prep": None, "modelo": None}

    assert acelerado.compilar(Otro()) is None
    servicio = cargar_modelo(_ajustes(False))
    servicio.modelo = Otro()
    from api.modelo import acelerar

    assert acelerar(servicio).rapido is None
    assert "estructura conocida" in servicio.info.motor_inferencia


def test_el_motor_no_toca_pandas_por_peticion(servicios, monkeypatch):
    _, rapido = servicios
    llamado = []
    monkeypatch.setattr(pd, "DataFrame", lambda *a, **k: llamado.append(1))
    fila = {c: float(v) for c, v in zip(VARIABLES_P1, acelerado.entradas_sinteticas(1)[0])}
    assert rapido.predecir([fila]).shape == (1,)
    assert not llamado


def test_la_ruta_del_modelo_existe():
    assert Path(SERVICIO_MODELO_DIR, "registro.json").is_file()
