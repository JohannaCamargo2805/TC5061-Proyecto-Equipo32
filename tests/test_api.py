"""Pruebas de la API REST de la entrega 4 (FastAPI).

* Recuperación del modelo: un Model Registry temporal (SQLite) con dos versiones de
  un modelo P1 sintético; ``scripts/exportar_modelo_registry.py`` lo descarga por
  alias y por versión, y la API lo sirve verificando las huellas SHA-256.
* Contrato: ``/health``, ``/modelo``, ``/predict`` y ``/predict/lote`` responden con
  el esquema de salida y las mismas predicciones que el modelo registrado.
* Validación: cada regla de entrada (tipo, rango y consistencia) rechaza con 422.
* Modelo real: si existe ``models/servicio`` (la copia del registry que se empaqueta en
  la imagen), la API reproduce sus predicciones y, con los datos reales, la métrica
  ``test_mae`` registrada en MLflow.
"""

import json
import shutil
import sys

from fastapi.testclient import TestClient
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from api.config import Ajustes
from api.esquemas import EJEMPLO_ENTRADA, LOTE_MAXIMO, VARIABLES_P1, EntradaP1
from api.main import crear_app
from api.modelo import ErrorModelo, cargar_modelo
from mibici.config import PROJ_ROOT, SERVICIO_MODELO_DIR

sys.path.insert(0, str(PROJ_ROOT / "scripts"))
import exportar_modelo_registry as exportador  # noqa: E402

NOMBRE = "prueba-mibici-p1"
HAY_MODELO_REAL = (SERVICIO_MODELO_DIR / "registro.json").is_file()


def _datos_sinteticos(n: int = 400, semilla: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Filas con la misma forma que P1 (12 variables double) y un conteo de Poisson."""
    rng = np.random.default_rng(semilla)
    dia = rng.integers(0, 7, n)
    lags = rng.poisson(30, size=(n, 4)).astype("float64")
    x = pd.DataFrame({
        "dia_semana": dia, "fin_semana": (dia >= 5).astype(int), "feriado": rng.integers(0, 2, n),
        "lag_1": lags[:, 0], "lag_2": lags[:, 1], "lag_7": lags[:, 2], "lag_14": lags[:, 3],
        "media_7d": lags.mean(axis=1), "media_28d": lags.mean(axis=1) + 1,
        "media_mismo_dia_4sem": lags[:, 2:].mean(axis=1) + 1,
        "dias_activa": rng.integers(28, 300, n), "capacidad": rng.integers(11, 40, n),
    })[list(VARIABLES_P1)].astype("float64")
    y = rng.poisson(np.clip(x["media_7d"].to_numpy(), 1, None))
    return x, y


def _registrar(cliente: MlflowClient, experimento: str, x: pd.DataFrame, y: np.ndarray):
    modelo = Pipeline([("esc", StandardScaler()), ("modelo", PoissonRegressor(alpha=1e-3))]).fit(x, y)
    with mlflow.start_run(experiment_id=experimento) as corrida:
        mlflow.log_metrics({"val_mae": 4.2, "test_mae": 4.5, "test_wape": 0.15, "train_mae": 4.0,
                            "costo_inferencia_ms_1k": 3.4, "costo_entrenamiento_seg": 1.5})
        info = mlflow.sklearn.log_model(modelo, name="model", input_example=x.head(3),
                                        signature=infer_signature(x, modelo.predict(x)))
    return modelo, corrida.info.run_id, info.model_uri


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Registry temporal: v1 (@champion) y v2 (@challenger) de un modelo con firma P1."""
    anterior = mlflow.get_tracking_uri()
    raiz = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{(raiz / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    cliente = MlflowClient()
    experimento = cliente.create_experiment("prueba-api", artifact_location=(raiz / "artefactos").as_uri())
    x, y = _datos_sinteticos()
    cliente.create_registered_model(NOMBRE)
    modelos, corridas = {}, {}
    for version, alias, semilla in ((1, "champion", 42), (2, "challenger", 7)):
        x, y = _datos_sinteticos(semilla=semilla)
        modelo, run_id, fuente = _registrar(cliente, experimento, x, y)
        mv = cliente.create_model_version(NOMBRE, fuente, run_id)
        for k, v in {"git_commit": "commit-de-prueba", "data_version": "datos-de-prueba",
                     "model": "poisson"}.items():
            cliente.set_model_version_tag(NOMBRE, mv.version, k, v)
        cliente.set_registered_model_alias(NOMBRE, alias, mv.version)
        modelos[version], corridas[version] = modelo, run_id
    yield {"uri": uri, "modelos": modelos, "corridas": corridas, "experimento": experimento}
    mlflow.set_tracking_uri(anterior)


@pytest.fixture(scope="module")
def exportado(registry, tmp_path_factory):
    """Copia de @champion descargada con el script, como la que copia el Containerfile."""
    carpeta = tmp_path_factory.mktemp("modelos") / "servicio"
    registro = exportador.exportar(NOMBRE, 0, "champion", carpeta, registry["uri"])
    return carpeta, registro


def _ajustes(uri, tracking: str = "", verificar: bool = True) -> Ajustes:
    return Ajustes(modelo_uri=str(uri), tracking_uri=tracking, verificar_integridad=verificar)


@pytest.fixture(scope="module")
def cliente(exportado):
    with TestClient(crear_app(_ajustes(exportado[0]))) as c:
        yield c


def _pred(modelo, entradas: list[dict]) -> np.ndarray:
    filas = [EntradaP1(**e).fila() for e in entradas]
    return modelo.predict(pd.DataFrame(filas, columns=list(VARIABLES_P1)))


# --------------------------------------------------------------------------------------
# Recuperación desde el Model Registry
# --------------------------------------------------------------------------------------
def test_exportar_por_alias_resuelve_la_version(registry, exportado):
    carpeta, registro = exportado
    assert registro["modelo"] == NOMBRE and registro["version"] == 1
    assert registro["aliases"] == ["champion"] and registro["alias_solicitado"] == "champion"
    assert registro["aliases_del_modelo"] == {"challenger": 2, "champion": 1}
    assert registro["uri_registry"] == f"models:/{NOMBRE}/1"
    assert registro["run_id"] == registry["corridas"][1]
    assert registro["firma_entrada"] == list(VARIABLES_P1)
    assert registro["metricas"]["test_mae"] == 4.5 and registro["metricas"]["costo_inferencia_ms_1k"] == 3.4
    assert "train_mae" not in registro["metricas"] and "costo_entrenamiento_seg" not in registro["metricas"]
    assert registro["etiquetas_version"]["data_version"] == "datos-de-prueba"
    assert {"MLmodel", "input_example.json"} <= set(registro["sha256"])
    for relativa, huella in registro["sha256"].items():
        assert exportador.sha256(carpeta / relativa) == huella
    assert json.loads((carpeta / "registro.json").read_text(encoding="utf-8")) == registro


def test_exportar_por_version_explicita(registry, tmp_path):
    registro = exportador.exportar(NOMBRE, 2, "champion", tmp_path / "v2", registry["uri"])
    assert registro["version"] == 2 and registro["aliases"] == ["challenger"]
    assert registro["alias_solicitado"] is None and registro["run_id"] == registry["corridas"][2]


def test_exportar_no_borra_carpetas_fuera_de_models(registry, tmp_path):
    ajena = tmp_path / "ajena"
    ajena.mkdir()
    (ajena / "importante.txt").write_text("no borrar", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        exportador.exportar(NOMBRE, 1, "champion", ajena, registry["uri"])
    assert (ajena / "importante.txt").read_text(encoding="utf-8") == "no borrar"


def test_registro_no_guarda_credenciales_de_la_uri():
    usuario, clave = "ana", "cla" + "ve"  # credencial FALSA armada en tiempo de ejecución
    uri = f"http://{usuario}:{clave}@mlflow:5000"
    assert exportador.uri_sin_credenciales(uri) == "http://mlflow:5000"
    assert exportador.uri_sin_credenciales("http://127.0.0.1:5000") == "http://127.0.0.1:5000"


# --------------------------------------------------------------------------------------
# Carga del modelo en la API
# --------------------------------------------------------------------------------------
def test_integridad_detecta_un_archivo_alterado(exportado, tmp_path):
    copia = tmp_path / "servicio"
    shutil.copytree(exportado[0], copia)
    with open(copia / "conda.yaml", "a", encoding="utf-8") as f:
        f.write("# cambio\n")
    with pytest.raises(ErrorModelo, match="SHA-256"):
        cargar_modelo(_ajustes(copia))
    assert cargar_modelo(_ajustes(copia, verificar=False)).info.version == 1


def test_sin_registro_json_no_arranca(exportado, tmp_path):
    copia = tmp_path / "servicio"
    shutil.copytree(exportado[0], copia)
    (copia / "registro.json").unlink()
    with pytest.raises(ErrorModelo, match="registro.json"):
        cargar_modelo(_ajustes(copia))


def test_carpeta_sin_modelo_no_arranca(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELO_URI", str(tmp_path))  # la app lee MODELO_URI al arrancar
    with pytest.raises(ErrorModelo, match="exportar_modelo_registry"), TestClient(crear_app()):
        pass


def test_firma_distinta_no_arranca(registry, tmp_path):
    x, y = _datos_sinteticos()
    x = x.drop(columns="capacidad")
    _, run_id, fuente = _registrar(MlflowClient(), registry["experimento"], x, y)
    MlflowClient().create_registered_model("prueba-firma-distinta")
    MlflowClient().create_model_version("prueba-firma-distinta", fuente, run_id)
    carpeta = tmp_path / "servicio"
    exportador.exportar("prueba-firma-distinta", 1, "champion", carpeta, registry["uri"])
    with pytest.raises(ErrorModelo, match="firma"):
        cargar_modelo(_ajustes(carpeta))


@pytest.mark.parametrize(("uri", "version"), [(f"models:/{NOMBRE}@champion", 1),
                                              (f"models:/{NOMBRE}/2", 2)])
def test_carga_directa_desde_el_registry(registry, uri, version):
    servicio = cargar_modelo(_ajustes(uri, registry["uri"]))
    assert servicio.info.version == version and servicio.info.origen == "registry"
    assert servicio.info.uri_registry == f"models:/{NOMBRE}/{version}"
    np.testing.assert_allclose(servicio.predecir([EntradaP1(**EJEMPLO_ENTRADA).fila()]),
                               np.clip(_pred(registry["modelos"][version], [EJEMPLO_ENTRADA]), 0, None))


def test_uri_del_registry_invalida():
    with pytest.raises(ErrorModelo, match="no es válida"):
        cargar_modelo(_ajustes("models:/sin-version"))


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
def test_health_y_modelo(cliente, registry):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "modelo": NOMBRE, "version": 1, "origen": "empaquetado"}
    info = cliente.get("/modelo").json()
    assert info["version"] == 1 and info["alias"] == ["champion"] and info["integridad_verificada"]
    assert info["run_id"] == registry["corridas"][1]
    assert [v["nombre"] for v in info["variables"]] == list(VARIABLES_P1)
    assert info["metricas"]["test_mae"] == 4.5
    assert cliente.get("/").json()["documentacion"] == "/docs"


def test_predict_devuelve_la_prediccion_del_modelo_registrado(cliente, registry):
    r = cliente.post("/predict", json=EJEMPLO_ENTRADA)
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    esperado = float(_pred(registry["modelos"][1], [EJEMPLO_ENTRADA])[0])
    assert cuerpo["viajes_esperados"] == pytest.approx(esperado, abs=5e-4)
    assert cuerpo["viajes_esperados_redondeado"] == int(np.floor(esperado + 0.5))
    assert cuerpo["estacion_id"] == 2 and cuerpo["fecha"] == "2025-11-04"
    assert cuerpo["modelo"] == {"nombre": NOMBRE, "version": 1, "alias": ["champion"],
                                "run_id": registry["corridas"][1]}
    assert r.headers["X-Modelo"] == f"{NOMBRE}/1" and float(r.headers["X-Tiempo-Respuesta-ms"]) >= 0


def test_predict_lote_conserva_el_orden(cliente, registry):
    instancias = [{**EJEMPLO_ENTRADA, "estacion_id": i, "lag_1": 10 * i, "media_7d": 60.0 + i}
                  for i in range(1, 6)]
    r = cliente.post("/predict/lote", json={"instancias": instancias})
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert cuerpo["n"] == 5 and [p["estacion_id"] for p in cuerpo["predicciones"]] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([p["viajes_esperados"] for p in cuerpo["predicciones"]],
                               _pred(registry["modelos"][1], instancias), atol=5e-4)


def test_acepta_numeros_enteros_escritos_como_decimales(cliente):
    """Clientes que envían los conteos como 61.0 (p. ej. desde pandas) no se rechazan."""
    r = cliente.post("/predict", json={**EJEMPLO_ENTRADA, "lag_1": 61.0, "capacidad": 15.0})
    assert r.status_code == 200, r.text


def test_advertencia_fuera_del_rango_de_entrenamiento(cliente):
    fuera = cliente.post("/predict", json=EJEMPLO_ENTRADA).json()["advertencias"]
    assert len(fuera) == 1 and fuera[0].startswith("dias_activa=307")
    dentro = cliente.post("/predict", json={**EJEMPLO_ENTRADA, "dias_activa": 200}).json()
    assert dentro["advertencias"] == []
    # lag_14 llega a 325 en TRAIN (los otros rezagos a 550): 400 ya es extrapolación.
    lag14 = {**EJEMPLO_ENTRADA, "dias_activa": 200, "lag_14": 400, "media_mismo_dia_4sem": 120.0,
             "media_28d": 70.0}
    avisos = cliente.post("/predict", json=lag14).json()["advertencias"]
    assert len(avisos) == 1 and avisos[0].startswith("lag_14=400")


INVALIDAS = {
    "falta-capacidad": ({k: v for k, v in EJEMPLO_ENTRADA.items() if k != "capacidad"}, "capacidad"),
    "campo-desconocido": ({**EJEMPLO_ENTRADA, "lluvia": 1}, "lluvia"),
    "dia-semana-7": ({**EJEMPLO_ENTRADA, "dia_semana": 7}, "dia_semana"),
    "lag-negativo": ({**EJEMPLO_ENTRADA, "lag_1": -3}, "lag_1"),
    "lag-no-entero": ({**EJEMPLO_ENTRADA, "lag_2": 27.5}, "lag_2"),
    "lag-texto": ({**EJEMPLO_ENTRADA, "lag_7": "65"}, "lag_7"),
    "feriado-booleano": ({**EJEMPLO_ENTRADA, "feriado": True}, "feriado"),
    "lag-implausible": ({**EJEMPLO_ENTRADA, "lag_14": 50_000}, "lag_14"),
    "capacidad-cero": ({**EJEMPLO_ENTRADA, "capacidad": 0}, "capacidad"),
    "historia-corta": ({**EJEMPLO_ENTRADA, "dias_activa": 10}, "dias_activa"),
    "fecha-invalida": ({**EJEMPLO_ENTRADA, "fecha": "2025-13-40"}, "fecha"),
    "fin-semana-inconsistente": ({**EJEMPLO_ENTRADA, "fin_semana": 1}, "fin_semana=1"),
    "fecha-otro-dia": ({**EJEMPLO_ENTRADA, "fecha": "2025-11-05"}, "dia_semana=1"),
    "media-7d-imposible": ({**EJEMPLO_ENTRADA, "media_7d": 5.0}, "media_7d=5.0"),
    "media-4sem-imposible": ({**EJEMPLO_ENTRADA, "media_mismo_dia_4sem": 10.0}, "media_mismo_dia_4sem"),
    "media-28d-imposible": ({**EJEMPLO_ENTRADA, "media_28d": 10.0}, "media_28d=10.0"),
    "estacion-texto": ({**EJEMPLO_ENTRADA, "estacion_id": "2"}, "estacion_id"),
    "estacion-booleano": ({**EJEMPLO_ENTRADA, "estacion_id": True}, "estacion_id"),
    "estacion-implausible": ({**EJEMPLO_ENTRADA, "estacion_id": 10**12}, "estacion_id"),
    "fecha-numero": ({**EJEMPLO_ENTRADA, "fecha": 1762214400}, "fecha"),
}


@pytest.mark.parametrize("caso", list(INVALIDAS), ids=list(INVALIDAS))
def test_entradas_invalidas_se_rechazan_con_422(cliente, caso):
    cuerpo, se_menciona = INVALIDAS[caso]
    r = cliente.post("/predict", json=cuerpo)
    assert r.status_code == 422, r.text
    assert se_menciona in json.dumps(r.json()["detail"], ensure_ascii=False)


def test_nan_se_rechaza(cliente):
    texto = json.dumps({**EJEMPLO_ENTRADA, "media_7d": 0.0}).replace('"media_7d": 0.0', '"media_7d": NaN')
    r = cliente.post("/predict", content=texto, headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "media_7d" in r.text


@pytest.mark.parametrize("n", [0, LOTE_MAXIMO + 1], ids=["lote-vacio", "lote-excedido"])
def test_lote_fuera_de_limites(cliente, n):
    r = cliente.post("/predict/lote", json={"instancias": [EJEMPLO_ENTRADA] * n})
    assert r.status_code == 422


def test_lote_con_una_instancia_invalida_indica_su_posicion(cliente):
    r = cliente.post("/predict/lote", json={"instancias": [EJEMPLO_ENTRADA,
                                                           {**EJEMPLO_ENTRADA, "dia_semana": 9}]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:3] == ["body", "instancias", 1]


# --------------------------------------------------------------------------------------
# Modelo real empaquetado (models/servicio, descargado del registry del proyecto)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cliente_real():
    if not HAY_MODELO_REAL:
        pytest.skip("falta models/servicio: python scripts/exportar_modelo_registry.py")
    with TestClient(crear_app(_ajustes(SERVICIO_MODELO_DIR))) as c:
        yield c


def test_modelo_empaquetado_es_p1_champion(cliente_real):
    info = cliente_real.get("/modelo").json()
    assert info["nombre"] == "mibici-equipo32-p1-demanda" and info["version"] >= 1
    assert "champion" in info["alias"] and info["integridad_verificada"]
    assert info["data_version"] == json.loads((PROJ_ROOT / "data" / "manifest.json")
                                              .read_text(encoding="utf-8"))["data_version"]


def test_api_reproduce_el_modelo_empaquetado(cliente_real):
    """Mismas predicciones que mlflow.pyfunc sobre el input_example registrado."""
    ejemplo = json.loads((SERVICIO_MODELO_DIR / "input_example.json").read_text(encoding="utf-8"))
    x = pd.DataFrame(ejemplo["data"], columns=ejemplo["columns"])
    esperado = mlflow.pyfunc.load_model(str(SERVICIO_MODELO_DIR)).predict(x)
    instancias = [{k: (float(v) if k.startswith("media") else int(v)) for k, v in fila.items()}
                  for fila in x.to_dict(orient="records")]
    r = cliente_real.post("/predict/lote", json={"instancias": instancias})
    assert r.status_code == 200, r.text
    np.testing.assert_allclose([p["viajes_esperados"] for p in r.json()["predicciones"]], esperado,
                               atol=5e-4)


@pytest.mark.datos
@pytest.mark.lento
def test_api_reproduce_test_mae_con_datos_reales(cliente_real):
    """Las 124 mil filas reales pasan la validación y la API reproduce el test_mae registrado."""
    from mibici.config import (
        ESTACIONES_PARQUET,
        P1_FERIADOS,
        P1_HORIZONTE_DIAS,
        P1_LAGS_DIAS,
        P1_TEST_END,
        P1_TEST_START,
        P1_TRAIN_START,
        P1_VENTANAS_MEDIA_DIAS,
        VIAJES_PARQUET,
    )
    from mibici.features import P1_FEATURES, p1_agregar_features, p1_construir_panel_diario
    from mibici.modeling import p1_regresion as p1

    panel = p1_construir_panel_diario(p1.cargar_viajes(VIAJES_PARQUET), P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(panel, p1.cargar_capacidad(ESTACIONES_PARQUET), P1_FERIADOS,
                             P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS)
    df = df.dropna(subset=P1_FEATURES)
    for fila in df[P1_FEATURES].to_dict(orient="records"):
        EntradaP1(**fila)  # ninguna fila real se rechaza
    prueba = p1.cortar(df, P1_TEST_START, P1_TEST_END)
    instancias = [{k: (float(v) if k.startswith("media") else int(v)) for k, v in fila.items()}
                  for fila in prueba[P1_FEATURES].to_dict(orient="records")]
    predicciones = []
    for i in range(0, len(instancias), LOTE_MAXIMO):
        r = cliente_real.post("/predict/lote", json={"instancias": instancias[i:i + LOTE_MAXIMO]})
        assert r.status_code == 200, r.text
        predicciones += [p["viajes_esperados"] for p in r.json()["predicciones"]]
    mae = float(np.mean(np.abs(prueba["viajes"].to_numpy() - np.asarray(predicciones))))
    registrado = json.loads((SERVICIO_MODELO_DIR / "registro.json").read_text(encoding="utf-8"))
    assert mae == pytest.approx(registrado["metricas"]["test_mae"], abs=1e-3)
