"""Reproducibilidad del entrenamiento: misma semilla => mismas predicciones y métricas.

Se usan los constructores y el método ``ajustar_evaluar`` de ``mibici/modeling/tareas.py``,
que son los mismos que ejecuta ``mlflow run . -e train``. La semilla viene de
``config.SEED``. La última prueba (``@pytest.mark.datos``) entrena dos veces con los datos
reales y compara las métricas que quedan registradas en MLflow.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from mibici.config import (
    P1_FERIADOS,
    P1_LAGS_DIAS,
    P1_MODELOS,
    P1_VENTANAS_MEDIA_DIAS,
    P2_MODELOS,
    SEED,
)
from mibici.features import P1_FEATURES, P2_FEATURES, P3_MODEL_FEATURES
from mibici.features import p1_agregar_features, p1_construir_panel_diario
from mibici.modeling.tareas import P1Tarea, P2Tarea, P3Tarea

from .conftest import generar_viajes

# Costos de tiempo: dependen de la carga de la máquina, no del modelo.
NO_DETERMINISTAS = {"costo_entrenamiento_seg", "costo_inferencia_ms_1k"}


def _deterministas(metricas: dict) -> dict:
    return {k: v for k, v in metricas.items() if k not in NO_DETERMINISTAS}


def _mismas_metricas(a: dict, b: dict, rel: float = 0.0) -> None:
    """Mismas llaves y mismos valores (NaN == NaN); ``rel`` > 0 tolera redondeo de punto flotante."""
    a, b = _deterministas(a), _deterministas(b)
    assert a.keys() == b.keys()
    for k in a:
        if np.isnan(a[k]) and np.isnan(b[k]):
            continue
        assert a[k] == pytest.approx(b[k], rel=rel, abs=0), k


@pytest.fixture(scope="module")
def tarea_p1():
    """P1Tarea con datos sintéticos construidos con las funciones reales de variables."""
    viajes = generar_viajes(dias=120, estaciones=6, inicio="2025-01-01")
    capacidad = pd.Series({i: 10 + 2 * i for i in range(1, 7)})
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-05-01")
    df = p1_agregar_features(panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS,
                             P1_VENTANAS_MEDIA_DIAS).dropna(subset=P1_FEATURES)
    fecha = df.index.get_level_values("fecha")
    partes = {"train": df[fecha < "2025-03-15"],
              "val": df[(fecha >= "2025-03-15") & (fecha < "2025-04-01")],
              "test": df[fecha >= "2025-04-01"]}
    tarea = P1Tarea()
    tarea.df = partes
    tarea.x = {k: d[P1_FEATURES].astype("float64") for k, d in partes.items()}
    tarea.y = {k: d["viajes"] for k, d in partes.items()}
    return tarea


@pytest.mark.parametrize("modelo", ["naive", "poisson", "hgb"])
def test_p1_misma_semilla_mismas_metricas(tarea_p1, modelo):
    """Modelos de un solo hilo: resultados idénticos bit a bit."""
    params = dict(P1_MODELOS[modelo]["params"])
    a = tarea_p1.ajustar_evaluar(tarea_p1.construir(modelo, params), "a", con_test=False)
    b = tarea_p1.ajustar_evaluar(tarea_p1.construir(modelo, params), "b", con_test=False)
    _mismas_metricas(a.metricas, b.metricas)
    for k in ("train", "val"):
        np.testing.assert_array_equal(a.extra["pred"][k], b.extra["pred"][k])


def test_p1_bosque_aleatorio_mismos_arboles(tarea_p1):
    """Random forest con ``n_jobs=-1``: con la misma semilla los árboles son idénticos.

    Las predicciones promedian los árboles en paralelo y el orden de la suma puede variar
    entre hilos, así que métricas y predicciones se comparan con tolerancia de 1e-9
    (diferencias del orden de 1e-15, redondeo de punto flotante).
    """
    params = {**P1_MODELOS["rf"]["params"], "n_estimators": 30}
    a = tarea_p1.ajustar_evaluar(tarea_p1.construir("rf", params), "a", con_test=False)
    b = tarea_p1.ajustar_evaluar(tarea_p1.construir("rf", params), "b", con_test=False)
    for ta, tb in zip(a.modelo.estimators_, b.modelo.estimators_):
        np.testing.assert_array_equal(ta.tree_.feature, tb.tree_.feature)
        np.testing.assert_array_equal(ta.tree_.threshold, tb.tree_.threshold)
        np.testing.assert_array_equal(ta.tree_.value, tb.tree_.value)
    _mismas_metricas(a.metricas, b.metricas, rel=1e-9)
    np.testing.assert_allclose(a.extra["pred"]["val"], b.extra["pred"]["val"], rtol=1e-12)


def test_p1_la_semilla_controla_el_resultado(tarea_p1):
    """Control: con otra semilla el bosque aleatorio cambia; la igualdad no es casualidad."""
    params = {**P1_MODELOS["rf"]["params"], "n_estimators": 30}
    x, y = tarea_p1.x["train"], tarea_p1.y["train"]
    pred_semilla = RandomForestRegressor(**params, random_state=SEED).fit(x, y).predict(x)
    pred_otra = RandomForestRegressor(**params, random_state=SEED + 1).fit(x, y).predict(x)
    assert not np.array_equal(pred_semilla, pred_otra)


@pytest.fixture(scope="module")
def tarea_p2():
    rng = np.random.default_rng(SEED)
    partes = {}
    for nombre, inicio, dias in (("train", "2025-01-01", 28), ("val", "2025-10-01", 21)):
        horas = pd.date_range(inicio, periods=24 * dias, freq="h")
        idx = pd.MultiIndex.from_product([[1, 2, 3], horas], names=["estacion", "hora"])
        x = pd.DataFrame(rng.normal(size=(len(idx), len(P2_FEATURES))), index=idx,
                         columns=P2_FEATURES)
        y = ((x["neto_3h"] + rng.normal(scale=0.8, size=len(idx))) > 1.6).astype("int8")
        partes[nombre] = (x, y)
    tarea = P2Tarea()
    tarea.x = {k: v[0] for k, v in partes.items()}
    tarea.y = {k: v[1] for k, v in partes.items()}
    tarea.tipo = {
        k: pd.Series(np.where(v[1] == 1, rng.choice(["sin_bicis", "sin_anclajes"], len(v[1])),
                              "normal"))
        for k, v in partes.items()
    }
    return tarea


def test_p2_misma_semilla_mismo_umbral_y_metricas(tarea_p2):
    params = dict(P2_MODELOS["hgb"]["params"], max_iter=60)
    a = tarea_p2.ajustar_evaluar(tarea_p2.construir("hgb", params), "a", con_test=False)
    b = tarea_p2.ajustar_evaluar(tarea_p2.construir("hgb", params), "b", con_test=False)
    assert a.extra["umbral"] == b.extra["umbral"]
    _mismas_metricas(a.metricas, b.metricas)
    np.testing.assert_array_equal(a.modelo.predict(tarea_p2.x["val"]),
                                  b.modelo.predict(tarea_p2.x["val"]))


@pytest.mark.parametrize(("modelo", "params"), [
    ("kmeans", {"k": 4, "n_init": 5}),
    ("gmm", {"k": 4, "covariance_type": "diag", "n_init": 2}),
    ("agglomerative", {"k": 4, "linkage": "ward"}),
])
def test_p3_misma_semilla_mismos_grupos(modelo, params):
    rng = np.random.default_rng(SEED)
    centros = rng.uniform(0, 1, size=(4, len(P3_MODEL_FEATURES)))
    datos = np.vstack([c + rng.normal(scale=0.05, size=(30, len(P3_MODEL_FEATURES)))
                       for c in centros])
    f = pd.DataFrame(np.abs(datos), columns=P3_MODEL_FEATURES)
    tarea = P3Tarea()
    lab_a = tarea.construir(modelo, params).fit(f)[-1].labels_
    lab_b = tarea.construir(modelo, params).fit(f)[-1].labels_
    np.testing.assert_array_equal(lab_a, lab_b)


@pytest.mark.datos
@pytest.mark.lento
def test_entrenamiento_real_registrado_en_mlflow_es_reproducible(tmp_path, monkeypatch):
    """Dos corridas de ``train`` con los datos reales registran las mismas métricas en MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from mibici.modeling.train import ejecutar, resolver

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.chdir(tmp_path)  # los artefactos quedan en la carpeta temporal
    modelo, params, familia, etapa, cambios = resolver(
        "p1", "rf", {"n_estimators": "20", "max_samples": "0.2"})
    metricas = [ejecutar("p1", modelo, params, familia, etapa, f"repro-{i}", cambios)
                for i in (1, 2)]
    _mismas_metricas(metricas[0], metricas[1], rel=1e-9)

    client = MlflowClient(tracking_uri=uri)
    corridas = client.search_runs(
        [e.experiment_id for e in client.search_experiments()],
        filter_string="tags.mlflow.runName LIKE 'repro-%'")
    assert len(corridas) == 2
    _mismas_metricas(corridas[0].data.metrics, corridas[1].data.metrics, rel=1e-9)
    assert {r.data.params["random_state"] for r in corridas} == {str(SEED)}
    assert len({r.data.tags["data_version"] for r in corridas}) == 1
    mlflow.set_tracking_uri(None)
