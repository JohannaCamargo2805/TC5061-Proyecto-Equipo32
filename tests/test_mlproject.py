"""Pruebas del MLflow Project: estructura del MLproject, valores por defecto y selección."""

import re

import numpy as np
import pandas as pd
import pytest
import typer
import yaml

from mibici import config
from mibici.config import PROJ_ROOT
from mibici.modeling.estimadores import AglomerativoConCentroides, Log1pColumnas, NaiveEstacional
from mibici.modeling.seleccion import ordenar
from mibici.modeling.train import convertir, resolver


@pytest.fixture(scope="module")
def mlproject():
    return yaml.safe_load((PROJ_ROOT / "MLproject").read_text(encoding="utf-8"))


def test_entry_points_y_entorno(mlproject):
    assert mlproject["python_env"] == "python_env.yaml"
    assert {"ingest", "train", "search", "register", "verify"} <= set(mlproject["entry_points"])
    train = mlproject["entry_points"]["train"]["parameters"]
    assert {"question", "model", "n_estimators", "learning_rate", "k"} <= set(train)


def test_mlproject_no_repite_rutas_fechas_ni_semillas(mlproject):
    """Los valores por defecto vienen de config.py: en el MLproject solo hay palabras clave."""
    patron = re.compile(r"(\d{4}-\d{2}-\d{2})|(/|\\)|(^\d+$)")
    for ep in mlproject["entry_points"].values():
        for nombre, p in ep.get("parameters", {}).items():
            assert not patron.search(str(p.get("default", ""))), (nombre, p)


def test_python_env_igual_a_requirements():
    env = yaml.safe_load((PROJ_ROOT / "python_env.yaml").read_text(encoding="utf-8"))
    req = {
        linea.split("==")[0].lower(): linea.split("==")[1]
        for linea in (PROJ_ROOT / "requirements.txt").read_text().splitlines()
        if "==" in linea
    }
    for dep in env["dependencies"]:
        nombre, version = dep.split("==")
        assert req[nombre.lower()] == version, dep


def test_valores_por_defecto_desde_config():
    modelo, params, _, _, cambios = resolver("p1", "config", {"alpha": "config", "max_iter": '""'})
    assert modelo == config.P1_MODELO_DEFECTO
    assert params["alpha"] == config.P1_LINEAL_ALPHA and cambios == {}
    modelo, params, _, etapa, _ = resolver("p3", "config", {}, stage_of_work="baseline")
    assert (modelo, params["k"], etapa) == ("kmeans", config.P3_BASELINE_K, "baseline")


def test_hiperparametros_se_convierten_y_validan():
    _, params, _, _, cambios = resolver("p1", "rf", {"n_estimators": "300", "max_depth": "None"})
    assert params["n_estimators"] == 300 and params["max_depth"] is None
    assert cambios == {"n_estimators": 300, "max_depth": None}
    assert convertir("0.05") == 0.05 and convertir("balanced") == "balanced"
    with pytest.raises(typer.BadParameter):
        resolver("p1", "poisson", {"n_estimators": "10"})


def test_regla_un_error_estandar_prefiere_el_modelo_simple():
    tabla = pd.DataFrame({
        "run_id": ["rf", "poisson", "hgb"],
        "val_mae": [5.07, 5.10, 5.33],
        "val_mae_se": [0.09, 0.10, 0.11],
        "tamano_modelo_mb": [29.0, 0.2, 2.9],
    })
    ranking = ordenar(tabla, "p1")
    assert list(ranking["run_id"]) == ["poisson", "rf", "hgb"]
    assert not ranking.set_index("run_id").loc["hgb", "dentro_1ee"]


def test_restricciones_de_p3():
    tabla = pd.DataFrame({
        "run_id": ["k2", "k4", "gmm"],
        "train_silhouette": [0.20, 0.19, 0.16],
        "n_grupos": [2, 4, 4],
        "ari_historico_reducido": [0.88, 0.94, 0.85],
        "estabilidad_semillas_ari": [1.0, 0.98, 0.86],
    })
    assert ordenar(tabla, "p3").iloc[0]["run_id"] == "k4"


def test_estimadores_propios():
    x = pd.DataFrame({"salidas_dia": [0.0, 9.0], "otra": [1.0, 2.0], "llegadas_dia": [3.0, 0.0]})
    out = Log1pColumnas().fit(x).transform(x)
    assert list(out.columns) == list(x.columns)
    assert out.loc[1, "salidas_dia"] == pytest.approx(np.log1p(9.0))
    assert NaiveEstacional(7).fit(None).predict(pd.DataFrame({"lag_7": [3, 4]})).tolist() == [3, 4]
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    ag = AglomerativoConCentroides(n_clusters=2).fit(pts)
    assert (ag.predict(pts) == ag.labels_).all()
