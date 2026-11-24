"""Pruebas del proyecto final: promoción a producción y rollback (``mibici.modeling.promocion``).

Un Model Registry temporal (SQLite) reproduce la situación real del proyecto: v1 en
``@champion``, v2 sin alias y v3 reentrenada en ``@challenger``, más una corrida de A/B con la
decisión. Se prueba que la promoción solo ocurre si pasan todas las compuertas, que deja el
estado de producción de cada versión y el respaldo para el rollback, y que es idempotente.
"""

import json
import shutil

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import ALIAS_ANTERIOR, ALIAS_PRODUCCION, ALIAS_RETADOR, REGISTRY_NAMES
from mibici.features import P1_FEATURES
from mibici.modeling import promocion as pr

NOMBRE = REGISTRY_NAMES["p1"]
EXPERIMENTO = "prueba-promocion"


def _x(n: int = 120, semilla: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.poisson(20, size=(n, len(P1_FEATURES))).astype("float64"), columns=P1_FEATURES)
    return x, rng.poisson(np.clip(x["media_7d"].to_numpy(), 1, None))


def _version(cliente, experimento, semilla, metricas=None, columnas=None):
    x, y = _x(semilla=semilla)
    if columnas:
        x = x.rename(columns=columnas)
    modelo = Pipeline([("esc", StandardScaler()), ("glm", PoissonRegressor(alpha=1e-3))]).fit(x, y)
    with mlflow.start_run(experiment_id=experimento) as run:
        mlflow.log_metrics(metricas or {"val_mae": 5.0})
        info = mlflow.sklearn.log_model(modelo, name="model", signature=infer_signature(x, modelo.predict(x)))
    mv = cliente.create_model_version(NOMBRE, info.model_uri, run.info.run_id)
    cliente.set_model_version_tag(NOMBRE, mv.version, "question", "p1")
    return int(mv.version)


def _corrida_ab(cliente, experimento, va, vb, promover=True, reglas_ok=True):
    reglas = {"1_ic95_delta_mae_bajo_cero": reglas_ok, "2_sin_empeorar_mas_de_2pct_en_ningun_mes": True,
              "3_sesgo_no_peor": True, "4_costo_inferencia_aceptable": True}
    resultado = {"decision": {"reglas": reglas, "ganadora": "B" if promover else "A", "promover_B": promover},
                 "metricas": {"A": {"mae": 5.87}, "B": {"mae": 5.66}},
                 "bootstrap": {"ic95_delta_mae": [-0.32, -0.12]},
                 "wilcoxon_mae_diario": {"p_valor": 0.0004},
                 "evaluacion": {"desde": "2026-07-01", "hasta_excl": "2026-10-01"}}
    with mlflow.start_run(experiment_id=experimento, run_name=f"p1-ab-test-v{va}-vs-v{vb}") as run:
        mlflow.set_tags({"stage_of_work": "ab_test", "version_A": str(va), "version_B": str(vb),
                         "ganadora": "B" if promover else "A"})
        mlflow.log_metric("promover_B", float(promover))
        mlflow.log_dict(resultado, "ab_test/ab_resultado.json")
    return cliente.get_run(run.info.run_id)


@pytest.fixture(scope="module")
def plantilla(tmp_path_factory):
    """Registry base, creado una sola vez: cada prueba trabaja sobre una copia de su mlflow.db."""
    anterior = mlflow.get_tracking_uri()
    raiz = tmp_path_factory.mktemp("plantilla")
    mlflow.set_tracking_uri(f"sqlite:///{(raiz / 'mlflow.db').as_posix()}")
    cliente = MlflowClient()
    exp = cliente.create_experiment(EXPERIMENTO, artifact_location=(raiz / "artefactos").as_uri())
    cliente.create_registered_model(NOMBRE)
    v1 = _version(cliente, exp, 1)
    v2 = _version(cliente, exp, 2)
    v3 = _version(cliente, exp, 3, {"val_mae": 5.6, "pasa_compuerta": 1.0,
                                    "val_mae_nuevo": 7.27, "val_mae_champion": 7.47})
    cliente.set_registered_model_alias(NOMBRE, ALIAS_PRODUCCION, str(v1))
    cliente.set_registered_model_alias(NOMBRE, ALIAS_RETADOR, str(v3))
    mlflow.set_tracking_uri(anterior)
    return {"db": raiz / "mlflow.db", "exp": exp, "v": (v1, v2, v3)}


@pytest.fixture
def registry(plantilla, tmp_path):
    anterior = mlflow.get_tracking_uri()
    shutil.copyfile(plantilla["db"], tmp_path / "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENTO)
    yield {"cliente": MlflowClient(), "exp": plantilla["exp"], "v": plantilla["v"]}
    mlflow.set_tracking_uri(anterior)


def _aliases(cliente) -> dict[str, int]:
    return {a: int(v) for a, v in cliente.get_registered_model(NOMBRE).aliases.items()}


def test_la_corrida_de_ab_mas_reciente_y_su_decision(registry):
    c, exp = registry["cliente"], registry["exp"]
    _corrida_ab(c, exp, 1, 2, promover=False)
    _corrida_ab(c, exp, 1, 3)
    corrida = pr.ultima_corrida_ab(c, EXPERIMENTO)
    decision = pr.leer_decision(c, corrida)
    assert (decision["version_A"], decision["version_B"], decision["promover_B"]) == (1, 3, True)
    assert all(decision["reglas"].values()) and decision["metricas"]["ic95_delta_mae"] == [-0.32, -0.12]


def test_promueve_y_deja_respaldo_y_estados(registry):
    c = registry["cliente"]
    decision = pr.leer_decision(c, _corrida_ab(c, registry["exp"], 1, 3))
    res = pr.promover(c, decision)
    assert res["promovida"] and res["todas"] and res["version_produccion"] == 3
    assert _aliases(c) == {ALIAS_PRODUCCION: 3, ALIAS_ANTERIOR: 1}  # @challenger ya no apunta a v3
    estados = {v: e["estado"] for v, e in res["estados"].items()}
    assert estados == {1: "respaldo", 2: "archivada", 3: "produccion"}
    assert c.get_model_version(NOMBRE, "3").tags["estado_produccion"] == "produccion"
    assert "A/B" in c.get_model_version(NOMBRE, "3").tags["promovida"]
    assert "reemplazada por v3" in c.get_model_version(NOMBRE, "1").tags["retirada_de_produccion"]
    assert c.get_registered_model(NOMBRE).tags["version_produccion"] == "3"
    # La corrida de la promoción guarda compuertas y plan
    run_id = pr.registrar_corrida(res)
    corrida = c.get_run(run_id)
    assert corrida.data.tags["stage_of_work"] == "promotion" and corrida.data.metrics["promovida"] == 1.0
    assert c.get_model_version(NOMBRE, "3").tags["promocion_run_id"] == run_id


def test_es_idempotente(registry):
    c = registry["cliente"]
    decision = pr.leer_decision(c, _corrida_ab(c, registry["exp"], 1, 3))
    pr.promover(c, decision)
    segunda = pr.promover(c, decision)
    assert segunda["ya_en_produccion"] and not segunda["promovida"]
    assert _aliases(c)[ALIAS_PRODUCCION] == 3


def test_simular_no_cambia_el_registry(registry):
    c = registry["cliente"]
    decision = pr.leer_decision(c, _corrida_ab(c, registry["exp"], 1, 3))
    res = pr.promover(c, decision, simular=True)
    assert res["simulacion"] and not res["promovida"] and len(res["plan"]) == 4
    assert _aliases(c) == {ALIAS_PRODUCCION: 1, ALIAS_RETADOR: 3}


@pytest.mark.parametrize("caso", ["ab_no_promueve", "regla_falla", "decision_obsoleta"])
def test_rechaza_si_falla_una_compuerta(registry, caso):
    c, exp = registry["cliente"], registry["exp"]
    if caso == "decision_obsoleta":  # el A/B comparó v2 contra v3, pero producción es v1
        corrida = _corrida_ab(c, exp, 2, 3)
    else:
        corrida = _corrida_ab(c, exp, 1, 3, promover=caso != "ab_no_promueve",
                              reglas_ok=caso != "regla_falla")
    with pytest.raises(pr.PromocionRechazada):
        pr.promover(c, pr.leer_decision(c, corrida))
    assert _aliases(c) == {ALIAS_PRODUCCION: 1, ALIAS_RETADOR: 3}


def test_rechaza_una_version_sin_el_contrato_de_la_api(registry):
    c, exp = registry["cliente"], registry["exp"]
    v4 = _version(c, exp, 4, columnas={"lag_1": "rezago_1"})
    evaluacion = pr.evaluar_compuertas(c, pr.leer_decision(c, _corrida_ab(c, exp, 1, v4)))
    assert not evaluacion["todas"]
    assert not evaluacion["compuertas"]["3_B_lista_y_con_el_contrato_de_la_API"]["ok"]


def test_rechaza_si_no_paso_su_compuerta_de_validacion(registry):
    c, exp = registry["cliente"], registry["exp"]
    v4 = _version(c, exp, 4, {"pasa_compuerta": 0.0, "val_mae_nuevo": 8.0, "val_mae_champion": 7.5})
    evaluacion = pr.evaluar_compuertas(c, pr.leer_decision(c, _corrida_ab(c, exp, 1, v4)))
    assert not evaluacion["compuertas"]["4_B_paso_su_compuerta_de_validacion"]["ok"]


def test_rollback_en_un_paso(registry):
    c = registry["cliente"]
    pr.promover(c, pr.leer_decision(c, _corrida_ab(c, registry["exp"], 1, 3)))
    res = pr.rollback(c)
    assert res["version_produccion"] == 1
    assert _aliases(c) == {ALIAS_PRODUCCION: 1, ALIAS_ANTERIOR: 3}
    assert {v: e["estado"] for v, e in res["estados"].items()} == {1: "produccion", 2: "archivada",
                                                                   3: "respaldo"}


def test_rollback_sin_respaldo_falla(registry):
    with pytest.raises(pr.PromocionRechazada):
        pr.rollback(registry["cliente"])


def test_estado_segun_aliases():
    assert pr.estado_de(["champion", "anterior"]) == "produccion"
    assert pr.estado_de(["challenger"]) == "retador"
    assert pr.estado_de([]) == "archivada"


def test_entry_point_promote_en_el_mlproject():
    import yaml

    from mibici.config import PROJ_ROOT

    ep = yaml.safe_load((PROJ_ROOT / "MLproject").read_text(encoding="utf-8"))["entry_points"]["promote"]
    assert set(ep["parameters"]) == {"ab_run_id", "accion"}
    assert ep["command"].startswith("python -m mibici.modeling.promocion")
    assert json.dumps(pr.ACCIONES) == json.dumps(["promover", "simular", "rollback"])
