"""Entrega 5: versión alternativa de P1 reentrenada con datos recientes (Continuous Training).

Entrena el MISMO pipeline de la v1 (Poisson lineal, ``alpha=1e-3``, las 12 variables de
``mibici.features``) con una ventana de datos que ya incluye el periodo posterior al
entrenamiento original, la registra en MLflow y la publica en el Model Registry como
nueva versión de ``mibici-equipo32-p1-demanda`` con el alias ``challenger``::

    python -m mibici.modeling.reentrenamiento          # requiere el servidor de MLflow
    mlflow run . -e retrain --env-manager local

Partición (``config.py``): TRAIN ene 2025-may 2026, VAL jun 2026 (compuerta contra la
v1) y TEST jul-sep 2026, el periodo del monitoreo, que ninguna versión vio y en el que
``mibici.modeling.ab_testing`` compara las dos versiones.
"""

import json
from pathlib import Path
import tempfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import typer

from mibici import monitoreo as mon
from mibici.config import (
    AB_EVALUACION,
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    P1_BASELINE_LAG_DIAS,
    P1_MODELOS,
    REENTRENO_MODELO,
    REENTRENO_RUN_NAME,
    REENTRENO_TRAIN,
    REENTRENO_VAL,
    REGISTRY_NAMES,
    SEED,
    VIAJES_PRODUCCION_PARQUET,
)
from mibici.features import P1_FEATURES
from mibici.modeling import seguimiento as seg
from mibici.modeling.registry import registrar
from mibici.modeling.tareas import P1Tarea

app = typer.Typer(add_completion=False)


class P1Reentreno(P1Tarea):
    """P1 con la ventana de reentrenamiento: mismas variables y métricas que la v1."""

    fuente = VIAJES_PRODUCCION_PARQUET

    def preparar(self, data_version: str | None = None) -> None:
        panel = mon.construir_panel(fin=AB_EVALUACION[1])
        self.df = {"train": mon.periodo(panel, *REENTRENO_TRAIN),
                   "val": mon.periodo(panel, *REENTRENO_VAL),
                   "test": mon.periodo(panel, *AB_EVALUACION)}
        self.x = {k: d[P1_FEATURES].astype("float64") for k, d in self.df.items()}
        self.y = {k: d["viajes"] for k, d in self.df.items()}
        self.base_test = self.df["test"][f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64")
        for k, d in self.df.items():
            logger.info(f"P1 reentreno {k}: {len(d):,} estación-días, "
                        f"{d.index.get_level_values('estacion').nunique()} estaciones")

    def _metricas_estabilidad(self, pred_test: np.ndarray) -> dict:
        d = pd.DataFrame({"y": self.y["test"].to_numpy(), "p": np.clip(pred_test, 0, None)},
                         index=self.y["test"].index)
        fecha = d.index.get_level_values("fecha")
        err = (d["y"] - d["p"]).abs()
        por_mes = {f"test_mae_{p.year}_{p.month:02d}": float(v)
                   for p, v in err.groupby(fecha.to_period("M")).mean().items()}
        return {**por_mes, "test_mae_semanal_std": float(err.groupby(fecha.to_period("W")).mean().std())}

    def artefactos(self, res, nombre: str, carpeta: Path) -> dict[str, list[Path]]:
        pred = np.clip(res.extra["pred"]["test"], 0, None)
        test = self.df["test"]
        total = pd.DataFrame({"real": test["viajes"].to_numpy(), "pred": pred,
                              "base": self.base_test}, index=test.index).groupby(level="fecha").sum()
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(total.index, total["real"], color=mon.OSCURO, lw=2, label="Real")
        ax.plot(total.index, total["base"], color=mon.GRIS, lw=1.3, ls="--", label="Baseline (t-7 días)")
        ax.plot(total.index, total["pred"], color=mon.AZUL, lw=1.6, label=nombre)
        ax.set_title("P1 reentrenado · viajes totales por día en TEST (jul-sep 2026)")
        ax.set_ylabel("Viajes iniciados")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        ruta_serie = carpeta / "p1_reentreno_total_diario_test.png"
        fig.savefig(ruta_serie, dpi=150, bbox_inches="tight")
        plt.close(fig)
        ruta_pred = carpeta / "p1_reentreno_predicciones_test.csv"
        test[["viajes"]].assign(pred=pred).reset_index().to_csv(ruta_pred, index=False)
        return {"plots": [ruta_serie], "evaluation": [ruta_pred]}

    def particion(self) -> dict:
        return {"train_inicio": REENTRENO_TRAIN[0], "val_inicio": REENTRENO_VAL[0],
                "test_inicio": AB_EVALUACION[0], "test_fin": AB_EVALUACION[1],
                "horizonte_dias": 1, "features": len(P1_FEATURES)}


def data_version_combinada() -> tuple[str, dict]:
    """``<data_version 2025>+<data_version 2026>`` y el manifiesto combinado."""
    m25, m26 = seg.leer_manifiesto(), mon.leer_manifiesto_produccion()
    combinado = {"data_version": f"{m25['data_version']}+{m26['data_version']}",
                 "archivos": {**m25["archivos"], **m26["archivos"]},
                 "total_viajes": m25["total_viajes"] + m26["total_viajes"],
                 "manifiestos": {"2025": m25["data_version"], "2026": m26["data_version"]}}
    return combinado["data_version"], combinado


def compuerta_val(tarea: P1Reentreno, nuevo, client: MlflowClient) -> dict:
    """MAE de VAL (jun 2026) del modelo nuevo contra el @champion vigente."""
    campeon = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAMES['p1']}@{ALIAS_PRODUCCION}")
    version = client.get_model_version_by_alias(REGISTRY_NAMES["p1"], ALIAS_PRODUCCION).version
    y = tarea.y["val"]
    mae = {n: float(np.abs(y - np.clip(m.predict(tarea.x["val"]), 0, None)).mean())
           for n, m in (("nuevo", nuevo), ("champion", campeon))}
    return {"val_mae_champion": mae["champion"], "val_mae_nuevo": mae["nuevo"],
            "champion_version": int(version), "pasa_compuerta": mae["nuevo"] < mae["champion"]}


def ejecutar(alias: str = ALIAS_RETADOR, registrar_version: bool = True) -> dict:
    seg.configurar_tracking()
    client = MlflowClient()
    data_version, manifiesto = data_version_combinada()
    tarea = P1Reentreno()
    tarea.preparar()
    params = dict(P1_MODELOS[REENTRENO_MODELO]["params"])
    estimador = tarea.construir(REENTRENO_MODELO, params)
    etiquetas = seg.etiquetas_base("p1", P1_MODELOS[REENTRENO_MODELO]["familia"], "retraining",
                                   data_version)
    etiquetas.update({"model": REENTRENO_MODELO, "entrega": "monitoreo-ab",
                      "motivo": "drift detectado por Evidently (entrega 5)"})
    with seg.corrida(REENTRENO_RUN_NAME, etiquetas) as run:
        mlflow.log_params({"modelo_resuelto": REENTRENO_MODELO, "estimador": type(estimador).__name__,
                           "random_state": SEED, "data_version": data_version,
                           "hiperparametros_modificados": "{}",
                           **{f"hp_{k}": v for k, v in params.items()}, **tarea.particion(),
                           "cambio_respecto_a_v1": "solo la ventana de datos (mismo pipeline e hiperparametros)"})
        res = tarea.ajustar_evaluar(estimador, REENTRENO_RUN_NAME)
        compuerta = compuerta_val(tarea, res.modelo, client)
        mlflow.log_metrics({**res.metricas, **{k: float(v) for k, v in compuerta.items()}})
        mlflow.set_tag("metrica_seleccion", "val_mae")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for carpeta, rutas in tarea.artefactos(res, REENTRENO_RUN_NAME, tmp).items():
                seg.registrar_archivos(rutas, carpeta)
            ruta_m = tmp / "metricas.json"
            ruta_m.write_text(json.dumps({**res.metricas, **compuerta}, indent=2, default=float))
            seg.registrar_archivos([ruta_m], "evaluation")
        seg.registrar_version_datos("p1", manifiesto, tarea.conjuntos(), tarea.fuente,
                                    tarea.objetivo, tarea.resumen())
        seg.registrar_rango_entrenamiento(tarea.x["train"])  # advertencias de la API
        seg.registrar_config({"question": "p1", "model": REENTRENO_MODELO, "stage_of_work": "retraining",
                              "run_name": REENTRENO_RUN_NAME, "hiperparametros": params,
                              "particion": tarea.particion(), "git": seg.git_info(),
                              "data_version": data_version, "compuerta_val": compuerta})
        info = seg.registrar_modelo(res.modelo, tarea.ejemplo(), {
            "question": "p1", "model": REENTRENO_MODELO, "data_version": data_version,
            "git_commit": etiquetas["git_commit"]})
        mlflow.set_tag("logged_model_id", info.model_id)
        run_id = run.info.run_id
    logger.info(f"Compuerta VAL (jun 2026): nuevo={compuerta['val_mae_nuevo']:.4f} "
                f"champion v{compuerta['champion_version']}={compuerta['val_mae_champion']:.4f}")
    resultado = {"run_id": run_id, "metricas": res.metricas, "compuerta": compuerta}
    if registrar_version:
        anterior = None
        try:
            anterior = client.get_model_version_by_alias(REGISTRY_NAMES["p1"], alias)
        except mlflow.exceptions.MlflowException:
            pass
        resultado["registro"] = registrar(client, "p1", run_id, alias)
        if anterior is not None and int(anterior.version) != resultado["registro"]["version"]:
            client.set_model_version_tag(REGISTRY_NAMES["p1"], anterior.version, "alias_anterior",
                                         f"{alias} hasta la entrega 5")
        client.set_model_version_tag(REGISTRY_NAMES["p1"], resultado["registro"]["version"],
                                     "origen", "reentrenamiento por drift (entrega 5)")
    logger.success(json.dumps({k: v for k, v in resultado.items() if k != "metricas"},
                              default=str, ensure_ascii=False))
    return resultado


@app.command()
def main(
    alias: str = typer.Option(ALIAS_RETADOR, help="Alias de la nueva versión en el registry"),
    registrar_version: bool = typer.Option(True, "--registrar/--sin-registrar"),
):
    """Reentrena P1 con datos hasta may 2026 y lo registra como nueva versión (challenger)."""
    ejecutar(alias, registrar_version)


if __name__ == "__main__":
    app()
