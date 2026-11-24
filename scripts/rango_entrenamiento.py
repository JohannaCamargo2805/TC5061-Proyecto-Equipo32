"""Registra el rango de TRAIN de una versión ya registrada de P1 (``data/rango_entrenamiento.json``).

Las corridas nuevas de P1 (``train`` y ``retrain``) ya lo registran solas. Este script lo agrega a
las corridas de versiones anteriores, reconstruyendo su TRAIN con las mismas funciones con que se
entrenaron, para que ``scripts/exportar_modelo_registry.py`` lo copie en ``registro.json`` y la API
advierta la extrapolación respecto de la versión que sirve (y no de la v1)::

    # con el servidor de MLflow y los Parquet de data/processed
    python scripts/rango_entrenamiento.py --version 3
    python scripts/rango_entrenamiento.py --alias champion

Reconstruye el TRAIN según la etapa de la corrida: ``retraining`` (reentrenamiento de la
entrega 5, ene 2025-may 2026) o la partición original de P1 (ene-sep 2025).
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow import MlflowClient  # noqa: E402
import typer  # noqa: E402

from mibici.config import ALIAS_PRODUCCION, REGISTRY_NAMES  # noqa: E402
from mibici.modeling import seguimiento as seg  # noqa: E402

app = typer.Typer(add_completion=False)


def train_de(etapa: str):
    """Variables de TRAIN reconstruidas como en la corrida original."""
    if etapa == "retraining":
        from mibici.modeling.reentrenamiento import P1Reentreno

        tarea = P1Reentreno()
    else:
        from mibici.modeling.tareas import P1Tarea

        tarea = P1Tarea()
    tarea.preparar()
    return tarea.x["train"]


@app.command()
def main(
    version: int = typer.Option(0, help="Versión registrada; 0 = la de --alias"),
    alias: str = typer.Option(ALIAS_PRODUCCION, help="Alias que se resuelve si no se da --version"),
):
    """Calcula el rango de TRAIN de la versión y lo registra en su corrida de origen."""
    seg.configurar_tracking()
    cliente = MlflowClient()
    nombre = REGISTRY_NAMES["p1"]
    mv = (cliente.get_model_version(nombre, str(version)) if version
          else cliente.get_model_version_by_alias(nombre, alias))
    corrida = cliente.get_run(mv.run_id)
    etapa = corrida.data.tags.get("stage_of_work", "")
    x = train_de(etapa)
    rango = seg.registrar_rango_entrenamiento(x, run_id=mv.run_id)
    print(f"{nombre} v{mv.version} (corrida {corrida.data.tags.get('mlflow.runName')}, {etapa}): "
          f"TRAIN {len(x):,} filas -> data/rango_entrenamiento.json")
    print(json.dumps(rango, ensure_ascii=False))


if __name__ == "__main__":
    app()
