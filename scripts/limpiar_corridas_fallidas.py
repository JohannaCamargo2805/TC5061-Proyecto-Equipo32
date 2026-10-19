"""Archiva (borrado lógico) las corridas fallidas o lanzadas con el árbol de Git sucio.

Solo afecta corridas del MLflow Project (las que tienen ``mlflow.project.entryPoint``)
en el experimento ``mibici-equipo32``. El borrado es lógico: las corridas quedan en
el filtro *State: Deleted* de la interfaz y pueden restaurarse con
``mlflow runs restore --run-id <id>``. Así el experimento oficial conserva solo
corridas reproducibles desde su ``git_commit``::

    python scripts/limpiar_corridas_fallidas.py
"""

from mlflow import MlflowClient

from mibici.config import ENTREGA, MLFLOW_EXPERIMENT
from mibici.modeling.seguimiento import configurar_tracking


def main() -> None:
    configurar_tracking()
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        return
    n = 0
    for run in client.search_runs([exp.experiment_id], max_results=5000):
        t = run.data.tags
        if "mlflow.project.entryPoint" not in t:
            continue
        fallida = run.info.status in ("FAILED", "KILLED")
        sucia = t.get("entrega") == ENTREGA and t.get("git_sucio") == "True"
        if fallida or sucia:
            client.delete_run(run.info.run_id)
            n += 1
            print(f"archivada {run.info.run_id} {t.get('mlflow.runName', '')} "
                  f"({'fallida' if fallida else 'árbol sucio'})")
    print(f"Corridas archivadas: {n}")


if __name__ == "__main__":
    main()
