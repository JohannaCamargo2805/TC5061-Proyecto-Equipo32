"""Agrega las etiquetas de la entrega 2 a las corridas de la Propuesta de proyecto.

Las corridas de la entrega anterior (experimento ``mibici-equipo32``) solo tenían
``problema`` y ``etapa``. Para compararlas en la interfaz junto con las nuevas se
les agregan ``question``, ``model_family``, ``stage_of_work`` y ``entrega=propuesta``.
No se cambia ninguna métrica, parámetro ni artefacto. Es idempotente::

    python scripts/etiquetar_corridas_propuesta.py
"""

import mlflow
from mlflow import MlflowClient

from mibici.config import ENTREGA_ANTERIOR, MLFLOW_EXPERIMENT
from mibici.modeling.seguimiento import configurar_tracking

FAMILIAS = {
    "p1_baseline_naive_semanal": "naive_estacional",
    "p1_candidato_poisson_lineal": "glm_poisson",
    "p1_candidato_hgb_poisson": "gradient_boosting",
    "p2_baseline_logreg": "lineal_logistica",
    "p2_candidato_hgb": "gradient_boosting",
    "p3_baseline_kmeans_k2": "particional_centroides",
    "p3_candidato_agglomerative_k4": "jerarquico_aglomerativo",
    "p3_kmeans_final": "particional_centroides",
}


def main() -> None:
    configurar_tracking()
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        print("El experimento aún no existe: nada que etiquetar.")
        return
    n = 0
    for run in client.search_runs([exp.experiment_id], max_results=5000):
        t = run.data.tags
        nombre = t.get("mlflow.runName", "")
        if nombre not in FAMILIAS or t.get("entrega") not in (None, ENTREGA_ANTERIOR):
            continue
        etapa = "baseline" if t.get("etapa") == "baseline" else "propuesta"
        for k, v in {"question": t["problema"].lower(), "model_family": FAMILIAS[nombre],
                     "stage_of_work": etapa, "entrega": ENTREGA_ANTERIOR}.items():
            client.set_tag(run.info.run_id, k, v)
        n += 1
        print(f"{nombre}: stage_of_work={etapa}")
    print(f"Corridas de la propuesta etiquetadas: {n} (tracking: {mlflow.get_tracking_uri()})")


if __name__ == "__main__":
    main()
