"""Punto de entrada ``verify``: carga el modelo en producción por nombre y alias.

Para cada pregunta carga ``models:/<nombre>@champion`` (sin referirse a la
corrida), predice sobre el conjunto de prueba y comprueba que la métrica coincide
con la que registró la corrida de origen. Deja la evidencia en una corrida::

    mlflow run . -e verify
    mlflow run . -e verify -P question=p2 -P alias=challenger
"""

import json

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score
import typer

from mibici.config import ALIAS_PRODUCCION, REGISTRY_NAMES
from mibici.modeling import seguimiento as seg
from mibici.modeling.tareas import TAREAS

app = typer.Typer(add_completion=False)

METRICA = {"p1": "test_mae", "p2": "test_f1", "p3": "test_silhouette"}


def verificar(client: MlflowClient, question: str, alias: str, data_version: str) -> dict:
    nombre = REGISTRY_NAMES[question]
    uri = f"models:/{nombre}@{alias}"
    version = client.get_model_version_by_alias(nombre, alias)
    original = client.get_run(version.run_id).data.metrics[METRICA[question]]
    tarea = TAREAS[question]()
    tarea.preparar(data_version=data_version)
    modelo = mlflow.pyfunc.load_model(uri)  # solo nombre + alias
    if question == "p1":
        pred = np.clip(np.asarray(modelo.predict(tarea.x["test"])), 0, None)
        valor = mean_absolute_error(tarea.y["test"], pred)
        muestra = tarea.x["test"].head(10).assign(prediccion=pred[:10])
    elif question == "p2":
        pred = np.asarray(modelo.predict(tarea.x["test"])).astype(int)
        valor = f1_score(tarea.y["test"], pred)
        muestra = tarea.x["test"].head(10).assign(prediccion=pred[:10])
    else:
        pred = np.asarray(modelo.predict(tarea.f_test)).astype(int)
        sk = mlflow.sklearn.load_model(uri)
        valor = silhouette_score(sk[:-1].transform(tarea.f_test), pred)
        muestra = tarea.f_test.head(10).assign(grupo=pred[:10])
    coincide = bool(np.isclose(valor, original, rtol=1e-6, atol=1e-9))
    logger.info(f"{uri} (v{version.version}): {METRICA[question]}={valor:.6f} "
                f"registrado={original:.6f} coincide={coincide}")
    return {"question": question, "uri": uri, "version": int(version.version),
            "run_id": version.run_id, "metrica": METRICA[question], "valor_recalculado": valor,
            "valor_registrado": original, "coincide": coincide,
            "muestra": muestra.reset_index(drop=True).to_dict(orient="records")}


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all"),
    alias: str = typer.Option(ALIAS_PRODUCCION, help="Alias a cargar (champion o challenger)"),
):
    """Carga cada modelo por nombre y alias y verifica su métrica de prueba."""
    seg.configurar_tracking()
    client = MlflowClient()
    manifiesto = seg.leer_manifiesto()
    preguntas = ["p1", "p2", "p3"] if question == "all" else [question.lower()]
    etiquetas = seg.etiquetas_base("p1", "verificacion", "verify", manifiesto["data_version"])
    etiquetas.update({"question": question, "problema": question.upper(),
                      "pregunta": "Verificación de carga por nombre y alias de los modelos en producción"})
    with seg.corrida(f"verify-{question}-{alias}", etiquetas):
        resultados = []
        for q in preguntas:
            r = verificar(client, q, alias, manifiesto["data_version"])
            resultados.append(r)
            mlflow.log_metrics({f"{q}_{r['metrica']}_recalculado": r["valor_recalculado"],
                                f"{q}_coincide": float(r["coincide"]),
                                f"{q}_version": r["version"]})
        mlflow.log_dict(json.loads(pd.Series(resultados).to_json(orient="values")),
                        "evaluation/verificacion_registry.json")
        if not all(r["coincide"] for r in resultados):
            raise SystemExit("La métrica recalculada no coincide con la registrada")
        logger.success("Todos los modelos en producción cargan por nombre y alias y reproducen su métrica")


if __name__ == "__main__":
    app()
