"""Punto de entrada ``register``: Model Registry con el esquema de aliases.

Registra la corrida elegida de una pregunta como nueva versión del modelo
``mibici-equipo32-<pN>-<nombre>`` y le asigna el alias ``champion`` (producción).
Opcionalmente asigna ``challenger`` a la siguiente mejor corrida. Cada versión
conserva su linaje (experimento, corrida de origen, commit y versión de datos)
como etiquetas y en su descripción::

    mlflow run . -e register -P question=p1 -P run_id=<id> -P challenger_run_id=<id>

Si ``run_id`` va vacío se elige automáticamente la mejor corrida candidata de la
pregunta según la métrica de selección de ``config.py`` (calculada con VAL).
"""

import json

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    ENTREGA,
    MLFLOW_EXPERIMENT,
    PREGUNTAS,
    REGISTRY_NAMES,
    SELECCION,
    TAREAS,
)
from mibici.modeling import seguimiento as seg
from mibici.modeling.seleccion import ordenar

app = typer.Typer(add_completion=False)

DESCRIPCIONES = {
    "p1": ("Regresión de viajes iniciados por estación para el día siguiente (MiBici, Guadalajara). "
           "Usuario: operador del sistema para planear flota y turnos. Entrada: 12 variables "
           "conocidas al cierre del día anterior (rezagos, medias móviles, calendario, capacidad). "
           "Salida: viajes esperados por estación. Métrica de negocio: MAE y WAPE."),
    "p2": ("Clasificación del riesgo de que una estación se quede sin bicicletas o sin anclajes en "
           "las próximas 2 horas. Usuario: equipo de rebalanceo. Entrada: 15 variables disponibles "
           "en la hora de decisión. Salida: 1 = riesgo (el umbral de decisión, elegido en "
           "validación, vive dentro del modelo). Métricas: PR-AUC, F1, recall y precisión."),
    "p3": ("Segmentación de estaciones por patrón de uso (intensidad, balance, fin de semana y "
           "bloques horarios). Usuario: planeación urbana y expansión. Entrada: 12 variables de "
           "perfil por estación. Salida: grupo de la estación. Métricas: silueta, Davies-Bouldin "
           "y estabilidad temporal (ARI)."),
}


def candidatas(client: MlflowClient, question: str) -> pd.DataFrame:
    """Corridas candidatas terminadas de la pregunta, ordenadas con la regla de config.py."""
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(f"tags.question = '{question}' and tags.stage_of_work = 'candidate' "
                       f"and tags.entrega = '{ENTREGA}' and attributes.status = 'FINISHED'"),
    )
    if not runs:
        return pd.DataFrame()
    tabla = pd.DataFrame([{"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName"),
                           "model": r.data.tags.get("model"), **r.data.metrics} for r in runs])
    return ordenar(tabla, question)


def version_de_corrida(client: MlflowClient, nombre: str, run) -> object:
    """Devuelve la versión ya registrada para la corrida o crea una nueva."""
    for v in client.search_model_versions(f"name = '{nombre}'"):
        if v.run_id == run.info.run_id:
            return v
    try:
        client.get_registered_model(nombre)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(nombre)
    model_id = run.data.tags.get("logged_model_id")
    fuente = f"models:/{model_id}" if model_id else f"runs:/{run.info.run_id}/model"
    return mlflow.register_model(fuente, nombre)


def linaje(client: MlflowClient, run, exp_name: str) -> dict:
    t, m = run.data.tags, run.data.metrics
    criterio = SELECCION[t["question"]]["metrica"]
    return {
        "experimento": exp_name,
        "source_run_id": run.info.run_id,
        "source_run_name": t.get("mlflow.runName", ""),
        "git_commit": t.get("mlflow.source.git.commit", t.get("git_commit", "")),
        "git_rama": t.get("git_rama", ""),
        "data_version": t.get("data_version", ""),
        "mlproject_entry_point": t.get("mlflow.project.entryPoint", ""),
        "model_family": t.get("model_family", ""),
        "model": t.get("model", ""),
        "question": t.get("question", ""),
        "metrica_seleccion": f"{criterio}={m.get(criterio, float('nan')):.4f}",
    }


def registrar(client: MlflowClient, question: str, run_id: str, alias: str) -> dict:
    nombre = REGISTRY_NAMES[question]
    run = client.get_run(run_id)
    if run.data.tags.get("question") != question:
        raise typer.BadParameter(f"La corrida {run_id} no es de {question}")
    version = version_de_corrida(client, nombre, run)
    lin = linaje(client, run, MLFLOW_EXPERIMENT)
    client.update_registered_model(nombre, description=DESCRIPCIONES[question])
    for k, v in {"question": question, "tarea": TAREAS[question], "equipo": "equipo32",
                 "pregunta": PREGUNTAS[question], "esquema_ciclo_vida": "aliases"}.items():
        client.set_registered_model_tag(nombre, k, v)
    for k, v in lin.items():
        client.set_model_version_tag(nombre, version.version, k, v)
    metricas = {k: round(v, 4) for k, v in run.data.metrics.items()
                if k.startswith(("val_", "test_")) and not k.endswith("prevalencia")}
    client.update_model_version(
        nombre, version.version,
        description=(f"Versión '{alias}' de {question.upper()}. Corrida de origen "
                     f"{lin['source_run_name']} ({run_id}) en el experimento {MLFLOW_EXPERIMENT}; "
                     f"commit {lin['git_commit'][:7]}; data_version {lin['data_version']}; "
                     f"modelo {lin['model']} ({lin['model_family']}). {lin['metrica_seleccion']}. "
                     f"Métricas: {json.dumps(metricas, ensure_ascii=False)}"),
    )
    client.set_registered_model_alias(nombre, alias, version.version)
    client.set_tag(run_id, f"registry_{alias}", f"{nombre} v{version.version}")
    logger.success(f"{nombre} v{version.version} -> @{alias} (corrida {run_id})")
    return {"modelo": nombre, "version": int(version.version), "alias": alias, "run_id": run_id,
            "uri": f"models:/{nombre}@{alias}"}


@app.command()
def main(
    question: str = typer.Option("p1", help="Pregunta: p1, p2 o p3"),
    run_id: str = typer.Option("", help="Corrida que pasa a producción (champion)"),
    challenger_run_id: str = typer.Option("", help="Corrida retadora (challenger), opcional"),
):
    """Registra el modelo de la pregunta y lo lleva a producción con el alias champion."""
    question = question.lower()
    seg.configurar_tracking()
    client = MlflowClient()
    ranking = candidatas(client, question)
    if not run_id:
        if ranking.empty:
            raise typer.BadParameter(f"No hay corridas candidatas de {question}")
        run_id = ranking.iloc[0]["run_id"]
    if not challenger_run_id and not ranking.empty:
        resto = ranking.loc[ranking["run_id"] != run_id, "run_id"]
        challenger_run_id = resto.iloc[0] if len(resto) else ""
    manifiesto = seg.leer_manifiesto()
    etiquetas = seg.etiquetas_base(question, "registry", "registry", manifiesto["data_version"])
    with seg.corrida(f"{question}-registry", etiquetas):
        resultado = [registrar(client, question, run_id, ALIAS_PRODUCCION)]
        if challenger_run_id:
            resultado.append(registrar(client, question, challenger_run_id, ALIAS_RETADOR))
        mlflow.log_params({"registered_model": REGISTRY_NAMES[question],
                           "champion_resuelto": run_id, "challenger_resuelto": challenger_run_id,
                           "esquema": "aliases (champion/challenger)"})
        criterio = SELECCION[question]
        cols = [c for c in ["run_id", "run_name", "model", criterio["metrica"],
                            criterio.get("error_estandar"), criterio.get("complejidad"),
                            "cumple_restricciones", "dentro_1ee", "motivo"] if c and c in ranking]
        mlflow.log_dict({"versiones": resultado, "criterio": criterio,
                         "ranking": json.loads(ranking[cols].to_json(orient="records"))},
                        "registry/registro.json")
        mlflow.log_text(ranking[cols].to_csv(index=False), "registry/ranking_candidatas.csv")


if __name__ == "__main__":
    app()
