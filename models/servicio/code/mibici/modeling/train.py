"""Punto de entrada ``train`` del MLflow Project.

Entrena UN modelo para UNA pregunta y lo registra en la corrida que abre
``mlflow run``. Ejemplos::

    mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300
    mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05
    mlflow run . -e train -P question=p3 -P model=kmeans -P k=4

Los hiperparámetros que no se pasan (valor vacío) se toman del catálogo de
``config.py`` (``P1_MODELOS``, ``P2_MODELOS``, ``P3_MODELOS``); la partición
temporal, las rutas y la semilla también vienen de ``config.py``.
"""

import json
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import typer

from mibici.config import (
    BUSQUEDAS,
    CATALOGO_MODELOS,
    MLFLOW_EXPERIMENT,
    MODELO_DEFECTO,
    P1_OUTPUT_DIR,
    P3_BASELINE,
    REGISTRY_NAMES,
    SEED,
    SELECCION,
)
from mibici.modeling import seguimiento as seg
from mibici.modeling.tareas import TAREAS

app = typer.Typer(add_completion=False)

HIPERPARAMETROS = [
    "n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples",
    "learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization", "alpha", "C",
    "class_weight", "k", "linkage", "covariance_type", "n_init", "lag_dias",
]


def convertir(valor: str):
    """Convierte el texto de ``-P nombre=valor`` al tipo de Python adecuado."""
    texto = str(valor).strip()
    if texto.lower() in {"none", "null"}:
        return None
    if texto.lower() in {"true", "false"}:
        return texto.lower() == "true"
    for tipo in (int, float):
        try:
            return tipo(texto)
        except ValueError:
            continue
    return texto


def resolver(question: str, model: str, overrides: dict, stage_of_work: str = "",
             extras: dict | None = None) -> tuple:
    """Modelo, hiperparámetros finales, familia y etapa a partir del catálogo de config.py.

    Un valor vacío significa "usar el valor de config.py". Para P3, ``stage_of_work=baseline``
    usa la configuración baseline de la entrega anterior (K-Means con k = 2).
    """
    question = question.lower()
    model = "" if seg.es_defecto(model) else model
    stage_of_work = "" if seg.es_defecto(stage_of_work) else stage_of_work
    if question not in CATALOGO_MODELOS:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3 (recibido: {question!r})")
    catalogo = CATALOGO_MODELOS[question]
    if question == "p3" and stage_of_work == "baseline" and not model:
        model = P3_BASELINE["model"]
    model = model or MODELO_DEFECTO[question]
    if model not in catalogo:
        raise typer.BadParameter(f"model para {question} debe ser uno de {sorted(catalogo)}")
    entrada = catalogo[model]
    params = dict(entrada["params"])
    if question == "p3" and stage_of_work == "baseline":
        params.update(P3_BASELINE["params"])
    cambios = {k: convertir(v) for k, v in overrides.items() if not seg.es_defecto(v)}
    desconocidos = set(cambios) - set(params)
    if desconocidos:
        raise typer.BadParameter(
            f"Hiperparámetros {sorted(desconocidos)} no aplican al modelo {model!r} de {question}; "
            f"válidos: {sorted(params)}"
        )
    params.update(cambios)
    if extras:  # hiperparámetros adicionales vía extra_params (JSON), sin validar contra el catálogo
        params.update(extras)
        cambios = {**cambios, **extras}
    etapa = stage_of_work or entrada["etapa"]
    return model, params, entrada["familia"], etapa, cambios


def mejores_de_busqueda(question: str, model: str, referencia: str) -> tuple[dict, str]:
    """Hiperparámetros de la mejor combinación de una búsqueda (corrida padre).

    ``referencia`` es el run_id de la corrida padre o ``latest`` para la búsqueda
    terminada más reciente de esa pregunta y modelo.
    """
    from mlflow import MlflowClient

    seg.configurar_tracking()
    client = MlflowClient()
    if referencia == "latest":
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        padres = client.search_runs(
            [exp.experiment_id],
            filter_string=(f"tags.question = '{question}' and tags.search_role = 'parent' "
                           f"and tags.model = '{model}' and attributes.status = 'FINISHED'"),
            order_by=["attributes.start_time DESC"], max_results=1)
        if not padres:
            raise typer.BadParameter(f"No hay búsquedas terminadas de {question}/{model}")
        referencia = padres[0].info.run_id
    padre = client.get_run(referencia)
    rejilla = json.loads(padre.data.params["grid_resuelta"])
    mejor = client.get_run(padre.data.tags["best_run_id"])
    return {k: convertir(mejor.data.params[f"hp_{k}"]) for k in rejilla}, referencia


def ejecutar(question: str, model: str, params: dict, familia: str, etapa: str, run_name: str,
             cambios: dict, padre_busqueda: str = "") -> dict:
    """Prepara datos, entrena, evalúa y registra en MLflow todo lo que respalda el resultado."""
    seg.configurar_tracking()
    manifiesto = seg.leer_manifiesto()
    P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tarea = TAREAS[question]()
    tarea.preparar(data_version=manifiesto["data_version"])
    estimador = tarea.construir(model, params)
    nombre = run_name or f"{question}-{etapa}-{model}"
    etiquetas = seg.etiquetas_base(question, familia, etapa, manifiesto["data_version"])
    etiquetas["model"] = model
    if padre_busqueda:
        etiquetas["search_parent"] = padre_busqueda

    with seg.corrida(nombre, etiquetas) as run:
        mlflow.log_params({
            "modelo_resuelto": model,
            "estimador": type(estimador).__name__,
            "random_state": SEED,
            "data_version": manifiesto["data_version"],
            "hiperparametros_modificados": json.dumps(cambios, sort_keys=True) or "{}",
            **{f"hp_{k}": v for k, v in params.items()},
            **tarea.particion(),
        })
        res = tarea.ajustar_evaluar(estimador, nombre)
        mlflow.log_metrics(res.metricas)
        criterio = SELECCION[question]
        mlflow.set_tag("metrica_seleccion", criterio["metrica"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for carpeta, rutas in tarea.artefactos(res, nombre, tmp).items():
                seg.registrar_archivos(rutas, carpeta)
            ruta_m = tmp / "metricas.json"
            ruta_m.write_text(json.dumps(res.metricas, indent=2))
            seg.registrar_archivos([ruta_m], "evaluation")

        seg.registrar_version_datos(question, manifiesto, tarea.conjuntos(), tarea.fuente,
                                    tarea.objetivo, tarea.resumen())
        seg.registrar_config({
            "question": question, "model": model, "stage_of_work": etapa, "run_name": nombre,
            "hiperparametros": params, "hiperparametros_modificados": cambios,
            "particion": tarea.particion(), "seleccion": criterio,
            "registry_name": REGISTRY_NAMES[question], "git": seg.git_info(),
            "data_version": manifiesto["data_version"],
        })
        info = seg.registrar_modelo(res.modelo, tarea.ejemplo(), {
            "question": question, "model": model, "data_version": manifiesto["data_version"],
            "git_commit": etiquetas["git_commit"],
        })
        mlflow.set_tag("logged_model_id", info.model_id)
        logger.success(
            f"{nombre} ({run.info.run_id}): {criterio['metrica']}="
            f"{res.metricas.get(criterio['metrica'], float('nan')):.4f}"
        )
    return res.metricas


@app.command()
def main(
    question: str = typer.Option("p1", help="Pregunta: p1, p2 o p3"),
    model: str = typer.Option("config", help="Modelo del catálogo; config = el de config.py"),
    stage: str = typer.Option("config", help="baseline o candidate; config = el del catálogo"),
    run_name: str = typer.Option("auto", help="Nombre de la corrida; auto = <pregunta>-<etapa>-<modelo>"),
    n_estimators: str = "config", max_depth: str = "config", min_samples_leaf: str = "config",
    max_features: str = "config", max_samples: str = "config", learning_rate: str = "config",
    max_iter: str = "config", max_leaf_nodes: str = "config", l2_regularization: str = "config",
    alpha: str = "config", C: str = typer.Option("config", "--C"), class_weight: str = "config",
    k: str = "config", linkage: str = "config", covariance_type: str = "config",
    n_init: str = "config", lag_dias: str = "config",
    extra_params: str = typer.Option("{}", help="JSON con otros hiperparámetros"),
    from_search: str = typer.Option("no", help="run_id de una búsqueda o 'latest': usa su mejor combinación"),
):
    """Entrena y registra un modelo de la pregunta indicada."""
    valores = locals()
    overrides = {h: valores[h] for h in HIPERPARAMETROS}
    extras = {} if seg.es_defecto(extra_params) else json.loads(extra_params)
    run_name = "" if seg.es_defecto(run_name) else run_name
    model = "" if seg.es_defecto(model) else model
    padre = ""
    if not seg.es_defecto(from_search):
        modelo_busqueda = model or BUSQUEDAS[question.lower()]["model"]
        mejores, padre = mejores_de_busqueda(question.lower(), modelo_busqueda, from_search)
        overrides.update({k: str(v) for k, v in mejores.items()})
        model = modelo_busqueda
        run_name = run_name or f"{question.lower()}-candidate-{modelo_busqueda}-tuned"
    modelo, params, familia, etapa, cambios = resolver(question, model, overrides, stage, extras)
    logger.info(f"train: question={question} model={modelo} etapa={etapa} params={params}")
    ejecutar(question.lower(), modelo, params, familia, etapa, run_name, cambios, padre)


if __name__ == "__main__":
    app()
