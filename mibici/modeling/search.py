"""Punto de entrada ``search``: búsqueda de hiperparámetros organizada como nested runs.

La corrida que abre ``mlflow run`` es la PADRE (``stage_of_work=search``); cada
combinación de la rejilla es una corrida HIJA anidada. Las hijas solo calculan
métricas de TRAIN y VALIDACIÓN (P1, P2) o métricas internas y de estabilidad con
datos de TRAIN (P3): el conjunto de prueba no participa en la selección.

Al terminar, la padre registra la tabla de resultados, una gráfica y el comando
``mlflow run`` para entrenar la mejor combinación como corrida candidata::

    mlflow run . -e search -P question=p1
    mlflow run . -e search -P question=p3 -P grid='{"k": [3, 4, 5]}'
"""

import itertools
import json
from pathlib import Path
import tempfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import typer

from mibici.config import BUSQUEDAS, CATALOGO_MODELOS, P1_OUTPUT_DIR, SEED, SELECCION
from mibici.modeling import seguimiento as seg
from mibici.modeling.seleccion import ordenar
from mibici.modeling.tareas import TAREAS

app = typer.Typer(add_completion=False)


def mejor_combinacion(tabla: pd.DataFrame, question: str) -> pd.Series:
    """Mejor combinación por la métrica de selección (y restricciones de P3), sin regla 1 EE.

    La regla de un error estándar se aplica después, al elegir entre candidatas en el
    Model Registry; aquí interesa la combinación con mejor desempeño en validación.
    """
    return ordenar(tabla, question, un_ee=False).iloc[0]


def grafica_busqueda(tabla: pd.DataFrame, question: str, grid: dict, ruta: Path) -> Path:
    criterio = SELECCION[question]["metrica"]
    claves = list(grid)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    etiquetas = tabla[[f"hp_{c}" for c in claves]].astype(str).agg(" | ".join, axis=1)
    orden = tabla[criterio].sort_values(ascending=SELECCION[question]["orden"] == "min").index
    colores = ["#dc2626" if i == orden[0] else "#2563eb" for i in tabla.index]
    ax.bar(range(len(tabla)), tabla[criterio], color=colores)
    ax.set_xticks(range(len(tabla)), etiquetas, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(criterio)
    ax.set_xlabel(" | ".join(claves))
    ax.set_title(f"{question.upper()} · Búsqueda de hiperparámetros (rojo = mejor por {criterio})")
    ax.grid(axis="y", alpha=0.3)
    if "val_mae" in criterio or "val_pr_auc" in criterio:
        lo, hi = tabla[criterio].min(), tabla[criterio].max()
        ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.3)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


@app.command()
def main(
    question: str = typer.Option("p1", help="Pregunta: p1, p2 o p3"),
    model: str = typer.Option("config", help="Modelo; config = el definido en BUSQUEDAS de config.py"),
    grid: str = typer.Option("config", help="Rejilla JSON; config = la de config.py"),
    run_name: str = typer.Option("auto", help="Nombre de la corrida padre"),
):
    """Ejecuta la búsqueda y registra una corrida padre con una hija por combinación."""
    question = question.lower()
    cfg = BUSQUEDAS[question]
    model = cfg["model"] if seg.es_defecto(model) else model
    rejilla = cfg["grid"] if seg.es_defecto(grid) else json.loads(grid)
    run_name = "" if seg.es_defecto(run_name) else run_name
    entrada = CATALOGO_MODELOS[question][model]
    combinaciones = [dict(zip(rejilla, v)) for v in itertools.product(*rejilla.values())]

    seg.configurar_tracking()
    manifiesto = seg.leer_manifiesto()
    P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tarea = TAREAS[question]()
    tarea.preparar(data_version=manifiesto["data_version"])
    nombre_padre = run_name or f"{question}-search-{model}"
    etiquetas = seg.etiquetas_base(question, entrada["familia"], "search", manifiesto["data_version"])
    etiquetas.update({"model": model, "search_role": "parent"})
    criterio = SELECCION[question]

    with seg.corrida(nombre_padre, etiquetas) as padre:
        mlflow.log_params({
            "modelo_resuelto": model, "random_state": SEED,
            "grid_resuelta": json.dumps(rejilla), "n_combinaciones": len(combinaciones),
            "metrica_seleccion": f"{criterio['orden']}({criterio['metrica']})",
            "restricciones": json.dumps(criterio.get("restricciones", {})),
            "data_version": manifiesto["data_version"], **tarea.particion(),
        })
        filas = []
        for i, combo in enumerate(combinaciones, start=1):
            params = {**entrada["params"], **combo}
            sufijo = "-".join(f"{k}{v}" for k, v in combo.items())
            nombre = f"{nombre_padre}-{i:02d}-{sufijo}"
            etiquetas_hija = {**etiquetas, "search_role": "child", "search_parent": padre.info.run_id,
                              "search_index": str(i)}
            with seg.corrida(nombre, etiquetas_hija, nested=True) as hija:
                mlflow.log_params({"model": model, **{f"hp_{k}": v for k, v in params.items()}})
                res = tarea.ajustar_evaluar(tarea.construir(model, params), nombre, con_test=False)
                mlflow.log_metrics(res.metricas)
                filas.append({"run_id": hija.info.run_id, "run_name": nombre,
                              **{f"hp_{k}": v for k, v in params.items()}, **res.metricas})
            logger.info(f"[{i}/{len(combinaciones)}] {nombre}: {criterio['metrica']}="
                        f"{res.metricas[criterio['metrica']]:.4f}")

        tabla = pd.DataFrame(filas)
        mejor = mejor_combinacion(tabla, question)
        mlflow.log_metrics({f"best_{k}": float(mejor[k]) for k in res.metricas if k in mejor})
        mlflow.set_tags({"best_run_id": mejor["run_id"], "best_run_name": mejor["run_name"]})
        comando = "mlflow run . -e train -P question={} -P model={} {}".format(
            question, model, " ".join(f"-P {k}={mejor[f'hp_{k}']}" for k in rejilla))
        mlflow.set_tag("comando_candidato", comando + f" -P run_name={question}-candidate-{model}-tuned")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ruta_tabla = tmp / "resultados_busqueda.csv"
            tabla.sort_values(criterio["metrica"], ascending=criterio["orden"] == "min").to_csv(
                ruta_tabla, index=False)
            seg.registrar_archivos([ruta_tabla], "evaluation")
            seg.registrar_archivos([grafica_busqueda(tabla, question, rejilla,
                                                     tmp / f"{question}_busqueda.png")], "plots")
        seg.registrar_version_datos(question, manifiesto, tarea.conjuntos(), tarea.fuente,
                                    tarea.objetivo, tarea.resumen())
        seg.registrar_config({"question": question, "model": model, "grid": rejilla,
                              "seleccion": criterio,
                              "git": seg.git_info(), "data_version": manifiesto["data_version"]})
        logger.success(f"Búsqueda {nombre_padre}: mejor={mejor['run_name']} -> {comando}")


if __name__ == "__main__":
    app()
