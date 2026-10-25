"""Utilidades comunes de MLflow Tracking para las corridas del MLflow Project.

Centraliza lo que toda corrida candidata debe registrar:

* etiquetas para filtrar (``question``, ``model_family``, ``stage_of_work``,
  ``equipo``, ``git_commit``, ``data_version``, ``entrega``);
* la versión de los datos (manifiesto como artefacto y conjuntos con
  ``mlflow.log_input``);
* los archivos de configuración con que se ejecutó la corrida;
* el modelo con firma (signature) y ejemplo de entrada.

Los artefactos se organizan en subcarpetas: ``config/``, ``data/``, ``plots/``
y ``evaluation/``; el modelo se registra como *logged model* ``model``.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from loguru import logger
import mlflow
import pandas as pd
import skops.io as sio

from mibici.config import (
    ENTREGA,
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PREGUNTAS,
    PROJ_ROOT,
    TAREAS,
    TEAM_ID,
)

# Valores de los parámetros del MLproject que significan "usar el valor de config.py" o
# "decidir automáticamente". Se aceptan también las comillas vacías que produce cmd en Windows.
SIN_VALOR = {"", "config", "auto", "no", '""', "''"}


def es_defecto(valor) -> bool:
    """True si el parámetro de ``mlflow run`` no trae un valor explícito."""
    return valor is None or str(valor).strip().lower() in SIN_VALOR


RESPONSABLES = {"p1": "Johanna Camargo", "p2": "Manuel Cardona", "p3": "Gerardo Conde"}
ARCHIVOS_CONFIG = ["mibici/config.py", "MLproject", "python_env.yaml", "requirements.txt"]


def configurar_tracking() -> None:
    """Usa el servidor de tracking de ``MLFLOW_TRACKING_URI`` y el experimento del equipo.

    Dentro de ``mlflow run`` el experimento ya viene fijado por el proyecto; fuera
    de él (por ejemplo, en pruebas) se selecciona ``mibici-equipo32``.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    if not os.getenv("MLFLOW_RUN_ID") and not os.getenv("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(MLFLOW_EXPERIMENT)


def git_info() -> dict:
    """Commit actual, rama y si el árbol de trabajo tiene cambios sin confirmar."""
    try:
        def _git(*args):
            return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()

        commit = _git("rev-parse", "HEAD")
        rama = _git("rev-parse", "--abbrev-ref", "HEAD")
        sucio = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "desconocido", "git_rama": "desconocida", "git_sucio": "True"}
    if sucio:
        logger.warning("Hay cambios sin commit: la corrida no será reproducible desde git_commit")
    return {"git_commit": commit, "git_rama": rama, "git_sucio": str(sucio)}


def leer_manifiesto() -> dict:
    """Manifiesto de la ingesta (``data/manifest.json``) con la ``data_version``."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No existe data/manifest.json: ejecuta primero `mlflow run . -e ingest`."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def etiquetas_base(question: str, model_family: str, stage_of_work: str, data_version: str) -> dict:
    """Etiquetas comunes de todas las corridas de la entrega (incluye las obligatorias)."""
    return {
        "question": question,
        "model_family": model_family,
        "stage_of_work": stage_of_work,
        "entrega": ENTREGA,
        "equipo": TEAM_ID,
        "problema": question.upper(),  # etiqueta de la entrega anterior (compatibilidad)
        "tipo": TAREAS[question],
        "pregunta": PREGUNTAS[question],
        "responsable": RESPONSABLES[question],
        "data_version": data_version,
        **git_info(),
    }


@contextmanager
def corrida(run_name: str, etiquetas: dict, nested: bool = False):
    """Abre la corrida de ``mlflow run`` (o una hija si ``nested``) y le pone nombre y etiquetas."""
    if nested:
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.set_tags(etiquetas)
            yield run
        return
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tags(etiquetas)
        yield run


def registrar_config(run_config: dict) -> None:
    """Guarda en ``config/`` los archivos de configuración y los parámetros resueltos."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for rel in ARCHIVOS_CONFIG:
            origen = PROJ_ROOT / rel
            if origen.exists():
                shutil.copy(origen, tmp / origen.name)
        (tmp / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        mlflow.log_artifacts(str(tmp), artifact_path="config")


def registrar_version_datos(question: str, manifiesto: dict, conjuntos: dict, fuente: Path,
                            objetivo: str | None, resumen: dict) -> None:
    """Versión de los datos: manifiesto y resumen en ``data/`` y conjuntos con ``log_input``.

    ``conjuntos`` mapea el contexto (``training``, ``validation``, ``testing``) a un
    DataFrame. Solo se registran metadatos (esquema, número de filas y huella
    digital), nunca los registros, y ninguna columna contiene datos personales.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "manifest.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
        (tmp / "resumen_conjuntos.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        mlflow.log_artifacts(str(tmp), artifact_path="data")
    for contexto, df in conjuntos.items():
        dataset = mlflow.data.from_pandas(
            df, source=str(fuente.relative_to(PROJ_ROOT)), targets=objetivo,
            name=f"{question}_{contexto}",
        )
        mlflow.log_input(dataset, context=contexto, tags={"data_version": manifiesto["data_version"]})


def registrar_archivos(rutas: list[Path], carpeta: str) -> None:
    """Sube cada archivo a la subcarpeta ``carpeta`` de los artefactos de la corrida."""
    for ruta in rutas:
        mlflow.log_artifact(str(ruta), artifact_path=carpeta)


def tamano_modelo_mb(modelo) -> float:
    """Tamaño del modelo serializado con skops (MB): costo de almacenamiento y carga."""
    return len(sio.dumps(modelo)) / 1e6


def registrar_modelo(modelo, ejemplo: pd.DataFrame, metadatos: dict):
    """Registra el modelo con firma inferida a partir del ejemplo de entrada.

    Se serializa con skops (formato por defecto de MLflow 3); los tipos propios del
    proyecto se declaran como confiables porque los entrenamos nosotros, y el
    paquete ``mibici`` viaja con el modelo (``code_paths``).
    """
    confiables = sio.get_untrusted_types(data=sio.dumps(modelo))
    return mlflow.sklearn.log_model(
        modelo,
        name="model",
        input_example=ejemplo,
        skops_trusted_types=confiables,
        code_paths=[str(PROJ_ROOT / "mibici")],
        metadata=metadatos,
    )


class Cronometro:
    """Mide segundos transcurridos: ``with Cronometro() as c: ...; c.segundos``."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self._t0
        return False


def costo_inferencia_ms_1k(modelo, x: pd.DataFrame, metodo: str = "predict") -> float:
    """Milisegundos por cada 1,000 filas al predecir sobre ``x`` (costo de inferencia)."""
    with Cronometro() as c:
        getattr(modelo, metodo)(x)
    return 1000 * c.segundos / max(len(x), 1) * 1000
