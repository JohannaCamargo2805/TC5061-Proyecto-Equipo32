"""Registra en el MLflow de Docker Compose la versión de P1 empaquetada en ``models/servicio``.

Es el servicio ``registrar-modelo`` de ``docker-compose.yml``: un trabajo de una sola vez que corre
cuando el servidor ``mlflow`` ya está sano y antes de que arranquen la API y el worker, que
descargan el modelo del registry por alias. Así el sistema se levanta completo con un solo
``docker compose up`` en cualquier máquina, sin depender del registry local de la autora (sus
artefactos son rutas de Windows que no existen dentro de un contenedor).

Pasos:

1. Verifica el SHA-256 de cada archivo del modelo contra ``registro.json`` (las mismas huellas que
   comprueba la API): se registra exactamente la versión que se descargó del Model Registry del
   proyecto, no una copia alterada.
2. Carga el modelo y predice ``input_example.json``: si no carga o no devuelve conteos finitos y no
   negativos, el trabajo falla y Compose no arranca la API.
3. Si el registry ya tiene una versión con la misma huella, solo asegura el alias y termina
   (idempotente: ``docker compose up`` se puede repetir sin crear versiones duplicadas).
4. Si no, crea la corrida ``registro-compose-v<versión>`` con el linaje de la versión original
   (corrida, commit, data_version) como etiquetas y sus métricas, sube la carpeta como artefacto
   ``model``, la registra con el mismo nombre y le pone el alias y las etiquetas originales.

Variables de entorno: ``MLFLOW_TRACKING_URI`` (``http://mlflow:5000``), ``MODELO_DIR``
(``/modelo``), ``MODELO_ALIAS`` (``champion``), ``EXPERIMENTO`` (``mibici-equipo32``).
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODELO_DIR = Path(os.getenv("MODELO_DIR", "/modelo"))
ALIAS = os.getenv("MODELO_ALIAS", "champion")
EXPERIMENTO = os.getenv("EXPERIMENTO", "mibici-equipo32")
ETIQUETA_HUELLA = "sha256_model_skops"


def log(mensaje: str) -> None:
    print(f"[registrar-modelo] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_integridad(registro: dict) -> None:
    huellas = registro.get("sha256") or {}
    if "model.skops" not in huellas:
        raise SystemExit("registro.json no trae la huella SHA-256 de model.skops")
    for relativa, esperada in huellas.items():
        ruta = MODELO_DIR / relativa
        if not ruta.is_file():
            raise SystemExit(f"falta {relativa} en {MODELO_DIR}")
        if sha256(ruta) != esperada:
            raise SystemExit(f"{relativa} no coincide con su SHA-256 de registro.json")
    log(f"integridad verificada: {len(huellas)} archivos coinciden con registro.json")


def probar_modelo() -> float:
    modelo = mlflow.pyfunc.load_model(str(MODELO_DIR), suppress_warnings=True)
    ejemplo = json.loads((MODELO_DIR / "input_example.json").read_text(encoding="utf-8"))
    x = pd.DataFrame(ejemplo["data"], columns=ejemplo["columns"]).astype("float64")
    pred = [float(v) for v in modelo.predict(x)]
    if not pred or not all(math.isfinite(v) and v >= -1e-9 for v in pred):
        raise SystemExit(f"el modelo devolvió predicciones inválidas: {pred}")
    log(f"el modelo carga y predice el ejemplo: {pred[0]:.3f} viajes")
    return pred[0]


def esperar_servidor(cliente: MlflowClient, segundos: int = 120) -> None:
    limite = time.monotonic() + segundos
    while True:
        try:
            cliente.search_experiments(max_results=1)
            return
        except Exception as error:  # el servidor aún arranca
            if time.monotonic() > limite:
                raise SystemExit(f"MLflow no responde en {TRACKING}: {error}") from error
            time.sleep(2)


def version_existente(cliente: MlflowClient, nombre: str, huella: str):
    try:
        versiones = cliente.search_model_versions(f"name = '{nombre}'")
    except MlflowException:
        return None
    for mv in versiones:
        if (mv.tags or {}).get(ETIQUETA_HUELLA) == huella:
            return mv
    return None


def registrar(cliente: MlflowClient, registro: dict, huella: str, prediccion: float):
    nombre = registro["modelo"]
    version_origen = int(registro["version"])
    etiquetas = dict(registro.get("etiquetas_version", {}))
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=f"registro-compose-v{version_origen}") as corrida:
        mlflow.set_tags(
            {
                "origen": "docker-compose (models/servicio)",
                "version_origen": str(version_origen),
                "uri_registry_origen": registro["uri_registry"],
                "source_run_id": registro["run_id"],
                "git_commit": etiquetas.get("git_commit", ""),
                "data_version": etiquetas.get("data_version", ""),
                "model": etiquetas.get("model", ""),
                "question": etiquetas.get("question", "p1"),
            }
        )
        metricas = {k: float(v) for k, v in registro.get("metricas", {}).items()}
        metricas["prediccion_ejemplo"] = prediccion
        mlflow.log_metrics(metricas)
        mlflow.log_artifacts(str(MODELO_DIR), artifact_path="model")
        run_id = corrida.info.run_id
    try:
        cliente.create_registered_model(nombre, description=registro.get("descripcion", ""))
    except MlflowException:
        pass  # ya existe
    fuente = f"{mlflow.get_run(run_id).info.artifact_uri}/model"
    mv = cliente.create_model_version(
        nombre, source=fuente, run_id=run_id, description=registro.get("descripcion", "")
    )
    for _ in range(30):
        if cliente.get_model_version(nombre, mv.version).status == "READY":
            break
        time.sleep(1)
    for clave, valor in {
        **etiquetas,
        ETIQUETA_HUELLA: huella,
        "version_origen": str(version_origen),
    }.items():
        cliente.set_model_version_tag(nombre, mv.version, clave, str(valor))
    log(f"registrada {nombre} v{mv.version} (origen: v{version_origen}, corrida {run_id})")
    return mv


def main() -> int:
    registro = json.loads((MODELO_DIR / "registro.json").read_text(encoding="utf-8"))
    nombre = registro["modelo"]
    verificar_integridad(registro)
    prediccion = probar_modelo()
    mlflow.set_tracking_uri(TRACKING)
    cliente = MlflowClient()
    esperar_servidor(cliente)
    huella = registro["sha256"]["model.skops"]
    mv = version_existente(cliente, nombre, huella)
    if mv is None:
        mv = registrar(cliente, registro, huella, prediccion)
    else:
        log(f"{nombre} v{mv.version} ya estaba registrada con la misma huella: no se duplica")
    cliente.set_registered_model_alias(nombre, ALIAS, mv.version)
    log(f"alias @{ALIAS} -> v{mv.version}; la API puede cargar models:/{nombre}@{ALIAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
