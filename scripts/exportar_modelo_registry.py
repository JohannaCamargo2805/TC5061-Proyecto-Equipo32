"""Recupera del Model Registry de MLflow el modelo que sirve la API (entrega 4).

Resuelve el modelo por **nombre y versión** (o por alias, por defecto ``champion``),
descarga sus artefactos a ``models/servicio/`` y escribe ``registro.json`` con la
versión, la corrida de origen, el linaje (commit y versión de datos), las métricas y
el SHA-256 de cada archivo. El ``Containerfile`` copia esa carpeta a la imagen y la
API verifica las huellas al arrancar, así el contenedor sirve exactamente la versión
registrada aunque el servidor de MLflow no esté disponible::

    # con el servidor de tracking levantado (scripts/servidor_mlflow.ps1 o make server)
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000      # PowerShell: $env:MLFLOW_TRACKING_URI="..."
    python scripts/exportar_modelo_registry.py --version 1          # nombre + versión
    python scripts/exportar_modelo_registry.py --alias champion     # la versión en producción
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit, urlunsplit

# Antes de importar mlflow: sin barra de progreso ni sugerencias en la salida.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from mibici.config import (  # noqa: E402
    ALIAS_PRODUCCION,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJ_ROOT,
    REGISTRY_NAMES,
    SERVICIO_MODELO_DIR,
    SERVICIO_PREGUNTA,
)

app = typer.Typer(add_completion=False)
REGISTRO = "registro.json"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def huellas(carpeta: Path) -> dict[str, str]:
    return {p.relative_to(carpeta).as_posix(): sha256(p)
            for p in sorted(carpeta.rglob("*")) if p.is_file() and p.name != REGISTRO}


def uri_sin_credenciales(uri: str) -> str:
    """Quita usuario y contraseña de la URI antes de escribirla en un archivo versionado."""
    partes = urlsplit(uri)
    if partes.username or partes.password:
        host = partes.hostname or ""
        if partes.port:
            host = f"{host}:{partes.port}"
        partes = partes._replace(netloc=host)
    return urlunsplit(partes)


def metricas_de_corrida(metricas: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in sorted(metricas.items())
            if k.startswith(("val_", "test_")) and not k.endswith("prevalencia")}


def _fecha(ms) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _preparar_destino(destino: Path) -> None:
    """Solo reemplaza carpetas dentro de models/ para no borrar nada por error."""
    destino = destino.resolve()
    if destino.exists():
        if MODELS_DIR.resolve() not in destino.parents:
            raise typer.BadParameter(f"{destino} ya existe y no está dentro de {MODELS_DIR}")
        shutil.rmtree(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)


def exportar(nombre: str, version: int, alias: str, destino: Path, tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient()
    registrado = cliente.get_registered_model(nombre)
    aliases_modelo = dict(registrado.aliases or {})
    if version:
        mv = cliente.get_model_version(nombre, str(version))
        alias_solicitado = None
    else:
        mv = cliente.get_model_version_by_alias(nombre, alias)
        alias_solicitado = alias
    if mv.status != "READY":
        raise typer.BadParameter(f"{nombre} v{mv.version} está en estado {mv.status}, no READY")
    corrida = cliente.get_run(mv.run_id)
    uri = f"models:/{nombre}/{mv.version}"

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp))
        if not (local / "MLmodel").is_file():
            raise typer.BadParameter(f"{uri} no contiene un archivo MLmodel")
        _preparar_destino(destino)
        shutil.copytree(local, destino)

    modelo = mlflow.models.Model.load(str(destino / "MLmodel"))
    firma = modelo.signature.inputs.input_names() if modelo.signature else []
    registro = {
        "modelo": nombre,
        "version": int(mv.version),
        "aliases": sorted(a for a, v in aliases_modelo.items() if str(v) == str(mv.version)),
        "alias_solicitado": alias_solicitado,
        "aliases_del_modelo": {a: int(v) for a, v in sorted(aliases_modelo.items())},
        "uri_registry": uri,
        "tracking_uri": uri_sin_credenciales(tracking_uri),
        "run_id": mv.run_id,
        "model_id": modelo.model_id,
        "source": mv.source,
        "estado": mv.status,
        "registrado": _fecha(mv.creation_timestamp),
        "descripcion": mv.description,
        "etiquetas_version": dict(sorted((mv.tags or {}).items())),
        "metricas": metricas_de_corrida(corrida.data.metrics),
        "firma_entrada": list(firma),
        "mlflow_version": mlflow.__version__,
        "recuperado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": huellas(destino),
    }
    (destino / REGISTRO).write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    return registro


def verificar(destino: Path) -> list[float]:
    """Carga la copia con mlflow.pyfunc y predice su input_example: prueba que es utilizable."""
    modelo = mlflow.pyfunc.load_model(str(destino))
    ejemplo = json.loads((destino / "input_example.json").read_text(encoding="utf-8"))
    x = pd.DataFrame(ejemplo["data"], columns=ejemplo["columns"])
    return [round(float(v), 3) for v in modelo.predict(x)]


@app.command()
def main(
    nombre: str = typer.Option(REGISTRY_NAMES[SERVICIO_PREGUNTA], help="Modelo registrado"),
    version: int = typer.Option(0, help="Versión registrada; 0 = la versión a la que apunta --alias"),
    alias: str = typer.Option(ALIAS_PRODUCCION, help="Alias que se resuelve si no se da --version"),
    destino: Path = typer.Option(SERVICIO_MODELO_DIR, help="Carpeta que copia el Containerfile"),
):
    """Descarga una versión del Model Registry a models/servicio con su registro.json."""
    tracking = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    registro = exportar(nombre, version, alias, destino, tracking)
    e, m = registro["etiquetas_version"], registro["metricas"]
    aliases = ", ".join(f"@{a} -> v{v}" for a, v in registro["aliases_del_modelo"].items())
    pedida = f"alias @{registro['alias_solicitado']}" if registro["alias_solicitado"] else "versión explícita"
    print(f"Model Registry      : {registro['tracking_uri']}")
    print(f"Modelo registrado   : {registro['modelo']}  ({aliases})")
    print(f"Versión recuperada  : v{registro['version']} por {pedida}  ->  {registro['uri_registry']}")
    print(f"Corrida de origen   : {e.get('source_run_name', '')} ({registro['run_id']}), estado {registro['estado']}")
    print(f"Linaje              : commit {e.get('git_commit', '')[:7]}, data_version {e.get('data_version', '')}, "
          f"modelo {e.get('model', '')} ({e.get('model_family', '')})")
    print(f"Métricas            : val_mae={m.get('val_mae', float('nan')):.3f}  test_mae={m.get('test_mae', float('nan')):.3f}  "
          f"test_wape={m.get('test_wape', float('nan')):.3f}")
    print(f"Firma de entrada    : {len(registro['firma_entrada'])} variables: {', '.join(registro['firma_entrada'])}")
    relativa = destino.resolve().relative_to(PROJ_ROOT) if PROJ_ROOT in destino.resolve().parents else destino
    print(f"Descargado en       : {relativa} ({len(registro['sha256'])} archivos, SHA-256 en {REGISTRO})")
    print(f"Verificación        : mlflow.pyfunc carga la copia y predice el input_example -> {verificar(destino)}")


if __name__ == "__main__":
    app()
