"""Ingesta reproducible de datos de MiBici — Equipo 32.

Descarga los meses definidos en config.py, normaliza columnas y
codificación, guarda Parquet y versiona los datos con un manifiesto
de hashes. Ejecutable sin pasos manuales:

    python -m mibici.dataset
    mlflow run . -e ingest                       # como punto de entrada del MLflow Project
    mlflow run . -e ingest -P raw_source=<dir>   # copia local de los CSV (sin red)

Si se ejecuta dentro de ``mlflow run``, la corrida registra el manifiesto como
artefacto y la ``data_version`` como etiqueta (etapa ``ingest``).
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import unicodedata

from loguru import logger
import pandas as pd
import requests
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    NOMENCLATURA_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VIAJES_PARQUET,
    VIAJES_URLS,
)

app = typer.Typer()

HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico TC5061)"}

# Mapeo de nombres de columna: los archivos de MiBici cambian
# nombres y acentos entre años; todo se normaliza a este esquema.
COLUMNAS_VIAJES = {
    "viaje_id": "viaje_id",
    "usuario_id": "usuario_id",
    "genero": "genero",
    "ano_de_nacimiento": "anio_nacimiento",
    "anio_de_nacimiento": "anio_nacimiento",
    "a_o_de_nacimiento": "anio_nacimiento",
    "inicio_del_viaje": "inicio_viaje",
    "fin_del_viaje": "fin_viaje",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}


# Columnas con datos personales que se eliminan en la ingesta.
COLUMNAS_PERSONALES = ["usuario_id", "genero", "anio_nacimiento"]


def _normalizar_nombre(col: str) -> str:
    """Quita acentos, pasa a minúsculas y deja solo [a-z0-9_]."""
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = col.strip().lower().replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in col)


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path, raw_source: Path | None = None) -> Path:
    """Descarga un archivo si no existe ya en data/raw (idempotente).

    Si ``raw_source`` apunta a una carpeta con una copia local del archivo (con el
    nombre original del portal o el nombre de ``data/raw``), se copia en lugar de
    descargarlo. El SHA-256 del manifiesto permite comprobar que es el mismo archivo.
    """
    if destino.exists():
        logger.info(f"Ya existe, no se descarga: {destino.name}")
        return destino
    if raw_source is not None:
        for nombre in (url.rsplit("/", 1)[-1], destino.name):
            local = Path(raw_source) / nombre
            if local.exists():
                logger.info(f"Copia local: {local} -> {destino.name}")
                shutil.copy(local, destino)
                return destino
    logger.info(f"Descargando {url}")
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)
    return destino


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV tolerando las codificaciones usadas por MiBici."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def normalizar_viajes(df: pd.DataFrame, origen: str) -> pd.DataFrame:
    """Normaliza nombres de columna, tipos de fecha y agrega el mes origen."""
    df.columns = [_normalizar_nombre(c) for c in df.columns]
    df = df.rename(columns={c: COLUMNAS_VIAJES[c] for c in df.columns if c in COLUMNAS_VIAJES})
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"], errors="coerce")
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"], errors="coerce")
    df["mes_archivo"] = origen
    # Minimización de datos personales (aviso de privacidad de MiBici/AMIM): el
    # pseudónimo de usuario, el género y el año de nacimiento no se usan en P1-P3,
    # así que no se guardan en los Parquet procesados.
    return df.drop(columns=[c for c in COLUMNAS_PERSONALES if c in df.columns])


def _registrar_en_mlflow(manifiesto: dict, total_estaciones: int) -> None:
    """Si corre dentro de ``mlflow run``, deja la versión de datos en la corrida de ingesta."""
    if not os.getenv("MLFLOW_RUN_ID"):
        return
    import mlflow

    from mibici.config import TEAM_ID
    from mibici.modeling.seguimiento import git_info

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "ingest-mibici")
        mlflow.set_tags({"stage_of_work": "ingest", "question": "all", "entrega": "mlflow-project",
                         "equipo": TEAM_ID, "model_family": "datos",
                         "data_version": manifiesto["data_version"], **git_info()})
        mlflow.log_params({"archivos": len(manifiesto["archivos"]),
                           "meses": ",".join(VIAJES_URLS)})
        mlflow.log_metrics({"total_viajes": manifiesto["total_viajes"],
                            "total_estaciones": total_estaciones})
        mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")


@app.command()
def main(raw_source: str = typer.Option("config", help="Carpeta con copias locales de los CSV")):
    # "config" (o vacío) = descargar de las URLs de config.py
    sin_valor = raw_source.strip().lower() in {"", "config", "auto", "no", '""', "''"}
    fuente_local = None if sin_valor else Path(raw_source)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "archivos": {},
    }

    # ---- 1. Viajes ----
    partes = []
    for mes, url in VIAJES_URLS.items():
        ruta = descargar(url, RAW_DATA_DIR / f"viajes_{mes}.csv", fuente_local)
        df = normalizar_viajes(leer_csv(ruta), origen=mes)
        manifiesto["archivos"][ruta.name] = {
            "url": url,
            "sha256": _sha256(ruta),
            "registros": len(df),
        }
        partes.append(df)
    viajes = pd.concat(partes, ignore_index=True)

    # ---- 2. Nomenclatura de estaciones ----
    ruta_nom = descargar(NOMENCLATURA_URL, RAW_DATA_DIR / "nomenclatura.csv", fuente_local)
    estaciones = leer_csv(ruta_nom)
    estaciones.columns = [_normalizar_nombre(c) for c in estaciones.columns]
    manifiesto["archivos"][ruta_nom.name] = {
        "url": NOMENCLATURA_URL,
        "sha256": _sha256(ruta_nom),
        "registros": len(estaciones),
    }

    # ---- 3. Guardar en formato eficiente ----
    viajes.to_parquet(VIAJES_PARQUET, index=False)
    estaciones.to_parquet(ESTACIONES_PARQUET, index=False)

    # ---- 4. Versión de los datos ----
    hashes = sorted(v["sha256"] for v in manifiesto["archivos"].values())
    manifiesto["data_version"] = hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]
    manifiesto["total_viajes"] = len(viajes)
    # Si los datos no cambiaron se conserva la fecha original: el manifiesto versionado en
    # Git no se modifica y el árbol queda limpio para que git_commit sea reproducible.
    if MANIFEST_PATH.exists():
        anterior = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        sin_fecha = {k: v for k, v in manifiesto.items() if k != "generado"}
        if sin_fecha == {k: v for k, v in anterior.items() if k != "generado"}:
            manifiesto["generado"] = anterior["generado"]
    # newline="\n": en Windows evita reescribir el archivo con CRLF (Git lo vería modificado).
    MANIFEST_PATH.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                             encoding="utf-8", newline="\n")

    logger.success(
        f"Ingesta completa: {len(viajes):,} viajes, {len(estaciones)} estaciones, "
        f"data_version={manifiesto['data_version']}"
    )
    _registrar_en_mlflow(manifiesto, len(estaciones))


if __name__ == "__main__":
    app()