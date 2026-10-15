"""Ingesta reproducible y versionada de los datos abiertos de MiBici.

Uso desde la terminal (desde la raíz del repositorio)::

    python -m mibici.dataset ingest   # descarga, normaliza, guarda Parquet y manifiesto
    python -m mibici.dataset verify   # comprueba que los datos locales coinciden con el manifiesto

El proceso no requiere pasos manuales:

1. Descarga los CSV mensuales del rango de fechas de ``config.py`` y la
   nomenclatura de estaciones (si un archivo ya existe en ``data/raw`` no se
   vuelve a descargar: ``data/raw`` es inmutable).
2. Detecta la codificación (UTF-8 / CP1252), el separador, los nombres de
   columnas (que cambian entre años) y el formato de fecha de cada archivo, y
   los normaliza a un esquema canónico.
3. Elimina las columnas con datos personales (minimización) y aplica reglas de
   limpieza documentadas, contando cuántos registros descarta cada regla.
4. Guarda un Parquet por mes en ``data/interim/trips`` y las estaciones en
   ``data/interim/stations.parquet``.
5. Escribe ``references/data_manifest.json`` con el número de registros y el
   hash SHA-256 de cada archivo crudo y procesado. El manifiesto se versiona en
   Git y de él se deriva ``data_version``, que se registra en MLflow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici import config

app = typer.Typer(help="Ingesta reproducible y versionada de datos de MiBici.")

CANONICAL_TRIP_COLUMNS = [
    "trip_id",
    "start_time",
    "end_time",
    "origin_station_id",
    "destination_station_id",
    "duration_s",
]

# Patrones (sobre nombres normalizados) para mapear las variantes de cada año
# al esquema canónico. El orden importa: se asigna la primera coincidencia.
TRIP_COLUMN_PATTERNS: list[tuple[str, str]] = [
    ("trip_id", r"(^|_)viaje_?id$|^id_?viaje$|^trip_?id$"),
    ("user_id", r"usuario|^user"),
    ("gender", r"^g.{0,3}nero$|^sexo$|^gender$"),
    ("birth_year", r"nacimiento|^birth"),
    ("age", r"^edad$|^age$"),
    ("start_time", r"inicio|^start|^fecha_?hora_?inicio"),
    ("end_time", r"(^|_)fin(_|$)|^end|^fecha_?hora_?fin|termino"),
    ("origin_station_id", r"origen|^origin|^start_?station"),
    ("destination_station_id", r"destino|^destination|^end_?station"),
    ("duration_raw", r"duraci"),
]

STATION_COLUMN_PATTERNS: list[tuple[str, str]] = [
    ("station_id", r"^id$|^id_?estacion$|^station_?id$|^estacion_?id$"),
    ("code", r"^obcn$|^codigo$|^code$|^clave$"),
    ("name", r"^name$|^nombre"),
    ("location", r"^location$|^ubicacion$|^zona$|^municipio$|^colonia$"),
    ("lat", r"^lat"),
    ("lon", r"^lon|^lng"),
    ("capacity", r"capac|anclaje|^docks?$|^slots?$|puertos"),
    ("status", r"^status$|^estatus$|^estado$"),
]

DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
]


# ---------------------------------------------------------------------------
# Utilidades de fechas y descubrimiento de archivos
# ---------------------------------------------------------------------------
def month_keys(start: str, end: str) -> list[str]:
    """Lista de meses ``AAAA_MM`` entre ``start`` y ``end`` (inclusive)."""
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [f"{p.year:04d}_{p.month:02d}" for p in periods]


def discover_trip_urls(portal_url: str = config.MIBICI_PORTAL_URL) -> dict[str, str]:
    """Lee el portal de datos abiertos y devuelve ``{AAAA_MM: url}``.

    Solo se usa para meses que no estén fijados en ``config.TRIP_URLS``.
    """
    html = requests.get(portal_url, timeout=60, headers=_headers()).text
    urls: dict[str, str] = {}
    pattern = r"""href=["']([^"']*datos_abiertos_(\d{4})_(\d{2})[^"']*\.csv)["']"""
    for href, year, month in re.findall(pattern, html):
        full = requests.compat.urljoin(portal_url, href)
        urls.setdefault(f"{year}_{month}", full)
    return urls


def resolve_trip_urls(months: list[str], discover: bool = False) -> dict[str, str]:
    """Obtiene la URL de cada mes: primero ``config.TRIP_URLS``, luego el portal."""
    urls = {m: config.TRIP_URLS[m] for m in months if m in config.TRIP_URLS}
    missing = [m for m in months if m not in urls]
    if missing or discover:
        logger.info(f"Buscando en el portal {len(missing)} mes(es) sin URL fija...")
        found = discover_trip_urls()
        for m in missing:
            if m not in found:
                raise typer.BadParameter(f"No se encontró el archivo del mes {m} en el portal.")
            urls[m] = found[m]
    return dict(sorted(urls.items()))


def _headers() -> dict[str, str]:
    return {"User-Agent": "TC5061-MiBici-ingesta/1.0 (proyecto academico)"}


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> None:
    """Descarga ``url`` a ``dest`` por bloques, con reintentos y escritura atómica."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=_headers()) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    fh.writelines(r.iter_content(chunk_size=1 << 20))
            tmp.replace(dest)
            return
        except requests.RequestException as exc:
            logger.warning(f"Intento {attempt}/{retries} falló para {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar {url}")


def sha256_file(path: Path) -> str:
    """Hash SHA-256 del contenido binario de un archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(df: pd.DataFrame) -> str:
    """Hash del *contenido* de un DataFrame (independiente del formato en disco)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h = hashlib.sha256(hashed.tobytes())
    h.update("|".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Normalización de archivos
# ---------------------------------------------------------------------------
def detect_encoding(path: Path) -> str:
    """Devuelve ``utf-8-sig`` si el archivo es UTF-8 válido; si no, ``cp1252``."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(path: Path, encoding: str) -> str:
    """Elige el separador más frecuente en la línea de encabezado."""
    with open(path, encoding=encoding, errors="replace") as fh:
        header = fh.readline()
    counts = {sep: header.count(sep) for sep in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get)


def normalize_column_name(name: str) -> str:
    """Quita BOM, acentos y símbolos; devuelve ``snake_case`` en minúsculas."""
    name = str(name).replace("﻿", "").strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return name


def map_columns(columns: list[str], patterns: list[tuple[str, str]]) -> dict[str, str]:
    """Mapea columnas originales a nombres canónicos usando ``patterns``.

    Las columnas que no coinciden con ningún patrón conservan su nombre
    normalizado. Si dos columnas compiten por el mismo nombre canónico, se
    conserva la primera.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        norm = normalize_column_name(col)
        target = norm
        for canonical, pattern in patterns:
            if canonical not in used and re.search(pattern, norm):
                target = canonical
                break
        used.add(target)
        mapping[col] = target
    return mapping


def detect_datetime_format(
    values: pd.Series, year: int | None = None, month: int | None = None
) -> str:
    """Elige el formato de fecha que interpreta mejor una muestra de valores.

    Si se conoce el mes del archivo, se privilegia el formato cuyo mes coincide
    (resuelve la ambigüedad día/mes de fechas como 03/04/2025).
    """
    sample = values.dropna().astype(str).str.strip()
    sample = sample.sample(min(len(sample), 2000), random_state=config.SEED)
    best_fmt, best_score = None, -1.0
    for fmt in DATETIME_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        ok = parsed.notna().mean()
        if ok == 0:
            continue
        score = ok
        if year is not None and month is not None:
            in_month = ((parsed.dt.year == year) & (parsed.dt.month == month)).mean()
            score = ok + in_month
        if score > best_score:
            best_fmt, best_score = fmt, score
    if best_fmt is None:
        raise ValueError(f"Formato de fecha no reconocido. Ejemplos: {sample.head(3).tolist()}")
    return best_fmt


def read_raw_trips(path: Path, year: int, month: int) -> tuple[pd.DataFrame, dict]:
    """Lee un CSV mensual y lo lleva al esquema canónico (sin limpiar)."""
    encoding = detect_encoding(path)
    sep = detect_delimiter(path, encoding)
    df = pd.read_csv(path, encoding=encoding, sep=sep, dtype=str, keep_default_na=True)
    original_columns = list(df.columns)
    mapping = map_columns(original_columns, TRIP_COLUMN_PATTERNS)
    df = df.rename(columns=mapping)
    required = ["start_time", "end_time", "origin_station_id", "destination_station_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {missing}. Columnas: {original_columns}")

    fmt_start = detect_datetime_format(df["start_time"], year, month)
    fmt_end = detect_datetime_format(df["end_time"])
    out = pd.DataFrame(
        {
            "trip_id": pd.to_numeric(df["trip_id"], errors="coerce").astype("Int64")
            if "trip_id" in df.columns
            else pd.array([pd.NA] * len(df), dtype="Int64"),
            "start_time": pd.to_datetime(
                df["start_time"].str.strip(), format=fmt_start, errors="coerce"
            ),
            "end_time": pd.to_datetime(
                df["end_time"].str.strip(), format=fmt_end, errors="coerce"
            ),
            "origin_station_id": pd.to_numeric(df["origin_station_id"], errors="coerce").astype(
                "Int32"
            ),
            "destination_station_id": pd.to_numeric(
                df["destination_station_id"], errors="coerce"
            ).astype("Int32"),
        }
    )
    out["start_time"] = out["start_time"].astype("datetime64[ns]")
    out["end_time"] = out["end_time"].astype("datetime64[ns]")
    personal = [c for c in config.PERSONAL_COLUMNS if c in df.columns]
    info = {
        "encoding": encoding,
        "delimiter": sep,
        "original_columns": original_columns,
        "column_mapping": {k: v for k, v in mapping.items()},
        "datetime_format_start": fmt_start,
        "datetime_format_end": fmt_end,
        "personal_columns_dropped": personal,
        "raw_rows": len(df),
    }
    return out, info


def clean_trips(df: pd.DataFrame, year: int, month: int) -> tuple[pd.DataFrame, dict]:
    """Aplica las reglas de limpieza y devuelve el conteo descartado por regla.

    Reglas (en orden): campos obligatorios nulos o no interpretables, ``trip_id``
    duplicado, inicio fuera del mes del archivo, duración menor a
    ``MIN_TRIP_SECONDS`` (incluye negativas) y mayor a ``MAX_TRIP_HOURS``.
    """
    dropped: dict[str, int] = {}

    def _drop(mask: pd.Series, rule: str) -> None:
        nonlocal df
        dropped[rule] = int(mask.sum())
        df = df.loc[~mask]

    required = ["start_time", "end_time", "origin_station_id", "destination_station_id"]
    _drop(df[required].isna().any(axis=1), "campos_obligatorios_nulos")
    if df["trip_id"].notna().any():
        _drop(df["trip_id"].notna() & df["trip_id"].duplicated(keep="first"), "trip_id_duplicado")
    else:
        dropped["trip_id_duplicado"] = 0
    in_month = (df["start_time"].dt.year == year) & (df["start_time"].dt.month == month)
    _drop(~in_month, "inicio_fuera_del_mes")
    duration = (df["end_time"] - df["start_time"]).dt.total_seconds()
    _drop(duration < config.MIN_TRIP_SECONDS, "duracion_menor_1_min")
    duration = (df["end_time"] - df["start_time"]).dt.total_seconds()
    _drop(duration > config.MAX_TRIP_HOURS * 3600, "duracion_mayor_24_h")

    df = df.copy()
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds().astype("int32")
    df = df.sort_values(["start_time", "trip_id"], kind="mergesort").reset_index(drop=True)
    return df[CANONICAL_TRIP_COLUMNS], dropped


def read_stations(path: Path) -> tuple[pd.DataFrame, dict]:
    """Lee y normaliza la nomenclatura de estaciones (id, nombre, lat, lon, capacidad)."""
    encoding = detect_encoding(path)
    sep = detect_delimiter(path, encoding)
    df = pd.read_csv(path, encoding=encoding, sep=sep, dtype=str)
    original_columns = list(df.columns)
    df = df.rename(columns=map_columns(original_columns, STATION_COLUMN_PATTERNS))
    out = pd.DataFrame(
        {"station_id": pd.to_numeric(df["station_id"], errors="coerce").astype("Int32")}
    )
    for col in ["code", "name", "location", "status"]:
        if col in df.columns:
            out[col] = df[col].astype("string").str.strip()
    for col in ["lat", "lon"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    out["capacity"] = (
        pd.to_numeric(df["capacity"], errors="coerce").astype("Int16")
        if "capacity" in df.columns
        else pd.array([pd.NA] * len(df), dtype="Int16")
    )
    # Una capacidad <= 0 no es válida (estación inactiva o dato faltante).
    out["capacity"] = out["capacity"].where(out["capacity"] > 0)
    out = out.dropna(subset=["station_id"]).drop_duplicates("station_id")
    out = out.sort_values("station_id").reset_index(drop=True)
    info = {
        "encoding": encoding,
        "delimiter": sep,
        "original_columns": original_columns,
        "rows": len(out),
        "stations_with_capacity": int(out["capacity"].notna().sum()),
    }
    return out, info


# ---------------------------------------------------------------------------
# Manifiesto y versión de datos
# ---------------------------------------------------------------------------
def compute_data_version(manifest: dict) -> str:
    """Deriva un identificador corto y determinista de la versión de los datos.

    Depende del rango de fechas, la versión del pipeline de limpieza y los
    hashes de todos los archivos crudos: si cambia cualquiera, cambia la versión.
    """
    core = {
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "pipeline_version": manifest["pipeline_version"],
        "raw": sorted(f["raw_sha256"] for f in manifest["files"]),
        "stations_raw": manifest["stations"]["raw_sha256"],
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]
    start = manifest["date_start"][:7].replace("-", "")
    end = manifest["date_end"][:7].replace("-", "")
    return f"v{manifest['pipeline_version']}-{start}_{end}-{digest}"


def load_manifest(path: Path = config.DATA_MANIFEST_PATH) -> dict:
    """Lee el manifiesto versionado."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m mibici.dataset ingest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_data_version(path: Path = config.DATA_MANIFEST_PATH) -> str:
    """Versión de datos que se registra como parámetro en MLflow."""
    return load_manifest(path)["data_version"]


def _rel(path: Path) -> str:
    return path.relative_to(config.PROJ_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    start: str = typer.Option(config.DATE_START, help="Fecha inicial AAAA-MM-DD"),
    end: str = typer.Option(config.DATE_END, help="Fecha final AAAA-MM-DD (inclusive)"),
    discover: bool = typer.Option(False, help="Forzar búsqueda de URLs en el portal"),
) -> None:
    """Descarga, normaliza, guarda en Parquet y versiona los datos de MiBici."""
    months = month_keys(start, end)
    urls = resolve_trip_urls(months, discover=discover)
    previous = load_manifest() if config.DATA_MANIFEST_PATH.exists() else None
    prev_hashes = {f["name"]: f["raw_sha256"] for f in previous["files"]} if previous else {}

    config.RAW_TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    config.INTERIM_TRIPS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Estaciones
    st_raw = config.RAW_DATA_DIR / config.STATIONS_RAW_NAME
    if not st_raw.exists():
        logger.info(f"Descargando nomenclatura de estaciones: {config.STATIONS_URL}")
        download_file(config.STATIONS_URL, st_raw)
    stations, st_info = read_stations(st_raw)
    stations.to_parquet(config.STATIONS_INTERIM_PATH, index=False)
    stations_entry = {
        "name": st_raw.name,
        "url": config.STATIONS_URL,
        "raw_sha256": sha256_file(st_raw),
        "raw_bytes": st_raw.stat().st_size,
        **st_info,
        "processed_file": _rel(config.STATIONS_INTERIM_PATH),
        "processed_content_sha256": content_sha256(stations),
    }

    # 2) Viajes, un mes a la vez (memoria acotada)
    files, station_ids_seen = [], set()
    for key in tqdm(months, desc="Meses"):
        year, month = int(key[:4]), int(key[5:])
        raw_path = config.RAW_TRIPS_DIR / f"datos_abiertos_{key}.csv"
        if not raw_path.exists():
            logger.info(f"Descargando {urls[key]}")
            download_file(urls[key], raw_path)
        raw_hash = sha256_file(raw_path)
        if raw_path.name in prev_hashes and prev_hashes[raw_path.name] != raw_hash:
            logger.warning(
                f"{raw_path.name}: el hash difiere del manifiesto versionado; "
                "MiBici pudo republicar el archivo. Se generará una nueva versión de datos."
            )
        trips, info = read_raw_trips(raw_path, year, month)
        trips, dropped = clean_trips(trips, year, month)
        out_path = config.INTERIM_TRIPS_DIR / f"trips_{key}.parquet"
        trips.to_parquet(out_path, index=False)
        station_ids_seen |= set(trips["origin_station_id"].dropna().astype(int))
        station_ids_seen |= set(trips["destination_station_id"].dropna().astype(int))
        files.append(
            {
                "name": raw_path.name,
                "month": key,
                "url": urls[key],
                "raw_sha256": raw_hash,
                "raw_bytes": raw_path.stat().st_size,
                **info,
                "dropped": dropped,
                "processed_file": _rel(out_path),
                "processed_rows": len(trips),
                "processed_content_sha256": content_sha256(trips),
            }
        )
        logger.info(f"{key}: {info['raw_rows']:,} crudos -> {len(trips):,} limpios")
        del trips

    known = set(stations["station_id"].astype(int))
    manifest = {
        "dataset": "MiBici - viajes por mes y nomenclatura de estaciones",
        "source": config.MIBICI_PORTAL_URL,
        "date_start": start,
        "date_end": end,
        "pipeline_version": config.PIPELINE_VERSION,
        "cleaning_rules": {
            "min_trip_seconds": config.MIN_TRIP_SECONDS,
            "max_trip_hours": config.MAX_TRIP_HOURS,
            "personal_columns_dropped": config.PERSONAL_COLUMNS,
        },
        "totals": {
            "months": len(files),
            "raw_rows": sum(f["raw_rows"] for f in files),
            "clean_trips": sum(f["processed_rows"] for f in files),
            "stations_in_nomenclature": len(stations),
            "stations_with_trips": len(station_ids_seen),
            "stations_with_trips_in_nomenclature": len(station_ids_seen & known),
            "station_ids_not_in_nomenclature": sorted(station_ids_seen - known),
        },
        "stations": stations_entry,
        "files": files,
    }
    manifest["data_version"] = compute_data_version(manifest)
    config.DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": el manifiesto es idéntico en Windows, macOS y Linux (sin CRLF).
    config.DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    t = manifest["totals"]
    logger.success(
        f"Ingesta completa | versión {manifest['data_version']} | {t['months']} meses | "
        f"{t['raw_rows']:,} registros crudos | {t['clean_trips']:,} viajes limpios | "
        f"{t['stations_in_nomenclature']} estaciones en nomenclatura"
    )


@app.command()
def verify() -> None:
    """Comprueba que los archivos locales coinciden con el manifiesto versionado."""
    manifest = load_manifest()
    problems = 0
    st = manifest["stations"]
    checks = [(config.RAW_DATA_DIR / st["name"], st["raw_sha256"], None, None)]
    for f in manifest["files"]:
        checks.append(
            (
                config.RAW_TRIPS_DIR / f["name"],
                f["raw_sha256"],
                config.PROJ_ROOT / f["processed_file"],
                f["processed_content_sha256"],
            )
        )
    for raw_path, raw_hash, proc_path, proc_hash in tqdm(checks, desc="Verificando"):
        if not raw_path.exists() or sha256_file(raw_path) != raw_hash:
            logger.error(f"No coincide o falta: {raw_path.name}")
            problems += 1
        if proc_path is not None and (
            not proc_path.exists() or content_sha256(pd.read_parquet(proc_path)) != proc_hash
        ):
            logger.error(f"No coincide o falta: {proc_path.name}")
            problems += 1
    if problems:
        logger.error(f"{problems} archivo(s) no coinciden con {manifest['data_version']}")
        raise typer.Exit(code=1)
    logger.success(f"Datos locales idénticos a la versión {manifest['data_version']}")


if __name__ == "__main__":
    app()
