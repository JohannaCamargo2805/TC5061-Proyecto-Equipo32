"""Configuración central del proyecto MiBici.

Todas las rutas, fechas, semillas, URLs de datos e hiperparámetros viven aquí.
Ningún otro módulo debe repetir estos valores: si se necesita cambiar el rango
de fechas, la partición o una semilla, se cambia solo en este archivo y el
cambio queda versionado en Git (y registrado como parámetro en MLflow).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Carga variables de entorno desde .env si existe (nunca se versiona).
load_dotenv()

# ---------------------------------------------------------------------------
# Rutas (estructura de Cookiecutter Data Science)
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_TRIPS_DIR = RAW_DATA_DIR / "trips"  # CSV originales, inmutables
INTERIM_TRIPS_DIR = INTERIM_DATA_DIR / "trips"  # Parquet normalizado por mes
STATIONS_INTERIM_PATH = INTERIM_DATA_DIR / "stations.parquet"

MODELS_DIR = PROJ_ROOT / "models"
REFERENCES_DIR = PROJ_ROOT / "references"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Manifiesto de versionado de datos: es el ÚNICO archivo de datos que se versiona
# en Git (no contiene datos, solo conteos y hashes de cada archivo).
DATA_MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"

# Tablas analíticas (salida de features.py)
P1_DATASET_PATH = PROCESSED_DATA_DIR / "p1_demanda_estacion_hora.parquet"
P2_DATASET_PATH = PROCESSED_DATA_DIR / "p2_riesgo_estacion_hora.parquet"
P3_TRAIN_PROFILE_PATH = PROCESSED_DATA_DIR / "p3_perfiles_estacion_train.parquet"
P3_TEST_PROFILE_PATH = PROCESSED_DATA_DIR / "p3_perfiles_estacion_test.parquet"

# ---------------------------------------------------------------------------
# Equipo y experimento
# ---------------------------------------------------------------------------
TEAM_NUMBER = "32"  # equipo asignado en Canvas (TC5061, Gpo 10)
TEAM_ID = f"equipo{TEAM_NUMBER}"
MLFLOW_EXPERIMENT_NAME = f"mibici-{TEAM_ID}"
# Backend local (SQLite) y artefactos locales; ambos excluidos de Git.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Datos: fuente, rango de fechas y partición temporal
# ---------------------------------------------------------------------------
MIBICI_PORTAL_URL = "https://mibici.net/es/datos-abiertos/"

# 24 meses = dos ciclos anuales completos (incluye dos temporadas de lluvias).
DATE_START = "2024-10-01"
DATE_END = "2026-09-30"  # inclusivo

# Partición temporal (sin barajar). Cada bloque es posterior al anterior, de
# modo que el modelo nunca ve información del futuro al entrenar.
SPLITS = {
    "train": ("2024-10-01", "2026-03-31"),  # 18 meses
    "val": ("2026-04-01", "2026-06-30"),  # 3 meses: selección de umbrales/modelos
    "test": ("2026-07-01", "2026-09-30"),  # 3 meses: evaluación final (holdout)
}

# Enlaces publicados en el portal (los identificadores de carpeta no son
# predecibles, por eso se fijan aquí). dataset.py también puede descubrirlos
# leyendo el portal si se amplía el rango de fechas.
TRIP_URLS = {
    "2024_10": "https://mibici.net/site/assets/files/91758/datos_abiertos_2024_10.csv",
    "2024_11": "https://mibici.net/site/assets/files/93516/datos_abiertos_2024_11.csv",
    "2024_12": "https://mibici.net/site/assets/files/94962/datos_abiertos_2024_12.csv",
    "2025_01": "https://mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    "2025_02": "https://mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv",
    "2025_03": "https://mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv",
    "2025_04": "https://mibici.net/site/assets/files/103813/datos_abiertos_2025_04.csv",
    "2025_05": "https://mibici.net/site/assets/files/105513/datos_abiertos_2025_05.csv",
    "2025_06": "https://mibici.net/site/assets/files/107088/datos_abiertos_2025_06.csv",
    "2025_07": "https://mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv",
    "2025_08": "https://mibici.net/site/assets/files/111021/datos_abiertos_2025_08.csv",
    "2025_09": "https://mibici.net/site/assets/files/113216/datos_abiertos_2025_09.csv",
    "2025_10": "https://mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "2025_11": "https://mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "2025_12": "https://mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv",
    "2026_01": "https://mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "2026_02": "https://mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "2026_03": "https://mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "2026_04": "https://mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026_05": "https://mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026_06": "https://mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "2026_07": "https://mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "2026_08": "https://mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "2026_09": "https://mibici.net/site/assets/files/131825/datos_abiertos_2026_09.csv",
}
STATIONS_URL = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
STATIONS_RAW_NAME = "nomenclatura_2026_09.csv"

# Reglas de limpieza (si cambian, sube PIPELINE_VERSION para generar una nueva
# versión de datos).
PIPELINE_VERSION = "1.0"
MIN_TRIP_SECONDS = 60  # viajes < 1 min: desanclajes fallidos, no demanda real
MAX_TRIP_HOURS = 24  # viajes > 24 h: bicis no ancladas/registros anómalos
# Columnas con datos personales: se eliminan en la ingesta (minimización).
PERSONAL_COLUMNS = ["user_id", "gender", "birth_year", "age"]

# Horario de servicio publicado por MiBici: 05:00 a 00:59. Se usa para definir
# el "día operativo" (de 05:00 a 04:59 del día siguiente).
OPERATING_DAY_START_HOUR = 5
SERVICE_HOURS = 20  # 05:00-00:59

# Días de descanso obligatorio (art. 74 de la Ley Federal del Trabajo) y
# Jueves/Viernes Santo, dentro del rango de fechas.
HOLIDAYS = [
    "2024-10-01", "2024-11-18", "2024-12-25",
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-17", "2025-04-18",
    "2025-05-01", "2025-09-16", "2025-11-17", "2025-12-25",
    "2026-01-01", "2026-02-02", "2026-03-16", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-09-16",
]  # fmt: skip

# ---------------------------------------------------------------------------
# P1 - Regresión: viajes iniciados por estación y hora, horizonte día siguiente
# ---------------------------------------------------------------------------
P1_HORIZON_HOURS = 24  # la predicción se emite al cierre del día D-1 para todo D
P1_LAGS_HOURS = [24, 48, 168, 336]
P1_BASELINE_LAG_HOURS = 168  # ingenuo estacional: misma hora de la semana pasada
P1_HGB_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "early_stopping": False,
}

# ---------------------------------------------------------------------------
# P2 - Clasificación: riesgo de vaciado/llenado en las próximas horas
# ---------------------------------------------------------------------------
P2_HORIZON_HOURS = 3  # ventana futura evaluada después de la hora de decisión
P2_INITIAL_FILL = 0.50  # ocupación supuesta al iniciar el día operativo
P2_LOW_FILL = 0.10  # <= 10 % de la capacidad: riesgo de quedarse sin bicis
P2_HIGH_FILL = 0.90  # >= 90 % de la capacidad: riesgo de quedarse sin anclajes
P2_HGB_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": False,
}
P2_THRESHOLD_BETA = 2.0  # F2: el costo de no anticipar un riesgo > falsa alarma

# ---------------------------------------------------------------------------
# P3 - Clustering de estaciones por patrón de uso
# ---------------------------------------------------------------------------
# Bloques horarios (hora de inicio inclusive, fin exclusivo) para los perfiles.
P3_TIME_BLOCKS = {
    "madrugada": (0, 6),
    "pico_manana": (6, 10),
    "mediodia": (10, 14),
    "tarde": (14, 17),
    "pico_tarde": (17, 21),
    "noche": (21, 24),
}
P3_K_RANGE = list(range(3, 9))  # k candidatos; se elige por silueta en train
P3_MIN_DAILY_TRIPS = 1.0  # excluye estaciones casi sin uso en el periodo

# ---------------------------------------------------------------------------
# Logging: si tqdm está instalado, enruta loguru a tqdm.write
# https://github.com/Delgan/loguru/issues/135
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
