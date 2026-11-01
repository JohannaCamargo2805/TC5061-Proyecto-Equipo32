"""Fixtures compartidas de las pruebas del proyecto MiBici (Equipo 32).

* Las pruebas unitarias usan datos sintéticos pequeños y corren en cualquier máquina.
* Las pruebas marcadas ``@pytest.mark.datos`` usan los Parquet reales de
  ``data/processed`` (fuera de Git) y se omiten si no existen. Para generarlos:
  ``python -m mibici.dataset``.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mibici.config import ESTACIONES_PARQUET, SEED, VIAJES_PARQUET

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

HAY_DATOS = VIAJES_PARQUET.exists() and ESTACIONES_PARQUET.exists()


def pytest_collection_modifyitems(config, items):
    """Omite las pruebas con datos reales cuando los Parquet no existen."""
    if HAY_DATOS:
        return
    omitir = pytest.mark.skip(reason="faltan data/processed/*.parquet: python -m mibici.dataset")
    for item in items:
        if "datos" in item.keywords:
            item.add_marker(omitir)


def generar_viajes(dias: int = 70, estaciones: int = 6, inicio: str = "2025-01-01",
                   semilla: int = SEED) -> pd.DataFrame:
    """Viajes sintéticos con estacionalidad semanal y horaria (Poisson, semilla fija)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for dia in pd.date_range(inicio, periods=dias, freq="D"):
        factor = 0.6 if dia.dayofweek >= 5 else 1.0
        for est in range(1, estaciones + 1):
            n = rng.poisson(lam=factor * (8 + 3 * est))
            horas = rng.choice(np.arange(6, 23), size=n)
            minutos = rng.integers(0, 60, size=n)
            duraciones = rng.integers(120, 2400, size=n)
            destinos = rng.integers(1, estaciones + 1, size=n)
            for h, m, d, dest in zip(horas, minutos, duraciones, destinos):
                ini = dia + pd.Timedelta(hours=int(h), minutes=int(m))
                filas.append((ini, ini + pd.Timedelta(seconds=int(d)), est, int(dest)))
    return pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])


@pytest.fixture(scope="session")
def viajes_sinteticos() -> pd.DataFrame:
    return generar_viajes()


@pytest.fixture(scope="session")
def capacidad_sintetica() -> pd.Series:
    return pd.Series({i: 10 + 2 * i for i in range(1, 7)}, name="dpcapacity")
