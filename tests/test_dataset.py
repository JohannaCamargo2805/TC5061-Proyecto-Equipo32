"""Pruebas de la normalización de la ingesta (dataset.py)."""

import pandas as pd

from mibici import config
from mibici.dataset import (
    TRIP_COLUMN_PATTERNS,
    clean_trips,
    compute_data_version,
    detect_datetime_format,
    map_columns,
    month_keys,
    normalize_column_name,
)


def test_month_keys_inclusive():
    assert month_keys("2024-10-01", "2025-01-31") == ["2024_10", "2024_11", "2024_12", "2025_01"]


def test_normalize_column_name_handles_bom_accents_and_spaces():
    assert normalize_column_name("﻿Año de nacimiento") == "ano_de_nacimiento"
    assert normalize_column_name("Inicio del viaje ") == "inicio_del_viaje"


def test_map_columns_variants_between_years():
    variants = [
        ["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento", "Inicio_del_viaje",
         "Fin_del_viaje", "Origen_Id", "Destino_Id"],
        ["viaje_id", "usuario_id", "genero", "ano_de_nacimiento", "inicio_del_viaje",
         "fin_del_viaje", "origen_id", "destino_id"],
        ["Viaje_Id", "Usuario_Id", "GÃ©nero", "AÃ±o_de_nacimiento", "Inicio_del_viaje",
         "Fin_del_viaje", "Origen_Id", "Destino_Id"],
    ]
    expected = {"trip_id", "user_id", "gender", "birth_year", "start_time", "end_time",
                "origin_station_id", "destination_station_id"}
    for cols in variants:
        assert set(map_columns(cols, TRIP_COLUMN_PATTERNS).values()) == expected


def test_detect_datetime_format_resolves_day_month_ambiguity():
    # Archivo de abril de 2025 con fechas dd/mm/aaaa: 03/04 debe leerse como 3 de abril.
    values = pd.Series([f"{d:02d}/04/2025 08:15:00" for d in range(1, 29)])
    assert detect_datetime_format(values, 2025, 4) == "%d/%m/%Y %H:%M:%S"
    iso = pd.Series(["2025-04-03 08:15:00", "2025-04-12 23:01:59"])
    assert detect_datetime_format(iso, 2025, 4) == "%Y-%m-%d %H:%M:%S"


def test_clean_trips_rules_and_counts():
    t = pd.Timestamp("2025-04-10 08:00:00")
    df = pd.DataFrame(
        {
            "trip_id": pd.array([1, 2, 2, 3, 4, 5, 6], dtype="Int64"),
            "start_time": [t, t, t, t, t, pd.Timestamp("2025-03-31 23:00"), t],
            "end_time": [t + pd.Timedelta(minutes=12), t + pd.Timedelta(minutes=5),
                         t + pd.Timedelta(minutes=5), t + pd.Timedelta(seconds=20),
                         t + pd.Timedelta(hours=30), pd.Timestamp("2025-03-31 23:10"), pd.NaT],
            "origin_station_id": pd.array([1, 1, 1, 2, 2, 3, 3], dtype="Int32"),
            "destination_station_id": pd.array([2, 2, 2, 2, 3, 1, 1], dtype="Int32"),
        }
    )
    out, dropped = clean_trips(df, 2025, 4)
    assert dropped == {
        "campos_obligatorios_nulos": 1,
        "trip_id_duplicado": 1,
        "inicio_fuera_del_mes": 1,
        "duracion_menor_1_min": 1,
        "duracion_mayor_24_h": 1,
    }
    assert out["trip_id"].tolist() == [1, 2]
    assert out["duration_s"].tolist() == [720, 300]
    assert not set(config.PERSONAL_COLUMNS) & set(out.columns)


def test_data_version_changes_when_raw_hash_changes():
    m = {
        "date_start": "2024-10-01", "date_end": "2026-09-30", "pipeline_version": "1.0",
        "files": [{"raw_sha256": "a"}, {"raw_sha256": "b"}], "stations": {"raw_sha256": "s"},
    }
    v1 = compute_data_version(m)
    assert v1 == compute_data_version(m)  # determinista
    m["files"][1]["raw_sha256"] = "c"
    assert compute_data_version(m) != v1
    assert v1.startswith("v1.0-202410_202609-")
