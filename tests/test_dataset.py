"""Pruebas de la ingesta (``mibici/dataset.py``): normalización, codificación, hashes y descarga."""

import hashlib
import json

import pandas as pd
import pytest
import requests

from mibici import dataset
from mibici.config import MANIFEST_PATH


@pytest.mark.parametrize(
    ("original", "esperado"),
    [
        ("Año de nacimiento", "ano_de_nacimiento"),
        ("Inicio del viaje", "inicio_del_viaje"),
        ("  Origen Id  ", "origen_id"),
        ("A?o_de_nacimiento", "a_o_de_nacimiento"),  # carácter mal decodificado
        ("Género", "genero"),
        ("", ""),  # caso límite: encabezado vacío
    ],
)
def test_normalizar_nombre(original, esperado):
    assert dataset._normalizar_nombre(original) == esperado


def _crudo(col_anio: str = "Año de nacimiento") -> pd.DataFrame:
    return pd.DataFrame({
        "Viaje Id": [1, 2],
        "Usuario Id": [10, 11],
        "Género": ["M", "F"],
        col_anio: [1990, 1985],
        "Inicio del viaje": ["2025-01-01 08:00:00", "2025-01-01 09:30:00"],
        "Fin del viaje": ["2025-01-01 08:15:00", "2025-01-01 09:40:00"],
        "Origen Id": [5, 6],
        "Destino Id": [6, 5],
    })


def test_normalizar_viajes_esquema_y_minimizacion():
    df = dataset.normalizar_viajes(_crudo(), origen="2025_01")
    assert list(df.columns) == ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id",
                                "destino_id", "mes_archivo"]
    assert pd.api.types.is_datetime64_any_dtype(df["inicio_viaje"])
    assert (df["mes_archivo"] == "2025_01").all()


@pytest.mark.parametrize("col_anio", ["Año de nacimiento", "Anio de nacimiento", "A?o de nacimiento"])
def test_variantes_del_anio_se_eliminan(col_anio):
    """Los archivos de MiBici cambian el nombre de la columna; en todos los casos se elimina."""
    df = dataset.normalizar_viajes(_crudo(col_anio), origen="2025_01")
    assert not set(dataset.COLUMNAS_PERSONALES) & set(df.columns)
    assert not any("nacimiento" in c for c in df.columns)


def test_fecha_invalida_queda_como_nat():
    crudo = _crudo()
    crudo.loc[1, "Fin del viaje"] = "no es fecha"
    df = dataset.normalizar_viajes(crudo, origen="2025_01")
    assert df["fin_viaje"].isna().tolist() == [False, True]


@pytest.mark.parametrize("codificacion", ["utf-8", "latin-1"])
def test_leer_csv_tolera_codificaciones(tmp_path, codificacion):
    ruta = tmp_path / "viajes.csv"
    ruta.write_bytes("Año,Género\n1990,Mujer\n".encode(codificacion))
    df = dataset.leer_csv(ruta)
    assert list(df.columns) == ["Año", "Género"]
    assert df.loc[0, "Género"] == "Mujer"


@pytest.mark.parametrize(
    ("contenido", "esperado"),
    [
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),  # vacío
    ],
)
def test_sha256(tmp_path, contenido, esperado):
    ruta = tmp_path / "archivo.bin"
    ruta.write_bytes(contenido)
    assert dataset._sha256(ruta) == esperado


@pytest.fixture
def sin_red(monkeypatch):
    """Cualquier intento de descarga hace fallar la prueba."""

    def prohibido(*args, **kwargs):
        raise AssertionError("no debería descargar")

    monkeypatch.setattr(dataset.requests, "get", prohibido)


def test_descargar_es_idempotente(tmp_path, sin_red):
    destino = tmp_path / "viajes_2025_01.csv"
    destino.write_text("ya descargado")
    assert dataset.descargar("https://x/datos.csv", destino) == destino
    assert destino.read_text() == "ya descargado"


def test_descargar_usa_copia_local(tmp_path, sin_red):
    local = tmp_path / "local"
    local.mkdir()
    (local / "datos_abiertos_2025_01.csv").write_text("copia local")
    destino = tmp_path / "raw" / "viajes_2025_01.csv"
    destino.parent.mkdir()
    dataset.descargar("https://x/datos_abiertos_2025_01.csv", destino, raw_source=local)
    assert destino.read_text() == "copia local"


def test_descargar_error_http_no_deja_archivo(tmp_path, monkeypatch):
    class Respuesta:
        content = b""

        def raise_for_status(self):
            raise requests.HTTPError("404")

    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: Respuesta())
    destino = tmp_path / "viajes.csv"
    with pytest.raises(requests.HTTPError):
        dataset.descargar("https://x/no-existe.csv", destino)
    assert not destino.exists()


def test_manifiesto_versionado_es_consistente():
    """La data_version del manifiesto en Git es el hash de los SHA-256 de cada archivo."""
    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    hashes = sorted(a["sha256"] for a in manifiesto["archivos"].values())
    assert manifiesto["data_version"] == hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]
    assert manifiesto["total_viajes"] == sum(
        a["registros"] for n, a in manifiesto["archivos"].items() if n.startswith("viajes_"))
