"""Pruebas estáticas del Containerfile, .dockerignore y requirements-api.txt (entrega 4).

No construyen la imagen (eso se hace con ``podman build`` / ``docker build``, ver README):
comprueban las decisiones que la hacen reproducible y segura. Imagen base con versión
exacta e igual a la del entrenamiento, dependencias fijas e iguales a las del modelo
registrado, usuario sin privilegios, puerto, HEALTHCHECK y un contexto de construcción
que no incluye datos, credenciales ni el entorno virtual.
"""

from pathlib import Path
import re

import pytest
import yaml

from mibici.config import PROJ_ROOT, SERVICIO_MODELO_DIR

CONTAINERFILE = (PROJ_ROOT / "Containerfile").read_text(encoding="utf-8")
HAY_MODELO = (SERVICIO_MODELO_DIR / "MLmodel").is_file()


def _instrucciones(texto: str) -> list[tuple[str, str]]:
    """(INSTRUCCIÓN, argumentos) uniendo las líneas continuadas con '\\'."""
    texto = re.sub(r"\\\n\s*", " ", texto)
    lineas = [ln.strip() for ln in texto.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return [(ln.split(None, 1)[0].upper(), ln.split(None, 1)[1] if " " in ln else "") for ln in lineas]


def _etapa_final() -> list[tuple[str, str]]:
    instrucciones = _instrucciones(CONTAINERFILE)
    ultimo_from = max(i for i, (op, _) in enumerate(instrucciones) if op == "FROM")
    return instrucciones[ultimo_from:]


def _pines(ruta: Path) -> dict[str, str]:
    pines = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if "==" in linea:
            nombre, version = linea.split("==")
            pines[nombre.strip().lower().replace("_", "-")] = version.strip()
    return pines


def _imagen_base() -> str:
    return re.search(r"^ARG IMAGEN_PYTHON=(\S+)", CONTAINERFILE, re.MULTILINE).group(1)


def test_imagen_base_con_version_exacta_y_nombre_completo():
    imagen = _imagen_base()
    assert re.fullmatch(r"docker\.io/library/python:3\.14\.\d+-slim-trixie", imagen), imagen
    assert [args for op, args in _instrucciones(CONTAINERFILE) if op == "FROM"] == [
        "${IMAGEN_PYTHON} AS dependencias", "${IMAGEN_PYTHON}"]


@pytest.mark.skipif(not HAY_MODELO, reason="falta models/servicio")
def test_python_de_la_imagen_igual_al_del_entrenamiento():
    mlmodel = yaml.safe_load((SERVICIO_MODELO_DIR / "MLmodel").read_text(encoding="utf-8"))
    entrenamiento = mlmodel["flavors"]["python_function"]["python_version"]
    assert _imagen_base().split(":")[1].removesuffix("-slim-trixie") == entrenamiento


def test_usuario_sin_privilegios():
    usuarios = [args for op, args in _etapa_final() if op == "USER"]
    assert usuarios and usuarios[-1].split(":")[0] not in {"root", "0"}


def test_puerto_healthcheck_y_comando():
    final = dict(_etapa_final())
    assert final["EXPOSE"] == "8000"
    assert "/health" in final["HEALTHCHECK"] and "--start-period" in final["HEALTHCHECK"]
    assert final["CMD"].startswith('["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"')
    assert "MODELO_URI=/app/models/servicio" in final["ENV"]


def test_solo_copia_la_api_el_modelo_y_sus_dependencias():
    origenes = set()
    for op, args in _instrucciones(CONTAINERFILE):
        if op == "COPY" and not args.startswith("--from"):
            origenes.update(args.split()[:-1])
    assert origenes == {"requirements-api.txt", "api/", "models/servicio/"}
    for origen in origenes - {"models/servicio/"}:
        assert (PROJ_ROOT / origen).exists(), origen


def _patron(p: str) -> re.Pattern:
    regex = ""
    i = 0
    while i < len(p):
        if p.startswith("**", i):
            regex, i = regex + ".*", i + 2
        elif p[i] == "*":
            regex, i = regex + "[^/]*", i + 1
        elif p[i] == "?":
            regex, i = regex + "[^/]", i + 1
        else:
            regex, i = regex + re.escape(p[i]), i + 1
    return re.compile(regex)


def _excluido(ruta: str, reglas: list[str]) -> bool:
    """Semántica de .dockerignore: gana la última regla que coincide (o la de un directorio padre)."""
    partes = ruta.split("/")
    prefijos = ["/".join(partes[: i + 1]) for i in range(len(partes))]
    excluido = False
    for regla in reglas:
        negada = regla.startswith("!")
        patron = _patron(regla.lstrip("!").rstrip("/"))
        if any(patron.fullmatch(p) for p in prefijos):
            excluido = not negada
    return excluido


@pytest.mark.parametrize("ruta", [".env", ".venv/Scripts/python.exe", "data/processed/viajes.parquet",
                                  "mlflow.db", "mlruns/1/models/m-1/artifacts/model.skops",
                                  ".git/config", "notebooks/1.0-ljcm-exploracion-demanda-p1.ipynb",
                                  "mibici/config.py", "api/__pycache__/main.cpython-314.pyc"])
def test_dockerignore_deja_fuera_datos_credenciales_y_entorno(ruta):
    reglas = [ln.strip() for ln in (PROJ_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")]
    assert _excluido(ruta, reglas), ruta


@pytest.mark.parametrize("ruta", ["requirements-api.txt", "api/main.py", "api/esquemas.py",
                                  "models/servicio/MLmodel", "models/servicio/registro.json",
                                  "models/servicio/model.skops"])
def test_dockerignore_incluye_lo_que_copia_el_containerfile(ruta):
    reglas = [ln.strip() for ln in (PROJ_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")]
    assert not _excluido(ruta, reglas), ruta


def test_dependencias_de_la_imagen_con_version_fija():
    for linea in (PROJ_ROOT / "requirements-api.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[\w.]+", linea), linea
    directas = _pines(PROJ_ROOT / "requirements-api.in")
    completas = _pines(PROJ_ROOT / "requirements-api.txt")
    assert directas.items() <= completas.items()


def test_mismas_versiones_que_el_entorno_de_desarrollo():
    api, desarrollo = _pines(PROJ_ROOT / "requirements-api.txt"), _pines(PROJ_ROOT / "requirements.txt")
    for paquete in ("scikit-learn", "numpy", "pandas", "fastapi", "starlette", "pydantic", "uvicorn"):
        assert api[paquete] == desarrollo[paquete], paquete
    assert api["mlflow-skinny"] == desarrollo["mlflow"]


@pytest.mark.skipif(not HAY_MODELO, reason="falta models/servicio")
def test_mismas_versiones_con_las_que_se_registro_el_modelo():
    """Servir con otras versiones de scikit-learn/numpy/pandas/scipy/skops puede cambiar las
    predicciones o impedir la carga. mlflow se excluye a propósito: el modelo se registró con
    3.17.1 y el proyecto usa 3.17.2 por CVE-2026-53995 (mismo formato de modelo)."""
    api, modelo = _pines(PROJ_ROOT / "requirements-api.txt"), _pines(SERVICIO_MODELO_DIR / "requirements.txt")
    for paquete in ("scikit-learn", "numpy", "pandas", "scipy", "skops"):
        assert api[paquete] == modelo[paquete], paquete
