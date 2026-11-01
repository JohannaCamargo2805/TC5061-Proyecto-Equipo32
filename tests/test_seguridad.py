"""Pruebas de las prácticas de seguridad del repositorio.

El escáner de secretos debe detectar secretos reales (y no confundirlos con los falsos
positivos revisados), ``.gitignore`` debe cubrir los archivos sensibles, la configuración
debe leerse de variables de entorno y bandit no debe reportar hallazgos.
"""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mibici import config
from mibici.config import PROJ_ROOT

sys.path.insert(0, str(PROJ_ROOT / "scripts"))
import escanear_secretos  # noqa: E402

# Secretos FALSOS construidos en tiempo de ejecución (no quedan como texto en el repo).
LLAVE_AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
TOKEN_GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0" + "k1L2m3N4o5P6q7R8"
CABECERA_PEM = "-----BEGIN " + "RSA PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("archivo", "contenido"),
    [
        ("config.py", f'AWS_ACCESS_KEY_ID = "{LLAVE_AWS}"\n'),
        ("deploy.sh", f"export GITHUB_TOKEN={TOKEN_GITHUB}\n"),
        ("llave.txt", f"{CABECERA_PEM}\nMIIEowIBAAKCAQEA\n"),
    ],
    ids=["llave-aws", "token-github", "llave-privada-pem"],  # sin imprimir los valores falsos
)
def test_detecta_secretos_reales(archivo, contenido):
    hallazgos = escanear_secretos._escanear(archivo, contenido.encode())
    assert hallazgos, "el escáner no detectó el secreto"
    assert all(h["falso_positivo"] is None for h in hallazgos)
    assert all(LLAVE_AWS not in h["vista"] and TOKEN_GITHUB not in h["vista"] for h in hallazgos)


@pytest.mark.parametrize(
    ("archivo", "linea", "esperado"),
    [
        ("data/manifest.json", '  "sha256": "' + "ab" * 32 + '",', "manifiesto"),
        ("notebooks/x.ipynb", '"image/png": "iVBORw0KGgoAAAANSUhEUgAA",', "PNG"),
        ("reports/seguridad/secretos.json", '  "commit": "' + "c0ffee12" * 5 + '",', "Git"),
        ("src/config.py", '"sha256": "' + "ab" * 32 + '"', None),  # fuera del manifiesto
        ("notebooks/x.ipynb", f'"token": "{TOKEN_GITHUB}"', None),
    ],
    ids=["hash-en-manifiesto", "png-en-notebook", "hash-git-en-reporte", "hash-fuera-del-manifiesto",
         "token-en-notebook"],
)
def test_falsos_positivos_solo_con_reglas_explicitas(archivo, linea, esperado):
    motivo = escanear_secretos._clasificar(archivo, linea)
    if esperado is None:
        assert motivo is None
    else:
        assert esperado in motivo


@pytest.mark.skipif(not (PROJ_ROOT / ".git").exists() or shutil.which("git") is None,
                    reason="requiere el repositorio de Git (no aplica a una copia descargada en ZIP)")
def test_gitignore_cubre_archivos_sensibles():
    assert escanear_secretos.revisar_ignorados() == []


def test_git_no_instalado_da_error_claro(monkeypatch):
    from mibici import git_seguro

    monkeypatch.setattr(git_seguro.shutil, "which", lambda nombre: None)
    with pytest.raises(FileNotFoundError, match="git no está instalado"):
        git_seguro.git("status")


def test_env_example_sin_valores_secretos():
    texto = (PROJ_ROOT / ".env.example").read_text(encoding="utf-8")
    for linea in texto.splitlines():
        if any(p in linea.upper() for p in ("PASSWORD", "TOKEN", "SECRET")) and "=" in linea:
            assert linea.lstrip().startswith("#") or linea.split("=", 1)[1].strip() == ""


def test_configuracion_desde_variables_de_entorno(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://servidor-de-prueba:5000")
    try:
        assert importlib.reload(config).MLFLOW_TRACKING_URI == "http://servidor-de-prueba:5000"
    finally:
        monkeypatch.delenv("MLFLOW_TRACKING_URI")
        importlib.reload(config)


def test_dependencias_con_version_fija():
    """Cada dependencia lleva versión exacta (==): instalaciones reproducibles y auditables."""
    for linea in (PROJ_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea and not linea.startswith("-e"):
            assert "==" in linea, linea


def test_bandit_sin_hallazgos(tmp_path):
    reporte = tmp_path / "bandit.json"
    subprocess.run([sys.executable, "-m", "bandit", "-c", "pyproject.toml", "-r", "mibici",
                    "scripts", "-f", "json", "-o", str(reporte), "-q"], cwd=PROJ_ROOT, check=False)
    resultados = json.loads(Path(reporte).read_text(encoding="utf-8"))["results"]
    assert resultados == [], [(r["test_id"], r["filename"], r["line_number"]) for r in resultados]
