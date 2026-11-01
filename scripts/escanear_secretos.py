"""Busca secretos (contraseñas, tokens, llaves privadas) en el repositorio y en TODO su historial.

1. ``detect-secrets`` sobre cada archivo versionado (``git ls-files``).
2. ``detect-secrets`` sobre cada versión de cada archivo de texto en el historial de Git
   (todas las ramas): un secreto borrado en un commit posterior sigue expuesto en el historial.
3. Comprueba que los archivos sensibles (``.env``, datos, base de MLflow, llaves) estén
   ignorados por ``.gitignore`` y que ninguno esté versionado.

Los detectores de entropía de detect-secrets también marcan cadenas que no son secretos.
Cada hallazgo se clasifica con reglas explícitas (``FALSOS_POSITIVOS``): hashes SHA-256 de
los manifiestos de datos, vectores de prueba de SHA-256 e imágenes PNG en base64 dentro
de los notebooks. Solo los hallazgos que no encajan en una regla cuentan como secretos.
El reporte nunca guarda la cadena detectada, solo sus primeros caracteres.

Uso::

    python scripts/escanear_secretos.py            # escribe reports/seguridad/secretos.json

Termina con código 1 si encuentra algún hallazgo (útil para automatizarlo).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

from mibici.config import PROJ_ROOT
from mibici.git_seguro import GitError, git

SALIDA = PROJ_ROOT / "reports" / "seguridad" / "secretos.json"
BINARIOS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".xlsx", ".pptx", ".parquet",
            ".bundle", ".zip", ".gz", ".joblib", ".pkl", ".db"}
MAX_BYTES = 5_000_000

# Rutas que NUNCA deben versionarse; cada una debe quedar ignorada por .gitignore.
SENSIBLES = [
    ".env", ".env.local", "data/raw/viajes_2025_01.csv", "data/processed/viajes.parquet",
    "data/interim/cache.parquet", "mlflow.db", "mlruns/0/meta.yaml", "mlartifacts/x/model.pkl",
    "models/p3_model_bundle.joblib", "credentials.json", "llave.pem", ".pypirc",
]


# (regla, condición sobre el archivo y la línea marcada). Todas fueron revisadas a mano.
FALSOS_POSITIVOS = [
    ("hash SHA-256 o data_version del manifiesto de datos",
     lambda archivo, linea: archivo.endswith("manifest.json")
     and re.search(r'"(\w*sha256|data_version)": "[0-9a-f]{12,64}"', linea) is not None),
    ("vector de prueba de SHA-256 (cadenas 'abc' y vacía)",
     lambda archivo, linea: archivo.startswith("tests/")
     and any(hashlib.sha256(v).hexdigest() in linea for v in (b"abc", b""))),
    ("imagen PNG en base64 dentro de la salida de un notebook",
     lambda archivo, linea: archivo.endswith(".ipynb")
     and re.search(r'"image/png": "iVBORw0KGgo', linea) is not None),
]


def _clasificar(archivo: str, linea: str) -> str | None:
    for motivo, regla in FALSOS_POSITIVOS:
        if regla(archivo, linea):
            return motivo
    return None


def _escanear(nombre: str, contenido: bytes) -> list[dict]:
    """Escanea un contenido con los plugins por defecto de detect-secrets."""
    lineas = contenido.decode("utf-8", errors="replace").splitlines()
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / PurePosixPath(nombre).name
        ruta.write_bytes(contenido)
        secretos = SecretsCollection()
        with default_settings():
            secretos.scan_file(str(ruta))
    hallazgos = []
    for lista in secretos.json().values():
        for s in lista:
            n = s.get("line_number") or 0
            texto = lineas[n - 1] if 0 < n <= len(lineas) else ""
            hallazgos.append({"tipo": s["type"], "linea": n,
                              "falso_positivo": _clasificar(nombre, texto),
                              "vista": texto.strip()[:12] + "…"})
    return hallazgos


def escanear_arbol() -> list[dict]:
    """Archivos versionados en el commit actual."""
    hallazgos = []
    for nombre in git("ls-files").splitlines():
        ruta = PROJ_ROOT / nombre
        if ruta.suffix.lower() in BINARIOS or not ruta.is_file():
            continue
        for h in _escanear(nombre, ruta.read_bytes()):
            hallazgos.append({"archivo": nombre, **h})
    return hallazgos


def escanear_historial() -> tuple[int, list[dict]]:
    """Cada versión (blob) de cada archivo de texto en todas las ramas del historial."""
    vistos, hallazgos = set(), []
    for linea in git("rev-list", "--all", "--objects").splitlines():
        partes = linea.split(" ", 1)
        if len(partes) != 2 or partes[0] in vistos:
            continue
        sha, nombre = partes
        if PurePosixPath(nombre).suffix.lower() in BINARIOS:
            continue
        if git("cat-file", "-t", sha) != "blob" or int(git("cat-file", "-s", sha)) > MAX_BYTES:
            continue
        vistos.add(sha)
        contenido = git("cat-file", "-p", sha).encode("utf-8", errors="replace")
        for h in _escanear(nombre, contenido):
            hallazgos.append({"archivo": nombre, "blob": sha[:12], **h})
    return len(vistos), hallazgos


def revisar_ignorados() -> list[dict]:
    """Cada ruta sensible debe estar ignorada y ningún archivo así debe estar versionado."""
    problemas = []
    for ruta in SENSIBLES:
        try:
            git("check-ignore", "-q", "--no-index", ruta)
        except GitError:  # check-ignore devuelve 1 si la ruta NO está ignorada
            problemas.append({"ruta": ruta, "problema": "no está ignorada por .gitignore"})
    versionados = git("ls-files").splitlines()
    patrones = (".env", ".pem", ".key", ".parquet", ".csv", ".db", ".joblib", ".pkl")
    for nombre in versionados:
        base = PurePosixPath(nombre).name
        if base != ".env.example" and (base.startswith(".env") or base.endswith(patrones[1:])):
            problemas.append({"ruta": nombre, "problema": "archivo sensible versionado"})
    return problemas


def main() -> int:
    arbol = escanear_arbol()
    n_blobs, historial = escanear_historial()
    ignorados = revisar_ignorados()
    commits = int(git("rev-list", "--all", "--count"))
    reales = [h for h in arbol + historial if not h["falso_positivo"]]
    revisados = [h for h in arbol + historial if h["falso_positivo"]]
    motivos = {}
    for h in revisados:
        motivos[h["falso_positivo"]] = motivos.get(h["falso_positivo"], 0) + 1
    resumen = {
        "herramienta": "detect-secrets (plugins por defecto) + git check-ignore",
        "secretos_reales": len(reales),
        "falsos_positivos_revisados": motivos,
        "commit": git("rev-parse", "HEAD"),
        "archivos_versionados": len(git("ls-files").splitlines()),
        "commits_en_historial": commits,
        "versiones_de_archivo_revisadas": n_blobs,
        "hallazgos_arbol_actual": arbol,
        "hallazgos_historial": historial,
        "rutas_sensibles_revisadas": SENSIBLES,
        "problemas_gitignore": ignorados,
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8",
                      newline="\n")
    total = len(reales) + len(ignorados)
    print(f"Archivos versionados revisados: {resumen['archivos_versionados']}")
    print(f"Versiones de archivo en el historial ({commits} commits): {n_blobs}")
    print(f"Coincidencias de detect-secrets: {len(arbol)} en el árbol actual, "
          f"{len(historial)} en el historial")
    for motivo, n in motivos.items():
        print(f"  - falso positivo revisado ({n}): {motivo}")
    print(f"Secretos reales: {len(reales)}")
    print(f"Rutas sensibles revisadas: {len(SENSIBLES)}; sin ignorar o versionadas: "
          f"{len(ignorados)}")
    for h in reales + ignorados:
        print("  - REVISAR:", h)
    print(f"Resultado: {'SIN HALLAZGOS' if total == 0 else f'{total} HALLAZGOS'} -> {SALIDA}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
