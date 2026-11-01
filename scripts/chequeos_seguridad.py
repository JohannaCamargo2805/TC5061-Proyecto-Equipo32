"""Chequeos de seguridad del repositorio con un solo comando.

1. **bandit**: análisis estático del código (``mibici/`` y ``scripts/``).
2. **pip-audit**: vulnerabilidades conocidas (PyPI/OSV) en las versiones fijadas de
   ``requirements.txt`` y en todos los paquetes instalados del entorno (incluye las
   dependencias transitivas).
3. **Secretos**: ``scripts/escanear_secretos.py`` (detect-secrets en el árbol y en todo el
   historial de Git, más la revisión de ``.gitignore``).

Uso::

    python scripts/chequeos_seguridad.py

Guarda los reportes en ``reports/seguridad/`` (JSON y texto de cada herramienta y
``RESUMEN.md``) y termina con código 1 si alguna herramienta encuentra un problema.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

# Revisado (bandit B404): solo se ejecutan bandit y pip-audit con argumentos fijos.
import subprocess  # nosec B404
import sys

from mibici.config import PROJ_ROOT

SALIDA = PROJ_ROOT / "reports" / "seguridad"


def _herramienta(*args: str, salida_texto: Path | None = None) -> int:
    """Ejecuta ``python -m <herramienta>`` con argumentos fijos y guarda su salida de texto."""
    # Revisado (bandit B603): intérprete actual, sin shell y con argumentos escritos aquí.
    proc = subprocess.run([sys.executable, "-m", *args], cwd=PROJ_ROOT,  # nosec B603
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=False)  # el código de salida se evalúa con los reportes JSON
    texto = (proc.stdout or "") + (proc.stderr or "")
    if salida_texto is not None:
        salida_texto.write_text(texto, encoding="utf-8", newline="\n")
    return proc.returncode


def bandit() -> dict:
    base = ["bandit", "-c", "pyproject.toml", "-r", "mibici", "scripts"]
    _herramienta(*base, "-f", "txt", "-o", str(SALIDA / "bandit.txt"))
    _herramienta(*base, "-f", "json", "-o", str(SALIDA / "bandit.json"))
    datos = json.loads((SALIDA / "bandit.json").read_text(encoding="utf-8"))
    totales = datos["metrics"]["_totals"]
    severidad = {s: int(totales.get(f"SEVERITY.{s.upper()}", 0)) for s in ("high", "medium", "low")}
    return {
        "herramienta": "bandit",
        "lineas_analizadas": int(totales.get("loc", 0)),
        "hallazgos": len(datos["results"]),
        "por_severidad": severidad,
        "revisados_nosec": int(totales.get("skipped_tests", 0)),
        "detalle": [{"id": r["test_id"], "severidad": r["issue_severity"],
                     "archivo": r["filename"], "linea": r["line_number"], "texto": r["issue_text"]}
                    for r in datos["results"]],
        "ok": len(datos["results"]) == 0,
    }


def pip_audit(nombre: str, *args: str) -> dict:
    ruta = SALIDA / f"pip_audit_{nombre}.json"
    _herramienta("pip_audit", *args, "--progress-spinner", "off", "--desc", "-f", "json",
                 "-o", str(ruta))
    _herramienta("pip_audit", *args, "--progress-spinner", "off",
                 salida_texto=SALIDA / f"pip_audit_{nombre}.txt")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    deps = datos.get("dependencies", [])
    vulns = [{"paquete": d["name"], "version": d.get("version"), "id": v["id"],
              "alias": v.get("aliases", []), "corregida_en": v.get("fix_versions", [])}
             for d in deps for v in d.get("vulns", [])]
    return {"herramienta": f"pip-audit ({nombre})", "paquetes_auditados": len(deps),
            "omitidos": [d["name"] for d in deps if d.get("skip_reason")],
            "vulnerabilidades": vulns, "ok": not vulns}


def secretos() -> dict:
    sys.path.insert(0, str(PROJ_ROOT / "scripts"))
    import escanear_secretos

    codigo = escanear_secretos.main()
    datos = json.loads(escanear_secretos.SALIDA.read_text(encoding="utf-8"))
    return {"herramienta": "detect-secrets + .gitignore", "secretos_reales": datos["secretos_reales"],
            "versiones_revisadas": datos["versiones_de_archivo_revisadas"],
            "falsos_positivos_revisados": datos["falsos_positivos_revisados"],
            "problemas_gitignore": datos["problemas_gitignore"], "ok": codigo == 0}


def escribir_resumen(res: dict) -> Path:
    b, pr, pe, s = res["bandit"], res["pip_audit_requirements"], res["pip_audit_entorno"], \
        res["secretos"]
    estado = {True: "sin hallazgos", False: "REVISAR"}
    lineas = [
        "# Resumen de los chequeos de seguridad", "",
        (f"Generado por `python scripts/chequeos_seguridad.py` el {res['fecha']} "
        f"(Python {res['python']}, {res['sistema']})."), "",
        "| Chequeo | Herramienta | Resultado | Estado |", "|---|---|---|---|",
        (f"| Código (análisis estático) | bandit | {b['hallazgos']} hallazgos en "
        f"{b['lineas_analizadas']:,} líneas (alta {b['por_severidad']['high']}, media "
        f"{b['por_severidad']['medium']}, baja {b['por_severidad']['low']}); "
        f"{b['revisados_nosec']} revisados con `# nosec` | {estado[b['ok']]} |"),
        (f"| Dependencias fijadas | pip-audit -r requirements.txt | {pr['paquetes_auditados']} "
        f"paquetes, {len(pr['vulnerabilidades'])} vulnerabilidades | {estado[pr['ok']]} |"),
        (f"| Entorno instalado (incluye transitivas) | pip-audit | {pe['paquetes_auditados']} "
        f"paquetes, {len(pe['vulnerabilidades'])} vulnerabilidades | {estado[pe['ok']]} |"),
        (f"| Secretos en árbol e historial | detect-secrets | {s['versiones_revisadas']} versiones "
        f"de archivo, {s['secretos_reales']} secretos reales | {estado[s['ok']]} |"),
        (f"| Archivos sensibles fuera de Git | git check-ignore | "
        f"{len(s['problemas_gitignore'])} problemas | {estado[not s['problemas_gitignore']]} |"),
        "",
    ]
    for clave in ("pip_audit_requirements", "pip_audit_entorno"):
        for v in res[clave]["vulnerabilidades"]:
            lineas.append(f"- {res[clave]['herramienta']}: {v['paquete']} {v['version']} — "
                          f"{v['id']} (corregida en {', '.join(v['corregida_en']) or 'sin versión'})")
    if s["falsos_positivos_revisados"]:
        lineas += ["", "Coincidencias de detect-secrets revisadas (no son secretos):", ""]
        lineas += [f"- {n} × {motivo}" for motivo, n in s["falsos_positivos_revisados"].items()]
    ruta = SALIDA / "RESUMEN.md"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8", newline="\n")
    return ruta


def main() -> int:
    SALIDA.mkdir(parents=True, exist_ok=True)
    print("== 1/4 bandit (código) ==")
    res = {"bandit": bandit()}
    print(f"   hallazgos: {res['bandit']['hallazgos']}  {res['bandit']['por_severidad']}")
    print("== 2/4 pip-audit (requirements.txt) ==")
    res["pip_audit_requirements"] = pip_audit("requirements", "-r", "requirements.txt",
                                              "--no-deps", "--disable-pip")
    print(f"   vulnerabilidades: {len(res['pip_audit_requirements']['vulnerabilidades'])}")
    print("== 3/4 pip-audit (entorno instalado) ==")
    res["pip_audit_entorno"] = pip_audit("entorno")
    print(f"   vulnerabilidades: {len(res['pip_audit_entorno']['vulnerabilidades'])}")
    print("== 4/4 secretos y .gitignore ==")
    res["secretos"] = secretos()
    res.update(fecha=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               python=platform.python_version(), sistema=platform.platform())
    (SALIDA / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                         encoding="utf-8", newline="\n")
    ruta = escribir_resumen(res)
    print()
    print(ruta.read_text(encoding="utf-8"))
    ok = all(res[k]["ok"] for k in ("bandit", "pip_audit_requirements", "pip_audit_entorno",
                                    "secretos"))
    print("RESULTADO GLOBAL:", "SIN HALLAZGOS" if ok else "HAY HALLAZGOS POR REVISAR")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
