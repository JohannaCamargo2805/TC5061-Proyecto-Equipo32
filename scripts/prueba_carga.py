"""Prueba de carga de punta a punta: levanta el sistema, corre Locust y mide recursos (entrega 6).

Solo usa la biblioteca estándar de Python y la CLI de Docker, así que corre igual en Windows
(Docker Desktop), Linux y en el runner de GitHub Actions::

    python scripts/prueba_carga.py                                  # escalones, réplica 1 vCPU
    python scripts/prueba_carga.py --nombre escalado --workers 2 --cpus 2
    python scripts/prueba_carga.py --nombre optimizado --acelerar    # motor en NumPy
    python scripts/prueba_carga.py --escenario humo --duracion 60   # la usa el pipeline de CI/CD
    python scripts/prueba_carga.py --escenario lote --duracion 60     # costo por inferencia en lote
    python scripts/prueba_carga.py --escenario asincrono --duracion 60

Pasos:

1. ``docker compose up -d --build --wait`` con el tamaño de réplica pedido (``API_WORKERS``,
   ``API_CPUS``): el sistema queda sano antes de medir.
2. Guarda el entorno (``entorno.json``): CPUs y memoria del motor de contenedores, versión de
   Docker, commit de Git, modelo que sirve la API y parámetros de la prueba.
3. Muestrea ``docker stats`` (CPU %, memoria, red) de todos los contenedores del proyecto en un
   hilo aparte mientras corre la prueba (``recursos.jsonl``).
4. Corre Locust sin interfaz dentro del contenedor ``locust`` (perfil ``carga``): CSV con el
   historial, reporte HTML y el registro de cada petición (``peticiones.csv.gz``).
5. Genera las gráficas y el resumen por escalón (``carga/graficas.py``, en el mismo contenedor).

Todo queda en ``reports/carga/<nombre>/``.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess  # nosec B404: solo se ejecutan docker y git con listas de argumentos fijas
import sys
import threading
import time
import urllib.request

RAIZ = Path(__file__).resolve().parents[1]
PROYECTO = "mibici-equipo32"
DOCKER = shutil.which("docker") or "docker"
GIT = shutil.which("git") or "git"


def ejecutar(comando: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("> " + " ".join(comando), flush=True)
    return subprocess.run(comando, cwd=RAIZ, **kwargs)  # nosec B603: sin shell


def salida(comando: list[str]) -> str:
    try:
        return subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True,  # nosec B603
                              timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _json_url(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # nosec B310: http://127.0.0.1 fijo
            return json.loads(r.read().decode("utf-8"))
    except Exception as error:
        return {"error": str(error)}


class Muestreador(threading.Thread):
    """``docker stats --no-stream`` en bucle: una línea JSON por contenedor y muestra."""

    def __init__(self, destino: Path, intervalo: float):
        super().__init__(daemon=True)
        self.destino, self.intervalo = destino, intervalo
        self.alto = threading.Event()
        self.muestras = 0

    def run(self):
        formato = "{{json .}}"
        with open(self.destino, "w", encoding="utf-8") as f:
            while not self.alto.is_set():
                inicio = time.time()
                texto = salida([DOCKER, "stats", "--no-stream", "--format", formato])
                for linea in texto.splitlines():
                    try:
                        dato = json.loads(linea)
                    except json.JSONDecodeError:
                        continue
                    if not dato.get("Name", "").startswith(PROYECTO):
                        continue
                    for clave in ("ID", "Container"):  # identificadores efímeros: no aportan
                        dato.pop(clave, None)
                    dato["ts"] = round(inicio, 3)
                    f.write(json.dumps(dato, ensure_ascii=False) + "\n")
                    self.muestras += 1
                f.flush()
                self.alto.wait(max(0.0, self.intervalo - (time.time() - inicio)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nombre", default=None, help="carpeta en reports/carga (por defecto el escenario)")
    p.add_argument("--escenario", default="escalones", choices=["escalones", "humo", "lote", "asincrono"])
    p.add_argument("--escalones", default="10:45,25:45,50:45,100:45,150:45,200:45,250:45,300:45")
    p.add_argument("--duracion", type=int, default=60, help="segundos (humo, lote y asincrono)")
    p.add_argument("--workers", type=int, default=1, help="procesos de Uvicorn de la API")
    p.add_argument("--cpus", default="1.0", help="límite de CPU del contenedor de la API")
    p.add_argument("--memoria", default="1g", help="límite de memoria del contenedor de la API")
    p.add_argument("--acelerar", action="store_true",
                   help="motor de inferencia en NumPy verificado (ACELERAR_INFERENCIA=1)")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre muestras de stats")
    p.add_argument("--slo-p95-ms", type=float, default=300.0, help="objetivo de latencia p95")
    p.add_argument("--sin-construir", action="store_true", help="no pasar --build a compose up")
    p.add_argument("--compose", nargs="*", default=[], help="archivos -f adicionales de compose")
    p.add_argument("--apagar", action="store_true", help="docker compose down al terminar")
    a = p.parse_args()

    nombre = a.nombre or a.escenario
    carpeta = RAIZ / "reports" / "carga" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        carpeta.chmod(0o777)  # el contenedor de Locust escribe con un usuario sin privilegios
    for viejo in carpeta.iterdir():
        if viejo.is_file():
            viejo.unlink()

    compose = [DOCKER, "compose"]
    for archivo in a.compose:
        compose += ["-f", archivo]
    entorno = {**os.environ, "API_WORKERS": str(a.workers), "API_CPUS": str(a.cpus),
               "API_MEMORIA": a.memoria, "API_ACELERAR": "1" if a.acelerar else "0",
               "ESCENARIO": a.escenario,
               "ESCALONES": a.escalones, "DURACION": str(a.duracion)}

    arriba = compose + ["up", "-d", "--wait"] + ([] if a.sin_construir else ["--build"])
    if ejecutar(arriba, env=entorno).returncode != 0:
        print("El sistema no quedó sano; revisa 'docker compose logs'.", file=sys.stderr)
        return 1
    puerto = entorno.get("API_PUERTO", "8000")
    modelo = _json_url(f"http://127.0.0.1:{puerto}/modelo")

    info = {}
    try:
        info = json.loads(salida([DOCKER, "info", "--format", "{{json .}}"]) or "{}")
    except json.JSONDecodeError:
        pass
    registro = {
        "nombre": nombre,
        "escenario": a.escenario,
        "escalones": a.escalones if a.escenario == "escalones" else None,
        "duracion_s": a.duracion if a.escenario != "escalones" else None,
        "api": {"workers": a.workers, "cpus": float(a.cpus), "memoria": a.memoria,
                "acelerar": a.acelerar},
        "slo_p95_ms": a.slo_p95_ms,
        "inicio_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "anfitrion": {"sistema": platform.platform(), "equipo": platform.node(),
                      "python": platform.python_version()},
        "motor": {k: info.get(k) for k in ("ServerVersion", "OperatingSystem", "KernelVersion",
                                            "NCPU", "MemTotal", "Architecture")},
        "compose": salida([DOCKER, "compose", "version", "--short"]),
        "git_commit": salida([GIT, "rev-parse", "--short", "HEAD"]),
        "modelo": {k: modelo.get(k) for k in ("nombre", "version", "alias", "origen",
                                              "uri_registry", "integridad_verificada",
                                              "motor_inferencia")},
    }
    (carpeta / "entorno.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    muestreador = Muestreador(carpeta / "recursos.jsonl", a.intervalo)
    muestreador.start()
    time.sleep(max(4.0, a.intervalo * 2))  # línea base en reposo antes de la carga
    destino = f"/proyecto/reports/carga/{nombre}"
    locust = compose + [
        "--profile", "carga", "run", "--rm", "--no-deps",
        "-e", f"SALIDA_PETICIONES={destino}/peticiones.csv.gz",
        "locust", "--headless", "--host", "http://api:8000",
        "--csv", f"{destino}/locust", "--csv-full-history",
        "--html", f"{destino}/locust_reporte.html",
        "--exit-code-on-error", "0", "--only-summary",
    ]
    resultado = ejecutar(locust, env=entorno)
    time.sleep(max(4.0, a.intervalo * 2))  # regreso al reposo
    muestreador.alto.set()
    muestreador.join(timeout=30)
    print(f"docker stats: {muestreador.muestras} muestras en recursos.jsonl", flush=True)
    if resultado.returncode != 0:
        print("Locust terminó con error", file=sys.stderr)
        return resultado.returncode

    analisis = compose + ["--profile", "carga", "run", "--rm", "--no-deps",
                          "--entrypoint", "python", "locust", "/proyecto/carga/graficas.py", destino]
    codigo = ejecutar(analisis, env=entorno).returncode
    if a.apagar:
        ejecutar(compose + ["down"], env=entorno)
    print(f"Resultados en {carpeta.relative_to(RAIZ)}", flush=True)
    return codigo


if __name__ == "__main__":
    if shutil.which("docker") is None:
        sys.exit("No se encontró la CLI de Docker en el PATH")
    sys.exit(main())
