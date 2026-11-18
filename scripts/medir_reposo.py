"""Memoria y CPU en reposo del servidor de MLflow y de la API según su configuración (entrega 6).

    python scripts/medir_reposo.py            # escribe reports/finops/reposo.json

Mide con ``docker stats`` lo que cuesta tener encendido cada servicio sin tráfico, que en el
análisis FinOps es costo fijo (se paga aunque no haya inferencias):

* MLflow con la configuración de fábrica de 3.17 (2 procesos web + ejecutor de trabajos de GenAI),
  sin el ejecutor y con 1 solo proceso (la que usa ``docker-compose.yml``).
* La API con 1 y con 2 procesos de Uvicorn (cada proceso carga su propia copia del modelo).

Al terminar deja el sistema con la configuración por defecto. Solo biblioteca estándar.
"""

import json
import os
from pathlib import Path
import subprocess  # nosec B404: solo docker compose con listas de argumentos fijas
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prueba_carga import DOCKER, RAIZ, salida  # noqa: E402

CONFIGURACIONES = [
    ("mlflow", "fabrica_2_procesos_con_trabajos", {"MLFLOW_WORKERS": "2", "MLFLOW_TRABAJOS": "true"}),
    ("mlflow", "2_procesos_sin_trabajos", {"MLFLOW_WORKERS": "2", "MLFLOW_TRABAJOS": "false"}),
    ("mlflow", "1_proceso_sin_trabajos", {"MLFLOW_WORKERS": "1", "MLFLOW_TRABAJOS": "false"}),
    ("api", "1_proceso", {"API_WORKERS": "1"}),
    ("api", "2_procesos", {"API_WORKERS": "2", "API_CPUS": "2"}),
]


def muestras(servicio: str, n: int = 4) -> dict:
    datos = []
    for _ in range(n):
        texto = salida([DOCKER, "stats", "--no-stream", "--format", "{{json .}}",
                        f"mibici-equipo32-{servicio}-1"])
        for linea in texto.splitlines():
            d = json.loads(linea)
            mem = d["MemUsage"].split("/")[0].strip()
            valor = float(mem[:-3]) * (1024 if mem.endswith("GiB") else 1)
            datos.append((float(d["CPUPerc"].rstrip("%")), valor))
        time.sleep(1)
    return {"cpu_pct": round(sum(c for c, _ in datos) / len(datos), 2),
            "mem_mib": round(sum(m for _, m in datos) / len(datos), 1)}


def main() -> int:
    compose = [DOCKER, "compose"] + sum((["-f", f] for f in sys.argv[1:]), [])
    resultados = []
    for servicio, nombre, variables in CONFIGURACIONES:
        entorno = {**os.environ, **variables}
        print(f"> {servicio}: {nombre} {variables}", flush=True)
        if subprocess.run(compose + ["up", "-d", "--wait", servicio], cwd=RAIZ,  # nosec B603
                          env=entorno).returncode != 0:
            return 1
        time.sleep(20)  # que termine de arrancar y se estabilice
        medida = muestras(servicio)
        print(f"  {medida}", flush=True)
        resultados.append({"servicio": servicio, "configuracion": nombre, **variables, **medida})
    subprocess.run(compose + ["up", "-d", "--wait"], cwd=RAIZ)  # nosec B603: valores por defecto
    destino = RAIZ / "reports" / "finops"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "reposo.json").write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"Escrito {destino / 'reposo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
