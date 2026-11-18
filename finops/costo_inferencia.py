"""Costo por inferencia del sistema MiBici P1 a partir de la prueba de carga (entrega 6).

    python finops/costo_inferencia.py            # desde la raíz del repositorio

Combina lo **medido** en la computadora de la autora (``reports/carga/*/resumen.json`` y
``reports/finops/reposo.json``) con **precios de lista** de nube (``precios_referencia.toml``):

1. **Costo unitario de cómputo** (por inferencia en línea y en lote): el precio por hora de una
   réplica del tamaño probado dividido entre las inferencias que atiende en una hora, a la
   utilización objetivo (60 %: se escala antes de saturar).
2. **Costo fijo de la plataforma**: MLflow + Redis + worker encendidos todo el mes, con el tamaño
   de máquina que pide su memoria medida en reposo (y cuánto costaría con la configuración de
   fábrica de MLflow).
3. **Escenarios de volumen** (lote diario de las 470 estaciones, app con 1 M y con 30 M de
   peticiones al mes) y la curva de costo por cada 1,000 inferencias contra el volumen mensual,
   para tres formas de desplegar: réplica siempre encendida (Fargate x86 o EC2 ARM) y Cloud Run
   con cobro por petición (escala a cero).

Escribe ``reports/finops/costos.json``, ``escenarios.csv`` y ``05_costo_por_volumen.png``.
Solo usa la biblioteca estándar más matplotlib (corre en el contenedor ``locust``).
"""

import csv
import json
import math
from pathlib import Path
import sys
import tomllib

RAIZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
CARGA = RAIZ / "reports" / "carga"
SALIDA = RAIZ / "reports" / "finops"
SEG_MES = 30 * 24 * 3600


def leer(ruta: Path) -> dict | list | None:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def hora_fargate(p: dict, vcpu: float, gb: float) -> float:
    return vcpu * p["vcpu_hora"] + gb * p["gb_hora"]


def gb_fargate(vcpu: float, gb_medido: float) -> float:
    """Memoria mínima válida en Fargate para ese vCPU (1 vCPU: 2-8 GB; 0.5: 1-4; 0.25: 0.5-2)."""
    minimo = {0.25: 0.5, 0.5: 1.0, 1.0: 2.0, 2.0: 4.0, 4.0: 8.0}.get(vcpu, 2.0 * vcpu)
    return max(minimo, math.ceil(gb_medido * 2) / 2)


def maquina_para(ec2: dict, gib_necesarios: float) -> tuple[str, dict]:
    """La instancia más barata de la tabla con memoria suficiente (deja 25 % para el sistema)."""
    candidatas = [(n, d) for n, d in ec2.items() if d["gib"] * 0.75 >= gib_necesarios]
    return min(candidatas, key=lambda x: x[1]["hora"])


def main() -> int:
    precios = tomllib.loads((RAIZ / "finops" / "precios_referencia.toml").read_text("utf-8"))
    sup = precios["supuestos"]
    u, horas = sup["utilizacion_objetivo"], sup["horas_mes"]
    base = leer(CARGA / "base" / "resumen.json")
    if not base or not base.get("capacidad"):
        print("Falta reports/carga/base/resumen.json (python scripts/prueba_carga.py --nombre base)")
        return 1
    escalado = leer(CARGA / "escalado" / "resumen.json")
    lote = leer(CARGA / "lote" / "resumen.json")
    asinc = leer(CARGA / "asincrono" / "resumen.json")
    reposo = leer(SALIDA / "reposo.json") or []

    cap = base["capacidad"]
    vcpu = float(base["api"]["cpus"])
    mem_api_gib = cap["mem_api_mib"] / 1024
    rps = cap["rps"]  # peticiones por segundo de la mezcla de la app en el escalón de capacidad
    cr = precios["cloud_run"]
    ec2 = precios["ec2"]
    gb_fg = gb_fargate(vcpu, mem_api_gib)
    hora = {
        "Fargate x86": hora_fargate(precios["fargate"]["x86"], vcpu, gb_fg),
        "Fargate ARM": hora_fargate(precios["fargate"]["arm"], vcpu, gb_fg),
        "Fargate Spot x86": hora_fargate(precios["fargate"]["spot_x86"], vcpu, gb_fg),
        "EC2 c7g.medium (ARM)": ec2["c7g.medium"]["hora"],
        "Cloud Run por instancia": 3600 * (vcpu * cr["por_instancia"]["vcpu_segundo"]
                                           + 1.0 * cr["por_instancia"]["gib_segundo"]),
    }
    por_s_cr = vcpu * cr["por_peticion"]["vcpu_segundo"] + 1.0 * cr["por_peticion"]["gib_segundo"]

    # 1. Costo unitario de cómputo en línea (USD por cada 1,000 peticiones a capacidad).
    unitario = {k: 1000 * h / (rps * u * 3600) for k, h in hora.items()}
    unitario["Cloud Run por petición"] = 1000 * (por_s_cr / (rps * u)
                                                 + cr["por_peticion"]["millon_peticiones"] / 1e6)
    cpu_s_pet = cap["cpu_api_pct"] / 100 / rps

    # Lote: inferencias por segundo con lotes de 100 seguidos (réplica al 100 %: es un trabajo).
    inf_lote = None
    if lote and lote.get("escalones"):
        inf_lote = lote["escalones"][-1]["inferencias_por_s"]
    lote_por_1000 = {k: 1000 * h / (inf_lote * 3600) for k, h in hora.items()} if inf_lote else {}

    # 2. Plataforma fija: MLflow + Redis + worker (memoria medida en reposo).
    def mem_reposo(servicio, configuracion=None):
        filas = [r for r in reposo if r["servicio"] == servicio
                 and (configuracion is None or r["configuracion"] == configuracion)]
        return filas[-1]["mem_mib"] / 1024 if filas else None

    mlflow_opt = mem_reposo("mlflow", "1_proceso_sin_trabajos") or \
        base.get("reposo", {}).get("mlflow", {}).get("mem_mib", 400) / 1024
    mlflow_fab = mem_reposo("mlflow", "fabrica_2_procesos_con_trabajos")
    otros = sum(base.get("reposo", {}).get(s, {}).get("mem_mib", 0) for s in ("redis", "worker"))
    plat_opt_gib = mlflow_opt + otros / 1024
    nombre_opt, maq_opt = maquina_para(ec2, plat_opt_gib)
    plataforma = {"memoria_gib": round(plat_opt_gib, 2), "maquina": nombre_opt,
                  "usd_mes": round(maq_opt["hora"] * horas, 2)}
    if mlflow_fab:
        plat_fab_gib = mlflow_fab + otros / 1024
        nombre_fab, maq_fab = maquina_para(ec2, plat_fab_gib)
        plataforma["fabrica"] = {"memoria_gib": round(plat_fab_gib, 2), "maquina": nombre_fab,
                                 "usd_mes": round(maq_fab["hora"] * horas, 2)}
    fijo = plataforma["usd_mes"]

    # 3. Curvas de costo contra volumen mensual.
    def siempre_encendido(v, h):
        # Flota fija dimensionada para la hora pico (sin autoescalado), mínimo una réplica.
        pico = v / SEG_MES * sup["pico_sobre_promedio"]
        replicas = max(1, math.ceil(pico / (rps * u)))
        return replicas * h * horas + fijo

    def cloud_run(v):
        promedio = v / SEG_MES
        # Con poco tráfico cada petición paga al menos 100 ms de instancia; con mucho, la
        # concurrencia reparte la instancia y el tiempo facturado tiende a 1 / (capacidad × u).
        seg = cr["por_peticion"]["redondeo_s"] if promedio < 1 else 1 / (rps * u)
        return v * (seg * por_s_cr + cr["por_peticion"]["millon_peticiones"] / 1e6) + fijo

    opciones = {
        "Fargate x86 siempre encendido": lambda v: siempre_encendido(v, hora["Fargate x86"]),
        "EC2 c7g.medium (ARM) siempre encendido":
            lambda v: siempre_encendido(v, hora["EC2 c7g.medium (ARM)"]),
        "Cloud Run por petición (escala a cero)": cloud_run,
    }
    volumenes = [10 ** (4 + i / 8) for i in range(33)]  # 1e4 .. 1e8
    curvas = {n: [(v, 1000 * f(v) / v) for v in volumenes] for n, f in opciones.items()}

    # Lote diario: trabajo de Cloud Run que arranca, predice las 470 estaciones y termina.
    n_mes = sup["estaciones"] * sup["dias_mes"]
    seg_trabajo = sup["arranque_trabajo_s"] + (sup["estaciones"] / inf_lote if inf_lote else 1)
    # Los trabajos de Cloud Run se cobran como la facturación por instancia.
    trabajo_mes = sup["dias_mes"] * seg_trabajo * (
        vcpu * cr["por_instancia"]["vcpu_segundo"] + 1.0 * cr["por_instancia"]["gib_segundo"])
    escenarios = [
        {"escenario": "Lote diario (uso actual: 470 estaciones x 30 días)", "inferencias_mes": n_mes,
         "opcion": "Trabajo diario en Cloud Run (sin plataforma encendida)",
         "usd_mes": trabajo_mes},
        {"escenario": "Lote diario (uso actual: 470 estaciones x 30 días)", "inferencias_mes": n_mes,
         "opcion": f"Sistema completo siempre encendido ({nombre_opt})",
         "usd_mes": fijo},
    ]
    for v, nombre in ((1e6, "App: 1 M peticiones/mes"), (3e7, "App: 30 M peticiones/mes")):
        for opcion, f in opciones.items():
            escenarios.append({"escenario": nombre, "inferencias_mes": int(v), "opcion": opcion,
                               "usd_mes": f(v)})
    for e in escenarios:
        e["usd_mes"] = round(e["usd_mes"], 2)
        e["usd_por_1000"] = round(1000 * e["usd_mes"] / e["inferencias_mes"], 4)
        pico = e["inferencias_mes"] / SEG_MES * sup["pico_sobre_promedio"]
        e["replicas_en_pico"] = max(1, math.ceil(pico / (rps * u)))

    # Escalado: ¿2 procesos en 2 vCPU rinden el doble? (costo por petición constante si es lineal)
    escala = None
    if escalado and escalado.get("capacidad"):
        c2 = escalado["capacidad"]
        escala = {"rps_1": rps, "rps_2": c2["rps"], "vcpu_2": escalado["api"]["cpus"],
                  "eficiencia": round(c2["rps"] / (rps * escalado["api"]["cpus"] / vcpu), 3),
                  "p95_ms_2": c2["p95_ms"]}

    resultado = {
        "fuente_precios": precios["fuentes"],
        "supuestos": sup,
        "medido": {
            "replica": base["api"],
            "capacidad_rps": rps,
            "capacidad_usuarios": cap["usuarios"],
            "p95_ms_en_capacidad": cap["p95_ms"],
            "cpu_api_pct_en_capacidad": cap["cpu_api_pct"],
            "cpu_s_por_peticion": round(cpu_s_pet, 6),
            "mem_api_mib": cap["mem_api_mib"],
            "inferencias_por_s_lote": inf_lote,
            "asincrono": (asinc or {}).get("asincrono"),
            "escalado": escala,
        },
        "precio_hora_replica": {k: round(v, 5) for k, v in hora.items()},
        "fargate_gb_reservados": gb_fg,
        "usd_por_millon_en_linea": {k: round(1000 * v, 4) for k, v in unitario.items()},
        "usd_por_millon_en_lote": {k: round(1000 * v, 4) for k, v in lote_por_1000.items()},
        "ahorro_lote_vs_linea": round(unitario["Fargate x86"] / lote_por_1000["Fargate x86"], 1)
        if lote_por_1000 else None,
        "plataforma_fija": plataforma,
        "segundos_trabajo_diario": round(seg_trabajo, 1),
        "escenarios": escenarios,
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "costos.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    with open(SALIDA / "escenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(escenarios[0]))
        w.writeheader()
        w.writerows(escenarios)
    grafica(curvas, escenarios[0], SALIDA / "05_costo_por_volumen.png")
    print(json.dumps({k: resultado[k] for k in ("usd_por_millon_en_linea", "usd_por_millon_en_lote",
                                               "plataforma_fija")}, indent=2, ensure_ascii=False))
    for e in escenarios:
        print(f"{e['escenario']:<55} {e['opcion']:<45} ${e['usd_mes']:>9.2f}/mes "
              f"${e['usd_por_1000']:.4f} por 1,000")
    return 0


def grafica(curvas: dict, lote: dict, ruta: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = ["#2a78d6", "#eb6834", "#1baf7a"]  # paleta categórica de referencia, orden fijo
    texto_2, rejilla = "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": rejilla, "axes.grid": True,
                         "grid.color": rejilla, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.color": texto_2,
                         "ytick.color": texto_2, "axes.labelcolor": texto_2,
                         "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for (nombre, puntos), color in zip(curvas.items(), colores):
        xs, ys = zip(*puntos)
        ax.plot(xs, ys, color=color, lw=2, label=nombre)
    ax.scatter([lote["inferencias_mes"]], [lote["usd_por_1000"]], s=36, color="#0b0b0b",
               zorder=3, label="Lote diario como trabajo de Cloud Run")
    ax.annotate(f"lote diario: ${lote['usd_por_1000']:.3f} por 1,000",
                (lote["inferencias_mes"], lote["usd_por_1000"]), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color=texto_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inferencias (peticiones) al mes")
    ax.set_ylabel("USD por cada 1,000 inferencias")
    ax.set_title("Costo por inferencia contra volumen (incluye plataforma fija)", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
