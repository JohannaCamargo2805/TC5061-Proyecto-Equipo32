"""Resumen por escalón y gráficas de una corrida de la prueba de carga (entrega 6).

    python carga/graficas.py reports/carga/<nombre>

Lee lo que deja ``scripts/prueba_carga.py`` en la carpeta (``peticiones.csv.gz``, el historial de
Locust, ``recursos.jsonl`` de ``docker stats`` y ``entorno.json``) y escribe:

* ``escalones.csv``: por escalón de usuarios, en su tramo estable (sin los primeros 10 s):
  peticiones por segundo, latencia p50/p95/p99 **exacta** (de cada petición, no de las ventanas
  móviles de Locust), % de error y CPU/memoria medias de la API.
* ``resumen.json``: lo anterior más la capacidad de una réplica (el escalón con más peticiones por
  segundo que cumple el SLO de p95 y < 1 % de errores), el CPU por inferencia y el consumo en reposo.
* ``01_desempeno.png``, ``02_capacidad.png``, ``03_recursos.png`` y ``04_endpoints.png``.
"""

import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta de referencia (skill dataviz): categórica en orden fijo y rampa ordinal azul.
TEXTO, TEXTO_2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e0"
SERVICIOS = {"api": "#2a78d6", "worker": "#eb6834", "mlflow": "#1baf7a", "redis": "#eda100"}
PERCENTILES = {"p50": "#86b6ef", "p95": "#2a78d6", "p99": "#104281"}
SERIE, CRITICO = "#2a78d6", "#e34948"
ESTABLE_S = 10  # se descartan los primeros segundos de cada escalón (arranque de usuarios)


def _estilo():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": REJILLA, "axes.linewidth": 0.8, "axes.labelcolor": TEXTO_2,
        "axes.titlecolor": TEXTO, "axes.titlelocation": "left", "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def _mib(texto: str) -> float:
    """'273.7MiB' -> 273.7 ; '1.2GiB' -> 1228.8 ; '512kB' -> 0.49"""
    m = re.match(r"([\d.]+)\s*([KMG]i?B|B|kB)", texto.strip())
    if not m:
        return float("nan")
    valor, unidad = float(m.group(1)), m.group(2)
    factor = {"B": 1 / 2**20, "kB": 1e3 / 2**20, "KiB": 1 / 1024, "KB": 1e3 / 2**20,
              "MiB": 1.0, "MB": 1e6 / 2**20, "GiB": 1024.0, "GB": 1e9 / 2**20}[unidad]
    return valor * factor


def leer_recursos(ruta: Path) -> pd.DataFrame:
    filas = []
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            d = json.loads(linea)
            nombre = d.get("Name", "")
            servicio = re.sub(r"^mibici-equipo32[-_]|[-_]\d+$", "", nombre)
            if servicio.startswith("locust"):
                servicio = "locust"
            filas.append({
                "ts": d["ts"], "servicio": servicio,
                "cpu_pct": float(d.get("CPUPerc", "0%").rstrip("%") or 0),
                "mem_mib": _mib(d.get("MemUsage", "0B / 0B").split("/")[0]),
            })
    return pd.DataFrame(filas, columns=["ts", "servicio", "cpu_pct", "mem_mib"])


def tramos(historial: pd.DataFrame, peticiones: pd.DataFrame) -> list[dict]:
    """Escalones: filas consecutivas del historial de Locust con el mismo número de usuarios."""
    agregado = historial[historial["Name"] == "Aggregated"].sort_values("Timestamp")
    agregado = agregado[agregado["User Count"] > 0]
    resultado = []
    for usuarios, grupo in agregado.groupby((agregado["User Count"].diff() != 0).cumsum()):
        u = int(grupo["User Count"].iloc[0])
        inicio, fin = float(grupo["Timestamp"].min()), float(grupo["Timestamp"].max())
        if resultado and resultado[-1]["usuarios"] == u:
            resultado[-1]["fin"] = fin
            continue
        resultado.append({"usuarios": u, "inicio": inicio, "fin": fin})
    if not resultado:  # corrida muy corta: un solo tramo con todas las peticiones
        resultado = [{"usuarios": 0, "inicio": peticiones["inicio"].min(),
                      "fin": peticiones["inicio"].max()}]
    return resultado


def resumen_tramo(t: dict, pet: pd.DataFrame, rec: pd.DataFrame) -> dict:
    ini, fin = t["inicio"] + ESTABLE_S, t["fin"]
    if fin - ini < 5:  # tramo corto: se usa completo
        ini = t["inicio"]
    ventana = pet[(pet["inicio"] >= ini) & (pet["inicio"] < fin)]
    sinc = ventana[ventana["tipo"] != "ASYNC"]
    dur = max(fin - ini, 1e-9)
    pred = sinc[sinc["nombre"] == "/predict"]
    lote = sinc[sinc["nombre"].str.startswith("/predict/lote")]
    tamano = int(re.search(r"\[(\d+)\]", lote["nombre"].iloc[0]).group(1)) if len(lote) else 0
    api = rec[(rec["servicio"] == "api") & (rec["ts"] >= ini) & (rec["ts"] < fin)]
    q = sinc["ms"].quantile([0.5, 0.95, 0.99]) if len(sinc) else pd.Series([np.nan] * 3)
    qp = pred["ms"].quantile([0.5, 0.95, 0.99]) if len(pred) else pd.Series([np.nan] * 3)
    return {
        "usuarios": t["usuarios"],
        "inicio": round(ini, 1),
        "segundos": round(dur, 1),
        "peticiones": int(len(sinc)),
        "rps": round(len(sinc) / dur, 2),
        "rps_predict": round(len(pred) / dur, 2),
        "inferencias_por_s": round((len(pred) + len(lote) * tamano) / dur, 2),
        "p50_ms": round(float(q.iloc[0]), 1),
        "p95_ms": round(float(q.iloc[1]), 1),
        "p99_ms": round(float(q.iloc[2]), 1),
        "predict_p50_ms": round(float(qp.iloc[0]), 1),
        "predict_p95_ms": round(float(qp.iloc[1]), 1),
        "predict_p99_ms": round(float(qp.iloc[2]), 1),
        "errores_pct": round(100 * (1 - sinc["ok"].mean()), 3) if len(sinc) else 0.0,
        "cpu_api_pct": round(float(api["cpu_pct"].mean()), 1) if len(api) else np.nan,
        "mem_api_mib": round(float(api["mem_mib"].max()), 1) if len(api) else np.nan,
    }


def figura_desempeno(pet, hist, tr, carpeta, t0):
    s = pet[pet["tipo"] != "ASYNC"].copy()
    s["t"] = s["inicio"] - t0
    s["bin"] = (s["t"] // 5) * 5
    s = s[s["bin"] + 5 <= s["t"].max()]  # sin la última ventana incompleta
    g = s.groupby("bin")
    serie = pd.DataFrame({
        "rps": g.size() / 5,
        "p50": g["ms"].quantile(0.5), "p95": g["ms"].quantile(0.95), "p99": g["ms"].quantile(0.99),
        "err": 100 * (1 - g["ok"].mean()),
    })
    ag = hist[hist["Name"] == "Aggregated"].copy()
    ag["t"] = ag["Timestamp"] - t0
    fig, ejes = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.3, 1.6, 0.9]})
    ejes[0].step(ag["t"], ag["User Count"], where="post", color=SERIE, lw=2)
    ejes[0].set_title("Usuarios simulados")
    ejes[1].plot(serie.index, serie["rps"], color=SERIE, lw=2)
    ejes[1].set_title("Peticiones por segundo atendidas")
    for clave, color in PERCENTILES.items():
        ejes[2].plot(serie.index, serie[clave], color=color, lw=2, label=clave)
    ejes[2].set_yscale("log")
    ejes[2].set_title("Latencia (ms, escala logarítmica)")
    ejes[2].legend(loc="upper left", ncols=3)
    ejes[3].plot(serie.index, serie["err"], color=CRITICO, lw=2)
    ejes[3].set_title("Errores (%)")
    ejes[3].set_ylim(bottom=0, top=max(1.0, float(serie["err"].max()) * 1.2))
    ejes[3].set_xlabel("Segundos desde el inicio de la prueba (ventanas de 5 s)")
    for eje in ejes:
        for t in tr[1:]:
            eje.axvline(t["inicio"] - t0, color=REJILLA, lw=0.8, zorder=0)
    fig.align_ylabels(ejes)
    fig.savefig(carpeta / "01_desempeno.png")
    plt.close(fig)


def figura_capacidad(df, slo, carpeta):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = df["usuarios"]
    a.plot(x, df["rps"], color=SERIE, lw=2, marker="o", ms=4)
    a.set_title("Peticiones por segundo")
    a.set_xlabel("Usuarios simulados")
    a.set_ylim(bottom=0)
    b.plot(x, df["p95_ms"], color=SERIE, lw=2, marker="o", ms=4)
    b.axhline(slo, color=CRITICO, lw=1)
    b.annotate(f"SLO p95 = {slo:.0f} ms", (x.min(), slo), xytext=(0, 4),
               textcoords="offset points", color=TEXTO_2, fontsize=8)
    b.set_yscale("log")
    b.set_title("Latencia p95 (ms)")
    b.set_xlabel("Usuarios simulados")
    fig.tight_layout()
    fig.savefig(carpeta / "02_capacidad.png")
    plt.close(fig)


def figura_recursos(rec, hist, carpeta, t0):
    if rec.empty:
        return
    rec = rec.copy()
    rec["t"] = rec["ts"] - t0
    servicios = [s for s in SERVICIOS if s in set(rec["servicio"])]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    for s in servicios:
        d = rec[rec["servicio"] == s].sort_values("t")
        a.plot(d["t"], d["cpu_pct"], color=SERVICIOS[s], lw=2, label=s)
        b.plot(d["t"], d["mem_mib"], color=SERVICIOS[s], lw=2, label=s)
        b.annotate(s, (d["t"].iloc[-1], d["mem_mib"].iloc[-1]), xytext=(4, 0),
                   textcoords="offset points", color=TEXTO_2, va="center", fontsize=8)
    a.set_title("CPU por contenedor (% de un núcleo; docker stats)")
    a.set_ylim(bottom=0)
    a.legend(loc="upper left", ncols=len(servicios))
    b.set_title("Memoria por contenedor (MiB)")
    b.set_ylim(bottom=0)
    b.set_xlabel("Segundos desde el inicio de la prueba")
    fig.align_ylabels((a, b))
    fig.savefig(carpeta / "03_recursos.png")
    plt.close(fig)


def figura_endpoints(pet, tramo, carpeta):
    v = pet[(pet["inicio"] >= tramo["inicio"]) & (pet["inicio"] < tramo["inicio"] + tramo["segundos"])]
    if v.empty:
        return
    q = v.groupby("nombre")["ms"].quantile([0.5, 0.95]).unstack()
    q = q.sort_values(0.95)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(q) + 1.0))
    y = np.arange(len(q))
    ax.barh(y + 0.2, q[0.5], height=0.36, color=PERCENTILES["p50"], label="p50")
    ax.barh(y - 0.2, q[0.95], height=0.36, color=PERCENTILES["p95"], label="p95")
    for i, (p50, p95) in enumerate(zip(q[0.5], q[0.95])):
        ax.annotate(f"{p95:.0f} ms", (p95, i - 0.2), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXTO_2)
    ax.set_yticks(y, q.index)
    ax.set_xlabel("Latencia (ms)")
    ax.set_title(f"Latencia por endpoint con {tramo['usuarios']} usuarios")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(carpeta / "04_endpoints.png")
    plt.close(fig)


def main(carpeta: Path) -> int:
    _estilo()
    entorno = json.loads((carpeta / "entorno.json").read_text(encoding="utf-8"))
    slo = float(entorno.get("slo_p95_ms", 300))
    pet = pd.read_csv(carpeta / "peticiones.csv.gz")
    hist = pd.read_csv(carpeta / "locust_stats_history.csv")
    rec = leer_recursos(carpeta / "recursos.jsonl")
    t0 = float(pet["inicio"].min())
    tr = tramos(hist, pet)
    df = pd.DataFrame([resumen_tramo(t, pet, rec) for t in tr])
    df.to_csv(carpeta / "escalones.csv", index=False)

    cumplen = df[(df["p95_ms"] <= slo) & (df["errores_pct"] < 1.0)]
    capacidad = cumplen.loc[cumplen["rps"].idxmax()].to_dict() if len(cumplen) else None
    reposo = rec[rec["ts"] < t0].groupby("servicio").agg(
        cpu_pct=("cpu_pct", "mean"), mem_mib=("mem_mib", "mean")).round(1)
    pico = rec[rec["ts"] >= t0].groupby("servicio").agg(
        cpu_pct_max=("cpu_pct", "max"), mem_mib_max=("mem_mib", "max")).round(1)
    cpu_api = (capacidad or {}).get("cpu_api_pct", np.nan)
    inferencias = (capacidad or {}).get("inferencias_por_s", np.nan)
    asinc = pet[pet["tipo"] == "ASYNC"]
    resumen = {
        "nombre": entorno["nombre"],
        "escenario": entorno["escenario"],
        "api": entorno["api"],
        "motor": entorno.get("motor"),
        "slo_p95_ms": slo,
        "peticiones_totales": int(len(pet)),
        "errores_totales_pct": round(100 * (1 - pet["ok"].mean()), 3),
        "escalones": df.to_dict(orient="records"),
        "capacidad": capacidad,
        # CPU-segundos que consume la API por inferencia en el escalón de capacidad: la base del
        # costo por inferencia (finops/costo_inferencia.py).
        "cpu_s_por_inferencia": round(cpu_api / 100 / inferencias, 6)
        if capacidad and inferencias else None,
        "reposo": reposo.to_dict(orient="index"),
        "pico": pico.to_dict(orient="index"),
        "asincrono": {
            "tareas": int(len(asinc)),
            "p50_ms": round(float(asinc["ms"].quantile(0.5)), 1),
            "p95_ms": round(float(asinc["ms"].quantile(0.95)), 1),
            "fallidas_pct": round(100 * (1 - asinc["ok"].mean()), 2),
        } if len(asinc) else None,
    }
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False,
                                                     default=float), encoding="utf-8")
    figura_desempeno(pet, hist, tr, carpeta, t0)
    if len(df) > 1:
        figura_capacidad(df, slo, carpeta)
    figura_recursos(rec, hist, carpeta, t0)
    ref = capacidad or df.iloc[-1].to_dict()
    figura_endpoints(pet, ref, carpeta)
    print(df.to_string(index=False))
    if capacidad:
        print(f"Capacidad de una réplica: {capacidad['rps']:.1f} pet/s con {capacidad['usuarios']} "
              f"usuarios (p95 {capacidad['p95_ms']:.0f} ms <= {slo:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
