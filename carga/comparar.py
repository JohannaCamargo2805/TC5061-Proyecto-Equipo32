"""Compara la capacidad de las configuraciones probadas (entrega 6).

    python carga/comparar.py reports/carga

Lee ``escalones.csv`` de las corridas ``base`` (1 proceso, 1 vCPU, pyfunc), ``escalado``
(2 procesos, 2 vCPU) y ``optimizado`` (1 proceso, 1 vCPU, motor NumPy) y dibuja, contra los
usuarios simulados, las peticiones por segundo atendidas y la latencia p95 con el SLO. Escribe
``comparacion_configuraciones.png`` y ``comparacion_configuraciones.csv``.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CORRIDAS = {
    "base": ("Base: 1 proceso, 1 vCPU, pyfunc", "#2a78d6", "o"),
    "escalado": ("Escalado: 2 procesos, 2 vCPU, pyfunc", "#eb6834", "s"),
    "optimizado": ("Optimizado: 1 proceso, 1 vCPU, motor NumPy", "#1baf7a", "^"),
}
SLO_MS = 300
TEXTO_2, REJILLA, CRITICO = "#52514e", "#e6e5e0", "#e34948"


def main(carpeta: Path) -> int:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": REJILLA, "axes.grid": True,
                         "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.color": TEXTO_2,
                         "ytick.color": TEXTO_2, "axes.labelcolor": TEXTO_2,
                         "legend.frameon": False, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    tablas = []
    for clave, (nombre, color, marca) in CORRIDAS.items():
        ruta = carpeta / clave / "escalones.csv"
        if not ruta.exists():
            continue
        df = pd.read_csv(ruta)
        df.insert(0, "corrida", clave)
        tablas.append(df)
        a.plot(df["usuarios"], df["rps"], color=color, marker=marca, ms=4, lw=2, label=nombre)
        b.plot(df["usuarios"], df["p95_ms"], color=color, marker=marca, ms=4, lw=2)
    a.set_title("Peticiones por segundo atendidas")
    a.set_xlabel("Usuarios simulados")
    a.set_ylim(bottom=0)
    b.axhline(SLO_MS, color=CRITICO, lw=1)
    b.annotate(f"SLO p95 = {SLO_MS} ms", (b.get_xlim()[1], SLO_MS), xytext=(-2, -11),
               textcoords="offset points", ha="right", color=TEXTO_2, fontsize=8)
    b.set_yscale("log")
    b.set_title("Latencia p95 (ms, escala log)")
    b.set_xlabel("Usuarios simulados")
    fig.legend(loc="lower center", ncols=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(carpeta / "comparacion_configuraciones.png", dpi=150, bbox_inches="tight")
    pd.concat(tablas).to_csv(carpeta / "comparacion_configuraciones.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/carga")))
