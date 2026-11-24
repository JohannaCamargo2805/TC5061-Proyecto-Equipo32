"""Entrega 5: A/B testing entre dos versiones de P1 registradas en MLflow.

Compara la versión en producción (A, ``@champion``) con la reentrenada (B,
``@challenger``) sobre el MISMO conjunto de evaluación del periodo actual (jul-sep
2026), que ninguna de las dos usó para entrenar ni para elegir hiperparámetros::

    python -m mibici.modeling.ab_testing                       # A=@champion, B=@challenger
    python -m mibici.modeling.ab_testing --a 1 --b 3           # por número de versión
    mlflow run . -e abtest --env-manager local

Metodología (``config.py``, reglas fijadas antes de ver el resultado):

* Diseño **pareado**: cada estación-día recibe la predicción de A y la de B, así que la
  diferencia de errores no depende de qué estaciones o días tocaron a cada brazo.
* Métrica primaria: MAE (la de selección del proyecto); secundarias: RMSE, WAPE, R²,
  sesgo, error del total diario y devianza de Poisson.
* Incertidumbre: **bootstrap por bloques de día** (los errores de las ~440 estaciones de
  un mismo día están correlacionados: clima, eventos), 2000 réplicas, IC 95 %, y prueba
  de Wilcoxon de rangos con signo sobre el MAE diario de A y B.
* Salvaguardas por mes, por tipo de día y por cohorte de estación, sesgo y costo.
* Para contrastar, simula un A/B "en línea" que reparte las estaciones 50/50 por hash:
  muestra por qué el diseño pareado necesita muchos menos datos para decidir.
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_poisson_deviance
import typer

from mibici import monitoreo as mon
from mibici.config import (
    AB_ALFA,
    AB_BOOTSTRAP_REPS,
    AB_EVALUACION,
    AB_MAX_COSTO_RELATIVO,
    AB_TOLERANCIA_MES,
    AB_TOLERANCIA_SESGO,
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    MONITOREO_DIR,
    MONITOREO_REFERENCIA,
    REGISTRY_NAMES,
    SEED,
)
from mibici.features import P1_FEATURES
from mibici.modeling import p1_regresion as p1
from mibici.modeling import seguimiento as seg

app = typer.Typer(add_completion=False)
TIPOS_DIA = {0: "laborable", 1: "fin de semana", 2: "feriado"}


# =============================================================================
# Estadística del A/B
# =============================================================================
def metricas(y: np.ndarray, pred: np.ndarray, fecha: pd.Index) -> dict:
    pred = np.clip(pred, 0, None)
    serie = pd.Series(y, index=pd.MultiIndex.from_arrays([np.arange(len(y)), fecha],
                                                          names=["fila", "fecha"]))
    m = {k.split("_", 1)[1]: v for k, v in p1.evaluar(serie, pred, "x").items()}
    m["devianza_poisson"] = float(mean_poisson_deviance(y, np.clip(pred, 1e-6, None)))
    return m


def bootstrap_dias(err_a: np.ndarray, err_b: np.ndarray, y: np.ndarray, fecha: pd.Index,
                   reps: int = AB_BOOTSTRAP_REPS, semilla: int = SEED) -> pd.DataFrame:
    """Réplicas de MAE_B - MAE_A, RMSE_B - RMSE_A y WAPE_B - WAPE_A remuestreando días."""
    d = pd.DataFrame({"ea": err_a, "eb": err_b, "ea2": err_a ** 2, "eb2": err_b ** 2,
                      "y": y, "n": 1.0, "fecha": fecha})
    por_dia = d.groupby("fecha")[["ea", "eb", "ea2", "eb2", "y", "n"]].sum().to_numpy()
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(por_dia), size=(reps, len(por_dia)))
    s = por_dia[idx].sum(axis=1)  # (reps, 6)
    ea, eb, ea2, eb2, ys, n = s.T
    return pd.DataFrame({"delta_mae": (eb - ea) / n,
                         "delta_rmse": np.sqrt(eb2 / n) - np.sqrt(ea2 / n),
                         "delta_wape": (eb - ea) / ys,
                         "cambio_mae_rel": eb / ea - 1})


def intervalo(x: pd.Series, alfa: float = AB_ALFA) -> tuple[float, float]:
    return float(x.quantile(alfa / 2)), float(x.quantile(1 - alfa / 2))


def asignar_brazo(estacion: int) -> str:
    """Asignación determinista 50/50 por hash del id (como un enrutador de tráfico)."""
    return "A" if int(hashlib.sha256(str(int(estacion)).encode()).hexdigest(), 16) % 2 == 0 else "B"


def simulacion_en_linea(t: pd.DataFrame, reps: int = AB_BOOTSTRAP_REPS,
                        semilla: int = SEED) -> dict:
    """A/B en línea simulado: cada estación ve solo una versión según su brazo."""
    t = t.assign(brazo=t["estacion"].map(asignar_brazo))
    t["err_servido"] = np.where(t["brazo"] == "A", t["err_a"], t["err_b"])
    por_est = t.groupby(["brazo", "estacion"])["err_servido"].agg(["sum", "size"])
    rng = np.random.default_rng(semilla)
    reps_d = []
    for _ in range(reps):
        mae = {}
        for brazo in ("A", "B"):
            g = por_est.loc[brazo].to_numpy()
            r = g[rng.integers(0, len(g), len(g))].sum(axis=0)
            mae[brazo] = r[0] / r[1]
        reps_d.append(mae["B"] - mae["A"])
    mae_brazo = t.groupby("brazo")["err_servido"].mean()
    lo, hi = np.quantile(reps_d, [AB_ALFA / 2, 1 - AB_ALFA / 2])
    return {"estaciones_A": int(t.loc[t["brazo"] == "A", "estacion"].nunique()),
            "estaciones_B": int(t.loc[t["brazo"] == "B", "estacion"].nunique()),
            "mae_brazo_A": float(mae_brazo["A"]), "mae_brazo_B": float(mae_brazo["B"]),
            "delta_mae": float(mae_brazo["B"] - mae_brazo["A"]),
            "ic95": [float(lo), float(hi)], "ancho_ic95": float(hi - lo)}


def segmentos(t: pd.DataFrame) -> pd.DataFrame:
    """MAE de A y B por mes, tipo de día, cohorte de estación y nivel de demanda."""
    t = t.copy()
    t["mes"] = t["fecha"].dt.strftime("%Y-%m")
    t["tipo_dia"] = np.where(t["feriado"] == 1, 2, t["fin_semana"]).astype(int)
    t["tipo_dia"] = t["tipo_dia"].map(TIPOS_DIA)
    t["nivel"] = pd.qcut(t.groupby("estacion")["viajes"].transform("mean"), 4,
                         labels=["Q1 baja", "Q2", "Q3", "Q4 alta"]).astype(str)
    filas = []
    for dimension in ("mes", "tipo_dia", "cohorte", "nivel"):
        for valor, g in t.groupby(dimension):
            filas.append({"dimension": dimension, "segmento": valor, "filas": len(g),
                          "estaciones": g["estacion"].nunique(),
                          "mae_A": g["err_a"].mean(), "mae_B": g["err_b"].mean(),
                          "sesgo_A": (g["pred_a"] - g["viajes"]).mean(),
                          "sesgo_B": (g["pred_b"] - g["viajes"]).mean()})
    s = pd.DataFrame(filas)
    s["cambio_mae_pct"] = 100 * (s["mae_B"] / s["mae_A"] - 1)
    return s


def decidir(m_a: dict, m_b: dict, ic: tuple, seg_df: pd.DataFrame, costo_a: float,
            costo_b: float) -> dict:
    """Aplica las cuatro reglas de config.py y devuelve la versión que debe quedar vigente."""
    meses = seg_df[seg_df["dimension"] == "mes"]
    reglas = {
        "1_ic95_delta_mae_bajo_cero": ic[1] < 0,
        "2_sin_empeorar_mas_de_2pct_en_ningun_mes": bool(
            (meses["cambio_mae_pct"] <= 100 * AB_TOLERANCIA_MES).all()),
        "3_sesgo_no_peor": abs(m_b["sesgo"]) <= abs(m_a["sesgo"]) + AB_TOLERANCIA_SESGO,
        "4_costo_inferencia_aceptable": costo_b <= AB_MAX_COSTO_RELATIVO * max(costo_a, 1e-9),
    }
    gana_b = all(reglas.values())
    return {"reglas": reglas, "ganadora": "B" if gana_b else "A", "promover_B": gana_b}


# =============================================================================
# Gráficas
# =============================================================================
def graficas(t: pd.DataFrame, boot: pd.DataFrame, ic: tuple, seg_df: pd.DataFrame,
             nombres: dict, salida: Path) -> list[Path]:
    rutas = []
    dia = t.groupby("fecha")[["err_a", "err_b", "viajes", "pred_a", "pred_b"]].agg(
        {"err_a": "mean", "err_b": "mean", "viajes": "sum", "pred_a": "sum", "pred_b": "sum"})
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    axs[0].plot(dia.index, dia["viajes"] / 1000, color=mon.OSCURO, lw=2, label="Real")
    axs[0].plot(dia.index, dia["pred_a"] / 1000, color=mon.GRIS, lw=1.5, label=nombres["A"])
    axs[0].plot(dia.index, dia["pred_b"] / 1000, color=mon.AZUL, lw=1.5, label=nombres["B"])
    axs[0].set_ylabel("Miles de viajes / día")
    axs[0].set_title("Total diario del sistema: real contra cada versión")
    axs[1].plot(dia.index, dia["err_a"], color=mon.GRIS, lw=1.5, label=nombres["A"])
    axs[1].plot(dia.index, dia["err_b"], color=mon.AZUL, lw=1.5, label=nombres["B"])
    axs[1].set_ylabel("MAE del día (viajes por estación)")
    axs[1].set_title("MAE diario de A y B sobre las mismas estación-días")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.autofmt_xdate()
    rutas.append(salida / "ab_serie_diaria.png")
    fig.savefig(rutas[-1], dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
    axs[0].hist(boot["delta_mae"], bins=50, color=mon.AZUL, alpha=0.85)
    for v in ic:
        axs[0].axvline(v, color=mon.OSCURO, ls="--", lw=1)
    axs[0].axvline(0, color=mon.ROJO, lw=1.5, label="sin diferencia")
    axs[0].set_title(f"Bootstrap por días: MAE_B − MAE_A\nIC 95 % [{ic[0]:.3f}, {ic[1]:.3f}]",
                     fontsize=10)
    axs[0].set_xlabel("viajes por estación-día (negativo = B mejor)")
    axs[0].legend(fontsize=8)
    s = seg_df[seg_df["dimension"].isin(["mes", "cohorte", "tipo_dia"])]
    x = np.arange(len(s))
    axs[1].bar(x - 0.2, s["mae_A"], 0.4, color=mon.GRIS, label=nombres["A"])
    axs[1].bar(x + 0.2, s["mae_B"], 0.4, color=mon.AZUL, label=nombres["B"])
    axs[1].set_xticks(x, s["segmento"], rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("MAE")
    axs[1].set_title("MAE por segmento", fontsize=10)
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    rutas.append(salida / "ab_bootstrap_segmentos.png")
    fig.savefig(rutas[-1], dpi=160, bbox_inches="tight")
    plt.close(fig)
    return rutas


# =============================================================================
# Orquestación
# =============================================================================
def uri(ref: str) -> str:
    """``champion`` -> models:/<p1>@champion ; ``3`` -> models:/<p1>/3 ; o una URI completa."""
    if ref.startswith("models:/") or Path(ref).exists():
        return ref
    nombre = REGISTRY_NAMES["p1"]
    return f"models:/{nombre}/{ref}" if ref.isdigit() else f"models:/{nombre}@{ref}"


def version_de(client: MlflowClient | None, ref: str) -> dict:
    if client is None or not uri(ref).startswith("models:/"):
        return {"version": None, "run_id": None, "ref": ref}
    nombre = REGISTRY_NAMES["p1"]
    v = (client.get_model_version(nombre, ref) if ref.isdigit()
         else client.get_model_version_by_alias(nombre, ref))
    return {"version": int(v.version), "run_id": v.run_id, "ref": ref,
            "aliases": list(getattr(v, "aliases", []) or [])}


def comparar(a: str = ALIAS_PRODUCCION, b: str = ALIAS_RETADOR, salida: Path = MONITOREO_DIR,
             client: MlflowClient | None = None) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    info = {"A": version_de(client, a), "B": version_de(client, b)}
    nombres = {k: f"{k}: v{v['version']}" if v["version"] else f"{k}: {v['ref']}"
               for k, v in info.items()}
    modelos = {k: mlflow.sklearn.load_model(uri(r)) for k, r in (("A", a), ("B", b))}

    panel = mon.construir_panel(fin=AB_EVALUACION[1])
    ev = mon.periodo(panel, *AB_EVALUACION)
    x = ev[P1_FEATURES].astype("float64")
    t = ev[P1_FEATURES + ["viajes"]].reset_index()
    costos = {}
    for k, m in modelos.items():
        t[f"pred_{k.lower()}"] = np.clip(m.predict(x), 0, None)
        costos[k] = seg.costo_inferencia_ms_1k(m, x)
    t["err_a"] = (t["viajes"] - t["pred_a"]).abs()
    t["err_b"] = (t["viajes"] - t["pred_b"]).abs()
    primera = panel.reset_index().groupby("estacion")["fecha"].min()
    t["cohorte"] = np.where(t["estacion"].map(primera) >= pd.Timestamp(MONITOREO_REFERENCIA[1]),
                            "estación nueva (abrió después del TRAIN de v1)", "estación existente")

    y, fecha = t["viajes"].to_numpy(), pd.Index(t["fecha"])
    m_a, m_b = metricas(y, t["pred_a"].to_numpy(), fecha), metricas(y, t["pred_b"].to_numpy(), fecha)
    boot = bootstrap_dias(t["err_a"].to_numpy(), t["err_b"].to_numpy(), y, fecha)
    ic = intervalo(boot["delta_mae"])
    dia = t.groupby("fecha")[["err_a", "err_b"]].mean()
    wilcoxon = stats.wilcoxon(dia["err_b"], dia["err_a"], alternative="two-sided")
    por_est = t.groupby("estacion")[["err_a", "err_b"]].mean()
    seg_df = segmentos(t)
    decision = decidir(m_a, m_b, ic, seg_df, costos["A"], costos["B"])
    en_linea = simulacion_en_linea(t)

    tabla = pd.DataFrame({"A": m_a, "B": m_b}).T
    tabla["costo_inferencia_ms_1k"] = [costos["A"], costos["B"]]
    tabla.insert(0, "version", [info["A"]["version"], info["B"]["version"]])
    tabla["diferencia"] = None
    resultado = {
        "evaluacion": {"desde": AB_EVALUACION[0], "hasta_excl": AB_EVALUACION[1], "filas": len(t),
                       "dias": int(t["fecha"].nunique()), "estaciones": int(t["estacion"].nunique())},
        "versiones": info,
        "metricas": {"A": m_a, "B": m_b},
        "costo_inferencia_ms_1k": costos,
        "delta": {"mae": m_b["mae"] - m_a["mae"], "mae_rel_pct": 100 * (m_b["mae"] / m_a["mae"] - 1),
                  "rmse": m_b["rmse"] - m_a["rmse"], "wape": m_b["wape"] - m_a["wape"]},
        "bootstrap": {"reps": AB_BOOTSTRAP_REPS, "unidad": "día",
                      "ic95_delta_mae": list(ic),
                      "ic95_delta_rmse": list(intervalo(boot["delta_rmse"])),
                      "ic95_delta_wape": list(intervalo(boot["delta_wape"])),
                      "ic95_cambio_mae_rel_pct": [100 * v for v in intervalo(boot["cambio_mae_rel"])],
                      "p_bootstrap_b_no_mejor": float((boot["delta_mae"] >= 0).mean())},
        "wilcoxon_mae_diario": {"estadistico": float(wilcoxon.statistic),
                                "p_valor": float(wilcoxon.pvalue),
                                "dias_b_mejor": int((dia["err_b"] < dia["err_a"]).sum()),
                                "dias": len(dia)},
        "estaciones_b_mejor_pct": float(100 * (por_est["err_b"] < por_est["err_a"]).mean()),
        "simulacion_en_linea": en_linea,
        "decision": decision,
    }
    coef = mon.terminos_glm(modelos["A"]).merge(mon.terminos_glm(modelos["B"]), on="termino",
                                                 how="outer", suffixes=("_A", "_B"))
    if not coef.empty:
        coef["cambio"] = coef["coeficiente_B"] - coef["coeficiente_A"]
        coef.to_csv(salida / "ab_coeficientes.csv", index=False)
    tabla.drop(columns="diferencia").to_csv(salida / "ab_metricas.csv")
    seg_df.to_csv(salida / "ab_segmentos.csv", index=False)
    boot.describe().to_csv(salida / "ab_bootstrap_resumen.csv")
    (salida / "ab_resultado.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False,
                                                         default=float), encoding="utf-8")
    rutas = graficas(t, boot, ic, seg_df, nombres, salida)
    logger.success(f"A/B: MAE A={m_a['mae']:.4f} B={m_b['mae']:.4f} "
                   f"ΔMAE IC95=[{ic[0]:.4f}, {ic[1]:.4f}] -> gana {decision['ganadora']}")
    return {"resultado": resultado, "graficas": rutas, "salida": salida}


def registrar_en_mlflow(res: dict, client: MlflowClient) -> str:
    r = res["resultado"]
    manifiesto = seg.leer_manifiesto()
    etiquetas = seg.etiquetas_base("p1", "ab_testing", "ab_test", manifiesto["data_version"])
    va, vb = r["versiones"]["A"]["version"], r["versiones"]["B"]["version"]
    etiquetas.update({"entrega": "monitoreo-ab", "version_A": str(va), "version_B": str(vb),
                      "ganadora": r["decision"]["ganadora"],
                      "data_version_produccion": mon.leer_manifiesto_produccion()["data_version"]})
    nombre = f"p1-ab-test-v{va}-vs-v{vb}"
    with seg.corrida(nombre, etiquetas) as run:
        mlflow.log_params({"evaluacion": "{} a {}".format(*AB_EVALUACION),
                           "A": f"v{va} ({r['versiones']['A']['ref']})",
                           "B": f"v{vb} ({r['versiones']['B']['ref']})",
                           "metrica_primaria": "mae", "bootstrap_reps": AB_BOOTSTRAP_REPS,
                           "bootstrap_unidad": "dia", "alfa": AB_ALFA})
        for k in ("A", "B"):
            mlflow.log_metrics({f"{k}_{m}": v for m, v in r["metricas"][k].items()})
        mlflow.log_metrics({"delta_mae": r["delta"]["mae"], "delta_mae_rel_pct": r["delta"]["mae_rel_pct"],
                            "ic95_delta_mae_inf": r["bootstrap"]["ic95_delta_mae"][0],
                            "ic95_delta_mae_sup": r["bootstrap"]["ic95_delta_mae"][1],
                            "wilcoxon_p": r["wilcoxon_mae_diario"]["p_valor"],
                            "estaciones_b_mejor_pct": r["estaciones_b_mejor_pct"],
                            "promover_B": float(r["decision"]["promover_B"])})
        mlflow.log_artifacts(str(res["salida"]), artifact_path="ab_test")
        run_id = run.info.run_id
    nombre_reg = REGISTRY_NAMES["p1"]
    for k, v in (("A", va), ("B", vb)):
        if v is not None:
            gana = r["decision"]["ganadora"] == k
            client.set_model_version_tag(nombre_reg, str(v), "ab_test_jul_sep_2026",
                                         f"{'gana' if gana else 'pierde'} (corrida {run_id})")
    return run_id


@app.command()
def main(
    a: str = typer.Option(ALIAS_PRODUCCION, help="Versión A: alias, número o URI"),
    b: str = typer.Option(ALIAS_RETADOR, help="Versión B: alias, número o URI"),
    mlflow_log: bool = typer.Option(True, "--mlflow/--sin-mlflow"),
):
    """Compara dos versiones de P1 en jul-sep 2026 y decide cuál debe quedar vigente."""
    client = None
    if uri(a).startswith("models:/") or uri(b).startswith("models:/"):
        seg.configurar_tracking()
        client = MlflowClient()
    res = comparar(a, b, client=client)
    if mlflow_log and client is not None:
        logger.success(f"Corrida del A/B registrada: {registrar_en_mlflow(res, client)}")


if __name__ == "__main__":
    app()
