"""Entrega 5: monitoreo del modelo P1 con Evidently (data drift y calidad de datos).

El modelo en producción (``mibici-equipo32-p1-demanda@champion``, v1) se entrenó con
ene-sep 2025. Este módulo compara esos datos de entrenamiento (conjunto de
**referencia**) con un periodo posterior de los datos abiertos de MiBici (conjunto
**actual**, jul-sep 2026) y deja tres reportes de Evidently y sus tablas::

    python -m mibici.monitoreo ingesta --raw-source <carpeta con los CSV de 2026>
    python -m mibici.monitoreo drift                 # requiere el servidor de MLflow
    mlflow run . -e monitor --env-manager local      # lo mismo como punto de entrada

Reportes (``reports/monitoreo/``):

* ``drift_datos.html``: ``DataDriftPreset`` (drift por variable, la objetivo y la
  predicción de la v1) + ``DataSummaryPreset`` (calidad de datos) con pruebas.
* ``drift_psi.html``: el mismo drift con el índice de estabilidad poblacional (PSI),
  como segunda opinión independiente del método por defecto.
* ``desempeno_v1.html``: ``RegressionPreset`` de la v1 en su periodo de prueba
  (nov-dic 2025) contra el periodo actual: cuánto se degrada el error.

Las variables se construyen con las mismas funciones de ``mibici.features`` que usó el
entrenamiento; el panel se arma de ene 2025 a sep 2026 de forma continua, así los
rezagos de los primeros días de julio usan los viajes reales de junio.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    DRIFT_PSI_UMBRAL,
    DRIFT_UMBRAL_COLUMNA,
    DRIFT_UMBRAL_CONJUNTO,
    ESTACIONES_PARQUET,
    MANIFEST_PRODUCCION_PATH,
    MONITOREO_ACTUAL,
    MONITOREO_DIR,
    MONITOREO_REFERENCIA,
    P1_CATEGORICAS,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VENTANAS_MEDIA_DIAS,
    PROCESSED_DATA_DIR,
    PRODUCCION_URLS,
    RAW_DATA_DIR,
    REGISTRY_NAMES,
    VIAJES_PARQUET,
    VIAJES_PRODUCCION_PARQUET,
)
from mibici.dataset import _sha256, descargar, leer_csv, normalizar_viajes
from mibici.features import P1_FEATURES, p1_agregar_features, p1_construir_panel_diario
from mibici.modeling import p1_regresion as p1

app = typer.Typer(add_completion=False)

# Evidently envía telemetría de uso solo desde su interfaz web; se desactiva igual.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")

OBJETIVO = "viajes"
PREDICCION = "prediccion_v1"
NUMERICAS = [c for c in P1_FEATURES if c not in P1_CATEGORICAS]
AZUL, GRIS, OSCURO, ROJO, NARANJA = "#2563eb", "#9ca3af", "#1f2937", "#dc2626", "#ea580c"


# =============================================================================
# 1. Datos de producción (2026)
# =============================================================================
def ingesta_produccion(raw_source: Path | None = None) -> dict:
    """Ingiere los meses de ``PRODUCCION_URLS`` con el mismo código que la entrega 1.

    Escribe ``data/processed/viajes_produccion.parquet`` y un manifiesto aparte con el
    SHA-256 de cada CSV, de modo que la ``data_version`` de 2025 no cambia.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = {"generado": datetime.now(timezone.utc).isoformat(), "archivos": {}}
    partes = []
    for mes, url in PRODUCCION_URLS.items():
        ruta = descargar(url, RAW_DATA_DIR / f"viajes_{mes}.csv", raw_source)
        df = normalizar_viajes(leer_csv(ruta), origen=mes)
        manifiesto["archivos"][ruta.name] = {"url": url, "sha256": _sha256(ruta),
                                             "registros": len(df)}
        partes.append(df)
    viajes = pd.concat(partes, ignore_index=True)
    viajes.to_parquet(VIAJES_PRODUCCION_PARQUET, index=False)
    hashes = sorted(v["sha256"] for v in manifiesto["archivos"].values())
    manifiesto["data_version"] = hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]
    manifiesto["total_viajes"] = len(viajes)
    if MANIFEST_PRODUCCION_PATH.exists():
        anterior = json.loads(MANIFEST_PRODUCCION_PATH.read_text(encoding="utf-8"))
        if {k: v for k, v in anterior.items() if k != "generado"} == {
                k: v for k, v in manifiesto.items() if k != "generado"}:
            manifiesto["generado"] = anterior["generado"]
    MANIFEST_PRODUCCION_PATH.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                        encoding="utf-8", newline="\n")
    logger.success(f"Producción 2026: {len(viajes):,} viajes, "
                   f"data_version={manifiesto['data_version']}")
    return manifiesto


def leer_manifiesto_produccion() -> dict:
    if not MANIFEST_PRODUCCION_PATH.exists():
        raise FileNotFoundError("Falta data/manifest_produccion.json: "
                                "python -m mibici.monitoreo ingesta --raw-source <carpeta>")
    return json.loads(MANIFEST_PRODUCCION_PATH.read_text(encoding="utf-8"))


# =============================================================================
# 2. Panel de P1 de ene 2025 a sep 2026 y conjuntos de referencia y actual
# =============================================================================
def construir_panel(fin: str = MONITOREO_ACTUAL[1],
                    viajes_2025: Path = VIAJES_PARQUET,
                    viajes_2026: Path = VIAJES_PRODUCCION_PARQUET) -> pd.DataFrame:
    """Panel estación × día con las 12 variables de P1, continuo desde ene 2025.

    Usa ``cargar_viajes`` (filtro de duración 60 s - 24 h) y las funciones de
    ``mibici.features``: es exactamente la construcción del entrenamiento.
    """
    partes = [p1.cargar_viajes(viajes_2025)]
    if Path(viajes_2026).exists():
        partes.append(p1.cargar_viajes(viajes_2026))
    viajes = pd.concat(partes, ignore_index=True)
    capacidad = p1.cargar_capacidad(ESTACIONES_PARQUET)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, fin)
    return p1_agregar_features(panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS,
                               P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS)


def periodo(panel: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filas de ``[inicio, fin)`` con las 12 variables completas, como en el entrenamiento."""
    return p1.cortar(panel, inicio, fin).dropna(subset=P1_FEATURES)


def a_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas planas (estación, fecha, variables y objetivo) con tipos estables."""
    t = df[P1_FEATURES + [OBJETIVO]].reset_index()
    for c in NUMERICAS + [OBJETIVO]:
        t[c] = t[c].astype("float64")
    for c in P1_CATEGORICAS:
        t[c] = t[c].astype("int64")
    return t


def conjuntos(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Referencia (TRAIN de la v1), actual (jul-sep 2026) y prueba de la v1 (nov-dic 2025)."""
    return {
        "referencia": a_tabla(periodo(panel, *MONITOREO_REFERENCIA)),
        "actual": a_tabla(periodo(panel, *MONITOREO_ACTUAL)),
        "prueba_v1": a_tabla(periodo(panel, P1_TEST_START, P1_TEST_END)),
    }


def describir(nombre: str, t: pd.DataFrame) -> dict:
    fecha = t["fecha"]
    return {"conjunto": nombre, "desde": str(fecha.min().date()), "hasta": str(fecha.max().date()),
            "dias": int(fecha.nunique()), "filas": len(t), "estaciones": int(t["estacion"].nunique()),
            "viajes_totales": int(t[OBJETIVO].sum()),
            "media_viajes_estacion_dia": round(float(t[OBJETIVO].mean()), 3)}


# =============================================================================
# 3. Reportes de Evidently
# =============================================================================
def definicion(con_prediccion: bool = True):
    """DataDefinition de Evidently: tipos de columna, objetivo y predicción de regresión."""
    from evidently import DataDefinition, Regression

    return DataDefinition(
        id_column="estacion",
        numerical_columns=NUMERICAS + [OBJETIVO] + ([PREDICCION] if con_prediccion else []),
        categorical_columns=list(P1_CATEGORICAS),
        timestamp="fecha",
        regression=[Regression(target=OBJETIVO, prediction=PREDICCION)] if con_prediccion else None,
    )


def _dataset(t: pd.DataFrame, con_prediccion: bool = True):
    from evidently import Dataset

    columnas = ["estacion", "fecha"] + P1_FEATURES + [OBJETIVO] + ([PREDICCION] if con_prediccion else [])
    return Dataset.from_pandas(t[columnas], data_definition=definicion(con_prediccion))


def reporte_drift(referencia: pd.DataFrame, actual: pd.DataFrame, metodo: str | None = None,
                  umbral: float | None = None, con_resumen: bool = True):
    """Corre ``DataDriftPreset`` (y ``DataSummaryPreset``) y devuelve el *snapshot*.

    ``metodo=None`` usa el criterio por defecto de Evidently para muestras grandes
    (Wasserstein normalizada / Jensen-Shannon, umbral 0.1); ``metodo="psi"`` usa PSI.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    columnas = P1_FEATURES + [OBJETIVO] + ([PREDICCION] if PREDICCION in actual else [])
    drift = DataDriftPreset(columns=columnas, method=metodo, threshold=umbral,
                            drift_share=DRIFT_UMBRAL_CONJUNTO)
    metricas = [drift, DataSummaryPreset()] if con_resumen else [drift]
    reporte = Report(metricas, include_tests=True)
    con_pred = PREDICCION in actual
    return reporte.run(_dataset(actual, con_pred), _dataset(referencia, con_pred))


def reporte_desempeno(prueba: pd.DataFrame, actual: pd.DataFrame):
    """``RegressionPreset``: la v1 en su prueba (nov-dic 2025) contra jul-sep 2026."""
    from evidently import Report
    from evidently.presets import RegressionPreset

    return Report([RegressionPreset()], include_tests=False).run(_dataset(actual), _dataset(prueba))


def tabla_drift(snapshot, metodo_esperado: str | None = None) -> pd.DataFrame:
    """Puntaje, método, umbral y veredicto de drift por columna a partir del snapshot."""
    filas = []
    for m in snapshot.dict()["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            filas.append({"columna": cfg["column"], "metodo": cfg.get("method"),
                          "umbral": cfg.get("threshold"), "puntaje": float(m["value"])})
    t = pd.DataFrame(filas)
    if t.empty:
        raise ValueError("El snapshot no trae métricas ValueDrift")
    # Para p-valores el drift es puntaje < umbral; para distancias, puntaje >= umbral.
    es_pvalor = t["metodo"].fillna("").str.contains("p_value|test", case=False)
    t["drift"] = np.where(es_pvalor, t["puntaje"] < t["umbral"], t["puntaje"] >= t["umbral"])
    t["tipo"] = np.where(t["columna"].isin(P1_CATEGORICAS), "categórica", "numérica")
    return t


def resumen_conjunto(snapshot) -> dict:
    """Número y proporción de columnas con drift (métrica DriftedColumnsCount)."""
    for m in snapshot.dict()["metrics"]:
        if m.get("config", {}).get("type", "").endswith("DriftedColumnsCount"):
            v = m["value"]
            return {"columnas_con_drift": int(v["count"]), "proporcion_con_drift": float(v["share"])}
    raise ValueError("El snapshot no trae DriftedColumnsCount")


def resultados_pruebas(snapshot) -> pd.DataFrame:
    """Pruebas (tests) de Evidently con su estado: SUCCESS, FAIL, WARNING o ERROR."""
    pruebas = snapshot.dict().get("tests", [])
    return pd.DataFrame([{"prueba": t.get("name"), "estado": t.get("status"),
                          "descripcion": t.get("description")} for t in pruebas])


# =============================================================================
# 4. Estadísticos de interpretación y efecto sobre la v1
# =============================================================================
def comparar_estadisticos(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana, p95 y fracción de valores fuera del rango de entrenamiento."""
    filas = []
    for c in NUMERICAS + [OBJETIVO] + ([PREDICCION] if PREDICCION in cur else []):
        lo, hi = ref[c].min(), ref[c].max()
        filas.append({
            "columna": c, "media_ref": ref[c].mean(), "media_act": cur[c].mean(),
            "cambio_media_pct": 100 * (cur[c].mean() / ref[c].mean() - 1) if ref[c].mean() else np.nan,
            "mediana_ref": ref[c].median(), "mediana_act": cur[c].median(),
            "p95_ref": ref[c].quantile(0.95), "p95_act": cur[c].quantile(0.95),
            "max_ref": hi, "max_act": cur[c].max(),
            "fuera_rango_pct": 100 * float(((cur[c] < lo) | (cur[c] > hi)).mean()),
        })
    for c in P1_CATEGORICAS:
        filas.append({"columna": c, "media_ref": ref[c].mean(), "media_act": cur[c].mean(),
                      "cambio_media_pct": 100 * (cur[c].mean() / ref[c].mean() - 1)
                      if ref[c].mean() else np.nan,
                      "fuera_rango_pct": 100 * float((~cur[c].isin(ref[c].unique())).mean())})
    return pd.DataFrame(filas)


def terminos_glm(modelo) -> pd.DataFrame:
    """Coeficientes del GLM Poisson por término transformado (vacío si no es un GLM)."""
    try:
        prep, glm = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    except (AttributeError, KeyError):
        return pd.DataFrame(columns=["termino", "coeficiente"])
    nombres = []
    for nombre, trans, cols in prep.transformers_:
        if nombre == "remainder" or trans == "drop":
            continue
        if hasattr(trans, "categories_"):  # OneHotEncoder del día de la semana
            nombres += [f"{nombre}__{n}" for n in trans.get_feature_names_out(cols)]
        else:
            nombres += [f"{nombre}__{c}" for c in cols]
    return pd.DataFrame({"termino": nombres, "coeficiente": glm.coef_})


def efecto_en_modelo(modelo, ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Cuánto mueve el drift de cada variable la predicción de la v1 (Poisson lineal).

    En un GLM Poisson con enlace log, ``log(mu) = b0 + sum(b_j * z_j)``. Si la media de
    la variable transformada ``z_j`` se desplaza en ``dz_j`` del conjunto de referencia
    al actual, la predicción media se multiplica por ``exp(b_j * dz_j)`` con todo lo
    demás fijo. Para modelos sin coeficientes devuelve una tabla vacía.
    """
    t = terminos_glm(modelo)
    if t.empty:
        return pd.DataFrame()
    prep, nombres = modelo.named_steps["prep"], t["termino"].tolist()
    zr = pd.DataFrame(prep.transform(ref[P1_FEATURES].astype("float64")), columns=nombres)
    zc = pd.DataFrame(prep.transform(cur[P1_FEATURES].astype("float64")), columns=nombres)
    t = t.assign(z_media_ref=zr.mean().to_numpy(), z_media_act=zc.mean().to_numpy())
    t["delta_z"] = t["z_media_act"] - t["z_media_ref"]
    t["factor_prediccion"] = np.exp(t["coeficiente"] * t["delta_z"])
    t["variable"] = t["termino"].str.split("__").str[-1]
    t.loc[t["termino"].str.startswith("dia__"), "variable"] = "dia_semana"
    return t.sort_values("factor_prediccion", key=lambda s: (np.log(s)).abs(), ascending=False)


# =============================================================================
# 5. Gráficas propias (complementan las de Evidently en el reporte)
# =============================================================================
def grafica_puntajes(tabla: pd.DataFrame, tabla_psi: pd.DataFrame, ruta: Path) -> Path:
    t = tabla.set_index("columna")[["puntaje", "drift"]].join(
        tabla_psi.set_index("columna")[["puntaje", "drift"]], rsuffix="_psi")
    t = t.sort_values("puntaje")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, col, umbral, titulo in ((axs[0], "puntaje", DRIFT_UMBRAL_COLUMNA,
                                     "Wasserstein normalizada / Jensen-Shannon"),
                                    (axs[1], "puntaje_psi", DRIFT_PSI_UMBRAL, "PSI")):
        colores = [ROJO if v >= umbral else AZUL for v in t[col]]
        ax.barh(t.index, t[col], color=colores)
        ax.axvline(umbral, color=OSCURO, ls="--", lw=1, label=f"umbral {umbral}")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_title(titulo, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Drift por variable: referencia ene-sep 2025 vs. actual jul-sep 2026 "
                 "(rojo = drift)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafica_distribuciones(ref: pd.DataFrame, cur: pd.DataFrame, ruta: Path) -> Path:
    cols = ["lag_1", "media_28d", "dias_activa", "capacidad", OBJETIVO,
            PREDICCION if PREDICCION in cur else "media_7d"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, c in zip(axs.ravel(), cols):
        hi = float(np.nanpercentile(np.r_[ref[c], cur[c]], 99.5))
        bins = np.linspace(0, hi if hi > 0 else 1, 50)
        ax.hist(ref[c].clip(upper=hi), bins=bins, density=True, alpha=0.55, color=GRIS,
                label="referencia (ene-sep 2025)")
        ax.hist(cur[c].clip(upper=hi), bins=bins, density=True, alpha=0.55, color=AZUL,
                label="actual (jul-sep 2026)")
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Distribuciones de referencia y actuales (densidad)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafica_serie(panel: pd.DataFrame, ruta: Path) -> Path:
    """Viajes del sistema por semana de ene 2025 a sep 2026 con los periodos marcados."""
    total = panel["viajes"].groupby(level="fecha").sum()
    semanal = total.resample("W").sum().iloc[1:-1]
    activas = (panel["viajes"] > 0).groupby(level="fecha").sum().resample("W").mean().iloc[1:-1]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(semanal.index, semanal / 1000, color=OSCURO, lw=1.8)
    for (a, b), color, nombre in ((MONITOREO_REFERENCIA, GRIS, "referencia (TRAIN v1)"),
                                  (MONITOREO_ACTUAL, AZUL, "actual")):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=color, alpha=0.18, label=nombre)
    ax.set_ylabel("Miles de viajes por semana")
    ax2 = ax.twinx()
    ax2.plot(activas.index, activas, color=NARANJA, lw=1.2, ls="--", label="estaciones con viajes/día")
    ax2.set_ylabel("Estaciones con al menos 1 viaje", color=NARANJA)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("Demanda semanal del sistema MiBici, ene 2025 - sep 2026")
    ax.grid(alpha=0.3)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


# =============================================================================
# 6. Orquestación
# =============================================================================
def cargar_modelo(uri: str):
    """Modelo de scikit-learn por URI de MLflow (``models:/...@alias``) o carpeta local."""
    import mlflow

    return mlflow.sklearn.load_model(uri)


def monitorear(modelo_uri: str, salida: Path = MONITOREO_DIR) -> dict:
    """Construye los conjuntos, corre los tres reportes y guarda tablas y gráficas."""
    salida.mkdir(parents=True, exist_ok=True)
    panel = construir_panel()
    datos = conjuntos(panel)
    modelo = cargar_modelo(modelo_uri)
    for t in datos.values():
        t[PREDICCION] = np.clip(modelo.predict(t[P1_FEATURES].astype("float64")), 0, None)
    ref, cur, prueba = datos["referencia"], datos["actual"], datos["prueba_v1"]
    descripcion = pd.DataFrame([describir(k, v) for k, v in datos.items()])
    logger.info("\n" + descripcion.to_string(index=False))

    snap = reporte_drift(ref, cur)
    snap.save_html(str(salida / "drift_datos.html"))
    (salida / "drift_datos.json").write_text(snap.json(), encoding="utf-8")
    snap_psi = reporte_drift(ref, cur, metodo="psi", umbral=DRIFT_PSI_UMBRAL, con_resumen=False)
    snap_psi.save_html(str(salida / "drift_psi.html"))
    # Mismo drift solo con las estaciones que ya existían en la referencia: separa el
    # efecto de composición (estaciones nuevas) del cambio de comportamiento.
    existentes = cur[cur["estacion"].isin(set(ref["estacion"]))]
    snap_ex = reporte_drift(ref, existentes, con_resumen=False)
    snap_ex.save_html(str(salida / "drift_estaciones_existentes.html"))
    snap_des = reporte_desempeno(prueba, cur)
    snap_des.save_html(str(salida / "desempeno_v1.html"))
    (salida / "desempeno_v1.json").write_text(snap_des.json(), encoding="utf-8")

    tabla = tabla_drift(snap)
    tabla_psi = tabla_drift(snap_psi)
    tabla = tabla.merge(tabla_psi[["columna", "puntaje", "drift"]].rename(
        columns={"puntaje": "psi", "drift": "drift_psi"}), on="columna", how="left")
    tabla = tabla.merge(tabla_drift(snap_ex)[["columna", "puntaje", "drift"]].rename(
        columns={"puntaje": "puntaje_existentes", "drift": "drift_existentes"}), on="columna",
        how="left")
    estad = comparar_estadisticos(ref, cur)
    tabla = tabla.merge(estad, on="columna", how="left").sort_values("puntaje", ascending=False)
    tabla.to_csv(salida / "drift_por_variable.csv", index=False)
    efecto = efecto_en_modelo(modelo, ref, cur)
    efecto.to_csv(salida / "efecto_drift_v1.csv", index=False)
    pruebas = resultados_pruebas(snap)
    pruebas.to_csv(salida / "pruebas_evidently.csv", index=False)
    descripcion.to_csv(salida / "conjuntos.csv", index=False)

    rendimiento = {}
    for nombre, t in (("referencia", ref), ("prueba_v1", prueba), ("actual", cur)):
        y = t[OBJETIVO].set_axis(pd.MultiIndex.from_frame(t[["estacion", "fecha"]]))
        rendimiento[nombre] = p1.evaluar(y, t[PREDICCION].to_numpy(), "v1")

    graficas = [grafica_puntajes(tabla, tabla_psi, salida / "drift_puntajes.png"),
                grafica_distribuciones(ref, cur, salida / "drift_distribuciones.png"),
                grafica_serie(panel, salida / "demanda_semanal.png")]
    resumen = {
        "modelo": modelo_uri,
        "conjuntos": descripcion.to_dict(orient="records"),
        "criterio": {"metodo": "Evidently por defecto (Wasserstein normalizada / Jensen-Shannon)",
                     "umbral_columna": DRIFT_UMBRAL_COLUMNA, "umbral_conjunto": DRIFT_UMBRAL_CONJUNTO,
                     "psi_umbral": DRIFT_PSI_UMBRAL},
        **resumen_conjunto(snap),
        "columnas_evaluadas": len(tabla),
        "conjunto_con_drift": resumen_conjunto(snap)["proporcion_con_drift"] >= DRIFT_UMBRAL_CONJUNTO,
        "con_drift": tabla.loc[tabla["drift"], "columna"].tolist(),
        "con_drift_psi": tabla.loc[tabla["drift_psi"].fillna(False).astype(bool), "columna"].tolist(),
        "estaciones_existentes": {**resumen_conjunto(snap_ex), "filas": len(existentes),
                                  "con_drift": tabla.loc[tabla["drift_existentes"].fillna(False)
                                                         .astype(bool), "columna"].tolist()},
        "pruebas_evidently": pruebas["estado"].value_counts().to_dict() if len(pruebas) else {},
        "desempeno_v1": rendimiento,
    }
    (salida / "resumen_drift.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    logger.success(f"Drift: {resumen['columnas_con_drift']} de {len(tabla)} columnas "
                   f"({resumen['proporcion_con_drift']:.0%}); con drift: {resumen['con_drift']}")
    return {"resumen": resumen, "tabla": tabla, "efecto": efecto, "graficas": graficas,
            "salida": salida}


def registrar_en_mlflow(res: dict, modelo_uri: str) -> str:
    """Corrida ``p1-monitoreo-drift`` en el experimento del equipo con métricas y reportes."""
    import mlflow

    from mibici.modeling import seguimiento as seg

    seg.configurar_tracking()
    manifiesto = seg.leer_manifiesto()
    prod = leer_manifiesto_produccion()
    etiquetas = seg.etiquetas_base("p1", "monitoreo", "monitoring", manifiesto["data_version"])
    etiquetas.update({"entrega": "monitoreo-ab", "data_version_produccion": prod["data_version"],
                      "modelo_monitoreado": modelo_uri})
    resumen, salida = res["resumen"], res["salida"]
    nombre = "p1-monitoreo-drift"
    with mlflow.start_run(run_name=nombre) if not os.getenv("MLFLOW_RUN_ID") else \
            mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", nombre)
        mlflow.set_tags(etiquetas)
        mlflow.log_params({"referencia": "{} a {}".format(*MONITOREO_REFERENCIA),
                           "actual": "{} a {}".format(*MONITOREO_ACTUAL),
                           "umbral_columna": DRIFT_UMBRAL_COLUMNA,
                           "umbral_conjunto": DRIFT_UMBRAL_CONJUNTO, "psi_umbral": DRIFT_PSI_UMBRAL,
                           "modelo_uri": modelo_uri})
        mlflow.log_metrics({"columnas_con_drift": resumen["columnas_con_drift"],
                            "proporcion_con_drift": resumen["proporcion_con_drift"]})
        mlflow.log_metrics({f"drift_{r.columna}": r.puntaje for r in res["tabla"].itertuples()})
        mlflow.log_metrics({f"psi_{r.columna}": r.psi for r in res["tabla"].itertuples()
                            if pd.notna(r.psi)})
        for conjunto, m in resumen["desempeno_v1"].items():
            mlflow.log_metrics({f"{conjunto}_{k}": v for k, v in m.items()})
        mlflow.log_artifacts(str(salida), artifact_path="monitoreo")
        return run.info.run_id


@app.command()
def ingesta(raw_source: str = typer.Option("config", help="Carpeta con copias locales de los CSV")):
    """Ingiere los meses de producción (2026) con su manifiesto y data_version."""
    sin_valor = raw_source.strip().lower() in {"", "config", "auto", "no", '""', "''"}
    ingesta_produccion(None if sin_valor else Path(raw_source))


@app.command()
def drift(
    modelo: str = typer.Option(f"models:/{REGISTRY_NAMES['p1']}@champion",
                               help="Modelo monitoreado (URI de MLflow o carpeta local)"),
    mlflow_log: bool = typer.Option(True, "--mlflow/--sin-mlflow",
                                    help="Registrar la corrida de monitoreo en MLflow"),
):
    """Reportes de Evidently (drift, calidad y desempeño) del modelo P1 en producción."""
    if modelo.startswith("models:/"):
        from mibici.modeling import seguimiento as seg

        seg.configurar_tracking()
    res = monitorear(modelo)
    if mlflow_log:
        run_id = registrar_en_mlflow(res, modelo)
        logger.success(f"Corrida de monitoreo registrada: {run_id}")


if __name__ == "__main__":
    app()
