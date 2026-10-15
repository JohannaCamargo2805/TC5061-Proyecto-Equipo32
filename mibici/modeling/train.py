"""Experimento base (baselines y modelos candidatos) registrado en MLflow Tracking.

Uso desde la raíz del repositorio::

    python -m mibici.modeling.train                 # P1, P2 y P3 (baseline + candidato)
    python -m mibici.modeling.train --question P2   # solo una pregunta
    python -m mibici.modeling.train --no-candidates # solo baselines

Buenas prácticas implementadas:

* Un experimento por proyecto (``mibici-equipo32``) y runs con nombre descriptivo.
* Parámetros de reproducción: rango de fechas, partición, versión de datos,
  semilla, modelo e hiperparámetros.
* Etiquetas: equipo, pregunta, tipo de modelo y commit de Git del código.
* Métricas nombradas con el conjunto en que se calculan (``test_mae``, ``val_f2``...).
* Ningún dato personal ni credencial se registra (solo agregados por estación).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from loguru import logger
import mlflow
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici import config, plots
from mibici.dataset import load_manifest
from mibici.features import P1_FEATURES, P2_FEATURES, P3_DESCRIPTIVE, P3_FEATURES

app = typer.Typer(help="Experimento base de MiBici registrado en MLflow.")


# ---------------------------------------------------------------------------
# Utilidades de trazabilidad
# ---------------------------------------------------------------------------
def git_info() -> dict[str, str]:
    """Commit, rama y estado del árbol de trabajo (para ligar cada run a su código)."""

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=config.PROJ_ROOT, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "desconocido"

    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(dirty) and dirty != "desconocido").lower(),
    }


def setup_mlflow() -> str:
    """Configura el tracking local y devuelve el id del experimento del equipo."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        config.MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT_NAME,
            artifact_location=config.MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
            tags={"equipo": config.TEAM_ID, "curso": "TC5061", "proyecto": "MiBici"},
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def common_params() -> dict:
    """Parámetros compartidos que permiten reproducir cualquier run."""
    manifest = load_manifest()
    return {
        "data_version": manifest["data_version"],
        "data_start": config.DATE_START,
        "data_end": config.DATE_END,
        "data_clean_trips": manifest["totals"]["clean_trips"],
        "split_train": "/".join(config.SPLITS["train"]),
        "split_val": "/".join(config.SPLITS["val"]),
        "split_test": "/".join(config.SPLITS["test"]),
        "seed": config.SEED,
        "pipeline_version": config.PIPELINE_VERSION,
    }


def common_tags(question: str, run_type: str, description: str) -> dict:
    """Etiquetas de equipo, pregunta, tipo de run y commit de Git."""
    info = git_info()
    if info["git_dirty"] == "true":
        logger.warning(
            "Hay cambios sin commit: el run no será reproducible desde el commit registrado."
        )
    return {
        "equipo": config.TEAM_ID,
        # Sustituye el usuario del sistema operativo (dato personal) por el equipo.
        "mlflow.user": config.TEAM_ID,
        "pregunta": question,
        "tipo_run": run_type,
        "mlflow.note.content": description,
        **info,
    }


def log_common_artifacts(features: list[str], extra: dict | None = None) -> None:
    """Registra el manifiesto de datos y la lista de variables del modelo."""
    mlflow.log_artifact(str(config.DATA_MANIFEST_PATH), artifact_path="data")
    mlflow.log_dict({"features": features, **(extra or {})}, "features.json")


def log_datasets(question: str) -> None:
    """Liga el run con la versión de datos (columna *Dataset* de la interfaz de MLflow).

    Se registra una referencia ligera: fuente = portal de MiBici y digest = hash
    de ``data_version`` (no se copian datos al servidor de MLflow).
    """
    version = load_manifest()["data_version"]
    source = HTTPDatasetSource(url=config.MIBICI_PORTAL_URL)
    for context in ("train", "val", "test"):
        ds = MetaDataset(
            source=source, name=f"mibici-{question.lower()}-{context}", digest=version[-10:]
        )
        mlflow.log_input(ds, context=context)


# Tipos internos de HistGradientBoosting que MLflow (skops) pide confirmar como
# confiables; el modelo lo entrena este mismo código, no proviene de terceros.
TRUSTED_SKOPS_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def log_model(model, input_example: pd.DataFrame) -> None:
    """Registra el modelo entrenado como artefacto del run (no se versiona en Git)."""
    mlflow.sklearn.log_model(
        model, name="model", input_example=input_example, skops_trusted_types=TRUSTED_SKOPS_TYPES
    )


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df.loc[df["split"] == name]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def regression_metrics(y: np.ndarray, y_hat: np.ndarray, prefix: str) -> dict[str, float]:
    """MAE, RMSE, WAPE y sesgo medio, con el prefijo del conjunto evaluado."""
    err = y_hat - y
    return {
        f"{prefix}_mae": float(mean_absolute_error(y, y_hat)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
        f"{prefix}_wape": float(np.abs(err).sum() / max(np.abs(y).sum(), 1e-9)),
        f"{prefix}_bias": float(err.mean()),
    }


def station_day_mae(df: pd.DataFrame, pred_col: str) -> float:
    """MAE del total diario por estación (granularidad de planeación de flota)."""
    d = df.groupby(["station_id", "op_date"])[["y", pred_col]].sum()
    return float((d[pred_col] - d["y"]).abs().mean())


def classification_metrics(
    y: np.ndarray, score: np.ndarray, y_hat: np.ndarray, prefix: str
) -> dict[str, float]:
    """PR-AUC, ROC-AUC, precisión, exhaustividad, F1 y F2 del conjunto evaluado."""
    return {
        f"{prefix}_pr_auc": float(average_precision_score(y, score)),
        f"{prefix}_roc_auc": float(roc_auc_score(y, score)),
        f"{prefix}_precision": float(precision_score(y, y_hat, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, y_hat, zero_division=0)),
        f"{prefix}_f1": float(fbeta_score(y, y_hat, beta=1, zero_division=0)),
        f"{prefix}_f2": float(fbeta_score(y, y_hat, beta=2, zero_division=0)),
        f"{prefix}_positive_rate": float(np.mean(y)),
        f"{prefix}_predicted_positive_rate": float(np.mean(y_hat)),
    }


def best_threshold(
    y: np.ndarray, score: np.ndarray, beta: float = config.P2_THRESHOLD_BETA
) -> float:
    """Umbral que maximiza F-beta en validación (nunca se elige con prueba)."""
    grid = np.unique(np.quantile(score, np.linspace(0.50, 0.995, 120)))
    scores = [fbeta_score(y, score >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def cluster_metrics(X: np.ndarray, labels: np.ndarray, prefix: str) -> dict[str, float]:
    """Silueta, Davies-Bouldin y Calinski-Harabasz en el espacio de uso estandarizado."""
    return {
        f"{prefix}_silhouette": float(silhouette_score(X, labels, random_state=config.SEED)),
        f"{prefix}_davies_bouldin": float(davies_bouldin_score(X, labels)),
        f"{prefix}_calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


# ---------------------------------------------------------------------------
# P1 - Regresión
# ---------------------------------------------------------------------------
def run_p1(candidates: bool = True) -> None:
    """P1: viajes iniciados por estación y hora del día siguiente."""
    df = pd.read_parquet(config.P1_DATASET_PATH)
    base_col = f"departures_lag_{config.P1_BASELINE_LAG_HOURS}"
    # Se evalúa en filas donde el baseline está definido (mismas filas para todos).
    evaluable = df[base_col].notna()
    train, val = _split(df, "train"), _split(df[evaluable], "val")
    test = _split(df[evaluable], "test").copy()
    common = {
        **common_params(),
        "granularity": "estacion x hora de servicio (05:00-00:59)",
        "target": "viajes iniciados (departures)",
        "horizon_hours": config.P1_HORIZON_HOURS,
        "n_stations": int(df["station_id"].nunique()),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        desc = (
            "Baseline P1: ingenuo estacional semanal. Predice para cada estación y hora el "
            "número de viajes iniciados a la misma hora y día de la semana anterior."
        )
        with mlflow.start_run(
            run_name="P1-baseline-ingenuo-estacional-semanal",
            tags=common_tags("P1", "baseline", desc),
        ):
            mlflow.log_params(
                {
                    **common,
                    "model": "SeasonalNaive",
                    "season_lag_hours": config.P1_BASELINE_LAG_HOURS,
                }
            )
            test["pred_baseline"] = test[base_col]
            metrics = regression_metrics(
                test["y"].to_numpy(), test["pred_baseline"].to_numpy(), "test"
            )
            metrics |= regression_metrics(val["y"].to_numpy(), val[base_col].to_numpy(), "val")
            metrics["test_mae_station_day"] = station_day_mae(test, "pred_baseline")
            mlflow.log_metrics(metrics)
            log_common_artifacts([base_col])
            log_datasets("P1")
            mlflow.log_artifact(
                str(
                    plots.p1_week_plot(
                        test, {"Baseline (t-168 h)": "pred_baseline"}, tmp / "p1_real_vs_pred.png"
                    )
                )
            )
            mlflow.log_artifact(
                str(
                    plots.p1_error_by_hour(
                        test, {"Baseline": "pred_baseline"}, tmp / "p1_mae_por_hora.png"
                    )
                )
            )
            logger.success(
                f"P1 baseline | test_mae={metrics['test_mae']:.3f} test_wape={metrics['test_wape']:.3f}"
            )

        if not candidates:
            return
        desc = (
            "Candidato P1: HistGradientBoostingRegressor con pérdida Poisson (conteos) sobre "
            "rezagos >= 24 h, promedios a la misma hora, calendario y perfil histórico de la estación."
        )
        with mlflow.start_run(
            run_name="P1-candidato-hgb-poisson-rezagos", tags=common_tags("P1", "candidato", desc)
        ):
            params = {f"hgb_{k}": v for k, v in config.P1_HGB_PARAMS.items()}
            mlflow.log_params(
                {
                    **common,
                    "model": "HistGradientBoostingRegressor",
                    **params,
                    "n_features": len(P1_FEATURES),
                }
            )
            model = HistGradientBoostingRegressor(**config.P1_HGB_PARAMS, random_state=config.SEED)
            model.fit(train[P1_FEATURES], train["y"])
            test["pred_hgb"] = np.clip(model.predict(test[P1_FEATURES]), 0, None)
            val_pred = np.clip(model.predict(val[P1_FEATURES]), 0, None)
            metrics = regression_metrics(test["y"].to_numpy(), test["pred_hgb"].to_numpy(), "test")
            metrics |= regression_metrics(val["y"].to_numpy(), val_pred, "val")
            metrics["test_mae_station_day"] = station_day_mae(test, "pred_hgb")
            mlflow.log_metrics(metrics)
            log_common_artifacts(P1_FEATURES)
            log_datasets("P1")
            pred_cols = {"Baseline (t-168 h)": "pred_baseline", "HGB Poisson": "pred_hgb"}
            mlflow.log_artifact(
                str(plots.p1_week_plot(test, pred_cols, tmp / "p1_real_vs_pred.png"))
            )
            mlflow.log_artifact(
                str(
                    plots.p1_error_by_hour(
                        test,
                        {"Baseline": "pred_baseline", "HGB Poisson": "pred_hgb"},
                        tmp / "p1_mae_por_hora.png",
                    )
                )
            )
            log_model(model, test[P1_FEATURES].head(5))
            logger.success(
                f"P1 candidato | test_mae={metrics['test_mae']:.3f} test_wape={metrics['test_wape']:.3f}"
            )


# ---------------------------------------------------------------------------
# P2 - Clasificación
# ---------------------------------------------------------------------------
def run_p2(candidates: bool = True) -> None:
    """P2: riesgo de vaciado/llenado en las próximas horas."""
    df = pd.read_parquet(config.P2_DATASET_PATH)
    train, val, test = _split(df, "train"), _split(df, "val"), _split(df, "test")
    y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
    common = {
        **common_params(),
        "granularity": "estacion x hora de decision",
        "target": "riesgo de ocupacion <= low o >= high en (t, t+h]",
        "horizon_hours": config.P2_HORIZON_HOURS,
        "initial_fill": config.P2_INITIAL_FILL,
        "low_fill": config.P2_LOW_FILL,
        "high_fill": config.P2_HIGH_FILL,
        "threshold_selection": f"max F{config.P2_THRESHOLD_BETA:g} en validacion",
        "n_stations": int(df["station_id"].nunique()),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_positive_rate": round(float(train["y"].mean()), 4),
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        desc = (
            "Baseline P2: regla sobre la ocupación estimada actual. Riesgo si |ocupación - 0.5| "
            "supera un umbral elegido en validación (la estación ya se acerca a vaciarse o llenarse)."
        )
        with mlflow.start_run(
            run_name="P2-baseline-regla-ocupacion-actual", tags=common_tags("P2", "baseline", desc)
        ):
            s_val = (val["fill_ratio_now"] - 0.5).abs().to_numpy()
            s_test = (test["fill_ratio_now"] - 0.5).abs().to_numpy()
            thr = best_threshold(y_val, s_val)
            mlflow.log_params(
                {**common, "model": "ReglaOcupacionActual", "decision_threshold": round(thr, 4)}
            )
            metrics = classification_metrics(y_test, s_test, s_test >= thr, "test")
            metrics |= classification_metrics(y_val, s_val, s_val >= thr, "val")
            mlflow.log_metrics(metrics)
            log_common_artifacts(["fill_ratio_now"])
            log_datasets("P2")
            mlflow.log_artifact(
                str(
                    plots.confusion_plot(
                        y_test,
                        s_test >= thr,
                        tmp / "p2_matriz_confusion.png",
                        "P2 · Baseline (prueba)",
                    )
                )
            )
            mlflow.log_artifact(
                str(plots.p2_rate_by_hour(train, tmp / "p2_tasa_riesgo_por_hora.png"))
            )
            mlflow.log_artifact(
                str(plots.p2_pr_curve(y_test, {"Baseline": s_test}, tmp / "p2_curva_pr.png"))
            )
            logger.success(
                f"P2 baseline | test_pr_auc={metrics['test_pr_auc']:.3f} test_recall={metrics['test_recall']:.3f}"
            )

        if not candidates:
            return
        desc = (
            "Candidato P2: HistGradientBoostingClassifier (class_weight balanceado) con ocupación "
            "estimada, flujos recientes, flujos de la misma ventana ayer/semana pasada y calendario."
        )
        with mlflow.start_run(
            run_name="P2-candidato-hgb-clasificador-3h", tags=common_tags("P2", "candidato", desc)
        ):
            model = HistGradientBoostingClassifier(
                **config.P2_HGB_PARAMS, random_state=config.SEED
            )
            model.fit(train[P2_FEATURES], train["y"])
            p_val = model.predict_proba(val[P2_FEATURES])[:, 1]
            p_test = model.predict_proba(test[P2_FEATURES])[:, 1]
            thr = best_threshold(y_val, p_val)
            params = {f"hgb_{k}": v for k, v in config.P2_HGB_PARAMS.items()}
            mlflow.log_params(
                {
                    **common,
                    "model": "HistGradientBoostingClassifier",
                    **params,
                    "decision_threshold": round(thr, 4),
                    "n_features": len(P2_FEATURES),
                }
            )
            metrics = classification_metrics(y_test, p_test, p_test >= thr, "test")
            metrics |= classification_metrics(y_val, p_val, p_val >= thr, "val")
            mlflow.log_metrics(metrics)
            log_common_artifacts(P2_FEATURES)
            log_datasets("P2")
            mlflow.log_artifact(
                str(
                    plots.confusion_plot(
                        y_test, p_test >= thr, tmp / "p2_matriz_confusion.png", "P2 · HGB (prueba)"
                    )
                )
            )
            mlflow.log_artifact(
                str(
                    plots.p2_pr_curve(
                        y_test,
                        {"Baseline (regla)": s_test, "HGB": p_test},
                        tmp / "p2_curva_pr.png",
                    )
                )
            )
            log_model(model, test[P2_FEATURES].head(5))
            logger.success(
                f"P2 candidato | test_pr_auc={metrics['test_pr_auc']:.3f} test_recall={metrics['test_recall']:.3f}"
            )


# ---------------------------------------------------------------------------
# P3 - Clustering
# ---------------------------------------------------------------------------
PHRASES = {
    "dep_pico_manana": "salidas pico AM", "arr_pico_manana": "llegadas pico AM",
    "dep_pico_tarde": "salidas pico PM", "arr_pico_tarde": "llegadas pico PM",
    "dep_mediodia": "salidas mediodía", "arr_mediodia": "llegadas mediodía",
    "dep_tarde": "salidas tarde", "arr_tarde": "llegadas tarde",
    "dep_noche": "salidas noche", "arr_noche": "llegadas noche",
    "dep_madrugada": "salidas madrugada", "arr_madrugada": "llegadas madrugada",
    "weekend_share": "fin de semana", "log_daily_trips": "alto volumen",
    "net_flow_ratio": "acumula bicis", "mean_duration_min": "viajes largos",
    "round_trip_share": "viajes circulares",
}  # fmt: skip


def name_clusters(prof: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Nombre descriptivo de cada grupo: sus dos rasgos de patrón más altos (z-score)
    y, si destaca, su volumen (variable descriptiva)."""
    cols = P3_FEATURES + ["log_daily_trips"]
    z = (prof[cols] - prof[cols].mean()) / prof[cols].std(ddof=0).replace(0, 1)
    means = z.groupby(labels).mean()
    names = {}
    for c, row in means.iterrows():
        top = row[P3_FEATURES].sort_values(ascending=False).index[:2]
        label = " + ".join(PHRASES.get(f, f) for f in top)
        if row["log_daily_trips"] > 0.5:
            label += " (alto volumen)"
        elif row["log_daily_trips"] < -0.5:
            label += " (bajo volumen)"
        names[int(c)] = label
    return names


def run_p3(candidates: bool = True) -> None:
    """P3: grupos de estaciones con patrones de uso similares y su ubicación."""
    tr = pd.read_parquet(config.P3_TRAIN_PROFILE_PATH)
    te = pd.read_parquet(config.P3_TEST_PROFILE_PATH)
    scaler = StandardScaler().fit(tr[P3_FEATURES])
    X_tr = scaler.transform(tr[P3_FEATURES])
    X_te = scaler.transform(te[P3_FEATURES])
    common_ids = np.intersect1d(tr["station_id"], te["station_id"])
    in_te = te["station_id"].isin(common_ids).to_numpy()

    sil = [
        silhouette_score(
            X_tr,
            KMeans(k, n_init=20, random_state=config.SEED).fit_predict(X_tr),
            random_state=config.SEED,
        )
        for k in config.P3_K_RANGE
    ]
    k = config.P3_K_RANGE[int(np.argmax(sil))]
    common = {
        **common_params(),
        "granularity": "estacion (perfil del periodo)",
        "profile_period_train": "/".join(config.SPLITS["train"]),
        "profile_period_test": "/".join(config.SPLITS["test"]),
        "k": k,
        "k_range": f"{config.P3_K_RANGE[0]}-{config.P3_K_RANGE[-1]}",
        "k_selection": "max silueta en train",
        "min_daily_trips": config.P3_MIN_DAILY_TRIPS,
        "n_stations_train": len(tr),
        "n_stations_test": len(te),
    }

    def _stability(model_fn, Xa, Xb) -> float:
        """ARI entre los grupos de train (aplicados a test) y un reajuste con datos de test."""
        a = model_fn().fit(Xa).predict(Xb)[in_te]
        b = model_fn().fit(Xb).predict(Xb)[in_te]
        return float(adjusted_rand_score(a, b))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        desc = (
            "Baseline P3: zonificación geográfica. K-Means sobre latitud/longitud con el mismo k; "
            "se evalúa en el espacio de patrones de uso para saber si la cercanía basta para agrupar por uso."
        )
        with mlflow.start_run(
            run_name="P3-baseline-kmeans-geografico", tags=common_tags("P3", "baseline", desc)
        ):
            geo = ["lat", "lon"]
            km_geo = KMeans(k, n_init=20, random_state=config.SEED).fit(tr[geo])
            lab_tr, lab_te = km_geo.labels_, km_geo.predict(te[geo])
            mlflow.log_params({**common, "model": "KMeans(lat, lon)", "kmeans_n_init": 20})
            metrics = cluster_metrics(X_tr, lab_tr, "train") | cluster_metrics(
                X_te, lab_te, "test"
            )
            metrics["test_stability_ari"] = _stability(
                lambda: KMeans(k, n_init=20, random_state=config.SEED),
                tr[geo].to_numpy(),
                te[geo].to_numpy(),
            )
            mlflow.log_metrics(metrics)
            log_common_artifacts(geo, {"evaluation_space": P3_FEATURES})
            log_datasets("P3")
            names = {c: "zona geográfica" for c in range(k)}
            mlflow.log_artifact(
                str(
                    plots.p3_cluster_map(
                        tr,
                        lab_tr,
                        names,
                        tmp / "p3_mapa_grupos.png",
                        "P3 · Baseline: zonas geográficas",
                    )
                )
            )
            mlflow.log_artifact(
                str(
                    plots.p3_k_selection(
                        list(config.P3_K_RANGE), sil, k, tmp / "p3_seleccion_k.png"
                    )
                )
            )
            logger.success(
                f"P3 baseline | train_silhouette={metrics['train_silhouette']:.3f} k={k}"
            )

        if not candidates:
            return
        desc = (
            "Candidato P3: K-Means sobre el patrón de uso estandarizado (proporción de salidas y "
            "llegadas por bloque horario en días hábiles, balance neto y proporción de fin de semana)."
        )
        with mlflow.start_run(
            run_name="P3-candidato-kmeans-perfil-uso", tags=common_tags("P3", "candidato", desc)
        ):
            km = KMeans(k, n_init=20, random_state=config.SEED).fit(X_tr)
            lab_tr, lab_te = km.labels_, km.predict(X_te)
            mlflow.log_params(
                {
                    **common,
                    "model": "KMeans(perfil de uso)",
                    "kmeans_n_init": 20,
                    "n_features": len(P3_FEATURES),
                }
            )
            metrics = cluster_metrics(X_tr, lab_tr, "train") | cluster_metrics(
                X_te, lab_te, "test"
            )
            metrics["test_stability_ari"] = _stability(
                lambda: KMeans(k, n_init=20, random_state=config.SEED), X_tr, X_te
            )
            mlflow.log_metrics(metrics)
            names = name_clusters(tr, lab_tr)
            log_common_artifacts(
                P3_FEATURES, {"cluster_names": names, "descriptive_features": P3_DESCRIPTIVE}
            )
            log_datasets("P3")
            mlflow.log_artifact(
                str(
                    plots.p3_cluster_map(
                        tr,
                        lab_tr,
                        names,
                        tmp / "p3_mapa_grupos.png",
                        "P3 · Grupos de estaciones por patrón de uso",
                    )
                )
            )
            mlflow.log_artifact(
                str(
                    plots.p3_profile_heatmap(
                        tr,
                        lab_tr,
                        P3_FEATURES + P3_DESCRIPTIVE,
                        tmp / "p3_perfil_grupos.png",
                        n_cluster_features=len(P3_FEATURES),
                    )
                )
            )
            mlflow.log_artifact(
                str(
                    plots.p3_k_selection(
                        list(config.P3_K_RANGE), sil, k, tmp / "p3_seleccion_k.png"
                    )
                )
            )
            assign = tr[
                ["station_id", "name", "lat", "lon", "daily_trips", *P3_DESCRIPTIVE[1:]]
            ].assign(cluster=lab_tr)
            assign["cluster_name"] = assign["cluster"].map(names)
            assign.to_csv(tmp / "p3_asignacion_estaciones.csv", index=False)
            mlflow.log_artifact(str(tmp / "p3_asignacion_estaciones.csv"))
            pipe = make_pipeline(scaler, km)
            log_model(pipe, tr[P3_FEATURES].head(5))
            logger.success(
                f"P3 candidato | train_silhouette={metrics['train_silhouette']:.3f} k={k}"
            )


@app.command()
def main(
    question: str = typer.Option("all", help="P1, P2, P3 o all"),
    candidates: bool = typer.Option(True, help="Incluir modelos candidatos además de baselines"),
) -> None:
    """Ejecuta el experimento base y lo registra en MLflow."""
    np.random.seed(config.SEED)
    setup_mlflow()
    q = question.upper()
    runners = {"P1": run_p1, "P2": run_p2, "P3": run_p3}
    for name, fn in runners.items():
        if q in ("ALL", name):
            logger.info(f"=== {name} ===")
            fn(candidates=candidates)
    logger.success(
        f"Runs registrados en el experimento '{config.MLFLOW_EXPERIMENT_NAME}' ({config.MLFLOW_TRACKING_URI})"
    )


if __name__ == "__main__":
    app()
