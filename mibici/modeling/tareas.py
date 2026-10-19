"""Tareas del MLflow Project: datos, modelos, evaluación y gráficas por pregunta.

Cada clase reutiliza las funciones de la entrega anterior (``features.py`` y los
módulos ``p1_regresion``, ``p2_clasificacion`` y ``p3_clustering``) para que la
partición temporal, las variables y las métricas sean EXACTAMENTE las mismas que
las del baseline registrado en la Propuesta de proyecto. Así la comparación
contra el baseline es válida.

* ``P1Tarea``: regresión de viajes por estación y día (Johanna Camargo).
* ``P2Tarea``: clasificación de riesgo de vaciado/llenado (Manuel Cardona).
* ``P3Tarea``: clustering de estaciones por patrón de uso (Gerardo Conde).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    silhouette_score,
)
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    CACHE_DIR,
    ESTACIONES_PARQUET,
    P1_BASELINE_LAG_DIAS,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_VENTANAS_MEDIA_DIAS,
    P2_FRACCION_CAPACIDAD,
    P2_HORIZONTE_HORAS,
    P2_TEST_END,
    P2_TEST_START,
    P2_TRAIN_START,
    P2_VAL_START,
    P3_MAX_DURATION_SECONDS,
    P3_MAX_INACTIVITY_DAYS,
    P3_MIN_DURATION_SECONDS,
    P3_MIN_HISTORY_DAYS,
    P3_REDUCED_START,
    P3_SEMILLAS_ESTABILIDAD,
    P3_TEST_END,
    P3_TEST_START,
    P3_TRAIN_END,
    P3_TRAIN_START,
    SEED,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    P2_FEATURES,
    P3_MODEL_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
    p3_agregar_entropias,
    p3_construir_features,
    p3_filtrar_viajes,
)
from mibici.modeling import p1_regresion as p1
from mibici.modeling import p2_clasificacion as p2
from mibici.modeling import p3_clustering as p3
from mibici.modeling.estimadores import (
    AglomerativoConCentroides,
    Log1pColumnas,
    MezclaGaussiana,
    NaiveEstacional,
)
from mibici.modeling.seguimiento import Cronometro, costo_inferencia_ms_1k, tamano_modelo_mb

AZUL, GRIS, OSCURO, ROJO = "#2563eb", "#9ca3af", "#1f2937", "#dc2626"


@dataclass
class Resultado:
    """Lo que devuelve una tarea al ajustar y evaluar un modelo."""

    modelo: object  # modelo final que se registra (con umbral, pipeline, etc.)
    metricas: dict
    extra: dict = field(default_factory=dict)


def _guardar(fig, ruta: Path) -> Path:
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


class Tarea:
    """Interfaz común de las tres preguntas."""

    question: str = ""
    objetivo: str | None = None
    fuente: Path = VIAJES_PARQUET
    features: tuple[str, ...] = ()

    def preparar(self, data_version: str | None = None) -> None:
        raise NotImplementedError

    def construir(self, modelo: str, params: dict):
        raise NotImplementedError

    def ajustar_evaluar(self, estimador, nombre: str, con_test: bool = True) -> Resultado:
        raise NotImplementedError

    def artefactos(self, res: Resultado, nombre: str, carpeta: Path) -> dict[str, list[Path]]:
        raise NotImplementedError

    def conjuntos(self) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def resumen(self) -> dict:
        raise NotImplementedError

    def ejemplo(self) -> pd.DataFrame:
        raise NotImplementedError

    def particion(self) -> dict:
        raise NotImplementedError


# =============================================================================
# P1 · Regresión de viajes iniciados por estación y día
# =============================================================================
class P1Tarea(Tarea):
    question = "p1"
    objetivo = "viajes"
    features = P1_FEATURES

    def preparar(self, data_version: str | None = None) -> None:
        viajes = p1.cargar_viajes(VIAJES_PARQUET)
        capacidad = p1.cargar_capacidad(ESTACIONES_PARQUET)
        panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
        df = p1_agregar_features(
            panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
        )
        partes = {
            "train": p1.cortar(df, P1_TRAIN_START, P1_VAL_START),
            "val": p1.cortar(df, P1_VAL_START, P1_TEST_START),
            "test": p1.cortar(df, P1_TEST_START, P1_TEST_END),
        }
        self.df = {k: d.dropna(subset=P1_FEATURES) for k, d in partes.items()}
        self.x = {k: d[P1_FEATURES].astype("float64") for k, d in self.df.items()}
        self.y = {k: d["viajes"] for k, d in self.df.items()}
        self.base_test = self.df["test"][f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64")
        for k, d in self.df.items():
            logger.info(f"P1 {k}: {len(d):,} estación-días, "
                        f"{d.index.get_level_values('estacion').nunique()} estaciones")

    def construir(self, modelo: str, params: dict):
        if modelo == "naive":
            return NaiveEstacional(lag_dias=int(params["lag_dias"]))
        if modelo == "poisson":
            return p1.construir_lineal().set_params(
                modelo__alpha=float(params["alpha"]), modelo__max_iter=int(params["max_iter"])
            )
        if modelo == "hgb":
            return HistGradientBoostingRegressor(**params, random_state=SEED)
        if modelo == "rf":
            return RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
        raise ValueError(f"Modelo desconocido para P1: {modelo}")

    def _metricas_estabilidad(self, pred_test: np.ndarray) -> dict:
        """MAE por mes de prueba y dispersión del MAE semanal (estabilidad en el tiempo)."""
        d = pd.DataFrame({"y": self.y["test"].to_numpy(), "p": np.clip(pred_test, 0, None)},
                         index=self.y["test"].index)
        fecha = d.index.get_level_values("fecha")
        err = (d["y"] - d["p"]).abs()
        semanal = err.groupby(fecha.to_period("W")).mean()
        return {
            "test_mae_nov": float(err[fecha.month == 11].mean()),
            "test_mae_dic": float(err[fecha.month == 12].mean()),
            "test_mae_semanal_std": float(semanal.std()),
        }

    def ajustar_evaluar(self, estimador, nombre: str, con_test: bool = True) -> Resultado:
        with Cronometro() as c:
            estimador.fit(self.x["train"], self.y["train"])
        conjuntos = ("train", "val", "test") if con_test else ("train", "val")
        pred = {k: estimador.predict(self.x[k]) for k in conjuntos}
        metricas = {}
        for k in conjuntos:
            metricas.update(p1.evaluar(self.y[k], pred[k], k))
        # Error estándar del MAE de validación (MAE diario de octubre): mide el ruido de VAL.
        err_dia = (self.y["val"] - np.clip(pred["val"], 0, None)).abs().groupby(
            level="fecha").mean()
        metricas["val_mae_se"] = float(err_dia.std(ddof=1) / np.sqrt(len(err_dia)))
        metricas["costo_entrenamiento_seg"] = c.segundos
        metricas["costo_inferencia_ms_1k"] = costo_inferencia_ms_1k(estimador, self.x["val"])
        metricas["tamano_modelo_mb"] = tamano_modelo_mb(estimador)
        if con_test:
            metricas.update(self._metricas_estabilidad(pred["test"]))
        return Resultado(estimador, metricas, {"pred": pred})

    def artefactos(self, res: Resultado, nombre: str, carpeta: Path) -> dict[str, list[Path]]:
        pred = np.clip(res.extra["pred"]["test"], 0, None)
        y = self.y["test"].to_numpy()
        serie, por_dia, predicciones = p1.guardar_graficas(nombre, self.df["test"], pred,
                                                           self.base_test)
        plots = [serie, por_dia]

        fig, ax = plt.subplots(figsize=(5.6, 5))
        hb = ax.hexbin(y, pred, gridsize=45, bins="log", cmap="Blues", mincnt=1)
        lim = max(y.max(), pred.max())
        ax.plot([0, lim], [0, lim], color=ROJO, lw=1.2, ls="--", label="predicho = real")
        ax.set_xlabel("Viajes reales (estación-día)")
        ax.set_ylabel("Viajes predichos")
        ax.set_title(f"P1 · Real contra predicho en TEST\n{nombre}")
        ax.legend(loc="upper left")
        fig.colorbar(hb, ax=ax, label="estación-días (escala log)")
        plots.append(_guardar(fig, carpeta / "p1_real_vs_predicho_test.png"))

        resid = y - pred
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].scatter(pred, resid, s=3, alpha=0.15, color=AZUL)
        axs[0].axhline(0, color=ROJO, lw=1)
        axs[0].set_xlabel("Viajes predichos")
        axs[0].set_ylabel("Residual (real − predicho)")
        axs[0].set_title("Residuales contra predicho (TEST)")
        axs[1].hist(np.clip(resid, -60, 60), bins=80, color=AZUL, alpha=0.85)
        axs[1].axvline(0, color=ROJO, lw=1)
        axs[1].set_xlabel("Residual (recortado a ±60)")
        axs[1].set_ylabel("estación-días")
        axs[1].set_title(f"Distribución de residuales · sesgo medio {(-resid).mean():+.2f}")
        for a in axs:
            a.grid(alpha=0.3)
        plots.append(_guardar(fig, carpeta / "p1_residuales_test.png"))

        evaluacion = [predicciones]
        if not isinstance(res.modelo, (NaiveEstacional, Pipeline)):
            plots_imp, csv_imp = _importancia_permutacion(
                res.modelo, self.x["val"], self.y["val"], "neg_mean_absolute_error",
                "Aumento del MAE al permutar (VAL)", f"P1 · Importancia de variables\n{nombre}",
                carpeta / "p1_importancia_permutacion",
            )
            plots.append(plots_imp)
            evaluacion.append(csv_imp)
        return {"plots": plots, "evaluation": evaluacion}

    def conjuntos(self) -> dict[str, pd.DataFrame]:
        return {
            "training": self.df["train"][P1_FEATURES + ["viajes"]].reset_index(),
            "validation": self.df["val"][P1_FEATURES + ["viajes"]].reset_index(),
            "testing": self.df["test"][P1_FEATURES + ["viajes"]].reset_index(),
        }

    def resumen(self) -> dict:
        return {
            k: {
                "filas": len(d),
                "estaciones": int(d.index.get_level_values("estacion").nunique()),
                "desde": str(d.index.get_level_values("fecha").min().date()),
                "hasta": str(d.index.get_level_values("fecha").max().date()),
                "media_viajes": float(d["viajes"].mean()),
            }
            for k, d in self.df.items()
        }

    def ejemplo(self) -> pd.DataFrame:
        return self.x["test"].head(5).reset_index(drop=True)

    def particion(self) -> dict:
        return {"train_inicio": P1_TRAIN_START, "val_inicio": P1_VAL_START,
                "test_inicio": P1_TEST_START, "test_fin": P1_TEST_END,
                "horizonte_dias": P1_HORIZONTE_DIAS, "features": len(P1_FEATURES)}


# =============================================================================
# P2 · Clasificación de riesgo de quedarse sin bicicletas o sin anclajes
# =============================================================================
class P2Tarea(Tarea):
    question = "p2"
    objetivo = "riesgo"
    features = P2_FEATURES

    def _ruta_cache(self, data_version: str) -> Path:
        firma = f"h{P2_HORIZONTE_HORAS}_f{P2_FRACCION_CAPACIDAD}_{P2_TRAIN_START}_{P2_TEST_END}"
        return CACHE_DIR / f"p2_conjuntos_{data_version}_{firma}.parquet"

    def preparar(self, data_version: str | None = None) -> None:
        cache = self._ruta_cache(data_version) if data_version else None
        if cache is not None and cache.exists():
            logger.info(f"P2: usando caché {cache.name}")
            todo = pd.read_parquet(cache)
        else:
            todo = self._construir()
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                todo.to_parquet(cache)
        self.df = {k: d.drop(columns="conjunto") for k, d in todo.groupby("conjunto", sort=False)}
        self.df = {k: self.df[k] for k in ("train", "val", "test")}
        self.x = {k: d[P2_FEATURES].astype("float64") for k, d in self.df.items()}
        self.y = {k: d["riesgo"] for k, d in self.df.items()}
        self.tipo = {k: d["tipo_riesgo"] for k, d in self.df.items()}
        for k, d in self.df.items():
            logger.info(f"P2 {k}: {len(d):,} estación-horas, prevalencia {d['riesgo'].mean():.2%}")

    def _construir(self) -> pd.DataFrame:
        """Mismos pasos que ``p2_clasificacion.main`` (entrega anterior)."""
        viajes = p2.cargar_viajes(VIAJES_PARQUET)
        capacidad = p2.cargar_capacidad(ESTACIONES_PARQUET)
        panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
        df = p2_agregar_etiqueta(panel, P2_HORIZONTE_HORAS, capacidad, P2_FRACCION_CAPACIDAD)
        df = p2_agregar_features(df, P2_HORIZONTE_HORAS)
        train = p2.cortar(df, P2_TRAIN_START, P2_VAL_START)
        val = p2.cortar(df, P2_VAL_START, P2_TEST_START)
        test = p2.cortar(df, P2_TEST_START, P2_TEST_END)
        viajes_train = viajes.loc[viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)]
        est_train = pd.Index(viajes_train["origen_id"]).union(viajes_train["destino_id"])
        val = val.loc[val.index.get_level_values("estacion").isin(est_train)]
        test = test.loc[test.index.get_level_values("estacion").isin(est_train)]
        historico = p2_ajustar_historico(train)
        train = p2_historico_pasado(train)
        val, test = (p2_aplicar_historico(d, historico) for d in (val, test))
        columnas = P2_FEATURES + ["riesgo", "tipo_riesgo", "neto_futuro"]
        partes = []
        for nombre, d in (("train", train), ("val", val), ("test", test)):
            d = d.dropna(subset=P2_FEATURES)[columnas].copy()
            d["conjunto"] = nombre
            partes.append(d)
        return pd.concat(partes)

    def construir(self, modelo: str, params: dict):
        if modelo == "logreg":
            return p2.construir_baseline().set_params(
                modelo__C=float(params["C"]), modelo__class_weight=params["class_weight"],
                modelo__max_iter=int(params["max_iter"]),
            )
        if modelo == "hgb":
            return HistGradientBoostingClassifier(**params, random_state=SEED)
        if modelo == "rf":
            return RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)
        raise ValueError(f"Modelo desconocido para P2: {modelo}")

    def _f1_por_periodo(self, pred: np.ndarray) -> dict:
        d = pd.DataFrame({"y": self.y["test"].to_numpy(), "p": pred}, index=self.y["test"].index)
        hora = d.index.get_level_values("hora")
        semana = hora.to_period("W")
        semanal = pd.Series([f1_score(g["y"], g["p"], zero_division=0)
                             for _, g in d.groupby(semana)])
        return {
            "test_f1_nov": float(f1_score(d.loc[hora.month == 11, "y"], d.loc[hora.month == 11, "p"])),
            "test_f1_dic": float(f1_score(d.loc[hora.month == 12, "y"], d.loc[hora.month == 12, "p"])),
            "test_f1_semanal_std": float(semanal.std()),
        }

    def ajustar_evaluar(self, estimador, nombre: str, con_test: bool = True) -> Resultado:
        with Cronometro() as c:
            estimador.fit(self.x["train"], self.y["train"])
        conjuntos = ("train", "val", "test") if con_test else ("train", "val")
        prob = {k: estimador.predict_proba(self.x[k])[:, 1] for k in conjuntos}
        umbral = p2.elegir_umbral(self.y["val"], prob["val"])  # nunca con TEST
        metricas = {"umbral_decision": umbral}
        for k in conjuntos:
            metricas.update(p2.evaluar(self.y[k], prob[k], umbral, k, self.tipo[k]))
        # El umbral vive dentro del modelo registrado: predict() devuelve 0/1 listos para operar.
        final = FixedThresholdClassifier(FrozenEstimator(estimador), threshold=umbral,
                                         response_method="predict_proba")
        final.fit(self.x["val"], self.y["val"])  # FrozenEstimator: no reentrena
        # Error estándar del PR-AUC de validación (PR-AUC por semana de octubre).
        semana = self.y["val"].index.get_level_values("hora").to_period("W")
        pr_sem = pd.DataFrame({"y": self.y["val"].to_numpy(), "p": prob["val"]}).groupby(
            np.asarray(semana)).apply(lambda g: average_precision_score(g["y"], g["p"]))
        metricas["val_pr_auc_se"] = float(pr_sem.std(ddof=1) / np.sqrt(len(pr_sem)))
        metricas["costo_entrenamiento_seg"] = c.segundos
        metricas["costo_inferencia_ms_1k"] = costo_inferencia_ms_1k(final, self.x["val"])
        metricas["tamano_modelo_mb"] = tamano_modelo_mb(final)
        if con_test:
            metricas.update(self._f1_por_periodo((prob["test"] >= umbral).astype(int)))
        return Resultado(final, metricas, {"prob": prob, "umbral": umbral, "base": estimador})

    def artefactos(self, res: Resultado, nombre: str, carpeta: Path) -> dict[str, list[Path]]:
        umbral = res.extra["umbral"]
        y_t, p_t = self.y["test"], res.extra["prob"]["test"]
        y_v, p_v = self.y["val"], res.extra["prob"]["val"]
        pred = (p_t >= umbral).astype(int)
        plots = []

        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
        ConfusionMatrixDisplay.from_predictions(y_t, pred, display_labels=["normal", "riesgo"],
                                                ax=axs[0], colorbar=False, values_format=",")
        axs[0].set_title("Conteos (TEST)")
        ConfusionMatrixDisplay.from_predictions(y_t, pred, display_labels=["normal", "riesgo"],
                                                ax=axs[1], colorbar=False, normalize="true",
                                                values_format=".2f", cmap="Oranges")
        axs[1].set_title("Normalizada por clase real (recall)")
        fig.suptitle(f"P2 · Matriz de confusión con umbral {umbral:.3f} · {nombre}")
        plots.append(_guardar(fig, carpeta / "p2_matriz_confusion_test.png"))

        fig, ax = plt.subplots(figsize=(6.4, 5))
        for y, p, etiqueta, color in ((y_v, p_v, "VAL (oct)", GRIS), (y_t, p_t, "TEST (nov-dic)", AZUL)):
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=color, lw=1.8, label=etiqueta)
            ax.axhline(y.mean(), color=color, lw=0.8, ls=":")
        pr = (pred[y_t.to_numpy() == 1].mean(), y_t.to_numpy()[pred == 1].mean() if pred.sum() else 0)
        ax.scatter([pr[0]], [pr[1]], color=ROJO, zorder=5, label=f"umbral {umbral:.3f} (TEST)")
        ax.set_xlabel("Recall (eventos de riesgo detectados)")
        ax.set_ylabel("Precisión (alertas correctas)")
        ax.set_title(f"P2 · Curva precisión-recall\n{nombre}")
        ax.legend()
        ax.grid(alpha=0.3)
        plots.append(_guardar(fig, carpeta / "p2_curva_precision_recall.png"))

        prec, rec, umb = precision_recall_curve(y_v, p_v)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(umb, f1[:-1], color=AZUL, label="F1")
        ax.plot(umb, prec[:-1], color=GRIS, lw=1, label="precisión")
        ax.plot(umb, rec[:-1], color=OSCURO, lw=1, ls="--", label="recall")
        ax.axvline(umbral, color=ROJO, lw=1.2, label=f"umbral elegido {umbral:.3f}")
        ax.set_xlabel("Umbral de probabilidad")
        ax.set_title("P2 · Elección del umbral en VALIDACIÓN (máximo F1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plots.append(_guardar(fig, carpeta / "p2_umbral_validacion.png"))

        reporte = carpeta / "p2_reporte_clasificacion_test.txt"
        reporte.write_text(classification_report(y_t, pred, target_names=["normal", "riesgo"]))
        evaluacion = [reporte]
        base = res.extra["base"]
        if not isinstance(base, Pipeline):
            plots_imp, csv_imp = _importancia_permutacion(
                base, self.x["val"], y_v, "average_precision", "Caída del PR-AUC al permutar (VAL)",
                f"P2 · Importancia de variables\n{nombre}", carpeta / "p2_importancia_permutacion",
                n=60_000,
            )
            plots.append(plots_imp)
            evaluacion.append(csv_imp)
        return {"plots": plots, "evaluation": evaluacion}

    def conjuntos(self) -> dict[str, pd.DataFrame]:
        cols = P2_FEATURES + ["riesgo"]
        return {
            "training": self.df["train"][cols].reset_index(),
            "validation": self.df["val"][cols].reset_index(),
            "testing": self.df["test"][cols].reset_index(),
        }

    def resumen(self) -> dict:
        return {
            k: {
                "filas": len(d),
                "estaciones": int(d.index.get_level_values("estacion").nunique()),
                "desde": str(d.index.get_level_values("hora").min()),
                "hasta": str(d.index.get_level_values("hora").max()),
                "prevalencia_riesgo": float(d["riesgo"].mean()),
            }
            for k, d in self.df.items()
        }

    def ejemplo(self) -> pd.DataFrame:
        return self.x["test"].head(5).reset_index(drop=True)

    def particion(self) -> dict:
        return {"train_inicio": P2_TRAIN_START, "val_inicio": P2_VAL_START,
                "test_inicio": P2_TEST_START, "test_fin": P2_TEST_END,
                "horizonte_horas": P2_HORIZONTE_HORAS,
                "fraccion_capacidad": P2_FRACCION_CAPACIDAD, "features": len(P2_FEATURES)}


# =============================================================================
# P3 · Clustering de estaciones por patrón de uso
# =============================================================================
class P3Tarea(Tarea):
    question = "p3"
    objetivo = None
    features = P3_MODEL_FEATURES

    def preparar(self, data_version: str | None = None) -> None:
        """Mismos pasos que ``p3_clustering.main`` (entrega anterior)."""
        viajes = pd.read_parquet(VIAJES_PARQUET)
        self.estaciones = pd.read_parquet(ESTACIONES_PARQUET)
        viajes = p3_filtrar_viajes(viajes, P3_MIN_DURATION_SECONDS, P3_MAX_DURATION_SECONDS)
        train = viajes.loc[
            (viajes["inicio_viaje"] >= pd.Timestamp(P3_TRAIN_START))
            & (viajes["inicio_viaje"] <= pd.Timestamp(P3_TRAIN_END))
            & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_START))
        ]
        test = viajes.loc[
            (viajes["inicio_viaje"] >= pd.Timestamp(P3_TEST_START))
            & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_END))
        ]
        kw = {"max_inactivity_days": P3_MAX_INACTIVITY_DAYS, "min_history_days": P3_MIN_HISTORY_DAYS}
        self.f_train, _ = p3_construir_features(train, pd.Timestamp(P3_TRAIN_END), **kw)
        self.f_test, _ = p3_construir_features(
            test, pd.Timestamp(P3_TEST_END) - pd.Timedelta(seconds=1), **kw)
        reducido = train.loc[train["inicio_viaje"] >= pd.Timestamp(P3_REDUCED_START)]
        self.f_reducido, _ = p3_construir_features(reducido, pd.Timestamp(P3_TRAIN_END), **kw)
        self.entropias = p3_agregar_entropias(self.f_train, train)
        logger.info(f"P3: estaciones TRAIN={len(self.f_train)} TEST={len(self.f_test)} "
                    f"reducido={len(self.f_reducido)}")

    def construir(self, modelo: str, params: dict):
        prep = [("log1p", Log1pColumnas(("salidas_dia", "llegadas_dia"))), ("escala", StandardScaler())]
        k = int(params["k"])
        if modelo == "kmeans":
            est = KMeans(n_clusters=k, n_init=int(params.get("n_init", 20)), random_state=SEED)
        elif modelo == "agglomerative":
            est = AglomerativoConCentroides(n_clusters=k, linkage=params.get("linkage", "ward"))
        elif modelo == "gmm":
            est = MezclaGaussiana(n_clusters=k, covariance_type=params.get("covariance_type", "diag"),
                                  n_init=int(params.get("n_init", 5)), random_state=SEED)
        else:
            raise ValueError(f"Modelo desconocido para P3: {modelo}")
        return Pipeline(prep + [("modelo", est)])

    def ajustar_evaluar(self, estimador, nombre: str, con_test: bool = True) -> Resultado:
        with Cronometro() as c:
            estimador.fit(self.f_train)
        x_tr = pd.DataFrame(estimador[:-1].transform(self.f_train), index=self.f_train.index,
                            columns=P3_MODEL_FEATURES)
        lab_tr = pd.Series(estimador[-1].labels_, index=self.f_train.index)
        m = {
            "train_silhouette": silhouette_score(x_tr, lab_tr),
            "train_davies_bouldin": davies_bouldin_score(x_tr, lab_tr),
            "train_calinski_harabasz": calinski_harabasz_score(x_tr, lab_tr),
            "tam_min_cluster_pct": float(lab_tr.value_counts(normalize=True).min() * 100),
            "n_grupos": int(lab_tr.nunique()),
            "estaciones_train": len(x_tr),
        }
        # Estabilidad con datos de TRAIN: misma receta ajustada solo con abr-oct.
        reducido = clone(estimador).fit(self.f_reducido)
        lab_red = pd.Series(reducido[-1].labels_, index=self.f_reducido.index)
        comunes = lab_tr.index.intersection(lab_red.index)
        m["ari_historico_reducido"] = adjusted_rand_score(lab_tr[comunes], lab_red[comunes])
        m["nmi_historico_reducido"] = normalized_mutual_info_score(lab_tr[comunes], lab_red[comunes])
        # Estabilidad ante la semilla (1.0 si el algoritmo es determinista).
        if "random_state" in estimador[-1].get_params():
            aris = [adjusted_rand_score(lab_tr, clone(estimador).set_params(modelo__random_state=s)
                                        .fit(self.f_train)[-1].labels_)
                    for s in P3_SEMILLAS_ESTABILIDAD]
            m["estabilidad_semillas_ari"] = float(np.mean(aris))
        else:
            m["estabilidad_semillas_ari"] = 1.0
        m["costo_entrenamiento_seg"] = c.segundos
        m["costo_inferencia_ms_1k"] = costo_inferencia_ms_1k(estimador, self.f_test)
        m["tamano_modelo_mb"] = tamano_modelo_mb(estimador)
        extra = {"lab_tr": lab_tr, "x_tr": x_tr}
        if con_test:
            x_te = estimador[:-1].transform(self.f_test)
            lab_te = pd.Series(estimador.predict(self.f_test), index=self.f_test.index)
            comunes = lab_tr.index.intersection(lab_te.index)
            m.update({
                "test_silhouette": silhouette_score(x_te, lab_te),
                "test_davies_bouldin": davies_bouldin_score(x_te, lab_te),
                "test_calinski_harabasz": calinski_harabasz_score(x_te, lab_te),
                "ari_train_test": adjusted_rand_score(lab_tr[comunes], lab_te[comunes]),
                "nmi_train_test": normalized_mutual_info_score(lab_tr[comunes], lab_te[comunes]),
                "estabilidad_exacta_test": float((lab_tr[comunes] == lab_te[comunes]).mean()),
                "estaciones_test": len(x_te),
                "estaciones_comunes": len(comunes),
            })
            extra.update({"lab_te": lab_te, "comunes": comunes})
        return Resultado(estimador, m, extra)

    def nombres_perfil(self, labels: pd.Series) -> dict[int, str]:
        """Nombres de negocio de la entrega anterior si k = 4; si no, "Grupo i"."""
        if labels.nunique() == 4:
            try:
                return p3.asignar_perfiles(self.f_train, labels)
            except (KeyError, ValueError):
                pass
        return {int(c): f"Grupo {int(c)}" for c in sorted(labels.unique())}

    def artefactos(self, res: Resultado, nombre: str, carpeta: Path) -> dict[str, list[Path]]:
        lab_tr, x_tr = res.extra["lab_tr"], res.extra["x_tr"]
        perfiles = self.nombres_perfil(lab_tr)
        orden = sorted(perfiles)
        colores = plt.get_cmap("tab10")
        plots, evaluacion = [], []

        pca = PCA(n_components=2, random_state=SEED).fit(x_tr)
        comp = pca.transform(x_tr)
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for i, c in enumerate(orden):
            sel = (lab_tr == c).to_numpy()
            ax.scatter(comp[sel, 0], comp[sel, 1], s=28, alpha=0.75, color=colores(i),
                       edgecolors="white", linewidths=0.3, label=f"{perfiles[c]} ({sel.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} de la varianza)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"P3 · Grupos de estaciones en el plano PCA (TRAIN)\n{nombre}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        plots.append(_guardar(fig, carpeta / "p3_pca_grupos.png"))

        z = x_tr.assign(grupo=lab_tr.to_numpy()).groupby("grupo").mean().loc[orden]
        fig, ax = plt.subplots(figsize=(11, 0.9 + 0.55 * len(orden)))
        im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(z.shape[1]), z.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(orden)), [f"{perfiles[c]} (n={int((lab_tr == c).sum())})" for c in orden],
                      fontsize=8)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j, i, f"{z.iat[i, j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="media estandarizada (z)")
        ax.set_title(f"P3 · Perfil de uso de cada grupo (TRAIN) · {nombre}")
        plots.append(_guardar(fig, carpeta / "p3_perfiles_grupos.png"))

        tabla = self.f_train.assign(grupo=lab_tr.to_numpy()).groupby("grupo").mean()
        tabla.insert(0, "perfil", [perfiles[int(c)] for c in tabla.index])
        tabla.insert(1, "estaciones", lab_tr.value_counts().reindex(tabla.index).to_numpy())
        if "lab_te" in res.extra:
            com = res.extra["comunes"]
            igual = (lab_tr[com] == res.extra["lab_te"][com])
            tabla["estabilidad_test_pct"] = igual.groupby(lab_tr[com]).mean().reindex(tabla.index) * 100
        ruta_tabla = carpeta / "p3_tabla_perfiles.csv"
        tabla.round(4).to_csv(ruta_tabla)
        evaluacion.append(ruta_tabla)

        geo = (self.f_train.assign(cluster=lab_tr.to_numpy())
               .join(self.entropias[["entropia_destinos", "entropia_origenes"]])
               .reset_index()
               .merge(self.estaciones[["id", "name", "latitude", "longitude"]],
                      left_on="estacion", right_on="id", how="left").drop(columns="id"))
        geo["perfil"] = geo["cluster"].map(perfiles)
        estado = pd.Series("no_comparable", index=self.f_train.index)
        if "lab_te" in res.extra:
            com = res.extra["comunes"]
            estado[com] = np.where(lab_tr[com] == res.extra["lab_te"][com], "estable", "cambio")
        geo["estado_test"] = geo["estacion"].map(estado)
        asignaciones = carpeta / "p3_asignacion_estaciones.csv"
        geo[["estacion", "name", "latitude", "longitude", "cluster", "perfil", "estado_test"]].to_csv(
            asignaciones, index=False)
        evaluacion.append(asignaciones)

        fig, ax = plt.subplots(figsize=(7, 7))
        for i, c in enumerate(orden):
            sel = geo["cluster"] == c
            ax.scatter(geo.loc[sel, "longitude"], geo.loc[sel, "latitude"], s=14, color=colores(i),
                       label=perfiles[c], alpha=0.85)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(f"P3 · Ubicación de los grupos de estaciones\n{nombre}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        plots.append(_guardar(fig, carpeta / "p3_mapa_grupos.png"))

        if set(perfiles.values()) == set(p3.COLORES_PERFIL):
            ruta_mapa = carpeta / "p3_mapa_interactivo.html"
            p3.generar_mapa(geo.dropna(subset=["latitude", "longitude"]), ruta_mapa)
            plots.append(ruta_mapa)
        (carpeta / "p3_perfiles.json").write_text(json.dumps(
            {str(k): v for k, v in perfiles.items()}, ensure_ascii=False, indent=2))
        evaluacion.append(carpeta / "p3_perfiles.json")
        return {"plots": plots, "evaluation": evaluacion}

    def conjuntos(self) -> dict[str, pd.DataFrame]:
        return {"training": self.f_train.reset_index(), "testing": self.f_test.reset_index()}

    def resumen(self) -> dict:
        return {
            "train": {"estaciones": len(self.f_train), "desde": P3_TRAIN_START, "hasta": P3_TRAIN_END},
            "test": {"estaciones": len(self.f_test), "desde": P3_TEST_START, "hasta": P3_TEST_END},
            "reducido": {"estaciones": len(self.f_reducido), "desde": P3_REDUCED_START,
                         "hasta": P3_TRAIN_END},
        }

    def ejemplo(self) -> pd.DataFrame:
        return self.f_test.head(5).reset_index(drop=True)

    def particion(self) -> dict:
        return {"train_inicio": P3_TRAIN_START[:10], "train_fin": P3_TRAIN_END[:10],
                "test_inicio": P3_TEST_START[:10], "test_fin": "2025-12-31",
                "reducido_inicio": P3_REDUCED_START[:10], "features": len(P3_MODEL_FEATURES)}


def _importancia_permutacion(modelo, x, y, scoring, xlabel, titulo, base: Path, n: int = 40_000):
    """Importancia por permutación en una muestra fija de VAL; devuelve (png, csv)."""
    muestra = x.sample(min(n, len(x)), random_state=SEED)
    res = permutation_importance(modelo, muestra, y.loc[muestra.index], scoring=scoring,
                                 n_repeats=3, random_state=SEED, n_jobs=1)
    imp = pd.Series(res.importances_mean, index=x.columns).sort_values()
    csv = base.with_suffix(".csv")
    imp.rename("importancia").to_csv(csv)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.index, imp.values, color=AZUL)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=0.3)
    return _guardar(fig, base.with_suffix(".png")), csv


TAREAS = {"p1": P1Tarea, "p2": P2Tarea, "p3": P3Tarea}
