"""Gráficas reutilizables (se guardan como PNG y se registran como artefactos en MLflow)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

# Paleta categórica validada para daltonismo (orden fijo, nunca cíclico).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------
def p1_week_plot(
    df: pd.DataFrame, pred_cols: dict[str, str], path: Path, n_stations: int = 3
) -> Path:
    """Viajes reales vs. predicción en la primera semana de prueba (estaciones con más viajes
    en esa semana). Las horas sin servicio (01:00-04:59) se dejan en blanco."""
    t0 = df["ts"].min().normalize() + pd.Timedelta(hours=5)
    t1 = t0 + pd.Timedelta(days=7)
    week = df[(df["ts"] >= t0) & (df["ts"] < t1)]
    top = week.groupby("station_id")["y"].sum().nlargest(n_stations).index
    hours = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq="h")
    fig, axes = plt.subplots(n_stations, 1, figsize=(11, 2.6 * n_stations), sharex=True)
    for ax, sid in zip(np.atleast_1d(axes), top):
        d = week[week["station_id"] == sid].set_index("ts").reindex(hours)
        ax.plot(d.index, d["y"], color=INK, lw=1.6, label="Real")
        for i, (label, col) in enumerate(pred_cols.items()):
            ax.plot(d.index, d[col], color=PALETTE[i], lw=1.6, label=label)
        ax.set_title(f"Estación {sid}", loc="left", fontsize=10)
        ax.set_ylabel("Viajes/hora")
    np.atleast_1d(axes)[0].legend(ncol=len(pred_cols) + 1, loc="upper right")
    fig.suptitle(
        "P1 · Viajes iniciados por hora: real vs. predicción (primera semana de prueba)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, path)


def p1_error_by_hour(df: pd.DataFrame, pred_cols: dict[str, str], path: Path) -> Path:
    """MAE por hora de servicio (05:00 a 00:59) en el conjunto de prueba."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    order = (df["hour"] - 5) % 24  # 0 = 05:00 ... 19 = 00:00
    for i, (label, col) in enumerate(pred_cols.items()):
        mae = (df[col] - df["y"]).abs().groupby(order).mean()
        ax.plot(
            mae.index, mae.values, marker=MARKERS[i], ms=6, lw=2, color=PALETTE[i], label=label
        )
    ticks = list(range(0, 20, 2))
    ax.set_xticks(ticks, [f"{(t + 5) % 24:02d}:00" for t in ticks])
    ax.set_xlabel("Hora de servicio")
    ax.set_ylabel("MAE (viajes/hora)")
    ax.set_title("P1 · Error absoluto medio por hora (prueba)", loc="left")
    ax.legend()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------
def p2_pr_curve(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> Path:
    """Curva precisión-exhaustividad (adecuada con clases desbalanceadas)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for i, (label, s) in enumerate(scores.items()):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=2, color=PALETTE[i], label=label)
    ax.axhline(y.mean(), color=INK_2, lw=1, ls="--", label=f"Tasa base = {y.mean():.3f}")
    ax.set_xlabel("Exhaustividad (recall)")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("P2 · Curva precisión-exhaustividad (prueba)", loc="left")
    ax.legend(loc="upper right")
    return _save(fig, path)


def confusion_plot(y: np.ndarray, y_hat: np.ndarray, path: Path, title: str) -> Path:
    """Matriz de confusión normalizada por clase real."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y, y_hat, display_labels=["Sin riesgo", "Riesgo"], normalize="true",
        values_format=".2f", cmap="Blues", colorbar=False, ax=ax,
    )  # fmt: skip
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    return _save(fig, path)


def p2_rate_by_hour(df: pd.DataFrame, path: Path) -> Path:
    """Tasa de riesgo (etiqueta aproximada) por hora de decisión y tipo de día."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (wk, label) in enumerate([(0, "Entre semana"), (1, "Fin de semana")]):
        r = df[df["is_weekend"] == wk].groupby("hour")["y"].mean()
        ax.plot(r.index, r.values, marker=MARKERS[i], ms=6, lw=2, color=PALETTE[i], label=label)
    ax.set_xlabel("Hora de decisión")
    ax.set_ylabel("Proporción con riesgo en 3 h")
    ax.set_title("P2 · Frecuencia de la etiqueta de riesgo por hora", loc="left")
    ax.legend()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------
def p3_cluster_map(
    prof: pd.DataFrame, labels: np.ndarray, names: dict[int, str], path: Path, title: str
) -> Path:
    """Mapa de estaciones (longitud/latitud) coloreadas y con marcador por grupo."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(
            prof.loc[m, "lon"], prof.loc[m, "lat"], s=26, color=PALETTE[c % 8], marker=MARKERS[c % 8],
            edgecolor=SURFACE, linewidth=0.6, label=f"G{c}: {names.get(c, '')} (n={m.sum()})",
        )  # fmt: skip
    lat0 = np.deg2rad(prof["lat"].mean())
    ax.set_aspect(1 / np.cos(lat0))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return _save(fig, path)


def p3_profile_heatmap(
    prof: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    path: Path,
    n_cluster_features: int | None = None,
) -> Path:
    """Promedio estandarizado (z-score) de cada variable por grupo. Si se indica
    ``n_cluster_features``, una línea separa las variables de agrupamiento de las
    descriptivas."""
    z = (prof[features] - prof[features].mean()) / prof[features].std(ddof=0).replace(0, 1)
    means = z.groupby(labels).mean()
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(means) + 2.2))
    im = ax.imshow(means.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(means)), [f"G{c}" for c in means.index])
    for (i, j), v in np.ndenumerate(means.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.grid(False)
    if n_cluster_features is not None and n_cluster_features < len(features):
        ax.axvline(n_cluster_features - 0.5, color=INK, lw=2)
        ax.text(
            n_cluster_features - 0.6, -0.65, "agrupamiento ←", ha="right", fontsize=9, color=INK_2
        )
        ax.text(
            n_cluster_features - 0.4, -0.65, "→ descriptivas", ha="left", fontsize=9, color=INK_2
        )
    fig.colorbar(im, ax=ax, fraction=0.025, label="Promedio estandarizado (z)")
    ax.set_title("P3 · Perfil de uso promedio por grupo", loc="left", pad=18)
    return _save(fig, path)


def p3_k_selection(ks: list[int], sil: list[float], best_k: int, path: Path) -> Path:
    """Silueta por número de grupos k (selección en entrenamiento)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, sil, marker="o", ms=7, lw=2, color=PALETTE[0])
    ax.axvline(best_k, color=INK_2, ls="--", lw=1)
    ax.annotate(
        f"k = {best_k}", (best_k, max(sil)), xytext=(6, -2), textcoords="offset points", color=INK
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Silueta (train)")
    ax.set_title("P3 · Selección del número de grupos", loc="left")
    return _save(fig, path)
