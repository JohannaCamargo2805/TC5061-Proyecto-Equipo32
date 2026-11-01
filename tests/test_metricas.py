"""Pruebas de las métricas y de la elección del umbral de decisión."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling import p1_regresion as p1
from mibici.modeling import p2_clasificacion as p2


def _serie_p1(valores):
    idx = pd.MultiIndex.from_product(
        [[1, 2], pd.date_range("2025-11-01", periods=len(valores) // 2)],
        names=["estacion", "fecha"])
    return pd.Series(valores, index=idx, dtype="float64")


def test_metricas_p1_prediccion_perfecta():
    y = _serie_p1([3, 5, 8, 0, 2, 7])
    m = p1.evaluar(y, y.to_numpy(), "test")
    assert m["test_mae"] == 0 and m["test_rmse"] == 0 and m["test_wape"] == 0
    assert m["test_r2"] == pytest.approx(1.0) and m["test_sesgo"] == 0


def test_metricas_p1_recorta_predicciones_negativas():
    """Un conteo no puede ser negativo: -5 se evalúa como 0."""
    y = _serie_p1([0, 4, 2, 0])
    m = p1.evaluar(y, np.array([-5.0, 4.0, 2.0, -1.0]), "val")
    assert m["val_mae"] == 0


def test_metricas_p1_valores_conocidos():
    y = _serie_p1([10, 10, 10, 10])
    m = p1.evaluar(y, np.array([12.0, 8.0, 10.0, 10.0]), "val")
    assert m["val_mae"] == pytest.approx(1.0)
    assert m["val_wape"] == pytest.approx(4 / 40)
    assert m["val_rmse"] == pytest.approx(np.sqrt(2.0))


def test_umbral_separa_clases_perfectamente():
    y = pd.Series([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    umbral = p2.elegir_umbral(y, prob)
    assert 0.3 < umbral <= 0.8
    assert ((prob >= umbral).astype(int) == y.to_numpy()).all()


def test_metricas_p2_y_accuracy_trivial():
    y = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    tipo = pd.Series(["normal"] * 4 + ["sin_bicis", "sin_anclajes"] + ["normal"] * 3 + ["sin_bicis"])
    prob = np.array([0.1, 0.2, 0.1, 0.6, 0.9, 0.2, 0.1, 0.3, 0.2, 0.8])
    m = p2.evaluar(y, prob, 0.5, "test", tipo)
    assert m["test_accuracy_trivial"] == pytest.approx(0.7)  # siempre "normal"
    assert m["test_recall"] == pytest.approx(2 / 3)
    assert m["test_precision"] == pytest.approx(2 / 3)
    assert m["test_recall_sin_bicis"] == 1.0 and m["test_recall_sin_anclajes"] == 0.0


def test_metricas_p2_sin_positivos_predichos():
    """Caso límite: el modelo nunca alerta; la precisión no truena (zero_division=0)."""
    y = pd.Series([0, 1, 0, 1])
    tipo = pd.Series(["normal", "sin_bicis", "normal", "sin_anclajes"])
    m = p2.evaluar(y, np.zeros(4), 0.5, "val", tipo)
    assert m["val_precision"] == 0 and m["val_recall"] == 0 and m["val_f1"] == 0
