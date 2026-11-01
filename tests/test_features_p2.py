"""Pruebas de las funciones de P2 (riesgo de vaciado/llenado): panel, etiqueta y variables."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    P2_FEATURES,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
)
from mibici.modeling import p2_clasificacion as p2


def _viaje(inicio, fin, origen, destino):
    return {"inicio_viaje": pd.Timestamp(inicio), "fin_viaje": pd.Timestamp(fin),
            "origen_id": origen, "destino_id": destino}


@pytest.fixture
def viajes():
    return pd.DataFrame([
        _viaje("2025-01-01 08:10", "2025-01-01 08:40", 1, 2),
        _viaje("2025-01-01 08:20", "2025-01-01 09:05", 1, 2),
        _viaje("2025-01-01 10:00", "2025-01-01 10:30", 2, 1),
    ])


def test_panel_horario_neto_y_ceros(viajes):
    panel = p2_construir_panel(viajes, "2025-01-01", "2025-01-02")
    assert panel.loc[(1, pd.Timestamp("2025-01-01 08:00")), "salidas"] == 2
    assert panel.loc[(2, pd.Timestamp("2025-01-01 08:00")), "llegadas"] == 1
    assert panel.loc[(2, pd.Timestamp("2025-01-01 09:00")), "llegadas"] == 1
    assert (panel["neto"] == panel["llegadas"] - panel["salidas"]).all()
    assert panel.loc[(1, pd.Timestamp("2025-01-01 15:00")), ["salidas", "llegadas"]].sum() == 0


def test_panel_inicia_en_la_primera_actividad(viajes):
    panel = p2_construir_panel(viajes, "2025-01-01", "2025-01-02")
    assert panel.loc[1].index.min() == pd.Timestamp("2025-01-01 08:00")
    assert panel.loc[2].index.min() == pd.Timestamp("2025-01-01 08:00")  # primera llegada


def _panel_neto(netos, estacion=1):
    horas = pd.date_range("2025-01-01", periods=len(netos), freq="h")
    idx = pd.MultiIndex.from_product([[estacion], horas], names=["estacion", "hora"])
    return pd.DataFrame({"neto": netos, "salidas": 0, "llegadas": 0}, index=idx)


@pytest.mark.parametrize(
    ("netos", "riesgo", "tipo"),
    [
        ([0, -2, -2, 0], 1, "sin_bicis"),  # |−4| = 0.4 × 10: justo en el umbral (límite)
        ([0, -2, -1, 0], 0, "normal"),  # |−3| < 4
        ([0, 3, 1, 0], 1, "sin_anclajes"),
    ],
)
def test_etiqueta_en_el_umbral(netos, riesgo, tipo):
    df = p2_agregar_etiqueta(_panel_neto(netos), horizonte=2, capacidad=pd.Series({1: 10}),
                             fraccion=0.4)
    primera = df.iloc[0]
    assert primera["riesgo"] == riesgo and primera["tipo_riesgo"] == tipo


def test_etiqueta_descarta_horas_sin_ventana_y_capacidad_cero():
    df = p2_agregar_etiqueta(_panel_neto([0, 1, 2, 3, 4]), 2, pd.Series({1: 10}), 0.4)
    assert len(df) == 3  # las 2 últimas horas no tienen ventana futura completa
    sin_cap = p2_agregar_etiqueta(_panel_neto([0, 1, 2, 3]), 2, pd.Series({1: 0}), 0.4)
    assert sin_cap.empty


@pytest.mark.parametrize("horizonte", [0, 24, -1])
def test_horizonte_fuera_de_rango(horizonte):
    with pytest.raises(ValueError):
        p2_agregar_features(_panel_neto([0] * 5).assign(capacidad=10), horizonte)


def test_features_no_usan_el_futuro():
    """Cambiar el flujo de t+1 en adelante no altera ninguna variable de la hora t."""
    rng = np.random.default_rng(0)
    netos = rng.integers(-3, 4, size=24 * 9)
    base = _panel_neto(netos).assign(capacidad=12)
    alterado = base.copy()
    t = 24 * 8
    alterado.iloc[t + 1:, alterado.columns.get_loc("neto")] += 50
    cols = [c for c in P2_FEATURES if c not in ("tasa_riesgo_hist", "neto_futuro_medio_hist")]
    f_base = p2_agregar_features(base, 2)[cols].iloc[: t + 1]
    f_alt = p2_agregar_features(alterado, 2)[cols].iloc[: t + 1]
    pd.testing.assert_frame_equal(f_base, f_alt)


def test_historico_pasado_excluye_la_fila_actual():
    horas = pd.to_datetime(["2025-01-06 08:00", "2025-01-07 08:00", "2025-01-08 08:00"])
    idx = pd.MultiIndex.from_product([[1], horas], names=["estacion", "hora"])
    train = pd.DataFrame({"riesgo": [1, 0, 1], "neto_futuro": [4.0, 0.0, 2.0], "hora_dia": 8,
                          "fin_semana": 0}, index=idx)
    out = p2_historico_pasado(train)
    assert np.isnan(out["tasa_riesgo_hist"].iloc[0])  # sin pasado
    assert out["tasa_riesgo_hist"].tolist()[1:] == [1.0, 0.5]
    assert out["neto_futuro_medio_hist"].tolist()[1:] == [4.0, 2.0]


def test_historico_de_train_se_aplica_sin_recalcular():
    idx = pd.MultiIndex.from_tuples([(1, pd.Timestamp("2025-01-06 08:00"))],
                                    names=["estacion", "hora"])
    train = pd.DataFrame({"riesgo": [1], "neto_futuro": [5.0], "hora_dia": 8, "fin_semana": 0,
                          "estacion": [1]}, index=idx).drop(columns="estacion")
    hist = p2_ajustar_historico(train.reset_index())
    nuevo = pd.DataFrame({"hora_dia": [8, 9], "fin_semana": [0, 0]},
                         index=pd.MultiIndex.from_tuples(
                             [(1, pd.Timestamp("2025-11-03 08:00")),
                              (1, pd.Timestamp("2025-11-03 09:00"))], names=["estacion", "hora"]))
    out = p2_aplicar_historico(nuevo, hist)
    assert out["tasa_riesgo_hist"].iloc[0] == 1.0
    assert np.isnan(out["tasa_riesgo_hist"].iloc[1])  # grupo que no existe en TRAIN


def test_cortar_excluye_ventanas_que_cruzan_el_limite():
    df = _panel_neto([0] * 10)
    out = p2.cortar(df, "2025-01-01 00:00", "2025-01-01 06:00")
    horas = out.index.get_level_values("hora")
    assert horas.max() == pd.Timestamp("2025-01-01 03:00")  # 03:00 + 2 h = 05:00 < 06:00
