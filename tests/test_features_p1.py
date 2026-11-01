"""Pruebas de las variables de P1: el panel es contiguo y no hay fuga de información."""

import pandas as pd
import pytest

from mibici.features import LeakageError, p1_agregar_features, p1_construir_panel_diario


@pytest.fixture
def viajes():
    # Estación 1: 1 viaje el día d (d = 1..20). Estación 2 empieza el día 10.
    filas = []
    for d in range(1, 21):
        filas += [{"inicio_viaje": pd.Timestamp(2025, 1, d, 8), "origen_id": 1}] * d
        if d >= 10:
            filas.append({"inicio_viaje": pd.Timestamp(2025, 1, d, 9), "origen_id": 2})
    return pd.DataFrame(filas)


def test_panel_contiguo_desde_primera_actividad(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    assert panel.loc[1].shape[0] == 20
    assert panel.loc[2].index.min() == pd.Timestamp("2025-01-10")
    assert panel.loc[(1, pd.Timestamp("2025-01-05")), "viajes"] == 5


def test_rezagos_solo_usan_dias_anteriores(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1, 7], [7])
    fila = df.loc[(1, pd.Timestamp("2025-01-15"))]
    assert fila["lag_1"] == 14
    assert fila["lag_7"] == 8
    assert fila["media_7d"] == pytest.approx(sum(range(8, 15)) / 7)  # días 8..14, no incluye el 15


def test_rezago_menor_al_horizonte_lanza_error(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    with pytest.raises(LeakageError):
        p1_agregar_features(panel, pd.Series(dtype=float), [], [0, 7], [7], horizonte=1)


# ---------- Casos límite (entrega de pruebas y seguridad) ----------
def test_viajes_fuera_del_rango_dan_panel_vacio(viajes):
    """Si toda la actividad es posterior al rango pedido, ninguna estación entra al panel."""
    panel = p1_construir_panel_diario(viajes, "2024-12-01", "2024-12-31")
    assert panel.empty


def test_calendario_feriado_y_fin_de_semana(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20}), ["2025-01-01"], [1, 7], [7])
    assert df.loc[(1, pd.Timestamp("2025-01-01")), "feriado"] == 1  # Año Nuevo
    assert df.loc[(1, pd.Timestamp("2025-01-02")), "feriado"] == 0
    assert df.loc[(1, pd.Timestamp("2025-01-04")), "fin_semana"] == 1  # sábado
    assert df.loc[(1, pd.Timestamp("2025-01-06")), "fin_semana"] == 0  # lunes


def test_capacidad_desconocida_se_rellena_con_cero(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20}), [], [1], [7])  # falta la estación 2
    assert (df.xs(2, level="estacion")["capacidad"] == 0).all()
    assert (df.xs(1, level="estacion")["capacidad"] == 20).all()


def test_sin_historia_suficiente_los_rezagos_son_nulos(viajes):
    """Los primeros k días de cada estación no tienen lag_k: quedan NaN (el pipeline los descarta)."""
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1, 7], [7])
    est2 = df.xs(2, level="estacion")
    assert est2["lag_7"].iloc[:7].isna().all()
    assert est2["lag_7"].iloc[7:].notna().all()
    assert est2["dias_activa"].tolist() == list(range(len(est2)))


def test_features_identicas_si_cambia_el_futuro(viajes_sinteticos, capacidad_sintetica):
    """Sin fuga: alterar los viajes de un día no cambia las variables de ese día ni de antes."""
    corte = pd.Timestamp("2025-02-15")
    base = p1_construir_panel_diario(viajes_sinteticos, "2025-01-01", "2025-03-11")
    alterado = base.copy()
    alterado.loc[alterado.index.get_level_values("fecha") >= corte, "viajes"] *= 10
    args = (capacidad_sintetica, [], [1, 2, 7, 14], [7, 28])
    f_base = p1_agregar_features(base, *args).drop(columns="viajes")
    f_alt = p1_agregar_features(alterado, *args).drop(columns="viajes")
    hasta = f_base.index.get_level_values("fecha") <= corte
    pd.testing.assert_frame_equal(f_base.loc[hasta], f_alt.loc[hasta])
