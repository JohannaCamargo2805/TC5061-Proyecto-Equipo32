"""Pruebas de las funciones de P3 (clustering de estaciones)."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    P3_MODEL_FEATURES,
    p3_construir_features,
    p3_entropia_normalizada,
    p3_filtrar_viajes,
)


@pytest.mark.parametrize(
    ("segundos", "se_conserva"),
    [(59, False), (60, True), (600, True), (86_400, True), (86_401, False)],
)
def test_filtro_de_duracion_en_los_limites(segundos, se_conserva):
    ini = pd.Timestamp("2025-03-01 10:00")
    viajes = pd.DataFrame({"inicio_viaje": [ini], "fin_viaje": [ini + pd.Timedelta(seconds=segundos)],
                           "origen_id": [1], "destino_id": [2]})
    assert len(p3_filtrar_viajes(viajes)) == int(se_conserva)


@pytest.mark.parametrize(
    ("serie", "esperado"),
    [
        (pd.Series([5, 5, 5]), 0.0),  # una sola categoría
        (pd.Series([1, 2, 3, 4]), 1.0),  # uniforme
        (pd.Series([], dtype=int), 0.0),  # vacía (límite)
    ],
)
def test_entropia_normalizada(serie, esperado):
    assert p3_entropia_normalizada(serie) == pytest.approx(esperado)


def test_entropia_entre_cero_y_uno():
    rng = np.random.default_rng(1)
    valor = p3_entropia_normalizada(pd.Series(rng.integers(0, 6, 500)))
    assert 0 < valor < 1


def test_perfiles_validos(viajes_sinteticos):
    fin = viajes_sinteticos["fin_viaje"].max()
    features, cobertura = p3_construir_features(viajes_sinteticos, fin, min_history_days=30)
    assert list(features.columns) == P3_MODEL_FEATURES
    assert features.notna().all().all()
    salida = [c for c in P3_MODEL_FEATURES if c.startswith("salida_")]
    llegada = [c for c in P3_MODEL_FEATURES if c.startswith("llegada_")]
    np.testing.assert_allclose(features[salida].sum(axis=1), 1)
    np.testing.assert_allclose(features[llegada].sum(axis=1), 1)
    assert features["balance_relativo"].between(-1, 1).all()
    assert features["share_fin_semana"].between(0, 1).all()
    assert cobertura["eligible_p3"].all()


def test_estacion_con_poca_historia_no_es_elegible(viajes_sinteticos):
    ini = pd.Timestamp("2025-03-10 08:00")
    nueva = pd.DataFrame({"inicio_viaje": [ini], "fin_viaje": [ini + pd.Timedelta(minutes=10)],
                          "origen_id": [99], "destino_id": [1]})
    viajes = pd.concat([viajes_sinteticos, nueva], ignore_index=True)
    features, cobertura = p3_construir_features(viajes, viajes["fin_viaje"].max())
    assert 99 not in features.index
    assert not cobertura.set_index("estacion").loc[99, "eligible_p3"]


def test_sin_estaciones_elegibles_lanza_error(viajes_sinteticos):
    with pytest.raises(ValueError, match="No hay estaciones elegibles"):
        p3_construir_features(viajes_sinteticos, viajes_sinteticos["fin_viaje"].max(),
                              min_history_days=10_000)
