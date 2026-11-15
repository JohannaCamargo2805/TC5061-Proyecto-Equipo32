"""Pruebas de la entrega 5: monitoreo con Evidently, reentrenamiento y A/B testing de P1.

Las unitarias usan datos sintéticos; las marcadas ``datos`` usan los Parquet reales
(2025 y producción 2026) y se omiten si no existen.
"""

import numpy as np
import pandas as pd
import pytest

from mibici import config
from mibici import monitoreo as mon
from mibici.features import P1_FEATURES
from mibici.modeling import ab_testing as ab

HAY_PRODUCCION = config.VIAJES_PRODUCCION_PARQUET.exists() and config.VIAJES_PARQUET.exists()


# ---------------------------------------------------------------------------
# Configuración: periodos sin traslape ni fuga
# ---------------------------------------------------------------------------
def test_periodos_sin_traslape_y_sin_fuga():
    ts = pd.Timestamp
    ref_ini, ref_fin = map(ts, config.MONITOREO_REFERENCIA)
    act_ini, act_fin = map(ts, config.MONITOREO_ACTUAL)
    _, tr_fin = map(ts, config.REENTRENO_TRAIN)
    val_ini, val_fin = map(ts, config.REENTRENO_VAL)
    ev_ini, ev_fin = map(ts, config.AB_EVALUACION)
    # La referencia es exactamente el TRAIN de la v1
    assert (ref_ini, ref_fin) == (ts(config.P1_TRAIN_START), ts(config.P1_VAL_START))
    # El periodo actual es posterior a todo lo que vio la v1 (incluida su prueba)
    assert act_ini >= ts(config.P1_TEST_END)
    # Reentreno: TRAIN < VAL < evaluación del A/B, sin traslape
    assert tr_fin <= val_ini < val_fin <= ev_ini < ev_fin
    # El A/B se evalúa sobre el periodo del monitoreo
    assert (ev_ini, ev_fin) == (act_ini, act_fin)


def test_feriados_2026_son_los_de_la_ley():
    feriados = pd.to_datetime([f for f in config.P1_FERIADOS if f.startswith("2026")])
    assert len(feriados) == 9
    lunes = {"2026-02-02", "2026-03-16", "2026-11-16"}  # primer lunes de feb, tercero de mar y nov
    assert all(pd.Timestamp(d).dayofweek == 0 for d in lunes)
    assert {"2026-04-02", "2026-04-03"} <= {str(d.date()) for d in feriados}  # Jueves y Viernes Santo
    # Los de 2025 no cambiaron (la v1 se reproduce igual)
    assert [f for f in config.P1_FERIADOS if f.startswith("2025")] == [
        "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-17", "2025-04-18",
        "2025-05-01", "2025-09-16", "2025-11-17", "2025-12-25"]


def test_reglas_de_decision_fijas_en_config():
    assert config.AB_METRICA == "mae"
    assert config.AB_BOOTSTRAP_REPS >= 1000
    assert 0 < config.AB_ALFA <= 0.05


# ---------------------------------------------------------------------------
# Evidently: el reporte detecta el drift donde lo hay y solo ahí
# ---------------------------------------------------------------------------
def _tabla_sintetica(n: int, desplazamiento: float = 0.0, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2025-01-01", periods=n // 10, freq="D").repeat(10)[:n]
    t = pd.DataFrame({"estacion": np.tile(np.arange(10), n // 10)[:n], "fecha": fechas})
    for c in P1_FEATURES:
        t[c] = rng.poisson(30, n).astype("float64")
    t["dia_semana"] = t["fecha"].dt.dayofweek.astype("int64")
    t["fin_semana"] = (t["dia_semana"] >= 5).astype("int64")
    t["feriado"] = 0
    t["dias_activa"] = rng.integers(28, 250, n).astype("float64") + desplazamiento
    t["viajes"] = rng.poisson(30, n).astype("float64")
    return t


def test_evidently_detecta_drift_solo_en_la_variable_desplazada():
    ref = _tabla_sintetica(3000, semilla=1)
    cur = _tabla_sintetica(2000, desplazamiento=400, semilla=2)
    snap = mon.reporte_drift(ref, cur, con_resumen=False)
    tabla = mon.tabla_drift(snap).set_index("columna")
    assert bool(tabla.loc["dias_activa", "drift"])
    assert not tabla.drop(index="dias_activa")["drift"].any()
    assert set(tabla.index) == set(P1_FEATURES) | {"viajes"}
    resumen = mon.resumen_conjunto(snap)
    assert resumen["columnas_con_drift"] == 1


def test_evidently_psi_y_html(tmp_path):
    ref = _tabla_sintetica(2000, semilla=3)
    cur = _tabla_sintetica(2000, desplazamiento=400, semilla=4)
    snap = mon.reporte_drift(ref, cur, metodo="psi", umbral=config.DRIFT_PSI_UMBRAL)
    tabla = mon.tabla_drift(snap)
    assert tabla["metodo"].str.contains("PSI", case=False).all()
    assert tabla.set_index("columna").loc["dias_activa", "drift"]
    ruta = tmp_path / "r.html"
    snap.save_html(str(ruta))
    assert ruta.stat().st_size > 10_000
    assert len(mon.resultados_pruebas(snap)) > 0


def test_estadisticos_fuera_de_rango():
    ref = _tabla_sintetica(1000, semilla=5)
    cur = _tabla_sintetica(1000, desplazamiento=1000, semilla=6)
    est = mon.comparar_estadisticos(ref, cur).set_index("columna")
    assert est.loc["dias_activa", "fuera_rango_pct"] == 100
    assert est.loc["lag_1", "fuera_rango_pct"] < 5


def test_efecto_en_modelo_glm():
    from mibici.modeling.p1_regresion import construir_lineal

    ref = _tabla_sintetica(2000, semilla=7)
    modelo = construir_lineal().fit(ref[P1_FEATURES], ref["viajes"])
    sin_cambio = mon.efecto_en_modelo(modelo, ref, ref)
    assert np.allclose(sin_cambio["factor_prediccion"], 1.0)
    cur = _tabla_sintetica(2000, desplazamiento=300, semilla=8)
    efecto = mon.efecto_en_modelo(modelo, ref, cur).set_index("termino")
    assert efecto.loc["num__dias_activa", "delta_z"] > 3
    assert len(mon.terminos_glm(modelo)) == 7 + 11  # 7 días en one-hot + 11 numéricas


# ---------------------------------------------------------------------------
# A/B testing
# ---------------------------------------------------------------------------
def _errores(n_dias=60, por_dia=50, mejora=0.5, semilla=0):
    rng = np.random.default_rng(semilla)
    fecha = pd.Index(np.repeat(pd.date_range("2026-07-01", periods=n_dias), por_dia))
    ea = rng.gamma(2.0, 2.5, len(fecha))
    eb = np.clip(ea - mejora + rng.normal(0, 0.3, len(fecha)), 0, None)
    y = rng.poisson(30, len(fecha)).astype(float)
    return ea, eb, y, fecha


def test_bootstrap_por_dias_detecta_mejora_y_es_reproducible():
    ea, eb, y, fecha = _errores()
    boot = ab.bootstrap_dias(ea, eb, y, fecha, reps=500)
    lo, hi = ab.intervalo(boot["delta_mae"])
    assert hi < 0
    assert lo <= (eb.mean() - ea.mean()) <= hi
    assert boot.equals(ab.bootstrap_dias(ea, eb, y, fecha, reps=500))  # semilla fija


def test_bootstrap_sin_diferencia_incluye_cero():
    ea, _, y, fecha = _errores()
    rng = np.random.default_rng(9)
    eb = np.abs(ea + rng.normal(0, 0.5, len(ea)))
    eb = eb * (ea.mean() / eb.mean())
    lo, hi = ab.intervalo(ab.bootstrap_dias(ea, eb, y, fecha, reps=500)["delta_mae"])
    assert lo < 0 < hi


def test_asignacion_de_brazo_determinista_y_balanceada():
    brazos = [ab.asignar_brazo(e) for e in range(1, 1001)]
    assert brazos == [ab.asignar_brazo(e) for e in range(1, 1001)]
    assert 0.45 < brazos.count("A") / len(brazos) < 0.55


def test_decision_aplica_las_cuatro_reglas():
    m_a = {"mae": 6.0, "sesgo": -2.0}
    m_b = {"mae": 5.7, "sesgo": -1.0}
    seg_ok = pd.DataFrame({"dimension": ["mes", "mes"], "cambio_mae_pct": [-3.0, 1.0]})
    assert ab.decidir(m_a, m_b, (-0.4, -0.1), seg_ok, 0.3, 0.3)["ganadora"] == "B"
    # IC que cruza el cero: se queda A
    assert ab.decidir(m_a, m_b, (-0.4, 0.05), seg_ok, 0.3, 0.3)["ganadora"] == "A"
    # Un mes empeora más de 2 %: se queda A
    seg_mal = pd.DataFrame({"dimension": ["mes", "mes"], "cambio_mae_pct": [-6.0, 2.5]})
    assert ab.decidir(m_a, m_b, (-0.4, -0.1), seg_mal, 0.3, 0.3)["ganadora"] == "A"
    # Más sesgo o más costo: se queda A
    assert ab.decidir(m_a, {"mae": 5.7, "sesgo": -3.0}, (-0.4, -0.1), seg_ok, 0.3, 0.3)[
        "ganadora"] == "A"
    assert ab.decidir(m_a, m_b, (-0.4, -0.1), seg_ok, 0.3, 0.9)["ganadora"] == "A"


def test_uri_por_alias_version_o_carpeta(tmp_path):
    assert ab.uri("champion") == "models:/mibici-equipo32-p1-demanda@champion"
    assert ab.uri("3") == "models:/mibici-equipo32-p1-demanda/3"
    assert ab.uri(str(tmp_path)) == str(tmp_path)


# ---------------------------------------------------------------------------
# Datos reales
# ---------------------------------------------------------------------------
@pytest.mark.datos
@pytest.mark.lento
@pytest.mark.skipif(not HAY_PRODUCCION, reason="falta data/processed/viajes_produccion.parquet")
def test_conjuntos_reales_reproducen_los_de_la_v1():
    """La referencia es el TRAIN de la v1 y su prueba reproduce el test_mae registrado."""
    import json

    import mlflow

    panel = mon.construir_panel()
    datos = mon.conjuntos(panel)
    assert datos["referencia"]["fecha"].max() < pd.Timestamp(config.P1_VAL_START)
    assert datos["actual"]["fecha"].min() >= pd.Timestamp(config.MONITOREO_ACTUAL[0])
    assert len(datos["prueba_v1"]) == 22_565
    registro = json.loads((config.SERVICIO_MODELO_DIR / "registro.json").read_text(encoding="utf-8"))
    modelo = mlflow.sklearn.load_model(str(config.SERVICIO_MODELO_DIR))
    t = datos["prueba_v1"]
    mae = np.abs(t["viajes"] - np.clip(modelo.predict(t[P1_FEATURES]), 0, None)).mean()
    assert mae == pytest.approx(registro["metricas"]["test_mae"], abs=1e-4)
