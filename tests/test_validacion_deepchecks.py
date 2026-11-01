"""Validación de datos con DeepChecks (``mibici/validacion.py``).

1. Con datos sintéticos se comprueba que las suites DETECTAN los problemas que deben
   detectar (nulos, duplicados, tipos mezclados, drift, fuga por fechas, reglas de negocio)
   y que todos los chequeos corren sin error con las versiones fijadas de las librerías.
2. Con los datos reales (``@pytest.mark.datos``) se comprueba la integridad del dataset de
   MiBici y de las tablas de modelado, y se fijan los hallazgos documentados en el README.
"""

import numpy as np
import pandas as pd
import pytest

from mibici import validacion as v
from mibici.config import SEED

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _estados(resultado) -> dict:
    return {f["chequeo"]: f["estado"] for f in v.resumir(resultado)}


def _tabla(n: int = 600, desplazamiento: float = 0.0, inicio: str = "2025-01-01",
           semilla: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    return pd.DataFrame({
        "fecha": pd.date_range(inicio, periods=n, freq="h"),
        "x_num": rng.normal(desplazamiento, 1, n),
        "x_conteo": rng.poisson(5 + 5 * desplazamiento, n).astype(float),
        "x_cat": rng.choice(["a", "b", "c"], n),
        "y": rng.integers(0, 2, n),
    })


def _dataset(df: pd.DataFrame):
    return v.Dataset(df, label="y", label_type="binary", datetime_name="fecha",
                     features=["x_num", "x_conteo", "x_cat"], cat_features=["x_cat"])


SIN_REGLAS: list = []
DETECTADO = {"no aprobado", "advertencia"}  # DeepChecks marca algunos hallazgos como WARN

# ---------------------------------------------------------------------------
# 1. Las suites detectan problemas plantados (datos sintéticos)
# ---------------------------------------------------------------------------


def test_datos_limpios_aprueban_integridad():
    estados = _estados(v.suite_integridad("limpio", SIN_REGLAS, "binary").run(_dataset(_tabla())))
    assert "error" not in estados.values(), estados
    for chequeo in ("PercentOfNulls", "Data Duplicates", "Mixed Data Types", "Mixed Nulls"):
        assert estados[chequeo] == "aprobado", (chequeo, estados[chequeo])


def test_detecta_nulos_duplicados_y_tipos_mezclados():
    df = _tabla()
    df.loc[:30, "x_num"] = np.nan  # nulos
    df = pd.concat([df, df.iloc[100:140]], ignore_index=True)  # filas duplicadas
    df["x_cat"] = df["x_cat"].astype(object)
    df.loc[200:203, "x_cat"] = 7  # números dentro de una columna de texto
    estados = _estados(v.suite_integridad("sucio", SIN_REGLAS, "binary").run(_dataset(df)))
    assert estados["PercentOfNulls"] in DETECTADO
    assert estados["Data Duplicates"] in DETECTADO
    assert estados["Mixed Data Types"] in DETECTADO


def test_detecta_nulos_de_distinto_tipo():
    df = _tabla()
    df["x_cat"] = df["x_cat"].astype(object)
    df.loc[:5, "x_cat"] = None
    df.loc[6:10, "x_cat"] = "N/A"
    estados = _estados(v.suite_integridad("nulos", SIN_REGLAS, "binary").run(_dataset(df)))
    assert estados["Mixed Nulls"] in DETECTADO


def test_reglas_de_negocio_de_viajes():
    ini = pd.to_datetime(["2025-01-03 08:00", "2025-01-03 09:00", "2025-01-03 10:00",
                          "2025-01-03 11:00"])
    viajes = pd.DataFrame({
        "viaje_id": [1, 2, 2, 4],  # id repetido
        "inicio_viaje": ini,
        "fin_viaje": ini + pd.to_timedelta([600, 30, 900, -60], unit="s"),  # corto y negativo
        "origen_id": [10, 10, 11, 99],  # 99 no existe en el catálogo
        "destino_id": [11, 10, 10, 10],
        "mes_archivo": ["2025_01", "2025_01", "2025_01", "2025_02"],  # mes incoherente
    })
    df, reglas = v.preparar_viajes(viajes, ids_estaciones={10, 11})
    valor = v.ReglasDeNegocio(reglas).run(v.dataset_viajes(df)).value
    assert valor["viaje_id duplicado"] == 1
    assert valor["fin anterior al inicio"] == 1
    assert valor["estación de origen fuera del catálogo"] == 1
    assert valor["mes del archivo distinto al mes de inicio"] == 1
    assert valor["duración menor a 60 s (se descarta en el pipeline)"] == 2
    assert valor["origen igual a destino (viaje circular)"] == 1
    check = v.ReglasDeNegocio(reglas).add_condition_reglas_criticas_sin_violaciones()
    assert not check.run(v.dataset_viajes(df)).passed_conditions()


def test_reglas_informativas_no_reprueban():
    """Solo las reglas críticas definen la condición; las informativas solo se reportan."""
    df = pd.DataFrame({"x": [1, -1, 2]})
    reglas = [v.ReglaNegocio("negativo", lambda d: d["x"] < 0, critica=False)]
    check = v.ReglasDeNegocio(reglas).add_condition_reglas_criticas_sin_violaciones()
    resultado = check.run(v.Dataset(df, cat_features=[]))
    assert resultado.value == {"negativo": 1}
    assert resultado.passed_conditions()


def test_detecta_drift_entre_entrenamiento_y_prueba():
    train = _dataset(_tabla(desplazamiento=0.0))
    test = _dataset(_tabla(desplazamiento=1.5, inicio="2025-03-01", semilla=SEED + 1))
    resultado = v.suite_entrenamiento_prueba("drift", "binary").run(train_dataset=train,
                                                                    test_dataset=test)
    estados = _estados(resultado)
    assert "error" not in estados.values(), estados
    assert estados["Feature Drift"] in DETECTADO
    assert estados["Multivariate Drift"] in DETECTADO
    assert estados["Date Train-Test Leakage (overlap)"] == "aprobado"
    drift = v.valor_de(resultado, "Feature Drift")
    assert drift["x_num"]["Drift score"] > 0.2


def test_sin_drift_aprueba():
    train = _dataset(_tabla(n=2000))
    test = _dataset(_tabla(n=2000, inicio="2025-04-01", semilla=SEED + 1))
    estados = _estados(v.suite_entrenamiento_prueba("estable", "binary").run(
        train_dataset=train, test_dataset=test))
    assert estados["Feature Drift"] == "aprobado"
    assert estados["Multivariate Drift"] == "aprobado"
    assert estados["Label Drift"] == "aprobado"


def test_detecta_fuga_por_fechas():
    """Si prueba incluye fechas anteriores al fin de entrenamiento, hay fuga temporal."""
    train = _dataset(_tabla(n=500))
    test = _dataset(_tabla(n=500, inicio="2025-01-10", semilla=SEED + 1))  # se traslapa
    estados = _estados(v.suite_entrenamiento_prueba("fuga", "binary").run(
        train_dataset=train, test_dataset=test))
    assert estados["Date Train-Test Leakage (overlap)"] in DETECTADO


def test_compatibilidad_correlacion_entre_variables():
    """Con pandas 3 la condición original reportaba todos los pares; el ajuste lo corrige."""
    df = _tabla()
    df["x_copia"] = df["x_num"] * 2 + 1  # único par correlacionado
    ds = v.Dataset(df, label="y", label_type="binary", datetime_name="fecha",
                   features=["x_num", "x_conteo", "x_copia"], cat_features=[])
    check = v.dc.FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        0.9, 0)
    detalle = check.run(ds).conditions_results[0].details
    assert "('x_copia', 'x_num')" in detalle and "x_conteo" not in detalle


def test_resumen_markdown(tmp_path):
    resultado = v.suite_integridad("md", SIN_REGLAS, "binary").run(_dataset(_tabla()))
    resumen = {"conjunto": "demo", "descripcion": "sintético", "filas": {"train": 600},
               "suites": {"integridad_train": {"reporte": "demo.html",
                                               "chequeos": v.resumir(resultado)}}}
    texto = v.escribir_resumen_md([resumen], tmp_path).read_text(encoding="utf-8")
    assert "## demo" in texto and "| Data Duplicates | aprobado |" in texto


# ---------------------------------------------------------------------------
# 2. Datos reales de MiBici (se omiten si no existen los Parquet)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def viajes_reales():
    return v.conjunto_viajes(muestra=50_000)


@pytest.mark.datos
def test_viajes_reales_reglas_criticas(viajes_reales):
    """Integridad del dataset completo (4.5 millones de viajes)."""
    total = sum(viajes_reales.filas.values())
    for clave in ("train", "test"):
        check = v.ReglasDeNegocio(viajes_reales.reglas, viajes_reales.completos[clave])
        valor = check.run(viajes_reales.train).value
        for regla in ("viaje_id duplicado", "fecha de inicio o fin nula", "fin anterior al inicio",
                      "estación de origen fuera del catálogo",
                      "estación de destino fuera del catálogo",
                      "mes del archivo distinto al mes de inicio"):
            assert valor[regla] == 0, (clave, regla)
        # Hallazgo documentado: ~1,076 pares de viajes repetidos con id consecutivo (0.024 %).
        assert valor["viaje repetido (mismo origen, destino, inicio y fin)"] / total < 5e-4


@pytest.mark.datos
def test_viajes_reales_sin_fuga_entre_conjuntos(viajes_reales):
    estados = _estados(v.suite_entrenamiento_prueba(
        "viajes", con_indice=True).run(train_dataset=viajes_reales.train,
                                      test_dataset=viajes_reales.test))
    assert "error" not in estados.values(), estados
    assert estados["Date Train-Test Leakage (overlap)"] == "aprobado"
    assert estados["Index Train-Test Leakage"] == "aprobado"


@pytest.mark.datos
@pytest.mark.lento
def test_tabla_p1_integridad_y_hallazgos():
    conjunto = v.conjunto_p1()
    resultados = v.validar(conjunto)
    for clave in ("integridad_train", "integridad_test"):
        estados = _estados(resultados[clave])
        assert "error" not in estados.values(), estados
        for chequeo in ("PercentOfNulls", "Data Duplicates", "Reglas de negocio MiBici",
                        "Mixed Data Types"):
            assert estados[chequeo] == "aprobado", (clave, chequeo)
    tt = resultados["train_test"]
    assert _estados(tt)["Date Train-Test Leakage (overlap)"] == "aprobado"
    # Hallazgo documentado: solo `dias_activa` (contador que crece con el tiempo) tiene drift.
    drift = v.valor_de(tt, "Feature Drift")
    con_drift = {k for k, d in drift.items() if d["Drift score"] >= v.UMBRAL_DRIFT_VARIABLE}
    assert con_drift == {"dias_activa"}
    # Sin ese contador ya no hay drift: ni por variable ni multivariado.
    sin_tiempo = _estados(resultados["train_test_sin_tiempo"])
    assert sin_tiempo["Feature Drift"] == "aprobado"
    assert sin_tiempo["Multivariate Drift"] == "aprobado"


@pytest.mark.datos
@pytest.mark.lento
def test_perfiles_p3_integridad_y_estabilidad():
    resultados = v.validar(v.conjunto_p3())
    for clave in ("integridad_train", "integridad_test"):
        estados = _estados(resultados[clave])
        for chequeo in ("PercentOfNulls", "Data Duplicates", "Reglas de negocio MiBici"):
            assert estados[chequeo] == "aprobado", (clave, chequeo)
    assert _estados(resultados["train_test"])["Multivariate Drift"] == "aprobado"
