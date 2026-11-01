"""Validación de datos con DeepChecks: integridad, calidad y drift del dataset MiBici.

Se validan cuatro conjuntos, siempre con la misma partición temporal de ``config.py``:

* ``viajes``: los viajes procesados por la ingesta (``data/processed/viajes.parquet``).
  Entrenamiento = viajes que inician antes de ``TEST_START``; prueba = a partir de esa fecha.
* ``p1``: tabla de modelado de P1 (estación × día, variables ``p1_*`` y etiqueta ``viajes``).
* ``p2``: tabla de modelado de P2 (estación × hora, variables ``p2_*`` y etiqueta ``riesgo``).
* ``p3``: perfiles de estación de P3 (variables ``p3_*``, sin etiqueta).

Para cada conjunto se corren dos suites:

* **Integridad** (un conjunto): nulos, tipos mezclados, valores únicos, duplicados,
  caracteres especiales, variantes de texto, atípicos, correlaciones, desbalance de clases
  y un chequeo propio, :class:`ReglasDeNegocio`, con las reglas del dominio MiBici.
* **Entrenamiento contra prueba**: drift por variable, drift de la etiqueta, drift
  multivariado, categorías nuevas, fuga por fechas o por índice y tamaño de los conjuntos.

Uso (genera los reportes HTML y el resumen en ``reports/deepchecks``)::

    python -m mibici.validacion                 # los cuatro conjuntos
    python -m mibici.validacion --conjuntos p1  # solo uno

Las pruebas de ``tests/test_validacion_deepchecks.py`` usan estas mismas suites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

from mibici import compat_deepchecks

compat_deepchecks.aplicar()  # antes de importar DeepChecks

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.check_result import CheckFailure
from deepchecks.tabular import Context, Dataset, SingleDatasetCheck, Suite
import deepchecks.tabular.checks as dc
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    P2_TEST_START,
    P2_TRAIN_START,
    REPORTS_DIR,
    SEED,
    TEST_START,
    VIAJES_PARQUET,
)

SALIDA_DIR = REPORTS_DIR / "deepchecks"
CONJUNTOS = ("viajes", "p1", "p2", "p3")

# Umbrales de las condiciones (documentados en el README).
UMBRAL_NULOS = 0.0  # ninguna variable de entrada puede tener nulos
UMBRAL_DUPLICADOS = 0.0  # ninguna fila duplicada
UMBRAL_DRIFT_VARIABLE = 0.2  # Kolmogorov-Smirnov (numéricas) o Cramér's V (categóricas)
UMBRAL_DRIFT_MULTIVARIADO = 0.25  # drift score = 2·AUC − 1 del clasificador de dominio
UMBRAL_PPS_FUGA = 0.9  # PPS de una variable con la etiqueta por encima de esto sugiere fuga
MUESTRA_P2 = 200_000  # filas por conjunto de P2 (estación × hora tiene ~3 millones)
MUESTRA_VIAJES = 300_000  # filas por conjunto de viajes para los chequeos estadísticos


# =============================================================================
# Chequeo propio: reglas de negocio de MiBici
# =============================================================================
Regla = Callable[[pd.DataFrame], pd.Series]


@dataclass
class ReglaNegocio:
    """Una regla de dominio: ``evaluar(df)`` devuelve True en las filas que la VIOLAN."""

    nombre: str
    evaluar: Regla
    critica: bool = True  # crítica: debe cumplirse siempre; informativa: solo se reporta


class ReglasDeNegocio(SingleDatasetCheck):
    """Cuenta las violaciones de reglas del dominio MiBici en TODAS las filas del conjunto.

    Las reglas críticas tienen la condición "0 violaciones"; las informativas se reportan
    (por ejemplo, viajes de menos de 60 s que el pipeline descarta después).

    Si el ``Dataset`` de DeepChecks es una muestra (viajes: 4.5 millones de filas), ``datos``
    recibe el conjunto completo para que las reglas se evalúen sobre todas las filas.
    """

    def __init__(self, reglas: list[ReglaNegocio], datos: pd.DataFrame | None = None, **kwargs):
        super().__init__(**kwargs)
        self.reglas = reglas
        self.datos = datos

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        df = self.datos if self.datos is not None else context.get_data_by_kind(dataset_kind).data
        filas = []
        for r in self.reglas:
            violaciones = int(np.asarray(r.evaluar(df), dtype=bool).sum())
            filas.append({
                "regla": r.nombre,
                "tipo": "crítica" if r.critica else "informativa",
                "violaciones": violaciones,
                "porcentaje": round(100 * violaciones / max(len(df), 1), 4),
            })
        tabla = pd.DataFrame(filas)
        valor = {f["regla"]: f["violaciones"] for f in filas}
        return CheckResult(valor, header="Reglas de negocio MiBici",
                           display=[f"Filas revisadas: {len(df):,}", tabla])

    def add_condition_reglas_criticas_sin_violaciones(self):
        criticas = [r.nombre for r in self.reglas if r.critica]

        def condicion(valor: dict) -> ConditionResult:
            fallas = {k: v for k, v in valor.items() if k in criticas and v > 0}
            if fallas:
                return ConditionResult(ConditionCategory.FAIL, f"Reglas violadas: {fallas}")
            return ConditionResult(ConditionCategory.PASS,
                                   f"{len(criticas)} reglas críticas sin violaciones")

        return self.add_condition("Las reglas críticas no tienen violaciones", condicion)


# =============================================================================
# Suites
# =============================================================================
def suite_integridad(nombre: str, reglas: list[ReglaNegocio], tarea: str | None = None,
                     datos_completos: pd.DataFrame | None = None) -> Suite:
    """Suite de integridad y calidad de UN conjunto. ``tarea``: regression, binary o None.

    ``datos_completos``: conjunto completo para las reglas de negocio cuando el ``Dataset``
    de DeepChecks es una muestra.
    """
    checks = [
        dc.PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(UMBRAL_NULOS),
        dc.MixedNulls().add_condition_different_nulls_less_equal_to(1),
        # Cualquier mezcla de tipos reprueba (por defecto DeepChecks ignora mezclas < 1 %).
        dc.MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.0, 0.5)),
        dc.IsSingleValue().add_condition_not_single_value(),
        dc.DataDuplicates(n_samples=1_000_000).add_condition_ratio_less_or_equal(UMBRAL_DUPLICADOS),
        dc.SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(),
        dc.StringMismatch().add_condition_no_variants(),
        dc.OutlierSampleDetection(n_samples=5_000, timeout=60, random_state=SEED),
        dc.FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
            threshold=0.9, n_pairs=5),
        ReglasDeNegocio(reglas, datos_completos).add_condition_reglas_criticas_sin_violaciones(),
    ]
    if tarea is not None:
        checks.append(dc.FeatureLabelCorrelation(random_state=SEED)
                      .add_condition_feature_pps_less_than(UMBRAL_PPS_FUGA))
    if tarea == "binary":
        checks += [dc.ClassImbalance(), dc.ConflictingLabels()]
    return Suite(f"Integridad · {nombre}", *checks)


def suite_entrenamiento_prueba(nombre: str, tarea: str | None = None, con_fecha: bool = True,
                               con_indice: bool = False, columnas_tiempo: list | None = None
                               ) -> Suite:
    """Suite de validación entre el conjunto de entrenamiento y el de prueba.

    ``columnas_tiempo``: contadores que crecen con el calendario (p. ej. ``dias_activa``). Por
    construcción separan entrenamiento de prueba, así que se agrega un segundo drift
    multivariado sin ellos para medir el drift del resto de las variables.
    """
    checks = [
        dc.DatasetsSizeComparison().add_condition_test_size_greater_or_equal(100),
        dc.FeatureDrift(n_samples=100_000, random_state=SEED).add_condition_drift_score_less_than(
            max_allowed_categorical_score=UMBRAL_DRIFT_VARIABLE,
            max_allowed_numeric_score=UMBRAL_DRIFT_VARIABLE),
        dc.MultivariateDrift(n_samples=10_000, random_state=SEED)
        .add_condition_overall_drift_value_less_than(UMBRAL_DRIFT_MULTIVARIADO),
        dc.NewCategoryTrainTest(),
        dc.TrainTestSamplesMix(),
    ]
    if columnas_tiempo:
        checks.append(
            dc.MultivariateDrift(n_samples=10_000, random_state=SEED,
                                 ignore_columns=list(columnas_tiempo))
            .add_condition_overall_drift_value_less_than(UMBRAL_DRIFT_MULTIVARIADO))
    if con_fecha:
        checks += [
            dc.DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
            dc.DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0),
        ]
    if con_indice:
        checks.append(dc.IndexTrainTestLeakage().add_condition_ratio_less_or_equal(0))
    if tarea is not None:
        checks += [dc.LabelDrift(random_state=SEED).add_condition_drift_score_less_than(
                       UMBRAL_DRIFT_VARIABLE),
                   dc.FeatureLabelCorrelationChange(random_state=SEED)]
    if tarea == "binary":
        checks.append(dc.NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(0))
    return Suite(f"Entrenamiento contra prueba · {nombre}", *checks)


# =============================================================================
# Conjuntos (mismas funciones y partición que el pipeline del proyecto)
# =============================================================================
@dataclass
class ConjuntoValidacion:
    """Un conjunto listo para DeepChecks: entrenamiento, prueba y reglas de negocio."""

    nombre: str
    descripcion: str
    train: Dataset
    test: Dataset
    reglas: list[ReglaNegocio]
    tarea: str | None = None  # regression | binary | None
    con_fecha: bool = True
    con_indice: bool = False
    filas: dict = field(default_factory=dict)
    completos: dict = field(default_factory=dict)  # train/test completos para las reglas
    columnas_tiempo: list = field(default_factory=list)  # contadores que crecen con el tiempo


def reglas_viajes(ids_estaciones: set) -> list[ReglaNegocio]:
    """Reglas de dominio de los viajes procesados."""
    return [
        ReglaNegocio("viaje_id duplicado", lambda d: d["viaje_id"].duplicated()),
        ReglaNegocio("viaje repetido (mismo origen, destino, inicio y fin)",
                     lambda d: d.duplicated(subset=["origen_id", "destino_id", "inicio_viaje",
                                                    "fin_viaje"])),
        ReglaNegocio("fecha de inicio o fin nula",
                     lambda d: d["inicio_viaje"].isna() | d["fin_viaje"].isna()),
        ReglaNegocio("fin anterior al inicio", lambda d: d["duracion_seg"] < 0),
        ReglaNegocio("estación de origen fuera del catálogo",
                     lambda d: ~d["origen_id"].isin(ids_estaciones)),
        ReglaNegocio("estación de destino fuera del catálogo",
                     lambda d: ~d["destino_id"].isin(ids_estaciones)),
        ReglaNegocio("mes del archivo distinto al mes de inicio",
                     lambda d: d["inicio_viaje"].dt.strftime("%Y_%m") != d["mes_archivo"]),
        ReglaNegocio("duración menor a 60 s (se descarta en el pipeline)",
                     lambda d: d["duracion_seg"] < 60, critica=False),
        ReglaNegocio("duración mayor a 24 h (se descarta en el pipeline)",
                     lambda d: d["duracion_seg"] > 86_400, critica=False),
        ReglaNegocio("origen igual a destino (viaje circular)",
                     lambda d: d["origen_id"] == d["destino_id"], critica=False),
    ]


def preparar_viajes(viajes: pd.DataFrame, ids_estaciones: set) -> tuple[pd.DataFrame, list]:
    """Agrega las columnas derivadas que se validan (duración, hora y día de la semana)."""
    df = viajes.copy()
    df["duracion_seg"] = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds()
    df["hora_inicio"] = df["inicio_viaje"].dt.hour
    df["dia_semana"] = df["inicio_viaje"].dt.dayofweek
    return df, reglas_viajes(ids_estaciones)


def dataset_viajes(df: pd.DataFrame) -> Dataset:
    return Dataset(
        df, index_name="viaje_id", datetime_name="inicio_viaje",
        # mes_archivo no es variable: separa entrenamiento de prueba por construcción.
        features=["origen_id", "destino_id", "duracion_seg", "hora_inicio", "dia_semana"],
        cat_features=["origen_id", "destino_id", "dia_semana"],
    )


def conjunto_viajes(viajes: pd.DataFrame | None = None, estaciones: pd.DataFrame | None = None,
                    muestra: int = MUESTRA_VIAJES) -> ConjuntoValidacion:
    """Viajes antes de ``TEST_START`` contra viajes desde esa fecha.

    Las reglas de negocio revisan TODOS los viajes; los chequeos estadísticos de DeepChecks
    usan una muestra aleatoria (semilla del proyecto) de ``muestra`` viajes por conjunto.
    """
    viajes = pd.read_parquet(VIAJES_PARQUET) if viajes is None else viajes
    estaciones = pd.read_parquet(ESTACIONES_PARQUET) if estaciones is None else estaciones
    df, reglas = preparar_viajes(viajes, set(estaciones["id"]))
    corte = df["inicio_viaje"] < pd.Timestamp(TEST_START)
    completos = {"train": df.loc[corte], "test": df.loc[~corte]}
    muestras = {k: d.sample(n=min(muestra, len(d)), random_state=SEED).sort_values("inicio_viaje")
                for k, d in completos.items()}
    return ConjuntoValidacion(
        "viajes", "Viajes procesados por la ingesta (data/processed/viajes.parquet); reglas de "
                  f"negocio sobre todos los viajes y chequeos estadísticos sobre una muestra de "
                  f"{muestra:,} viajes por conjunto",
        dataset_viajes(muestras["train"]), dataset_viajes(muestras["test"]), reglas,
        con_indice=True, filas={k: len(d) for k, d in completos.items()}, completos=completos,
    )


def reglas_p1() -> list[ReglaNegocio]:
    from mibici.features import P1_FEATURES

    return [
        ReglaNegocio("llave estación-fecha duplicada",
                     lambda d: d.duplicated(subset=["estacion", "fecha"])),
        ReglaNegocio("viajes negativos", lambda d: d["viajes"] < 0),
        ReglaNegocio("variable de entrada nula", lambda d: d[P1_FEATURES].isna().any(axis=1)),
        ReglaNegocio("rezago o media negativa",
                     lambda d: (d[[c for c in P1_FEATURES if c.startswith(("lag_", "media_"))]]
                                < 0).any(axis=1)),
        ReglaNegocio("día de la semana fuera de 0-6", lambda d: ~d["dia_semana"].between(0, 6)),
        ReglaNegocio("indicador (fin de semana/feriado) distinto de 0/1",
                     lambda d: ~d["fin_semana"].isin([0, 1]) | ~d["feriado"].isin([0, 1])),
        ReglaNegocio("estación sin capacidad en el catálogo", lambda d: d["capacidad"] <= 0,
                     critica=False),
    ]


def dataset_p1(df: pd.DataFrame) -> Dataset:
    from mibici.features import P1_FEATURES

    return Dataset(df, label="viajes", label_type="regression", datetime_name="fecha",
                   features=["estacion"] + P1_FEATURES,
                   cat_features=["estacion", "dia_semana", "fin_semana", "feriado"])


def conjunto_p1() -> ConjuntoValidacion:
    from mibici.modeling.tareas import P1Tarea

    tarea = P1Tarea()
    tarea.preparar()
    train, test = (tarea.df[k].reset_index() for k in ("train", "test"))
    return ConjuntoValidacion(
        "p1", "P1 · tabla estación × día (TRAIN ene-sep contra TEST nov-dic)",
        dataset_p1(train), dataset_p1(test), reglas_p1(), tarea="regression",
        filas={"train": len(train), "test": len(test)}, columnas_tiempo=["dias_activa"],
    )


def reglas_p2() -> list[ReglaNegocio]:
    from mibici.features import P2_FEATURES

    return [
        ReglaNegocio("llave estación-hora duplicada",
                     lambda d: d.duplicated(subset=["estacion", "hora"])),
        ReglaNegocio("etiqueta distinta de 0/1", lambda d: ~d["riesgo"].isin([0, 1])),
        ReglaNegocio("variable de entrada nula", lambda d: d[P2_FEATURES].isna().any(axis=1)),
        ReglaNegocio("capacidad no positiva", lambda d: d["capacidad"] <= 0),
        ReglaNegocio("hora del día fuera de 0-23", lambda d: ~d["hora_dia"].between(0, 23)),
        ReglaNegocio("variable de entrada no finita",
                     lambda d: ~np.isfinite(d[P2_FEATURES].to_numpy(dtype=float)).all(axis=1)),
    ]


def dataset_p2(df: pd.DataFrame) -> Dataset:
    from mibici.features import P2_FEATURES

    return Dataset(df, label="riesgo", label_type="binary", datetime_name="hora",
                   features=P2_FEATURES, cat_features=["dia_semana", "fin_semana"])


def conjunto_p2(muestra: int = MUESTRA_P2) -> ConjuntoValidacion:
    """P2 desde la caché de ``P2Tarea`` (o la construye). Muestra aleatoria con la semilla."""
    from mibici.modeling.seguimiento import leer_manifiesto
    from mibici.modeling.tareas import P2Tarea

    tarea = P2Tarea()
    tarea.preparar(data_version=leer_manifiesto()["data_version"])
    partes = {}
    for k in ("train", "test"):
        d = tarea.df[k].reset_index()
        partes[k] = d.sample(n=min(muestra, len(d)), random_state=SEED).sort_values("hora")
        partes[k] = partes[k].drop(columns=["tipo_riesgo", "neto_futuro"])  # miran al futuro
    return ConjuntoValidacion(
        "p2", f"P2 · tabla estación × hora (TRAIN {P2_TRAIN_START[:7]}-sep contra TEST "
              f"{P2_TEST_START[:7]}-dic; muestra de {muestra:,} filas por conjunto)",
        dataset_p2(partes["train"]), dataset_p2(partes["test"]), reglas_p2(), tarea="binary",
        filas={k: len(v) for k, v in partes.items()},
    )


def reglas_p3() -> list[ReglaNegocio]:
    from mibici.features import P3_MODEL_FEATURES

    shares = [c for c in P3_MODEL_FEATURES if c.endswith("_share") and c != "share_fin_semana"]
    salida = [c for c in shares if c.startswith("salida_")]
    llegada = [c for c in shares if c.startswith("llegada_")]
    return [
        ReglaNegocio("estación duplicada", lambda d: d.index.duplicated()),
        ReglaNegocio("variable nula", lambda d: d[P3_MODEL_FEATURES].isna().any(axis=1)),
        ReglaNegocio("intensidad negativa",
                     lambda d: (d[["salidas_dia", "llegadas_dia"]] < 0).any(axis=1)),
        ReglaNegocio("proporción fuera de [0, 1]",
                     lambda d: ~d[shares + ["share_fin_semana"]].stack().between(0, 1)
                     .groupby(level=0).all()),
        ReglaNegocio("balance relativo fuera de [-1, 1]",
                     lambda d: ~d["balance_relativo"].between(-1, 1)),
        ReglaNegocio("bloques horarios de salida no suman 1",
                     lambda d: ~np.isclose(d[salida].sum(axis=1), 1)),
        ReglaNegocio("bloques horarios de llegada no suman 1",
                     lambda d: ~np.isclose(d[llegada].sum(axis=1), 1)),
    ]


def dataset_p3(df: pd.DataFrame) -> Dataset:
    from mibici.features import P3_MODEL_FEATURES

    return Dataset(df, features=P3_MODEL_FEATURES, cat_features=[])


def conjunto_p3() -> ConjuntoValidacion:
    from mibici.modeling.tareas import P3Tarea

    tarea = P3Tarea()
    tarea.preparar()
    return ConjuntoValidacion(
        "p3", "P3 · perfiles por estación (TRAIN 1 ene-19 oct contra TEST 20 oct-31 dic)",
        dataset_p3(tarea.f_train), dataset_p3(tarea.f_test), reglas_p3(), con_fecha=False,
        filas={"train": len(tarea.f_train), "test": len(tarea.f_test)},
    )


CONSTRUCTORES = {"viajes": conjunto_viajes, "p1": conjunto_p1, "p2": conjunto_p2,
                 "p3": conjunto_p3}


# =============================================================================
# Ejecución y resumen
# =============================================================================
def resumir(resultado) -> list[dict]:
    """Una fila por chequeo: estado y detalle de sus condiciones."""
    filas = []
    for r in resultado.results:
        if isinstance(r, CheckFailure):
            filas.append({"chequeo": r.header, "estado": "error",
                          "detalle": f"{type(r.exception).__name__}: {r.exception}"})
            continue
        conds = r.conditions_results
        if not conds:
            estado = "informativo"
        elif all(c.category == ConditionCategory.PASS for c in conds):
            estado = "aprobado"
        elif any(c.category in (ConditionCategory.FAIL, ConditionCategory.ERROR) for c in conds):
            estado = "no aprobado"
        else:
            estado = "advertencia"
        detalle = " | ".join(f"{c.name}: {c.details}" for c in conds)
        if not conds:  # chequeo descriptivo: se resume su valor
            detalle = _valor_breve(r.get_header(), r.value)
        filas.append({"chequeo": r.get_header(), "estado": estado, "detalle": detalle})
    return filas


def _valor_breve(encabezado: str, valor) -> str:
    """Resumen legible del valor de un chequeo sin condición."""
    try:
        if encabezado == "Class Imbalance":
            return "Proporción por clase: " + ", ".join(
                f"{k}: {float(v):.2%}" for k, v in dict(valor).items())
        if encabezado == "Conflicting Labels":
            return f"Muestras con etiquetas en conflicto: {valor['percent_of_conflicting_samples']:.2%}"
        if encabezado == "Train Test Samples Mix":
            return f"Muestras de prueba que aparecen en entrenamiento: {valor['ratio']:.2%}"
        if encabezado == "New Category Train Test":
            nuevas = list(valor.index[valor["# New Categories"] > 0])
            return f"Columnas con categorías nuevas en prueba: {nuevas or 'ninguna'}"
        if encabezado == "Outlier Sample Detection":
            return f"Se listan las {min(5, len(valor))} muestras más atípicas (ver reporte HTML)"
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    texto = json.dumps(_a_json(valor), ensure_ascii=False)
    return texto[:300] + ("…" if len(texto) > 300 else "")


def valor_de(resultado, encabezado: str):
    """``value`` del chequeo con ese encabezado (para las pruebas y el resumen)."""
    for r in resultado.results:
        if not isinstance(r, CheckFailure) and r.get_header() == encabezado:
            return r.value
    raise KeyError(encabezado)


def validar(conjunto: ConjuntoValidacion) -> dict:
    """Corre las dos suites sobre un conjunto y devuelve los resultados de DeepChecks."""
    def integridad(clave: str, dataset: Dataset):
        suite = suite_integridad(conjunto.nombre, conjunto.reglas, conjunto.tarea,
                                 conjunto.completos.get(clave))
        return suite.run(dataset)

    tt = suite_entrenamiento_prueba(conjunto.nombre, conjunto.tarea, conjunto.con_fecha,
                                    conjunto.con_indice, conjunto.columnas_tiempo)
    return {
        "integridad_train": integridad("train", conjunto.train),
        "integridad_test": integridad("test", conjunto.test),
        "train_test": tt.run(train_dataset=conjunto.train, test_dataset=conjunto.test),
    }


def _a_json(valor):
    try:
        return json.loads(json.dumps(valor, default=lambda o: o.item() if hasattr(o, "item")
                                     else str(o)))
    except (TypeError, ValueError):
        return str(valor)


def guardar(nombre: str, descripcion: str, filas: dict, resultados: dict, salida: Path) -> dict:
    """Guarda un HTML por suite y devuelve el resumen del conjunto."""
    salida.mkdir(parents=True, exist_ok=True)
    resumen = {"conjunto": nombre, "descripcion": descripcion, "filas": filas, "suites": {}}
    for clave, res in resultados.items():
        ruta = salida / f"{nombre}_{clave}.html"
        ruta.unlink(missing_ok=True)  # DeepChecks no sobrescribe: crearía "nombre (1).html"
        res.save_as_html(str(ruta), as_widget=False)
        resumen["suites"][clave] = {"reporte": ruta.name, "chequeos": resumir(res)}
    tt = resultados["train_test"]
    for encabezado, clave in (("Multivariate Drift", "drift_multivariado"),
                              ("Feature Drift", "drift_por_variable"),
                              ("Label Drift", "drift_etiqueta")):
        try:
            resumen[clave] = _a_json(valor_de(tt, encabezado))
        except KeyError:
            pass
    resumen["reglas_negocio"] = {
        k: _a_json(valor_de(resultados[k], "Reglas de negocio MiBici"))
        for k in ("integridad_train", "integridad_test")
    }
    return resumen


def escribir_resumen_md(resumenes: list[dict], salida: Path) -> Path:
    """RESUMEN.md con el estado de cada chequeo por conjunto y suite."""
    leyenda = ("Generado por `python -m mibici.validacion`. Estados: **aprobado** (todas sus "
               "condiciones se cumplen), **no aprobado** (alguna falla), **advertencia** "
               "(DeepChecks marca el hallazgo como aviso), **informativo** (chequeo sin "
               "condición, solo describe) o **error**.")
    lineas = ["# Resumen de la validación de datos con DeepChecks", "", leyenda, ""]
    for r in resumenes:
        filas = (f"Filas: entrenamiento {r['filas'].get('train', 0):,} · prueba "
                 f"{r['filas'].get('test', 0):,}")
        lineas += [f"## {r['conjunto']} — {r['descripcion']}", "", filas, ""]
        for suite, datos in r["suites"].items():
            lineas += [f"### {suite} (`{datos['reporte']}`)", "",
                       "| Chequeo | Estado | Detalle |", "|---|---|---|"]
            for c in datos["chequeos"]:
                detalle = c["detalle"].replace("|", "/").replace("\n", " ")[:400]
                lineas.append(f"| {c['chequeo']} | {c['estado']} | {detalle} |")
            lineas.append("")
    ruta = salida / "RESUMEN.md"
    ruta.write_text("\n".join(lineas), encoding="utf-8", newline="\n")
    return ruta


app = typer.Typer(add_completion=False)


@app.command()
def main(
    conjuntos: str = typer.Option(",".join(CONJUNTOS), help="Lista separada por comas"),
    carpeta: str = typer.Option(str(SALIDA_DIR), "--salida", help="Carpeta de los reportes"),
):
    """Corre DeepChecks sobre los conjuntos indicados y guarda reportes y resumen."""
    salida = Path(carpeta)
    resumenes = []
    for nombre in [c.strip() for c in conjuntos.split(",") if c.strip()]:
        t0 = time.perf_counter()
        logger.info(f"DeepChecks · preparando {nombre}")
        conjunto = CONSTRUCTORES[nombre]()
        resultados = validar(conjunto)
        resumenes.append(guardar(nombre, conjunto.descripcion, conjunto.filas, resultados,
                                 salida))
        estados = [c["estado"] for s in resumenes[-1]["suites"].values() for c in s["chequeos"]]
        logger.success(f"{nombre}: {len(estados)} chequeos en {time.perf_counter() - t0:.0f} s "
                       f"(no aprobados: {estados.count('no aprobado')}, "
                       f"errores: {estados.count('error')})")
    (salida / "resumen.json").write_text(
        json.dumps(resumenes, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    escribir_resumen_md(resumenes, salida)
    logger.success(f"Reportes en {salida}")


if __name__ == "__main__":
    app()
