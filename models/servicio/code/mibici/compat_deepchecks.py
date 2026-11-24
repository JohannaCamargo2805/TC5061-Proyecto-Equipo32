"""Capa de compatibilidad para usar DeepChecks 0.19 con las versiones fijadas del proyecto.

DeepChecks 0.19.1 (la versión más reciente en PyPI) todavía usa APIs que ya no existen en
las versiones que fija ``requirements.txt`` (NumPy 2.5, pandas 3.0, scikit-learn 1.9). Sin
estos ajustes, ``import deepchecks.tabular`` falla o varios chequeos terminan con error.

Los ajustes son mínimos, explícitos y solo afectan a DeepChecks:

1. ``np.Inf``, ``np.NINF`` y ``np.NaN`` (eliminados en NumPy 2.0) se definen como alias de
   ``np.inf``, ``-np.inf`` y ``np.nan`` si no existen.
2. El scorer ``"max_error"`` (renombrado ``"neg_max_error"`` en scikit-learn) se registra
   como alias para que DeepChecks pueda construir su tabla de métricas por defecto. Este
   proyecto no usa los chequeos de desempeño de modelos que lo consumen.
3. ``DataFrame.applymap`` (eliminado en pandas 3.0) se define como alias de ``DataFrame.map``.
4. ``np.quantile``/``np.percentile`` ya no aceptan ``interpolation=`` (ahora es ``method=``):
   solo dentro de los dos módulos de DeepChecks que lo usan, ``np`` se sustituye por un
   envoltorio que traduce el argumento.
5. ``MultivariateDrift`` calcula la importancia por permutación del clasificador de dominio
   con un envoltorio de modelo que scikit-learn 1.6+ ya no acepta (``__sklearn_tags__``).
   Ese cálculo es solo explicativo: si falla, el chequeo sigue reportando el AUC del
   clasificador de dominio (la medida de drift) sin el desglose por variable.
6. La condición de ``FeatureFeatureCorrelation`` usa ``DataFrame.stack()``, que en pandas 3.0
   ya no descarta los NaN: sin el ajuste reporta TODOS los pares como correlacionados. Se
   reemplaza por la misma lógica con ``dropna()``.

7. ``pkg_resources`` desapareció en setuptools 81. Fijar ``setuptools<81`` reintroduce
   vulnerabilidades conocidas (CVE-2026-59890 y GHSA-xmj3-9gf7-h52w, detectadas con
   pip-audit), así que, si no existe, se registra un sustituto mínimo con lo único que
   DeepChecks usa: ``parse_version`` y ``working_set``.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import sys
import types
import warnings

import numpy as np
import pandas as pd

_APLICADA = False


class _NumpyConInterpolation:
    """``numpy`` que acepta el argumento antiguo ``interpolation=`` en quantile/percentile."""

    def __getattr__(self, nombre):
        return getattr(np, nombre)

    @staticmethod
    def _traducir(funcion):
        @functools.wraps(funcion)
        def envoltura(*args, interpolation=None, **kwargs):
            if interpolation is not None:
                kwargs.setdefault("method", interpolation)
            return funcion(*args, **kwargs)

        return envoltura

    @property
    def quantile(self):
        return self._traducir(np.quantile)

    @property
    def percentile(self):
        return self._traducir(np.percentile)


class _Distribucion:
    """Lo que DeepChecks lee de cada elemento de ``pkg_resources.working_set``."""

    def __init__(self, nombre: str):
        self.key = nombre.lower()
        self.project_name = nombre


def _pkg_resources_minimo() -> types.ModuleType:
    from packaging.version import parse

    modulo = types.ModuleType("pkg_resources")
    modulo.parse_version = parse
    modulo.working_set = [_Distribucion(d.metadata["Name"])
                          for d in importlib.metadata.distributions() if d.metadata["Name"]]
    return modulo


def aplicar() -> None:
    """Aplica los ajustes una sola vez (idempotente). Llamar antes de importar DeepChecks."""
    global _APLICADA
    if _APLICADA:
        return
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

    # 7. pkg_resources (setuptools >= 81 ya no lo incluye).
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        sys.modules["pkg_resources"] = _pkg_resources_minimo()

    # 1. Constantes eliminadas en NumPy 2.0.
    for nombre, valor in {"Inf": np.inf, "NINF": -np.inf, "NaN": np.nan}.items():
        if nombre not in np.__dict__:
            setattr(np, nombre, valor)

    # 2. Scorer renombrado en scikit-learn.
    from sklearn.metrics import _scorer

    if "max_error" not in _scorer._SCORERS and "neg_max_error" in _scorer._SCORERS:
        _scorer._SCORERS["max_error"] = _scorer._SCORERS["neg_max_error"]

    # 3. Método eliminado en pandas 3.0.
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map

    # 4. interpolation= -> method= solo en los módulos de DeepChecks que lo usan.
    for modulo in (
        "deepchecks.tabular.checks.data_integrity.outlier_sample_detection",
        "deepchecks.tabular.checks.data_integrity.string_length_out_of_bounds",
    ):
        importlib.import_module(modulo).np = _NumpyConInterpolation()

    # 5. Importancia del clasificador de dominio en MultivariateDrift.
    mdu = importlib.import_module("deepchecks.core.check_utils.multivariate_drift_utils")
    original = mdu.calculate_feature_importance_or_none

    @functools.wraps(original)
    def importancia_tolerante(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except AttributeError as error:  # envoltorio sin __sklearn_tags__ (scikit-learn >= 1.6)
            if "__sklearn_tags__" not in str(error):
                raise
            return None, None

    mdu.calculate_feature_importance_or_none = importancia_tolerante

    # 6. Condición de FeatureFeatureCorrelation con pandas 3 (stack conserva NaN).
    from deepchecks.core import ConditionCategory, ConditionResult
    from deepchecks.tabular.checks import FeatureFeatureCorrelation

    def condicion_pares(self, threshold: float = 0.9, n_pairs: int = 0):
        """Igual que la original de DeepChecks, pero descartando los NaN de ``stack()``."""

        def condition(result):
            arriba = result[result > threshold].stack().dropna().index.to_list()
            pares = [(i, j) for (i, j) in arriba if i < j]  # sin diagonal ni duplicados
            if len(pares) > n_pairs:
                return ConditionResult(ConditionCategory.FAIL,
                                       f"Correlation is greater than {threshold} for pairs {pares}")
            return ConditionResult(ConditionCategory.PASS,
                                   f"All correlations are less than {threshold} except pairs {pares}")

        return self.add_condition(f"Not more than {n_pairs} pairs are correlated above {threshold}",
                                  condition)

    FeatureFeatureCorrelation.add_condition_max_number_of_pairs_above_threshold = condicion_pares
    _APLICADA = True
