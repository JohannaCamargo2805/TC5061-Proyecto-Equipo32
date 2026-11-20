"""Motor de inferencia en NumPy equivalente al Pipeline de P1 (entrega 6, optimización FinOps).

La prueba de carga mostró que casi todo el CPU de una predicción es sobrecarga de la llamada
(``pyfunc.predict`` de MLflow y el ``ColumnTransformer`` de scikit-learn sobre un DataFrame: ~6 ms
tanto para 1 fila como para 100), no el cálculo del modelo. El modelo de P1 es una regresión de
Poisson sobre una transformación lineal sencilla, así que la misma predicción se puede calcular
con unas cuantas operaciones de NumPy:

    exp([one-hot(dia_semana), (log1p(conteos) - μ) / σ, (numéricas - μ) / σ] · β + β0)

``compilar`` lee los parámetros **del modelo registrado** (categorías, medias, escalas,
coeficientes) y ``verificar`` compara sus predicciones con las del Pipeline original sobre 2,000
entradas sintéticas: si la diferencia relativa supera 1e-9, o el Pipeline no tiene exactamente la
estructura esperada, la API sigue usando ``pyfunc`` (lo dice en ``GET /modelo``). Se activa con
``ACELERAR_INFERENCIA=1``; por defecto la API usa el camino de MLflow.
"""

from collections.abc import Callable

import numpy as np

from api.esquemas import VARIABLES_P1

Motor = Callable[[np.ndarray], np.ndarray]
COLUMNAS = list(VARIABLES_P1)
TOLERANCIA = 1e-9


def _pipeline(modelo):
    """El Pipeline de scikit-learn dentro del PyFuncModel de MLflow (o el propio Pipeline)."""
    impl = getattr(modelo, "_model_impl", None)
    return getattr(impl, "sklearn_model", modelo)


def compilar(modelo) -> Motor | None:
    """Devuelve la función en NumPy, o None si el modelo no tiene la estructura conocida."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    p = _pipeline(modelo)
    try:
        prep, glm = p.named_steps["prep"], p.named_steps["modelo"]
        if not (
            isinstance(p, Pipeline)
            and isinstance(prep, ColumnTransformer)
            and isinstance(glm, PoissonRegressor)
        ):
            return None
        t = {nombre: (obj, cols) for nombre, obj, cols in prep.transformers_}
        if set(t) - {"remainder"} != {"dia", "conteos", "num"}:
            return None
        if "remainder" in t and t["remainder"][0] != "drop":
            return None
        ohe, cols_dia = t["dia"]
        conteos, cols_conteos = t["conteos"]
        num, cols_num = t["num"]
        log, esc_c = conteos.named_steps["log"], conteos.named_steps["esc"]
        if not (
            isinstance(ohe, OneHotEncoder)
            and ohe.handle_unknown == "ignore"
            and len(cols_dia) == 1
            and isinstance(log, FunctionTransformer)
            and log.func is np.log1p
            and isinstance(esc_c, StandardScaler)
            and isinstance(num, StandardScaler)
            and ohe.drop is None
        ):
            return None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None

    idx = {c: i for i, c in enumerate(COLUMNAS)}
    i_dia = idx[cols_dia[0]]
    i_c = [idx[c] for c in cols_conteos]
    i_n = [idx[c] for c in cols_num]
    categorias = np.asarray(ohe.categories_[0], dtype="float64")
    mu_c, sd_c = np.asarray(esc_c.mean_), np.asarray(esc_c.scale_)
    mu_n, sd_n = np.asarray(num.mean_), np.asarray(num.scale_)
    beta, beta0 = np.asarray(glm.coef_), float(glm.intercept_)
    if beta.shape[0] != categorias.size + len(i_c) + len(i_n):
        return None

    def predecir(x: np.ndarray) -> np.ndarray:
        dia = (x[:, [i_dia]] == categorias).astype("float64")
        c = (np.log1p(x[:, i_c]) - mu_c) / sd_c
        n = (x[:, i_n] - mu_n) / sd_n
        return np.exp(np.hstack([dia, c, n]) @ beta + beta0)

    return predecir


def entradas_sinteticas(n: int = 2000, semilla: int = 20261120) -> np.ndarray:
    """Filas plausibles (y algunas extremas) con las 12 variables, en el orden de VARIABLES_P1."""
    rng = np.random.default_rng(semilla)
    dia = rng.integers(0, 7, n)
    conteos = rng.lognormal(3.2, 1.0, (n, 7)).round()
    conteos[:20] = 0  # estaciones sin viajes
    datos = {
        "dia_semana": dia,
        "fin_semana": (dia >= 5).astype(int),
        "feriado": rng.integers(0, 2, n),
        "dias_activa": rng.integers(0, 4000, n),
        "capacidad": rng.integers(5, 60, n),
    }
    for j, c in enumerate(
        ["lag_1", "lag_2", "lag_7", "lag_14", "media_7d", "media_28d", "media_mismo_dia_4sem"]
    ):
        datos[c] = conteos[:, j]
    return np.column_stack([np.asarray(datos[c], dtype="float64") for c in COLUMNAS])


def verificar(motor: Motor, modelo) -> float:
    """Diferencia relativa máxima contra el Pipeline original (debe ser <= TOLERANCIA)."""
    import pandas as pd

    x = entradas_sinteticas()
    referencia = np.asarray(_pipeline(modelo).predict(pd.DataFrame(x, columns=COLUMNAS)))
    rapido = motor(x)
    return float(np.max(np.abs(rapido - referencia) / np.maximum(np.abs(referencia), 1e-12)))
