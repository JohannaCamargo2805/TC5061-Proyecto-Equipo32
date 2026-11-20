"""¿En qué se va el CPU de una predicción? Perfil de la inferencia de P1 (entrega 6, FinOps).

Corre dentro de la imagen de la API, con el mismo límite de CPU que el servicio::

    docker compose run --rm --no-deps -v ./scripts:/scripts:ro -v ./carga:/carga:ro \
        --entrypoint python api /scripts/perfil_inferencia.py > reports/finops/perfil_inferencia.json

Mide, con 300 repeticiones y entradas realistas (``carga/peticiones.py``), el tiempo por llamada de
cada capa: validación con Pydantic, armado del DataFrame, ``pyfunc.predict`` de MLflow (lo que usa
la API), el ``Pipeline`` de scikit-learn directo y una **versión en NumPy** de la misma
transformación (one-hot, log1p, estandarización y exp del modelo Poisson). Comprueba que la versión
en NumPy da las mismas predicciones (diferencia máxima < 1e-9) antes de reportar su tiempo, para
cuantificar cuánto costo por inferencia es sobrecarga de la llamada y no cálculo del modelo.
"""

import json
import os
import platform
import sys
import time

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.getenv("CARGA_DIR", "/carga"))
# El paquete api: /app dentro de la imagen; la raíz del repositorio al correrlo localmente.
sys.path.insert(0, os.getenv("APP_DIR", "/app"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from peticiones import generar  # noqa: E402

from api.esquemas import VARIABLES_P1, EntradaP1  # noqa: E402

MODELO = os.getenv("MODELO_DIR", "/app/models/servicio")
COLUMNAS = list(VARIABLES_P1)


def cronometrar(funcion, repeticiones: int) -> float:
    """Milisegundos por llamada (mediana de 5 bloques)."""
    funcion()
    bloques = []
    for _ in range(5):
        inicio = time.perf_counter()
        for _ in range(repeticiones // 5):
            funcion()
        bloques.append((time.perf_counter() - inicio) / (repeticiones // 5) * 1000)
    return round(float(np.median(bloques)), 4)


def numpy_equivalente(pipeline):
    """La misma transformación y predicción del Pipeline, como operaciones de NumPy."""
    ct = pipeline.named_steps["prep"]
    poisson = pipeline.named_steps["modelo"]
    ohe = ct.named_transformers_["dia"]
    conteos = ct.named_transformers_["conteos"]
    num = ct.named_transformers_["num"]
    columnas = {nombre: cols for nombre, _, cols in ct.transformers_ if nombre != "remainder"}
    idx = {c: i for i, c in enumerate(COLUMNAS)}
    i_dia = idx[columnas["dia"][0]]
    i_conteos = [idx[c] for c in columnas["conteos"]]
    i_num = [idx[c] for c in columnas["num"]]
    categorias = ohe.categories_[0].astype("float64")
    esc_c = conteos.named_steps["esc"]
    media_c, escala_c = esc_c.mean_, esc_c.scale_
    media_n, escala_n = num.mean_, num.scale_
    coef, intercepto = poisson.coef_, poisson.intercept_

    def predecir(x: np.ndarray) -> np.ndarray:
        dia = (x[:, [i_dia]] == categorias).astype("float64")
        c = (np.log1p(x[:, i_conteos]) - media_c) / escala_c
        n = (x[:, i_num] - media_n) / escala_n
        return np.exp(np.hstack([dia, c, n]) @ coef + intercepto)

    return predecir


def main() -> int:
    modelo = mlflow.pyfunc.load_model(MODELO, suppress_warnings=True)
    pipeline = modelo._model_impl.sklearn_model
    rapido = numpy_equivalente(pipeline)
    entradas = generar(100)
    validadas = [EntradaP1.model_validate(e) for e in entradas]
    filas = [v.fila() for v in validadas]
    x1 = pd.DataFrame(filas[:1], columns=COLUMNAS).astype("float64")
    x100 = pd.DataFrame(filas, columns=COLUMNAS).astype("float64")
    a1, a100 = x1.to_numpy(), x100.to_numpy()
    diferencia = float(np.max(np.abs(rapido(a100) - pipeline.predict(x100))))
    if diferencia > 1e-9:
        raise SystemExit(f"la versión en NumPy no coincide: {diferencia}")
    n = 300
    resultado = {
        "entorno": {"python": platform.python_version(), "cpus_visibles": os.cpu_count(),
                    "numpy": np.__version__, "pandas": pd.__version__, "mlflow": mlflow.__version__},
        "repeticiones": n,
        "diferencia_maxima_numpy_vs_pipeline": diferencia,
        "ms_por_llamada": {
            "validacion_pydantic_1": cronometrar(lambda: EntradaP1.model_validate(entradas[0]), n),
            "dataframe_1": cronometrar(
                lambda: pd.DataFrame(filas[:1], columns=COLUMNAS).astype("float64"), n),
            "pyfunc_mlflow_1": cronometrar(lambda: modelo.predict(x1), n),
            "pyfunc_mlflow_100": cronometrar(lambda: modelo.predict(x100), n),
            "pipeline_sklearn_1": cronometrar(lambda: pipeline.predict(x1), n),
            "pipeline_sklearn_100": cronometrar(lambda: pipeline.predict(x100), n),
            "numpy_1": cronometrar(lambda: rapido(a1), n),
            "numpy_100": cronometrar(lambda: rapido(a100), n),
        },
    }
    m = resultado["ms_por_llamada"]
    resultado["ms_por_inferencia"] = {
        "pyfunc_individual": m["pyfunc_mlflow_1"],
        "pyfunc_en_lote_100": round(m["pyfunc_mlflow_100"] / 100, 5),
        "numpy_individual": m["numpy_1"],
        "numpy_en_lote_100": round(m["numpy_100"] / 100, 6),
    }
    print(json.dumps(resultado, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
