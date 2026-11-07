"""Carga del modelo P1 y predicción.

Dos orígenes, según ``MODELO_URI`` (ver ``config.py``):

* **Empaquetado** (por defecto): la carpeta ``models/servicio`` que
  ``scripts/exportar_modelo_registry.py`` descargó del Model Registry por nombre y
  versión. Se verifica el SHA-256 de cada archivo contra ``registro.json`` antes de
  cargarla, así la imagen sirve exactamente la versión registrada.
* **Registry en vivo**: una URI ``models:/<nombre>/<versión>`` o
  ``models:/<nombre>@<alias>`` que se resuelve contra ``MLFLOW_TRACKING_URI`` al
  arrancar. Un alias se resuelve una sola vez: el proceso sirve una versión fija.

En ambos casos se comprueba que la firma del modelo tenga exactamente las 12
variables de ``esquemas.VARIABLES_P1``; si no, la API no arranca.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from api.config import Ajustes
from api.esquemas import DESCRIPCIONES, VARIABLES_P1, InfoModelo, ModeloResumen, Variable

ARCHIVO_REGISTRO = "registro.json"
SALIDA = "Viajes que se espera inicien en la estación el día D (conteo, regresión de Poisson)."
_URI_REGISTRY = re.compile(r"^models:/(?P<nombre>[^/@]+)(?:/(?P<version>\d+)|@(?P<alias>[\w-]+))$")


class ErrorModelo(RuntimeError):
    """El modelo no se pudo cargar o devolvió algo inválido."""


@dataclass
class ModeloServicio:
    modelo: object  # mlflow.pyfunc.PyFuncModel
    info: InfoModelo

    @property
    def resumen(self) -> ModeloResumen:
        i = self.info
        return ModeloResumen(nombre=i.nombre, version=i.version, alias=i.alias, run_id=i.run_id)

    def predecir(self, filas: list[dict[str, float]]) -> np.ndarray:
        """Predice en una sola llamada; devuelve conteos finitos y no negativos."""
        x = pd.DataFrame(filas, columns=list(VARIABLES_P1)).astype("float64")
        pred = np.asarray(self.modelo.predict(x), dtype="float64").reshape(-1)
        if pred.shape[0] != len(filas) or not np.all(np.isfinite(pred)):
            raise ErrorModelo(
                "el modelo devolvió una predicción no válida (NaN, infinito o tamaño distinto)"
            )
        return np.clip(pred, 0, None)  # igual que en la evaluación: no hay viajes negativos


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def metricas_de_corrida(metricas: dict[str, float]) -> dict[str, float]:
    """Validación y prueba (justifican la versión) más costo de inferencia y tamaño del modelo."""
    return {
        k: round(float(v), 6)
        for k, v in sorted(metricas.items())
        if k.startswith(("val_", "test_", "costo_inferencia", "tamano_modelo"))
        and not k.endswith("prevalencia")
    }


def _variables() -> list[Variable]:
    return [Variable(nombre=v, tipo="double", descripcion=DESCRIPCIONES[v]) for v in VARIABLES_P1]


def _verificar_firma(modelo) -> None:
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        raise ErrorModelo(
            "el modelo no tiene firma de entrada; no se puede validar el contrato de la API"
        )
    nombres = tuple(esquema.input_names())
    if nombres != VARIABLES_P1:
        raise ErrorModelo(
            f"la firma del modelo {list(nombres)} no coincide con las variables de la API "
            f"{list(VARIABLES_P1)}"
        )


def _verificar_integridad(carpeta: Path, registro: dict) -> None:
    esperados = registro.get("sha256") or {}
    if "model.skops" not in esperados and "MLmodel" not in esperados:
        raise ErrorModelo(f"{ARCHIVO_REGISTRO} no trae las huellas SHA-256 del modelo")
    for relativa, huella in esperados.items():
        ruta = carpeta / relativa
        if not ruta.is_file():
            raise ErrorModelo(f"falta el archivo {relativa} del modelo empaquetado")
        if sha256(ruta) != huella:
            raise ErrorModelo(
                f"{relativa} no coincide con el SHA-256 de {ARCHIVO_REGISTRO}: "
                "el modelo empaquetado no es el que se descargó del registry"
            )


def _desde_carpeta(carpeta: Path, verificar: bool) -> ModeloServicio:
    if not (carpeta / "MLmodel").is_file():
        raise ErrorModelo(
            f"no hay un modelo de MLflow en {carpeta}. Descárgalo del registry con "
            "'python scripts/exportar_modelo_registry.py' o usa MODELO_URI=models:/..."
        )
    ruta_registro = carpeta / ARCHIVO_REGISTRO
    if not ruta_registro.is_file():
        raise ErrorModelo(f"falta {ruta_registro}: sin él no se sabe qué versión del registry es")
    registro = json.loads(ruta_registro.read_text(encoding="utf-8"))
    if verificar:
        _verificar_integridad(carpeta, registro)
    # suppress_warnings: el aviso de MLflow compara TODO el entorno del entrenamiento (pytest, rich,
    # psutil...). Las versiones que sí importan las fija requirements-api.txt y las comprueba
    # tests/test_contenedor.py contra models/servicio/requirements.txt.
    modelo = mlflow.pyfunc.load_model(str(carpeta), suppress_warnings=True)
    _verificar_firma(modelo)
    etiquetas = registro.get("etiquetas_version", {})
    info = InfoModelo(
        nombre=registro["modelo"],
        version=int(registro["version"]),
        alias=list(registro.get("aliases", [])),
        run_id=registro["run_id"],
        model_id=registro.get("model_id"),
        origen="empaquetado",
        uri_registry=registro["uri_registry"],
        git_commit=etiquetas.get("git_commit"),
        data_version=etiquetas.get("data_version"),
        tipo_modelo=etiquetas.get("model"),
        metricas=registro.get("metricas", {}),
        variables=_variables(),
        salida=SALIDA,
        recuperado=registro.get("recuperado"),
        integridad_verificada=verificar,
    )
    return ModeloServicio(modelo=modelo, info=info)


def _desde_registry(uri: str, tracking_uri: str) -> ModeloServicio:
    partes = _URI_REGISTRY.match(uri)
    if not partes:
        raise ErrorModelo(
            f"MODELO_URI={uri!r} no es válida: usa models:/<nombre>/<versión> "
            "o models:/<nombre>@<alias>"
        )
    nombre, version, alias = partes["nombre"], partes["version"], partes["alias"]
    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient()
    mv = (
        cliente.get_model_version(nombre, version)
        if version
        else cliente.get_model_version_by_alias(nombre, alias)
    )
    # Se carga por versión: si el alias se mueve después, este proceso sigue sirviendo la misma.
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{mv.version}", suppress_warnings=True)
    _verificar_firma(modelo)
    etiquetas = dict(mv.tags or {})
    info = InfoModelo(
        nombre=nombre,
        version=int(mv.version),
        alias=sorted(getattr(mv, "aliases", None) or ([alias] if alias else [])),
        run_id=mv.run_id,
        model_id=modelo.metadata.model_id,
        origen="registry",
        uri_registry=f"models:/{nombre}/{mv.version}",
        git_commit=etiquetas.get("git_commit"),
        data_version=etiquetas.get("data_version"),
        tipo_modelo=etiquetas.get("model"),
        metricas=metricas_de_corrida(cliente.get_run(mv.run_id).data.metrics),
        variables=_variables(),
        salida=SALIDA,
        recuperado=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        integridad_verificada=False,
    )
    return ModeloServicio(modelo=modelo, info=info)


def cargar_modelo(ajustes: Ajustes) -> ModeloServicio:
    if ajustes.desde_registry:
        return _desde_registry(ajustes.modelo_uri, ajustes.tracking_uri)
    return _desde_carpeta(Path(ajustes.modelo_uri), ajustes.verificar_integridad)
