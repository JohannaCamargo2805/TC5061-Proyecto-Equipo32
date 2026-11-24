"""Pruebas del proyecto final: la versión en producción es la misma en todas partes.

La versión de ``models/servicio/registro.json`` (la que exportó el Model Registry con
``@champion``) debe coincidir con la etiqueta de la imagen en Docker Compose y en el pipeline, con
el ``MODELO_VERSION`` del Containerfile y con la Model Card; la predicción de referencia que
comprueba el pipeline debe ser la del modelo empaquetado; y la petición de la demo debe pasar la
validación de la API.
"""

import json
import re

import pytest
import yaml

from api.esquemas import EntradaP1
from mibici.config import PROJ_ROOT, SERVICIO_MODELO_DIR

REGISTRO = SERVICIO_MODELO_DIR / "registro.json"
HAY_MODELO = (SERVICIO_MODELO_DIR / "MLmodel").is_file() and REGISTRO.is_file()
CI = yaml.safe_load((PROJ_ROOT / ".github" / "workflows" / "ci-cd.yml").read_text(encoding="utf-8"))
COMPOSE = yaml.safe_load((PROJ_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
pytestmark = pytest.mark.skipif(not HAY_MODELO, reason="falta models/servicio")


@pytest.fixture(scope="module")
def registro() -> dict:
    return json.loads(REGISTRO.read_text(encoding="utf-8"))


def test_la_copia_es_champion_en_produccion(registro):
    assert "champion" in registro["aliases"]
    assert registro["etiquetas_version"].get("estado_produccion") == "produccion"
    assert registro["rango_entrenamiento"] and len(registro["rango_entrenamiento"]) == 9


def test_misma_version_en_imagen_compose_y_pipeline(registro):
    v = registro["version"]
    etiqueta = f"mibici-p1-api:modelo-v{v}"
    for servicio in ("api", "worker"):
        assert COMPOSE["services"][servicio]["image"].endswith(f"{etiqueta}}}"), servicio
    assert COMPOSE["services"]["api"]["build"]["args"]["MODELO_VERSION"] == str(v)
    for archivo in ("Containerfile", "Dockerfile"):
        assert f"ARG MODELO_VERSION={v}\n" in (PROJ_ROOT / archivo).read_text(encoding="utf-8")
    matriz = CI["jobs"]["imagenes"]["strategy"]["matrix"]["imagen"]
    api = next(i for i in matriz if i["nombre"] == "mibici-p1-api")
    assert api["local"] == etiqueta and api["version"] == f"modelo-v{v}"
    texto_ci = (PROJ_ROOT / ".github" / "workflows" / "ci-cd.yml").read_text(encoding="utf-8")
    assert set(re.findall(r"mibici-p1-api:modelo-v\d+", texto_ci)) == {etiqueta}


def test_prediccion_de_referencia_del_pipeline(registro):
    """El valor que el CI exige a la API en contenedor es lo que predice la copia empaquetada."""
    from api.config import Ajustes
    from api.modelo import cargar_modelo, predicciones

    servicio = cargar_modelo(Ajustes(modelo_uri=str(SERVICIO_MODELO_DIR), tracking_uri="",
                                     verificar_integridad=True))
    entrada = EntradaP1(**json.loads((PROJ_ROOT / "docs" / "ejemplos" / "peticion_predict.json")
                                    .read_text(encoding="utf-8")))
    valor = predicciones(servicio, [entrada])[0].viajes_esperados
    assert valor == pytest.approx(float(CI["env"]["PREDICCION_REFERENCIA"]), abs=1e-3)


def test_peticion_de_la_demo_es_valida():
    for nombre in ("peticion_demo.json", "peticion_extrapolacion.json"):
        EntradaP1(**json.loads((PROJ_ROOT / "docs" / "ejemplos" / nombre).read_text(encoding="utf-8")))
    entrada = EntradaP1(**json.loads((PROJ_ROOT / "docs" / "ejemplos" / "peticion_demo.json")
                                    .read_text(encoding="utf-8")))
    assert entrada.estacion_id == 286 and str(entrada.fecha) == "2026-09-30"


def test_model_card_y_documentos(registro):
    tarjeta = (PROJ_ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert f"**v{registro['version']}** (`@champion`)" in tarjeta
    for seccion in ("Uso previsto", "Factores", "Métricas", "Datos de evaluación", "Datos de entrenamiento",
                    "Análisis cuantitativo", "Consideraciones éticas", "Limitaciones"):
        assert re.search(rf"^## \d+\. {seccion}", tarjeta, re.MULTILINE), seccion
    assert registro["run_id"] in (PROJ_ROOT / "docs" / "model_registry.md").read_text(encoding="utf-8")
