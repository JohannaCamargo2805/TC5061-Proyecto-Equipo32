#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-proyecto-equipo32
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
TRACKING_URI = http://127.0.0.1:5000
EXPERIMENT = mibici-equipo32
ENV_MANAGER = local
MOTOR ?= podman
# La etiqueta de la imagen lleva la versión del registry que está en models/servicio/registro.json
MODELO_VERSION = $(shell $(PYTHON_INTERPRETER) -c "import json; print(json.load(open('models/servicio/registro.json'))['version'])")
IMAGEN = mibici-p1-api:modelo-v$(MODELO_VERSION)
CONTENEDOR = mibici-p1-api

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Servidor de tracking con SQLite en http://127.0.0.1:5000 (requerido por el Model Registry)
.PHONY: server
server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

## Ingesta reproducible y versionado de datos (mlflow run . -e ingest)
.PHONY: data
data:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow run . -e ingest --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENT)

## P1: entrena el modelo por defecto de config.py (mlflow run . -e train -P question=p1)
.PHONY: p1
p1:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow run . -e train -P question=p1 --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENT)

## P2: entrena el modelo por defecto de config.py
.PHONY: p2
p2:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow run . -e train -P question=p2 --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENT)

## P3: entrena el modelo por defecto de config.py
.PHONY: p3
p3:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow run . -e train -P question=p3 --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENT)

## Protocolo completo de la entrega 2: baselines, candidatas, búsquedas, registry y verificación
.PHONY: experimentos
experimentos:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) ENV_MANAGER=$(ENV_MANAGER) bash scripts/ejecutar_entrega2.sh

## Carga los modelos en producción por nombre y alias (models:/<nombre>@champion)
.PHONY: verify
verify:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow run . -e verify --env-manager $(ENV_MANAGER) --experiment-name $(EXPERIMENT)

## Todas las pruebas: unitarias, reproducibilidad, DeepChecks y seguridad (un solo comando)
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Solo las pruebas que no necesitan los datos reales de data/processed
.PHONY: test-rapido
test-rapido:
	$(PYTHON_INTERPRETER) -m pytest -m "not datos"

## Validación de datos con DeepChecks: reportes HTML y RESUMEN.md en reports/deepchecks
.PHONY: deepchecks
deepchecks:
	$(PYTHON_INTERPRETER) -m mibici.validacion

## Chequeos de seguridad: bandit, pip-audit y secretos; reportes en reports/seguridad
.PHONY: seguridad
seguridad:
	$(PYTHON_INTERPRETER) scripts/chequeos_seguridad.py

## Pruebas + DeepChecks + seguridad (en Windows: scripts\ejecutar_calidad.ps1)
.PHONY: calidad
calidad: test deepchecks seguridad

## Entrega 4: recupera del Model Registry la versión @champion de P1 a models/servicio
.PHONY: modelo-servicio
modelo-servicio:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PYTHON_INTERPRETER) scripts/exportar_modelo_registry.py --alias champion

## Entrega 4: API en local sin contenedor (http://127.0.0.1:8000/docs)
.PHONY: api
api:
	$(PYTHON_INTERPRETER) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

## Entrega 4: construye la imagen de la API (MOTOR=podman por defecto; MOTOR=docker también sirve)
.PHONY: imagen
imagen:
	$(MOTOR) build -f Containerfile --build-arg MODELO_VERSION=$(MODELO_VERSION) -t $(IMAGEN) .

## Entrega 4: levanta el contenedor en el puerto 8000
.PHONY: contenedor
contenedor:
	-$(MOTOR) rm -f $(CONTENEDOR)
	$(MOTOR) run -d --name $(CONTENEDOR) -p 8000:8000 $(IMAGEN)

## Entrega 4: petición de prueba al contenedor (espera a /health y envía docs/ejemplos/peticion_predict.json)
.PHONY: probar-api
probar-api:
	curl -s --retry 60 --retry-delay 1 --retry-all-errors http://localhost:8000/health && echo
	curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
		--data-binary @docs/ejemplos/peticion_predict.json && echo

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
