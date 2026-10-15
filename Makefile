#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo32
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

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



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Descarga, normaliza, guarda en Parquet y versiona los datos de MiBici
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset ingest

## Verifica que los datos locales coinciden con references/data_manifest.json
.PHONY: verify-data
verify-data:
	$(PYTHON_INTERPRETER) -m mibici.dataset verify

## Construye las tablas de variables de P1, P2 y P3 en data/processed
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features build

## Ejecuta el experimento base (baselines + candidatos) y lo registra en MLflow
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train

## Pipeline completo: ingesta -> variables -> experimento base
.PHONY: pipeline
pipeline: data features train

## Abre la interfaz de MLflow en http://127.0.0.1:5000
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
