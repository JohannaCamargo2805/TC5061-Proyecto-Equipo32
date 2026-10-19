#!/usr/bin/env bash
# Protocolo completo de la entrega 2 (MLflow Project) en Linux/macOS.
# Requiere el servidor de tracking activo (make server o scripts/servidor_mlflow.ps1 en Windows)
# y el entorno del proyecto activado (o usar --env-manager uv).
set -uo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_EXPERIMENT_NAME=mibici-equipo32 MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore
ENV_MANAGER="${ENV_MANAGER:-local}"
run() { echo ">> mlflow run . -e $*"; local ep=$1; shift; local ps=(); for p in "$@"; do ps+=(-P "$p"); done
        mlflow run . -e "$ep" --env-manager "$ENV_MANAGER" "${ps[@]}"; }

python -m pytest -q tests
run ingest "raw_source=${MIBICI_RAW_SOURCE:-}"
python scripts/etiquetar_corridas_propuesta.py
# P1
run train question=p1 model=naive stage_of_work=baseline
run train question=p1 model=poisson
run train question=p1 model=hgb
run train question=p1 model=rf n_estimators=200
run search question=p1
run train question=p1 from_search=latest
# P2
run train question=p2 model=logreg stage_of_work=baseline
run train question=p2 model=hgb
run train question=p2 model=rf
run search question=p2
run train question=p2 from_search=latest
# P3
run train question=p3 stage_of_work=baseline
run train question=p3 model=kmeans
run train question=p3 model=agglomerative
run train question=p3 model=gmm
run search question=p3
# Model Registry y verificación
for q in p1 p2 p3; do run register question=$q; done
run verify question=all alias=champion
