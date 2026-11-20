# Bitácora de la entrega 2: MLflow Project, experimentación y Model Registry

Equipo 32 · TC5061 Operaciones de aprendizaje automático · Proyecto integrador MiBici

## Ejecución oficial

- **Dónde:** computadora de Leydy Johanna Camargo Medina (Windows 11 Pro), con su usuario y su cuenta de
  GitHub. Servidor de tracking local `mlflow server --backend-store-uri sqlite:///mlflow.db` en
  `http://127.0.0.1:5000`.
- **Cuándo:** 2026-10-19 14:36:05 a 2026-10-19 15:35:51 (hora del centro de México).
- **Entorno:** Python 3.14.8, pandas 3.0.6, numpy 2.5.4,
  scikit-learn 1.9.1, mlflow 3.17.1 (mismas versiones que `python_env.yaml`).
- **Código:** rama `feat/mlflow-project-registry`, commit `e6c1a60`. 21 corridas lanzadas con
  `mlflow run` (ingesta, 16 de entrenamiento y búsqueda, 3 de registro y 1 de verificación) y 33 corridas hijas
  de búsquedas; corridas con cambios sin commit: 0.
- **Datos:** `data_version = 7aa5e436d887` (12 meses de 2025 + nomenclatura), verificada con el SHA-256 de
  cada archivo en `data/manifest.json`.
- **Protocolo:** `scripts/ejecutar_entrega2.ps1` (pruebas, ingesta, P1, P2, P3, búsquedas, registro y verificación).

| Comando | Resultado | Duración (s) |
|---|---|---|
| `mlflow run . -e ingest --env-manager local -P raw_source=config` | OK | 74 |
| `mlflow run . -e train --env-manager local -P question=p1 -P model=naive -P stage=baseline` | OK | 56 |
| `mlflow run . -e train --env-manager local -P question=p1 -P model=poisson` | OK | 53 |
| `mlflow run . -e train --env-manager local -P question=p1 -P model=hgb` | OK | 89 |
| `mlflow run . -e train --env-manager local -P question=p1 -P model=rf -P n_estimators=200` | OK | 109 |
| `mlflow run . -e search --env-manager local -P question=p1` | OK | 297 |
| `mlflow run . -e train --env-manager local -P question=p1 -P from_search=latest` | OK | 85 |
| `mlflow run . -e train --env-manager local -P question=p2 -P model=logreg -P stage=baseline` | OK | 333 |
| `mlflow run . -e train --env-manager local -P question=p2 -P model=hgb` | OK | 413 |
| `mlflow run . -e train --env-manager local -P question=p2 -P model=rf` | OK | 376 |
| `mlflow run . -e search --env-manager local -P question=p2` | OK | 1,083 |
| `mlflow run . -e train --env-manager local -P question=p2 -P from_search=latest` | OK | 205 |
| `mlflow run . -e train --env-manager local -P question=p3 -P stage=baseline` | OK | 37 |
| `mlflow run . -e train --env-manager local -P question=p3 -P model=kmeans` | OK | 156 |
| `mlflow run . -e train --env-manager local -P question=p3 -P model=agglomerative` | OK | 32 |
| `mlflow run . -e train --env-manager local -P question=p3 -P model=gmm` | OK | 31 |
| `mlflow run . -e search --env-manager local -P question=p3` | OK | 30 |
| `mlflow run . -e register --env-manager local -P question=p1` | OK | 14 |
| `mlflow run . -e register --env-manager local -P question=p2` | OK | 13 |
| `mlflow run . -e register --env-manager local -P question=p3` | OK | 14 |
| `mlflow run . -e verify --env-manager local -P question=all -P alias=champion` | OK | 31 |

## Incidencias y cómo se resolvieron

1. **Parámetros vacíos en Windows.** El primer intento (19 de octubre de 2026, 14:14) falló en todas las corridas de
   entrenamiento: `mlflow run` ejecuta el comando con `cmd /c` y los parámetros con valor `""` llegaban
   como el texto `""`, que MLflow rechazaba como `run_id`. Se reemplazaron los valores vacíos del
   `MLproject` por palabras clave (`config`, `auto`, `no`) que el código resuelve con `config.py`
   (commit `3636d46`).
2. **Manifiesto con CRLF.** En Windows `write_text` convertía los saltos de línea y Git veía
   `data/manifest.json` modificado, por lo que la ingesta del primer intento quedó con `git_sucio=True`.
   Se escribe con `newline="\n"` (commit `f974508`).
3. **Árbol de trabajo sin actualizar.** En el segundo intento la rama local apuntaba al commit nuevo, pero los
   archivos seguían en la versión anterior. El protocolo detectó cambios sin commit en el paso 1, antes de
   lanzar ninguna corrida, y se detuvo. Ahora el lanzador hace `git reset --hard` al commit del paquete y no
   corre si `git diff --quiet HEAD` encuentra cambios.
4. **Corridas del primer intento.** Las 12 corridas fallidas y la ingesta con árbol sucio se archivaron con
   borrado lógico (recuperables con `mlflow runs restore`) mediante `scripts/limpiar_corridas_fallidas.py`.
   Siguen visibles en la interfaz con el filtro *State: Deleted*. Así el experimento oficial solo contiene
   corridas reproducibles desde su commit.

## Prueba de clon limpio con el entorno declarado

- **Cuándo y dónde:** 19 de octubre de 2026, 14:11–14:12 (hora del centro de México), en un entorno Linux de
  prueba, antes de la ejecución oficial.
- **Qué se hizo:** clon limpio del repositorio y `mlflow run` **sin** `--env-manager local`, para que MLflow
  creara el entorno de `python_env.yaml` con `uv` (CPython 3.14.6, dependencias fijadas).
- **Comandos:** `mlflow run . -e ingest --env-manager uv -P raw_source=<copia local de los CSV>` y
  `mlflow run . -e train --env-manager uv -P question=p1 -P model=poisson`, contra un servidor de tracking
  local en un experimento de prueba (`clon-limpio-uv`).
- **Resultado:** la ingesta produjo `data_version = 7aa5e436d887` (4,532,032 viajes, 484 estaciones) y el
  modelo Poisson obtuvo `val_mae = 5.1037`, los mismos valores que la ejecución oficial.

## Publicación

- La rama se publicó el 25 de octubre de 2026 en el repositorio de entrega
  [`JohannaCamargo2805/TC5061-Proyecto-Equipo32`](https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32) con la cuenta de Leydy Johanna Camargo Medina, y
  se integró a `main` con el Pull Request [#5](https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32/pull/5) (*Create a merge commit*).
- Ese repositorio tenía en `main` solo la estructura base de cookiecutter. Para conservar el historial del
  equipo y los hashes de los commits, la rama une ambos historiales con un merge que mantiene el árbol del
  proyecto (`git merge --allow-unrelated-histories -s ours`).
- Las corridas guardan en `mlflow.source.git.repoURL` la URL del repositorio del equipo porque se lanzaron
  desde ese clon. El commit `e6c1a60` es el mismo en los dos repositorios, así que el linaje
  (`git_commit`, `data_version`) sigue siendo verificable.

## Resultados

| Pregunta | Corrida | Métrica de selección (VAL) | Métrica de prueba | Registry |
|---|---|---|---|---|
| P1 | `p1-baseline-naive` | `val_mae` = 6.232 ± 0.116 | `test_mae` = 7.219 |  |
| P1 | `p1-candidate-poisson` | `val_mae` = 5.104 ± 0.103 | `test_mae` = 5.410 | @champion (v1) |
| P1 | `p1-candidate-hgb` | `val_mae` = 5.326 ± 0.113 | `test_mae` = 6.339 |  |
| P1 | `p1-candidate-rf` | `val_mae` = 5.068 ± 0.090 | `test_mae` = 5.718 |  |
| P1 | `p1-candidate-hgb-tuned` | `val_mae` = 5.129 ± 0.095 | `test_mae` = 5.904 | @challenger (v2) |
| P2 | `p2-baseline-logreg` | `val_pr_auc` = 0.415 ± 0.006 | `test_pr_auc` = 0.351 |  |
| P2 | `p2-candidate-hgb` | `val_pr_auc` = 0.484 ± 0.009 | `test_pr_auc` = 0.394 | @champion (v1) |
| P2 | `p2-candidate-rf` | `val_pr_auc` = 0.470 ± 0.008 | `test_pr_auc` = 0.383 |  |
| P2 | `p2-candidate-hgb-tuned` | `val_pr_auc` = 0.488 ± 0.008 | `test_pr_auc` = 0.402 | @challenger (v2) |
| P3 | `p3-baseline-kmeans` | `train_silhouette` = 0.198 | `test_silhouette` = 0.186 |  |
| P3 | `p3-candidate-kmeans` | `train_silhouette` = 0.186 | `test_silhouette` = 0.167 | @champion (v1) |
| P3 | `p3-candidate-agglomerative` | `train_silhouette` = 0.109 | `test_silhouette` = 0.122 |  |
| P3 | `p3-candidate-gmm` | `train_silhouette` = 0.162 | `test_silhouette` = 0.132 | @challenger (v2) |

