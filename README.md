# TC5061-Proyecto-Equipo32
Proyecto integrador de TC5061 (Operaciones de aprendizaje automático): propuesta de solución de ML
sobre los datos abiertos de [MiBici](https://mibici.net/es/datos-abiertos/), el sistema de bicicleta
pública del Área Metropolitana de Guadalajara. Datos: ene–dic 2025 (4,532,032 viajes, 484 estaciones).

El proyecto responde tres preguntas de negocio:

| # | Pregunta | Tipo de tarea | Usuario |
|---|----------|---------------|---------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión |

El ML Canvas y el diagrama de arquitectura se encuentran en `references/`.

## Entrega 2: MLflow Project, experimentación y Model Registry

**Repositorio de entrega:** <https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32>. Conserva el historial completo del repositorio del equipo
([`a01840537/TC5061-Proyecto-Equipo32`](https://github.com/a01840537/TC5061-Proyecto-Equipo32), donde se integró la Propuesta de proyecto con los
Pull Requests #1–#12 de ese repositorio) y recibe la entrega 2 en `main` mediante el Pull Request
[#5](https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32/pull/5). Los commits son los mismos en ambos repositorios; por eso las etiquetas
`git_commit` de las corridas (`e6c1a60`) se pueden consultar aquí.

El repositorio es un **MLflow Project** (`MLproject` + `python_env.yaml`): cualquier integrante
ejecuta la ingesta, el entrenamiento, las búsquedas y el registro con un solo comando
`mlflow run`, contra un **servidor local de tracking respaldado por SQLite** (requerido por el
Model Registry). Todas las corridas de esta entrega se lanzaron así, no desde un notebook.

### 1. Servidor de tracking

```bash
# Terminal 1 (déjala abierta). En Windows: powershell -ExecutionPolicy Bypass -File scripts\servidor_mlflow.ps1
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
    --host 127.0.0.1 --port 5000

# Terminal 2: apunta MLflow al servidor y al experimento único del equipo
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000          # PowerShell: $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_EXPERIMENT_NAME=mibici-equipo32             # PowerShell: $env:MLFLOW_EXPERIMENT_NAME="mibici-equipo32"
```

`mlflow.db`, `mlruns/` y `mlartifacts/` están en `.gitignore`: la base de datos de MLflow, los
artefactos y los modelos serializados no se versionan.

### 2. Puntos de entrada del `MLproject`

| Punto de entrada | Qué hace | Parámetros principales |
|---|---|---|
| `ingest` | Ejecuta `dataset.py`: descarga/normaliza los CSV y genera `data/manifest.json` (`data_version`) | `raw_source` (carpeta con copia local de los CSV, opcional) |
| `train` | Entrena **un** modelo de **una** pregunta y registra métricas, artefactos y modelo con firma | `question` (p1, p2, p3), `model`, `stage` (baseline/candidate), hiperparámetros (`n_estimators`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `alpha`, `C`, `k`, `linkage`, …), `from_search` |
| `search` | Búsqueda de hiperparámetros: una corrida **padre** y una corrida **hija anidada** por combinación | `question`, `model`, `grid` |
| `register` | Registra la corrida elegida en el Model Registry y le asigna los aliases `champion`/`challenger` | `question`, `run_id`, `challenger_run_id` |
| `verify` | Carga `models:/<nombre>@champion` y comprueba que reproduce la métrica registrada | `question`, `alias` |

Los parámetros con valor `config` (o `auto`) toman su valor de `mibici/config.py` (catálogos `P1_MODELOS`,
`P2_MODELOS`, `P3_MODELOS`, rejillas `BUSQUEDAS`, partición temporal, rutas y semilla): el
`MLproject` no repite rutas, fechas ni semillas. Se usan palabras clave y no cadenas vacías porque
`mlflow run` en Windows ejecuta el comando con `cmd /c`, que no transmite bien los argumentos vacíos.

### 3. Comandos para reproducir desde un clon limpio

```bash
git clone https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32.git
cd TC5061-Proyecto-Equipo32
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# (servidor y variables de entorno del paso 1)

mlflow run . -e ingest --env-manager local             # data_version = 7aa5e436d887
mlflow run . -e train  --env-manager local -P question=p1 -P model=poisson               # P1 champion
mlflow run . -e train  --env-manager local -P question=p2 -P model=hgb                   # P2 champion
mlflow run . -e train  --env-manager local -P question=p3 -P model=kmeans -P k=4         # P3 champion
```

- `--env-manager local` usa el entorno ya activado (instalado con `requirements.txt`, mismas
  versiones que `python_env.yaml`). Sin esa opción, MLflow crea el entorno declarado en
  `python_env.yaml` (`--env-manager uv` si tienes `uv`, o `virtualenv` con `pyenv`).
- Si no hay red hacia el portal de MiBici, la ingesta puede copiar los CSV de una carpeta local
  (`-P raw_source=<carpeta>`); el SHA-256 del manifiesto confirma que son los mismos archivos.
- Protocolo completo de la entrega (baselines, candidatas, búsquedas, registry y verificación):
  `scripts/ejecutar_entrega2.ps1` (Windows) o `scripts/ejecutar_entrega2.sh` / `make experimentos`
  (Linux/macOS).

Ejemplos de variaciones:

```bash
mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300
mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05 -P max_leaf_nodes=63
mlflow run . -e search -P question=p1                    # rejilla de BUSQUEDAS["p1"] (nested runs)
mlflow run . -e train -P question=p1 -P from_search=latest   # entrena la mejor combinación
```

### 4. Qué registra cada corrida candidata

| Elemento | Dónde |
|---|---|
| Etiquetas para filtrar | `question`, `model_family`, `stage_of_work` (`baseline`, `search`, `candidate`), `entrega`, `equipo`, `git_commit`, `git_sucio`, `data_version`, `responsable` (y `mlflow.source.git.commit` que agrega `mlflow run`) |
| Modelo con firma y ejemplo de entrada | *Logged model* `model` (skops; el paquete `mibici` viaja con el modelo en `code_paths`) |
| Versión de los datos | `data/manifest.json`, `data/resumen_conjuntos.json` y los conjuntos TRAIN/VAL/TEST con `mlflow.log_input` (solo esquema, filas y huella digital) |
| Gráficas | `plots/`: P1 real contra predicho, residuales, total diario y MAE por día; P2 matriz de confusión, curva precisión-recall y elección del umbral; P3 PCA, perfiles de los grupos y mapa |
| Configuración | `config/`: `config.py`, `MLproject`, `python_env.yaml`, `requirements.txt` y `run_config.json` (parámetros resueltos) |
| Evaluación | `evaluation/`: métricas, predicciones de prueba, importancia por permutación y tablas |

Métricas: las mismas de la entrega anterior con prefijo `train_`, `val_` y `test_` (P1: MAE, RMSE,
WAPE, R², sesgo, error del total diario; P2: PR-AUC, F1, recall, precisión, recall por tipo;
P3: silueta, Davies-Bouldin, Calinski-Harabasz y ARI de estabilidad), más estabilidad
(`test_mae_semanal_std`, `test_f1_semanal_std`, `estabilidad_semillas_ari`), error estándar de
validación (`val_mae_se`, `val_pr_auc_se`) y costo (`costo_entrenamiento_seg`,
`costo_inferencia_ms_1k`, `tamano_modelo_mb`). No se registran datos personales: la ingesta
elimina el pseudónimo, el género y el año de nacimiento.

### 5. Model Registry: esquema de aliases

| Pregunta | Modelo registrado | Producción | Retador |
|---|---|---|---|
| P1 | `mibici-equipo32-p1-demanda` | `@champion` | `@challenger` |
| P2 | `mibici-equipo32-p2-riesgo` | `@champion` | `@challenger` |
| P3 | `mibici-equipo32-p3-grupos` | `@champion` | `@challenger` |

```bash
mlflow run . -e register -P question=p1        # run_id=auto: aplica la regla de selección de config.py
mlflow run . -e verify                         # carga models:/<nombre>@champion de las tres preguntas
```

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo32-p1-demanda@champion")
```

Usamos **aliases** y no las etapas clásicas (`Staging`/`Production`) porque las etapas están
obsoletas desde MLflow 2.9 y los aliases son el esquema vigente: un alias es un puntero
mutable a una versión, así que pasar a producción o hacer *rollback* es mover `@champion`
sin que los consumidores cambien su URI, y `@challenger` deja lista la siguiente mejor versión
para comparación. Cada versión guarda su linaje como etiquetas (`experimento`,
`source_run_id`, `git_commit`, `data_version`, `mlproject_entry_point`) y en su descripción.

**Regla de selección** (`SELECCION` en `config.py`, se calcula solo con TRAIN/VAL): P1 y P2
usan la regla de un error estándar (entre los candidatos a menos de 1 EE del mejor en
validación se elige el de menor complejidad); P3 maximiza la silueta entre los modelos con al
menos 3 grupos, `ari_historico_reducido >= 0.8` y `estabilidad_semillas_ari >= 0.9`.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Entregas: Propuesta de proyecto (semana 5) y MLflow Project con experimentación y Model Registry (semana 6).

### 6. Resultados de la entrega 2

Corridas oficiales lanzadas con `mlflow run` desde el commit `e6c1a60` (árbol limpio,
`data_version = 7aa5e436d887`) en el experimento `mibici-equipo32`. Detalle, análisis y capturas en
`TC5061_Reporte_MLFlow_Equipo32.pdf`; bitácora en [`docs/entrega2_bitacora.md`](docs/entrega2_bitacora.md).

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


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── MLproject          <- MLflow Project: puntos de entrada ingest, train, search, register y verify
├── python_env.yaml    <- Entorno del MLflow Project (mismas versiones que requirements.txt)
├── Makefile           <- Comandos: `make server`, `make data`, `make experimentos`, `make verify`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── scripts            <- Protocolo de la entrega 2 (Windows/Linux) y servidor de tracking
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── p1_regresion.py     <- P1: viajes diarios por estación (baseline + 2 candidatos, MLflow)
    │   ├── p2_clasificacion.py <- P2: riesgo de quedarse sin bicis/anclajes (MLflow)
    │   ├── p3_clustering.py    <- P3: clustering de estaciones por patrón de uso (MLflow)
    │   ├── tareas.py           <- Entrega 2: datos, modelos, métricas y gráficas de P1, P2 y P3
    │   ├── train.py            <- Punto de entrada `train` (una corrida por modelo)
    │   ├── search.py           <- Punto de entrada `search` (nested runs)
    │   ├── registry.py         <- Punto de entrada `register` (Model Registry con aliases)
    │   ├── predict.py          <- Punto de entrada `verify` (carga models:/<nombre>@champion)
    │   ├── seguimiento.py      <- Etiquetas, versión de datos, config y modelo con firma en MLflow
    │   ├── seleccion.py        <- Regla de selección (1 error estándar / restricciones de P3)
    │   └── estimadores.py      <- Estimadores propios (baseline ingenuo, log1p, aglomerativo, GMM)
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Reproducibilidad y MLOps

El proyecto utiliza una estructura basada en Cookiecutter Data Science. La lógica final se mantiene dentro del paquete `mibici`; los notebooks se reservan para exploración.

El entorno fue probado con Python 3.14.4. Las dependencias directas se encuentran fijadas por versión en `requirements.txt`.
También se verificó la ejecución completa (ingesta, pruebas, P1, P2 y P3) con **Python 3.12.13** desde un clon limpio.

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt


```

La consistencia del entorno puede verificarse con:

```bash
python -m pip check
```

### Datos

La ingesta se ejecuta desde el módulo:

```bash
python -m mibici.dataset
```

`dataset.py` descarga los datos públicos de MiBici definidos en `config.py`, genera los archivos procesados y crea `data/manifest.json`.

El manifiesto incluye una `data_version` calculada a partir de los archivos de entrada. Esta versión se registra posteriormente en MLflow para mantener trazabilidad entre datos y experimentos.

Los datos crudos, intermedios y procesados no se versionan en Git.

## Reproducción de la Propuesta de proyecto (entrega 1)

Desde la entrega 2 la reproducción oficial se hace con el **MLflow Project** (ver
[Entrega 2](#entrega-2-mlflow-project-experimentación-y-model-registry)): `mlflow run . -e ingest`
y `mlflow run . -e train ...` contra el servidor de tracking. Los scripts de la propuesta
(`python -m mibici.modeling.p1_regresion`, `p2_clasificacion` y `p3_clustering`) se conservan
como referencia; sus baselines se reproducen ahora con `stage=baseline`:

```bash
mlflow run . -e train -P question=p1 -P model=naive  -P stage=baseline   # p1_baseline_naive_semanal
mlflow run . -e train -P question=p2 -P model=logreg -P stage=baseline   # p2_baseline_logreg
mlflow run . -e train -P question=p3 -P stage=baseline                   # p3_baseline_kmeans_k2
python -m pytest tests                                                          # pruebas
```

La ingesta debe producir `data_version = 7aa5e436d887`; si el valor cambia, los archivos de MiBici
descargados no son los mismos que se usaron en los experimentos.

**Privacidad.** Los CSV de MiBici incluyen un pseudónimo de usuario, el género y el año de nacimiento.
Como ninguna pregunta los necesita, `dataset.py` los elimina antes de escribir los Parquet procesados
(minimización de datos). Los datos se usan con fines académicos citando la fuente (portal de datos
abiertos de MiBici) y no se redistribuyen en este repositorio.

## P1 — Regresión de viajes diarios por estación

**Pregunta:** ¿cuántos viajes se iniciarán en cada estación el día siguiente? Usuario: operador del
sistema (planeación de flota y turnos). Granularidad: estación × día; horizonte: 1 día (la predicción
del día D se emite al cierre de D-1).

```bash
python -m mibici.modeling.p1_regresion
```

- **Partición temporal:** TRAIN ene-sep 2025, VAL oct 2025 (selección del modelo), TEST nov-dic 2025.
- **Variables** (`mibici/features.py`, funciones `p1_*`): calendario del día objetivo (día de la semana,
  fin de semana, feriado), rezagos de 1, 2, 7 y 14 días, medias móviles de 7 y 28 días que terminan en
  D-1, promedio del mismo día de las 4 semanas previas, antigüedad y capacidad de la estación.
  `p1_agregar_features` lanza `LeakageError` si se pide un rezago menor que el horizonte.
- **Modelos:** baseline ingenuo semanal (`t − 7 días`), candidato lineal `PoissonRegressor`
  (conteos en `log1p`) y candidato `HistGradientBoostingRegressor` con pérdida Poisson.
- **Métricas:** `train_/val_/test_` + `mae`, `rmse`, `wape`, `r2`, `sesgo` y `mape_total_dia`.
- **Runs:** `p1_baseline_naive_semanal`, `p1_candidato_poisson_lineal`, `p1_candidato_hgb_poisson`.

## P2 — Clasificación de riesgo de disponibilidad

**Pregunta:** ¿una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas
2 horas? Usuario: equipo de rebalanceo.

```bash
python -m mibici.modeling.p2_clasificacion
```

- **Etiqueta:** riesgo si `|neto(t+1 … t+2)| >= 0.4 × capacidad` (`p2_agregar_etiqueta`).
- **Partición temporal:** TRAIN ene-sep, VAL oct (umbral de decisión que maximiza F1), TEST nov-dic 2025.
- **Modelos:** baseline regresión logística balanceada y candidato `HistGradientBoostingClassifier`.
- **Runs:** `p2_baseline_logreg`, `p2_candidato_hgb`.

## P3 — Clustering de estaciones

La pregunta P3 busca agrupar estaciones de MiBici de acuerdo con sus patrones de uso y evaluar si esos perfiles se mantienen estables en el tiempo.

La configuración de P3 se encuentra centralizada en `mibici/config.py`. El pipeline final se ejecuta con:

```bash
python -m mibici.modeling.p3_clustering
```

### Partición temporal

Para evitar fuga de información se utiliza una separación temporal sin mezcla aleatoria:

- TRAIN: 1 de enero al 19 de octubre de 2025.
- TEST: 20 de octubre al 31 de diciembre de 2025.
- Ventana histórica reducida: 1 de abril al 19 de octubre de 2025.

Los viajes con duración menor a 60 segundos o mayor a 24 horas se excluyen. También se eliminan viajes que cruzan los límites temporales entre TRAIN y TEST.

### Construcción de características

El feature engineering se encuentra en `mibici/features.py`.

Una estación se considera elegible cuando:

- tiene al menos 30 días de historial;
- presenta menos de 60 días de inactividad al cierre del periodo.

Para cada estación se generan 12 características de comportamiento relacionadas con intensidad, balance, uso en fin de semana y distribución de salidas y llegadas por bloque horario.

Las variables de intensidad se transforman con `log1p` y posteriormente se aplica `StandardScaler`. El escalador se ajusta exclusivamente con TRAIN.

### Comparación de modelos

El pipeline reproduce tres alternativas:

- Baseline: K-Means con k=2.
- Candidato: Agglomerative Clustering con enlace Ward y k=4.
- Modelo final: K-Means con k=4.

La comparación utiliza Silhouette, Davies-Bouldin y Calinski-Harabasz. Para el modelo final también se evalúa estabilidad temporal mediante ARI, NMI y proporción de estaciones que conservan su cluster.

### MLflow Tracking

Todos los runs del proyecto se registran en el experimento único `mibici-equipo32`;
la pregunta atendida se distingue mediante tags (`problema: P1/P2/P3`).

Los runs tienen nombres descriptivos:

- `p3_baseline_kmeans_k2`
- `p3_candidato_agglomerative_k4`
- `p3_kmeans_final`

Además de los tres runs de P3, el experimento contiene los runs de P1 (`p1_baseline_naive_semanal`,
`p1_candidato_poisson_lineal`, `p1_candidato_hgb_poisson`) y de P2 (`p2_baseline_logreg`,
`p2_candidato_hgb`). Todos llevan los tags `equipo = equipo32`, `problema` (P1/P2/P3), `pregunta`,
`etapa` (baseline/candidato), `git_commit` y `data_version`.

Cada run registra, según corresponda:

- versión de los datos (`data_version`);
- commit de Git (`git_commit`);
- pregunta atendida;
- rango temporal;
- semilla y modelo;
- hiperparámetros;
- reglas de filtrado y elegibilidad;
- número de características;
- transformación y escalamiento;
- métricas diferenciadas por conjunto, como `train_silhouette` y `test_silhouette`.

El run final registra como artefactos el modelo serializado, la visualización PCA, el mapa interactivo, el resultado geográfico y la tabla resumen de clusters.

La interfaz local de MLflow puede iniciarse desde la raíz del repositorio con:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

y consultarse en `http://localhost:5000`.

## Git y control de versiones

Los datos de MiBici, modelos serializados, base de datos de MLflow, artefactos de MLflow y resultados generados por P3 se excluyen mediante `.gitignore`.

El código se desarrolla en ramas independientes y todo cambio hacia `main` debe incorporarse mediante Pull Request revisado por otro integrante del equipo.

Convenciones: ramas `feat/`, `fix/`, `docs/` + descripción corta; mensajes de commit
siguiendo [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/). En el
repositorio del equipo la rama `main` está protegida; cada integrante abre al menos un PR y revisa al
menos uno.

Para P3 se utiliza la rama `feature/p3-clustering`.

## Equipo 32

| Integrante | Matrícula | GitHub | Rol |
|------------|-----------|--------|-----|
| Rolando Calderón Carballo | A01840537 | `a01840537` | Infraestructura: repo, ingesta reproducible, versionado de datos, configuración |
| Leydy Johanna Camargo Medina | A01840540 | `JohannaCamargo2805` | P1 — Regresión: demanda de viajes por estación |
| Manuel Alejandro Cardona Esquivel | A01795954 | `alexcard33` | P2 — Clasificación: riesgo de disponibilidad por estación |
| Gerardo Iván Conde Reynoso | A01840602 | `GIvanConder` | P3 — Clustering: patrones de uso de estaciones |

El detalle de qué hizo cada integrante (commits, Pull Requests y revisiones) está en
[`CONTRIBUCIONES.md`](CONTRIBUCIONES.md).