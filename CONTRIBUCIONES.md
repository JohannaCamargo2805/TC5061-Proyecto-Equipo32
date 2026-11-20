# Contribuciones del Equipo 32

Registro de qué hizo cada integrante, construido a partir del historial de Git y de los Pull Requests
del repositorio (`git log --all` y pestaña *Pull requests* de GitHub). Sirve como evidencia de la
evaluación individual: commits propios, PR abiertos y revisiones hechas a otros integrantes.

| Integrante | Matrícula | GitHub | Responsabilidad |
|---|---|---|---|
| Rolando Calderón Carballo | A01840537 | `a01840537` (dueño del repo) | Infraestructura: estructura CCDS, ingesta, versionado de datos, configuración, README, ML Canvas y arquitectura |
| Gerardo Iván Conde Reynoso | A01840602 | `GIvanConder` | P3: clustering de estaciones por patrón de uso |
| Manuel Alejandro Cardona Esquivel | A01795954 | `alexcard33` | P2: clasificación de riesgo de vaciado/llenado |
| Leydy Johanna Camargo Medina | A01840540 | `JohannaCamargo2805` | P1: regresión de viajes diarios por estación; cierre de la entrega |

## Rolando Calderón Carballo (`a01840537`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `75e3826` | `main` | `chore`: estructura base generada con cookiecutter-data-science |
| `4e4f6cb` | `feat/ingesta` | `feat`: ingesta reproducible de viajes y estaciones con versionado por manifiesto (`dataset.py`, `config.py`) |
| `ba4bcef` | `fix/experimento-unico-y-manifiesto` | `fix`: experimento único `mibici-equipo32` y `data/manifest.json` versionado en Git |
| `da671ef` | `docs/canvas-y-arquitectura` | `docs`: README completo, ML Canvas y diagrama de arquitectura en `references/` |

- **PR abiertos:** #1 `feat: ingesta reproducible y configuracion central`, #3 `fix: experimento unico del equipo y manifiesto versionado`, #4 `docs: ML Canvas y diagrama de arquitectura`.
- **Revisiones:** aprobó el PR #2 (P3, Gerardo).
- **Integró:** PR #1.

## Gerardo Iván Conde Reynoso (`GIvanConder`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `bcd2039` | `feature/p3-clustering` | `feat(p3)`: inicializa el módulo de clustering |
| `42bc1e0` | `feature/p3-clustering` | `feat(p3)`: pipeline reproducible de clustering (`p3_clustering.py`, funciones `p3_*` de `features.py`, configuración P3, sección P3 del README) |

- **PR abiertos:** #2 `feat(p3): implementa pipeline reproducible de clustering`.
- **Revisiones:** aprobó el PR #1 (ingesta, Rolando) y el PR #5 (P2, Manuel).
- **Integró:** PR #2.

## Manuel Alejandro Cardona Esquivel (`alexcard33`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `9289545` | `feature/acardona/p2-clasificacion` | P2: `p2_clasificacion.py`, funciones `p2_*` de `features.py`, configuración P2, objetivo `p2` del Makefile y `.gitignore` |

- **PR abiertos:** #5 `Agregado pregunta 2`.
- **Integró:** PR #3 y PR #5.
- **Nota:** el commit `9289545` se hizo con un correo que no está asociado a la
  cuenta `alexcard33`; GitHub lo muestra sin usuario. Al agregar ese correo en *Settings → Emails* de
  GitHub, el commit queda atribuido a su cuenta.

## Leydy Johanna Camargo Medina (`JohannaCamargo2805`)

Rama `feat/p1-regresion-y-cierre` (PR de cierre de la entrega):

| Commit | Qué hizo |
|---|---|
| `d32c7e1` | `fix(build)`: corrige el Makefile (el objetivo `p2` tenía un error de separador) y agrega `data`, `p1`, `p2`, `p3`, `train` y `mlflow-ui` |
| `3325b7d` | `fix(dataset)`: elimina pseudónimo de usuario, género y año de nacimiento antes de guardar los Parquet (minimización de datos) |
| `8b1f983` | `docs(features)`: documenta todas las funciones de P2 y P3 y la regla común de no fuga |
| `4a592fa` | `feat(p1)`: P1 completa: configuración, funciones `p1_*` en `features.py`, `modeling/p1_regresion.py` (baseline ingenuo semanal, Poisson lineal y HGB Poisson), pruebas `tests/test_features_p1.py` |
| `b9f1a31` | `feat(mlflow)`: agrega la etiqueta `equipo = equipo32` a los runs de P2 y P3 |
| `b089c6c` | `docs(notebooks)`: notebook de exploración `1.0-ljcm-exploracion-demanda-p1.ipynb` |
| `6754760` | `docs(readme)`: reproducción desde un clon limpio, secciones P1 y P2, privacidad y tabla del equipo con matrícula y usuario |

- Ejecutó los runs oficiales de P1, P2 y P3 desde el commit `b10a1fb` (árbol limpio) en el experimento
  `mibici-equipo32` con `data_version = 7aa5e436d887`, y tomó las capturas de MLflow del reporte.
- **Integró:** PR #4.

## Revisiones cruzadas

Los números de esta tabla son los Pull Requests de la entrega 1 en el repositorio del equipo
([`a01840537/TC5061-Proyecto-Equipo32`](https://github.com/a01840537/TC5061-Proyecto-Equipo32)).

| PR | Autor | Revisó / aprobó | Integró |
|---|---|---|---|
| #1 | Rolando | Gerardo (aprobado) | Rolando |
| #2 | Gerardo | Rolando (aprobado) | Gerardo |
| #3 | Rolando | — | Manuel |
| #4 | Rolando | — | Johanna |
| #5 | Manuel | Gerardo (aprobado) | Manuel |
| Cierre (P1) | Johanna | pendiente: Manuel | — |

## Entrega 2: MLflow Project, experimentación y Model Registry

### Leydy Johanna Camargo Medina (`JohannaCamargo2805`)

Rama `feat/mlflow-project-registry`, integrada a `main` del repositorio de entrega
([`JohannaCamargo2805/TC5061-Proyecto-Equipo32`](https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32)) con el Pull Request [#5](https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32/pull/5):

| Commit | Qué hizo |
|---|---|
| `38f4a42` | feat(config): parametros de la entrega 2 (tracking, registry, catalogo de modelos y busquedas) |
| `108b388` | feat(dataset): ingesta como punto de entrada del MLflow Project |
| `4af9ee0` | feat(modeling): entrenamiento por pregunta con artefactos, version de datos y modelo con firma |
| `601c073` | feat(search): busqueda de hiperparametros con nested runs y regla de seleccion |
| `1ee6482` | feat(registry): Model Registry con aliases champion/challenger y verificacion por alias |
| `afdb307` | feat(mlproject): MLproject con puntos de entrada ingest, train, search, register y verify |
| `046b791` | test: pruebas del MLproject, valores por defecto y regla de seleccion |
| `d1b4d41` | build: protocolo de la entrega 2, servidor de tracking y objetivos del Makefile |
| `2598d88` | docs(readme): servidor de tracking, comandos de mlflow run y esquema del registry |
| `d04a312` | fix(scripts): solo ASCII en los .ps1 para Windows PowerShell 5.1 (lee los scripts como ANSI) |
| `24c501e` | fix(modeling): escribir los JSON de artefactos en UTF-8 (Windows usa cp1252 por defecto) |
| `3636d46` | fix(mlproject): valores por defecto con palabras clave en lugar de cadenas vacias |
| `f974508` | fix(dataset): escribir el manifiesto con LF y UTF-8 |
| `55e579d` | build(scripts): archivar corridas fallidas o con arbol sucio antes de la ejecucion oficial |
| `e6c1a60` | build(scripts): archivar tambien corridas huerfanas en estado RUNNING |

- Ejecutó en su computadora, con `mlflow run`, todas las corridas oficiales de la entrega (commit
  `e6c1a60`, árbol limpio) y el registro de los tres modelos en el Model Registry.
- Tomó las capturas de la interfaz de MLflow y elaboró el reporte `TC5061_Reporte_MLFlow_Equipo32.pdf`.
- Publicó la entrega en el repositorio de entrega con su cuenta, junto con los commits de documentación y
  de integración del historial del equipo que acompañan al Pull Request.

### Revisiones y Pull Requests de la entrega 2

| Integrante | Pull Request abierto | Revisión realizada |
|---|---|---|
| Leydy Johanna Camargo Medina | #5 (repositorio de entrega) | pendiente de registrar |
| Rolando Calderón Carballo | pendiente | pendiente |
| Manuel Alejandro Cardona Esquivel | pendiente | pendiente |
| Gerardo Iván Conde Reynoso | pendiente | pendiente |

> Cada integrante debe abrir al menos un Pull Request y revisar al menos uno en esta entrega; esta tabla se
> actualiza conforme se registren en GitHub.

## Entrega 3: pruebas automatizadas, validación de datos y seguridad (versión individual)

Por la mecánica que acordó la coordinación con el profesor, cada integrante puede entregar su propia
versión de las actividades en equipo que faltan. Esta versión es de **Leydy Johanna Camargo Medina**
(`JohannaCamargo2805`) y está en `main` de este repositorio. Ramas y commits (todos a su nombre):

| Rama | Commits | Qué contiene |
|---|---|---|
| `test/pruebas-unitarias` | 8 | `tests/conftest.py`, pruebas de ingesta, variables de P1, P2 y P3, métricas y reproducibilidad del entrenamiento |
| `test/validacion-deepchecks` | 4 | `mibici/compat_deepchecks.py`, `mibici/validacion.py` y `tests/test_validacion_deepchecks.py` |
| `chore/seguridad` | 9 | `mibici/git_seguro.py`, `.env.example`, actualización de dependencias, `scripts/escanear_secretos.py`, `scripts/chequeos_seguridad.py` y `tests/test_seguridad.py` |
| `fix/ajustes-primera-ejecucion` | 7 | `scripts/ejecutar_calidad.ps1`, objetivos de `make` y correcciones tras la primera ejecución en Windows |
| `fix/revision-independiente` | 4 | Correcciones de una revisión independiente (duplicados sin `viaje_id`, archivos sensibles, prueba de `.gitignore`, resumen con commit) |
| `docs/readme-evidencia` | 3 | README, este archivo, reportes de DeepChecks y de seguridad, capturas de la ejecución y una regla del escáner para los hashes de Git de su propio reporte |

Ejecutó las pruebas, DeepChecks y los chequeos de seguridad en su computadora con
`scripts\ejecutar_calidad.ps1 -Capturas`; las capturas y la salida completa están en `docs/evidencia/`
y `reports/calidad/`. Los commits incluyen la línea `Co-Authored-By: Claude` porque se usó Claude
(Anthropic) como asistente de programación.

## Entrega 4: el modelo P1 como API REST en un contenedor (versión individual)

Misma mecánica que la entrega 3: versión de **Leydy Johanna Camargo Medina** (`JohannaCamargo2805`),
en `main` de este repositorio. Ramas y commits (todos a su nombre):

| Rama | Commits | Qué contiene |
|---|---|---|
| `feat/registry-exportar-modelo` | 2 | Constantes de la entrega en `mibici/config.py` y `scripts/exportar_modelo_registry.py` (modelo por nombre y versión o alias, `registro.json` con linaje y SHA-256) |
| `feat/api-fastapi` | 7 | Paquete `api/` (esquemas Pydantic con validación de tipo, rango y consistencia; carga verificada del modelo; endpoints `/health`, `/modelo`, `/predict` y `/predict/lote`), dependencias, `tests/test_api.py` y `docs/ejemplos/` |
| `build/contenedor` | 5 | `requirements-api.in/.txt`, `Containerfile`, `.dockerignore`, `tests/test_contenedor.py`, `scripts/ejecutar_servicio.ps1`, objetivos de `make` y auditoría de las dependencias de la imagen |
| `build/modelo-servicio-v1` | 2 | `models/servicio/` (v1 de `mibici-equipo32-p1-demanda` descargada del registry) y `.gitattributes` |
| `fix/escaner-registro-modelo` | 1 | Regla revisada del escáner de secretos para las huellas de `registro.json` |
| `docs/readme-servicio` | 3 | README, este archivo, capturas (`docs/evidencia/entrega4/`) y salida de la primera ejecución (`reports/servicio/`, `reports/seguridad/`) |
| `fix/revision-independiente-e4` | 5 | Correcciones de una revisión independiente (validación de `estacion_id`, `fecha` y `lag_14`, costo de inferencia en `registro.json`, base por *digest*, versión como argumento, `make probar-api`, script de Windows) |
| `docs/revision-e4` | 4 | Regla del escáner para las respuestas de la API, evidencia de la ejecución final, reportes de seguridad y README |
| `build/dockerfile-por-defecto` | 2 | `Dockerfile` idéntico al `Containerfile` (Docker solo busca `Dockerfile`) y README |

Recuperó el modelo del registry, construyó la imagen con Docker Desktop, levantó el contenedor y probó
la API en su computadora con `scripts\ejecutar_servicio.ps1 -Motor docker -Exportar -Capturas`; corrió
además todas las pruebas, los chequeos de seguridad y la construcción desde un clon limpio. Los commits
incluyen la línea `Co-Authored-By: Claude` porque se usó Claude (Anthropic) como asistente de programación.

## Entrega 5: monitoreo de drift con Evidently y A/B testing (versión individual)

Misma mecánica que las entregas 3 y 4: versión de **Leydy Johanna Camargo Medina** (`JohannaCamargo2805`),
en `main` de este repositorio. Ramas y commits (todos a su nombre), integradas con `git merge --no-ff`:

| Rama | Commits | Qué contiene |
|---|---|---|
| `feat/monitoreo-evidently` | 4 | Sección *Entrega 5* de `mibici/config.py` (periodos, feriados 2026, umbrales y reglas del A/B), `evidently` en las dependencias, `mibici/monitoreo.py` (ingesta 2026, conjuntos de referencia y actual, cuatro reportes de Evidently, efecto del drift en la v1 y corrida en MLflow), `MLproject` y `make monitoreo` |
| `feat/reentrenamiento-ab-testing` | 3 | `mibici/modeling/reentrenamiento.py` (v3 con datos hasta may 2026, compuerta en jun 2026, registro como `@challenger`), `mibici/modeling/ab_testing.py` (A/B pareado, bootstrap por días, Wilcoxon, segmentos, simulación en línea y reglas de decisión), `MLproject` y `make` |
| `test/monitoreo-ab` | 2 | `tests/test_monitoreo_ab.py` (13 pruebas) y reglas del escáner para los datos de 2026 |
| `docs/reporte-drift-ab` | 6 | Reglas revisadas del escáner para los HTML de Evidently, evidencia de la ejecución (`reports/monitoreo/`, `docs/evidencia/entrega5/`), README y este archivo |

Ejecutó en su computadora la ingesta de 2026, los reportes de Evidently, el reentrenamiento y registro de
la v3 en el Model Registry, el A/B testing y las capturas de MLflow (`ENTREGA5_1_ejecutar.cmd`), y redactó el
reporte `TC5061_Drift_ABTest_Equipo32.pdf`. Los commits incluyen la línea `Co-Authored-By: Claude` porque se
usó Claude (Anthropic) como asistente de programación.

## Entrega 6: microservicios, CI/CD y FinOps (versión individual)

Misma mecánica que las entregas 3 a 5: versión de **Leydy Johanna Camargo Medina** (`JohannaCamargo2805`),
en `main` de este repositorio. Ramas y commits (todos a su nombre), integradas con `git merge --no-ff`:

| Rama | Commits | Qué contiene |
|---|---|---|
| `feat/prediccion-asincrona` | 4 | Verificación de huellas SHA-256 también al cargar desde el registry, `POST /predict/asincrono` y `GET /tareas/{id}` con Redis y RQ, `api/worker.py` y pruebas con fakeredis |
| `feat/compose-orquestacion` | 2 | Imagen del servidor de MLflow 3.17.2, registro idempotente del modelo y `docker-compose.yml` (MLflow, registro, API, Redis y worker; perfil de Locust) |
| `feat/prueba-carga-finops` | 4 | Escenario de Locust con peticiones realistas, `scripts/prueba_carga.py` con `docker stats`, gráficas por escalón, `finops/costo_inferencia.py`, `scripts/medir_reposo.py` y `tests/test_entrega6.py` |
| `ci/github-actions` | 3 | `.github/workflows/ci-cd.yml` (calidad, pruebas en matriz, imágenes con caché y GHCR, sistema completo y despliegue), `bandit` sobre los módulos nuevos y corrección de DeepChecks con pandas 3 |
| `perf/motor-numpy` | 3 | `api/acelerado.py` (motor en NumPy verificado contra el Pipeline), `scripts/perfil_inferencia.py`, costo con el motor optimizado y medición de MLflow de fábrica |
| `fix/carga-ventanas`, `fix/perfil-inferencia`, `fix/finops-eficiencia` | 3 | Correcciones del análisis encontradas al revisar los resultados |
| `feat/finops-motor-numpy`, `feat/carga-comparacion`, `feat/grafica-desempeno-compacta` | 3 | Escenarios con el motor medido y gráficas para el reporte |
| `docs/operacion-finops` | 3 | Resultados de las pruebas (`reports/carga/`, `reports/finops/`), evidencia (`docs/evidencia/entrega6/`), regla del escáner de secretos para las tablas de resultados, README y este archivo |
| `fix/titulo-endpoints` | 1 | Título de la gráfica por endpoint con el número de usuarios entero (gráficas regeneradas) |
| `docs/evidencia-ci` | 1 | Capturas de la ejecución #1 de GitHub Actions en verde y de la imagen publicada en GHCR, README y este archivo |

Levantó el sistema con Docker Compose y corrió en su computadora las cinco pruebas de carga (base, escalado,
optimizado, lote y asíncrona), las mediciones en reposo y el perfil de la inferencia
(`ENTREGA6_1_ejecutar.cmd` a `ENTREGA6_2c_mlflow_fabrica.cmd`), publicó el repositorio, con lo que el pipeline de
GitHub Actions corrió en verde en `main` y en las ramas (ejecución #1), y redactó el reporte `TC5061_Operacion_FinOps_Equipo32.pdf`. Los commits incluyen la línea
`Co-Authored-By: Claude` porque se usó Claude (Anthropic) como asistente de programación.
