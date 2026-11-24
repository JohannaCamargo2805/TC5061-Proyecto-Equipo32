# Model Card · `mibici-equipo32-p1-demanda` (v3, en producción)

Ficha del modelo que sirve la API del proyecto integrador de TC5061 (Equipo 32). Sigue la estructura de
Mitchell et al. (2019), *Model Cards for Model Reporting*, adaptada al ciclo de MLOps del proyecto: cada dato
de esta ficha sale de una corrida de MLflow, de un reporte versionado en `reports/` o del código, y se indica
dónde verificarlo.

| | |
|---|---|
| **Modelo registrado** | `mibici-equipo32-p1-demanda` en el Model Registry de MLflow 3.17.2 |
| **Versión en producción** | **v3** (`@champion`), promovida el 24 de noviembre de 2026 por el A/B `p1-ab-test-v1-vs-v3` |
| **Respaldo (rollback)** | v1 (`@anterior`): `python -m mibici.modeling.promocion --accion rollback` |
| **Dueña** | Leydy Johanna Camargo Medina (A01840540) · responsable de P1 en el Equipo 32 |
| **Curso** | TC5061 Operaciones de aprendizaje automático · Tecnológico de Monterrey · Prof. Iván Reyes Amezcua |
| **Licencia del código** | MIT (`LICENSE`) · datos: portal de datos abiertos de MiBici |
| **Contacto** | Issues del repositorio `JohannaCamargo2805/TC5061-Proyecto-Equipo32` |

**Contenido:** [1. Detalles](#1-detalles-del-modelo) · [2. Uso previsto](#2-uso-previsto) · [3. Factores](#3-factores) ·
[4. Métricas](#4-métricas) · [5. Datos de evaluación](#5-datos-de-evaluación) · [6. Datos de entrenamiento](#6-datos-de-entrenamiento) ·
[7. Análisis cuantitativo](#7-análisis-cuantitativo) · [8. Consideraciones éticas](#8-consideraciones-éticas) ·
[9. Limitaciones y recomendaciones](#9-limitaciones-y-recomendaciones) · [10. Gobernanza y operación](#10-gobernanza-y-operación) ·
[11. Historial de versiones](#11-historial-de-versiones)

## 1. Detalles del modelo

* **Tarea (P1):** regresión de conteos. Predice cuántos viajes se iniciarán en una estación de MiBici (sistema de
  bicicleta pública del Área Metropolitana de Guadalajara) el **día siguiente** (horizonte de 1 día).
* **Algoritmo:** regresión de Poisson lineal con enlace logarítmico (`PoissonRegressor`, `alpha = 1e-3`,
  `max_iter = 1000`) dentro de un `Pipeline` de scikit-learn 1.9.1: `dia_semana` en *one-hot*; rezagos y medias
  con `log1p` + estandarización; el resto estandarizado (`mibici/modeling/p1_regresion.py::construir_lineal`).
  La salida se recorta a ≥ 0 y la API la redondea al entero más cercano.
* **Por qué este algoritmo:** en la entrega 2 se compararon un baseline ingenuo (t − 7), Poisson lineal, HGB con
  pérdida de Poisson (con búsqueda de hiperparámetros) y bosque aleatorio. La regla de **un error estándar**
  (`config.py::SELECCION`, calculada solo con VAL) eligió el Poisson lineal: está a menos de 1 EE del mejor y es
  el más simple (0.18 MB, 2 ms por 1,000 predicciones, coeficientes interpretables).
* **Entradas:** 12 variables numéricas conocidas al cierre del día anterior (firma del modelo en MLflow, idéntica
  al contrato de la API en `api/esquemas.py`):

  | Variable | Significado |
  |---|---|
  | `dia_semana`, `fin_semana`, `feriado` | Calendario del día que se predice (feriados de ley 2025–2026) |
  | `lag_1`, `lag_2`, `lag_7`, `lag_14` | Viajes iniciados en la estación hace 1, 2, 7 y 14 días |
  | `media_7d`, `media_28d`, `media_mismo_dia_4sem` | Promedios móviles de 7 y 28 días y del mismo día de la semana |
  | `dias_activa` | Días de historia de la estación (mínimo 28) |
  | `capacidad` | Anclajes instalados |

* **Salida:** `viajes_esperados` (media de Poisson, real ≥ 0) y `viajes_esperados_redondeado`.
* **Versión:** v3 = el mismo pipeline e hiperparámetros que la v1, **reentrenado con una ventana más reciente**
  (Continuous Training disparado por el drift de la entrega 5). Corrida `p1-reentreno-ventana-2026`
  (`1d685b6b91744c979f3f7ffab5220819`), commit `62b4cf0`, `data_version` `7aa5e436d887+5fb2c4834870`.
* **Artefacto:** modelo de MLflow con *flavor* `sklearn` serializado con **skops** (no `pickle`), firma de entrada,
  `input_example` y el código del paquete `mibici` en `code_paths`. La copia que se empaqueta en la imagen está en
  `models/servicio/` con el SHA-256 de cada archivo en `registro.json`; la API y el registro del Compose las
  verifican antes de cargarla.

## 2. Uso previsto

**Usuario principal:** el área de operación de MiBici, para planear la flota y los turnos de rebalanceo del día
siguiente (una predicción por estación cada noche, o bajo demanda desde una aplicación).

| Uso | ¿Previsto? | Comentario |
|---|---|---|
| Estimar la demanda de mañana por estación para priorizar el rebalanceo | **Sí** | El caso de diseño; MAE ≈ 5.7 viajes por estación-día |
| Sumar las predicciones para estimar la demanda total del sistema | **Sí** | Error del total diario ≈ 8 % (`mape_total_dia`) |
| Horizontes de más de un día | No | Las variables suponen conocer el día D − 1 |
| Estaciones con menos de 28 días de historia | No | La API las rechaza (`dias_activa ≥ 28`) |
| Decisiones sobre personas usuarias (precios, sanciones, perfiles) | **No** | El modelo no usa ni produce datos personales |
| Planeación de nuevas estaciones o expansión | No | Es la pregunta P3 (clustering), no P1 |
| Eventos extraordinarios (cierres viales, conciertos, contingencias) | Con cautela | No hay variables de eventos ni de clima |

## 3. Factores

Factores que cambian el desempeño y con los que se desagregó la evaluación (sección 7):

* **Tiempo:** mes y tipo de día (laborable, fin de semana, feriado); septiembre tiene la mayor demanda y el
  mayor error.
* **Cohorte de estación:** estaciones existentes en el TRAIN de la v1 frente a las **103 estaciones nuevas** que
  abrieron entre junio y septiembre de 2026 (crecimiento de la red de 369 a 472 estaciones).
* **Nivel de demanda:** cuartiles de viajes por estación; el error absoluto crece con el volumen (propio de los
  conteos de Poisson).
* **Antigüedad (`dias_activa`):** crece cada día; fuera del rango de TRAIN el modelo extrapola y la API lo advierte.

No se evaluó por zona geográfica ni por características socioeconómicas del entorno de cada estación (ver
sección 8).

## 4. Métricas

| Métrica | Por qué |
|---|---|
| **MAE** (viajes por estación-día) — métrica de selección | Error en las unidades del operador; robusta a picos |
| RMSE | Penaliza los errores grandes (días de mucha demanda) |
| WAPE | Error relativo ponderado por volumen, comparable entre periodos |
| R² | Varianza explicada |
| Sesgo (predicho − real) | Subestimar deja estaciones sin bicicletas; se vigila por separado |
| Error del total diario (`mape_total_dia`) | Calidad de la suma de todo el sistema |
| Devianza de Poisson | Ajuste probabilístico al conteo |

Umbrales de decisión fijados **antes** de ver resultados (`mibici/config.py`): compuerta de VAL (el modelo nuevo
debe tener menor MAE que el de producción) y cuatro reglas del A/B: IC 95 % de ΔMAE completamente bajo cero,
ningún mes peor por más de 2 %, |sesgo| no peor por más de 0.5 viajes y costo de inferencia menor a 2×.

## 5. Datos de evaluación

* **TEST de la v3 = periodo del A/B:** 1 de julio a 30 de septiembre de 2026, **40,512 estación-días**, 472
  estaciones, 92 días. Ninguna versión usó estos datos para entrenar ni para elegir hiperparámetros.
* **VAL (compuerta):** junio de 2026, 11,201 estación-días, 388 estaciones.
* Las variables se calculan con las mismas funciones del entrenamiento (`mibici/features.py`) sobre un panel
  continuo enero 2025 – septiembre 2026 (`mibici/monitoreo.py::construir_panel`), así no hay diferencia entre
  entrenamiento y servicio.

## 6. Datos de entrenamiento

* **Fuente:** datos abiertos de viajes de MiBici (<https://mibici.net/es/datos-abiertos/>): enero–diciembre de
  2025 (4,532,032 viajes, `data_version` `7aa5e436d887`) y enero–septiembre de 2026 (`5fb2c4834870`). Cada
  ingesta guarda un manifiesto con el SHA-256 de los archivos (`data/manifest.json`, `data/manifest_produccion.json`).
* **TRAIN de la v3:** 1 de enero 2025 – 31 de mayo 2026 (las primeras 4 semanas sirven solo para calcular rezagos),
  **180,254 estación-días** de 371 estaciones.
* **Preprocesamiento:** viajes de 60 s a 24 h; agregación por estación y día de inicio; las estaciones sin 28 días
  de historia se excluyen; capacidad del catálogo de estaciones.
* **Minimización de datos:** la ingesta **elimina el pseudónimo de usuario, el género y el año de nacimiento**
  antes de guardar los Parquet (`mibici/dataset.py::COLUMNAS_PERSONALES`); el modelo solo ve conteos agregados por
  estación-día. En MLflow se registra el esquema y la huella de cada conjunto (`mlflow.log_input`), nunca las filas.
* **Rango de TRAIN** (`data/rango_entrenamiento.json` de la corrida; viaja en `models/servicio/registro.json`):
  rezagos 0–550, `media_7d` 0–281.1, `media_28d` 0–263.5, `media_mismo_dia_4sem` 0–319, `dias_activa` 28–515,
  `capacidad` 11–39. Fuera de él la API responde con una advertencia de extrapolación.

## 7. Análisis cuantitativo

**Desempeño general en TEST (jul–sep 2026)**, A/B pareado sobre las mismas 40,512 estación-días
(`reports/monitoreo/ab_resultado.json`; corrida `p1-ab-test-v1-vs-v3`, `93d20255…`):

| Métrica | v1 (anterior) | **v3 (producción)** | v3 − v1 [IC 95 %] |
|---|---:|---:|---:|
| MAE | 5.873 | **5.657** | −0.215 [−0.316, −0.119] |
| RMSE | 10.783 | **10.374** | −0.409 [−0.589, −0.234] |
| WAPE | 19.73 % | **19.00 %** | −0.72 pp |
| R² | 0.8989 | **0.9064** | +0.0075 |
| Sesgo | −1.961 | **−1.056** | +0.905 |
| Error del total diario | 9.02 % | **8.00 %** | −1.02 pp |
| Costo de inferencia (ms por 1,000) | 1.99 | **1.92** | similar |

IC por *bootstrap* de días (2,000 réplicas); la v3 tiene menor MAE en 59 de 92 días (Wilcoxon, p = 0.0004).

**Desagregado** (`ab_segmentos.csv`, artefacto de la corrida del A/B):

| Factor | Segmento | Estación-días | MAE v1 | MAE v3 | Cambio | Sesgo v3 |
|---|---|---:|---:|---:|---:|---:|
| Mes | Julio 2026 | 12,624 | 5.577 | 5.568 | −0.1 % | +0.47 |
| | Agosto 2026 | 13,795 | 4.847 | 4.792 | −1.1 % | −0.20 |
| | Septiembre 2026 | 14,093 | 7.141 | 6.584 | −7.8 % | −3.27 |
| Tipo de día | Laborable | 28,593 | 5.966 | 5.697 | −4.5 % | −0.88 |
| | Fin de semana | 11,447 | 5.598 | 5.519 | −1.4 % | −1.37 |
| | Feriado | 472 | 6.868 | 6.611 | −3.8 % | −3.76 |
| Cohorte | Estación existente (369) | 33,948 | 6.410 | 6.162 | −3.9 % | −1.17 |
| | Estación nueva (103) | 6,564 | 3.094 | 3.047 | −1.5 % | −0.46 |
| Nivel de demanda | Q1 baja | 10,162 | 2.251 | 2.250 | −0.0 % | +0.12 |
| | Q2 | 10,096 | 3.756 | 3.682 | −2.0 % | −0.51 |
| | Q3 | 10,153 | 5.885 | 5.630 | −4.3 % | −1.32 |
| | Q4 alta | 10,101 | 11.620 | 11.087 | −4.6 % | −2.52 |

**Lectura:** la v3 no empeora ningún segmento. Persiste una **subestimación en septiembre** (sesgo −3.27; el
sistema creció ~29 % ese mes) y en las estaciones de mayor demanda (Q4, −2.52): son los casos donde el operador
debe agregar margen.

**Monitoreo de datos (Evidently 0.7.23, `reports/monitoreo/drift_datos.html`):** frente al TRAIN de la v1 hubo
drift en 10 de 14 columnas (71.4 %), explicado por la composición de la red (estaciones nuevas con menos viajes)
y por `dias_activa`, que crece con el tiempo; con solo las estaciones existentes únicamente `dias_activa` conserva
drift. Ese diagnóstico disparó el reentrenamiento que produjo la v3.

## 8. Consideraciones éticas

* **Datos personales:** el modelo no usa ni infiere datos de personas; la ingesta descarta los tres campos
  personales del portal y lo comprueban las pruebas (`tests/test_dataset.py`). El escaneo de secretos y de rutas
  sensibles corre en cada *push* (`scripts/escanear_secretos.py`).
* **Equidad territorial:** un modelo de demanda puede reforzar desigualdades si la operación solo atiende donde ya
  hay más viajes. Las estaciones nuevas y las de baja demanda tienen el menor error absoluto, pero la predicción
  **no debe** usarse para retirar servicio de zonas de baja demanda sin un análisis social aparte; no se evaluó el
  desempeño por zona socioeconómica.
* **Riesgo de la subestimación:** subestimar la demanda deja estaciones sin bicicletas. El sesgo se vigila como
  regla explícita del A/B y se reporta por segmento.
* **Transparencia:** el modelo es lineal en el espacio log y sus coeficientes se registran por versión
  (`ab_coeficientes.csv`); toda decisión de producción queda en MLflow con su evidencia.
* **Uso de IA en el desarrollo:** el código y la documentación se elaboraron con apoyo de Claude (Anthropic); los
  commits lo declaran con `Co-Authored-By`. Las ejecuciones se hicieron en la computadora de la autora.

## 9. Limitaciones y recomendaciones

* **Sin clima ni eventos:** la lluvia, los cierres viales y los eventos masivos no están en las variables; el
  error crece esos días.
* **`dias_activa` crece con el calendario:** el máximo de TRAIN de la v3 es 515 días y en jul–sep 2026 las
  estaciones existentes ya tienen entre 547 y 637, así que la API les advierte extrapolación en esa variable (es
  el drift estructural que detectó Evidently). El A/B muestra que la v3 aun así mejora a la v1 en esas
  estaciones (−3.9 % de MAE), pero la recomendación para el siguiente ciclo es acotar `dias_activa` (p. ej. a 365)
  o quitarla.
* **Estaciones nuevas:** necesitan 28 días de historia antes de recibir predicción.
* **Estaciones fuera del catálogo (hallazgo del proyecto final):** las estaciones 405 y 407 aparecen en los viajes
  de jul–sep 2026 pero no en la nomenclatura de estaciones de septiembre de 2026; el panel les imputa
  `capacidad = 0` y la API las rechaza con 422 (la capacidad debe ser ≥ 1). Son 171 de 40,512 estación-días
  (0.4 %); las métricas del A/B las incluyen. Se recomienda una regla de validación de datos para estaciones sin
  catálogo y actualizar la nomenclatura en cada ingesta (`tests/test_api.py` lo vigila).
* **Etiqueta con un día de retraso:** el desempeño real solo se conoce al día siguiente; se recomienda monitorear
  el MAE móvil de 7 días (disparador: +10 %) y el sesgo (|sesgo| > 1.5) además del drift de Evidently.
* **Registro de peticiones:** la API no guarda las peticiones; el monitoreo usa los datos abiertos. Para producción
  real conviene registrar entradas y predicciones (con retención limitada) y comparar contra lo observado.

## 10. Gobernanza y operación

| Aspecto | Cómo se cumple | Evidencia |
|---|---|---|
| **Linaje** | Cada versión guarda como etiquetas su corrida de origen, commit, `data_version`, punto de entrada del MLproject y métrica de selección; la corrida guarda datos, configuración, gráficas y modelo | MLflow: versión 3 del modelo y corrida `p1-reentreno-ventana-2026`; `models/servicio/registro.json` |
| **Estado de producción** | Aliases: `@champion` = producción, `@challenger` = retador, `@anterior` = respaldo; etiqueta `estado_produccion` en cada versión (produccion · retador · respaldo · archivada) | `docs/model_registry.md` |
| **Promoción controlada** | Solo con las cuatro compuertas de `mibici/modeling/promocion.py`; registra la corrida `p1-promocion-v1-a-v3` | `mlflow run . -e promote` |
| **Rollback** | Un paso: `@champion` regresa a `@anterior` | `--accion rollback` |
| **Integridad** | SHA-256 de cada archivo del modelo, verificado por la API y por el registro en Docker Compose | `registro.json` |
| **Despliegue** | Imagen `mibici-p1-api:modelo-v3` construida y probada por GitHub Actions y publicada en GHCR; el Compose la levanta con MLflow, Redis y worker | `.github/workflows/ci-cd.yml` |
| **Monitoreo** | Evidently (drift y calidad), A/B pareado, prueba de carga con Locust y consumo con `docker stats` | `reports/monitoreo/`, `reports/carga/` |
| **Costo** | USD 1.03 por millón de peticiones en Fargate x86 (USD 0.26 con el motor NumPy); lote diario de todas las estaciones: USD 0.03 al mes | `reports/finops/costos.json` |

## 11. Historial de versiones

| Versión | Corrida | Datos de TRAIN | MAE en su TEST | Estado |
|---|---|---|---|---|
| v1 | `p1-candidate-poisson` (entrega 2) | ene–sep 2025 · 90,230 estación-días | 5.410 (nov–dic 2025); 5.873 en jul–sep 2026 | `respaldo` (`@anterior`) |
| v2 | `p1-candidate-hgb-tuned` (entrega 2) | ene–sep 2025 | 5.904 (nov–dic 2025) | `archivada` |
| **v3** | `p1-reentreno-ventana-2026` (entrega 5) | ene 2025–may 2026 · 180,254 estación-días | **5.657** (jul–sep 2026) | **`produccion` (`@champion`)** |

**Referencia:** Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji,
I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness,
Accountability, and Transparency*, 220–229. <https://doi.org/10.1145/3287560.3287596>
