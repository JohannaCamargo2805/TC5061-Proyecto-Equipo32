# Model Registry de P1: linaje, estado de producción y ciclo de Continuous Training

Modelo registrado: **`mibici-equipo32-p1-demanda`** (MLflow 3.17.2, servidor local con SQLite,
experimento `mibici-equipo32`). Estado al 24 de noviembre de 2026, después de la promoción del proyecto final
(salida de `estado_registry` en la computadora de la autora y capturas en `docs/evidencia/final/`).

## 1. Esquema de ciclo de vida: aliases + etiqueta de estado

MLflow 2.9 dejó obsoletas las etapas `Staging`/`Production`; el proyecto usa **aliases** (punteros mutables a una
versión) y una etiqueta explícita por versión:

| Alias | Significado | `estado_produccion` | Quién lo mueve |
|---|---|---|---|
| `@champion` | La versión que sirve la API | `produccion` | `promote` (solo si pasan las compuertas) o `rollback` |
| `@challenger` | Candidata pendiente de A/B | `retador` | `register` / `retrain` |
| `@anterior` | La que estaba en producción antes de la última promoción | `respaldo` | `promote` |
| (sin alias) | Se conserva para auditoría y linaje | `archivada` | — |

La primera línea de la descripción de cada versión repite su estado (`[Estado: produccion · @champion]`) para que
se vea en la interfaz de MLflow, y el modelo registrado lleva `version_produccion` y `ultima_promocion`.
Los consumidores usan siempre `models:/mibici-equipo32-p1-demanda@champion`: promover o revertir es mover el alias,
sin cambiar código ni URI.

## 2. Versiones y linaje

| Versión | Alias / estado | Corrida de origen (run_id) | Commit | `data_version` | Modelo | val_mae | test_mae (su TEST) |
|---|---|---|---|---|---|---:|---:|
| **v3** | **`@champion` · produccion** | `p1-reentreno-ventana-2026` (`1d685b6b91744c979f3f7ffab5220819`) | `62b4cf0` | `7aa5e436d887+5fb2c4834870` | Poisson lineal | 7.271 (jun 2026) | **5.657** (jul–sep 2026) |
| v2 | archivada | `p1-candidate-hgb-tuned` (`41ce1a92611a4a23a31577308360f9e6`) | `e6c1a60` | `7aa5e436d887` | HGB Poisson | 5.129 | 5.904 (nov–dic 2025) |
| v1 | `@anterior` · respaldo | `p1-candidate-poisson` (`38260a3c12e84155b76d9c126b7be2e1`) | `e6c1a60` | `7aa5e436d887` | Poisson lineal | 5.104 | 5.410 (nov–dic 2025) |

Cada versión guarda como etiquetas su linaje (`experimento`, `source_run_id`, `source_run_name`, `git_commit`,
`git_rama`, `data_version`, `mlproject_entry_point`, `model_family`, `metrica_seleccion`) y su historia
(`ab_test_jul_sep_2026`, `promovida`, `retirada_de_produccion`, `promocion_run_id`). La corrida de origen guarda el
manifiesto de datos (SHA-256 de cada archivo de MiBici), los conjuntos con `mlflow.log_input`, la configuración
(`config.py`, `MLproject`, `python_env.yaml`, `requirements.txt`), las gráficas, la evaluación, el rango de TRAIN y
el modelo con firma. Cadena completa de la v3:

```
datos abiertos MiBici 2025 (7aa5e436d887) + 2026 (5fb2c4834870)
  └─ p1-monitoreo-drift (b6e0a185…)          Evidently: drift en 10 de 14 columnas → disparo
       └─ p1-reentreno-ventana-2026 (1d685b6b…)   commit 62b4cf0 · TRAIN ene 2025–may 2026 → v3 @challenger
            └─ p1-ab-test-v1-vs-v3 (93d20255…)     A/B pareado jul–sep 2026: gana v3 (4/4 reglas)
                 └─ p1-promocion-v1-a-v3 (95d2c591…)  compuertas 4/4 → v3 @champion, v1 @anterior
                      └─ models/servicio/registro.json   SHA-256 de 31 archivos → imagen mibici-p1-api:modelo-v3
```

## 3. Compuertas de la promoción (`mibici/modeling/promocion.py`)

Resultado real de `python -m mibici.modeling.promocion` (corrida `95d2c591faf546b6963c742fc26d3635`):

| Compuerta | Resultado |
|---|---|
| 1. El A/B más reciente decide promover B con sus cuatro reglas | sí · `p1-ab-test-v1-vs-v3`, ganadora B, reglas 4/4 |
| 2. A sigue siendo `@champion` (la decisión no es obsoleta) | sí · `@champion` = v1 y el A/B comparó v1 |
| 3. B está READY, es de P1 y tiene la firma de la API | sí · v3 READY, 12 variables de entrada |
| 4. B pasó su compuerta de validación | sí · MAE en VAL (jun 2026) 7.2710 contra 7.4668 del champion |

## 4. Operación

```bash
mlflow run . -e promote --env-manager local                    # promueve si pasan las compuertas
mlflow run . -e promote --env-manager local -P accion=simular  # solo muestra compuertas y plan
mlflow run . -e promote --env-manager local -P accion=rollback # @champion vuelve a @anterior
python scripts/exportar_modelo_registry.py --alias champion    # copia para la imagen (models/servicio)
make ciclo-ct                                                  # monitor → retrain → abtest → promote → copia
```

La imagen de la API empaqueta la versión exportada y la API verifica su SHA-256 al arrancar. En Docker Compose,
el servicio `registrar-modelo` vuelve a registrar esa copia en un MLflow limpio (que numera desde 1) con
`@champion` y la etiqueta `version_origen = 3`; por eso `/health` y `/modelo` responden `version` (la del registry
del Compose) y `version_origen` (la del registry del proyecto).
