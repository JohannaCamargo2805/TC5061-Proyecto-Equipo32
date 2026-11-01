# Resumen de la validación de datos con DeepChecks

Generado por `python -m mibici.validacion`. Estados: **aprobado** (todas sus condiciones se cumplen), **no aprobado** (alguna falla), **advertencia** (DeepChecks marca el hallazgo como aviso), **informativo** (chequeo sin condición, solo describe) o **error**.

## viajes — Viajes procesados por la ingesta (data/processed/viajes.parquet); reglas de negocio sobre todos los viajes y chequeos estadísticos sobre una muestra de 300,000 viajes por conjunto

Filas: entrenamiento 3,816,284 · prueba 715,748

### integridad_train (`viajes_integridad_train.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Reglas de negocio MiBici | no aprobado | Las reglas críticas no tienen violaciones: Reglas violadas: {'viaje repetido (mismo origen, destino, inicio y fin)': 898} |
| Data Duplicates | advertencia | Duplicate data ratio is less or equal to 0%: Found 0.0017% duplicate data |
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 5 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 9 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 9 columns passed: found 0 columns with negligible types mix, and 9 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 9 relevant columns |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 9 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [] |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### integridad_test (`viajes_integridad_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Reglas de negocio MiBici | no aprobado | Las reglas críticas no tienen violaciones: Reglas violadas: {'viaje repetido (mismo origen, destino, inicio y fin)': 178} |
| Data Duplicates | advertencia | Duplicate data ratio is less or equal to 0%: Found 0.01% duplicate data |
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 5 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 9 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 9 columns passed: found 0 columns with negligible types mix, and 9 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 9 relevant columns |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 9 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [] |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### train_test (`viajes_train_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Datasets Size Comparison | aprobado | Test dataset size is greater or equal to 100: Test dataset contains 300000 samples |
| Feature Drift | aprobado | categorical drift score < 0.2 and numerical drift score < 0.2: Passed for 5 columns out of 5 columns. Found column "dia_semana" has the highest categorical drift score: 0.04 Found column "hora_inicio" has the highest numerical drift score: 0.01 |
| Multivariate Drift | aprobado | Drift value is less than 0.25: Found drift value of: 0.04, corresponding to a domain classifier AUC of: 0.52 |
| Date Train-Test Leakage (overlap) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Date Train-Test Leakage (duplicates) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Index Train-Test Leakage | aprobado | Ratio of leaking indices is less or equal to 0%: No index leakage found |
| New Category Train Test | informativo | Columnas con categorías nuevas en prueba: ['origen_id', 'destino_id'] |
| Train Test Samples Mix | informativo | Muestras de prueba que aparecen en entrenamiento: 0.22% |

## p1 — P1 · tabla estación × día (TRAIN ene-sep contra TEST nov-dic)

Filas: entrenamiento 90,230 · prueba 22,565

### integridad_train (`p1_integridad_train.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Feature-Feature Correlation | no aprobado | Not more than 5 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('lag_1', 'media_7d'), ('lag_1', 'media_28d'), ('lag_2', 'media_7d'), ('lag_2', 'media_28d'), ('lag_7', 'media_7d'), ('lag_7', 'media_28d'), ('lag_7', 'media_mismo_dia_4sem'), ('lag_14', 'lag_7'), ('lag_14', 'media_7d'), ('lag_14', 'media_28d'), ('lag_14', 'media_mismo_dia_4sem'), ('media_7d', 'media_mismo_d |
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 13 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 15 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 15 columns passed: found 0 columns with negligible types mix, and 15 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 15 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 15 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 6 reglas críticas sin violaciones |
| Feature Label Correlation | aprobado | Features' Predictive Power Score is less than 0.9: Passed for 13 relevant columns |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### integridad_test (`p1_integridad_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Feature-Feature Correlation | no aprobado | Not more than 5 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('lag_1', 'media_7d'), ('lag_1', 'media_28d'), ('lag_2', 'media_7d'), ('lag_2', 'media_28d'), ('lag_7', 'media_7d'), ('lag_7', 'media_28d'), ('lag_7', 'media_mismo_dia_4sem'), ('lag_14', 'lag_7'), ('lag_14', 'media_28d'), ('lag_14', 'media_mismo_dia_4sem'), ('media_7d', 'media_mismo_dia_4sem'), ('media_28d', |
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 13 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 15 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 15 columns passed: found 0 columns with negligible types mix, and 15 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 15 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 15 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 6 reglas críticas sin violaciones |
| Feature Label Correlation | aprobado | Features' Predictive Power Score is less than 0.9: Passed for 13 relevant columns |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### train_test (`p1_train_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Feature Drift | no aprobado | categorical drift score < 0.2 and numerical drift score < 0.2: Failed for 1 out of 13 columns. Found 1 numeric columns with Kolmogorov-Smirnov above threshold: {'dias_activa': '0.99'} |
| Multivariate Drift | no aprobado | Drift value is less than 0.25: Found drift value of: 1, corresponding to a domain classifier AUC of: 1 |
| Datasets Size Comparison | aprobado | Test dataset size is greater or equal to 100: Test dataset contains 22565 samples |
| Date Train-Test Leakage (overlap) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Date Train-Test Leakage (duplicates) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Label Drift | aprobado | Label drift score < 0.2: Label's drift score Kolmogorov-Smirnov is 0.04 |
| New Category Train Test | informativo | Columnas con categorías nuevas en prueba: ['estacion'] |
| Train Test Samples Mix | informativo | Muestras de prueba que aparecen en entrenamiento: 0.00% |
| Feature Label Correlation Change | informativo | {"train": {"media_mismo_dia_4sem": 0.6919572864660195, "lag_7": 0.6528148812293175, "lag_14": 0.6487656899808292, "estacion": 0.5936447517158643, "media_7d": 0.5655082479608995, "lag_1": 0.5569665185277356, "media_28d": 0.506803306174321, "lag_2": 0.4320600082054459, "dia_semana": 0.0, "fin_semana":… |

### train_test_sin_tiempo (`p1_train_test_sin_tiempo.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Feature Drift | aprobado | categorical drift score < 0.2 and numerical drift score < 0.2: Passed for 12 columns out of 12 columns. Found column "feriado" has the highest categorical drift score: 0.02 Found column "lag_1" has the highest numerical drift score: 0.04 |
| Multivariate Drift | aprobado | Drift value is less than 0.25: Found drift value of: 3.18E-4, corresponding to a domain classifier AUC of: 0.5 |

## p2 — P2 · tabla estación × hora (TRAIN ene-sep contra TEST nov-dic; muestra de 200,000 filas por conjunto)

Filas: entrenamiento 200,000 · prueba 200,000

### integridad_train (`p2_integridad_train.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 15 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 18 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 18 columns passed: found 0 columns with negligible types mix, and 18 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 18 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 18 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [('neto_3h', 'neto_3h_rel')] |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 6 reglas críticas sin violaciones |
| Feature Label Correlation | aprobado | Features' Predictive Power Score is less than 0.9: Passed for 15 relevant columns |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |
| Class Imbalance | informativo | Proporción por clase: 0: 96.00%, 1: 4.00% |
| Conflicting Labels | informativo | Muestras con etiquetas en conflicto: 0.00% |

### integridad_test (`p2_integridad_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 15 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 18 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 18 columns passed: found 0 columns with negligible types mix, and 18 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 18 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 18 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [('neto_3h', 'neto_3h_rel')] |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 6 reglas críticas sin violaciones |
| Feature Label Correlation | aprobado | Features' Predictive Power Score is less than 0.9: Passed for 15 relevant columns |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |
| Class Imbalance | informativo | Proporción por clase: 0: 97.00%, 1: 3.00% |
| Conflicting Labels | informativo | Muestras con etiquetas en conflicto: 0.01% |

### train_test (`p2_train_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Datasets Size Comparison | aprobado | Test dataset size is greater or equal to 100: Test dataset contains 200000 samples |
| Feature Drift | aprobado | categorical drift score < 0.2 and numerical drift score < 0.2: Passed for 15 columns out of 15 columns. Found column "dia_semana" has the highest categorical drift score: 0.03 Found column "tasa_riesgo_hist" has the highest numerical drift score: 0.07 |
| Multivariate Drift | aprobado | Drift value is less than 0.25: Found drift value of: 0.13, corresponding to a domain classifier AUC of: 0.57 |
| Date Train-Test Leakage (overlap) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Date Train-Test Leakage (duplicates) | aprobado | Date leakage ratio is less or equal to 0%: No leaked dates found |
| Label Drift | aprobado | Label drift score < 0.2: Label's drift score Cramer's V is 0.02 |
| New Label Train Test | aprobado | Number of new label values is less or equal to 0: Found 0 new labels in test data: [] |
| New Category Train Test | informativo | Columnas con categorías nuevas en prueba: ninguna |
| Train Test Samples Mix | informativo | Muestras de prueba que aparecen en entrenamiento: 9.62% |
| Feature Label Correlation Change | informativo | {"train": {"tasa_riesgo_hist": 0.09291480986547047, "neto_futuro_medio_hist": 0.044622427195699185, "neto_h1": 0.01246598924338325, "neto_3h": 0.005643745888651105, "hora_dia": 5.425807220965487e-07, "dia_semana": 5.425807220965487e-07, "fin_semana": 5.425807220965487e-07, "neto_h0": 5.4258072209654… |

## p3 — P3 · perfiles por estación (TRAIN 1 ene-19 oct contra TEST 20 oct-31 dic)

Filas: entrenamiento 367 · prueba 367

### integridad_train (`p3_integridad_train.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 12 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 12 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 12 columns passed: found 0 columns with negligible types mix, and 12 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 12 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 12 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [('llegadas_dia', 'salidas_dia')] |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 7 reglas críticas sin violaciones |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### integridad_test (`p3_integridad_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| PercentOfNulls | aprobado | Percent of null values in each column is not greater than 0%: Passed for 12 relevant columns |
| Mixed Nulls | aprobado | Number of different null types is less or equal to 1: Passed for 12 relevant columns |
| Mixed Data Types | aprobado | Rare data types in column are either more than 50% or less than 0% of the data: 12 columns passed: found 0 columns with negligible types mix, and 12 columns without any types mix |
| Single Value in Column | aprobado | Does not contain only a single value: Passed for 12 relevant columns |
| Data Duplicates | aprobado | Duplicate data ratio is less or equal to 0%: Found 0% duplicate data |
| Special Characters | aprobado | Ratio of samples containing solely special character is less or equal to 0.1%: Passed for 12 relevant columns |
| String Mismatch | aprobado | No string variants: Passed for 1 relevant column |
| Feature-Feature Correlation | aprobado | Not more than 5 pairs are correlated above 0.9: All correlations are less than 0.9 except pairs [('llegadas_dia', 'salidas_dia')] |
| Reglas de negocio MiBici | aprobado | Las reglas críticas no tienen violaciones: 7 reglas críticas sin violaciones |
| Outlier Sample Detection | informativo | Se listan las 5 muestras más atípicas (ver reporte HTML) |

### train_test (`p3_train_test.html`)

| Chequeo | Estado | Detalle |
|---|---|---|
| Datasets Size Comparison | aprobado | Test dataset size is greater or equal to 100: Test dataset contains 367 samples |
| Feature Drift | aprobado | categorical drift score < 0.2 and numerical drift score < 0.2: Passed for 12 columns out of 12 columns. Found column "salida_mediodia_share" has the highest numerical drift score: 0.11 |
| Multivariate Drift | aprobado | Drift value is less than 0.25: Found drift value of: 0.14, corresponding to a domain classifier AUC of: 0.57 |
| New Category Train Test | informativo | Columnas con categorías nuevas en prueba: ninguna |
| Train Test Samples Mix | informativo | Muestras de prueba que aparecen en entrenamiento: 0.00% |
