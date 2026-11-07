# Resumen de los chequeos de seguridad

Generado por `python scripts/chequeos_seguridad.py` el 2026-11-07 00:31 UTC (Python 3.14.8, Windows-11-10.0.26200-SP0).

| Chequeo | Herramienta | Resultado | Estado |
|---|---|---|---|
| Código (análisis estático) | bandit | 0 hallazgos en 5,615 líneas (alta 0, media 0, baja 0); 5 revisados con `# nosec` | sin hallazgos |
| Dependencias fijadas | pip-audit -r requirements.txt | 30 paquetes, 0 vulnerabilidades | sin hallazgos |
| Imagen de la API (entrega 4) | pip-audit -r requirements-api.txt | 53 paquetes, 0 vulnerabilidades | sin hallazgos |
| Entorno instalado (incluye transitivas) | pip-audit | 209 paquetes, 0 vulnerabilidades | sin hallazgos |
| Secretos en árbol e historial | detect-secrets | 282 versiones de archivo, 0 secretos reales | sin hallazgos |
| Archivos sensibles fuera de Git | git check-ignore | 0 problemas | sin hallazgos |


Coincidencias de detect-secrets revisadas (no son secretos):

- 14 × hash SHA-256 o data_version del manifiesto de datos
- 54 × SHA-256 de archivos, commit o data_version en el registro del modelo empaquetado
- 6 × imagen PNG en base64 dentro de la salida de un notebook
- 7 × hash de commit o de blob de Git en el reporte del propio escáner
- 4 × commit, data_version o id de MLflow en las respuestas guardadas de la API (evidencia)
- 2 × vector de prueba de SHA-256 (cadenas 'abc' y vacía)
