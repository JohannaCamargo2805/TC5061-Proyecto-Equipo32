# Resumen de los chequeos de seguridad

Generado por `python scripts/chequeos_seguridad.py` el 2026-11-01 23:29 UTC (Python 3.14.8, Windows-11-10.0.26200-SP0).

| Chequeo | Herramienta | Resultado | Estado |
|---|---|---|---|
| Código (análisis estático) | bandit | 0 hallazgos en 4,800 líneas (alta 0, media 0, baja 0); 5 revisados con `# nosec` | sin hallazgos |
| Dependencias fijadas | pip-audit -r requirements.txt | 25 paquetes, 0 vulnerabilidades | sin hallazgos |
| Entorno instalado (incluye transitivas) | pip-audit | 206 paquetes, 0 vulnerabilidades | sin hallazgos |
| Secretos en árbol e historial | detect-secrets | 167 versiones de archivo, 0 secretos reales | sin hallazgos |
| Archivos sensibles fuera de Git | git check-ignore | 0 problemas | sin hallazgos |


Coincidencias de detect-secrets revisadas (no son secretos):

- 14 × hash SHA-256 o data_version del manifiesto de datos
- 6 × imagen PNG en base64 dentro de la salida de un notebook
- 2 × vector de prueba de SHA-256 (cadenas 'abc' y vacía)
