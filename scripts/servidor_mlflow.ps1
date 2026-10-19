# Servidor local de MLflow Tracking respaldado por SQLite (requerido por el Model Registry).
# Uso (desde la raíz del repositorio):  powershell -ExecutionPolicy Bypass -File scripts\servidor_mlflow.ps1
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$mlflow = Join-Path $repo ".venv\Scripts\mlflow.exe"
if (-not (Test-Path $mlflow)) { $mlflow = "mlflow" }
Write-Host "Servidor MLflow en http://127.0.0.1:5000 (backend sqlite:///mlflow.db). No cierres esta ventana." -ForegroundColor Green
& $mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
