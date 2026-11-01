# Pruebas, validacion de datos y chequeos de seguridad del proyecto MiBici (Equipo 32).
#
# Uso (PowerShell, desde la raiz del repositorio):
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_calidad.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_calidad.ps1 -Capturas
#
# Pasos: 1) pytest  2) DeepChecks (python -m mibici.validacion)  3) seguridad
# (python scripts\chequeos_seguridad.py). La salida completa de cada paso queda en
# reports\calidad\*.txt. Con -Capturas guarda una captura de pantalla al final de cada
# paso en docs\evidencia\ (evidencia para el README).
# Solo ASCII: Windows PowerShell 5.1 lee los .ps1 como ANSI.
param([switch]$Capturas)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$venv = Join-Path $repo ".venv\Scripts"
if (Test-Path $venv) { $env:PATH = "$venv;$env:PATH"; $env:VIRTUAL_ENV = (Split-Path $venv -Parent) }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:MLFLOW_DISABLE_AGENT_HINT = "1"
$logs = Join-Path $repo "reports\calidad"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$evidencia = Join-Path $repo "docs\evidencia"
if ($Capturas) {
    New-Item -ItemType Directory -Force -Path $evidencia | Out-Null
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class Dpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
    [Dpi]::SetProcessDPIAware() | Out-Null
}

function Captura([string]$nombre) {
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
    $bmp.Save((Join-Path $evidencia $nombre), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}

function Paso([string]$id, [string]$titulo, [string]$comando) {
    Clear-Host
    $fecha = Get-Date -Format "yyyy-MM-dd HH:mm"
    Write-Host "TC5061 Equipo 32 | $titulo | $fecha | $env:COMPUTERNAME" -ForegroundColor Cyan
    Write-Host "> $comando" -ForegroundColor Yellow
    $log = Join-Path $logs "$id.txt"
    # cmd /c une stdout y stderr antes de PowerShell (evita NativeCommandError en PS 5.1).
    $salida = cmd /c "$comando 2>&1"
    $codigo = $LASTEXITCODE
    $salida | ForEach-Object { Write-Host $_ }
    $salida | Out-File -FilePath $log -Encoding utf8
    Write-Host ""
    $color = "Green"; if ($codigo -ne 0) { $color = "Red" }
    Write-Host "Codigo de salida: $codigo   (salida completa en reports\calidad\$id.txt)" -ForegroundColor $color
    if ($Capturas) { Start-Sleep -Seconds 2; Captura "$id.png" }
    return $codigo
}

$git = (git log -1 --format="%h %s" 2>$null)
$r1 = Paso "01_pytest" "Pruebas unitarias, de reproducibilidad, de datos y de seguridad" "python -m pytest -v -p no:cacheprovider"
$r2 = Paso "02_deepchecks" "Validacion de datos con DeepChecks" "python -m mibici.validacion"
$r3 = Paso "03_seguridad" "Chequeos de seguridad (bandit, pip-audit, detect-secrets)" "python scripts\chequeos_seguridad.py"

Clear-Host
Write-Host "TC5061 Equipo 32 | Resumen de la ejecucion | $(Get-Date -Format 'yyyy-MM-dd HH:mm') | $env:COMPUTERNAME" -ForegroundColor Cyan
Write-Host "Commit: $git"
Write-Host ""
$resumen = @(
    [pscustomobject]@{ Paso = "1. pytest"; Codigo = $r1; Detalle = ((Get-Content (Join-Path $logs "01_pytest.txt") | Select-Object -Last 1)) },
    [pscustomobject]@{ Paso = "2. DeepChecks"; Codigo = $r2; Detalle = "reports\deepchecks\RESUMEN.md" },
    [pscustomobject]@{ Paso = "3. Seguridad"; Codigo = $r3; Detalle = ((Get-Content (Join-Path $logs "03_seguridad.txt") | Select-Object -Last 1)) }
)
$resumen | Format-Table -AutoSize -Wrap | Out-String -Width 200 | Write-Host
$archivoResumen = Join-Path $logs "00_resumen.txt"
"TC5061 Equipo 32 | Resumen de la ejecucion | $(Get-Date -Format 'yyyy-MM-dd HH:mm') | $env:COMPUTERNAME" | Out-File $archivoResumen -Encoding utf8
"Commit: $git" | Out-File $archivoResumen -Encoding utf8 -Append
$resumen | Format-Table -AutoSize -Wrap | Out-String -Width 200 | Out-File $archivoResumen -Encoding utf8 -Append
if ($Capturas) { Start-Sleep -Seconds 2; Captura "00_resumen.png" }
if (($r1 -ne 0) -or ($r3 -ne 0)) { exit 1 }
exit 0
