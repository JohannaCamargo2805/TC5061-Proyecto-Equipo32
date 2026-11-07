# API REST de MiBici P1 en contenedor (entrega 4, Equipo 32): recupera el modelo del Model
# Registry, construye la imagen, levanta el contenedor y prueba la API con peticiones reales.
#
# Uso (PowerShell, desde la raiz del repositorio):
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_servicio.ps1                  # podman si existe; si no, docker
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_servicio.ps1 -Motor docker
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_servicio.ps1 -Exportar -Capturas
#
# -Exportar : antes de construir descarga del Model Registry la version @champion a
#             models\servicio (scripts\exportar_modelo_registry.py). Necesita el servidor de
#             MLflow levantado (scripts\servidor_mlflow.ps1) en MLFLOW_TRACKING_URI.
# -Exportar solo funciona donde esta el registry del proyecto (sus artefactos son rutas locales);
#             sin -Exportar se usa el modelo que ya esta versionado en models\servicio.
# -Capturas : modo evidencia: capturas en docs\evidencia\entrega4\ y salida en reports\servicio\
#             (reemplaza la evidencia versionada). Sin -Capturas la salida va a %TEMP%\mibici_servicio.
# El contenedor queda corriendo al final (docs en http://localhost:8000/docs).
# Solo ASCII: Windows PowerShell 5.1 lee los .ps1 como ANSI.
param([string]$Motor = "", [switch]$Exportar, [switch]$Capturas, [int]$Puerto = 8000)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$venv = Join-Path $repo ".venv\Scripts"
if (Test-Path $venv) { $env:PATH = "$venv;$env:PATH"; $env:VIRTUAL_ENV = (Split-Path $venv -Parent) }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:MLFLOW_DISABLE_AGENT_HINT = "1"
if (-not $env:MLFLOW_TRACKING_URI) { $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" }
if (-not $Motor) {
    if (Get-Command podman -ErrorAction SilentlyContinue) { $Motor = "podman" } else { $Motor = "docker" }
}
$contenedor = "mibici-p1-api"
$url = "http://localhost:$Puerto"
$logs = Join-Path $env:TEMP "mibici_servicio"; $logsTexto = $logs
if ($Capturas) { $logs = Join-Path $repo "reports\servicio"; $logsTexto = "reports\servicio" }
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$evidencia = Join-Path $repo "docs\evidencia\entrega4"
if ($Capturas) {
    New-Item -ItemType Directory -Force -Path $evidencia | Out-Null
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class Ventana { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n); [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f); }'
    [Ventana]::SetProcessDPIAware() | Out-Null
    # Consola maximizada y siempre al frente para que las capturas la muestren completa.
    $h = [Ventana]::GetConsoleWindow()
    try { $raw = $Host.UI.RawUI; $b = $raw.BufferSize; $b.Width = 200; $b.Height = 5000; $raw.BufferSize = $b } catch {}
    [Ventana]::ShowWindow($h, 3) | Out-Null
    [Ventana]::SetWindowPos($h, [IntPtr](-1), 0, 0, 0, 0, 3) | Out-Null
}

function Captura([string]$nombre) {
    if (-not $Capturas) { return }
    Start-Sleep -Seconds 2
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
    $bmp.Save((Join-Path $evidencia $nombre), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}

function Encabezado([string]$titulo) {
    Clear-Host
    $fecha = Get-Date -Format "yyyy-MM-dd HH:mm"
    Write-Host "TC5061 Equipo 32 | Entrega 4 | $titulo | $fecha | $env:COMPUTERNAME | motor: $Motor" -ForegroundColor Cyan
}

# Ejecuta un comando, lo muestra y guarda su salida; cmd /c une stdout y stderr (PS 5.1).
function Ejecutar([string]$comando, [string]$log, [switch]$Agregar) {
    Write-Host "> $comando" -ForegroundColor Yellow
    $salida = cmd /c "$comando 2>&1"
    $codigo = $LASTEXITCODE
    $salida | ForEach-Object { Write-Host $_ }
    if ($Agregar) { $salida | Out-File -FilePath $log -Encoding utf8 -Append }
    else { $salida | Out-File -FilePath $log -Encoding utf8 }
    return $codigo
}

function Final([int]$codigo, [string]$log) {
    Write-Host ""
    $color = "Green"; if ($codigo -ne 0) { $color = "Red" }
    Write-Host "Codigo de salida: $codigo   (salida completa en $logsTexto\$log)" -ForegroundColor $color
}

# POST/GET con curl.exe: muestra la peticion, el estado HTTP, las cabeceras de trazabilidad y el JSON.
function Peticion([string]$metodo, [string]$ruta, [string]$cuerpo, [string]$id) {
    $respuesta = Join-Path $logs "$id.json"
    $cabeceras = Join-Path $logs "$id.headers.txt"
    $comando = "curl.exe -s -X $metodo $url$ruta"
    if ($cuerpo) { $comando += " -H `"Content-Type: application/json`" --data-binary `"@$cuerpo`"" }
    Write-Host "> $comando" -ForegroundColor Yellow
    if ($cuerpo) {
        & curl.exe -s -X $metodo "$url$ruta" -H "Content-Type: application/json" --data-binary "@$cuerpo" -D $cabeceras -o $respuesta
    } else {
        & curl.exe -s -X $metodo "$url$ruta" -D $cabeceras -o $respuesta
    }
    $lineas = Get-Content $cabeceras | Where-Object { $_ -match "^(HTTP/|x-modelo|x-tiempo)" }
    $lineas | ForEach-Object { Write-Host $_ -ForegroundColor Green }
    $estado = [int](($lineas | Select-Object -First 1) -split " ")[1]
    $json = cmd /c "python -m json.tool --no-ensure-ascii `"$respuesta`" 2>&1"
    $json | ForEach-Object { Write-Host $_ }
    Set-Content -Path $respuesta -Value $json -Encoding utf8
    return $estado
}

$git = (git log -1 --format="%h %s" 2>$null)
$resultados = @()

# ---------- 1. Recuperar el modelo del Model Registry ----------
if ($Exportar) {
    Encabezado "1. Recuperar el modelo del Model Registry de MLflow"
    try { Invoke-WebRequest "$env:MLFLOW_TRACKING_URI/health" -UseBasicParsing -TimeoutSec 5 | Out-Null }
    catch {
        Write-Host "No responde el servidor de MLflow en $env:MLFLOW_TRACKING_URI." -ForegroundColor Red
        Write-Host "Levantalo en otra terminal: powershell -ExecutionPolicy Bypass -File scripts\servidor_mlflow.ps1"
        exit 1
    }
    Write-Host "MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI" -ForegroundColor Yellow
    $r = Ejecutar "python scripts\exportar_modelo_registry.py --nombre mibici-equipo32-p1-demanda --alias champion" (Join-Path $logs "01_registry.txt")
    Final $r "01_registry.txt"
    Captura "01_registry.png"
    $resultados += [pscustomobject]@{ Paso = "1. Modelo desde el registry"; Resultado = $r }
    if ($r -ne 0) { exit 1 }
}
$registro = Get-Content (Join-Path $repo "models\servicio\registro.json") -Raw | ConvertFrom-Json
$imagen = "mibici-p1-api:modelo-v$($registro.version)"

# ---------- 2. Construir la imagen ----------
Encabezado "2. Construir la imagen ($imagen)"
$null = Ejecutar "$Motor version" (Join-Path $logs "02_motor_version.txt")
Write-Host ""
$r = Ejecutar "$Motor build -f Containerfile --build-arg MODELO_VERSION=$($registro.version) -t $imagen ." (Join-Path $logs "02_build.txt")
if ($r -eq 0) { Write-Host ""; $null = Ejecutar "$Motor images $($imagen.Split(':')[0])" (Join-Path $logs "02_build.txt") -Agregar }
Final $r "02_build.txt"
Captura "02_build.png"
$resultados += [pscustomobject]@{ Paso = "2. $Motor build"; Resultado = $r }
if ($r -ne 0) { exit 1 }

# ---------- 3. Levantar el contenedor ----------
Encabezado "3. Levantar el contenedor y esperar /health"
cmd /c "$Motor rm -f $contenedor >nul 2>&1"
$r = Ejecutar "$Motor run -d --name $contenedor -p ${Puerto}:8000 $imagen" (Join-Path $logs "03_contenedor.txt")
$listo = $false
for ($i = 1; $i -le 60 -and $r -eq 0; $i++) {
    Start-Sleep -Seconds 2
    try {
        $salud = Invoke-RestMethod "$url/health" -TimeoutSec 3
        if ($salud.estado -eq "ok") { $listo = $true }
    } catch {}
    $estadoSalud = (cmd /c "$Motor inspect --format `"{{.State.Health.Status}}`" $contenedor 2>nul")
    if ($listo -and ($estadoSalud -eq "healthy" -or $Motor -eq "podman")) { break }
}
Write-Host ""
$null = Ejecutar "$Motor ps --filter name=$contenedor" (Join-Path $logs "03_contenedor.txt") -Agregar
Write-Host ""
$null = Ejecutar "$Motor logs $contenedor" (Join-Path $logs "03_contenedor.txt") -Agregar
if (-not $listo) { $r = 1 }
Final $r "03_contenedor.txt"
Captura "03_contenedor.png"
$resultados += [pscustomobject]@{ Paso = "3. $Motor run + /health"; Resultado = $r }
if ($r -ne 0) { exit 1 }

# ---------- 4. Peticion de prueba a /predict ----------
Encabezado "4. Peticion de prueba al contenedor: POST /predict"
$null = Ejecutar "$Motor ps --filter name=$contenedor --format `"table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}`"" (Join-Path $logs "04_predict.txt")
Write-Host ""
Write-Host "Cuerpo de la peticion (docs\ejemplos\peticion_predict.json):" -ForegroundColor Yellow
Get-Content (Join-Path $repo "docs\ejemplos\peticion_predict.json") | ForEach-Object { Write-Host $_ }
Write-Host ""
$s4 = Peticion "POST" "/predict" "docs/ejemplos/peticion_predict.json" "04_respuesta_predict"
Final ([int]($s4 -ne 200)) "04_respuesta_predict.json"
Captura "04_predict.png"
$resultados += [pscustomobject]@{ Paso = "4. POST /predict"; Resultado = "HTTP $s4" }

# ---------- 5. Lote e informacion del modelo ----------
Encabezado "5. POST /predict/lote (3 estaciones) y GET /modelo"
$s5 = Peticion "POST" "/predict/lote" "docs/ejemplos/peticion_lote.json" "05_respuesta_lote"
Write-Host ""
$s5b = Peticion "GET" "/health" "" "05_respuesta_health"
Final ([int](($s5 -ne 200) -or ($s5b -ne 200))) "05_respuesta_lote.json"
Captura "05_lote.png"
$resultados += [pscustomobject]@{ Paso = "5. POST /predict/lote"; Resultado = "HTTP $s5" }

Encabezado "6. GET /modelo: version del registry que sirve el contenedor"
$s6 = Peticion "GET" "/modelo" "" "06_respuesta_modelo"
Final ([int]($s6 -ne 200)) "06_respuesta_modelo.json"
Captura "06_modelo.png"
$resultados += [pscustomobject]@{ Paso = "6. GET /modelo"; Resultado = "HTTP $s6" }

# ---------- 7. Validacion: una entrada invalida se rechaza con 422 ----------
Encabezado "7. Validacion de entradas: peticion invalida -> 422"
Write-Host "docs\ejemplos\peticion_invalida.json: fin_semana=1 en martes y media_7d=5.0 (imposible con esos rezagos)" -ForegroundColor Yellow
Write-Host ""
$s7 = Peticion "POST" "/predict" "docs/ejemplos/peticion_invalida.json" "07_respuesta_invalida"
Final ([int]($s7 -ne 422)) "07_respuesta_invalida.json"
Captura "07_validacion.png"
$resultados += [pscustomobject]@{ Paso = "7. Entrada invalida"; Resultado = "HTTP $s7 (se espera 422)" }

# ---------- Resumen ----------
Encabezado "Resumen"
Write-Host "Commit: $git"
Write-Host "Imagen: $imagen   Contenedor: $contenedor   URL: $url   Documentacion: $url/docs"
Write-Host ""
$resultados | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$resumen = Join-Path $logs "00_resumen.txt"
"TC5061 Equipo 32 | Entrega 4 | Resumen | $(Get-Date -Format 'yyyy-MM-dd HH:mm') | $env:COMPUTERNAME | motor: $Motor" | Out-File $resumen -Encoding utf8
"Commit: $git" | Out-File $resumen -Encoding utf8 -Append
"Imagen: $imagen" | Out-File $resumen -Encoding utf8 -Append
$resultados | Format-Table -AutoSize | Out-String -Width 200 | Out-File $resumen -Encoding utf8 -Append
Write-Host "El contenedor sigue corriendo. Para detenerlo: $Motor rm -f $contenedor" -ForegroundColor Cyan
Captura "00_resumen.png"
if (($s4 -ne 200) -or ($s5 -ne 200) -or ($s6 -ne 200) -or ($s7 -ne 422)) { exit 1 }
exit 0
