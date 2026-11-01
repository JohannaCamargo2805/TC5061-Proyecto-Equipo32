"""Llamadas a ``git`` para la trazabilidad de las corridas (commit, rama y árbol sucio).

Centraliza el único uso de ``subprocess`` del paquete para cumplir con bandit:

* B607 (ruta parcial): el ejecutable se resuelve con ``shutil.which`` a una ruta absoluta.
* B603 (sin shell): se pasa una lista de argumentos fijos escritos en el código, nunca
  texto del usuario, y sin ``shell=True``; por eso la advertencia se marca como revisada.
"""

from pathlib import Path
import shutil

# Revisado (bandit B404): subprocess solo se usa para ejecutar git con argumentos fijos.
import subprocess  # nosec B404

from mibici.config import PROJ_ROOT

GitError = subprocess.CalledProcessError


def ruta_git() -> str:
    """Ruta absoluta de ``git``; ``FileNotFoundError`` si no está instalado."""
    ruta = shutil.which("git")
    if ruta is None:
        raise FileNotFoundError("git no está instalado o no está en el PATH")
    return ruta


def git(*args: str, cwd: Path = PROJ_ROOT) -> str:
    """Ejecuta ``git <args>`` en el repositorio y devuelve la salida sin espacios finales."""
    # Revisado (bandit B603): ejecutable con ruta absoluta, sin shell y con argumentos fijos.
    return subprocess.check_output([ruta_git(), *args], cwd=cwd, text=True).strip()  # nosec B603
