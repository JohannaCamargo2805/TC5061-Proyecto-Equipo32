"""Genera la narración del video con Microsoft Edge TTS (voz femenina es-MX-DaliaNeural).

Uso: python voz.py <carpeta_video> [+6%]
Lee narracion.json, escribe voz/<id>.mp3 y voz/duraciones.json (duración de cada segmento con ffprobe).
"""
import asyncio
import json
import pathlib
import subprocess
import sys

import edge_tts

VOZ = "es-MX-DaliaNeural"
base = pathlib.Path(sys.argv[1])
rate = sys.argv[2] if len(sys.argv) > 2 else "+6%"
segmentos = json.loads((base / "narracion.json").read_text(encoding="utf-8"))
salida = base / "voz"
salida.mkdir(exist_ok=True)


async def generar(seg):
    destino = salida / f"{seg['id']}.mp3"
    for intento in range(5):
        try:
            await edge_tts.Communicate(seg["texto"], VOZ, rate=rate).save(str(destino))
            print("ok", seg["id"], flush=True)
            return
        except Exception as e:  # red intermitente: reintenta
            print("reintento", seg["id"], intento, repr(e), flush=True)
            await asyncio.sleep(3)
    raise SystemExit(f"no se pudo generar {seg['id']}")


async def main():
    for seg in segmentos:
        await generar(seg)


asyncio.run(main())
duraciones = {}
for seg in segmentos:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(salida / f"{seg['id']}.mp3")],
        capture_output=True, text=True, check=True)
    duraciones[seg["id"]] = round(float(r.stdout.strip()), 3)
total = round(sum(duraciones.values()), 2)
(salida / "duraciones.json").write_text(
    json.dumps({"voz": VOZ, "rate": rate, "edge_tts": edge_tts.__version__, "duraciones": duraciones, "total": total},
               ensure_ascii=False, indent=1), encoding="utf-8")
print("TOTAL_SEGUNDOS", total)
