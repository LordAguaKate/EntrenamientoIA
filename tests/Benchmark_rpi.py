"""
Benchmark físico en Raspberry Pi 5: mide latencia de inferencia, consumo de
CPU/RAM y estabilidad en ejecución continua del modelo .tflite ya desplegado.
Pensado para correrse EN el dispositivo, no en la máquina de desarrollo
(secciones 3.6.4, 3.6.5 y 3.6.6 de la tesina).

Uso en la Raspberry Pi:
    pip install tflite-runtime numpy pillow psutil
    python benchmark_rpi.py --images-dir /ruta/a/imagenes/de/prueba

Si no tienes tflite-runtime instalado (por ejemplo, para probar primero en
tu laptop), el script cae automáticamente a tensorflow.lite.

Salida: reporte impreso en consola + un JSON con el resumen
(benchmark_resultados.json) para pegar las cifras directamente en la tesina.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    RUNTIME_USADO = "tflite_runtime"
except ImportError:
    import tensorflow.lite as tflite
    RUNTIME_USADO = "tensorflow.lite"

try:
    import psutil
except ImportError:
    raise SystemExit(
        "Falta psutil. Instálalo con: pip install psutil"
    )

from PIL import Image


def cargar_imagenes(images_dir: Path, input_size, max_imagenes: int):
    """Carga y preprocesa hasta max_imagenes imágenes desde un directorio (recursivo)."""
    extensiones = {".jpg", ".jpeg", ".png"}
    rutas = [p for p in images_dir.rglob("*") if p.suffix.lower() in extensiones]
    if not rutas:
        raise FileNotFoundError(f"No se encontraron imágenes en {images_dir}")

    rutas = rutas[:max_imagenes]
    imagenes = []
    for ruta in rutas:
        img = Image.open(ruta).convert("RGB").resize(input_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        imagenes.append(np.expand_dims(arr, axis=0))
    return imagenes, [str(r) for r in rutas]


def benchmark_latencia(interpreter, imagenes, n_warmup: int):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencias_ms = []
    for i, img in enumerate(imagenes):
        img = img.astype(input_details[0]["dtype"])

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= n_warmup:  # se descartan las primeras como calentamiento
            latencias_ms.append(elapsed_ms)

    return latencias_ms


def benchmark_estabilidad(interpreter, imagenes, duracion_segundos: int, intervalo_muestreo: float):
    """Ejecuta inferencias en bucle durante 'duracion_segundos' y muestrea CPU/RAM."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    proceso = psutil.Process()
    muestras_cpu = []
    muestras_ram_mb = []

    t_fin = time.time() + duracion_segundos
    t_ultima_muestra = 0.0
    n_inferencias = 0
    idx = 0

    psutil.cpu_percent(interval=None)  # descarta la primera lectura (siempre 0.0)

    while time.time() < t_fin:
        img = imagenes[idx % len(imagenes)].astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        n_inferencias += 1
        idx += 1

        ahora = time.time()
        if ahora - t_ultima_muestra >= intervalo_muestreo:
            muestras_cpu.append(psutil.cpu_percent(interval=None))
            muestras_ram_mb.append(proceso.memory_info().rss / (1024 * 1024))
            t_ultima_muestra = ahora

    return {
        "n_inferencias": n_inferencias,
        "cpu_percent": muestras_cpu,
        "ram_mb": muestras_ram_mb,
    }


def resumen(valores):
    if not valores:
        return {}
    return {
        "media": round(statistics.mean(valores), 2),
        "std": round(statistics.pstdev(valores), 2) if len(valores) > 1 else 0.0,
        "min": round(min(valores), 2),
        "max": round(max(valores), 2),
        "p95": round(float(np.percentile(valores, 95)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark físico del modelo en Raspberry Pi 5")
    parser.add_argument("--model", type=str, default="modelo_residuos_rpi_V3_05.tflite")
    parser.add_argument("--images-dir", type=str, required=True,
                         help="Carpeta con imágenes de prueba (p. ej. tests/images o un subconjunto del test set)")
    parser.add_argument("--max-imagenes", type=int, default=200)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--duracion-estabilidad", type=int, default=300,
                         help="Segundos de ejecución continua para la prueba de estabilidad (default 5 min)")
    parser.add_argument("--intervalo-muestreo", type=float, default=1.0,
                         help="Segundos entre cada muestra de CPU/RAM durante la prueba de estabilidad")
    args = parser.parse_args()

    model_path = Path(args.model)
    images_dir = Path(args.images_dir)

    print(f"Runtime detectado: {RUNTIME_USADO}")
    print(f"Cargando modelo: {model_path}")
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_shape = interpreter.get_input_details()[0]["shape"]
    input_size = (int(input_shape[1]), int(input_shape[2]))

    print(f"Cargando imágenes de prueba desde: {images_dir}")
    imagenes, rutas = cargar_imagenes(images_dir, input_size, args.max_imagenes)
    print(f"  {len(imagenes)} imágenes cargadas.\n")

    print("--- BENCHMARK DE LATENCIA ---")
    latencias_ms = benchmark_latencia(interpreter, imagenes, args.n_warmup)
    resumen_latencia = resumen(latencias_ms)
    for k, v in resumen_latencia.items():
        print(f"  {k}: {v} ms")

    print(f"\n--- PRUEBA DE ESTABILIDAD ({args.duracion_estabilidad}s de ejecución continua) ---")
    resultado_estabilidad = benchmark_estabilidad(
        interpreter, imagenes, args.duracion_estabilidad, args.intervalo_muestreo
    )
    resumen_cpu = resumen(resultado_estabilidad["cpu_percent"])
    resumen_ram = resumen(resultado_estabilidad["ram_mb"])
    print(f"  Inferencias totales: {resultado_estabilidad['n_inferencias']}")
    print(f"  CPU %: {resumen_cpu}")
    print(f"  RAM MB: {resumen_ram}")

    reporte = {
        "dispositivo": platform.uname()._asdict(),
        "runtime": RUNTIME_USADO,
        "modelo": str(model_path),
        "n_imagenes_latencia": len(imagenes),
        "n_warmup_descartadas": args.n_warmup,
        "latencia_ms": resumen_latencia,
        "estabilidad": {
            "duracion_segundos": args.duracion_estabilidad,
            "n_inferencias": resultado_estabilidad["n_inferencias"],
            "cpu_percent": resumen_cpu,
            "ram_mb": resumen_ram,
        },
    }

    out_path = Path("benchmark_resultados.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {out_path.resolve()}")


if __name__ == "__main__":
    main()