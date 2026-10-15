"""
Evalúa el modelo cuantizado (.tflite) sobre el mismo test set usado para
evaluar el modelo .keras, con el fin de completar la comparación de
precisión antes/después de la cuantización (sección 3.5.3 de la tesina).

Uso (desde la raíz del repo, o copiando este archivo a tests/):
    python evaluate_tflite.py

Requiere: tensorflow, numpy, scikit-learn (ya están en requirements.txt)
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURACIÓN DE RUTAS ---
# Asume que este archivo vive en tests/, igual que test_camara_realtime.py.
# Si lo colocas en otro lugar, ajusta BASE_DIR manualmente.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw" / "waste_classification" / "test"

TFLITE_MODEL_PATH = MODELS_DIR / "modelo_residuos_rpi_V3_05.tflite"
IMG_SIZE = (224, 224)

# Valores de referencia del modelo .keras ya evaluado en
# evaluacion_modelo_MobileNetV3.ipynb, para poder imprimir la comparación
# antes/después sin tener que volver a correr ese notebook.
REFERENCE_KERAS_ACCURACY = 0.9652  # 96.52%
REFERENCE_KERAS_LOSS = 0.1284
REFERENCE_KERAS_SIZE_MB = 32.47
REFERENCE_TFLITE_SIZE_MB = 3.24


def cargar_test_set():
    """Genera el set de prueba en el mismo orden que evaluacion_modelo_MobileNetV3.ipynb."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontraron los datos en: {DATA_DIR}")

    datagen = ImageDataGenerator()
    test_gen = datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode="categorical",
        shuffle=False,  # importante: mantener el orden para comparar contra y_true
    )
    class_labels = list(test_gen.class_indices.keys())
    return test_gen, class_labels


def evaluar_tflite(model_path: Path, test_gen, num_samples: int):
    """Corre el intérprete de TFLite imagen por imagen y devuelve predicciones y latencias."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    y_true = []
    y_pred = []
    latencias_ms = []

    test_gen.reset()
    for i in range(num_samples):
        batch_x, batch_y = next(test_gen)
        img = batch_x.astype(input_dtype)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Se descarta la primera inferencia del promedio de latencia por el
        # overhead de "calentamiento" del intérprete (carga de kernels, cachés).
        if i > 0:
            latencias_ms.append(elapsed_ms)

        y_true.append(int(np.argmax(batch_y[0])))
        y_pred.append(int(np.argmax(output)))

        if (i + 1) % 500 == 0:
            print(f"  Procesadas {i + 1}/{num_samples} imágenes...")

    return np.array(y_true), np.array(y_pred), np.array(latencias_ms)


def main():
    if not TFLITE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {TFLITE_MODEL_PATH}. "
            "Verifica que hayas corrido la conversión en clasificacion_3.0.ipynb."
        )

    print(f"Cargando test set desde: {DATA_DIR}")
    test_gen, class_labels = cargar_test_set()
    num_samples = test_gen.samples
    print(f"Clases detectadas: {class_labels}")
    print(f"Total de imágenes de prueba: {num_samples}\n")

    print(f"Evaluando modelo cuantizado: {TFLITE_MODEL_PATH.name}")
    y_true, y_pred, latencias_ms = evaluar_tflite(TFLITE_MODEL_PATH, test_gen, num_samples)

    accuracy_tflite = float(np.mean(y_true == y_pred))

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN (.tflite)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

    print("MATRIZ DE CONFUSIÓN (.tflite)")
    print(confusion_matrix(y_true, y_pred))

    print("\n" + "=" * 60)
    print("COMPARACIÓN ANTES/DESPUÉS DE LA CUANTIZACIÓN")
    print("=" * 60)
    print(f"{'Métrica':<28}{'.keras (original)':<22}{'.tflite (cuantizado)':<22}")
    print(f"{'Tamaño (MB)':<28}{REFERENCE_KERAS_SIZE_MB:<22}{REFERENCE_TFLITE_SIZE_MB:<22}")
    print(
        f"{'Accuracy (test set)':<28}"
        f"{REFERENCE_KERAS_ACCURACY * 100:.2f}%{'':<15}"
        f"{accuracy_tflite * 100:.2f}%"
    )
    degradacion_pp = (REFERENCE_KERAS_ACCURACY - accuracy_tflite) * 100
    print(f"\nDegradación de precisión por cuantización: {degradacion_pp:.2f} puntos porcentuales")

    if latencias_ms.size > 0:
        print("\n" + "=" * 60)
        print("LATENCIA DE INFERENCIA (referencial, medida en esta máquina — NO es la")
        print("latencia final de la Raspberry Pi 5, ver benchmark_rpi.py para esa medición)")
        print("=" * 60)
        print(f"  Media : {latencias_ms.mean():.2f} ms")
        print(f"  Std   : {latencias_ms.std():.2f} ms")
        print(f"  Mín   : {latencias_ms.min():.2f} ms")
        print(f"  Máx   : {latencias_ms.max():.2f} ms")
        print(f"  P95   : {np.percentile(latencias_ms, 95):.2f} ms")


if __name__ == "__main__":
    main()