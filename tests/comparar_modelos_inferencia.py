"""
Genera una tabla comparativa de modelos ligeros ejecutando inferencias
en tiempo real sobre el test set completo (waste_classification/test).

A diferencia de comparar_modelos.py (que lee resultados de notebooks ya
ejecutados), este script carga cada modelo, predice las 4534 imágenes y
calcula las métricas desde cero. Útil cuando un notebook aún no fue
ejecutado o cuando se quiere verificar los resultados de forma independiente.

Modelos comparados:
    - MobileNetV3-Large          → modelo_residuos_rpi_V3_05.keras
    - EfficientNetB0             → modelo_residuos_EfficientNetB0_v1.keras
    - ShuffleNetV2 (desde cero)  → modelo_residuos_shufflenetV2_nativo.keras
    - ShuffleNetV2 (ImageNet)    → modelo_residuos_shufflenetV2_imagenet.keras

Los modelos cuyo archivo .keras no exista todavía se omiten.

Uso (desde la raíz del repo):
    python tests/comparar_modelos_inferencia.py

Requiere: tensorflow, numpy, scikit-learn, tabulate
    pip install tabulate   (opcional, para mejor formato)
"""

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from tabulate import tabulate
    TABULATE_DISPONIBLE = True
except ImportError:
    TABULATE_DISPONIBLE = False
    print("⚠️  'tabulate' no está instalada. Se usará formato de texto plano.")
    print("   Para instalarla: pip install tabulate\n")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------------------------------------------------
# Asume que este archivo vive en tests/, igual que Evaluate_tflite.py
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw" / "waste_classification" / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Clases del dataset (orden alfabético, igual que en los notebooks)
CLASES = ["aluminio", "carton", "otros", "plastico", "vidrio"]


# ---------------------------------------------------------------------------
# FUNCIÓN PERSONALIZADA DE SHUFFLENETV2
# ---------------------------------------------------------------------------
# ShuffleNetV2 fue construido con una operación matemática propia que no
# forma parte de las capas estándar de Keras. Hay que registrarla para que
# load_model() pueda reconstruir el modelo correctamente.
def channel_shuffle(x):
    """Reorganiza los canales de un tensor para mezclar información entre grupos."""
    _, h, w, c = x.shape.as_list()
    x = tf.reshape(x, [-1, h, w, 2, c // 2])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x


# ---------------------------------------------------------------------------
# DEFINICIÓN DE LOS MODELOS A COMPARAR
# ---------------------------------------------------------------------------
# Se define después de channel_shuffle para poder referenciarla directamente.
MODELOS = [
    {
        "nombre": "MobileNetV3-Large",
        "keras": MODELS_DIR / "modelo_residuos_rpi_V3_05.keras",
        "tflite": MODELS_DIR / "modelo_residuos_rpi_V3_05.tflite",
        "custom_objects": {},
    },
    {
        "nombre": "EfficientNetB0",
        "keras": MODELS_DIR / "modelo_residuos_EfficientNetB0_v1.keras",
        "tflite": MODELS_DIR / "modelo_residuos_EfficientNetB0_v1.tflite",
        "custom_objects": {},
    },
    {
        # Entrenado desde cero (ShuffleNetV2.ipynb), sin pesos de ImageNet.
        # Se conserva como referencia de cuánto aporta el transfer learning.
        "nombre": "ShuffleNetV2 (desde cero)",
        "keras": MODELS_DIR / "modelo_residuos_shufflenetV2_nativo.keras",
        "tflite": MODELS_DIR / "modelo_residuos_shufflenetV2_nativo.tflite",
        # Registramos la función personalizada para que Keras reconstruya el modelo
        "custom_objects": {"channel_shuffle": channel_shuffle},
    },
    {
        # Con pesos de ImageNet y el mismo protocolo de dos fases que los otros
        # modelos (clasificacion_ShuffleNetV2_preentrenado.ipynb).
        # No usa capas Lambda, así que no necesita custom_objects.
        "nombre": "ShuffleNetV2 (ImageNet)",
        "keras": MODELS_DIR / "modelo_residuos_shufflenetV2_imagenet.keras",
        "tflite": MODELS_DIR / "modelo_residuos_shufflenetV2_imagenet.tflite",
        "custom_objects": {},
    },
]


def construir_test_gen():
    """Crea el generador del set de prueba con shuffle=False para evaluación reproducible."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No se encontraron los datos en: {DATA_DIR}\n"
            "Asegúrate de que el dataset está en data/raw/waste_classification/test"
        )
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # ¡Crítico! Mantener el orden para que y_true empate con y_pred
    )
    return gen


def evaluar_modelo(config: dict, test_gen) -> dict:
    """Carga un modelo .keras, corre inferencias sobre test_gen y devuelve sus métricas."""
    nombre = config["nombre"]
    keras_path = config["keras"]
    tflite_path = config.get("tflite")
    custom_objects = config["custom_objects"]

    if not keras_path.exists():
        print(f"   ⚠️  No se encontró: {keras_path.name}. Se omite.")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluando: {nombre}")
    print(f"   Archivo: {keras_path.name}")
    print(f"{'='*60}")

    model = load_model(keras_path, custom_objects=custom_objects, safe_mode=False)

    # Evaluación estándar de Keras (loss y accuracy globales)
    # Se resetea el generador antes de cada uso para garantizar el mismo orden de imágenes
    test_gen.reset()
    t_inicio = time.perf_counter()
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    tiempo_evaluate_s = time.perf_counter() - t_inicio

    # Predicciones individuales para calcular métricas por clase
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(predictions, axis=1)  # Clase con mayor probabilidad
    y_true = test_gen.classes                # Etiquetas reales del generador

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASES, output_dict=True
    )

    errors = np.where(y_pred != y_true)[0]

    # Accuracy por clase calculada desde la diagonal de la matriz de confusión
    acc_por_clase = {}
    for i, cls in enumerate(CLASES):
        total_cls = np.sum(y_true == i)
        acc_por_clase[cls] = cm[i][i] / total_cls if total_cls > 0 else 0.0

    # Tamaños en disco
    size_keras_mb = keras_path.stat().st_size / (1024 * 1024)
    size_tflite_mb = (
        tflite_path.stat().st_size / (1024 * 1024)
        if tflite_path and tflite_path.exists()
        else None
    )

    # Latencia promedio estimada (tiempo total de evaluate / número de imágenes)
    latencia_ms = (tiempo_evaluate_s / len(y_true)) * 1000

    print(f"\n   Accuracy  : {accuracy*100:.2f}%")
    print(f"   Loss      : {loss:.4f}")
    print(f"   Errores   : {len(errors)} / {len(y_true)}")
    print(f"   Latencia* : {latencia_ms:.1f} ms/imagen  (*referencial, no en Raspberry Pi)")

    return {
        "nombre": nombre,
        "accuracy": accuracy,
        "loss": loss,
        "errores": len(errors),
        "total": len(y_true),
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "acc_por_clase": acc_por_clase,
        "size_keras_mb": size_keras_mb,
        "size_tflite_mb": size_tflite_mb,
        "latencia_ms": latencia_ms,
        "cm": cm,
        "report": report,
    }


# ---------------------------------------------------------------------------
# IMPRESIÓN DE TABLAS
# ---------------------------------------------------------------------------

def _imprimir_tabla(headers: list, filas: list):
    """Imprime con tabulate si está disponible, o en texto plano si no."""
    if TABULATE_DISPONIBLE:
        print(tabulate(filas, headers=headers, tablefmt="github"))
        return

    col_w = [
        max(len(str(h)), max(len(str(f[i])) for f in filas))
        for i, h in enumerate(headers)
    ]
    separador = "  ".join("-" * w for w in col_w)
    print("  ".join(str(h).ljust(col_w[i]) for i, h in enumerate(headers)))
    print(separador)
    for fila in filas:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(fila)))


def imprimir_tabla_comparativa(resultados: list):
    """Imprime la tabla principal, el desglose por clase y las matrices de confusión."""

    print("\n\n" + "=" * 70)
    print("TABLA COMPARATIVA DE MODELOS — TEST SET (waste_classification)")
    print(f"Total de imágenes: {resultados[0]['total']}  |  Clases: {len(CLASES)}")
    print("=" * 70)

    # --- Tabla principal de métricas globales ---
    headers_main = [
        "Modelo", "Accuracy", "Loss", "Errores",
        "F1 Macro", "Precisión Macro", "Recall Macro",
        "Tam .keras (MB)", "Tam .tflite (MB)", "Latencia* (ms/img)",
    ]
    filas_main = [
        [
            r["nombre"],
            f"{r['accuracy']*100:.2f}%",
            f"{r['loss']:.4f}",
            f"{r['errores']} / {r['total']}",
            f"{r['f1_macro']:.4f}",
            f"{r['precision_macro']:.4f}",
            f"{r['recall_macro']:.4f}",
            f"{r['size_keras_mb']:.2f}",
            f"{r['size_tflite_mb']:.2f}" if r["size_tflite_mb"] else "—",
            f"{r['latencia_ms']:.1f}",
        ]
        for r in resultados
    ]
    _imprimir_tabla(headers_main, filas_main)
    print("\n* Latencia referencial medida en esta máquina, NO en Raspberry Pi.")

    # --- Tabla de accuracy por clase ---
    print("\n\nACCURACY POR CLASE")
    print("=" * 70)

    headers_cls = ["Clase"] + [r["nombre"] for r in resultados]
    filas_cls = [
        [cls] + [f"{r['acc_por_clase'].get(cls, 0.0)*100:.2f}%" for r in resultados]
        for cls in CLASES
    ]
    _imprimir_tabla(headers_cls, filas_cls)

    # --- Matrices de confusión en texto ---
    print("\n\nMATRICES DE CONFUSIÓN")
    print("=" * 70)
    for r in resultados:
        print(f"\n{r['nombre']}:")
        header_cm = ["Real \\ Pred"] + CLASES
        filas_cm = [
            [CLASES[i]] + [str(v) for v in r["cm"][i]]
            for i in range(len(CLASES))
        ]
        _imprimir_tabla(header_cm, filas_cm)

    # --- Resumen del ganador ---
    mejor_acc = max(resultados, key=lambda r: r["accuracy"])
    mejor_f1 = max(resultados, key=lambda r: r["f1_macro"])
    menor_tam = min(resultados, key=lambda r: r["size_keras_mb"])

    print("\n\nRESUMEN")
    print("=" * 70)
    print(f"✅ Mejor accuracy global : {mejor_acc['nombre']}  →  {mejor_acc['accuracy']:.2%}")
    print(f"✅ Mejor F1 Macro        : {mejor_f1['nombre']}  →  {mejor_f1['f1_macro']:.4f}")
    print(f"✅ Modelo más ligero (.keras): {menor_tam['nombre']}  →  {menor_tam['size_keras_mb']:.2f} MB")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset no encontrado en: {DATA_DIR}")

    print(f"Dataset : {DATA_DIR}")

    test_gen = construir_test_gen()
    print(f"Imágenes de prueba : {test_gen.samples}")
    print(f"Clases             : {list(test_gen.class_indices.keys())}")

    resultados = []
    for config in MODELOS:
        resultado = evaluar_modelo(config, test_gen)
        if resultado is not None:
            resultados.append(resultado)

    if not resultados:
        print("\n❌ No se pudo evaluar ningún modelo. Verifica que los archivos .keras existen en models/")
        return

    imprimir_tabla_comparativa(resultados)


if __name__ == "__main__":
    main()
