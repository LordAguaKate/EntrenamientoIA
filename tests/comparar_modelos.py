"""
Genera una tabla comparativa de modelos ligeros leyendo los resultados
directamente desde los archivos .ipynb ya ejecutados, sin volver a cargar
ni correr inferencias sobre ningún modelo.

Notebooks leídos:
    - evaluacion_modelo_MobileNetV3.ipynb
    - evaluacion_modelo_EfficientNetB0.ipynb
    - evaluacion_modelo_ShuffleNetV2.ipynb               (entrenado desde cero)
    - evaluacion_modelo_ShuffleNetV2_preentrenado.ipynb  (pesos de ImageNet)

El script busca en los outputs de cada notebook:
    - Accuracy y Loss  (celda de model.evaluate)
    - Matriz de confusión en texto plano  (celda de print(cm))
    - Reporte de clasificación por clase  (classification_report)

Si un notebook no tiene outputs guardados (aún no fue ejecutado),
se marca como "Pendiente" en la tabla.

Uso (desde la raíz del repo):
    python tests/comparar_modelos.py

Requiere: tabulate  (opcional, para mejor formato)
    pip install tabulate
"""

import json
import re
from pathlib import Path

try:
    from tabulate import tabulate
    TABULATE_DISPONIBLE = True
except ImportError:
    TABULATE_DISPONIBLE = False
    print("⚠️  'tabulate' no está instalada. Se usará formato de texto plano.")
    print("   Para instalarla: pip install tabulate\n")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = BASE_DIR / "models"

# Cada entrada vincula el nombre del modelo con su notebook de evaluación
# y el nombre del archivo .keras para leer su tamaño en disco.
MODELOS = [
    {
        "nombre": "MobileNetV3-Large",
        "notebook": NOTEBOOKS_DIR / "evaluacion_modelo_MobileNetV3.ipynb",
        "keras": MODELS_DIR / "modelo_residuos_rpi_V3_05.keras",
        "tflite": MODELS_DIR / "modelo_residuos_rpi_V3_05.tflite",
    },
    {
        "nombre": "EfficientNetB0",
        "notebook": NOTEBOOKS_DIR / "evaluacion_modelo_EfficientNetB0.ipynb",
        "keras": MODELS_DIR / "modelo_residuos_EfficientNetB0_v1.keras",
        "tflite": MODELS_DIR / "modelo_residuos_EfficientNetB0_v1.tflite",
    },
    {
        # Entrenado desde cero, sin pesos de ImageNet
        "nombre": "ShuffleNetV2 (desde cero)",
        "notebook": NOTEBOOKS_DIR / "evaluacion_modelo_ShuffleNetV2.ipynb",
        "keras": MODELS_DIR / "modelo_residuos_shufflenetV2_nativo.keras",
        "tflite": MODELS_DIR / "modelo_residuos_shufflenetV2_nativo.tflite",
    },
    {
        # Con pesos de ImageNet y el mismo protocolo de dos fases que los otros modelos
        "nombre": "ShuffleNetV2 (ImageNet)",
        "notebook": NOTEBOOKS_DIR / "evaluacion_modelo_ShuffleNetV2_preentrenado.ipynb",
        "keras": MODELS_DIR / "modelo_residuos_shufflenetV2_imagenet.keras",
        "tflite": MODELS_DIR / "modelo_residuos_shufflenetV2_imagenet.tflite",
    },
]

# Clases del dataset (orden alfabético, igual que en los notebooks)
CLASES = ["aluminio", "carton", "otros", "plastico", "vidrio"]


# ---------------------------------------------------------------------------
# EXTRACCIÓN DE RESULTADOS DESDE LOS NOTEBOOKS
# ---------------------------------------------------------------------------

def leer_outputs_notebook(path: Path) -> list[str]:
    """Lee un .ipynb y devuelve todos los textos de output concatenados por celda."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    outputs_por_celda = []
    for cell in nb.get("cells", []):
        texto = ""
        for output in cell.get("outputs", []):
            # Los outputs pueden ser 'stream' (print) o 'execute_result'
            if "text" in output:
                texto += "".join(output["text"])
            elif "data" in output and "text/plain" in output["data"]:
                texto += "".join(output["data"]["text/plain"])
        if texto.strip():
            outputs_por_celda.append(texto)

    return outputs_por_celda


def extraer_accuracy_loss(outputs: list[str]) -> tuple[float, float] | tuple[None, None]:
    """Busca 'Precisión (Accuracy): XX.XX%' y 'Pérdida (Loss): X.XXXX' en los outputs."""
    accuracy = None
    loss = None

    for texto in outputs:
        if accuracy is None:
            # Busca el patrón del print de resultados finales
            m = re.search(r"Precisi[oó]n.*?Accuracy.*?:\s*([\d.]+)%", texto, re.IGNORECASE)
            if m:
                accuracy = float(m.group(1)) / 100.0

        if loss is None:
            m = re.search(r"P[eé]rdida.*?Loss.*?:\s*([\d.]+)", texto, re.IGNORECASE)
            if m:
                loss = float(m.group(1))

        if accuracy is not None and loss is not None:
            break

    return accuracy, loss


def extraer_matriz_confusion(outputs: list[str]) -> list[list[int]] | None:
    """
    Busca el output de 'print(cm)' que imprime la matriz como arrays de numpy:
        [[ 869    0    8    9    2]
         [   6  710  10    4    0]
         ...
    """
    for texto in outputs:
        # Detectamos el bloque con corchetes dobles típico de numpy
        if "[[" in texto and "]]" in texto:
            filas = re.findall(r"\[\s*([\d\s]+)\]", texto)
            if len(filas) >= len(CLASES):
                try:
                    matriz = [[int(n) for n in fila.split()] for fila in filas[:len(CLASES)]]
                    if all(len(f) == len(CLASES) for f in matriz):
                        return matriz
                except ValueError:
                    continue
    return None


def extraer_reporte_clases(outputs: list[str]) -> dict[str, dict] | None:
    """
    Parsea el classification_report de sklearn que imprime columnas:
        precision  recall  f1-score  support
    para cada clase.
    """
    for texto in outputs:
        if "precision" in texto and "recall" in texto and "f1-score" in texto:
            resultado = {}
            for cls in CLASES:
                # Busca la línea de la clase: "aluminio    0.97    0.98    0.97    888"
                patron = rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
                m = re.search(patron, texto)
                if m:
                    resultado[cls] = {
                        "precision": float(m.group(1)),
                        "recall": float(m.group(2)),
                        "f1": float(m.group(3)),
                        "support": int(m.group(4)),
                    }
            # También extraemos el macro avg
            m_macro = re.search(
                r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", texto
            )
            if m_macro:
                resultado["macro avg"] = {
                    "precision": float(m_macro.group(1)),
                    "recall": float(m_macro.group(2)),
                    "f1": float(m_macro.group(3)),
                }
            if resultado:
                return resultado
    return None


def tamanio_archivo_mb(path: Path) -> str:
    """Devuelve el tamaño en MB como string, o '—' si no existe."""
    if path and path.exists():
        return f"{path.stat().st_size / (1024 * 1024):.2f}"
    return "—"


def procesar_modelo(config: dict) -> dict:
    """Lee el notebook del modelo y extrae todas las métricas disponibles."""
    nombre = config["nombre"]
    nb_path = config["notebook"]

    resultado_base = {
        "nombre": nombre,
        "accuracy": None,
        "loss": None,
        "errores": None,
        "f1_macro": None,
        "precision_macro": None,
        "recall_macro": None,
        "acc_por_clase": {cls: None for cls in CLASES},
        "size_keras": tamanio_archivo_mb(config.get("keras")),
        "size_tflite": tamanio_archivo_mb(config.get("tflite")),
        "pendiente": False,
    }

    if not nb_path.exists():
        print(f"   ⚠️  Notebook no encontrado: {nb_path.name}")
        resultado_base["pendiente"] = True
        return resultado_base

    outputs = leer_outputs_notebook(nb_path)

    if not outputs:
        print(f"   ⚠️  El notebook '{nb_path.name}' no tiene outputs guardados (pendiente de ejecutar).")
        resultado_base["pendiente"] = True
        return resultado_base

    # Accuracy y Loss
    accuracy, loss = extraer_accuracy_loss(outputs)
    resultado_base["accuracy"] = accuracy
    resultado_base["loss"] = loss

    # Matriz de confusión → accuracy por clase
    cm = extraer_matriz_confusion(outputs)
    if cm:
        for i, cls in enumerate(CLASES):
            total_cls = sum(cm[i])
            resultado_base["acc_por_clase"][cls] = cm[i][i] / total_cls if total_cls > 0 else 0.0

        total = sum(sum(fila) for fila in cm)
        aciertos = sum(cm[i][i] for i in range(len(CLASES)))
        resultado_base["errores"] = total - aciertos
        resultado_base["total"] = total

    # F1, Precisión, Recall del reporte de clasificación
    reporte = extraer_reporte_clases(outputs)
    if reporte and "macro avg" in reporte:
        resultado_base["f1_macro"] = reporte["macro avg"]["f1"]
        resultado_base["precision_macro"] = reporte["macro avg"]["precision"]
        resultado_base["recall_macro"] = reporte["macro avg"]["recall"]

    return resultado_base


# ---------------------------------------------------------------------------
# IMPRESIÓN DE TABLAS
# ---------------------------------------------------------------------------

def _fmt(valor, fmt=".4f", sufijo="", pendiente_label="Pendiente"):
    """Formatea un valor numérico o devuelve 'Pendiente' si es None."""
    if valor is None:
        return pendiente_label
    return f"{valor:{fmt}}{sufijo}"


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
    """Imprime la tabla principal y el desglose por clase."""

    # El total sale de sumar la matriz de confusión (dato exacto, no una estimación)
    total_imagenes = next((r["total"] for r in resultados if r.get("total")), None)

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA DE MODELOS — TEST SET (waste_classification)")
    if total_imagenes:
        print(f"Total de imágenes evaluadas: {total_imagenes}  |  Clases: {len(CLASES)}")
    print("=" * 70)

    # --- Tabla principal ---
    headers_main = [
        "Modelo", "Accuracy", "Loss", "Errores",
        "F1 Macro", "Precisión Macro", "Recall Macro",
        "Tam .keras (MB)", "Tam .tflite (MB)",
    ]
    filas_main = [
        [
            r["nombre"],
            _fmt(r["accuracy"], ".2%"),
            _fmt(r["loss"], ".4f"),
            f"{r['errores']}" if r["errores"] is not None else "Pendiente",
            _fmt(r["f1_macro"], ".4f"),
            _fmt(r["precision_macro"], ".4f"),
            _fmt(r["recall_macro"], ".4f"),
            r["size_keras"],
            r["size_tflite"],
        ]
        for r in resultados
    ]
    _imprimir_tabla(headers_main, filas_main)

    # --- Tabla de accuracy por clase ---
    print("\n\nACCURACY POR CLASE")
    print("=" * 70)

    headers_cls = ["Clase"] + [r["nombre"] for r in resultados]
    filas_cls = [
        [cls] + [_fmt(r["acc_por_clase"].get(cls), ".2%") for r in resultados]
        for cls in CLASES
    ]
    _imprimir_tabla(headers_cls, filas_cls)

    # --- Resumen del ganador (solo entre modelos con resultados) ---
    con_resultados = [r for r in resultados if r["accuracy"] is not None]
    if con_resultados:
        mejor_acc = max(con_resultados, key=lambda r: r["accuracy"])
        print(f"\n✅ Mejor accuracy global: {mejor_acc['nombre']}  →  {mejor_acc['accuracy']:.2%}")

        con_f1 = [r for r in con_resultados if r["f1_macro"] is not None]
        if con_f1:
            mejor_f1 = max(con_f1, key=lambda r: r["f1_macro"])
            print(f"✅ Mejor F1 Macro:        {mejor_f1['nombre']}  →  {mejor_f1['f1_macro']:.4f}")

    pendientes = [r["nombre"] for r in resultados if r["pendiente"]]
    if pendientes:
        print(f"\n⏳ Pendientes de ejecutar: {', '.join(pendientes)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Leyendo resultados desde los notebooks de evaluación...\n")

    resultados = []
    for config in MODELOS:
        print(f"→ {config['nombre']} ({config['notebook'].name})")
        resultado = procesar_modelo(config)
        resultados.append(resultado)

    imprimir_tabla_comparativa(resultados)


if __name__ == "__main__":
    main()
