"""
Convierte los pesos de ShuffleNetV2 x1.0 preentrenados en ImageNet (torchvision)
a un archivo .npz con el formato de tensores que espera Keras.

¿Por qué existe este script?
    Keras no incluye ShuffleNetV2 en keras.applications, así que no hay forma
    de pedir weights='imagenet' como con MobileNetV3 o EfficientNetB0.
    La fuente oficial de pesos preentrenados es torchvision. Este script los
    descarga UNA sola vez y los deja en models/pretrained/ para que el notebook
    clasificacion_ShuffleNetV2_preentrenado.ipynb los cargue solo con numpy.

Solo hace falta correrlo si el .npz no existe. Requiere torch y torchvision,
que NO están en requirements.txt a propósito: instalarlos en el venv del
proyecto puede pisar las librerías CUDA de TensorFlow. Se corre en un entorno
aislado con uv (desde la raíz del repo):

    uv run --no-project --python 3.12 \
        --with torch --with torchvision --with numpy \
        --index https://download.pytorch.org/whl/cpu \
        tests/convertir_pesos_shufflenetv2.py

Salida:
    models/pretrained/shufflenet_v2_x1_0_imagenet.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0

BASE_DIR = Path(__file__).resolve().parent.parent
SALIDA_DEFAULT = BASE_DIR / "models" / "pretrained" / "shufflenet_v2_x1_0_imagenet.npz"

# Normalización con la que torchvision entrenó el modelo en ImageNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def convertir(state_dict) -> dict:
    """Traduce el state_dict de PyTorch al layout de tensores de Keras.

    El nombre de cada capa es la ruta del módulo de torchvision cambiando
    '.' por '_' (p. ej. 'stage2.0.branch1.0' -> 'stage2_0_branch1_0'); el
    notebook nombra sus capas igual, y así se emparejan sin ambigüedad.
    """
    pesos = {}
    for clave, tensor in state_dict.items():
        if clave.endswith("num_batches_tracked"):
            continue  # contador interno de PyTorch, Keras no lo usa

        modulo, parametro = clave.rsplit(".", 1)
        nombre = modulo.replace(".", "_")
        arr = tensor.detach().cpu().numpy()

        if arr.ndim == 4:
            if arr.shape[1] == 1 and arr.shape[0] > 1 and arr.shape[2] == 3:
                # Convolución depthwise: (C, 1, H, W) -> (H, W, C, 1)
                pesos[f"{nombre}__kernel"] = arr.transpose(2, 3, 0, 1)
            else:
                # Convolución normal: (O, I, H, W) -> (H, W, I, O)
                pesos[f"{nombre}__kernel"] = arr.transpose(2, 3, 1, 0)
        elif modulo == "fc":
            # Capa densa de 1000 clases: solo se guarda para poder verificar el porte
            if parametro == "weight":
                pesos["fc__kernel"] = arr.T  # (out, in) -> (in, out)
            else:
                pesos["fc__bias"] = arr
        else:
            # BatchNorm: weight/bias/running_mean/running_var
            destino = {
                "weight": "gamma",
                "bias": "beta",
                "running_mean": "moving_mean",
                "running_var": "moving_variance",
            }[parametro]
            pesos[f"{nombre}__{destino}"] = arr

    return pesos


def main():
    parser = argparse.ArgumentParser(description="Exporta ShuffleNetV2 x1.0 (ImageNet) de torchvision a .npz")
    parser.add_argument("--salida", type=Path, default=SALIDA_DEFAULT)
    parser.add_argument(
        "--referencia", type=Path, default=None,
        help="Opcional: guarda una imagen aleatoria y los logits de PyTorch para verificar el porte en Keras",
    )
    args = parser.parse_args()

    pesos_tv = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    modelo = shufflenet_v2_x1_0(weights=pesos_tv).eval()

    pesos = convertir(modelo.state_dict())
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida, **pesos)

    n_params = sum(v.size for v in pesos.values())
    print(f"Pesos de origen : torchvision {pesos_tv} ({pesos_tv.url})")
    print(f"Tensores        : {len(pesos)}  ({n_params:,} valores)")
    print(f"Guardado en     : {args.salida}  ({args.salida.stat().st_size / 1024**2:.2f} MB)")

    if args.referencia:
        # Imagen en el rango 0-255, igual que la que entrega ImageDataGenerator
        rng = np.random.default_rng(0)
        imagen = rng.uniform(0, 255, size=(4, 224, 224, 3)).astype(np.float32)
        normalizada = (imagen / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        with torch.no_grad():
            logits = modelo(torch.from_numpy(normalizada.transpose(0, 3, 1, 2))).numpy()
        np.savez(args.referencia, imagen=imagen, logits=logits)
        print(f"Referencia      : {args.referencia}")


if __name__ == "__main__":
    main()
