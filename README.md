# ♻️ Clasificación Inteligente de Residuos (Edge AI)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/IoT-Raspberry_Pi-cbb03b?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Proyecto de Transfer Learning para clasificar residuos domésticos utilizando MobileNetV2, optimizado para despliegue en dispositivos IoT (Raspberry Pi).**

## 📖 Descripción del Proyecto

Este proyecto aborda el problema de la clasificación de basura utilizando Deep Learning. El objetivo es entrenar un modelo robusto capaz de distinguir entre **30 clases de residuos** (plástico, vidrio, cartón, biológico, etc.) y desplegarlo eficientemente en una Raspberry Pi.

El flujo de trabajo incluye:
1.  **Ingesta de Datos:** Descarga automatizada y cross-platform del dataset.
2.  **Preprocesamiento:** Data Augmentation agresivo para evitar overfitting.
3.  **Modelado:** Transfer Learning con **MobileNetV2** (Google) + Fine Tuning.
4.  **Despliegue:** Conversión a **TensorFlow Lite (.tflite)** para inferencia en tiempo real en el borde (Edge).


## 📂 Estructura del Proyecto

El proyecto sigue el estándar *Cookiecutter Data Science*:

```text
├── data/
│   ├── raw/                 # Datos originales (descargados automáticamente)
│   └── processed/           # Datos transformados 
├── models/                  # Modelos entrenados (.keras = maestro, .tflite = Raspberry Pi)
│   ├── MobileNetV2/         # modelo_residuos_rpi*, más filtrado_ordenamiento/
│   ├── MobileNetV3/         # modelo_residuos_rpi_V3_*
│   ├── EfficientNetB0/      # modelo_residuos_EfficientNetB0_v1
│   └── ShuffleNetV2/        # modelo_residuos_shufflenetV2_imagenet + pretrained/
├── notebooks/
│   ├── MobileNetV2/         # Entrenamiento (30 y 5 clases) y filtrado_ordenamiento/
│   ├── MobileNetV3/         # Entrenamiento y evaluación de MobileNetV3-Large
│   ├── EfficientNetB0/      # Entrenamiento y evaluación de EfficientNetB0
│   └── ShuffleNetV2/        # Entrenamiento y evaluación de ShuffleNetV2 (ImageNet)
├── entrenamientoia/         # Código fuente y scripts auxiliares
├── requirements.txt         # Dependencias del proyecto
├── Makefile                 # Comandos de automatización (Linux/Mac)
└── README.md                # Documentación
``` 

## 🛠️ Instalación y Configuración
Sigue estos pasos para preparar tu entorno de desarrollo. Este proyecto está configurado para funcionar tanto en Windows como en Linux.

1. Clonar el repositorio


```bash
git clone https://github.com/LordAguaKate/EntrenamientoIA.git
cd EntrenamientoIA
```

2. Crear el Entorno Virtual (VENV)

Opción A: Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Opción B: Windows (PowerShell/CMD)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Instalar Dependencias

Una vez activo el entorno (deberías ver (.venv) en tu terminal):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**Nota:** El archivo requirements.txt incluye kagglehub, tensorflow, matplotlib, notebook, entre otros.



## 🚀 Cómo Entrenar el Modelo
Todo el flujo de entrenamiento reside en el notebook principal.

1. Inicia Jupyter Lab o Notebook:

```bash
jupyter notebook
```

2. Abre el notebook de entrenamiento del modelo que quieras, por ejemplo: notebooks/MobileNetV2/entrenamiento_MobileNetV2_30clases.ipynb.

3. Ejecuta las celdas en orden acendente.

**Resultado:** Al finalizar, encontrarás los archivos .keras y .tflite en la subcarpeta de su modelo dentro de models/ (por ejemplo, models/MobileNetV2/).



## 🍓 Despliegue en Raspberry Pi

Para llevar este modelo al mundo físico:

1. Transferencia: Copia el archivo models/MobileNetV2/modelo_residuos_rpi.tflite a tu Raspberry Pi.

2. Entorno en la Pi: No necesitas instalar TensorFlow completo (es muy pesado). Usa el runtime ligero:

```bash
pip install tflite-runtime numpy pillow
```

3. Script de Inferencia (Ejemplo Rápido):

```bash
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Cargar modelo
interpreter = tflite.Interpreter(model_path="modelo_residuos_rpi.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada/salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocesar imagen (igual que en el entrenamiento)
img = Image.open("basura.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

# Inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Clase predicha:", np.argmax(output_data))
```


## 🤝 Contribuciones

1. Haz un Fork del proyecto.

2. Crea una rama para tu feature (git checkout -b feature/NuevaMejora).

3. Haz tus cambios y Commit (git commit -m 'Añadir X mejora').

4. Haz Push (git push origin feature/NuevaMejora).

5. Abre un Pull Request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.


