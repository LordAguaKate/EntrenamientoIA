import random
import shutil
from pathlib import Path

# 1. Configuración de parámetros
split_ratio = 0.8  # 80% para entrenamiento, 20% para prueba

# 2. Configuración de rutas usando Pathlib
# Asumiendo que ejecutas este script estando dentro de la carpeta 'tests'
base_dir = Path(".")
source_dir = base_dir / "imagenes_originales"
dest_dir = base_dir / "waste_classification"
train_dir = dest_dir / "train"
test_dir = dest_dir / "test"

# 3. Categorías basadas en la estructura de tus carpetas
categorias = ["aluminio", "carton", "otros", "plastico", "vidrio"]

total_train = 0
total_test = 0

print("Iniciando el ordenamiento del dataset por categorías...\n")
print("-" * 50)

# 4. Iterar sobre cada categoría para procesarlas de manera independiente
for categoria in categorias:
    cat_source = source_dir / categoria
    cat_train = train_dir / categoria
    cat_test = test_dir / categoria

    # Crear las subcarpetas de destino si aún no existen
    cat_train.mkdir(parents=True, exist_ok=True)
    cat_test.mkdir(parents=True, exist_ok=True)

    if not cat_source.exists():
        print(f"Advertencia: La carpeta {cat_source} no existe. Omitiendo...")
        continue

    # Obtener lista de archivos válidos dentro de la subcarpeta de la categoría
    files = [f.name for f in cat_source.iterdir() if f.is_file()]

    if not files:
        print(f"Advertencia: No hay imágenes en {categoria}.")
        continue

    # Mezclar aleatoriamente para evitar sesgos en el entrenamiento
    random.shuffle(files)

    # Calcular índice de división (80/20 por cada clase)
    split_point = int(len(files) * split_ratio)

    train_files = files[:split_point]
    test_files = files[split_point:]

    # Copiar archivos a la carpeta 'train' de su respectiva categoría
    for f in train_files:
        shutil.copy2(cat_source / f, cat_train / f)

    # Copiar archivos a la carpeta 'test' de su respectiva categoría
    for f in test_files:
        shutil.copy2(cat_source / f, cat_test / f)

    total_train += len(train_files)
    total_test += len(test_files)

    print(
        f"✓ Categoría '{categoria.capitalize()}': {len(train_files)} a train | {len(test_files)} a test."
    )

print("-" * 50)
print("¡Distribución de imágenes completada exitosamente!")
print(f"Total en directorio Train: {total_train} imágenes")
print(f"Total en directorio Test : {total_test} imágenes")
