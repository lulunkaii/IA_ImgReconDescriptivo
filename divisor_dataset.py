import os
import shutil
import random
from pathlib import Path

# Ruta actual de tus imágenes
original_dataset_dir = "animals/animals"  # ajusta si tu ruta es distinta
output_base_dir = "dataset"
train_ratio = 0.8  # 80% para entrenamiento

random.seed(42)  # Para resultados reproducibles

# Crear carpetas base
for split in ['train', 'val']:
    for class_name in os.listdir(original_dataset_dir):
        split_dir = os.path.join(output_base_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)

# Recorremos cada clase
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    images = os.listdir(class_dir)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Copiar imágenes
    for img_name in train_imgs:
        src = os.path.join(class_dir, img_name)
        dst = os.path.join(output_base_dir, 'train', class_name, img_name)
        shutil.copy(src, dst)

    for img_name in val_imgs:
        src = os.path.join(class_dir, img_name)
        dst = os.path.join(output_base_dir, 'val', class_name, img_name)
        shutil.copy(src, dst)

print("✅ División completada. Revisa la carpeta 'dataset/'")
