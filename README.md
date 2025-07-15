# IA_ImgReconDescriptivo

## Descripción del Proyecto
Este proyecto aborda el problema de interpretabilidad de los ***Vision Transformer (ViT)*** y su necesidad de una gran cantidad de datos para entrenamiento en comparación con las Redes Convolucionales (CNNs), llegando a necesitar millones de imágenes a cambio de ser un modelo más flexible.


Este proyecto implementa un Vision Transformer basándose en el paper ***"Vision Transformer for Small-Size Datasets"***, en el cual se aborda el entrenamiento de un **ViT** con una baja cantidad de datos utilizando los siguientes algoritmos:
- **Shifted Patch Tokenization (SPT)**: Técnica para mejorar la extracción de características.
- **Locality Self-Attention (LSA)**: Mecanismo de atención que enfatiza características locales.

Para poder mejorar la interpretabilidad del modelo, se implementó la visualización de los mapas de atención correspondientes para cada imagen, permitiendo al desarrollador verificar en que partes de la imagen se concentra el modelo.

## Datasets Utilizados

### 1. CIFAR-10
- **Descripción**: Dataset estándar de clasificación de imágenes con 10 clases.
- **Clases**: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión.
- **Dataset**: 60,000 imágenes (6,000 por clase).
- **Características**:
  - **Entrenamiento**: 50,000 imágenes (5,000 por clase).
  - **Validación**: 10,000 imágenes (1,000 por clase).
  - **Resolución**: 32x32 píxeles.
  - **Canales**: RGB (3 canales).
- **Preprocesamiento**:
  - Redimensionado a 32x32.
  - AutoAugment con política CIFAR-10.
  - Normalización: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616).

### 2. Dataset de Animales (Kaggle)
- **Descripción**: Dataset personalizado de animales obtenido de Kaggle.
- **Clases**: 90 diferentes especies de animales.
- **Dataser**: 5,400 imágenes (60 por clase).
- **Características**:
  - **Entrenamiento**: 4,320 imágenes (48 por clase).
  - **Validación**: 1,080 imágenes (12 por clase).
  - **Resolución**: Redimensionado a 224x224 píxeles.
  - **Canales**: RGB (3 canales).
- **Preprocesamiento**:
  - Redimensionado a 224x224.
  - Flip horizontal aleatorio (p=0.5).
  - Normalización: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010).

## Arquitectura del Modelo

### Configuración para CIFAR-10
```
- Tamaño de imagen: 32x32
- Tamaño de patch: 4x4
- Dimensión del embedding: 192
- Profundidad: 9 capas
- Cabezas de atención: 12
- Dimensión de cabeza: 16
- Dropout: 0.1
```

### Configuración para Dataset Kaggle
```
- Tamaño de imagen: 224x224
- Tamaño de patch: 16x16
- Dimensión del embedding: 192
- Profundidad: 6 capas
- Cabezas de atención: 8
- Dimensión de cabeza: 64
- Dropout: 0.1
```

## Experimentos Realizados

El proyecto incluye múltiples configuraciones experimentales:

1. **CIFAR-10 con LSA y SPT**: Configuración completa con todas las mejoras.
2. **CIFAR-10 sin LSA ni SPT**: Configuración base para comparación.
3. **Dataset Kaggle con Cross-Validation**: Validación robusta en 5 pliegues.
4. **Dataset Kaggle sin SPT ni LSA**: Configuración de control.
