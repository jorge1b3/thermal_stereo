# Modelo de Profundidad con Imágenes Térmicas

Este proyecto proporciona herramientas para cargar y procesar un conjunto de datos con imágenes térmicas (izquierda y derecha) y de profundidad, así como entrenar un modelo de redes neuronales para estimar la profundidad a partir de las imágenes térmicas.

## Estructura de Directorios

```
raw
├── frick_1
│   ├── depth_filtered
│   ├── img_left
│   └── img_right
├── frick_2_test
│   ├── depth_filtered
│   ├── img_left
│   └── img_right
├── frick_2_train
│   ├── depth_filtered
│   ├── img_left
│   └── img_right
├── frick_2_val
│   ├── depth_filtered
│   ├── img_left
│   └── img_right
... (y más subdirectorios)
```

## Estructura del Código

- `src/data/dataset.py`: Contiene la clase `ThermalDepthDataset` que implementa un dataset de PyTorch.
- `src/utils/data_loader.py`: Proporciona utilidades para cargar conjuntos de datos de entrenamiento, prueba y validación.
- `src/models/models.py`: Contiene diferentes arquitecturas de modelos para estimación de profundidad.
- `src/main.py`: Script principal para demostrar cómo usar los cargadores de datos.
- `src/example.py`: Script de ejemplo para visualizar muestras de los conjuntos de datos.
- `src/train.py`: Script para entrenar un modelo de estimación de profundidad con Weights & Biases (wandb).
- `src/visualize.py`: Script para visualizar los resultados del modelo entrenado.

## Uso

### Cargar Datos

Para cargar tanto los datos de entrenamiento como de prueba, puedes ejecutar:

```bash
python -m src.main --data_dir raw
```

Para cargar solo los datos de entrenamiento:

```bash
python -m src.main --data_dir raw --mode train
```

Para cargar solo los datos de prueba:

```bash
python -m src.main --data_dir raw --mode test
```

### Visualizar Ejemplos

Para visualizar ejemplos de los conjuntos de datos:

```bash
python -m src.example
```

Esto generará visualizaciones de muestras de los conjuntos de entrenamiento y prueba.

### Entrenar Modelo

Para entrenar el modelo de estimación de profundidad:

```bash
python -m src.train --data_dir raw --batch_size 8 --num_epochs 20 --lr 3e-4 --model_type basic
```

El entrenamiento utilizará Weights & Biases (wandb) para seguimiento y visualización. Los pesos del modelo se guardarán en el directorio `models/`.

Parámetros principales:
- `--data_dir`: Directorio donde se encuentran los datos (por defecto: "raw")
- `--batch_size`: Tamaño del lote para entrenamiento (por defecto: 8)
- `--num_epochs`: Número de épocas de entrenamiento (por defecto: 20)
- `--lr`: Tasa de aprendizaje (por defecto: 3e-4)
- `--model_type`: Tipo de arquitectura a usar (opciones: "basic", "resnet", "unet")
- `--initial_filters`: Número inicial de filtros en el modelo (por defecto: 64)
- `--dropout_rate`: Tasa de dropout para regularización (por defecto: 0.2)

### Visualizar Resultados del Modelo

Para visualizar las predicciones del modelo entrenado:

```bash
python -m src.visualize --model_path models/model_trained.pth --data_dir raw --num_samples 5 --model_type basic
```

Asegúrate de especificar el mismo tipo de modelo que usaste para entrenar utilizando el parámetro `--model_type`.

Esto generará visualizaciones de predicciones de profundidad comparadas con los valores reales.

### Uso Programático

También puedes usar el cargador de datos directamente en tu código:

```python
from src.utils.data_loader import load_train_dataset, load_test_dataset

# Cargar conjunto de entrenamiento
train_loader = load_train_dataset(root_dir='raw', batch_size=8)

# Cargar conjunto de prueba
test_loader = load_test_dataset(root_dir='raw', batch_size=8)

# Iterar sobre los lotes
for batch in train_loader:
    # Acceder a las imágenes
    left_images = batch['left']    # Imágenes térmicas izquierdas
    right_images = batch['right']  # Imágenes térmicas derechas
    depth_images = batch['depth']  # Imágenes de profundidad
    
    # Hacer algo con las imágenes...
    break
```

## Transformaciones

Puedes aplicar transformaciones personalizadas pasando una función de transformación al cargar los datos:

```python
from src.utils.data_loader import load_train_dataset
import torch

# Ejemplo de transformación que convierte las imágenes a tensores de PyTorch
def transform(sample):
    return {
        'left': torch.from_numpy(sample['left']).float(),
        'right': torch.from_numpy(sample['right']).float(),
        'depth': torch.from_numpy(sample['depth']).float(),
        'subdir': sample['subdir'],
        'filename': sample['filename']
    }

# Cargar conjunto de entrenamiento con transformación
train_loader = load_train_dataset(
    root_dir='raw',
    batch_size=8,
    transform=transform
)
```

## Modelo de Estimación de Profundidad

El proyecto incluye un modelo de red neuronal convolucional para estimar mapas de profundidad a partir de pares de imágenes térmicas. El modelo está implementado en PyTorch y tiene la siguiente arquitectura:

1. **Codificador para imagen térmica izquierda**: Procesa la imagen térmica izquierda.
2. **Codificador para imagen térmica derecha**: Procesa la imagen térmica derecha.
3. **Decodificador**: Combina las características de ambas imágenes y reconstruye el mapa de profundidad.

### Entrenamiento con Weights & Biases

El proceso de entrenamiento utiliza [Weights & Biases](https://wandb.ai/) para seguimiento de experimentos. Esto permite:

- Visualizar curvas de pérdida durante el entrenamiento
- Comparar múltiples experimentos con diferentes hiperparámetros
- Registrar muestras de predicciones de profundidad
- Almacenar los artefactos del modelo para uso posterior

Para iniciar el entrenamiento, primero debes [crear una cuenta en Weights & Biases](https://wandb.ai/signup) y autenticarte:

```bash
pip install wandb
wandb login
```

Luego puedes iniciar el entrenamiento como se describe en la sección de uso.