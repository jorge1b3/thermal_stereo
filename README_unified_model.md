# Modelo Unificado de Estimación de Profundidad para Imágenes Térmicas

Este proyecto implementa un modelo unificado para estimación de profundidad a partir de imágenes térmicas, basado en el enfoque NeWCRF (Neural Window Fully Connected Conditional Random Field). El modelo puede operar tanto en modo estéreo (usando un par de imágenes térmicas) como en modo monocular (usando una sola imagen).

## Características

- **Arquitectura unificada**: Funciona con una sola imagen (monocular) o un par de imágenes (estéreo).
- **Basado en NeWCRF**: Implementa un campo aleatorio condicional mediante capas de atención.
- **Entrenamiento multi-escala**: Proporciona predicciones en múltiples escalas para un aprendizaje supervisado más efectivo.
- **Volumen de costos**: Utiliza un volumen de costos para mejorar la estimación de disparidad en el caso estéreo.
- **Optimizado**: Reduce las imágenes a la mitad para acelerar el entrenamiento.

## Estructura del Proyecto

- `src/models/unified_depth_model.py` - Implementación del modelo unificado
- `src/train_unified.py` - Script para entrenar el modelo
- `src/visualize_unified.py` - Script para visualizar resultados
- `src/data/dataset.py` - Conjunto de datos para imágenes térmicas y mapas de profundidad

## Requisitos

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Wandb (para seguimiento de experimentos)
- OpenCV

## Uso

### Entrenamiento

Para entrenar el modelo:

```bash
python -m src.train_unified --data_dir raw --batch_size 4 --num_epochs 20 --lr 0.0001 --output_dir models --initial_filters 32 --max_disp 192
```

Parámetros importantes:
- `--data_dir`: Directorio con los datos (carpetas frick_*, hawkins_*, etc.)
- `--batch_size`: Tamaño del batch para entrenamiento
- `--initial_filters`: Número de filtros iniciales en el modelo (afecta tamaño/velocidad)
- `--max_disp`: Máxima disparidad para el volumen de costos

### Visualización

Para visualizar resultados después del entrenamiento:

```bash
python -m src.visualize_unified --model_path models/unified_depth_model_best.pth --data_dir raw --split test --output_dir outputs --num_samples 10
```

Parámetros:
- `--model_path`: Ruta al modelo guardado
- `--split`: Conjunto de datos a visualizar ('test', 'val')
- `--num_samples`: Número de muestras aleatorias para visualizar

## Detalles del Modelo

### Componentes principales

1. **Extracción de características**
   - Utiliza un encoder similar a Swin Transformer para extraer características de las imágenes
   - Extrae mapas de características en cuatro escalas

2. **Módulo de agrupación de pirámides (PPM)**
   - Agrega información contextual global

3. **Construcción de volumen de costos**
   - Crea un volumen de costos mediante correlación de características
   - En el caso monocular, utiliza un volumen de ceros como placeholder

4. **Bloques NeWCRF**
   - Implementan el campo aleatorio condicional mediante capas de atención
   - Calculan potenciales unarios y pairwise para refinar las predicciones

5. **Predicción de disparidad/profundidad**
   - Convierte las características en volúmenes de probabilidad
   - Realiza una suma ponderada para obtener las predicciones finales

### Formulación matemática

- **Volumen de costos**: C(d, x, y) = (1/Nc) < fL(x, y), fR(x-d, y) >
- **Potencial unario**: ψu = θu(X)
- **Potencial pairwise**: ψp = SoftMax(Q·K^T + P)·X
- **Predicción final**: Dpred = ∑(k=0 a Dmax-1) k · pk

## Referencias

- NeWCRF: Neural Window Fully-Connected CRF for Stereo Matching
- Unified Depth Network: Depth Estimation from a Single Image or Stereo Pairs
