#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para evaluar el modelo de profundidad en un 10% del conjunto de prueba.
Calcula métricas de rendimiento y genera visualizaciones de la predicción de profundidad.

Uso:
    python challenge.py --model_path models/unified_depth_model_unified_lr_1.000e-04_bs_4_ep_5_frac_0.2_final_20250531_160602.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import logging
import random
import json
from pathlib import Path

# Importaciones específicas del proyecto
from src.models.unified_depth_model import UnifiedDepthModel, prepare_inputs
from src.data.dataset import ThermalDepthDataset

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Procesa los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Evaluar modelo de profundidad en conjunto de test")
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--data_dir', type=str, default='raw',
                      help='Directorio que contiene los datos')
    parser.add_argument('--output_dir', type=str, default='challenge_results',
                      help='Directorio para guardar resultados')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Tamaño del batch para inferencia')
    parser.add_argument('--test_fraction', type=float, default=0.1,
                      help='Fracción del conjunto de prueba a utilizar (0.0-1.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Número de trabajadores para carga de datos')
    parser.add_argument('--save_images', action='store_true',
                      help='Guardar visualizaciones de las profundidades predichas')
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para reproducibilidad')
    
    return parser.parse_args()

def load_model(model_path):
    """
    Carga el modelo desde el checkpoint guardado.
    """
    logger.info(f"Cargando modelo desde {model_path}")
    
    # Determinar si el modelo es un estado de diccionario simple o un checkpoint completo
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extraer configuración del modelo si está disponible
    initial_filters = 32  # valor por defecto
    if 'config' in checkpoint:
        initial_filters = checkpoint['config'].get('initial_filters', 16)
        logger.info(f"Usando configuración del checkpoint: initial_filters={initial_filters}")
    
    # Crear instancia del modelo
    model = UnifiedDepthModel(in_channels=3, base_filters=initial_filters, max_disp=192)
    
    # Cargar pesos del modelo
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Modelo cargado exitosamente")
    except Exception as e:
        logger.warning(f"Error al cargar el modelo: {e}")
        logger.warning("Intentando carga flexible...")
        # Intenta cargar el modelo de forma flexible (ignorar claves que no coincidan)
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
            
        model_dict = model.state_dict()
        # Filtrar las claves que coinciden en ambos diccionarios
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # Actualizar el diccionario del modelo
        model_dict.update(pretrained_dict)
        # Cargar el diccionario actualizado
        model.load_state_dict(model_dict, strict=False)
        logger.info(f"Modelo cargado con {len(pretrained_dict)}/{len(model_dict)} parámetros")
        
    return model

def calculate_metrics(pred_depth, gt_depth, mask=None):
    """
    Calcula métricas de evaluación para profundidad.
    
    Args:
        pred_depth: Profundidad predicha
        gt_depth: Profundidad de groundtruth
        mask: Máscara opcional para áreas válidas
    
    Returns:
        dict: Diccionario con varias métricas
    """
    # Asegurar que ambas profundidades tengan el mismo tamaño antes de convertirlas a numpy
    if pred_depth.shape[2:] != gt_depth.shape[2:]:
        logger.info("Redimensionando pred_depth de {} a {}".format(pred_depth.shape, gt_depth.shape[2:]))
        pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[2:], mode='bilinear', align_corners=True)
    
    # Asegurarse de que sean arrays de numpy
    pred_depth = pred_depth.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    
    if mask is not None:
        mask = mask.squeeze().cpu().numpy().astype(bool)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    
    # Calcular métricas
    abs_diff = np.abs(pred_depth - gt_depth)
    abs_rel = np.mean(abs_diff / (gt_depth + 1e-10))
    sq_rel = np.mean(((pred_depth - gt_depth)**2) / (gt_depth + 1e-10))
    rmse = np.sqrt(np.mean((pred_depth - gt_depth)**2))
    rmse_log = np.sqrt(np.mean((np.log(pred_depth + 1e-10) - np.log(gt_depth + 1e-10))**2))
    
    # Calcular thresholds (delta_1, delta_2, delta_3)
    thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()
    
    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
    }

def visualize_depth(input_image, pred_depth, gt_depth, output_path):
    """
    Genera y guarda visualizaciones de la profundidad predicha vs. ground truth.
    
    Args:
        input_image: Imagen térmica de entrada
        pred_depth: Profundidad predicha
        gt_depth: Profundidad ground truth
        output_path: Ruta donde guardar la visualización
    """
    # Asegurar que ambas profundidades tengan el mismo tamaño antes de convertirlas a numpy
    if pred_depth.shape[2:] != gt_depth.shape[2:]:
        logger.info("Redimensionando pred_depth de {} a {} para visualización".format(pred_depth.shape, gt_depth.shape[2:]))
        pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[2:], mode='bilinear', align_corners=True)
    
    # Convertir a numpy y asegurar que sean 2D
    input_image = input_image.squeeze().cpu().numpy()
    pred_depth = pred_depth.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    
    # Si input_image tiene 3 canales, usar el promedio
    if len(input_image.shape) == 3:
        if input_image.shape[0] == 3:  # Si los canales están en la primera dimensión (C, H, W)
            input_image = input_image.mean(0)
        elif input_image.shape[2] == 3:  # Si los canales están en la última dimensión (H, W, C)
            input_image = input_image.mean(2)
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen térmica
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title("Imagen Térmica")
    axes[0].axis('off')
    
    # Profundidad predicha
    im_pred = axes[1].imshow(pred_depth, cmap='inferno')
    axes[1].set_title("Profundidad Predicha")
    axes[1].axis('off')
    fig.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Profundidad ground truth
    im_gt = axes[2].imshow(gt_depth, cmap='inferno')
    axes[2].set_title("Profundidad Ground Truth")
    axes[2].axis('off')
    fig.colorbar(im_gt, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Guardar figura
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fijar semillas para reproducibilidad
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar el modelo
    model = load_model(args.model_path)
    model.to(device)
    model.eval()
    
    # Cargar dataset de prueba
    try:
        test_dataset = ThermalDepthDataset(root_dir=args.data_dir, split='test', transform=None)
        logger.info(f"Dataset de test cargado con {len(test_dataset)} muestras")
    except Exception:
        logger.warning("No se pudo cargar el dataset con split='test', intentando cargar directorios que contengan 'test'")
        test_dataset = ThermalDepthDataset(root_dir=args.data_dir, transform=None)
        # Filtrar solo subdirectorios con 'test' en el nombre
        test_samples = [s for s in test_dataset.samples if 'test' in s['subdir']]
        if not test_samples:
            logger.error("No se encontraron muestras de test")
            return
        test_dataset.samples = test_samples
        logger.info("Dataset de test filtrado con {} muestras".format(len(test_dataset)))
    
    # Seleccionar una fracción aleatoria del conjunto de test
    test_size = len(test_dataset)
    subset_size = int(test_size * args.test_fraction)
    logger.info("Usando {} muestras ({:.1f}%) del conjunto de test".format(subset_size, args.test_fraction*100))
    
    indices = random.sample(range(test_size), subset_size)
    test_subset = Subset(test_dataset, indices)
    
    # Crear dataloader
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Preparar para métricas y resultados
    all_metrics = []
    sample_count = 0
    
    # Para guardar nombres de archivo y predicciones
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluando")):
            # Preparar entradas
            left, right, depth = prepare_inputs(batch, device)
            
            try:
                # Forward pass
                outputs = model(left, right)
                
                # Obtener predicción de profundidad (usar stereo o mono según lo que esté disponible)
                if "stereo_depth" in outputs and outputs["stereo_depth"] is not None:
                    pred_depth = outputs["stereo_depth"]
                    depth_type = "stereo"
                elif "mono_depth" in outputs and outputs["mono_depth"] is not None:
                    pred_depth = outputs["mono_depth"]
                    depth_type = "mono"
                else:
                    # Fallback a la primera salida disponible
                    first_key = list(outputs.keys())[0]
                    pred_depth = outputs[first_key]
                    depth_type = first_key
                    
                logger.info("Usando mapa de profundidad de tipo: {}".format(depth_type))
            except Exception as e:
                logger.error("Error en forward pass: {}".format(e))
                continue
            
            # Calcular métricas por imagen
            for i in range(pred_depth.size(0)):
                metrics = calculate_metrics(pred_depth[i:i+1], depth[i:i+1])
                all_metrics.append(metrics)
                
                # Guardar nombre de archivo y predicción
                filename = batch['filename'][i] if isinstance(batch, dict) else f"sample_{sample_count:05d}"
                subdir = batch['subdir'][i] if isinstance(batch, dict) else "unknown"
                key = f"{subdir}/{filename}"
                
                predictions[key] = {
                    "metrics": metrics,
                    "depth_min": float(pred_depth[i].min().item()),
                    "depth_max": float(pred_depth[i].max().item()),
                    "depth_mean": float(pred_depth[i].mean().item())
                }
                
                # Visualizar y guardar algunas imágenes si se solicita
                if args.save_images:
                    if sample_count % max(10, subset_size // 20) == 0:  # Guardar ~20 imágenes
                        vis_path = os.path.join(args.output_dir, f"depth_vis_{sample_count:05d}.png")
                        visualize_depth(left[i], pred_depth[i], depth[i], vis_path)
                
                sample_count += 1
    
    # Calcular estadísticas globales
    global_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    
    # Guardar resultados
    results = {
        "model_path": args.model_path,
        "test_samples": sample_count,
        "global_metrics": global_metrics,
        "predictions": predictions
    }
    
    results_path = os.path.join(args.output_dir, "challenge_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Imprimir métricas
    logger.info("Métricas globales:")
    for key, value in global_metrics.items():
        logger.info("  {}: {:.4f}".format(key, value))
    
    logger.info("\nResultados guardados en {}".format(results_path))
    if args.save_images:
        logger.info("Visualizaciones guardadas en {}".format(args.output_dir))

if __name__ == "__main__":
    main()
