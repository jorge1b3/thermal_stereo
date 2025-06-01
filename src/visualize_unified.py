#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para visualizar los resultados del modelo unificado de estimación de profundidad.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

from src.models.unified_depth_model import UnifiedDepthModel, prepare_inputs
from src.data.dataset import ThermalDepthDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizar resultados del modelo unificado"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Ruta al modelo guardado"
    )
    parser.add_argument(
        "--data_dir", type=str, default="raw", help="Directorio de datos"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split a visualizar (test, val)"
    )
    parser.add_argument(
        "--initial_filters",
        type=int,
        default=32,
        help="Número de filtros iniciales del modelo",
    )
    parser.add_argument(
        "--max_disp", type=int, default=192, help="Disparidad máxima del modelo"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directorio para guardar visualizaciones",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Número de muestras a visualizar"
    )
    parser.add_argument(
        "--use_super_resolution", action="store_true", default=True,
        help="Usar módulos de super-resolución y refinamiento"
    )
    parser.add_argument(
        "--use_hybrid_refinement", action="store_true", default=True,
        help="Usar módulo híbrido de refinamiento ViT-CNN"
    )

    return parser.parse_args()


def create_depth_colormap():
    """Crear un colormap para visualizar mapas de profundidad"""
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]  # Azul a rojo
    return LinearSegmentedColormap.from_list("depth_colormap", colors, N=256)


def visualize_predictions(model, dataloader, device, args):
    """Visualizar predicciones del modelo"""
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Colormap para visualizar profundidad
    cmap = create_depth_colormap()

    # Poner el modelo en modo evaluación
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:
                break

            # Preparar datos para el modelo
            left, right, depth = prepare_inputs(batch, device)

            # Obtener predicciones
            outputs = model(left, right)  # Predicción estéreo
            outputs_mono = model(left, None)  # Predicción monocular

            # Obtener predicciones
            depth_stereo = outputs["stereo_depth"]
            depth_mono = outputs_mono["mono_depth"]

            # Convertir a numpy para visualización (usando el primer elemento del batch)
            left_np = left[0].permute(1, 2, 0).cpu().numpy()
            right_np = right[0].permute(1, 2, 0).cpu().numpy()
            gt_depth_np = depth[0, 0].cpu().numpy()
            pred_stereo_np = depth_stereo[0, 0].cpu().numpy()
            pred_mono_np = depth_mono[0, 0].cpu().numpy()

            # Normalizar para visualización
            left_np = (left_np - left_np.min()) / (left_np.max() - left_np.min() + 1e-8)
            right_np = (right_np - right_np.min()) / (
                right_np.max() - right_np.min() + 1e-8
            )

            # Normalizar mapas de profundidad para visualización
            gt_depth_norm = (gt_depth_np - gt_depth_np.min()) / (
                gt_depth_np.max() - gt_depth_np.min() + 1e-8
            )
            pred_stereo_norm = (pred_stereo_np - pred_stereo_np.min()) / (
                pred_stereo_np.max() - pred_stereo_np.min() + 1e-8
            )
            pred_mono_norm = (pred_mono_np - pred_mono_np.min()) / (
                pred_mono_np.max() - pred_mono_np.min() + 1e-8
            )

            # Calcular error entre predicción y ground truth
            error_stereo = np.abs(pred_stereo_norm - gt_depth_norm)
            error_mono = np.abs(pred_mono_norm - gt_depth_norm)

            # Crear figura para visualización
            plt.figure(figsize=(15, 10))

            # Primera fila: Imágenes térmicas y ground truth
            plt.subplot(3, 3, 1)
            plt.title("Imagen Térmica Izquierda")
            plt.imshow(left_np, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 2)
            plt.title("Imagen Térmica Derecha")
            plt.imshow(right_np, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 3, 3)
            plt.title("Profundidad Ground Truth")
            plt.imshow(gt_depth_norm, cmap=cmap)
            plt.colorbar()
            plt.axis("off")

            # Segunda fila: Profundidad predicha estéreo
            plt.subplot(3, 3, 4)
            plt.title("Profundidad Predicha Estéreo")
            plt.imshow(pred_stereo_norm, cmap=cmap)
            plt.colorbar()
            plt.axis("off")

            plt.subplot(3, 3, 5)
            plt.title("Error Estéreo")
            plt.imshow(error_stereo, cmap="hot")
            plt.colorbar()
            plt.axis("off")

            # Tercera fila: Profundidad predicha monocular
            plt.subplot(3, 3, 7)
            plt.title("Profundidad Predicha Monocular")
            plt.imshow(pred_mono_norm, cmap=cmap)
            plt.colorbar()
            plt.axis("off")

            plt.subplot(3, 3, 8)
            plt.title("Error Monocular")
            plt.imshow(error_mono, cmap="hot")
            plt.colorbar()
            plt.axis("off")

            # Calcular métricas
            rmse_stereo = np.sqrt(np.mean((pred_stereo_norm - gt_depth_norm) ** 2))
            rmse_mono = np.sqrt(np.mean((pred_mono_norm - gt_depth_norm) ** 2))

            plt.subplot(3, 3, 6)
            plt.title("Métricas")
            plt.text(0.1, 0.7, f"RMSE Estéreo: {rmse_stereo:.4f}", fontsize=12)
            plt.text(0.1, 0.3, f"RMSE Mono: {rmse_mono:.4f}", fontsize=12)
            plt.axis("off")

            # Obtener nombre del archivo
            filename = batch.get("filename", [f"sample_{i}.png"])[0]

            # Guardar figura
            save_path = os.path.join(args.output_dir, f"viz_{filename}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            print(f"Guardada visualización {i+1}/{args.num_samples}: {save_path}")


def main():
    """Función principal"""
    args = parse_args()

    # Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar dataset
    dataset = ThermalDepthDataset(
        root_dir=args.data_dir, split=args.split, transform=None
    )

    # Crear dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Visualizamos de uno en uno
        shuffle=True,
        num_workers=2,
    )

    # Crear modelo
    model = UnifiedDepthModel(
        in_channels=3, 
        base_filters=args.initial_filters, 
        max_disp=args.max_disp,
        use_super_resolution=args.use_super_resolution,
        use_hybrid_refinement=args.use_hybrid_refinement if hasattr(args, 'use_hybrid_refinement') else True
    )

    # Cargar pesos del modelo
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Modelo cargado desde {args.model_path}")

    # Visualizar predicciones
    visualize_predictions(model, dataloader, device, args)

    print(f"Visualizaciones guardadas en {args.output_dir}")


if __name__ == "__main__":
    main()
