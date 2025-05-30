#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.data_loader import load_test_dataset
from src.models import get_model, prepare_inputs

def visualize_predictions(model, test_loader, device, save_dir="results", num_samples=5):
    """
    Visualiza las predicciones del modelo comparadas con las profundidades reales.
    
    Args:
        model: El modelo entrenado.
        test_loader: Cargador de datos de prueba.
        device: Dispositivo para inferencia (CPU o GPU).
        save_dir: Directorio donde guardar las visualizaciones.
        num_samples: Número de muestras a visualizar.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Preparar datos
            left, right, depth_true = prepare_inputs(batch, device)
            
            # Obtener predicciones
            depth_pred = model(left, right)
            
            # Preparar para visualización
            left_img = left[0].permute(1, 2, 0).cpu().numpy()
            right_img = right[0].permute(1, 2, 0).cpu().numpy()
            depth_true_img = depth_true[0].squeeze().cpu().numpy()
            depth_pred_img = depth_pred[0].squeeze().cpu().numpy()
            
            # Normalizar imágenes si es necesario
            left_img = np.clip(left_img, 0, 1)
            right_img = np.clip(right_img, 0, 1)
            
            # Crear visualización
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            axes[0, 0].imshow(left_img)
            axes[0, 0].set_title('Imagen Térmica Izquierda')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(right_img)
            axes[0, 1].set_title('Imagen Térmica Derecha')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(depth_true_img, cmap='viridis')
            axes[1, 0].set_title('Profundidad Real')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(depth_pred_img, cmap='viridis')
            axes[1, 1].set_title('Profundidad Predicha')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i+1}.png"))
            plt.close(fig)
            
            # También guardar la diferencia entre la predicción y la verdad
            fig, ax = plt.subplots(figsize=(6, 6))
            diff = np.abs(depth_pred_img - depth_true_img)
            im = ax.imshow(diff, cmap='hot')
            ax.set_title('Error Absoluto')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"error_{i+1}.png"))
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualiza las predicciones del modelo")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al modelo entrenado")
    parser.add_argument("--data_dir", type=str, default="raw", help="Directorio raíz donde se encuentran los datos")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamaño del lote (debe ser 1 para visualización)")
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "resnet", "unet"], 
                        help="Tipo de modelo a visualizar (basic, resnet, unet)")
    parser.add_argument("--initial_filters", type=int, default=64, help="Número inicial de filtros del modelo")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Tasa de dropout del modelo")
    parser.add_argument("--save_dir", type=str, default="results", help="Directorio donde guardar resultados")
    parser.add_argument("--num_samples", type=int, default=5, help="Número de muestras a visualizar")
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo
    model = get_model(
        model_name=args.model_type,
        initial_filters=args.initial_filters,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Modelo cargado desde: {args.model_path}")
    
    # Cargar datos de prueba
    test_loader = load_test_dataset(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=1
    )
    
    print(f"Conjunto de prueba cargado con {len(test_loader.dataset)} muestras")
    
    # Visualizar predicciones
    visualize_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        num_samples=args.num_samples
    )
    
    print(f"Visualizaciones guardadas en: {args.save_dir}")


if __name__ == "__main__":
    main()
