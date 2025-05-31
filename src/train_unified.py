#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de entrenamiento para el modelo unificado de estimación de profundidad.
"""

import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import numpy as np
import wandb
from tqdm import tqdm

from src.models.unified_depth_model import UnifiedDepthModel, prepare_inputs, smooth_l1_loss
from src.data.dataset import ThermalDepthDataset

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo unificado de estimación de profundidad")
    
    # Parámetros de datos
    parser.add_argument('--data_dir', type=str, default='raw', help='Directorio de datos')
    parser.add_argument('--batch_size', type=int, default=4, help='Tamaño de batch')
    parser.add_argument('--num_workers', type=int, default=4, help='Número de workers para carga de datos')
    parser.add_argument('--val_split', type=float, default=0.1, help='Proporción de datos para validación')
    
    # Parámetros del modelo
    parser.add_argument('--model_name', type=str, default='unified_depth_model', help='Nombre del modelo')
    parser.add_argument('--initial_filters', type=int, default=32, help='Número de filtros iniciales')
    parser.add_argument('--max_disp', type=int, default=192, help='Disparidad máxima')
    
    # Parámetros de entrenamiento
    parser.add_argument('--num_epochs', type=int, default=20, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.0001, help='Tasa de aprendizaje')
    parser.add_argument('--output_dir', type=str, default='models', help='Directorio para guardar modelos')
    
    return parser.parse_args()

def train(model, train_loader, val_loader, optimizer, device, args):
    """
    Función principal de entrenamiento.
    """
    # Configuración de pesos para las diferentes escalas
    scale_weights = [0.5, 0.7, 0.85, 1.0]  # para 4 escalas
    
    # Crear directorio para guardar modelos si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicializar el mejor error de validación
    best_val_loss = float('inf')
    
    # Bucle de entrenamiento por épocas
    for epoch in range(args.num_epochs):
        logging.info(f"Época {epoch+1}/{args.num_epochs}")
        
        # ===== ENTRENAMIENTO =====
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader)
        for batch_idx, batch in enumerate(progress_bar):
            # Preparar los datos de entrada
            left, right, depth = prepare_inputs(batch, device)
            
            # Limpiar gradientes
            optimizer.zero_grad()
            
            # Forward pass
            output = model(left, right)
            
            # Calcular pérdida (multi-escala)
            loss = 0.0
            
            # Para cada escala, calcular la pérdida
            for i, (mono_depth, stereo_depth) in enumerate(zip(
                output['multi_scale_mono_depth'], output['multi_scale_stereo_depth'])):
                
                # Aplicar peso a la escala correspondiente
                scale_loss = (
                    smooth_l1_loss(mono_depth, depth) + 
                    smooth_l1_loss(stereo_depth, depth)
                ) * scale_weights[i]
                
                loss += scale_loss
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            # Actualizar estadísticas
            train_loss += loss.item()
            
            # Actualizar barra de progreso
            progress_bar.set_description(f"Pérdida: {loss.item():.4f}")
            
            # Registrar en wandb cada 50 batches
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_loader),
                    "train_batch_loss": loss.item()
                })
        
        # Calcular pérdida media de entrenamiento
        train_loss /= len(train_loader)
        
        # ===== VALIDACIÓN =====
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validación"):
                left, right, depth = prepare_inputs(batch, device)
                
                output = model(left, right)
                
                # Calcular pérdida de validación (solo para la predicción final)
                val_loss_batch = smooth_l1_loss(output['stereo_depth'], depth)
                val_loss += val_loss_batch.item()
        
        # Calcular pérdida media de validación
        val_loss /= len(val_loader)
        
        # Registrar métricas en wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        logging.info(f"Época {epoch+1}/{args.num_epochs}, "
                    f"Pérdida entrenamiento: {train_loss:.4f}, "
                    f"Pérdida validación: {val_loss:.4f}")
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, f"{args.model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            logging.info(f"Mejor modelo guardado con pérdida de validación: {val_loss:.4f}")
        
        # Guardar checkpoint cada 5 épocas
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint guardado en época {epoch+1}")
    
    # Guardar modelo final
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    logging.info(f"Modelo final guardado en {final_model_path}")


def main():
    """Función principal para configurar y ejecutar el entrenamiento."""
    args = parse_args()
    
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"Argumentos: {args}")
    
    # Inicializar wandb para seguimiento de experimentos
    wandb.init(
        project="thermal_depth",
        name=f"{args.model_name}_unified_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}",
        config=vars(args)
    )
    
    # Crear dataset
    dataset = ThermalDepthDataset(
        root_dir=args.data_dir,
        transform=None  # Las transformaciones se hacen en prepare_inputs
    )
    
    # Dividir en conjunto de entrenamiento y validación
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logging.info(f"Conjuntos de datos: {len(train_dataset)} entrenamiento, {len(val_dataset)} validación")
    
    # Crear modelo
    model = UnifiedDepthModel(
        in_channels=3,  # 3 canales para imágenes térmicas (replicadas)
        base_filters=args.initial_filters,
        max_disp=args.max_disp
    )
    model.to(device)
    
    logging.info(f"Creado modelo unificado con {args.initial_filters} filtros iniciales y disparidad máxima {args.max_disp}")
    
    # Configurar optimizador
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Entrenar modelo
    train(model, train_loader, val_loader, optimizer, device, args)
    
    # Cerrar wandb
    wandb.finish()


if __name__ == "__main__":
    main()
