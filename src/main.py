#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from src.utils.data_loader import load_train_dataset, load_test_dataset, get_dataset_stats

def main():
    """
    Script principal para cargar y mostrar información sobre los conjuntos de datos.
    """
    parser = argparse.ArgumentParser(description='Cargador de datos para imágenes térmicas y de profundidad')
    parser.add_argument('--data_dir', type=str, default='raw', help='Directorio raíz de los datos')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño del lote')
    parser.add_argument('--num_workers', type=int, default=4, help='Número de workers para la carga de datos')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', 
                        help='Modo de carga: train, test, o ambos')
    args = parser.parse_args()
    
    # Verificar si el directorio de datos existe
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: El directorio de datos '{data_dir}' no existe.")
        return
    
    # Cargar datos según el modo elegido
    if args.mode == 'train' or args.mode == 'both':
        print("Cargando conjunto de datos de entrenamiento...")
        train_loader = load_train_dataset(
            root_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        if train_loader:
            print(f"Conjunto de entrenamiento cargado con {len(train_loader.dataset)} muestras")
            stats = get_dataset_stats(train_loader.dataset)
            print(f"Subdirectorios incluidos: {stats['subdirs']}")
            
            # Mostrar información de una muestra
            sample_batch = next(iter(train_loader))
            print("\nEjemplo de lote:")
            print(f"Forma de imagen térmica izquierda: {sample_batch['left'].shape}")
            print(f"Forma de imagen térmica derecha: {sample_batch['right'].shape}")
            print(f"Forma de imagen de profundidad: {sample_batch['depth'].shape}")
    
    if args.mode == 'test' or args.mode == 'both':
        print("\nCargando conjunto de datos de prueba...")
        test_loader = load_test_dataset(
            root_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        if test_loader:
            print(f"Conjunto de prueba cargado con {len(test_loader.dataset)} muestras")
            stats = get_dataset_stats(test_loader.dataset)
            print(f"Subdirectorios incluidos: {stats['subdirs']}")
            
            # Mostrar información de una muestra
            sample_batch = next(iter(test_loader))
            print("\nEjemplo de lote:")
            print(f"Forma de imagen térmica izquierda: {sample_batch['left'].shape}")
            print(f"Forma de imagen térmica derecha: {sample_batch['right'].shape}")
            print(f"Forma de imagen de profundidad: {sample_batch['depth'].shape}")

if __name__ == "__main__":
    main()
