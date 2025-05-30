#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import ThermalDepthDataset

def load_dataset(root_dir, batch_size=8, num_workers=4, transform=None):
    """
    Carga los conjuntos de datos de entrenamiento, prueba y validación.
    
    Args:
        root_dir (str): Directorio raíz donde se encuentran los datos.
        batch_size (int): Tamaño del lote para los dataloaders.
        num_workers (int): Número de workers para la carga de datos.
        transform (callable, optional): Transformaciones a aplicar a los datos.
    
    Returns:
        dict: Diccionario con los dataloaders 'train', 'test' y 'val'.
    """
    dataloaders = {}
    
    # Crear conjuntos de datos para entrenamiento, prueba y validación
    for split in ['train', 'test', 'val']:
        dataset = ThermalDepthDataset(
            root_dir=root_dir,
            split=split,
            transform=transform
        )
        
        if len(dataset) > 0:
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),  # Solo mezclar datos de entrenamiento
                num_workers=num_workers,
                pin_memory=True
            )
    
    return dataloaders

def load_train_dataset(root_dir, batch_size=8, num_workers=4, transform=None):
    """
    Carga solo el conjunto de datos de entrenamiento.
    
    Args:
        root_dir (str): Directorio raíz donde se encuentran los datos.
        batch_size (int): Tamaño del lote para los dataloaders.
        num_workers (int): Número de workers para la carga de datos.
        transform (callable, optional): Transformaciones a aplicar a los datos.
    
    Returns:
        DataLoader: Dataloader para el conjunto de entrenamiento.
    """
    train_dataset = ThermalDepthDataset(
        root_dir=root_dir,
        split='train',
        transform=transform
    )
    
    if len(train_dataset) == 0:
        print("¡Advertencia! No se encontraron datos de entrenamiento.")
        return None
    
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

def load_test_dataset(root_dir, batch_size=8, num_workers=4, transform=None):
    """
    Carga solo el conjunto de datos de prueba.
    
    Args:
        root_dir (str): Directorio raíz donde se encuentran los datos.
        batch_size (int): Tamaño del lote para los dataloaders.
        num_workers (int): Número de workers para la carga de datos.
        transform (callable, optional): Transformaciones a aplicar a los datos.
    
    Returns:
        DataLoader: Dataloader para el conjunto de prueba.
    """
    test_dataset = ThermalDepthDataset(
        root_dir=root_dir,
        split='test',
        transform=transform
    )
    
    if len(test_dataset) == 0:
        print("¡Advertencia! No se encontraron datos de prueba.")
        return None
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

def get_dataset_stats(dataset):
    """
    Calcula estadísticas básicas del conjunto de datos.
    
    Args:
        dataset (ThermalDepthDataset): El conjunto de datos.
    
    Returns:
        dict: Diccionario con estadísticas del conjunto de datos.
    """
    stats = {
        'total_samples': len(dataset),
        'subdirs': set(),
    }
    
    for sample in dataset.samples:
        stats['subdirs'].add(sample['subdir'])
    
    stats['subdirs'] = list(stats['subdirs'])
    stats['num_subdirs'] = len(stats['subdirs'])
    
    return stats
