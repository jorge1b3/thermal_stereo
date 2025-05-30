#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import glob

class ThermalDepthDataset(Dataset):
    """
    Dataset para cargar imágenes térmicas (izquierda y derecha) y de profundidad.
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directorio raíz donde se encuentran los datos.
            split (str): 'train', 'test' o 'val'. 
            transform (callable, optional): Transformaciones opcionales que se pueden aplicar a las imágenes.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Identificar los subdirectorios que corresponden al split especificado
        self.subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and split in d]
        
        # Si no hay directorios con el split especificado pero hay uno sin sufijo (como 'frick_1'),
        # lo usamos solo para 'train'
        if not self.subdirs and split == 'train':
            self.subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and 
                           not any(s in d for s in ['_test', '_train', '_val'])]
        
        self.samples = []
        
        # Para cada subdirectorio, encontrar todas las imágenes disponibles
        for subdir in self.subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            
            # Obtener archivos de profundidad
            depth_dir = os.path.join(subdir_path, 'depth_filtered')
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
            
            # Para cada archivo de profundidad, verificar que existan las imágenes térmicas correspondientes
            for depth_file in depth_files:
                filename = os.path.basename(depth_file)
                
                left_file = os.path.join(subdir_path, 'img_left', filename)
                right_file = os.path.join(subdir_path, 'img_right', filename)
                
                # Solo agregar si existen ambos archivos térmicos
                if os.path.exists(left_file) and os.path.exists(right_file):
                    self.samples.append({
                        'depth': depth_file,
                        'left': left_file,
                        'right': right_file,
                        'subdir': subdir
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.samples[idx]
        
        # Cargar imágenes
        depth_img = Image.open(sample['depth'])
        left_img = Image.open(sample['left'])
        right_img = Image.open(sample['right'])
        
        # Convertir a numpy arrays
        depth_array = np.array(depth_img)
        left_array = np.array(left_img)
        right_array = np.array(right_img)
        
        # Crear diccionario con las imágenes
        data = {
            'depth': depth_array,
            'left': left_array,
            'right': right_array,
            'subdir': sample['subdir'],
            'filename': os.path.basename(sample['depth'])
        }
        
        # Aplicar transformaciones si se proporcionan
        if self.transform:
            data = self.transform(data)
        
        return data
