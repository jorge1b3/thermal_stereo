#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo unificado para estimación de profundidad a partir de imágenes térmicas.
Implementa un modelo basado en NeWCRF (Neural Window Fully Connected Conditional Random Field)
que puede funcionar tanto en modo monocular como estéreo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WindowAttention(nn.Module):
    """
    Módulo de atención con ventana deslizante para implementar el mecanismo de NeWCRF.
    """
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Capas lineales para query, key, value
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # Proyección final
        self.proj = nn.Linear(dim, dim)
        
        # Position embedding para la atención relativa
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Índices para acceder al position embedding
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        
        # Proyecciones lineales y reshape
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Atención
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Añadir position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        # Aplicar atención al valor
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class NeWCRFBlock(nn.Module):
    """
    Bloque NeWCRF (Neural Window Fully Connected CRF)
    """
    def __init__(self, dim, window_size=8, num_heads=8, mlp_ratio=4.0):
        super(NeWCRFBlock, self).__init__()
        self.dim = dim
        self.window_size = window_size
        
        # Normalización para la entrada
        self.norm1 = nn.LayerNorm(dim)
        
        # Atención de ventana
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        # Normalización para MLP
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP para procesar el resultado de la atención
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        # Proyección para el potencial unario
        self.unary_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, feature):
        # x: resultado de predicción anterior o característica de entrada inicial
        # feature: característica concatenada (puede incluir volumen de costos)
        
        # Calcular potencial unario
        unary_potential = self.unary_proj(x)
        
        # Procesamiento de ventana para atención
        H, W = feature.shape[-2:]
        feature_window = self._window_partition(feature.flatten(2).transpose(-1, -2), self.window_size)
        
        # Calcular potencial pairwise a través de la atención
        norm_feature = self.norm1(feature_window)
        pairwise_potential = self.attn(norm_feature)
        
        # Combinar potenciales
        x = unary_potential + self._window_reverse(pairwise_potential, self.window_size, H, W)
        
        # Procesamiento final MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _window_partition(self, x, window_size):
        """Particiona la imagen en ventanas"""
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.view(B, H, W, C)
        
        # Acolchado si es necesario
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_padded, W_padded = H + pad_h, W + pad_w
        
        # Reorganizar para ventanas
        x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows
    
    def _window_reverse(self, windows, window_size, H, W):
        """Revierte la partición de ventanas a imagen completa"""
        # Calculamos las dimensiones acolchadas
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_padded, W_padded = H + pad_h, W + pad_w
        
        B = int(windows.shape[0] // ((H_padded // window_size) * (W_padded // window_size)))
        x = windows.view(B, H_padded // window_size, W_padded // window_size, 
                         window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
        
        # Quitar el acolchado si se aplicó
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        return x.view(B, H * W, -1)


class SwinTransformerEncoder(nn.Module):
    """
    Encoder basado en Swin Transformer simplificado para extracción de características
    """
    def __init__(self, in_channels, base_filters=64, depths=[2, 2, 6, 2]):
        super(SwinTransformerEncoder, self).__init__()
        
        # Capa inicial de convolución
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        # Bloques Swin para diferentes escalas
        self.stages = nn.ModuleList()
        num_features = [base_filters, base_filters*2, base_filters*4, base_filters*8]
        
        for i in range(4):
            # Downsample excepto para el primer bloque
            downsample = (i > 0)
            in_dim = num_features[i-1] if i > 0 else base_filters
            out_dim = num_features[i]
            
            if downsample:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.GELU()
                ))
            
            # Bloques NeWCRF (simplificado como bloques de convolucion)
            layer = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.GELU()
            )
            self.stages.append(layer)
    
    def forward(self, x):
        features = []
        
        # Embedding inicial
        x = self.patch_embed(x)  # 1/4 escala
        features.append(x)
        
        # Pasar por los bloques
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i % 2 == 1:  # Guardar feature maps después de cada etapa de procesamiento
                features.append(x)
        
        return features


class PyramidPoolingModule(nn.Module):
    """
    Módulo de agrupación de pirámides (PPM) para agregar información contextual global
    """
    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.sizes = [1, 2, 3, 6]
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for size in self.sizes
        ])
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.size()[2:]
        
        outputs = []
        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
            outputs.append(out)
        
        outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        return self.conv_out(outputs)


class CostVolumeBuilder(nn.Module):
    """
    Constructor de volumen de costos para estimación de disparidad estéreo
    """
    def __init__(self, max_disp=192):
        super(CostVolumeBuilder, self).__init__()
        self.max_disp = max_disp
    
    def forward(self, feat_l, feat_r=None):
        B, C, H, W = feat_l.shape
        
        # Si no hay imagen derecha, crear un volumen de costos lleno de ceros (caso monocular)
        if feat_r is None:
            return torch.zeros(B, self.max_disp//4, H, W, device=feat_l.device)
        
        # Crear volumen de costos de correlación
        cost_volume = []
        
        for d in range(0, self.max_disp//4):
            if d > 0:
                # Desplazar la característica derecha por disparidad d
                feat_r_shifted = F.pad(
                    feat_r[:, :, :, :-d], (d, 0, 0, 0), mode="constant", value=0
                )
            else:
                feat_r_shifted = feat_r
            
            # Calcular correlación para esta disparidad
            corr = torch.mean(feat_l * feat_r_shifted, dim=1, keepdim=True)
            cost_volume.append(corr)
        
        # Apilar para formar volumen de costos 3D [B, D, H, W]
        cost_volume = torch.cat(cost_volume, dim=1)
        
        return cost_volume


class DisparityPredictor(nn.Module):
    """
    Predictor de disparidad/profundidad inversa
    """
    def __init__(self, in_channels, max_disp=192):
        super(DisparityPredictor, self).__init__()
        self.max_disp = max_disp
        
        # Convoluciones para predicción
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Conv2d(in_channels//2, self.max_disp//4, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        prob_volume = self.conv2(x)
        
        # Aplicar softmax a lo largo de la dimensión de disparidad
        prob_volume = F.softmax(prob_volume, dim=1)
        
        # Calcular predicción como suma ponderada
        disp_range = torch.arange(0, self.max_disp//4, device=x.device, dtype=torch.float)
        disp_range = disp_range.view(1, -1, 1, 1)
        
        pred = torch.sum(prob_volume * disp_range, dim=1, keepdim=True)
        
        # Escalar a rango completo de disparidad
        pred = pred * 4.0
        
        return pred, prob_volume


class UnifiedDepthModel(nn.Module):
    """
    Modelo unificado para estimación de profundidad monocular y estéreo basado en NeWCRF
    """
    def __init__(self, in_channels=3, base_filters=64, max_disp=192):
        super(UnifiedDepthModel, self).__init__()
        self.max_disp = max_disp
        self.scales = 4
        
        # Encoder para extracción de características (se aplica tanto a la imagen izquierda como a la derecha)
        self.encoder = SwinTransformerEncoder(in_channels, base_filters=base_filters//2)
        
        # Pyramid Pooling Module para contexto global
        num_features = [base_filters//2, base_filters, base_filters*2, base_filters*4]
        self.ppm = PyramidPoolingModule(num_features[3], num_features[3])
        
        # Constructor de volumen de costos
        self.cost_volume_builders = nn.ModuleList([
            CostVolumeBuilder(max_disp=max_disp // (2**i)) for i in range(self.scales)
        ])
        
        # Bloques NeWCRF para cada escala
        self.newcrf_blocks = nn.ModuleList()
        for i in range(self.scales):
            scale_features = num_features[i]
            layers = nn.ModuleList([
                NeWCRFBlock(dim=scale_features, window_size=8, num_heads=8),
                NeWCRFBlock(dim=scale_features, window_size=8, num_heads=8)
            ])
            self.newcrf_blocks.append(layers)
        
        # Predictores de disparidad/profundidad para cada escala
        self.disp_predictors = nn.ModuleList([
            DisparityPredictor(num_features[i], max_disp=max_disp // (2**i)) for i in range(self.scales)
        ])
        
        # Capas de fusión para características + volume de costos
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features[i] + (max_disp // (2**i) // 4), num_features[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features[i]),
                nn.ReLU(inplace=True)
            ) for i in range(self.scales)
        ])
        
        # Capas de upsampling para predicciones a escala completa
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=2**(i+2), stride=2**(i+2), padding=0),
                nn.ReLU(inplace=True)
            ) for i in range(self.scales)
        ])
    
    def forward(self, left, right=None):
        """
        Args:
            left: Imagen térmica izquierda [B, C, H, W]
            right: Imagen térmica derecha [B, C, H, W] o None para el caso monocular
        """
        # Reducir tamaño a la mitad para acelerar entrenamiento
        orig_size = (left.shape[2], left.shape[3])
        left = F.interpolate(left, scale_factor=0.5, mode='bilinear', align_corners=True)
        if right is not None:
            right = F.interpolate(right, scale_factor=0.5, mode='bilinear', align_corners=True)
        
        # Extraer características mediante el encoder compartido
        left_features = self.encoder(left)
        right_features = None if right is None else self.encoder(right)
        
        # Aplicar PPM a la última escala
        left_features[3] = self.ppm(left_features[3])
        if right_features is not None:
            right_features[3] = self.ppm(right_features[3])
        
        # Almacenar predicciones de cada escala
        disp_preds = []
        prob_volumes = []
        
        # Procesamiento en cada escala, de la más pequeña (1/32) a la más grande (1/4)
        for scale in range(self.scales-1, -1, -1):
            # Obtener características de la escala actual
            feat_l = left_features[scale]
            feat_r = None if right_features is None else right_features[scale]
            
            # Construir volumen de costos
            cost_volume = self.cost_volume_builders[scale](feat_l, feat_r)
            
            # Fusionar características con volumen de costos
            B, C, H, W = feat_l.shape
            feat_with_cost = torch.cat([feat_l, cost_volume], dim=1)
            feat_fused = self.fusion_layers[scale](feat_with_cost)
            
            # Procesar con bloques NeWCRF
            # Convertir a formato para NeWCRF
            feat_fused_reshaped = feat_fused.flatten(2).transpose(-1, -2)  # [B, H*W, C]
            x = feat_fused_reshaped
            
            # Pasar por bloques NeWCRF
            for block in self.newcrf_blocks[scale]:
                x = block(x, feat_fused)
            
            # Volver a formato de imagen
            x = x.transpose(-1, -2).view(B, C, H, W)
            
            # Predecir disparidad/profundidad
            disp, prob = self.disp_predictors[scale](x)
            
            # Upscale para tener la misma resolución que la entrada original
            disp_full_res = self.upsampling[scale](disp)
            
            # Redimensionar al tamaño original
            disp_full_res = F.interpolate(disp_full_res, size=orig_size, mode='bilinear', align_corners=True)
            
            disp_preds.append(disp_full_res)
            prob_volumes.append(prob)
        
        return {
            'mono_depth': disp_preds[0],  # Predicción en escala 1/4 (la más detallada)
            'stereo_depth': disp_preds[0],
            'multi_scale_mono_depth': disp_preds,
            'multi_scale_stereo_depth': disp_preds,
            'prob_volumes': prob_volumes
        }


def prepare_inputs(batch, device):
    """
    Prepara los datos de entrada para el modelo.
    
    Args:
        batch: Puede ser un diccionario con claves 'left', 'right', 'depth' o una tupla (left, right, depth)
        device: dispositivo donde se moverán los tensores
    
    Returns:
        left, right, depth: Tensores procesados para el modelo
    """
    # Determinar si batch es un diccionario o una tupla
    if isinstance(batch, dict):
        left = batch['left']
        right = batch.get('right', None)  # Podría ser None en caso monocular
        depth = batch['depth']
    else:
        left, right, depth = batch
    
    # Asegurar que son tensores y moverlos al device
    left = left.to(device) if torch.is_tensor(left) else torch.tensor(left, dtype=torch.float32).to(device)
    
    if right is not None:
        right = right.to(device) if torch.is_tensor(right) else torch.tensor(right, dtype=torch.float32).to(device)
    
    depth = depth.to(device) if torch.is_tensor(depth) else torch.tensor(depth, dtype=torch.float32).to(device)
    
    # Reorganizar dimensiones para formato de imagen (B, C, H, W)
    if len(left.shape) == 4 and left.shape[3] in [1, 3]:  # Si es [B, H, W, C]
        left = left.permute(0, 3, 1, 2)
    elif len(left.shape) == 3:  # Si es [B, H, W]
        left = left.unsqueeze(1)
        # Si el modelo espera 3 canales pero la imagen tiene 1, replicamos el canal
        left = left.repeat(1, 3, 1, 1)
    
    if right is not None:
        if len(right.shape) == 4 and right.shape[3] in [1, 3]:  # Si es [B, H, W, C]
            right = right.permute(0, 3, 1, 2)
        elif len(right.shape) == 3:  # Si es [B, H, W]
            right = right.unsqueeze(1)
            # Si el modelo espera 3 canales pero la imagen tiene 1, replicamos el canal
            right = right.repeat(1, 3, 1, 1)
    
    depth = depth.unsqueeze(1) if len(depth.shape) == 3 else depth
    
    # Convertir a float32 antes de cualquier operación
    left = left.float()
    if right is not None:
        right = right.float()
    depth = depth.float()
    
    # Normalizar si es necesario (0-65535 -> 0-1 para imágenes uint16)
    left = left / 65535.0 if left.max() > 1.0 else left
    if right is not None:
        right = right / 65535.0 if right.max() > 1.0 else right
    depth = depth / 65535.0 if depth.max() > 1.0 else depth
    
    return left, right, depth


def smooth_l1_loss(pred, target, mask=None):
    """
    Función de pérdida Smooth L1.
    
    Args:
        pred: predicción del modelo
        target: valor objetivo (ground truth)
        mask: máscara opcional para aplicar en áreas válidas
    """
    loss = F.smooth_l1_loss(pred, target, reduction='none')
    
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)
    else:
        return loss.mean()


def get_unified_depth_model(initial_filters=64, max_disp=192):
    """
    Crea un nuevo modelo unificado de estimación de profundidad.
    
    Args:
        initial_filters: número de filtros iniciales para el modelo
        max_disp: disparidad máxima para el volumen de costos
    
    Returns:
        modelo: Una instancia del modelo UnifiedDepthModel
    """
    return UnifiedDepthModel(in_channels=3, base_filters=initial_filters, max_disp=max_disp)
