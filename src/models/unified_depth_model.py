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
        self.scale = head_dim**-0.5

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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
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
        q = (
            self.q(x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # Atención
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Añadir position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
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
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)
        )

        # Proyección para el potencial unario
        self.unary_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x, feature):
        # x: resultado de predicción anterior o característica de entrada inicial
        # feature: característica concatenada (puede incluir volumen de costos)

        try:
            # Calcular potencial unario
            unary_potential = self.unary_proj(x)

            # Procesamiento de ventana para atención
            H, W = feature.shape[-2:]

            # Asegurarse de que los datos estén en formato adecuado antes de procesar
            feature_flattened = feature.flatten(2).transpose(-1, -2)

            # Verificar que las dimensiones sean consistentes
            if feature_flattened.size(1) != H * W:
                print(
                    f"Advertencia: dimensiones inconsistentes. feature_flattened: {feature_flattened.size()}, HxW: {H*W}"
                )
                # Ajustar el tensor para que coincida con las dimensiones esperadas
                B, N, C = feature_flattened.size()
                if N > H * W:
                    feature_flattened = feature_flattened[:, : H * W, :]
                else:
                    # Rellenar con ceros si es necesario
                    padded = torch.zeros(B, H * W, C, device=feature_flattened.device)
                    padded[:, :N, :] = feature_flattened
                    feature_flattened = padded

            feature_window = self._window_partition(feature_flattened, self.window_size)

            # Calcular potencial pairwise a través de la atención
            norm_feature = self.norm1(feature_window)
            pairwise_potential = self.attn(norm_feature)

            # Combinar potenciales
            x = unary_potential + self._window_reverse(
                pairwise_potential, self.window_size, H, W
            )

            # Procesamiento final MLP
            x = x + self.mlp(self.norm2(x))

        except RuntimeError as e:
            print(f"Error en forward de NewCRFBlock: {e}")
            # En caso de error, devolver la entrada original para no interrumpir el flujo
            print("Devolviendo entrada original como fallback")
            return x

        return x

    def _window_partition(self, x, window_size):
        """Particiona la imagen en ventanas"""
        B, N, C = x.shape
        # Verificamos si N es un cuadrado perfecto
        H = int(np.sqrt(N))
        W = H
        # Si no es un cuadrado perfecto, ajustamos W para que H*W sea igual a N
        if H * W != N:
            # Buscamos factores para H y W que sean más cercanos a ser iguales
            for i in range(int(np.sqrt(N)), 0, -1):
                if N % i == 0:
                    H = i
                    W = N // i
                    break

        # Aseguramos que los datos se puedan reorganizar correctamente
        try:
            x = x.view(B, H, W, C)
        except RuntimeError:
            # Si hay un error, imprimimos información de depuración y ajustamos las dimensiones
            print(
                f"Error de forma: intentando dar forma a tensor de tamaño {x.size()} a {B}x{H}x{W}x{C}"
            )
            # Redimensionar x para que sea compatible
            x = x[:, : H * W, :].contiguous()
            x = x.view(B, H, W, C)

        # Acolchado si es necesario
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        H_padded, W_padded = H + pad_h, W + pad_w

        # Reorganizar para ventanas
        x = x.view(
            B,
            H_padded // window_size,
            window_size,
            W_padded // window_size,
            window_size,
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size * window_size, C)
        )
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        """Revierte la partición de ventanas a imagen completa"""
        # Calculamos las dimensiones acolchadas
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_padded, W_padded = H + pad_h, W + pad_w

        # Verificar y ajustar si H o W no son compatibles con window_size
        if H_padded % window_size != 0:
            H_padded = (H_padded // window_size + 1) * window_size
        if W_padded % window_size != 0:
            W_padded = (W_padded // window_size + 1) * window_size

        # Calcular B de manera segura
        try:
            B = int(
                windows.shape[0]
                // ((H_padded // window_size) * (W_padded // window_size))
            )
        except ZeroDivisionError:
            # Si hay un error en la división, usamos un valor predeterminado para B
            # Este es un caso de error que no debería ocurrir si las dimensiones son correctas
            print(
                f"Error al calcular B. H_padded={H_padded}, W_padded={W_padded}, window_size={window_size}"
            )
            B = int(windows.shape[0])  # Usar un valor seguro como respaldo
        # Reorganizar el tensor de manera segura
        try:
            x = windows.view(
                B,
                H_padded // window_size,
                W_padded // window_size,
                window_size,
                window_size,
                -1,
            )
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
        except RuntimeError as e:
            # En caso de error, imprimir información y hacer un ajuste alternativo
            print(f"Error al reorganizar ventanas: {e}")
            print(
                f"Dimensiones: B={B}, H_padded={H_padded}, W_padded={W_padded}, window_size={window_size}"
            )
            # Intentar una estrategia alternativa: redimensionar las ventanas
            C = windows.size(-1)
            windows_resized = torch.zeros(
                B * (H_padded // window_size) * (W_padded // window_size),
                window_size * window_size,
                C,
                device=windows.device,
            )
            # Copiar valores disponibles
            min_size = min(windows.size(0), windows_resized.size(0))
            windows_resized[:min_size] = windows[:min_size]

            # Intentar de nuevo con el tensor redimensionado
            x = windows_resized.view(
                B,
                H_padded // window_size,
                W_padded // window_size,
                window_size,
                window_size,
                C,
            )
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, C)

        # Quitar el acolchado si se aplicó
        # Asegurar que no intentamos acceder más allá del tamaño del tensor
        if pad_h > 0 or pad_w > 0:
            H_actual = min(H, x.size(1))
            W_actual = min(W, x.size(2))
            x = x[:, :H_actual, :W_actual, :].contiguous()

        return x.view(B, H * W, -1)


class SwinTransformerEncoder(nn.Module):
    """
    Encoder basado en Swin Transformer simplificado para extracción de características
    """

    def __init__(self, in_channels, base_filters=64, depths=[2, 2, 6, 2]):
        super(SwinTransformerEncoder, self).__init__()

        # Capa inicial de convolución con GroupNorm en lugar de BatchNorm2d
        # Reducción del stride de 4 a 2 para mantener mayor resolución
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, base_filters), base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, base_filters), base_filters),
            nn.GELU(),
        )

        # Bloques Swin para diferentes escalas
        self.stages = nn.ModuleList()
        num_features = [
            base_filters,
            base_filters * 2,
            base_filters * 4,
            base_filters * 8,
        ]

        for i in range(4):
            # Downsample excepto para el primer bloque
            downsample = i > 0
            in_dim = num_features[i - 1] if i > 0 else base_filters
            out_dim = num_features[i]

            if downsample:
                # Usar GroupNorm en lugar de BatchNorm2d
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                        nn.GroupNorm(min(16, out_dim), out_dim),
                        nn.GELU(),
                    )
                )

            # Bloques NeWCRF (simplificado como bloques de convolucion)
            # Usar GroupNorm en lugar de BatchNorm2d
            layer = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(16, out_dim), out_dim),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(16, out_dim), out_dim),
                nn.GELU(),
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
            if (
                i % 2 == 1
            ):  # Guardar feature maps después de cada etapa de procesamiento
                features.append(x)

        return features


class PyramidPoolingModule(nn.Module):
    """
    Módulo de agrupación de pirámides (PPM) para agregar información contextual global
    """

    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.sizes = [1, 2, 3, 6]

        # Crear branches usando GroupNorm en lugar de BatchNorm2d
        self.branches = nn.ModuleList()
        for size in self.sizes:
            channels_quarter = in_channels // 4
            # Asegurar que num_groups sea compatible con channels_quarter
            num_groups = min(8, channels_quarter)
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, channels_quarter, kernel_size=1),
                # GroupNorm es más robusto con tensores pequeños y batch_size=1
                nn.GroupNorm(num_groups, channels_quarter),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # Usar GroupNorm también en la capa de salida
        num_out_groups = min(16, out_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_out_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.size()[2:]

        # Verificar que la entrada no sea demasiado pequeña
        B, C, H, W = x.shape
        if H <= 1 or W <= 1:
            # Si el tamaño espacial es muy pequeño, aplicar un procesamiento alternativo
            print(f"Advertencia: PPM recibió un tensor con forma {x.shape}, aplicando procesamiento especial")
            # Simplemente aplicar convolución 1x1 para ajustar canales
            return nn.Conv2d(C, C, kernel_size=1, bias=False).to(x.device)(x)

        outputs = []
        for branch in self.branches:
            try:
                out = branch(x)
                # Usar interpolación solo si es necesario
                if out.shape[2:] != size:
                    out = F.interpolate(out, size=size, mode="bilinear", align_corners=True)
                outputs.append(out)
            except RuntimeError as e:
                print(f"Error en rama PPM: {e}")
                # En caso de error, saltear esta rama
                continue

        # Asegurarse de que haya al menos una salida
        if not outputs:
            print("Advertencia: No hay ramas PPM exitosas, devolviendo entrada original")
            return x

        # Agregar la entrada original a las salidas
        outputs.append(x)
        
        # Concatenar todas las salidas
        outputs = torch.cat(outputs, dim=1)
        
        # Aplicar capa de convolución final
        try:
            return self.conv_out(outputs)
        except RuntimeError as e:
            print(f"Error en conv_out PPM: {e}, devolviendo entrada original")
            return x


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
            return torch.zeros(B, self.max_disp // 4, H, W, device=feat_l.device)

        # Crear volumen de costos de correlación
        cost_volume = []

        for d in range(0, self.max_disp // 4):
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

        # Convoluciones para predicción, usando GroupNorm en lugar de BatchNorm2d
        channels_half = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_half, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, channels_half), channels_half),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(in_channels // 2, self.max_disp // 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        prob_volume = self.conv2(x)

        # Aplicar softmax a lo largo de la dimensión de disparidad
        prob_volume = F.softmax(prob_volume, dim=1)

        # Calcular predicción como suma ponderada
        disp_range = torch.arange(
            0, self.max_disp // 4, device=x.device, dtype=torch.float
        )
        disp_range = disp_range.view(1, -1, 1, 1)

        pred = torch.sum(prob_volume * disp_range, dim=1, keepdim=True)

        # Escalar a rango completo de disparidad
        pred = pred * 4.0

        return pred, prob_volume


class UnifiedDepthModel(nn.Module):
    """
    Modelo unificado para estimación de profundidad monocular y estéreo basado en NeWCRF
    """

    def __init__(self, in_channels=3, base_filters=64, max_disp=192, use_super_resolution=True, use_hybrid_refinement=True):
        super(UnifiedDepthModel, self).__init__()
        self.max_disp = max_disp
        self.scales = 4
        self.use_super_resolution = use_super_resolution
        self.use_hybrid_refinement = use_hybrid_refinement

        # Encoder para extracción de características (se aplica tanto a la imagen izquierda como a la derecha)
        self.encoder = SwinTransformerEncoder(
            in_channels, base_filters=base_filters // 2
        )
        
        # Módulos opcionales de super-resolución y refinamiento
        if self.use_super_resolution:
            # Añadir módulo de super-resolución para mejorar detalles en el mapa de profundidad
            self.super_resolution = EdgeAwareSuperResolutionModule(
                in_channels=base_filters // 2,  # base_filters // 2 corresponde a num_features[0]
                scale_factor=2  # Factor de escala para la resolución final
            )
            
            # Red de refinamiento para reducir el efecto de "parches" y mejorar detalles finos
            self.refinement_network = DepthRefinementNetwork(
                in_channels=base_filters // 2  # base_filters // 2 corresponde a num_features[0]
            )
            
            # Módulo híbrido ViT-CNN para combinar imagen original con mapa de profundidad
            if self.use_hybrid_refinement:
                self.hybrid_refinement = HybridDepthRefinementModule(
                    in_channels=in_channels,  # Número de canales de la imagen original
                    depth_features=base_filters // 2  # Características del mapa de profundidad
                )

        # Pyramid Pooling Module para contexto global
        num_features = [
            base_filters // 2,
            base_filters,
            base_filters * 2,
            base_filters * 4,
        ]
        self.ppm = PyramidPoolingModule(num_features[3], num_features[3])

        # Constructor de volumen de costos
        self.cost_volume_builders = nn.ModuleList(
            [CostVolumeBuilder(max_disp=max_disp // (2**i)) for i in range(self.scales)]
        )

        # Bloques NeWCRF para cada escala
        self.newcrf_blocks = nn.ModuleList()
        for i in range(self.scales):
            scale_features = num_features[i]
            layers = nn.ModuleList(
                [
                    NeWCRFBlock(dim=scale_features, window_size=8, num_heads=8),
                    NeWCRFBlock(dim=scale_features, window_size=8, num_heads=8),
                ]
            )
            self.newcrf_blocks.append(layers)

        # Predictores de disparidad/profundidad para cada escala
        self.disp_predictors = nn.ModuleList(
            [
                DisparityPredictor(num_features[i], max_disp=max_disp // (2**i))
                for i in range(self.scales)
            ]
        )

        # Capas de fusión para características + volume de costos usando GroupNorm
        self.fusion_layers = nn.ModuleList()
        for i in range(self.scales):
            out_channels = num_features[i]
            # Determinar número de grupos para GroupNorm (como máximo 16, mínimo 1)
            num_groups = min(16, out_channels)
            
            fusion_layer = nn.Sequential(
                nn.Conv2d(
                    num_features[i] + (max_disp // (2**i) // 4),
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
            )
            self.fusion_layers.append(fusion_layer)

        # Capas de upsampling para predicciones a escala completa
        self.upsampling = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        1, 1, kernel_size=2 ** (i + 2), stride=2 ** (i + 2), padding=0
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(self.scales)
            ]
        )

    def forward(self, left, right=None):
        """
        Args:
            left: Imagen térmica izquierda [B, C, H, W]
            right: Imagen térmica derecha [B, C, H, W] o None para el caso monocular
        """
        # Trabajar directamente con el tamaño original sin reducción de escala
        
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
        for scale in range(self.scales - 1, -1, -1):
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

            # Upscale para tener la resolución completa
            # Nota: Como no hacemos downsampling inicial, solo necesitamos usar upsampling
            # para compensar el downsampling del encoder
            disp_full_res = self.upsampling[scale](disp)

            disp_preds.append(disp_full_res)
            prob_volumes.append(prob)

        # Si la super-resolución está habilitada, aplicarla
        if self.use_super_resolution:
            try:
                # Aplicar módulo de super-resolución para mejorar la calidad del mapa de profundidad
                enhanced_depth = self.super_resolution(left_features[0], disp_preds[0])
                
                # Verificar y ajustar las dimensiones si es necesario antes de refinar
                if left_features[0].shape[2:] != disp_preds[0].shape[2:]:
                    # Redimensionar el mapa de profundidad para que coincida con las características
                    disp_preds_resized = F.interpolate(
                        disp_preds[0], 
                        size=left_features[0].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    disp_preds_resized = disp_preds[0]
                    
                # Refinar el mapa de profundidad combinando múltiples resoluciones
                refined_depth = self.refinement_network(left_features[0], disp_preds_resized)
                
                # Asegurarnos de que todos los mapas de profundidad tengan el tamaño original de la entrada
                original_size = left.shape[2:]
                
                # Redimensionar los mapas de profundidad finales para que coincidan con la entrada original
                if refined_depth.shape[2:] != original_size:
                    refined_depth = F.interpolate(refined_depth, size=original_size, mode='bilinear', align_corners=True)
                    enhanced_depth = F.interpolate(enhanced_depth, size=original_size, mode='bilinear', align_corners=True)
                
                # Si el refinamiento híbrido está habilitado, aplicarlo como paso final
                if self.use_hybrid_refinement:
                    try:
                        # Usar el módulo de refinamiento híbrido ViT-CNN para combinar imagen original con profundidad
                        hybrid_refined_depth = self.hybrid_refinement(left, refined_depth)
                        
                        # Asegurar que la salida tenga las dimensiones correctas
                        if hybrid_refined_depth.shape[2:] != original_size:
                            hybrid_refined_depth = F.interpolate(
                                hybrid_refined_depth, 
                                size=original_size, 
                                mode='bilinear', 
                                align_corners=True
                            )
                        
                        # Usar el mapa de profundidad refinado por el híbrido como salida final
                        return {
                            "mono_depth": hybrid_refined_depth,  # Predicción con refinamiento híbrido
                            "stereo_depth": hybrid_refined_depth,
                            "multi_scale_mono_depth": [hybrid_refined_depth, refined_depth, enhanced_depth] + disp_preds,
                            "multi_scale_stereo_depth": [hybrid_refined_depth, refined_depth, enhanced_depth] + disp_preds,
                            "prob_volumes": prob_volumes,
                        }
                        
                    except Exception as e:
                        print(f"Error en el módulo de refinamiento híbrido: {e}. Usando solo refinamiento estándar.")
                        # Continuar con el refinamiento estándar si falla el híbrido
                
                # Si no hay refinamiento híbrido o falló, usar la versión refinada estándar
                return {
                    "mono_depth": refined_depth,  # Predicción mejorada con super-resolución
                    "stereo_depth": refined_depth,
                    "multi_scale_mono_depth": [refined_depth, enhanced_depth] + disp_preds,  # Incluimos todas las escalas
                    "multi_scale_stereo_depth": [refined_depth, enhanced_depth] + disp_preds,
                    "prob_volumes": prob_volumes,
                }
                
            except Exception as e:
                # Si hay algún error, desactivar la super-resolución para esta ejecución
                print(f"Error en los módulos de super-resolución: {e}. Usando salida básica.")
                self.use_super_resolution = False
        
        # Versión sin super-resolución (ya sea por estar desactivada o por error)
        return {
            "mono_depth": disp_preds[0],  # Predicción en escala 1/4 (la más detallada)
            "stereo_depth": disp_preds[0],
            "multi_scale_mono_depth": disp_preds,
            "multi_scale_stereo_depth": disp_preds,
            "prob_volumes": prob_volumes,
        }


class EdgeAwareSuperResolutionModule(nn.Module):
    """
    Módulo de súper-resolución con preservación de bordes para mapas de profundidad.
    Este módulo mejora la resolución del mapa de profundidad mientras preserva los bordes y detalles finos.
    Implementación simplificada y robusta para evitar errores de dimensiones.
    """

    def __init__(self, in_channels, scale_factor=2):
        super(EdgeAwareSuperResolutionModule, self).__init__()
        self.scale_factor = scale_factor
        
        # Número de grupos para GroupNorm
        num_groups = min(8, in_channels)
        
        # Detector de bordes simplificado
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Red principal de super-resolución: más simple y robusta
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, in_channels*4), in_channels*4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)  # Aumenta la resolución espacial x2
        )
        
        # Capa de refinamiento final con menos parámetros
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, in_channels // 2), in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x, depth_map):
        """
        Args:
            x: Características extraídas del encoder [B, C, H, W]
            depth_map: Mapa de profundidad a mejorar [B, 1, H, W]
        """
        # Verificar dimensiones de entrada
        B, C, H, W = x.shape
        _, _, Hd, Wd = depth_map.shape
        
        # Si las dimensiones son muy pequeñas, simplemente devolver depth_map
        if H <= 1 or W <= 1:
            return depth_map
            
        # Detectar bordes
        edge_features = self.edge_detector(x)
        x = x + edge_features  # Conexión residual
        
        # Super-resolución de las características
        up_features = self.up_conv(x)
        
        # Redimensionar depth_map para que coincida con up_features
        if up_features.shape[2:] != depth_map.shape[2:]:
            up_depth = F.interpolate(
                depth_map, 
                size=up_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        else:
            up_depth = depth_map
        
        # Concatenar características y profundidad
        concat_features = torch.cat([up_features, up_depth], dim=1)
        
        # Refinar el mapa de profundidad
        refinement = self.refine(concat_features)
        
        # Conexión residual, pero evitando la suma directa si las dimensiones no coinciden
        if refinement.shape == up_depth.shape:
            refined_depth = up_depth + refinement * 0.1  # Factor de escala para estabilidad
        else:
            # Dimensiones diferentes, solo usar el refinamiento
            refined_depth = refinement
        
        return refined_depth


class DepthRefinementNetwork(nn.Module):
    """
    Red de refinamiento de mapas de profundidad simplificada y optimizada.
    Esta implementación mejora la robustez para evitar errores dimensionales
    y reduce la complejidad para un entrenamiento más estable.
    """
    def __init__(self, in_channels):
        super(DepthRefinementNetwork, self).__init__()
        
        # Reducir número de canales para menor complejidad
        mid_channels = max(16, in_channels // 2)
        
        # Detector de bordes simplificado
        self.edge_detection = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, mid_channels), mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Capa de fusión simplificada
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels + 1, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, mid_channels), mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Capa final de predicción
        self.predict = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)
        )
        
        # Factor de inicialización para la capa final (para estabilidad)
        # Inicializar con valores pequeños para que el refinamiento sea sutil al principio
        for m in self.predict.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, depth_map):
        """
        Args:
            features: Características del encoder [B, C, H, W]
            depth_map: Mapa de profundidad inicial [B, 1, H, W]
        """
        # Verificar dimensiones
        B, C, H, W = features.shape
        _, _, Hd, Wd = depth_map.shape
        
        # Si las dimensiones son muy pequeñas, simplemente devolver depth_map
        if H <= 1 or W <= 1 or Hd <= 1 or Wd <= 1:
            return depth_map
            
        # Extracción de bordes a partir de características
        edge_features = self.edge_detection(features)
        
        # Asegurar que depth_map tenga las mismas dimensiones que edge_features
        if edge_features.shape[2:] != depth_map.shape[2:]:
            depth_resized = F.interpolate(
                depth_map, 
                size=edge_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        else:
            depth_resized = depth_map
        
        # Concatenar features con depth map
        concat_features = torch.cat([edge_features, depth_resized], dim=1)
        
        # Procesar con capas de fusión
        refined_features = self.fusion(concat_features)
        
        # Predicción final (residual)
        refinement = self.predict(refined_features)
        
        # Aplicar refinamiento con un factor de escala para estabilidad
        result = depth_resized + refinement * 0.1
        
        # Si las dimensiones originales son diferentes, redimensionar el resultado
        if result.shape[2:] != depth_map.shape[2:]:
            result = F.interpolate(
                result,
                size=depth_map.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        return result


class HybridDepthRefinementModule(nn.Module):
    """
    Módulo híbrido que combina características de la imagen térmica original 
    con el mapa de profundidad generado para mejorar la coherencia espacial y los bordes.
    
    Este módulo implementa un enfoque híbrido ViT-CNN donde:
    1. Extrae características de la imagen térmica original mediante CNNs
    2. Procesa el mapa de profundidad mediante un encoder específico
    3. Combina características mediante un mecanismo de atención
    4. Refina el mapa final mediante un decoder tipo U-Net para recuperar bordes y detalles
    """
    def __init__(self, in_channels=3, depth_features=32, mid_channels=64):
        super(HybridDepthRefinementModule, self).__init__()
        
        # Extractor de características de la imagen térmica
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.GELU()
        )
        
        # Extractor de características del mapa de profundidad
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, depth_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, depth_features), depth_features),
            nn.GELU(),
            nn.Conv2d(depth_features, depth_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(8, depth_features), depth_features),
            nn.GELU()
        )
        
        # Mecanismo de atención para combinar características
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels + depth_features, mid_channels, kernel_size=1),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.Sigmoid()
        )
        
        # Transformador simplificado basado en convoluciones para procesar información global
        self.transformer_block = nn.Sequential(
            nn.Conv2d(mid_channels + depth_features, mid_channels*2, kernel_size=1),
            nn.GroupNorm(min(8, mid_channels*2), mid_channels*2),
            nn.GELU(),
            # Un tipo de atención simple basada en convoluciones
            nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=3, padding=1, groups=mid_channels*2),
            nn.GroupNorm(min(8, mid_channels*2), mid_channels*2),
            nn.GELU(),
            nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=1),
            nn.GroupNorm(min(8, mid_channels*2), mid_channels*2),
            nn.GELU()
        )
        
        # Extractor de bordes para preservar detalles finos
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels//2, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, mid_channels//2), mid_channels//2),
            nn.GELU(),
            nn.Conv2d(mid_channels//2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder tipo U-Net para refinar y recuperar bordes - con conexiones skip
        self.decoder_level1 = nn.Sequential(
            # Nivel 1
            nn.Conv2d(mid_channels*2 + mid_channels + 1, mid_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, mid_channels*2), mid_channels*2),
            nn.GELU()
        )
        
        self.decoder_level2 = nn.Sequential(
            # Nivel 2 - Refinamiento de bordes
            nn.Conv2d(mid_channels*2 + 1, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.GELU()
        )
        
        # Capa de fusión final
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(mid_channels + 1, mid_channels//2, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, mid_channels//2), mid_channels//2),
            nn.GELU(),
            nn.Conv2d(mid_channels//2, 1, kernel_size=1)
        )
        
        # Factor de suavizado para la conexión residual
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Inicialización cuidadosa para estabilidad
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización cuidadosa de pesos para estabilidad durante el entrenamiento"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # Para las capas de salida, inicializar con valores más pequeños
        for m in self.fusion_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, original_image, depth_map):
        """
        Refina el mapa de profundidad usando la imagen original como guía
        
        Args:
            original_image: Imagen térmica original [B, C, H, W]
            depth_map: Mapa de profundidad generado [B, 1, H, W]
        """
        # Asegurar dimensiones correctas
        B, C, H, W = original_image.shape
        _, _, Hd, Wd = depth_map.shape
        
        if H != Hd or W != Wd:
            depth_map = F.interpolate(
                depth_map, 
                size=(H, W),
                mode='bilinear',
                align_corners=True
            )
        
        # Extraer características de la imagen y del mapa de profundidad
        img_features = self.image_encoder(original_image)
        depth_features = self.depth_encoder(depth_map)
        
        # Extraer información de bordes para guiar el proceso
        edge_map = self.edge_extractor(original_image)
        
        # Calcular mapa de atención entre imagen y profundidad
        combined_features = torch.cat([img_features, depth_features], dim=1)
        attention_map = self.attention(combined_features)
        
        # Aplicar atención a las características de la imagen
        attended_features = img_features * attention_map
        
        # Procesar con el bloque transformador
        transformer_features = self.transformer_block(combined_features)
        
        # Decoder tipo U-Net con conexiones skip para recuperar detalles
        # Nivel 1
        decoder_input_1 = torch.cat([transformer_features, attended_features, edge_map], dim=1)
        decoder_features_1 = self.decoder_level1(decoder_input_1)
        
        # Nivel 2 con conexión skip desde el mapa de bordes
        decoder_input_2 = torch.cat([decoder_features_1, edge_map], dim=1)
        decoder_features_2 = self.decoder_level2(decoder_input_2)
        
        # Fusión final con el mapa de profundidad original
        fusion_input = torch.cat([decoder_features_2, depth_map], dim=1)
        refined_depth = self.fusion_layer(fusion_input)
        
        # Conexión residual con el mapa original para preservar estructura
        final_depth = depth_map + self.alpha * refined_depth
        
        return final_depth


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
        left = batch["left"]
        right = batch.get("right", None)  # Podría ser None en caso monocular
        depth = batch["depth"]
    else:
        left, right, depth = batch

    # Asegurar que son tensores y moverlos al device
    left = (
        left.to(device)
        if torch.is_tensor(left)
        else torch.tensor(left, dtype=torch.float32).to(device)
    )

    if right is not None:
        right = (
            right.to(device)
            if torch.is_tensor(right)
            else torch.tensor(right, dtype=torch.float32).to(device)
        )

    depth = (
        depth.to(device)
        if torch.is_tensor(depth)
        else torch.tensor(depth, dtype=torch.float32).to(device)
    )

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
    # Asegurarse de que pred y target tengan las mismas dimensiones espaciales
    if pred.shape[2:] != target.shape[2:]:
        print(f"Redimensionando tensores para el cálculo de pérdida: pred {pred.shape}, target {target.shape}")
        # Siempre redimensionar la predicción para que coincida con el objetivo (ground truth)
        # Esto es más apropiado ya que el ground truth representa la escala real que queremos alcanzar
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)
        
        print(f"Nuevas dimensiones: pred {pred.shape}, target {target.shape}")
    
    loss = F.smooth_l1_loss(pred, target, reduction="none")

    if mask is not None:
        # Asegurarse de que la máscara también tenga el mismo tamaño
        if mask.shape[2:] != loss.shape[2:]:
            mask = F.interpolate(mask.float(), size=loss.shape[2:], mode='nearest').bool()
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)
    else:
        return loss.mean()


def get_unified_depth_model(initial_filters=64, max_disp=192, use_super_resolution=True, use_hybrid_refinement=True):
    """
    Crea un nuevo modelo unificado de estimación de profundidad.

    Args:
        initial_filters: número de filtros iniciales para el modelo
        max_disp: disparidad máxima para el volumen de costos
        use_super_resolution: si True, habilita los módulos de super-resolución y refinamiento
        use_hybrid_refinement: si True, habilita el módulo híbrido ViT-CNN para refinamiento de profundidad

    Returns:
        modelo: Una instancia del modelo UnifiedDepthModel
    """
    return UnifiedDepthModel(
        in_channels=3, 
        base_filters=initial_filters, 
        max_disp=max_disp,
        use_super_resolution=use_super_resolution,
        use_hybrid_refinement=use_hybrid_refinement
    )
