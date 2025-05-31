#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalDepthModel(nn.Module):
    """
    Modelo para procesar imágenes térmicas (izquierda y derecha) y predecir profundidad.
    """

    def __init__(self, initial_filters=64, dropout_rate=0.2):
        super(ThermalDepthModel, self).__init__()

        # Codificador para imagen térmica izquierda
        self.left_encoder = nn.Sequential(
            nn.Conv2d(3, initial_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                initial_filters * 2, initial_filters * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )

        # Codificador para imagen térmica derecha
        self.right_encoder = nn.Sequential(
            nn.Conv2d(3, initial_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                initial_filters * 2, initial_filters * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )

        # Decodificador para reconstrucción de profundidad
        self.decoder = nn.Sequential(
            nn.Conv2d(
                initial_filters * 8, initial_filters * 4, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                initial_filters * 4, initial_filters * 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(initial_filters * 2, initial_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(initial_filters, 1, kernel_size=3, padding=1),
        )

    def forward(self, left, right):
        # Codificar imágenes térmicas
        left_features = self.left_encoder(left)
        right_features = self.right_encoder(right)

        # Concatenar características
        combined = torch.cat([left_features, right_features], dim=1)

        # Decodificar para generar mapa de profundidad
        depth = self.decoder(combined)

        return depth


class ThermalDepthModelResNet(nn.Module):
    """
    Modelo alternativo basado en bloques residuales para procesar
    imágenes térmicas y predecir profundidad.
    """

    def __init__(self, initial_filters=64, dropout_rate=0.2):
        super(ThermalDepthModelResNet, self).__init__()

        # Codificador para imagen térmica izquierda
        self.left_conv1 = nn.Conv2d(
            3, initial_filters, kernel_size=7, stride=2, padding=3
        )
        self.left_bn1 = nn.BatchNorm2d(initial_filters)
        self.left_res1 = self._make_residual_block(initial_filters, initial_filters * 2)
        self.left_res2 = self._make_residual_block(
            initial_filters * 2, initial_filters * 4
        )
        self.left_res3 = self._make_residual_block(
            initial_filters * 4, initial_filters * 4
        )
        self.left_dropout = nn.Dropout(dropout_rate)

        # Codificador para imagen térmica derecha
        self.right_conv1 = nn.Conv2d(
            3, initial_filters, kernel_size=7, stride=2, padding=3
        )
        self.right_bn1 = nn.BatchNorm2d(initial_filters)
        self.right_res1 = self._make_residual_block(
            initial_filters, initial_filters * 2
        )
        self.right_res2 = self._make_residual_block(
            initial_filters * 2, initial_filters * 4
        )
        self.right_res3 = self._make_residual_block(
            initial_filters * 4, initial_filters * 4
        )
        self.right_dropout = nn.Dropout(dropout_rate)

        # Decodificador con bloques residuales
        self.up_conv1 = nn.ConvTranspose2d(
            initial_filters * 8, initial_filters * 4, kernel_size=4, stride=2, padding=1
        )
        self.up_bn1 = nn.BatchNorm2d(initial_filters * 4)

        self.up_conv2 = nn.ConvTranspose2d(
            initial_filters * 4, initial_filters * 2, kernel_size=4, stride=2, padding=1
        )
        self.up_bn2 = nn.BatchNorm2d(initial_filters * 2)

        self.up_conv3 = nn.ConvTranspose2d(
            initial_filters * 2, initial_filters, kernel_size=4, stride=2, padding=1
        )
        self.up_bn3 = nn.BatchNorm2d(initial_filters)

        self.final_conv = nn.Conv2d(initial_filters, 1, kernel_size=3, padding=1)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left, right):
        # Codificar imagen izquierda
        left = F.relu(self.left_bn1(self.left_conv1(left)))
        left = self.left_res1(left)
        left = self.left_res2(left)
        left = self.left_res3(left)
        left = self.left_dropout(left)

        # Codificar imagen derecha
        right = F.relu(self.right_bn1(self.right_conv1(right)))
        right = self.right_res1(right)
        right = self.right_res2(right)
        right = self.right_res3(right)
        right = self.right_dropout(right)

        # Combinar características
        combined = torch.cat([left, right], dim=1)

        # Decoder
        x = F.relu(self.up_bn1(self.up_conv1(combined)))
        x = F.relu(self.up_bn2(self.up_conv2(x)))
        x = F.relu(self.up_bn3(self.up_conv3(x)))
        depth = self.final_conv(x)

        return depth


class ThermalDepthModelUNet(nn.Module):
    """
    Modelo basado en arquitectura U-Net para procesar imágenes térmicas y predecir profundidad.
    Incluye conexiones de salto (skip connections) entre codificador y decodificador.
    """

    def __init__(self, initial_filters=64, dropout_rate=0.2):
        super(ThermalDepthModelUNet, self).__init__()

        # Codificador para imagen izquierda
        self.left_enc1 = self._make_encoder_block(3, initial_filters)
        self.left_enc2 = self._make_encoder_block(initial_filters, initial_filters * 2)
        self.left_enc3 = self._make_encoder_block(
            initial_filters * 2, initial_filters * 4
        )
        self.left_enc4 = self._make_encoder_block(
            initial_filters * 4, initial_filters * 8
        )

        # Codificador para imagen derecha
        self.right_enc1 = self._make_encoder_block(3, initial_filters)
        self.right_enc2 = self._make_encoder_block(initial_filters, initial_filters * 2)
        self.right_enc3 = self._make_encoder_block(
            initial_filters * 2, initial_filters * 4
        )
        self.right_enc4 = self._make_encoder_block(
            initial_filters * 4, initial_filters * 8
        )

        # Punto más bajo (cuello de botella)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                initial_filters * 16, initial_filters * 16, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                initial_filters * 16, initial_filters * 16, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(initial_filters * 16),
            nn.ReLU(inplace=True),
        )

        # Decodificador
        self.dec1 = self._make_decoder_block(initial_filters * 16, initial_filters * 8)
        self.dec2 = self._make_decoder_block(initial_filters * 16, initial_filters * 4)
        self.dec3 = self._make_decoder_block(initial_filters * 8, initial_filters * 2)
        self.dec4 = self._make_decoder_block(initial_filters * 4, initial_filters)

        # Capa final
        self.final_conv = nn.Conv2d(initial_filters * 2, 1, kernel_size=1)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left, right):
        # Codificación de la imagen izquierda
        left1 = self.left_enc1(left)
        left2 = self.left_enc2(left1)
        left3 = self.left_enc3(left2)
        left4 = self.left_enc4(left3)

        # Codificación de la imagen derecha
        right1 = self.right_enc1(right)
        right2 = self.right_enc2(right1)
        right3 = self.right_enc3(right2)
        right4 = self.right_enc4(right3)

        # Fusión de características
        combined = torch.cat([left4, right4], dim=1)

        # Cuello de botella
        bottleneck = self.bottleneck(combined)

        # Decodificación con conexiones de salto
        x = self.dec1(bottleneck)
        x = torch.cat([x, torch.cat([left3, right3], dim=1)], dim=1)

        x = self.dec2(x)
        x = torch.cat([x, torch.cat([left2, right2], dim=1)], dim=1)

        x = self.dec3(x)
        x = torch.cat([x, torch.cat([left1, right1], dim=1)], dim=1)

        x = self.dec4(x)

        # Capa final
        depth = self.final_conv(x)

        return depth


# Función para preparar las imágenes para el modelo
def prepare_inputs(batch, device):
    """
    Prepara los datos de entrada para el modelo.
    """
    # Convertir imágenes a tensores y normalizar
    left = (
        batch["left"].to(device)
        if torch.is_tensor(batch["left"])
        else torch.tensor(batch["left"], dtype=torch.float32).to(device)
    )
    right = (
        batch["right"].to(device)
        if torch.is_tensor(batch["right"])
        else torch.tensor(batch["right"], dtype=torch.float32).to(device)
    )
    depth = (
        batch["depth"].to(device)
        if torch.is_tensor(batch["depth"])
        else torch.tensor(batch["depth"], dtype=torch.float32).to(device)
    )

    # Reorganizar dimensiones para formato de imagen (B, C, H, W)
    if len(left.shape) == 4 and left.shape[3] in [1, 3]:  # Si es [B, H, W, C]
        left = left.permute(0, 3, 1, 2)
    elif len(left.shape) == 3:  # Si es [B, H, W]
        left = left.unsqueeze(1)
        # Si el modelo espera 3 canales pero la imagen tiene 1, replicamos el canal
        left = left.repeat(1, 3, 1, 1)

    if len(right.shape) == 4 and right.shape[3] in [1, 3]:  # Si es [B, H, W, C]
        right = right.permute(0, 3, 1, 2)
    elif len(right.shape) == 3:  # Si es [B, H, W]
        right = right.unsqueeze(1)
        # Si el modelo espera 3 canales pero la imagen tiene 1, replicamos el canal
        right = right.repeat(1, 3, 1, 1)

    depth = depth.unsqueeze(1) if len(depth.shape) == 3 else depth

    # Convertir a float32 antes de cualquier operación
    left = left.float()
    right = right.float()
    depth = depth.float()

    # Normalizar si es necesario (0-255 -> 0-1)
    left = left / 255.0 if left.max() > 1.0 else left
    right = right / 255.0 if right.max() > 1.0 else right
    depth = depth / 255.0 if depth.max() > 1.0 else depth

    return left, right, depth


# Diccionario para mapear nombres de modelos a sus clases
MODEL_REGISTRY = {
    "basic": ThermalDepthModel,
    "resnet": ThermalDepthModelResNet,
    "unet": ThermalDepthModelUNet,
}


def get_model(model_name="basic", **kwargs):
    """
    Obtiene una instancia del modelo según su nombre.

    Args:
        model_name (str): Nombre del modelo ('basic', 'resnet', 'unet')
        **kwargs: Parámetros adicionales para el modelo

    Returns:
        modelo: Una instancia del modelo solicitado
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' no encontrado. Opciones disponibles: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)
