#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de entrenamiento unificado para modelos de estimación de profundidad.
Soporta tanto el modelo tradicional como el modelo unificado.
El modo de entrenamiento se puede controlar a través de la variable de entorno MODEL_TYPE.

Uso:
    # Para entrenar el modelo tradicional:
    MODEL_TYPE=traditional python -m src.train_combined

    # Para entrenar el modelo unificado:
    MODEL_TYPE=unified python -m src.train_combined
"""

import os
import logging
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

# Importaciones condicionales según el tipo de modelo
MODEL_TYPE = os.environ.get("MODEL_TYPE", "traditional").lower()

if MODEL_TYPE == "unified":
    from src.models.unified_depth_model import (
        UnifiedDepthModel,
        prepare_inputs,
        smooth_l1_loss,
    )
    from src.data.dataset import ThermalDepthDataset
else:  # traditional
    from src.utils.data_loader import load_train_dataset, load_test_dataset
    from src.models.models import get_model, prepare_inputs


# Configuración de logging
def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo de estimación de profundidad"
    )

    # Parámetros de datos
    parser.add_argument(
        "--data_dir", type=str, default="raw", help="Directorio de datos"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8 if MODEL_TYPE == "traditional" else 4,
        help="Tamaño de batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para carga de datos",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proporción de datos para validación",
    )

    # Parámetros del modelo
    parser.add_argument(
        "--model_name",
        type=str,
        default="thermal_depth_model"
        if MODEL_TYPE == "traditional"
        else "unified_depth_model",
        help="Nombre del modelo",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="basic" if MODEL_TYPE == "traditional" else "unified",
        choices=["basic", "resnet", "unet", "unified"],
        help="Tipo de arquitectura del modelo",
    )
    parser.add_argument(
        "--initial_filters",
        type=int,
        default=64 if MODEL_TYPE == "traditional" else 32,
        help="Número de filtros iniciales",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Tasa de dropout para regularización",
    )

    # Parámetros específicos del modelo unificado
    if MODEL_TYPE == "unified":
        parser.add_argument(
            "--max_disp", type=int, default=192, help="Disparidad máxima"
        )
        parser.add_argument(
            "--use_super_resolution", action="store_true", 
            help="Usar módulos de super-resolución y refinamiento"
        )
        parser.add_argument(
            "--use_hybrid_refinement", action="store_true", default=True,
            help="Usar módulo híbrido de refinamiento ViT-CNN"
        )

    # Parámetros de entrenamiento
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Número máximo de épocas"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4 if MODEL_TYPE == "traditional" else 1e-4,
        help="Tasa de aprendizaje inicial",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Regularización L2 (weight decay)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directorio para guardar modelos",
    )

    # Parámetros para early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Número de épocas a esperar para early stopping",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.015,
        help="Mejora relativa mínima para continuar el entrenamiento (porcentaje, ej: 0.01 = 1%)",
    )

    # Parámetros para scheduler
    parser.add_argument(
        "--scheduler_t_max",
        type=int,
        default=None,
        help="T_max para CosineAnnealingLR (por defecto num_epochs)",
    )
    parser.add_argument(
        "--scheduler_eta_min",
        type=float,
        default=1e-6,
        help="Eta mínimo para CosineAnnealingLR",
    )

    # Parámetros de recuperación
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Ruta del checkpoint para reanudar el entrenamiento",
    )
    
    # Parámetro de fracción del dataset
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fracción del dataset a utilizar (0.0-1.0). Útil para pruebas rápidas.",
    )

    return parser.parse_args()


class EarlyStopping:
    """
    Clase para realizar early stopping durante el entrenamiento.
    Detiene el entrenamiento cuando la mejora relativa de la métrica de validación
    es menor que un umbral durante 'patience' épocas.
    Al final, carga el mejor modelo guardado.
    """

    def __init__(self, patience=5, min_delta=0.01, path="checkpoint.pt", verbose=True):
        self.patience = patience
        self.min_delta = min_delta  # Ahora representa un porcentaje relativo (ej: 0.01 = 1%)
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model = None
        self.best_optimizer = None
        self.best_epoch = -1
    
    def __call__(self, val_loss, model, optimizer, epoch):
        score = -val_loss
        
        if self.best_score is None:
            # Primera evaluación
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        else:
            # Calcular mejora relativa: (nuevo - antiguo) / abs(antiguo)
            # Para score (negativo de val_loss), mejora si es mayor
            if self.best_score != 0:
                relative_improvement = (score - self.best_score) / abs(self.best_score)
            else:
                # Si el mejor score es 0, usamos mejora absoluta
                relative_improvement = score - self.best_score
            
            if relative_improvement < self.min_delta:
                # No hay mejora significativa
                self.counter += 1
                if self.verbose:
                    logging.info(
                        f"EarlyStopping counter: {self.counter} de {self.patience} "
                        f"(mejora relativa: {relative_improvement:.2%}, umbral: {self.min_delta:.2%})"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
                    # Cargar el mejor modelo guardado
                    self.load_best_model(model, optimizer)
            else:
                # Hay mejora significativa, resetear contador
                self.best_score = score
                self.save_checkpoint(val_loss, model, optimizer, epoch)
                self.counter = 0
                if self.verbose:
                    logging.info(
                        f"Mejora relativa significativa: {relative_improvement:.2%} > {self.min_delta:.2%}"
                    )
    
    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Guardar modelo cuando hay mejora en la pérdida de validación"""
        if self.verbose:
            logging.info(
                f"Pérdida de validación disminuida ({self.val_loss_min:.6f} --> {val_loss:.6f}). Guardando modelo..."
            )
        
        # Guardar estado para recuperación posterior
        self.best_model = {key: val.cpu().clone() for key, val in model.state_dict().items()}
        self.best_optimizer = optimizer.state_dict()
        self.best_epoch = epoch
        
        # Guardar en disco
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.path,
        )
        self.val_loss_min = val_loss
    
    def load_best_model(self, model, optimizer):
        """Cargar el mejor modelo guardado"""
        if self.best_model is not None and self.best_epoch >= 0:
            if self.verbose:
                logging.info(
                    f"Cargando el mejor modelo de la época {self.best_epoch} "
                    f"con pérdida de validación: {self.val_loss_min:.6f}"
                )
            # Cargar el mejor modelo
            model.load_state_dict(self.best_model)
            optimizer.load_state_dict(self.best_optimizer)


def train_traditional(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, args
):
    """
    Función de entrenamiento para el modelo tradicional.
    """
    # Crear directorio para guardar modelos si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Nombre del experimento para wandb y guardado
    fraction_suffix = f"_frac_{args.dataset_fraction:.1f}" if args.dataset_fraction < 1.0 else ""
    experiment_name = f"{args.model_name}_{args.model_type}_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}{fraction_suffix}"

    # Configurar early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        path=os.path.join(args.output_dir, f"{experiment_name}_best.pth"),
        verbose=True,
    )

    # Variables para registro
    best_val_loss = float("inf")
    start_epoch = 0

    # Si estamos reanudando el entrenamiento
    if args.resume_from and os.path.isfile(args.resume_from):
        logging.info(f"Reanudando entrenamiento desde checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logging.info(
            f"Reanudando desde época {start_epoch}, mejor pérdida: {best_val_loss:.4f}"
        )

    # Bucle de entrenamiento por épocas
    for epoch in range(start_epoch, args.num_epochs):
        # ===== ENTRENAMIENTO =====
        model.train()
        running_loss = 0.0
        train_mae = 0.0
        batch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Preparar datos
            left, right, depth = prepare_inputs(batch, device)

            # Forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(left, right)
            loss = criterion(outputs, depth)
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item() * left.size(0)
            batch_losses.append(loss.item())
            train_mae += torch.abs(outputs - depth).mean().item() * left.size(0)

            # Registrar pérdida por lote
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "batch": batch_idx + epoch * len(train_loader),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Actualizar barra de progreso
            progress_bar.set_description(f"Época {epoch+1}, Loss: {loss.item():.4f}")

        # Calcular estadísticas de la época
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = train_mae / len(train_loader.dataset)

        # ===== VALIDACIÓN =====
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validación"):
                # Preparar datos
                left, right, depth = prepare_inputs(batch, device)

                # Forward
                outputs = model(left, right)
                loss = criterion(outputs, depth)

                # Estadísticas
                val_loss += loss.item() * left.size(0)
                val_mae += torch.abs(outputs - depth).mean().item() * left.size(0)

        # Calcular estadísticas de validación
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae / len(val_loader.dataset)

        # Ajustar learning rate con el scheduler
        scheduler.step()

        logging.info(
            f"Época {epoch+1}/{args.num_epochs} - Train Loss: {epoch_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val MAE: {val_mae:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Registrar métricas en wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_mae": epoch_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Aplicar early stopping
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            logging.info("Early stopping activado")
            break

        # Guardar checkpoint cada 5 épocas (solo guardar localmente)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"{experiment_name}_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": epoch_loss,  # para train_traditional usamos epoch_loss
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint guardado en época {epoch+1}")

    # Cargar el mejor modelo antes de guardar el final
    if early_stopping.best_model is not None:
        logging.info(f"Cargando el mejor modelo de la época {early_stopping.best_epoch}")
        model.load_state_dict(early_stopping.best_model)
        optimizer.load_state_dict(early_stopping.best_optimizer)
        val_loss = early_stopping.val_loss_min
    
    # Generar timestamp único para el nombre del archivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar el modelo final (que es el mejor según val_loss)
    final_model_path = os.path.join(args.output_dir, f"{experiment_name}_final_{timestamp}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "best_epoch": early_stopping.best_epoch,
            "timestamp": timestamp,
            "config": vars(args),
        },
        final_model_path,
    )
    logging.info(f"Mejor modelo (época {early_stopping.best_epoch}) guardado como modelo final en {final_model_path}")

    # También guardar una copia con el nombre estándar (sobrescribe la anterior)
    standard_path = os.path.join(args.output_dir, "model_trained.pth")
    torch.save(model.state_dict(), standard_path)
    logging.info(f"Modelo estándar guardado en {standard_path}")

    # Registrar la ruta del modelo guardado para referencia futura
    models_registry_path = os.path.join(args.output_dir, "models_registry.txt")
    with open(models_registry_path, "a") as f:
        f.write(f"{timestamp} | {experiment_name} | Época: {early_stopping.best_epoch} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}\n")
    logging.info(f"Registro de modelo añadido a {models_registry_path}")

    # Generar y guardar algunas predicciones de ejemplo con el mejor modelo
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        left, right, depth = prepare_inputs(batch, device)
        outputs = model(left, right)

        # Convertir para visualización
        depth_pred = outputs.squeeze().cpu().numpy()
        depth_true = depth.squeeze().cpu().numpy()
        left_img = left[0].permute(1, 2, 0).cpu().numpy()
        right_img = right[0].permute(1, 2, 0).cpu().numpy()

        # Crear figuras con colormap inferno para profundidad
        fig_true, ax_true = plt.subplots()
        img_true = ax_true.imshow(depth_true, cmap='inferno')
        ax_true.set_title("Profundidad Real")
        ax_true.axis('off')
        fig_true.colorbar(img_true)
        
        fig_pred, ax_pred = plt.subplots()
        img_pred = ax_pred.imshow(depth_pred, cmap='inferno')
        ax_pred.set_title("Profundidad Predicha")
        ax_pred.axis('off')
        fig_pred.colorbar(img_pred)
        
        # Registrar imágenes en wandb
        wandb.log(
            {
                "sample_left": wandb.Image(
                    left_img, caption="Imagen Térmica Izquierda"
                ),
                "sample_right": wandb.Image(
                    right_img, caption="Imagen Térmica Derecha"
                ),
                "sample_depth_true": wandb.Image(
                    fig_true, caption="Profundidad Real"
                ),
                "sample_depth_pred": wandb.Image(
                    fig_pred, caption="Profundidad Predicha"
                ),
            }
        )
        plt.close(fig_true)
        plt.close(fig_pred)
        
    # No subimos los pesos a wandb, solo la configuración


def train_unified(model, train_loader, val_loader, optimizer, scheduler, device, args):
    """
    Función de entrenamiento para el modelo unificado.
    """
    # Configuración de pesos para las diferentes escalas
    # El modelo puede devolver más de 4 escalas debido al mapa mejorado, así que hacemos la lista más larga
    scale_weights = [1.0, 0.85, 0.7, 0.5, 0.3, 0.2]  # Extendemos para soportar hasta 6 escalas
                                                     # y damos más peso a las escalas de mayor resolución

    # Crear directorio para guardar modelos si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Nombre del experimento para wandb y guardado
    fraction_suffix = f"_frac_{args.dataset_fraction:.1f}" if args.dataset_fraction < 1.0 else ""
    experiment_name = f"{args.model_name}_unified_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}{fraction_suffix}"

    # Configurar early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        path=os.path.join(args.output_dir, f"{experiment_name}_best.pth"),
        verbose=True,
    )

    # Variables para registro
    start_epoch = 0

    # Si estamos reanudando el entrenamiento
    if args.resume_from and os.path.isfile(args.resume_from):
        logging.info(f"Reanudando entrenamiento desde checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Reanudando desde época {start_epoch}")

    # Si el modelo no está en modo de super-resolución, acortar la lista de pesos
    if not hasattr(model, 'use_super_resolution') or not model.use_super_resolution:
        scale_weights = scale_weights[2:]  # Usar solo los pesos para las 4 escalas básicas
        logging.info("Modelo en modo básico (sin super-resolución), ajustando pesos de escalas")

    # Bucle de entrenamiento por épocas
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"Época {epoch+1}/{args.num_epochs}")

        # ===== ENTRENAMIENTO =====
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}")
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
            mono_depths = output["multi_scale_mono_depth"]
            stereo_depths = output["multi_scale_stereo_depth"]
            
            # Asegurar que sólo utilizamos tantas escalas como tengamos pesos definidos
            num_scales = min(len(mono_depths), len(stereo_depths), len(scale_weights))
            
            for i in range(num_scales):
                mono_depth = mono_depths[i]
                stereo_depth = stereo_depths[i]
                
                # Verificar y ajustar tamaños si son diferentes
                if mono_depth.shape[2:] != depth.shape[2:]:
                    # Redimensionar los mapas de profundidad predichos para que coincidan con el ground truth
                    mono_depth_resized = F.interpolate(
                        mono_depth, size=depth.shape[2:], mode='bilinear', align_corners=True
                    )
                    stereo_depth_resized = F.interpolate(
                        stereo_depth, size=depth.shape[2:], mode='bilinear', align_corners=True
                    )
                else:
                    mono_depth_resized = mono_depth
                    stereo_depth_resized = stereo_depth
                
                # Aplicar peso a la escala correspondiente
                # Usar el peso correspondiente al índice, o el último peso si estamos fuera de rango
                weight = scale_weights[i] if i < len(scale_weights) else scale_weights[-1]
                scale_loss = (
                    smooth_l1_loss(mono_depth_resized, depth)
                    + smooth_l1_loss(stereo_depth_resized, depth)
                ) * weight

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
                    "train_batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
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
                # Usamos la predicción principal (mono o stereo)
                pred_depth = output["stereo_depth"]
                
                # Asegurarnos de que las dimensiones coincidan
                if pred_depth.shape[2:] != depth.shape[2:]:
                    pred_depth = F.interpolate(pred_depth, size=depth.shape[2:], mode='bilinear', align_corners=True)
                    
                val_loss_batch = smooth_l1_loss(pred_depth, depth)
                val_loss += val_loss_batch.item()

        # Calcular pérdida media de validación
        val_loss /= len(val_loader)

        # Ajustar learning rate con el scheduler
        scheduler.step()

        # Registrar métricas en wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        logging.info(
            f"Época {epoch+1}/{args.num_epochs}, "
            f"Pérdida entrenamiento: {train_loss:.4f}, "
            f"Pérdida validación: {val_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Aplicar early stopping
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            logging.info("Early stopping activado")
            break

        # Guardar checkpoint cada 5 épocas (solo guardar localmente)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"{experiment_name}_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint guardado en época {epoch+1}")

    # Cargar el mejor modelo antes de guardar el final
    if early_stopping.best_model is not None:
        logging.info(f"Cargando el mejor modelo de la época {early_stopping.best_epoch}")
        model.load_state_dict(early_stopping.best_model)
        optimizer.load_state_dict(early_stopping.best_optimizer)
        val_loss = early_stopping.val_loss_min
    
    # Generar timestamp único para el nombre del archivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelo final (que es el mejor según val_loss)
    final_model_path = os.path.join(args.output_dir, f"{experiment_name}_final_{timestamp}.pth")
    
    # Guardar información adicional sobre super-resolución
    config_dict = vars(args)
    if hasattr(model, 'use_super_resolution'):
        config_dict['use_super_resolution'] = model.use_super_resolution
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_epoch": early_stopping.best_epoch,
            "timestamp": timestamp,
            "config": config_dict,
        },
        final_model_path,
    )
    logging.info(f"Mejor modelo (época {early_stopping.best_epoch}) guardado como modelo final en {final_model_path}")

    # También guardar una copia con el nombre estándar (sobrescribe la anterior)
    standard_path = os.path.join(args.output_dir, "unified_model_trained.pth")
    torch.save(model.state_dict(), standard_path)
    logging.info(f"Modelo estándar guardado en {standard_path}")
    
    # Registrar la ruta del modelo guardado para referencia futura
    models_registry_path = os.path.join(args.output_dir, "models_registry.txt")
    with open(models_registry_path, "a") as f:
        f.write(f"{timestamp} | {experiment_name} | Época: {early_stopping.best_epoch} | Val Loss: {val_loss:.4f}\n")
    logging.info(f"Registro de modelo añadido a {models_registry_path}")

    # Crear una visualización de ejemplo
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        left, right, depth = prepare_inputs(batch, device)
        output = model(left, right)

        # Convertir para visualización
        stereo_pred = output["stereo_depth"].squeeze(1).cpu().numpy()[0]
        mono_pred = output["mono_depth"].squeeze(1).cpu().numpy()[0]
        depth_true = depth.squeeze(1).cpu().numpy()[0]
        left_img = left[0].permute(1, 2, 0).cpu().numpy()
        right_img = right[0].permute(1, 2, 0).cpu().numpy()

        # Crear figuras con colormap inferno para profundidad
        fig_true, ax_true = plt.subplots()
        img_true = ax_true.imshow(depth_true, cmap='inferno')
        ax_true.set_title("Profundidad Real")
        ax_true.axis('off')
        fig_true.colorbar(img_true)
        
        fig_stereo, ax_stereo = plt.subplots()
        img_stereo = ax_stereo.imshow(stereo_pred, cmap='inferno')
        ax_stereo.set_title("Profundidad Estéreo Predicha")
        ax_stereo.axis('off')
        fig_stereo.colorbar(img_stereo)
        
        fig_mono, ax_mono = plt.subplots()
        img_mono = ax_mono.imshow(mono_pred, cmap='inferno')
        ax_mono.set_title("Profundidad Mono Predicha")
        ax_mono.axis('off')
        fig_mono.colorbar(img_mono)

        # Registrar imágenes en wandb
        wandb.log(
            {
                "sample_left": wandb.Image(
                    left_img, caption="Imagen Térmica Izquierda"
                ),
                "sample_right": wandb.Image(
                    right_img, caption="Imagen Térmica Derecha"
                ),
                "sample_depth_true": wandb.Image(
                    fig_true, caption="Profundidad Real"
                ),
                "sample_stereo_pred": wandb.Image(
                    fig_stereo, caption="Profundidad Estéreo Predicha"
                ),
                "sample_mono_pred": wandb.Image(
                    fig_mono, caption="Profundidad Mono Predicha"
                ),
            }
        )
        plt.close(fig_true)
        plt.close(fig_stereo)
        plt.close(fig_mono)
        
    # No subimos los pesos a wandb, solo la configuración


def main():
    """Función principal para configurar y ejecutar el entrenamiento."""
    args = parse_args()

    # Configuración de nombre para el experimento
    fraction_suffix = f"_frac_{args.dataset_fraction:.1f}" if args.dataset_fraction < 1.0 else ""
    experiment_name = f"{args.model_name}_{args.model_type}_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}{fraction_suffix}"

    # Configurar logging con archivo
    log_file = os.path.join("logs", f"{experiment_name}.log")
    os.makedirs("logs", exist_ok=True)
    setup_logging(log_file)

    # Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"Modo de entrenamiento: {MODEL_TYPE}")
    logging.info(f"Argumentos: {args}")

    # Inicializar wandb para seguimiento de experimentos
    wandb.init(project="thermal_depth", name=experiment_name, config=vars(args))

    # Crear datasets según el modo
    if MODEL_TYPE == "unified":
        # === CARGA DE DATOS PARA MODELO UNIFICADO ===
        dataset = ThermalDepthDataset(
            root_dir=args.data_dir,
            transform=None,  # Las transformaciones se hacen en prepare_inputs
        )
        
        # Aplicar fracción del dataset si es menor que 1.0
        if args.dataset_fraction < 1.0:
            total_size = len(dataset)
            subset_size = int(total_size * args.dataset_fraction)
            indices = torch.randperm(total_size)[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)
            logging.info(f"Usando una fracción del {args.dataset_fraction:.1%} del dataset ({subset_size} de {total_size} muestras)")

        # Dividir en conjunto de entrenamiento y validación
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Crear data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        logging.info(
            f"Conjuntos de datos: {len(train_dataset)} entrenamiento, {len(val_dataset)} validación"
        )

        # Crear modelo unificado
        model = UnifiedDepthModel(
            in_channels=3,  # 3 canales para imágenes térmicas (replicadas)
            base_filters=args.initial_filters,
            max_disp=args.max_disp,
            use_super_resolution=args.use_super_resolution,
            use_hybrid_refinement=args.use_hybrid_refinement if hasattr(args, 'use_hybrid_refinement') else True
        )
        model.to(device)

        logging.info(
            f"Creado modelo unificado con {args.initial_filters} filtros iniciales, "
            f"disparidad máxima {args.max_disp}, "
            f"super-resolución: {args.use_super_resolution}, "
            f"refinamiento híbrido: {getattr(args, 'use_hybrid_refinement', True)}"
        )

    else:
        # === CARGA DE DATOS PARA MODELO TRADICIONAL ===
        # Cargar los datasets de entrenamiento y validación con posible submuestreo
        if args.dataset_fraction < 1.0:
            # Cuando usamos una fracción del dataset, tenemos que crear los datasets primero
            train_dataset = ThermalDepthDataset(
                root_dir=args.data_dir,
                split="train",
                transform=None
            )
            test_dataset = ThermalDepthDataset(
                root_dir=args.data_dir,
                split="test",
                transform=None
            )
            
            # Aplicar la fracción a ambos datasets
            total_train = len(train_dataset)
            total_test = len(test_dataset)
            train_size = int(total_train * args.dataset_fraction)
            test_size = int(total_test * args.dataset_fraction)
            
            # Crear subsets aleatorios
            train_indices = torch.randperm(total_train)[:train_size]
            test_indices = torch.randperm(total_test)[:test_size]
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            test_subset = torch.utils.data.Subset(test_dataset, test_indices)
            
            # Crear data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                test_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            logging.info(f"Usando fracción del {args.dataset_fraction:.1%} del dataset:")
            logging.info(f"  - Entrenamiento: {train_size} de {total_train} muestras")
            logging.info(f"  - Validación: {test_size} de {total_test} muestras")
        else:
            # Usar los loaders estándar sin modificación
            train_loader = load_train_dataset(
                root_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            # Usar conjunto específico de validación
            val_loader = load_test_dataset(
                root_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

        # Crear modelo tradicional
        model = get_model(
            model_name=args.model_type,
            initial_filters=args.initial_filters,
            dropout_rate=args.dropout_rate,
        ).to(device)

        logging.info(
            f"Creado modelo tradicional ({args.model_type}) con {args.initial_filters} filtros iniciales y dropout {args.dropout_rate}"
        )

    # Configurar optimizador
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Configurar scheduler
    t_max = args.scheduler_t_max if args.scheduler_t_max else args.num_epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=args.scheduler_eta_min
    )

    # Entrenar modelo según el modo
    if MODEL_TYPE == "unified":
        train_unified(
            model, train_loader, val_loader, optimizer, scheduler, device, args
        )
    else:
        # Usar MAE para regresión de profundidad
        criterion = nn.L1Loss()
        train_traditional(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args,
        )

    # Guardar archivo de configuración local
    os.makedirs("config", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join("config", f"{experiment_name}_config_{timestamp}.txt")
    with open(config_path, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Experimento: {experiment_name}\n")
        f.write("-" * 50 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Subir solo el archivo de configuración como artefacto (no los pesos)
    artifact_config = wandb.Artifact("run_config", type="config")
    artifact_config.add_file(config_path)
    wandb.log_artifact(artifact_config)

    # Cerrar wandb
    wandb.finish()


if __name__ == "__main__":
    main()
