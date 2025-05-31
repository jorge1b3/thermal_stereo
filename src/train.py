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
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
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
        default=5,
        help="Número de épocas a esperar para early stopping",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Cambio mínimo en la pérdida para considerarla como mejora",
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

    return parser.parse_args()


class EarlyStopping:
    """
    Clase para realizar early stopping durante el entrenamiento.
    Detiene el entrenamiento cuando la métrica de validación no mejora durante 'patience' épocas.
    """

    def __init__(self, patience=5, min_delta=0, path="checkpoint.pt", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} de {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Guardar modelo cuando hay mejora en la pérdida de validación"""
        if self.verbose:
            logging.info(
                f"Pérdida de validación disminuida ({self.val_loss_min:.6f} --> {val_loss:.6f}). Guardando modelo..."
            )
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


def train_traditional(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, args
):
    """
    Función de entrenamiento para el modelo tradicional.
    """
    # Crear directorio para guardar modelos si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Nombre del experimento para wandb y guardado
    experiment_name = f"{args.model_name}_{args.model_type}_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}"

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

        # Guardar checkpoint cada 5 épocas
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
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint guardado en época {epoch+1}")

    # Guardar el modelo final
    final_model_path = os.path.join(args.output_dir, f"{experiment_name}_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
        },
        final_model_path,
    )
    logging.info(f"Modelo final guardado en {final_model_path}")

    # También guardar una copia con el nombre estándar
    standard_path = os.path.join(args.output_dir, "model_trained.pth")
    torch.save(model.state_dict(), standard_path)
    logging.info(f"Modelo estándar guardado en {standard_path}")

    # Generar y guardar algunas predicciones de ejemplo
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
                    depth_true, caption="Profundidad Real"
                ),
                "sample_depth_pred": wandb.Image(
                    depth_pred, caption="Profundidad Predicha"
                ),
            }
        )

    # Subir los modelos como artifacts
    artifact = wandb.Artifact("model_weights", type="model")
    artifact.add_file(os.path.join(args.output_dir, f"{experiment_name}_best.pth"))
    artifact.add_file(final_model_path)
    artifact.add_file(standard_path)
    wandb.log_artifact(artifact)


def train_unified(model, train_loader, val_loader, optimizer, scheduler, device, args):
    """
    Función de entrenamiento para el modelo unificado.
    """
    # Configuración de pesos para las diferentes escalas
    scale_weights = [0.5, 0.7, 0.85, 1.0]  # para 4 escalas

    # Crear directorio para guardar modelos si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Nombre del experimento para wandb y guardado
    experiment_name = f"{args.model_name}_unified_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}"

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
            for i, (mono_depth, stereo_depth) in enumerate(
                zip(
                    output["multi_scale_mono_depth"], output["multi_scale_stereo_depth"]
                )
            ):
                # Aplicar peso a la escala correspondiente
                scale_loss = (
                    smooth_l1_loss(mono_depth, depth)
                    + smooth_l1_loss(stereo_depth, depth)
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
                wandb.log(
                    {
                        "batch": batch_idx + epoch * len(train_loader),
                        "train_batch_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

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
                val_loss_batch = smooth_l1_loss(output["stereo_depth"], depth)
                val_loss += val_loss_batch.item()

        # Calcular pérdida media de validación
        val_loss /= len(val_loader)

        # Ajustar learning rate con el scheduler
        scheduler.step()

        # Registrar métricas en wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

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

        # Guardar checkpoint cada 5 épocas
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

    # Guardar modelo final
    final_model_path = os.path.join(args.output_dir, f"{experiment_name}_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        },
        final_model_path,
    )
    logging.info(f"Modelo final guardado en {final_model_path}")

    # También guardar una copia con el nombre estándar
    standard_path = os.path.join(args.output_dir, "unified_model_trained.pth")
    torch.save(model.state_dict(), standard_path)
    logging.info(f"Modelo estándar guardado en {standard_path}")

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
                    depth_true, caption="Profundidad Real"
                ),
                "sample_stereo_pred": wandb.Image(
                    stereo_pred, caption="Profundidad Estéreo Predicha"
                ),
                "sample_mono_pred": wandb.Image(
                    mono_pred, caption="Profundidad Mono Predicha"
                ),
            }
        )

    # Subir los modelos como artifacts
    artifact = wandb.Artifact("model_weights", type="model")
    artifact.add_file(os.path.join(args.output_dir, f"{experiment_name}_best.pth"))
    artifact.add_file(final_model_path)
    artifact.add_file(standard_path)
    wandb.log_artifact(artifact)


def main():
    """Función principal para configurar y ejecutar el entrenamiento."""
    args = parse_args()

    # Configuración de nombre para el experimento
    experiment_name = f"{args.model_name}_{args.model_type}_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}"

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
        )
        model.to(device)

        logging.info(
            f"Creado modelo unificado con {args.initial_filters} filtros iniciales y disparidad máxima {args.max_disp}"
        )

    else:
        # === CARGA DE DATOS PARA MODELO TRADICIONAL ===
        # Cargar los datasets de entrenamiento y validación
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

    # Guardar archivo de configuración
    os.makedirs("config", exist_ok=True)
    config_path = os.path.join("config", f"{experiment_name}_config.txt")
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Subir el archivo de configuración como artefacto
    artifact_config = wandb.Artifact("run_config", type="config")
    artifact_config.add_file(config_path)
    wandb.log_artifact(artifact_config)

    # Cerrar wandb
    wandb.finish()


if __name__ == "__main__":
    main()
