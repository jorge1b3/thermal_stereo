#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import logging
import wandb
from torch.utils.data import DataLoader
from src.utils.data_loader import load_train_dataset, load_test_dataset


def get_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo para imágenes térmicas y de profundidad")
    parser.add_argument("--data_dir", type=str, default="raw", help="Directorio raíz donde se encuentran los datos")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño del lote")
    parser.add_argument("--num_epochs", type=int, default=20, help="Épocas de entrenamiento")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Directorio para guardar el modelo")
    parser.add_argument("--model_name", type=str, default="thermal_depth_model", help="Nombre base del modelo")
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "resnet", "unet"], help="Tipo de modelo a entrenar (basic, resnet, unet)")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Tasa de dropout para regularización")
    parser.add_argument("--initial_filters", type=int, default=64, help="Número inicial de filtros en la primera capa")
    parser.add_argument("--num_workers", type=int, default=4, help="Número de workers para la carga de datos")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fracción de datos para validación")
    return parser.parse_args()


# Esta función ya se ha movido al archivo de modelos


if __name__ == "__main__":
    args = get_args()
    
    # Creamos directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Nombre de experimento basado en hiperparámetros y nombre de modelo
    wandb_name = f"{args.model_name}_{args.model_type}_lr_{args.lr:.3e}_bs_{args.batch_size}_ep_{args.num_epochs}"
    
    # Asegurarse de que exista el directorio para los logs
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("logs", f"{wandb_name}.log"))
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"Argumentos: {args}")

    # Inicializar wandb con nombre personalizado
    wandb.init(project="thermal_depth", name=wandb_name, config=vars(args))
    
    # Cargar los datasets de entrenamiento y validación
    train_dataloader = load_train_dataset(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Si hay un conjunto específico de validación, usarlo
    # De lo contrario, usar un subconjunto del conjunto de prueba
    test_dataloader = load_test_dataset(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Crear modelo con parámetros configurables
    model = get_model(
        model_name=args.model_type,
        initial_filters=args.initial_filters, 
        dropout_rate=args.dropout_rate
    ).to(device)
    
    logging.info(f"Creado modelo con {args.initial_filters} filtros iniciales y dropout {args.dropout_rate}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()  # MAE para regresión de profundidad

    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        train_mae = 0.0
        batch_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
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
            wandb.log({"batch_loss": loss.item(), "batch": batch_idx + epoch * len(train_dataloader)})
            
            # Mostrar progreso cada 10 lotes
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        # Calcular estadísticas de la época
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_mae = train_mae / len(train_dataloader.dataset)
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Preparar datos
                left, right, depth = prepare_inputs(batch, device)
                
                # Forward
                outputs = model(left, right)
                loss = criterion(outputs, depth)
                
                # Estadísticas
                val_loss += loss.item() * left.size(0)
                val_mae += torch.abs(outputs - depth).mean().item() * left.size(0)
        
        # Calcular estadísticas de validación
        val_loss = val_loss / len(test_dataloader.dataset)
        val_mae = val_mae / len(test_dataloader.dataset)
        
        logging.info(
            f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val MAE: {val_mae:.4f}"
        )
        
        # Registrar métricas en wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_mae": epoch_mae,
            "val_loss": val_loss,
            "val_mae": val_mae
        })
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, f"{wandb_name}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Mejor modelo guardado con pérdida de validación: {val_loss:.4f}")
            wandb.summary["best_val_loss"] = val_loss
            wandb.summary["best_val_mae"] = val_mae
            wandb.summary["best_epoch"] = epoch + 1
    
    # Guardar el modelo final
    final_model_path = os.path.join(args.output_dir, f"{wandb_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Modelo final guardado en {final_model_path}")
    
    # También guardar una copia con el nombre estándar
    standard_path = os.path.join(args.output_dir, "model_trained.pth")
    torch.save(model.state_dict(), standard_path)
    logging.info(f"Modelo estándar guardado en {standard_path}")
    
    # Guardar métricas finales en wandb.summary
    wandb.summary["final_val_loss"] = val_loss
    wandb.summary["final_val_mae"] = val_mae
    
    # Generar y guardar algunas predicciones de ejemplo
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        left, right, depth = prepare_inputs(batch, device)
        outputs = model(left, right)
        
        # Convertir para visualización
        depth_pred = outputs.squeeze().cpu().numpy()
        depth_true = depth.squeeze().cpu().numpy()
        left_img = left[0].permute(1, 2, 0).cpu().numpy()
        right_img = right[0].permute(1, 2, 0).cpu().numpy()
        
        # Registrar imágenes en wandb
        wandb.log({
            "sample_left": wandb.Image(left_img, caption="Imagen Térmica Izquierda"),
            "sample_right": wandb.Image(right_img, caption="Imagen Térmica Derecha"),
            "sample_depth_true": wandb.Image(depth_true, caption="Profundidad Real"),
            "sample_depth_pred": wandb.Image(depth_pred, caption="Profundidad Predicha")
        })
    
    # Subir los modelos como artifacts
    artifact = wandb.Artifact("model_weights", type="model")
    artifact.add_file(best_model_path)
    artifact.add_file(final_model_path)
    artifact.add_file(standard_path)
    wandb.log_artifact(artifact)
    
    # Asegurarse de que exista el directorio para las configuraciones
    os.makedirs("config", exist_ok=True)
    
    # Subir el archivo de configuración usado
    config_path = os.path.join("config", f"{wandb_name}_config.txt")
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    artifact_config = wandb.Artifact("run_config", type="config")
    artifact_config.add_file(config_path)
    wandb.log_artifact(artifact_config)
    
    wandb.finish()
