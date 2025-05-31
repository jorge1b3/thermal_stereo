#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import load_train_dataset, load_test_dataset


def visualize_sample(sample, save_path=None):
    """
    Visualiza un ejemplo de los datos.

    Args:
        sample (dict): Muestra que contiene 'depth', 'left' y 'right'.
        save_path (str, optional): Ruta para guardar la visualización.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mostrar imagen térmica izquierda
    axes[0].imshow(sample["left"], cmap="inferno")
    axes[0].set_title("Imagen Térmica Izquierda")
    axes[0].axis("off")

    # Mostrar imagen térmica derecha
    axes[1].imshow(sample["right"], cmap="inferno")
    axes[1].set_title("Imagen Térmica Derecha")
    axes[1].axis("off")

    # Mostrar imagen de profundidad
    axes[2].imshow(sample["depth"], cmap="viridis")
    axes[2].set_title("Imagen de Profundidad")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    """
    Script de ejemplo para mostrar cómo usar los cargadores de datos.
    """
    # Directorio donde se encuentran los datos
    data_dir = os.path.abspath("raw")

    print("Cargando conjunto de datos de entrenamiento...")
    train_loader = load_train_dataset(
        root_dir=data_dir,
        batch_size=1,  # Usamos batch_size=1 para visualización
        num_workers=1,
    )

    print("Cargando conjunto de datos de prueba...")
    test_loader = load_test_dataset(
        root_dir=data_dir,
        batch_size=1,  # Usamos batch_size=1 para visualización
        num_workers=1,
    )

    if train_loader:
        print(
            f"Conjunto de entrenamiento cargado con {len(train_loader.dataset)} muestras"
        )

        # Obtener y visualizar una muestra del conjunto de entrenamiento
        sample_batch = next(iter(train_loader))
        train_sample = {
            "left": sample_batch["left"][0].numpy(),
            "right": sample_batch["right"][0].numpy(),
            "depth": sample_batch["depth"][0].numpy(),
        }

        print("\nVisualizando muestra de entrenamiento...")
        visualize_sample(train_sample, save_path="train_sample.png")
    else:
        print("No se pudo cargar el conjunto de entrenamiento.")

    if test_loader:
        print(f"Conjunto de prueba cargado con {len(test_loader.dataset)} muestras")

        # Obtener y visualizar una muestra del conjunto de prueba
        sample_batch = next(iter(test_loader))
        test_sample = {
            "left": sample_batch["left"][0].numpy(),
            "right": sample_batch["right"][0].numpy(),
            "depth": sample_batch["depth"][0].numpy(),
        }

        print("\nVisualizando muestra de prueba...")
        visualize_sample(test_sample, save_path="test_sample.png")
    else:
        print("No se pudo cargar el conjunto de prueba.")


if __name__ == "__main__":
    main()
