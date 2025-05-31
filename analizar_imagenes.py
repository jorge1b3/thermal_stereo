#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analizar imágenes de los conjuntos de datos en raw/frick_2_test.
Examina una imagen de cada una de las carpetas: depth_filtered, img_left, img_right
y muestra información sobre su formato, tipo de datos, forma, etc.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import cv2


def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen y muestra su información.

    Args:
        ruta_imagen (str): Ruta a la imagen a analizar
    """
    print(f"\nAnalizando imagen: {ruta_imagen}")

    # Leer imagen con PIL
    img_pil = Image.open(ruta_imagen)
    print(f"Información PIL:")
    print(f"  Formato: {img_pil.format}")
    print(f"  Tamaño: {img_pil.size}")
    print(f"  Modo: {img_pil.mode}")

    # # Leer la misma imagen con OpenCV para obtener más información
    # img_cv = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)

    # if img_cv is not None:
    #     print(f"\nInformación OpenCV:")
    #     print(f"  Shape: {img_cv.shape}")
    #     print(f"  Tipo de datos: {img_cv.dtype}")
    #     print(f"  Valor mínimo: {img_cv.min()}")
    #     print(f"  Valor máximo: {img_cv.max()}")
    #     print(f"  Valores únicos: {np.unique(img_cv).shape[0]}")

    #     # Si es una imagen de profundidad, mostrar estadísticas adicionales
    #     if "depth" in ruta_imagen:
    #         print(f"  Profundidad media: {img_cv.mean():.2f}")
    #         print(f"  Profundidad mediana: {np.median(img_cv):.2f}")
    # else:
    #     print("Error: No se pudo leer la imagen con OpenCV")

    # Leer la imagen como un array de NumPy directamente
    img_np = np.array(img_pil)
    print(f"\nInformación NumPy:")
    print(f"  Shape: {img_np.shape}")
    print(f"  Tipo de datos: {img_np.dtype}")


def main():
    """Función principal que analiza imágenes en diferentes carpetas."""
    # Directorio base
    base_dir = "./raw/frick_2_test"

    # Nombre de imagen para analizar (asegúrate que existe en las tres carpetas)
    nombre_imagen = "00467.png"

    # Carpetas a analizar
    carpetas = ["depth_filtered", "img_left", "img_right"]

    print("=" * 60)
    print(f"ANÁLISIS DE LA IMAGEN {nombre_imagen} EN DIFERENTES CARPETAS")
    print("=" * 60)

    # Analizar la misma imagen en cada carpeta
    for carpeta in carpetas:
        ruta_imagen = os.path.join(base_dir, carpeta, nombre_imagen)
        print("\n" + "=" * 40)
        print(f"CARPETA: {carpeta}")
        print("=" * 40)

        if os.path.exists(ruta_imagen):
            analizar_imagen(ruta_imagen)
        else:
            print(f"Error: La imagen {nombre_imagen} no existe en {carpeta}")


if __name__ == "__main__":
    main()
