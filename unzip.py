#!/usr/bin/env python3
import zipfile
import sys
import os

def descomprimir_zip(archivo_zip, destino="."):
    """
    Descomprime un archivo .zip en el directorio de destino.
    """
    if not zipfile.is_zipfile(archivo_zip):
        print(f"Error: {archivo_zip} no es un archivo .zip válido.")
        return

    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(destino)
        print(f"Archivo {archivo_zip} descomprimido en {destino}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 descomprimir_zip.py archivo.zip [directorio_destino]")
    else:
        archivo_zip = sys.argv[1]
        destino = sys.argv[2] if len(sys.argv) >= 3 else "."
        descomprimir_zip(archivo_zip, destino)

