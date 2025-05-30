# Inicializador del paquete models
from .models import ThermalDepthModel, ThermalDepthModelResNet, ThermalDepthModelUNet, prepare_inputs, get_model, MODEL_REGISTRY

__all__ = [
    'ThermalDepthModel',
    'ThermalDepthModelResNet', 
    'ThermalDepthModelUNet',
    'prepare_inputs',
    'get_model',
    'MODEL_REGISTRY'
]
