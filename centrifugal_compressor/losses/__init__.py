# File: centrifugal_compressor/losses/__init__.py

"""
Loss calculation framework

Exports:
- LossCalculator: Main interface for loss calculations
- available_models: List available loss models
- Individual loss component classes (for advanced users)
"""

from .calculator import LossCalculator, available_models, compare_models
from .models import AVAILABLE_MODELS

# Optional: Export individual models for advanced users
from .models import (
    OhLossModel,
    ZhangSet1LossModel,
    ZhangSet2LossModel,
    ZhangSet3LossModel,
    SchiffmannLossModel,
    MeroniLossModel,
)

__all__ = [
    'LossCalculator',
    'available_models',
    'compare_models',
    'AVAILABLE_MODELS',
    
    # Individual models
    'OhLossModel',
    'ZhangSet1LossModel',
    'ZhangSet2LossModel',
    'ZhangSet3LossModel',
    'SchiffmannLossModel',
    'MeroniLossModel',
]
