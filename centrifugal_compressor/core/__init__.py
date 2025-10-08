# File: centrifugal_compressor/core/__init__.py

"""
Core utilities for centrifugal compressor calculations

This module provides:
- Geometry definitions and operating conditions
- Thermodynamic property calculations using CoolProp
- Trigonometric functions in degrees (cosd, sind, tand)
- Performance correlations (moody friction, japikse friction)

Usage:
    from centrifugal_compressor.core import Geometry, OperatingCondition, Fluid
    from centrifugal_compressor.core import cosd, sind, tand
    
    geom = Geometry(r1=0.02, r2s=0.025, r2h=0.01, ...)
    fluid = Fluid('Air')
    state = fluid.thermoprop('PT', 101325, 300)
"""

# ============================================================================
# Trigonometric functions (degrees) - from geometry.py
# ============================================================================
from .geometry import cosd, sind, tand

# ============================================================================
# Geometry and operating conditions - from geometry.py
# ============================================================================
from .geometry import (
    Geometry,
    OperatingCondition,
    create_example_geometry,
    create_example_operating_condition,
    compare_hydraulic_diameter_methods,
)

# ============================================================================
# Thermodynamics - from thermodynamics.py
# ============================================================================
from .thermodynamics import (
    FluidState,
    Fluid,
    ThermoException,
    static_from_total,
    total_from_static,
)

# ============================================================================
# Correlations - from correlations.py
# ============================================================================
from .correlations import (
    moody,
    japikse_friction,
)

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Trigonometric functions
    'cosd',
    'sind',
    'tand',
    
    # Geometry
    'Geometry',
    'OperatingCondition',
    'create_example_geometry',
    'create_example_operating_condition',
    'compare_hydraulic_diameter_methods',
    
    # Thermodynamics
    'FluidState',
    'Fluid',
    'ThermoException',
    'static_from_total',
    'total_from_static',
    
    # Correlations
    'moody',
    'japikse_friction',
]

# ============================================================================
# Package metadata
# ============================================================================
__version__ = '0.1.0'
__author__ = 'Centrifugal Compressor Team'
__description__ = 'Core utilities for centrifugal compressor design and analysis'
