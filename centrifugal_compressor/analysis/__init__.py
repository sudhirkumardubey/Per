"""
Analysis tools for centrifugal compressor performance

Includes:
- Surge prediction
- Performance maps
- Off-design analysis
"""

from .surge import (
    SurgeAnalysis,
    SurgePredictor,
    check_surge,
    surge_critical_angle,
    calculate_surge_margin,
)

__all__ = [
    'SurgeAnalysis',
    'SurgePredictor',
    'check_surge',
    'surge_critical_angle',
    'calculate_surge_margin',
]
