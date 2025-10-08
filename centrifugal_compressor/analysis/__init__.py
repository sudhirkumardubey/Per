"""
Analysis tools for centrifugal compressor performance

Includes:
- Surge prediction
- Performance maps
- Off-design analysis
"""

from .surge import (
    SurgeAnalysis,
    check_surge,

)

__all__ = [
    'SurgeAnalysis',
    'check_surge',
]
