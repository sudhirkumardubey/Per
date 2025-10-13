# File: centrifugal_compressor/core/correlations.py
"""
Correlations for centrifugal compressor performance prediction
EXACT implementation from RadComp correlations.py
"""

import math
import numpy as np
from scipy import optimize


def moody(Re: float, r: float) -> float:
    """
    Calculate Moody's friction coefficient
    
    EXACT implementation from RadComp
    
    Args:
        Re: Reynolds number
        r: Relative roughness (ε/D)
    
    Returns:
        float: Moody friction coefficient (f/4)
    
    Note: Returns f/4, not f (Darcy friction factor divided by 4)
          For laminar: returns 16/Re (which is 64/Re/4)
          For turbulent: solves Colebrook equation and divides by 4
    
    Reference: RadComp correlations.py
    """
    if Re < 2300.0:
        return 64.0 / Re / 4.0
    
    def colebrook(x: float) -> float:
        x_val = x[0] if isinstance(x, np.ndarray) else float(x)
        return -2 * math.log10(r / 3.72 + 2.51 / Re / x_val**0.5) - 1 / x_val**0.5
    
    result = optimize.fsolve(colebrook, 0.02)
    return float(result[0]) / 4.0


def japikse_friction(Re: float, k: float = 0.02) -> float:
    """
    Japikse friction coefficient for vaneless diffuser
    Used in diffuser.py
    
    Args:
        Re: Reynolds number
        k: Empirical constant (default 0.02)
    
    Returns:
        float: Friction coefficient Cf
    
    Reference: RadComp diffuser.py line:
               Cf = k * (1.8e5 / Re) ** 0.2
    """
    if Re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {Re}")
    
    Cf = k * (1.8e5 / Re) ** 0.2
    return Cf


# Calculate Geometric Parameters
def calculate_number_of_blades(PR: float, splitter: str) -> float:
    """
    Calculate the number of blades based on pressure ratio and splitter type.
    
    Parameters:
    - PR: Pressure ratio (dimensionless)
    - splitter: Splitter type ("WITH" or "WITHOUT")
    
    Returns:
    - Number of blades (dimensionless)
    """
    if splitter == "WITHOUT":
        return 12.03 + 2.544 * PR
    elif splitter == "WITH":
        return -4.527 * math.exp(1.865 / PR) + 32.22
    return 0