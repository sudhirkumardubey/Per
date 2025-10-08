"""
Surge Prediction Module - Adapted from RadComp

Predicts surge onset based on critical flow angle in vaneless diffuser
Based on Japikse & Baines (1994) empirical correlations

Author: Centrifugal Compressor Team
Date: October 2025
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.polynomial import polynomial

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.components.impeller import Impeller
from centrifugal_compressor.components.diffuser import Diffuser


# ============================================================================
# SURGE PREDICTION COEFFICIENTS (Japikse & Baines)
# ============================================================================

def _generate_surge_fits():
    """
    Generate polynomial fits for surge critical angle prediction
    
    Based on empirical data from Japikse & Baines (1994)
    for vaneless diffuser surge onset
    
    Returns:
        c_12, c_20: Polynomial coefficients for length ratios 1.2 and 2.0
    """
    # Mach number values
    mach_values = np.array([0, 0.4, 0.8, 1.2, 1.6])
    
    # Width-to-radius ratio
    b_ratio = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    
    # Polynomial degree
    deg = np.array([3, 3])
    
    # Critical angle data for r5/r4 = 1.2
    a_12 = np.array([
        [80.78, 80.00, 78.59, 76.41, 73.90],
        [76.71, 75.47, 73.28, 70.47, 67.19],
        [73.91, 72.97, 70.63, 66.25, 60.00],
        [72.81, 71.87, 69.53, 64.53, 55.63],
        [72.19, 71.25, 68.75, 63.59, 54.22],
    ])
    
    # Critical angle data for r5/r4 = 2.0
    a_20 = np.array([
        [80.78, 80.16, 78.59, 76.41, 73.91],
        [76.56, 77.19, 73.44, 70.63, 67.19],
        [74.06, 71.56, 68.75, 64.84, 60.31],
        [70.47, 69.38, 66.25, 61.25, 55.16],
        [69.22, 68.13, 64.84, 59.38, 52.97],
    ])
    
    def polyfit2d(x, y, z, deg):
        """2D polynomial fit"""
        xx, yy = np.meshgrid(x, y)
        lhs = polynomial.polyvander2d(xx.ravel(), yy.ravel(), deg).T
        rhs = z.ravel().T
        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1
        rcond = xx.size * np.finfo(xx.dtype).eps
        c1, _, _, _ = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
        c1 = (c1.T / scl).T
        return c1
    
    c12 = polyfit2d(mach_values, b_ratio, a_12, deg)
    c20 = polyfit2d(mach_values, b_ratio, a_20, deg)
    
    shape = deg + 1
    return c12.reshape(shape), c20.reshape(shape)


# Generate coefficients at module load time
_C_12, _C_20 = _generate_surge_fits()


# ============================================================================
# SURGE CALCULATION FUNCTIONS
# ============================================================================

def surge_critical_angle(
    r5: float, 
    r4: float, 
    b4: float, 
    mach_impeller_exit: float
) -> float:
    """
    Calculate critical flow angle for surge onset in vaneless diffuser
    
    Based on Japikse & Baines (1994) empirical correlation
    
    Args:
        r5: Vaneless diffuser exit radius [m]
        r4: Impeller exit radius (vaneless inlet) [m]
        b4: Impeller exit width [m]
        mach_impeller_exit: Mach number at impeller exit [-]
    
    Returns:
        alpha_critical: Critical absolute flow angle at r4 [degrees]
                        Surge occurs when alpha4 < alpha_critical
    
    Reference:
        Japikse, D., & Baines, N. C. (1994). Introduction to Turbomachinery
    """
    # Geometry ratios
    b_r_ratio = b4 / r4  # Width-to-radius ratio
    length_ratio = r5 / r4  # Diffuser length ratio
    
    # Interpolate critical angle for r5/r4 = 1.2
    angle_12 = polynomial.polyval2d(mach_impeller_exit, b_r_ratio, _C_12)
    
    # Interpolate critical angle for r5/r4 = 2.0
    angle_20 = polynomial.polyval2d(mach_impeller_exit, b_r_ratio, _C_20)
    
    # Linear interpolation between 1.2 and 2.0
    if length_ratio < 1.2:
        alpha_r = angle_12
    elif length_ratio > 2.0:
        alpha_r = angle_20
    else:
        alpha_r = angle_12 + (angle_20 - angle_12) * (length_ratio - 1.2) / (2.0 - 1.2)
    
    # Apply correction factor (RadComp empirical adjustment)
    alpha_critical = 90.0 - 0.35 * (90.0 - alpha_r)
    
    return alpha_critical


def calculate_surge_margin(
    alpha_actual: float,
    alpha_critical: float
) -> float:
    """
    Calculate surge margin
    
    Args:
        alpha_actual: Actual flow angle at impeller exit [degrees]
        alpha_critical: Critical flow angle for surge [degrees]
    
    Returns:
        surge_margin: Percentage margin to surge [%]
                      Positive = safe, Negative = in surge
    """
    # Surge margin: how far from critical angle
    # SM = (α_actual - α_critical) / α_critical * 100%
    surge_margin = (alpha_actual - alpha_critical) / alpha_critical * 100.0
    
    return surge_margin


# ============================================================================
# SURGE ANALYSIS DATACLASS
# ============================================================================

@dataclass
class SurgeAnalysis:
    """
    Surge analysis results for centrifugal compressor
    
    Attributes:
        alpha_critical: Critical flow angle for surge onset [deg]
        alpha_actual: Actual flow angle at impeller exit [deg]
        surge_margin: Margin to surge [%] (positive = safe)
        in_surge: Flag indicating if operating in surge region
        mach_impeller_exit: Mach number at impeller exit [-]
    """
    alpha_critical: float = math.nan
    alpha_actual: float = math.nan
    surge_margin: float = math.nan
    in_surge: bool = False
    mach_impeller_exit: float = math.nan
    
    # Geometry parameters (for reference)
    r4: float = math.nan
    r5: float = math.nan
    b4: float = math.nan
    length_ratio: float = math.nan
    width_ratio: float = math.nan
    
    def __repr__(self):
        status = "⚠️ SURGE" if self.in_surge else "✓ STABLE"
        return (f"SurgeAnalysis({status}, "
                f"α_actual={self.alpha_actual:.2f}°, "
                f"α_crit={self.alpha_critical:.2f}°, "
                f"SM={self.surge_margin:+.2f}%)")


# ============================================================================
# SURGE PREDICTOR
# ============================================================================

@dataclass
class SurgePredictor:
    """
    Predicts surge onset for centrifugal compressor
    
    Usage:
        predictor = SurgePredictor(geom, impeller, diffuser)
        analysis = predictor.analyze()
        print(f"Surge margin: {analysis.surge_margin:.2f}%")
    """
    
    geom: Geometry
    impeller: Impeller
    diffuser: Optional[Diffuser] = None
    
    def analyze(self) -> SurgeAnalysis:
        """
        Perform surge analysis
        
        Returns:
            SurgeAnalysis object with all surge parameters
        """
        # Get impeller exit conditions
        alpha_actual = self.impeller.out.alpha  # Flow angle [deg]
        mach_exit = self.impeller.out.mach  # Mach number
        
        # Geometry
        r4 = self.geom.r4
        r5 = self.geom.r5
        b4 = self.geom.b4
        
        # Calculate critical angle for surge
        alpha_critical = surge_critical_angle(r5, r4, b4, mach_exit)
        
        # Calculate surge margin
        margin = calculate_surge_margin(alpha_actual, alpha_critical)
        
        # Determine if in surge
        in_surge = alpha_actual < alpha_critical
        
        # Create analysis object
        analysis = SurgeAnalysis(
            alpha_critical=alpha_critical,
            alpha_actual=alpha_actual,
            surge_margin=margin,
            in_surge=in_surge,
            mach_impeller_exit=mach_exit,
            r4=r4,
            r5=r5,
            b4=b4,
            length_ratio=r5 / r4,
            width_ratio=b4 / r4,
        )
        
        return analysis
    
    def is_stable(self, safety_margin: float = 10.0) -> bool:
        """
        Check if operating point is stable (with safety margin)
        
        Args:
            safety_margin: Minimum required margin [%] (default: 10%)
        
        Returns:
            True if stable with sufficient margin, False otherwise
        """
        analysis = self.analyze()
        return analysis.surge_margin >= safety_margin
    
    def predict_surge_mass_flow(self, op: OperatingCondition) -> Optional[float]:
        """
        Predict mass flow rate at surge onset (not implemented - requires iteration)
        
        Would need to:
        1. Vary mass flow
        2. Recalculate impeller/diffuser
        3. Find point where alpha = alpha_critical
        
        Returns:
            mass_flow_surge: Mass flow at surge [kg/s] (or None if not implemented)
        """
        # TODO: Implement iterative surge line prediction
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_surge(
    geom: Geometry,
    impeller: Impeller,
    diffuser: Optional[Diffuser] = None
) -> SurgeAnalysis:
    """
    Convenience function to quickly check surge status
    
    Args:
        geom: Geometry object
        impeller: Impeller results
        diffuser: Diffuser results (optional)
    
    Returns:
        SurgeAnalysis object
    
    Example:
        >>> surge = check_surge(geom, impeller, diffuser)
        >>> print(f"Surge margin: {surge.surge_margin:.2f}%")
        >>> if surge.in_surge:
        ...     print("⚠️ WARNING: Operating in surge region!")
    """
    predictor = SurgePredictor(geom, impeller, diffuser)
    return predictor.analyze()
