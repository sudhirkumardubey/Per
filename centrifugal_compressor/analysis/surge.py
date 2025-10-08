"""
Surge Prediction Module - RadComp Slope Method
================================================

Predicts surge onset using characteristic slope criterion (dH/dQ).

This is the method used in RadComp: numerically calculate the slope
of the compressor characteristic curve. Positive slope indicates
unstable operation (surge region).

Theory:
-------
For stable compressor operation, the characteristic must have negative slope:
    dH/dQ < 0  (where H = head, Q = mass flow)

If dH/dQ ≥ 0, the system is in the surge region.

Author: Centrifugal Compressor Team  
Date: October 2025

References:
-----------
[1] Greitzer, E. M., 1976, "Surge and Rotating Stall in Axial Flow 
    Compressors," ASME J. Eng. Power, 98(2), pp. 190-198.
[2] RadComp source code: https://github.com/cyrilpic/radcomp
"""

import math
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition


# ============================================================================
# SURGE ANALYSIS DATACLASS
# ============================================================================

@dataclass
class SurgeAnalysis:
    """
    Surge analysis results using characteristic slope method
    
    Attributes:
        in_surge: True if positive slope detected (unstable)
        d_head_d_flow: Numerical derivative dH/dQ [J/kg per kg/s]
        head_1: Head at nominal operating point [J/kg]
        head_2: Head at perturbed operating point [J/kg]
        flow_1: Mass flow at nominal point [kg/s]
        flow_2: Mass flow at perturbed point [kg/s]
        delta_factor: Perturbation factor used (e.g., 1.005 = +0.5%)
        converged: True if both calculations succeeded
        threshold: Threshold used for surge detection
    """
    in_surge: bool
    d_head_d_flow: Optional[float] = None
    head_1: float = math.nan
    head_2: float = math.nan
    flow_1: float = math.nan
    flow_2: float = math.nan
    delta_factor: float = 1.005
    converged: bool = False
    threshold: float = -1e-4
    
    def __repr__(self):
        if not self.converged:
            return "SurgeAnalysis(FAILED - calculation did not converge)"
        
        status = "⚠️  SURGE" if self.in_surge else "✅ STABLE"
        slope_sign = "POSITIVE" if self.d_head_d_flow > 0 else "NEGATIVE"
        return (f"SurgeAnalysis({status}, "
                f"dH/dQ={self.d_head_d_flow:.2f} J/kg/(kg/s) [{slope_sign}])")
    
    def print_detailed(self):
        """Print detailed surge analysis report"""
        print("\n" + "="*70)
        print("  SURGE ANALYSIS - CHARACTERISTIC SLOPE METHOD")
        print("="*70)
        
        if not self.converged:
            print("  ❌ STATUS: CALCULATION FAILED")
            print("     One or both operating points did not converge")
            print("="*70 + "\n")
            return
        
        # Status
        if self.in_surge:
            print("  ❌ STATUS: SURGE DETECTED")
            print("     Compressor characteristic has POSITIVE slope")
            print("     System is UNSTABLE at this operating point")
        else:
            print("  ✅ STATUS: STABLE OPERATION")
            print("     Compressor characteristic has NEGATIVE slope")
        
        print("\n" + "-"*70)
        print("  CHARACTERISTIC SLOPE")
        print("-"*70)
        print(f"  dH/dQ:                {self.d_head_d_flow:.2f} J/kg/(kg/s)")
        print(f"  Threshold:            {self.threshold:.6f} J/kg/(kg/s)")
        
        if self.d_head_d_flow > 0:
            print(f"  Slope type:           POSITIVE (unstable)")
        else:
            print(f"  Slope type:           NEGATIVE (stable)")
        
        print("\n" + "-"*70)
        print("  OPERATING POINTS")
        print("-"*70)
        print(f"  Nominal point:")
        print(f"    Mass flow:          {self.flow_1:.4f} kg/s")
        print(f"    Head:               {self.head_1:.0f} J/kg")
        
        print(f"\n  Perturbed point (+{(self.delta_factor-1)*100:.1f}% flow):")
        print(f"    Mass flow:          {self.flow_2:.4f} kg/s")
        print(f"    Head:               {self.head_2:.0f} J/kg")
        
        delta_head = self.head_2 - self.head_1
        print(f"\n  Change in head:       {delta_head:+.2f} J/kg")
        print(f"  Change in flow:       {self.flow_2 - self.flow_1:+.6f} kg/s")
        
        print("\n" + "-"*70)
        print("  PHYSICAL INTERPRETATION")
        print("-"*70)
        
        if self.in_surge:
            print("  • Increasing flow INCREASES head (wrong direction!)")
            print("  • Characteristic curve has positive slope")
            print("  • This operating point is fundamentally UNSTABLE")
            print("  • Surge/rotating stall will occur")
            print("\n  RECOMMENDED ACTIONS:")
            print("    1. Increase mass flow significantly")
            print("    2. Reduce rotational speed")
            print("    3. Check for downstream blockage")
            print("    4. Review design geometry")
        else:
            print("  • Increasing flow DECREASES head (correct direction)")
            print("  • Characteristic curve has negative slope")
            print("  • Operating point is STABLE")
            print("  • No surge risk at this condition")
        
        print("="*70 + "\n")


# ============================================================================
# SURGE CHECKING FUNCTION
# ============================================================================

def check_surge(
    geom: Geometry,
    op: OperatingCondition,
    stage_calculator,
    delta_factor: float = 1.005,
    loss_model: str = 'schiffmann',
    threshold: float = -1e-4
) -> SurgeAnalysis:
    """
    Check surge using characteristic slope method (FIXED VERSION)
    """
    
    try:
        # ====================================================================
        # STEP 1: Calculate at nominal operating point
        # ====================================================================
        stage_1 = stage_calculator(geom, op, loss_model)
        
        # Check if calculation converged
        if hasattr(stage_1, 'converged') and not stage_1.converged:
            return SurgeAnalysis(
                in_surge=False,
                converged=False,
                threshold=threshold
            )
        
        if hasattr(stage_1, 'invalid_flag') and stage_1.invalid_flag:
            return SurgeAnalysis(
                in_surge=False,
                converged=False,
                threshold=threshold
            )
        
        # ====================================================================
        # FIX: Extract head (specific work) correctly
        # ====================================================================
        # Head = specific work = ΔH [J/kg] (NOT total power!)
        # For stage: use inlet to diffuser outlet enthalpy rise
        
        head_1 = stage_1.diffuser.outlet.total.H - stage_1.inlet.H  # Stage: inlet to diffuser outlet
        flow_1 = op.mass_flow
        
        # Validate head is positive
        if head_1 <= 0:
            return SurgeAnalysis(
                in_surge=False,
                converged=False,
                threshold=threshold
            )
        
        # ====================================================================
        # STEP 2: Create perturbed operating condition
        # ====================================================================
        op_perturbed = deepcopy(op)
        op_perturbed.mass_flow *= delta_factor
        
        # ====================================================================
        # STEP 3: Calculate at perturbed point
        # ====================================================================
        stage_2 = stage_calculator(geom, op_perturbed, loss_model)
        
        # Check if perturbed calculation converged
        if hasattr(stage_2, 'converged') and not stage_2.converged:
            return SurgeAnalysis(
                in_surge=False,
                head_1=head_1,
                flow_1=flow_1,
                converged=False,
                threshold=threshold
            )
        
        if hasattr(stage_2, 'invalid_flag') and stage_2.invalid_flag:
            return SurgeAnalysis(
                in_surge=False,
                head_1=head_1,
                flow_1=flow_1,
                converged=False,
                threshold=threshold
            )
        
        # Extract head at perturbed point
        head_2 = stage_2.diffuser.outlet.total.H - stage_2.inlet.H  # Stage: inlet to diffuser outlet
        flow_2 = op_perturbed.mass_flow
        
        # Validate head is positive
        if head_2 <= 0:
            return SurgeAnalysis(
                in_surge=False,
                head_1=head_1,
                flow_1=flow_1,
                converged=False,
                threshold=threshold
            )
        
        # ====================================================================
        # STEP 4: Calculate numerical derivative (slope)
        # ====================================================================
        d_head_d_flow = (head_2 - head_1) / (flow_2 - flow_1)
        
        # ====================================================================
        # STEP 5: Check stability criterion
        # ====================================================================
        # Surge if slope > threshold (i.e., positive or near-zero)
        in_surge = (d_head_d_flow > threshold)
        
        return SurgeAnalysis(
            in_surge=in_surge,
            d_head_d_flow=d_head_d_flow,
            head_1=head_1,
            head_2=head_2,
            flow_1=flow_1,
            flow_2=flow_2,
            delta_factor=delta_factor,
            converged=True,
            threshold=threshold
        )
        
    except Exception as e:
        import warnings
        warnings.warn(f"Surge calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return SurgeAnalysis(
            in_surge=False,
            converged=False,
            threshold=threshold
        )
