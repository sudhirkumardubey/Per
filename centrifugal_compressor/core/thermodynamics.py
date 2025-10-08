# File: centrifugal_compressor/core/thermodynamics.py
"""
Thermodynamic property calculations
Hybrid implementation combining RadComp structure with improvements
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import CoolProp.CoolProp as CP


class ThermoException(Exception):
    """Exception raised when thermodynamic property calculation fails"""
    pass


@dataclass(frozen=True)
class FluidState:
    """
    Thermodynamic state properties
    Based on RadComp ThermoProp but renamed for clarity
    
    Note: frozen=True makes instances immutable (best practice for state data)
    """
    P: float = math.nan      # Pressure (Pa)
    T: float = math.nan      # Temperature (K)
    D: float = math.nan      # Density (kg/m³)
    H: float = math.nan      # Specific enthalpy (J/kg)
    S: float = math.nan      # Specific entropy (J/kg-K)
    A: float = math.nan      # Speed of sound (m/s)
    V: float = math.nan      # Dynamic viscosity (Pa-s)
    cp: float = math.nan     # Specific heat for constant pressure mass based
    cv: float = math.nan     # Specific heat for constant volume mass based
    phase: str = ""          # Phase description
    fluid: 'Fluid' = field(default=None, repr=False)  # Reference to fluid
    
    @property
    def is_valid(self) -> bool:
        """Check if state has valid pressure"""
        return not math.isnan(self.P)
    
    def is_two_phase(self) -> bool:
        """
        Check if state is in two-phase region
        
        Uses CoolProp phase string returned by PhaseSI()
        
        Returns:
            True if phase == 'twophase', False otherwise
        """
        return self.phase.lower() == 'twophase'


class Fluid:
    """
    Fluid property calculator using CoolProp
    Matches RadComp Fluid interface
    """
    
    def __init__(self, fluid_name: str = "Air"):
        """
        Initialize fluid calculator
        
        Args:
            fluid_name: CoolProp fluid identifier (e.g., "Air", "CO2", "R134a")
        """
        self.name = fluid_name
        
        # Validate fluid availability in CoolProp
        try:
            CP.PropsSI('M', fluid_name)
        except:
            raise ValueError(f"Fluid '{fluid_name}' not available in CoolProp")
    
    def thermo_prop(self, mode: str, val1: float, val2: float) -> FluidState:
        """
        Calculate thermodynamic properties
        
        Args:
            mode: Property pair identifier ("PT", "PH", "PS", "PD", "HS")
            val1: First property value
            val2: Second property value
        
        Returns:
            FluidState object
        
        Example:
            >>> fluid = Fluid("Air")
            >>> state = fluid.thermo_prop("PT", 101325, 300)
        """
        try:
            if mode == "PT":
                P, T = val1, val2
                return FluidState(
                    P=P,
                    T=T,
                    D=CP.PropsSI('D', 'P', P, 'T', T, self.name),
                    H=CP.PropsSI('H', 'P', P, 'T', T, self.name),
                    S=CP.PropsSI('S', 'P', P, 'T', T, self.name),
                    A=CP.PropsSI('A', 'P', P, 'T', T, self.name),
                    V=CP.PropsSI('V', 'P', P, 'T', T, self.name),
                    cp=CP.PropsSI('Cpmass', 'P', P, 'T', T, self.name),
                    cv=CP.PropsSI('Cvmass', 'P', P, 'T', T, self.name),
                    phase=CP.PhaseSI('P', P, 'T', T, self.name),
                    fluid=self
                )
            
            elif mode == "PH":
                P, H = val1, val2
                return FluidState(
                    P=P,
                    H=H,
                    T=CP.PropsSI('T', 'P', P, 'H', H, self.name),
                    D=CP.PropsSI('D', 'P', P, 'H', H, self.name),
                    S=CP.PropsSI('S', 'P', P, 'H', H, self.name),
                    A=CP.PropsSI('A', 'P', P, 'H', H, self.name),
                    V=CP.PropsSI('V', 'P', P, 'H', H, self.name),
                    cp=CP.PropsSI('Cpmass', 'P', P, 'H', H, self.name),
                    cv=CP.PropsSI('Cvmass', 'P', P, 'H', H, self.name),
                    phase=CP.PhaseSI('P', P, 'H', H, self.name),
                    fluid=self
                )
            
            elif mode == "PS":
                P, S = val1, val2
                return FluidState(
                    P=P,
                    S=S,
                    T=CP.PropsSI('T', 'P', P, 'S', S, self.name),
                    D=CP.PropsSI('D', 'P', P, 'S', S, self.name),
                    H=CP.PropsSI('H', 'P', P, 'S', S, self.name),
                    A=CP.PropsSI('A', 'P', P, 'S', S, self.name),
                    V=CP.PropsSI('V', 'P', P, 'S', S, self.name),
                    cp=CP.PropsSI('Cpmass', 'P', P, 'S', S, self.name),
                    cv=CP.PropsSI('Cvmass', 'P', P, 'S', S, self.name),
                    phase=CP.PhaseSI('P', P, 'S', S, self.name),
                    fluid=self
                )
            
            elif mode == "PD":
                P, D = val1, val2
                return FluidState(
                    P=P,
                    D=D,
                    T=CP.PropsSI('T', 'P', P, 'D', D, self.name),
                    H=CP.PropsSI('H', 'P', P, 'D', D, self.name),
                    S=CP.PropsSI('S', 'P', P, 'D', D, self.name),
                    A=CP.PropsSI('A', 'P', P, 'D', D, self.name),
                    V=CP.PropsSI('V', 'P', P, 'D', D, self.name),
                    cp=CP.PropsSI('Cpmass', 'P', P, 'D', D, self.name),
                    cv=CP.PropsSI('Cvmass', 'P', P, 'D', D, self.name),
                    phase=CP.PhaseSI('P', P, 'D', D, self.name),
                    fluid=self
                )
            
            elif mode == "HS":
                H, S = val1, val2
                return FluidState(
                    H=H,
                    S=S,
                    P=CP.PropsSI('P', 'H', H, 'S', S, self.name),
                    T=CP.PropsSI('T', 'H', H, 'S', S, self.name),
                    D=CP.PropsSI('D', 'H', H, 'S', S, self.name),
                    A=CP.PropsSI('A', 'H', H, 'S', S, self.name),
                    V=CP.PropsSI('V', 'H', H, 'S', S, self.name),
                    cp=CP.PropsSI('Cpmass', 'H', H, 'S', S, self.name),
                    cv=CP.PropsSI('Cvmass', 'H', H, 'S', S, self.name),
                    phase=CP.PhaseSI('H', H, 'S', S, self.name),
                    fluid=self
                )
            
            else:
                raise ValueError(f"Unknown mode: {mode}. Use PT, PH, PS, PD, or HS")
        
        except Exception as e:
            raise ThermoException(f"Property calculation failed for {mode}({val1}, {val2}): {e}")


def static_from_total(total: FluidState, velocity: float) -> FluidState:
    """
    Calculate static properties from total properties and velocity
    
    Uses isentropic relation: h_static = h_total - V²/2, s_static = s_total
    
    Args:
        total: Total (stagnation) properties
        velocity: Flow velocity (m/s)
    
    Returns:
        Static FluidState
    """
    if not total.is_valid:
        raise ThermoException("Total properties must be valid")
    
    if total.fluid is None:
        raise ThermoException("FluidState must have fluid reference")
    
    # Static enthalpy from energy equation
    H_static = total.H - 0.5 * velocity**2
    
    # Isentropic process: entropy unchanged
    S_static = total.S
    
    # Calculate static state using HS
    return total.fluid.thermo_prop("HS", H_static, S_static)


def total_from_static(static: FluidState, velocity: float) -> FluidState:
    """
    Calculate total properties from static properties and velocity
    
    Uses isentropic relation: h_total = h_static + V²/2, s_total = s_static
    
    Args:
        static: Static properties
        velocity: Flow velocity (m/s)
    
    Returns:
        Total FluidState
    """
    if not static.is_valid:
        raise ThermoException("Static properties must be valid")
    
    if static.fluid is None:
        raise ThermoException("FluidState must have fluid reference")
    
    # Total enthalpy from energy equation
    H_total = static.H + 0.5 * velocity**2
    
    # Isentropic process: entropy unchanged
    S_total = static.S
    
    # Calculate total state using HS
    return static.fluid.thermo_prop("HS", H_total, S_total)
