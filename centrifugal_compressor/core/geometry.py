# File: centrifugal_compressor/core/geometry.py
"""
Geometry definitions for centrifugal compressor components
Based on RadComp structure with both hydraulic diameter methods
"""

import math
from dataclasses import dataclass, fields
from typing import List, Union, Tuple


def cosd(degrees: float) -> float:
    """Cosine of angle in degrees"""
    return math.cos(math.radians(degrees))


def sind(degrees: float) -> float:
    """Sine of angle in degrees"""
    return math.sin(math.radians(degrees))


def tand(degrees: float) -> float:
    """Tangent of angle in degrees"""
    return math.tan(math.radians(degrees))


@dataclass
class Geometry:
    """
    Complete centrifugal compressor geometry
    Exactly matches RadComp structure
    """
    # Station 1: Inducer inlet
    r1: float                    # Inducer inlet radius (m)
    
    # Station 2: Impeller inlet
    r2s: float                   # Impeller inlet tip radius (m)
    r2h: float                   # Impeller inlet hub radius (m)
    beta2: float                 # Impeller blade angle at inlet at rms radius (degrees)
    beta2s: float                # Impeller blade angle at inlet at tip radius (degrees)
    alpha2: float                # Inlet flow angle (degrees)
    
    # Station 4: Impeller outlet
    r4: float                    # Impeller outlet tip radius (m)
    b4: float                    # Impeller outlet width at tip (m)
    beta4: float                 # Impeller outlet blade angle (degrees)
    
    # Station 5: Vaneless Diffuser outlet
    r5: float                    # Vaneless Diffuser outlet radius (m)
    b5: float                    # Vaneless Diffuser outlet width (m)
    
    # Station 6: Vaned Diffuser
    beta5: float                 # Vaned Diffuser inlet flow angle (degrees)
    r6: float                    # Vaned Diffuser outlet radius (m)
    b6: float                    # Vaned Diffuser outlet width (m)
    beta6: float                 # Vaned Diffuser outlet blade angle (degrees)
    
    # Blade counts
    n_blades: int                # Number of blades in the impeller
    n_splits: int                # Number of splitter blades in the impeller
    n_vanes: int                 # Number of vanes in the diffuser
    
    # Blade thickness
    blade_le: float              # Blade thickness at the leading edge (m)
    blade_te: float              # Blade thickness at the trailing edge (m)
    
    # Clearances
    tip_cl: float                # Tip clearance (m)
    back_cl: float               # Backface clearance (m)
    
    # Surface properties
    rough_inducer: float         # Surface roughness of the inducer (m)
    
    # Lengths
    l_inducer: float             # Length of the inducer (m)
    l_comp: float                # Length of the compressor (m)
    
    # Blockage factors [inducer, impeller_inlet, impeller_throat, impeller_outlet, diffuser, vaned_diffuser]
    blockage: List[float]        # Blockage factors for each station
    
    @property
    def r2rms(self) -> float:
        """Impeller inlet radius at root-mean-square"""
        return math.sqrt((self.r2s**2 + self.r2h**2) / 2.0)
    
    @property
    def A1_eff(self) -> float:
        """Effective flow area at inducer inlet"""
        return math.pi * self.r1**2 * self.blockage[0]
    
    @property
    def A2_eff(self) -> float:
        """Effective flow area at impeller inlet"""
        return math.pi * (self.r2s**2 - self.r2h**2) * self.blockage[1] * cosd(self.alpha2)
    
    @property
    def A_x(self) -> float:
        """Flow area at station 2"""
        return math.pi * (self.r2s**2 - self.r2h**2) * self.blockage[1] * cosd(self.beta2)
    
    @property
    def A_y(self) -> float:
        """Flow area at station 3 at Impeller throat"""
        return ((math.pi * (self.r2s**2 - self.r2h**2) * cosd(self.beta2) - 
                (self.r2s - self.r2h) * self.blade_le * self.n_blades) * self.blockage[2])
    
    @property
    def A4_eff(self) -> float:
        """Effective flow area at impeller outlet"""
        return 2 * math.pi * self.r4 * self.b4 * self.blockage[3]
    
    @property
    def beta2_opt(self) -> float:
        """Optimal flow angle at impeller inlet (degrees)"""
        return math.degrees(math.atan(self.A_x / self.A_y * tand(self.beta2)))
    
    @property
    def A5_eff(self) -> float:
        """Effective flow area at vaneless diffuser outlet"""
        blockage = self.blockage[4] if len(self.blockage) > 4 else 1.0
        return 2 * math.pi * self.r5 * self.b5 * blockage
    
    @property
    def A6_eff(self) -> float:
        """Effective flow area at vaned diffuser outlet"""
        blockage = self.blockage[5] if len(self.blockage) > 5 else 1.0
        return 2 * math.pi * self.r6 * self.b6 * blockage
    
    @property
    def slip(self) -> float:
        """Slip factor (Wiesner correlation)"""
        slip_factor = 1.0 - cosd(self.beta4)**0.5 / (self.n_blades + self.n_splits) ** 0.7
        
        sf_star = sind(19 + 0.2 * (90 - self.beta4))
        r1r2_lim = (slip_factor - sf_star) / (1 - sf_star)
        
        if (self.r2rms / self.r4) > r1r2_lim:
            exp = ((90 - self.beta4) / 10) ** 0.5
            slip_factor = slip_factor * (1 - ((((self.r2rms / self.r4) - r1r2_lim) / (1 - r1r2_lim))) ** exp)
        
        return slip_factor
    
    @property
    def hydraulic_diameter_galvas(self) -> Tuple[float, float]:
        """
        RadComp original hydraulic diameter calculation
        
        Returns:
            tuple: (Dh, Lh) in meters
        
        Note: This is the ORIGINAL RadComp formula
        Contains empirical factor 0.3048 (1 foot) in Lh calculation
        """
        la = self.r2h / self.r2s
        
        # Hydraulic diameter
        term1 = 1.0 / (self.n_blades / math.pi / cosd(self.beta4) + 2.0 * self.r4 / self.b4)
        
        term2_denom = (2.0 / (1.0 - la) + 
                       2.0 * self.n_blades / math.pi / (1 + la) * 
                       math.sqrt(1 + (1 + la**2 / 2) * tand(self.beta2s)**2))
        
        term2 = self.r2s / self.r4 / term2_denom
        
        Dh = 2 * self.r4 * (term1 + term2)
        
        # Hydraulic length (contains magic number 0.3048)
        Lh = self.r4 * (1 - self.r2rms * 2 / 0.3048) / cosd(self.beta4)
        
        return Dh, Lh
    
    @property
    def hydraulic_diameter_jansen(self) -> Tuple[float, float]:
        """
        Jansen (1967) hydraulic diameter calculation
        Used by Oh, Zhang et al. loss models
        
        Returns:
            tuple: (Dh, Lb) in meters
        
        This is the STANDARD formulation used in all modern loss models
        """
        # Hydraulic diameter - outlet contribution
        Dh_outlet = (2 * self.r4 * cosd(self.beta4) / 
                     (self.n_blades / math.pi + 2 * self.r4 * cosd(self.beta4) / self.b4))
        
        # Hydraulic diameter - inlet contribution
        Dh_inlet = (0.5 * (self.r2s + self.r2h) / self.r4 * cosd(self.beta2) / 
                    (self.n_blades / math.pi + 
                     (self.r2s + self.r2h) / (self.r2s - self.r2h) * cosd(self.beta2)))
        
        Dh = Dh_outlet + Dh_inlet
        
        # Blade length - Aungier formulation
        L_ax = (0.5 * (self.r4 - self.r2s) + self.b4)
        Lb = ((math.pi / 8) * 
              (2 * self.r4 - (self.r2s + self.r2h) - self.b4 + 2 * L_ax) * 
              (2 / (cosd(self.beta2) + cosd(self.beta4))))
        
        return Dh, Lb
    
    
    def get_hydraulic_parameters(self, method: str = 'jansen') -> Tuple[float, float]:
        """
        Get hydraulic diameter and blade length using specified method
        
        Args:
            method: Calculation method
                'Galvas' - Original RadComp formula
                'jansen' - Jansen (1967) used by Oh, Zhang (default)
        
        Returns:
            tuple: (Dh, Lb) in meters
        """
        if method == 'Galvas':
            return self.hydraulic_diameter_galvas
        elif method == 'jansen':
            return self.hydraulic_diameter_jansen
        else:
            raise ValueError(f"Unknown method: {method}. Use 'Galvas', or 'jansen' ")

    @classmethod
    def from_dict(cls, data: dict, blockage: Union[List[float], None] = None):
        """Create a Geometry instance from a dictionary"""
        safe_names = [f.name for f in fields(cls)]
        d = {}
        
        if blockage is None and "blockage1" in data:
            blockage = [data[f"blockage{i+1}"] for i in range(6)]
        
        if blockage is None:
            raise ValueError("Blockage needs to be provided as an argument or in data.")
        
        for k, v in data.items():
            if k.lower() in safe_names:
                d[k.lower()] = v
        
        d["blockage"] = blockage
        
        return cls(**d)


# ============================================================================
# OPERATING CONDITIONS
# ============================================================================
from typing import List, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
if TYPE_CHECKING:
    from .thermodynamics import Fluid, FluidState
@dataclass
class OperatingCondition:
    """
    Operating conditions for centrifugal compressor
    
    Defines mass flow, rotational speed, and inlet state.
    Automatically initializes fluid and inlet thermodynamic state.
    
    Attributes:
        mass_flow: Mass flow rate (kg/s)
        omega: Rotational speed (rad/s)
        P_inlet: Inlet total pressure (Pa)
        T_inlet: Inlet total temperature (K)
        fluid_name: Working fluid name (default: "Air")
    
    Properties:
        rpm: Rotational speed in RPM
        n_rot: Alias for omega (rad/s)
        fluid: Fluid object (lazy initialization)
        inlet_state: Inlet total thermodynamic state (lazy initialization)
    
    Example:
        >>> op = OperatingCondition.from_rpm(
        ...     mass_flow=0.5, rpm=50000,
        ...     P_inlet=101325, T_inlet=300
        ... )
        >>> print(op.inlet_state.P, op.inlet_state.T)
        101325 300
    """
    # Core operating parameters
    mass_flow: float             # Mass flow rate (kg/s)
    omega: float                 # Rotational speed (rad/s)
    P_inlet: float               # Inlet total pressure (Pa)
    T_inlet: float               # Inlet total temperature (K)
    fluid_name: str = "Air"      # Working fluid

    # Internal cached objects (lazy initialization)
    _fluid: 'Fluid' = field(default=None, init=False, repr=False)
    _inlet_state: 'FluidState' = field(default=None, init=False, repr=False)
    
    @property
    def rpm(self) -> float:
        """Rotational speed in RPM"""
        return self.omega * 60 / (2 * math.pi)
    
    @property
    def n_rot(self) -> float:
        """Rotational speed in rad/s (alias for omega)"""
        return self.omega
    
    @property
    def fluid(self) -> 'Fluid':
        """
        Working fluid object (lazy initialization)
        
        Creates Fluid instance on first access and caches it.
        """
        if self._fluid is None:
            # Import here to avoid circular dependency
            from .thermodynamics import Fluid
            self._fluid = Fluid(self.fluid_name)
        return self._fluid
    
    @property
    def inlet_state(self) -> 'FluidState':
        """
        Inlet total thermodynamic state (lazy initialization)
        
        Creates FluidState from P_inlet and T_inlet on first access.
        """
        if self._inlet_state is None:
            self._inlet_state = self.fluid.thermo_prop('PT', self.P_inlet, self.T_inlet)
        return self._inlet_state
    
    @classmethod
    def from_rpm(cls, mass_flow: float, rpm: float, 
                 P_inlet: float, T_inlet: float, 
                 fluid_name: str = "Air") -> "OperatingCondition":
        """
        Create operating condition from RPM instead of rad/s
        
        Args:
            mass_flow: Mass flow rate (kg/s)
            rpm: Rotational speed (RPM)
            P_inlet: Inlet total pressure (Pa)
            T_inlet: Inlet total temperature (K)
            fluid_name: Working fluid (default: "Air")
        
        Returns:
            OperatingCondition with omega calculated from RPM
        
        Example:
            >>> op = OperatingCondition.from_rpm(
            ...     mass_flow=0.5, rpm=50000,
            ...     P_inlet=101325, T_inlet=300
            ... )
        """
        omega = rpm * 2 * math.pi / 60
        return cls(mass_flow, omega, P_inlet, T_inlet, fluid_name)

def create_example_geometry() -> Geometry:
    """Create example geometry matching typical RadComp values"""
    return Geometry(
        r1=0.025,
        r2s=0.025,
        r2h=0.010,
        beta2=-60,
        beta2s=-65,
        alpha2=0,
        r4=0.050,
        b4=0.004,
        beta4=-50,
        r5=0.070,
        b5=0.004,
        beta5=70,
        r6=0.090,
        b6=0.004,
        beta6=45,
        n_blades=12,
        n_splits=0,
        n_vanes=18,
        blade_le=0.001,
        blade_te=0.0005,
        tip_cl=0.0001,
        back_cl=0.0001,
        rough_inducer=1.5e-6,
        l_inducer=0.01,
        l_comp=0.05,
        blockage=[0.95, 0.95, 0.90, 0.95, 0.95, 0.95],
    )


def create_example_operating_condition() -> OperatingCondition:
    """Create example operating condition"""
    return OperatingCondition.from_rpm(
        mass_flow=0.5,
        rpm=50000,
        P_inlet=101325,
        T_inlet=300,
        fluid_name="Air"
    )


# Comparison utility
def compare_hydraulic_diameter_methods(geom: Geometry) -> dict:
    """
    Compare all three hydraulic diameter calculation methods
    
    Args:
        geom: Geometry object
    
    Returns:
        dict: Comparison results
    """
    Dh_radcomp, Lh_radcomp = geom.hydraulic_diameter_galvas
    Dh_jansen, Lb_jansen = geom.hydraulic_diameter_jansen
    
    return {
        'galvas': {'Dh_mm': Dh_radcomp*1000, 'Lb_mm': Lh_radcomp*1000},
        'jansen': {'Dh_mm': Dh_jansen*1000, 'Lb_mm': Lb_jansen*1000},
    }
