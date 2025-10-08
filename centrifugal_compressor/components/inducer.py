# File: centrifugal_compressor/components/inducer.py

"""
Inducer (axial inlet section) for centrifugal compressor

Based on RadComp implementation with updated imports to match unified structure.
Calculates inlet flow conditions and friction losses using Darcy-Weisbach equation.

Velocity Notation:
    v = absolute velocity (stationary frame)
    w = relative velocity (rotating frame)
    u = blade speed (= ω * r)

Stations:
    Station 1: Inducer inlet (axial entry)
    Station 2: Inducer outlet / Impeller inlet
"""

import math
from dataclasses import InitVar, dataclass, field, fields
from typing import Type, TypeVar
from scipy import optimize

# Import from unified core package
from ..core import (
    Geometry,
    OperatingCondition,
    FluidState,
    ThermoException,
    static_from_total,
    moody,
)

State = TypeVar("State", bound="InducerState")


@dataclass
class InducerState:
    """
    Thermodynamic state at an inducer station
    
    Velocity notation:
        v = absolute velocity (m/s)
        w = relative velocity (m/s) - not used in inducer (axial, no rotation)
        u = blade speed (m/s) - zero at inducer inlet
    
    Attributes:
        total: Total (stagnation) properties
        static: Static properties
        isentropic: Isentropic state at same total pressure
        A_eff: Effective flow area (m²)
        v: Absolute velocity magnitude (m/s)
        m_abs: Absolute Mach number
        alpha: Flow angle from meridional (degrees)
    """
    total: FluidState = field(default_factory=FluidState)
    static: FluidState = field(default_factory=FluidState)
    isentropic: FluidState = field(default_factory=FluidState)
    
    A_eff: float = math.nan  # Effective flow area (m²)
    v: float = math.nan      # Absolute velocity (m/s)
    m_abs: float = math.nan  # Absolute Mach number
    alpha: float = math.nan  # Flow angle (degrees)
    
    @property
    def is_not_set(self) -> bool:
        """Check if state has been initialized"""
        return math.isnan(self.total.P)
    
    @classmethod
    def from_state(cls: Type[State], state: "InducerState") -> State:
        """Create a copy of an InducerState instance"""
        cls_fields = [f.name for f in fields(cls)]
        content = {k: v for k, v in state.__dict__.items() if k in cls_fields}
        return cls(**content)


@dataclass
class Inducer:
    """
    Inducer component (axial inlet section)
    
    Calculates flow through the inducer using continuity equation
    and Darcy-Weisbach friction loss model.
    
    The inducer is axial with no rotation, so:
        - Absolute velocity v = meridional velocity
        - Relative velocity w = v (no blade motion)
        - Blade speed u = 0 at inlet
    
    Args:
        geom: Geometry definition
        op: Operating conditions
    
    Attributes:
        in1: Inlet state (Station 1)
        out: Outlet state (Station 2 - impeller inlet)
        dh0s: Isentropic enthalpy rise (J/kg)
        eff: Inducer efficiency
        choke_flag: True if flow is choked
        heat: External heat addition (W)
    """
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    
    in1: InducerState = field(init=False)
    out: InducerState = field(default_factory=InducerState)
    
    dh0s: float = math.nan   # Isentropic enthalpy rise (J/kg)
    eff: float = math.nan    # Efficiency
    choke_flag: bool = False
    heat: float = 0.0        # Heat addition (W)
    
    def __post_init__(self, geom: Geometry, op: OperatingCondition) -> None:
        """Initialize inducer and calculate if needed"""
        self.in1 = InducerState(total=op.inlet_state)
        
        if self.out.is_not_set:
            try:
                self.calculate(geom, op)
            except ThermoException as e:
                self.choke_flag = True
                print(f"ThermoException occurred: {e}")

    def calculate(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Calculate inducer performance
        
        Solves for:
        1. Inlet velocity (Station 1) from continuity
        2. Outlet state (Station 2) with friction losses
        
        Uses Darcy-Weisbach equation for pressure drop:
            ΔP = 4 * Cf * (L/D) * (1/2) * ρ * v²
        
        Args:
            geom: Geometry definition
            op: Operating conditions
        """
        in_total = self.in1.total
        
        # ====================================================================
        # STEP 1: Solve for inlet velocity (Station 1)
        # ====================================================================
        def resolve_v1(x):
            """Continuity equation at Station 1"""
            v1 = x  # Absolute velocity
            try:
                stat1 = static_from_total(in_total, v1)
                err = (op.mass_flow - geom.A1_eff * v1 * stat1.D) / op.mass_flow
            except ThermoException:
                return 1e3
            return err
        
        # Initial guess for inlet velocity
        v1_guess = op.mass_flow / geom.A1_eff / in_total.D
        
        # Check for choke at inlet
        if v1_guess / in_total.A > 1.5:
            self.choke_flag = True
            return
        
        # Solve for inlet velocity
        sol = optimize.root(resolve_v1, x0=v1_guess)
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return
        
        v1 = sol.x[0]
        
        # Assign inlet state
        self.in1.v = v1
        self.in1.A_eff = geom.A1_eff
        self.in1.static = static_from_total(in_total, v1)
        self.in1.m_abs = v1 / self.in1.static.A
        
        # Check for choke in inducer passage
        if self.in1.m_abs * geom.A1_eff / geom.A2_eff >= 0.99:
            self.choke_flag = True
            return
        
        # ====================================================================
        # STEP 2: Solve for outlet state (Station 2) with friction
        # ====================================================================
        def resolve_out(x):
            """
            Coupled equations:
            1. Continuity at Station 2
            2. Pressure drop from Darcy-Weisbach
            """
            v2, Pout = x  
            
            try:
                # Total state at outlet (with heat addition)
                tot2 = op.fluid.thermo_prop(
                    "PH", 
                    Pout, 
                    in_total.H + self.heat / op.mass_flow
                )
                stat2 = static_from_total(tot2, v2)
                
                # Continuity error
                err_cont = (op.mass_flow - geom.A2_eff * v2 * stat2.D) / op.mass_flow
                
                # Friction loss (Darcy-Weisbach)
                Re = v2 * 2 * geom.r2s * stat2.D / stat2.V
                Cf = moody(Re, geom.rough_inducer / (2 * geom.r2s))
                dP = 4 * Cf * geom.l_inducer * stat2.D * v2**2 / (4 * geom.r2s)

                # Pressure error (total pressure drop = friction loss)
                Pout_calc = in_total.P - dP
                err_press = (Pout_calc - tot2.P) / in_total.P
                
            except ThermoException:
                return [1e3, 1e3]
            
            return [err_cont, err_press]
        
        # Initial guess for outlet
        v2_guess = op.mass_flow / geom.A2_eff / self.in1.static.D
        Re_guess = v2_guess * 2 * geom.r2s * self.in1.static.D / self.in1.static.V
        Cf_guess = moody(Re_guess, geom.rough_inducer / (2 * geom.r2s))
        dP_guess = 4 * Cf_guess * geom.l_inducer * v2_guess**2 / (4 * geom.r2s) * self.in1.static.D
        Pout_guess = in_total.P - dP_guess
        
        # Solve coupled system
        sol = optimize.root(resolve_out, x0=[v2_guess, Pout_guess], tol=1e-4)
        
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return
        
        v2, Pout = sol.x
        
        # ====================================================================
        # STEP 3: Assign outlet state
        # ====================================================================
        self.out = InducerState(
            total=op.fluid.thermo_prop("PH", Pout, in_total.H + self.heat / op.mass_flow),
            isentropic=op.fluid.thermo_prop("PS", Pout, in_total.S),
        )
        
        self.out.v = v2
        self.out.static = static_from_total(self.out.total, v2)
        self.out.m_abs = v2 / self.out.static.A
        self.out.A_eff = geom.A2_eff
        
        # Calculate performance metrics
        self.dh0s = self.out.isentropic.H - self.in1.total.H
        delta_h = self.out.total.H - self.in1.total.H
        
        if abs(delta_h) <= 1e-6:
            self.eff = math.copysign(math.inf, self.dh0s)
        else:
            self.eff = self.dh0s / delta_h


# ============================================================================
# Package exports
# ============================================================================
__all__ = ['Inducer', 'InducerState']
