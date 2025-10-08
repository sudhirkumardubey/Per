"""
Unified Diffuser Module - Handles Vaneless, Vaned, or Both

Automatically detects configuration from geometry:
1. Vaneless only: r5 > r4, n_vanes = 0
2. Vaned only: r6 > r5, n_vanes > 0, r5 ≈ r4 (no vaneless section)
3. Vaneless + Vaned: r5 > r4 AND r6 > r5, n_vanes > 0 (both in series)

Author: Centrifugal Compressor Team
Date: October 2025
"""

import math
from dataclasses import InitVar, dataclass, field
from typing import Dict, Optional, Literal
from math import cos, pi, sin, tan, sqrt, atan2, asin
import numpy as np
from scipy import optimize

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.core.thermodynamics import Fluid, FluidState, ThermoException, total_from_static, static_from_total
from centrifugal_compressor.components.impeller import Impeller, ImpellerState
from centrifugal_compressor.losses.calculator import LossCalculator


# ============================================================================
# Helper functions
# ============================================================================

def radians(deg: float) -> float:
    return deg * pi / 180

def degrees(rad: float) -> float:
    return rad * 180 / pi

def cosd(deg: float) -> float:
    return cos(radians(deg))

def sind(deg: float) -> float:
    return sin(radians(deg))

def tand(deg: float) -> float:
    return tan(radians(deg))

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DiffuserState:
    """Diffuser state at inlet or outlet"""
    # Thermodynamic states
    total: FluidState = field(default_factory=FluidState)
    static: FluidState = field(default_factory=FluidState)
    
    # Velocities [m/s]
    v: float = math.nan      # Absolute velocity
    v_m: float = math.nan    # Meridional component
    v_t: float = math.nan    # Tangential component
    
    # Flow angles [deg]
    alpha: float = math.nan  # Absolute flow angle
    
    # Mach number
    mach: float = math.nan
    
    @classmethod
    def from_impeller_state(cls, imp_state: ImpellerState):
        """Create diffuser state from impeller outlet"""
        return cls(
            total=imp_state.total,
            static=imp_state.static,
            v=imp_state.v,
            v_m=imp_state.v_m,
            v_t=imp_state.v_t,
            alpha=imp_state.alpha,
            mach=imp_state.v / imp_state.static.a if hasattr(imp_state.static, 'a') else math.nan
        )


@dataclass
class DiffuserLosses:
    """Diffuser loss breakdown"""
    breakdown: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    
    # Individual loss properties
    @property
    def friction(self) -> float:
        return self.breakdown.get('friction', 0.0)
    
    @property
    def incidence(self) -> float:
        return self.breakdown.get('incidence', 0.0)


# ============================================================================
# VANELESS DIFFUSER (RadComp Physics Engine)
# ============================================================================

@dataclass
class VanelessDiffuser:
    """
    Vaneless diffuser using RadComp 1D marching solution
    
    Physics:
    - Conservation of angular momentum: r*v_theta = const
    - Continuity equation with friction
    - Stepwise integration from r4 to r5
    """
    
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    imp: InitVar[Impeller]
    
    # States
    inlet: DiffuserState = field(init=False)
    outlet: DiffuserState = field(default_factory=DiffuserState)
    
    # Performance
    losses: DiffuserLosses = field(default_factory=DiffuserLosses)
    eff_ts: float = math.nan  # Total-to-static efficiency
    pr_ts: float = math.nan   # Pressure ratio
    dh0s: float = math.nan

    # Flags
    choke_flag: bool = False
    
    # Calculation parameters
    n_steps: int = 15
    
    def __post_init__(self, geom: Geometry, op: OperatingCondition, inlet_state: DiffuserState):
        """Initialize and calculate diffuser performance"""
        self.inlet = inlet_state
        self._calculate(geom, op)
    
    def _calculate(self, geom: Geometry, op: OperatingCondition):
        """
        RadComp 1D marching solution with friction
        
        References:
        - Japikse & Baines (1994)
        - Conrad et al. (1979)
        """
        # Discretize geometry
        r = np.linspace(geom.r4, geom.r5, 1 + self.n_steps, endpoint=True)
        dr = np.diff(r)
        b = np.linspace(geom.b4, geom.b5, 1 + self.n_steps, endpoint=True)
        
        # Hydraulic diameter
        Dh = np.sqrt(8 * r[:-1] * b[1:] * geom.blockage[4])
        
        # Effective area
        A_eff = 2 * r[1:] * b[1:] * pi * geom.blockage[4]
        
        # Friction coefficient correlation (Japikse)
        k_friction = 0.02
        
        def marching_solver(v_m_guess, return_state=False):
            """Solve for velocity distribution"""
            state = DiffuserState.from_impeller_state(self.inlet)
            errors = []
            
            for i in range(self.n_steps):
                # Calculate friction coefficient
                Re = state.v * state.static.D * b[i+1] / state.static.V
                Cf = k_friction * (1.8e5 / Re) ** 0.2
                
                # Path length
                ds = sqrt((dr[i] / tan(math.radians(90 - state.alpha)))**2 + dr[i]**2)
                
                # Pressure loss due to friction
                dp0_friction = 4.0 * Cf * ds * state.v**2 * state.static.D / (2 * Dh[i])
                
                # Tangential velocity (angular momentum conservation)
                v_t4 = state.v * sin(math.radians(state.alpha))
                v_m4 = state.v * cos(math.radians(state.alpha))
                
                # Tangential velocity change with friction
                dv_t_dr = -(
                    v_t4 / r[i] 
                    + Cf * state.v**2 * sin(math.radians(state.alpha)) / (v_m4 * b[i+1])
                ) * dr[i]
                
                v_t5 = v_t4 + dv_t_dr
                
                # Total pressure at next station
                P0_next = state.total.P - dp0_friction
                
                # Check for validity
                if P0_next <= 0 or P0_next < op.inlet_state.P:
                    errors.extend([1e4] * (self.n_steps - i))
                    return errors
                
                # New total state
                try:
                    total_next = op.fluid.thermo_prop('PH', P0_next, state.total.H)
                except ThermoException:
                    errors.extend([1e4] * (self.n_steps - i))
                    return errors
                
                # Meridional velocity from guess
                v_m5 = v_m_guess[i]
                v5 = sqrt(v_m5**2 + v_t5**2)
                
                # Check for choke
                if v5 > 1.25 * total_next.A:
                    errors.extend([1e4] * (self.n_steps - i))
                    return errors
                
                # Static state
                try:
                    static_next = static_from_total(total_next, v5)
                except ThermoException:
                    errors.extend([1e4] * (self.n_steps - i))
                    return errors
                
                # Continuity error
                mass_flow_calc = A_eff[i] * v_m5 * static_next.D
                errors.append((op.mass_flow - mass_flow_calc) / op.mass_flow)
                
                # Update state
                state.v = v5
                state.v_m = v_m5
                state.v_t = v_t5
                state.alpha = math.degrees(math.asin(v_t5 / v5))
                state.total = total_next
                state.static = static_next
                state.mach = v5 / static_next.A
                
                # Check Mach number
                if state.mach >= 0.99:
                    errors[-1] += state.mach - 0.99
            
            if return_state:
                return errors, state
            return errors
        
        # Initial guess (momentum conservation)
        v_m4 = self.inlet.v * cos(math.radians(self.inlet.alpha))
        v_m_guess = v_m4 * r[:-1] / r[1:]

        # Check inlet choking
        if v_m4 / self.inlet.static.A >= 0.99:
            self.choke_flag = True
            return
        
        # Solve system
        sol = optimize.root(marching_solver, x0=v_m_guess, method='hybr')
        
        if not sol.success or (np.abs(sol.fun) > 0.001).any():
            self.choke_flag = True
            return
        
        # Extract solution
        _, outlet_state = marching_solver(sol.x, return_state=True)

        # Final check
        if outlet_state.mach >= 0.99:
            self.choke_flag = True
            return
        
        self.outlet = outlet_state
        
        # Calculate performance metrics
        self._calculate_performance(op)
    
    def _calculate_performance(self, op: OperatingCondition):
        """Calculate efficiency and pressure ratio"""
        # Isentropic outlet
        try:
            outlet_is = op.fluid.thermo_prop('PS', self.outlet.total.P, self.inlet.total.S)
        except ThermoException:
            return
        
        # Losses
        self.losses.total = self.outlet.total.H - outlet_is.H
        self.losses.breakdown['friction'] = self.losses.total
        
        # Efficiency
        dh_isen = outlet_is.H - self.inlet.total.H
        dh_actual = self.outlet.total.H - self.inlet.total.H
        
        if abs(dh_actual) > 1e-6:
            self.eff_ts = dh_isen / dh_actual
        else:
            self.eff_ts = math.copysign(math.inf, dh_isen)
        
        # Pressure ratio
        self.pr_ts = self.outlet.static.P / self.inlet.total.P

"""
Vaned Diffuser - Using LossCalculator for loss computation

"""
# ============================================================================
# VANED DIFFUSER (LossCalculator)
# ============================================================================

@dataclass
class VanedDiffuser:
    """
    Vaned diffuser using TurboFlow flow solver + LossCalculator
    
    Flow Model:
    - 1D velocity triangles (continuity + geometry)
    - Deviation model (optional)
    - Loss calculation via LossCalculator
    
    Supported loss models:
    - 'meroni': Full vaned diffuser implementation (Conrad method)
    - Others: Fallback to empirical correlations
    """
    
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    inlet_state: InitVar[DiffuserState]
    loss_model: str = 'meroni'
    
    inlet: DiffuserState = field(init=False)
    outlet: DiffuserState = field(default_factory=DiffuserState)
    losses: DiffuserLosses = field(default_factory=DiffuserLosses)
    eff_ts: float = math.nan
    pr_ts: float = math.nan
    pr_tt: float = math.nan
    choke_flag: bool = False
    
    def __post_init__(self, geom: Geometry, op: OperatingCondition, inlet_state: DiffuserState):
        """Initialize and calculate"""
        self.inlet = inlet_state
        self.loss_calc = LossCalculator(self.loss_model)
        self._calculate(geom, op)
    
    def _calculate(self, geom: Geometry, op: OperatingCondition):
        """
        Main calculation routine
        
        Steps:
        1. Solve for outlet velocity triangle (continuity + geometry)
        2. Calculate losses using LossCalculator
        3. Apply losses to get outlet thermodynamic state
        4. Compute performance metrics
        """
        # ====================================================================
        # STEP 1: Outlet velocity triangle
        # ====================================================================
        
        # Inlet
        v_in = self.inlet.v
        v_m_in = self.inlet.v_m
        v_t_in = self.inlet.v_t
        alpha_in = self.inlet.alpha
        h0_in = self.inlet.total.H
        s_in = self.inlet.total.S
        
        # Geometry
        r5, r6 = geom.r5, geom.r6
        b5, b6 = geom.b5, geom.b6
        theta_in = geom.beta5   # Vane inlet angle [deg]
        theta_out = geom.beta6  # Vane outlet angle [deg]
        A_in = geom.A5_eff
        A_out = geom.A6_eff
        
        # Flow deviation (simple assumption: alpha_out = theta_out)
        # Can be replaced with Aungier deviation model if available
        alpha_out = theta_out
        
        # Continuity: v_m_out ≈ v_m_in * A_in / A_out
        v_m_out_guess = v_m_in * A_in / A_out
        v_t_out_guess = v_m_out_guess * tand(alpha_out)
        v_out_guess = sqrt(v_m_out_guess**2 + v_t_out_guess**2)
        
        # Check for choking
        if v_out_guess / self.inlet.total.a > 0.99:
            self.choke_flag = True
            return
        
        # ====================================================================
        # STEP 2: Calculate losses using LossCalculator
        # ====================================================================
        
        # Prepare velocity dictionary for loss calculator
        velocities = {
            'v_in': v_in,
            'v_out': v_out_guess,
            'v_m_in': v_m_in,
            'v_m_out': v_m_out_guess,
            'v_t_in': v_t_in,
            'v_t_out': v_t_out_guess,
            'alpha_in': alpha_in,
            'alpha_out': alpha_out,
            'w_in': v_in,  # For vaned diffuser, w ≈ v (no rotation)
            'w_out': v_out_guess,
        }
        
        # Estimate outlet static state (for loss calculator)
        h_out_guess = h0_in - 0.5 * v_out_guess**2
        
        try:
            static_out_guess = op.fluid.thermo_prop('HS', h_out_guess, s_in)
            total_out_guess = op.fluid.thermo_prop('HS', h0_in, s_in)
        except ThermoException:
            self.choke_flag = True
            return
        
        # Compute losses
        try:
            loss_dict = self.loss_calc.compute_vaned_diffuser_losses(
                geom=geom,
                inlet_state=self.inlet.static,
                outlet_state=static_out_guess,
                velocities=velocities,
                Cf=0.005  # Friction coefficient
            )
        except NotImplementedError:
            # Fallback to empirical correlations
            loss_dict = self._compute_losses_empirical(
                geom, v_in, v_out_guess, v_m_in, v_m_out_guess,
                alpha_in, alpha_out, theta_in
            )
        
        self.losses.breakdown = loss_dict.copy()
        self.losses.total = sum(loss_dict.values())
        
        # ====================================================================
        # STEP 3: Apply losses and solve for outlet state
        # ====================================================================
        
        # Update entropy with losses
        T_avg = (self.inlet.static.T + static_out_guess.T) / 2
        s_out = s_in + self.losses.total / T_avg
        
        # Update outlet thermodynamic states
        h_out = h0_in - 0.5 * v_out_guess**2
        
        try:
            static_out_final = op.fluid.thermo_prop('HS', h_out, s_out)
            total_out_final = op.fluid.thermo_prop('HS', h0_in, s_out)
        except ThermoException:
            self.choke_flag = True
            return
        
        # Refine with continuity
        rho_out = static_out_final.rho
        m_calc = rho_out * v_m_out_guess * A_out
        
        # Check mass flow error
        mass_flow_error = abs(m_calc - op.mass_flow) / op.mass_flow
        
        if mass_flow_error > 0.01:
            # Adjust velocity
            v_m_out = v_m_out_guess * op.mass_flow / m_calc
            v_t_out = v_m_out * tand(alpha_out)
            v_out = sqrt(v_m_out**2 + v_t_out**2)
            
            # Re-calculate static state
            h_out = h0_in - 0.5 * v_out**2
            try:
                static_out_final = op.fluid.thermo_prop('HS', h_out, s_out)
                total_out_final = op.fluid.thermo_prop('HS', h0_in, s_out)
            except ThermoException:
                self.choke_flag = True
                return
        else:
            v_m_out = v_m_out_guess
            v_t_out = v_t_out_guess
            v_out = v_out_guess
        
        # ====================================================================
        # STEP 4: Store outlet state
        # ====================================================================
        
        self.outlet.total = total_out_final
        self.outlet.static = static_out_final
        self.outlet.v = v_out
        self.outlet.v_m = v_m_out
        self.outlet.v_t = v_t_out
        self.outlet.alpha = alpha_out
        self.outlet.mach = v_out / static_out_final.a
        
        self.converged = True
        
        # ====================================================================
        # STEP 5: Calculate performance metrics
        # ====================================================================
        
        self._calculate_performance(op)
    
    def _compute_losses_empirical(
        self, geom, v_in, v_out, v_m_in, v_m_out,
        alpha_in, alpha_out, theta_in
    ) -> dict:
        """
        Fallback empirical loss correlations
        Returns losses in J/kg
        """
        r5, r6 = geom.r5, geom.r6
        b5, b6 = geom.b5, geom.b6
        z = geom.n_vanes
        
        # Incidence loss
        alpha_inc = alpha_in - theta_in
        Y_inc = 0.5 * (v_in * sind(alpha_inc))**2
        
        # Skin friction loss
        L_vane = sqrt((r6 - r5)**2 + (radians((theta_in + alpha_out) / 2) * (r5 + r6) / 2)**2)
        D_h = 2 * r5 * b5 / (z / pi + 2)
        Re = self.inlet.static.rho * v_in * D_h / self.inlet.static.mu
        Cf = 0.04 / Re**0.16
        Y_sf = Cf * (L_vane / D_h) * 0.5 * ((v_in + v_out) / 2)**2
        
        # Total loss
        Y_tot = Y_inc + Y_sf
        
        return {
            'incidence': Y_inc,
            'skin_friction': Y_sf,
            'loss_total': Y_tot,
        }
    
    def _calculate_performance(self, op: OperatingCondition):
        """Calculate efficiency and pressure ratios"""
        try:
            outlet_is = op.fluid.thermo_prop('PS', self.outlet.total.P, self.inlet.total.S)
        except ThermoException:
            return
        
        # Efficiency
        dh_isen = outlet_is.H - self.inlet.total.H
        dh_actual = self.outlet.total.H - self.inlet.total.H
        
        if abs(dh_actual) > 1e-6:
            self.eff_ts = dh_isen / dh_actual
        else:
            self.eff_ts = math.copysign(math.inf, dh_isen)
        
        # Pressure ratios
        self.pr_ts = self.outlet.static.P / self.inlet.total.P
        self.pr_tt = self.outlet.total.P / self.inlet.total.P
        
        # Static pressure recovery coefficient
        dynamic_head = 0.5 * self.inlet.static.rho * self.inlet.v**2
        self.cp_static = (self.outlet.static.P - self.inlet.static.P) / dynamic_head


# ============================================================================
# UNIFIED DIFFUSER (Auto-detects configuration)
# ============================================================================

@dataclass
class Diffuser:
    """
    Unified diffuser that automatically handles:
    
    1. Vaneless only: r5 > r4, n_vanes = 0
    2. Vaned only: n_vanes > 0, r5 ≈ r4 (no vaneless section)
    3. Vaneless + Vaned: r5 > r4, r6 > r5, n_vanes > 0 (both in series)
    
    Configuration is auto-detected from Geometry.
    """
    
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    imp: InitVar[Impeller]
    loss_model: str = 'meroni'
    
    # Configuration
    config: Literal['vaneless_only', 'vaned_only', 'both'] = field(init=False)
    
    # Components (populated based on config)
    vaneless: Optional[VanelessDiffuser] = None
    vaned: Optional[VanedDiffuser] = None
    
    # Unified interface (delegates to appropriate component)
    inlet: DiffuserState = field(init=False)
    outlet: DiffuserState = field(init=False)
    losses: DiffuserLosses = field(init=False)
    eff_ts: float = math.nan
    pr_ts: float = math.nan
    choke_flag: bool = False
    
    def __post_init__(self, geom: Geometry, op: OperatingCondition, imp: Impeller):
        """Auto-detect configuration and initialize appropriate diffuser(s)"""
        
        # Convert impeller outlet to diffuser inlet
        inlet_state = DiffuserState.from_impeller_state(imp.out)
        
        # Auto-detect configuration
        self.config = self._detect_configuration(geom)
        
        # Initialize based on configuration
        if self.config == 'vaneless_only':
            self._initialize_vaneless_only(geom, op, inlet_state)
        
        elif self.config == 'vaned_only':
            self._initialize_vaned_only(geom, op, inlet_state)
        
        elif self.config == 'both':
            self._initialize_both(geom, op, inlet_state)
    
    def _detect_configuration(self, geom: Geometry) -> Literal['vaneless_only', 'vaned_only', 'both']:
        """
        Auto-detect diffuser configuration from geometry
        
        Logic:
        - Vaneless only: r5 > r4 AND n_vanes = 0
        - Vaned only: r6 > r5 AND n_vanes > 0 AND (r5 - r4) < threshold
        - Both: r5 > r4 AND r6 > r5 AND n_vanes > 0
        """
        r4, r5, r6 = geom.r4, geom.r5, geom.r6
        n_vanes = geom.n_vanes
        
        # Threshold: vaneless section exists if (r5 - r4) > 2% of r4
        vaneless_threshold = 0.02 * r4
        has_vaneless_section = (r5 - r4) > vaneless_threshold
        has_vaned_section = (r6 > r5) and (n_vanes > 0)
        
        if has_vaneless_section and not has_vaned_section:
            return 'vaneless_only'
        
        elif not has_vaneless_section and has_vaned_section:
            return 'vaned_only'
        
        elif has_vaneless_section and has_vaned_section:
            return 'both'
        
        else:
            # No diffuser at all (r5 = r4, r6 = r5, n_vanes = 0)
            return 'vaneless_only'  # Default fallback
    
    def _initialize_vaneless_only(self, geom: Geometry, op: OperatingCondition, inlet_state: DiffuserState):
        """Initialize vaneless diffuser only"""
        self.vaneless = VanelessDiffuser(geom, op, inlet_state)
        
        # Delegate to vaneless
        self.inlet = self.vaneless.inlet
        self.outlet = self.vaneless.outlet
        self.losses = self.vaneless.losses
        self.eff_ts = self.vaneless.eff_ts
        self.pr_ts = self.vaneless.pr_ts
        self.choke_flag = self.vaneless.choke_flag
    
    def _initialize_vaned_only(self, geom: Geometry, op: OperatingCondition, inlet_state: DiffuserState):
        """Initialize vaned diffuser only (impeller outlet → vaned diffuser)"""
        self.vaned = VanedDiffuser(geom, op, inlet_state, self.loss_model)
        
        # Delegate to vaned
        self.inlet = self.vaned.inlet
        self.outlet = self.vaned.outlet
        self.losses = self.vaned.losses
        self.eff_ts = self.vaned.eff_ts
        self.pr_ts = self.vaned.pr_ts
        self.choke_flag = self.vaned.choke_flag
    
    def _initialize_both(self, geom: Geometry, op: OperatingCondition, inlet_state: DiffuserState):
        """Initialize both vaneless and vaned in series"""
        
        # Step 1: Vaneless diffuser (impeller outlet → station 5)
        self.vaneless = VanelessDiffuser(geom, op, inlet_state)
        
        if self.vaneless.choke_flag:
            self.choke_flag = True
            self.inlet = inlet_state
            self.outlet = self.vaneless.outlet
            self.losses = self.vaneless.losses
            return
        
        # Step 2: Vaned diffuser (vaneless outlet → station 6)
        self.vaned = VanedDiffuser(geom, op, self.vaneless.outlet, self.loss_model)
        
        if self.vaned.choke_flag:
            self.choke_flag = True
        
        # Combine results
        self.inlet = inlet_state
        self.outlet = self.vaned.outlet
        
        # Combine losses
        combined_losses = {}
        for key, value in self.vaneless.losses.breakdown.items():
            combined_losses[f'vaneless_{key}'] = value
        for key, value in self.vaned.losses.breakdown.items():
            combined_losses[f'vaned_{key}'] = value
        
        self.losses = DiffuserLosses(
            breakdown=combined_losses,
            total=self.vaneless.losses.total + self.vaned.losses.total
        )
        
        # Overall efficiency (inlet → final outlet)
        self._calculate_combined_performance(op)
    
    def _calculate_combined_performance(self, op: OperatingCondition):
        """Calculate overall diffuser performance (both vaneless + vaned)"""
        try:
            outlet_is = op.fluid.thermo_prop('PS', self.outlet.total.P, self.inlet.total.S)
        except ThermoException:
            return
        
        dh_isen = outlet_is.H - self.inlet.total.H
        dh_actual = self.outlet.total.H - self.inlet.total.H
        
        if abs(dh_actual) > 1e-6:
            self.eff_ts = dh_isen / dh_actual
        else:
            self.eff_ts = math.copysign(math.inf, dh_isen)
        
        self.pr_ts = self.outlet.static.P / self.inlet.total.P
    
    def __repr__(self):
        """Custom representation showing configuration"""
        return (f"Diffuser(config='{self.config}', "
                f"eff_ts={self.eff_ts:.4f}, "
                f"pr_ts={self.pr_ts:.4f}, "
                f"choke={self.choke_flag})")