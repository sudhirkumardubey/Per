# File: centrifugal_compressor/components/impeller.py

"""
Centrifugal impeller with flexible loss model integration

Combines RadComp physics engine with TurboFlow loss framework:
- RadComp: Proven iterative solver, velocity triangles, rothalpy conservation
- TurboFlow: Unified loss calculator supporting 6 models

Architecture:
    1. Physics Engine (RadComp-based): SAME for all models
       - Velocity triangles at inlet, throat, outlet
       - Rothalpy conservation in rotating frame
       - Coupled discharge triangle solver
    
    2. Loss Calculation (Model-agnostic via LossCalculator)
       - All 6 models handled through unified interface
       - Model selected at initialization: loss_model='schiffmann'
    
    3. Performance Metrics
       - Efficiency, pressure ratio, loss breakdown

Velocity Notation:
    v = absolute velocity (stationary frame) [m/s]
    w = relative velocity (rotating frame) [m/s]
    u = blade speed = ω × r [m/s]

Stations:
    Station 2: Impeller inlet (from inducer)
    Station 3: Throat (minimum flow area, used for choke check)
    Station 4: Impeller outlet (discharge)

Loss Models Supported:
    - 'schiffmann' (RadComp default)
    - 'oh' (Oh 2007)
    - 'zhang_set1', 'zhang_set2', 'zhang_set3' (Zhang variations)
    - 'meroni' (Meroni Table 4)

References:
    - RadComp: Original physics implementation
    - TurboFlow: Loss model framework
"""

import math
from dataclasses import InitVar, dataclass, field
from typing import Dict, Optional
from scipy import optimize

# Unified imports from core package
from ..core import (
    Geometry,
    OperatingCondition,
    FluidState,
    ThermoException,
    static_from_total,
    total_from_static,
    cosd, sind, tand,
)

# Import inducer for state inheritance
from .inducer import Inducer, InducerState

# Import loss calculator
from ..losses.calculator import LossCalculator


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImpellerState(InducerState):
    """Thermodynamic state at impeller station with spanwise velocity distribution"""
    
    # Rotating frame
    relative: FluidState = field(default_factory=FluidState)
    
    # Velocities (RMS location)
    w: float = math.nan      # Relative velocity at RMS
    w_m: float = math.nan    # Relative meridional
    w_t: float = math.nan    # Relative tangential
    v_m: float = math.nan    # Absolute meridional
    v_t: float = math.nan    # Absolute tangential
    u: float = math.nan      # Blade speed at RMS
    
    # Spanwise velocities
    ws: float = math.nan     # Relative velocity at shroud (tip)
    wh: float = math.nan     # Relative velocity at hub
    
    # Mach numbers
    m_abs_m: float = math.nan   # Meridional Mach
    m_rel: float = math.nan     # Relative Mach at RMS
    m_rels: float = math.nan    # Relative Mach at shroud
    
    # Flow angles
    beta: float = math.nan   # Relative flow angle at RMS



@dataclass
class ImpellerLosses:
    """
    Impeller loss breakdown - MODEL AGNOSTIC
    
    Flexible storage for any loss model via dictionary.
    Common losses accessible via properties.
    
    All losses in J/kg (specific enthalpy rise due to irreversibilities).
    
    Attributes:
        total: Total impeller losses (J/kg)
        internal: Internal losses affecting pressure rise (J/kg)
        external: External/parasitic losses (J/kg)
        breakdown: Dictionary of individual losses by name
    """
    total: float = math.nan     # Total losses (J/kg)
    internal: float = math.nan  # Internal losses (J/kg)
    external: float = math.nan  # External/parasitic losses (J/kg)
    
    # Flexible storage for any model's losses
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Convenience properties for common losses
    @property
    def incidence(self) -> float:
        return self.breakdown.get('incidence', 0.0)
    
    @property
    def skin_friction(self) -> float:
        return self.breakdown.get('skin_friction', 0.0)
    
    @property
    def blade_loading(self) -> float:
        return self.breakdown.get('blade_loading', 0.0)
    
    @property
    def clearance(self) -> float:
        return self.breakdown.get('clearance', 0.0)
    
    @property
    def disc_friction(self) -> float:
        return self.breakdown.get('disc_friction', 0.0)
    
    @property
    def recirculation(self) -> float:
        return self.breakdown.get('recirculation', 0.0)
    
    @property
    def mixing(self) -> float:
        return self.breakdown.get('mixing', 0.0)
    
    @property
    def leakage(self) -> float:
        return self.breakdown.get('leakage', 0.0)
    
    def __repr__(self):
        mechanisms = ', '.join(self.breakdown.keys()) if self.breakdown else 'none'
        return f"ImpellerLosses(total={self.total:.1f} J/kg, [{mechanisms}])"


@dataclass
class Impeller:
    """
    Centrifugal impeller with flexible loss model selection
    
    Args:
        geom: Geometry definition
        op: Operating conditions
        ind: Inducer component (provides inlet state)
        loss_model: Loss model ('schiffmann', 'oh', 'zhang_set1', etc.)
    
    Attributes:
        inlet: Impeller inlet state (Station 2)
        throat: Throat state (Station 3)
        outlet: Impeller outlet state (Station 4)
        losses: Loss breakdown
        dh0s: Isentropic enthalpy rise (J/kg)
        eff: Total-to-total efficiency
        choke_flag: True if flow is choked
        wet: True if two-phase flow detected
    """
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    ind: InitVar[Inducer]
    loss_model: str = 'schiffmann'
    
    # Thermodynamic states
    inlet: ImpellerState = field(init=False)
    throat: ImpellerState = field(default_factory=ImpellerState)
    outlet: ImpellerState = field(default_factory=ImpellerState)
    
    # Losses
    losses: ImpellerLosses = field(default_factory=ImpellerLosses)
    
    # Performance
    dh0s: float = math.nan
    eff: float = math.nan
    eff_tt: float = math.nan  
    eff_ts: float = math.nan  
    pr_tt: float = math.nan   
    pr_ts: float = math.nan   
    de_haller: float = math.nan  
    
    # Flags
    choke_flag: bool = False
    wet: bool = False
    
    # Internal (hidden from repr)
    _loss_calculator: Optional[LossCalculator] = field(
        default=None, init=False, repr=False
    )
    _geom: Optional[Geometry] = field(default=None, init=False, repr=False)
    _op: Optional[OperatingCondition] = field(default=None, init=False, repr=False)
    
    def __post_init__(
        self, geom: Geometry, op: OperatingCondition, ind: Inducer
    ) -> None:
        """Initialize and calculate impeller"""
        self._geom = geom
        self._op = op
        
        # Initialize loss calculator
        self._loss_calculator = LossCalculator(self.loss_model)
        
        # Copy inducer outlet as impeller inlet
        self.inlet = ImpellerState.from_state(ind.outlet)
        
        # Run calculation
        if self.outlet.is_not_set:
            try:
                self.calculate(geom, op)
            except ThermoException:
                self.choke_flag = True

        # ========================================================================
    # MAIN CALCULATION METHOD
    # ========================================================================
    
    def calculate(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Main impeller calculation
        
        Orchestrates the complete calculation sequence:
        1. Inlet triangle (Station 2)
        2. Throat check (Station 3)
        3. Discharge triangle solver (Station 4)
        4. Loss calculation (via LossCalculator)
        5. Performance metrics
        
        Args:
            geom: Geometry definition
            op: Operating conditions
        """
        # Step 1: Calculate inlet triangle
        self._calculate_inlet_triangle(geom, op)
        
        # Step 2: Calculate throat and check for choke
        self._calculate_throat(geom, op)
        
        if self.throat.m_rels >= 0.99:
            self.choke_flag = True
            return
        
        # Step 3: Solve discharge triangle
        self._solve_discharge_triangle(geom, op)
        
        # Step 4: Calculate performance metrics
        self._calculate_performance(geom, op)
    
    # ========================================================================
    # VELOCITY TRIANGLE CALCULATIONS
    # ========================================================================
    
    def _calculate_inlet_triangle(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Calculate impeller inlet velocity triangle (Station 2)
        
        Computes velocities at three radial locations:
        - Shroud (tip): r2s - highest blade speed, highest relative velocity
        - RMS (root-mean-square): r2rms - average conditions
        - Hub: r2h - lowest blade speed
        
        Takes inducer outlet state and computes:
        - Absolute flow angle: α2 (from geometry)
        - Blade speeds: u2 at shroud, RMS, hub
        - Velocity components: v_m, v_t decomposed using α2
        - Relative velocities: w2 at shroud, RMS, hub
        - Flow angles: β2 (relative)
        - Mach numbers
        """
        inlet = self.inlet
        
        # ════════════════════════════════════════════════════════════════
        # ABSOLUTE FLOW ANGLE (from geometry specification)
        # ════════════════════════════════════════════════════════════════
        alpha2 = geom.alpha2  # Absolute flow angle at inlet (degrees)
        inlet.alpha = alpha2
        
        # ════════════════════════════════════════════════════════════════
        # ABSOLUTE VELOCITY COMPONENTS (decomposed using alpha2)
        # ════════════════════════════════════════════════════════════════
        # α2 = arctan(v_t / v_m)
        v2_t = inlet.v * sind(alpha2)  # Tangential component (swirl)
        v2_m = inlet.v * cosd(alpha2)  # Meridional component (axial)
        
        # ════════════════════════════════════════════════════════════════
        # SHROUD (TIP) - Station 2s
        # ════════════════════════════════════════════════════════════════
        u2s = geom.r2s * op.omega  # Blade speed at shroud
        w2t_s = u2s - v2_t         # Relative tangential at shroud
        beta2_s = -math.degrees(math.atan(w2t_s / v2_m))  # Flow angle at shroud
        w2_s = v2_m / math.cos(math.radians(beta2_s))     # Relative velocity at shroud
        
        # ════════════════════════════════════════════════════════════════
        # RMS (ROOT-MEAN-SQUARE) - Station 2 (main calculation)
        # ════════════════════════════════════════════════════════════════
        u2 = geom.r2rms * op.omega  # Blade speed at RMS radius
        w2t = u2 - v2_t              # Relative tangential at RMS
        beta2 = -math.degrees(math.atan(w2t / v2_m))      # Flow angle at RMS
        w2 = v2_m / math.cos(math.radians(beta2))         # Relative velocity at RMS
        
        # ════════════════════════════════════════════════════════════════
        # HUB - Station 2h
        # ════════════════════════════════════════════════════════════════
        u2h = geom.r2h * op.omega  # Blade speed at hub
        w2t_h = u2h - v2_t         # Relative tangential at hub
        beta2_h = -math.degrees(math.atan(w2t_h / v2_m))  # Flow angle at hub
        w2_h = v2_m / math.cos(math.radians(beta2_h))     # Relative velocity at hub
        
        # ════════════════════════════════════════════════════════════════
        # STORE RESULTS IN STATE (RMS values + shroud/hub)
        # ════════════════════════════════════════════════════════════════
        # Main (RMS) values
        inlet.u = u2
        inlet.v_m = v2_m
        inlet.v_t = v2_t
        inlet.w = w2
        inlet.w_m = v2_m  # Meridional component same in relative frame
        inlet.w_t = w2t   # Tangential component in relative frame
        inlet.beta = beta2
        
        # Shroud and hub values (for loss calculations and choke check)
        inlet.ws = w2_s   # Relative velocity at shroud (critical for choke)
        inlet.wh = w2_h   # Relative velocity at hub
        
        # ════════════════════════════════════════════════════════════════
        # RELATIVE FRAME THERMODYNAMICS (using RMS velocity)
        # ════════════════════════════════════════════════════════════════
        try:
            inlet.relative = total_from_static(inlet.static, w2)
        except Exception as e:
            self.wet = True
            return

        # ════════════════════════════════════════════════════════════════
        # MACH NUMBERS
        # ════════════════════════════════════════════════════════════════
        inlet.m_rel = w2 / inlet.static.A       # Relative Mach (RMS)
        inlet.m_rels = w2_s / inlet.static.A    # Relative Mach at shroud (for choke)
        inlet.m_abs = inlet.v / inlet.static.A    # Absolute Mach

        
    # ========================================================================
    # SECTION 2: THROAT CALCULATION (STATION 3)
    # ========================================================================
    
    def _calculate_throat(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Calculate throat conditions (Station 3) - choke detection point
        
        ╔════════════════════════════════════════════════════════════════╗
        ║  FLEXIBLE LOSS MODEL APPROACH:                                 ║
        ║  Uses LossCalculator.compute_incidence_loss() with inlet data ║
        ║  Model-specific or Stanitz-Galvas fallback                     ║
        ║  Ensures consistency with selected loss model                  ║
        ╚════════════════════════════════════════════════════════════════╝
        
        The throat is the minimum flow area where:
        1. Flow may choke (M_rel ≥ 0.99)
        2. Incidence losses occur
        3. Entropy rises in relative frame
        
        Purpose:
        - Early choke detection before expensive discharge calculation
        - Model-consistent incidence loss calculation
        - Prepare state for discharge solver
        
        Args:
            geom: Geometry definition
            op: Operating conditions
        """
        inlet = self.inlet
        throat = self.throat
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: Calculate incidence loss (Stanitz-Galvas)
        # ════════════════════════════════════════════════════════════════
        # Optimal blade angle accounting for flow area change
        beta2_opt = math.degrees(
            math.atan(geom.A_x / geom.A_y * tand(geom.beta2))
        )
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: Calculate incidence loss using FLEXIBLE LOSS MODEL
        # ════════════════════════════════════════════════════════════════
        # Prepare inputs with ONLY available inlet data
        incidence_inputs = {
            # Geometry
            'beta_blade_inlet': geom.beta2,
            'beta_opt': beta2_opt,
            'n_blades': geom.n_blades,
            'area_ratio': geom.A_x / geom.A_y,
            
            # Inlet velocities
            'beta_flow_inlet': inlet.beta,
            'v_inlet': inlet.v,
            'v_m_inlet': inlet.v_m,
            'v_t_inlet': inlet.v_t,
            
            # Inlet thermodynamics
            'rho_inlet': inlet.static.D,
            'mu_inlet': inlet.static.V,
            'P_inlet_static': inlet.static.P,
            'T_inlet_static': inlet.static.T,
            'P_inlet_total': inlet.total.P,
            'T_inlet_total': inlet.total.T,
        }

        # CALL FLEXIBLE LOSS MODEL
        try:
            dh_incidence = self._loss_calculator.compute_incidence_loss(**incidence_inputs)
        except Exception as e:
            # If loss calculator fails, calculation cannot continue
            print(f"Error: Incidence loss calculation failed: {e}")
            self.choke_flag = True
            return
        
        # Store model-consistent incidence loss
        # self.losses.incidence = dh_incidence
        self.losses.breakdown['incidence'] = dh_incidence
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: Update relative frame state with incidence loss
        # ════════════════════════════════════════════════════════════════
        # Entropy rises due to incidence (irreversible process)
        # Enthalpy in relative frame: H_rel,3 = H_rel,2 - dh_inc
        try:
            # Calculate entropy rise
            rel3_with_loss = op.fluid.thermo_prop(
                'HS',
                inlet.relative.H - dh_incidence,
                inlet.relative.S
            )
            
        except ThermoException as e:
            print(f"Throat thermodynamics failed: {e}")
            self.choke_flag = True
            return
        
        # Store relative frame state (at constant relative enthalpy)
        throat.relative = op.fluid.thermo_prop('PH',rel3_with_loss.P, inlet.relative.H)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: Solve for throat velocity from continuity
        # ════════════════════════════════════════════════════════════════
        # 1D approximation: ṁ = ρ(w3) * A_throat * w3
        
        def throat_continuity(w3: float) -> float:
            """
            Residual function for throat velocity
            
            Args:
                w3: Throat relative velocity (m/s)
            
            Returns:
                Mass flow error (kg/s)
            """
            try:
                # Static state from relative frame total
                stat3 = static_from_total(inlet.relative, w3)
                
                # Mass flow at throat
                mass_flow_calc = geom.A_y * w3 * stat3.D
                
                # Residual
                return (mass_flow_calc - op.mass_flow) / op.mass_flow
                
            except ThermoException:
                return 1e4  # Penalize invalid solutions
        
        # Initial guess: 65% of acoustic velocity
        w3_guess = 0.65 * inlet.relative.A
        
        # Solve for throat velocity
        sol = optimize.root(throat_continuity, x0=w3_guess)
        
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return
        
        w3_throat = sol.x[0]
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: Calculate throat thermodynamic states
        # ════════════════════════════════════════════════════════════════
        # Static state in relative frame
        throat.static = static_from_total(inlet.relative, w3_throat)
        
        # Absolute velocity at throat (accounting for area change)
        # Continuity in absolute frame: ρ * A * v_m = constant
        v3_m = inlet.v_m * (geom.A_x / geom.A_y)
        
        # Assume flow angle constant through throat region
        if abs(cosd(geom.alpha2)) > 1e-6:
            v3 = v3_m / cosd(geom.alpha2)
        else:
            v3 = v3_m
        
        # Absolute frame total state
        throat.total = total_from_static(throat.static, v3)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 6: Store throat velocities
        # ════════════════════════════════════════════════════════════════
        throat.w = w3_throat        # Relative velocity magnitude
        throat.v = v3               # Absolute velocity magnitude
        throat.v_m = v3_m           # Meridional component
        
        # ════════════════════════════════════════════════════════════════
        # STEP 7: Calculate Mach numbers
        # ════════════════════════════════════════════════════════════════
        throat.m_rel = w3_throat / throat.static.A    # Relative Mach at throat
        throat.m_abs = v3 / throat.static.A           # Absolute Mach


    # ========================================================================
    # SECTION 3: DISCHARGE TRIANGLE SOLVER (STATION 4) - FULLY ADAPTIVE
    # ========================================================================
    
    def _solve_discharge_triangle(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Solve discharge triangle (Station 4) - RADCOMP PATTERN + FLEXIBLE LOSSES
        
        ╔════════════════════════════════════════════════════════════════╗
        ║  FULLY ADAPTIVE APPROACH:                                      ║
        ║  1. Calculate states (velocities, thermodynamics)              ║
        ║  2. Call LossCalculator → returns available losses             ║
        ║  3. Dynamically separate internal vs external losses           ║
        ║  4. Iterate until converged                                    ║
        ╚════════════════════════════════════════════════════════════════╝
        
        Key Innovation:
        - Doesn't assume which losses exist
        - Adapts to whatever the selected model returns
        - Uses LossCalculator metadata to classify losses
        """
        inlet = self.inlet
        outlet = self.outlet
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: Isentropic reference state
        # ════════════════════════════════════════════════════════════════
        h4_rel_isen = 0.5 * (op.omega * geom.r4)**2 - 0.5 * inlet.u**2 + inlet.relative.H
        
        try:
            tp4_rel_isen = op.fluid.thermo_prop('HS', h4_rel_isen, inlet.relative.S)
        except ThermoException as e:
            print(f"Isentropic state calculation failed: {e}")
            self.choke_flag = True
            return
        
        # phase check
        if tp4_rel_isen.is_two_phase():
            print(f"Two-phase flow detected at outlet: phase='{tp4_rel_isen.phase}'")
            print(f"Wet compression: condensation droplets invalidate analysis")
            self.wet = True
            return
            
        A4_total = geom.A4_eff
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: ADAPTIVE COUPLED SOLVER
        # ════════════════════════════════════════════════════════════════
        
        def resolve_discharge_triangle(x):
            """
            Residual function with FULLY ADAPTIVE loss handling
            
            Args:
                x: [beta4_flow, w4, dh_losses_external, p4_rel]
            """
            beta4_flow, w4, dh_losses_ext, p4_rel = x
            
            # Safety bounds
            dh_loss_ext = max(0.0, dh_losses_ext)
            
            p4_r = p4_rel 
            if p4_rel <= 0:
                p4_r = tp4_rel_isen.P
            err = []
            
            try:
                # ════════════════════════════════════════════════════
                # PART A: CALCULATE STATES
                # ════════════════════════════════════════════════════
                tp4_rel = op.fluid.thermo_prop('PH', p4_r, h4_rel_isen + dh_loss_ext)
                A4_rel = A4_total * cosd(beta4_flow)
                tp4_stat = static_from_total(tp4_rel, w4)
                
                # Continuity
                mass_flow_calc = A4_rel * w4 * tp4_stat.D
                err.append((mass_flow_calc - op.mass_flow) / op.mass_flow)
                
                # Velocity triangle
                v4_m = op.mass_flow / (A4_total * tp4_stat.D)
                v4_t = v4_m * tand(geom.beta4) + geom.slip * op.omega * geom.r4
                v4 = math.sqrt(v4_t**2 + v4_m**2)
                alpha4 = math.degrees(math.atan2(v4_t, v4_m))
                
                w4_t = op.omega * geom.r4 - v4_t
                w4_calc = math.sqrt(w4_t**2 + v4_m**2)
                beta4_calc = -math.degrees(math.asin(w4_t / w4_calc))
                
                err.append((beta4_calc - beta4_flow) / 60.0)
                
                tp4_tot = total_from_static(tp4_stat, v4)
                enthalpy_rise = tp4_tot.H - inlet.total.H
                
                # ════════════════════════════════════════════════════
                # PART B: ADAPTIVE LOSS CALCULATION
                # ════════════════════════════════════════════════════
                
                # Prepare comprehensive inputs
                loss_inputs = self._prepare_loss_inputs(
                    geom, op, inlet, tp4_stat, tp4_tot,
                    v4, v4_m, v4_t, w4, beta4_flow, alpha4,
                    enthalpy_rise
                )

                # CALL FLEXIBLE LOSS CALCULATOR
                loss_dict = self._loss_calculator.compute_impeller_losses(**loss_inputs)
                
                # ADAPTIVE CLASSIFICATION
                # Query LossCalculator for which losses are external
                external_losses, internal_losses = self._classify_losses(loss_dict)
                
                dh_losses_ext_calc = sum(external_losses.values())
                dh_losses_int_calc = sum(internal_losses.values())
                
                # ════════════════════════════════════════════════════
                # PART C: RESIDUALS
                # ════════════════════════════════════════════════════
                
                # External energy balance (rothalpy)
                err.append((dh_losses_ext_calc - dh_loss_ext) / inlet.relative.H)
                
                # Internal energy balance (entropy/pressure)
                try:
                    tp4_check = op.fluid.thermo_prop(
                        'HS',
                        h4_rel_isen - dh_losses_int_calc,
                        inlet.relative.S
                    )
                    err.append((tp4_check.P - tp4_rel.P) / inlet.relative.P)
                except:
                    err.append(0.01)  # Small residual if calc fails
                
            except (ThermoException, ValueError, ZeroDivisionError):
                err = [1e4, 1e4, 1e4, 1e4]
            
            return err
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: INITIAL GUESS
        # ════════════════════════════════════════════════════════════════
        beta4_guess = geom.beta4 - 10.0
        w4_guess = op.mass_flow / (geom.A4_eff * tp4_rel_isen.D * cosd(beta4_guess))
        
        # External loss guess (disc friction estimate based on Daily and Nece)
        Rey = 2.0 * op.omega * geom.r4**2 * tp4_rel_isen.D / tp4_rel_isen.V
        if Rey < 3e5:
            Kf = 0.102 * (geom.back_cl / geom.r4) ** 0.1 * Rey ** (-0.2)
        else:
            Kf = 3.7 * (geom.back_cl / geom.r4) ** 0.1 * Rey ** (-0.5)
        dh_ext_guess = 0.25 * tp4_rel_isen.D * (op.omega * geom.r4)**3 * Kf / op.mass_flow
        
        p4_rel_guess = tp4_rel_isen.P
        
        x0 = [beta4_guess, w4_guess, dh_ext_guess, p4_rel_guess]
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: SOLVE COUPLED SYSTEM
        # ════════════════════════════════════════════════════════════════
        sol = optimize.root(
            resolve_discharge_triangle,
            x0,
            method='hybr',
            tol=1e-4
        )
        
        if not sol.success or (abs(sol.fun) > 0.01).any():
            self.choke_flag = True
            return
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: EXTRACT CONVERGED SOLUTION
        # ════════════════════════════════════════════════════════════════
        beta4_flow, w4, dh_losses_ext, p4_rel = sol.x
        
        # Final states
        tp4_rel = op.fluid.thermo_prop('PH', p4_rel, h4_rel_isen + max(0, dh_losses_ext))
        tp4_stat = static_from_total(tp4_rel, w4)
        
        v4_m = op.mass_flow / (geom.A4_eff * tp4_stat.D)
        v4_t = v4_m * tand(geom.beta4) + geom.slip * op.omega * geom.r4
        v4 = math.sqrt(v4_t**2 + v4_m**2)
        alpha4 = math.degrees(math.atan2(v4_t, v4_m))
        
        tp4_tot = total_from_static(tp4_stat, v4)
        tp4_isen = op.fluid.thermo_prop('PS', tp4_tot.P, inlet.total.S)

        enthalpy_rise = tp4_tot.H - inlet.total.H

        # ════════════════════════════════════════════════════════════════
        # STEP 6: FINAL LOSS CALCULATION (CONVERGED VELOCITIES)
        # ════════════════════════════════════════════════════════════════
        
        # Prepare inputs with final converged velocities
        final_loss_inputs = self._prepare_loss_inputs(
            geom, op, inlet, tp4_stat, tp4_tot,
            v4, v4_m, v4_t, w4, beta4_flow, alpha4,
            enthalpy_rise
        )
        
        try:
            # Calculate FINAL losses with converged solution
            final_loss_dict = self._loss_calculator.compute_impeller_losses(**final_loss_inputs)
            
            # Store complete loss breakdown
            self._store_loss_breakdown(final_loss_dict)
            
            # Calculate total loss
            total_loss = sum(final_loss_dict.values())
            
        except Exception as e:
            print(f"Warning: Final loss calculation failed: {e}")
        
        # ════════════════════════════════════════════════════════════════
        # STEP 7: STORE OUTLET STATE
        # ════════════════════════════════════════════════════════════════
        outlet.total = tp4_tot
        outlet.static = tp4_stat
        outlet.isentropic = tp4_isen
        outlet.relative = tp4_rel
        
        outlet.v = v4
        outlet.v_m = v4_m
        outlet.v_t = v4_t
        outlet.u = op.omega * geom.r4
        outlet.w = w4
        outlet.w_m = v4_m
        outlet.w_t = op.omega * geom.r4 - v4_t
        outlet.beta = beta4_flow
        outlet.alpha = alpha4
        outlet.A_eff = geom.A4_eff
        
        outlet.m_abs = v4 / tp4_stat.A
        outlet.m_abs_m = v4_m / tp4_stat.A
        outlet.m_rel = w4 / tp4_stat.A
        
        # Choke checks
        if outlet.m_rel >= 0.99 or outlet.m_abs_m >= 0.99:
            self.choke_flag = True
            return
        
        if outlet.total.P <= inlet.total.P:
            self.choke_flag = True
            return
    
    
    # ========================================================================
    # SECTION 4: PERFORMANCE CALCULATION
    # ========================================================================
    
    def _calculate_performance(self, geom: Geometry, op: OperatingCondition) -> None:
        """
        Calculate impeller performance metrics
        
        Computed after discharge triangle and losses are known:
        
        1. Isentropic enthalpy rise (ideal work)
        2. Euler work (theoretical work)
        3. Actual enthalpy rise (real work)
        4. Total-to-total efficiency
        5. Total-to-static efficiency
        6. Pressure ratio
        7. Loss coefficients
        
        All metrics computed from converged states and final losses.
        """
        inlet = self.inlet
        outlet = self.outlet
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: ISENTROPIC ENTHALPY RISE (ideal compression work)
        # ════════════════════════════════════════════════════════════════
        # Enthalpy rise if compression were reversible (no losses)
        # From inlet total to outlet total at constant entropy
        self.dh0s = outlet.isentropic.H - inlet.total.H
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: ACTUAL ENTHALPY RISE (real compression work)
        # ════════════════════════════════════════════════════════════════
        dh0_actual = outlet.total.H - inlet.total.H
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: EULER WORK (theoretical work from velocity triangles)
        # ════════════════════════════════════════════════════════════════
        # From Euler turbomachine equation: dh_euler = u₄v₄ₜ - u₂v₂ₜ
        dh_euler = outlet.u * outlet.v_t - inlet.u * inlet.v_t
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: TOTAL-TO-TOTAL EFFICIENCY
        # ════════════════════════════════════════════════════════════════
        # η_tt = (isentropic work) / (actual work)
        #      = dh0s / dh0_actual
        if dh0_actual > 0:
            self.eff_tt = self.dh0s / dh0_actual
        else:
            self.eff_tt = 0.0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: TOTAL-TO-STATIC EFFICIENCY
        # ════════════════════════════════════════════════════════════════
        # η_ts = (isentropic work to static pressure) / (actual work)
        # Accounts for kinetic energy loss at outlet
        try:
            out_isen_static = op.fluid.thermo_prop('PS', outlet.static.P, inlet.total.S)
            dhs = out_isen_static.H - inlet.total.H
            self.eff_ts = dhs / dh0_actual if dh0_actual > 0 else 0.0
        except ThermoException:
            self.eff_ts = math.nan
        
        # ════════════════════════════════════════════════════════════════
        # STEP 6: PRESSURE RATIO
        # ════════════════════════════════════════════════════════════════
        self.pr_tt = outlet.total.P / inlet.total.P  # Total-to-total
        self.pr_ts = outlet.static.P / inlet.total.P  # Total-to-static
        
        # ════════════════════════════════════════════════════════════════
        # STEP 7: LOSS COEFFICIENTS
        # ════════════════════════════════════════════════════════════════
        # Separate internal and external losses
        external_keys = self._loss_calculator.get_external_loss_keys()
        
        self.losses.internal = sum(
            v for k, v in self.losses.breakdown.items() 
            if k not in external_keys
        )
        self.losses.external = sum(
            v for k, v in self.losses.breakdown.items() 
            if k in external_keys
        )
        
        # Total loss
        self.losses.total = self.losses.internal + self.losses.external
        
        # Loss coefficient (based on Euler work)
        if dh_euler > 0:
            self.loss_coeff = self.losses.total / dh_euler
        else:
            self.loss_coeff = math.nan
        
        # ════════════════════════════════════════════════════════════════
        # STEP 8: DIAGNOSTIC METRICS (optional)
        # ════════════════════════════════════════════════════════════════
        
        # De Haller number (check for excessive diffusion)
        # De Haller > 0.7 is desirable
        self.de_haller = outlet.w / inlet.w if inlet.w > 0 else math.nan


    # ════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ════════════════════════════════════════════════════════════════════
    def _prepare_loss_inputs(
        self, geom, op, inlet, tp4_stat, tp4_tot,
        v4, v4_m, v4_t, w4, beta4_flow, alpha4, enthalpy_rise
    ) -> dict:
        """Prepare comprehensive inputs for LossCalculator"""
        return {
            # Geometry
            'geom': geom,
            
            # States
            'inlet_state': inlet.static,
            'outlet_state': tp4_stat,
            
            # Velocities dictionary - ✅ FIXED: Use underscore naming
            'velocities': {
                # Inlet
                'w_in': inlet.w,        # ✅ Changed from 'win'
                'v_in': inlet.v,        # ✅ Changed from 'vin'
                'u_in': inlet.u,        # ✅ Changed from 'uin'
                'v_m_inlet': inlet.v_m,
                'v_t_in': inlet.v_t,    # ✅ Changed from 'vtin'
                'w_t_in': inlet.w_t,    # ✅ Changed from 'wtin'
                'v_m_in': inlet.v_m,    # ✅ Changed from 'vmin'
                
                # Outlet
                'w_out': w4,          # ✅ Changed from 'wout'
                'v_out': v4,          # ✅ Changed from 'vout'
                'u_out': op.omega * geom.r4,  # ✅ Changed from 'uout'
                'v_m_out': v4_m,      # ✅ Changed from 'vmout'
                'v_t_out': v4_t,      # ✅ Changed from 'vtout'
                'w_t_out': op.omega * geom.r4 - v4_t,  # ✅ Changed from 'wtout'
                
                # Flow angles
                'beta_flow_inlet': inlet.beta,
                'beta_flow_outlet': beta4_flow,
                'alpha_out': alpha4,  # ✅ Changed from 'alphaout'
                
                # Throat
                'w_th': self.throat.w if hasattr(self.throat, 'w') else inlet.w * 0.9,
            },
            
            # Operating conditions
            'mass_flow': op.mass_flow,
            
            # Performance
            'enthalpy_rise': enthalpy_rise,
            
            # Flow angle
            'beta_flow': beta4_flow,
            
            # Throat state
            'throat_state': self.throat.static if hasattr(self.throat, 'static') else None,
        }


    def _classify_losses(self, loss_dict: dict) -> tuple:
        """
        Dynamically classify losses as internal vs external
        
        Uses LossCalculator metadata to determine which losses
        affect rothalpy (external) vs entropy (internal)
        
        Returns:
            (external_losses, internal_losses) - both dicts in J/kg
        """
        # Ask LossCalculator which losses are external
        external_keys = self._loss_calculator.get_external_loss_keys()
        
        external_losses = {}
        internal_losses = {}
        
        for key, value in loss_dict.items():
            if key in external_keys:
                external_losses[key] = value
            else:
                internal_losses[key] = value
        
        return external_losses, internal_losses

    def _store_loss_breakdown(self, loss_dict: dict) -> None:
        """Store complete loss breakdown in self.losses"""
        self.losses.breakdown = loss_dict.copy()
        self.losses.total = sum(loss_dict.values())
