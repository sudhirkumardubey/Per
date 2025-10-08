# File: centrifugal_compressor/components/stage.py
"""
Stage - Unified centrifugal compressor stage integration

Integrates: Inducer → Impeller → Diffuser
Auto-detects diffuser type (vaneless, vaned, or both)
and computes overall performance metrics.

Author: Centrifugal Compressor Team
Date: October 2025
Reference: Japikse & Baines (1994), RadComp Framework
"""

import math
from dataclasses import InitVar, dataclass, field
from typing import Optional

# ============================================================================ #
# Unified imports (match existing structure)
# ============================================================================ #
from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.core.thermodynamics import FluidState, ThermoException
from centrifugal_compressor.components.inducer import Inducer
from centrifugal_compressor.components.impeller import Impeller
from centrifugal_compressor.components.diffuser import Diffuser


# ============================================================================ #
# STAGE CLASS
# ============================================================================ #
@dataclass
class Stage:
    """
    Represents a single centrifugal compressor stage.

    Automatically constructs and computes:
        Inducer → Impeller → Diffuser (vaneless/vaned/both)
    from geometry and operating conditions.

    Attributes:
        geom: Geometry object
        op: OperatingCondition object
        loss_model: Loss model identifier ('meroni', 'oh', 'zhang_set1', etc.)

        inducer: Inducer component (if present)
        impeller: Impeller component
        diffuser: Diffuser component (auto-configured)

        inlet: Stage inlet total state
        outlet: Stage outlet total state

        pr_tt: Total-to-total pressure ratio
        pr_ts: Total-to-static pressure ratio
        eff_tt: Total-to-total efficiency
        eff_ts: Total-to-static efficiency
        power: Shaft power [W]
        torque: Shaft torque [N·m]
        psi: Head coefficient (Δh₀ / U²)
        phi: Flow coefficient (Vₘ / U)
        mu: Work input coefficient (≈ ψ)
    """

    # Inputs
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    loss_model: str = "meroni"

    # Components
    inducer: Optional[Inducer] = None
    impeller: Impeller = field(init=False)
    diffuser: Diffuser = field(init=False)

    # Inlet/outlet states
    inlet: FluidState = field(default_factory=FluidState)
    outlet: FluidState = field(default_factory=FluidState)

    # Overall performance
    pr_tt: float = math.nan
    pr_ts: float = math.nan
    eff_tt: float = math.nan
    eff_ts: float = math.nan
    power: float = math.nan
    torque: float = math.nan
    psi: float = math.nan
    phi: float = math.nan
    mu: float = math.nan

    # Internal parameters
    tip_speed: float = math.nan

    # Flags
    converged: bool = False
    choke_flag: bool = False
    invalid_flag: bool = False

    # ====================================================================== #
    def __post_init__(self, geom: Geometry, op: OperatingCondition):
        """Initialize stage and trigger full calculation."""
        self.inlet = op.inlet_state
        self.tip_speed = geom.r4 * op.omega
        self._calculate(geom, op)

    # ====================================================================== #
    def _calculate(self, geom: Geometry, op: OperatingCondition):
        """Compute complete stage flowpath and overall performance."""
        try:
            # --------------------------------------------------------------
            # Step 1: Inducer (optional)
            # --------------------------------------------------------------
            if geom.r1 >= geom.r2s - 1e-6:
                self.inducer = Inducer(geom, op)
                if self.inducer.choke_flag:
                    self.choke_flag = True
                    self.invalid_flag = True
                    return
            else:
                self.inducer = None

            # --------------------------------------------------------------
            # Step 2: Impeller
            # --------------------------------------------------------------
            self.impeller = Impeller(geom, op, self.inducer, self.loss_model)

            if self.impeller.choke_flag or getattr(self.impeller, "wet", False):
                self.invalid_flag = True
                self.choke_flag = True
                return

            # --------------------------------------------------------------
            # Step 3: Diffuser
            # --------------------------------------------------------------
            self.diffuser = Diffuser(geom, op, self.impeller, self.loss_model)

            if self.diffuser.choke_flag:
                self.choke_flag = True
                return

            # --------------------------------------------------------------
            # Step 4: Overall stage performance
            # --------------------------------------------------------------
            self._calculate_performance(geom, op)

            self.converged = not (self.invalid_flag or self.choke_flag)

        except Exception as e:
            print(f"[Stage] Calculation failed: {e}")
            self.invalid_flag = True

    # ====================================================================== #
    def _calculate_performance(self, geom: Geometry, op: OperatingCondition):
        """Compute overall performance metrics from inlet to diffuser outlet."""
        inlet_state = op.inlet_state
        outlet_total = self.diffuser.outlet.total
        outlet_static = self.diffuser.outlet.static

        # Pressure ratios
        self.pr_tt = outlet_total.P / inlet_state.P
        self.pr_ts = outlet_static.P / inlet_state.P

        # Efficiency
        try:
            outlet_is = op.fluid.thermo_prop("PS", outlet_total.P, inlet_state.S)
            dh_is = outlet_is.H - inlet_state.H
            dh_actual = outlet_total.H - inlet_state.H
            self.eff_tt = dh_is / dh_actual if abs(dh_actual) > 1e-6 else math.nan
        except ThermoException:
            self.invalid_flag = True
            return

        if dh_actual < 0 or self.pr_tt < 1:
            self.invalid_flag = True
            return False
        
        # Derived parameters
        self.power = op.mass_flow * (outlet_total.H - inlet_state.H)
        self.torque = self.power / op.omega if op.omega > 0 else math.nan

        # Dimensionless coefficients
        self.psi = (outlet_total.H - inlet_state.H) / (self.tip_speed**2)
        self.mu = self.psi
        self.phi = op.mass_flow / (inlet_state.D * self.tip_speed * geom.r4**2)

    # ====================================================================== #
    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"Stage Summary:\n"
            f"  Diffuser Config: {self.diffuser.config}\n"
            f"  PR_tt = {self.pr_tt:.4f}\n"
            f"  PR_ts = {self.pr_ts:.4f}\n"
            f"  Eff_tt = {self.eff_tt*100:.2f}%\n"
            f"  Power = {self.power/1000:.2f} kW\n"
            f"  Tip speed = {self.tip_speed:.2f} m/s\n"
        )

    # ====================================================================== #
    def __repr__(self):
        return (
            f"Stage(PR_tt={self.pr_tt:.4f}, Eff_tt={self.eff_tt:.4f}, "
            f"config='{getattr(self.diffuser, 'config', 'N/A')}', "
            f"choke={self.choke_flag}, invalid={self.invalid_flag})"
        )
