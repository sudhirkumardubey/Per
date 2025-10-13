# File: centrifugal_compressor/losses/components.py
"""
Loss correlations for centrifugal compressor components
"""

import math
import numpy as np
from typing import Dict, Any

# GOOD: Import at module level (once)
from ..core.correlations import moody, japikse_friction
from ..core import cosd, sind, tand  # Helper functions from geometry


# ============================================================================
# SECTION 1: INDUCER LOSSES (DWE)
# ============================================================================

class InducerFrictionLoss:
    """Inducer friction loss using Darcy-Weisbach equation"""
    
    @staticmethod
    def darcy_weisbach(geom, fluid_state, velocity):
        """
        Calculate pressure drop due to friction in inducer
        
        Reference: Darcy-Weisbach equation
        """
        Re = velocity * 2 * geom.r2s * fluid_state.D / fluid_state.V
        relative_roughness = geom.rough_inducer / (2 * geom.r2s)
        Cf = moody(Re, relative_roughness)
        delP = 4 * Cf * geom.l_inducer * fluid_state.D * velocity**2 / (4 * geom.r2s)
        
        return delP

# ============================================================================
# SECTION 2: IMPELLER LOSSES
# ============================================================================

class ImpellerSkinFrictionLoss:
    """
    Impeller skin friction losses due to boundary layer on blade surfaces
    
    Available methods:
    - jansen_method: Common to all TurboFlow models (Oh, Zhang Set 1/2/3, Custom)
    """
    
    @staticmethod
    def jansen_method(geom, inlet_state, outlet_state, velocities):
        """
        Skin friction loss using Jansen correlation with hydraulic diameter
        
        Formula (identical across all TurboFlow models):
            L_b = π/8 * (2*r_out - (r_in_tip + r_in_hub) - b_out + 2*L_ax) * 
                  (2/(cos(β_in) + cos(β_out)))
            D_h = 2*r_out*cos(β_out)/(z/π + 2*r_out*cos(β_out)/b_out) + 
                  0.5*(r_in_tip+r_in_hub)/r_out*cos(β_in)/
                  (z/π + (r_in_tip+r_in_hub)/(r_in_tip-r_in_hub)*cos(β_in))
            Re = u_out * D_h * ρ_in / μ_in
            Cf = 0.0412 * Re^(-0.1925)
            Y_sf = 2 * Cf * L_b/D_h * w_avg²
        
        Args:
            geom: Geometry with attributes:
                - r4: Outlet radius (m)
                - r2s: Inlet shroud radius (m)
                - r2h: Inlet hub radius (m)
                - b4: Outlet width (m)
                - beta4: Outlet blade angle (deg)
                - beta2: Inlet blade angle (deg)
                - n_blades: Number of blades
            inlet_state: FluidState at inlet with:
                - D: Density (kg/m³)
                - V: Dynamic viscosity (Pa·s)
            outlet_state: FluidState at outlet
            velocities: dict with keys:
                - 'u_out': Blade speed at outlet (m/s)
                - 'v_in': Absolute velocity at inlet (m/s)
                - 'v_out': Absolute velocity at outlet (m/s)
                - 'w_in': Relative velocity at inlet (m/s)
                - 'w_out': Relative velocity at outlet (m/s)
        
        Returns:
            float: Skin friction loss coefficient Y_sf (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set1/2/3.py
        """
        # Extract geometry parameters
        r_out = geom.r4
        r_in_tip = geom.r2s
        r_in_hub = geom.r2h
        b_out = geom.b4
        beta_out = geom.beta4
        beta_in = geom.beta2
        z = geom.n_blades
        
        # Calculate axial length
        L_ax = 0.5 * (geom.r4 - geom.r2s) + geom.b4
        
        # Blade length (L_b): Aungier
        L_b = (np.pi/8 * (r_out*2 - (r_in_tip + r_in_hub) - b_out + 2*L_ax) * 
               (2/(cosd(beta_in) + cosd(beta_out))))
        
        # Hydraulic diameter (D_h) - two-term formula
        term1 = (2*r_out*cosd(beta_out) / 
                 (z/np.pi + 2*r_out*cosd(beta_out)/b_out))
        
        term2 = (0.5*(r_in_tip+r_in_hub)/r_out*cosd(beta_in) /
                 (z/np.pi + (r_in_tip + r_in_hub)/(r_in_tip - r_in_hub)*cosd(beta_in)))
        
        D_h = term1 + term2
        
        # Average relative velocity (Jansen)
        w_avg = (velocities['v_in'] + velocities['v_out'] + 
                 3*velocities['w_in'] + 3*velocities['w_out']) / 8
        
        # Reynolds number
        u_out = velocities['u_out']
        Re = u_out * D_h * inlet_state.D / inlet_state.V
        
        # Friction coefficient (Jansen correlation)
        Cf = 0.0412 * Re**(-0.1925)
        
        # Skin friction loss coefficient
        Y_sf = 2 * Cf * L_b / D_h * w_avg**2
        
        return Y_sf


class ImpellerBladeLoadingLoss:
    """
    Impeller blade loading losses due to flow diffusion and separation
    
    Available methods:
    - oh_coppage_method: Oh et al. (1997) - uses diffusion factor
    - zhang_aungier_method: Zhang et al. - uses velocity difference
    """
    
    @staticmethod
    def oh_coppage_method(geom, inlet_state, outlet_state, velocities, enthalpy_rise):
        """
        Blade loading loss - Oh/Coppage method using diffusion factor
        
        Formula:
            Df = 1 - w_out/w_in + 0.75*(h0_out - h0_in)*w_out/
                 ((z/π*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out²)
            Y_bld = 0.05 * Df² * u_out²
        
        Args:
            geom: Geometry (r4, r2s, n_blades)
            inlet_state, outlet_state: FluidState objects  
            velocities: dict with 'w_in', 'w_out', 'u_out' (m/s)
            enthalpy_rise: h0_out - h0_in (J/kg)
        
        Returns:
            float: Blade loading loss coefficient Y_bld (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py
        """
        r_out = geom.r4
        r_in_tip = geom.r2s
        z = geom.n_blades
        
        w_in = velocities['w_in']
        w_out = velocities['w_out']
        u_out = velocities['u_out']
        
        # Diffusion factor (Coppage)
        Df = (1 - w_out/w_in + 
              0.75*enthalpy_rise*w_out/
              ((z/np.pi*(1-r_in_tip/r_out) + 2*r_in_tip/r_out)*w_in*u_out**2))
        
        # Blade loading loss
        Y_bld = 0.05 * Df**2 * u_out**2
        
        return Y_bld
    
    @staticmethod
    def zhang_aungier_method(geom, velocities):
        """
        Blade loading loss - Zhang/Aungier method using velocity difference
        
        Formula:
            L_b = π/8*(2*r_out-(r_in_tip+r_in_hub)-b_out+2*L_ax)*
                  (2/(cos(β_in)+cos(β_out)))
            ΔW = 2π*2*r_out*v_t_out/(z*L_b)
            Y_bld = ΔW²/48
        
        Args:
            geom: Geometry (full)
            velocities: dict with 'v_t_out' (tangential velocity at outlet, m/s)
        
        Returns:
            float: Blade loading loss coefficient Y_bld (dimensionless)
        
        Reference: TurboFlow loss_model_zhang_set1/2/3.py
        """
        r_out = geom.r4
        r_in_tip = geom.r2s
        r_in_hub = geom.r2h
        b_out = geom.b4
        beta_in = geom.beta2
        beta_out = geom.beta4
        z = geom.n_blades
        
        # Axial length
        L_ax = 0.5 * (geom.r4 - geom.r2s) + geom.b4
        
        # Blade length
        L_b = (np.pi/8 * (r_out*2 - (r_in_tip + r_in_hub) - b_out + 2*L_ax) *
               (2/(cosd(beta_in) + cosd(beta_out))))
        
        # Velocity difference on blade
        v_t_out = velocities['v_t_out']

        delta_W = 2*np.pi*2*r_out*v_t_out / (z*L_b)
        
        # Blade loading loss
        Y_bld = delta_W**2 / 48
        
        return Y_bld


class ImpellerIncidenceLoss:
    """
    Impeller incidence losses due to flow angle mismatch at blade leading edge
    
    Available methods:
    - oh_conrad_method: Oh et al. - simple tangential velocity squared
    - zhang_aungier_method: Zhang et al. - angle-based approach
    - radcomp_stanitz_galvas_method: RadComp - optimal angle calculation
    """
    
    @staticmethod
    def oh_conrad_method(velocities):
        """
        Incidence loss - Oh/Conrad method
        
        Formula:
            Y_inc = 0.5 * 0.5 * w_t_in²
        
        Args:
            velocities: dict with 'w_t_in' (tangential component of relative velocity, m/s)
        
        Returns:
            float: Incidence loss coefficient Y_inc (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py
        """
        w_t_in = velocities['w_t_in']
        f_inc = 0.5  # Incidence factor
        Y_inc = f_inc * 0.5 * w_t_in**2
        
        return Y_inc
    
    @staticmethod
    def zhang_aungier_method(geom, velocities):
        """
        Incidence loss - Zhang/Aungier method
        
        Formula:
            Y_inc = 0.4 * (w_in - v_m_in/cos(θ_in))²
        
        Args:
            geom: Geometry with beta2 (leading edge angle, deg)
            velocities: dict with 'w_in' (m/s), 'v_m_in' (meridional velocity, m/s)
        
        Returns:
            float: Incidence loss coefficient Y_inc (dimensionless)
        
        Reference: TurboFlow loss_model_zhang_set1/2/3.py
        """
        w_in = velocities['w_in']
        v_m_in = velocities['v_m_in']
        theta_in = geom.beta2  # Leading edge angle
        
        Y_inc = 0.4 * (w_in - v_m_in/cosd(theta_in))**2
        
        return Y_inc
    
    @staticmethod
    def stanitz_galvas_method(geom, velocities, beta_flow):
        """
        Incidence loss - RadComp Stanitz-Galvas method
        
        Calculates optimal blade angle based on area ratio, then computes
        loss from deviation between actual flow angle and optimal.
        
        Formula:
            β2_opt = arctan(A_x/A_y * tan(β2))
            dh_inc = 0.5 * (w_in * sin(|β_flow| - |β2_opt|))²
        
        Args:
            geom: Geometry with A_x, A_y, beta2
            velocities: dict with 'w_in' (relative velocity at inlet, m/s)
            beta_flow: float, actual flow angle at inlet (degrees)
        
        Returns:
            float: Incidence loss enthalpy (J/kg) - NOT dimensionless!
        
        Reference: RadComp impeller.py
        Note: Returns J/kg, not coefficient
        """
        # Optimal blade angle accounting for area ratio effects
        beta2_opt = math.degrees(
            math.atan(geom.A_x / geom.A_y * tand(geom.beta2))
        )
        
        # Incidence loss from angle mismatch
        w_in = velocities['w_in']
        angle_diff = abs(abs(beta_flow) - abs(beta2_opt))
        
        dh_inc = 0.5 * (w_in * sind(angle_diff))**2
        
        return dh_inc


class ImpellerClearanceLoss:
    """
    Impeller tip clearance losses due to leakage flow over blade tips
    
    Available methods:
    - jansen_method: Common to all TurboFlow models
    """
    
    @staticmethod
    def jansen_method(geom, inlet_state, outlet_state, velocities):
        """
        Tip clearance loss - Jansen method
        
        Formula:
            Y_cl = 0.6 * t_cl/b_out * v_t_out *
                   sqrt(4π/(b_out*z) * (r_in_tip²-r_in_hub²)/
                   (r_out-r_in_tip) * v_t_out * v_in / (1+ρ_out/ρ_in))
        
        Args:
            geom: Geometry with tip_cl, b4, r4, r2s, r2h, n_blades
            inlet_state, outlet_state: FluidState with density D
            velocities: dict with 'v_t_out' (m/s), 'v_in' or 'v_m_in' (m/s)
        
        Returns:
            float: Clearance loss coefficient Y_cl (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set1.py
        """
        t_cl = geom.tip_cl
        b_out = geom.b4
        r_out = geom.r4
        r_in_tip = geom.r2s
        r_in_hub = geom.r2h
        z = geom.n_blades
        
        v_t_out = velocities['v_t_out']
        v_in = velocities.get('v_in', velocities.get('v_m_in', 0))
        
        rho_out = outlet_state.D
        rho_in = inlet_state.D
        
        # Clearance loss coefficient
        term1 = 4*np.pi / (b_out*z)
        term2 = (r_in_tip**2 - r_in_hub**2) / (r_out - r_in_tip)
        term3 = v_t_out * v_in / (1 + rho_out/rho_in)
        
        Y_cl = 0.6 * t_cl/b_out * v_t_out * np.sqrt(term1 * term2 * term3)
        
        return Y_cl


class ImpellerDiscFrictionLoss:
    """
    Impeller disc friction losses (parasitic windage on backface)
    
    Available methods:
    - daily_nece_method: Common to all models
    """
    
    @staticmethod
    def daily_nece_method(geom, inlet_state, outlet_state, velocities, mass_flow):
        """
        Disc friction loss - Daily and Nece correlation
        
        Formula:
            Re = ρ_out * u_out * r_out / μ_out
            f_df = 2.67/Re^0.5         if Re < 3e5
            f_df = 0.0622/Re^0.2       if Re >= 3e5
            Y_df = f_df * ρ_avg * r_out² * u_out³ / (4 * m_dot)
        
        Args:
            geom: Geometry with r4
            inlet_state, outlet_state: FluidState with D (density), V (viscosity)
            velocities: dict with 'u_out' (blade speed, m/s)
            mass_flow: Mass flow rate (kg/s)
        
        Returns:
            float: Disc friction loss coefficient Y_df (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set1.py
        """
        r_out = geom.r4
        u_out = velocities['u_out']
        rho_out = outlet_state.D
        mu_out = outlet_state.V
        
        # Reynolds number based on outer radius
        Re = 2 * rho_out * u_out * r_out / mu_out
        
        # Friction factor (piecewise - turbulent transition at Re=3e5)
        if Re < 3e5:
            f_df = 2.67 / Re**0.5
        else:
            f_df = 0.0622 / Re**0.2
        
        # Average density
        rho_avg = (inlet_state.D + outlet_state.D) / 2
        
        # Disc friction loss
        Y_df = f_df * rho_avg * r_out**2 * u_out**3 / (4 * mass_flow)
        
        return Y_df


class ImpellerRecirculationLoss:
    """
    Impeller recirculation losses (parasitic recirculation at exit)
    
    Available methods:
    - oh_method: Oh et al. - uses sinh function of exit angle
    - zhang_coppage_method: Zhang et al. - uses sqrt(tan) of exit angle
    """
    
    @staticmethod
    def oh_method(geom, inlet_state, outlet_state, velocities, enthalpy_rise):
        """
        Recirculation loss - Oh method with hyperbolic sine
        
        Formula:
            Df = 1 - w_out/w_in + 0.75*(h0_out-h0_in)*w_out/
                 ((z/π*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out²)
            Y_rc = 8e-5 * sinh(3.5*(α_out*π/180)³) * Df² * u_out²
        
        Args:
            geom: Geometry (r4, r2s, n_blades)
            inlet_state, outlet_state: FluidState objects
            velocities: dict with w_in, w_out, u_out, alpha_out (deg)
            enthalpy_rise: h0_out - h0_in (J/kg)
        
        Returns:
            float: Recirculation loss coefficient Y_rc (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py
        """
        r_out = geom.r4
        r_in_tip = geom.r2s
        z = geom.n_blades
        
        w_in = velocities['w_in']
        w_out = velocities['w_out']
        u_out = velocities['u_out']
        alpha_out = velocities['alpha_out']
        
        # Diffusion factor (same as blade loading)
        Df = (1 - w_out/w_in + 
              0.75*enthalpy_rise*w_out/
              ((z/np.pi*(1-r_in_tip/r_out) + 2*r_in_tip/r_out)*w_in*u_out**2))
        
        # Recirculation loss with hyperbolic sine
        Y_rc = 8e-5 * np.sinh(3.5*(alpha_out*np.pi/180)**3) * Df**2 * u_out**2
        
        return Y_rc
    
    @staticmethod
    def coppage_method(geom, velocities):
        """
        Recirculation loss - Zhang/Coppage method with sqrt(tan)
        
        Formula:
            Df = 1 - w_out/w_in + 0.75*(|u_out*v_t_out|-|u_in*v_t_in|)*w_out/
                 ((z/π*(1-r_in_tip/r_out)+2*r_in_tip/r_out)*w_in*u_out²)
            Y_rc = 0.02 * sqrt(tan(α_out)) * Df² * u_out²
        
        Args:
            geom: Geometry (r4, r2s, n_blades)
            velocities: dict with w_in, w_out, u_in, u_out, v_t_in, v_t_out, alpha_out
        
        Returns:
            float: Recirculation loss coefficient Y_rc (dimensionless)
        
        Reference: TurboFlow loss_model_zhang_set1.py
        """
        r_out = geom.r4
        r_in_tip = geom.r2s
        z = geom.n_blades
        
        w_in = velocities['w_in']
        w_out = velocities['w_out']
        u_out = velocities['u_out']
        u_in = velocities['u_in']
        v_t_in = velocities['v_t_in']
        v_t_out = velocities['v_t_out']
        alpha_out = velocities['alpha_out']
        
        # Euler work
        euler_work = abs(u_out*v_t_out) - abs(u_in*v_t_in)
        
        # Modified diffusion factor for Zhang
        Df = (1 - w_out/w_in +
              0.75*euler_work*w_out/
              ((z/np.pi*(1-r_in_tip/r_out) + 2*r_in_tip/r_out)*w_in*u_out**2))
        
        # Recirculation loss with sqrt(tan)
        Y_rc = 0.02 * np.sqrt(tand(alpha_out)) * Df**2 * u_out**2
        
        return Y_rc


class ImpellerLeakageLoss:
    """
    Impeller leakage losses through tip clearances
    
    Available methods:
    - jansen_method: Zhang Set 1 and Set 3 - Jansen correlation
    - aungier_method: Oh and Zhang Set 2 - Aungier correlation
    """
    
    @staticmethod
    def jansen_method(geom, inlet_state, outlet_state, velocities):
        """
        Leakage loss - Zhang Set 1 and Set 3 - Jansen method
        
        Formula (from loss_model_zhang_set1.py and loss_model_zhang_set3.py):
            α = (r_in_tip - r_hub_in) / (r_out - r_in_tip) / (1 + ρ_out/ρ_in)
            Y_lk = 0.6 * t_cl/b_out * v_out * sqrt(4π/(b_out*z) * α * v_t_out * v_in)
        
        Args:
            geom: Geometry with tip_cl, b4, r4, r2s, r2h, n_blades
            inlet_state, outlet_state: FluidState with density D
            velocities: dict with 'v_out', 'v_t_out', 'v_in' (m/s)
        
        Returns:
            float: Leakage loss coefficient Y_lk (dimensionless)
        
        Reference: TurboFlow loss_model_zhang_set1.py, loss_model_zhang_set3.py
                  compute_parasitic_losses() function
        """
        t_cl = geom.tip_cl
        b_out = geom.b4
        r_out = geom.r4
        r_in_tip = geom.r2s
        r_hub_in = geom.r2h
        z = geom.n_blades
        
        v_out = velocities['v_out']
        v_t_out = velocities['v_t_out']
        v_in = velocities['v_in']
        
        rho_out = outlet_state.D
        rho_in = inlet_state.D
        
        # Alpha parameter
        alpha = (r_in_tip - r_hub_in) / (r_out - r_in_tip) / (1 + rho_out/rho_in)
        
        # Leakage loss
        Y_lk = 0.6 * t_cl/b_out * v_out * np.sqrt(4*np.pi/(b_out*z) * alpha * v_t_out * v_in)
        
        return Y_lk
    
    @staticmethod
    def aungier_method(geom, inlet_state, outlet_state, velocities, mass_flow):
        """
        Leakage loss - Oh and Zhang Set 2 - Aungier method
        
        Formula (from loss_model_oh.py and loss_model_zhang_set2.py):
            r_avg = (r_mean_in + r_out) / 2
            b_avg = (b_in + b_out) / 2
            Δp = m_dot * (r_out*v_t_out - r_mean_in*v_t_in) / (z*r_avg*b_avg*L_m)
            u_cl = 0.816 * sqrt(2*Δp / ρ_out)
            m_cl = ρ_out * z * t_cl * L_m * u_cl
            Y_lk = m_cl * u_cl * u_out / (2 * m_dot)
        
        Args:
            geom: Geometry with:
                - tip_cl: Tip clearance (m)
                - b2: Inlet width (m)
                - b4: Outlet width (m)
                - r4: Outlet radius (m)
                - r2_mean: Mean inlet radius (m) 
                - l_meridional: Meridional length (m)
                - n_blades: Number of blades
            inlet_state, outlet_state: FluidState with D (density)
            velocities: dict with 'v_t_in', 'v_t_out', 'u_out' (m/s)
            mass_flow: Mass flow rate (kg/s)
        
        Returns:
            float: Leakage loss coefficient Y_lk (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set2.py
                  compute_parasitic_losses() function
        """
        t_cl = geom.tip_cl
        b_in = geom.r2s - geom.r2h
        b_out = geom.b4
        r_out = geom.r4
        r_mean_in = (geom.r2s + geom.r2h) / 2  # Mean radius at inlet
        z = geom.n_blades
        
        v_t_in = velocities['v_t_in']
        v_t_out = velocities['v_t_out']
        u_out = velocities['u_out']
        
        rho_out = outlet_state.D
        
        # Average radius and width
        r_avg = (r_mean_in + r_out) / 2
        b_avg = (b_in + b_out) / 2

        L_ax = 0.5 * (geom.r4 - geom.r2s) + geom.b4
        R_ay = (r_out - r_mean_in)
        L_m = np.sqrt(L_ax**2 + R_ay**2)
        
        # Pressure difference across clearance
        delta_p = mass_flow * (r_out*v_t_out - r_mean_in*v_t_in) / (z*r_avg*b_avg*L_m)
        
        # Clearance velocity (orifice flow with discharge coefficient 0.816)
        u_cl = 0.816 * np.sqrt(2*delta_p / rho_out)
        
        # Leakage mass flow
        m_cl = rho_out * z * t_cl * L_m * u_cl
        
        # Leakage loss
        Y_lk = m_cl * u_cl * u_out / (2 * mass_flow)
        
        return Y_lk


class ImpellerMixingLoss:
    """
    Impeller wake mixing losses
    
    Available methods:
    - johnston_dean_method: Oh and Zhang Set 2 - wake-based
    - aungier_method: Zhang Set 1 and Set 3 - separation-based
    """
    
    @staticmethod
    def johnston_dean_method(geom, velocities, wake_width=0.05):
        """
        Mixing loss - Johnston and Dean method (Oh, Zhang Set 2, Custom)
        
        Formula:
            b_star = b_diffuser/b_out  (typically b_star = 1.0)
            Y_mix = 1/(1+tan²(α_out)) * ((1-wake_width-b_star)/(1-wake_width))² * v_out²/2
        
        Args:
            geom: Geometry with b4 (outlet width, m)
            velocities: dict with:
                - 'alpha_out': Flow angle at outlet (deg)
                - 'v_out': Absolute velocity at outlet (m/s)
            wake_width: Wake width parameter (default 0.05)
        
        Returns:
            float: Mixing loss Y_mix (m²/s²)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set2.py, loss_model_custom.py
        """
        alpha_out = velocities['alpha_out']
        v_out = velocities['v_out']
        
        # Assume no change in width from impeller exit to diffuser inlet
        b_star = 1.0
        
        # Johnston and Dean mixing loss
        Y_mix = (1/(1 + tand(alpha_out)**2) * 
                 ((1 - wake_width - b_star)/(1 - wake_width))**2 * 
                 0.5 * v_out**2)
        
        return Y_mix
    
    @staticmethod
    def aungier_method(geom, velocities):
        """
        Mixing loss - Aungier method (Zhang Set 1, Set 3)
        
        Based on equivalent diffusion ratio and separation velocity.
        
        Formula:
            W_out = sqrt((v_m_out*A_out/(2π*r_out*b_out))² + w_t_out²)
            W_max = (w_in + w_out + ΔW)/2
            D_eq = W_max/w_out
            W_sep = w_out  (if D_eq ≤ 2.0) or w_out*D_eq/2  (if D_eq > 2.0)
            Y_mix = 0.5*(W_sep - W_out)²
        
        Args:
            geom: Geometry with r4, b4, A_out, n_blades
            velocities: dict with:
                - 'v_m_out': Meridional velocity at outlet (m/s)
                - 'w_in': Relative velocity at inlet (m/s)
                - 'w_out': Relative velocity at outlet (m/s)
                - 'w_t_out': Tangential component of relative velocity at outlet (m/s)
                - 'v_t_out': Tangential absolute velocity at outlet (m/s)
                - 'delta_W': Velocity difference on blade (from blade loading calc)
        
        Returns:
            float: Mixing loss Y_mix (m²/s²)
        
        Reference: TurboFlow loss_model_zhang_set1.py, loss_model_zhang_set3.py
        """
        r_in_tip = geom.r2s
        r_in_hub = geom.r2h
        beta_in = geom.beta2
        
        r_out = geom.r4
        b_out = geom.b4
        beta_out = geom.beta4
        A_out = geom.A4_eff
        z = geom.n_blades

        v_m_out = velocities['v_m_out']
        w_in = velocities['w_in']
        w_out = velocities['w_out']
        w_t_out = velocities['w_t_out']

        # Axial length
        L_ax = 0.5 * (geom.r4 - geom.r2s) + geom.b4
        
        # Blade length
        L_b = (np.pi/8 * (r_out*2 - (r_in_tip + r_in_hub) - b_out + 2*L_ax) *
               (2/(cosd(beta_in) + cosd(beta_out))))
        
        # Velocity difference on blade
        v_t_out = velocities['v_t_out']

        delta_W = 2*np.pi*2*r_out*v_t_out / (z*L_b)
        
        # Actual outlet relative velocity considering blockage
        W_out = np.sqrt((v_m_out * A_out / (2*np.pi*r_out*b_out))**2 + w_t_out**2)
        
        # Maximum velocity in passage
        W_max = (w_in + w_out + delta_W) / 2
        
        # Equivalent diffusion ratio
        D_eq = W_max / w_out
        
        # Separation velocity (piecewise based on D_eq)
        if D_eq <= 2.0:
            W_sep = w_out
        else:
            W_sep = w_out * D_eq / 2
        
        # Mixing loss
        Y_mix = 0.5 * (W_sep - W_out)**2
        
        return Y_mix


class ImpellerChokeLoss:
    """
    Impeller choke losses (Aungier)
    
    Available in: Zhang Set 1, Set 2, Set 3
    """
    
    @staticmethod
    def aungier_method(geom, inlet_state, velocities, mass_flow):
        """
        Choke loss - Aungier method
        
        Based on critical area and choking parameter.
        
        Formula:
            R = cp - cv
            A_crit = m_dot/(p0*sqrt(γ/(R*T0))*(2/(γ+1))^((γ+1)/(2*(γ-1))))
            Cr = sqrt(A_in*cos(θ_in)/A_th)
            Cr = min(Cr, 1-(A_in*cos(θ_in)/A_th-1)²)
            X = 11-10*(Cr*A_th)/A_crit
            Y_ch = 0.5*w_in²*(0.05*X+X⁷)  if X > 0, else 0
        
        Args:
            geom: Geometry with:
                - A_in: Inlet area (m²)
                - A_throat: Throat area (m²)
                - theta_in: Leading edge angle (deg)
            inlet_state: FluidState with:
                - P0: Total pressure (Pa)
                - T0: Total temperature (K)
                - cp: Specific heat at constant pressure (J/kg·K)
                - cv: Specific heat at constant volume (J/kg·K)
                - gamma: Specific heat ratio
            velocities: dict with 'w_in' (relative velocity at inlet, m/s)
            mass_flow: Mass flow rate (kg/s)
        
        Returns:
            float: Choke loss Y_ch (m²/s²)
        
        Reference: TurboFlow loss_model_zhang_set1/2/3.py
        """
        A_in = geom.A2_eff
        A_th = geom.A_y
        theta_in = geom.beta2  # Leading edge angle
        
        p0_in = inlet_state.P
        T0_in = inlet_state.T
        cp = inlet_state.cp
        cv = inlet_state.cv
        gamma = cp / cv
        
        w_in = velocities['w_in']
        
        # Gas constant
        R = cp - cv
        
        # Critical area (choked flow area)
        A_crit = (mass_flow / (p0_in * np.sqrt(gamma/(R*T0_in)) * 
                  (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))))
        
        # Area contraction ratio
        Cr = np.sqrt(A_in * cosd(theta_in) / A_th)
        Cr = min(Cr, 1 - (A_in * cosd(theta_in) / A_th - 1)**2)
        
        # Choking parameter
        X = 11 - 10 * (Cr * A_th) / A_crit
        
        # Choke loss (only if X > 0)
        if X > 0:
            Y_ch = 0.5 * w_in**2 * (0.05*X + X**7)
        else:
            Y_ch = 0.0
        
        return Y_ch


class ImpellerShockLoss:
    """
    Impeller shock losses due to supersonic flow
    
    Available methods:
    - whitfield_baines_method: Zhang Set 2, Set 3 - compressible flow
    - custom_method: Custom model - Mach number based
    """
    
    @staticmethod
    def whitfield_baines_method(inlet_state, throat_state, velocities):
        """
        Shock loss - Whitfield and Baines method (Zhang Set 2, Set 3)
        
        Formula:
            Y_sh = w_in²/2 * (1-(w_th/w_in)² - 2/((γ-1)*Ma_rel_in²) * 
                   ((p_th/p_in)^(γ/(γ-1)) - 1))
        
        Args:
            inlet_state: FluidState with:
                - P: Static pressure (Pa)
                - Ma_rel: Relative Mach number
                - gamma: Specific heat ratio
            throat_state: FluidState with:
                - P: Static pressure at throat (Pa)
            velocities: dict with:
                - 'w_in': Relative velocity at inlet (m/s)
                - 'w_th': Relative velocity at throat (m/s)
        
        Returns:
            float: Shock loss Y_sh (m²/s²)
        
        Reference: TurboFlow loss_model_zhang_set2.py, loss_model_zhang_set3.py
        """
        w_in = velocities['w_in']
        w_th = velocities['w_th']
        
        p_in = inlet_state.P
        p_th = throat_state.P
        Ma_rel_in = w_in / inlet_state.A
        gamma = inlet_state.cp / inlet_state.cv
        
        # Whitfield and Baines shock loss
        Y_sh = (w_in**2 / 2 * 
                (1 - (w_th/w_in)**2 - 
                 2/((gamma-1)*Ma_rel_in**2) * ((p_th/p_in)**(gamma/(gamma-1)) - 1)))
        
        return max(0.0, Y_sh)  # Ensure non-negative
    
    @staticmethod
    def custom_method(inlet_state, throat_state, velocities):
        """
        Shock loss - Custom method (TurboFlow Custom model)
        
        Simplified Mach-based shock loss.
        
        Formula:
            w_th = w_m_in * A_in / A_throat
            Y_shock = 0.56 * max(0, ((w_th/a_in)² - 1)³)
        
        Args:
            inlet_state: FluidState with:
                - a: Speed of sound (m/s)
            throat_state: Not used in this method
            velocities: dict with:
                - 'w_m_in': Meridional relative velocity at inlet (m/s)
                - 'A_in': Inlet area (m²)
                - 'A_throat': Throat area (m²)
        
        Returns:
            float: Shock loss Y_shock (m²/s²)
        
        Reference: TurboFlow loss_model_custom.py
        """
        w_m_in = velocities['w_m_in']
        A_in = velocities['A_in']
        A_throat = velocities['A_throat']
        a_in = inlet_state.A
        
        # Throat velocity (continuity)
        w_th = w_m_in * A_in / A_throat
        
        # Shock loss (only if supersonic)
        Mach_th = w_th / a_in
        Y_shock = 0.56 * max(0, (Mach_th**2 - 1)**3)
        
        return Y_shock


class ImpellerEntranceDiffusionLoss:
    """
    Impeller entrance diffusion losses (Aungier)
    
    Available in: Zhang Set 1, Set 2, Set 3
    """
    
    @staticmethod
    def aungier_method(velocities, Y_inc):
        """
        Entrance diffusion loss - Aungier method
        
        Formula:
            Y_dif = max(0, 0.4*(w_in - w_th)² - Y_inc)
        
        Args:
            velocities: dict with:
                - 'w_in': Relative velocity at inlet (m/s)
                - 'w_th': Relative velocity at throat (m/s)
            Y_inc: Incidence loss (m²/s²) - already computed
        
        Returns:
            float: Entrance diffusion loss Y_dif (m²/s²)
        
        Reference: TurboFlow loss_model_zhang_set1/2/3.py
        """
        w_in = velocities['w_in']
        w_th = velocities['w_th']
        
        # Diffusion loss minus incidence (avoid double-counting)
        Y_dif = max(0, 0.4*(w_in - w_th)**2 - Y_inc)
        
        return Y_dif


# ============================================================================
# SECTION 3: VANELESS DIFFUSER LOSSES
# ============================================================================

class VanelessDiffuserLoss:
    """
    Vaneless diffuser losses
    
    Available methods:
    - stanitz_method: - thermodynamic approach
    - japikse_method: Schiffmann - friction-based stepwise integration
    """
    
    @staticmethod
    def stanitz_method(inlet_state, outlet_state):
        """
        Vaneless diffuser loss - Stanitz method 
        
        Thermodynamic approach based on pressure ratios and specific heats.
        
        Formula :
            γ = cp/cv
            α = (γ - 1)/γ
            Y_tot = cp*T0_in*((p_out/p0_out)^α - (p_out/p0_in)^α)
        
        Args:
            inlet_state: FluidState at diffuser inlet (station 4) with:
                - T0: Total temperature (K)
                - P0: Total pressure (Pa)
                - P: Static pressure (Pa)
                - cp: Specific heat at constant pressure (J/kg·K)
                - cv: Specific heat at constant volume (J/kg·K)
                - V_abs: Absolute velocity (m/s)
            outlet_state: FluidState at diffuser outlet (station 5) with:
                - P: Static pressure (Pa)
                - P0: Total pressure (Pa)
        
        Returns:
            float: Loss coefficient Y (dimensionless)
        
        Reference: TurboFlow loss_model_oh.py, loss_model_zhang_set1/2/3.py
                  compute_vaneless_diffuser_losses()
        """
        # Extract parameters
        T0_in = inlet_state.T0
        p0_in = inlet_state.P0
        p_out = outlet_state.P
        p0_out = outlet_state.P0
        cp = inlet_state.cp
        cv = inlet_state.cv
        v_in = inlet_state.V_abs
        
        # Specific heat ratio
        gamma = cp / cv
        alpha = (gamma - 1) / gamma
        
        # Stanitz loss formula
        Y_tot = cp * T0_in * ((p_out/p0_out)**alpha - (p_out/p0_in)**alpha)
    
        return Y_tot
    
    @staticmethod
    def japikse_method(geom, inlet_state, outlet_state, n_steps=15):
        """
        Vaneless diffuser loss - Japikse method 
        
        Stepwise integration approach with friction losses at each step.
        More detailed than Stanitz, accounts for geometry changes.
        
        Formula:
            At each radial step i:
            Re = c * ρ / μ * b
            Cf = k * (1.8e5 / Re)^0.2  (Japikse friction factor)
            dp0 = 4 * Cf * ds * c² * ρ / (2 * D_h)
        
        Args:
            geom: Geometry with:
                - r4: Diffuser inlet radius (m)
                - r5: Diffuser outlet radius (m)
                - b4: Diffuser inlet width (m)
                - b5: Diffuser outlet width (m)
                - blockage[4]: Blockage factor at station 4
            inlet_state: FluidState at diffuser inlet
            outlet_state: FluidState at diffuser outlet (for comparison)
            n_steps: Number of integration steps (default 15)
        
        Returns:
            float: Total pressure loss dp0_total (Pa)
    
        Note: Returns pressure loss in Pa, not dimensionless coefficient
        """
        # Radial discretization
        r = np.linspace(geom.r4, geom.r5, 1 + n_steps)
        dr = np.diff(r)
        
        # Width variation (linear interpolation)
        b = np.linspace(geom.b4, geom.b5, 1 + n_steps)
        
        # Hydraulic diameter at each step
        D_h = np.sqrt(8 * r[:-1] * b[1:] * geom.blockage[4])
        
        # Japikse friction constant
        k = 0.02
        
        # Initialize
        c = inlet_state.V_abs
        alpha = inlet_state.alpha  # Flow angle (deg)
        rho = inlet_state.D
        mu = inlet_state.V
        
        dp0_total = 0.0
        
        # Stepwise integration
        for i in range(n_steps):
            # Reynolds number
            Re = c * rho / mu * b[i + 1]
            
            # Japikse friction factor
            Cf = k * (1.8e5 / Re)**0.2
            
            # Path length along streamline
            ds = np.sqrt((dr[i] / tand(90 - alpha))**2 + dr[i]**2)
            
            # Pressure loss in this step
            dp0 = 4.0 * Cf * ds * c**2 * rho / 2 / D_h[i]
            dp0_total += dp0
            
            # Update velocity (conservation of angular momentum)
            # c_t * r = constant
            c_t = c * sind(alpha)
            c_t_new = c_t * r[i] / r[i+1]
            
            # Meridional velocity approximately constant
            c_m = c * cosd(alpha)
            
            # New total velocity
            c = np.sqrt(c_m**2 + c_t_new**2)
            alpha = math.degrees(math.asin(c_t_new / c))
        
        return dp0_total


# ============================================================================
# SECTION 4: VANED DIFFUSER LOSSES (TurboFlow Custom Model)
# ============================================================================

class VanedDiffuserLoss:
    """
    Vaned diffuser losses
    
    Available methods:
    - conrad_method: TurboFlow Custom model - incidence and skin friction
    """
    
    @staticmethod
    def conrad_method(geom, inlet_state, outlet_state, velocities, Cf=0.005):
        """
        Vaned diffuser loss - Conrad method (Meroni paper)
        
        Computes incidence and skin friction losses for vaned diffusers.
        
        Formula (from TurboFlow loss_model_custom.py):
            # Skin friction:
            L_b = (r_out - r_in) / cos((θ_in + θ_out)/2)
            D_h = opening*b_throat/(opening+b_throat) + pitch*b_out/(pitch+b_out)
            Y_sf = 2*Cf*v_m_out²*L_b/D_h
            
            # Incidence:
            α_des = θ_out  (design angle = trailing edge angle)
            Y_inc = 0.6*sin(|α_out - α_des|)*0.5*v_out²
        
        Args:
            geom: Geometry with:
                - r5: Diffuser inlet radius (m)
                - r6: Diffuser outlet radius (m)
                - theta_vane_in: Leading edge angle (deg)
                - theta_vane_out: Trailing edge angle (deg)
                - b6: Width at outlet (m)
                - b_throat_vane: Throat width (m)
                - vane_opening: Vane opening/spacing (m)
                - vane_pitch: Vane pitch (m)
            inlet_state: FluidState at diffuser inlet
            outlet_state: FluidState at diffuser outlet
            velocities: dict with:
                - 'v_in': Absolute velocity at inlet (m/s)
                - 'v_out': Absolute velocity at outlet (m/s)
                - 'v_m_out': Meridional velocity at outlet (m/s)
                - 'alpha_out': Flow angle at outlet (deg)
            Cf: Skin friction coefficient (default 0.005)
        
        Returns:
            dict: Dictionary with keys 'incidence', 'skin_friction', 'loss_total'
        
        Reference: TurboFlow loss_model_custom.py
                  compute_vaned_diffuser_losses()
        """
        # Extract geometry
        r_in = geom.r5
        r_out = geom.r6
        theta_in = geom.theta_vane_in
        theta_out = geom.theta_vane_out
        b_out = geom.b6
        b_throat = geom.b_throat_vane
        opening = geom.vane_opening
        pitch = geom.vane_pitch
        
        # Extract velocities
        v_in = velocities['v_in']
        v_out = velocities['v_out']
        v_m_out = velocities['v_m_out']
        alpha_out = velocities['alpha_out']
        
        # Blade length
        L_b = (r_out - r_in) / cosd((theta_in + theta_out) / 2)
        
        # Hydraulic diameter (parallel formula for two passages)
        D_h = (opening * b_throat / (opening + b_throat) + 
               pitch * b_out / (pitch + b_out))
        
        # Skin friction loss
        Y_sf = 2 * Cf * v_m_out**2 * L_b / D_h
        
        # Incidence loss
        alpha_des = theta_out  # Design angle equals trailing edge angle
        Y_inc = 0.6 * sind(abs(alpha_out - alpha_des)) * 0.5 * v_out**2
        
        return {
            'incidence': Y_inc,
            'skin_friction': Y_sf,
            'loss_total': Y_inc + Y_sf
        }


# ============================================================================
# SECTION 5: VOLUTE LOSSES (TurboFlow Custom Model)
# ============================================================================

class VoluteLoss:
    """
    Volute/scroll losses
    
    Available methods:
    - roberto_method: TurboFlow Custom model - expansion losses
    """
    
    @staticmethod
    def roberto_method(geom, inlet_state, alpha_in):
        """
        Volute loss - Roberto's method
        
        Computes three components of volute losses:
        1. Radial kinetic energy dissipation
        2. Tangential kinetic energy loss in scroll expansion
        3. Tangential kinetic energy loss in cone expansion
        
        Formula (from TurboFlow loss_model_custom.py):
            ζ_rad = cos²(α_in)
            ζ_scroll = sin²(α_in)*(1 - A_in*sin(α_in)/A_scroll)²
            ζ_cone = cos²(α_in)*(A_in/A_scroll)²*(1 - A_scroll/A_out)²
            ζ_total = ζ_rad + ζ_scroll + ζ_cone
        
        Args:
            geom: Geometry with:
                - A_volute_in: Volute inlet area (m²)
                - A_scroll: Scroll cross-sectional area (m²)
                - A_volute_out: Volute outlet area (m²)
            inlet_state: FluidState at volute inlet
            alpha_in: Flow angle at volute inlet (deg)
        
        Returns:
            dict: Dictionary with keys 'loss_radial', 'loss_scroll', 
                  'loss_cone', 'loss_total'
        
        Reference: TurboFlow loss_model_custom.py
                  compute_volute_losses()
        """
        # Extract geometry
        A_in = geom.A_volute_in
        A_scroll = geom.A_scroll
        A_out = geom.A_volute_out
        
        # Dissipation of radial kinetic energy
        zeta_rad = cosd(alpha_in)**2
        
        # Dissipation due to expansion in scroll
        zeta_scroll = (sind(alpha_in)**2 * 
                      (1 - A_in * sind(alpha_in) / A_scroll)**2)
        
        # Dissipation due to expansion in cone
        zeta_cone = (cosd(alpha_in)**2 * (A_in / A_scroll)**2 * 
                    (1 - A_scroll / A_out)**2)
        
        # Total loss coefficient
        zeta_total = zeta_rad + zeta_scroll + zeta_cone
        
        return {
            'loss_radial': zeta_rad,
            'loss_scroll': zeta_scroll,
            'loss_cone': zeta_cone,
            'loss_total': zeta_total
        }
