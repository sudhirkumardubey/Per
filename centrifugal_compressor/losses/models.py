# File: centrifugal_compressor/losses/models.py

"""
Loss model definitions - EXACT combinations from TurboFlow

STEP 1: OH MODEL (verified against loss_model_oh.py)
"""

from typing import Dict, Any
from .components import *


# ============================================================================
# BASE MODEL CLASS
# ============================================================================

class BaseLossModel:
    """Base class for all loss models"""
    
    def __init__(self, name: str):
        self.name = name
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, velocities, mass_flow, **kwargs):
        """Override in subclasses - returns RAW loss values (not scaled)"""
        raise NotImplementedError(f"{self.name} model must implement compute_impeller_losses")
    
    def compute_vaneless_diffuser_losses(self, inlet_state, outlet_state, **kwargs):
        """Override in subclasses"""
        raise NotImplementedError(f"{self.name} model does not support vaneless diffuser")
    
    def compute_vaned_diffuser_losses(self, inlet_state, outlet_state, **kwargs):
        """Override in subclasses"""
        raise NotImplementedError(f"{self.name} model does not support vaned diffuser")


# ============================================================================
# OH MODEL (1997) - ✅ VERIFIED AGAINST loss_model_oh.py
# ============================================================================

class OhLossModel(BaseLossModel):
    """
    Oh et al. (1997) loss model
    
    EXACT implementation from TurboFlow loss_model_oh.py
    
    Impeller losses (8 total):
    1. Incidence: Conrad (f_inc=0.5, Y_inc = 0.5*0.5*w_t_in²)
    2. Blade loading: Coppage (Df-based, Y_bld = 0.05*Df²*u_out²)
    3. Skin friction: Jansen (hydraulic diameter method)
    4. Clearance: Jansen (tip clearance)
    5. Mixing: Johnston & Dean (wake mixing)
    6. Recirculation: Oh (sinh method, Y_rc = 8e-5*sinh(...)*Df²*u_out²)
    7. Disc friction: Daily & Nece
    8. Leakage: Aungier (meridional length method)
    
    Vaneless diffuser: Stanitz
    """
    
    def __init__(self):
        super().__init__("Oh")
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, velocities, mass_flow, enthalpy_rise):
        """
        Compute impeller losses - Oh model
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet
            outlet_state: FluidState at outlet
            velocities: dict with keys:
                - 'v_in', 'v_out': Absolute velocities
                - 'v_m_in': Meridional velocity at inlet
                - 'v_t_in', 'v_t_out': Tangential velocities
                - 'w_in', 'w_out': Relative velocities
                - 'w_t_in': Tangential relative velocity at inlet
                - 'u_out': Blade speed at outlet
                - 'alpha_out': Flow angle at outlet
                - 'wake_width': Wake width parameter (optional, default 0.05)
            mass_flow: Mass flow rate (kg/s)
            enthalpy_rise: h0_out - h0_in (J/kg)
        
        Returns:
            dict: RAW loss values (m²/s² or J/kg) before scaling
        """
        losses = {}
        
        # 1. Incidence (Conrad) - f_inc = 0.5
        # Y_inc = 0.5*0.5*w_t_in²
        losses['incidence'] = ImpellerIncidenceLoss.oh_conrad_method(velocities)
        
        # 2. Blade loading (Coppage with Df)
        # Y_bld = 0.05*Df²*u_out²
        losses['blade_loading'] = ImpellerBladeLoadingLoss.oh_coppage_method(
            geom, inlet_state, outlet_state, velocities, enthalpy_rise
        )
        
        # 3. Skin friction (Jansen)
        # Y_sf = 2*Cf*L_b/D_h*w_avg²
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 4. Clearance (Jansen)
        # Y_cl = 0.6*t_cl/b_out*v_t_out*sqrt(...)
        losses['clearance'] = ImpellerClearanceLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 5. Mixing (Johnston & Dean)
        # Y_mix = 1/(1+tan²(α))*((1-w-b*)/(1-w))²*0.5*v_out²
        wake_width = velocities.get('wake_width', 0.05)
        losses['mixing'] = ImpellerMixingLoss.johnston_dean_method(
            geom, velocities, wake_width
        )
        
        # 6. Recirculation (Oh - sinh method)
        # Y_rc = 8e-5*sinh(3.5*(α*π/180)³)*Df²*u_out²
        losses['recirculation'] = ImpellerRecirculationLoss.oh_method(
            geom, inlet_state, outlet_state, velocities, enthalpy_rise
        )
        
        # 7. Disc friction (Daily & Nece)
        # Y_df = f_df*ρ_avg*r²*u³/(4*m_dot)
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 8. Leakage (Aungier - meridional length method)
        # Y_lk = m_cl*u_cl*u_out/(2*m_dot)
        losses['leakage'] = ImpellerLeakageLoss.aungier_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        return losses
    
    def compute_vaneless_diffuser_losses(self, inlet_state, outlet_state):
        """
        Vaneless diffuser - Stanitz method
        
        Returns:
            dict: {'loss_total': Y_tot} in J/kg
        """
        Y_tot = VanelessDiffuserLoss.stanitz_method(inlet_state, outlet_state)
        return {'loss_total': Y_tot}

# ============================================================================
# ZHANG SET 1 MODEL - ✅ VERIFIED AGAINST loss_model_zhang_set1.py
# ============================================================================

class ZhangSet1LossModel(BaseLossModel):
    """
    Zhang Set 1 loss model
    
    EXACT implementation from TurboFlow loss_model_zhang_set1.py
    
    Impeller losses (11 total):
    1. Incidence: Aungier (Y_inc = 0.4*(w_in - v_m_in/cos(θ))²)
    2. Blade loading: Aungier (Y_bld = (ΔW)²/48)
    3. Skin friction: Jansen
    4. Clearance: Jansen  
    5. Mixing: Aungier (separation-based)
    6. Entrance diffusion: Aungier (Y_dif = max(0, 0.4*(w_in-w_th)² - Y_inc))
    7. Choke: Aungier (critical area method)
    8. Recirculation: Coppage (Y_rc = 0.02*sqrt(tan(α))*Df²*u²)
    9. Disc friction: Daily & Nece
    10. Leakage: Jansen (Y_lk = 0.6*t_cl/b*v*sqrt(...))
    
    Vaneless diffuser: Stanitz
    """
    
    def __init__(self):
        super().__init__("Zhang_Set1")
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, throat_state, velocities, mass_flow):
        """
        Compute impeller losses - Zhang Set 1
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet
            outlet_state: FluidState at outlet
            throat_state: FluidState at throat
            velocities: dict with keys (MUST include):
                - 'v_in', 'v_out', 'v_m_in', 'v_m_out', 'v_t_in', 'v_t_out'
                - 'w_in', 'w_out', 'w_t_in', 'w_t_out', 'w_th'
                - 'u_in', 'u_out'
                - 'alpha_out'
                - 'delta_W': Velocity difference on blade (from blade loading calc)
            mass_flow: Mass flow rate (kg/s)
        
        Returns:
            dict: RAW loss values (m²/s²) before scaling
        """
        losses = {}
        
        # 1. Incidence (Aungier)
        # Y_inc = 0.4*(w_in - v_m_in/cos(θ_in))²
        losses['incidence'] = ImpellerIncidenceLoss.zhang_aungier_method(geom, velocities)
        
        # 2. Blade loading (Aungier - ΔW²/48)
        # Y_bld = (ΔW)²/48
        losses['blade_loading'] = ImpellerBladeLoadingLoss.zhang_aungier_method(geom, velocities)
        
        # 3. Skin friction (Jansen)
        # Y_sf = 2*Cf*L_b/D_h*w_avg²
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 4. Clearance (Jansen)
        # Y_cl = 0.6*t_cl/b_out*v_t_out*sqrt(4π/(b*z)*...)
        losses['clearance'] = ImpellerClearanceLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 5. Mixing (Aungier)
        # W_out, W_max, D_eq, W_sep → Y_mix = 0.5*(W_sep - W_out)²
        losses['mixing'] = ImpellerMixingLoss.aungier_method(geom, velocities)
        
        # 6. Entrance diffusion (Aungier)
        # Y_dif = max(0, 0.4*(w_in - w_th)² - Y_inc)
        losses['entrance_diffusion'] = ImpellerEntranceDiffusionLoss.aungier_method(
            velocities, losses['incidence']
        )
        
        # 7. Choke (Aungier)
        # A_crit, Cr, X → Y_ch = 0.5*w_in²*(0.05*X + X⁷)
        losses['choke'] = ImpellerChokeLoss.aungier_method(
            geom, inlet_state, velocities, mass_flow
        )
        
        # 8. Recirculation (Coppage - sqrt(tan))
        # Y_rc = 0.02*sqrt(tan(α_out))*Df²*u_out²
        losses['recirculation'] = ImpellerRecirculationLoss.coppage_method(geom, velocities)
        
        # 9. Disc friction (Daily & Nece)
        # Y_df = f_df*ρ_avg*r²*u³/(4*m)
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 10. Leakage (Jansen)
        # Y_lk = 0.6*t_cl/b*v_out*sqrt(4π/(b*z)*α*v_t*v_in)
        losses['leakage'] = ImpellerLeakageLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        return losses
    
    def compute_vaneless_diffuser_losses(self, inlet_state, outlet_state):
        """Vaneless diffuser - Stanitz method"""
        Y_tot = VanelessDiffuserLoss.stanitz_method(inlet_state, outlet_state)
        return {'loss_total': Y_tot}

# ============================================================================
# ZHANG SET 2 MODEL - ✅ VERIFIED AGAINST loss_model_zhang_set2.py
# ============================================================================

class ZhangSet2LossModel(BaseLossModel):
    """
    Zhang Set 2 loss model
    
    EXACT implementation from TurboFlow loss_model_zhang_set2.py
    
    KEY DIFFERENCES from Set 1:
    - Blade loading: Coppage (Df-based, like Oh) instead of Aungier
    - Mixing: Johnston & Dean (like Oh) instead of Aungier
    - Shock: Whitfield & Baines (compressible) instead of none
    - Leakage: Aungier (meridional) instead of Jansen
    
    Impeller losses (12 total):
    1. Incidence: Aungier (Y_inc = 0.4*(w_in - v_m_in/cos(θ))²)
    2. Blade loading: Coppage (Df-based, Y_bld = 0.05*Df²*u_out²)
    3. Skin friction: Jansen
    4. Clearance: Jansen
    5. Mixing: Johnston & Dean (wake mixing)
    6. Entrance diffusion: Aungier
    7. Choke: Aungier
    8. Shock: Whitfield & Baines (Y_sh = w²/2*(1-(w_th/w)²-2/(...)))
    9. Recirculation: Coppage (sqrt(tan))
    10. Disc friction: Daily & Nece
    11. Leakage: Aungier (meridional length)
    
    Vaneless diffuser: Stanitz
    """
    
    def __init__(self):
        super().__init__("Zhang_Set2")
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, throat_state, velocities, mass_flow, enthalpy_rise):
        """
        Compute impeller losses - Zhang Set 2
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet (needs Ma_rel, gamma)
            outlet_state: FluidState at outlet
            throat_state: FluidState at throat (needs P for shock loss)
            velocities: dict with keys (MUST include):
                - 'v_in', 'v_out', 'v_m_in', 'v_m_out', 'v_t_in', 'v_t_out'
                - 'w_in', 'w_out', 'w_t_in', 'w_t_out', 'w_th'
                - 'u_in', 'u_out'
                - 'alpha_out'
                - 'wake_width' (optional, default 0.05)
            mass_flow: Mass flow rate (kg/s)
            enthalpy_rise: h0_out - h0_in (J/kg)
        
        Returns:
            dict: RAW loss values (m²/s²) before scaling
        """
        losses = {}
        
        # 1. Incidence (Aungier) - SAME as Set 1
        # Y_inc = 0.4*(w_in - v_m_in/cos(θ))²
        losses['incidence'] = ImpellerIncidenceLoss.zhang_aungier_method(geom, velocities)
        
        # 2. Blade loading (Coppage - Df²) - DIFFERENT from Set 1
        # Y_bld = 0.05*Df²*u_out²
        losses['blade_loading'] = ImpellerBladeLoadingLoss.oh_coppage_method(
            geom, inlet_state, outlet_state, velocities, enthalpy_rise
        )
        
        # 3. Skin friction (Jansen) - SAME
        # Y_sf = 2*Cf*L_b/D_h*w_avg²
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 4. Clearance (Jansen) - SAME
        # Y_cl = 0.6*t_cl/b*v_t*sqrt(...)
        losses['clearance'] = ImpellerClearanceLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 5. Mixing (Johnston & Dean) - DIFFERENT from Set 1
        # Y_mix = 1/(1+tan²(α))*((1-w-b*)/(1-w))²*0.5*v²
        wake_width = velocities.get('wake_width', 0.05)
        losses['mixing'] = ImpellerMixingLoss.johnston_dean_method(
            geom, velocities, wake_width
        )
        
        # 6. Entrance diffusion (Aungier) - SAME
        # Y_dif = max(0, 0.4*(w_in - w_th)² - Y_inc)
        losses['entrance_diffusion'] = ImpellerEntranceDiffusionLoss.aungier_method(
            velocities, losses['incidence']
        )
        
        # 7. Choke (Aungier) - SAME
        # A_crit, Cr, X → Y_ch
        losses['choke'] = ImpellerChokeLoss.aungier_method(
            geom, inlet_state, velocities, mass_flow
        )
        
        # 8. Shock (Whitfield & Baines) - NEW in Set 2
        # Y_sh = w²/2*(1-(w_th/w)²-2/((γ-1)*Ma²)*((p_th/p)^(γ/(γ-1))-1))
        losses['shock'] = ImpellerShockLoss.whitfield_baines_method(
            inlet_state, throat_state, velocities
        )
        
        # 9. Recirculation (Coppage) - SAME
        # Y_rc = 0.02*sqrt(tan(α))*Df²*u²
        losses['recirculation'] = ImpellerRecirculationLoss.coppage_method(geom, velocities)
        
        # 10. Disc friction (Daily & Nece) - SAME
        # Y_df = f_df*ρ_avg*r²*u³/(4*m)
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 11. Leakage (Aungier) - DIFFERENT from Set 1
        # Y_lk = m_cl*u_cl*u_out/(2*m)
        losses['leakage'] = ImpellerLeakageLoss.aungier_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        return losses
    
    def compute_vaneless_diffuser_losses(self, inlet_state, outlet_state):
        """Vaneless diffuser - Stanitz method"""
        Y_tot = VanelessDiffuserLoss.stanitz_method(inlet_state, outlet_state)
        return {'loss_total': Y_tot}

# ============================================================================
# ZHANG SET 3 MODEL - ✅ VERIFIED AGAINST loss_model_zhang_set3.py
# ============================================================================

class ZhangSet3LossModel(BaseLossModel):
    """
    Zhang Set 3 loss model
    
    EXACT implementation from TurboFlow loss_model_zhang_set3.py
    
    KEY DIFFERENCES from Set 1 & Set 2:
    - Clearance: Rogers (simplified 0.1*t_cl/b*u²) instead of Jansen
    - All other losses same as Set 1 (Aungier-based)
    
    Impeller losses (11 total):
    1. Incidence: Aungier (Y_inc = 0.4*(w_in - v_m_in/cos(θ))²)
    2. Blade loading: Aungier (Y_bld = (ΔW)²/48)
    3. Skin friction: Jansen
    4. Clearance: Rogers (Y_cl = 0.1*t_cl/b*u²) ⭐ SIMPLIFIED
    5. Mixing: Aungier (separation-based)
    6. Entrance diffusion: Aungier
    7. Choke: Aungier
    8. Shock: Whitfield & Baines
    9. Recirculation: Coppage
    10. Disc friction: Daily & Nece
    11. Leakage: Jansen
    
    Vaneless diffuser: Stanitz
    """
    
    def __init__(self):
        super().__init__("Zhang_Set3")
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, throat_state, velocities, mass_flow):
        """
        Compute impeller losses - Zhang Set 3
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet (needs Ma_rel, gamma)
            outlet_state: FluidState at outlet
            throat_state: FluidState at throat
            velocities: dict with keys (MUST include):
                - 'v_in', 'v_out', 'v_m_in', 'v_m_out', 'v_t_in', 'v_t_out'
                - 'w_in', 'w_out', 'w_t_in', 'w_t_out', 'w_th'
                - 'u_in', 'u_out'
                - 'alpha_out'
                - 'delta_W': Velocity difference on blade
            mass_flow: Mass flow rate (kg/s)
        
        Returns:
            dict: RAW loss values (m²/s²) before scaling
        """
        losses = {}
        
        # 1. Incidence (Aungier) - SAME as Set 1
        # Y_inc = 0.4*(w_in - v_m_in/cos(θ))²
        losses['incidence'] = ImpellerIncidenceLoss.zhang_aungier_method(geom, velocities)
        
        # 2. Blade loading (Aungier) - SAME as Set 1
        # Y_bld = (ΔW)²/48
        losses['blade_loading'] = ImpellerBladeLoadingLoss.zhang_aungier_method(geom, velocities)
        
        # 3. Skin friction (Jansen) - SAME
        # Y_sf = 2*Cf*L_b/D_h*w_avg²
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 4. Clearance (Rogers - SIMPLIFIED) - ⭐ DIFFERENT from Set 1 & Set 2
        # Y_cl = 0.1*t_cl/b_out*u_out²
        t_cl = geom.tip_cl
        b_out = geom.b4
        u_out = velocities['u_out']
        losses['clearance'] = 0.1 * t_cl / b_out * u_out**2
        
        # 5. Mixing (Aungier) - SAME as Set 1
        # W_out, W_max, D_eq, W_sep → Y_mix = 0.5*(W_sep - W_out)²
        losses['mixing'] = ImpellerMixingLoss.aungier_method(geom, velocities)
        
        # 6. Entrance diffusion (Aungier) - SAME
        # Y_dif = max(0, 0.4*(w_in - w_th)² - Y_inc)
        losses['entrance_diffusion'] = ImpellerEntranceDiffusionLoss.aungier_method(
            velocities, losses['incidence']
        )
        
        # 7. Choke (Aungier) - SAME
        # A_crit, Cr, X → Y_ch
        losses['choke'] = ImpellerChokeLoss.aungier_method(
            geom, inlet_state, velocities, mass_flow
        )
        
        # 8. Shock (Whitfield & Baines) - SAME as Set 2
        # Y_sh = w²/2*(1-(w_th/w)²-2/((γ-1)*Ma²)*((p_th/p)^(...)-1))
        losses['shock'] = ImpellerShockLoss.whitfield_baines_method(
            inlet_state, throat_state, velocities
        )
        
        # 9. Recirculation (Coppage) - SAME
        # Y_rc = 0.02*sqrt(tan(α))*Df²*u²
        losses['recirculation'] = ImpellerRecirculationLoss.coppage_method(geom, velocities)
        
        # 10. Disc friction (Daily & Nece) - SAME
        # Y_df = f_df*ρ_avg*r²*u³/(4*m)
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 11. Leakage (Jansen) - SAME as Set 1
        # Y_lk = 0.6*t_cl/b*v*sqrt(4π/(b*z)*α*v_t*v_in)
        losses['leakage'] = ImpellerLeakageLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        return losses
    
    def compute_vaneless_diffuser_losses(self, inlet_state, outlet_state):
        """Vaneless diffuser - Stanitz method"""
        Y_tot = VanelessDiffuserLoss.stanitz_method(inlet_state, outlet_state)
        return {'loss_total': Y_tot}

# ============================================================================
# SCHIFFMANN MODEL - ✅ VERIFIED AGAINST RadComp files
# ============================================================================

class SchiffmannLossModel(BaseLossModel):
    """
    Schiffmann loss model (Schiffmann)
    
    EXACT implementation from RadComp (inducer.py, impeller.py, diffuser.py)
    
    Inducer losses:
    - Friction: Darcy-Weisbach (ΔP = 4*Cf*L*ρ*c²/(4*r))
    
    Impeller losses (6 total):
    1. Incidence: Stanitz-Galvas (dh_inc = 0.5*(w*sin(|β_flow-β_opt|))²)
    2. Skin friction: Jansen (Coppage-Galvas variant)
    3. Blade loading: Coppage-Rodgers (0.05*Df²*u²)
    4. Clearance: Jansen-Brasz (tip clearance)
    5. Disc friction: Daily & Nece
    6. Recirculation: Coppage (0.02*Df²*tan(α)*u²)
    
    Vaneless diffuser:
    - Japikse stepwise integration (15 steps, Cf = k*(1.8e5/Re)^0.2)
    """
    
    def __init__(self):
        super().__init__("RadComp")
    
    def compute_inducer_losses(self, geom, fluid_state, velocity):
        """
        Inducer friction loss - Darcy-Weisbach
        
        Returns ΔP in Pa (NOT scaled)
        """
        delP = InducerFrictionLoss.darcy_weisbach(geom, fluid_state, velocity)
        return {'friction': delP}
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, velocities, mass_flow, beta_flow):
        """
        Compute impeller losses - RadComp/Schiffmann model
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet (station 2)
            outlet_state: FluidState at outlet (station 4)
            velocities: dict with keys:
                - 'w_in': Relative velocity at inlet (m/s)
                - 'w_out': Relative velocity at outlet (m/s)
                - 'c_in': Absolute velocity at inlet (m/s)
                - 'c_out': Absolute velocity at outlet (m/s)
                - 'c_out_t': Tangential absolute velocity at outlet (m/s)
                - 'alpha_out': Flow angle at outlet (deg)
            mass_flow: Mass flow rate (kg/s)
            beta_flow: Flow angle in relative frame (deg)
        
        Returns:
            dict: RAW loss values (J/kg) before scaling
        """
        losses = {}
        
        # 1. Incidence (Stanitz-Galvas) - Returns J/kg
        # dh_inc = 0.5*(w*sin(|β_flow - β_opt|))²
        losses['incidence'] = ImpellerIncidenceLoss.stanitz_galvas_method(
            geom, velocities, beta_flow
        )
        
        # 2. Skin friction (Jansen - Coppage/Galvas variant)
        # Returns J/kg: 4*Cf*L_h*w_avg²/(2*D_h)
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 3. Blade loading (Coppage-Rodgers)
        # Returns J/kg: 0.05*Df²*u_out²
        # Note: Df calculated with Coppage formula including aerodynamic term
        losses['blade_loading'] = ImpellerBladeLoadingLoss.oh_coppage_method(
            geom, inlet_state, outlet_state, velocities,
            enthalpy_rise=(outlet_state.H - inlet_state.H)
        )
        
        # 4. Clearance (Jansen-Brasz)
        # Returns J/kg: 0.6*t_cl/b*c_t/u*sqrt(...)*u²
        losses['clearance'] = ImpellerClearanceLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 5. Disc friction (Daily & Nece)
        # Returns J/kg: Kf*ρ*n*r³/m*u²
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 6. Recirculation (Coppage)
        # Returns J/kg: 0.02*Df²*tan(α)*u²
        losses['recirculation'] = ImpellerRecirculationLoss.coppage_method(
            geom, velocities
        )
        
        return losses
    
    def compute_vaneless_diffuser_losses(self, geom, inlet_state, outlet_state, n_steps=15):
        """
        Vaneless diffuser - Japikse method (stepwise integration)
        
        Returns dp0_total in Pa (NOT scaled)
        
        Implementation details:
        - 15 radial steps from r4 to r5
        - Friction: Cf = k*(1.8e5/Re)^0.2, k=0.02
        - Hydraulic diameter: Dh = sqrt(8*r*b*blockage)
        - dp0 = 4*Cf*ds*c²*ρ/(2*Dh)
        """
        dp0_total = VanelessDiffuserLoss.japikse_method(
            geom, inlet_state, outlet_state, n_steps
        )
        return {'loss_total': dp0_total}

# ============================================================================
# MERONI MODEL - ✅ BASED ON TABLE 4 (Meroni et al.)
# ============================================================================

class MeroniLossModel(BaseLossModel):
    """
    Meroni et al. loss model
    
    Based on Table 4 loss correlations (Meroni paper)
    
    Impeller losses (7 total):
    1. Incidence: Galvas (Stanitz-Galvas, dh_inc = 0.5*w²*sin²|β_i,opt-β_i|)
    2. Friction: Galvas (4*Cf*L_b/d_hb*W̄²)
    3. Blade loading: Coppage (Δh_bl = 0.05*Df²*U²)
    4. Clearance: Jansen (Δh_cl = 0.6*ε*C_2t*sqrt(...))
    5. Mixing: Johnston and Dean (Δh_mix = 1/(1+tan²(α₂))*((...))²*c₂²/2)
    6. Disc friction: Daily and Nece (Δh_df = 0.25*c_d*U3*r₁²/ṁ)
    7. Recirculation: Oh et al. (Δh_rc = 8×10⁻⁵*sinh(3.5*α₂³)*Df²*U²)
    
    Vaned diffuser losses (2 total):
    1. Incidence: Conrad (Δh_inc,rot = 0.6*sin²|β₃-β₃,opt|*0.5*W₃²)
    2. Friction: Conrad (Δh_fvd,sf = 2*C_fvd*C²m,vd*L_b,vd/d_hb,vd)
    
    Reference: Meroni et al. paper, Table 4
    """
    
    def __init__(self):
        super().__init__("Meroni")
    
    def compute_impeller_losses(self, geom, inlet_state, outlet_state, velocities, mass_flow, enthalpy_rise, beta_flow):
        """
        Compute impeller losses - Meroni model
        
        Args:
            geom: Geometry object
            inlet_state: FluidState at inlet
            outlet_state: FluidState at outlet
            velocities: dict with keys:
                - 'w_in', 'w_out': Relative velocities
                - 'v_in', 'v_out', 'v_t_out': Absolute velocities
                - 'u_out': Blade speed
                - 'alpha_out': Flow angle
                - 'wake_width': Optional (default 0.05)
            mass_flow: Mass flow rate (kg/s)
            enthalpy_rise: h0_out - h0_in (J/kg)
            beta_flow: Flow angle in relative frame (deg)
        
        Returns:
            dict: RAW loss values (J/kg or m²/s²) before scaling
        """
        losses = {}
        
        # 1. Incidence (Galvas - Stanitz-Galvas method)
        # dh_inc = 0.5*w²*sin²(|β_i,opt - β_i|)
        losses['incidence'] = ImpellerIncidenceLoss.stanitz_galvas_method(
            geom, velocities, beta_flow
        )
        
        # 2. Friction (Galvas - via Jansen method which uses Coppage/Galvas)
        # Δh_sf = 4*Cf*L_b/d_hb*W̄²
        losses['skin_friction'] = ImpellerSkinFrictionLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 3. Blade loading (Coppage)
        # Δh_bl = 0.05*Df²*U_out²
        losses['blade_loading'] = ImpellerBladeLoadingLoss.oh_coppage_method(
            geom, inlet_state, outlet_state, velocities, enthalpy_rise
        )
        
        # 4. Clearance (Jansen)
        # Δh_cl = 0.6*ε*C_2t*sqrt(4π/(b₂*z)*α*(r₂-r₁s)/(1+ρ₂/ρ₁)*C₂t*C₁)
        losses['clearance'] = ImpellerClearanceLoss.jansen_method(
            geom, inlet_state, outlet_state, velocities
        )
        
        # 5. Mixing (Johnston and Dean)
        # Δh_mix = 1/(1+tan²(α₂))*((1-ε_w-b*)/(1-ε_w))²*0.5*c₂²
        wake_width = velocities.get('wake_width', 0.05)
        losses['mixing'] = ImpellerMixingLoss.johnston_dean_method(
            geom, velocities, wake_width
        )
        
        # 6. Disc friction (Daily and Nece)
        # Δh_df = 0.25*c_d*U²*ρ_i*r₁²/ṁ where ρ_i = (ρ₁ + ρ₂)/2
        losses['disc_friction'] = ImpellerDiscFrictionLoss.daily_nece_method(
            geom, inlet_state, outlet_state, velocities, mass_flow
        )
        
        # 7. Recirculation (Oh et al.)
        # Δh_rc = 8×10⁻⁵*sinh(3.5*α₂³)*Df²*U²
        losses['recirculation'] = ImpellerRecirculationLoss.oh_method(
            geom, inlet_state, outlet_state, velocities, enthalpy_rise
        )
        
        return losses
    
    def compute_vaned_diffuser_losses(self, geom, inlet_state, outlet_state, velocities, Cf=0.005):
        """
        Vaned diffuser losses - Conrad method (Meroni)
        
        Returns dict with 'incidence', 'skin_friction', 'loss_total'
        """
        return VanedDiffuserLoss.conrad_method(
            geom, inlet_state, outlet_state, velocities, Cf
        )

# ============================================================================
# MODEL REGISTRY
# ============================================================================

AVAILABLE_MODELS = {
    'oh': OhLossModel,
    'zhang_set1': ZhangSet1LossModel,
    'zhang_set2': ZhangSet2LossModel,
    'zhang_set3': ZhangSet3LossModel,
    'schiffmann': SchiffmannLossModel,
    'meroni': MeroniLossModel,
}

def get_model(model_name: str) -> BaseLossModel:
    """
    Factory function to get loss model instance
    
    Args:
        model_name: Currently only 'oh' available
    
    Returns:
        Loss model instance
    
    Raises:
        ValueError: If model_name not recognized
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not available. Choose from: {available}")
    
    return AVAILABLE_MODELS[model_name]()
