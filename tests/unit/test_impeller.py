"""
pytest unit tests for impeller.py

Uses real compressor test cases:
- Sandia: CO2 compressor
- Eckardt A: Air compressor

Run with:
    pytest test_impeller.py -v
    pytest test_impeller.py -v -k "sandia"  # Only Sandia tests
    pytest test_impeller.py -v -k "eckardt"  # Only Eckardt tests
"""

import pytest
import math

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.components.impeller import Impeller
from centrifugal_compressor.components.inducer import Inducer


# ============================================================================
# TEST DATA - Real Compressor Geometries
# ============================================================================

@pytest.fixture
def sandia_geometry():
    """Sandia CO2 compressor geometry"""
    return Geometry(
        r1=0.01,
        r2s=0.0094,
        r2h=0.00254,
        beta2=-45.0,
        beta2s=-50.0,
        alpha2=0.0,
        r4=0.01868,
        b4=0.00171,
        beta4=-50.0,
        r5=0.0191,
        b5=0.00171,
        beta5=70.0,  # Assumed for vaneless diffuser
        r6=0.025,    # Assumed (not in spec)
        b6=0.00171,  # Assumed
        beta6=45.0,  # Assumed
        n_blades=6,
        n_splits=6,
        n_vanes=12,  # Assumed
        blade_le=762.0e-6,  # blade_e in spec
        blade_te=500.0e-6,  # Assumed (not in spec)
        tip_cl=254.0e-6,    # clearance
        back_cl=254.0e-6,   # backface
        rough_inducer=1.0e-4,  # rug_ind
        l_inducer=0.02,
        l_comp=0.1137,
        blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # blockage1-5
    )


@pytest.fixture
def sandia_operating_condition():
    """Sandia operating conditions"""
    return OperatingCondition(
        mass_flow=2.15,      # Assumed (not specified)
        omega=5235.98,        # Assumed ~6000 RPM
        P_inlet=769.0e3,    # 769 kPa
        T_inlet=305.97,     # K
        fluid_name='CO2'
    )


@pytest.fixture
def eckardt_a_geometry():
    """Eckardt A air compressor geometry"""
    return Geometry(
        r1=0.14,
        r2s=0.14,
        r2h=0.06,
        beta2=-53.0,
        beta2s=-63.0,
        alpha2=0.0,
        r4=0.2,
        b4=0.0267,
        beta4=-30.0,
        r5=0.538,  # Large vaneless diffuser
        b5=0.0136,
        beta5=70.0,  # Assumed
        r6=0.60,     # Assumed (not in spec)
        b6=0.0136,   # Assumed
        beta6=45.0,  # Assumed
        n_blades=20,
        n_splits=0,
        n_vanes=30,  # Assumed
        blade_le=2.11e-3,  # blade_e
        blade_te=1.5e-3,   # Assumed
        tip_cl=213.0e-6,   # clearance
        back_cl=235.0e-6,  # backface
        rough_inducer=2.0e-5,  # rug_inducer (using rug_imp value)
        l_inducer=0.02,
        l_comp=0.13,
        blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )


@pytest.fixture
def eckardt_a_operating_condition():
    """Eckardt A operating conditions"""
    return OperatingCondition(
        mass_flow=5.32,      # Assumed (typical for this size)
        omega=1466.07,       # Assumed ~10000 RPM
        P_inlet=1.01e5,     # 1 atm
        T_inlet=288.15,     # K
        fluid_name='Air'
    )


# ============================================================================
# TEST CLASS 1: Sandia Compressor
# ============================================================================

class TestSandiaCompressor:
    """Test Sandia CO2 compressor"""
    
    @pytest.fixture
    def sandia_inducer(self, sandia_geometry, sandia_operating_condition):
        """Create Sandia inducer"""
        return Inducer(sandia_geometry, sandia_operating_condition)
    
    @pytest.fixture
    def sandia_impeller(self, sandia_geometry, sandia_operating_condition, sandia_inducer):
        """Create Sandia impeller"""
        return Impeller(
            sandia_geometry,
            sandia_operating_condition,
            sandia_inducer,
            loss_model='schiffmann'
        )
    
    def test_sandia_geometry_valid(self, sandia_geometry):
        """Test Sandia geometry is valid"""
        assert sandia_geometry.r1 > 0
        assert sandia_geometry.r2s > sandia_geometry.r2h
        assert sandia_geometry.r4 > sandia_geometry.r2s
        assert sandia_geometry.n_blades == 6
        assert sandia_geometry.n_splits == 6
        
        # Check computed r2rms
        r2rms = sandia_geometry.r2rms
        assert sandia_geometry.r2h < r2rms < sandia_geometry.r2s
    
    def test_sandia_operating_condition_valid(self, sandia_operating_condition):
        """Test Sandia operating condition is valid"""
        assert sandia_operating_condition.P_inlet == 769.0e3
        assert sandia_operating_condition.T_inlet == 305.97
        assert sandia_operating_condition.fluid_name == 'CO2'
    
    def test_sandia_impeller_creates(self, sandia_impeller):
        """Test Sandia impeller can be created"""
        assert sandia_impeller is not None
        assert sandia_impeller.in2 is not None
    
    def test_sandia_inlet_velocities(self, sandia_impeller):
        """Test Sandia inlet velocities are positive"""
        if sandia_impeller.choke_flag:
            pytest.skip("Flow choked")
        
        assert sandia_impeller.in2.v > 0
        assert sandia_impeller.in2.w > 0
        assert sandia_impeller.in2.u > 0
    
    def test_sandia_velocity_triangle(self, sandia_impeller):
        """Test Sandia velocity triangle closure"""
        if sandia_impeller.choke_flag:
            pytest.skip("Flow choked")
        
        in2 = sandia_impeller.in2
        w_calc = math.sqrt(in2.v_m**2 + (in2.u - in2.v_t)**2)
        rel_error = abs(in2.w - w_calc) / in2.w
        assert rel_error < 0.01, f"Triangle error {rel_error:.3%}"
    
    def test_sandia_pressure_rise(self, sandia_impeller):
        """Test Sandia pressure increases"""
        if sandia_impeller.choke_flag or sandia_impeller.wet:
            pytest.skip("Flow choked or wet")
        
        assert sandia_impeller.out.total.P > sandia_impeller.in2.total.P
    
    def test_sandia_efficiency(self, sandia_impeller):
        """Test Sandia efficiency in range"""
        if sandia_impeller.choke_flag or sandia_impeller.wet:
            pytest.skip("Flow choked or wet")
        
        assert 0 < sandia_impeller.eff_tt < 1.0
        assert 0.4 < sandia_impeller.eff_tt < 0.95


# ============================================================================
# TEST CLASS 2: Eckardt A Compressor
# ============================================================================

class TestEckardtACompressor:
    """Test Eckardt A air compressor"""
    
    @pytest.fixture
    def eckardt_inducer(self, eckardt_a_geometry, eckardt_a_operating_condition):
        """Create Eckardt A inducer"""
        return Inducer(eckardt_a_geometry, eckardt_a_operating_condition)
    
    @pytest.fixture
    def eckardt_impeller(self, eckardt_a_geometry, eckardt_a_operating_condition, eckardt_inducer):
        """Create Eckardt A impeller"""
        return Impeller(
            eckardt_a_geometry,
            eckardt_a_operating_condition,
            eckardt_inducer,
            loss_model='schiffmann'
        )
    
    def test_eckardt_geometry_valid(self, eckardt_a_geometry):
        """Test Eckardt A geometry is valid"""
        assert eckardt_a_geometry.r1 > 0
        assert eckardt_a_geometry.r2s == 0.14  # Equal radii at inlet
        assert eckardt_a_geometry.r2h == 0.06
        assert eckardt_a_geometry.r4 == 0.2
        assert eckardt_a_geometry.n_blades == 20
        assert eckardt_a_geometry.n_splits == 0
    
    def test_eckardt_operating_condition_valid(self, eckardt_a_operating_condition):
        """Test Eckardt A operating condition is valid"""
        assert eckardt_a_operating_condition.P_inlet == 1.01e5
        assert eckardt_a_operating_condition.T_inlet == 288.15
        assert eckardt_a_operating_condition.fluid_name == 'Air'
    
    def test_eckardt_impeller_creates(self, eckardt_impeller):
        """Test Eckardt A impeller can be created"""
        assert eckardt_impeller is not None
        assert eckardt_impeller.in2 is not None
    
    def test_eckardt_inlet_velocities(self, eckardt_impeller):
        """Test Eckardt A inlet velocities are positive"""
        if eckardt_impeller.choke_flag:
            pytest.skip("Flow choked")
        
        assert eckardt_impeller.in2.v > 0
        assert eckardt_impeller.in2.w > 0
        assert eckardt_impeller.in2.u > 0
    
    def test_eckardt_velocity_triangle(self, eckardt_impeller):
        """Test Eckardt A velocity triangle closure"""
        if eckardt_impeller.choke_flag:
            pytest.skip("Flow choked")
        
        in2 = eckardt_impeller.in2
        w_calc = math.sqrt(in2.v_m**2 + (in2.u - in2.v_t)**2)
        rel_error = abs(in2.w - w_calc) / in2.w
        assert rel_error < 0.01
    
    def test_eckardt_spanwise_ordering(self, eckardt_impeller):
        """Test Eckardt A spanwise velocity ordering"""
        if eckardt_impeller.choke_flag:
            pytest.skip("Flow choked")
        
        in2 = eckardt_impeller.in2
        assert in2.wh < in2.w < in2.ws
    
    def test_eckardt_pressure_rise(self, eckardt_impeller):
        """Test Eckardt A pressure increases"""
        if eckardt_impeller.choke_flag or eckardt_impeller.wet:
            pytest.skip("Flow choked or wet")
        
        assert eckardt_impeller.out.total.P > eckardt_impeller.in2.total.P
        pr = eckardt_impeller.out.total.P / eckardt_impeller.in2.total.P
        assert 1.1 < pr < 5.0
    
    def test_eckardt_efficiency(self, eckardt_impeller):
        """Test Eckardt A efficiency in range"""
        if eckardt_impeller.choke_flag or eckardt_impeller.wet:
            pytest.skip("Flow choked or wet")
        
        assert 0 < eckardt_impeller.eff_tt < 1.0
        assert 0.4 < eckardt_impeller.eff_tt < 0.96


# ============================================================================
# TEST CLASS 3: Comparison Tests
# ============================================================================

class TestCompressorComparison:
    """Compare Sandia and Eckardt compressors"""
    
    def test_sandia_vs_eckardt_size(self, sandia_geometry, eckardt_a_geometry):
        """Test that Eckardt A is much larger than Sandia"""
        # Eckardt A should be ~20x larger
        assert eckardt_a_geometry.r4 > 10 * sandia_geometry.r4
        assert eckardt_a_geometry.b4 > 10 * sandia_geometry.b4
    
    def test_sandia_vs_eckardt_blade_count(self, sandia_geometry, eckardt_a_geometry):
        """Test blade count differences"""
        # Sandia has fewer main blades but more splitters
        assert sandia_geometry.n_blades < eckardt_a_geometry.n_blades
        assert sandia_geometry.n_splits > eckardt_a_geometry.n_splits
        
        # Total blade count
        total_sandia = sandia_geometry.n_blades + sandia_geometry.n_splits
        total_eckardt = eckardt_a_geometry.n_blades + eckardt_a_geometry.n_splits
        assert total_sandia == 12
        assert total_eckardt == 20


# ============================================================================
# TEST CLASS 4: Loss Model Tests
# ============================================================================

class TestLossModelsWithRealGeometry:
    """Test all loss models with real geometries"""
    
    @pytest.mark.parametrize("loss_model", [
        'oh', 'schiffmann', 'meroni',
        'zhang_set1', 'zhang_set2', 'zhang_set3'
    ])
    def test_sandia_all_loss_models(self, sandia_geometry, sandia_operating_condition, loss_model):
        """Test all loss models with Sandia geometry"""
        inducer = Inducer(sandia_geometry, sandia_operating_condition)
        impeller = Impeller(
            sandia_geometry,
            sandia_operating_condition,
            inducer,
            loss_model=loss_model
        )
        
        # Should complete without exception
        assert impeller is not None
        
        if not (impeller.choke_flag or impeller.wet):
            assert impeller.losses.total >= 0
    
    @pytest.mark.parametrize("loss_model", [
        'oh', 'schiffmann', 'meroni',
        'zhang_set1', 'zhang_set2', 'zhang_set3'
    ])
    def test_eckardt_all_loss_models(self, eckardt_a_geometry, eckardt_a_operating_condition, loss_model):
        """Test all loss models with Eckardt A geometry"""
        inducer = Inducer(eckardt_a_geometry, eckardt_a_operating_condition)
        impeller = Impeller(
            eckardt_a_geometry,
            eckardt_a_operating_condition,
            inducer,
            loss_model=loss_model
        )
        
        # Should complete without exception
        assert impeller is not None
        
        if not (impeller.choke_flag or impeller.wet):
            assert impeller.losses.total >= 0


# ============================================================================
# TEST CLASS 5: Physics Validation
# ============================================================================

class TestPhysicsValidation:
    """Validate physics equations"""
    
    def test_euler_work_sandia(self, sandia_geometry, sandia_operating_condition):
        """Test Euler turbomachine equation for Sandia"""
        inducer = Inducer(sandia_geometry, sandia_operating_condition)
        impeller = Impeller(sandia_geometry, sandia_operating_condition, inducer, loss_model='schiffmann')
        
        if impeller.choke_flag or impeller.wet:
            pytest.skip("Flow choked or wet")
        
        dh_actual = impeller.out.total.H - impeller.in2.total.H
        dh_euler = (impeller.out.u * impeller.out.v_t - 
                   impeller.in2.u * impeller.in2.v_t)
        
        # Should be within 30% (losses, slip, etc.)
        assert abs(dh_actual - dh_euler) / dh_euler < 0.3
    
    def test_continuity_eckardt(self, eckardt_a_geometry, eckardt_a_operating_condition):
        """Test mass continuity for Eckardt A"""
        inducer = Inducer(eckardt_a_geometry, eckardt_a_operating_condition)
        impeller = Impeller(eckardt_a_geometry, eckardt_a_operating_condition, inducer, loss_model='schiffmann')
        
        if impeller.choke_flag or impeller.wet:
            pytest.skip("Flow choked or wet")
        
        # Check outlet continuity
        mass_calc = (impeller.out.static.D * 
                    eckardt_a_geometry.A4_eff * 
                    impeller.out.v_m)
        
        error = abs(mass_calc - eckardt_a_operating_condition.mass_flow) / \
                eckardt_a_operating_condition.mass_flow
        
        assert error < 0.02, f"Mass error {error:.3%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
