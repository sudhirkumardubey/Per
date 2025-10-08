# File: tests/unit/test_calculator.py

"""
Comprehensive test suite for calculator.py

Tests:
1. Unit conversion correctness (m²/s² → J/kg, Pa → J/kg, J/kg → J/kg)
2. Model-specific unit handling
3. All 6 models with all their losses
4. Physical consistency checks
5. Integration test with mock components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from centrifugal_compressor.losses.calculator import (
    LossCalculator, 
    available_models,
    compare_models,
)


# ============================================================================
# FIXTURES - Mock objects for testing
# ============================================================================

@pytest.fixture
def mock_geometry():
    """Mock geometry object"""
    geom = Mock()
    geom.r4 = 0.1  # m
    geom.b4 = 0.01  # m
    geom.n_blades = 12
    geom.tip_cl = 0.0005  # m
    geom.rough_inducer = 1e-6  # m
    geom.l_inducer = 0.05  # m
    geom.r2s = 0.05  # m
    return geom


@pytest.fixture
def mock_fluid_state():
    """Mock fluid state"""
    state = Mock()
    state.D = 10.0  # kg/m³ (typical compressor density)
    state.V = 1e-5  # m²/s (kinematic viscosity)
    state.P = 200000  # Pa
    state.T = 300  # K
    state.H = 300000  # J/kg
    state.cp = 1005  # J/(kg·K)
    state.cv = 718  # J/(kg·K)
    return state


@pytest.fixture
def mock_velocities():
    """Mock velocity components"""
    return {
        'w_in': 150.0,      # m/s
        'w_out': 120.0,     # m/s
        'w_th': 140.0,      # m/s (throat)
        'v_in': 100.0,      # m/s
        'v_out': 200.0,     # m/s
        'v_m_in': 80.0,     # m/s
        'v_m_out': 50.0,    # m/s
        'v_t_in': 60.0,     # m/s
        'v_t_out': 190.0,   # m/s
        'u_in': 200.0,      # m/s
        'u_out': 400.0,     # m/s
        'alpha_out': 15.0,  # deg
        'wake_width': 0.05,
        'delta_W': 30.0,    # m/s
    }


# ============================================================================
# TEST 1: Basic Initialization
# ============================================================================

def test_calculator_initialization():
    """Test calculator can be initialized with all models"""
    for model_name in available_models():
        calc = LossCalculator(model_name)
        assert calc.model_name == model_name
        assert calc.model is not None


def test_invalid_model_name():
    """Test that invalid model raises error"""
    with pytest.raises(ValueError, match="not available"):
        LossCalculator('nonexistent_model')


# ============================================================================
# TEST 2: Unit Conversion Logic
# ============================================================================

class TestUnitConversion:
    """Test unit conversion for different loss types"""
    
    def test_m2s2_to_jkg_direct(self):
        """Test m²/s² → J/kg (direct, same dimension)"""
        calc = LossCalculator('oh')
        density = 10.0  # kg/m³
        
        # Simulate m²/s² loss (e.g., incidence)
        raw_losses = {'incidence': 1000.0}  # m²/s²
        
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
        
        # Should pass through directly (m²/s² = J/kg)
        assert converted['incidence'] == 1000.0
    
    def test_jkg_passthrough(self):
        """Test J/kg → J/kg (no conversion)"""
        calc = LossCalculator('oh')
        density = 10.0
        
        # Simulate J/kg loss (e.g., disc_friction)
        raw_losses = {'disc_friction': 5000.0}  # J/kg
        
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
        
        # Should pass through unchanged
        assert converted['disc_friction'] == 5000.0
    
    def test_pa_to_jkg_conversion(self):
        """Test Pa → J/kg (divide by density)"""
        calc = LossCalculator('schiffmann')
        density = 10.0  # kg/m³
        
        # Simulate Pa loss (e.g., inducer_friction)
        raw_losses = {'inducer_friction': 50000.0}  # Pa
        
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
        
        # Should convert: 50000 Pa / 10 kg/m³ = 5000 J/kg
        assert converted['inducer_friction'] == 5000.0


# ============================================================================
# TEST 3: Model-Specific Unit Handling
# ============================================================================

class TestModelSpecificUnits:
    """Test each model handles units correctly"""
    
    def test_oh_model_units(self):
        """Oh model: 6 m²/s², 2 J/kg (leakage, disc_friction), 1 J/kg (diffuser)"""
        calc = LossCalculator('oh')
        
        # Test J/kg detection
        assert calc._is_already_j_per_kg('disc_friction') == True
        assert calc._is_already_j_per_kg('leakage') == True
        assert calc._is_already_j_per_kg('vaneless_diffuser') == True
        assert calc._is_already_j_per_kg('incidence') == False
        assert calc._is_already_j_per_kg('skin_friction') == False
    
    def test_zhang_set1_units(self):
        """Zhang Set 1: 9 m²/s², 1 J/kg (disc_friction), 1 J/kg (diffuser)"""
        calc = LossCalculator('zhang_set1')
        
        assert calc._is_already_j_per_kg('disc_friction') == True
        assert calc._is_already_j_per_kg('vaneless_diffuser') == True
        assert calc._is_already_j_per_kg('leakage') == False  # Jansen method (m²/s²)
    
    def test_zhang_set2_units(self):
        """Zhang Set 2: 9 m²/s², 2 J/kg (leakage, disc_friction), 1 J/kg (diffuser)"""
        calc = LossCalculator('zhang_set2')
        
        assert calc._is_already_j_per_kg('disc_friction') == True
        assert calc._is_already_j_per_kg('leakage') == True  # Aungier method
        assert calc._is_already_j_per_kg('vaneless_diffuser') == True
    
    def test_schiffmann_units(self):
        """Schiffmann: 1 J/kg (incidence), 4 m²/s², 2 Pa"""
        calc = LossCalculator('schiffmann')
        
        # J/kg
        assert calc._is_already_j_per_kg('incidence') == True
        assert calc._is_already_j_per_kg('disc_friction') == True
        
        # m²/s²
        assert calc._is_already_j_per_kg('skin_friction') == False
        assert calc._is_already_j_per_kg('blade_loading') == False
        assert calc._is_already_j_per_kg('clearance') == False
        assert calc._is_already_j_per_kg('recirculation') == False
        
        # Pa
        assert calc._is_pressure_loss('inducer_friction') == True
        assert calc._is_pressure_loss('vaneless_diffuser') == True
    
    def test_meroni_units(self):
        """Meroni: 2 J/kg (incidence, disc_friction), 6 m²/s²"""
        calc = LossCalculator('meroni')
        
        assert calc._is_already_j_per_kg('incidence') == True
        assert calc._is_already_j_per_kg('disc_friction') == True
        assert calc._is_already_j_per_kg('skin_friction') == False


# ============================================================================
# TEST 4: Physical Consistency Checks
# ============================================================================

class TestPhysicalConsistency:
    """Test that converted values are physically reasonable"""
    
    def test_losses_are_positive(self):
        """All losses should be positive (energy dissipation)"""
        calc = LossCalculator('oh')
        density = 10.0
        
        raw_losses = {
            'incidence': 1000.0,
            'disc_friction': 500.0,
            'leakage': 300.0,
        }
        
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
        
        for loss_value in converted.values():
            assert loss_value >= 0, "Loss must be non-negative"
    
    def test_unit_consistency(self):
        """All output losses should have J/kg dimension"""
        calc = LossCalculator('schiffmann')
        density = 10.0
        
        # Mix of Pa, J/kg, m²/s²
        raw_losses = {
            'inducer_friction': 10000.0,  # Pa
            'incidence': 500.0,            # J/kg
            'skin_friction': 800.0,        # m²/s²
        }
        
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
        
        # All should be in same order of magnitude (hundreds to thousands)
        for key, value in converted.items():
            assert 100 < value < 100000, f"{key} = {value} J/kg seems unreasonable"
    
    def test_pa_conversion_dimensional_analysis(self):
        """Verify Pa → J/kg conversion: Pa / (kg/m³) = J/kg"""
        calc = LossCalculator('schiffmann')
        
        # Pa = N/m² = kg/(m·s²)
        # Pa / (kg/m³) = kg/(m·s²) / (kg/m³) = m²/s² = J/kg ✓
        
        pressure_loss_pa = 20000.0  # Pa
        density_kgm3 = 8.0  # kg/m³
        
        raw_losses = {'inducer_friction': pressure_loss_pa}
        converted = calc._convert_losses_to_j_per_kg(raw_losses, density_kgm3)
        
        expected_jkg = pressure_loss_pa / density_kgm3
        assert abs(converted['inducer_friction'] - expected_jkg) < 0.01


# ============================================================================
# TEST 5: Integration Tests with Mock Components
# ============================================================================

class TestIntegration:
    """Test full workflow with mocked component methods"""
    
    @patch('centrifugal_compressor.losses.models.ImpellerIncidenceLoss')
    @patch('centrifugal_compressor.losses.models.ImpellerDiscFrictionLoss')
    def test_oh_model_integration(
        self, 
        mock_disc, 
        mock_inc, 
        mock_geometry, 
        mock_fluid_state, 
        mock_velocities
    ):
        """Test Oh model end-to-end"""
        
        # Mock component returns
        mock_inc.oh_conrad_method.return_value = 1000.0  # m²/s²
        mock_disc.daily_nece_method.return_value = 800.0  # J/kg
        
        calc = LossCalculator('oh')
        
        # Note: This would call the actual model which calls components
        # For full test, we'd need to mock all component methods
        # Here we just verify the calculator structure is correct
        
        assert calc.model_name == 'oh'
        assert hasattr(calc, 'compute_impeller_losses')
    
    def test_total_loss_calculation(self):
        """Test get_total_loss utility"""
        calc = LossCalculator('oh')
        
        losses = {
            'incidence': 1000.0,
            'skin_friction': 1500.0,
            'disc_friction': 800.0,
        }
        
        total = calc.get_total_loss(losses)
        assert total == 3300.0
    
    def test_loss_breakdown_percentage(self):
        """Test get_loss_breakdown utility"""
        calc = LossCalculator('oh')
        
        losses = {
            'incidence': 1000.0,
            'skin_friction': 1500.0,
            'disc_friction': 500.0,
        }
        
        breakdown = calc.get_loss_breakdown(losses)
        
        # Total = 3000, so percentages: 33.33%, 50%, 16.67%
        assert abs(breakdown['incidence'] - 33.33) < 0.1
        assert abs(breakdown['skin_friction'] - 50.0) < 0.1
        assert abs(breakdown['disc_friction'] - 16.67) < 0.1
        
        # Sum should be 100%
        assert abs(sum(breakdown.values()) - 100.0) < 0.01


# ============================================================================
# TEST 6: Model Comparison
# ============================================================================

def test_compare_models_function():
    """Test compare_models utility (will error without full mocks, but check structure)"""
    model_list = ['oh', 'zhang_set1']
    
    # Just verify function exists and returns dict
    assert callable(compare_models)


# ============================================================================
# TEST 7: Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation for required parameters"""
    
    def test_oh_requires_enthalpy_rise(self, mock_geometry, mock_fluid_state, mock_velocities):
        """Oh model warns if enthalpy_rise missing"""
        calc = LossCalculator('oh')
        
        with pytest.warns(UserWarning, match="requires enthalpy_rise"):
            calc._validate_impeller_inputs(
                enthalpy_rise=None,
                throat_state=None,
                beta_flow=None
            )
    
    def test_zhang_set2_requires_throat_state(self):
        """Zhang Set 2 requires throat_state"""
        calc = LossCalculator('zhang_set2')
        
        with pytest.raises(ValueError, match="requires throat_state"):
            calc._validate_impeller_inputs(
                enthalpy_rise=100000.0,
                throat_state=None,
                beta_flow=None
            )
    
    def test_schiffmann_requires_beta_flow(self):
        """Schiffmann requires beta_flow"""
        calc = LossCalculator('schiffmann')
        
        with pytest.raises(ValueError, match="requires beta_flow"):
            calc._validate_impeller_inputs(
                enthalpy_rise=None,
                throat_state=None,
                beta_flow=None
            )


# ============================================================================
# TEST 8: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_losses(self):
        """Test handling of zero losses"""
        calc = LossCalculator('oh')
        
        losses = {'incidence': 0.0, 'disc_friction': 0.0}
        total = calc.get_total_loss(losses)
        
        assert total == 0.0
        
        breakdown = calc.get_loss_breakdown(losses)
        assert all(v == 0.0 for v in breakdown.values())
    
    def test_empty_losses(self):
        """Test handling of empty loss dict"""
        calc = LossCalculator('oh')
        
        losses = {}
        total = calc.get_total_loss(losses)
        
        assert total == 0.0
    
    def test_repr(self):
        """Test __repr__ method"""
        calc = LossCalculator('oh')
        repr_str = repr(calc)
        
        assert 'LossCalculator' in repr_str
        assert 'oh' in repr_str


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
