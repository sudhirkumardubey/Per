# File: tests/unit/test_thermodynamics.py
"""
Unit tests for thermodynamics module
Validates against RadComp implementation and CoolProp
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from centrifugal_compressor.core.thermodynamics import (
    FluidState,           # ✅ CHANGED from ThermalProperties
    Fluid,
    static_from_total,
    total_from_static,
    ThermoException       # ✅ CHANGED from ThermoPropertiesError
)


class TestFluidState:
    """Test FluidState dataclass structure"""
    
    def test_default_initialization(self):
        """Test that default values are NaN"""
        state = FluidState()
        
        assert math.isnan(state.P)
        assert math.isnan(state.T)
        assert math.isnan(state.D)
        assert math.isnan(state.H)
        assert math.isnan(state.S)
        assert math.isnan(state.A)
        assert math.isnan(state.V)
        assert state.phase == ""
        assert state.fluid is None
        assert not state.is_valid
    
    def test_explicit_initialization(self):
        """Test explicit property assignment"""
        state = FluidState(
            P=101325,
            T=300,
            D=1.2,
            H=300000,
            S=1000,
            A=340,
            V=1.8e-5
        )
        
        assert state.P == 101325
        assert state.T == 300
        assert state.is_valid
    
    def test_immutability(self):
        """Test that FluidState is immutable (frozen=True)"""
        state = FluidState(P=101325, T=300)
        
        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            state.P = 200000
    
    def test_is_valid_property(self):
        """Test validity check based on pressure"""
        state_invalid = FluidState()
        assert not state_invalid.is_valid
        
        state_valid = FluidState(P=101325, T=300)
        assert state_valid.is_valid


class TestFluid:
    """Test Fluid class"""
    
    def test_initialization_air(self):
        """Test Air fluid initialization"""
        air = Fluid("Air")
        assert air.name == "Air"
    
    def test_initialization_invalid_fluid(self):
        """Test error handling for invalid fluid"""
        with pytest.raises(ValueError, match="not available"):
            Fluid("InvalidFluid123")
    
    def test_thermo_prop_method_exists(self):
        """Verify method is 'thermo_prop'"""
        air = Fluid("Air")
        assert hasattr(air, "thermo_prop")
        assert callable(air.thermo_prop)


class TestThermoPropModes:
    """Test all thermo_prop calculation modes"""
    
    @pytest.fixture
    def air(self):
        return Fluid("Air")
    
    def test_pt_mode_standard_conditions(self, air):
        """Test PT mode at standard atmospheric conditions"""
        P = 101325  # Pa
        T = 288.15  # K (15°C)
        
        state = air.thermo_prop("PT", P, T)
        
        assert state.P == pytest.approx(P, rel=1e-10)
        assert state.T == pytest.approx(T, rel=1e-10)
        assert state.D == pytest.approx(1.225, rel=0.01)
        assert state.A == pytest.approx(340.3, rel=0.01)
        assert state.is_valid
        assert state.fluid == air  # ✅ NEW: Check fluid reference
        assert not math.isnan(state.H)
        assert not math.isnan(state.S)
        assert not math.isnan(state.V)
    
    def test_pt_mode_high_pressure(self, air):
        """Test PT mode at elevated pressure"""
        P = 10e5  # 10 bar
        T = 400   # K
        
        state = air.thermo_prop("PT", P, T)
        
        assert state.P == pytest.approx(P, rel=1e-10)
        assert state.T == pytest.approx(T, rel=1e-10)
        assert state.D > 1.225
        assert state.is_valid
        assert state.fluid == air  # ✅ NEW
    
    def test_ph_mode(self, air):
        """Test PH mode"""
        P = 101325
        T = 300
        
        ref = air.thermo_prop("PT", P, T)
        H_ref = ref.H
        
        state = air.thermo_prop("PH", P, H_ref)
        
        assert state.P == pytest.approx(P, rel=1e-10)
        assert state.H == pytest.approx(H_ref, rel=1e-10)
        assert state.T == pytest.approx(T, rel=1e-4)
        assert state.fluid == air  # ✅ NEW
    
    def test_ps_mode_isentropic_compression(self, air):
        """Test PS mode for isentropic compression"""
        P1 = 101325
        T1 = 300
        PR = 2.0
        
        state1 = air.thermo_prop("PT", P1, T1)
        S1 = state1.S
        
        P2 = P1 * PR
        state2 = air.thermo_prop("PS", P2, S1)
        
        assert state2.P == pytest.approx(P2, rel=1e-10)
        assert state2.S == pytest.approx(S1, rel=1e-6)
        assert state2.T > T1
        
        # Check isentropic relation
        gamma = 1.4
        T2_ideal = T1 * (PR)**((gamma-1)/gamma)
        assert state2.T == pytest.approx(T2_ideal, rel=0.01)
    
    def test_pd_mode(self, air):
        """Test PD mode"""
        P = 101325
        T = 300
        
        ref = air.thermo_prop("PT", P, T)
        D_ref = ref.D
        
        state = air.thermo_prop("PD", P, D_ref)
        
        assert state.P == pytest.approx(P, rel=1e-10)
        assert state.D == pytest.approx(D_ref, rel=1e-10)
        assert state.T == pytest.approx(T, rel=1e-4)
    
    def test_hs_mode(self, air):
        """Test HS mode (critical for static/total conversions)"""
        P = 101325
        T = 300
        
        ref = air.thermo_prop("PT", P, T)
        H_ref = ref.H
        S_ref = ref.S
        
        state = air.thermo_prop("HS", H_ref, S_ref)
        
        assert state.H == pytest.approx(H_ref, rel=1e-10)
        assert state.S == pytest.approx(S_ref, rel=1e-10)
        assert state.T == pytest.approx(T, rel=1e-4)
        assert state.P == pytest.approx(P, rel=1e-3)
    
    def test_invalid_mode(self, air):
        """Test error handling for invalid mode"""
        with pytest.raises(ThermoException, match="Unknown mode"):
            air.thermo_prop("XX", 101325, 300)
    
    def test_invalid_values(self, air):
        """Test error handling for invalid property values"""
        with pytest.raises(ThermoException):
            air.thermo_prop("PT", -1000, 300)


class TestStaticTotalConversion:
    """Test static/total conversions"""
    
    @pytest.fixture
    def air(self):
        return Fluid("Air")
    
    def test_static_from_total_zero_velocity(self, air):
        """Static equals total at zero velocity"""
        total = air.thermo_prop("PT", 101325, 300)
        static = static_from_total(total, 0.0)
        
        assert static.P == pytest.approx(total.P, rel=1e-6)
        assert static.T == pytest.approx(total.T, rel=1e-6)
        assert static.H == pytest.approx(total.H, rel=1e-6)
        assert static.S == pytest.approx(total.S, rel=1e-6)
    
    def test_static_from_total_subsonic(self, air):
        """Test static from total at subsonic speed"""
        total = air.thermo_prop("PT", 101325, 300)
        velocity = 100  # m/s
        
        static = static_from_total(total, velocity)
        
        assert static.P < total.P
        assert static.T < total.T
        assert static.H < total.H
        assert static.S == pytest.approx(total.S, rel=1e-6)
        
        # Energy equation
        h_diff = total.H - static.H
        assert h_diff == pytest.approx(0.5 * velocity**2, rel=1e-6)
    
    def test_total_from_static_subsonic(self, air):
        """Test total from static"""
        static = air.thermo_prop("PT", 101325, 300)
        velocity = 100  # m/s
        
        total = total_from_static(static, velocity)
        
        assert total.P > static.P
        assert total.T > static.T
        assert total.H > static.H
        assert total.S == pytest.approx(static.S, rel=1e-6)
        
        h_diff = total.H - static.H
        assert h_diff == pytest.approx(0.5 * velocity**2, rel=1e-6)
    
    def test_roundtrip_static_total_static(self, air):
        """Test roundtrip: static → total → static"""
        static_orig = air.thermo_prop("PT", 101325, 300)
        velocity = 50
        
        total = total_from_static(static_orig, velocity)
        static_calc = static_from_total(total, velocity)
        
        assert static_calc.P == pytest.approx(static_orig.P, rel=1e-4)
        assert static_calc.T == pytest.approx(static_orig.T, rel=1e-4)
        assert static_calc.H == pytest.approx(static_orig.H, rel=1e-6)
        assert static_calc.S == pytest.approx(static_orig.S, rel=1e-6)
    
    def test_missing_fluid_reference(self, air):
        """Test error when fluid reference is missing"""
        # Create state without fluid reference
        invalid = FluidState(P=101325, T=300, H=300000, S=1000)
        
        with pytest.raises(ThermoException, match="must have fluid"):
            static_from_total(invalid, 100)
    
    def test_invalid_total_state(self):
        """Test error for invalid total state"""
        invalid = FluidState()  # All NaN
        
        with pytest.raises(ThermoException, match="must be valid"):
            static_from_total(invalid, 100)


class TestDifferentFluids:
    """Test with different working fluids"""
    
    def test_co2(self):
        """Test CO2 fluid"""
        co2 = Fluid("CO2")
        assert co2.name == "CO2"
        
        state = co2.thermo_prop("PT", 101325, 300)
        assert state.is_valid
        assert state.D > 1.5  # CO2 denser than air
        assert state.fluid == co2
    
    def test_nitrogen(self):
        """Test Nitrogen fluid"""
        n2 = Fluid("Nitrogen")
        assert n2.name == "Nitrogen"
        
        state = n2.thermo_prop("PT", 101325, 300)
        assert state.is_valid
        assert state.fluid == n2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
