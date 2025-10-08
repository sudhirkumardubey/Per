# File: tests/unit/test_correlations.py
"""
Unit tests for correlations module
Tests EXACT RadComp correlations.py implementation
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from centrifugal_compressor.core.correlations import (
    moody,
    japikse_friction,
)


class TestMoody:
    """Test Moody friction coefficient (EXACT RadComp implementation)"""
    
    def test_laminar_returns_16_over_Re(self):
        """Test laminar flow returns 16/Re (not 64/Re)"""
        Re = 2000
        r = 0.001
        f = moody(Re, r)
        
        # RadComp returns 64/Re/4 = 16/Re for laminar
        expected = 16.0 / Re
        assert f == pytest.approx(expected)
    
    def test_laminar_threshold(self):
        """Test transition at Re = 2300"""
        r = 0.001
        
        # Just below threshold - should use laminar
        Re_lam = 2299
        f_lam = moody(Re_lam, r)
        expected_lam = 16.0 / Re_lam
        assert f_lam == pytest.approx(expected_lam)
        
        # Just above threshold - should use Colebrook
        Re_turb = 2301
        f_turb = moody(Re_turb, r)
        # Should NOT equal 16/Re
        assert f_turb != pytest.approx(16.0 / Re_turb)
    
    def test_turbulent_smooth_pipe(self):
        """Test turbulent flow with smooth pipe"""
        Re = 100000
        r = 0.0  # Smooth pipe
        f = moody(Re, r)
        
        # Should be positive and reasonable
        # For smooth pipe, f/4 ≈ 0.004 for Re=100000
        assert 0.002 < f < 0.01
    
    def test_turbulent_rough_pipe(self):
        """Test turbulent flow with rough pipe"""
        Re = 100000
        r_smooth = 0.0
        r_rough = 0.01
        
        f_smooth = moody(Re, r_smooth)
        f_rough = moody(Re, r_rough)
        
        # Rough pipe should have higher friction
        assert f_rough > f_smooth
    
    def test_reynolds_effect_turbulent(self):
        """Test Reynolds number effect in turbulent regime"""
        r = 0.001
        Re_low = 10000
        Re_high = 100000
        
        f_low = moody(Re_low, r)
        f_high = moody(Re_high, r)
        
        # Higher Re should give lower friction
        assert f_high < f_low
    
    def test_returns_f_divided_by_4(self):
        """Test that result is f/4, not f"""
        Re = 100000
        r = 0.001
        
        f_quarter = moody(Re, r)
        
        # The full Darcy friction factor would be 4 times this
        # For Re=100000, smooth pipe, Darcy f ≈ 0.018
        # So f/4 ≈ 0.0045
        assert 0.003 < f_quarter < 0.006


class TestJapikseFriction:
    """Test Japikse friction coefficient (from diffuser.py)"""
    
    def test_default_k(self):
        """Test with default k=0.02"""
        Re = 180000  # Close to reference value 1.8e5
        Cf = japikse_friction(Re)
        
        # At Re = 1.8e5, should be close to k
        assert Cf == pytest.approx(0.02, rel=0.01)
    
    def test_reynolds_dependence(self):
        """Test Reynolds number dependence"""
        Re1 = 90000
        Re2 = 180000
        
        Cf1 = japikse_friction(Re1)
        Cf2 = japikse_friction(Re2)
        
        # Cf decreases with Re
        assert Cf1 > Cf2
    
    def test_custom_k(self):
        """Test with custom k value"""
        Re = 100000
        k_custom = 0.03
        
        Cf_default = japikse_friction(Re, k=0.02)
        Cf_custom = japikse_friction(Re, k=k_custom)
        
        # Should scale linearly with k
        assert Cf_custom / Cf_default == pytest.approx(k_custom / 0.02)
    
    def test_typical_range(self):
        """Test that Cf is in typical range"""
        Re = 100000
        Cf = japikse_friction(Re)
        
        # Typical range for skin friction coefficient
        assert 0.001 < Cf < 0.1
    
    def test_invalid_reynolds(self):
        """Test error for invalid Reynolds number"""
        with pytest.raises(ValueError, match="positive"):
            japikse_friction(-1000)
        
        with pytest.raises(ValueError, match="positive"):
            japikse_friction(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
