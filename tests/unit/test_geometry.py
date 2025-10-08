# File: tests/unit/test_geometry.py
"""
Unit tests for geometry module
Validates RadComp geometry structure and hydraulic diameter calculations
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from centrifugal_compressor.core.geometry import (
    Geometry,
    OperatingCondition,
    create_example_geometry,
    create_example_operating_condition,
    compare_hydraulic_diameter_methods,
    cosd,
    sind,
    tand,
)


class TestHelperFunctions:
    """Test trigonometric helper functions"""
    
    def test_cosd(self):
        """Test cosine in degrees"""
        assert cosd(0) == pytest.approx(1.0)
        assert cosd(90) == pytest.approx(0.0, abs=1e-10)
        assert cosd(180) == pytest.approx(-1.0)
        assert cosd(-60) == pytest.approx(0.5)
    
    def test_sind(self):
        """Test sine in degrees"""
        assert sind(0) == pytest.approx(0.0, abs=1e-10)
        assert sind(90) == pytest.approx(1.0)
        assert sind(180) == pytest.approx(0.0, abs=1e-10)
        assert sind(30) == pytest.approx(0.5)
    
    def test_tand(self):
        """Test tangent in degrees"""
        assert tand(0) == pytest.approx(0.0, abs=1e-10)
        assert tand(45) == pytest.approx(1.0)
        assert tand(-45) == pytest.approx(-1.0)


class TestGeometryCreation:
    """Test Geometry dataclass creation"""
    
    def test_valid_geometry_r1_equal_r2s(self):
        """Test geometry with r1 = r2s"""
        geom = Geometry(
            r1=0.025,
            r2s=0.025,
            r2h=0.010,
            beta2=-60,
            beta2s=-65,
            alpha2=0,
            r4=0.050,
            b4=0.004,
            beta4=-50,
            r5=0.070,
            b5=0.004,
            beta5=70,
            r6=0.090,
            b6=0.004,
            beta6=45,
            n_blades=12,
            n_splits=0,
            n_vanes=18,
            blade_le=0.001,
            blade_te=0.0005,
            tip_cl=0.0001,
            back_cl=0.0001,
            rough_inducer=1.5e-6,
            l_inducer=0.01,
            l_comp=0.05,
            blockage=[0.95, 0.95, 0.90, 0.95, 0.95, 0.95],
        )
        
        assert geom.r1 == 0.025
        assert geom.r1 == geom.r2s
        assert geom.n_blades == 12
    
    def test_valid_geometry_r1_greater_than_r2s(self):
        """Test geometry with r1 > r2s"""
        geom = Geometry(
            r1=0.030,
            r2s=0.025,
            r2h=0.010,
            beta2=-60,
            beta2s=-65,
            alpha2=0,
            r4=0.050,
            b4=0.004,
            beta4=-50,
            r5=0.070,
            b5=0.004,
            beta5=70,
            r6=0.090,
            b6=0.004,
            beta6=45,
            n_blades=12,
            n_splits=0,
            n_vanes=0,
            blade_le=0.001,
            blade_te=0.001,
            tip_cl=0.0,
            back_cl=0.0,
            rough_inducer=1.5e-6,
            l_inducer=0.01,
            l_comp=0.05,
            blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        
        assert geom.r1 > geom.r2s
        assert geom.r1 == 0.030
    
    def test_angles_in_degrees(self):
        """Test that angles are in degrees"""
        geom = create_example_geometry()
        
        assert geom.beta2 == -60
        assert geom.beta4 == -50
        assert geom.alpha2 == 0


class TestGeometryProperties:
    """Test calculated properties"""
    
    @pytest.fixture
    def geom(self):
        return create_example_geometry()
    
    def test_r2rms_calculation(self, geom):
        """Test root-mean-square radius"""
        expected = math.sqrt((geom.r2s**2 + geom.r2h**2) / 2.0)
        assert geom.r2rms == pytest.approx(expected)
        assert geom.r2h < geom.r2rms < geom.r2s
    
    def test_area_calculations(self, geom):
        """Test area calculations"""
        # A1_eff
        A1_expected = math.pi * geom.r1**2 * geom.blockage[0]
        assert geom.A1_eff == pytest.approx(A1_expected)
        
        # A2_eff
        A2_expected = (math.pi * (geom.r2s**2 - geom.r2h**2) * 
                      geom.blockage[1] * cosd(geom.alpha2))
        assert geom.A2_eff == pytest.approx(A2_expected)
        
        # A4_eff
        A4_expected = 2 * math.pi * geom.r4 * geom.b4 * geom.blockage[3]
        assert geom.A4_eff == pytest.approx(A4_expected)
        
        # A5_eff
        A5_expected = 2 * math.pi * geom.r5 * geom.b5 * geom.blockage[4]
        assert geom.A5_eff == pytest.approx(A5_expected)
    
    def test_slip_factor(self, geom):
        """Test slip factor"""
        slip = geom.slip
        
        assert 0 < slip < 1
        
        slip_basic = 1.0 - cosd(geom.beta4)**0.5 / (geom.n_blades + geom.n_splits) ** 0.7
        assert slip <= slip_basic


class TestHydraulicDiameterGalvas:
    """Test Galvas (RadComp original) hydraulic diameter"""
    
    @pytest.fixture
    def geom(self):
        return create_example_geometry()
    
    def test_galvas_returns_tuple(self, geom):
        """Test Galvas method returns (Dh, Lh) tuple"""
        result = geom.hydraulic_diameter_galvas
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        Dh, Lh = result
        assert Dh > 0
        assert Lh > 0
    
    def test_galvas_typical_values(self, geom):
        """Test Galvas gives typical values"""
        Dh, Lh = geom.hydraulic_diameter_galvas
        
        # Dh should be in reasonable range (1-20 mm)
        assert 0.001 < Dh < 0.020
        
        # Lh should be larger than Dh
        assert Lh > Dh
        
        # Lh should be reasonable (10-200 mm)
        assert 0.010 < Lh < 0.200


class TestHydraulicDiameterJansen:
    """Test Jansen (loss model standard) hydraulic diameter"""
    
    @pytest.fixture
    def geom(self):
        return create_example_geometry()
    
    def test_jansen_returns_tuple(self, geom):
        """Test Jansen method returns (Dh, Lb) tuple"""
        result = geom.hydraulic_diameter_jansen
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        Dh, Lb = result
        assert Dh > 0
        assert Lb > 0
    
    def test_jansen_typical_values(self, geom):
        """Test Jansen gives typical values"""
        Dh, Lb = geom.hydraulic_diameter_jansen
        
        # Dh should be in reasonable range (5-50 mm for typical compressors)
        assert 0.005 < Dh < 0.050
        
        # Lb should be in reasonable range (20-100 mm)
        assert 0.020 < Lb < 0.100
    
    def test_jansen_formula_components(self, geom):
        """Test Jansen formula components"""
        # Calculate manually
        Dh_outlet = (2 * geom.r4 * cosd(geom.beta4) /
                     (geom.n_blades / math.pi + 2 * geom.r4 * cosd(geom.beta4) / geom.b4))
        
        Dh_inlet = (0.5 * (geom.r2s + geom.r2h) / geom.r4 * cosd(geom.beta2) /
                    (geom.n_blades / math.pi +
                     (geom.r2s + geom.r2h) / (geom.r2s - geom.r2h) * cosd(geom.beta2)))
        
        Dh_expected = Dh_outlet + Dh_inlet
        
        Dh_actual, _ = geom.hydraulic_diameter_jansen
        
        assert Dh_actual == pytest.approx(Dh_expected, rel=1e-6)
    
    def test_jansen_L_ax_approximation(self, geom):
        """Test L_ax approximation in Jansen"""
        # L_ax should be approximately (r4 - r2s)/2 + b4
        L_ax_approx = 0.5 * (geom.r4 - geom.r2s) + geom.b4
        
        # This is what's used in the formula
        assert L_ax_approx > 0


class TestGetHydraulicParameters:
    """Test flexible getter method"""
    
    @pytest.fixture
    def geom(self):
        return create_example_geometry()
    
    def test_get_galvas_method(self, geom):
        """Test getting Galvas method"""
        Dh, Lh = geom.get_hydraulic_parameters(method='Galvas')
        Dh_direct, Lh_direct = geom.hydraulic_diameter_galvas
        
        assert Dh == pytest.approx(Dh_direct)
        assert Lh == pytest.approx(Lh_direct)
    
    def test_get_jansen_method(self, geom):
        """Test getting Jansen method"""
        Dh, Lb = geom.get_hydraulic_parameters(method='jansen')
        Dh_direct, Lb_direct = geom.hydraulic_diameter_jansen
        
        assert Dh == pytest.approx(Dh_direct)
        assert Lb == pytest.approx(Lb_direct)
    
    def test_default_is_jansen(self, geom):
        """Test default method is Jansen"""
        Dh_default, Lb_default = geom.get_hydraulic_parameters()
        Dh_jansen, Lb_jansen = geom.hydraulic_diameter_jansen
        
        assert Dh_default == pytest.approx(Dh_jansen)
        assert Lb_default == pytest.approx(Lb_jansen)
    
    def test_invalid_method_raises_error(self, geom):
        """Test invalid method raises ValueError"""
        with pytest.raises(ValueError, match="Unknown method"):
            geom.get_hydraulic_parameters(method='invalid')


class TestCompareHydraulicDiameterMethods:
    """Test comparison utility function"""
    
    def test_comparison_returns_dict(self):
        """Test comparison returns proper dictionary"""
        geom = create_example_geometry()
        result = compare_hydraulic_diameter_methods(geom)
        
        assert isinstance(result, dict)
        assert 'galvas' in result
        assert 'jansen' in result
    
    def test_comparison_has_correct_keys(self):
        """Test comparison dict has correct structure"""
        geom = create_example_geometry()
        result = compare_hydraulic_diameter_methods(geom)
        
        for method in ['galvas', 'jansen']:
            assert 'Dh_mm' in result[method]
            assert 'Lb_mm' in result[method]
            assert result[method]['Dh_mm'] > 0
            assert result[method]['Lb_mm'] > 0
    
    def test_galvas_vs_jansen_difference(self):
        """Test that Galvas and Jansen give different results"""
        geom = create_example_geometry()
        result = compare_hydraulic_diameter_methods(geom)
        
        # They should be significantly different
        Dh_diff = abs(result['galvas']['Dh_mm'] - result['jansen']['Dh_mm'])
        assert Dh_diff > 1.0  # At least 1mm difference


class TestOperatingCondition:
    """Test OperatingCondition"""
    
    def test_valid_operating_condition(self):
        """Test creation"""
        op = OperatingCondition(
            mass_flow=0.5,
            omega=5000,
            P_inlet=101325,
            T_inlet=300,
            fluid_name="Air"
        )
        
        assert op.mass_flow == 0.5
        assert op.omega == 5000
    
    def test_rpm_property(self):
        """Test RPM calculation"""
        omega = 5000
        op = OperatingCondition(
            mass_flow=0.5,
            omega=omega,
            P_inlet=101325,
            T_inlet=300,
        )
        
        rpm_expected = omega * 60 / (2 * math.pi)
        assert op.rpm == pytest.approx(rpm_expected)
    
    def test_from_rpm_constructor(self):
        """Test from RPM"""
        rpm = 50000
        op = OperatingCondition.from_rpm(
            mass_flow=0.5,
            rpm=rpm,
            P_inlet=101325,
            T_inlet=300,
        )
        
        assert op.rpm == pytest.approx(rpm)
        
        omega_expected = rpm * 2 * math.pi / 60
        assert op.omega == pytest.approx(omega_expected)


class TestFromDict:
    """Test dictionary construction"""
    
    def test_from_dict_with_blockage_list(self):
        """Test from_dict with blockage list"""
        data = {
            'r1': 0.020,
            'r2s': 0.025,
            'r2h': 0.010,
            'beta2': -60,
            'beta2s': -65,
            'alpha2': 0,
            'r4': 0.050,
            'b4': 0.004,
            'beta4': -50,
            'r5': 0.070,
            'b5': 0.004,
            'beta5': 70,
            'r6': 0.090,
            'b6': 0.004,
            'beta6': 45,
            'n_blades': 12,
            'n_splits': 0,
            'n_vanes': 18,
            'blade_le': 0.001,
            'blade_te': 0.0005,
            'tip_cl': 0.0001,
            'back_cl': 0.0001,
            'rough_inducer': 1.5e-6,
            'l_inducer': 0.01,
            'l_comp': 0.05,
        }
        blockage = [0.95, 0.95, 0.90, 0.95, 0.95, 0.95]
        
        geom = Geometry.from_dict(data, blockage=blockage)
        
        assert geom.r1 == 0.020
        assert geom.n_blades == 12
        assert geom.blockage == blockage


class TestExampleCreators:
    """Test example functions"""
    
    def test_example_geometry_valid(self):
        """Test example geometry"""
        geom = create_example_geometry()
        
        assert isinstance(geom, Geometry)
        assert geom.n_blades >= 3
        assert geom.r2h < geom.r2s
        assert len(geom.blockage) >= 4
    
    def test_example_operating_condition_valid(self):
        """Test example operating condition"""
        op = create_example_operating_condition()
        
        assert isinstance(op, OperatingCondition)
        assert op.mass_flow > 0
        assert op.rpm > 0


class TestRadiusRelationships:
    """Test r1 vs r2s relationships"""
    
    def test_r1_less_than_r2s(self):
        """r1 < r2s is valid"""
        geom = Geometry(
            r1=0.015,
            r2s=0.025,
            r2h=0.010,
            beta2=-60, beta2s=-65, alpha2=0,
            r4=0.050, b4=0.004, beta4=-50,
            r5=0.070, b5=0.004,
            beta5=70, r6=0.090, b6=0.004, beta6=45,
            n_blades=12, n_splits=0, n_vanes=0,
            blade_le=0.001, blade_te=0.001,
            tip_cl=0.0, back_cl=0.0, rough_inducer=1.5e-6,
            l_inducer=0.01, l_comp=0.05,
            blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        assert geom.r1 < geom.r2s
    
    def test_r1_equal_to_r2s(self):
        """r1 = r2s is valid"""
        geom = Geometry(
            r1=0.025,
            r2s=0.025,
            r2h=0.010,
            beta2=-60, beta2s=-65, alpha2=0,
            r4=0.050, b4=0.004, beta4=-50,
            r5=0.070, b5=0.004,
            beta5=70, r6=0.090, b6=0.004, beta6=45,
            n_blades=12, n_splits=0, n_vanes=0,
            blade_le=0.001, blade_te=0.001,
            tip_cl=0.0, back_cl=0.0, rough_inducer=1.5e-6,
            l_inducer=0.01, l_comp=0.05,
            blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        assert geom.r1 == geom.r2s
    
    def test_r1_greater_than_r2s(self):
        """r1 > r2s is valid"""
        geom = Geometry(
            r1=0.030,
            r2s=0.025,
            r2h=0.010,
            beta2=-60, beta2s=-65, alpha2=0,
            r4=0.050, b4=0.004, beta4=-50,
            r5=0.070, b5=0.004,
            beta5=70, r6=0.090, b6=0.004, beta6=45,
            n_blades=12, n_splits=0, n_vanes=0,
            blade_le=0.001, blade_te=0.001,
            tip_cl=0.0, back_cl=0.0, rough_inducer=1.5e-6,
            l_inducer=0.01, l_comp=0.05,
            blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        assert geom.r1 > geom.r2s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
