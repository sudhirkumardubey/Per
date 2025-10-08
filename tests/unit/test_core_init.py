# File: tests/unit/test_core_init.py (CORRECTED VERSION)

"""Quick test to verify core/__init__.py works"""

def test_imports():
    """Test all imports from core package"""
    
    print("Testing core package imports...")
    
    # Test 1: Trigonometric functions
    from centrifugal_compressor.core import cosd, sind, tand
    assert abs(cosd(0) - 1.0) < 1e-10
    assert abs(sind(90) - 1.0) < 1e-10
    assert abs(tand(45) - 1.0) < 1e-10
    print("✅ Trig functions: PASS")
    
    # Test 2: Geometry
    from centrifugal_compressor.core import Geometry, OperatingCondition
    geom = Geometry(
        r1=0.020, r2s=0.025, r2h=0.010,
        beta2=-60, beta2s=-65, alpha2=0,
        r4=0.050, b4=0.004, beta4=-50,
        r5=0.070, b5=0.004,
        beta5=70, r6=0.090, b6=0.004, beta6=45,
        n_blades=12, n_splits=0, n_vanes=18,
        blade_le=0.001, blade_te=0.0005,
        tip_cl=0.0001, back_cl=0.0001,
        rough_inducer=1.5e-6,
        l_inducer=0.01, l_comp=0.05,
        blockage=[0.95, 0.95, 0.90, 0.95, 0.95, 0.95],
    )
    assert geom.r4 == 0.050
    print("✅ Geometry: PASS")
    
    # Test 3: Operating condition
    op = OperatingCondition.from_rpm(
        mass_flow=0.5,
        rpm=50000,
        P_inlet=101325,
        T_inlet=300,
        fluid_name='Air'
    )
    assert abs(op.rpm - 50000) < 0.1
    print("✅ OperatingCondition: PASS")
    
    # Test 4: Thermodynamics
    from centrifugal_compressor.core import Fluid, FluidState
    fluid = Fluid('Air')
    state = fluid.thermo_prop('PT', 101325, 300)
    
    # ✅ FIXED: is_valid is a property, not a method!
    assert state.is_valid  # No parentheses!
    assert 1.0 < state.D < 2.0  # Air density at standard conditions
    print("✅ Thermodynamics: PASS")
    
    # Test 5: Static/Total conversions
    from centrifugal_compressor.core import static_from_total, total_from_static
    total = fluid.thermo_prop('PT', 101325, 300)
    static = static_from_total(total, velocity=100.0)
    assert static.P < total.P  # Static pressure should be lower
    
    # Round trip test
    total_back = total_from_static(static, velocity=100.0)
    assert abs(total_back.P - total.P) / total.P < 0.01  # Within 1%
    print("✅ Static/Total conversions: PASS")
    
    # Test 6: Correlations
    from centrifugal_compressor.core import moody, japikse_friction
    Cf = moody(Re=10000, r=0.001)
    assert Cf > 0
    Cf_japikse = japikse_friction(Re=10000)
    assert Cf_japikse > 0
    print("✅ Correlations: PASS")
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("FILE 1 (core/__init__.py) is working correctly!")


if __name__ == '__main__':
    test_imports()
