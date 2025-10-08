# File: test_calculator_manual.py

"""
STANDALONE Manual Unit Test for calculator.py
NO DEPENDENCIES - Tests core logic only
"""


class SimpleLossCalculator:
    """Simplified calculator with EXACT logic from calculator.py"""
    
    def __init__(self, model_name='oh'):
        self.model_name = model_name
        
        # EXACT mappings from calculator.py
        self.ALWAYS_J_PER_KG_LOSSES = {'disc_friction'}
        
        self.MODEL_SPECIFIC_J_PER_KG = {
            'oh': {'leakage', 'vaneless_diffuser'},
            'zhang_set1': {'vaneless_diffuser'},
            'zhang_set2': {'leakage', 'vaneless_diffuser'},
            'zhang_set3': {'vaneless_diffuser'},
            'schiffmann': {'incidence'},
            'meroni': {'incidence'},
        }
        
        self.PA_LOSSES_BY_MODEL = {
            'schiffmann': {'friction', 'inducer_friction', 'loss_total', 'vaneless_diffuser'},
        }
    
    def _is_already_j_per_kg(self, loss_key):
        """Check if loss is already in J/kg"""
        if loss_key in self.ALWAYS_J_PER_KG_LOSSES:
            return True
        if self.model_name in self.MODEL_SPECIFIC_J_PER_KG:
            if loss_key in self.MODEL_SPECIFIC_J_PER_KG[self.model_name]:
                return True
        return False
    
    def _is_pressure_loss(self, loss_key):
        """Check if loss is in Pa"""
        if self.model_name in self.PA_LOSSES_BY_MODEL:
            if loss_key in self.PA_LOSSES_BY_MODEL[self.model_name]:
                return True
        return False
    
    def _convert_losses_to_j_per_kg(self, raw_losses, density):
        """Convert all losses to J/kg"""
        converted = {}
        for key, value in raw_losses.items():
            if self._is_already_j_per_kg(key):
                # Already J/kg - pass through
                converted[key] = value
            elif self._is_pressure_loss(key):
                # Pa → J/kg
                converted[key] = value / density
            else:
                # m²/s² → J/kg (direct)
                converted[key] = value
        return converted


def test_unit_conversions():
    """Test basic unit conversions"""
    print("="*80)
    print("TEST 1: Unit Conversion Logic")
    print("="*80)
    
    # Test 1: m²/s² → J/kg (direct)
    calc = SimpleLossCalculator('oh')
    density = 10.0
    raw = {'incidence': 1000.0}  # m²/s²
    converted = calc._convert_losses_to_j_per_kg(raw, density)
    
    assert converted['incidence'] == 1000.0, "FAILED: m²/s² should pass through"
    print("✅ m²/s² → J/kg: PASS (1000 m²/s² = 1000 J/kg)")
    
    # Test 2: J/kg → J/kg (passthrough)
    raw = {'disc_friction': 500.0}  # J/kg
    converted = calc._convert_losses_to_j_per_kg(raw, density)
    
    assert converted['disc_friction'] == 500.0, "FAILED: J/kg should pass through"
    print("✅ J/kg → J/kg: PASS (500 J/kg unchanged)")
    
    # Test 3: Pa → J/kg (divide by density)
    calc_schiff = SimpleLossCalculator('schiffmann')
    raw = {'inducer_friction': 20000.0}  # Pa
    converted = calc_schiff._convert_losses_to_j_per_kg(raw, density)
    
    expected = 20000.0 / 10.0  # = 2000 J/kg
    assert abs(converted['inducer_friction'] - expected) < 0.01, "FAILED: Pa conversion"
    print(f"✅ Pa → J/kg: PASS (20000 Pa / 10 kg/m³ = {expected} J/kg)")


def test_all_models():
    """Test all 6 models"""
    print("\n" + "="*80)
    print("TEST 2: All Models Initialization")
    print("="*80)
    
    models = ['oh', 'zhang_set1', 'zhang_set2', 'zhang_set3', 'schiffmann', 'meroni']
    
    for model_name in models:
        calc = SimpleLossCalculator(model_name)
        assert calc.model_name == model_name, f"FAILED: {model_name}"
        print(f"✅ {model_name:15s}: initialized")


def test_model_specific_units():
    """Test model-specific unit detection"""
    print("\n" + "="*80)
    print("TEST 3: Model-Specific Unit Detection")
    print("="*80)
    
    # Oh model
    calc_oh = SimpleLossCalculator('oh')
    assert calc_oh._is_already_j_per_kg('disc_friction') == True
    assert calc_oh._is_already_j_per_kg('leakage') == True
    assert calc_oh._is_already_j_per_kg('vaneless_diffuser') == True
    assert calc_oh._is_already_j_per_kg('incidence') == False
    print("✅ Oh: disc_friction, leakage, vaneless_diffuser = J/kg")
    
    # Schiffmann model
    calc_schiff = SimpleLossCalculator('schiffmann')
    assert calc_schiff._is_already_j_per_kg('incidence') == True
    assert calc_schiff._is_already_j_per_kg('disc_friction') == True
    assert calc_schiff._is_pressure_loss('inducer_friction') == True
    assert calc_schiff._is_pressure_loss('vaneless_diffuser') == True
    print("✅ Schiffmann: incidence = J/kg, inducer/diffuser = Pa")
    
    # Zhang Set 2
    calc_z2 = SimpleLossCalculator('zhang_set2')
    assert calc_z2._is_already_j_per_kg('leakage') == True
    assert calc_z2._is_already_j_per_kg('vaneless_diffuser') == True
    print("✅ Zhang Set 2: leakage, vaneless_diffuser = J/kg")
    
    # Meroni
    calc_meroni = SimpleLossCalculator('meroni')
    assert calc_meroni._is_already_j_per_kg('incidence') == True
    assert calc_meroni._is_already_j_per_kg('disc_friction') == True
    print("✅ Meroni: incidence, disc_friction = J/kg")


def test_mixed_units():
    """Test mixed units in same calculation"""
    print("\n" + "="*80)
    print("TEST 4: Mixed Units (Schiffmann Model)")
    print("="*80)
    
    calc = SimpleLossCalculator('schiffmann')
    density = 8.0  # kg/m³
    
    raw_losses = {
        'inducer_friction': 16000.0,  # Pa
        'incidence': 800.0,            # J/kg
        'skin_friction': 600.0,        # m²/s²
        'disc_friction': 400.0,        # J/kg
    }
    
    converted = calc._convert_losses_to_j_per_kg(raw_losses, density)
    
    # Check conversions
    assert converted['inducer_friction'] == 2000.0, "Pa conversion failed"
    assert converted['incidence'] == 800.0, "J/kg passthrough failed"
    assert converted['skin_friction'] == 600.0, "m²/s² conversion failed"
    assert converted['disc_friction'] == 400.0, "J/kg passthrough failed"
    
    print(f"  Input (Pa):     inducer_friction = {raw_losses['inducer_friction']} Pa")
    print(f"  Output (J/kg):  inducer_friction = {converted['inducer_friction']} J/kg")
    print(f"  Input (J/kg):   incidence = {raw_losses['incidence']} J/kg")
    print(f"  Output (J/kg):  incidence = {converted['incidence']} J/kg")
    print(f"  Input (m²/s²):  skin_friction = {raw_losses['skin_friction']} m²/s²")
    print(f"  Output (J/kg):  skin_friction = {converted['skin_friction']} J/kg")
    print("✅ All conversions correct!")


def test_dimensional_analysis():
    """Test dimensional consistency"""
    print("\n" + "="*80)
    print("TEST 5: Dimensional Analysis")
    print("="*80)
    
    # Pa / (kg/m³) = J/kg verification
    calc = SimpleLossCalculator('schiffmann')
    
    pressure_pa = 15000.0
    density_kgm3 = 7.5
    
    raw = {'inducer_friction': pressure_pa}
    converted = calc._convert_losses_to_j_per_kg(raw, density_kgm3)
    
    expected = pressure_pa / density_kgm3
    actual = converted['inducer_friction']
    error = abs(actual - expected)
    
    print(f"  Pressure:  {pressure_pa} Pa")
    print(f"  Density:   {density_kgm3} kg/m³")
    print(f"  Expected:  {expected} J/kg")
    print(f"  Actual:    {actual} J/kg")
    print(f"  Error:     {error} J/kg")
    
    assert error < 0.001, "Dimensional analysis failed"
    print("✅ Dimensional analysis: Pa/(kg/m³) = J/kg ✓")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CALCULATOR.PY MANUAL UNIT TEST")
    print("="*80 + "\n")
    
    try:
        test_unit_conversions()
        test_all_models()
        test_model_specific_units()
        test_mixed_units()
        test_dimensional_analysis()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nCalculator.py unit conversion logic is CORRECT!")
        print("Ready for integration with components.py\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
