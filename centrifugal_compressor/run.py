#!/usr/bin/env python3
"""
Run script for complete centrifugal compressor stage

Usage:
    $ python run.py

This script:
  1. Builds geometry and operating conditions
  2. Runs the complete stage (Inducer → Impeller → Diffuser)
  3. Prints performance summary

Author: Centrifugal Compressor Team
Date: October 2025
"""

import sys
import os
import math


# --------------------------------------------------------------------------- #
# Ensure project root is in path
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.components.stage import Stage

# --------------------------------------------------------------------------- #
# Define geometry (Eckardt A-type)
# --------------------------------------------------------------------------- #
geom = Geometry(
    # Inducer inlet
    r1=0.14,

    # Impeller inlet
    r2s=0.14,
    r2h=0.06,
    beta2=-53.0,
    beta2s=-63.0,
    alpha2=0.0,

    # Impeller outlet
    r4=0.20,
    b4=0.0267,
    beta4=-30.0,

    # Vaneless diffuser
    r5=0.538,
    b5=0.0136,

    # Vaned diffuser (disabled here)
    beta5=70.0,
    r6=0.09,
    b6=0.006,
    beta6=45.0,

    # Blade counts
    n_blades=20,
    n_splits=0,
    n_vanes=0,  # 0 = vaneless diffuser

    # Blade thickness
    blade_le=2.11e-3,
    blade_te=2.11e-3,

    # Clearances
    tip_cl=213e-6,
    back_cl=235e-6,

    # Surface properties
    rough_inducer=2e-5,

    # Lengths
    l_inducer=0.02,
    l_comp=0.13,

    # Blockage factors [inducer, impeller_inlet, throat, outlet, diffuser, vaned_diffuser]
    blockage=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

# --------------------------------------------------------------------------- #
# Define operating conditions
# --------------------------------------------------------------------------- #
op = OperatingCondition(
    mass_flow=5.32,        # kg/s
    omega=1466.1,          # rad/s (≈ 14,000 RPM)
    P_inlet=101325.0,      # Pa
    T_inlet=288.15,        # K
    fluid_name="Air"
)

# --------------------------------------------------------------------------- #
# Run the stage calculation
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print(" CENTRIFUGAL COMPRESSOR STAGE - RUN MODE")
print("=" * 80 + "\n")

stage = Stage(geom, op, loss_model="schiffmann") # loss models: "none", "meroni", "schiffmann", "oh", "zhang_set1", "zhang_set2", "zhang_set3"

# --------------------------------------------------------------------------- #
# Display results
# --------------------------------------------------------------------------- #
if stage.invalid_flag:
    print("✗ Stage calculation failed.")
    print(f"   - Choked: {stage.choke_flag}")
    print(f"   - Invalid: {stage.invalid_flag}")
    sys.exit(1)

print(stage.summary())

print("=" * 80)
print("✓ STAGE CALCULATION COMPLETE")
print("=" * 80)

# --------------------------------------------------------------------------- #
# Optional: Export summary data
# --------------------------------------------------------------------------- #
results = {
    "PR_tt": stage.pr_tt,
    "PR_ts": stage.pr_ts,
    "Eff_tt": stage.eff_tt * 100,  # %
    "Power_kW": stage.power / 1000,
    "Torque_Nm": stage.torque,
    "Psi": stage.psi,
    "Phi": stage.phi,
}

print("\nResults dictionary (for programmatic use):")
for k, v in results.items():
    print(f"  {k:<10} = {v:.5f}")

print("\n" + "=" * 80 + "\n")
