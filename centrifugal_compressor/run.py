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
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Ensure project root is in path
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.components.stage import Stage
from centrifugal_compressor.analysis.surge import check_surge



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
    mass_flow=4.54,        # kg/s
    omega=1466.1,          # rad/s (≈ 14,000 RPM)
    P_inlet=101325.0,      # Pa
    T_inlet=288.15,        # K
    fluid_name="Air"
)

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("  CENTRIFUGAL COMPRESSOR STAGE - ANALYSIS WITH SURGE PREDICTION")
print("="*80 + "\n")

print("CONFIGURATION:")
print(f"  Loss model:         schiffmann")
print(f"  Diffuser type:      Vaneless only (r5/r4 = {geom.r5/geom.r4:.3f})")
print(f"  Operating point:    ṁ = {op.mass_flow:.2f} kg/s, N = {op.omega*60/(2*3.14159):.0f} rpm")
print()

# --------------------------------------------------------------------------- #
# Run the stage calculation
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("Running stage analysis...")
print("=" * 80 + "\n")

stage = Stage(geom, op, loss_model="meroni") # loss models: "none", "meroni", "schiffmann", "oh", "zhang_set1", "zhang_set2", "zhang_set3"


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


# --------------------------------------------------------------------------- #
# DETAILED COMPONENT BREAKDOWN (Optional)
# --------------------------------------------------------------------------- #
if True:  # Set to False to skip detailed output
    print("\n" + "="*80)
    print("  DETAILED COMPONENT ANALYSIS")
    print("="*80 + "\n")
    
    # Inducer
    if stage.inducer is not None:
        print("─" * 80)
        print("INDUCER:")
        print("─" * 80)
        print(f"  Inlet Pressure (total):     {stage.inducer.inlet.total.P/1000:.2f} kPa")
        print(f"  Inlet Temperature (total):  {stage.inducer.inlet.total.T:.2f} K")
        print(f"  Outlet Pressure (total):    {stage.inducer.outlet.total.P/1000:.2f} kPa")
        print(f"  Outlet Temperature (total): {stage.inducer.outlet.total.T:.2f} K")
        print(f"  Inlet Velocity:             {stage.inducer.inlet.v:.2f} m/s")
        print(f"  Outlet Velocity:            {stage.inducer.outlet.v:.2f} m/s")
        print(f"  Inlet Mach:                 {stage.inducer.inlet.m_abs:.4f}")
        print(f"  Outlet Mach:                {stage.inducer.outlet.m_abs:.4f}")
        # print(f"  Efficiency:                 {stage.inducer.eff*100:.2f}%")
        print()
    
    # Impeller
    print("─" * 80)
    print("IMPELLER:")
    print("─" * 80)
    print(f"  Inlet Pressure (total):     {stage.impeller.inlet.total.P/1000:.2f} kPa")
    print(f"  Inlet Temperature (total):  {stage.impeller.inlet.total.T:.2f} K")
    print(f"  Outlet Pressure (total):    {stage.impeller.outlet.total.P/1000:.2f} kPa")
    print(f"  Outlet Temperature (total): {stage.impeller.outlet.total.T:.2f} K")
    print(f"  Pressure Ratio (tt):        {stage.impeller.pr_tt:.4f}")
    print(f"  Efficiency (tt):            {stage.impeller.eff_tt*100:.2f}%")
    print(f"  De Haller Number:           {stage.impeller.de_haller:.4f}")
    print(f"  Inlet Relative Velocity:    {stage.impeller.inlet.w:.2f} m/s")
    print(f"  Outlet Relative Velocity:   {stage.impeller.outlet.w:.2f} m/s")
    print(f"  Inlet Flow Angle:           {stage.impeller.inlet.beta:.2f}°")
    print(f"  Outlet Flow Angle:          {stage.impeller.outlet.beta:.2f}°")
    print(f"  Inlet Blade Speed:          {stage.impeller.inlet.u:.2f} m/s")
    print(f"  Outlet Blade Speed:         {stage.impeller.outlet.u:.2f} m/s")
    
    # Throat information
    if hasattr(stage.impeller.throat, 'static') and stage.impeller.throat.static.P > 0:
        print(f"  Throat Relative Mach:       {stage.impeller.throat.m_rel:.4f}")
        print(f"  Throat Relative Velocity:   {stage.impeller.throat.w:.2f} m/s")
    
    # Loss breakdown
    if stage.impeller.losses.breakdown:
        print(f"\n  Loss Breakdown:")
        for loss_name, loss_value in stage.impeller.losses.breakdown.items():
            print(f"    {loss_name:<20}: {loss_value:>8.2f} J/kg")
        print(f"    {'Total Losses':<20}: {stage.impeller.losses.total:>8.2f} J/kg")
    print()
    
    # Diffuser
    print("─" * 80)
    print("DIFFUSER:")
    print("─" * 80)
    print(f"  Configuration:              {stage.diffuser.config}")
    print(f"  Inlet Pressure (total):     {stage.diffuser.inlet.total.P/1000:.2f} kPa")
    print(f"  Inlet Temperature (total):  {stage.diffuser.inlet.total.T:.2f} K")
    print(f"  Outlet Pressure (total):    {stage.diffuser.outlet.total.P/1000:.2f} kPa")
    print(f"  Outlet Temperature (total): {stage.diffuser.outlet.total.T:.2f} K")
    print(f"  Pressure Recovery:          {stage.diffuser.outlet.static.P:.4f}")
    
    print("="*80 + "\n")