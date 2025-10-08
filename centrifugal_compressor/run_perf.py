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


# --- Performance map plotting (append to run.py) --------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

# import generator and exporter from your analysis module
from centrifugal_compressor.analysis.performance import (
    generate_performance_map,
    export_performance_map_csv,
    PerformanceMap,
)

# Optional: grid interpolation for smooth contour (requires scipy)
try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def plot_speed_lines_vs_pr(perf_map: PerformanceMap, ax=None, savepath=None):
    """Plot mass_flow vs pressure ratio (PR_tt) for each speed line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    for sl in perf_map.speed_lines:
        mass = [p.mass_flow for p in sl.points if p.is_valid]
        pr   = [p.pr_tt     for p in sl.points if p.is_valid]
        if not mass:
            continue
        ax.plot(mass, pr, '-o', markersize=4, label=f"{sl.speed_rpm:.0f} RPM")
        # mark surge & choke if available
        if sl.surge_point and sl.surge_point.is_valid:
            ax.plot(sl.surge_point.mass_flow, sl.surge_point.pr_tt, 's', ms=8, mfc='none', mew=1.5)
        if sl.choke_point and sl.choke_point.is_valid:
            ax.plot(sl.choke_point.mass_flow, sl.choke_point.pr_tt, 'X', ms=8)
    ax.set_xlabel("Mass flow (kg/s)")
    ax.set_ylabel("Pressure ratio PR_tt (-)")
    ax.set_title("Speed lines: Mass flow vs PR")
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=200)

def plot_efficiency_map(perf_map: PerformanceMap, ax=None, savepath=None):
    """Scatter (or contour) of efficiency vs mass flow and speed."""
    points = []
    for sl in perf_map.speed_lines:
        for p in sl.points:
            if p.is_valid:
                points.append((sl.speed_fraction, p.mass_flow, p.eff_tt))
    if not points:
        return
    speed_frac = np.array([pt[0] for pt in points])
    mass_flow  = np.array([pt[1] for pt in points])
    eff        = np.array([pt[2] for pt in points])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    # If scipy available, produce smooth contour by interpolating on grid
    if HAS_SCIPY:
        # build regular grid
        SF = np.linspace(speed_frac.min(), speed_frac.max(), 80)
        MF = np.linspace(mass_flow.min(), mass_flow.max(), 80)
        SFg, MFg = np.meshgrid(SF, MF)
        Effg = griddata((speed_frac, mass_flow), eff, (SFg, MFg), method='cubic')
        c = ax.contourf(MFg, SFg, Effg, levels=15, cmap='viridis')
        cb = plt.colorbar(c, ax=ax)
        cb.set_label("Eff_tt (-)")
        ax.set_xlabel("Mass flow (kg/s)")
        ax.set_ylabel("Speed fraction")
        ax.set_title("Efficiency contour (interpolated)")
    else:
        sc = ax.scatter(mass_flow, speed_frac, c=eff, s=30, cmap='viridis')
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Eff_tt (-)")
        ax.set_xlabel("Mass flow (kg/s)")
        ax.set_ylabel("Speed fraction")
        ax.set_title("Efficiency scatter (Eff_tt colored)")

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=200)

def plot_phi_psi_map(perf_map: PerformanceMap, ax=None, savepath=None):
    """Plot phi vs psi colored by efficiency; show peak efficiency marker."""
    points = []
    for sl in perf_map.speed_lines:
        for p in sl.points:
            if p.is_valid:
                points.append((p.phi, p.psi, p.eff_tt))
    if not points:
        return
    phi = np.array([pt[0] for pt in points])
    psi = np.array([pt[1] for pt in points])
    eff = np.array([pt[2] for pt in points])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(phi, psi, c=eff, s=30, cmap='plasma')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Eff_tt (-)")
    ax.set_xlabel("Flow coefficient φ")
    ax.set_ylabel("Head coefficient ψ")
    ax.set_title("φ - ψ map (colored by efficiency)")
    ax.grid(True)

    # annotate peak efficiency
    idx = np.nanargmax(eff)
    ax.plot(phi[idx], psi[idx], 'k*', ms=12, label='Peak eff')
    ax.legend()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=200)

# ------------------- run generator and plot -------------------
def generate_and_plot(geometry, design_op, out_dir="performance_output", loss_model="schiffmann"):
    os.makedirs(out_dir, exist_ok=True)

    print("Generating performance map (this may take a while)...")
    perf_map = generate_performance_map(geometry, design_op, loss_model=loss_model, speed_fractions=[0.5,0.7,0.9,1.0,1.1], n_points=20)

    # csv export
    csv_file = os.path.join(out_dir, "performance_map.csv")
    export_performance_map_csv(perf_map, csv_file)

    # plots
    plot_speed_lines_vs_pr(perf_map, savepath=os.path.join(out_dir, "speed_lines_pr.png"))
    plot_efficiency_map(perf_map, savepath=os.path.join(out_dir, "efficiency_map.png"))
    plot_phi_psi_map(perf_map, savepath=os.path.join(out_dir, "phi_psi_map.png"))

    print("Performance plots and CSV written to:", os.path.abspath(out_dir))
    return perf_map

# Example call: uncomment if you want to run right after running single-stage above
perf_map = generate_and_plot(geom, op, out_dir="performance_output", loss_model="schiffmann")
plt.show()
