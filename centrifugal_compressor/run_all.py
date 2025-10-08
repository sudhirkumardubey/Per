#!/usr/bin/env python3
"""
Batch run for all compressors defined in data/known_compressors.yml

This script:
  1. Loads all compressors from the YAML file
  2. Runs the full compressor stage model for each
  3. Prints a tabulated summary (PR and efficiency)
  4. Optionally saves results to CSV

Usage:
    python run_all.py
"""

import os
import sys
import math
import yaml
import pandas as pd

# ---------------------------------------------------------------------------- #
# Ensure project root in path
# ---------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.components.stage import Stage

# ---------------------------------------------------------------------------- #
# Locate YAML file
# ---------------------------------------------------------------------------- #
DATA_FILE = os.path.join(ROOT, "centrifugal_compressor", "data", "known_compressors.yml")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

# ---------------------------------------------------------------------------- #
# Load YAML data
# ---------------------------------------------------------------------------- #
with open(DATA_FILE, "r") as f:
    compressors = yaml.safe_load(f)

print("\n" + "=" * 90)
print(f" RUNNING {len(compressors)} COMPRESSORS FROM {DATA_FILE}")
print("=" * 90)

# ---------------------------------------------------------------------------- #
# Helper function to build blockage array and handle missing fields
# ---------------------------------------------------------------------------- #
def build_geometry(geom_data):
    # Construct blockage array if individual fields are given
    blockage = [
        geom_data.get("blockage1", 1.0),
        geom_data.get("blockage2", 1.0),
        geom_data.get("blockage3", 1.0),
        geom_data.get("blockage4", 1.0),
        geom_data.get("blockage5", 1.0),
        1.0,  # default for vaned diffuser (if not used)
    ]
    for i in range(1, 6):
        geom_data.pop(f"blockage{i}", None)
    geom_data["blockage"] = blockage


    # Map field names if different from Geometry class
    if "blade_e" in geom_data:
        blade_e_val = geom_data.pop("blade_e")
        geom_data["blade_le"] = blade_e_val
        geom_data["blade_te"] = blade_e_val

    if "rug_ind" in geom_data:
        geom_data["rough_inducer"] = geom_data.pop("rug_ind")

    if "clearance" in geom_data:
        geom_data["tip_cl"] = geom_data.pop("clearance")

    if "backface" in geom_data:
        geom_data["back_cl"] = geom_data.pop("backface")

    if "l_ind" in geom_data:
        geom_data["l_inducer"] = geom_data.pop("l_ind")

    if "l_comp" in geom_data:
        geom_data["l_comp"] = geom_data.pop("l_comp")

    # --- Fill in default diffuser parameters if missing ---
    defaults = {
        "beta5": 70.0,     # vaned diffuser inlet angle
        "r6": geom_data.get("r5", 1.2 * geom_data.get("r4", 0.05)),  # default slightly larger than r5
        "b6": geom_data.get("b5", 0.5 * geom_data.get("b4", 0.005)), # half of previous width
        "beta6": 45.0,     # outlet vane angle
        "n_vanes": 0,      # 0 = vaneless diffuser (default)
    }
    for key, val in defaults.items():
        geom_data.setdefault(key, val)

    return Geometry(**geom_data)

# ---------------------------------------------------------------------------- #
# Run each compressor and collect results
# ---------------------------------------------------------------------------- #
results = []

for comp in compressors:
    name = comp.get("name", "Unnamed")
    geom_data = comp.get("geom", {})
    cond_data = comp.get("conditions", {})

    # Evaluate numerical expressions safely (like 180000 * np.pi / 30)
    for key, val in cond_data.items():
        if isinstance(val, str) and ("np." in val or "*" in val or "/" in val):
            import numpy as np
            cond_data[key] = eval(val, {"np": np, "math": math})

    try:
        geom = build_geometry(geom_data)

        def _to_float(val):
            """Convert YAML numeric string (like '1.01e5') safely to float."""
            if isinstance(val, (int, float)):
                return val
            try:
                return float(eval(str(val), {"__builtins__": {}}, {}))
            except Exception:
                return float(val)

        op = OperatingCondition(
            mass_flow=_to_float(cond_data.get("mass_flow")),
            omega=_to_float(cond_data.get("omega")),
            P_inlet=_to_float(cond_data.get("in_P")),
            T_inlet=_to_float(cond_data.get("in_T")),
            fluid_name=cond_data.get("fluid", "Air")
        )

        loss_model = cond_data.get("loss_model", "schiffmann")

        print(f"\n--- Running compressor: {name} ---")
        stage = Stage(geom, op, loss_model=loss_model)

        if stage.invalid_flag or getattr(stage, "choke_flag", False):
            print(f"   - Choked: {getattr(stage, 'choke_flag', False)}")
            print(f"   - Wet region: {getattr(stage, 'wet', False)}")
            print(f"   - Invalid thermodynamics: {stage.invalid_flag}")
            continue

        res = {
            "Name": name,
            "PR_tt": stage.pr_tt,
            "PR_ts": stage.pr_ts,
            "Eff_tt(%)": stage.eff_tt * 100,
            "Eff_ts(%)": stage.eff_ts * 100 if hasattr(stage, "eff_ts") else None,
            "Power(kW)": stage.power / 1000,
        }
        results.append(res)
        print(f"✓ {name}: PR_tt={res['PR_tt']:.3f}, η_tt={res['Eff_tt(%)']:.2f}%")

    except Exception as e:
        print(f"✗ {name}: Error → {e}")

# ---------------------------------------------------------------------------- #
# Print summary table
# ---------------------------------------------------------------------------- #
print("\n" + "=" * 90)
print(" SUMMARY OF ALL COMPRESSORS")
print("=" * 90)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:.3f}"))

    # Save to CSV
    out_csv = os.path.join(ROOT, "centrifugal_compressor", "data", "results_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Results saved to: {out_csv}")
else:
    print("No successful runs found.")

print("=" * 90)
print("✓ BATCH EVALUATION COMPLETE")
print("=" * 90)
