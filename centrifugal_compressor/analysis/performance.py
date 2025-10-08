# File: centrifugal_compressor/analysis/performance.py
"""
Performance Map Generation for Centrifugal Compressor

Generates complete compressor characteristic maps by:
1. Sweeping speed lines (40% to 110% of design speed)
2. Sweeping mass flow from surge to choke for each speed
3. Computing non-dimensional coefficients
4. Detecting operating limits

Based on RadComp architecture with systematic operating point exploration.

Author: Centrifugal Compressor Team
Date: October 2025
Reference: Japikse & Baines (1994), Casey & Robinson (2010)
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from centrifugal_compressor.core.geometry import Geometry, OperatingCondition
from centrifugal_compressor.core.thermodynamics import FluidState, Fluid
from centrifugal_compressor.components.stage import Stage
from Archive.surge import SurgePredictor, SurgeAnalysis


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OperatingPoint:
    """
    Single operating point on performance map
    
    Contains all relevant performance and thermodynamic data
    """
    # Operating conditions
    speed_rpm: float = math.nan          # Rotational speed [RPM]
    mass_flow: float = math.nan          # Mass flow rate [kg/s]
    omega: float = math.nan              # Angular velocity [rad/s]
    
    # Dimensionless parameters
    phi: float = math.nan                # Flow coefficient [-]
    psi: float = math.nan                # Head coefficient [-]
    mu: float = math.nan                 # Work input coefficient [-]
    
    # Performance metrics
    pr_tt: float = math.nan              # Total-to-total pressure ratio [-]
    pr_ts: float = math.nan              # Total-to-static pressure ratio [-]
    eff_tt: float = math.nan             # Total-to-total efficiency [-]
    eff_ts: float = math.nan             # Total-to-static efficiency [-]
    
    # Power and torque
    power: float = math.nan              # Shaft power [W]
    torque: float = math.nan             # Shaft torque [N·m]
    
    # Surge analysis
    surge_margin: float = math.nan       # Surge margin [%]
    alpha_actual: float = math.nan       # Actual flow angle [deg]
    alpha_critical: float = math.nan     # Critical flow angle [deg]
    
    # Thermodynamic states
    inlet_total_pressure: float = math.nan    # [Pa]
    inlet_total_temp: float = math.nan        # [K]
    outlet_total_pressure: float = math.nan   # [Pa]
    outlet_total_temp: float = math.nan       # [K]
    
    # Flags
    converged: bool = False
    choked: bool = False
    surged: bool = False
    invalid: bool = False
    
    @property
    def is_valid(self) -> bool:
        """Check if operating point is valid"""
        return self.converged and not (self.choked or self.surged or self.invalid)


@dataclass
class SpeedLine:
    """
    Single speed line on performance map
    
    Contains all operating points at constant rotational speed
    """
    speed_rpm: float                     # Rotational speed [RPM]
    speed_fraction: float                # Fraction of design speed [-]
    points: List[OperatingPoint] = field(default_factory=list)
    
    # Operating limits
    surge_point: Optional[OperatingPoint] = None
    choke_point: Optional[OperatingPoint] = None
    peak_efficiency_point: Optional[OperatingPoint] = None
    
    def add_point(self, point: OperatingPoint):
        """Add operating point to speed line"""
        self.points.append(point)
    
    def find_limits(self):
        """Identify surge, choke, and peak efficiency points"""
        valid_points = [p for p in self.points if p.is_valid]
        
        if not valid_points:
            return
        
        # Find surge point (lowest valid mass flow)
        self.surge_point = min(valid_points, key=lambda p: p.mass_flow)
        
        # Find choke point (highest valid mass flow)
        self.choke_point = max(valid_points, key=lambda p: p.mass_flow)
        
        # Find peak efficiency
        self.peak_efficiency_point = max(valid_points, key=lambda p: p.eff_tt)
    
    @property
    def mass_flow_range(self) -> Tuple[float, float]:
        """Get mass flow range (surge to choke)"""
        if self.surge_point and self.choke_point:
            return (self.surge_point.mass_flow, self.choke_point.mass_flow)
        return (math.nan, math.nan)
    
    @property
    def pressure_ratio_range(self) -> Tuple[float, float]:
        """Get pressure ratio range"""
        valid_prs = [p.pr_tt for p in self.points if p.is_valid]
        if valid_prs:
            return (min(valid_prs), max(valid_prs))
        return (math.nan, math.nan)


@dataclass
class PerformanceMap:
    """
    Complete compressor performance map
    
    Contains multiple speed lines spanning operating envelope
    """
    design_speed_rpm: float
    design_mass_flow: float
    speed_lines: List[SpeedLine] = field(default_factory=list)
    
    # Map metadata
    geometry: Optional[Geometry] = None
    fluid_name: str = "Air"
    inlet_pressure: float = math.nan
    inlet_temperature: float = math.nan
    loss_model: str = "meroni"
    
    def add_speed_line(self, speed_line: SpeedLine):
        """Add speed line to map"""
        self.speed_lines.append(speed_line)
    
    def get_speed_line(self, speed_fraction: float, tol: float = 0.01) -> Optional[SpeedLine]:
        """Get speed line by speed fraction"""
        for sl in self.speed_lines:
            if abs(sl.speed_fraction - speed_fraction) < tol:
                return sl
        return None
    
    @property
    def surge_line(self) -> List[OperatingPoint]:
        """Extract surge line from all speed lines"""
        surge_points = []
        for sl in self.speed_lines:
            if sl.surge_point:
                surge_points.append(sl.surge_point)
        return surge_points
    
    @property
    def choke_line(self) -> List[OperatingPoint]:
        """Extract choke line from all speed lines"""
        choke_points = []
        for sl in self.speed_lines:
            if sl.choke_point:
                choke_points.append(sl.choke_point)
        return choke_points
    
    @property
    def efficiency_islands(self) -> Dict[float, List[OperatingPoint]]:
        """
        Extract efficiency islands (contours)
        
        Returns dict: efficiency_level -> list of points
        """
        islands = {}
        levels = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        
        for level in levels:
            island_points = []
            for sl in self.speed_lines:
                for pt in sl.points:
                    if pt.is_valid and abs(pt.eff_tt - level) < 0.02:
                        island_points.append(pt)
            if island_points:
                islands[level] = island_points
        
        return islands


# ============================================================================
# PERFORMANCE MAP GENERATOR
# ============================================================================

class PerformanceMapGenerator:
    """
    Generates complete performance maps for centrifugal compressors
    
    Methodology:
    1. Define speed lines (fractions of design speed)
    2. For each speed line, sweep mass flow from low to high
    3. Run stage calculation for each (speed, mass_flow) combination
    4. Detect surge and choke limits
    5. Extract performance metrics
    
    Usage:
        generator = PerformanceMapGenerator(geom, design_op, loss_model='meroni')
        perf_map = generator.generate(
            speed_fractions=[0.5, 0.7, 0.9, 1.0, 1.1],
            n_points=20
        )
    """
    
    def __init__(
        self,
        geometry: Geometry,
        design_operating_condition: OperatingCondition,
        loss_model: str = "meroni"
    ):
        """
        Initialize performance map generator
        
        Args:
            geometry: Compressor geometry
            design_operating_condition: Design point operating condition
            loss_model: Loss model to use (meroni, oh, schiffmann, etc.)
        """
        self.geom = geometry
        self.design_op = design_operating_condition
        self.loss_model = loss_model
        
        # Design point parameters
        self.design_speed_rpm = design_operating_condition.rpm
        self.design_mass_flow = design_operating_condition.mass_flow
        
        # Convergence settings
        self.max_iterations = 50
        self.convergence_tol = 1e-4
    
    def generate(
        self,
        speed_fractions: List[float] = None,
        n_points: int = 20,
        mass_flow_range: Tuple[float, float] = (0.5, 1.3),
        surge_safety_margin: float = 5.0
    ) -> PerformanceMap:
        """
        Generate complete performance map
        
        Args:
            speed_fractions: List of speed fractions (e.g., [0.6, 0.8, 1.0, 1.1])
            n_points: Number of points per speed line
            mass_flow_range: Mass flow range as fraction of design (min, max)
            surge_safety_margin: Safety margin for surge detection [%]
        
        Returns:
            PerformanceMap object with all speed lines
        """
        if speed_fractions is None:
            speed_fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1]
        
        perf_map = PerformanceMap(
            design_speed_rpm=self.design_speed_rpm,
            design_mass_flow=self.design_mass_flow,
            geometry=self.geom,
            fluid_name=self.design_op.fluid_name,
            inlet_pressure=self.design_op.P_inlet,
            inlet_temperature=self.design_op.T_inlet,
            loss_model=self.loss_model
        )
        
        print("\n" + "="*70)
        print("PERFORMANCE MAP GENERATION")
        print("="*70)
        print(f"Design Speed: {self.design_speed_rpm:.1f} RPM")
        print(f"Design Mass Flow: {self.design_mass_flow:.4f} kg/s")
        print(f"Loss Model: {self.loss_model}")
        print(f"Speed Lines: {len(speed_fractions)}")
        print(f"Points per Line: {n_points}")
        print("="*70 + "\n")
        
        for speed_frac in speed_fractions:
            speed_line = self._generate_speed_line(
                speed_frac,
                n_points,
                mass_flow_range,
                surge_safety_margin
            )
            perf_map.add_speed_line(speed_line)
            
            # Print progress
            valid_count = len([p for p in speed_line.points if p.is_valid])
            print(f"  Speed {speed_frac*100:5.1f}%: {valid_count}/{n_points} valid points | "
                  f"Surge: {speed_line.surge_point is not None} | "
                  f"Choke: {speed_line.choke_point is not None}")
        
        print("\n" + "="*70)
        print("PERFORMANCE MAP GENERATION COMPLETE")
        print("="*70 + "\n")
        
        return perf_map
    
    def _generate_speed_line(
        self,
        speed_fraction: float,
        n_points: int,
        mass_flow_range: Tuple[float, float],
        surge_safety_margin: float
    ) -> SpeedLine:
        """
        Generate single speed line
        
        Args:
            speed_fraction: Fraction of design speed
            n_points: Number of points
            mass_flow_range: Mass flow range (min_frac, max_frac)
            surge_safety_margin: Surge safety margin [%]
        
        Returns:
            SpeedLine object with computed points
        """
        speed_rpm = self.design_speed_rpm * speed_fraction
        omega = speed_rpm * 2 * math.pi / 60
        
        speed_line = SpeedLine(
            speed_rpm=speed_rpm,
            speed_fraction=speed_fraction
        )
        
        # Define mass flow sweep
        m_min = self.design_mass_flow * mass_flow_range[0]
        m_max = self.design_mass_flow * mass_flow_range[1]
        mass_flows = np.linspace(m_min, m_max, n_points)
        
        for m_dot in mass_flows:
            point = self._compute_operating_point(
                omega,
                speed_rpm,
                m_dot,
                surge_safety_margin
            )
            speed_line.add_point(point)
        
        # Identify limits
        speed_line.find_limits()
        
        return speed_line
    
    def _compute_operating_point(
        self,
        omega: float,
        speed_rpm: float,
        mass_flow: float,
        surge_safety_margin: float
    ) -> OperatingPoint:
        """
        Compute single operating point
        
        Args:
            omega: Angular velocity [rad/s]
            speed_rpm: Rotational speed [RPM]
            mass_flow: Mass flow rate [kg/s]
            surge_safety_margin: Surge margin threshold [%]
        
        Returns:
            OperatingPoint with all computed data
        """
        point = OperatingPoint(
            speed_rpm=speed_rpm,
            mass_flow=mass_flow,
            omega=omega
        )
        
        try:
            # Create operating condition with correct parameters
            op = OperatingCondition(
                mass_flow=mass_flow,
                omega=omega,
                P_inlet=self.design_op.P_inlet,
                T_inlet=self.design_op.T_inlet,
                fluid_name=self.design_op.fluid_name
            )
            
            # Run stage calculation
            stage = Stage(self.geom, op, loss_model=self.loss_model)
            
            # Check validity
            if stage.invalid_flag or stage.choke_flag:
                point.invalid = stage.invalid_flag
                point.choked = stage.choke_flag
                return point
            
            # Extract performance metrics
            point.pr_tt = stage.pr_tt
            point.pr_ts = stage.pr_ts
            point.eff_tt = stage.eff_tt
            point.eff_ts = stage.eff_ts
            point.power = stage.power
            point.torque = stage.torque
            
            # Dimensionless coefficients
            point.phi = stage.phi
            point.psi = stage.psi
            point.mu = stage.mu
            
            # Thermodynamic states
            point.inlet_total_pressure = stage.inlet.P
            point.inlet_total_temp = stage.inlet.T
            point.outlet_total_pressure = stage.diffuser.outlet.total.P
            point.outlet_total_temp = stage.diffuser.outlet.total.T
            
            # Surge analysis
            surge_predictor = SurgePredictor(self.geom, stage.impeller, stage.diffuser)
            surge_analysis = surge_predictor.analyze()
            
            point.surge_margin = surge_analysis.surge_margin
            point.alpha_actual = surge_analysis.alpha_actual
            point.alpha_critical = surge_analysis.alpha_critical
            point.surged = surge_analysis.in_surge or (surge_analysis.surge_margin < surge_safety_margin)
            
            # Convergence flag
            point.converged = stage.converged
            
        except Exception as e:
            point.invalid = True
            warnings.warn(f"Operating point failed: {e}")
        
        return point


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_performance_map(
    geometry: Geometry,
    design_operating_condition: OperatingCondition,
    loss_model: str = "meroni",
    speed_fractions: List[float] = None,
    n_points: int = 20
) -> PerformanceMap:
    """
    Convenience function to generate performance map
    
    Args:
        geometry: Compressor geometry
        design_operating_condition: Design point
        loss_model: Loss model name
        speed_fractions: List of speed fractions
        n_points: Points per speed line
    
    Returns:
        Complete PerformanceMap
    
    Example:
        >>> perf_map = generate_performance_map(geom, op_design)
        >>> print(f"Surge line: {len(perf_map.surge_line)} points")
    """
    generator = PerformanceMapGenerator(geometry, design_operating_condition, loss_model)
    return generator.generate(speed_fractions, n_points)


def export_performance_map_csv(perf_map: PerformanceMap, filename: str):
    """
    Export performance map to CSV file
    
    Args:
        perf_map: PerformanceMap object
        filename: Output CSV file path
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'speed_rpm', 'speed_fraction', 'mass_flow', 'pr_tt', 'pr_ts',
            'eff_tt', 'eff_ts', 'power', 'phi', 'psi', 'surge_margin',
            'converged', 'choked', 'surged', 'invalid'
        ])
        
        # Data
        for sl in perf_map.speed_lines:
            for pt in sl.points:
                writer.writerow([
                    pt.speed_rpm, sl.speed_fraction, pt.mass_flow,
                    pt.pr_tt, pt.pr_ts, pt.eff_tt, pt.eff_ts, pt.power,
                    pt.phi, pt.psi, pt.surge_margin,
                    pt.converged, pt.choked, pt.surged, pt.invalid
                ])
    
    print(f"Performance map exported to: {filename}")
