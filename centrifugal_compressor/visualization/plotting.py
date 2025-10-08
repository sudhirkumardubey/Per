try:
    import CoolProp as CP
    from CoolProp.Plots import PropertyPlot, StateContainer
except ImportError:
    CP, PropertyPlot, StateContainer = None, None, None

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any, Tuple, Dict, List, Union
from ..components.stage import Stage


def plot_compressor_cycle(stage: Stage, plot_type: str = "Ts", fluid_name: str = "Air") -> Optional[Any]:
    """
    Plot thermodynamic cycle on T-s or P-h diagram using CoolProp.
    
    Parameters:
    -----------
    stage : Stage
        Completed Stage object with converged solution
    plot_type : str
        Type of plot: "Ts" (Temperature-Entropy) or "Ph" (Pressure-Enthalpy)
    fluid_name : str
        Fluid name for CoolProp (default: "Air")
    
    Returns:
    --------
    plot : PropertyPlot
        CoolProp PropertyPlot object
    
    Example:
    --------
    >>> stage = Stage(geom, op, loss_model="schiffmann")
    >>> plot = plot_compressor_cycle(stage, "Ts")
    >>> plt.show()
    """
    if PropertyPlot is None or CP is None:
        raise ImportError("CoolProp is required for plotting functionality. Please install CoolProp.")
    
    if stage.invalid_flag or stage.choke_flag:
        raise ValueError("Cannot plot cycle for invalid or choked stage")
    
    plot = PropertyPlot(fluid_name.capitalize(), plot_type)
    plot.calc_isolines(CP.iQ, num=11)
    plot.calc_isolines(CP.iP, num=15)

    states = StateContainer()

    # State 0: Stage inlet (compressor inlet)
    states[0, "H"] = stage.inlet.H
    states[0, "S"] = stage.inlet.S
    states[0, "T"] = stage.inlet.T
    states[0, "P"] = stage.inlet.P

    # State 1: Inducer outlet / Impeller inlet (if inducer exists)
    if stage.inducer is not None:
        states[1, "H"] = stage.inducer.outlet.total.H
        states[1, "S"] = stage.inducer.outlet.total.S
        states[1, "T"] = stage.inducer.outlet.total.T
        states[1, "P"] = stage.inducer.outlet.total.P
        
        # State 2: Impeller outlet
        states[2, "H"] = stage.impeller.outlet.total.H
        states[2, "S"] = stage.impeller.outlet.total.S
        states[2, "T"] = stage.impeller.outlet.total.T
        states[2, "P"] = stage.impeller.outlet.total.P
        
        # State 3: Diffuser outlet (stage outlet)
        states[3, "H"] = stage.diffuser.outlet.total.H
        states[3, "S"] = stage.diffuser.outlet.total.S
        states[3, "T"] = stage.diffuser.outlet.total.T
        states[3, "P"] = stage.diffuser.outlet.total.P
    else:
        # No inducer: skip to impeller
        states[1, "H"] = stage.impeller.outlet.total.H
        states[1, "S"] = stage.impeller.outlet.total.S
        states[1, "T"] = stage.impeller.outlet.total.T
        states[1, "P"] = stage.impeller.outlet.total.P
        
        # State 2: Diffuser outlet (stage outlet)
        states[2, "H"] = stage.diffuser.outlet.total.H
        states[2, "S"] = stage.diffuser.outlet.total.S
        states[2, "T"] = stage.diffuser.outlet.total.T
        states[2, "P"] = stage.diffuser.outlet.total.P

    plot.draw_process(states)
    return plot


def plot_compressor_map_dual(
    map_data: Dict,
    compressor_name: str,
    design_point: Optional[Tuple[float, float, float]] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Create traditional dual compressor maps (pressure ratio and efficiency vs mass flow).
    
    Parameters:
    -----------
    map_data : dict
        Dictionary containing:
        - 'n_rot_grid': array of rotational speeds
        - 'm_flow_grid': array of mass flows  
        - 'pr_results': array of pressure ratios
        - 'eta_results': array of efficiencies
        - 'valid_results': array of boolean validity flags
    compressor_name : str
        Name of the compressor for plot titles
    design_point : tuple, optional
        (mass_flow, pressure_ratio, efficiency) for design point highlighting
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    (ax1, ax2) : tuple of Axes
        Pressure ratio and efficiency plot axes
    """
    
    # Extract data
    n_rot_grid = map_data['n_rot_grid']
    m_flow_grid = map_data['m_flow_grid']
    pr_results = map_data['pr_results']
    eta_results = map_data['eta_results']
    valid_results = map_data['valid_results']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors and markers for different speed lines
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    # Group data by speed (rounded to nearest 100 rad/s for grouping)
    speed_tolerance = 100  # rad/s
    unique_speeds = []
    speed_groups = {}
    
    for i, speed in enumerate(n_rot_grid):
        # Round speed to nearest tolerance for grouping
        rounded_speed = round(speed / speed_tolerance) * speed_tolerance
        if rounded_speed not in speed_groups:
            speed_groups[rounded_speed] = []
            unique_speeds.append(rounded_speed)
        speed_groups[rounded_speed].append(i)
    
    # Sort speeds
    unique_speeds.sort()
    
    # Set up axes
    ax1.set_xlabel('$\\dot{m}$ [kg/s]')
    ax1.set_ylabel('$PR_{tt}$ [-]')
    ax1.set_title(f'{compressor_name} - Pressure Ratio Map')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('$\\dot{m}$ [kg/s]')
    ax2.set_ylabel('$\\eta_{tt}$ [-]')
    ax2.set_title(f'{compressor_name} - Efficiency Map')
    ax2.grid(True, alpha=0.3)
    
    # Filter for valid points
    valid_mask = valid_results & (~np.isnan(pr_results)) & (~np.isnan(eta_results))
    m_valid = m_flow_grid[valid_mask]
    pr_valid = pr_results[valid_mask]
    eta_valid = eta_results[valid_mask]
    
    # Plot each speed line
    plotted_speeds = []
    for i, speed in enumerate(unique_speeds):
        if i >= len(colors):  # Cycle through colors if we have more speeds than colors
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
        else:
            color = colors[i]
            marker = markers[i]
        
        # Get indices for this speed
        indices = speed_groups[speed]
        
        # Filter for valid points only
        valid_indices = [idx for idx in indices if valid_results[idx] and not np.isnan(pr_results[idx]) and not np.isnan(eta_results[idx])]
        
        if len(valid_indices) > 0:
            # Get data for this speed line
            m_speed = m_flow_grid[valid_indices]
            pr_speed = pr_results[valid_indices]
            eta_speed = eta_results[valid_indices]
            
            # Sort by mass flow for proper line connection
            sort_idx = np.argsort(m_speed)
            m_speed = m_speed[sort_idx]
            pr_speed = pr_speed[sort_idx]
            eta_speed = eta_speed[sort_idx]
            
            # Convert speed to RPM for legend
            rpm = speed * 30 / np.pi / 1000  # krpm
            label = f'{rpm:.0f} krpm'
            
            # Plot pressure ratio
            ax1.plot(m_speed, pr_speed, marker=marker, color=color, linewidth=1.5, 
                    markersize=6, label=label, markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
            
            # Plot efficiency
            ax2.plot(m_speed, eta_speed, marker=marker, color=color, linewidth=1.5,
                    markersize=6, label=label, markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
            
            plotted_speeds.append(speed)
    
    # Set axis limits and formatting
    if len(m_valid) > 0:
        m_min, m_max = np.min(m_valid), np.max(m_valid)
        m_range = m_max - m_min
        ax1.set_xlim(max(0.5, m_min - 0.1*m_range), m_max + 0.1*m_range)
        ax2.set_xlim(max(0.5, m_min - 0.1*m_range), m_max + 0.1*m_range)
    
    if len(pr_valid) > 0:
        pr_min, pr_max = np.min(pr_valid), np.max(pr_valid)
        pr_range = pr_max - pr_min
        ax1.set_ylim(max(1.0, pr_min - 0.1*pr_range), pr_max + 0.1*pr_range)
    
    if len(eta_valid) > 0:
        eta_min, eta_max = np.min(eta_valid), np.max(eta_valid)
        ax2.set_ylim(max(0.6, eta_min - 0.05), min(1.0, eta_max + 0.05))
    
    # Add legends
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Highlight design point if provided
    if design_point:
        design_m, design_pr, design_eta = design_point
        ax1.scatter([design_m], [design_pr], s=150, c='red', marker='*', 
                   edgecolor='black', linewidth=2, zorder=10, label='Design Point')
        ax2.scatter([design_m], [design_eta], s=150, c='red', marker='*',
                   edgecolor='black', linewidth=2, zorder=10, label='Design Point')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_compressor_map_enhanced(
    map_data: Dict,
    compressor_name: str,
    design_point: Optional[Tuple[float, float, float]] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_efficiency_contours: bool = True,
    show_surge_choke_lines: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create enhanced compressor performance map with surge/choke lines and efficiency contours.
    
    Parameters:
    -----------
    map_data : dict
        Dictionary containing map data arrays
    compressor_name : str
        Name of the compressor for plot title
    design_point : tuple, optional
        (mass_flow, pressure_ratio, efficiency) for design point highlighting
    figsize : tuple
        Figure size (width, height)
    show_efficiency_contours : bool
        Whether to show efficiency contour lines
    show_surge_choke_lines : bool
        Whether to show approximate surge and choke lines
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Plot axes
    """
    
    # Extract data
    n_rot_grid = map_data['n_rot_grid']
    m_flow_grid = map_data['m_flow_grid']
    pr_results = map_data['pr_results']
    eta_results = map_data['eta_results']
    valid_results = map_data['valid_results']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Set up main plot
    ax.set_xlabel('Mass Flow Rate, $\\dot{m}$ [kg/s]', fontsize=12)
    ax.set_ylabel('Total Pressure Ratio, $PR_{tt}$ [-]', fontsize=12)
    ax.set_title(f'{compressor_name} Compressor Performance Map', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Filter for valid points
    valid_mask = valid_results & (~np.isnan(pr_results)) & (~np.isnan(eta_results))
    m_valid = m_flow_grid[valid_mask]
    pr_valid = pr_results[valid_mask]
    eta_valid = eta_results[valid_mask]
    
    # Group data by speed
    speed_tolerance = 100  # rad/s
    unique_speeds = []
    speed_groups = {}
    
    for i, speed in enumerate(n_rot_grid):
        rounded_speed = round(speed / speed_tolerance) * speed_tolerance
        if rounded_speed not in speed_groups:
            speed_groups[rounded_speed] = []
            unique_speeds.append(rounded_speed)
        speed_groups[rounded_speed].append(i)
    
    unique_speeds.sort()
    
    # Define enhanced colors and markers
    colors = ['#000000', '#0066CC', '#CC0000', '#00AA00', '#FF8800', '#8800CC', '#00CCCC', '#CC8800']
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    plotted_speeds = []
    surge_points_m = []
    surge_points_pr = []
    choke_points_m = []
    choke_points_pr = []
    
    for i, speed in enumerate(unique_speeds):
        if i >= len(colors):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
        else:
            color = colors[i]
            marker = markers[i]
        
        # Get indices for this speed
        indices = speed_groups[speed]
        valid_indices = [idx for idx in indices if valid_results[idx] and not np.isnan(pr_results[idx])]
        
        if len(valid_indices) >= 2:  # Need at least 2 points for a meaningful speed line
            m_speed = m_flow_grid[valid_indices]
            pr_speed = pr_results[valid_indices]
            eta_speed = eta_results[valid_indices]
            
            # Sort by mass flow
            sort_idx = np.argsort(m_speed)
            m_speed = m_speed[sort_idx]
            pr_speed = pr_speed[sort_idx]
            eta_speed = eta_speed[sort_idx]
            
            # Convert speed to RPM for legend
            rpm = speed * 30 / np.pi / 1000  # krpm
            label = f'{rpm:.0f} krpm'
            
            # Plot speed line
            ax.plot(m_speed, pr_speed, marker=marker, color=color, linewidth=2.0, 
                    markersize=8, label=label, markerfacecolor='white', markeredgecolor=color, 
                    markeredgewidth=2, alpha=0.8)
            
            # Collect surge and choke points if requested
            if show_surge_choke_lines and len(m_speed) > 0:
                # Surge point (leftmost, lowest flow)
                surge_points_m.append(m_speed[0])
                surge_points_pr.append(pr_speed[0])
                
                # Choke point (rightmost, highest flow) 
                choke_points_m.append(m_speed[-1])
                choke_points_pr.append(pr_speed[-1])
            
            plotted_speeds.append(speed)
    
    # Draw approximate surge line (left boundary)
    if show_surge_choke_lines and len(surge_points_m) > 1:
        surge_sort = np.argsort(surge_points_pr)
        surge_m_sorted = np.array(surge_points_m)[surge_sort]
        surge_pr_sorted = np.array(surge_points_pr)[surge_sort]
        ax.plot(surge_m_sorted, surge_pr_sorted, '--', color='red', linewidth=3, 
                alpha=0.7, label='Approximate Surge Line')
    
    # Draw approximate choke line (right boundary)
    if show_surge_choke_lines and len(choke_points_m) > 1:
        choke_sort = np.argsort(choke_points_pr)
        choke_m_sorted = np.array(choke_points_m)[choke_sort]
        choke_pr_sorted = np.array(choke_points_pr)[choke_sort]
        ax.plot(choke_m_sorted, choke_pr_sorted, '--', color='orange', linewidth=3,
                alpha=0.7, label='Approximate Choke Line')
    
    # Add efficiency contours as background
    if show_efficiency_contours and len(m_valid) > 0:
        try:
            from scipy.interpolate import griddata
            
            # Create regular grid
            m_grid = np.linspace(np.min(m_valid), np.max(m_valid), 50)
            pr_grid = np.linspace(np.min(pr_valid), np.max(pr_valid), 50)
            M_grid, PR_grid = np.meshgrid(m_grid, pr_grid)
            
            # Interpolate efficiency
            eta_grid = griddata((m_valid, pr_valid), eta_valid, (M_grid, PR_grid), method='cubic', fill_value=np.nan)
            
            # Plot efficiency contours
            contour_levels = np.arange(0.7, 1.0, 0.05)
            cs = ax.contour(M_grid, PR_grid, eta_grid, levels=contour_levels, colors='gray', 
                           alpha=0.4, linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        except ImportError:
            print("Note: scipy required for efficiency contours")
        except:
            print("Note: Could not generate efficiency contours due to sparse data")
    
    # Highlight design point
    if design_point:
        design_m, design_pr, design_eta = design_point
        ax.scatter([design_m], [design_pr], s=200, c='red', marker='*', 
                  edgecolor='black', linewidth=3, zorder=15, label='Design Point')
    
    # Set limits and formatting
    if len(m_valid) > 0:
        m_range = np.max(m_valid) - np.min(m_valid)
        ax.set_xlim(np.min(m_valid) - 0.1*m_range, np.max(m_valid) + 0.1*m_range)
    if len(pr_valid) > 0:
        pr_range = np.max(pr_valid) - np.min(pr_valid)
        ax.set_ylim(np.min(pr_valid) - 0.1*pr_range, np.max(pr_valid) + 0.1*pr_range)
    
    # Enhanced legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
              fontsize=10, ncol=1, bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    return fig, ax


def analyze_compressor_map(map_data: Dict, compressor_name: str) -> Dict:
    """
    Analyze compressor map data and return performance summary.
    
    Parameters:
    -----------
    map_data : dict
        Dictionary containing map data arrays
    compressor_name : str
        Name of the compressor
        
    Returns:
    --------
    analysis : dict
        Dictionary with performance analysis results
    """
    
    # Extract data
    n_rot_grid = map_data['n_rot_grid']
    m_flow_grid = map_data['m_flow_grid']
    pr_results = map_data['pr_results']
    eta_results = map_data['eta_results']
    valid_results = map_data['valid_results']
    
    # Filter for valid points
    valid_mask = valid_results & (~np.isnan(pr_results)) & (~np.isnan(eta_results))
    m_valid = m_flow_grid[valid_mask]
    pr_valid = pr_results[valid_mask]
    eta_valid = eta_results[valid_mask]
    n_valid = n_rot_grid[valid_mask]
    
    if len(m_valid) == 0:
        return {"error": "No valid data points found"}
    
    # Calculate performance metrics
    best_eta_idx = np.argmax(eta_valid)
    
    analysis = {
        "compressor_name": compressor_name,
        "total_points": len(valid_results),
        "valid_points": np.sum(valid_mask),
        "convergence_rate": np.sum(valid_mask) / len(valid_results) * 100,
        
        "operating_envelope": {
            "mass_flow_range": [float(np.min(m_valid)), float(np.max(m_valid))],
            "pressure_ratio_range": [float(np.min(pr_valid)), float(np.max(pr_valid))],
            "efficiency_range": [float(np.min(eta_valid)), float(np.max(eta_valid))],
            "speed_range_krpm": [float(np.min(n_valid)*30/np.pi/1000), float(np.max(n_valid)*30/np.pi/1000)]
        },
        
        "best_efficiency_point": {
            "efficiency": float(eta_valid[best_eta_idx]),
            "mass_flow": float(m_valid[best_eta_idx]),
            "pressure_ratio": float(pr_valid[best_eta_idx]),
            "speed_krpm": float(n_valid[best_eta_idx]*30/np.pi/1000)
        }
    }
    
    return analysis


# ============================================================================
# PERFORMANCE MAP PLOTTING FOR CENTRIFUGAL COMPRESSOR
# ============================================================================

def plot_performance_map_from_data(
    perf_map,
    compressor_name: str = "Centrifugal Compressor",
    plot_type: str = "dual",
    design_point: Optional[Tuple[float, float, float]] = None,
    figsize: Tuple[int, int] = (14, 6),
    show_surge_line: bool = True,
    show_choke_line: bool = True,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
    """
    Plot performance map from PerformanceMap object.
    
    Parameters:
    -----------
    perf_map : PerformanceMap
        Performance map object from analysis.performance module
    compressor_name : str
        Name for plot title
    plot_type : str
        "dual" for PR and efficiency side-by-side, "enhanced" for single combined plot
    design_point : tuple, optional
        (mass_flow, pressure_ratio, efficiency) to highlight
    figsize : tuple
        Figure size
    show_surge_line : bool
        Draw surge line boundary
    show_choke_line : bool
        Draw choke line boundary
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig, axes : Figure and Axes objects
    
    Example:
    --------
    >>> from centrifugal_compressor.analysis.performance import generate_performance_map
    >>> perf_map = generate_performance_map(geom, op, loss_model="schiffmann")
    >>> fig, axes = plot_performance_map_from_data(perf_map, show_surge_line=True)
    >>> plt.show()
    """
    
    # Extract data from PerformanceMap object
    if plot_type == "dual":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Define colors and markers
    colors = ['#000000', '#0066CC', '#CC0000', '#00AA00', '#FF8800', '#8800CC', '#00CCCC', '#CC8800']
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p']
    
    # Plot each speed line
    surge_points_m = []
    surge_points_pr = []
    choke_points_m = []
    choke_points_pr = []
    
    for i, speed_line in enumerate(perf_map.speed_lines):
        # Get valid points
        valid_points = [pt for pt in speed_line.points if pt.is_valid]
        
        if len(valid_points) == 0:
            continue
        
        # Extract data
        mass_flows = [pt.mass_flow for pt in valid_points]
        pr_vals = [pt.pr_tt for pt in valid_points]
        eff_vals = [pt.eff_tt for pt in valid_points]
        
        # Sort by mass flow
        sort_idx = np.argsort(mass_flows)
        mass_flows = np.array(mass_flows)[sort_idx]
        pr_vals = np.array(pr_vals)[sort_idx]
        eff_vals = np.array(eff_vals)[sort_idx]
        
        # Select color and marker
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        label = f'{speed_line.speed_rpm:.0f} RPM'
        
        # Plot pressure ratio
        ax1.plot(mass_flows, pr_vals, marker=marker, color=color, linewidth=2.0,
                markersize=6, label=label, markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.5, alpha=0.8)
        
        # Plot efficiency (if dual mode)
        if ax2 is not None:
            ax2.plot(mass_flows, eff_vals, marker=marker, color=color, linewidth=2.0,
                    markersize=6, label=label, markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=1.5, alpha=0.8)
        
        # Collect surge and choke points
        if show_surge_line and speed_line.surge_point:
            surge_points_m.append(speed_line.surge_point.mass_flow)
            surge_points_pr.append(speed_line.surge_point.pr_tt)
        
        if show_choke_line and speed_line.choke_point:
            choke_points_m.append(speed_line.choke_point.mass_flow)
            choke_points_pr.append(speed_line.choke_point.pr_tt)
    
    # Draw surge line
    if show_surge_line and len(surge_points_m) > 1:
        surge_sort = np.argsort(surge_points_pr)
        surge_m_sorted = np.array(surge_points_m)[surge_sort]
        surge_pr_sorted = np.array(surge_points_pr)[surge_sort]
        ax1.plot(surge_m_sorted, surge_pr_sorted, '--', color='red', linewidth=3,
                alpha=0.7, label='Surge Line')
    
    # Draw choke line
    if show_choke_line and len(choke_points_m) > 1:
        choke_sort = np.argsort(choke_points_pr)
        choke_m_sorted = np.array(choke_points_m)[choke_sort]
        choke_pr_sorted = np.array(choke_points_pr)[choke_sort]
        ax1.plot(choke_m_sorted, choke_pr_sorted, '--', color='orange', linewidth=3,
                alpha=0.7, label='Choke Line')
    
    # Format axes
    ax1.set_xlabel('Mass Flow Rate, $\\dot{m}$ [kg/s]', fontsize=12)
    ax1.set_ylabel('Total Pressure Ratio, $PR_{tt}$ [-]', fontsize=12)
    ax1.set_title(f'{compressor_name} - Pressure Ratio Map', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    if ax2 is not None:
        ax2.set_xlabel('Mass Flow Rate, $\\dot{m}$ [kg/s]', fontsize=12)
        ax2.set_ylabel('Total-to-Total Efficiency, $\\eta_{tt}$ [-]', fontsize=12)
        ax2.set_title(f'{compressor_name} - Efficiency Map', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Highlight design point
    if design_point:
        design_m, design_pr, design_eta = design_point
        ax1.scatter([design_m], [design_pr], s=200, c='red', marker='*',
                   edgecolor='black', linewidth=3, zorder=15, label='Design Point')
        if ax2 is not None:
            ax2.scatter([design_m], [design_eta], s=200, c='red', marker='*',
                       edgecolor='black', linewidth=3, zorder=15, label='Design Point')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, (ax1, ax2) if ax2 is not None else ax1


def plot_phi_psi_map(
    perf_map,
    compressor_name: str = "Centrifugal Compressor",
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot dimensionless performance map (φ-ψ diagram).
    
    Parameters:
    -----------
    perf_map : PerformanceMap
        Performance map object
    compressor_name : str
        Name for plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig, ax : Figure and Axes objects
    
    Example:
    --------
    >>> fig, ax = plot_phi_psi_map(perf_map)
    >>> plt.show()
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all valid points
    phi_vals = []
    psi_vals = []
    eff_vals = []
    
    for speed_line in perf_map.speed_lines:
        for pt in speed_line.points:
            if pt.is_valid:
                phi_vals.append(pt.phi)
                psi_vals.append(pt.psi)
                eff_vals.append(pt.eff_tt)
    
    if len(phi_vals) == 0:
        print("Warning: No valid points to plot")
        return fig, ax
    
    phi_vals = np.array(phi_vals)
    psi_vals = np.array(psi_vals)
    eff_vals = np.array(eff_vals)
    
    # Scatter plot colored by efficiency
    scatter = ax.scatter(phi_vals, psi_vals, c=eff_vals, s=50, cmap='viridis',
                        edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Efficiency $\\eta_{tt}$ [-]', fontsize=11)
    
    # Find and mark peak efficiency point
    peak_idx = np.argmax(eff_vals)
    ax.scatter(phi_vals[peak_idx], psi_vals[peak_idx], s=300, marker='*',
              c='red', edgecolors='black', linewidth=2, zorder=10, label='Peak Efficiency')
    
    # Format
    ax.set_xlabel('Flow Coefficient, $\\phi$ [-]', fontsize=12)
    ax.set_ylabel('Head Coefficient, $\\psi$ [-]', fontsize=12)
    ax.set_title(f'{compressor_name} - Dimensionless Performance ($\\phi$-$\\psi$)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_surge_margin_analysis(
    perf_map,
    compressor_name: str = "Centrifugal Compressor",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot surge margin across operating envelope.
    
    Parameters:
    -----------
    perf_map : PerformanceMap
        Performance map object with surge analysis
    compressor_name : str
        Name for plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig, ax : Figure and Axes objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect points with surge margin data
    mass_flows = []
    surge_margins = []
    speeds = []
    
    for speed_line in perf_map.speed_lines:
        for pt in speed_line.points:
            if pt.is_valid and not np.isnan(pt.surge_margin):
                mass_flows.append(pt.mass_flow)
                surge_margins.append(pt.surge_margin)
                speeds.append(speed_line.speed_rpm)
    
    if len(mass_flows) == 0:
        print("Warning: No surge margin data available")
        return fig, ax
    
    mass_flows = np.array(mass_flows)
    surge_margins = np.array(surge_margins)
    speeds = np.array(speeds)
    
    # Scatter plot colored by speed
    scatter = ax.scatter(mass_flows, surge_margins, c=speeds, s=40, cmap='plasma',
                        edgecolors='black', linewidth=0.5, alpha=0.7)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed [RPM]', fontsize=11)
    
    # Add zero line (surge boundary)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Surge Limit')
    
    # Format
    ax.set_xlabel('Mass Flow Rate, $\\dot{m}$ [kg/s]', fontsize=12)
    ax.set_ylabel('Surge Margin [%]', fontsize=12)
    ax.set_title(f'{compressor_name} - Surge Margin Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax
