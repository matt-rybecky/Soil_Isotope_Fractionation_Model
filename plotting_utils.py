"""
Plotting Utilities for Soil Evaporation Model
============================================

Reusable plotting functions and visualization components for the soil evaporation
application. Provides standardized plot creation with consistent styling.

Author: Enhanced Python Implementation for Soil Evaporation Research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from functools import wraps


class PlotStyle:
    """Professional black and white plotting style constants."""
    
    # Professional black and white styling
    BLACK = '#000000'
    DARK_GRAY = '#333333'
    MEDIUM_GRAY = '#666666'
    LIGHT_GRAY = '#999999'
    ERROR_COLOR = '#000000'  # Keep error messages black
    
    # Line styles for multiple series (all black/gray)
    LINE_STYLES = ['-', '--', '-.', ':']
    LINE_WIDTHS = [1.5, 1.5, 1.5, 1.5]
    MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    MARKER_COLORS = ['white', 'white', 'white', 'white']
    MARKER_EDGE_COLORS = ['black', 'black', 'black', 'black']
    
    # Standard settings
    LINE_WIDTH = 1.5
    MARKER_SIZE = 6
    MARKER_EDGE_WIDTH = 1.0
    
    # Font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10
    
    # Figure settings
    DPI = 100
    FIGURE_SIZE = (10, 6)  # Reduced from (12, 8) to prevent cutoff
    
    # Heatmap settings
    HEATMAP_COLORMAP = 'viridis'
    
    @classmethod
    def get_line_style(cls, index: int) -> Dict[str, Any]:
        """Get line style for multiple series plots."""
        i = index % len(cls.LINE_STYLES)
        return {
            'color': cls.BLACK,
            'linestyle': cls.LINE_STYLES[i],
            'linewidth': cls.LINE_WIDTHS[i]
        }
    
    @classmethod
    def get_marker_style(cls, index: int) -> Dict[str, Any]:
        """Get hollow marker style for scatter plots."""
        i = index % len(cls.MARKER_STYLES)
        return {
            'marker': cls.MARKER_STYLES[i],
            'facecolor': cls.MARKER_COLORS[i],
            'edgecolor': cls.MARKER_EDGE_COLORS[i],
            'markersize': cls.MARKER_SIZE,
            'markeredgewidth': cls.MARKER_EDGE_WIDTH
        }


# ================================
# Utility Functions
# ================================

def plot_error_handler(func):
    """Decorator for consistent error handling in plot methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            self.figure.clear()
            return func(self, *args, **kwargs)
        except Exception as e:
            self._show_error_plot(f"Error in {func.__name__}: {str(e)}")
            return False
    return wrapper

def validate_plot_data(results: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, str]:
    """Validate that required data is present and non-empty."""
    if not all(key in results for key in required_keys):
        missing = [key for key in required_keys if key not in results]
        return False, f"Missing required data: {', '.join(missing)}"
    
    for key in required_keys:
        data = results[key]
        if isinstance(data, (list, np.ndarray)) and len(data) == 0:
            return False, f"No data available for {key}"
    
    return True, ""

def extract_final_profiles(results: Dict[str, Any], variable_names: List[str]) -> Dict[str, np.ndarray]:
    """Extract final timestep data for multiple variables."""
    profiles = {}
    for var_name in variable_names:
        if var_name in results:
            data = np.array(results[var_name])
            if data.ndim > 1:
                profiles[var_name] = data[-1, :]  # Last timestep
            else:
                profiles[var_name] = data
    return profiles

def setup_depth_plot_styling(ax, xlabel: str, ylabel: str = "Depth (cm)", title: str = ""):
    """Apply consistent styling to depth profile plots."""
    ax.set_xlabel(xlabel, fontsize=PlotStyle.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=PlotStyle.LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=PlotStyle.TITLE_SIZE)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def add_statistics_box(ax, stats_text: str, position: str = 'top'):
    """Add statistics text box to plot."""
    y_pos = 0.98 if position == 'top' else 0.02
    va = 'top' if position == 'top' else 'bottom'
    
    ax.text(0.02, y_pos, stats_text, transform=ax.transAxes,
           verticalalignment=va, 
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', alpha=0.8))

def create_subplot_grid(num_plots: int):
    """Create optimal subplot grid for given number of plots."""
    if num_plots == 1:
        return (1, 1)
    elif num_plots == 2:
        return (1, 2)
    elif num_plots <= 4:
        return (2, 2) if num_plots > 3 else (1, 3)
    else:
        # For more than 4 plots, use a roughly square grid
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = int(np.ceil(num_plots / rows))
        return (rows, cols)

class SoilEvaporationPlotter:
    """
    Professional black and white plotting for soil evaporation model results.
    
    Consolidates plot creation with consistent styling and error handling.
    """
    
    def __init__(self, figure: Figure = None):
        """
        Initialize plotter with optional figure.
        
        Args:
            figure: Matplotlib Figure object. If None, creates new figure.
        """
        if figure is None:
            self.figure = Figure(figsize=PlotStyle.FIGURE_SIZE, dpi=PlotStyle.DPI)
        else:
            self.figure = figure
        
        self.current_plot_type = None
        self.current_data = None
    
    @plot_error_handler
    def create_depth_profiles_plot(self, model_data: Dict[str, Any]) -> bool:
        """
        Create depth profiles plot for water content, temperature, and isotopes.
        
        Args:
            model_data: Model results dictionary
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Validate required data
        required_keys = ['water_content', 'depth_nodes']
        is_valid, error_msg = validate_plot_data(results, required_keys)
        if not is_valid:
            self._show_error_plot(error_msg)
            return False
        
        depths = np.array(results['depth_nodes'])
        
        # Determine available variables
        available_vars = ['water_content']
        var_configs = [
            ('water_content', 'Water Content (cm³/cm³)', 'Water Content Profile'),
        ]
        
        if 'temperature' in results and len(results['temperature']) > 0:
            available_vars.append('temperature')
            var_configs.append(('temperature', 'Temperature (°C)', 'Temperature Profile'))
        
        if 'd18O' in results and len(results['d18O']) > 0:
            available_vars.append('d18O')
            var_configs.append(('d18O', 'δ¹⁸O (‰)', 'δ¹⁸O Profile'))
        
        if 'delta_prime' in results and len(results['delta_prime']) > 0:
            available_vars.append('delta_prime')
            var_configs.append(('delta_prime', "Δ'¹⁷O (per meg)", "Δ'¹⁷O Profile"))
        
        # Extract final profiles
        profiles = extract_final_profiles(results, available_vars)
        
        # Create subplot grid
        num_plots = len(var_configs)
        rows, cols = create_subplot_grid(num_plots)
        
        # Create plots
        for i, (var_name, xlabel, title) in enumerate(var_configs):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            
            if var_name in profiles:
                # Use professional black and white styling
                marker_style = PlotStyle.get_marker_style(i)
                ax.plot(profiles[var_name], depths, 
                       color=PlotStyle.BLACK, 
                       linewidth=PlotStyle.LINE_WIDTH,
                       linestyle='-',
                       marker=marker_style['marker'],
                       markerfacecolor=marker_style['facecolor'],
                       markeredgecolor=marker_style['edgecolor'],
                       markersize=marker_style['markersize'],
                       markeredgewidth=marker_style['markeredgewidth'])
                
                setup_depth_plot_styling(ax, xlabel, "Depth (cm)", title)
        
        self.figure.suptitle('Final Depth Profiles', 
                           fontsize=PlotStyle.TITLE_SIZE + 2, 
                           fontweight='bold')
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = "depth_profiles"
        self.current_data = model_data
        
        return True
    
    @plot_error_handler
    def create_time_series_plot(self, model_data: Dict[str, Any]) -> bool:
        """
        Create time series plot for surface evolution.
        
        Args:
            model_data: Model results dictionary
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Validate required data
        required_keys = ['water_content', 'd18O', 'delta_prime', 'times_days']
        is_valid, error_msg = validate_plot_data(results, required_keys)
        if not is_valid:
            self._show_error_plot(error_msg)
            return False
        
        # Extract data
        water_content = np.array(results['water_content'])
        d18O = np.array(results['d18O'])
        delta_prime = np.array(results['delta_prime'])
        times = np.array(results['times_days'])
        
        # Surface values (first depth index)
        surface_data = [
            (water_content[:, 0], 'Water Content\n(cm³/cm³)'),
            (d18O[:, 0], 'δ¹⁸O (‰)'),
            (delta_prime[:, 0], "Δ'¹⁷O\n(per meg)")
        ]
        
        # Create subplots
        for i, (data, ylabel) in enumerate(surface_data):
            ax = self.figure.add_subplot(3, 1, i + 1)
            
            # Professional black and white styling
            line_style = PlotStyle.get_line_style(i)
            ax.plot(times, data, **line_style)
            
            ax.set_ylabel(ylabel, fontsize=PlotStyle.LABEL_SIZE)
            ax.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
            
            if i == 0:
                ax.set_title('Surface Evolution Over Time', fontsize=PlotStyle.TITLE_SIZE)
            if i == len(surface_data) - 1:
                ax.set_xlabel('Time (days)', fontsize=PlotStyle.LABEL_SIZE)
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = "time_series"
        self.current_data = model_data
        
        return True
    
    
    
    @plot_error_handler
    def create_isotope_evolution_plot(self, model_data: Dict[str, Any]) -> bool:
        """
        Create isotope evolution plot showing d18O vs delta_prime.
        
        Args:
            model_data: Model results dictionary
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Validate required data
        required_keys = ['d18O', 'delta_prime', 'times_days']
        is_valid, error_msg = validate_plot_data(results, required_keys)
        if not is_valid:
            self._show_error_plot(error_msg)
            return False
        
        d18O = np.array(results['d18O'])
        delta_prime = np.array(results['delta_prime'])
        times = np.array(results['times_days'])
        
        # Create plot
        ax = self.figure.add_subplot(111)
        
        # Surface values evolution
        surface_d18O = d18O[:, 0]
        surface_delta_prime = delta_prime[:, 0]
        
        # Professional black and white styling
        # Scatter plot with viridis colormap for time progression
        scatter = ax.scatter(surface_d18O, surface_delta_prime, 
                           c=times, cmap=PlotStyle.HEATMAP_COLORMAP, 
                           s=30, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # Add connecting line
        ax.plot(surface_d18O, surface_delta_prime, 
               color=PlotStyle.BLACK, alpha=0.3, linewidth=1, zorder=1)
        
        # Mark start and end points with hollow markers
        start_marker = PlotStyle.get_marker_style(0)
        end_marker = PlotStyle.get_marker_style(1)
        
        ax.plot(surface_d18O[0], surface_delta_prime[0], 
               label='Start', linestyle='None',
               marker=start_marker['marker'],
               markerfacecolor=start_marker['facecolor'],
               markeredgecolor=start_marker['edgecolor'],
               markersize=8,
               markeredgewidth=start_marker['markeredgewidth'],
               zorder=3)
        ax.plot(surface_d18O[-1], surface_delta_prime[-1], 
               label='End', linestyle='None',
               marker=end_marker['marker'],
               markerfacecolor=end_marker['facecolor'],
               markeredgecolor=end_marker['edgecolor'],
               markersize=8,
               markeredgewidth=end_marker['markeredgewidth'],
               zorder=3)
        
        ax.set_xlabel('δ¹⁸O (‰)', fontsize=PlotStyle.LABEL_SIZE)
        ax.set_ylabel("Δ'¹⁷O (per meg)", fontsize=PlotStyle.LABEL_SIZE)
        ax.set_title('Surface Isotope Evolution Path', fontsize=PlotStyle.TITLE_SIZE)
        ax.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
        ax.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        # Add colorbar
        cbar = self.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Time (days)', fontsize=PlotStyle.LABEL_SIZE)
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = "isotope_evolution"
        self.current_data = model_data
        
        return True
    
    def _show_error_plot(self, error_message: str):
        """
        Show error message on plot with professional styling.
        
        Args:
            error_message: Error message to display
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f"Error: {error_message}", 
               ha='center', va='center', fontsize=PlotStyle.LABEL_SIZE,
               color=PlotStyle.ERROR_COLOR, wrap=True,
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
    
    def save_current_plot(self, filepath: Path, dpi: int = 300, format: str = 'png') -> bool:
        """
        Save current plot to file.
        
        Args:
            filepath: Path to save file
            dpi: Resolution for saved image
            format: Image format ('png', 'pdf', 'svg')
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self.figure.savefig(filepath, dpi=dpi, format=format, 
                              bbox_inches='tight', facecolor='white')
            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False
    
    @plot_error_handler
    def create_mass_balance_plot(self, model_data: Dict[str, Any]) -> bool:
        """
        Create mass balance tracking plot showing mass changes over time.
        
        Shows time series of cumulative mass tracking:
        actual mass lost from the soil column.
        
        Args:
            model_data: Model results dictionary with standardized format
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Check for mass balance data
        if 'mass_balance_data' not in results:
            self._show_error_plot("No mass balance data available")
            return False
        
        mb_data = results['mass_balance_data']
        
        # Validate required mass balance data  
        required_keys = ['times', 'total_mass', 'mass_lost', 'evaporation_mm']
        missing_keys = [key for key in required_keys if key not in mb_data]
        if missing_keys:
            self._show_error_plot(f"Missing mass balance data: {missing_keys}")
            return False
        
        # Extract mass balance data
        times = np.array(mb_data['times'])
        total_mass = np.array(mb_data['total_mass'])          # Current total mass in soil
        mass_lost = np.array(mb_data['mass_lost'])            # Cumulative mass lost (evaporated)
        evaporation_mm = np.array(mb_data['evaporation_mm'])  # Evaporation in mm equivalent
        
        if len(times) < 2:
            self._show_error_plot("Insufficient mass balance data")
            return False
        
        # Calculate instantaneous evaporation rate (mm/day)
        dt = np.diff(times)
        dt[dt == 0] = 1e-10  # Avoid division by zero
        
        evap_rate = np.zeros_like(times)
        if len(times) > 1:
            evap_rate[1:] = np.diff(evaporation_mm) / dt  # mm/day
        
        # Create subplots
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        
        # Plot total mass remaining in soil
        mass_style = PlotStyle.get_line_style(0)
        evap_style = PlotStyle.get_line_style(1)
        
        ax1.plot(times, total_mass, label='Total Mass in Soil', **mass_style)
        ax1.set_ylabel('Total Mass (mol)', fontsize=PlotStyle.LABEL_SIZE)
        ax1.set_title('Soil Water Mass Evolution', fontsize=PlotStyle.TITLE_SIZE)
        ax1.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
        ax1.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        # Plot evaporation 
        ax2.plot(times, evaporation_mm, label='Cumulative Evaporation', **evap_style)
        ax2.set_ylabel('Cumulative Evaporation (mm)', fontsize=PlotStyle.LABEL_SIZE)
        ax2.set_xlabel('Time (days)', fontsize=PlotStyle.LABEL_SIZE)
        ax2.set_title('Total Evaporation from Mass Balance', fontsize=PlotStyle.TITLE_SIZE)
        ax2.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
        ax2.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        # Add summary text
        if len(times) > 0:
            final_evap = evaporation_mm[-1]
            max_rate = np.max(evap_rate) if len(evap_rate) > 0 else 0
            
            # Add summary text box
            summary_text = f"Total Evaporation: {final_evap:.2f} mm\nMax Rate: {max_rate:.3f} mm/day"
            ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
                    fontsize=PlotStyle.LEGEND_SIZE, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = "mass_balance"
        self.current_data = model_data
        
        return True
    
    @plot_error_handler
    def create_cumulative_evaporation_plot(self, model_data: Dict[str, Any]) -> bool:
        """
        Create cumulative evaporation plot with instantaneous rate.
        
        Args:
            model_data: Model results dictionary with standardized format
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Validate required data
        required_keys = ['water_content', 'times_days', 'depth_nodes']
        is_valid, error_msg = validate_plot_data(results, required_keys)
        if not is_valid:
            self._show_error_plot(error_msg)
            return False
        
        # Extract data
        water_content = np.array(results['water_content'])
        times_days = np.array(results['times_days'])
        depth_nodes = np.array(results['depth_nodes'])
        
        if len(water_content) < 2:
            self._show_error_plot("Insufficient time series data for evaporation calculation")
            return False
        
        # Calculate evaporation metrics
        depth_step = depth_nodes[1] - depth_nodes[0] if len(depth_nodes) > 1 else 1.0
        total_water = np.sum(water_content * depth_step, axis=1)
        cumulative_evap = total_water[0] - total_water
        
        dt = np.diff(times_days)
        dt[dt == 0] = 1e-10
        evap_rate = np.diff(cumulative_evap) / dt
        
        # Create subplots
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        
        # Professional black and white styling
        cumulative_style = PlotStyle.get_line_style(0)
        cumulative_style['linewidth'] = PlotStyle.LINE_WIDTH * 1.5
        
        rate_style = PlotStyle.get_line_style(1)
        
        # Cumulative evaporation plot
        ax1.plot(times_days, cumulative_evap, label='Cumulative Evaporation', **cumulative_style)
        ax1.set_ylabel('Cumulative Evaporation (cm)', fontsize=PlotStyle.LABEL_SIZE)
        ax1.set_title('Cumulative Evaporation', fontsize=PlotStyle.TITLE_SIZE)
        ax1.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
        ax1.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        # Evaporation rate plot
        ax2.plot(times_days[1:], evap_rate, label='Evaporation Rate', **rate_style)
        ax2.set_xlabel('Time (days)', fontsize=PlotStyle.LABEL_SIZE)
        ax2.set_ylabel('Evaporation Rate (cm/day)', fontsize=PlotStyle.LABEL_SIZE)
        ax2.set_title('Instantaneous Evaporation Rate', fontsize=PlotStyle.TITLE_SIZE)
        ax2.grid(True, alpha=0.3, color=PlotStyle.LIGHT_GRAY)
        ax2.legend(fontsize=PlotStyle.LEGEND_SIZE)
        
        # Add statistics
        if len(cumulative_evap) > 0:
            final_evap = cumulative_evap[-1]
            avg_rate = final_evap / max(times_days[-1], 1e-6) if times_days[-1] > 0 else 0
            max_rate = np.max(evap_rate) if len(evap_rate) > 0 else 0
            
            stats_text = (f'Total: {final_evap:.4f} cm | '
                         f'Avg Rate: {avg_rate:.4f} cm/day | '
                         f'Max Rate: {max_rate:.4f} cm/day')
            add_statistics_box(ax1, stats_text)
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = "cumulative_evaporation"
        self.current_data = model_data
        
        return True
    
    @plot_error_handler
    def create_evolution_heatmap(self, model_data: Dict[str, Any], data_type: str) -> bool:
        """
        Create unified evolution heatmap for any variable type.
        
        Args:
            model_data: Model results dictionary with standardized format
            data_type: Variable type ('water_content', 'temperature', 'd18O', 'delta_prime')
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        # Define variable configurations
        var_configs = {
            'water_content': {
                'title': 'Soil Water Content Evolution',
                'label': 'Water Content (cm³/cm³)',
                'units': 'cm³/cm³',
                'precision': 4
            },
            'temperature': {
                'title': 'Temperature Evolution in Soil Column',
                'label': 'Temperature (°C)',
                'units': '°C',
                'precision': 2
            },
            'd18O': {
                'title': 'δ¹⁸O Evolution in Soil Column',
                'label': 'δ¹⁸O (‰)',
                'units': '‰',
                'precision': 3
            },
            'delta_prime': {
                'title': "Δ'¹⁷O Evolution in Soil Column",
                'label': "Δ'¹⁷O (per meg)",
                'units': 'per meg',
                'precision': 3
            }
        }
        
        if data_type not in var_configs:
            self._show_error_plot(f"Unsupported data type: {data_type}")
            return False
        
        config = var_configs[data_type]
        
        # Validate required data
        required_keys = [data_type, 'times_days', 'depth_nodes']
        is_valid, error_msg = validate_plot_data(results, required_keys)
        if not is_valid:
            self._show_error_plot(error_msg)
            return False
        
        # Extract data
        data = np.array(results[data_type])  # Shape: (n_times, n_depths)
        times_days = np.array(results['times_days'])
        depth_nodes = np.array(results['depth_nodes'])
        
        # Create heatmap
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Create meshgrid and transpose data for correct orientation
        T, D = np.meshgrid(times_days, depth_nodes)
        data_T = data.T  # (depths x times)
        
        # Create heatmap with viridis colormap
        im = ax.pcolormesh(T, D, data_T, shading='auto', cmap=PlotStyle.HEATMAP_COLORMAP)
        
        # Professional formatting
        ax.set_xlabel('Time (days)', fontsize=PlotStyle.LABEL_SIZE)
        ax.set_ylabel('Depth (cm)', fontsize=PlotStyle.LABEL_SIZE)
        ax.set_title(config['title'], fontsize=PlotStyle.TITLE_SIZE)
        ax.invert_yaxis()  # Depth increases downward
        
        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label(config['label'], fontsize=PlotStyle.LABEL_SIZE)
        
        # Add statistics
        initial_values = np.mean(data[0, :])
        final_values = np.mean(data[-1, :])
        change = final_values - initial_values
        precision = config['precision']
        
        if data_type == 'water_content':
            stats_text = (f'Initial: {initial_values:.{precision}f} | '
                         f'Final: {final_values:.{precision}f} | '
                         f'Loss: {abs(change):.{precision}f} {config["units"]}')
        else:
            stats_text = (f'Initial: {initial_values:.{precision}f} | '
                         f'Final: {final_values:.{precision}f} | '
                         f'Change: {change:+.{precision}f} {config["units"]}')
        
        add_statistics_box(ax, stats_text)
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = f"{data_type}_evolution_heatmap"
        self.current_data = model_data
        
        return True
    
    # Convenience methods for backward compatibility
    def create_soil_drying_heatmap(self, model_data: Dict[str, Any]) -> bool:
        """Create soil water content evolution heatmap."""
        return self.create_evolution_heatmap(model_data, 'water_content')
    
    def create_isotope_evolution_heatmap(self, model_data: Dict[str, Any], isotope_type: str = 'd18O') -> bool:
        """Create isotope evolution heatmap."""
        return self.create_evolution_heatmap(model_data, isotope_type)
    
    def create_temperature_evolution_heatmap(self, model_data: Dict[str, Any]) -> bool:
        """Create temperature evolution heatmap."""
        return self.create_evolution_heatmap(model_data, 'temperature')

    @plot_error_handler
    def create_single_variable_depth_plot(self, model_data: Dict[str, Any], variable_name: str, 
                                        ylabel: str, units: str, title_prefix: str = "") -> bool:
        """
        Create single variable vs depth plot.
        
        Args:
            model_data: Model results dictionary
            variable_name: Name of variable to plot
            ylabel: Y-axis label
            units: Units for the variable
            title_prefix: Optional prefix for title
            
        Returns:
            True if plot created successfully, False otherwise
        """
        results = model_data.get('results', {})
        
        if not results or 'depth_nodes' not in results:
            self._show_error_plot(f"Missing required data for {ylabel} plot")
            return False
        
        depth = np.array(results['depth_nodes'])
        
        # Extract data (try final state first, then time series)
        plot_data = None
        if f'final_{variable_name}' in results:
            plot_data = np.array(results[f'final_{variable_name}'])
        elif variable_name in results:
            data = np.array(results[variable_name])
            plot_data = data[-1, :] if data.ndim > 1 else data
        
        if plot_data is None:
            self._show_error_plot(f"No {ylabel} data available")
            return False
        
        # Create professional depth plot
        ax = self.figure.add_subplot(111)
        
        # Professional black and white styling with hollow markers
        marker_style = PlotStyle.get_marker_style(0)
        ax.plot(plot_data, depth, 
               color=PlotStyle.BLACK, 
               linewidth=PlotStyle.LINE_WIDTH,
               marker=marker_style['marker'],
               markerfacecolor=marker_style['facecolor'],
               markeredgecolor=marker_style['edgecolor'],
               markersize=marker_style['markersize'],
               markeredgewidth=marker_style['markeredgewidth'])
        
        # Format labels
        xlabel = f'{ylabel} ({units})' if units else ylabel
        title = f'{title_prefix} - {ylabel} Profile' if title_prefix else f'{ylabel} Profile'
        
        setup_depth_plot_styling(ax, xlabel, "Depth (cm)", title)
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = f"{variable_name}_depth_profile"
        self.current_data = model_data
        
        return True
    
    @plot_error_handler
    def create_multi_run_comparison_plot(self, runs_data: List[Dict[str, Any]], run_labels: List[str], 
                                       variable_name: str, ylabel: str, units: str, 
                                       overlay_data=None) -> bool:
        """
        Create comparison plot for multiple model runs with optional overlay data.
        
        Args:
            runs_data: List of model results dictionaries
            run_labels: List of labels for each run
            variable_name: Name of variable to plot
            ylabel: Y-axis label
            units: Units for the variable
            overlay_data: Optional experimental data to overlay
            
        Returns:
            True if plot created successfully, False otherwise
        """
        if not runs_data or not run_labels:
            self._show_error_plot(f"No data provided for {ylabel} comparison")
            return False
        
        # Extract depth data from first run (assume all have same depth grid)
        first_results = runs_data[0]
        if 'depth_nodes' not in first_results:
            self._show_error_plot(f"Missing depth data for {ylabel} comparison")
            return False
        
        depth = np.array(first_results['depth_nodes'])
        
        # Create plot
        ax = self.figure.add_subplot(111)
        
        # Plot model runs with different line styles
        for i, (results, label) in enumerate(zip(runs_data, run_labels)):
            # Extract variable data
            plot_data = None
            if f'final_{variable_name}' in results:
                plot_data = np.array(results[f'final_{variable_name}'])
            elif variable_name in results:
                data = np.array(results[variable_name])
                plot_data = data[-1, :] if data.ndim > 1 else data
            
            if plot_data is not None:
                # Use different line styles for different runs
                line_style = PlotStyle.get_line_style(i)
                
                # When overlay is enabled, use lines only (no markers)
                if overlay_data is not None:
                    ax.plot(plot_data, depth,
                           color=line_style['color'],
                           linestyle=line_style['linestyle'],
                           linewidth=line_style['linewidth'],
                           label=label)
                else:
                    # Include markers when no overlay
                    marker_style = PlotStyle.get_marker_style(i)
                    ax.plot(plot_data, depth,
                           color=line_style['color'],
                           linestyle=line_style['linestyle'],
                           linewidth=line_style['linewidth'],
                           marker=marker_style['marker'],
                           markerfacecolor=marker_style['facecolor'],
                           markeredgecolor=marker_style['edgecolor'],
                           markersize=marker_style['markersize'],
                           markeredgewidth=marker_style['markeredgewidth'],
                           label=label)
        
        # Add overlay data if provided
        if overlay_data is not None:
            self._plot_overlay_data(ax, overlay_data, variable_name, depth)
        
        # Format labels and styling
        xlabel = f'{ylabel} ({units})' if units else ylabel
        title = f'{ylabel} Profile Comparison'
        if overlay_data is not None:
            title += ' with Data Overlay'
        
        setup_depth_plot_styling(ax, xlabel, "Depth (cm)", title)
        
        # Add legend
        ax.legend(frameon=True, fancybox=False, shadow=False, 
                 framealpha=1.0, edgecolor='black', fontsize='small')
        
        self.figure.tight_layout(pad=2.5)  # Increased padding to prevent cutoff
        self.figure.subplots_adjust(bottom=0.15, top=0.90, left=0.12, right=0.95)  # Extra margin control
        
        self.current_plot_type = f"{variable_name}_multi_run_comparison"
        self.current_data = {'runs_data': runs_data, 'overlay_data': overlay_data}
        
        return True
    
    def _plot_overlay_data(self, ax, overlay_data, variable_name, model_depth):
        """Plot experimental overlay data with color styling for multiple datasets."""
        try:
            import pandas as pd
            
            # Handle multiple overlay datasets
            if not overlay_data:
                return
            
            # Define colors for overlay data (different from black/white model data)
            overlay_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # If overlay_data is a list of dataset dictionaries
            if isinstance(overlay_data, list):
                datasets = overlay_data
            else:
                # Legacy support - single dataset as DataFrame
                datasets = [{'filename': 'Data', 'data': overlay_data}]
            
            for i, dataset in enumerate(datasets):
                # Extract data and filename
                if isinstance(dataset, dict) and 'data' in dataset:
                    data = dataset['data']
                    filename = dataset.get('filename', f'Dataset {i+1}')
                else:
                    # Legacy support
                    data = dataset
                    filename = f'Dataset {i+1}'
                
                # Convert to DataFrame if not already
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                
                # Map variable names for overlay data
                overlay_variable_name = variable_name
                if variable_name == 'delta_prime':  # Model uses 'delta_prime' for Δ'¹⁷O
                    overlay_variable_name = 'D17O'  # Overlay data uses 'D17O'
                
                # Check required columns
                required_cols = ['depth', overlay_variable_name]
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"Warning: Overlay data '{filename}' missing columns: {missing_cols}")
                    continue
                
                # Extract overlay data
                overlay_depth = data['depth'].values
                overlay_values = data[overlay_variable_name].values
                
                # Use different color for each dataset
                color = overlay_colors[i % len(overlay_colors)]
                
                # Plot with colored markers only (no connecting lines)
                ax.scatter(overlay_values, overlay_depth,
                          color=color,
                          s=60,  # Larger marker size (s is area, so 60 is quite large)
                          marker='o',
                          edgecolors='white',
                          linewidth=1.5,
                          label=f'{filename}',
                          alpha=0.9,
                          zorder=10)  # Ensure overlay points are on top
                   
        except Exception as e:
            print(f"Warning: Could not plot overlay data: {e}")


class PlotCanvas:
    """Wrapper for matplotlib canvas in tkinter with navigation toolbar."""
    
    def __init__(self, parent: tk.Widget, figure: Figure):
        """
        Initialize plot canvas.
        
        Args:
            parent: Parent tkinter widget
            figure: Matplotlib Figure object
        """
        self.parent = parent
        self.figure = figure
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(figure, parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        
        # Create navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
    
    def pack(self, **kwargs):
        """Pack canvas and toolbar."""
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, **kwargs)
    
    def grid(self, row: int, column: int = 0, **kwargs):
        """Grid canvas and toolbar."""
        # Remove sticky from kwargs to avoid conflicts
        kwargs.pop('sticky', None)
        self.toolbar.grid(row=row, column=column, sticky="ew", **kwargs)
        self.canvas_widget.grid(row=row+1, column=column, 
                               sticky="nsew", **kwargs)
    
    def destroy(self):
        """Clean up canvas and toolbar."""
        self.toolbar.destroy()
        self.canvas_widget.destroy()
    
    def draw(self):
        """Refresh the canvas."""
        self.canvas.draw()


