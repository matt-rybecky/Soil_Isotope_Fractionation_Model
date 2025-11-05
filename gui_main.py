#!/usr/bin/env python3
"""
Soil Evaporation Model - Main GUI Application

Interactive desktop application for soil water evaporation modeling with 
triple oxygen isotope tracking. Supports single model runs with visualization.

Authors: Dan Breecker (UT Austin), Matthew Rybecky (UNM), Catt Peshek (UNM)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')

from soil_model import SoilEvaporationModel
from gui_utils import DataPathManager, GUIUtils, StandardStyles
from plotting_utils import SoilEvaporationPlotter, PlotCanvas, PlotStyle
from config_manager import ConfigurationManager, GUIConfigManager, ModelConfiguration, load_config, ERA5Forcing
from output_data_manager import OutputDataManager


class SoilEvaporationApp:
    """Main soil evaporation model application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Soil Evaporation Model")
        self.root.geometry("1400x1000")
        self.root.minsize(1200, 900)  # Ensure minimum size for proper plot display
        
        # Initialize core components
        self.output_dir = Path(__file__).parent / 'output_results'
        self.output_dir.mkdir(exist_ok=True)
        
        self.path_manager = DataPathManager()
        self.config_manager = ConfigurationManager()
        self.gui_config_manager = GUIConfigManager(self.config_manager)
        self.config = self.config_manager.current_config
        self.output_data_manager = OutputDataManager(self.output_dir)
        
        # Initialize plotting
        self.results_figure = Figure(figsize=PlotStyle.FIGURE_SIZE, dpi=PlotStyle.DPI)
        self.plotter = SoilEvaporationPlotter(self.results_figure)
        
        # State variables
        self.model_results = None
        self.current_model = None
        
        self.create_widgets()
        self.refresh_data_sources()
    
    def refresh_data_sources(self):
        """Refresh ERA5 files and available runs."""
        self._refresh_era5_files()
        self._refresh_available_runs()
    
    def _refresh_era5_files(self):
        """Refresh ERA5 file list."""
        era5_files = []
        era5_data_dir = Path(__file__).parent / "ERA5_DATA"
        
        if era5_data_dir.exists():
            era5_files.extend([f.name for f in era5_data_dir.glob("*.csv")])
        
        # Check current directory
        current_dir = Path(__file__).parent
        era5_files.extend([f.name for f in current_dir.glob("era5*.csv")])
        era5_files.extend([f.name for f in current_dir.glob("sample_era5*.csv")])
        
        era5_files = sorted(list(set(era5_files)))
        
        if era5_files:
            self.era5_file_combo['values'] = era5_files
            if self.era5_file_var.get() not in era5_files:
                self.era5_file_var.set(era5_files[0])
        else:
            self.era5_file_combo['values'] = ["No ERA5 files found"]
            self.era5_file_var.set("No ERA5 files found")
    
    def _refresh_available_runs(self):
        """Refresh available model runs."""
        try:
            available_runs = self.output_data_manager.get_available_runs()
            
            if available_runs:
                run_options = []
                self.run_lookup = {}
                
                for i, run_info in enumerate(available_runs, 1):  # Start numbering from 1
                    # Load metadata for display
                    run_info = self.output_data_manager.load_run_metadata(run_info)
                    
                    if run_info.get('metadata'):
                        metadata = run_info['metadata']
                        timestamp = metadata.get('timestamp', run_info['timestamp'])
                        run_type = metadata.get('run_type', 'unknown')
                        success = "✓" if metadata.get('success', True) else "✗"
                        config_summary = run_info.get('config_summary', 'Unknown config')
                        
                        # Create numbered display format
                        display = f"#{i} {success} {config_summary} [{timestamp}]"
                    else:
                        display = f"#{i} ? {run_info['timestamp']} | Error loading details"
                    
                    run_options.append(display)
                    self.run_lookup[display] = run_info
                    # Store the run number for plot legends
                    run_info['run_number'] = i
                
                # Update listbox
                self.runs_listbox.delete(0, tk.END)
                for option in run_options:
                    self.runs_listbox.insert(tk.END, option)
            else:
                self.runs_listbox.delete(0, tk.END)
                self.runs_listbox.insert(tk.END, "No saved runs found")
                self.run_lookup = {}
                
        except Exception as e:
            print(f"Error refreshing runs: {e}")
            self.runs_listbox.delete(0, tk.END)
            self.runs_listbox.insert(tk.END, "Error loading runs")
            self.run_lookup = {}
    
    def on_runs_selected(self, event=None):
        """Handle selection of runs from listbox."""
        selected_indices = self.runs_listbox.curselection()
        selected_runs = []
        
        for index in selected_indices:
            run_display = self.runs_listbox.get(index)
            if run_display in getattr(self, 'run_lookup', {}):
                selected_runs.append(self.run_lookup[run_display])
        
        # Enable/disable plot button based on selection
        if selected_runs:
            self.create_plot_btn.config(state="normal")
            # Store selected runs for plotting
            self.selected_runs_data = selected_runs
            
            # Load first run for info display (maintains compatibility)
            try:
                first_run = selected_runs[0]
                complete_data = self.output_data_manager.load_complete_run(first_run['filepath'])
                
                if complete_data['metadata']['success']:
                    self.current_run_config = complete_data.get('configuration', {})
                    self.current_run_metadata = complete_data.get('metadata', {})
                    self._update_results_info()
                    
                    # Show multi-run ready message
                    self.results_figure.clear()
                    ax = self.results_figure.add_subplot(111)
                    num_runs = len(selected_runs)
                    run_text = "run" if num_runs == 1 else "runs"
                    ax.text(0.5, 0.5, f'{num_runs} {run_text} selected\n\nSelect plot type and create visualization', 
                            transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title(f'{num_runs} Model Run{"s" if num_runs > 1 else ""} Selected - Ready for Plotting')
                    self._ensure_plot_canvas_created()
                    if hasattr(self, 'plot_canvas') and self.plot_canvas:
                        self.plot_canvas.draw()
            except Exception as e:
                print(f"Error loading run info: {e}")
        else:
            self.create_plot_btn.config(state="disabled")
            self.selected_runs_data = []
    
    def _select_all_runs(self):
        """Select all available runs."""
        self.runs_listbox.selection_set(0, tk.END)
        self.on_runs_selected()
    
    def _clear_run_selection(self):
        """Clear all run selections."""
        self.runs_listbox.selection_clear(0, tk.END)
        self.on_runs_selected()
        
    
    def create_widgets(self):
        """Create main GUI components."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_model_tab()
        self.create_results_tab()
        
        # Register all GUI variables for automated config synchronization
        self._register_gui_variables_for_autosync()
    
    def create_model_tab(self):
        """Create model configuration tab."""
        model_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(model_frame, text="Model Configuration")
        
        # Configuration parameters
        config_frame = GUIUtils.create_section_frame(model_frame, "Model Parameters", row=0, columnspan=2)
        config_frame.grid(sticky=(tk.W, tk.E), pady=(0, 10))
        
        self._create_atmospheric_section(config_frame)
        self._create_soil_section(config_frame)
        self._create_numerical_section(config_frame)
        
        # Configure grid weights
        for i in range(2):
            config_frame.columnconfigure(i, weight=1)
    
    def _create_atmospheric_section(self, parent):
        """Create atmospheric parameters section with environmental forcing toggle."""
        atm_frame = GUIUtils.create_section_frame(parent, "Atmospheric Conditions", row=0)
        atm_frame.grid(sticky=(tk.W, tk.E, tk.N), padx=(0, 5))
        
        # Environmental forcing toggle
        self.environmental_forcing_var = tk.BooleanVar(value=True)
        forcing_frame = ttk.Frame(atm_frame)
        forcing_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        tk.Label(forcing_frame, text="Environmental Forcing:", font=StandardStyles.HEADER_FONT).grid(row=0, column=0, sticky=tk.W)
        self.forcing_toggle = tk.Checkbutton(forcing_frame, text="Use ERA5 Data", 
                                           variable=self.environmental_forcing_var,
                                           font=StandardStyles.BODY_FONT,
                                           command=self._on_environmental_forcing_changed)
        self.forcing_toggle.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # ERA5 section (shown when environmental forcing is enabled)
        self.era5_section_frame = ttk.Frame(atm_frame)
        self.era5_section_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self._create_era5_controls()
        
        # Create remaining sections (will be positioned dynamically)
        self._create_remaining_atmospheric_sections(atm_frame)
        
        # Initialize visibility
        self._update_environmental_forcing_visibility()
    
    def _create_remaining_atmospheric_sections(self, parent):
        """Create isotope sections (will be positioned dynamically)."""
        # Store parent for dynamic repositioning
        self.atm_parent = parent
        
        # Create constant atmospheric controls section (hidden when ERA5 is enabled)
        self.constant_atm_section_frame = ttk.Frame(parent)
        self._create_constant_atmospheric_controls()
        
        # Isotope parameters separator and entries
        self.isotope_separator = ttk.Separator(parent, orient='horizontal')
        self.isotope_frame = ttk.Frame(parent)
        
        self.gui_config_manager.create_parameter_entry(self.isotope_frame, "atmospheric.d18O_rain", "Rain δ¹⁸O (‰):", row=0)
        self.gui_config_manager.create_parameter_entry(self.isotope_frame, "atmospheric.D_prime_17O_rain", "Rain Δ'¹⁷O (per meg):", row=1)
        self.gui_config_manager.create_parameter_entry(self.isotope_frame, "atmospheric.d18O_vapor", "Vapor δ¹⁸O (‰):", row=2)
        self.gui_config_manager.create_parameter_entry(self.isotope_frame, "atmospheric.D_prime_17O_vapor", "Vapor Δ'¹⁷O (per meg):", row=3)
    
    def _create_era5_controls(self):
        """Create ERA5 data controls within the ERA5 section frame."""
        frame = self.era5_section_frame
        
        # ERA5 status display
        tk.Label(frame, text="ERA5 Data Status:", font=StandardStyles.HEADER_FONT).grid(row=0, column=0, sticky=tk.W, columnspan=2)
        self.era5_status_label = GUIUtils.create_status_label(frame, "Select ERA5 file", row=1, columnspan=2)
        
        # ERA5 data displays
        self.temp_range_var = tk.StringVar(value="--")
        temp_label, self.temp_range_display = GUIUtils.create_labeled_entry(frame, "Temperature Range:", self.temp_range_var, row=2)
        self.temp_range_display.config(state="readonly")
        
        self.humidity_range_var = tk.StringVar(value="--")
        humidity_label, self.humidity_range_display = GUIUtils.create_labeled_entry(frame, "Humidity Range:", self.humidity_range_var, row=3)
        self.humidity_range_display.config(state="readonly")
        
        self.time_period_var = tk.StringVar(value="--")
        time_label, self.time_period_display = GUIUtils.create_labeled_entry(frame, "Time Period:", self.time_period_var, row=4)
        self.time_period_display.config(state="readonly")
        
        # ERA5 file selection
        self._create_era5_file_selector(frame)
    
    def _on_environmental_forcing_changed(self):
        """Handle environmental forcing toggle change."""
        self._update_environmental_forcing_visibility()
    
    def _update_environmental_forcing_visibility(self):
        """Show/hide environmental forcing controls and reposition other sections."""
        if self.environmental_forcing_var.get():
            # ERA5 enabled - show ERA5 section, hide constant controls
            self.era5_section_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            self.constant_atm_section_frame.grid_remove()
            
            # Position isotope section after ERA5
            self.isotope_separator.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            self.isotope_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        else:
            # ERA5 disabled - hide ERA5 section, show constant controls
            self.era5_section_frame.grid_remove()
            self.constant_atm_section_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Position isotope section after constant controls
            self.isotope_separator.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            self.isotope_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def _create_constant_atmospheric_controls(self):
        """Create constant atmospheric parameter controls for non-ERA5 mode."""
        frame = self.constant_atm_section_frame
        
        # Status label
        tk.Label(frame, text="Constant Atmospheric Conditions:", font=StandardStyles.HEADER_FONT).grid(
            row=0, column=0, sticky=tk.W, columnspan=2)
        
        # Air Temperature control
        self.constant_air_temp_var = tk.DoubleVar(value=self.config.atmospheric.mean_air_temperature)
        ttk.Label(frame, text="Air Temperature (°C):").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=-50, to=50, increment=0.5, 
                   textvariable=self.constant_air_temp_var, width=10, format="%.1f").grid(row=1, column=1, padx=(5, 0))
        
        # Relative Humidity control
        self.constant_humidity_var = tk.DoubleVar(value=self.config.atmospheric.relative_humidity)
        ttk.Label(frame, text="Relative Humidity (%):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=0, to=100, increment=1, 
                   textvariable=self.constant_humidity_var, width=10, format="%.1f").grid(row=2, column=1, padx=(5, 0))
        
        # Air Pressure control (default to standard atmosphere)
        self.constant_pressure_var = tk.DoubleVar(value=101325.0)  # Pa
        ttk.Label(frame, text="Air Pressure (Pa):").grid(row=3, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=90000, to=110000, increment=100, 
                   textvariable=self.constant_pressure_var, width=10, format="%.0f").grid(row=3, column=1, padx=(5, 0))
        
        # Wind Speed control
        self.wind_speed_var = tk.DoubleVar(value=self.config.atmospheric.wind_speed)
        ttk.Label(frame, text="Wind Speed (m/s):").grid(row=4, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.1, to=20, increment=0.1, 
                   textvariable=self.wind_speed_var, width=10, format="%.1f").grid(row=4, column=1, padx=(5, 0))
        
        # Surface Roughness control
        self.surface_roughness_var = tk.DoubleVar(value=self.config.atmospheric.surface_roughness)
        ttk.Label(frame, text="Surface Roughness (cm):").grid(row=5, column=0, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.001, to=10, increment=0.001, 
                   textvariable=self.surface_roughness_var, width=10, format="%.3f").grid(row=5, column=1, padx=(5, 0))
        
        # Temperature diffusion toggle (only for constant mode)
        self.disable_temp_diffusion_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Disable Temperature Diffusion (keep soil temp constant)", 
                      variable=self.disable_temp_diffusion_var,
                      font=StandardStyles.BODY_FONT).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Note about simplified physics
        tk.Label(frame, text="Note: Simplified physics mode for exploring basic evaporation", 
                font=("Arial", 8), foreground="gray").grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def _create_era5_file_selector(self, parent):
        """Create ERA5 file selection widgets."""
        ttk.Label(parent, text="ERA5 Dataset:").grid(row=5, column=0, sticky=tk.W)
        era5_frame = ttk.Frame(parent)
        era5_frame.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        self.era5_file_var = tk.StringVar(value="sample_era5_summer2023.csv")
        self.era5_file_combo = ttk.Combobox(era5_frame, textvariable=self.era5_file_var, 
                                           width=25, state="readonly")
        self.era5_file_combo.grid(row=0, column=0, sticky=tk.W)
        self.era5_file_combo.bind('<<ComboboxSelected>>', self.on_era5_selection_changed)
        
        ttk.Button(era5_frame, text="↻", width=3, command=self._refresh_era5_files).grid(row=0, column=1, padx=(2, 0))
        ttk.Button(era5_frame, text="Browse...", command=self.browse_era5_file).grid(row=0, column=2, padx=(2, 0))
    
    def _create_soil_section(self, parent):
        """Create soil parameters section."""
        soil_frame = ttk.LabelFrame(parent, text="Soil Properties", padding="5")
        soil_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=(5, 0))
        
        # Basic soil parameters (excluding water content - now handled by profile section)
        basic_params = [
            ("Soil Depth (cm):", "depth_var", self.config.soil.depth, 50, 300, 1),
            ("Total Porosity:", "porosity_var", self.config.soil.total_porosity, 0.1, 0.9, 0.01),
            ("Tortuosity:", "tortuosity_var", self.config.soil.tortuosity, 0.1, 1.0, 0.01),
            ("Water Threshold:", "water_threshold_var", self.config.soil.water_content_threshold, 0.00001, 0.001, 0.00001),
            ("Depth Step (cm):", "depth_step_var", self.config.numerical.depth_step, 0.5, 5.0, 0.5)
        ]
        
        for i, (label, var_name, default, min_val, max_val, increment) in enumerate(basic_params):
            ttk.Label(soil_frame, text=label).grid(row=i, column=0, sticky=tk.W)
            var = tk.DoubleVar(value=default)
            setattr(self, var_name, var)
            
            format_str = "%.5f" if increment < 0.001 else None
            spin = ttk.Spinbox(soil_frame, from_=min_val, to=max_val, increment=increment, 
                              textvariable=var, width=10, format=format_str)
            spin.grid(row=i, column=1, sticky=tk.W, padx=(5, 0))
        
        # Water Content Profile Section
        water_content_rows = 4  # separator, header, type selection, params frame
        self._create_water_content_profile_section(soil_frame, start_row=len(basic_params))
        
        # Temperature Profile Section 
        temp_profile_start_row = len(basic_params) + water_content_rows
        self._create_temperature_profile_section(soil_frame, start_row=temp_profile_start_row)
    
    def _create_water_content_profile_section(self, parent, start_row):
        """Create water content profile initialization section."""
        # Section separator
        ttk.Separator(parent, orient='horizontal').grid(row=start_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Profile type selection
        ttk.Label(parent, text="Water Content Profile:", font=StandardStyles.HEADER_FONT).grid(row=start_row+1, column=0, columnspan=2, sticky=tk.W)
        
        profile_config = self.config.soil.water_content_profile
        self.profile_type_var = tk.StringVar(value=profile_config.profile_type)
        
        profile_frame = ttk.Frame(parent)
        profile_frame.grid(row=start_row+2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(profile_frame, text="Profile Type:").grid(row=0, column=0, sticky=tk.W)
        self.profile_type_combo = ttk.Combobox(profile_frame, 
                                             textvariable=self.profile_type_var,
                                             values=["constant", "linear", "exponential", "from_file"],
                                             state="readonly", width=12)
        self.profile_type_combo.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)
        self.profile_type_combo.bind('<<ComboboxSelected>>', self._on_profile_type_changed)
        
        # Dynamic parameter frames
        self.profile_params_frame = ttk.Frame(parent)
        self.profile_params_frame.grid(row=start_row+3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self._create_profile_parameter_frames()
        self._update_profile_visibility()
    
    def _create_profile_parameter_frames(self):
        """Create parameter input frames for each profile type."""
        profile_config = self.config.soil.water_content_profile
        
        # Constant profile frame
        self.constant_frame = ttk.Frame(self.profile_params_frame)
        self.constant_value_var = tk.DoubleVar(value=profile_config.constant_value)
        ttk.Label(self.constant_frame, text="Water Content:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.constant_frame, from_=0.001, to=0.1, increment=0.001, 
                   textvariable=self.constant_value_var, width=10, format="%.3f").grid(row=0, column=1, padx=(5, 0))
        ttk.Label(self.constant_frame, text="cm³/cm³").grid(row=0, column=2, padx=(5, 0))
        
        # Linear profile frame
        self.linear_frame = ttk.Frame(self.profile_params_frame)
        self.surface_value_var = tk.DoubleVar(value=profile_config.surface_value)
        self.bottom_value_var = tk.DoubleVar(value=profile_config.bottom_value)
        
        ttk.Label(self.linear_frame, text="Surface Value:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.linear_frame, from_=0.001, to=0.1, increment=0.001,
                   textvariable=self.surface_value_var, width=10, format="%.3f").grid(row=0, column=1, padx=(5, 0))
        ttk.Label(self.linear_frame, text="cm³/cm³").grid(row=0, column=2, padx=(5, 0))
        
        ttk.Label(self.linear_frame, text="Bottom Value:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(self.linear_frame, from_=0.001, to=0.1, increment=0.001,
                   textvariable=self.bottom_value_var, width=10, format="%.3f").grid(row=1, column=1, padx=(5, 0))
        ttk.Label(self.linear_frame, text="cm³/cm³").grid(row=1, column=2, padx=(5, 0))
        
        # Exponential profile frame
        self.exponential_frame = ttk.Frame(self.profile_params_frame)
        self.surface_value_exp_var = tk.DoubleVar(value=profile_config.surface_value_exp)
        self.decay_length_var = tk.DoubleVar(value=profile_config.decay_length)
        self.background_value_var = tk.DoubleVar(value=profile_config.background_value)
        
        ttk.Label(self.exponential_frame, text="Surface Value:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.exponential_frame, from_=0.001, to=0.1, increment=0.001,
                   textvariable=self.surface_value_exp_var, width=10, format="%.3f").grid(row=0, column=1, padx=(5, 0))
        ttk.Label(self.exponential_frame, text="cm³/cm³").grid(row=0, column=2, padx=(5, 0))
        
        ttk.Label(self.exponential_frame, text="Decay Length:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(self.exponential_frame, from_=1.0, to=100.0, increment=1.0,
                   textvariable=self.decay_length_var, width=10, format="%.1f").grid(row=1, column=1, padx=(5, 0))
        ttk.Label(self.exponential_frame, text="cm").grid(row=1, column=2, padx=(5, 0))
        
        ttk.Label(self.exponential_frame, text="Background:").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(self.exponential_frame, from_=0.0001, to=0.01, increment=0.0001,
                   textvariable=self.background_value_var, width=10, format="%.4f").grid(row=2, column=1, padx=(5, 0))
        ttk.Label(self.exponential_frame, text="cm³/cm³").grid(row=2, column=2, padx=(5, 0))
        
        # From file frame
        self.file_frame = ttk.Frame(self.profile_params_frame)
        self.profile_file_var = tk.StringVar(value=profile_config.profile_file or "No file selected")
        
        ttk.Label(self.file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        file_display = ttk.Label(self.file_frame, textvariable=self.profile_file_var, 
                               foreground=StandardStyles.PRIMARY_COLOR, wraplength=200)
        file_display.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Button(self.file_frame, text="Browse...", 
                  command=self._browse_profile_file).grid(row=1, column=0, pady=(5, 0), sticky=tk.W)
        
        ttk.Label(self.file_frame, text="Format: depth (cm), water_content (cm³/cm³)", 
                 font=("Arial", 8), foreground="gray").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def _on_profile_type_changed(self, event=None):
        """Handle profile type selection change."""
        self._update_profile_visibility()
    
    def _update_profile_visibility(self):
        """Show/hide parameter frames based on selected profile type."""
        profile_type = self.profile_type_var.get()
        
        # Hide all frames first
        for frame in [self.constant_frame, self.linear_frame, self.exponential_frame, self.file_frame]:
            frame.grid_remove()
        
        # Show the selected frame
        if profile_type == "constant":
            self.constant_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "linear":
            self.linear_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "exponential":
            self.exponential_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "from_file":
            self.file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def _browse_profile_file(self):
        """Open file browser for water content profile CSV."""
        
        file_path = filedialog.askopenfilename(
            title="Select Water Content Profile CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.profile_file_var.set(file_path)
    
    def _create_temperature_profile_section(self, parent, start_row):
        """Create temperature profile initialization section."""
        # Profile type selection
        tk.Label(parent, text="Temperature Profile:", font=StandardStyles.HEADER_FONT).grid(row=start_row, column=0, columnspan=2, sticky=tk.W)
        
        profile_config = self.config.temperature.temperature_profile
        self.temp_profile_type_var = tk.StringVar(value=profile_config.profile_type)
        
        profile_frame = ttk.Frame(parent)
        profile_frame.grid(row=start_row+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        tk.Label(profile_frame, text="Profile Type:").grid(row=0, column=0, sticky=tk.W)
        self.temp_profile_type_combo = ttk.Combobox(profile_frame, 
                                                   textvariable=self.temp_profile_type_var,
                                                   values=["constant", "linear", "exponential", "from_file"],
                                                   state="readonly", width=12)
        self.temp_profile_type_combo.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)
        self.temp_profile_type_combo.bind('<<ComboboxSelected>>', self._on_temp_profile_type_changed)
        
        # Dynamic parameter frames
        self.temp_profile_params_frame = ttk.Frame(parent)
        self.temp_profile_params_frame.grid(row=start_row+2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self._create_temp_profile_parameter_frames()
        self._update_temp_profile_visibility()
    
    def _create_temp_profile_parameter_frames(self):
        """Create parameter input frames for each temperature profile type."""
        profile_config = self.config.temperature.temperature_profile
        
        # Constant profile frame
        self.temp_constant_frame = ttk.Frame(self.temp_profile_params_frame)
        self.temp_constant_value_var = tk.DoubleVar(value=profile_config.constant_value)
        tk.Label(self.temp_constant_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_constant_frame, from_=-20, to=50, increment=0.5, 
                   textvariable=self.temp_constant_value_var, width=10, format="%.1f").grid(row=0, column=1, padx=(5, 0))
        tk.Label(self.temp_constant_frame, text="°C").grid(row=0, column=2, padx=(5, 0))
        
        # Linear profile frame
        self.temp_linear_frame = ttk.Frame(self.temp_profile_params_frame)
        self.temp_surface_value_var = tk.DoubleVar(value=profile_config.surface_value)
        self.temp_bottom_value_var = tk.DoubleVar(value=profile_config.bottom_value)
        
        tk.Label(self.temp_linear_frame, text="Surface Temp:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_linear_frame, from_=-20, to=50, increment=0.5,
                   textvariable=self.temp_surface_value_var, width=10, format="%.1f").grid(row=0, column=1, padx=(5, 0))
        tk.Label(self.temp_linear_frame, text="°C").grid(row=0, column=2, padx=(5, 0))
        
        tk.Label(self.temp_linear_frame, text="Bottom Temp:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_linear_frame, from_=-20, to=50, increment=0.5,
                   textvariable=self.temp_bottom_value_var, width=10, format="%.1f").grid(row=1, column=1, padx=(5, 0))
        tk.Label(self.temp_linear_frame, text="°C").grid(row=1, column=2, padx=(5, 0))
        
        # Exponential profile frame
        self.temp_exponential_frame = ttk.Frame(self.temp_profile_params_frame)
        self.temp_surface_value_exp_var = tk.DoubleVar(value=profile_config.surface_value_exp)
        self.temp_decay_length_var = tk.DoubleVar(value=profile_config.decay_length)
        self.temp_background_value_var = tk.DoubleVar(value=profile_config.background_value)
        
        tk.Label(self.temp_exponential_frame, text="Surface Temp:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_exponential_frame, from_=-20, to=50, increment=0.5,
                   textvariable=self.temp_surface_value_exp_var, width=10, format="%.1f").grid(row=0, column=1, padx=(5, 0))
        tk.Label(self.temp_exponential_frame, text="°C").grid(row=0, column=2, padx=(5, 0))
        
        tk.Label(self.temp_exponential_frame, text="Decay Length:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_exponential_frame, from_=5.0, to=200.0, increment=5.0,
                   textvariable=self.temp_decay_length_var, width=10, format="%.1f").grid(row=1, column=1, padx=(5, 0))
        tk.Label(self.temp_exponential_frame, text="cm").grid(row=1, column=2, padx=(5, 0))
        
        tk.Label(self.temp_exponential_frame, text="Background:").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(self.temp_exponential_frame, from_=-20, to=50, increment=0.5,
                   textvariable=self.temp_background_value_var, width=10, format="%.1f").grid(row=2, column=1, padx=(5, 0))
        tk.Label(self.temp_exponential_frame, text="°C").grid(row=2, column=2, padx=(5, 0))
        
        # From file frame
        self.temp_file_frame = ttk.Frame(self.temp_profile_params_frame)
        self.temp_profile_file_var = tk.StringVar(value=profile_config.profile_file or "No file selected")
        
        tk.Label(self.temp_file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        file_display = tk.Label(self.temp_file_frame, textvariable=self.temp_profile_file_var, 
                               foreground=StandardStyles.PRIMARY_COLOR, wraplength=200)
        file_display.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Button(self.temp_file_frame, text="Browse...", 
                  command=self._browse_temp_profile_file).grid(row=1, column=0, pady=(5, 0), sticky=tk.W)
        
        tk.Label(self.temp_file_frame, text="Format: depth (cm), temperature (°C)", 
                 font=("Arial", 8), foreground="gray").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def _on_temp_profile_type_changed(self, event=None):
        """Handle temperature profile type selection change."""
        self._update_temp_profile_visibility()
    
    def _update_temp_profile_visibility(self):
        """Show/hide parameter frames based on selected temperature profile type."""
        profile_type = self.temp_profile_type_var.get()
        
        # Hide all frames first
        for frame in [self.temp_constant_frame, self.temp_linear_frame, self.temp_exponential_frame, self.temp_file_frame]:
            frame.grid_remove()
        
        # Show the selected frame
        if profile_type == "constant":
            self.temp_constant_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "linear":
            self.temp_linear_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "exponential":
            self.temp_exponential_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        elif profile_type == "from_file":
            self.temp_file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def _browse_temp_profile_file(self):
        """Open file browser for temperature profile CSV."""
        
        file_path = filedialog.askopenfilename(
            title="Select Temperature Profile CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.temp_profile_file_var.set(file_path)
    
    def _create_numerical_section(self, parent):
        """Create run model section with numerical parameters and run button."""
        num_frame = ttk.LabelFrame(parent, text="Run Model", padding="10")
        num_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Run parameters and button
        params_frame = ttk.Frame(num_frame)
        params_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Run days parameter
        self.run_days_var = tk.DoubleVar(value=self.config.numerical.run_days)
        ttk.Label(params_frame, text="Run Days:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(params_frame, from_=1, to=365, textvariable=self.run_days_var, width=10).grid(row=0, column=1, padx=(5, 0))
        
        # Run model button
        self.run_model_btn = ttk.Button(params_frame, text="Run Model", 
                                       command=self.start_model_run, style="Accent.TButton")
        self.run_model_btn.grid(row=0, column=2, padx=(20, 0))
        
        # Progress status (minimal, no progress bar)
        self.progress_var = tk.StringVar(value="Ready to run model")
        ttk.Label(num_frame, textvariable=self.progress_var, font=("Arial", 9)).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    def create_results_tab(self):
        """Create results visualization tab with reorganized layout."""
        results_frame = ttk.Frame(self.notebook, padding="5")  # Reduced padding
        self.notebook.add(results_frame, text="Results & Plots")
        
        # Configure main results frame
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(3, weight=1)  # Main content area gets extra space
        
        # Top section: Data selection (row 0)
        data_frame = ttk.LabelFrame(results_frame, text="Data Selection", padding="5")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 3))
        data_frame.columnconfigure(1, weight=1)
        
        ttk.Label(data_frame, text="Select Runs:").grid(row=0, column=0, sticky=tk.NW)
        
        # Create scrollable listbox for multiple run selection
        runs_listbox_frame = ttk.Frame(data_frame)
        runs_listbox_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        runs_listbox_frame.columnconfigure(0, weight=1)
        runs_listbox_frame.rowconfigure(0, weight=1)
        
        self.runs_listbox = tk.Listbox(runs_listbox_frame, selectmode=tk.EXTENDED, height=3, 
                                      exportselection=False)
        self.runs_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.runs_listbox.bind('<<ListboxSelect>>', self.on_runs_selected)
        
        # Scrollbar for listbox
        runs_scrollbar = ttk.Scrollbar(runs_listbox_frame, orient=tk.VERTICAL, command=self.runs_listbox.yview)
        runs_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.runs_listbox.config(yscrollcommand=runs_scrollbar.set)
        
        # Control buttons
        controls_frame = ttk.Frame(data_frame)
        controls_frame.grid(row=0, column=2, sticky=(tk.N, tk.W), padx=(5, 0))
        
        ttk.Button(controls_frame, text="Refresh", command=self._refresh_available_runs).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(controls_frame, text="Select All", command=self._select_all_runs).grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        ttk.Button(controls_frame, text="Clear", command=self._clear_run_selection).grid(row=2, column=0, sticky=tk.W, pady=(2, 0))
        
        # Visualization controls section (row 1) 
        plot_frame = ttk.LabelFrame(results_frame, text="Visualization Options", padding="5")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=3)
        plot_frame.columnconfigure(1, weight=1)
        
        ttk.Label(plot_frame, text="Plot Type:").grid(row=0, column=0, sticky=tk.W)
        
        self.plot_types = [
            "Depth Profiles (All Variables)", "Water Content vs Depth", "Temperature vs Depth",
            "δ¹⁸O vs Depth", "Δ'¹⁷O vs Depth", "Surface Time Series",
            "Triple Oxygen Isotope Evolution", "Mass Balance Tracking", 
            "Water Content Evolution Heatmap", "Temperature Evolution Heatmap", 
            "δ¹⁸O Evolution Heatmap", "Δ'¹⁷O Evolution Heatmap"
        ]
        
        self.plot_type_var = tk.StringVar(value=self.plot_types[0])
        self.plot_type_combo = ttk.Combobox(plot_frame, textvariable=self.plot_type_var,
                                           values=self.plot_types, width=40, state="readonly")
        self.plot_type_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.create_plot_btn = ttk.Button(plot_frame, text="Create Plot", 
                                         command=self.create_selected_plot, state="disabled")
        self.create_plot_btn.grid(row=0, column=2, padx=5)
        
        # Data overlay controls section (row 2)
        self._create_data_overlay_section(results_frame)
        
        # Main content area (row 3) - uses a PanedWindow for resizable panes
        content_paned = ttk.PanedWindow(results_frame, orient='horizontal')
        content_paned.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(3, 0))
        
        # Left pane: Plot area
        plot_area_frame = ttk.LabelFrame(content_paned, text="Visualization", padding="2")
        content_paned.add(plot_area_frame, weight=2)  # 2/3 of space
        
        # Right pane: Results info panel  
        info_frame = ttk.LabelFrame(content_paned, text="Results Information", padding="2")
        content_paned.add(info_frame, weight=1)  # 1/3 of space
        
        # Setup plot canvas (lazy loading)
        self.plot_area_frame = plot_area_frame
        self.plot_canvas = None
        self._plot_canvas_created = False
        
        self.canvas_placeholder = ttk.Label(plot_area_frame, 
                                          text="Select a run and visualization option...", 
                                          font=('Arial', 12), foreground='gray')
        self.canvas_placeholder.grid(row=0, column=0, padx=20, pady=50)
        
        # Results info text
        info_text_frame = ttk.Frame(info_frame)
        info_text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_text_frame.columnconfigure(0, weight=1)
        info_text_frame.rowconfigure(0, weight=1)
        
        self.results_info_text = tk.Text(info_text_frame, wrap=tk.WORD, height=25, width=35, 
                                        font=('Consolas', 9), state='disabled')
        info_scrollbar = ttk.Scrollbar(info_text_frame, orient="vertical", command=self.results_info_text.yview)
        self.results_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.results_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure frame weights
        plot_area_frame.columnconfigure(0, weight=1)
        plot_area_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        self._show_empty_plot()
    
    def _create_data_overlay_section(self, parent):
        """Create data overlay controls section."""
        overlay_frame = ttk.LabelFrame(parent, text="Data Overlay", padding="5")
        overlay_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=3)
        overlay_frame.columnconfigure(2, weight=1)
        
        # Enable overlay toggle
        self.enable_overlay_var = tk.BooleanVar(value=False)
        tk.Checkbutton(overlay_frame, text="Enable Data Overlay", 
                      variable=self.enable_overlay_var,
                      command=self._on_overlay_toggle,
                      font=StandardStyles.BODY_FONT).grid(row=0, column=0, sticky=tk.W)
        
        # File selection button
        self.browse_data_btn = ttk.Button(overlay_frame, text="Browse Data File...", 
                                         command=self._browse_overlay_data, state="disabled")
        self.browse_data_btn.grid(row=0, column=1, padx=(10, 5))
        
        # Selected file display
        self.overlay_file_var = tk.StringVar(value="No file selected")
        self.overlay_file_label = ttk.Label(overlay_frame, textvariable=self.overlay_file_var, 
                                           foreground="gray", wraplength=300)
        self.overlay_file_label.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # File format info
        format_info = tk.Label(overlay_frame, 
                              text="Expected format: CSV with columns [depth, water_content, d18O, D17O, temperature]", 
                              font=("Arial", 8), foreground="gray")
        format_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Store overlay data
        self.overlay_data = None
    
    def _on_overlay_toggle(self):
        """Handle overlay enable/disable."""
        if self.enable_overlay_var.get():
            self.browse_data_btn.config(state="normal")
            self.overlay_file_label.config(foreground=StandardStyles.PRIMARY_COLOR)
        else:
            self.browse_data_btn.config(state="disabled")
            self.overlay_file_label.config(foreground="gray")
            self.overlay_file_var.set("No file selected")
            self.overlay_data = None
    
    def _browse_overlay_data(self):
        """Browse for overlay data file."""
        import pandas as pd
        
        file_path = filedialog.askopenfilename(
            title="Select Data Overlay CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and validate data
                data = pd.read_csv(file_path)
                required_columns = ['depth', 'water_content', 'd18O', 'D17O', 'temperature']
                
                if not all(col in data.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in data.columns]
                    messagebox.showerror("Invalid File", 
                                       f"CSV file missing required columns: {missing}\n\n"
                                       f"Required: {required_columns}")
                    return
                
                # Store overlay data
                self.overlay_data = data[required_columns].copy()
                filename = Path(file_path).name
                self.overlay_file_var.set(f"✓ {filename} ({len(data)} points)")
                self.overlay_file_label.config(foreground="green")
                
            except Exception as e:
                messagebox.showerror("File Error", f"Failed to load file: {str(e)}")
    
    def _ensure_plot_canvas_created(self):
        """Create plot canvas if needed."""
        if self._plot_canvas_created and hasattr(self, 'plot_canvas'):
            return
        
        try:
            # Destroy existing plot canvas if it exists
            if hasattr(self, 'plot_canvas') and self.plot_canvas:
                self.plot_canvas.destroy()
            
            # Destroy placeholder
            if hasattr(self, 'canvas_placeholder'):
                self.canvas_placeholder.destroy()
                
            self.plot_canvas = PlotCanvas(self.plot_area_frame, self.results_figure)
            self.plot_canvas.grid(row=0, column=0)  # Remove conflicting sticky parameter
            self._plot_canvas_created = True
            
        except Exception as e:
            print(f"Error creating plot canvas: {e}")
            self._plot_canvas_created = False
    
    def _show_empty_plot(self):
        """Show empty plot with instructions."""
        if self._plot_canvas_created:
            self.results_figure.clear()
            ax = self.results_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Run a model to see results\n\nSelect visualization options above', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Soil Evaporation Model")
            ax.axis('off')
            if hasattr(self, 'plot_canvas') and self.plot_canvas:
                self.plot_canvas.draw()
    
    def _show_error_plot(self, message):
        """Show error message in plot area."""
        self.results_figure.clear()
        ax = self.results_figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Error')
        self._ensure_plot_canvas_created()
        if hasattr(self, 'plot_canvas') and self.plot_canvas:
            self.plot_canvas.draw()
    
    def browse_era5_file(self):
        """Browse for ERA5 data file."""
        file_path = self.path_manager.browse_era5_data(parent=self.root)
        if file_path:
            self.path_manager.set_selected_file('era5_file', file_path)
            self.era5_file_var.set(file_path.name)
            self.update_era5_display()
    
    def on_era5_selection_changed(self, event):
        """Handle ERA5 file selection change."""
        self.update_era5_display()
    
    def update_era5_display(self):
        """Update ERA5 data summary display."""
        try:
            
            selected_file = self.era5_file_var.get()
            if not selected_file or selected_file == "No ERA5 files found":
                self.era5_status_label.config(text="No ERA5 file selected", foreground="orange")
                return
            
            # Find file path
            era5_file_path = self.path_manager.get_selected_file('era5_file')
            if not era5_file_path:
                era5_file_path = self.path_manager.era5_data_dir / selected_file
                if not era5_file_path.exists():
                    self.era5_status_label.config(text="❌ ERA5 file not found", foreground="red")
                    return
            
            # Load and display ERA5 data
            era5_forcing = ERA5Forcing(era5_file_path)
            stats = era5_forcing.get_summary_statistics()
            time_range = era5_forcing.get_time_range_days()
            
            self.era5_status_label.config(text="✓ Loaded successfully", foreground="green")
            
            temp_min, temp_max = stats['temperature_2m']['min'], stats['temperature_2m']['max']
            self.temp_range_var.set(f"{temp_min:.1f} to {temp_max:.1f}°C")
            
            rh_min, rh_max = stats['relative_humidity']['min'], stats['relative_humidity']['max']
            self.humidity_range_var.set(f"{rh_min:.1f} to {rh_max:.1f}%")
            
            self.time_period_var.set(f"{time_range:.1f} days")
            
        except Exception as e:
            self.era5_status_label.config(text="✗ Error loading", foreground="red")
            for var in [self.temp_range_var, self.humidity_range_var, self.time_period_var]:
                var.set("--")
    
    def _register_gui_variables_for_autosync(self):
        """Register all GUI variables with GUIConfigManager for automatic synchronization."""
        # Mapping of GUI variable names to their config parameter paths
        variable_mappings = {
            # Soil parameters
            'depth_var': 'soil.depth',
            'porosity_var': 'soil.total_porosity', 
            'tortuosity_var': 'soil.tortuosity',
            'water_threshold_var': 'soil.water_content_threshold',
            
            # Temperature profile parameters
            'temp_profile_type_var': 'temperature.temperature_profile.profile_type',
            'temp_constant_value_var': 'temperature.temperature_profile.constant_value',
            'temp_surface_value_var': 'temperature.temperature_profile.surface_value',
            'temp_bottom_value_var': 'temperature.temperature_profile.bottom_value',
            'temp_surface_value_exp_var': 'temperature.temperature_profile.surface_value_exp',
            'temp_decay_length_var': 'temperature.temperature_profile.decay_length',
            'temp_background_value_var': 'temperature.temperature_profile.background_value',
            'temp_profile_file_var': 'temperature.temperature_profile.profile_file',
            
            # Water content profile parameters
            'profile_type_var': 'soil.water_content_profile.profile_type',
            'constant_value_var': 'soil.water_content_profile.constant_value',
            'surface_value_var': 'soil.water_content_profile.surface_value',
            'bottom_value_var': 'soil.water_content_profile.bottom_value',
            'surface_value_exp_var': 'soil.water_content_profile.surface_value_exp',
            'decay_length_var': 'soil.water_content_profile.decay_length',
            'background_value_var': 'soil.water_content_profile.background_value',
            'profile_file_var': 'soil.water_content_profile.profile_file',
            
            # Numerical parameters
            'run_days_var': 'numerical.run_days',
            'depth_step_var': 'numerical.depth_step'
        }
        
        # Register each variable that exists
        for var_name, config_path in variable_mappings.items():
            if hasattr(self, var_name):
                gui_variable = getattr(self, var_name)
                self.gui_config_manager.register_gui_variable(config_path, gui_variable)
    
    def update_config_from_gui(self):
        """Update configuration from GUI values using automated sync."""
        # Use automated synchronization for all registered variables
        self.gui_config_manager.sync_config_from_gui()
        
        # Handle special cases that need manual processing
        # Check if environmental forcing is disabled
        if not self.environmental_forcing_var.get():
            # Update config with constant atmospheric values
            self.config.atmospheric.mean_air_temperature = self.constant_air_temp_var.get()
            self.config.atmospheric.relative_humidity = self.constant_humidity_var.get()
            # Store pressure in config (we'll add this field to config)
            self.config.atmospheric.air_pressure = self.constant_pressure_var.get()
            # Store boundary layer parameters
            self.config.atmospheric.wind_speed = self.wind_speed_var.get()
            self.config.atmospheric.surface_roughness = self.surface_roughness_var.get()
            # Store temperature diffusion flag
            self.config.temperature.disable_diffusion = self.disable_temp_diffusion_var.get()
        else:
            # ERA5 file (requires path validation)
            selected_file = self.era5_file_var.get()
            if selected_file != "No ERA5 files found":
                self.config.temperature.era5_data_file = selected_file
                self.config.temperature.era5_data_directory = "ERA5_DATA"
            # Enable temperature diffusion when using ERA5
            self.config.temperature.disable_diffusion = False
        
        # Store environmental forcing state
        self.config.temperature.use_era5_forcing = self.environmental_forcing_var.get()
        
        # Water content profile file validation
        if hasattr(self, 'profile_file_var'):
            profile_file = self.profile_file_var.get()
            if profile_file and profile_file != "No file selected":
                # Validate file exists
                file_path = Path(profile_file)
                if file_path.exists():
                    self.config.soil.water_content_profile.profile_file = str(file_path)
                else:
                    print(f"Warning: Water content profile file not found: {profile_file}")
                    self.config.soil.water_content_profile.profile_file = None
        
        # Temperature profile file validation
        if hasattr(self, 'temp_profile_file_var'):
            temp_profile_file = self.temp_profile_file_var.get()
            if temp_profile_file and temp_profile_file != "No file selected":
                # Validate file exists
                file_path = Path(temp_profile_file)
                if file_path.exists():
                    self.config.temperature.temperature_profile.profile_file = str(file_path)
                else:
                    print(f"Warning: Temperature profile file not found: {temp_profile_file}")
                    self.config.temperature.temperature_profile.profile_file = None
    
    def start_model_run(self):
        """Start model run."""
        self.update_config_from_gui()
        
        self.run_model_btn.config(state="disabled")
        self.progress_var.set("Running soil evaporation model...")
        self.root.update()
        
        self.run_model()
    
    def run_model(self):
        """Run the soil evaporation model."""
        try:
            print("Starting model run...")
            
            # Clear previous run data
            for attr in ['current_run_config', 'current_run_metadata']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            # Run model
            model = SoilEvaporationModel(self.config)
            results = model.run_simulation()
            
            self.model_results = model.results
            self.current_model = model
            
            # Save results
            self.output_data_manager.save_complete_run(
                config=self.config,
                model_results=self.model_results,
                run_type="single_run",
                execution_time=0.0,
                success=True
            )
            
            self.progress_var.set("✓ Model run completed!")
            self.enable_visualization_controls()
            self._update_results_info()
            self._refresh_available_runs()
            messagebox.showinfo("Success", "Model run completed successfully!")
            
        except Exception as e:
            print(f"Model run failed: {e}")
            self.model_results = None
            self.current_model = None
            self.progress_var.set("❌ Model run failed")
            messagebox.showerror("Error", f"Model run failed: {str(e)}")
        
        finally:
            self.run_model_btn.config(state="normal")
    
    def enable_visualization_controls(self):
        """Enable visualization controls."""
        self.create_plot_btn.config(state="normal")
    
    def create_selected_plot(self):
        """Create the selected plot type with multiple runs and optional overlay."""
        plot_type = self.plot_type_var.get()
        
        self.results_figure.clear()
        
        try:
            # Get selected runs data
            selected_runs = getattr(self, 'selected_runs_data', [])
            if not selected_runs:
                self._show_no_data_message(plot_type)
                return
            
            # Get overlay data if enabled
            overlay_data = None
            if getattr(self, 'enable_overlay_var', None) and self.enable_overlay_var.get():
                overlay_data = getattr(self, 'overlay_data', None)
            
            # Create multi-run plot with optional overlay
            self._create_multi_run_plot(plot_type, selected_runs, overlay_data)
            
            try:
                self._ensure_plot_canvas_created()
                if hasattr(self, 'plot_canvas') and self.plot_canvas:
                    self.plot_canvas.draw()
            except Exception as e:
                print(f"Error drawing plot canvas: {e}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")
    
    def _create_multi_run_plot(self, plot_type, selected_runs, overlay_data=None):
        """Create plot with multiple runs and optional overlay data."""
        # Load complete data for all selected runs
        runs_data = []
        run_labels = []
        
        for run_info in selected_runs:
            try:
                complete_data = self.output_data_manager.load_complete_run(run_info['filepath'])
                if complete_data['metadata']['success']:
                    runs_data.append(complete_data['results'])
                    # Create simple numbered label for plot legend
                    run_number = run_info.get('run_number', len(run_labels) + 1)
                    label = f"Run #{run_number}"
                    run_labels.append(label)
            except Exception as e:
                print(f"Error loading run {run_info.get('filepath', 'unknown')}: {e}")
        
        if not runs_data:
            self._show_no_data_message(plot_type)
            return
        
        # Check if this plot type supports multi-run comparison
        multi_run_plot_types = [
            "Water Content vs Depth", "Temperature vs Depth", "δ¹⁸O vs Depth", "Δ'¹⁷O vs Depth"
        ]
        
        if plot_type in multi_run_plot_types:
            # Use new multi-run plotting function
            variable_map = {
                "Water Content vs Depth": ("water_content", "Water Content", "cm³/cm³"),
                "Temperature vs Depth": ("temperature", "Temperature", "°C"),
                "δ¹⁸O vs Depth": ("d18O", "δ¹⁸O", "‰"),
                "Δ'¹⁷O vs Depth": ("delta_prime", "Δ'¹⁷O", "per meg")
            }
            
            variable, title, units = variable_map[plot_type]
            success = self.plotter.create_multi_run_comparison_plot(
                runs_data, run_labels, variable, title, units, overlay_data
            )
            
            if not success:
                self._show_error_plot(f"Failed to create {plot_type} comparison plot")
        else:
            # For single-run plot types, use first selected run
            if len(runs_data) > 1:
                messagebox.showwarning("Plot Type Limitation", 
                                     f"'{plot_type}' does not support multi-run comparison.\n"
                                     f"Using first selected run only.")
            
            # Use existing single-run plotting logic
            wrapped_data = {'results': runs_data[0]}
            self._create_plot(plot_type, wrapped_data)
    
    def _create_plot(self, plot_type, data):
        """Create plot based on type and data."""
        # Wrap data if needed
        wrapped_data = {'results': data} if 'results' not in data else data
        
        plot_map = {
            "Depth Profiles (All Variables)": self.plotter.create_depth_profiles_plot,
            "Water Content vs Depth": lambda d: self.plotter.create_single_variable_depth_plot(d, 'water_content', 'Water Content', 'cm³/cm³'),
            "Temperature vs Depth": lambda d: self.plotter.create_single_variable_depth_plot(d, 'temperature', 'Temperature', '°C'),
            "δ¹⁸O vs Depth": lambda d: self.plotter.create_single_variable_depth_plot(d, 'd18O', 'δ¹⁸O', '‰'),
            "Δ'¹⁷O vs Depth": lambda d: self.plotter.create_single_variable_depth_plot(d, 'delta_prime', "Δ'¹⁷O", 'per meg'),
            "Surface Time Series": self.plotter.create_time_series_plot,
            "Triple Oxygen Isotope Evolution": self.plotter.create_isotope_evolution_plot,
            "Mass Balance Tracking": self.plotter.create_mass_balance_plot,
            "Water Content Evolution Heatmap": self.plotter.create_soil_drying_heatmap,
            "Temperature Evolution Heatmap": self.plotter.create_temperature_evolution_heatmap,
            "δ¹⁸O Evolution Heatmap": lambda d: self.plotter.create_isotope_evolution_heatmap(d, 'd18O'),
            "Δ'¹⁷O Evolution Heatmap": lambda d: self.plotter.create_isotope_evolution_heatmap(d, 'delta_prime')
        }
        
        plot_func = plot_map.get(plot_type)
        if plot_func:
            success = plot_func(wrapped_data)
            if not success:
                self._show_no_data_message(plot_type)
        else:
            self._show_not_implemented_message(plot_type)
    
    def _show_no_data_message(self, plot_type):
        """Show no data available message."""
        self.results_figure.clear()
        ax = self.results_figure.add_subplot(111)
        ax.text(0.5, 0.5, f"No data available for {plot_type}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    def _show_not_implemented_message(self, plot_type):
        """Show not implemented message."""
        self.results_figure.clear()
        ax = self.results_figure.add_subplot(111)
        ax.text(0.5, 0.5, f"Plot type '{plot_type}' not yet implemented", 
                transform=ax.transAxes, ha='center', va='center', fontsize=14, color='orange')
        ax.axis('off')
    
    def _update_results_info(self):
        """Update results information panel."""
        if not hasattr(self, 'results_info_text'):
            return
        
        self.results_info_text.config(state='normal')
        self.results_info_text.delete(1.0, tk.END)
        
        # Display run information
        if hasattr(self, 'current_run_config') and hasattr(self, 'current_run_metadata'):
            info_text = self.output_data_manager.create_detailed_config_info(
                self.current_run_config, self.current_run_metadata)
            
            if self.model_results:
                info_text += "\nRESULTS SUMMARY\n" + "="*40 + "\n"
                info_text += f"Time steps: {len(self.model_results['times_days'])}\n"
                info_text += f"Final day: {self.model_results['times_days'][-1]:.1f}\n"
                info_text += f"Depth nodes: {len(self.model_results['depth_nodes'])}\n"
                info_text += f"Max depth: {self.model_results['depth_nodes'][-1]:.1f} cm\n"
                
        elif self.model_results:
            info_text = "CURRENT MODEL RUN\n" + "="*40 + "\n"
            info_text += f"Time steps: {len(self.model_results['times_days'])}\n"
            info_text += f"Final day: {self.model_results['times_days'][-1]:.1f}\n"
            info_text += f"Depth nodes: {len(self.model_results['depth_nodes'])}\n"
            info_text += f"Max depth: {self.model_results['depth_nodes'][-1]:.1f} cm\n"
            
        else:
            info_text = "SOIL EVAPORATION MODEL\n" + "="*40 + "\n"
            info_text += "No results available. Run a model to see results.\n\n"
            info_text += "AVAILABLE VISUALIZATIONS:\n" + "="*40 + "\n"
            info_text += "• Depth profiles\n• Time series evolution\n• Isotope plots\n• Evolution heatmaps\n"
        
        self.results_info_text.insert(1.0, info_text)
        self.results_info_text.config(state='disabled')


def main():
    """Main application entry point."""
    print("Starting Soil Evaporation Model")
    print("="*50)
    
    try:
        # Test that imports are working (already imported at module level)
        SoilEvaporationModel
        print("✓ Model modules loaded")
    except NameError as e:
        print(f"❌ Failed to import modules: {e}")
        return
    
    root = tk.Tk()
    app = SoilEvaporationApp(root)
    print("✓ Application ready")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n✓ Application closed")
    except Exception as e:
        print(f"❌ Application error: {e}")


if __name__ == "__main__":
    main()