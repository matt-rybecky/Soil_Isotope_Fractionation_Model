"""
Configuration Management System
==============================

Centralized configuration management with validation, serialization,
and GUI integration for the soil evaporation model.

This module combines configuration parameter definitions and management
functionality into a single comprehensive system.

Author: Enhanced Python Implementation for Soil Evaporation Research
Original: Dan Breecker, University of Texas at Austin
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import warnings
from contextlib import contextmanager


# ================================
# Utility Functions
# ================================

def validate_range(value: float, min_val: float, max_val: float, name: str, 
                  unit: str = "") -> None:
    """Shared validation for numeric ranges."""
    if not min_val <= value <= max_val:
        unit_str = f" {unit}" if unit else ""
        raise ValueError(f"{name} must be {min_val}-{max_val}{unit_str}, got {value}{unit_str}")

def validate_positive(value: float, name: str) -> None:
    """Shared validation for positive values."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def load_config_file(file_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    with open(file_path, 'r') as f:
        if suffix == '.json':
            return json.load(f)
        elif suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

def save_config_file(file_path: Path, config_dict: Dict[str, Any]) -> None:
    """Save configuration to JSON or YAML file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = file_path.suffix.lower()
    with open(file_path, 'w') as f:
        if suffix == '.json':
            json.dump(config_dict, f, indent=2, default=str)
        elif suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

@contextmanager
def config_error_handler(operation: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except Exception as e:
        print(f"Failed to {operation}: {e}")
        raise RuntimeError(f"Failed to {operation}: {e}")

# Type mapping for GUI variables
GUI_TYPE_MAP = {
    float: tk.DoubleVar,
    int: tk.IntVar,
    str: tk.StringVar,
    bool: tk.BooleanVar
}

# ================================
# Configuration Parameter Classes
# ================================

@dataclass
class AtmosphericParameters:
    """Atmospheric conditions for the model."""
    relative_humidity: float = 70.0  # %
    mean_air_temperature: float = 18.0  # degrees C
    annual_temperature_range: float = 7.0  # degrees C (half range)
    d18O_rain: float = -10.0  # per mil
    D_prime_17O_rain: float = 10.0  # per meg
    
    # Atmospheric vapor isotope composition (independent of rain)
    d18O_vapor: float = -15.0  # per mil - atmospheric water vapor
    D_prime_17O_vapor: float = 10.0   # per meg - atmospheric water vapor (Δ'¹⁷O)
    
    # Additional atmospheric parameters for constant mode
    air_pressure: float = 101325.0  # Pa - atmospheric pressure
    
    # Surface boundary layer parameters
    wind_speed: float = 2.0  # m/s - wind speed at reference height
    surface_roughness: float = 0.1  # cm - surface roughness length
    
    def __post_init__(self):
        """Validate atmospheric parameters."""
        validate_range(self.relative_humidity, 0, 100, "Relative humidity", "%")
        if not -50 <= self.mean_air_temperature <= 50:
            warnings.warn(f"Unusual air temperature: {self.mean_air_temperature}°C")
        validate_positive(self.annual_temperature_range, "Temperature range")
        validate_positive(self.air_pressure, "Air pressure")
        validate_positive(self.wind_speed, "Wind speed")
        validate_positive(self.surface_roughness, "Surface roughness")


@dataclass
class WaterContentProfile:
    """Water content initialization profile settings."""
    profile_type: str = "constant"  # "constant", "linear", "exponential", "from_file"
    
    # Constant profile
    constant_value: float = 0.005  # cm³/cm³
    
    # Linear profile  
    surface_value: float = 0.01    # cm³/cm³ at surface
    bottom_value: float = 0.005    # cm³/cm³ at bottom
    
    # Exponential profile
    surface_value_exp: float = 0.01  # cm³/cm³ at surface
    decay_length: float = 20.0      # cm - characteristic decay length
    background_value: float = 0.002  # cm³/cm³ asymptotic value
    
    # From file
    profile_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate water content profile parameters."""
        valid_types = ["constant", "linear", "exponential", "from_file"]
        if self.profile_type not in valid_types:
            raise ValueError(f"Profile type must be one of {valid_types}, got {self.profile_type}")
        
        # Validate all water content values are positive
        if self.constant_value < 0:
            raise ValueError(f"Constant value must be positive, got {self.constant_value}")
        if self.surface_value < 0:
            raise ValueError(f"Surface value must be positive, got {self.surface_value}")
        if self.bottom_value < 0:
            raise ValueError(f"Bottom value must be positive, got {self.bottom_value}")
        if self.surface_value_exp < 0:
            raise ValueError(f"Exponential surface value must be positive, got {self.surface_value_exp}")
        if self.background_value < 0:
            raise ValueError(f"Background value must be positive, got {self.background_value}")
        if self.decay_length <= 0:
            raise ValueError(f"Decay length must be positive, got {self.decay_length}")


@dataclass
class TemperatureProfile:
    """Temperature initialization profile settings."""
    profile_type: str = "constant"  # "constant", "linear", "exponential", "from_file"
    
    # Constant profile
    constant_value: float = 15.0  # °C
    
    # Linear profile  
    surface_value: float = 20.0    # °C at surface
    bottom_value: float = 15.0     # °C at bottom
    
    # Exponential profile
    surface_value_exp: float = 25.0  # °C at surface
    decay_length: float = 30.0       # cm - characteristic decay length
    background_value: float = 15.0   # °C asymptotic value
    
    # From file
    profile_file: Optional[str] = None
    
    def __post_init__(self):
        """Validate temperature profile parameters."""
        valid_types = ["constant", "linear", "exponential", "from_file"]
        if self.profile_type not in valid_types:
            raise ValueError(f"Profile type must be one of {valid_types}, got {self.profile_type}")
        
        # Validate temperature values are reasonable (absolute temperatures in Celsius)
        temp_values = [self.constant_value, self.surface_value, self.bottom_value, 
                      self.surface_value_exp, self.background_value]
        for temp in temp_values:
            if not -50 <= temp <= 60:
                warnings.warn(f"Unusual temperature value: {temp}°C")
        
        if self.decay_length <= 0:
            raise ValueError(f"Decay length must be positive, got {self.decay_length}")


@dataclass
class SoilParameters:
    """Soil physical properties."""
    depth: float = 150.0  # cm
    total_porosity: float = 0.5  # dimensionless
    tortuosity: float = 0.6  # dimensionless
    water_content_threshold: float = 0.00005  # cm³/cm³
    water_content_profile: WaterContentProfile = field(default_factory=WaterContentProfile)
    
    def __post_init__(self):
        """Validate soil parameters."""
        validate_positive(self.depth, "Soil depth")
        validate_range(self.total_porosity, 0.001, 0.999, "Total porosity")
        validate_range(self.tortuosity, 0.001, 1.0, "Tortuosity")
        
        # Validate water content profile against porosity
        profile = self.water_content_profile
        max_wc = max(profile.constant_value, profile.surface_value, profile.bottom_value, 
                    profile.surface_value_exp, profile.background_value)
        if max_wc > self.total_porosity:
            raise ValueError(f"Water content ({max_wc:.3f}) cannot exceed total porosity ({self.total_porosity:.3f})")


@dataclass
class NumericalParameters:
    """Numerical model settings."""
    depth_step: float = 2.0  # cm
    run_days: float = 70.0  # days
    max_iterations: Optional[int] = None  # auto-calculated if None
    store_interval: Optional[int] = None  # auto-calculated if None
    progress_interval: int = 25000  # iterations (hardcoded, not user-controllable)
    
    def __post_init__(self):
        """Validate numerical parameters."""
        validate_positive(self.depth_step, "Depth step")
        validate_positive(self.run_days, "Run days")


@dataclass 
class PhysicalConstants:
    """Physical constants for isotope calculations."""
    # Triple oxygen isotope parameters (hardcoded constants, not user-controllable)
    theta_eq: float = 0.529  # theta for equilibrium
    theta_diff: float = 0.511  # theta for diffusion
    RSMOW_1816: float = 2005.20e-6  # 18O/16O in SMOW
    RSMOW_1716: float = 379.9e-6  # 17O/16O in SMOW
    
    # Molar masses (g/mol)
    molar_mass_16O: float = 15.99491
    molar_mass_18O: float = 17.99916
    molar_mass_H: float = 1.00784
    molar_mass_O: float = 15.9994
    avg_atomic_mass_air: float = 28.97
    
    # Diffusion parameters (hardcoded constants, not user-controllable)
    D_H2O_air_20C: float = 0.242  # cm²/s at 20°C
    D_H216O_over_D_H218O: float = 1.028  # Merlivat 1978 empirical value
    
    # Temperature dependence
    temp_exponent: float = 1.823  # for diffusion coefficient temperature scaling


@dataclass
class WaterContentParameters:
    """Water content evolution settings."""
    # Water content evolution method
    evolution_method: str = "prescribed"  # 'prescribed', 'predicted', 'lab_model'
    
    evolution_model_type: str = "exponential_decay"  # 'exponential_decay', 'diffusion', 'empirical'
    
    # Initial profile settings
    use_initial_profile: bool = False
    initial_profile_file: Optional[str] = None
    initial_profile_depths: Optional[np.ndarray] = None
    initial_profile_values: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate water content parameters."""
        valid_methods = ['prescribed', 'predicted', 'lab_model']
        if self.evolution_method not in valid_methods:
            raise ValueError(f"Evolution method must be one of {valid_methods}")
        
        valid_model_types = ['exponential_decay', 'diffusion', 'empirical']
        if self.evolution_model_type not in valid_model_types:
            raise ValueError(f"Evolution model type must be one of {valid_model_types}")


@dataclass
class TemperatureParameters:
    """Temperature modeling settings."""
    # Temperature calculation method
    temperature_method: str = "analytical"  # 'analytical', 'era5_forced', 'energy_balance'
    
    # ERA5 integration
    use_era5_data: bool = False
    use_era5_forcing: bool = True  # Whether to use ERA5 or constant atmospheric conditions
    era5_data_directory: Optional[str] = None
    era5_data_file: Optional[str] = None  # Specific ERA5 file to use
    era5_start_date: Optional[str] = None
    era5_end_date: Optional[str] = None
    era5_latitude: Optional[float] = None
    era5_longitude: Optional[float] = None
    
    # Temperature diffusion control
    disable_diffusion: bool = False  # When True, keeps soil temperature constant at initial profile
    
    # Climate normals
    climate_location: str = "default"
    thermal_diffusivity: float = 1e-6  # m²/s
    
    # Temperature profile initialization
    temperature_profile: TemperatureProfile = field(default_factory=TemperatureProfile)
    
    # Surface coupling
    surface_coupling: float = 0.8  # How strongly surface follows air temperature
    
    def __post_init__(self):
        """Validate temperature parameters."""
        valid_methods = ['analytical', 'era5_forced', 'energy_balance']
        if self.temperature_method not in valid_methods:
            raise ValueError(f"Temperature method must be one of {valid_methods}")
        
        validate_positive(self.thermal_diffusivity, "Thermal diffusivity")
        validate_range(self.surface_coupling, 0, 1, "Surface coupling")


@dataclass
class OutputParameters:
    """Output and visualization settings."""
    save_plots: bool = True
    save_csv: bool = True
    save_netcdf: bool = False
    plot_dpi: int = 300
    plot_format: str = 'png'
    output_directory: str = '.'
    filename_prefix: str = 'soil_evap'
    show_plots: bool = True
    
    def __post_init__(self):
        """Validate output parameters."""
        validate_positive(self.plot_dpi, "Plot DPI")
        if self.plot_format not in ['png', 'pdf', 'svg', 'jpg']:
            raise ValueError(f"Unsupported plot format: {self.plot_format}")


@dataclass
class ModelConfiguration:
    """Complete model configuration."""
    atmospheric: AtmosphericParameters = field(default_factory=AtmosphericParameters)
    soil: SoilParameters = field(default_factory=SoilParameters)
    numerical: NumericalParameters = field(default_factory=NumericalParameters)
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    water_content: WaterContentParameters = field(default_factory=WaterContentParameters)
    temperature: TemperatureParameters = field(default_factory=TemperatureParameters)
    output: OutputParameters = field(default_factory=OutputParameters)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfiguration':
        """Create configuration from dictionary."""
        return cls(
            atmospheric=AtmosphericParameters(**config_dict.get('atmospheric', {})),
            soil=SoilParameters(**config_dict.get('soil', {})),
            numerical=NumericalParameters(**config_dict.get('numerical', {})),
            constants=PhysicalConstants(**config_dict.get('constants', {})),
            water_content=WaterContentParameters(**config_dict.get('water_content', {})),
            temperature=TemperatureParameters(**config_dict.get('temperature', {})),
            output=OutputParameters(**config_dict.get('output', {}))
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ModelConfiguration':
        """Load configuration from JSON or YAML file."""
        config_dict = load_config_file(file_path)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ModelConfiguration':
        """Load configuration from YAML file."""
        return cls.from_file(yaml_path)
    
    @classmethod  
    def from_json(cls, json_path: Union[str, Path]) -> 'ModelConfiguration':
        """Load configuration from JSON file."""
        return cls.from_file(json_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'atmospheric': asdict(self.atmospheric),
            'soil': asdict(self.soil),
            'numerical': asdict(self.numerical),
            'constants': asdict(self.constants),
            'water_content': asdict(self.water_content),
            'temperature': asdict(self.temperature),
            'output': asdict(self.output)
        }
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON or YAML file."""
        save_config_file(file_path, self.to_dict())
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        self.to_file(yaml_path)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        self.to_file(json_path)
    
    def validate(self) -> None:
        """Validate entire configuration for physical consistency."""
        # Check soil depth vs depth step
        if self.soil.depth / self.numerical.depth_step < 10:
            warnings.warn("Very few depth nodes - consider smaller depth step or larger soil depth")
        
        # Check time step stability (will be auto-calculated, but warn about potential issues)
        max_D_effective = (self.constants.D_H2O_air_20C * self.soil.tortuosity * 
                          self.soil.total_porosity)
        dt_max = self.numerical.depth_step**2 / (2 * max_D_effective)
        
        if dt_max < 1:  # Less than 1 second timestep
            warnings.warn(f"Very small timestep required: {dt_max:.2e} seconds - simulation may be slow")
        
        # Check if run duration is reasonable
        total_iterations = int(self.numerical.run_days * 24 * 60 * 60 / dt_max)
        if total_iterations > 1e7:
            warnings.warn(f"Very long simulation: {total_iterations:,} iterations - consider reducing run_days or increasing depth_step")


# =============================
# Configuration Factory Functions
# =============================

def create_default_config() -> ModelConfiguration:
    """Create default model configuration matching original MATLAB values."""
    return ModelConfiguration()


def load_config(config_path: Union[str, Path, Dict[str, Any], None] = None) -> ModelConfiguration:
    """
    Load model configuration from various sources.
    
    Args:
        config_path: Path to YAML/JSON file, dictionary, or None for defaults
        
    Returns:
        ModelConfiguration object
    """
    if config_path is None:
        return create_default_config()
    
    if isinstance(config_path, dict):
        return ModelConfiguration.from_dict(config_path)
    
    return ModelConfiguration.from_file(config_path)




# ================================
# Configuration Management Classes
# ================================

class ConfigurationManager:
    """
    Centralized configuration management system.
    
    Handles loading, saving, validation, and GUI synchronization
    of model configurations.
    """
    
    def __init__(self, default_config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            default_config_path: Path to default configuration file
        """
        self.default_config_path = default_config_path
        self.current_config = None
        self.config_history = []
        self.unsaved_changes = False
        
        # Load default configuration
        self.load_default_config()
    
    def load_default_config(self) -> ModelConfiguration:
        """
        Load default configuration.
        
        Returns:
            Default ModelConfiguration instance
        """
        try:
            self.current_config = load_config()
            self.unsaved_changes = False
            return self.current_config
        except Exception as e:
            raise RuntimeError(f"Failed to load default configuration: {e}")
    
    def load_config_from_file(self, file_path: Path) -> ModelConfiguration:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Loaded ModelConfiguration instance
        """
        with config_error_handler(f"load configuration from {file_path}"):
            config_dict = load_config_file(file_path)
            self.current_config = ModelConfiguration.from_dict(config_dict)
            self.unsaved_changes = False
            return self.current_config
    
    def save_config_to_file(self, file_path: Path, 
                           config: Optional[ModelConfiguration] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            config: Configuration to save (uses current if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if config is None:
                config = self.current_config
            
            if config is None:
                raise ValueError("No configuration to save")
            
            save_config_file(file_path, config.to_dict())
            self.unsaved_changes = False
            return True
            
        except Exception as e:
            print(f"Failed to save configuration to {file_path}: {e}")
            return False
    
    def validate_config(self, config: Optional[ModelConfiguration] = None) -> List[str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            List of validation error messages (empty if valid)
        """
        if config is None:
            config = self.current_config
        
        if config is None:
            return ["No configuration to validate"]
        
        errors = []
        
        try:
            # Use dataclass validation first (catches most issues)
            config.validate()
            
            # Additional cross-parameter validation
            # Water content profile validation is handled in SoilParameters.__post_init__
            
            if config.numerical.depth_step > config.soil.depth:
                errors.append("Depth step cannot be larger than total soil depth")
            
        except Exception as e:
            errors.append(str(e))
        
        return errors
    
    def update_config_parameter(self, parameter_path: str, value: Any) -> bool:
        """
        Update a specific configuration parameter.
        
        Args:
            parameter_path: Dot-separated path to parameter (e.g., 'atmospheric.relative_humidity' 
                          or 'soil.water_content_profile.profile_type')
            value: New parameter value
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if self.current_config is None:
                self.load_default_config()
            
            # Navigate through nested attributes
            path_parts = parameter_path.split('.')
            obj = self.current_config
            
            # Navigate to the parent object
            for part in path_parts[:-1]:
                obj = getattr(obj, part)
            
            # Set the final attribute
            setattr(obj, path_parts[-1], value)
            self.unsaved_changes = True
            return True
        
        except (ValueError, AttributeError) as e:
            print(f"Failed to update parameter {parameter_path}: {e}")
            return False
    
    def get_config_parameter(self, parameter_path: str) -> Any:
        """
        Get a specific configuration parameter value.
        
        Args:
            parameter_path: Dot-separated path to parameter (supports nested paths)
            
        Returns:
            Parameter value or None if not found
        """
        try:
            if self.current_config is None:
                return None
            
            # Navigate through nested attributes
            path_parts = parameter_path.split('.')
            obj = self.current_config
            
            for part in path_parts:
                obj = getattr(obj, part)
            
            return obj
            
        except (ValueError, AttributeError):
            return None
    
    
    def reset_to_defaults(self) -> ModelConfiguration:
        """
        Reset configuration to defaults.
        
        Returns:
            Default configuration
        """
        self.current_config = load_config()
        self.unsaved_changes = False
        return self.current_config
    
    def has_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes.
        
        Returns:
            True if there are unsaved changes, False otherwise
        """
        return self.unsaved_changes
    
    def create_config_copy(self) -> ModelConfiguration:
        """
        Create a deep copy of current configuration.
        
        Returns:
            Copy of current configuration
        """
        if self.current_config is None:
            self.load_default_config()
        
        # Convert to dict and back to create deep copy
        config_dict = self.current_config.to_dict()
        return ModelConfiguration.from_dict(config_dict)


class GUIConfigManager:
    """
    GUI integration for configuration management.
    
    Provides methods for synchronizing GUI widgets with configuration values.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize GUI configuration manager.
        
        Args:
            config_manager: ConfigurationManager instance
        """
        self.config_manager = config_manager
        self.gui_variables = {}
        self.parameter_widgets = {}
    
    def register_gui_variable(self, parameter_path: str, gui_variable: tk.Variable):
        """
        Register a GUI variable for automatic synchronization.
        
        Args:
            parameter_path: Dot-separated parameter path
            gui_variable: tkinter Variable object
        """
        self.gui_variables[parameter_path] = gui_variable
        
        # Set initial value from config
        initial_value = self.config_manager.get_config_parameter(parameter_path)
        if initial_value is not None:
            gui_variable.set(initial_value)
        
        # Set up trace for automatic updates
        gui_variable.trace_add('write', 
                              lambda *args: self._on_gui_variable_changed(parameter_path))
    
    def _on_gui_variable_changed(self, parameter_path: str):
        """
        Handle GUI variable change.
        
        Args:
            parameter_path: Parameter path that changed
        """
        gui_variable = self.gui_variables.get(parameter_path)
        if gui_variable:
            try:
                new_value = gui_variable.get()
                self.config_manager.update_config_parameter(parameter_path, new_value)
            except Exception as e:
                print(f"Error updating config parameter {parameter_path}: {e}")
    
    def sync_gui_from_config(self):
        """Synchronize all GUI variables from current configuration."""
        for parameter_path, gui_variable in self.gui_variables.items():
            config_value = self.config_manager.get_config_parameter(parameter_path)
            if config_value is not None:
                try:
                    gui_variable.set(config_value)
                except Exception as e:
                    print(f"Error syncing GUI variable {parameter_path}: {e}")
    
    def sync_config_from_gui(self):
        """Synchronize configuration from all GUI variables."""
        for parameter_path, gui_variable in self.gui_variables.items():
            try:
                gui_value = gui_variable.get()
                self.config_manager.update_config_parameter(parameter_path, gui_value)
            except Exception as e:
                print(f"Error syncing config parameter {parameter_path}: {e}")
    
    def create_parameter_entry(self, parent: tk.Widget, parameter_path: str, 
                              label_text: str, row: int = None, 
                              width: int = 15) -> tk.Entry:
        """
        Create entry widget automatically bound to configuration parameter.
        
        Args:
            parent: Parent widget
            parameter_path: Dot-separated parameter path
            label_text: Label text for the entry
            row: Optional grid row
            width: Entry width
            
        Returns:
            Entry widget
        """
        # Determine variable type based on current value
        current_value = self.config_manager.get_config_parameter(parameter_path)
        
        if isinstance(current_value, float):
            variable = tk.DoubleVar()
        elif isinstance(current_value, int):
            variable = tk.IntVar()
        else:
            variable = tk.StringVar()
        
        # Register variable
        self.register_gui_variable(parameter_path, variable)
        
        # Create widgets
        label = tk.Label(parent, text=label_text, font=("Arial", 10))
        entry = tk.Entry(parent, textvariable=variable, width=width, font=("Arial", 10))
        
        if row is not None:
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        
        self.parameter_widgets[parameter_path] = (label, entry)
        
        return entry
    
    def validate_gui_inputs(self) -> List[str]:
        """
        Validate all GUI inputs.
        
        Returns:
            List of validation error messages
        """
        # First sync config from GUI
        self.sync_config_from_gui()
        
        # Then validate the configuration
        return self.config_manager.validate_config()
    
    def show_config_dialog(self, parent: tk.Widget = None) -> Optional[Path]:
        """
        Show configuration file dialog.
        
        Args:
            parent: Parent widget for dialog
            
        Returns:
            Selected file path or None if cancelled
        """
        file_path = filedialog.askopenfilename(
            parent=parent,
            title="Load Configuration File",
            filetypes=[
                ("JSON files", "*.json"),
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            return Path(file_path)
        return None
    
    def show_save_config_dialog(self, parent: tk.Widget = None) -> Optional[Path]:
        """
        Show save configuration file dialog.
        
        Args:
            parent: Parent widget for dialog
            
        Returns:
            Selected save path or None if cancelled
        """
        file_path = filedialog.asksaveasfilename(
            parent=parent,
            title="Save Configuration File",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("YAML files", "*.yaml"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            return Path(file_path)
        return None
    
    def show_validation_errors(self, errors: List[str], parent: tk.Widget = None):
        """
        Show validation errors in dialog.
        
        Args:
            errors: List of error messages
            parent: Parent widget for dialog
        """
        if errors:
            error_message = "Configuration validation failed:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showerror("Configuration Errors", error_message, parent=parent)
        else:
            messagebox.showinfo("Configuration Valid", "Configuration is valid!", parent=parent)


# ================================
# ERA5 Utility Functions
# ================================

def validate_era5_data(data: pd.DataFrame) -> None:
    """Validate ERA5 data ranges and units."""
    temp_air = data['temperature_2m']
    if temp_air.min() < -100 or temp_air.max() > 60:
        warnings.warn(f"Air temperature range seems unrealistic: {temp_air.min():.1f} to {temp_air.max():.1f} °C")
    
    humidity = data['relative_humidity']
    if humidity.min() < 0 or humidity.max() > 100:
        warnings.warn(f"Relative humidity outside 0-100% range: {humidity.min():.1f} to {humidity.max():.1f}")
    
    pressure = data['surface_pressure']
    if pressure.min() < 50000 or pressure.max() > 110000:
        warnings.warn(f"Surface pressure seems unrealistic: {pressure.min():.0f} to {pressure.max():.0f} Pa")

def calculate_era5_derived_variables(data: pd.DataFrame) -> None:
    """Calculate derived atmospheric variables from ERA5 data."""
    T_air = data['temperature_2m']
    T_skin = data['skin_temperature']
    
    # Saturation vapor pressure using Magnus formula
    es_air = 6.112 * np.exp(17.67 * T_air / (T_air + 243.5)) * 100  # Pa
    es_skin = 6.112 * np.exp(17.67 * T_skin / (T_skin + 243.5)) * 100  # Pa
    
    data['saturation_vapor_pressure_air'] = es_air
    data['saturation_vapor_pressure_skin'] = es_skin
    
    # Actual vapor pressure and concentration
    RH = data['relative_humidity']
    data['vapor_pressure_air'] = es_air * RH / 100.0
    
    # Vapor concentration (mol/cm³)
    R = 8.3144  # J/mol/K
    T_air_K = T_air + 273.15
    data['vapor_concentration_air'] = data['vapor_pressure_air'] / (R * T_air_K) * 1e-6
    data['vapor_pressure_deficit'] = es_air - data['vapor_pressure_air']

def setup_era5_numpy_arrays(data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert ERA5 DataFrame to NumPy arrays for fast lookup."""
    numeric_columns = [col for col in data.columns if col != 'datetime']
    numpy_data = data[numeric_columns].to_numpy()
    column_indices = {col: idx for idx, col in enumerate(numeric_columns)}
    return numpy_data, column_indices

def interpolate_era5_conditions(numpy_data: np.ndarray, column_indices: Dict[str, int], 
                               time_index: float) -> Dict[str, float]:
    """Fast interpolation of ERA5 conditions using NumPy arrays."""
    if time_index >= numpy_data.shape[0] - 1:
        # Use last data point
        idx = numpy_data.shape[0] - 1
        row = numpy_data[idx]
        return {col: row[col_idx] for col, col_idx in column_indices.items()}
    elif time_index <= 0:
        # Use first data point
        row = numpy_data[0]
        return {col: row[col_idx] for col, col_idx in column_indices.items()}
    else:
        # Linear interpolation
        idx_low = int(np.floor(time_index))
        idx_high = idx_low + 1
        weight_high = time_index - idx_low
        weight_low = 1.0 - weight_high
        
        row_low = numpy_data[idx_low]
        row_high = numpy_data[idx_high]
        
        conditions = {}
        for col, col_idx in column_indices.items():
            val_low = row_low[col_idx]
            val_high = row_high[col_idx]
            conditions[col] = weight_low * val_low + weight_high * val_high
        
        return conditions

# ================================
# ERA5 Atmospheric Forcing Classes
# ================================

class ERA5Forcing:
    """
    Handle ERA5 atmospheric forcing data for soil evaporation model.
    
    Provides time-interpolated atmospheric conditions including:
    - Surface and air temperatures
    - Relative humidity and vapor pressure  
    - Atmospheric pressure
    - Wind speed and solar radiation
    
    PERFORMANCE OPTIMIZATION: Uses class-level cache to prevent reloading same file.
    """
    
    # Class-level cache to share data between instances
    _cached_data = {}  # {file_path: (data, numpy_data, column_indices, metadata)}
    
    def __init__(self, era5_file_path: str):
        """Initialize with ERA5 data file using class-level cache."""
        self.era5_file_path = Path(era5_file_path)
        
        # Check class-level cache first
        cache_key = str(self.era5_file_path.resolve())
        if cache_key in ERA5Forcing._cached_data:
            # Use cached data
            cached = ERA5Forcing._cached_data[cache_key]
            self.data = cached['data']
            self._numpy_data = cached['numpy_data']
            self._column_indices = cached['column_indices']
            self.start_time = cached['start_time']
            self.end_time = cached['end_time']
            self.dt_hours = cached['dt_hours']
            print(f"✓ ERA5 data loaded from cache ({self._numpy_data.shape})")
        else:
            # Initialize fresh
            self.data = None
            self.start_time = None
            self.end_time = None
            self.dt_hours = None
            self._numpy_data = None
            self._column_indices = None
            self._load_era5_data()
    
    def _load_era5_data(self):
        """Load and validate ERA5 data."""
        if not self.era5_file_path.exists():
            raise FileNotFoundError(f"ERA5 data file not found: {self.era5_file_path}")
        
        
        # Load CSV data
        self.data = pd.read_csv(self.era5_file_path)
        
        # Parse datetime column
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        
        # Extract time information
        self.start_time = self.data['datetime'].iloc[0]
        self.end_time = self.data['datetime'].iloc[-1]
        
        # Calculate time step (assume regular intervals)
        if len(self.data) > 1:
            self.dt_hours = (self.data['datetime'].iloc[1] - self.data['datetime'].iloc[0]).total_seconds() / 3600
        else:
            self.dt_hours = 1.0
        
        # Validate required columns
        required_cols = ['temperature_2m', 'skin_temperature', 'relative_humidity', 'surface_pressure']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ERA5 data: {missing_cols}")
        
        # Convert units and validate ranges
        self._validate_and_convert_units()
        
    
    def _validate_and_convert_units(self):
        """Validate and convert ERA5 data units."""
        validate_era5_data(self.data)
        calculate_era5_derived_variables(self.data)
        self._setup_numpy_arrays()
    
    
    def _setup_numpy_arrays(self):
        """Convert DataFrame to NumPy arrays for fast O(1) lookup."""
        self._numpy_data, self._column_indices = setup_era5_numpy_arrays(self.data)
        
        print(f"✓ ERA5 data converted to NumPy arrays ({self._numpy_data.shape})")
        
        # Cache the processed data
        cache_key = str(self.era5_file_path.resolve())
        ERA5Forcing._cached_data[cache_key] = {
            'data': self.data,
            'numpy_data': self._numpy_data,
            'column_indices': self._column_indices,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'dt_hours': self.dt_hours
        }
    
    def get_conditions_at_time(self, simulation_days: float) -> Dict[str, float]:
        """
        Get atmospheric conditions at a specific simulation time using fast NumPy lookup.
        
        PERFORMANCE OPTIMIZATION: Replaced expensive pandas DataFrame operations
        with fast NumPy array indexing.
        
        Args:
            simulation_days: Time since simulation start in days
            
        Returns:
            Dictionary of atmospheric conditions
        """
        # Convert simulation time to hours
        hours_since_start = simulation_days * 24
        
        # Handle time wrapping (repeat data if simulation is longer)
        total_hours = self._numpy_data.shape[0] * self.dt_hours
        if hours_since_start > total_hours:
            hours_since_start = hours_since_start % total_hours
        
        # Find interpolation indices and interpolate
        time_index = hours_since_start / self.dt_hours
        return interpolate_era5_conditions(self._numpy_data, self._column_indices, time_index)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of the ERA5 data."""
        stats = {}
        
        key_vars = ['temperature_2m', 'skin_temperature', 'relative_humidity', 
                   'surface_pressure', 'vapor_pressure_air', 'vapor_pressure_deficit']
        
        for var in key_vars:
            if var in self.data.columns:
                stats[var] = {
                    'mean': self.data[var].mean(),
                    'min': self.data[var].min(), 
                    'max': self.data[var].max(),
                    'std': self.data[var].std()
                }
        
        return stats
    
    def get_time_range_days(self) -> float:
        """Get the total time range of the data in days."""
        return (self.end_time - self.start_time).total_seconds() / (24 * 3600)


def load_default_era5_forcing() -> ERA5Forcing:
    """Load the default ERA5 forcing data."""
    default_path = Path(__file__).parent / "sample_era5_summer2023.csv"
    
    if not default_path.exists():
        raise FileNotFoundError(f"Default ERA5 data file not found: {default_path}")
    
    return ERA5Forcing(default_path)


def load_era5_forcing_from_config(config) -> ERA5Forcing:
    """Load ERA5 forcing data based on configuration."""
    
    # Check if specific file is configured
    if hasattr(config, 'temperature') and config.temperature.era5_data_file:
        era5_file = config.temperature.era5_data_file
        
        # If directory is also specified, combine them
        if config.temperature.era5_data_directory:
            era5_path = Path(config.temperature.era5_data_directory) / era5_file
        else:
            # Try relative to script directory first
            era5_path = Path(era5_file)
            if not era5_path.exists():
                # Try in ERA5_DATA directory
                era5_path = Path(__file__).parent / "ERA5_DATA" / era5_file
    
    # Fallback to directory-based search
    elif hasattr(config, 'temperature') and config.temperature.era5_data_directory:
        era5_dir = Path(config.temperature.era5_data_directory)
        
        # Find the first CSV file in the directory
        csv_files = list(era5_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in ERA5 directory: {era5_dir}")
        
        era5_path = csv_files[0]  # Use first CSV file found
    
    # Ultimate fallback to default
    else:
        return load_default_era5_forcing()
    
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 data file not found: {era5_path}")
    
    return ERA5Forcing(era5_path)