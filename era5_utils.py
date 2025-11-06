"""
ERA5 Data Processing Utilities
=============================

Handles ERA5 meteorological data loading, validation, and interpolation
for the soil evaporation model.

Author: Enhanced Python Implementation for Soil Evaporation Research
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def validate_era5_data(data: pd.DataFrame) -> None:
    """Validate ERA5 data format and required columns."""
    required_columns = ['datetime', 'temperature_2m', 'skin_temperature', 
                       'relative_humidity', 'surface_pressure']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required ERA5 columns: {missing_columns}")
    
    # Validate data ranges
    if data['temperature_2m'].min() < -100 or data['temperature_2m'].max() > 100:
        raise ValueError("Temperature values outside reasonable range (-100 to 100°C)")
    
    if data['relative_humidity'].min() < 0 or data['relative_humidity'].max() > 100:
        raise ValueError("Relative humidity must be between 0 and 100%")
    
    if data['surface_pressure'].min() < 50000 or data['surface_pressure'].max() > 120000:
        raise ValueError("Surface pressure outside reasonable range (50-120 kPa)")


def calculate_era5_derived_variables(data: pd.DataFrame) -> None:
    """Calculate derived meteorological variables from ERA5 data."""
    # Vapor pressure calculations
    data['vapor_pressure_air'] = (6.112 * np.exp(17.67 * data['temperature_2m'] / 
                                                 (data['temperature_2m'] + 243.5)) * 100 * 
                                 data['relative_humidity'] / 100.0)
    
    # Saturation vapor pressure
    data['saturation_vapor_pressure'] = (6.112 * np.exp(17.67 * data['temperature_2m'] / 
                                                        (data['temperature_2m'] + 243.5)) * 100)
    
    # Vapor pressure deficit
    data['vapor_pressure_deficit'] = data['saturation_vapor_pressure'] - data['vapor_pressure_air']
    
    # Vapor concentration in air (mol/cm³)
    R = 8.3144  # J/(mol·K)
    data['vapor_concentration_air'] = (data['vapor_pressure_air'] / 
                                      (R * (data['temperature_2m'] + 273.15))) * 1e-6


def setup_era5_numpy_arrays(data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert ERA5 DataFrame to NumPy arrays for fast lookup."""
    # Define column order for numpy array
    columns = ['temperature_2m', 'skin_temperature', 'relative_humidity', 
              'surface_pressure', 'vapor_pressure_air', 'vapor_pressure_deficit',
              'vapor_concentration_air']
    
    # Create column index mapping
    column_indices = {col: i for i, col in enumerate(columns)}
    
    # Extract data as numpy array
    numpy_data = data[columns].values
    
    return numpy_data, column_indices


def interpolate_era5_conditions(numpy_data: np.ndarray, column_indices: Dict[str, int], 
                               time_index: float) -> Dict[str, float]:
    """Fast interpolation of ERA5 conditions using numpy arrays."""
    # Handle edge cases
    if time_index <= 0:
        row = numpy_data[0, :]
    elif time_index >= numpy_data.shape[0] - 1:
        row = numpy_data[-1, :]
    else:
        # Linear interpolation
        lower_index = int(np.floor(time_index))
        upper_index = min(lower_index + 1, numpy_data.shape[0] - 1)
        fraction = time_index - lower_index
        
        row = (numpy_data[lower_index, :] * (1 - fraction) + 
               numpy_data[upper_index, :] * fraction)
    
    # Return as dictionary
    return {col: float(row[idx]) for col, idx in column_indices.items()}


class ERA5Forcing:
    """ERA5 meteorological forcing data handler with caching."""
    
    # Class-level cache to share data between instances
    _cached_data = {}  # {file_path: {numpy_data, column_indices, metadata}}
    
    def __init__(self, era5_file_path: str):
        """Initialize with ERA5 data file using class-level cache."""
        self.era5_file_path = Path(era5_file_path)
        
        # Check class-level cache first
        cache_key = str(self.era5_file_path.resolve())
        if cache_key in ERA5Forcing._cached_data:
            # Use cached data
            cached = ERA5Forcing._cached_data[cache_key]
            self._numpy_data = cached['numpy_data']
            self._column_indices = cached['column_indices']
            self.start_time = cached['start_time']
            self.end_time = cached['end_time']
            self.dt_hours = cached['dt_hours']
            print(f"✓ Using cached ERA5 data from {self.era5_file_path.name}")
        else:
            # Load new data
            self._load_era5_data()
    
    def _load_era5_data(self):
        """Load and validate ERA5 data."""
        if not self.era5_file_path.exists():
            raise FileNotFoundError(f"ERA5 data file not found: {self.era5_file_path}")
        
        # Load CSV data
        data = pd.read_csv(self.era5_file_path)
        
        # Parse datetime column
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Extract time information
        self.start_time = data['datetime'].iloc[0]
        self.end_time = data['datetime'].iloc[-1]
        
        # Calculate time step (assume regular intervals)
        if len(data) > 1:
            self.dt_hours = (data['datetime'].iloc[1] - data['datetime'].iloc[0]).total_seconds() / 3600
        else:
            self.dt_hours = 1.0
        
        # Validate required columns
        required_cols = ['temperature_2m', 'skin_temperature', 'relative_humidity', 'surface_pressure']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ERA5 data: {missing_cols}")
        
        # Convert units and validate ranges
        self._validate_and_convert_units(data)
    
    def _validate_and_convert_units(self, data: pd.DataFrame):
        """Validate and convert ERA5 data units."""
        validate_era5_data(data)
        calculate_era5_derived_variables(data)
        self._setup_numpy_arrays(data)
    
    def _setup_numpy_arrays(self, data: pd.DataFrame):
        """Convert DataFrame to NumPy arrays for fast O(1) lookup."""
        self._numpy_data, self._column_indices = setup_era5_numpy_arrays(data)
        
        print(f"✓ ERA5 data converted to NumPy arrays ({self._numpy_data.shape})")
        
        # Cache the processed data (numpy arrays only for memory efficiency)
        cache_key = str(self.era5_file_path.resolve())
        ERA5Forcing._cached_data[cache_key] = {
            # pandas DataFrame removed to save memory
            'numpy_data': self._numpy_data,
            'column_indices': self._column_indices,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'dt_hours': self.dt_hours
        }
        
        # Free the pandas DataFrame memory after conversion
        del data
    
    def get_conditions_at_time(self, simulation_days: float) -> Dict[str, float]:
        """
        Get atmospheric conditions at a specific simulation time using fast NumPy lookup.
        
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
        """Get summary statistics of the ERA5 data using numpy arrays."""
        stats = {}
        
        key_vars = ['temperature_2m', 'skin_temperature', 'relative_humidity', 
                   'surface_pressure', 'vapor_pressure_air', 'vapor_pressure_deficit']
        
        for var in key_vars:
            if var in self._column_indices:
                col_idx = self._column_indices[var]
                data_col = self._numpy_data[:, col_idx]
                stats[var] = {
                    'mean': float(np.mean(data_col)),
                    'min': float(np.min(data_col)), 
                    'max': float(np.max(data_col)),
                    'std': float(np.std(data_col))
                }
        
        return stats
    
    def get_time_range_days(self) -> float:
        """Get the total time range of the data in days."""
        return (self.end_time - self.start_time).total_seconds() / (24 * 3600)


def load_default_era5_forcing() -> ERA5Forcing:
    """Load default ERA5 forcing data."""
    default_path = Path(__file__).parent / "sample_era5_summer2023.csv"
    if not default_path.exists():
        raise FileNotFoundError(f"Default ERA5 file not found: {default_path}")
    return ERA5Forcing(str(default_path))


def load_era5_forcing_from_config(config) -> ERA5Forcing:
    """Load ERA5 forcing data from configuration."""
    try:
        # Check if ERA5 forcing is enabled
        if not hasattr(config.temperature, 'use_era5_forcing') or not config.temperature.use_era5_forcing:
            return None
        
        # Get ERA5 file path from config
        if hasattr(config.temperature, 'era5_data_file') and config.temperature.era5_data_file:
            # Check if it's just a filename or full path
            if hasattr(config.temperature, 'era5_data_directory') and config.temperature.era5_data_directory:
                era5_file_path = Path(__file__).parent / config.temperature.era5_data_directory / config.temperature.era5_data_file
            else:
                era5_file_path = Path(__file__).parent / config.temperature.era5_data_file
        elif hasattr(config.temperature, 'era5_file_path') and config.temperature.era5_file_path:
            era5_file_path = Path(config.temperature.era5_file_path)
        else:
            # Try default locations
            base_dir = Path(__file__).parent
            era5_data_dir = base_dir / "ERA5_DATA"
            
            # Look for ERA5 files in standard locations
            for pattern in ["*.csv", "era5*.csv", "sample_era5*.csv"]:
                files = list(era5_data_dir.glob(pattern))
                if files:
                    era5_file_path = files[0]
                    break
                files = list(base_dir.glob(pattern))
                if files:
                    era5_file_path = files[0]
                    break
            else:
                return load_default_era5_forcing()
        
        if not era5_file_path.exists():
            print(f"Warning: ERA5 file not found: {era5_file_path}")
            return load_default_era5_forcing()
        
        return ERA5Forcing(str(era5_file_path))
        
    except Exception as e:
        print(f"Warning: Failed to load ERA5 data: {e}")
        return load_default_era5_forcing()