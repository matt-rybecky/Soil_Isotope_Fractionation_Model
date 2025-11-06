"""
Output Data Management System
============================

Handles persistent storage and retrieval of model runs with configuration and results.
Enables viewing of previous model runs across multiple sessions.

Author: Enhanced Python Implementation for Soil Evaporation Research
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from config_manager import ModelConfiguration


@dataclass
class RunMetadata:
    """Metadata for a model run."""
    timestamp: str
    run_type: str  # "single_run"
    description: str
    version: str = "1.0"
    execution_time_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class OutputDataManager:
    """
    Manages persistent storage of model runs with configuration and results.
    
    Provides methods to save complete model runs and retrieve them for plotting.
    """
    
    def __init__(self, output_directory: Path = None):
        """
        Initialize output data manager.
        
        Args:
            output_directory: Directory to store output files (default: ./output_results)
        """
        if output_directory is None:
            output_directory = Path.cwd() / "output_results"
        
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # File naming patterns - compressed format for optimal performance 
        self.complete_pattern = "soil_evap_complete_*.npz"
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Handle dataclass/object
            try:
                return asdict(obj)
            except:
                return str(obj)
        else:
            return obj
    
    def _generate_timestamp(self) -> str:
        """Generate timestamp for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_description(self, run_type: str, config: ModelConfiguration) -> str:
        """Generate a user-friendly description for the run."""
        try:
            # Use config summary as description for better detail
            return self._create_config_summary(config.to_dict())
        except Exception:
            if run_type == "single_run":
                return f"Single Model Run - {config.numerical.run_days:.0f} days"
            else:
                return f"{run_type.title()} - {config.numerical.run_days:.0f} days"
    
    def save_complete_run(self, 
                         config: ModelConfiguration,
                         model_results: Dict[str, Any],
                         run_type: str = "single_run",
                         description: str = None,
                         execution_time: float = None,
                         success: bool = True,
                         error_message: str = None) -> Path:
        """
        Save complete model run with configuration and results.
        
        Args:
            config: Model configuration used for the run
            model_results: Complete model output data
            run_type: Type of run ("single_run")
            description: Optional custom description
            execution_time: Runtime in seconds
            success: Whether the run was successful
            error_message: Error message if run failed
            
        Returns:
            Path to saved file
        """
        timestamp = self._generate_timestamp()
        
        # Create metadata
        if description is None:
            description = self._generate_description(run_type, config)
            
        metadata = RunMetadata(
            timestamp=timestamp,
            run_type=run_type,
            description=description,
            execution_time_seconds=execution_time,
            success=success,
            error_message=error_message
        )
        
        # Optimized storage format: metadata as JSON + data as compressed NumPy
        # Metadata and configuration -> JSON (human readable, small size)
        metadata_config = {
            "metadata": asdict(metadata),
            "configuration": self._make_json_serializable(config.to_dict()),
            "format_version": "1.0"
        }
        
        # Results -> Compressed NumPy format (efficient for large scientific arrays)
        numpy_data = {}
        json_data = {}
        
        if model_results:
            for key, value in model_results.items():
                if isinstance(value, np.ndarray):
                    numpy_data[key] = value
                else:
                    json_data[key] = self._make_json_serializable(value)
        
        # Add non-array results to metadata file
        if json_data:
            metadata_config["scalar_results"] = json_data
        
        # Save to paired files for optimal I/O performance
        base_filename = f"soil_evap_complete_{timestamp}"
        metadata_file = self.output_directory / f"{base_filename}.json"
        data_file = self.output_directory / f"{base_filename}.npz"
        
        try:
            # Save metadata and configuration as readable JSON
            with open(metadata_file, 'w') as f:
                json.dump(metadata_config, f, indent=2)
            
            # Save large arrays as compressed NumPy format (10x faster I/O, ~90% smaller)
            if numpy_data:
                np.savez_compressed(data_file, **numpy_data)
            
            print(f"✓ Complete model run saved: {base_filename}")
            return data_file
            
        except Exception as e:
            print(f"❌ Error saving complete run: {e}")
            raise
    
    def load_complete_run(self, filepath: Path) -> Dict[str, Any]:
        """
        Load complete model run from compressed format.
        
        Args:
            filepath: Path to saved run data file (.npz)
            
        Returns:
            Dictionary with metadata, configuration, and results
        """
        filepath = Path(filepath)
        
        try:
            # Construct metadata file path from data file path
            base_name = filepath.stem  # Remove .npz extension
            metadata_filepath = filepath.parent / f"{base_name}.json"
            
            # Load metadata and configuration from JSON
            with open(metadata_filepath, 'r') as f:
                metadata_config = json.load(f)
            
            # Load compressed array data from NPZ
            compressed_data = np.load(filepath)
            results = dict(compressed_data)
            
            # Combine scalar results from metadata if present
            if "scalar_results" in metadata_config:
                results.update(metadata_config["scalar_results"])
                del metadata_config["scalar_results"]
            
            # Combine into expected format
            return {
                "metadata": metadata_config["metadata"],
                "configuration": metadata_config["configuration"], 
                "results": results,
                "format_version": metadata_config.get("format_version", "1.0")
            }
                
        except Exception as e:
            print(f"❌ Error loading run from {filepath}: {e}")
            raise
    
    def get_available_runs(self) -> List[Dict[str, Any]]:
        """
        Get list of all available complete runs with metadata (FAST VERSION).
        
        PERFORMANCE OPTIMIZATION: Only reads filenames, lazy loads metadata on demand.
        
        Returns:
            List of dictionaries with run information
        """
        runs = []
        
        # Find all complete run files (FAST - just filename scanning)
        complete_files = list(self.output_directory.glob(self.complete_pattern))
        
        for filepath in sorted(complete_files, reverse=True):  # Most recent first
            try:
                # Extract basic info from filename without reading file
                filename = filepath.name
                # Extract timestamp from filename: soil_evap_complete_YYYYMMDD_HHMMSS.npz
                if filename.startswith("soil_evap_complete_") and filename.endswith(".npz"):
                    timestamp_part = filename[19:-4]  # Remove prefix and .npz
                    
                    run_info = {
                        "filepath": filepath,
                        "filename": filename,
                        "timestamp": timestamp_part,
                        "metadata": None,  # Lazy load when needed
                        "config_summary": None,  # Lazy load when needed
                        "_loaded": False
                    }
                    runs.append(run_info)
                    
            except Exception as e:
                print(f"⚠️ Could not process run file {filepath}: {e}")
                continue
        
        return runs
    
    def load_run_metadata(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lazy load metadata for a specific run (called on demand).
        
        Args:
            run_info: Run info dictionary from get_available_runs()
            
        Returns:
            Updated run_info with loaded metadata
        """
        if run_info.get("_loaded", False):
            return run_info  # Already loaded
            
        npz_filepath = run_info["filepath"]
        
        # Construct corresponding JSON metadata file path
        base_name = npz_filepath.stem  # Remove .npz extension
        metadata_filepath = npz_filepath.parent / f"{base_name}.json"
        
        try:
            with open(metadata_filepath, 'r') as f:
                data = json.load(f)
            
            if "metadata" in data:
                run_info["metadata"] = data["metadata"]
                run_info["configuration"] = data.get("configuration", {})
                run_info["config_summary"] = self._create_config_summary(data.get("configuration", {}))
                run_info["_loaded"] = True
                
        except Exception as e:
            print(f"⚠️ Could not load metadata for {npz_filepath}: {e}")
            run_info["metadata"] = {"error": f"Failed to load: {e}"}
            run_info["config_summary"] = "Error loading config"
            
        return run_info
    
    def _create_config_summary(self, config: Dict[str, Any]) -> str:
        """Create a comprehensive summary of configuration parameters for easy identification.
        
        Format: #days | ERA5+diffusion flags | soil conditions | vapor conditions | atmospheric conditions
        """
        try:
            summary_parts = []
            
            # 1. Days
            if "numerical" in config and "run_days" in config["numerical"]:
                summary_parts.append(f"{config['numerical']['run_days']:.0f}d")
            
            # 2. ERA5 and diffusion flags
            flags = []
            
            # ERA5 flag
            era5_enabled = False
            if "temperature" in config and "era5_data_file" in config["temperature"]:
                era5_file = config["temperature"].get("era5_data_file")
                era5_enabled = bool(era5_file)
            flags.append("ERA5" if era5_enabled else "noERA5")
            
            # Temperature diffusion flag - follows actual model logic
            # ERA5 mode automatically enables temperature diffusion (see soil_model.py:1266)
            if era5_enabled:
                # When ERA5 is enabled, diffusion is always on - don't show any diffusion flag
                pass  # Temperature diffusion is implicit with ERA5
            else:
                # When no ERA5, show diffusion status based on config
                # Note: config uses 'disable_diffusion' (when True, diffusion is OFF)
                temp_diffusion_disabled = config.get("temperature", {}).get("disable_diffusion", False)
                flags.append("noTempDiff" if temp_diffusion_disabled else "TempDiff")
            
            summary_parts.append("+".join(flags))
            
            # 3. Soil conditions
            soil_parts = []
            if "soil" in config:
                soil_config = config["soil"]
                
                # Water content profile
                if "water_content_profile" in soil_config:
                    wc_profile = soil_config["water_content_profile"]
                    profile_type = wc_profile.get("profile_type", "const")
                    if profile_type == "constant":
                        soil_parts.append(f"WC=const({wc_profile.get('constant_value', 0):.3f})")
                    elif profile_type == "linear":
                        soil_parts.append(f"WC=lin({wc_profile.get('surface_value', 0):.3f}-{wc_profile.get('bottom_value', 0):.3f})")
                    elif profile_type == "exponential":
                        soil_parts.append(f"WC=exp(surf={wc_profile.get('surface_value_exp', 0):.3f})")
                    elif profile_type == "from_file":
                        soil_parts.append("WC=file")
                
                # Soil depth and porosity
                if "depth" in soil_config:
                    soil_parts.append(f"{soil_config['depth']:.0f}cm")
                if "total_porosity" in soil_config:
                    soil_parts.append(f"φ={soil_config['total_porosity']:.2f}")
            
            if soil_parts:
                summary_parts.append(" ".join(soil_parts))
            
            # 4. Vapor conditions (initial isotope conditions)
            vapor_parts = []
            if "isotope" in config:
                iso_config = config["isotope"]
                if "initial_d18O" in iso_config:
                    vapor_parts.append(f"δ18O={iso_config['initial_d18O']:.1f}‰")
                if "initial_d17O_excess" in iso_config:
                    vapor_parts.append(f"Δ'17O={iso_config['initial_d17O_excess']:.0f}pm")
            
            if vapor_parts:
                summary_parts.append(" ".join(vapor_parts))
            
            # 5. Atmospheric conditions
            atm_parts = []
            
            # Temperature profile
            if "temperature" in config and "temperature_profile" in config["temperature"]:
                temp_profile = config["temperature"]["temperature_profile"]
                profile_type = temp_profile.get("profile_type", "const")
                if profile_type == "constant":
                    atm_parts.append(f"T=const({temp_profile.get('constant_value', 0):.1f}°C)")
                elif profile_type == "linear":
                    atm_parts.append(f"T=lin({temp_profile.get('surface_value', 0):.1f}-{temp_profile.get('bottom_value', 0):.1f}°C)")
                elif profile_type == "exponential":
                    atm_parts.append(f"T=exp")
                elif profile_type == "from_file":
                    atm_parts.append("T=file")
            
            # Atmospheric isotopes
            if "atmospheric" in config:
                atm_config = config["atmospheric"]
                if "d18O_rain" in atm_config:
                    atm_parts.append(f"Rain={atm_config['d18O_rain']:.1f}‰")
                if "humidity" in atm_config:
                    atm_parts.append(f"RH={atm_config['humidity']:.0f}%")
            
            if atm_parts:
                summary_parts.append(" ".join(atm_parts))
            
            return " | ".join(summary_parts) if summary_parts else "N/A"
            
        except Exception:
            return "N/A"
    
    def create_detailed_config_info(self, config: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Create detailed configuration information for display in Results Information panel."""
        try:
            info_lines = []
            
            # Run metadata if available
            if metadata:
                info_lines.append("RUN METADATA")
                info_lines.append("=" * 40)
                info_lines.append(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
                info_lines.append(f"Run Type: {metadata.get('run_type', 'N/A')}")
                info_lines.append(f"Success: {'✓' if metadata.get('success', True) else '✗'}")
                if metadata.get('execution_time_seconds'):
                    info_lines.append(f"Execution Time: {metadata['execution_time_seconds']:.1f}s")
                info_lines.append("")
            
            # Numerical Configuration
            if "numerical" in config:
                info_lines.append("NUMERICAL PARAMETERS")
                info_lines.append("=" * 40)
                num_config = config["numerical"]
                info_lines.append(f"Run Days: {num_config.get('run_days', 'N/A')}")
                info_lines.append(f"Time Step: {num_config.get('time_step', 'N/A')} hours")
                info_lines.append(f"Depth Step: {num_config.get('depth_step', 'N/A')} cm")
                info_lines.append(f"Save Interval: {num_config.get('save_interval_hours', 'N/A')} hours")
                info_lines.append("")
            
            # Soil Configuration
            if "soil" in config:
                info_lines.append("SOIL PARAMETERS")
                info_lines.append("=" * 40)
                soil_config = config["soil"]
                info_lines.append(f"Depth: {soil_config.get('depth', 'N/A')} cm")
                info_lines.append(f"Total Porosity: {soil_config.get('total_porosity', 'N/A')}")
                info_lines.append(f"Tortuosity: {soil_config.get('tortuosity', 'N/A')}")
                info_lines.append(f"Water Content Threshold: {soil_config.get('water_content_threshold', 'N/A')}")
                
                # Water content profile details
                if 'water_content_profile' in soil_config:
                    wc_profile = soil_config['water_content_profile']
                    info_lines.append(f"Water Content Profile: {wc_profile.get('profile_type', 'N/A')}")
                    if wc_profile.get('profile_type') == 'constant':
                        info_lines.append(f"  Constant Value: {wc_profile.get('constant_value', 'N/A')}")
                    elif wc_profile.get('profile_type') == 'linear':
                        info_lines.append(f"  Surface Value: {wc_profile.get('surface_value', 'N/A')}")
                        info_lines.append(f"  Bottom Value: {wc_profile.get('bottom_value', 'N/A')}")
                    elif wc_profile.get('profile_type') == 'exponential':
                        info_lines.append(f"  Surface Value: {wc_profile.get('surface_value_exp', 'N/A')}")
                        info_lines.append(f"  Decay Length: {wc_profile.get('decay_length', 'N/A')}")
                        info_lines.append(f"  Background: {wc_profile.get('background_value', 'N/A')}")
                    elif wc_profile.get('profile_type') == 'from_file':
                        info_lines.append(f"  Profile File: {wc_profile.get('profile_file', 'N/A')}")
                info_lines.append("")
            
            # Temperature Configuration
            if "temperature" in config:
                info_lines.append("TEMPERATURE PARAMETERS")
                info_lines.append("=" * 40)
                temp_config = config["temperature"]
                info_lines.append(f"Deep Earth Temp: {temp_config.get('deep_earth_temperature', 'N/A')} °C")
                info_lines.append(f"Thermal Diffusivity: {temp_config.get('thermal_diffusivity', 'N/A')} cm²/s")
                # Note: config uses 'disable_diffusion' (when True, diffusion is OFF)
                temp_diffusion_disabled = temp_config.get('disable_diffusion', False)
                info_lines.append(f"Temperature Diffusion: {'Disabled' if temp_diffusion_disabled else 'Enabled'}")
                
                # ERA5 data status
                era5_file = temp_config.get('era5_data_file', None)
                info_lines.append(f"ERA5 Data: {'Yes' if era5_file else 'No'}")
                if era5_file:
                    info_lines.append(f"  ERA5 File: {Path(era5_file).name}")
                
                # Temperature profile details
                if 'temperature_profile' in temp_config:
                    temp_profile = temp_config['temperature_profile']
                    info_lines.append(f"Temperature Profile: {temp_profile.get('profile_type', 'N/A')}")
                    if temp_profile.get('profile_type') == 'constant':
                        info_lines.append(f"  Constant Value: {temp_profile.get('constant_value', 'N/A')} °C")
                    elif temp_profile.get('profile_type') == 'linear':
                        info_lines.append(f"  Surface Value: {temp_profile.get('surface_value', 'N/A')} °C")
                        info_lines.append(f"  Bottom Value: {temp_profile.get('bottom_value', 'N/A')} °C")
                    elif temp_profile.get('profile_type') == 'exponential':
                        info_lines.append(f"  Surface Value: {temp_profile.get('surface_value_exp', 'N/A')} °C")
                        info_lines.append(f"  Decay Length: {temp_profile.get('decay_length', 'N/A')}")
                        info_lines.append(f"  Background: {temp_profile.get('background_value', 'N/A')} °C")
                    elif temp_profile.get('profile_type') == 'from_file':
                        info_lines.append(f"  Profile File: {Path(temp_profile.get('profile_file', '')).name}")
                info_lines.append("")
            
            # Atmospheric Configuration
            if "atmospheric" in config:
                info_lines.append("ATMOSPHERIC PARAMETERS")
                info_lines.append("=" * 40)
                atm_config = config["atmospheric"]
                info_lines.append(f"Rain δ¹⁸O: {atm_config.get('d18O_rain', 'N/A')} ‰")
                info_lines.append(f"Rain Δ'¹⁷O: {atm_config.get('D_prime_17O_rain', 'N/A')} per meg")
                info_lines.append(f"Vapor δ¹⁸O: {atm_config.get('d18O_vapor', 'N/A')} ‰")
                info_lines.append(f"Vapor Δ'¹⁷O: {atm_config.get('D_prime_17O_vapor', 'N/A')} per meg")
                info_lines.append(f"Temperature: {atm_config.get('temperature', 'N/A')} °C")
                info_lines.append(f"Humidity: {atm_config.get('humidity', 'N/A')} %")
                info_lines.append(f"Pressure: {atm_config.get('pressure', 'N/A')} kPa")
                info_lines.append("")
            
            # Isotope Configuration
            if "isotope" in config:
                info_lines.append("ISOTOPE PARAMETERS")
                info_lines.append("=" * 40)
                iso_config = config["isotope"]
                info_lines.append(f"Initial δ¹⁸O: {iso_config.get('initial_d18O', 'N/A')} ‰")
                info_lines.append(f"Initial Δ'¹⁷O: {iso_config.get('initial_d17O_excess', 'N/A')} per meg")
                info_lines.append(f"Fractionation Model: {iso_config.get('fractionation_model', 'N/A')}")
                info_lines.append(f"θ Parameter: {iso_config.get('theta', 'N/A')}")
                info_lines.append(f"λ Parameter: {iso_config.get('lambda_value', 'N/A')}")
                info_lines.append("")
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"Error creating configuration info: {str(e)}"
    
    def cleanup_old_files(self, keep_recent: int = 50):
        """
        Clean up old files, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent files to keep
        """
        complete_files = sorted(self.output_directory.glob(self.complete_pattern))
        
        if len(complete_files) > keep_recent:
            files_to_remove = complete_files[:-keep_recent]
            for filepath in files_to_remove:
                try:
                    filepath.unlink()
                    print(f"🗑️ Removed old file: {filepath.name}")
                except Exception as e:
                    print(f"⚠️ Could not remove {filepath.name}: {e}")
    
    def get_run_by_timestamp(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific run by its timestamp.
        
        Args:
            timestamp: Timestamp string (e.g., "20250922_143015")
            
        Returns:
            Complete run data or None if not found
        """
        filename = f"soil_evap_complete_{timestamp}.npz"
        filepath = self.output_directory / filename
        
        if filepath.exists():
            return self.load_complete_run(filepath)
        else:
            return None


# Singleton instance for global use
output_data_manager = OutputDataManager()