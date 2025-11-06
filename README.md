# Soil Evaporation Model with Triple Oxygen Isotope Tracking

A comprehensive numerical model for simulating water evaporation from soils with simultaneous tracking of oxygen isotope compositions (δ¹⁸O, δ¹⁷O, Δ'¹⁷O).

## Authors & Attribution

- **Dan Breecker** (University of Texas at Austin) - Original scientific model development
- **Matthew Rybecky** (University of New Mexico) - Python implementation and optimization
- **Catt Peshek** (University of New Mexico) - Research collaboration and validation

## Citation

If you use this model in research, please cite:
[Manuscript in preparation - Breecker, D.L., Rybecky, M., Peshek, C.]

## Scientific Background

This model implements the Craig-Gordon equilibration theory for oxygen isotope fractionation during soil water evaporation. Key features include:

- **Vapor Diffusion**: Molecular diffusion of water isotopologues through soil pores using Fick's law
- **Equilibrium Fractionation**: Temperature-dependent liquid-vapor equilibrium based on Horita & Wesolowski (1994)
- **Kinetic Fractionation**: Mass-dependent diffusion rate differences between isotopologues
- **Atmospheric Forcing**: ERA5 meteorological data or constant atmospheric conditions
- **Boundary Layer Physics**: Aerodynamic resistance and flux/resistance coupling at soil surface

### Key Physics

1. **Diffusion Equation**: ∂C/∂t = ∂/∂z[D(z,T) × ∂C/∂z] for each isotopologue
2. **Craig-Gordon Equilibration**: Partitioning of water between liquid and vapor phases
3. **Fractionation Factors**: α₁₈₁₆ = exp[(-7.685 + 6713/T - 1.666×10⁶/T²) / 1000]
4. **Surface Flux**: Evaporation rate controlled by vapor pressure gradient and aerodynamic resistance

## Installation

### System Requirements

- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- Memory: 4GB RAM minimum, 8GB recommended
- Storage: 500MB for model and sample data

### Required Dependencies

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install plotly>=5.0.0
pip install pyyaml>=6.0
```

### GUI Dependencies

For the graphical interface, tkinter is required (usually included with Python):

**Linux users may need to install tkinter separately:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter

# Fedora
sudo dnf install python3-tkinter
```

### Download and Setup

1. **Clone or download** this repository to your local machine
2. **Navigate** to the model directory:
   ```bash
   cd Soil_Full_init
   ```
3. **Generate lookup tables** (required for fast execution):
   ```bash
   python generate_lookup_tables.py
   ```
4. **Test installation** by running the GUI:
   ```bash
   python gui_main.py
   ```

## Usage Guide

### Quick Start with GUI

1. **Launch the GUI application:**
   ```bash
   python gui_main.py
   ```

2. **Configure model parameters** using the interface:
   - Soil properties (porosity, depth, temperature profile)
   - Atmospheric conditions (temperature, humidity, isotope composition)
   - Numerical settings (simulation duration, time steps)

3. **Select input data** (optional):
   - ERA5 meteorological data (CSV format)
   - Field measurement data for validation

4. **Run simulation** and view results:
   - Real-time progress monitoring
   - Automatic plotting of results
   - Export options for data and figures

### Command Line Usage

For programmatic control or batch processing:

```python
from soil_model import SoilEvaporationModel
from config_manager import load_config

# Load configuration
config = load_config('config/default_config.yaml')

# Create and run model
model = SoilEvaporationModel(config)
model.setup_model()
results = model.run_simulation()

# Access results
print(f"Total evaporation: {results['evaporation_summary']['total_evaporation_mm']:.2f} mm")
```

### Sample Data

The model comes with sample datasets for testing and learning:

#### Sample ERA5 Data
- **File**: `sample_era5_summer2023.csv`
- **Location**: Representative mid-latitude site
- **Period**: Summer 2023 (typical evaporation conditions)
- **Variables**: Temperature, humidity, pressure, skin temperature
- **Temporal Resolution**: Hourly data

#### Field Data Examples
- **Location**: `field_data/` directory
- **Content**: Laboratory evaporation experiments
- **Format**: CSV files with time series measurements
- **Purpose**: Model validation and parameter calibration

## Model Configuration

### Configuration Files

The model uses YAML configuration files for parameter specification:

```yaml
# Example configuration structure
soil:
  depth: 30.0              # Soil column depth (cm)
  total_porosity: 0.4      # Total porosity (cm³/cm³)
  tortuosity: 0.66         # Tortuosity factor
  
atmospheric:
  mean_air_temperature: 25.0    # Air temperature (°C)
  relative_humidity: 50.0       # Relative humidity (%)
  d18O_vapor: -15.0            # δ¹⁸O of atmospheric vapor (‰)
  
numerical:
  run_days: 30.0           # Simulation duration (days)
  save_interval_hours: 6.0 # Output time resolution
```

### Parameter Sensitivity

Key parameters affecting model behavior:

1. **Soil Properties**:
   - Porosity: Controls vapor transport capacity
   - Tortuosity: Affects diffusion rate (typical range: 0.5-0.8)
   - Water content: Determines liquid/vapor partitioning

2. **Atmospheric Conditions**:
   - Temperature: Exponential effect on vapor pressure
   - Humidity: Controls evaporation driving force
   - Wind speed: Affects boundary layer resistance

3. **Isotope Composition**:
   - Initial δ¹⁸O: Starting isotope signature
   - Δ'¹⁷O: Triple oxygen isotope parameter

## Data Formats

### ERA5 Input Data

Required CSV format for meteorological forcing:

```csv
datetime,temperature_2m,skin_temperature,relative_humidity,surface_pressure
2023-06-01 00:00:00,22.5,24.1,65.2,101325
2023-06-01 01:00:00,21.8,23.4,68.1,101320
...
```

**Column Descriptions:**
- `datetime`: ISO format timestamp
- `temperature_2m`: Air temperature at 2m height (°C)
- `skin_temperature`: Soil surface temperature (°C)
- `relative_humidity`: Relative humidity (%)
- `surface_pressure`: Atmospheric pressure (Pa)

### Field Data Format

For model validation, use this CSV structure:

```csv
time_hours,temperature,humidity,evaporation_mm,d18O_vapor
0.0,25.0,50.0,0.0,-15.0
6.0,27.2,45.8,1.2,-12.5
12.0,29.1,42.3,2.8,-10.1
...
```

### Output Data

The model generates several output files:

#### Primary Results (`results_YYYYMMDD_HHMMSS.csv`)
- Time series of soil profiles
- Temperature, water content, isotope compositions
- Vapor and liquid concentrations at each depth

#### Evaporation Summary (`evaporation_summary.csv`)
- Time series of evaporation rates
- Cumulative evaporation
- Atmospheric conditions

#### Mass Balance (`mass_balance.csv`)
- Total water mass tracking
- Conservation verification
- Phase partitioning over time

## Interpretation of Results

### Key Output Variables

1. **Evaporation Rate (mm/day)**:
   - Instantaneous evaporation flux
   - Typically decreases exponentially with time
   - Sensitive to atmospheric conditions

2. **δ¹⁸O Evolution (‰)**:
   - Increasing values indicate evaporative enrichment
   - Surface > subsurface gradients develop
   - Asymptotic approach to atmospheric values

3. **Δ'¹⁷O Signatures (per meg)**:
   - Diagnostic of kinetic vs. equilibrium fractionation
   - Positive values indicate kinetic effects
   - Useful for process identification

### Physical Interpretation

- **Initial Phase**: Rapid evaporation with steep isotope gradients
- **Transition Phase**: Decreasing rates as surface water becomes enriched
- **Steady State**: Minimal evaporation, atmospheric equilibrium

### Troubleshooting Common Issues

1. **Mass Balance Errors**:
   - Check time step stability (automatically calculated)
   - Verify boundary condition consistency
   - Monitor numerical stability warnings

2. **Unrealistic Isotope Values**:
   - Verify atmospheric isotope inputs
   - Check fractionation factor calculations
   - Ensure reasonable temperature ranges

3. **Performance Issues**:
   - Use lookup tables (run `generate_lookup_tables.py`)
   - Reduce output frequency for long simulations
   - Consider parallel processing for batch runs

## Advanced Usage

### Batch Processing

For multiple simulations or parameter sensitivity analysis:

```python
import itertools
from soil_model import SoilEvaporationModel

# Parameter ranges
porosities = [0.3, 0.4, 0.5]
temperatures = [20, 25, 30]

# Run parameter sweep
for porosity, temp in itertools.product(porosities, temperatures):
    config = load_config('base_config.yaml')
    config.soil.total_porosity = porosity
    config.atmospheric.mean_air_temperature = temp
    
    model = SoilEvaporationModel(config)
    model.setup_model()
    results = model.run_simulation()
    
    # Save results with parameter information
    output_name = f"results_por{porosity}_temp{temp}.csv"
    # Process and save results...
```

### Custom Boundary Conditions

Implement time-varying atmospheric conditions:

```python
# Example: Diurnal temperature cycle
def custom_temperature(time_hours):
    """Custom temperature function with diurnal cycle."""
    daily_mean = 25.0  # °C
    amplitude = 10.0   # °C
    return daily_mean + amplitude * np.sin(2 * np.pi * time_hours / 24)

# Apply during simulation setup
config.atmospheric.use_custom_forcing = True
config.atmospheric.temperature_function = custom_temperature
```

## Scientific Applications

### Research Applications

1. **Paleoclimate Reconstruction**:
   - Soil carbonate isotope systematics
   - Evaporation vs. precipitation signals
   - Temperature proxy calibration

2. **Hydrology Studies**:
   - Soil-atmosphere water exchange
   - Evapotranspiration partitioning
   - Water balance quantification

3. **Laboratory Calibration**:
   - Experimental design guidance
   - Process understanding
   - Method validation

### Educational Use

- Graduate coursework in isotope geochemistry
- Demonstration of physical processes
- Numerical modeling training
- Research skill development

## Model Validation

The model has been validated against:

1. **Laboratory Experiments**:
   - Controlled evaporation studies
   - Known temperature and humidity conditions
   - Measured isotope evolution

2. **Field Observations**:
   - Natural soil profiles
   - Meteorological station data
   - Long-term monitoring sites

3. **Analytical Solutions**:
   - Simplified cases with known solutions
   - Mass balance verification
   - Limiting behavior checks

## License

This software is distributed under the MIT License. See `LICENSE` file for full terms.

## Support and Contributing

### Reporting Issues

Please report bugs, feature requests, or questions by:

1. **GitHub Issues**: Create an issue with detailed description
2. **Email Contact**: Reach out to the development team
3. **Documentation**: Check this README and code comments first

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

### Development Setup

For contributors:

```bash
# Clone repository
git clone <repository-url>
cd Soil_Full_init

# Install development dependencies
pip install -r requirements.txt
pip install pytest  # For testing

# Run tests
pytest tests/

# Generate documentation
python generate_lookup_tables.py
python -m doctest soil_model.py
```

## Acknowledgments

- **Funding**: [Funding source information]
- **Collaborators**: Research group members and advisors
- **Software**: Built with Python scientific computing ecosystem
- **Data**: ERA5 reanalysis data from Copernicus Climate Change Service

## References

1. Horita, J., & Wesolowski, D. J. (1994). Liquid-vapor fractionation of oxygen and hydrogen isotopes of water from the freezing to the critical temperature. *Geochimica et Cosmochimica Acta*, 58(16), 3425-3437.

2. Craig, H., & Gordon, L. I. (1965). Deuterium and oxygen 18 variations in the ocean and the marine atmosphere. In *Stable Isotopes in Oceanographic Studies and Paleotemperatures* (pp. 9-130).

3. Luz, B., & Barkan, E. (2010). Variations of 17O/16O and 18O/16O in meteoric waters. *Geochimica et Cosmochimica Acta*, 74(22), 6276-6286.

---

**Version**: 2.0  
**Last Updated**: November 2024  
**Documentation**: Comprehensive  
**Status**: Production Ready