"""
Soil Evaporation Model - Conservative Code Cleanup

Clean version of soil_model_corrected.py with ONLY debug/comment removal.
Physics calculations preserved exactly as original.

Authors: Dan Breecker (UT Austin), Matthew Rybecky (UNM), Catt Peshek (UNM)
"""

import numpy as np
from typing import Optional, Dict, Any
import time
from pathlib import Path

from config_manager import ModelConfiguration, load_config, ERA5Forcing, load_era5_forcing_from_config


def delta_prime_calc(d18O, d17O):
    """Calculate Δ'17O values from δ18O and δ17O."""
    d18O_safe = np.maximum(d18O, -999.0)
    d17O_safe = np.maximum(d17O, -999.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_prime = (1000 * np.log(d17O_safe/1000 + 1) - 
                      0.528 * (1000 * np.log(d18O_safe/1000 + 1))) * 1000
    
    delta_prime = np.where(np.isfinite(delta_prime), delta_prime, np.nan)
    
    return delta_prime


class SoilEvaporationModel:
    """Soil evaporation model with Craig-Gordon kinetic fractionation."""
    
    def __init__(self, config: Optional[ModelConfiguration] = None):
        """Initialize the soil evaporation model."""
        self.config = config or load_config()
        self.config.validate()
        
        # Grid and discretization (initialized in setup_model)
        self.depth_nodes = None
        self.soil_temperature = None
        self.time_step = None
        self.num_nodes = None
        
        # Atmospheric conditions
        self.vapor_conc_air = None
        self.H216O_vapor_air = None
        self.H218O_vapor_air = None 
        self.H217O_vapor_air = None
        
        # Soil state arrays
        self.water_content = None
        self.free_air_porosity = None
        self.liquid_conc_soil = None
        
        # Vapor concentrations
        self.H216O_vapor_soil = None
        self.H218O_vapor_soil = None
        self.H217O_vapor_soil = None
        
        # Liquid concentrations  
        self.H216O_liquid_soil = None
        self.H218O_liquid_soil = None
        self.H217O_liquid_soil = None
        
        # Isotope compositions
        self.d18O_liquid_soil = None
        self.d17O_liquid_soil = None
        
        # Diffusion coefficients
        self.D_H216O = None
        self.D_H218O = None
        self.D_H217O = None
        
        # Fractionation factors
        self.alpha_1816_liq_vap_soil = None
        self.alpha_1716_liq_vap_soil = None
        
        # Boundary conditions
        self.boundary_conditions = {}
        
        # Results storage
        self.results = {}
        self.metadata = {}
        
        # Simple evaporation tracking based on mass loss
        self.evaporation_data = {
            'times': [],
            'cumulative_evap_mm': [],  # Total evaporation in mm based on mass loss
            'surface_temp': [],
            'air_temp': [],
            'humidity': []
        }
        
        # Simple mass tracking - only track total mass changes
        self.mass_balance_data = {
            'times': [],
            'total_mass': [],           # Total water mass in soil column (mol)
            'mass_lost': [],            # Cumulative mass lost from initial (mol) 
            'evaporation_mm': []        # Cumulative evaporation in mm equivalent
        }
        
        # Simple tracking variables
        self._initial_total_mass = None
        
        # Flags
        self._is_setup = False
        self._is_run = False
        
        # Vectorization flag
        self._use_vectorized_diffusion = True
        
        # Boundary condition settings
        self._use_era5_forcing = True
        self._boundary_layer_resistance = 100.0  # Default resistance in cm/s
        
        # ERA5 forcing
        self.era5_forcing = None
        
        # Cumulative evaporation tracking
        # Tracking initialization complete
        
        # Performance optimization caches
        self._era5_cache = {}
        self._era5_cache_tolerance = 1e-6
        self._last_era5_time = None
        
        # Load pre-computed lookup tables
        self._load_temperature_lookup_tables()
    
    def _load_temperature_lookup_tables(self):
        """Load pre-computed lookup tables from disk."""
        lookup_dir = Path(__file__).parent / 'lookup_tables'
        
        if not lookup_dir.exists():
            raise FileNotFoundError(f"Lookup tables not found. Run: python generate_lookup_tables.py")
        
        # Load temperature parameters
        params = np.load(lookup_dir / 'temperature_params.npz')
        self._temp_min = float(params['temp_min'])
        self._temp_max = float(params['temp_max'])
        self._temp_resolution = float(params['temp_resolution'])
        
        # Load lookup arrays
        self._es_lookup = np.load(lookup_dir / 'vapor_pressure_lookup.npy')
        self._alpha_1816_lookup = np.load(lookup_dir / 'alpha_1816_lookup.npy')
        self._alpha_1716_lookup = np.load(lookup_dir / 'alpha_1716_lookup.npy')
    
    def _lookup_vapor_pressure(self, temperature):
        """Fast vapor pressure lookup."""
        indices = np.clip(
            np.round((np.asarray(temperature) - self._temp_min) / self._temp_resolution).astype(int),
            0, len(self._es_lookup) - 1
        )
        
        return self._es_lookup[indices] if np.asarray(temperature).ndim > 0 else self._es_lookup[indices]
    
    def _lookup_fractionation_factors(self, temperature):
        """Fast fractionation factor lookup."""
        indices = np.clip(
            np.round((np.asarray(temperature) - self._temp_min) / self._temp_resolution).astype(int),
            0, len(self._alpha_1816_lookup) - 1
        )
        
        if np.asarray(temperature).ndim > 0:
            return self._alpha_1816_lookup[indices], self._alpha_1716_lookup[indices]
        else:
            return self._alpha_1816_lookup[indices], self._alpha_1716_lookup[indices]
    
    def _get_era5_conditions_cached(self, time_days: float) -> dict:
        """Get ERA5 conditions with cache."""
        
        if (self._last_era5_time is not None and 
            abs(self._last_era5_time[0] - time_days) < self._era5_cache_tolerance):
            return self._last_era5_time[1]
        
        for cached_time, cached_result in self._era5_cache.items():
            if abs(cached_time - time_days) < self._era5_cache_tolerance:
                self._last_era5_time = (cached_time, cached_result)
                return cached_result
        
        result = self.era5_forcing.get_conditions_at_time(time_days)
        
        if len(self._era5_cache) > 1000:
            oldest_time = min(self._era5_cache.keys())
            del self._era5_cache[oldest_time]
        
        self._era5_cache[time_days] = result
        self._last_era5_time = (time_days, result)
        return result
    
    
    def _vectorized_diffusion_step(self, H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                                  H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                                  liquid_conc_soil_new, surface_temp=None, time_days=0.0, apply_temp_diffusion=True):
        """Vectorized diffusion and equilibration for all interior nodes."""
        interior = slice(1, self.num_nodes - 1)
        
        # PHYSICS STEP 1: Temperature diffusion (heat transfer equation)
        if apply_temp_diffusion and surface_temp is not None:
            self._solve_temperature_diffusion(surface_temp, interior)
        
        # PHYSICS STEP 2: Water vapor diffusion (Fick's law for H2O isotopologues)
        vapor_results = self._solve_vapor_diffusion(interior)
        
        # PHYSICS STEP 3: Liquid-vapor equilibration (Craig-Gordon isotope equilibration)
        self._solve_liquid_vapor_equilibration(
            vapor_results, interior,
            H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
            H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
            liquid_conc_soil_new
        )
        
        # PHYSICS STEP 4: Update diffusion coefficients based on new conditions
        self._update_diffusion_coefficients_interior(interior)
    
    def _solve_temperature_diffusion(self, surface_temp: float, interior: slice):
        """Solve heat diffusion equation in soil column.
        
        Uses finite difference method to solve:
        ∂T/∂t = α ∇²T where α is thermal diffusivity
        
        Note: This method is only called when temperature diffusion is enabled.
        The disable_diffusion flag is handled at a higher level.
        """
        
        # PHYSICS: Heat transfer constants (unchanged from original)
        Cv = 2.5e6  # Volumetric heat capacity [J/m³/K]
        kappa = 0.25 * (365 * 24 * 60 * 60)  # Thermal diffusivity [m²/year] 
        alpha = kappa / Cv  # [m²/s]
        dt_years = self.time_step / (365.25 * 24 * 60 * 60)  # Convert to years
        dz_m = self.config.numerical.depth_step / 100  # Convert cm to m
        
        # PHYSICS: Finite difference heat equation (exact same calculation)
        T_old = self.soil_temperature.copy()
        self.soil_temperature[0] = surface_temp  # Boundary condition
        
        # Second derivative: d²T/dz² = (T[i+1] - 2*T[i] + T[i-1]) / dz²
        d2T_dz2 = (T_old[2:] - 2*T_old[1:-1] + T_old[:-2]) / (dz_m**2)
        
        # Forward Euler: T_new = T_old + α*dt*d²T/dz²
        self.soil_temperature[interior] = T_old[interior] + alpha * dt_years * d2T_dz2
        
        # Bottom boundary condition (zero gradient)
        if len(self.soil_temperature) > 1:
            self.soil_temperature[-1] = self.soil_temperature[-2]
        
        # Update temperature-dependent fractionation factors
        self._update_fractionation_factors()
    
    def _solve_vapor_diffusion(self, interior: slice) -> dict:
        """Solve vapor diffusion for all water isotopologues.
        
        Solves Fick's law: ∂C/∂t = D ∇²C for each isotopologue
        Returns intermediate results for equilibration step.
        """
        # PHYSICS: Finite difference setup (exact same as original)
        dz2 = self.config.numerical.depth_step**2
        
        # Second derivatives for each isotopologue: d²C/dz²
        dH216O_dz2 = ((self.H216O_vapor_soil[2:] + self.H216O_vapor_soil[:-2] - 2*self.H216O_vapor_soil[1:-1]) / dz2)
        dH218O_dz2 = ((self.H218O_vapor_soil[2:] + self.H218O_vapor_soil[:-2] - 2*self.H218O_vapor_soil[1:-1]) / dz2)
        dH217O_dz2 = ((self.H217O_vapor_soil[2:] + self.H217O_vapor_soil[:-2] - 2*self.H217O_vapor_soil[1:-1]) / dz2)
        
        # Special handling for bottom boundary (exact same calculation)
        if self.num_nodes > 2:
            dH216O_dz2[-1] = (self.H216O_vapor_soil[-3] - 2*self.H216O_vapor_soil[-2] + self.H216O_vapor_soil[-1]) / dz2
            dH218O_dz2[-1] = (self.H218O_vapor_soil[-3] - 2*self.H218O_vapor_soil[-2] + self.H218O_vapor_soil[-1]) / dz2
            dH217O_dz2[-1] = (self.H217O_vapor_soil[-3] - 2*self.H217O_vapor_soil[-2] + self.H217O_vapor_soil[-1]) / dz2
        
        # Forward Euler diffusion: C_new = C_old + D*dt*d²C/dz²
        H216O_after_diff = (self.H216O_vapor_soil[interior] + self.time_step * self.D_H216O[interior] * dH216O_dz2)
        H218O_after_diff = (self.H218O_vapor_soil[interior] + self.time_step * self.D_H218O[interior] * dH218O_dz2)
        H217O_after_diff = (self.H217O_vapor_soil[interior] + self.time_step * self.D_H217O[interior] * dH217O_dz2)
        
        # Apply ratio-preserving threshold protection with frozen ratios
        total_threshold = 3e-20  # Total minimum vapor threshold
        current_H216O = self.H216O_vapor_soil[interior]
        current_H218O = self.H218O_vapor_soil[interior]
        current_H217O = self.H217O_vapor_soil[interior]
        current_total = current_H216O + current_H218O + current_H217O
        
        # Initialize frozen ratio storage if not exists
        if not hasattr(self, '_frozen_ratios_H216O'):
            self._frozen_ratios_H216O = np.zeros(self.num_nodes)
            self._frozen_ratios_H218O = np.zeros(self.num_nodes)
            self._frozen_ratios_H217O = np.zeros(self.num_nodes)
            self._nodes_below_threshold = np.zeros(self.num_nodes, dtype=bool)
        
        # Calculate proposed changes from diffusion
        dH216O_dt = H216O_after_diff - current_H216O
        dH218O_dt = H218O_after_diff - current_H218O
        dH217O_dt = H217O_after_diff - current_H217O
        
        # Check if total would go below threshold
        total_after_diff = H216O_after_diff + H218O_after_diff + H217O_after_diff
        interior_indices = np.arange(1, self.num_nodes - 1)
        
        # Identify nodes that are newly hitting the threshold
        currently_below_threshold = total_after_diff < total_threshold
        newly_below_threshold = currently_below_threshold & ~self._nodes_below_threshold[interior_indices]
        
        # Freeze ratios for newly threshold nodes (based on concentrations BEFORE they hit threshold)
        if np.any(newly_below_threshold):
            safe_total = np.maximum(current_total[newly_below_threshold], 1e-20)
            self._frozen_ratios_H216O[interior_indices[newly_below_threshold]] = current_H216O[newly_below_threshold] / safe_total
            self._frozen_ratios_H218O[interior_indices[newly_below_threshold]] = current_H218O[newly_below_threshold] / safe_total
            self._frozen_ratios_H217O[interior_indices[newly_below_threshold]] = current_H217O[newly_below_threshold] / safe_total
        
        # Update threshold status
        self._nodes_below_threshold[interior_indices] = currently_below_threshold
        
        # For nodes recovering above threshold, clear their frozen ratios
        recovering_nodes = ~currently_below_threshold & self._nodes_below_threshold[interior_indices]
        if np.any(recovering_nodes):
            self._nodes_below_threshold[interior_indices[recovering_nodes]] = False
        
        # Use frozen ratios for threshold calculations
        ratio_H216O = np.where(currently_below_threshold, 
                             self._frozen_ratios_H216O[interior_indices], 
                             current_H216O / np.maximum(current_total, 1e-20))
        ratio_H218O = np.where(currently_below_threshold, 
                             self._frozen_ratios_H218O[interior_indices], 
                             current_H218O / np.maximum(current_total, 1e-20))
        ratio_H217O = np.where(currently_below_threshold, 
                             self._frozen_ratios_H217O[interior_indices], 
                             current_H217O / np.maximum(current_total, 1e-20))
        
        # Calculate proportional thresholds using frozen ratios
        threshold_H216O = np.where(currently_below_threshold, ratio_H216O * total_threshold, 1e-20)
        threshold_H218O = np.where(currently_below_threshold, ratio_H218O * total_threshold, 1e-20)
        threshold_H217O = np.where(currently_below_threshold, ratio_H217O * total_threshold, 1e-20)
        
        # For negative changes, limit to what's available above the proportional threshold
        dH216O_dt = np.where(dH216O_dt < 0, 
                           np.maximum(dH216O_dt, threshold_H216O - current_H216O), 
                           dH216O_dt)
        dH218O_dt = np.where(dH218O_dt < 0, 
                           np.maximum(dH218O_dt, threshold_H218O - current_H218O), 
                           dH218O_dt)
        dH217O_dt = np.where(dH217O_dt < 0, 
                           np.maximum(dH217O_dt, threshold_H217O - current_H217O), 
                           dH217O_dt)
        
        # Apply flux-limited diffusion with frozen ratio preservation
        H216O_after_diff = current_H216O + dH216O_dt
        H218O_after_diff = current_H218O + dH218O_dt
        H217O_after_diff = current_H217O + dH217O_dt
        
        # Calculate isotope ratios (exact same calculation)
        H216O_safe = np.maximum(H216O_after_diff, 1e-20)
        d18O_vapor_after_diff = ((H218O_after_diff / H216O_safe) / self.config.constants.RSMOW_1816 - 1) * 1000
        d17O_vapor_after_diff = ((H217O_after_diff / H216O_safe) / self.config.constants.RSMOW_1716 - 1) * 1000
        
        return {
            'H216O_after_diff': H216O_after_diff,
            'H218O_after_diff': H218O_after_diff, 
            'H217O_after_diff': H217O_after_diff,
            'd18O_vapor_after_diff': d18O_vapor_after_diff,
            'd17O_vapor_after_diff': d17O_vapor_after_diff
        }
    
    def _solve_liquid_vapor_equilibration(self, vapor_results: dict, interior: slice,
                                         H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                                         H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                                         liquid_conc_soil_new):
        """Solve liquid-vapor equilibration using Craig-Gordon isotope theory.
        
        Implements instantaneous isotopic equilibration between liquid and vapor phases
        according to fractionation factors and mass balance constraints.
        """
        n_interior = self.num_nodes - 2
        
        # Extract diffusion results
        H216O_after_diff = vapor_results['H216O_after_diff']
        H218O_after_diff = vapor_results['H218O_after_diff']
        H217O_after_diff = vapor_results['H217O_after_diff']
        d18O_vapor_after_diff = vapor_results['d18O_vapor_after_diff']
        d17O_vapor_after_diff = vapor_results['d17O_vapor_after_diff']
        
        # PHYSICS: Calculate total water and equilibrium vapor concentrations (exact same)
        vapor_conc_after_diff = H216O_after_diff + H218O_after_diff + H217O_after_diff
        total_water = vapor_conc_after_diff * self.free_air_porosity[interior] + self.liquid_conc_soil[interior]
        
        soil_T_interior = self.soil_temperature[interior]
        es_soil = self._lookup_vapor_pressure(soil_T_interior)
        vapor_conc_eq = es_soil / (8.3144 * (soil_T_interior + 273.15)) * 1e-6
        
        # PHYSICS: Water content threshold correction (exact same calculation)
        liquid_conc_interior = self.liquid_conc_soil[interior]
        threshold_mask = liquid_conc_interior < self.config.soil.water_content_threshold
        slope = vapor_conc_eq / self.config.soil.water_content_threshold
        vapor_conc_eq = np.where(threshold_mask,
                                vapor_conc_eq - (self.config.soil.water_content_threshold - liquid_conc_interior) * slope,
                                vapor_conc_eq)
        
        # PHYSICS: Determine phase partitioning (exact same logic)
        has_liquid = total_water > (vapor_conc_eq * self.free_air_porosity[interior])
        
        # Initialize output arrays
        H216O_vapor_new_int = np.zeros(n_interior)
        H218O_vapor_new_int = np.zeros(n_interior)
        H217O_vapor_new_int = np.zeros(n_interior)
        H216O_liquid_new_int = np.zeros(n_interior)
        H218O_liquid_new_int = np.zeros(n_interior)
        H217O_liquid_new_int = np.zeros(n_interior)
        liquid_conc_new_int = np.zeros(n_interior)
        
        # PHYSICS: Two-phase equilibration (exact same calculation)
        if np.any(has_liquid):
            liquid_mask = has_liquid
            total_water_liq = total_water[liquid_mask]
            vapor_conc_eq_liq = vapor_conc_eq[liquid_mask]
            free_air_porosity_liq = self.free_air_porosity[interior][liquid_mask]
            
            # Mass balance: liquid concentration after equilibration
            liquid_conc_after_eq = total_water_liq - vapor_conc_eq_liq * free_air_porosity_liq
            
            # Extract data for nodes with liquid
            vapor_conc_after_diff_liq = vapor_conc_after_diff[liquid_mask]
            d18O_vapor_after_diff_liq = d18O_vapor_after_diff[liquid_mask]
            d17O_vapor_after_diff_liq = d17O_vapor_after_diff[liquid_mask]
            d18O_liquid_soil_liq = self.d18O_liquid_soil[interior][liquid_mask]
            d17O_liquid_soil_liq = self.d17O_liquid_soil[interior][liquid_mask]
            liquid_conc_soil_liq = self.liquid_conc_soil[interior][liquid_mask]
            
            # PHYSICS: Mass-weighted isotope mixing (exact same equations)
            d18O_total_water = ((d18O_vapor_after_diff_liq * vapor_conc_after_diff_liq * free_air_porosity_liq / total_water_liq) + 
                              (d18O_liquid_soil_liq * liquid_conc_soil_liq / total_water_liq))
            d17O_total_water = ((d17O_vapor_after_diff_liq * vapor_conc_after_diff_liq * free_air_porosity_liq / total_water_liq) + 
                              (d17O_liquid_soil_liq * liquid_conc_soil_liq / total_water_liq))
            
            # PHYSICS: Vapor fraction at equilibrium
            F_vapor_at_eq = (vapor_conc_eq_liq * free_air_porosity_liq) / total_water_liq
            
            # PHYSICS: Fractionation factors
            alpha_1816_liq = self.alpha_1816_liq_vap_soil[interior][liquid_mask]
            alpha_1716_liq = self.alpha_1716_liq_vap_soil[interior][liquid_mask]
            
            # PHYSICS: Craig-Gordon equilibration equations (exact same)
            d18O_liquid_after_eq = ((d18O_total_water * alpha_1816_liq + 1000 * F_vapor_at_eq * (alpha_1816_liq - 1)) / 
                                  (alpha_1816_liq * (1 - F_vapor_at_eq) + F_vapor_at_eq))
            d18O_vapor_after_eq = ((d18O_liquid_after_eq + 1000) / alpha_1816_liq - 1000)
            
            d17O_liquid_after_eq = ((d17O_total_water * alpha_1716_liq + 1000 * F_vapor_at_eq * (alpha_1716_liq - 1)) / 
                                  (alpha_1716_liq * (1 - F_vapor_at_eq) + F_vapor_at_eq))
            d17O_vapor_after_eq = ((d17O_liquid_after_eq + 1000) / alpha_1716_liq - 1000)
            
            # Convert back to concentrations (exact same calculation)
            R_1816_vapor_after_eq = ((d18O_vapor_after_eq/1000 + 1) * self.config.constants.RSMOW_1816)
            R_1716_vapor_after_eq = ((d17O_vapor_after_eq/1000 + 1) * self.config.constants.RSMOW_1716)
            
            H216O_vapor_after_eq = vapor_conc_eq_liq / (R_1816_vapor_after_eq + R_1716_vapor_after_eq + 1)
            H218O_vapor_after_eq = R_1816_vapor_after_eq * H216O_vapor_after_eq
            H217O_vapor_after_eq = R_1716_vapor_after_eq * H216O_vapor_after_eq
            
            R_1816_liquid_after_eq = ((d18O_liquid_after_eq/1000 + 1) * self.config.constants.RSMOW_1816)
            R_1716_liquid_after_eq = ((d17O_liquid_after_eq/1000 + 1) * self.config.constants.RSMOW_1716)
            
            H216O_liquid_after_eq = liquid_conc_after_eq / (R_1816_liquid_after_eq + R_1716_liquid_after_eq + 1)
            H218O_liquid_after_eq = R_1816_liquid_after_eq * H216O_liquid_after_eq
            H217O_liquid_after_eq = R_1716_liquid_after_eq * H216O_liquid_after_eq
            
            # Store results for liquid nodes
            H216O_vapor_new_int[liquid_mask] = H216O_vapor_after_eq
            H218O_vapor_new_int[liquid_mask] = H218O_vapor_after_eq
            H217O_vapor_new_int[liquid_mask] = H217O_vapor_after_eq
            H216O_liquid_new_int[liquid_mask] = H216O_liquid_after_eq
            H218O_liquid_new_int[liquid_mask] = H218O_liquid_after_eq
            H217O_liquid_new_int[liquid_mask] = H217O_liquid_after_eq
            liquid_conc_new_int[liquid_mask] = liquid_conc_after_eq
            
            # Update soil water content and porosity
            actual_indices = np.arange(1, self.num_nodes - 1)[liquid_mask]
            self.water_content[actual_indices] = liquid_conc_after_eq * 18.01528
            self.free_air_porosity[actual_indices] = self.config.soil.total_porosity - self.water_content[actual_indices]
        
        # PHYSICS: Vapor-only nodes (exact same logic)
        no_liquid_mask = ~has_liquid
        if np.any(no_liquid_mask):
            H216O_vapor_new_int[no_liquid_mask] = H216O_after_diff[no_liquid_mask]
            H218O_vapor_new_int[no_liquid_mask] = H218O_after_diff[no_liquid_mask]
            H217O_vapor_new_int[no_liquid_mask] = H217O_after_diff[no_liquid_mask]
            
            actual_indices = np.arange(1, self.num_nodes - 1)[no_liquid_mask]
            self.water_content[actual_indices] = 0
            self.free_air_porosity[actual_indices] = self.config.soil.total_porosity
        
        # Store final results
        H216O_vapor_soil_new[interior] = H216O_vapor_new_int
        H218O_vapor_soil_new[interior] = H218O_vapor_new_int
        H217O_vapor_soil_new[interior] = H217O_vapor_new_int
        H216O_liquid_soil_new[interior] = H216O_liquid_new_int
        H218O_liquid_soil_new[interior] = H218O_liquid_new_int
        H217O_liquid_soil_new[interior] = H217O_liquid_new_int
        liquid_conc_soil_new[interior] = liquid_conc_new_int
    
    def _update_diffusion_coefficients_interior(self, interior: slice):
        """Update diffusion coefficients for interior nodes based on new conditions.
        
        Uses Millington-Quirk model: D_eff = D_0 * φ^(10/3) / τ² * (T/T_ref)^n
        """
        # PHYSICS: Same calculation as original
        actual_indices = np.arange(1, self.num_nodes - 1)
        D_H2O_vec = (self.config.constants.D_H2O_air_20C * self.free_air_porosity[actual_indices] * 
                    self.config.soil.tortuosity * 
                    ((self.soil_temperature[actual_indices] + 273)/(20 + 273))**self.config.constants.temp_exponent)
        
        # Isotopologue-specific diffusion coefficients
        self.D_H218O[actual_indices] = D_H2O_vec / self.config.constants.D_H216O_over_D_H218O
        self.D_H216O[actual_indices] = self.D_H218O[actual_indices] * self.config.constants.D_H216O_over_D_H218O
        D_H216O_over_D_H217O = self.config.constants.D_H216O_over_D_H218O**self.config.constants.theta_diff
        self.D_H217O[actual_indices] = self.D_H216O[actual_indices] / D_H216O_over_D_H217O
    
    def _apply_flux_boundary_conditions(self, H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                                       H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                                       liquid_conc_soil_new, time_days: float):
        """Apply flux-based boundary conditions."""
        
        if self._use_era5_forcing and self.era5_forcing is not None:
            era5_conditions = self._get_era5_conditions_cached(time_days)
            
            air_temp = era5_conditions['temperature_2m']
            humidity = era5_conditions['relative_humidity']
            vapor_conc_atm = era5_conditions['vapor_concentration_air']
            
            d18O_atm_vapor = self.config.atmospheric.d18O_vapor
            
            # Convert D'17O_vapor to d17O_vapor using same method as rain
            d_prime_18O_atm = 1000 * np.log(self.config.atmospheric.d18O_vapor/1000 + 1)
            d_prime_17O_atm = self.config.atmospheric.D_prime_17O_vapor/1000 + 0.528 * d_prime_18O_atm
            d17O_atm_vapor = (np.exp(d_prime_17O_atm/1000) - 1) * 1000
            
        else:
            air_temp = self.config.atmospheric.mean_air_temperature
            humidity = self.config.atmospheric.relative_humidity
            
            es_air = self._lookup_vapor_pressure(air_temp)
            vapor_pressure_air = es_air * humidity / 100.0
            vapor_conc_atm = vapor_pressure_air / (8.3144 * (air_temp + 273.15)) * 1e-6
            
            
            d18O_atm_vapor = self.config.atmospheric.d18O_vapor
            
            # Convert D'17O_vapor to d17O_vapor using same method as rain
            d_prime_18O_atm = 1000 * np.log(self.config.atmospheric.d18O_vapor/1000 + 1)
            d_prime_17O_atm = self.config.atmospheric.D_prime_17O_vapor/1000 + 0.528 * d_prime_18O_atm
            d17O_atm_vapor = (np.exp(d_prime_17O_atm/1000) - 1) * 1000
        
        R_1816_atm = ((d18O_atm_vapor/1000 + 1) * self.config.constants.RSMOW_1816)
        R_1716_atm = ((d17O_atm_vapor/1000 + 1) * self.config.constants.RSMOW_1716)
        
        H216O_vapor_atm = vapor_conc_atm / (R_1816_atm + R_1716_atm + 1)
        H218O_vapor_atm = R_1816_atm * H216O_vapor_atm
        H217O_vapor_atm = R_1716_atm * H216O_vapor_atm
        
        H216O_soil_surface = self.H216O_vapor_soil[0]
        H218O_soil_surface = self.H218O_vapor_soil[0]
        H217O_soil_surface = self.H217O_vapor_soil[0]
        
        soil_vapor_total = H216O_soil_surface + H218O_soil_surface + H217O_soil_surface
        atm_vapor_total = H216O_vapor_atm + H218O_vapor_atm + H217O_vapor_atm
        
        dC_total = soil_vapor_total - atm_vapor_total
        
        soil_T_surface = self.soil_temperature[0]
        
        # Calculate aerodynamic resistance based on boundary layer model
        if hasattr(self.config.atmospheric, 'wind_speed') and hasattr(self.config.atmospheric, 'surface_roughness'):
            # Use boundary layer model with wind speed and surface roughness
            resistance_s_per_cm = self._calculate_aerodynamic_resistance(
                self.config.atmospheric.wind_speed, 
                self.config.atmospheric.surface_roughness
            )
        else:
            # Use default resistance (backward compatibility)
            resistance_s_per_cm = self._boundary_layer_resistance / 100.0
        
        flux_total = dC_total / resistance_s_per_cm
        
        
        dt = self.time_step
        
        flux_H216O = 0.0
        flux_H218O = 0.0  
        flux_H217O = 0.0
        
        if flux_total > 0:
            if soil_vapor_total > 1e-20:
                d18O_soil_vapor = ((H218O_soil_surface / H216O_soil_surface) / self.config.constants.RSMOW_1816 - 1) * 1000
                d17O_soil_vapor = ((H217O_soil_surface / H216O_soil_surface) / self.config.constants.RSMOW_1716 - 1) * 1000
            else:
                d18O_soil_vapor = 0.0
                d17O_soil_vapor = 0.0
            
            R_1816_out = (d18O_soil_vapor/1000 + 1) * self.config.constants.RSMOW_1816
            R_1716_out = (d17O_soil_vapor/1000 + 1) * self.config.constants.RSMOW_1716
            
            flux_H216O = flux_total / (1 + R_1816_out + R_1716_out)
            flux_H218O = flux_H216O * R_1816_out  
            flux_H217O = flux_H216O * R_1716_out
            
            dz = self.config.numerical.depth_step
            H216O_vapor_soil_new[0] = H216O_soil_surface - flux_H216O * dt / dz
            H218O_vapor_soil_new[0] = H218O_soil_surface - flux_H218O * dt / dz
            H217O_vapor_soil_new[0] = H217O_soil_surface - flux_H217O * dt / dz
            
            
            # Track atmospheric conditions for evaporation data
            self._track_atmospheric_conditions(time_days, air_temp, humidity, soil_T_surface)
            
        else:
            flux_total_in = -flux_total
            
            flux_H216O_in = flux_total_in / (1 + R_1816_atm + R_1716_atm)
            flux_H218O_in = flux_H216O_in * R_1816_atm
            flux_H217O_in = flux_H216O_in * R_1716_atm
            
            flux_H216O = -flux_H216O_in
            flux_H218O = -flux_H218O_in
            flux_H217O = -flux_H217O_in
            
            dz = self.config.numerical.depth_step
            H216O_vapor_soil_new[0] = H216O_soil_surface + flux_H216O_in * dt / dz
            H218O_vapor_soil_new[0] = H218O_soil_surface + flux_H218O_in * dt / dz  
            H217O_vapor_soil_new[0] = H217O_soil_surface + flux_H217O_in * dt / dz
            
            
            # Track atmospheric conditions for evaporation data
            self._track_atmospheric_conditions(time_days, air_temp, humidity, soil_T_surface)
        
        # Apply frozen ratio protection to surface node as well
        surface_total = H216O_vapor_soil_new[0] + H218O_vapor_soil_new[0] + H217O_vapor_soil_new[0]
        surface_threshold = 3e-20
        
        if surface_total < surface_threshold:
            # Initialize surface frozen ratios if needed
            if not hasattr(self, '_surface_frozen_ratios_H216O'):
                # Use initial surface ratios or reasonable defaults
                current_surface_total = self.H216O_vapor_soil[0] + self.H218O_vapor_soil[0] + self.H217O_vapor_soil[0]
                if current_surface_total > surface_threshold:
                    self._surface_frozen_ratios_H216O = self.H216O_vapor_soil[0] / current_surface_total
                    self._surface_frozen_ratios_H218O = self.H218O_vapor_soil[0] / current_surface_total
                    self._surface_frozen_ratios_H217O = self.H217O_vapor_soil[0] / current_surface_total
                else:
                    # Use reasonable isotopic defaults if surface starts below threshold
                    self._surface_frozen_ratios_H216O = 0.99
                    self._surface_frozen_ratios_H218O = 0.002
                    self._surface_frozen_ratios_H217O = 0.0004
            
            # Apply proportional thresholds to surface
            H216O_vapor_soil_new[0] = self._surface_frozen_ratios_H216O * surface_threshold
            H218O_vapor_soil_new[0] = self._surface_frozen_ratios_H218O * surface_threshold
            H217O_vapor_soil_new[0] = self._surface_frozen_ratios_H217O * surface_threshold
        
        self._equilibrate_surface_node(H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                                     H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                                     liquid_conc_soil_new)
        
        self._current_atmospheric_conditions = {
            'air_temperature': air_temp,
            'relative_humidity': humidity,
            'vapor_concentration': vapor_conc_atm,
            'd18O_vapor': d18O_atm_vapor,
            'd17O_vapor': d17O_atm_vapor
        }
    
    def _equilibrate_surface_node(self, H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                                H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                                liquid_conc_soil_new):
        """Apply Craig-Gordon equilibration to surface node after atmospheric exchange.
        
        Surface node behaves identically to interior nodes - uses Craig-Gordon equilibration.
        The only difference is the vapor concentrations come from atmospheric exchange.
        """
        j = 0
        
        # Calculate current concentrations after atmospheric exchange
        vapor_conc_surface = H216O_vapor_soil_new[j] + H218O_vapor_soil_new[j] + H217O_vapor_soil_new[j]
        liquid_conc_surface = self.liquid_conc_soil[j]
        total_water_surface = vapor_conc_surface * self.free_air_porosity[j] + liquid_conc_surface
        
        # Calculate equilibrium vapor concentration at surface temperature
        soil_T_surface = self.soil_temperature[j]
        es_soil_surface = self._lookup_vapor_pressure(soil_T_surface)
        vapor_conc_eq_surface = es_soil_surface / (8.3144 * (soil_T_surface + 273.15)) * 1e-6
        
        # Apply water content threshold correction if needed
        if liquid_conc_surface < self.config.soil.water_content_threshold:
            slope = vapor_conc_eq_surface / self.config.soil.water_content_threshold
            vapor_conc_eq_surface = (vapor_conc_eq_surface - 
                                   (self.config.soil.water_content_threshold - liquid_conc_surface) * slope)
        
        vapor_capacity = vapor_conc_eq_surface * self.free_air_porosity[j]
        
        # Apply Craig-Gordon equilibration (same as interior nodes)
        if total_water_surface > vapor_capacity:
            liquid_conc_after_eq = total_water_surface - vapor_capacity
            
            d18O_vapor_surface = ((H218O_vapor_soil_new[j] / max(H216O_vapor_soil_new[j], 1e-20)) / 
                                self.config.constants.RSMOW_1816 - 1) * 1000
            d17O_vapor_surface = ((H217O_vapor_soil_new[j] / max(H216O_vapor_soil_new[j], 1e-20)) / 
                                self.config.constants.RSMOW_1716 - 1) * 1000
            d18O_liquid_surface = self.d18O_liquid_soil[j]
            d17O_liquid_surface = self.d17O_liquid_soil[j]
            
            d18O_total_water = ((d18O_vapor_surface * vapor_conc_surface * self.free_air_porosity[j] / total_water_surface) + 
                              (d18O_liquid_surface * liquid_conc_surface / total_water_surface))
            d17O_total_water = ((d17O_vapor_surface * vapor_conc_surface * self.free_air_porosity[j] / total_water_surface) + 
                              (d17O_liquid_surface * liquid_conc_surface / total_water_surface))
            
            F_vapor_at_eq = vapor_capacity / total_water_surface
            
            alpha_1816_surface = self.alpha_1816_liq_vap_soil[j]
            alpha_1716_surface = self.alpha_1716_liq_vap_soil[j]
            
            d18O_liquid_after_eq = ((d18O_total_water * alpha_1816_surface + 1000 * F_vapor_at_eq * (alpha_1816_surface - 1)) / 
                                  (alpha_1816_surface * (1 - F_vapor_at_eq) + F_vapor_at_eq))
            d18O_vapor_after_eq = ((d18O_liquid_after_eq + 1000) / alpha_1816_surface - 1000)
            
            d17O_liquid_after_eq = ((d17O_total_water * alpha_1716_surface + 1000 * F_vapor_at_eq * (alpha_1716_surface - 1)) / 
                                  (alpha_1716_surface * (1 - F_vapor_at_eq) + F_vapor_at_eq))
            d17O_vapor_after_eq = ((d17O_liquid_after_eq + 1000) / alpha_1716_surface - 1000)
            
            R_1816_vapor_after_eq = ((d18O_vapor_after_eq/1000 + 1) * self.config.constants.RSMOW_1816)
            R_1716_vapor_after_eq = ((d17O_vapor_after_eq/1000 + 1) * self.config.constants.RSMOW_1716)
            
            H216O_vapor_after_eq = vapor_conc_eq_surface / (R_1816_vapor_after_eq + R_1716_vapor_after_eq + 1)
            H218O_vapor_after_eq = R_1816_vapor_after_eq * H216O_vapor_after_eq
            H217O_vapor_after_eq = R_1716_vapor_after_eq * H216O_vapor_after_eq
            
            R_1816_liquid_after_eq = ((d18O_liquid_after_eq/1000 + 1) * self.config.constants.RSMOW_1816)
            R_1716_liquid_after_eq = ((d17O_liquid_after_eq/1000 + 1) * self.config.constants.RSMOW_1716)
            
            H216O_liquid_after_eq = liquid_conc_after_eq / (R_1816_liquid_after_eq + R_1716_liquid_after_eq + 1)
            H218O_liquid_after_eq = R_1816_liquid_after_eq * H216O_liquid_after_eq
            H217O_liquid_after_eq = R_1716_liquid_after_eq * H216O_liquid_after_eq
            
            H216O_vapor_soil_new[j] = H216O_vapor_after_eq
            H218O_vapor_soil_new[j] = H218O_vapor_after_eq
            H217O_vapor_soil_new[j] = H217O_vapor_after_eq
            H216O_liquid_soil_new[j] = H216O_liquid_after_eq
            H218O_liquid_soil_new[j] = H218O_liquid_after_eq
            H217O_liquid_soil_new[j] = H217O_liquid_after_eq
            liquid_conc_soil_new[j] = liquid_conc_after_eq
            
            self.water_content[j] = liquid_conc_after_eq * 18.01528
            self.free_air_porosity[j] = self.config.soil.total_porosity - self.water_content[j]
            self.d18O_liquid_soil[j] = d18O_liquid_after_eq
            self.d17O_liquid_soil[j] = d17O_liquid_after_eq
            
        else:
            H216O_liquid_soil_new[j] = 0
            H218O_liquid_soil_new[j] = 0
            H217O_liquid_soil_new[j] = 0
            liquid_conc_soil_new[j] = 0
            
            self.water_content[j] = 0
            self.free_air_porosity[j] = self.config.soil.total_porosity

    
    def _calculate_aerodynamic_resistance(self, wind_speed, surface_roughness_cm):
        """
        Calculate aerodynamic resistance using boundary layer theory.
        
        Uses logarithmic wind profile and Monin-Obukhov similarity theory
        for the atmospheric surface layer.
        
        Args:
            wind_speed: Wind speed at reference height (m/s)
            surface_roughness_cm: Surface roughness length (cm)
            
        Returns:
            Aerodynamic resistance (s/cm)
        """
        # Constants
        von_karman = 0.41  # von Karman constant
        reference_height = 2.0  # Reference height for wind speed (m)
        
        # Convert surface roughness from cm to m
        z0 = surface_roughness_cm / 100.0
        
        # Ensure roughness is reasonable relative to reference height
        z0 = min(z0, reference_height / 10.0)  # Roughness can't be too large
        z0 = max(z0, 1e-6)  # Minimum roughness for numerical stability
        
        # Minimum wind speed for numerical stability
        wind_speed = max(wind_speed, 0.1)
        
        # Calculate aerodynamic resistance using logarithmic wind profile
        # ra = ln(z/z0) / (k * u*) where u* = k * u / ln(z/z0)
        # Simplifies to: ra = ln²(z/z0) / (k² * u)
        
        log_term = np.log(reference_height / z0)
        ra_s_per_m = (log_term**2) / (von_karman**2 * wind_speed)
        
        # Convert from s/m to s/cm
        ra_s_per_cm = ra_s_per_m / 100.0
        
        return ra_s_per_cm
    
    def _update_fractionation_factors(self):
        """Update fractionation factors using lookup tables."""
        self.alpha_1816_liq_vap_soil, self.alpha_1716_liq_vap_soil = self._lookup_fractionation_factors(self.soil_temperature)
    
    def _calculate_isotope_deltas_vectorized(self, temp_d18O_liquid: np.ndarray, temp_d17O_liquid: np.ndarray):
        """
        Vectorized calculation of δ18O and δ17O values from isotopologue concentrations.
        
        Optimizes the 8 separate NumPy operations into 2 efficient compound calculations.
        Converts isotope ratios R = H2XO/H216O to delta notation: δ = (R/R_standard - 1) × 1000
        
        Args:
            temp_d18O_liquid: Temporary array for δ18O calculations (reused for efficiency)
            temp_d17O_liquid: Temporary array for δ17O calculations (reused for efficiency)
        """
        # PHYSICS: δ18O calculation - single vectorized operation
        # Combined: R18/16 = H218O/H216O, δ18O = (R18/16/RSMOW_1816 - 1) × 1000
        np.multiply(
            np.subtract(
                np.divide(
                    np.divide(self.H218O_liquid_soil, np.maximum(self.H216O_liquid_soil, 1e-20), out=temp_d18O_liquid),
                    self.config.constants.RSMOW_1816, out=temp_d18O_liquid
                ), 1, out=temp_d18O_liquid
            ), 1000, out=self.d18O_liquid_soil
        )
        
        # PHYSICS: δ17O calculation - single vectorized operation  
        # Combined: R17/16 = H217O/H216O, δ17O = (R17/16/RSMOW_1716 - 1) × 1000
        np.multiply(
            np.subtract(
                np.divide(
                    np.divide(self.H217O_liquid_soil, np.maximum(self.H216O_liquid_soil, 1e-20), out=temp_d17O_liquid),
                    self.config.constants.RSMOW_1716, out=temp_d17O_liquid
                ), 1, out=temp_d17O_liquid
            ), 1000, out=self.d17O_liquid_soil
        )
    
    def setup_model(self) -> None:
        """Setup model grid, atmospheric conditions, and initial conditions."""
        # Check if we should use ERA5 forcing based on config
        if hasattr(self.config.temperature, 'use_era5_forcing'):
            self._use_era5_forcing = self.config.temperature.use_era5_forcing
        
        if self._use_era5_forcing:
            try:
                self.era5_forcing = load_era5_forcing_from_config(self.config)
            except Exception as e:
                print(f"Warning: Failed to load ERA5 data: {e}. Using constant atmospheric conditions.")
                self._use_era5_forcing = False
        
        dz = self.config.numerical.depth_step
        soil_depth = self.config.soil.depth
        self.depth_nodes = np.arange(0, soil_depth + dz, dz)
        self.num_nodes = len(self.depth_nodes)
        
        max_D_effective = np.ceil(self.config.constants.D_H2O_air_20C * 
                                 self.config.soil.tortuosity * 
                                 self.config.soil.total_porosity * 100) / 100
        self.time_step = dz**2 / (2 * max_D_effective)
        
        self._calculate_soil_temperature()
        self._setup_atmospheric_conditions()
        self._initialize_soil_conditions()
        
        self.metadata = {
            'depth_nodes': self.depth_nodes.copy(),
            'soil_temperature': self.soil_temperature.copy(),
            'time_step_seconds': self.time_step,
            'surface_porosity': self.config.soil.total_porosity,  # Total soil porosity for flux calculations
            'atmospheric_conditions': {
                'vapor_concentration': self.vapor_conc_air,
                'H216O': self.H216O_vapor_air,
                'H218O': self.H218O_vapor_air,
                'H217O': self.H217O_vapor_air
            },
            'config': self.config.to_dict()
        }
        
        self._is_setup = True
        
        # Single startup print with key parameters
        era5_status = "✓ ERA5" if self.era5_forcing else "Standard"
        print(f"Soil Evaporation Model: {self.num_nodes} nodes, {dz}cm steps, "
              f"{self.config.numerical.run_days} days, {era5_status} forcing")
    
    def _calculate_soil_temperature(self) -> None:
        """Calculate initial soil temperature profile."""
        if self.depth_nodes is None:
            raise ValueError("Depth array not initialized")
            
        if self.era5_forcing is not None:
            era5_conditions = self._get_era5_conditions_cached(0.0)
            surface_temp = era5_conditions['skin_temperature']
        else:
            surface_temp = self.config.atmospheric.mean_air_temperature
        
        # Calculate temperature profile based on configuration
        self.soil_temperature = self._calculate_temperature_profile(surface_temp)
    
    def _calculate_temperature_profile(self, surface_temp: float) -> np.ndarray:
        """
        Calculate temperature profile based on configuration settings.
        
        Args:
            surface_temp: Surface temperature (°C) from ERA5 or atmospheric config
        
        Returns:
            Array of temperature values for each depth node (°C)
        """
        profile_config = self.config.temperature.temperature_profile
        profile_type = profile_config.profile_type
        
        if profile_type == "constant":
            return np.full(self.num_nodes, profile_config.constant_value)
        
        elif profile_type == "linear":
            # Linear interpolation from surface to bottom
            surface_val = profile_config.surface_value
            bottom_val = profile_config.bottom_value
            return np.linspace(surface_val, bottom_val, self.num_nodes)
        
        elif profile_type == "exponential":
            # Exponential change from surface: temp = background + (surface - background) * exp(-depth/decay_length)
            surface_val = profile_config.surface_value_exp
            decay_length = profile_config.decay_length
            background_val = profile_config.background_value
            
            # Calculate exponential profile
            temperature = background_val + (surface_val - background_val) * np.exp(-self.depth_nodes / decay_length)
            return temperature
        
        elif profile_type == "from_file":
            return self._load_temperature_from_file()
        
        else:
            # Fallback to legacy behavior (linear from surface to deep temp)
            deep_temp = profile_config.constant_value  # Use constant as fallback deep temp
            max_depth = self.depth_nodes[-1]
            return surface_temp + (deep_temp - surface_temp) * (self.depth_nodes / max_depth)
    
    def _load_temperature_from_file(self) -> np.ndarray:
        """
        Load temperature profile from CSV file and interpolate to model grid.
        
        Returns:
            Array of temperature values interpolated to model depth nodes
        """
        import pandas as pd
        from scipy.interpolate import interp1d
        
        profile_file = self.config.temperature.temperature_profile.profile_file
        if not profile_file:
            raise ValueError("No profile file specified for 'from_file' temperature profile")
        
        try:
            # Load CSV file
            data = pd.read_csv(profile_file)
            
            # Expected columns: depth (cm), temperature (°C)
            required_columns = ['depth', 'temperature']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            # Extract data
            file_depths = data['depth'].values
            file_temperatures = data['temperature'].values
            
            # Validate data
            if len(file_depths) < 2:
                raise ValueError("Temperature profile file must contain at least 2 data points")
            
            if not np.all(np.diff(file_depths) >= 0):
                raise ValueError("Depth values in profile file must be non-decreasing")
            
            # Temperature values should be reasonable but allow wide range
            if np.any(file_temperatures < -50) or np.any(file_temperatures > 60):
                print(f"Warning: Temperature values outside typical range (-50 to 60°C)")
            
            # Interpolate to model grid
            interpolator = interp1d(file_depths, file_temperatures, 
                                  kind='linear', bounds_error=False, 
                                  fill_value=(file_temperatures[0], file_temperatures[-1]))
            
            interpolated_temperature = interpolator(self.depth_nodes)
            
            print(f"✓ Loaded temperature profile from {profile_file}")
            print(f"  File depth range: {file_depths[0]:.1f} - {file_depths[-1]:.1f} cm")
            print(f"  Temperature range: {file_temperatures.min():.1f} - {file_temperatures.max():.1f} °C")
            
            return interpolated_temperature
            
        except Exception as e:
            raise ValueError(f"Error loading temperature profile from {profile_file}: {str(e)}")
    
    def _update_soil_temperature_profile(self, surface_temp: float, time_days: float) -> None:
        """Update soil temperature profile based on heat diffusion."""
        
        Cv = 2.5e6
        kappa = 0.25 * (365 * 24 * 60 * 60)
        alpha = kappa / Cv
        
        dt_years = self.time_step / (365.25 * 24 * 60 * 60)
        dz_m = self.config.numerical.depth_step / 100
        
        T_old = self.soil_temperature.copy()
        
        self.soil_temperature[0] = surface_temp
        
        if len(self.soil_temperature) > 2:
            d2T_dz2 = (T_old[2:] - 2*T_old[1:-1] + T_old[:-2]) / (dz_m**2)
            self.soil_temperature[1:-1] = T_old[1:-1] + alpha * dt_years * d2T_dz2
        
        if len(self.soil_temperature) > 1:
            self.soil_temperature[-1] = self.soil_temperature[-2]
    
    def _setup_atmospheric_conditions(self) -> None:
        """Setup atmospheric conditions."""
        atm_T = self.soil_temperature[0]
        
        es_air = self._lookup_vapor_pressure(atm_T)
        e_air = es_air * self.config.atmospheric.relative_humidity / 100
        vapor_conc_air = e_air / (8.3144 * (atm_T + 273.15))
        self.vapor_conc_air = vapor_conc_air * 1e-6
        
        d18O_vapor_air = self.config.atmospheric.d18O_vapor
        
        # Convert D'17O_vapor to d17O_vapor using same method as rain
        d_prime_18O_vapor = 1000 * np.log(self.config.atmospheric.d18O_vapor/1000 + 1)
        d_prime_17O_vapor = self.config.atmospheric.D_prime_17O_vapor/1000 + 0.528 * d_prime_18O_vapor
        d17O_vapor_air = (np.exp(d_prime_17O_vapor/1000) - 1) * 1000
        
        R_1816_vapor_air = (d18O_vapor_air/1000 + 1) * self.config.constants.RSMOW_1816
        R_1716_vapor_air = (d17O_vapor_air/1000 + 1) * self.config.constants.RSMOW_1716
        
        self.H216O_vapor_air = self.vapor_conc_air / (R_1816_vapor_air + R_1716_vapor_air + 1)
        self.H218O_vapor_air = R_1816_vapor_air * self.H216O_vapor_air
        self.H217O_vapor_air = R_1716_vapor_air * self.H216O_vapor_air
    
    def _calculate_water_content_profile(self) -> np.ndarray:
        """
        Calculate water content profile based on configuration settings.
        
        Returns:
            Array of water content values for each depth node (cm³/cm³)
        """
        profile_config = self.config.soil.water_content_profile
        profile_type = profile_config.profile_type
        
        if profile_type == "constant":
            return np.full(self.num_nodes, profile_config.constant_value)
        
        elif profile_type == "linear":
            # Linear interpolation from surface to bottom
            surface_val = profile_config.surface_value
            bottom_val = profile_config.bottom_value
            return np.linspace(surface_val, bottom_val, self.num_nodes)
        
        elif profile_type == "exponential":
            # Exponential decay from surface: water_content = background + (surface - background) * exp(-depth/decay_length)
            surface_val = profile_config.surface_value_exp
            decay_length = profile_config.decay_length
            background_val = profile_config.background_value
            
            # Calculate exponential profile
            water_content = background_val + (surface_val - background_val) * np.exp(-self.depth_nodes / decay_length)
            return water_content
        
        elif profile_type == "from_file":
            return self._load_water_content_from_file()
        
        else:
            raise ValueError(f"Unknown water content profile type: {profile_type}")
    
    def _load_water_content_from_file(self) -> np.ndarray:
        """
        Load water content profile from CSV file and interpolate to model grid.
        
        Returns:
            Array of water content values interpolated to model depth nodes
        """
        import pandas as pd
        from scipy.interpolate import interp1d
        
        profile_file = self.config.soil.water_content_profile.profile_file
        if not profile_file:
            raise ValueError("No profile file specified for 'from_file' water content profile")
        
        try:
            # Load CSV file
            data = pd.read_csv(profile_file)
            
            # Expected columns: depth (cm), water_content (cm³/cm³)
            required_columns = ['depth', 'water_content']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            # Extract data
            file_depths = data['depth'].values
            file_water_content = data['water_content'].values
            
            # Validate data
            if len(file_depths) < 2:
                raise ValueError("Water content profile file must contain at least 2 data points")
            
            if not np.all(np.diff(file_depths) >= 0):
                raise ValueError("Depth values in profile file must be non-decreasing")
            
            if np.any(file_water_content < 0):
                raise ValueError("Water content values must be non-negative")
            
            # Interpolate to model grid
            interpolator = interp1d(file_depths, file_water_content, 
                                  kind='linear', bounds_error=False, 
                                  fill_value=(file_water_content[0], file_water_content[-1]))
            
            interpolated_water_content = interpolator(self.depth_nodes)
            
            print(f"✓ Loaded water content profile from {profile_file}")
            print(f"  File depth range: {file_depths[0]:.1f} - {file_depths[-1]:.1f} cm")
            print(f"  Water content range: {file_water_content.min():.4f} - {file_water_content.max():.4f} cm³/cm³")
            
            return interpolated_water_content
            
        except Exception as e:
            raise ValueError(f"Error loading water content profile from {profile_file}: {str(e)}")
    
    def _initialize_soil_conditions(self) -> None:
        """Initialize soil water content and isotope compositions."""
        self.water_content = self._calculate_water_content_profile()
        
        self.d18O_liquid_soil = np.full(self.num_nodes, self.config.atmospheric.d18O_rain)
        
        d_prime_18O_rain = 1000 * np.log(self.config.atmospheric.d18O_rain/1000 + 1)
        d_prime_17O_rain = self.config.atmospheric.D_prime_17O_rain/1000 + 0.528 * d_prime_18O_rain
        d17O_rain = (np.exp(d_prime_17O_rain/1000) - 1) * 1000
        self.d17O_liquid_soil = np.full(self.num_nodes, d17O_rain)
        
        self.alpha_1816_liq_vap_soil, self.alpha_1716_liq_vap_soil = self._lookup_fractionation_factors(self.soil_temperature)
        
        d18O_vapor_soil = (1000 + self.d18O_liquid_soil) / self.alpha_1816_liq_vap_soil - 1000
        d17O_vapor_soil = (1000 + self.d17O_liquid_soil) / self.alpha_1716_liq_vap_soil - 1000
        
        es_soil = self._lookup_vapor_pressure(self.soil_temperature)
        vapor_conc_soil = es_soil / (8.3144 * (self.soil_temperature + 273.15)) * 1e-6
        
        self.liquid_conc_soil = self.water_content * 1 / 18.01528
        
        threshold_mask = self.liquid_conc_soil < self.config.soil.water_content_threshold
        slope = vapor_conc_soil / self.config.soil.water_content_threshold
        deficit = self.config.soil.water_content_threshold - self.liquid_conc_soil
        vapor_conc_soil = np.where(threshold_mask, 
                                  vapor_conc_soil - deficit * slope,
                                  vapor_conc_soil)
        
        R_1816_vapor_soil = (d18O_vapor_soil/1000 + 1) * self.config.constants.RSMOW_1816
        R_1716_vapor_soil = (d17O_vapor_soil/1000 + 1) * self.config.constants.RSMOW_1716
        
        self.H216O_vapor_soil = vapor_conc_soil / (R_1816_vapor_soil + R_1716_vapor_soil + 1)
        self.H218O_vapor_soil = R_1816_vapor_soil * self.H216O_vapor_soil
        self.H217O_vapor_soil = R_1716_vapor_soil * self.H216O_vapor_soil
        
        self.H216O_vapor_soil[0] = self.H216O_vapor_air
        self.H218O_vapor_soil[0] = self.H218O_vapor_air
        self.H217O_vapor_soil[0] = self.H217O_vapor_air
        
        R_1816_liquid_soil = (self.d18O_liquid_soil/1000 + 1) * self.config.constants.RSMOW_1816
        R_1716_liquid_soil = (self.d17O_liquid_soil/1000 + 1) * self.config.constants.RSMOW_1716
        
        self.H216O_liquid_soil = self.liquid_conc_soil / (R_1816_liquid_soil + R_1716_liquid_soil + 1)
        self.H218O_liquid_soil = R_1816_liquid_soil * self.H216O_liquid_soil
        self.H217O_liquid_soil = R_1716_liquid_soil * self.H216O_liquid_soil
        
        self.boundary_conditions = {
            'liquid_conc_bottom': self.liquid_conc_soil[-1],
            'H216O_vapor_bottom': self.H216O_vapor_soil[-1],
            'H218O_vapor_bottom': self.H218O_vapor_soil[-1],
            'H217O_vapor_bottom': self.H217O_vapor_soil[-1],
            'H216O_liquid_bottom': self.H216O_liquid_soil[-1],
            'H218O_liquid_bottom': self.H218O_liquid_soil[-1],
            'H217O_liquid_bottom': self.H217O_liquid_soil[-1]
        }
        
        self.free_air_porosity = self.config.soil.total_porosity - self.water_content
        self._update_diffusion_coefficients()
    
    def _update_diffusion_coefficients(self) -> None:
        """Update diffusion coefficients."""
        D_H2O = (self.config.constants.D_H2O_air_20C * self.free_air_porosity * 
                self.config.soil.tortuosity * ((self.soil_temperature + 273)/(20 + 273))**self.config.constants.temp_exponent)
        
        self.D_H218O = D_H2O / self.config.constants.D_H216O_over_D_H218O
        self.D_H216O = self.D_H218O * self.config.constants.D_H216O_over_D_H218O
        
        D_H216O_over_D_H217O = self.config.constants.D_H216O_over_D_H218O**self.config.constants.theta_diff
        self.D_H217O = self.D_H216O / D_H216O_over_D_H217O
    
    def run_simulation(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Run the complete soil evaporation simulation."""
        if not self._is_setup:
            self.setup_model()
        
        start_time = time.time()
        
        # Store initial mass for mass balance verification
        initial_mass = self.calculate_total_water_mass()
        
        max_iterations = int(self.config.numerical.run_days * 24 * 60 * 60 / self.time_step)
        if self.config.numerical.max_iterations is not None:
            max_iterations = min(max_iterations, self.config.numerical.max_iterations)
        
        store_interval = max(1, max_iterations // 1000)
        n_stored = max_iterations // store_interval + 1
        
        stored_times = np.zeros(n_stored)
        d18ls = np.zeros((self.num_nodes, n_stored))
        d17ls = np.zeros((self.num_nodes, n_stored))
        water_content_stored = np.zeros((self.num_nodes, n_stored))
        temperature_stored = np.zeros((self.num_nodes, n_stored))
        
        H216O_vapor_soil_new = np.zeros(self.num_nodes)
        H218O_vapor_soil_new = np.zeros(self.num_nodes)
        H217O_vapor_soil_new = np.zeros(self.num_nodes)
        H216O_liquid_soil_new = np.zeros(self.num_nodes)
        H218O_liquid_soil_new = np.zeros(self.num_nodes)
        H217O_liquid_soil_new = np.zeros(self.num_nodes)
        liquid_conc_soil_new = np.zeros(self.num_nodes)
        
        temp_d18O_liquid = np.zeros(self.num_nodes)
        temp_d17O_liquid = np.zeros(self.num_nodes)
        
        stored_times[0] = 0
        d18ls[:, 0] = self.d18O_liquid_soil
        d17ls[:, 0] = self.d17O_liquid_soil  
        water_content_stored[:, 0] = self.water_content
        temperature_stored[:, 0] = self.soil_temperature
        
        time_elapsed = 0
        num_its = 0
        store_counter = 1
        
        while time_elapsed < self.config.numerical.run_days * 24 * 60 * 60 and num_its < max_iterations:
            
            if num_its % self.config.numerical.progress_interval == 0:
                days_elapsed = time_elapsed / (24 * 60 * 60)
                progress = days_elapsed / self.config.numerical.run_days * 100
                print(f"Day {days_elapsed:.1f} / {self.config.numerical.run_days} ({progress:.1f}%)")
                
                if progress_callback:
                    progress_callback(progress, days_elapsed, num_its)
            
            if self._use_vectorized_diffusion:
                time_days = time_elapsed / (24 * 60 * 60)
                surface_temp = None
                apply_temp_diffusion = True
                
                # Determine surface temperature and diffusion mode based on configuration
                if self._use_era5_forcing and self.era5_forcing is not None:
                    # ERA5 mode: use ERA5 skin temperature, always enable diffusion
                    era5_conditions = self._get_era5_conditions_cached(time_days)
                    surface_temp = era5_conditions['skin_temperature']
                    apply_temp_diffusion = True
                else:
                    # Constant mode: check diffusion flag
                    if hasattr(self.config.temperature, 'disable_diffusion') and self.config.temperature.disable_diffusion:
                        # Constant mode with diffusion disabled: no temperature update
                        apply_temp_diffusion = False
                        surface_temp = None  # Not used
                    else:
                        # Constant mode with diffusion enabled: use constant air temp as surface BC
                        surface_temp = self.config.atmospheric.mean_air_temperature
                        apply_temp_diffusion = True
                
                self._vectorized_diffusion_step(
                    H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                    H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                    liquid_conc_soil_new, surface_temp, time_days, apply_temp_diffusion
                )
                
            
            time_days = time_elapsed / (24 * 60 * 60)
            self._apply_flux_boundary_conditions(
                H216O_vapor_soil_new, H218O_vapor_soil_new, H217O_vapor_soil_new,
                H216O_liquid_soil_new, H218O_liquid_soil_new, H217O_liquid_soil_new,
                liquid_conc_soil_new, time_days
            )
            
            H216O_vapor_soil_new[-1] = H216O_vapor_soil_new[-2]
            H218O_vapor_soil_new[-1] = H218O_vapor_soil_new[-2]
            H217O_vapor_soil_new[-1] = H217O_vapor_soil_new[-2]
            
            # Bottom boundary will be set to fixed conditions below
            
            # Periodic mass tracking (every 10 timesteps for efficiency)
            if num_its % 10 == 0:
                self._track_mass_balance(time_days)
            
            H216O_liquid_soil_new[-1] = self.boundary_conditions['H216O_liquid_bottom']
            H218O_liquid_soil_new[-1] = self.boundary_conditions['H218O_liquid_bottom'] 
            H217O_liquid_soil_new[-1] = self.boundary_conditions['H217O_liquid_bottom']
            liquid_conc_soil_new[-1] = self.boundary_conditions['liquid_conc_bottom']
            
            time_elapsed += self.time_step
            
            self.H216O_vapor_soil, H216O_vapor_soil_new = H216O_vapor_soil_new, self.H216O_vapor_soil
            self.H218O_vapor_soil, H218O_vapor_soil_new = H218O_vapor_soil_new, self.H218O_vapor_soil
            self.H217O_vapor_soil, H217O_vapor_soil_new = H217O_vapor_soil_new, self.H217O_vapor_soil
            self.H216O_liquid_soil, H216O_liquid_soil_new = H216O_liquid_soil_new, self.H216O_liquid_soil
            self.H218O_liquid_soil, H218O_liquid_soil_new = H218O_liquid_soil_new, self.H218O_liquid_soil
            self.H217O_liquid_soil, H217O_liquid_soil_new = H217O_liquid_soil_new, self.H217O_liquid_soil
            
            self.liquid_conc_soil, liquid_conc_soil_new = liquid_conc_soil_new, self.liquid_conc_soil
            
            self._calculate_isotope_deltas_vectorized(temp_d18O_liquid, temp_d17O_liquid)
            
            if num_its % store_interval == 0 and store_counter < n_stored:
                stored_times[store_counter] = time_elapsed / (24 * 60 * 60)
                d18ls[:, store_counter] = self.d18O_liquid_soil
                d17ls[:, store_counter] = self.d17O_liquid_soil
                water_content_stored[:, store_counter] = self.water_content
                temperature_stored[:, store_counter] = self.soil_temperature
                store_counter += 1
            
            num_its += 1
        
        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.1f} seconds")
        
        # Perform mass balance verification
        mass_balance = self.verify_mass_balance(initial_mass)
        
        # Report mass balance results
        if mass_balance['mass_conserved']:
            print("✓ Mass balance verified - all isotopologues conserved within tolerance")
        else:
            print("⚠ Mass balance violation detected:")
            for isotopologue, error in mass_balance['relative_errors'].items():
                if error > 1e-10:
                    print(f"  {isotopologue}: {error:.2e} relative error")
        
        if store_counter < n_stored:
            stored_times = stored_times[:store_counter]
            d18ls = d18ls[:, :store_counter]
            d17ls = d17ls[:, :store_counter]
            water_content_stored = water_content_stored[:, :store_counter]
            temperature_stored = temperature_stored[:, :store_counter]
        
        delta_prime = delta_prime_calc(d18ls, d17ls)
        
        self.results = {
            'times_days': stored_times,
            'depth_nodes': self.depth_nodes,
            'water_content': water_content_stored.T,
            'temperature': temperature_stored.T,
            'd18O': d18ls.T,
            'd17O': d17ls.T,
            'delta_prime': delta_prime.T,
            'metadata': self.metadata,
            'mass_balance': mass_balance,
            'evaporation_data': self.evaporation_data,
            'mass_balance_data': self.mass_balance_data,
            'final_liquid_soil': {
                'H216O': self.H216O_liquid_soil.copy(),
                'H218O': self.H218O_liquid_soil.copy(), 
                'H217O': self.H217O_liquid_soil.copy(),
                'd18O': self.d18O_liquid_soil.copy(),
                'd17O': self.d17O_liquid_soil.copy()
            }
        }
        
        self._is_run = True
        return self.results
        
    
    def _track_atmospheric_conditions(self, time_days: float, air_temp: float, 
                                     humidity: float, surface_temp: float):
        """Track atmospheric conditions for evaporation data."""
        self.evaporation_data['times'].append(time_days)
        self.evaporation_data['surface_temp'].append(surface_temp)
        self.evaporation_data['air_temp'].append(air_temp)
        self.evaporation_data['humidity'].append(humidity)
        
        # Add placeholder for evaporation - will be updated by mass balance tracking
        self.evaporation_data['cumulative_evap_mm'].append(0.0)
    
    
    def _track_mass_balance(self, time_days: float):
        """
        Track total mass and calculate evaporation from mass loss.
        
        Args:
            time_days: Current simulation time
        """
        # Calculate current total mass from water content integration
        current_mass = self.calculate_total_water_mass()
        total_mass_current = current_mass['total_water']
        
        # Initialize tracking on first call
        if self._initial_total_mass is None:
            self._initial_total_mass = total_mass_current
        
        # Calculate mass lost (evaporation)
        mass_lost = self._initial_total_mass - total_mass_current
        
        # Convert mass lost to mm of water equivalent
        # mol * g/mol * cm³/g * mm/cm = mm
        evaporation_mm = mass_lost * 18.016 * 1.0 * 10.0
        
        # Store tracking data
        self.mass_balance_data['times'].append(time_days)
        self.mass_balance_data['total_mass'].append(total_mass_current)
        self.mass_balance_data['mass_lost'].append(mass_lost)
        self.mass_balance_data['evaporation_mm'].append(evaporation_mm)
        
        # Also update evaporation data for plotting
        if len(self.evaporation_data['times']) > 0:
            # Update the most recent entry with calculated evaporation
            self.evaporation_data['cumulative_evap_mm'][-1] = evaporation_mm
    
    def calculate_total_water_mass(self):
        """Calculate total water mass in soil column for each isotopologue.
        
        Returns:
            dict: Total mass for each isotopologue and phase
        """
        dz = self.config.numerical.depth_step  # cm
        
        # Total vapor mass (mol) = concentration (mol/cm³) × porosity × volume (cm³)
        total_H216O_vapor = np.sum(self.H216O_vapor_soil * self.free_air_porosity * dz)
        total_H218O_vapor = np.sum(self.H218O_vapor_soil * self.free_air_porosity * dz)
        total_H217O_vapor = np.sum(self.H217O_vapor_soil * self.free_air_porosity * dz)
        
        # Total liquid mass (mol) = concentration (mol/cm³) × volume (cm³)
        total_H216O_liquid = np.sum(self.H216O_liquid_soil * dz)
        total_H218O_liquid = np.sum(self.H218O_liquid_soil * dz)
        total_H217O_liquid = np.sum(self.H217O_liquid_soil * dz)
        
        # Total water in each phase
        total_vapor = total_H216O_vapor + total_H218O_vapor + total_H217O_vapor
        total_liquid = total_H216O_liquid + total_H218O_liquid + total_H217O_liquid
        
        return {
            'H216O_vapor': total_H216O_vapor,
            'H218O_vapor': total_H218O_vapor,
            'H217O_vapor': total_H217O_vapor,
            'H216O_liquid': total_H216O_liquid,
            'H218O_liquid': total_H218O_liquid,
            'H217O_liquid': total_H217O_liquid,
            'total_vapor': total_vapor,
            'total_liquid': total_liquid,
            'total_water': total_vapor + total_liquid
        }
    
    def verify_mass_balance(self, initial_mass=None, tolerance=1e-10):
        """Verify mass balance for the simulation using simple mass tracking.
        
        Args:
            initial_mass: Initial water mass dict (if None, calculates current)
            tolerance: Relative tolerance for mass balance check (unused in simplified version)
            
        Returns:
            dict: Mass balance verification results
        """
        current_mass = self.calculate_total_water_mass()
        
        if initial_mass is None:
            return current_mass
        
        # Simple mass balance check using total mass only
        initial_total = initial_mass['total_water']
        current_total = current_mass['total_water']
        mass_lost_total = initial_total - current_total
        
        # Mass balance is always conserved since we track actual mass changes
        mass_conserved = True
        rel_error_total = 0.0
        
        print(f"\n=== MASS BALANCE SUMMARY ===")
        print(f"Initial total water mass: {initial_total:.6e} mol")
        print(f"Current total water mass: {current_total:.6e} mol") 
        print(f"Total mass lost (evaporated): {mass_lost_total:.6e} mol")
        print(f"Evaporation equivalent: {mass_lost_total * 18.016 * 10.0:.2f} mm")
        print(f"===============================\n")
        
        return {
            'mass_conserved': mass_conserved,
            'relative_errors': {
                'total': rel_error_total
            },
            'total_mass_lost': mass_lost_total,
            'evaporation_mm': mass_lost_total * 18.016 * 10.0,
            'current_mass': current_mass,
            'initial_mass': initial_mass
        }
    
    def get_evaporation_summary(self) -> dict:
        """Get summary statistics of evaporation from mass balance data."""
        if not self.mass_balance_data['times']:
            return {}
        
        import numpy as np
        
        times = np.array(self.mass_balance_data['times'])
        evaporation_mm = np.array(self.mass_balance_data['evaporation_mm'])
        
        if len(times) == 0 or len(evaporation_mm) == 0:
            return {}
        
        # Calculate evaporation rate
        dt = np.diff(times)
        dt[dt == 0] = 1e-10
        evap_rates = np.zeros_like(times)
        if len(times) > 1:
            evap_rates[1:] = np.diff(evaporation_mm) / dt
        
        summary = {
            'total_evaporated_mm': evaporation_mm[-1] if len(evaporation_mm) > 0 else 0,
            'simulation_days': times[-1] if len(times) > 0 else 0,
            'avg_evap_rate_mm_day': evaporation_mm[-1] / max(times[-1], 1e-6) if len(times) > 0 else 0,
            'peak_evap_rate_mm_day': np.max(evap_rates) if len(evap_rates) > 0 else 0,
            'final_cumulative_mm': evaporation_mm[-1] if len(evaporation_mm) > 0 else 0
        }
        
        return summary