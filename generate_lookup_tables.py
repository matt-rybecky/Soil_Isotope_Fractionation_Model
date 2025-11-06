#!/usr/bin/env python3
"""
Lookup Table Generator for Soil Evaporation Model
================================================

Pre-computes temperature-dependent lookup tables for vapor pressure and 
fractionation factors to eliminate startup calculations.

Run once to generate lookup table files:
    python generate_lookup_tables.py
"""

import numpy as np
from pathlib import Path

def generate_lookup_tables():
    """Generate and save lookup tables for soil evaporation model."""
    
    # Temperature range: -50°C to +60°C with 0.01°C resolution
    temp_min = -50.0
    temp_max = 60.0
    temp_resolution = 0.01
    temp_range = np.arange(temp_min, temp_max + temp_resolution, temp_resolution)
    
    print(f"Generating lookup tables for {len(temp_range)} temperature points...")
    
    # Vapor pressure lookup: es = 6.112 * exp(17.67*T/(T+243.5)) * 100
    es_lookup = 6.112 * np.exp(17.67 * temp_range / (temp_range + 243.5)) * 100
    
    # Fractionation factors (Horita & Wesolowski 1994)
    # CORRECTED: The formula gives vapor/liquid, but model needs liquid/vapor
    T_K = temp_range + 273.15
    alpha_arg = (-7.685 + 6713.0/T_K - 1.666e6/(T_K**2)) / 1000.0
    alpha_1816_lookup = 1.0 / np.exp(alpha_arg)  # Invert to get liquid/vapor fractionation
    
    # Mass-dependent relationship (theta_eq = 0.529)
    alpha_1716_lookup = alpha_1816_lookup**0.529
    
    # Save lookup tables
    output_dir = Path(__file__).parent / 'lookup_tables'
    output_dir.mkdir(exist_ok=True)
    
    # Save temperature parameters
    np.savez(output_dir / 'temperature_params.npz',
             temp_min=temp_min, temp_max=temp_max, temp_resolution=temp_resolution)
    
    # Save lookup arrays
    np.save(output_dir / 'vapor_pressure_lookup.npy', es_lookup)
    np.save(output_dir / 'alpha_1816_lookup.npy', alpha_1816_lookup)
    np.save(output_dir / 'alpha_1716_lookup.npy', alpha_1716_lookup)
    
    print(f"✓ Lookup tables saved to {output_dir}")
    print(f"  Memory size: ~{(len(temp_range) * 3 * 8 / 1024):.1f} KB")
    
    return True

if __name__ == "__main__":
    generate_lookup_tables()