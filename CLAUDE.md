# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an HPVsim (Human Papillomavirus simulation) analysis repository for therapeutic vaccine (TxVx) studies across multiple countries. The project models HPV transmission, cancer progression, and intervention effectiveness using agent-based modeling.

## Core Architecture

### Main Components

- **Simulation Engine**: `run_sim.py` - Core simulation creation and configuration
- **Scenario Runner**: `run_scenarios.py` - Batch execution of scenarios across countries
- **Calibration**: `run_calibration.py` - Model calibration using Optuna optimization
- **Data Management**: `pars_data.py` - Parameter loading and country-specific data
- **Utilities**: `utils.py` - Common functions for data processing and visualization
- **Analyzers**: `analyzers.py` - Custom HPVsim analyzers (DALYs, costs, etc.)

### Key Data Files

- `locations.py` - Defines all supported countries
- `data/` - Country-specific epidemiological data (cancer cases, demographics, etc.)
- `*_pars.csv` - Parameter files for diagnostics and treatments

### Plotting and Analysis

- `plot_*.py` - Various plotting scripts for figures and results visualization
- `plot_results.py` - Main results plotting
- `plot_CEA.py` - Cost-effectiveness analysis plots

## Running Simulations

### Main Entry Points

1. **Single simulation**: `python run_sim.py`
2. **Scenario analysis**: `python run_scenarios.py`
3. **Calibration**: `python run_calibration.py`

### Debug Mode

All main scripts support debug mode for local testing:
- Set `debug = True` in script headers
- Reduces population sizes and enables serial execution
- Allows local testing before HPC deployment

### HPC Configuration

Production runs require HPC resources due to computational intensity:
- Uses multiprocessing with configurable worker counts
- Implements numpy threading controls for cluster environments
- MySQL storage backend for calibration results

## Key Dependencies

- `hpvsim` - Core HPV simulation framework
- `sciris` - Scientific computing utilities
- `optuna` - Optimization framework for calibration
- Standard scientific stack: numpy, pandas, matplotlib, seaborn

## Country-Specific Analysis

The project supports multi-country analysis with standardized data formats:
- Each country has dedicated data files in `data/`
- Parameters are loaded via `pars_data.py`
- Results are generated per country and can be aggregated

## Memory Management

- Uses `do_shrink = True` to reduce memory usage during runs
- Implements numpy threading limitations for cluster compatibility
- Supports large-scale simulations with 10k+ agents per simulation

## Common Workflows

### Adding New Countries

For new countries like Zambia and Cote d'Ivoire:

1. **Data Preparation**: Add country-specific data files to `data/` directory:
   - `{country}_cancer_cases.csv` - Cancer incidence data
   - `{country}_cancer_deaths.csv` - Cancer mortality data
   - `{country}_data.csv` - General epidemiological data
   - `{country}_age_pyramid.csv` - Population demographics
   - Optional: `{country}_hpv_prevalence.csv`, `{country}_cin_types.csv`

2. **Location Registration**: Add country names to `locations.py`

3. **Parameter Configuration**: Update `pars_data.py` to load new country data

4. **Calibration**: Run calibration for new countries using `run_calibration.py`
   - Configure `locations` list to include new countries
   - Set appropriate `n_trials` and `n_workers` based on available resources

### HPV Vaccine Delivery Scenarios

Common vaccination strategies to explore:

1. **Routine Vaccination**: School-based programs targeting specific age groups
2. **Catch-up Campaigns**: One-time vaccination of broader age ranges
3. **Multi-age Cohort**: Vaccination of multiple age groups simultaneously
4. **Coverage Variations**: Different vaccination coverage rates (e.g., 70%, 80%, 90%)

Configure scenarios in `pars_scenarios.py`:
- Define vaccination parameters (age targets, coverage, timing)
- Set intervention start dates and durations
- Specify vaccine efficacy profiles

### Typical Analysis Workflow

1. **Calibration Phase**:
   ```bash
   python run_calibration.py  # Set countries and run calibration
   ```

2. **Scenario Development**:
   - Define vaccination scenarios in `pars_scenarios.py`
   - Configure intervention parameters and timelines

3. **Scenario Execution**:
   ```bash
   python run_scenarios.py  # Run scenarios for calibrated countries
   ```

4. **Results Analysis**:
   - Use `plot_results.py` for standard outputs
   - Use `plot_CEA.py` for cost-effectiveness analysis
   - Custom analysis scripts as needed