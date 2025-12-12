"""
FMO Complex Exciton Transport Simulation
========================================

This script simulates exciton transport in the 7-site Fenna-Matthews-Olson (FMO)
complex using the Lindblad master equation with Markovian site-local dephasing.

Units:
- Energy: cm⁻¹ (wavenumbers)
- Time: picoseconds (ps)
- Dephasing rate γ: ps⁻¹

Usage:
    python File_1.py [--dephasing_rate GAMMA] [--perturb] [--sink] [--sweep]

Optional features:
    --perturb: Run ensemble with random site energy perturbations
    --sink: Include irreversible sink at reaction center site
    --sweep: Run parameter sweep over dephasing rate and sink rate

Author: Generated for exciton transport simulation
Reference: Adolphs & Renger (2006) - FMO Hamiltonian parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import qutip as qt
import argparse
from typing import Tuple, List, Optional, Dict
import os
from datetime import datetime


# =============================================================================
# Physical Constants and Unit Conversions
# =============================================================================

# Conversion factor: cm⁻¹ to ps⁻¹ (angular frequency units for Hamiltonian)
# 1 cm⁻¹ corresponds to frequency ν = c/λ = c * (1 cm⁻¹) = 2.998e10 Hz
# Angular frequency: ω = 2πν = 2π * 2.998e10 rad/s
# Convert to ps⁻¹: ω = 2π * 2.998e10 / 1e12 ≈ 0.1883 ps⁻¹
# Formula: ω (ps⁻¹) = E (cm⁻¹) * 2π * c * 1e-12, where c in cm/s
CM_TO_PS_INV = 2.0 * np.pi * constants.c * 1e-12 * 1e2  # cm⁻¹ to ps⁻¹
# constants.c is in m/s, so multiply by 100 to get cm/s, then by 1e-12 for ps


# =============================================================================
# FMO Hamiltonian Definition (Adolphs & Renger, 2006)
# =============================================================================

def get_fmo_hamiltonian() -> np.ndarray:
    """
    Construct the 7×7 FMO Hamiltonian matrix (units: cm⁻¹).
    
    Returns:
        7×7 numpy array representing the FMO Hamiltonian
    """
    # Site energies (diagonal elements) in cm⁻¹
    # Values from Adolphs & Renger (2006)
    site_energies = np.array([
        12410.0,  # Site 1
        12530.0,  # Site 2
        12210.0,  # Site 3
        12320.0,  # Site 4
        12480.0,  # Site 5
        12600.0,  # Site 6
        12120.0   # Site 7
    ])
    
    # Inter-site couplings (off-diagonal elements) in cm⁻¹
    # Only non-zero couplings are specified
    H = np.zeros((7, 7))
    
    # Set diagonal (site energies)
    np.fill_diagonal(H, site_energies)
    
    # Set off-diagonal couplings (symmetric matrix)
    # Coupling values from Adolphs & Renger (2006)
    couplings = {
        (0, 1): -87.7,   # Site 1-2
        (0, 2): 5.5,     # Site 1-3
        (1, 2): 30.8,    # Site 2-3
        (1, 3): -8.2,    # Site 2-4
        (2, 3): -7.8,    # Site 3-4
        (2, 4): 6.0,     # Site 3-5
        (3, 4): -13.7,   # Site 4-5
        (3, 5): -9.6,    # Site 4-6
        (4, 5): 18.7,    # Site 5-6
        (4, 6): 11.0,    # Site 5-7
        (5, 6): 9.9,     # Site 6-7
    }
    
    # Fill in couplings (symmetric)
    for (i, j), value in couplings.items():
        H[i, j] = value
        H[j, i] = value
    
    return H


def hamiltonian_to_qutip(H_cm: np.ndarray) -> qt.Qobj:
    """
    Convert Hamiltonian from cm⁻¹ to Qutip Qobj in units of ps⁻¹.
    
    Args:
        H_cm: 7×7 Hamiltonian matrix in cm⁻¹
        
    Returns:
        Qutip Qobj representing the Hamiltonian in ps⁻¹
    """
    # Convert from cm⁻¹ to ps⁻¹
    H_ps = H_cm * CM_TO_PS_INV
    return qt.Qobj(H_ps)


# =============================================================================
# Initial State Preparation
# =============================================================================

def get_initial_state(initial_site: int = 1) -> qt.Qobj:
    """
    Create initial density matrix with exciton localized on specified site.
    
    Args:
        initial_site: Site index (1-7) where exciton starts
        
    Returns:
        Qutip density matrix |initial_site⟩⟨initial_site|
    """
    if not (1 <= initial_site <= 7):
        raise ValueError("initial_site must be between 1 and 7")
    
    # Create basis state |i⟩
    psi = qt.basis(7, initial_site - 1)  # Qutip uses 0-indexing
    # Create density matrix |i⟩⟨i|
    rho0 = qt.ket2dm(psi)
    return rho0


# =============================================================================
# Lindblad Operators for Dephasing
# =============================================================================

def get_dephasing_operators(gamma: float) -> List[qt.Qobj]:
    """
    Create site-local pure-dephasing Lindblad operators.
    
    Each operator is L_i = sqrt(γ) |i⟩⟨i| for site i.
    
    Args:
        gamma: Dephasing rate in ps⁻¹
        
    Returns:
        List of 7 Lindblad collapse operators
    """
    operators = []
    sqrt_gamma = np.sqrt(gamma)
    
    for i in range(7):
        # Create |i⟩⟨i| projector
        L_i = sqrt_gamma * qt.projection(7, i, i)
        operators.append(L_i)
    
    return operators


def get_sink_operator(sink_site: int, k_sink: float) -> qt.Qobj:
    """
    Create irreversible sink Lindblad operator.
    
    Operator: sqrt(k_sink) |sink_site⟩⟨sink_site|
    This removes population from the sink site irreversibly.
    
    Args:
        sink_site: Site index (1-7) where sink is located
        k_sink: Sink rate in ps⁻¹
        
    Returns:
        Lindblad collapse operator for the sink
    """
    if not (1 <= sink_site <= 7):
        raise ValueError("sink_site must be between 1 and 7")
    
    sqrt_k = np.sqrt(k_sink)
    L_sink = sqrt_k * qt.projection(7, sink_site - 1, sink_site - 1)
    return L_sink


# =============================================================================
# Observables Computation
# =============================================================================

def compute_site_populations(rho: qt.Qobj) -> np.ndarray:
    """
    Compute site populations (diagonal elements of density matrix).
    
    Args:
        rho: Density matrix
        
    Returns:
        Array of 7 site populations
    """
    return np.real(np.diag(rho.full()))


def compute_l1_coherence(rho: qt.Qobj) -> float:
    """
    Compute ℓ1-norm coherence: sum of absolute values of off-diagonal elements.
    
    Args:
        rho: Density matrix
        
    Returns:
        ℓ1-norm coherence
    """
    rho_array = rho.full()
    # Sum over all off-diagonal elements
    coherence = np.sum(np.abs(rho_array)) - np.sum(np.abs(np.diag(rho_array)))
    return float(np.real(coherence))


def compute_purity(rho: qt.Qobj) -> float:
    """
    Compute purity: Tr(ρ²).
    
    Args:
        rho: Density matrix
        
    Returns:
        Purity (0 to 1)
    """
    rho_squared = rho * rho
    purity = float(np.real(rho_squared.tr()))
    return purity


# =============================================================================
# Time Evolution
# =============================================================================

def evolve_master_equation(
    H: qt.Qobj,
    rho0: qt.Qobj,
    collapse_operators: List[qt.Qobj],
    times: np.ndarray
) -> qt.Result:
    """
    Solve Lindblad master equation.
    
    Args:
        H: Hamiltonian (Qutip Qobj)
        rho0: Initial density matrix
        collapse_operators: List of Lindblad collapse operators
        times: Time points for evolution
        
    Returns:
        Qutip Result object containing evolved states
    """
    result = qt.mesolve(
        H, rho0, times, collapse_operators,
        options=qt.Options(nsteps=10000, atol=1e-10, rtol=1e-10)
    )
    return result


# =============================================================================
# Perturbation Ensemble
# =============================================================================

def perturb_hamiltonian(H: np.ndarray, perturbation_percent: float) -> np.ndarray:
    """
    Randomly perturb site energies by ±perturbation_percent.
    
    Args:
        H: Original Hamiltonian matrix
        perturbation_percent: Maximum perturbation percentage (e.g., 5.0 for ±5%)
        
    Returns:
        Perturbed Hamiltonian matrix
    """
    H_pert = H.copy()
    n_sites = H.shape[0]
    
    # Random perturbations for diagonal elements (site energies)
    perturbations = np.random.uniform(
        -perturbation_percent / 100.0,
        perturbation_percent / 100.0,
        n_sites
    )
    
    for i in range(n_sites):
        H_pert[i, i] *= (1.0 + perturbations[i])
    
    return H_pert


def run_ensemble(
    H_base: np.ndarray,
    rho0: qt.Qobj,
    gamma: float,
    times: np.ndarray,
    n_realizations: int = 100,
    perturbation_percent: float = 5.0,
    sink_site: Optional[int] = None,
    k_sink: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ensemble of simulations with random site energy perturbations.
    
    Args:
        H_base: Base Hamiltonian matrix
        rho0: Initial density matrix
        gamma: Dephasing rate (ps⁻¹)
        times: Time points
        n_realizations: Number of ensemble members
        perturbation_percent: Maximum perturbation percentage
        sink_site: Optional sink site (1-7)
        k_sink: Sink rate (ps⁻¹)
        
    Returns:
        Tuple of (avg_populations, avg_coherence, avg_purity) arrays
    """
    n_times = len(times)
    n_sites = 7
    
    # Storage arrays
    all_populations = np.zeros((n_realizations, n_times, n_sites))
    all_coherence = np.zeros((n_realizations, n_times))
    all_purity = np.zeros((n_realizations, n_times))
    
    print(f"Running ensemble with {n_realizations} realizations...")
    
    for r in range(n_realizations):
        if (r + 1) % 10 == 0:
            print(f"  Realization {r + 1}/{n_realizations}")
        
        # Perturb Hamiltonian
        H_pert = perturb_hamiltonian(H_base, perturbation_percent)
        H_qutip = hamiltonian_to_qutip(H_pert)
        
        # Build collapse operators
        collapse_ops = get_dephasing_operators(gamma)
        if sink_site is not None:
            collapse_ops.append(get_sink_operator(sink_site, k_sink))
        
        # Evolve
        result = evolve_master_equation(H_qutip, rho0, collapse_ops, times)
        
        # Store observables
        for t_idx, rho_t in enumerate(result.states):
            all_populations[r, t_idx, :] = compute_site_populations(rho_t)
            all_coherence[r, t_idx] = compute_l1_coherence(rho_t)
            all_purity[r, t_idx] = compute_purity(rho_t)
    
    # Compute averages
    avg_populations = np.mean(all_populations, axis=0)
    avg_coherence = np.mean(all_coherence, axis=0)
    avg_purity = np.mean(all_purity, axis=0)
    
    return avg_populations, avg_coherence, avg_purity


# =============================================================================
# Transfer Efficiency Calculation
# =============================================================================

def compute_transfer_efficiency(
    H: qt.Qobj,
    rho0: qt.Qobj,
    collapse_operators: List[qt.Qobj],
    times: np.ndarray,
    sink_site: int,
    return_trace: bool = False
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Compute transfer efficiency to sink site.
    
    Efficiency is defined as the population transferred to the sink
    over the simulation time.
    
    Args:
        H: Hamiltonian
        rho0: Initial density matrix
        collapse_operators: List of collapse operators (including sink)
        times: Time points
        sink_site: Sink site index (1-7)
        return_trace: If True, also return efficiency trace over time
        
    Returns:
        Transfer efficiency (0 to 1), and optionally efficiency trace
    """
    result = evolve_master_equation(H, rho0, collapse_operators, times)
    
    # Track sink population over time
    sink_pop_trace = []
    efficiency_trace = []
    cumulative_efficiency = 0.0
    
    for t_idx, rho_t in enumerate(result.states):
        pops = compute_site_populations(rho_t)
        sink_pop = pops[sink_site - 1]
        sink_pop_trace.append(sink_pop)
        
        # Cumulative efficiency (integrated up to current time)
        if t_idx > 0:
            dt = times[t_idx] - times[t_idx - 1]
            cumulative_efficiency += sink_pop * dt
        efficiency_trace.append(cumulative_efficiency)
    
    # Final efficiency: integrated sink population over full time
    efficiency = np.trapz(sink_pop_trace, times)
    
    if return_trace:
        return efficiency, np.array(efficiency_trace)
    return efficiency, None


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_site_populations(
    times: np.ndarray,
    populations: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot site populations vs time.
    
    Args:
        times: Time array (ps)
        populations: Array of shape (n_times, 7) with site populations
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for site in range(7):
        plt.plot(times, populations[:, site], label=f'Site {site + 1}', linewidth=2)
    
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('Site Population', fontsize=12)
    plt.title('FMO Site Populations vs Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_coherence(
    times: np.ndarray,
    coherence: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ℓ1-norm coherence vs time.
    
    Args:
        times: Time array (ps)
        coherence: Coherence array
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, coherence, 'b-', linewidth=2, label='ℓ₁-norm coherence')
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('Coherence', fontsize=12)
    plt.title('FMO ℓ₁-Norm Coherence vs Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_purity(
    times: np.ndarray,
    purity: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot purity vs time.
    
    Args:
        times: Time array (ps)
        purity: Purity array
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, purity, 'r-', linewidth=2, label='Purity Tr(ρ²)')
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('Purity', fontsize=12)
    plt.title('FMO Purity vs Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_efficiency_vs_coherence(
    coherence: np.ndarray,
    efficiency: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot efficiency vs coherence scatter plot.
    
    Args:
        coherence: Coherence values (can be time series or ensemble)
        efficiency: Efficiency values (can be time series or ensemble)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(coherence, efficiency, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('ℓ₁-Norm Coherence', fontsize=12)
    plt.ylabel('Transfer Efficiency', fontsize=12)
    plt.title('Transfer Efficiency vs Coherence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_matrix: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    save_path: Optional[str] = None
):
    """
    Plot 2D heatmap for parameter sweeps.
    
    Args:
        x_values: X-axis parameter values
        y_values: Y-axis parameter values
        z_matrix: 2D array of z values (shape: len(y_values) x len(x_values))
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(z_matrix, aspect='auto', origin='lower', 
                    extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
                    cmap='viridis', interpolation='bilinear')
    plt.colorbar(im, label='Value')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# Parameter Sweep Functions
# =============================================================================

def parameter_sweep(
    gamma_values: np.ndarray,
    k_sink_values: np.ndarray,
    t_max: float = 5.0,
    dt: float = 0.001,
    initial_site: int = 1,
    sink_site: int = 3
) -> Dict[str, np.ndarray]:
    """
    Perform parameter sweep over dephasing rate and sink rate.
    
    Args:
        gamma_values: Array of dephasing rates to sweep (ps⁻¹)
        k_sink_values: Array of sink rates to sweep (ps⁻¹)
        t_max: Maximum time (ps)
        dt: Time step (ps)
        initial_site: Initial site (1-7)
        sink_site: Sink site (1-7)
        
    Returns:
        Dictionary containing:
            - 'gamma': gamma values
            - 'k_sink': k_sink values
            - 'efficiency': 2D array of efficiencies (shape: len(k_sink) x len(gamma))
            - 'final_coherence': 2D array of final coherence values
            - 'final_purity': 2D array of final purity values
            - 'avg_coherence': 2D array of time-averaged coherence
    """
    times = np.arange(0, t_max + dt, dt)
    H_cm = get_fmo_hamiltonian()
    H_qutip = hamiltonian_to_qutip(H_cm)
    rho0 = get_initial_state(initial_site)
    
    n_gamma = len(gamma_values)
    n_k_sink = len(k_sink_values)
    
    efficiency_matrix = np.zeros((n_k_sink, n_gamma))
    final_coherence_matrix = np.zeros((n_k_sink, n_gamma))
    final_purity_matrix = np.zeros((n_k_sink, n_gamma))
    avg_coherence_matrix = np.zeros((n_k_sink, n_gamma))
    
    total_sims = n_gamma * n_k_sink
    sim_count = 0
    
    print(f"Running parameter sweep: {n_gamma} γ values × {n_k_sink} k_sink values = {total_sims} simulations")
    
    for i, k_sink in enumerate(k_sink_values):
        for j, gamma in enumerate(gamma_values):
            sim_count += 1
            if sim_count % 10 == 0 or sim_count == total_sims:
                print(f"  Progress: {sim_count}/{total_sims} ({(sim_count/total_sims)*100:.1f}%)")
            
            # Build collapse operators
            collapse_ops = get_dephasing_operators(gamma)
            collapse_ops.append(get_sink_operator(sink_site, k_sink))
            
            # Evolve
            result = evolve_master_equation(H_qutip, rho0, collapse_ops, times)
            
            # Compute observables
            coherence_trace = []
            purity_trace = []
            sink_pop_trace = []
            
            for rho_t in result.states:
                coherence_trace.append(compute_l1_coherence(rho_t))
                purity_trace.append(compute_purity(rho_t))
                pops = compute_site_populations(rho_t)
                sink_pop_trace.append(pops[sink_site - 1])
            
            # Store metrics
            efficiency_matrix[i, j] = np.trapz(sink_pop_trace, times)
            final_coherence_matrix[i, j] = coherence_trace[-1]
            final_purity_matrix[i, j] = purity_trace[-1]
            avg_coherence_matrix[i, j] = np.mean(coherence_trace)
    
    return {
        'gamma': gamma_values,
        'k_sink': k_sink_values,
        'efficiency': efficiency_matrix,
        'final_coherence': final_coherence_matrix,
        'final_purity': final_purity_matrix,
        'avg_coherence': avg_coherence_matrix
    }


# =============================================================================
# Data Saving Functions
# =============================================================================

def save_simulation_data(
    times: np.ndarray,
    populations: np.ndarray,
    coherence: np.ndarray,
    purity: np.ndarray,
    efficiency: Optional[float] = None,
    efficiency_trace: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    output_dir: str = "fmo_simulation_data"
):
    """
    Save simulation data to files.
    
    Args:
        times: Time array
        populations: Site populations array (n_times, 7)
        coherence: Coherence array
        purity: Purity array
        efficiency: Transfer efficiency (scalar)
        efficiency_trace: Efficiency over time
        params: Dictionary of simulation parameters
        output_dir: Output directory name
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save arrays as numpy files
    np.save(os.path.join(output_dir, f"times_{timestamp}.npy"), times)
    np.save(os.path.join(output_dir, f"populations_{timestamp}.npy"), populations)
    np.save(os.path.join(output_dir, f"coherence_{timestamp}.npy"), coherence)
    np.save(os.path.join(output_dir, f"purity_{timestamp}.npy"), purity)
    
    if efficiency_trace is not None:
        np.save(os.path.join(output_dir, f"efficiency_trace_{timestamp}.npy"), efficiency_trace)
    
    # Save metrics to text file
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.txt")
    with open(metrics_file, 'w') as f:
        f.write("FMO Simulation Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        if params:
            f.write("Simulation Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write("Coherence Metrics:\n")
        f.write(f"  Final coherence: {coherence[-1]:.6f}\n")
        f.write(f"  Average coherence: {np.mean(coherence):.6f}\n")
        f.write(f"  Max coherence: {np.max(coherence):.6f}\n")
        f.write(f"  Min coherence: {np.min(coherence):.6f}\n")
        f.write("\n")
        
        f.write("Purity Metrics:\n")
        f.write(f"  Final purity: {purity[-1]:.6f}\n")
        f.write(f"  Average purity: {np.mean(purity):.6f}\n")
        f.write(f"  Max purity: {np.max(purity):.6f}\n")
        f.write(f"  Min purity: {np.min(purity):.6f}\n")
        f.write("\n")
        
        if efficiency is not None:
            f.write("Efficiency Metrics:\n")
            f.write(f"  Transfer efficiency: {efficiency:.6f}\n")
            if efficiency_trace is not None:
                f.write(f"  Final efficiency (from trace): {efficiency_trace[-1]:.6f}\n")
    
    print(f"\nData saved to directory: {output_dir}/")
    print(f"  Timestamp: {timestamp}")


def save_sweep_data(
    sweep_results: Dict[str, np.ndarray],
    output_dir: str = "fmo_sweep_data"
):
    """
    Save parameter sweep results to files.
    
    Args:
        sweep_results: Dictionary from parameter_sweep()
        output_dir: Output directory name
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all arrays
    for key, value in sweep_results.items():
        np.save(os.path.join(output_dir, f"{key}_{timestamp}.npy"), value)
    
    # Save parameter sets
    params_file = os.path.join(output_dir, f"parameter_sets_{timestamp}.txt")
    with open(params_file, 'w') as f:
        f.write("Parameter Sweep Sets\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Gamma values (ps⁻¹):\n")
        for i, g in enumerate(sweep_results['gamma']):
            f.write(f"  [{i}] {g:.4f}\n")
        f.write(f"\nK_sink values (ps⁻¹):\n")
        for i, k in enumerate(sweep_results['k_sink']):
            f.write(f"  [{i}] {k:.4f}\n")
    
    print(f"\nSweep data saved to directory: {output_dir}/")
    print(f"  Timestamp: {timestamp}")


# =============================================================================
# Main Simulation Function
# =============================================================================

def run_simulation(
    gamma: float = 1.0,
    t_max: float = 5.0,
    dt: float = 0.001,
    initial_site: int = 1,
    use_ensemble: bool = False,
    n_realizations: int = 100,
    perturbation_percent: float = 5.0,
    use_sink: bool = False,
    sink_site: int = 3,
    k_sink: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float], Optional[np.ndarray]]:
    """
    Run the main FMO exciton transport simulation.
    
    Args:
        gamma: Dephasing rate (ps⁻¹)
        t_max: Maximum time (ps)
        dt: Time step (ps)
        initial_site: Initial site (1-7)
        use_ensemble: Whether to run ensemble with perturbations
        n_realizations: Number of ensemble realizations
        perturbation_percent: Maximum perturbation percentage
        use_sink: Whether to include sink
        sink_site: Sink site (1-7)
        k_sink: Sink rate (ps⁻¹)
        
    Returns:
        Tuple of (times, populations, coherence, purity, efficiency, efficiency_trace)
        efficiency and efficiency_trace are None if use_sink=False
    """
    # Time array
    times = np.arange(0, t_max + dt, dt)
    
    # Get Hamiltonian
    H_cm = get_fmo_hamiltonian()
    
    # Initial state
    rho0 = get_initial_state(initial_site)
    
    efficiency = None
    efficiency_trace = None
    
    if use_ensemble:
        # Run ensemble
        populations, coherence, purity = run_ensemble(
            H_cm, rho0, gamma, times, n_realizations,
            perturbation_percent, sink_site if use_sink else None, k_sink
        )
        # Note: ensemble doesn't compute efficiency yet
    else:
        # Single simulation
        H_qutip = hamiltonian_to_qutip(H_cm)
        
        # Build collapse operators
        collapse_ops = get_dephasing_operators(gamma)
        if use_sink:
            collapse_ops.append(get_sink_operator(sink_site, k_sink))
            print(f"Including sink at site {sink_site} with rate {k_sink} ps⁻¹")
        
        # Evolve
        print("Evolving master equation...")
        result = evolve_master_equation(H_qutip, rho0, collapse_ops, times)
        
        # Compute observables
        n_times = len(times)
        n_sites = 7
        populations = np.zeros((n_times, n_sites))
        coherence = np.zeros(n_times)
        purity = np.zeros(n_times)
        
        for t_idx, rho_t in enumerate(result.states):
            populations[t_idx, :] = compute_site_populations(rho_t)
            coherence[t_idx] = compute_l1_coherence(rho_t)
            purity[t_idx] = compute_purity(rho_t)
        
        if use_sink:
            efficiency, efficiency_trace = compute_transfer_efficiency(
                H_qutip, rho0, collapse_ops, times, sink_site, return_trace=True
            )
            print(f"Transfer efficiency to site {sink_site}: {efficiency:.4f}")
    
    return times, populations, coherence, purity, efficiency, efficiency_trace


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function demonstrating default simulation parameters."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='FMO Complex Exciton Transport Simulation'
    )
    parser.add_argument(
        '--dephasing_rate', type=float, default=1.0,
        help='Dephasing rate γ in ps⁻¹ (default: 1.0)'
    )
    parser.add_argument(
        '--perturb', action='store_true',
        help='Run ensemble with random site energy perturbations'
    )
    parser.add_argument(
        '--sink', action='store_true',
        help='Include irreversible sink at reaction center'
    )
    parser.add_argument(
        '--sink_site', type=int, default=3,
        help='Sink site index (1-7, default: 3)'
    )
    parser.add_argument(
        '--k_sink', type=float, default=0.1,
        help='Sink rate in ps⁻¹ (default: 0.1)'
    )
    parser.add_argument(
        '--n_realizations', type=int, default=100,
        help='Number of ensemble realizations (default: 100)'
    )
    parser.add_argument(
        '--perturbation_percent', type=float, default=5.0,
        help='Maximum perturbation percentage (default: 5.0)'
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help='Run parameter sweep over dephasing rate and sink rate'
    )
    parser.add_argument(
        '--gamma_min', type=float, default=0.1,
        help='Minimum gamma for sweep (default: 0.1)'
    )
    parser.add_argument(
        '--gamma_max', type=float, default=5.0,
        help='Maximum gamma for sweep (default: 5.0)'
    )
    parser.add_argument(
        '--gamma_n', type=int, default=10,
        help='Number of gamma values for sweep (default: 10)'
    )
    parser.add_argument(
        '--k_sink_min', type=float, default=0.01,
        help='Minimum k_sink for sweep (default: 0.01)'
    )
    parser.add_argument(
        '--k_sink_max', type=float, default=1.0,
        help='Maximum k_sink for sweep (default: 1.0)'
    )
    parser.add_argument(
        '--k_sink_n', type=int, default=10,
        help='Number of k_sink values for sweep (default: 10)'
    )
    parser.add_argument(
        '--save_data', action='store_true',
        help='Save simulation data to files'
    )
    
    args = parser.parse_args()
    
    # Run parameter sweep if requested
    if args.sweep:
        print("=" * 60)
        print("FMO Parameter Sweep Simulation")
        print("=" * 60)
        print(f"Gamma range: {args.gamma_min} → {args.gamma_max} ps⁻¹ ({args.gamma_n} values)")
        print(f"K_sink range: {args.k_sink_min} → {args.k_sink_max} ps⁻¹ ({args.k_sink_n} values)")
        print(f"Time range: 0 → 5 ps, dt = 0.001 ps")
        print(f"Initial site: 1, Sink site: {args.sink_site}")
        print("=" * 60)
        
        gamma_values = np.linspace(args.gamma_min, args.gamma_max, args.gamma_n)
        k_sink_values = np.linspace(args.k_sink_min, args.k_sink_max, args.k_sink_n)
        
        sweep_results = parameter_sweep(
            gamma_values, k_sink_values,
            t_max=5.0, dt=0.001, initial_site=1, sink_site=args.sink_site
        )
        
        # Generate heatmaps
        print("\nGenerating heatmaps...")
        plot_heatmap(
            sweep_results['gamma'], sweep_results['k_sink'],
            sweep_results['efficiency'],
            'Dephasing Rate γ (ps⁻¹)', 'Sink Rate k_sink (ps⁻¹)',
            'Transfer Efficiency Heatmap'
        )
        plot_heatmap(
            sweep_results['gamma'], sweep_results['k_sink'],
            sweep_results['final_coherence'],
            'Dephasing Rate γ (ps⁻¹)', 'Sink Rate k_sink (ps⁻¹)',
            'Final Coherence Heatmap'
        )
        plot_heatmap(
            sweep_results['gamma'], sweep_results['k_sink'],
            sweep_results['final_purity'],
            'Dephasing Rate γ (ps⁻¹)', 'Sink Rate k_sink (ps⁻¹)',
            'Final Purity Heatmap'
        )
        
        # Efficiency vs coherence scatter (flatten matrices)
        efficiency_flat = sweep_results['efficiency'].flatten()
        coherence_flat = sweep_results['final_coherence'].flatten()
        plot_efficiency_vs_coherence(coherence_flat, efficiency_flat)
        
        # Save sweep data
        if args.save_data:
            save_sweep_data(sweep_results)
        
        print("\nParameter sweep complete!")
        return
    
    # Run single simulation
    print("=" * 60)
    print("FMO Complex Exciton Transport Simulation")
    print("=" * 60)
    print(f"Dephasing rate γ = {args.dephasing_rate} ps⁻¹")
    print(f"Time range: 0 → 5 ps, dt = 0.001 ps")
    print(f"Initial site: 1")
    if args.sink:
        print(f"Sink site: {args.sink_site}, k_sink = {args.k_sink} ps⁻¹")
    print("=" * 60)
    
    times, populations, coherence, purity, efficiency, efficiency_trace = run_simulation(
        gamma=args.dephasing_rate,
        t_max=5.0,
        dt=0.001,
        initial_site=1,
        use_ensemble=args.perturb,
        n_realizations=args.n_realizations,
        perturbation_percent=args.perturbation_percent,
        use_sink=args.sink,
        sink_site=args.sink_site,
        k_sink=args.k_sink
    )
    
    # Generate plots
    print("\nGenerating plots...")
    plot_site_populations(times, populations)
    plot_coherence(times, coherence)
    plot_purity(times, purity)
    
    # Efficiency vs coherence plot (if sink is used)
    if args.sink and efficiency is not None and efficiency_trace is not None:
        # Use final coherence values vs efficiency trace
        plot_efficiency_vs_coherence(coherence, efficiency_trace)
    
    # Save data if requested
    if args.save_data:
        params = {
            'gamma': args.dephasing_rate,
            't_max': 5.0,
            'dt': 0.001,
            'initial_site': 1,
            'use_sink': args.sink,
            'sink_site': args.sink_site,
            'k_sink': args.k_sink if args.sink else None
        }
        save_simulation_data(
            times, populations, coherence, purity,
            efficiency, efficiency_trace, params
        )
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

