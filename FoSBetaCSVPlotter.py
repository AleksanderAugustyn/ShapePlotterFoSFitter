"""
Beta Parameters CSV Plotter
Reads CSV files with beta parameters and plots converged shapes
"""

import os
import re
import sys
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BetaDeformationCalculator import BetaDeformationCalculator
from ShapePlotterFoSFitter import FoSParameters, FoSShapeCalculator

matplotlib.use('Agg')  # Non-interactive backend for saving files


def extract_parameters_from_filename(filename: str) -> Tuple[int, int, int, int]:
    """
    Extract Z, N, points, and maxbeta from filename.

    Args:
        filename: CSV filename like 'beta_parameters_fitted_Z92_N144_points720_maxbeta20.csv'

    Returns:
        Tuple of (Z, N, points, maxbeta)
    """
    pattern = r'beta_parameters_fitted_Z(\d+)_N(\d+)_points(\d+)_maxbeta(\d+)\.csv'
    match = re.match(pattern, os.path.basename(filename))

    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")

    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def plot_converged_shape(row_data: pd.Series, z: int, n: int, output_filename: str, row_index: int):
    """
    Plot a single converged shape with FoS and beta representations.

    Args:
        row_data: DataFrame row with parameters
        z: Number of protons
        n: Number of neutrons
        output_filename: Base filename for output
        row_index: Index of the row in converged shapes
    """
    # Extract FoS parameters
    q2 = row_data['q2']
    a3 = row_data['a3']
    a4 = row_data['a4']
    a5 = row_data['a5']
    a6 = row_data['a6']
    applied_shift = row_data['applied_shift']

    # Calculate c from q2 and a4
    c = q2 + 1.0 + 1.5 * a4

    # Create FoS parameters
    fos_params = FoSParameters(
        protons=z,
        neutrons=n,
        c_elongation=c,
        q2=q2,
        a3=a3,
        a4=a4,
        a5=a5,
        a6=a6
    )

    # Calculate FoS shape
    calculator_fos = FoSShapeCalculator(fos_params)
    z_fos, rho_fos = calculator_fos.calculate_shape(n_points=720)

    # Apply shift to FoS shape
    z_fos_shifted = z_fos + applied_shift

    # Extract beta parameters (beta_1 through beta_20)
    beta_dict = {}
    for l in range(1, 21):
        beta_dict[l] = row_data[f'beta_{l}']

    # Create beta shape using spherical harmonics
    # First, we need to create a dummy theta array for reconstruction
    theta_beta = np.linspace(0, np.pi, 720)

    # Calculate the unnormalized beta shape
    radius_beta_unnorm = np.ones_like(theta_beta) * fos_params.radius0

    for l, beta_l in beta_dict.items():
        # Use BetaDeformationCalculator's spherical harmonic function
        from scipy.special import sph_harm_y
        ylm = sph_harm_y(l, 0, theta_beta, 0).real  # m=0 for axial symmetry
        radius_beta_unnorm += fos_params.radius0 * beta_l * ylm

    # Calculate volume fixing factor
    volume_unnorm = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(
        radius=radius_beta_unnorm, theta=theta_beta
    )
    sphere_volume = (4 / 3) * np.pi * fos_params.radius0 ** 3
    volume_fixing_factor = sphere_volume / volume_unnorm
    radius_fixing_factor = volume_fixing_factor ** (1 / 3)

    # Apply volume fixing to get normalized radius
    radius_beta = radius_fixing_factor * radius_beta_unnorm

    # Convert beta shape to cylindrical coordinates for plotting
    z_beta = radius_beta * np.cos(theta_beta)
    rho_beta = radius_beta * np.sin(theta_beta)

    # Calculate volumes and surface areas
    fos_volume = calculator_fos.calculate_volume_in_cylindrical_coordinates(z_fos_shifted, rho_fos)
    fos_surface_area = calculator_fos.calculate_surface_area_in_cylindrical_coordinates(z_fos_shifted, rho_fos)

    beta_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(
        radius=radius_beta, theta=theta_beta
    )
    beta_surface_area = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(
        radius=radius_beta, theta=theta_beta
    )

    # Extract RMSE values
    rmse_spherical = row_data['rmse_spherical']
    rmse_analytical = row_data['rmse_analytical']
    rmse_fitted = row_data['rmse_fitted']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot FoS shape (shifted)
    ax.plot(z_fos_shifted, rho_fos, 'b-', label='FoS shape (shifted)', linewidth=2)
    ax.plot(z_fos_shifted, -rho_fos, 'b-', linewidth=2)

    # Plot beta shape
    ax.plot(z_beta, rho_beta, 'r--', label='Beta reconstruction', linewidth=2)
    ax.plot(z_beta, -rho_beta, 'r--', linewidth=2)

    # Plot reference sphere
    theta_ref = np.linspace(0, 2 * np.pi, 200)
    sphere_x = fos_params.radius0 * np.cos(theta_ref)
    sphere_y = fos_params.radius0 * np.sin(theta_ref)
    ax.plot(sphere_x, sphere_y, 'k:', alpha=0.5, label=f'R₀={fos_params.radius0:.2f} fm')

    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set labels and title
    ax.set_xlabel('z (fm)', fontsize=12)
    ax.set_ylabel('ρ (fm)', fontsize=12)
    ax.set_title(f'Nuclear Shape (Z={z}, N={n}, A={z + n}) - Converged Shape #{row_index + 1}', fontsize=14)

    # Set plot limits
    max_val = max(np.max(np.abs(z_fos_shifted)), np.max(np.abs(rho_fos))) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Create info text
    info_text = (
        f"FoS Parameters:\n"
        f"c = {c:.3f}, q₂ = {q2:.3f}\n"
        f"a₃ = {a3:.3f}, a₄ = {a4:.3f}\n"
        f"a₅ = {a5:.3f}, a₆ = {a6:.3f}\n"
        f"Applied shift = {applied_shift:.3f} fm\n"
        f"\nBeta Parameters:\n"
    )

    # Add beta parameters
    beta_lines = []
    for l in range(1, 21):
        beta_val = beta_dict[l]
        beta_lines.append(f"β_{l} = {beta_val:+.6f}")

    # Format beta parameters in two columns
    for i in range(0, len(beta_lines), 2):
        if i + 1 < len(beta_lines):
            info_text += f"{beta_lines[i]:<15} {beta_lines[i + 1]}\n"
        else:
            info_text += f"{beta_lines[i]}\n"

    # Add RMSE information
    info_text += (
        f"\nRMSE Values:\n"
        f"Spherical conversion: {rmse_spherical:.4f} fm\n"
        f"Beta analytical: {rmse_analytical:.4f} fm\n"
        f"Beta fitted: {rmse_fitted:.4f} fm\n"
        f"\nVolume Information:\n"
        f"Reference sphere: {sphere_volume:.2f} fm³\n"
        f"FoS shape: {fos_volume:.2f} fm³\n"
        f"Beta shape: {beta_volume:.2f} fm³\n"
        f"Volume fixing factor: {volume_fixing_factor:.6f}\n"
        f"Radius fixing factor: {radius_fixing_factor:.6f}\n"
        f"\nSurface Area:\n"
        f"Reference sphere: {4 * np.pi * fos_params.radius0 ** 2:.2f} fm²\n"
        f"FoS shape: {fos_surface_area:.2f} fm²\n"
        f"Beta shape: {beta_surface_area:.2f} fm²"
    )

    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.85, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    # Save figure
    output_directory = "results/FitShapePlots"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"converged_shape_{row_index + 1}_Z{z}_N{n}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


def process_csv_file(csv_filename: str):
    """
    Process CSV file and create plots for all converged shapes.

    Args:
        csv_filename: Path to CSV file
    """
    # Extract parameters from filename
    try:
        z, n, points, maxbeta = extract_parameters_from_filename(csv_filename)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Read CSV file
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Filter for converged shapes
    converged_df = df[df['converged'] == 1]

    if len(converged_df) == 0:
        print("No converged shapes found in the CSV file.")
        return

    print(f"Found {len(converged_df)} converged shapes.")
    print(f"Nuclear parameters: Z={z}, N={n}, A={z + n}")

    # Create output filename base
    base_filename = os.path.splitext(csv_filename)[0]

    # Process each converged shape
    for idx, (_, row) in enumerate(converged_df.iterrows()):
        print(f"\nProcessing converged shape {idx + 1}/{len(converged_df)}...")
        try:
            plot_converged_shape(row, z, n, base_filename, idx)
        except Exception as e:
            print(f"Error processing shape {idx + 1}: {e}")
            continue

    print(f"\nProcessing complete. Created {len(converged_df)} plots.")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: FoSBetaCSVPlotter.py <csv_filename>")
        print("Example: FoSBetaCSVPlotter.py beta_parameters_fitted_Z92_N144_points720_maxbeta20.csv")
        sys.exit(1)

    csv_filename = sys.argv[1]

    if not os.path.exists(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        sys.exit(1)

    process_csv_file(csv_filename)


if __name__ == "__main__":
    main()
