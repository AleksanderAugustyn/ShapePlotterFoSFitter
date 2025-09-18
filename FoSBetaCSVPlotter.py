"""
Beta Parameters CSV Plotter with Parallel Processing
Reads CSV files with beta parameters and plots converged shapes using multiple processes
"""

import os
import re
import sys
import time
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
from multiprocessing import Pool, cpu_count, Manager
import argparse

from BetaDeformationCalculator import BetaDeformationCalculator
from ShapePlotterFoSFitter import FoSParameters, FoSShapeCalculator


def extract_parameters_from_filename(filename: str) -> Tuple[int, int, int, int]:
    """
    Extract Z, N, points, and maxbeta from the filename.

    Args:
        filename: CSV filename like 'combined_results_Z92_N144_points720_maxbeta20.csv'

    Returns:
        Tuple of (Z, N, points, maxbeta)
    """
    pattern = r'combined_results_Z(\d+)_N(\d+)_points(\d+)_maxbeta(\d+)\.csv'
    match = re.match(pattern, os.path.basename(filename))

    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")

    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def plot_single_shape(args):
    """
    Worker function to plot a single converged shape.

    Args:
        args: Tuple of (row_index, row_data, z, n, base_filename, total_shapes, counter_dict)
    """
    row_index, row_data, z, n, base_filename, total_shapes, counter_dict = args

    # Set matplotlib backend in each worker
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        # Extract FoS parameters
        c_elongation = row_data['c_elongation']
        q2 = row_data['q2']
        a3 = row_data['a3']
        a4 = row_data['a4']
        a5 = row_data['a5']
        a6 = row_data['a6']
        applied_shift = row_data['applied_shift']

        # Create FoS parameters
        fos_params = FoSParameters(
            protons=z,
            neutrons=n,
            c_elongation=c_elongation,
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

        # Create a beta shape using spherical harmonics
        theta_beta = np.linspace(0, np.pi, 720)

        # Calculate the unnormalized beta shape
        radius_beta_unnormalized = np.ones_like(theta_beta) * fos_params.radius0

        for l, beta_l in beta_dict.items():
            from scipy.special import sph_harm_y
            ylm = sph_harm_y(l, 0, theta_beta, 0.0).real  # m=0 for axial symmetry
            radius_beta_unnormalized += fos_params.radius0 * beta_l * ylm

        # Calculate volume fixing factor
        volume_unnormalized = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(
            radius=radius_beta_unnormalized, theta=theta_beta
        )
        sphere_volume = (4 / 3) * np.pi * fos_params.radius0 ** 3
        volume_fixing_factor = sphere_volume / volume_unnormalized
        radius_fixing_factor = volume_fixing_factor ** (1 / 3)

        # Apply volume fixing to get a normalized radius
        radius_beta = radius_fixing_factor * radius_beta_unnormalized

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

        # Extract energy values
        mass_excess = row_data['mass_excess']
        e_total = row_data['E_total']
        e_macro = row_data['E_macro']
        e_micro = row_data['E_micro']
        e_surf = row_data['E_surf']
        e_coulomb = row_data['E_coulomb']

        # Create a figure
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
        ax.set_title(f'Nuclear Shape (Z={z}, N={n}, A={z + n}) - Shape #{row_index + 1}', fontsize=14)

        # Set plot limits
        max_val = max(np.max(np.abs(z_fos_shifted)), np.max(np.abs(rho_fos))) * 1.2
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)

        # Add legend
        ax.legend(loc='upper left', fontsize=10)

        # Create info text
        info_text = (
            f"FoS Parameters:\n"
            f"c_elongation = {c_elongation:.3f}, q₂ = {q2:.3f}\n"
            f"a₃ = {a3:.3f}, a₄ = {a4:.3f}\n"
            f"a₅ = {a5:.3f}, a₆ = {a6:.3f}\n"
            f"Applied shift = {applied_shift:.3f} fm\n"
            f"\nBeta Parameters:\n"
        )

        # Add significant beta parameters
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
            f"Beta shape: {beta_surface_area:.2f} fm²\n"
            f"\nEnergy Components:\n"
            f"Mass excess: {mass_excess:.3f} MeV\n"
            f"E_total: {e_total:.3f} MeV\n"
            f"E_macro: {e_macro:.3f} MeV\n"
            f"E_micro: {e_micro:.3f} MeV\n"
            f"E_surface: {e_surf:.3f} MeV\n"
            f"E_coulomb: {e_coulomb:.3f} MeV"
        )

        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.85, 0.98, info_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props, fontfamily='monospace')

        # Save figure
        output_directory = "results/FitShapePlots"
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"{base_filename}_shape{row_index + 1:05d}_c{c_elongation:.3f}_a3{a3:.3f}_a4{a4:.3f}_a5{a5:.3f}_a6{a6:.3f}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Update progress counter
        with counter_dict['lock']:
            counter_dict['completed'] += 1
            completed = counter_dict['completed']

        # Print progress every 100 shapes or at completion
        if completed % 100 == 0 or completed == total_shapes:
            elapsed = time.time() - counter_dict['start_time']
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_shapes - completed) / rate if rate > 0 else 0
            print(f"Progress: {completed}/{total_shapes} ({100 * completed / total_shapes:.1f}%) - "
                  f"Rate: {rate:.1f} shapes/s - ETA: {eta / 60:.1f} min")

        return f"Success: {output_path}"

    except Exception as e:
        return f"Error processing shape {row_index + 1}: {str(e)}"


def process_csv_file(csv_filename: str, num_processes: int = None, batch_size: int = 1000):
    """
    Process CSV file and create plots for all shapes using parallel processing.

    Args:
        csv_filename: Path to CSV file
        num_processes: Number of processes to use (default: CPU count - 1)
        batch_size: Number of shapes to process in each batch
    """
    # Extract parameters from filename
    try:
        z, n, points, maxbeta = extract_parameters_from_filename(csv_filename)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Read the CSV file
    print(f"Reading CSV file: {csv_filename}")
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Process all shapes (no longer filtering by converged column)
    if len(df) == 0:
        print("No shapes found in the CSV file.")
        return

    print(f"Found {len(df)} shapes.")
    print(f"Nuclear parameters: Z={z}, N={n}, A={z + n}")

    # Create output filename base
    base_filename = f'Z{z}_N{n}_points{points}_maxbeta{maxbeta}'

    # Set the number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    print(f"Using {num_processes} processes for parallel processing.")

    # Create a manager for shared counter
    manager = Manager()
    counter_dict = manager.dict()
    counter_dict['completed'] = 0
    counter_dict['lock'] = manager.Lock()
    counter_dict['start_time'] = time.time()

    # Prepare arguments for parallel processing
    args_list = []
    for idx, (_, row) in enumerate(df.iterrows()):
        args_list.append((idx, row, z, n, base_filename, len(df), counter_dict))

    # Process in batches to avoid memory issues
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    print(f"Processing {len(args_list)} shapes in {total_batches} batches of up to {batch_size} shapes each.")

    start_time = time.time()
    error_count = 0

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(args_list))
        batch_args = args_list[batch_start:batch_end]

        # Process batch with pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(plot_single_shape, batch_args)

        # Count errors in this batch
        batch_errors = sum(1 for r in results if r.startswith("Error"))
        error_count += batch_errors

        if batch_errors > 0:
            print(f"Batch {batch_num + 1}/{total_batches} completed with {batch_errors} errors.")

    # Final statistics
    elapsed_time = time.time() - start_time
    success_count = len(df) - error_count

    print(f"\nProcessing complete!")
    print(f"Total time: {elapsed_time / 60:.1f} minutes")
    print(f"Average rate: {len(df) / elapsed_time:.1f} shapes/second")
    print(f"Successfully processed: {success_count}/{len(df)} shapes")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Plot nuclear shapes from beta parameters CSV file')
    parser.add_argument('csv_filename', help='CSV file containing beta parameters')
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('-b', '--batch-size', type=int, default=1000,
                        help='Number of shapes to process in each batch (default: 1000)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_filename):
        print(f"Error: File '{args.csv_filename}' not found.")
        sys.exit(1)

    process_csv_file(args.csv_filename, args.processes, args.batch_size)


if __name__ == "__main__":
    main()