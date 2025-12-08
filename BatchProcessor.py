"""
BatchProcessor.py
Handles the calculation and file saving for command-line batch execution.
"""

import os
import sys

import numpy as np

from CylindricalToSphericalConverter import CylindricalToSphericalConverter
from fos_parametrization import FoSParameters, FoSShapeCalculator


def calculate_and_save_shape(z, n, c_elongation, a3, a4, a5, a6, number_of_points, output_dir="."):
    """
    Calculate nuclear shape and save coordinates to files.
    """
    try:
        # Create parameters
        current_params = FoSParameters(
            protons=z,
            neutrons=n,
            c_elongation=c_elongation,
            a3=a3,
            a4=a4,
            a5=a5,
            a6=a6
        )

        # Calculate shape
        calculator_fos = FoSShapeCalculator(current_params)
        z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=number_of_points)

        # Prepare for conversion
        z_work = z_fos_cylindrical.copy()
        cumulative_shift = 0.0
        is_converted = False

        # Check if the shape can be unambiguously converted
        cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
        is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=number_of_points, tolerance=1e-9)

        # If not convertible, try shifting
        if not is_convertible:
            z_step = 0.1  # fm
            shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0
            z_length: float = float(np.max(z_fos_cylindrical)) - float(np.min(z_fos_cylindrical))
            max_shift = z_length / 2.0

            while abs(cumulative_shift) < max_shift:
                cumulative_shift += shift_direction * z_step
                z_work = z_fos_cylindrical + cumulative_shift

                cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=number_of_points, tolerance=1e-9):
                    is_convertible = True
                    is_converted = True
                    break

        if not is_convertible:
            print("Error: Shape cannot be converted to spherical coordinates", file=sys.stderr)
            return False

        # Convert to spherical coordinates
        theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=number_of_points)

        # Generate phi values from 0 to 2π with 360 steps
        phi_values = np.linspace(0, 2 * np.pi, 360)

        # Create filename
        params = [current_params.c_elongation, current_params.q2, a3, a4, a5, a6]
        filename_cylindrical = os.path.join(output_dir, f"cylindrical_coords_{z}_{n}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat")
        filename_spherical = os.path.join(output_dir, f"spherical_coords_{z}_{n}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat")
        filename_cartesian = os.path.join(output_dir, f"cartesian_coords_{z}_{n}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat")

        # Save to files
        with open(filename_cylindrical, 'w') as f:
            f.write("# Cylindrical coordinates: rho(fm) z(fm) Phi (radians)\n")
            f.write(f"# Z={z}, N={n}\n")
            f.write(f"# Parameters: c={current_params.c_elongation:.3f}, q2={current_params.q2:.3f}, a3={a3:.3f}, a4={a4:.3f}, a5={a5:.3f}, a6={a6:.3f}\n")
            if is_converted:
                f.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
            f.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
            f.write("# Format: rho(fm) z(fm) phi(radians)\n")

            for phi in phi_values:
                for i, (rho, z) in enumerate(zip(rho_fos_cylindrical, z_work)):
                    f.write(f"{rho:.6f} {z:.6f} {phi:.6f}\n")

        with open(filename_spherical, 'w') as f:
            f.write("# Spherical coordinates: r(fm) theta(radians) Phi (radians)\n")
            f.write(f"# Z={z}, N={n}\n")
            f.write(f"# Parameters: c={current_params.c_elongation:.3f}, q2={current_params.q2:.3f}, a3={a3:.3f}, a4={a4:.3f}, a5={a5:.3f}, a6={a6:.3f}\n")
            if is_converted:
                f.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
            f.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
            f.write("# Format: r(fm) theta(radians) phi(radians)\n")

            for phi in phi_values:
                for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                    f.write(f"{r:.6f} {theta:.6f} {phi:.6f}\n")

        with open(filename_cartesian, 'w') as f_cartesian:
            f_cartesian.write("# Cartesian coordinates: x(fm) y(fm) z(fm)\n")
            f_cartesian.write(f"# Z={z}, N={n}\n")
            f_cartesian.write(f"# Parameters: c={current_params.c_elongation:.3f}, q2={current_params.q2:.3f}, a3={a3:.3f}, a4={a4:.3f}, a5={a5:.3f}, a6={a6:.3f}\n")
            if is_converted:
                f_cartesian.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
            f_cartesian.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
            f_cartesian.write("# Format: x(fm) y(fm) z(fm)\n")

            for phi in phi_values:
                for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    f_cartesian.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        return True

    except Exception as e:
        print(f"Error during calculation: {e}", file=sys.stderr)
        return False
