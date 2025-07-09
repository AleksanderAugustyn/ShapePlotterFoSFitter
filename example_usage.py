"""
Example usage of FoSCoordinateTransformer
Demonstrates how to use the coordinate transformer with existing FoS classes
"""

import matplotlib.pyplot as plt
import numpy as np

from ShapePlotterFoSFitter import FoSParameters, FoSShapeCalculator
from fos_coordinate_transformer import FoSCoordinateTransformer


def demonstrate_coordinate_transformation():
    """Demonstrate the coordinate transformation functionality."""

    # Create nuclear parameters (Uranium-236 with some deformation)
    params = FoSParameters(
        protons=92,
        neutrons=144,
        c_elongation=1.3,  # Prolate shape
        a3=0.1,  # Small reflection asymmetry
        a4=0.2,  # Some necking
        a5=0.05,  # Higher order asymmetry
        a6=0.0
    )

    # Calculate FoS shape
    calculator = FoSShapeCalculator(params)
    z_coords, rho_coords = calculator.calculate_shape(n_points=500)

    # Create coordinate transformer
    transformer = FoSCoordinateTransformer(z_coords, rho_coords)

    # Transform to spherical coordinates
    theta, R = transformer.transform_to_spherical(n_theta=180)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Original FoS shape in cylindrical coordinates
    ax1.plot(z_coords, rho_coords, 'b-', linewidth=2, label='FoS shape')
    ax1.plot(z_coords, -rho_coords, 'b-', linewidth=2)
    ax1.set_xlabel('z (fm)')
    ax1.set_ylabel('ρ (fm)')
    ax1.set_title('Original FoS Shape: ρ(z)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Transformed spherical coordinates
    ax2.plot(np.degrees(theta), R, 'r-', linewidth=2, label='R(θ)')
    ax2.set_xlabel('θ (degrees)')
    ax2.set_ylabel('R (fm)')
    ax2.set_title('Transformed Shape: R(θ)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 180)

    # Plot 3: Polar plot of R(θ)
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(theta, R, 'g-', linewidth=2)
    ax3.set_title('Polar Plot: R(θ)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: 3D surface representation (cross-section)
    x, y, z_3d = transformer.get_cartesian_coordinates(n_theta=50, n_phi=50)

    # Plot a few meridional cross-sections
    ax4.set_aspect('equal')
    for i in range(0, 50, 10):
        ax4.plot(z_3d[i, :], x[i, :], alpha=0.7, linewidth=1)
    ax4.set_xlabel('z (fm)')
    ax4.set_ylabel('x (fm)')
    ax4.set_title('3D Cross-sections')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print some calculated properties
    print("Nuclear Shape Properties:")
    print(f"Original FoS volume: {calculator.calculate_volume(z_coords, rho_coords):.1f} fm³")
    print(f"Reference sphere volume: {params.sphere_volume:.1f} fm³")
    print(f"Surface area: {transformer.get_surface_area():.1f} fm²")

    I_parallel, I_perpendicular = transformer.get_moment_of_inertia()
    print(f"Moment of inertia (parallel): {I_parallel:.2e} fm⁵")
    print(f"Moment of inertia (perpendicular): {I_perpendicular:.2e} fm⁵")

    # Export spherical data
    transformer.export_spherical_data("spherical_coordinates.txt")
    print("Spherical coordinate data exported to 'spherical_coordinates.txt'")


def compare_shapes():
    """Compare different nuclear shapes in spherical coordinates."""

    shapes = [
        ("Spherical", {"c_elongation": 1.0, "a3": 0.0, "a4": 0.0}),
        ("Prolate", {"c_elongation": 1.4, "a3": 0.0, "a4": 0.0}),
        ("Oblate", {"c_elongation": 0.7, "a3": 0.0, "a4": 0.0}),
        ("Pear-shaped", {"c_elongation": 1.2, "a3": 0.3, "a4": 0.0}),
        ("Necked", {"c_elongation": 1.5, "a3": 0.0, "a4": 0.4}),
    ]

    plt.figure(figsize=(10, 8))

    for i, (name, params_dict) in enumerate(shapes):
        # Create parameters
        params = FoSParameters(protons=92, neutrons=144, **params_dict)

        # Calculate shape
        calculator = FoSShapeCalculator(params)
        z_coords, rho_coords = calculator.calculate_shape()

        # Transform to spherical
        transformer = FoSCoordinateTransformer(z_coords, rho_coords)
        theta, R = transformer.transform_to_spherical()

        # Plot
        plt.subplot(2, 3, i + 1, projection='polar')
        plt.plot(theta, R, linewidth=2, label=name)
        plt.title(name)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("FoS Coordinate Transformer Demo")
    print("=" * 40)

    demonstrate_coordinate_transformation()
    print("\n" + "=" * 40)
    compare_shapes()
