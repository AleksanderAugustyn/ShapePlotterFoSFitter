"""
Nuclear Shape Plotter using Fourier-over-Spheroid (FoS) Parametrization
Based on the formulation in Pomorski et al. (2023)
With volume normalization to ensure volume conservation

Now supports both interactive GUI mode and batch command-line mode.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.integrate import simpson

from BetaDeformationCalculator import BetaDeformationCalculator, FitResult
from CylindricalToSphericalConverter import CylindricalToSphericalConverter

matplotlib.use('TkAgg')


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int = 92
    neutrons: int = 144
    c_elongation: float = 1.0  # elongation
    a3: float = 0.0  # reflection asymmetry
    a4: float = 0.0  # neck parameter
    a5: float = 0.0  # higher order parameter
    a6: float = 0.0  # higher order parameter
    q2: float = 0.0  # entangled parameter: c = q2 + 1.0 + 1.5 * a4
    r0_constant: float = 1.16  # Radius constant in fm
    max_beta: int = 12  # Maximum number of beta parameters used for fitting

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons

    @property
    def radius0(self) -> float:
        """Radius of a spherical nucleus with the same number of nucleons."""
        return self.r0_constant * (self.nucleons ** (1 / 3))

    @property
    def z0(self) -> float:
        """Half-length of nucleus."""
        return self.c_elongation * self.radius0

    @property
    def a2(self) -> float:
        """Volume conservation constraint: a2 = a4/3 - a6/5 + ..."""
        return self.a4 / 3.0 - self.a6 / 5.0

    @property
    def sphere_surface_area(self) -> float:
        """Surface area of a sphere with the same nucleon number."""
        return 4 * np.pi * self.radius0 ** 2

    @property
    def sphere_volume(self) -> float:
        """Volume of a sphere with the same nucleon number."""
        return (4 / 3) * np.pi * self.radius0 ** 3

    @property
    def z_sh(self) -> float:
        """Shift to place the center of mass at origin."""
        # From the paper: z_sh = -3/(4π) z_0 (a_3 - a_5/2 + ...)
        # Error in the paper should be: +3/(2π) z_0 (a_3 - a_5/2 + ...)
        return 3.0 / (2.0 * np.pi) * self.z0 * (self.a3 - self.a5 / 2.0)


class FoSShapeCalculator:
    """Class for calculating Fourier-over-Spheroid shapes."""

    def __init__(self, params: FoSParameters):
        self.params = params

    def f_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the shape function f(u).
        # From paper: f(u) = 1 - u² - Σ[a_{2k} cos((k-1/2)πu) + a_{2k+1} sin(kπu)]
        # Error in the paper, should be: f(u) = 1 - u² - Σ[a_{2k} cos((2k-1/2)πu) + a_{2k+1} sin(kπu)]

        Args:
            u: normalized axial coordinate (z - z_sh) / z0

        Returns:
            f: shape function values
        """
        # Base spherical shape
        f: np.ndarray = 1.0 - u ** 2.0

        # Sum Fourier terms
        sum_terms: np.ndarray = np.zeros_like(u)

        # Add Fourier terms
        # k=1: a2 * cos((2 - 1) / 2 * π * u) + a3 sin(1 * π * u)
        sum_terms += self.params.a2 * np.cos((2.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a3 * np.sin(1.0 * np.pi * u)

        # k=2: a4 * cos((4 - 1) / 2 * π * u) + a5 sin(2πu)
        sum_terms += self.params.a4 * np.cos((4.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a5 * np.sin(2.0 * np.pi * u)

        # k=3: a6 * cos((6 - 1) / 2 * π * u)
        sum_terms += self.params.a6 * np.cos((6.0 - 1.0) / 2.0 * np.pi * u)

        return f - sum_terms

    def calculate_shape(self, n_points: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the shape coordinates in (z, ρ) space.

        For axially symmetric case (η=0):
        ρ²(z) = R₀² / c * f((z - z_sh)/z₀)

        Args:
            n_points: Number of points to calculate

        Returns:
            z: axial coordinates
            rho: radial coordinates
        """
        # Calculate shape without a shift first
        z_min: float = -self.params.z0 + self.params.z_sh
        z_max: float = self.params.z0 + self.params.z_sh
        z = np.linspace(z_min, z_max, n_points)

        # Calculate normalized u with shift in z
        u = (z - self.params.z_sh) / self.params.z0

        # Ensure u is in [-1, 1]
        u = np.clip(u, -1.0, 1.0)

        # Calculate f(u)
        f_vals = self.f_function(u)

        # Calculate ρ² = R₀² / c * f(u)
        rho_squared = self.params.radius0 ** 2 / self.params.c_elongation * f_vals

        # Handle negative values (set to 0)
        rho_squared = np.maximum(rho_squared, 0)

        # Calculate ρ
        rho = np.sqrt(rho_squared)

        return z, rho

    @staticmethod
    def calculate_surface_area_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray):
        """
        Calculate the surface area by numerical integration.
        S = 2π ∫ ρ(z) √(1 + (dρ/d_z)²) d_z

        Args:
            z: axial coordinates
            rho: radial coordinates

        Returns:
            surface_area: calculated surface area of the shape in fm²
        """
        if len(z) < 2 or len(rho) < 2:
            return 0.0

        # Calculate the derivative dρ/d_z using finite differences
        d_z = np.diff(z)
        d_rho = np.diff(rho)

        d_rho_dz = d_rho / d_z

        # Calculate the integrand
        integrand = 2 * np.pi * rho[:-1] * np.sqrt(1 + d_rho_dz ** 2)

        # Use simpson rule to calculate the integral
        surface_area: float = float(simpson(integrand, x=z[:-1]))

        return surface_area

    @staticmethod
    def calculate_volume_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate volume by numerical integration.
        V = π ∫ ρ²(z) d_z

        Args:
            z: axial coordinates
            rho: radial coordinates

        Returns:
            volume: calculated volume of the shape in fm³
        """
        # Use the simpson rule to calculate the integral
        volume: float = float(simpson(np.pi * rho[:-1] ** 2, x=z[:-1]))

        return volume

    @staticmethod
    def calculate_center_of_mass_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate the center of mass position along the z-axis.
        z_cm = ∫ z ρ²(z) dz / ∫ ρ²(z) dz
        """
        if len(z) < 2:
            return 0.0

        # Use simpson rule for numerical integration
        z_mid = (z[1:] + z[:-1]) / 2
        rho_mid = (rho[1:] + rho[:-1]) / 2

        # Numerator: ∫ z ρ²(z) dz
        numerator: float = float(simpson(z_mid * rho_mid ** 2, x=z[:-1]))

        # Denominator: ∫ ρ²(z) dz
        denominator: float = float(simpson(rho_mid ** 2, x=z[:-1]))

        if denominator == 0:
            return 0.0

        return numerator / denominator


def calculate_and_save_shape(z, n, q2, a3, a4, a5, a6, number_of_points, output_dir="."):
    """
    Calculate nuclear shape and save coordinates to files.

    Args:
        z: Number of protons
        n: Number of neutrons
        q2: Entangled parameter
        a3: Reflection asymmetry parameter
        a4: Neck parameter
        a5: Higher order parameter
        a6: Higher order parameter
        number_of_points: Number of points for shape calculation
        output_dir: Directory to save output files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Calculate c from q2 and a4
        c = q2 + 1.0 + 1.5 * a4

        # Create parameters
        current_params = FoSParameters(
            protons=z,
            neutrons=n,
            c_elongation=c,
            q2=q2,
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
            z_length = np.max(z_fos_cylindrical) - np.min(z_fos_cylindrical)
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
        params = [c, q2, a3, a4, a5, a6]
        filename_spherical = os.path.join(output_dir, f"spherical_coords_{z}_{n}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat")
        filename_cartesian = os.path.join(output_dir, f"cartesian_coords_{z}_{n}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat")

        # Save to files
        with open(filename_spherical, 'w') as f:
            # Write header for spherical coordinates
            f.write("# Spherical coordinates: r theta(radians) Phi (radians)\n")
            f.write(f"# Z={z}, N={n}\n")
            f.write(f"# Parameters: c={c:.3f}, q2={q2:.3f}, a3={a3:.3f}, a4={a4:.3f}, a5={a5:.3f}, a6={a6:.3f}\n")
            if is_converted:
                f.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
            f.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
            f.write("# Format: r(fm) theta (radians) phi (radians)\n")

            for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                for phi in phi_values:
                    f.write(f"{r:.6f} {theta:.6f} {phi:.6f}\n")

        with open(filename_cartesian, 'w') as f_cartesian:
            # Write header for Cartesian coordinates
            f_cartesian.write("# Cartesian coordinates: x y z\n")
            f_cartesian.write(f"# Z={z}, N={n}\n")
            f_cartesian.write(f"# Parameters: c={c:.3f}, q2={q2:.3f}, a3={a3:.3f}, a4={a4:.3f}, a5={a5:.3f}, a6={a6:.3f}\n")
            if is_converted:
                f_cartesian.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
            f_cartesian.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
            f_cartesian.write("# Format: x(fm) y(fm) z(fm)\n")

            for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                for phi in phi_values:
                    # Convert theta and phi to Cartesian coordinates
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    f_cartesian.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        return True

    except Exception as e:
        print(f"Error during calculation: {e}", file=sys.stderr)
        return False


def create_parser():
    """Create a command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Nuclear Shape Calculator using Fourier-over-Spheroid Parametrization',
        epilog='If no arguments are provided, the interactive GUI will launch.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('Z', type=int, nargs='?', help='Number of protons')
    parser.add_argument('N', type=int, nargs='?', help='Number of neutrons')
    parser.add_argument('q2', type=float, nargs='?', help='Entangled parameter q2')
    parser.add_argument('a3', type=float, nargs='?', help='Reflection asymmetry parameter')
    parser.add_argument('a4', type=float, nargs='?', help='Neck parameter')
    parser.add_argument('a5', type=float, nargs='?', help='Higher order parameter')
    parser.add_argument('a6', type=float, nargs='?', help='Higher order parameter')
    parser.add_argument('number_of_points', type=int, nargs='?', help='Number of points (180-3600)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for saved files (default: current directory)')

    return parser


def validate_parameters(z, n, q2, a3, a4, a5, a6, number_of_points):
    """Validate input parameters."""
    errors = []

    if z <= 0:
        errors.append("Z must be positive")
    if n <= 0:
        errors.append("N must be positive")

    for param_name, param_value in [('q2', q2), ('a3', a3), ('a4', a4), ('a5', a5), ('a6', a6)]:
        if abs(param_value) >= 2:
            errors.append(f"{param_name} must have absolute value less than 2")

    if not (180 <= number_of_points <= 3600):
        errors.append("number_of_points must be between 180 and 3600")

    return errors


class FoSShapePlotter:
    """Class for plotting Fourier-over-Spheroid nuclear shapes."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Default shape parameters
        self.initial_z: int = 92  # Uranium
        self.initial_n: int = 144
        self.initial_q2: float = 0.0
        self.initial_a3: float = 0.0
        self.initial_a4: float = 0.0
        self.initial_a5: float = 0.0
        self.initial_a6: float = 0.0

        # Calculate initial c from q2 and a4
        self.initial_c: float = self.initial_q2 + 1.0 + 1.5 * self.initial_a4

        # UI elements
        self.fig = None
        self.ax_plot = None
        self.ax_text = None
        self.line_fos = None
        self.line_fos_mirror = None
        self.line_fos_spherical = None
        self.line_fos_spherical_mirror = None
        self.line_fos_spherical_shifted_for_conversion = None
        self.line_fos_spherical_shifted_for_conversion_mirror = None
        self.line_beta = None
        self.line_beta_mirror = None
        self.line_beta_fitted = None
        self.line_beta_fitted_mirror = None
        self.reference_sphere_line = None
        self.cm_reference_point = None
        self.cm_fos_calculated = None
        self.cm_theoretical = None
        self.cm_spherical_fit_calculated = None
        self.cm_spherical_fit_calculated_shifted = None
        self.cm_beta_fit_calculated = None

        # Sliders
        self.slider_z = None
        self.slider_n = None
        self.slider_c = None
        self.slider_q2 = None
        self.slider_a3 = None
        self.slider_a4 = None
        self.slider_a5 = None
        self.slider_a6 = None
        self.slider_max_beta = None
        self.slider_number_of_points = None

        # Buttons
        self.reset_button = None
        self.save_button = None
        self.save_coordinates_to_files_button = None
        self.print_beta_button = None
        self.preset_buttons = []

        # Decrement/Increment buttons for sliders
        self.slider_buttons = {}

        # Flag to prevent infinite update loops
        self.updating = False

        # Initialize parameters
        self.nuclear_params = FoSParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            c_elongation=self.initial_c,
            q2=self.initial_q2,
            a3=self.initial_a3,
            a4=self.initial_a4,
            a5=self.initial_a5,
            a6=self.initial_a6
        )

        # Set up the interface
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(16.0, 8.0))

        # Create the main plot on the left side
        self.ax_plot = self.fig.add_subplot(121)

        # Create a text area on the right side
        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.set_xlim(0, 1)
        self.ax_text.set_ylim(0, 1)
        self.ax_text.axis('off')  # Hide axes for the text area

        plt.subplots_adjust(left=0.13, bottom=0.38, right=0.97, top=0.97, wspace=0.1)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_title('Nuclear Shape (Fourier-over-Spheroid Parametrization)', fontsize=14)
        self.ax_plot.set_xlabel('z (fm)', fontsize=12)
        self.ax_plot.set_ylabel('ρ (fm)', fontsize=12)

        # Initialize the shape plot
        calculator_fos = FoSShapeCalculator(self.nuclear_params)
        z, rho = calculator_fos.calculate_shape()

        # Create a reference sphere
        r0 = self.nuclear_params.radius0
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = r0 * np.cos(theta)
        sphere_y = r0 * np.sin(theta)

        # Plot the FoS shape and its mirror
        self.line_fos, = self.ax_plot.plot(z, rho, 'b-', label='FoS shape (normalized)', linewidth=2)
        self.line_fos_mirror, = self.ax_plot.plot(z, -rho, 'b-', linewidth=2)

        # Plot the FoS shape with CM in the origin in spherical coordinates
        self.line_fos_spherical, = self.ax_plot.plot([], [], 'y--', label='FoS shape (spherical)', linewidth=1.5)
        self.line_fos_spherical_mirror, = self.ax_plot.plot([], [], 'y--', linewidth=1.5)

        # Plot the FoS shape with CM shifted for conversion
        self.line_fos_spherical_shifted_for_conversion, = self.ax_plot.plot([], [], 'g:', label='FoS shape (spherical, shifted for conversion)', linewidth=1.5, alpha=0.5)
        self.line_fos_spherical_shifted_for_conversion_mirror, = self.ax_plot.plot([], [], 'g:', linewidth=1.5, alpha=0.5)

        # Plot the beta shape and its mirror (analytical)
        self.line_beta, = self.ax_plot.plot([], [], 'r--', label='Beta shape (analytical)', linewidth=2, alpha=0.7)
        self.line_beta_mirror, = self.ax_plot.plot([], [], 'r--', linewidth=2, alpha=0.7)

        # Plot the beta shape and its mirror (fitted)
        self.line_beta_fitted, = self.ax_plot.plot([], [], 'm-.', label='Beta shape (fitted)', linewidth=2, alpha=0.7)
        self.line_beta_fitted_mirror, = self.ax_plot.plot([], [], 'm-.', linewidth=2, alpha=0.7)

        self.reference_sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray', alpha=0.5, label=f'R₀={r0:.2f} fm')

        # Plot expected center of mass (at origin)
        self.cm_reference_point, = self.ax_plot.plot(0, 0, 'ro', label='Expected CM (Origin)', markersize=6, alpha=0.7)

        # Plot theoretical shift CM
        self.cm_theoretical, = self.ax_plot.plot(0, 0, 'b^', label='CM before theoretical shift', markersize=6, alpha=0.7)

        # Plot the calculated center of mass
        self.cm_fos_calculated, = self.ax_plot.plot(0, 0, 'go', label='Calculated CM (FoS)', markersize=6, alpha=0.7)
        self.cm_spherical_fit_calculated, = self.ax_plot.plot(0, 0, 'ms', label='Calculated CM (Spherical Fit)', markersize=6, alpha=0.7)
        self.cm_spherical_fit_calculated_shifted, = self.ax_plot.plot(0, 0, 'ms', label='Calculated CM (Spherical Fit, shifted)', markersize=6, alpha=0.5)
        self.cm_beta_fit_calculated, = self.ax_plot.plot(0, 0, 'cs', label='Calculated CM (Beta Fit)', markersize=6, alpha=0.7)

        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(0, 1))

    def create_slider_with_buttons(self, slider_name, y_position, label, value_minimal, value_maximal, value_initial, value_step):
        """Create a slider with decrement and increment buttons."""
        # Create a decrement button (left side)
        dec_ax = plt.axes((0.20, y_position, 0.016, 0.024))
        dec_button = Button(dec_ax, '-')

        # Create the slider
        slider_ax = plt.axes((0.25, y_position, 0.5, 0.024))
        slider = Slider(ax=slider_ax, label=label, valmin=value_minimal, valmax=value_maximal, valinit=value_initial, valstep=value_step)

        # Create an increment button (right side)
        inc_ax = plt.axes((0.78, y_position, 0.016, 0.024))
        inc_button = Button(inc_ax, '+')

        # Store buttons for later reference
        self.slider_buttons[slider_name] = {
            'dec_button': dec_button,
            'inc_button': inc_button,
            'slider': slider
        }

        return slider, dec_button, inc_button

    def setup_controls(self):
        """Set up all UI controls."""
        # Starting y position for sliders
        first_slider_y = 0.02
        slider_spacing = 0.03

        # Create sliders with buttons
        self.slider_z, _, _ = self.create_slider_with_buttons('z', first_slider_y, 'Z', 82, 120, self.initial_z, 1)
        self.slider_n, _, _ = self.create_slider_with_buttons('n', first_slider_y + slider_spacing, 'N', 120, 180, self.initial_n, 1)
        self.slider_c, _, _ = self.create_slider_with_buttons('c', first_slider_y + 2 * slider_spacing, 'c', 0.5, 3.0, self.initial_c, 0.01)
        self.slider_q2, _, _ = self.create_slider_with_buttons('q2', first_slider_y + 3 * slider_spacing, 'q₂', -0.5, 1.0, self.initial_q2, 0.01)
        self.slider_a3, _, _ = self.create_slider_with_buttons('a3', first_slider_y + 4 * slider_spacing, 'a₃', -0.6, 0.6, self.initial_a3, 0.01)
        self.slider_a4, _, _ = self.create_slider_with_buttons('a4', first_slider_y + 5 * slider_spacing, 'a₄', -0.75, 0.75, self.initial_a4, 0.01)
        self.slider_a5, _, _ = self.create_slider_with_buttons('a5', first_slider_y + 6 * slider_spacing, 'a₅', -0.5, 0.5, self.initial_a5, 0.01)
        self.slider_a6, _, _ = self.create_slider_with_buttons('a6', first_slider_y + 7 * slider_spacing, 'a₆', -0.5, 0.5, self.initial_a6, 0.01)
        self.slider_max_beta, _, _ = self.create_slider_with_buttons('max_beta', first_slider_y + 8 * slider_spacing, 'Max Betas Used For Fit', 1.0, 36.0, 12.0, 1.0)
        self.slider_number_of_points, _, _ = self.create_slider_with_buttons('number_of_points', first_slider_y + 9 * slider_spacing, 'Number of Points', 180, 3600, 720, 180)

        # Style font sizes for all sliders
        for slider in [self.slider_z, self.slider_n, self.slider_c, self.slider_q2,
                       self.slider_a3, self.slider_a4, self.slider_a5, self.slider_a6, self.slider_max_beta, self.slider_number_of_points]:
            slider.label.set_fontsize(14)
            slider.valtext.set_fontsize(14)

        # Style increment/decrement buttons
        for slider_name, buttons in self.slider_buttons.items():
            buttons['dec_button'].label.set_fontsize(16)
            buttons['inc_button'].label.set_fontsize(16)

        # Create buttons
        ax_reset = plt.axes((0.82, 0.37, 0.08, 0.032))
        self.reset_button = Button(ax=ax_reset, label='Reset')

        ax_save = plt.axes((0.82, 0.32, 0.08, 0.032))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        ax_save_spherical = plt.axes((0.82, 0.27, 0.08, 0.032))
        self.save_coordinates_to_files_button = Button(ax=ax_save_spherical, label='Save Spherical Fit Coords')

        ax_print_beta = plt.axes((0.82, 0.22, 0.08, 0.032))
        self.print_beta_button = Button(ax=ax_print_beta, label='Print All Beta Params')

        # Create preset buttons
        preset_labels = ['Sphere', 'Prolate', 'Oblate', 'Pear-shaped', 'Two-center', 'Scission']
        for i, label in enumerate(preset_labels):
            ax_preset = plt.axes((0.02, 0.6 - i * 0.06, 0.08, 0.032))
            btn = Button(ax=ax_preset, label=label)
            self.preset_buttons.append(btn)

    def apply_preset(self, preset_num):
        """Apply a predefined configuration."""
        presets = {
            0: {'c': 1.0, 'a3': 0.0, 'a4': 0.0, 'a5': 0.0, 'a6': 0.0},  # Sphere
            1: {'c': 1.5, 'a3': 0.0, 'a4': 0.0, 'a5': 0.0, 'a6': 0.0},  # Prolate
            2: {'c': 0.7, 'a3': 0.0, 'a4': 0.0, 'a5': 0.0, 'a6': 0.0},  # Oblate
            3: {'c': 1.2, 'a3': 0.2, 'a4': 0.0, 'a5': 0.0, 'a6': 0.0},  # Pear-shaped
            4: {'c': 2.0, 'a3': 0.0, 'a4': 0.5, 'a5': 0.0, 'a6': 0.0},  # Two-center
            5: {'c': 2.08, 'a3': 0.25, 'a4': 0.72, 'a5': 0.0, 'a6': 0.0}  # Scission
        }

        preset = presets[preset_num]
        self.updating = True

        # Set the parameters by iterating through the preset dictionary
        for param_name, value in preset.items():
            # Construct the slider attribute name, e.g., 'slider_a3' from key 'a3'
            slider_attr_name = f"slider_{param_name}"
            # Safely get the slider attribute; returns None if it doesn't exist
            slider = getattr(self, slider_attr_name, None)
            if slider:  # This checks if the slider exists and is not None
                slider.set_val(value)

        # Set c and calculate q2
        q2_val = preset['c'] - 1.0 - 1.5 * preset['a4']
        if self.slider_q2:
            self.slider_q2.set_val(q2_val)

        self.updating = False
        self.update_plot(None)

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)
        self.slider_c.on_changed(self.on_c_changed)
        self.slider_q2.on_changed(self.on_q2_changed)
        self.slider_a3.on_changed(self.update_plot)
        self.slider_a4.on_changed(self.on_a4_changed)
        self.slider_a5.on_changed(self.update_plot)
        self.slider_a6.on_changed(self.update_plot)
        self.slider_max_beta.on_changed(self.update_plot)
        self.slider_number_of_points.on_changed(self.update_plot)

        # Connect button handlers
        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)
        self.save_coordinates_to_files_button.on_clicked(self.save_coordinates_to_files)
        self.print_beta_button.on_clicked(self.print_all_beta_parameters)

        # Connect preset buttons
        for i, btn in enumerate(self.preset_buttons):
            btn.on_clicked(lambda event, num=i: self.apply_preset(num))

        # Connect slider increment/decrement buttons
        self.setup_slider_buttons()

    def setup_slider_buttons(self):
        """Set up event handlers for slider increment/decrement buttons."""

        # Helper function to create increment/decrement handlers
        def create_increment_handler(slider_val):
            def handler(_):
                current_val = slider_val.val
                new_val = min(current_val + slider_val.valstep, slider_val.valmax)
                slider_val.set_val(new_val)

            return handler

        def create_decrement_handler(slider_val):
            def handler(_):
                current_val = slider_val.val
                new_val = max(current_val - slider_val.valstep, slider_val.valmin)
                slider_val.set_val(new_val)

            return handler

        # Connect buttons for each slider
        for slider_name, buttons in self.slider_buttons.items():
            slider = buttons['slider']
            dec_button = buttons['dec_button']
            inc_button = buttons['inc_button']

            dec_button.on_clicked(create_decrement_handler(slider))
            inc_button.on_clicked(create_increment_handler(slider))

    def on_c_changed(self, val):
        """Handle changes to c slider - update q2."""
        if not self.updating:
            self.updating = True
            # q2 = c - 1.0 - 1.5 * a4
            q2_val = val - 1.0 - 1.5 * self.slider_a4.val
            self.slider_q2.set_val(q2_val)
            self.updating = False
        self.update_plot(val)

    def on_q2_changed(self, val):
        """Handle changes to q2 slider - update c."""
        if not self.updating:
            self.updating = True
            # c = q2 + 1.0 + 1.5 * a4
            c_val = val + 1.0 + 1.5 * self.slider_a4.val
            self.slider_c.set_val(c_val)
            self.updating = False
        self.update_plot(val)

    def on_a4_changed(self, val):
        """Handle changes to a4 slider - update c based on q2."""
        if not self.updating:
            self.updating = True
            # c = q2 + 1.0 + 1.5 * a4
            c_val = self.slider_q2.val + 1.0 + 1.5 * val
            self.slider_c.set_val(c_val)
            self.updating = False
        self.update_plot(val)

    def reset_values(self, _):
        """Reset all sliders to their initial values."""
        self.updating = True
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.slider_q2.set_val(self.initial_q2)
        self.slider_a3.set_val(self.initial_a3)
        self.slider_a4.set_val(self.initial_a4)
        self.slider_a5.set_val(self.initial_a5)
        self.slider_a6.set_val(self.initial_a6)
        self.slider_c.set_val(self.initial_c)
        self.slider_max_beta.set_val(12)
        self.slider_number_of_points.set_val(720)
        self.updating = False
        self.update_plot(None)

    def save_plot(self, _):
        """Save the current plot to a file."""
        params = [
            float(self.slider_c.val),
            float(self.slider_q2.val),
            float(self.slider_a3.val),
            float(self.slider_a4.val),
            float(self.slider_a5.val),
            float(self.slider_a6.val),
            float(self.slider_number_of_points.val),
            float(self.slider_max_beta.val)
        ]

        filename = f"fos_shape_{int(self.slider_z.val)}_{int(self.slider_n.val)}_" + \
                   f"{'_'.join(f'{p:.3f}' for p in params)}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    def save_coordinates_to_files(self, _):
        """Save the spherical coordinates to a file in the format: r theta Phi
           and Cartesian coordinates in the format: x y z.
        """
        try:
            # Get current parameters
            current_params = FoSParameters(
                protons=int(self.slider_z.val),
                neutrons=int(self.slider_n.val),
                c_elongation=self.slider_c.val,
                q2=self.slider_q2.val,
                a3=self.slider_a3.val,
                a4=self.slider_a4.val,
                a5=self.slider_a5.val,
                a6=self.slider_a6.val
            )

            # Calculate shape
            calculator_fos = FoSShapeCalculator(current_params)
            current_number_of_points = int(self.slider_number_of_points.val)
            z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=current_number_of_points)

            # Prepare for conversion
            z_work = z_fos_cylindrical.copy()
            cumulative_shift = 0.0
            is_converted = False

            # Check if the shape can be unambiguously converted
            cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
            is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9)

            # If not convertible, try shifting
            if not is_convertible:
                z_step = 0.1  # fm
                shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0
                z_length = np.max(z_fos_cylindrical) - np.min(z_fos_cylindrical)
                max_shift = z_length / 2.0

                while abs(cumulative_shift) < max_shift:
                    cumulative_shift += shift_direction * z_step
                    z_work = z_fos_cylindrical + cumulative_shift

                    cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                    if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9):
                        is_convertible = True
                        is_converted = True
                        break

            if not is_convertible:
                print("Error: Shape cannot be converted to spherical coordinates")
                return

            # Convert to spherical coordinates
            theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=current_number_of_points)

            # Generate phi values from 0 to 2π with 360 steps
            phi_values = np.linspace(0, 2 * np.pi, 360)

            # Create filename
            params = [self.slider_c.val, self.slider_q2.val, self.slider_a3.val,
                      self.slider_a4.val, self.slider_a5.val, self.slider_a6.val]
            filename_spherical = f"spherical_coords_{int(self.slider_z.val)}_{int(self.slider_n.val)}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat"
            filename_cartesian = f"cartesian_coords_{int(self.slider_z.val)}_{int(self.slider_n.val)}_" + f"{'_'.join(f'{p:.3f}' for p in params)}.dat"

            # Save to files
            with open(filename_spherical, 'w') as f:
                # Write header for spherical coordinates
                f.write("# Spherical coordinates: r theta(radians) Phi (radians)\n")
                f.write(f"# Z={int(self.slider_z.val)}, N={int(self.slider_n.val)}\n")
                f.write(f"# Parameters: c={self.slider_c.val:.3f}, q2={self.slider_q2.val:.3f}, a3={self.slider_a3.val:.3f}, a4={self.slider_a4.val:.3f}, a5={self.slider_a5.val:.3f}, a6={self.slider_a6.val:.3f}\n")
                if is_converted:
                    f.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
                f.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
                f.write("# Format: r(fm) theta (radians) phi (radians)\n")

                for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                    for phi in phi_values:
                        f.write(f"{r:.6f} {theta:.6f} {phi:.6f}\n")

            with open(filename_cartesian, 'w') as f_cartesian:
                # Write header for Cartesian coordinates
                f_cartesian.write("# Cartesian coordinates: x y z\n")
                f_cartesian.write(f"# Z={int(self.slider_z.val)}, N={int(self.slider_n.val)}\n")
                f_cartesian.write(f"# Parameters: c={self.slider_c.val:.3f}, q2={self.slider_q2.val:.3f}, a3={self.slider_a3.val:.3f}, a4={self.slider_a4.val:.3f}, a5={self.slider_a5.val:.3f}, a6={self.slider_a6.val:.3f}\n")
                if is_converted:
                    f_cartesian.write(f"# Shape was shifted by {cumulative_shift:.3f} fm for conversion\n")
                f_cartesian.write(f"# Total points: {len(theta_fos) * len(phi_values)}\n")
                f_cartesian.write("# Format: x(fm) y(fm) z(fm)\n")

                for i, (r, theta) in enumerate(zip(radius_fos, theta_fos)):
                    for phi in phi_values:
                        # Convert theta and phi to Cartesian coordinates
                        x = r * np.sin(theta) * np.cos(phi)
                        y = r * np.sin(theta) * np.sin(phi)
                        z = r * np.cos(theta)
                        f_cartesian.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

            print(f"Spherical coordinates saved as {filename_spherical}")
            print(f"Cartesian coordinates saved as {filename_cartesian}")
            print(f"Total points saved: {len(theta_fos) * len(phi_values)}")
            if is_converted:
                print(f"Note: Shape was shifted by {cumulative_shift:.3f} fm for conversion")
        except Exception as e:
            print(f"Error saving spherical coordinates: {e}")

    def print_all_beta_parameters(self, _):
        """Print all analytical and fitted beta parameters (including ones below 0.00095)."""
        try:
            # Get current parameters
            current_params = FoSParameters(
                protons=int(self.slider_z.val),
                neutrons=int(self.slider_n.val),
                c_elongation=self.slider_c.val,
                q2=self.slider_q2.val,
                a3=self.slider_a3.val,
                a4=self.slider_a4.val,
                a5=self.slider_a5.val,
                a6=self.slider_a6.val
            )

            # Calculate shape
            calculator_fos = FoSShapeCalculator(current_params)
            current_number_of_points = int(self.slider_number_of_points.val)
            z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=current_number_of_points)

            # Prepare for conversion
            z_work = z_fos_cylindrical.copy()
            cumulative_shift = 0.0
            is_converted = False

            # Check if the shape can be unambiguously converted
            cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
            is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9)

            # If not convertible, try shifting
            if not is_convertible:
                z_step = 0.1  # fm
                shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0
                z_length = np.max(z_fos_cylindrical) - np.min(z_fos_cylindrical)
                max_shift = z_length / 2.0

                while abs(cumulative_shift) < max_shift:
                    cumulative_shift += shift_direction * z_step
                    z_work = z_fos_cylindrical + cumulative_shift

                    cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                    if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9):
                        is_convertible = True
                        is_converted = True
                        break

            if not is_convertible:
                print("Error: Shape cannot be converted to spherical coordinates")
                return

            # Convert to spherical coordinates
            theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=current_number_of_points)

            # Calculate beta parameters using an analytical method
            spherical_to_beta_converter = BetaDeformationCalculator(theta=theta_fos, radius=radius_fos, number_of_nucleons=current_params.nucleons)
            l_max_value = int(self.slider_max_beta.val)
            beta_parameters = spherical_to_beta_converter.calculate_beta_parameters(l_max=l_max_value)

            # Calculate beta parameters using the RMSE fitting method
            try:
                fitting_results: FitResult = spherical_to_beta_converter.fit_beta_parameters_rmse(l_max=l_max_value)
                beta_parameters_fitted: Dict[int, float] = fitting_results['beta_fitted']
                scaling_factor_fitted = fitting_results['scaling_factor_fitted']
                rmse_fitting = fitting_results['rmse']
            except Exception as fit_error:
                print(f"Beta fitting error: {fit_error}")
                # Use analytical parameters as fallback
                beta_parameters_fitted = beta_parameters.copy()
                scaling_factor_fitted = 1.0
                rmse_fitting = -1.0  # Indicate fitting failed

            # Print header
            print(f"\n{'='*80}")
            print(f"ALL BETA PARAMETERS FOR Z={int(self.slider_z.val)}, N={int(self.slider_n.val)}")
            print(f"Parameters: c={self.slider_c.val:.3f}, q2={self.slider_q2.val:.3f}, a3={self.slider_a3.val:.3f}")
            print(f"           a4={self.slider_a4.val:.3f}, a5={self.slider_a5.val:.3f}, a6={self.slider_a6.val:.3f}")
            print(f"Max Beta Used: {l_max_value}")
            print(f"{'='*80}")

            # Print analytical beta parameters in a single row format
            print(f"\nANALYTICAL METHOD:")
            analytical_params = [f"β_{l}" for l in sorted(beta_parameters.keys())]
            analytical_values = [f"{beta_parameters[l]:+.3f}" for l in sorted(beta_parameters.keys())]
            print(f"Parameters: {', '.join(analytical_params)}")
            print(f"Values:     {', '.join(analytical_values)}")

            # Print fitted beta parameters in a single row format
            print(f"\nFITTED METHOD (RMSE minimization):")
            print(f"Scaling Factor: {scaling_factor_fitted:.6f}, RMSE: {rmse_fitting:.6f} fm")
            fitted_params = [f"β_{l}" for l in sorted(beta_parameters_fitted.keys())]
            fitted_values = [f"{beta_parameters_fitted[l]:+.3f}" for l in sorted(beta_parameters_fitted.keys())]
            print(f"Parameters: {', '.join(fitted_params)}")
            print(f"Values:     {', '.join(fitted_values)}")

            print(f"\n{'='*80}")

        except Exception as e:
            print(f"Error printing beta parameters: {e}")

    def update_plot(self, _):
        """Update the plot with new parameters."""
        # Get current parameters
        current_params = FoSParameters(
            protons=int(self.slider_z.val),
            neutrons=int(self.slider_n.val),
            c_elongation=self.slider_c.val,
            q2=self.slider_q2.val,
            a3=self.slider_a3.val,
            a4=self.slider_a4.val,
            a5=self.slider_a5.val,
            a6=self.slider_a6.val
        )

        # Calculate shape
        calculator_fos = FoSShapeCalculator(current_params)

        # Calculate shape with theoretical shift
        current_number_of_points = int(self.slider_number_of_points.val)
        z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=current_number_of_points)

        # Update FoS shape lines
        self.line_fos.set_data(z_fos_cylindrical, rho_fos_cylindrical)
        self.line_fos_mirror.set_data(z_fos_cylindrical, -rho_fos_cylindrical)

        # Update beta shape lines and spherical FoS shape lines
        center_of_mass_beta_fitted: float = 0.0
        fos_spherical_volume: float = 0.0
        beta_volume: float = 0.0
        beta_volume_fitted: float = 0.0
        fos_spherical_surface_area: float = 0.0
        beta_surface_area: float = 0.0
        beta_surface_area_fitted: float = 0.0
        radius_fos: np.ndarray = np.array([])
        theta_fos: np.ndarray = np.array([])
        radius_beta: np.ndarray = np.array([])
        theta_beta: np.ndarray = np.array([])
        conversion_root_mean_squared_error: float = 0.0
        rmse_beta_fit: float = 0.0
        rmse_fitting: float = 0.0
        scaling_factor_fitted: float = 0.0
        scaling_factor_volume: float = 0.0
        is_converted: bool = False
        cumulative_shift: float = 0.0

        try:
            # Make a copy of the original shape for shifting if needed
            z_work = z_fos_cylindrical.copy()

            # First, check if the shape can be unambiguously converted
            cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
            is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9)

            # If not convertible, try shifting
            if not is_convertible:
                # Determine a shift direction (opposite of z_sh)
                z_step = 0.1  # fm
                shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0

                # The maximum allowed shift is half the z-dimension length
                z_length = np.max(z_fos_cylindrical) - np.min(z_fos_cylindrical)
                max_shift = z_length / 2.0

                # Try shifting until convertible or reach the limit
                while abs(cumulative_shift) < max_shift:
                    cumulative_shift += shift_direction * z_step
                    z_work = z_fos_cylindrical + cumulative_shift

                    # Check if now convertible
                    cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                    if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9):
                        is_convertible = True
                        is_converted = True
                        break

            # If the shape is now convertible (either originally or after shifting), proceed with conversion
            if is_convertible:

                theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=current_number_of_points)
                z_fos_spherical, rho_fos_spherical = cylindrical_to_spherical_converter.convert_to_cartesian(n_theta=current_number_of_points)

                # Validate the conversion
                validation = cylindrical_to_spherical_converter.validate_conversion(n_samples=current_number_of_points)
                conversion_root_mean_squared_error = validation['root_mean_squared_error']

                # Update spherical FoS shape lines (shift back for plotting if needed)
                if abs(cumulative_shift) > 1e-10:
                    self.line_fos_spherical.set_data(rho_fos_spherical - cumulative_shift, z_fos_spherical)
                    self.line_fos_spherical_mirror.set_data(rho_fos_spherical - cumulative_shift, -z_fos_spherical)
                    self.line_fos_spherical_shifted_for_conversion.set_data(rho_fos_spherical, z_fos_spherical)
                    self.line_fos_spherical_shifted_for_conversion_mirror.set_data(rho_fos_spherical, -z_fos_spherical)
                else:
                    self.line_fos_spherical.set_data(rho_fos_spherical, z_fos_spherical)
                    self.line_fos_spherical_mirror.set_data(rho_fos_spherical, -z_fos_spherical)
                    self.line_fos_spherical_shifted_for_conversion.set_data([], [])
                    self.line_fos_spherical_shifted_for_conversion_mirror.set_data([], [])

                # Calculate beta parameters using an analytical method
                spherical_to_beta_converter = BetaDeformationCalculator(theta=theta_fos, radius=radius_fos, number_of_nucleons=current_params.nucleons)
                l_max_value = int(self.slider_max_beta.val)
                beta_parameters = spherical_to_beta_converter.calculate_beta_parameters(l_max=l_max_value)

                # Calculate beta parameters using the RMSE fitting method
                try:
                    fitting_results: FitResult = spherical_to_beta_converter.fit_beta_parameters_rmse(l_max=l_max_value)
                    beta_parameters_fitted: Dict[int, float] = fitting_results['beta_fitted']
                    scaling_factor_fitted = fitting_results['scaling_factor_fitted']
                    scaling_factor_volume = fitting_results['scaling_factor_volume']
                    rmse_fitting = fitting_results['rmse']
                except Exception as fit_error:
                    print(f"Beta fitting error: {fit_error}")
                    # Use analytical parameters as fallback
                    beta_parameters_fitted = beta_parameters.copy()
                    scaling_factor_fitted = 1.0
                    scaling_factor_volume = 1.0
                    rmse_fitting = -1.0  # Indicate fitting failed

                # Calculate the beta shape coordinates (analytical)
                theta_beta, radius_beta = spherical_to_beta_converter.reconstruct_shape(beta=beta_parameters, n_theta=current_number_of_points)

                # Convert back to Cartesian coordinates (analytical)
                z_beta = radius_beta * np.cos(theta_beta)
                rho_beta = radius_beta * np.sin(theta_beta)

                # Calculate the beta shape coordinates (fitted)
                theta_beta_fitted, radius_beta_fitted = spherical_to_beta_converter.reconstruct_shape(beta=beta_parameters_fitted, n_theta=current_number_of_points)

                # Convert back to Cartesian coordinates (fitted)
                z_beta_fitted = radius_beta_fitted * np.cos(theta_beta_fitted)
                rho_beta_fitted = radius_beta_fitted * np.sin(theta_beta_fitted)

                # Shift back the beta shapes if we had shifted for conversion
                if abs(cumulative_shift) > 1e-10:
                    z_beta = z_beta - cumulative_shift
                    z_beta_fitted = z_beta_fitted - cumulative_shift

                # Update beta shape lines (analytical)
                self.line_beta.set_data(z_beta, rho_beta)
                self.line_beta_mirror.set_data(z_beta, -rho_beta)

                # Update beta shape lines (fitted)
                self.line_beta_fitted.set_data(z_beta_fitted, rho_beta_fitted)
                self.line_beta_fitted_mirror.set_data(z_beta_fitted, -rho_beta_fitted)

                # Calculate the volume for the spherical fit and beta fits
                fos_spherical_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
                beta_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
                beta_volume_fitted = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_beta_fitted, theta=theta_beta_fitted)

                # Calculate the surface area for the spherical fit and beta fits
                fos_spherical_surface_area = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
                beta_surface_area = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
                beta_surface_area_fitted = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_beta_fitted, theta=theta_beta_fitted)

                # Calculate RMSE for the beta fit (analytical)
                rmse_beta_fit = np.sqrt(np.mean((radius_beta - radius_fos) ** 2))

                # Get significant beta parameters (analytical)
                beta_strings_analytical = [f"β_{l:<2} = {val:+.4f}" for l, val in sorted(beta_parameters.items()) if abs(val) >= 0.00095]

                # Get significant beta parameters (fitted)
                beta_strings_fitted = [f"β_{l:<2} = {val:+.4f}" for l, val in sorted(beta_parameters_fitted.items()) if abs(val) >= 0.00095]

                # Format analytical parameters
                if not beta_strings_analytical:
                    analytical_params_text = "No significant parameters found"
                else:
                    paired_analytical = []
                    for i in range(0, len(beta_strings_analytical), 2):
                        col1 = beta_strings_analytical[i]
                        if i + 1 < len(beta_strings_analytical):
                            col2 = beta_strings_analytical[i + 1]
                            paired_analytical.append(f"{col1:<20} {col2}")
                        else:
                            paired_analytical.append(col1)
                    analytical_params_text = "\n".join(paired_analytical)

                # Format fitted parameters
                if not beta_strings_fitted:
                    fitted_params_text = "No significant parameters found"
                else:
                    paired_fitted = []
                    for i in range(0, len(beta_strings_fitted), 2):
                        col1 = beta_strings_fitted[i]
                        if i + 1 < len(beta_strings_fitted):
                            col2 = beta_strings_fitted[i + 1]
                            paired_fitted.append(f"{col1:<20} {col2}")
                        else:
                            paired_fitted.append(col1)
                    fitted_params_text = "\n".join(paired_fitted)

                significant_beta_parameters = (
                        "ANALYTICAL METHOD:\n" + analytical_params_text + "\n\n" +
                        "FITTED METHOD (RMSE minimization):\n" + fitted_params_text
                )
            else:
                # Could not make shape convertible
                self.line_fos_spherical.set_data([], [])
                self.line_fos_spherical_mirror.set_data([], [])
                self.line_beta.set_data([], [])
                self.line_beta_mirror.set_data([], [])
                self.line_beta_fitted.set_data([], [])
                self.line_beta_fitted_mirror.set_data([], [])
                significant_beta_parameters = "Shape could not be made convertible"

        except Exception as e:
            print(f"Conversion error: {e}")
            self.line_fos_spherical.set_data([], [])
            self.line_fos_spherical_mirror.set_data([], [])
            self.line_beta.set_data([], [])
            self.line_beta_mirror.set_data([], [])
            self.line_beta_fitted.set_data([], [])
            self.line_beta_fitted_mirror.set_data([], [])
            significant_beta_parameters = "Calculation Error"

        # Update center of mass points
        self.cm_theoretical.set_data([current_params.z_sh], [0])
        center_of_mass_fos = calculator_fos.calculate_center_of_mass_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)
        self.cm_fos_calculated.set_data([center_of_mass_fos], [0])
        center_of_mass_spherical_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
        self.cm_spherical_fit_calculated.set_data([center_of_mass_spherical_fit - cumulative_shift], [0])
        self.cm_spherical_fit_calculated_shifted.set_data([center_of_mass_spherical_fit], [0])
        center_of_mass_beta_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
        self.cm_beta_fit_calculated.set_data([center_of_mass_beta_fit - cumulative_shift], [0])

        # Calculate the ratio of calculated CM to theoretical shift, if NaN, set to 0
        # cm_ratio = (
        #     calculator_fos.calculate_center_of_mass(z_fos_cylindrical, rho_fos_cylindrical) / current_params.z_sh
        #     if current_params.z_sh != 0 else 0.0
        # )

        # Update a reference sphere
        theta_reference_sphere = np.linspace(0, 2 * np.pi, 180)
        sphere_x = current_params.radius0 * np.cos(theta_reference_sphere)
        sphere_y = current_params.radius0 * np.sin(theta_reference_sphere)
        self.reference_sphere_line.set_data(sphere_x, sphere_y)
        self.reference_sphere_line.set_label(f'R₀={current_params.radius0:.2f} fm')

        # Update plot limits
        max_val = max(np.max(np.abs(z_fos_cylindrical)), np.max(np.abs(rho_fos_cylindrical))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Calculate the volume of the plotted shape for verification
        fos_shape_volume = calculator_fos.calculate_volume_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)

        # Calculate the surface area of the plotted shape
        fos_cylindrical_surface_area = calculator_fos.calculate_surface_area_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)

        # Calculate dimensions
        max_z = np.max(np.abs(z_fos_cylindrical))
        max_rho = np.max(rho_fos_cylindrical)

        # Calculate neck radius at a4 = 0.72 (scission point)
        if current_params.a4 > 0:
            neck_radius = min(rho_fos_cylindrical[len(rho_fos_cylindrical) // 2 - 10:len(rho_fos_cylindrical) // 2 + 10])
        else:
            neck_radius = max_rho

        # Add information text
        info_text = (
            f"Number of Shape Points: {current_number_of_points}\n"
            f"R₀ = {current_params.radius0:.3f} fm\n"
            f"\nParameter Relations:\n"
            f"a₂ (From Volume Conservation)= a₄/3 - a₆/5\n"
            f"a₂ = {current_params.a4:.3f} / 3.0 - {current_params.a6:.3f} / 5.0 = {current_params.a2:.3f}\n"
            f"c (Elongation) = q₂ + 1.0 + 1.5a₄\n"
            f"c = {current_params.q2:.3f} + 1.0 + 1.5 × {current_params.a4:.3f} = {current_params.c_elongation:.3f}\n"
            f"Aₕ = A/2 * (1.0 + a₃)\n"
            f"Aₕ = {current_params.nucleons:.3f} / 2 * (1.0 + {current_params.a3:.3f}) = {current_params.nucleons / 2.0 * (1.0 + current_params.a3):.3f}\n"
            f"\nShift Information:\n"
            f"Theoretical z_shift = {current_params.z_sh:.3f} fm\n"
            f"Shift Needed for Conversion: {cumulative_shift:.3f} fm\n"
            f"\nCenter of Mass:\n"
            f"Calculated CM (FoS Cylindrical): {center_of_mass_fos:.3f} fm\n"
            f"Calculated CM (FoS Spherical Fit): {center_of_mass_spherical_fit - cumulative_shift:.3f} fm\n"
            f"Calculated CM (Beta Analytical): {center_of_mass_beta_fit - cumulative_shift:.3f} fm\n"
            f"Calculated CM (Beta Fitted): {center_of_mass_beta_fitted - cumulative_shift:.3f} fm\n"
            f"Calculated CM (FoS Spherical Fit, Shifted): {center_of_mass_spherical_fit:.3f} fm\n"
            f"\nVolume Information:\n"
            f"Reference Sphere Volume: {current_params.sphere_volume:.3f} fm³\n"
            f"FoS (Cylindrical) Shape Volume: {fos_shape_volume:.3f} fm³\n"
            f"FoS (Spherical) Shape Volume: {fos_spherical_volume:.3f} fm³\n"
            f"Beta Shape Volume (Analytical): {beta_volume:.3f} fm³\n"
            f"Beta Shape Volume (Fitted): {beta_volume_fitted:.3f} fm³\n"
            f"\nSurface Information:\n"
            f"Reference Sphere Surface Area: {current_params.sphere_surface_area:.3f} fm²\n"
            f"FoS (Cylindrical) Shape Surface Area: {fos_cylindrical_surface_area:.3f} fm²\n"
            f"FoS (Spherical) Shape Surface Area: {fos_spherical_surface_area:.3f} fm²\n"
            f"Beta Shape Surface Area (Analytical): {beta_surface_area:.3f} fm² ({beta_surface_area / fos_cylindrical_surface_area * 100:.3f}% of FoS)\n"
            f"Beta Shape Surface Area (Fitted): {beta_surface_area_fitted:.3f} fm² ({beta_surface_area_fitted / fos_cylindrical_surface_area * 100:.3f}% of FoS)\n"
            f"\nShape Dimensions:\n"
            f"Max z: {max_z:.2f} fm\n"
            f"Max ρ: {max_rho:.2f} fm\n"
            f"Length along z: {abs(np.max(z_fos_cylindrical) - np.min(z_fos_cylindrical)):.2f} fm\n"
            f"Neck Radius: {neck_radius:.2f} fm\n"
            f"Calculated c (Elongation): {max_z / current_params.radius0:.3f}\n"
            f"\nConversion Information:\n"
            f"Shape is Originally Unambiguously Convertible: {'Yes' if not is_converted else 'No'}\n"
            f"\nFit Information:\n"
            f"RMSE (Spherical Coords Conversion): {conversion_root_mean_squared_error:.3f} fm\n"
            f"Absolute RMSE (Beta Analytical Method): {rmse_beta_fit:.3f} fm\n"
            f"Relative (RMSE/RMS(r)) RMSE (Beta Analytical Method): {rmse_beta_fit / np.sqrt(np.mean(radius_fos ** 2)) * 100:.3f} %\n"
            f"Absolute RMSE (Beta Fitting Method): {rmse_fitting:.3f} fm\n"
            f"Relative (RMSE/RMS(r)) RMSE (Beta Fitting Method): {rmse_fitting / np.sqrt(np.mean(radius_fos ** 2)) * 100:.3f} %\n"
            f"Radius Fixing Factor (Calculated): {scaling_factor_fitted:.6f}\n"
            f"Radius Fixing Factor Difference: {abs(scaling_factor_fitted - scaling_factor_volume):.6f}"
        )

        # Remove old text if it exists
        for artist in self.ax_text.texts:
            artist.remove()

        # Add new text to the left side of the right text area
        self.ax_text.text(-0.05, 1.0, info_text, transform=self.ax_text.transAxes,
                          fontsize=10, verticalalignment='top', horizontalalignment='left',
                          bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}, fontfamily='monospace')

        # Add beta parameters text to the right side of the right text area
        beta_text_obj = self.ax_text.text(0.55, 1.0, f"Significant Beta Parameters (>0.001):\n{significant_beta_parameters}",
                                          transform=self.ax_text.transAxes,
                                          fontsize=10, verticalalignment='top', horizontalalignment='left',
                                          bbox={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.5}, fontfamily='monospace')
        beta_text_obj._beta_text = True  # Mark this as beta text for removal

        # Update title
        self.ax_plot.set_title(
            f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, '
            f'A={current_params.nucleons})', fontsize=14
        )

        # Update legend
        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(-0.05, 1))

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot(None)
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    parser = create_parser()
    args = parser.parse_args()

    # Check if all required arguments are provided for batch mode
    if all(v is not None for v in [args.Z, args.N, args.q2, args.a3,
                                   args.a4, args.a5, args.a6, args.number_of_points]):
        # Batch mode
        errors = validate_parameters(args.Z, args.N, args.q2, args.a3,
                                     args.a4, args.a5, args.a6, args.number_of_points)

        if errors:
            print("Error: Invalid parameters:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

        # Create an output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir)
            except OSError as e:
                print(f"Error: Could not create output directory: {e}", file=sys.stderr)
                sys.exit(1)

        # Run calculation
        success = calculate_and_save_shape(
            args.Z, args.N, args.q2, args.a3, args.a4, args.a5, args.a6,
            args.number_of_points, args.output_dir
        )

        if not success:
            print("Error: Failed to calculate or save shape", file=sys.stderr)
            sys.exit(1)

        # Exit successfully (silent)
        sys.exit(0)

    elif any(v is not None for v in [args.Z, args.N, args.q2, args.a3,
                                     args.a4, args.a5, args.a6, args.number_of_points]):
        # Partial arguments provided - error
        print("Error: All parameters must be provided for batch mode", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    else:
        # No arguments - run GUI mode
        plotter = FoSShapePlotter()
        plotter.run()


if __name__ == '__main__':
    main()