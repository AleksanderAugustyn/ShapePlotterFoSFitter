"""
Nuclear Shape Plotter using Fourier-over-Spheroid (FoS) Parametrization
Based on the formulation in Pomorski et al. (2023)
With volume normalization to ensure volume conservation
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from BetaDeformationCalculator import BetaDeformationCalculator
from CylindricalToSphericalConverter import CylindricalToSphericalConverter

matplotlib.use('TkAgg')


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int
    neutrons: int
    c_elongation: float = 1.0  # elongation
    a3: float = 0.0  # reflection asymmetry
    a4: float = 0.0  # neck parameter
    a5: float = 0.0  # higher order parameter
    a6: float = 0.0  # higher order parameter
    q2: float = 0.0  # entangled parameter: c = q2 + 1.0 + 1.5 * a4
    r0_constant: float = 1.16  # Radius constant in fm

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
        f = 1.0 - u ** 2.0

        # Sum Fourier terms
        sum_terms = 0.0

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
        z_min = -self.params.z0 + self.params.z_sh
        z_max = self.params.z0 + self.params.z_sh
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
    def calculate_volume_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate volume by numerical integration.
        V = π ∫ ρ²(z) dz

        Args:
            z: axial coordinates
            rho: radial coordinates

        Returns:
            volume: calculated volume of the shape in fm³
        """
        # Use trapezoidal rule
        dz = np.diff(z)
        rho_mid = (rho[1:] + rho[:-1]) / 2
        volume = np.pi * np.sum(rho_mid ** 2 * dz)
        return volume

    @staticmethod
    def calculate_center_of_mass(z: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate the center of mass position along the z-axis.
        z_cm = ∫ z ρ²(z) dz / ∫ ρ²(z) dz
        """
        if len(z) < 2:
            return 0.0

        # Use trapezoidal rule for both integrals
        dz = np.diff(z)
        z_mid = (z[1:] + z[:-1]) / 2
        rho_mid = (rho[1:] + rho[:-1]) / 2

        # Numerator: ∫ z ρ²(z) dz
        numerator = np.sum(z_mid * rho_mid ** 2 * dz)

        # Denominator: ∫ ρ²(z) dz
        denominator = np.sum(rho_mid ** 2 * dz)

        if denominator == 0:
            return 0.0

        return numerator / denominator


class FoSShapePlotter:
    """Class for plotting Fourier-over-Spheroid nuclear shapes."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Default parameters
        self.initial_z = 92  # Uranium
        self.initial_n = 144
        self.initial_q2 = 0.0
        self.initial_a3 = 0.0
        self.initial_a4 = 0.0
        self.initial_a5 = 0.0
        self.initial_a6 = 0.0

        # Calculate initial c from q2 and a4
        self.initial_c = self.initial_q2 + 1.0 + 1.5 * self.initial_a4

        # UI elements
        self.fig = None
        self.ax_plot = None
        self.ax_text = None
        self.line_fos = None
        self.line_fos_mirror = None
        self.line_fos_spherical = None
        self.line_fos_spherical_mirror = None
        self.line_beta = None
        self.line_beta_mirror = None
        self.reference_sphere_line = None
        self.cm_reference_point = None
        self.cm_fos_calculated = None
        self.cm_theoretical = None
        self.cm_spherical_fit_calculated = None
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

        # Buttons
        self.reset_button = None
        self.save_button = None
        self.preset_buttons = []

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
        self.fig = plt.figure(figsize=(16, 8))

        # Create the main plot on the left side
        self.ax_plot = self.fig.add_subplot(121)

        # Create a text area on the right side
        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.set_xlim(0, 1)
        self.ax_text.set_ylim(0, 1)
        self.ax_text.axis('off')  # Hide axes for the text area

        plt.subplots_adjust(left=0.08, bottom=0.45, right=0.95, top=0.9, wspace=0.1)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_title('Nuclear Shape (Fourier-over-Spheroid Parametrization)', fontsize=14)
        self.ax_plot.set_xlabel('z (fm)', fontsize=12)
        self.ax_plot.set_ylabel('ρ (fm)', fontsize=12)

        # Initialize the shape plot
        calculator = FoSShapeCalculator(self.nuclear_params)
        z, rho = calculator.calculate_shape()

        # Create a reference sphere
        r0 = self.nuclear_params.radius0
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = r0 * np.cos(theta)
        sphere_y = r0 * np.sin(theta)

        # Plot the FoS shape and its mirror
        self.line_fos, = self.ax_plot.plot(z, rho, 'b-', label='FoS shape (normalized)', linewidth=2)
        self.line_fos_mirror, = self.ax_plot.plot(z, -rho, 'b-', linewidth=2)

        # Plot the FoS shape in spherical coordinates
        self.line_fos_spherical, = self.ax_plot.plot([], [], 'y--', label='FoS shape (spherical)', linewidth=1.5)
        self.line_fos_spherical_mirror, = self.ax_plot.plot([], [], 'y--', linewidth=1.5)

        # Plot the beta shape and its mirror
        self.line_beta, = self.ax_plot.plot([], [], 'r--', label='Beta shape (normalized)', linewidth=2, alpha=0.7)
        self.line_beta_mirror, = self.ax_plot.plot([], [], 'r--', linewidth=2, alpha=0.7)

        self.reference_sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray', alpha=0.5, label=f'R₀={r0:.2f} fm')

        # Plot expected center of mass (at origin)
        self.cm_reference_point, = self.ax_plot.plot(0, 0, 'ro', label='Expected CM (Origin)', markersize=8, alpha=0.7)

        # Plot theoretical shift CM
        self.cm_theoretical, = self.ax_plot.plot(0, 0, 'b^', label='CM before theoretical shift', markersize=8, alpha=0.7)

        # Plot the calculated center of mass
        self.cm_fos_calculated, = self.ax_plot.plot(0, 0, 'go', label='Calculated CM (FoS)', markersize=8, alpha=0.7)
        self.cm_spherical_fit_calculated, = self.ax_plot.plot(0, 0, 'ms', label='Calculated CM (Spherical Fit)', markersize=8, alpha=0.7)
        self.cm_beta_fit_calculated, = self.ax_plot.plot(0, 0, 'cs', label='Calculated CM (Beta Fit)', markersize=8, alpha=0.7)

        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(0, 1))

    def setup_controls(self):
        """Set up all UI controls."""
        # Starting y position for sliders
        first_slider_y = 0.02
        slider_spacing = 0.04

        # Create proton (Z) slider
        ax_z = plt.axes((0.25, first_slider_y, 0.5, 0.03))
        self.slider_z = Slider(ax=ax_z, label='Z', valmin=20, valmax=120, valinit=self.initial_z, valstep=1)

        # Create neutron (N) slider
        ax_n = plt.axes((0.25, first_slider_y + slider_spacing, 0.5, 0.03))
        self.slider_n = Slider(ax=ax_n, label='N', valmin=20, valmax=180, valinit=self.initial_n, valstep=1)

        # Create an elongation (c) slider
        ax_c = plt.axes((0.25, first_slider_y + 2 * slider_spacing, 0.5, 0.03))
        self.slider_c = Slider(ax=ax_c, label='c', valmin=0.5, valmax=3.0, valinit=self.initial_c, valstep=0.01)

        # Create q2 slider (entangled with c and a4)
        ax_q2 = plt.axes((0.25, first_slider_y + 3 * slider_spacing, 0.5, 0.03))
        self.slider_q2 = Slider(ax=ax_q2, label='q₂', valmin=-1.0, valmax=2.0, valinit=self.initial_q2, valstep=0.01)

        # Create a3 slider (reflection asymmetry)
        ax_a3 = plt.axes((0.25, first_slider_y + 4 * slider_spacing, 0.5, 0.03))
        self.slider_a3 = Slider(ax=ax_a3, label='a₃', valmin=-0.5, valmax=0.5, valinit=self.initial_a3, valstep=0.01)

        # Create a4 slider (neck parameter)
        ax_a4 = plt.axes((0.25, first_slider_y + 5 * slider_spacing, 0.5, 0.03))
        self.slider_a4 = Slider(ax=ax_a4, label='a₄', valmin=-0.5, valmax=0.75, valinit=self.initial_a4, valstep=0.01)

        # Create a5 slider
        ax_a5 = plt.axes((0.25, first_slider_y + 6 * slider_spacing, 0.5, 0.03))
        self.slider_a5 = Slider(ax=ax_a5, label='a₅', valmin=-0.3, valmax=0.3, valinit=self.initial_a5, valstep=0.01)

        # Create a6 slider
        ax_a6 = plt.axes((0.25, first_slider_y + 7 * slider_spacing, 0.5, 0.03))
        self.slider_a6 = Slider(ax=ax_a6, label='a₆', valmin=-0.3, valmax=0.3, valinit=self.initial_a6, valstep=0.01)

        # Style font sizes for all sliders
        for slider in [self.slider_z, self.slider_n, self.slider_c, self.slider_q2,
                       self.slider_a3, self.slider_a4, self.slider_a5, self.slider_a6]:
            slider.label.set_fontsize(12)
            slider.valtext.set_fontsize(12)

        # Create buttons
        ax_reset = plt.axes((0.8, 0.37, 0.1, 0.04))
        self.reset_button = Button(ax=ax_reset, label='Reset')

        ax_save = plt.axes((0.8, 0.32, 0.1, 0.04))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        # Create preset buttons
        preset_labels = ['Sphere', 'Prolate', 'Oblate', 'Pear-shaped', 'Two-center']
        for i, label in enumerate(preset_labels):
            ax_preset = plt.axes((0.02, 0.6 - i * 0.06, 0.1, 0.04))
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
        }

        preset = presets[preset_num]
        self.updating = True

        # Set the parameters
        self.slider_a3.set_val(preset['a3'])
        self.slider_a4.set_val(preset['a4'])
        self.slider_a5.set_val(preset['a5'])
        self.slider_a6.set_val(preset['a6'])

        # Set c and calculate q2
        self.slider_c.set_val(preset['c'])
        q2_val = preset['c'] - 1.0 - 1.5 * preset['a4']
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

        # Connect button handlers
        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)

        # Connect preset buttons
        for i, btn in enumerate(self.preset_buttons):
            btn.on_clicked(lambda event, num=i: self.apply_preset(num))

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
        self.updating = False
        self.update_plot(None)

    def save_plot(self, _):
        """Save the current plot to a file."""
        params = [self.slider_c.val, self.slider_q2.val, self.slider_a3.val,
                  self.slider_a4.val, self.slider_a5.val, self.slider_a6.val]
        filename = f"fos_shape_{int(self.slider_z.val)}_{int(self.slider_n.val)}_" + \
                   f"{'_'.join(f'{p:.3f}' for p in params)}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

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
        calculator = FoSShapeCalculator(current_params)

        # Calculate shape with theoretical shift
        z_fos, rho_fos = calculator.calculate_shape()

        # Update FoS shape lines
        self.line_fos.set_data(z_fos, rho_fos)
        self.line_fos_mirror.set_data(z_fos, -rho_fos)

        # Update beta shape lines and spherical FoS shape lines
        beta_volume: float = 0.0
        radius_fos: np.ndarray = np.array([])
        theta_fos: np.ndarray = np.array([])
        radius_beta: np.ndarray = np.array([])
        theta_beta: np.ndarray = np.array([])
        try:
            # First, convert z_fos, rho_fos to spherical coordinates
            cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_fos, rho_points=rho_fos)
            theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=720)
            y_fos, x_fos = cylindrical_to_spherical_converter.convert_to_cartesian(n_theta=720)

            # Validate the conversion
            validation = cylindrical_to_spherical_converter.validate_conversion(n_samples=360)
            conversion_error = validation['mean_error'] * 100  # Convert to percentage
            if conversion_error > 5.0:
                print(f"Warning: Conversion error is high ({conversion_error:.2f}%)")

            # Update spherical FoS shape lines
            # For plotting, we need to flip x and y since our original is (z, rho)
            # In the converter, x = r sin(θ) and y = r cos(θ)
            # But we want z on the horizontal axis and rho on vertical
            self.line_fos_spherical.set_data(x_fos, y_fos)
            self.line_fos_spherical_mirror.set_data(x_fos, -y_fos)

            # Calculate beta parameters
            spherical_to_beta_converter = BetaDeformationCalculator(theta=theta_fos, radius=radius_fos, number_of_nucleons=current_params.nucleons)
            beta_parameters = spherical_to_beta_converter.calculate_beta_parameters(l_max=32)

            # Calculate the beta shape coordinates
            theta_beta, radius_beta = spherical_to_beta_converter.reconstruct_shape(beta_parameters)

            # Convert back to Cartesian coordinates
            # For plotting, we need to flip x and y since our original is (z, rho)
            # In the converter, x = r sin(θ) and y = r cos(θ)
            # But we want z on the horizontal axis and rho on vertical

            y_beta = radius_beta * np.sin(theta_beta)
            x_beta = radius_beta * np.cos(theta_beta)

            # Update beta shape lines
            self.line_beta.set_data(x_beta, y_beta)
            self.line_beta_mirror.set_data(x_beta, -y_beta)

            # Calculate the volume and center of mass for the beta shape
            beta_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)


        except Exception as e:
            print(f"Conversion error: {e}")
            self.line_beta.set_data([], [])
            self.line_beta_mirror.set_data([], [])

        # Update center of mass points
        self.cm_theoretical.set_data([current_params.z_sh], [0])
        center_of_mass_fos = calculator.calculate_center_of_mass(z_fos, rho_fos)
        self.cm_fos_calculated.set_data([center_of_mass_fos], [0])
        center_of_mass_spherical_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
        self.cm_spherical_fit_calculated.set_data([center_of_mass_spherical_fit], [0])
        center_of_mass_beta_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
        self.cm_beta_fit_calculated.set_data([center_of_mass_beta_fit], [0])

        # Calculate the ratio of calculated CM to theoretical shift, if NaN, set to 0
        # cm_ratio = (
        #     calculator.calculate_center_of_mass(z_fos, rho_fos) / current_params.z_sh
        #     if current_params.z_sh != 0 else 0.0
        # )

        # Update a reference sphere
        theta_reference_sphere = np.linspace(0, 2 * np.pi, 200)
        sphere_x = current_params.radius0 * np.cos(theta_reference_sphere)
        sphere_y = current_params.radius0 * np.sin(theta_reference_sphere)
        self.reference_sphere_line.set_data(sphere_x, sphere_y)
        self.reference_sphere_line.set_label(f'R₀={current_params.radius0:.2f} fm')

        # Update plot limits
        max_val = max(np.max(np.abs(z_fos)), np.max(np.abs(rho_fos))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Calculate the volume of the plotted shape for verification
        shape_volume = calculator.calculate_volume_in_cylindrical_coordinates(z_fos, rho_fos)

        # Calculate dimensions
        max_z = np.max(np.abs(z_fos))
        max_rho = np.max(rho_fos)

        # Calculate neck radius at a4 = 0.72 (scission point)
        if current_params.a4 > 0:
            neck_radius = min(rho_fos[len(rho_fos) // 2 - 10:len(rho_fos) // 2 + 10])
        else:
            neck_radius = max_rho

        # Add information text
        info_text = (
            f"R₀ = {current_params.radius0:.3f} fm\n"
            f"a₂ = {current_params.a2:.3f} (volume conservation)\n"
            f"\nParameter Relations:\n"
            f"c = q₂ + 1.0 + 1.5a₄\n"
            f"c = {current_params.q2:.3f} + 1.0 + 1.5×{current_params.a4:.3f} = {current_params.c_elongation:.3f}\n"
            f"\nShift Information:\n"
            f"Theoretical z_shift = {current_params.z_sh:.3f} fm\n"
            f"\nCenter of Mass:\n"
            f"Calculated CM (FoS): {center_of_mass_fos:.3f} fm\n"
            f"Calculated CM (Spherical Fit): {center_of_mass_spherical_fit:.3f} fm\n"
            f"Calculated CM (Beta Fit): {center_of_mass_beta_fit:.3f} fm\n"
            # f"Ratio of calculated CM to theoretical shift: {cm_ratio:.3f}\n"
            f"\nVolume Information:\n"
            f"Reference sphere volume: {current_params.sphere_volume:.3f} fm³\n"
            f"FoS shape volume: {shape_volume:.3f} fm³\n"
            f"Beta shape volume: {beta_volume:.3f} fm³\n"
            f"\nShape dimensions:\n"
            f"Max z_fos: {max_z:.2f} fm\n"
            f"Max ρ: {max_rho:.2f} fm\n"
            f"Neck radius: {neck_radius:.2f} fm\n"
            f"Calculated c (elongation): {max_z / current_params.radius0:.3f}\n"
        )

        # Remove old text if it exists
        for artist in self.ax_text.texts:
            artist.remove()

        # Add new text to the right-side text area
        self.ax_text.text(0.05, 0.95, info_text, transform=self.ax_text.transAxes,
                          fontsize=10, verticalalignment='top', horizontalalignment='left',
                          bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

        # Update title
        self.ax_plot.set_title(
            f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, '
            f'A={current_params.nucleons})', fontsize=14
        )

        # Update legend
        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(-0.1, 1))

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot(None)
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    plotter = FoSShapePlotter()
    plotter.run()


if __name__ == '__main__':
    main()
