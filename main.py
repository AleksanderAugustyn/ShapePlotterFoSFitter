"""
Nuclear Shape Plotter using Fourier-over-Spheroid (FoS) Parametrization
Based on the parametrization described in "Fourier-over-Spheroid shape parametrization 
applied to nuclear fission dynamics" by K. Pomorski et al.
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

matplotlib.use('TkAgg')


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int
    neutrons: int
    c: float = 1.0  # Elongation parameter
    a3: float = 0.0  # Reflectional asymmetry
    a4: float = 0.0  # Neck size
    eta: float = 0.0  # Non-axiality
    r0: float = 1.16  # Radius constant in fm
    # Additional shape parameters
    a1: float = 0.0
    a2: float = 0.0  # Will be calculated from volume conservation
    a5: float = 0.0
    a6: float = 0.0

    def __post_init__(self):
        """Calculate a2 from volume conservation."""
        self.update_a2()

    def update_a2(self):
        """Update a2 based on volume conservation: a2 = a4/3 - a6/5 + ..."""
        self.a2 = self.a4 / 3.0 - self.a6 / 5.0

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons

    @property
    def radius0(self) -> float:
        """Radius of equivalent sphere."""
        return self.r0 * (self.nucleons ** (1 / 3))

    @property
    def z0(self) -> float:
        """Half-length parameter."""
        return self.c * self.radius0

    @property
    def z_shift(self) -> float:
        """Shift to place mass center at origin."""
        return -3.0 / (4.0 * np.pi) * self.z0 * (self.a3 - self.a5 / 2.0)


class FoSShapeCalculator:
    """Class for performing Fourier-over-Spheroid shape calculations."""

    def __init__(self, params: FoSParameters):
        self.params = params

    def f_function(self, u: np.ndarray) -> np.ndarray:
        """Calculate the f(u) function for the shape."""
        # Start with sphere: 1 - u²
        f = 1.0 - u ** 2

        # Add Fourier terms
        # a1 term: cos(0.5π u)
        f -= self.params.a1 * np.cos(0.5 * np.pi * u)

        # a2 term: sin(π u)
        f -= self.params.a2 * np.sin(np.pi * u)

        # a3 term: cos(1.5π u)
        f -= self.params.a3 * np.cos(1.5 * np.pi * u)

        # a4 term: sin(2π u)
        f -= self.params.a4 * np.sin(2.0 * np.pi * u)

        # a5 term: cos(2.5π u)
        f -= self.params.a5 * np.cos(2.5 * np.pi * u)

        # a6 term: sin(3π u)
        f -= self.params.a6 * np.sin(3.0 * np.pi * u)

        return f

    def calculate_shape_3d(self, z_points: np.ndarray, phi: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the shape in 3D (with non-axiality) for given z points and angle phi."""
        z_min = -self.params.z0 + self.params.z_shift
        z_max = self.params.z0 + self.params.z_shift

        # Clip z values to valid range
        z_clipped = np.clip(z_points, z_min, z_max)

        # Calculate u = (z - z_shift) / z0
        u = (z_clipped - self.params.z_shift) / self.params.z0

        # Calculate f(u)
        f_u = self.f_function(u)

        # Calculate rho²
        rho_squared = self.params.radius0 ** 2 * self.params.c * f_u

        # Apply non-axiality factor
        eta = self.params.eta
        non_axial_factor = (1 - eta ** 2) / (1 + eta ** 2 + 2 * eta * np.cos(2 * phi))

        rho_squared *= non_axial_factor

        # Ensure non-negative
        rho_squared = np.maximum(0, rho_squared)
        rho = np.sqrt(rho_squared)

        return rho, z_clipped

    def calculate_shape_2d(self, z_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate axially symmetric shape (eta=0 or averaged over phi)."""
        if self.params.eta == 0:
            # Truly axially symmetric
            return self.calculate_shape_3d(z_points, phi=0)
        else:
            # Average over phi for visualization
            n_phi = 100
            phi_values = np.linspace(0, 2 * np.pi, n_phi)
            rho_sum = np.zeros_like(z_points)

            for phi in phi_values:
                rho, z = self.calculate_shape_3d(z_points, phi)
                rho_sum += rho

            rho_avg = rho_sum / n_phi
            return rho_avg, z_points

    def calculate_volume(self, n_points: int = 2000) -> float:
        """Calculate volume by numerical integration."""
        z_min = -self.params.z0 + self.params.z_shift
        z_max = self.params.z0 + self.params.z_shift

        z = np.linspace(z_min, z_max, n_points)
        rho, _ = self.calculate_shape_2d(z)

        # Volume element: π ρ² dz
        dz = (z_max - z_min) / (n_points - 1)
        volume = np.pi * np.sum(rho ** 2) * dz

        return volume

    def calculate_center_of_mass(self, n_points: int = 2000) -> float:
        """Calculate z-coordinate of center of mass."""
        z_min = -self.params.z0 + self.params.z_shift
        z_max = self.params.z0 + self.params.z_shift

        z = np.linspace(z_min, z_max, n_points)
        rho, _ = self.calculate_shape_2d(z)

        # Volume element: π ρ² dz
        dz = (z_max - z_min) / (n_points - 1)
        volume_elements = np.pi * rho ** 2 * dz

        total_volume = np.sum(volume_elements)
        z_cm = np.sum(z * volume_elements) / total_volume

        return z_cm

    def get_sphere_volume(self) -> float:
        """Volume of equivalent sphere."""
        return (4.0 / 3.0) * np.pi * self.params.radius0 ** 3


class FoSShapePlotter:
    """Class for handling the plotting interface and user interaction."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Default values
        self.initial_z = 92  # Uranium
        self.initial_n = 144
        self.initial_c = 1.0
        self.initial_a3 = 0.0
        self.initial_a4 = 0.0
        self.initial_eta = 0.0

        # Create figure
        self.fig = None
        self.ax_plot = None
        self.line_scaled = None
        self.line_scaled_mirror = None
        self.line_unscaled = None
        self.line_unscaled_mirror = None
        self.sphere_line = None
        self.point_cm = None
        self.point_cm_unscaled = None

        # UI elements
        self.slider_z = None
        self.slider_n = None
        self.slider_c = None
        self.slider_a3 = None
        self.slider_a4 = None
        self.slider_eta = None
        self.buttons = []
        self.reset_button = None
        self.save_button = None
        self.config_buttons = []

        # Initialize parameters
        self.params = FoSParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            c=self.initial_c,
            a3=self.initial_a3,
            a4=self.initial_a4,
            eta=self.initial_eta
        )

        # Number of points for shape calculation
        self.n_points = 2000

        # Set up the interface
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_plot = self.fig.add_subplot(111)

        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True)
        self.ax_plot.set_title('Nuclear Shape (Fourier-over-Spheroid Parametrization)', fontsize=14)
        self.ax_plot.set_xlabel('Z (fm)', fontsize=12)
        self.ax_plot.set_ylabel('ρ (fm)', fontsize=12)

        # Initialize the shape plot
        z_points = np.linspace(-30, 30, self.n_points)

        # Create placeholder lines
        self.line_scaled, = self.ax_plot.plot([], [], 'b-', label='Scaled', linewidth=2)
        self.line_scaled_mirror, = self.ax_plot.plot([], [], 'b-', linewidth=2)
        self.line_unscaled, = self.ax_plot.plot([], [], 'r--', label='Unscaled', alpha=0.7)
        self.line_unscaled_mirror, = self.ax_plot.plot([], [], 'r--', alpha=0.7)

        # Reference sphere
        theta = np.linspace(0, 2 * np.pi, 200)
        R0 = self.params.radius0
        sphere_x = R0 * np.cos(theta)
        sphere_y = R0 * np.sin(theta)
        self.sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray',
                                              alpha=0.5, label='R₀')

        # Center of mass markers
        self.point_cm, = self.ax_plot.plot(0, 0, 'bo', label='CM', markersize=8)
        self.point_cm_unscaled, = self.ax_plot.plot(0, 0, 'ro', label='CM (unscaled)',
                                                    markersize=8, alpha=0.7)

        self.ax_plot.legend()

    def setup_controls(self):
        """Set up all UI controls."""
        # Starting y position for first slider
        first_slider_y = 0.02
        slider_spacing = 0.03

        # Proton (Z) controls
        y_pos = first_slider_y
        ax_z = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_z_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_z_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120,
                               valinit=self.initial_z, valstep=1)
        btn_z_decrease = Button(ax_z_decrease, '-')
        btn_z_increase = Button(ax_z_increase, '+')
        self.buttons.extend([btn_z_decrease, btn_z_increase])

        # Neutron (N) controls
        y_pos += slider_spacing
        ax_n = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_n_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_n_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180,
                               valinit=self.initial_n, valstep=1)
        btn_n_decrease = Button(ax_n_decrease, '-')
        btn_n_increase = Button(ax_n_increase, '+')
        self.buttons.extend([btn_n_decrease, btn_n_increase])

        # Elongation (c) controls
        y_pos += slider_spacing
        ax_c = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_c_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_c_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_c = Slider(ax=ax_c, label='c', valmin=0.7, valmax=3.0,
                               valinit=self.initial_c, valstep=0.025)
        btn_c_decrease = Button(ax_c_decrease, '-')
        btn_c_increase = Button(ax_c_increase, '+')
        self.buttons.extend([btn_c_decrease, btn_c_increase])

        # Asymmetry (a3) controls
        y_pos += slider_spacing
        ax_a3 = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_a3_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_a3_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_a3 = Slider(ax=ax_a3, label='a₃', valmin=-0.5, valmax=0.5,
                                valinit=self.initial_a3, valstep=0.025)
        btn_a3_decrease = Button(ax_a3_decrease, '-')
        btn_a3_increase = Button(ax_a3_increase, '+')
        self.buttons.extend([btn_a3_decrease, btn_a3_increase])

        # Neck (a4) controls
        y_pos += slider_spacing
        ax_a4 = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_a4_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_a4_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_a4 = Slider(ax=ax_a4, label='a₄', valmin=0.0, valmax=0.8,
                                valinit=self.initial_a4, valstep=0.025)
        btn_a4_decrease = Button(ax_a4_decrease, '-')
        btn_a4_increase = Button(ax_a4_increase, '+')
        self.buttons.extend([btn_a4_decrease, btn_a4_increase])

        # Non-axiality (eta) controls
        y_pos += slider_spacing
        ax_eta = plt.axes((0.25, y_pos, 0.5, 0.02))
        ax_eta_decrease = plt.axes((0.16, y_pos, 0.04, 0.02))
        ax_eta_increase = plt.axes((0.80, y_pos, 0.04, 0.02))

        self.slider_eta = Slider(ax=ax_eta, label='η', valmin=0.0, valmax=0.5,
                                 valinit=self.initial_eta, valstep=0.025)
        btn_eta_decrease = Button(ax_eta_decrease, '-')
        btn_eta_increase = Button(ax_eta_increase, '+')
        self.buttons.extend([btn_eta_decrease, btn_eta_increase])

        # Style font sizes
        for slider in [self.slider_z, self.slider_n, self.slider_c,
                       self.slider_a3, self.slider_a4, self.slider_eta]:
            slider.label.set_fontsize(12)
            slider.valtext.set_fontsize(12)

        # Create action buttons
        ax_reset = plt.axes((0.8, 0.25, 0.1, 0.04))
        self.reset_button = Button(ax=ax_reset, label='Reset')

        ax_save = plt.axes((0.8, 0.2, 0.1, 0.04))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        # Create configuration buttons
        config_labels = ['Sphere', 'Prolate', 'Asymmetric', 'Necked']
        for i, label in enumerate(config_labels):
            ax_config = plt.axes((0.02, 0.6 - i * 0.1, 0.1, 0.04))
            btn = Button(ax=ax_config, label=label)
            self.config_buttons.append(btn)

    def apply_configuration(self, config_num):
        """Apply a predefined configuration."""
        configs = {
            0: {'Z': 92, 'N': 144, 'c': 1.0, 'a3': 0.0, 'a4': 0.0, 'eta': 0.0},  # Sphere
            1: {'Z': 92, 'N': 144, 'c': 1.5, 'a3': 0.0, 'a4': 0.0, 'eta': 0.0},  # Prolate
            2: {'Z': 92, 'N': 144, 'c': 1.8, 'a3': 0.2, 'a4': 0.1, 'eta': 0.0},  # Asymmetric
            3: {'Z': 92, 'N': 144, 'c': 2.2, 'a3': 0.0, 'a4': 0.5, 'eta': 0.0}  # Necked
        }

        config = configs[config_num]
        self.slider_z.set_val(config['Z'])
        self.slider_n.set_val(config['N'])
        self.slider_c.set_val(config['c'])
        self.slider_a3.set_val(config['a3'])
        self.slider_a4.set_val(config['a4'])
        self.slider_eta.set_val(config['eta'])

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)
        self.slider_c.on_changed(self.update_plot)
        self.slider_a3.on_changed(self.update_plot)
        self.slider_a4.on_changed(self.update_plot)
        self.slider_eta.on_changed(self.update_plot)

        # Connect button handlers
        sliders = [self.slider_z, self.slider_n, self.slider_c,
                   self.slider_a3, self.slider_a4, self.slider_eta]

        for i, slider in enumerate(sliders):
            self.buttons[i * 2].on_clicked(self.create_button_handler(slider, -1))
            self.buttons[i * 2 + 1].on_clicked(self.create_button_handler(slider, 1))

        # Connect action buttons
        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)

        # Connect configuration buttons
        for i, btn in enumerate(self.config_buttons):
            btn.on_clicked(lambda event, num=i: self.apply_configuration(num))

    @staticmethod
    def create_button_handler(slider_obj: Slider, increment: int):
        """Create a button click handler for a slider object."""

        def handler(_):
            """Handle button click event."""
            new_val = slider_obj.val + increment * slider_obj.valstep
            if slider_obj.valmin <= new_val <= slider_obj.valmax:
                slider_obj.set_val(new_val)

        return handler

    def reset_values(self, _):
        """Reset all sliders to their initial values."""
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.slider_c.set_val(self.initial_c)
        self.slider_a3.set_val(self.initial_a3)
        self.slider_a4.set_val(self.initial_a4)
        self.slider_eta.set_val(self.initial_eta)

    def save_plot(self, _):
        """Save the current plot to a file."""
        params = [
            int(self.slider_z.val),
            int(self.slider_n.val),
            self.slider_c.val,
            self.slider_a3.val,
            self.slider_a4.val,
            self.slider_eta.val
        ]
        filename = f"fos_shape_{'_'.join(f'{p:.2f}' for p in params)}.png"
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self, _):
        """Update the plot with new parameters."""
        # Update parameters
        self.params.protons = int(self.slider_z.val)
        self.params.neutrons = int(self.slider_n.val)
        self.params.c = self.slider_c.val
        self.params.a3 = self.slider_a3.val
        self.params.a4 = self.slider_a4.val
        self.params.eta = self.slider_eta.val
        self.params.update_a2()  # Update a2 from volume conservation

        # Create calculator
        calculator = FoSShapeCalculator(self.params)

        # Calculate shape bounds
        z_min = -self.params.z0 + self.params.z_shift
        z_max = self.params.z0 + self.params.z_shift
        z_range = z_max - z_min

        # Create z points with some margin
        z_points = np.linspace(z_min - 0.1 * z_range, z_max + 0.1 * z_range, self.n_points)

        # Calculate unscaled shape
        rho_unscaled, z_unscaled = calculator.calculate_shape_2d(z_points)

        # Calculate volumes
        sphere_volume = calculator.get_sphere_volume()
        shape_volume = calculator.calculate_volume()
        volume_scale = (sphere_volume / shape_volume) ** (1 / 3)

        # Calculate center of mass
        cm_unscaled = calculator.calculate_center_of_mass()

        # Scale the shape
        rho_scaled = rho_unscaled * volume_scale
        z_scaled = (z_unscaled - cm_unscaled) * volume_scale

        # Update plots
        self.line_scaled.set_data(z_scaled, rho_scaled)
        self.line_scaled_mirror.set_data(z_scaled, -rho_scaled)
        self.line_unscaled.set_data(z_unscaled, rho_unscaled)
        self.line_unscaled_mirror.set_data(z_unscaled, -rho_unscaled)

        # Update center of mass markers
        self.point_cm.set_data([0], [0])  # Scaled shape is centered
        self.point_cm_unscaled.set_data([cm_unscaled], [0])

        # Update reference sphere
        r0 = self.params.radius0
        theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = r0 * np.cos(theta)
        sphere_y = r0 * np.sin(theta)
        self.sphere_line.set_data(sphere_x, sphere_y)

        # Update plot limits
        max_val = max(np.max(np.abs(z_scaled)), np.max(np.abs(rho_scaled))) * 1.2
        max_val = max(max_val, r0 * 1.2)
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Calculate dimensions
        max_z = np.max(np.abs(z_scaled))
        max_rho = np.max(np.abs(rho_scaled))
        total_length = 2 * max_z
        total_width = 2 * max_rho

        # Add info text
        info_text = (
            f"A = {self.params.nucleons} (Z={self.params.protons}, N={self.params.neutrons})\n"
            f"R₀ = {self.params.radius0:.3f} fm\n"
            f"Sphere volume: {sphere_volume:.1f} fm³\n"
            f"Shape volume: {shape_volume:.1f} fm³\n"
            f"Volume scale factor: {volume_scale:.4f}\n"
            f"CM (unscaled): {cm_unscaled:.3f} fm\n"
            f"Length: {total_length:.1f} fm\n"
            f"Width: {total_width:.1f} fm\n"
            f"Aspect ratio: {total_length / total_width:.2f}"
        )

        # Remove old text
        for artist in self.ax_plot.texts:
            artist.remove()

        # Add new text
        self.ax_plot.text(1.05 * max_val, 0.5 * max_val, info_text,
                          fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # Update title
        title = f'Nuclear Shape: FoS Parametrization (A={self.params.nucleons})'
        if self.params.eta > 0:
            title += f' [η={self.params.eta:.2f}]'
        self.ax_plot.set_title(title, fontsize=14)

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
