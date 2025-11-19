"""
FoSPlotter.py
Nuclear Shape Plotter GUI using Fourier-over-Spheroid (FoS) Parametrization.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, CheckButtons

from BetaDeformationCalculator import BetaDeformationCalculator, FitResult
from CylindricalToSphericalConverter import CylindricalToSphericalConverter
# Import dependencies
from FoSModel import FoSParameters, FoSShapeCalculator

# Set backend
matplotlib.use('TkAgg')


class FoSShapePlotter:
    """Class for plotting Fourier-over-Spheroid nuclear shapes."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Default shape parameters
        self.initial_z: int = 92  # Uranium
        self.initial_n: int = 144
        self.initial_c: float = 1.0  # elongation
        self.initial_a3: float = 0.0
        self.initial_a4: float = 0.0
        self.initial_a5: float = 0.0
        self.initial_a6: float = 0.0

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

        # Sliders and Buttons
        self.slider_z = None
        self.slider_n = None
        self.slider_c = None
        self.slider_a3 = None
        self.slider_a4 = None
        self.slider_a5 = None
        self.slider_a6 = None
        self.slider_max_beta = None
        self.slider_number_of_points = None
        self.reset_button = None
        self.save_button = None
        self.save_coordinates_to_files_button = None
        self.print_beta_button = None
        self.preset_buttons = []

        # Checkboxes
        self.show_text_checkbox = None
        self.show_beta_checkbox = None
        self.show_text_info = True
        self.show_beta_fitting = True

        self.slider_buttons = {}
        self.updating = False

        # Initialize parameters
        self.nuclear_params = FoSParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            c_elongation=self.initial_c,
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
        self.ax_text.axis('off')

        plt.subplots_adjust(left=0.13, bottom=0.38, right=0.97, top=0.97, wspace=0.1)

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

        # Initialize plots
        self.line_fos, = self.ax_plot.plot(z, rho, 'b-', label='FoS shape (normalized)', linewidth=2)
        self.line_fos_mirror, = self.ax_plot.plot(z, -rho, 'b-', linewidth=2)
        self.line_fos_spherical, = self.ax_plot.plot([], [], 'y--', label='FoS shape (spherical)', linewidth=1.5)
        self.line_fos_spherical_mirror, = self.ax_plot.plot([], [], 'y--', linewidth=1.5)
        self.line_fos_spherical_shifted_for_conversion, = self.ax_plot.plot([], [], 'g:', label='FoS shape (spherical, shifted for conversion)', linewidth=1.5, alpha=0.5)
        self.line_fos_spherical_shifted_for_conversion_mirror, = self.ax_plot.plot([], [], 'g:', linewidth=1.5, alpha=0.5)
        self.line_beta, = self.ax_plot.plot([], [], 'r--', label='Beta shape (analytical)', linewidth=2, alpha=0.7)
        self.line_beta_mirror, = self.ax_plot.plot([], [], 'r--', linewidth=2, alpha=0.7)
        self.line_beta_fitted, = self.ax_plot.plot([], [], 'm-.', label='Beta shape (fitted)', linewidth=2, alpha=0.7)
        self.line_beta_fitted_mirror, = self.ax_plot.plot([], [], 'm-.', linewidth=2, alpha=0.7)
        self.reference_sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray', alpha=0.5, label=f'R₀={r0:.2f} fm')

        # Initialize markers
        self.cm_reference_point, = self.ax_plot.plot(0, 0, 'ro', label='Expected CM (Origin)', markersize=6, alpha=0.7)
        self.cm_theoretical, = self.ax_plot.plot(0, 0, 'b^', label='CM before theoretical shift', markersize=6, alpha=0.7)
        self.cm_fos_calculated, = self.ax_plot.plot(0, 0, 'go', label='Calculated CM (FoS)', markersize=6, alpha=0.7)
        self.cm_spherical_fit_calculated, = self.ax_plot.plot(0, 0, 'ms', label='Calculated CM (Spherical Fit)', markersize=6, alpha=0.7)
        self.cm_spherical_fit_calculated_shifted, = self.ax_plot.plot(0, 0, 'ms', label='Calculated CM (Spherical Fit, shifted)', markersize=6, alpha=0.5)
        self.cm_beta_fit_calculated, = self.ax_plot.plot(0, 0, 'cs', label='Calculated CM (Beta Fit)', markersize=6, alpha=0.7)

        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(0, 1))

    def create_slider_with_buttons(self, slider_name, y_position, label, value_minimal, value_maximal, value_initial, value_step):
        dec_ax = plt.axes((0.20, y_position, 0.016, 0.024))
        dec_button = Button(dec_ax, '-')
        slider_ax = plt.axes((0.25, y_position, 0.5, 0.024))
        slider = Slider(ax=slider_ax, label=label, valmin=value_minimal, valmax=value_maximal, valinit=value_initial, valstep=value_step)
        inc_ax = plt.axes((0.78, y_position, 0.016, 0.024))
        inc_button = Button(inc_ax, '+')
        self.slider_buttons[slider_name] = {'dec_button': dec_button, 'inc_button': inc_button, 'slider': slider}
        return slider, dec_button, inc_button

    def setup_controls(self):
        first_slider_y = 0.02
        slider_spacing = 0.03
        self.slider_z, _, _ = self.create_slider_with_buttons('z', first_slider_y, 'Z', 82, 120, self.initial_z, 1)
        self.slider_n, _, _ = self.create_slider_with_buttons('n', first_slider_y + slider_spacing, 'N', 120, 180, self.initial_n, 1)
        self.slider_c, _, _ = self.create_slider_with_buttons('c', first_slider_y + 2 * slider_spacing, 'c', 0.5, 3.0, self.initial_c, 0.01)
        self.slider_a3, _, _ = self.create_slider_with_buttons('a3', first_slider_y + 3 * slider_spacing, 'a₃', -0.6, 0.6, self.initial_a3, 0.01)
        self.slider_a4, _, _ = self.create_slider_with_buttons('a4', first_slider_y + 4 * slider_spacing, 'a₄', -0.75, 0.75, self.initial_a4, 0.01)
        self.slider_a5, _, _ = self.create_slider_with_buttons('a5', first_slider_y + 5 * slider_spacing, 'a₅', -0.5, 0.5, self.initial_a5, 0.01)
        self.slider_a6, _, _ = self.create_slider_with_buttons('a6', first_slider_y + 6 * slider_spacing, 'a₆', -0.5, 0.5, self.initial_a6, 0.01)
        self.slider_max_beta, _, _ = self.create_slider_with_buttons('max_beta', first_slider_y + 7 * slider_spacing, 'Max Betas Used For Fit', 1.0, 64.0, 12.0, 1.0)
        self.slider_number_of_points, _, _ = self.create_slider_with_buttons('number_of_points', first_slider_y + 8 * slider_spacing, 'Number of Points', 180, 3600, 720, 180)

        for slider in [self.slider_z, self.slider_n, self.slider_c, self.slider_a3, self.slider_a4, self.slider_a5, self.slider_a6, self.slider_max_beta, self.slider_number_of_points]:
            slider.label.set_fontsize(10)
            slider.valtext.set_fontsize(10)

        for slider_name, buttons in self.slider_buttons.items():
            buttons['dec_button'].label.set_fontsize(16)
            buttons['inc_button'].label.set_fontsize(16)

        ax_show_text = plt.axes((0.82, 0.45, 0.08, 0.032))
        self.show_text_checkbox = CheckButtons(ax_show_text, ['Show Text Info'], [self.show_text_info])
        ax_show_beta = plt.axes((0.82, 0.42, 0.08, 0.032))
        self.show_beta_checkbox = CheckButtons(ax_show_beta, ['Show Beta Fitting'], [self.show_beta_fitting])

        ax_reset = plt.axes((0.82, 0.37, 0.08, 0.032))
        self.reset_button = Button(ax=ax_reset, label='Reset')
        ax_save = plt.axes((0.82, 0.32, 0.08, 0.032))
        self.save_button = Button(ax=ax_save, label='Save Plot')
        ax_save_spherical = plt.axes((0.82, 0.27, 0.08, 0.032))
        self.save_coordinates_to_files_button = Button(ax=ax_save_spherical, label='Save Spherical Fit Coords')
        ax_print_beta = plt.axes((0.82, 0.22, 0.08, 0.032))
        self.print_beta_button = Button(ax=ax_print_beta, label='Print All Beta Params')

        preset_labels = ['Sphere', 'Prolate', 'Oblate', 'Pear-shaped', 'Two-center', 'Scission']
        for i, label in enumerate(preset_labels):
            ax_preset = plt.axes((0.02, 0.6 - i * 0.06, 0.08, 0.032))
            btn = Button(ax=ax_preset, label=label)
            self.preset_buttons.append(btn)

    def apply_preset(self, preset_num):
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
        for param_name, value in preset.items():
            slider_attr_name = f"slider_{param_name}"
            slider = getattr(self, slider_attr_name, None)
            if slider:
                slider.set_val(value)
        self.updating = False
        self.update_plot(None)

    def setup_event_handlers(self):
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)
        self.slider_c.on_changed(self.update_plot)
        self.slider_a3.on_changed(self.update_plot)
        self.slider_a4.on_changed(self.update_plot)
        self.slider_a5.on_changed(self.update_plot)
        self.slider_a6.on_changed(self.update_plot)
        self.slider_max_beta.on_changed(self.update_plot)
        self.slider_number_of_points.on_changed(self.update_plot)

        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)
        self.save_coordinates_to_files_button.on_clicked(self.save_coordinates_to_files)
        self.print_beta_button.on_clicked(self.print_all_beta_parameters)
        self.show_text_checkbox.on_clicked(self.on_show_text_changed)
        self.show_beta_checkbox.on_clicked(self.on_show_beta_changed)

        for i, btn in enumerate(self.preset_buttons):
            btn.on_clicked(lambda event, num=i: self.apply_preset(num))
        self.setup_slider_buttons()

    def setup_slider_buttons(self):
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

        for slider_name, buttons in self.slider_buttons.items():
            buttons['dec_button'].on_clicked(create_decrement_handler(buttons['slider']))
            buttons['inc_button'].on_clicked(create_increment_handler(buttons['slider']))

    def on_show_text_changed(self, label):
        self.show_text_info = not self.show_text_info
        self.update_plot(None)

    def on_show_beta_changed(self, label):
        self.show_beta_fitting = not self.show_beta_fitting
        self.update_plot(None)

    def reset_values(self, _):
        self.updating = True
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.slider_c.set_val(self.initial_c)
        self.slider_a3.set_val(self.initial_a3)
        self.slider_a4.set_val(self.initial_a4)
        self.slider_a5.set_val(self.initial_a5)
        self.slider_a6.set_val(self.initial_a6)
        self.slider_max_beta.set_val(12)
        self.slider_number_of_points.set_val(720)
        self.updating = False
        self.update_plot(None)

    def save_plot(self, _):
        q2_calculated = float(self.slider_c.val) - 1.0 - 1.5 * float(self.slider_a4.val)
        params = [
            float(self.slider_c.val), q2_calculated, float(self.slider_a3.val),
            float(self.slider_a4.val), float(self.slider_a5.val), float(self.slider_a6.val),
            float(self.slider_number_of_points.val), float(self.slider_max_beta.val)
        ]
        filename = f"fos_shape_{int(self.slider_z.val)}_{int(self.slider_n.val)}_" + \
                   f"{'_'.join(f'{p:.3f}' for p in params)}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    def save_coordinates_to_files(self, _):
        # Delegate to BatchProcessor logic if possible, or reuse local logic for GUI integration
        # For simplicity, we will use a local implementation similar to BatchProcessor but adapted for GUI feedback
        from BatchProcessor import calculate_and_save_shape
        success = calculate_and_save_shape(
            int(self.slider_z.val), int(self.slider_n.val),
            self.slider_c.val, self.slider_a3.val, self.slider_a4.val,
            self.slider_a5.val, self.slider_a6.val,
            int(self.slider_number_of_points.val)
        )
        if success:
            print("Coordinates saved successfully to current directory.")
        else:
            print("Failed to save coordinates.")

    def print_all_beta_parameters(self, _):
        try:
            current_params = FoSParameters(
                protons=int(self.slider_z.val), neutrons=int(self.slider_n.val),
                c_elongation=self.slider_c.val, a3=self.slider_a3.val,
                a4=self.slider_a4.val, a5=self.slider_a5.val, a6=self.slider_a6.val
            )
            calculator_fos = FoSShapeCalculator(current_params)
            current_number_of_points = int(self.slider_number_of_points.val)
            z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=current_number_of_points)

            z_work = z_fos_cylindrical.copy()
            cumulative_shift = 0.0

            cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
            is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9)

            if not is_convertible:
                z_step = 0.1
                shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0
                z_length = float(np.max(z_fos_cylindrical)) - float(np.min(z_fos_cylindrical))
                max_shift = z_length / 2.0
                while abs(cumulative_shift) < max_shift:
                    cumulative_shift += shift_direction * z_step
                    z_work = z_fos_cylindrical + cumulative_shift
                    cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                    if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9):
                        is_convertible = True
                        break

            if not is_convertible:
                print("Error: Shape cannot be converted to spherical coordinates")
                return

            theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=current_number_of_points)
            spherical_to_beta_converter = BetaDeformationCalculator(theta=theta_fos, radius=radius_fos, number_of_nucleons=current_params.nucleons)
            l_max_value = int(self.slider_max_beta.val)
            beta_parameters = spherical_to_beta_converter.calculate_beta_parameters(l_max=l_max_value)

            try:
                fitting_results: FitResult = spherical_to_beta_converter.fit_beta_parameters_rmse(l_max=l_max_value)
                beta_parameters_fitted = fitting_results['beta_fitted']
                scaling_factor_fitted = fitting_results['scaling_factor_fitted']
                rmse_fitting = fitting_results['rmse']
            except Exception as fit_error:
                print(f"Beta fitting error: {fit_error}")
                beta_parameters_fitted = beta_parameters.copy()
                scaling_factor_fitted = 1.0
                rmse_fitting = -1.0

            print(f"\n{'=' * 80}")
            print(f"ALL BETA PARAMETERS FOR Z={int(self.slider_z.val)}, N={int(self.slider_n.val)}")
            q2_calculated = self.slider_c.val - 1.0 - 1.5 * self.slider_a4.val
            print(f"Parameters: c={self.slider_c.val:.3f}, q2={q2_calculated:.3f}, a3={self.slider_a3.val:.3f}")
            print(f"           a4={self.slider_a4.val:.3f}, a5={self.slider_a5.val:.3f}, a6={self.slider_a6.val:.3f}")
            print(f"Max Beta Used: {l_max_value}")
            print(f"{'=' * 80}")

            print(f"\nANALYTICAL METHOD:")
            analytical_params = [f"β_{l}" for l in sorted(beta_parameters.keys())]
            analytical_values = [f"{beta_parameters[l]:+.3f}" for l in sorted(beta_parameters.keys())]
            print(f"Parameters: {', '.join(analytical_params)}")
            print(f"Values:     {', '.join(analytical_values)}")

            print(f"\nFITTED METHOD (RMSE minimization):")
            print(f"Scaling Factor: {scaling_factor_fitted:.6f}, RMSE: {rmse_fitting:.6f} fm")
            fitted_params = [f"β_{l}" for l in sorted(beta_parameters_fitted.keys())]
            fitted_values = [f"{beta_parameters_fitted[l]:+.3f}" for l in sorted(beta_parameters_fitted.keys())]
            print(f"Parameters: {', '.join(fitted_params)}")
            print(f"Values:     {', '.join(fitted_values)}")
            print(f"\n{'=' * 80}")

        except Exception as e:
            print(f"Error printing beta parameters: {e}")

    def update_plot(self, _):
        current_params = FoSParameters(
            protons=int(self.slider_z.val), neutrons=int(self.slider_n.val),
            c_elongation=self.slider_c.val, a3=self.slider_a3.val,
            a4=self.slider_a4.val, a5=self.slider_a5.val, a6=self.slider_a6.val
        )
        calculator_fos = FoSShapeCalculator(current_params)
        current_number_of_points = int(self.slider_number_of_points.val)
        z_fos_cylindrical, rho_fos_cylindrical = calculator_fos.calculate_shape(n_points=current_number_of_points)

        self.line_fos.set_data(z_fos_cylindrical, rho_fos_cylindrical)
        self.line_fos_mirror.set_data(z_fos_cylindrical, -rho_fos_cylindrical)

        # Initialize vars
        center_of_mass_beta_fitted = 0.0
        fos_spherical_volume = 0.0
        beta_volume = 0.0
        beta_volume_fitted = 0.0
        fos_spherical_surface_area = 0.0
        beta_surface_area = 0.0
        beta_surface_area_fitted = 0.0
        radius_fos = np.array([])
        theta_fos = np.array([])
        conversion_root_mean_squared_error = 0.0
        rmse_beta_fit = 0.0
        rmse_fitting = 0.0
        scaling_factor_fitted = 0.0
        scaling_factor_volume = 0.0
        is_converted = False
        cumulative_shift = 0.0
        significant_beta_parameters = ""
        center_of_mass_spherical_fit = 0.0
        center_of_mass_beta_fit = 0.0

        if self.show_beta_fitting:
            try:
                z_work = z_fos_cylindrical.copy()
                cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                is_convertible = cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9)
                if not is_convertible:
                    z_step = 0.1
                    shift_direction = -1.0 if current_params.z_sh >= 0 else 1.0
                    z_length = float(np.max(z_fos_cylindrical)) - float(np.min(z_fos_cylindrical))
                    max_shift = z_length / 2.0
                    while abs(cumulative_shift) < max_shift:
                        cumulative_shift += shift_direction * z_step
                        z_work = z_fos_cylindrical + cumulative_shift
                        cylindrical_to_spherical_converter = CylindricalToSphericalConverter(z_points=z_work, rho_points=rho_fos_cylindrical)
                        if cylindrical_to_spherical_converter.is_unambiguously_convertible(n_points=current_number_of_points, tolerance=1e-9):
                            is_convertible = True
                            is_converted = True
                            break

                if is_convertible:
                    theta_fos, radius_fos = cylindrical_to_spherical_converter.convert_to_spherical(n_theta=current_number_of_points)
                    z_fos_spherical, rho_fos_spherical = cylindrical_to_spherical_converter.convert_to_cylindrical(n_theta=current_number_of_points)
                    validation = cylindrical_to_spherical_converter.validate_conversion(theta_converted=theta_fos, r_converted=radius_fos)
                    conversion_root_mean_squared_error = validation['rmse_combined']

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

                    spherical_to_beta_converter = BetaDeformationCalculator(theta=theta_fos, radius=radius_fos, number_of_nucleons=current_params.nucleons)
                    l_max_value = int(self.slider_max_beta.val)
                    beta_parameters = spherical_to_beta_converter.calculate_beta_parameters(l_max=l_max_value)

                    try:
                        fitting_results: FitResult = spherical_to_beta_converter.fit_beta_parameters_rmse(l_max=l_max_value)
                        beta_parameters_fitted = fitting_results['beta_fitted']
                        scaling_factor_fitted = fitting_results['scaling_factor_fitted']
                        scaling_factor_volume = fitting_results['scaling_factor_volume']
                        rmse_fitting = fitting_results['rmse']
                    except Exception as fit_error:
                        print(f"Beta fitting error: {fit_error}")
                        beta_parameters_fitted = beta_parameters.copy()
                        scaling_factor_fitted = 1.0
                        scaling_factor_volume = 1.0
                        rmse_fitting = -1.0

                    theta_beta, radius_beta = spherical_to_beta_converter.reconstruct_shape(beta=beta_parameters, n_theta=current_number_of_points)
                    z_beta = radius_beta * np.cos(theta_beta)
                    rho_beta = radius_beta * np.sin(theta_beta)

                    theta_beta_fitted, radius_beta_fitted = spherical_to_beta_converter.reconstruct_shape(beta=beta_parameters_fitted, n_theta=current_number_of_points)
                    z_beta_fitted = radius_beta_fitted * np.cos(theta_beta_fitted)
                    rho_beta_fitted = radius_beta_fitted * np.sin(theta_beta_fitted)

                    if abs(cumulative_shift) > 1e-10:
                        z_beta = z_beta - cumulative_shift
                        z_beta_fitted = z_beta_fitted - cumulative_shift

                    self.line_beta.set_data(z_beta, rho_beta)
                    self.line_beta_mirror.set_data(z_beta, -rho_beta)
                    self.line_beta_fitted.set_data(z_beta_fitted, rho_beta_fitted)
                    self.line_beta_fitted_mirror.set_data(z_beta_fitted, -rho_beta_fitted)

                    fos_spherical_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
                    beta_volume = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
                    beta_volume_fitted = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius=radius_beta_fitted, theta=theta_beta_fitted)
                    fos_spherical_surface_area = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
                    beta_surface_area = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
                    beta_surface_area_fitted = BetaDeformationCalculator.calculate_surface_area_in_spherical_coordinates(radius=radius_beta_fitted, theta=theta_beta_fitted)

                    rmse_beta_fit = np.sqrt(np.mean((radius_beta - radius_fos) ** 2))

                    beta_strings_analytical = [f"β_{l:<2} = {val:+.4f}" for l, val in sorted(beta_parameters.items()) if abs(val) >= 0.00095]
                    beta_strings_fitted = [f"β_{l:<2} = {val:+.4f}" for l, val in sorted(beta_parameters_fitted.items()) if abs(val) >= 0.00095]

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
                    self._clear_beta_lines()
                    significant_beta_parameters = "Shape could not be made convertible"
            except Exception as e:
                print(f"Conversion error: {e}")
                self._clear_beta_lines()
                significant_beta_parameters = "Calculation Error"
        else:
            self._clear_beta_lines()
            significant_beta_parameters = "Beta fitting disabled"

        # Update CM
        self.cm_theoretical.set_data([current_params.z_sh], [0])
        center_of_mass_fos = calculator_fos.calculate_center_of_mass_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)
        self.cm_fos_calculated.set_data([center_of_mass_fos], [0])

        if self.show_beta_fitting and len(radius_fos) > 0:
            center_of_mass_spherical_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_fos, theta=theta_fos)
            self.cm_spherical_fit_calculated.set_data([center_of_mass_spherical_fit - cumulative_shift], [0])
            self.cm_spherical_fit_calculated_shifted.set_data([center_of_mass_spherical_fit], [0])
            if len(radius_beta) > 0:
                center_of_mass_beta_fit = BetaDeformationCalculator.calculate_center_of_mass_in_spherical_coordinates(radius=radius_beta, theta=theta_beta)
                self.cm_beta_fit_calculated.set_data([center_of_mass_beta_fit - cumulative_shift], [0])
            else:
                self.cm_beta_fit_calculated.set_data([], [])
        else:
            self.cm_spherical_fit_calculated.set_data([], [])
            self.cm_spherical_fit_calculated_shifted.set_data([], [])
            self.cm_beta_fit_calculated.set_data([], [])

        theta_reference_sphere = np.linspace(0, 2 * np.pi, 180)
        sphere_x = current_params.radius0 * np.cos(theta_reference_sphere)
        sphere_y = current_params.radius0 * np.sin(theta_reference_sphere)
        self.reference_sphere_line.set_data(sphere_x, sphere_y)
        self.reference_sphere_line.set_label(f'R₀={current_params.radius0:.2f} fm')

        max_val = max(np.max(np.abs(z_fos_cylindrical)), np.max(np.abs(rho_fos_cylindrical))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        fos_shape_volume = calculator_fos.calculate_volume_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)
        fos_cylindrical_surface_area = calculator_fos.calculate_surface_area_in_cylindrical_coordinates(z_fos_cylindrical, rho_fos_cylindrical)
        max_z = np.max(z_fos_cylindrical)
        min_z = np.min(z_fos_cylindrical)
        max_rho = np.max(rho_fos_cylindrical)
        length_along_z_axis = float(z_fos_cylindrical[-1] - z_fos_cylindrical[0])
        neck_radius = min(rho_fos_cylindrical[len(rho_fos_cylindrical) // 2 - 10:len(rho_fos_cylindrical) // 2 + 10]) if current_params.a4 > 0 else max_rho

        basic_info = (
            f"Number of Shape Points: {current_number_of_points}\n"
            f"R₀ = 1.16 * {current_params.nucleons:.0f}^(1/3) = {current_params.radius0:.3f} fm\n"
            f"\nParameter Relations:\n"
            f"a₂ (From Volume Conservation)= a₄/3 - a₆/5\n"
            f"a₂ = {current_params.a4:.3f} / 3.0 - {current_params.a6:.3f} / 5.0 = {current_params.a2:.3f}\n"
            f"q₂ (c*) = c - 1.0 - 1.5a₄\n"
            f"q₂ = {current_params.c_elongation:.3f} - 1.0 - 1.5 × {current_params.a4:.3f} = {current_params.q2:.3f}\n"
            f"Aₕ = A/2 * (1.0 + a₃)\n"
            f"Aₕ = {current_params.nucleons:.3f} / 2 * (1.0 + {current_params.a3:.3f}) = {current_params.nucleons / 2.0 * (1.0 + current_params.a3):.3f}\n"
            f"\nCenter of Mass:\n"
            f"Calculated CM (FoS Cylindrical): {center_of_mass_fos:.3f} fm\n"
            f"Theoretical z_shift = {current_params.z_sh:.3f} fm\n"
            f"\nVolume Information:\n"
            f"Reference Sphere Volume: {current_params.sphere_volume:.3f} fm³\n"
            f"FoS (Cylindrical) Shape Volume: {fos_shape_volume:.3f} fm³\n"
            f"\nSurface Information:\n"
            f"Reference Sphere Surface Area: {current_params.sphere_surface_area:.3f} fm²\n"
            f"FoS (Cylindrical) Shape Surface Area: {fos_cylindrical_surface_area:.3f} fm²\n"
            f"\nShape Dimensions:\n"
            f"Max z: {max_z:.3f} fm\n"
            f"Min z: {min_z:.3f} fm\n"
            f"Max ρ: {max_rho:.3f} fm\n"
            f"Length along the z axis: {length_along_z_axis:.3f} fm\n"
            f"Neck Radius: {neck_radius:.3f} fm\n"
            f"Calculated c (z_length/2R₀): {length_along_z_axis / (2 * current_params.radius0):.3f}\n"
        )

        if self.show_beta_fitting and len(radius_fos) > 0:
            beta_info = (
                f"\nShift Information:\n"
                f"Shift Needed for Conversion: {cumulative_shift:.3f} fm\n"
                f"\nAdditional Center of Mass:\n"
                f"Calculated CM (FoS Spherical Fit): {center_of_mass_spherical_fit - cumulative_shift:.3f} fm\n"
                f"Calculated CM (Beta Analytical): {center_of_mass_beta_fit - cumulative_shift:.3f} fm\n"
                f"Calculated CM (Beta Fitted): {center_of_mass_beta_fitted - cumulative_shift:.3f} fm\n"
                f"Calculated CM (FoS Spherical Fit, Shifted): {center_of_mass_spherical_fit:.3f} fm\n"
                f"\nAdditional Volume Information:\n"
                f"FoS (Spherical) Shape Volume: {fos_spherical_volume:.3f} fm³\n"
                f"Beta Shape Volume (Analytical): {beta_volume:.3f} fm³\n"
                f"Beta Shape Volume (Fitted): {beta_volume_fitted:.3f} fm³\n"
                f"\nAdditional Surface Information:\n"
                f"FoS (Spherical) Shape Surface Area: {fos_spherical_surface_area:.3f} fm²\n"
                f"Beta Shape Surface Area (Analytical): {beta_surface_area:.3f} fm² ({beta_surface_area / fos_cylindrical_surface_area * 100:.3f}% of FoS)\n"
                f"Beta Shape Surface Area (Fitted): {beta_surface_area_fitted:.3f} fm² ({beta_surface_area_fitted / fos_cylindrical_surface_area * 100:.3f}% of FoS)\n"
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
            info_text = basic_info + beta_info
        else:
            info_text = basic_info

        for artist in self.ax_text.texts:
            artist.remove()

        if self.show_text_info:
            self.ax_text.text(-0.05, 1.0, info_text, transform=self.ax_text.transAxes,
                              fontsize=6, verticalalignment='top', horizontalalignment='left',
                              bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}, fontfamily='monospace')

        if self.show_beta_fitting:
            beta_text_obj = self.ax_text.text(0.55, 1.0, f"Significant Beta Parameters (>0.001):\n{significant_beta_parameters}",
                                              transform=self.ax_text.transAxes,
                                              fontsize=8, verticalalignment='top', horizontalalignment='left',
                                              bbox={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.5}, fontfamily='monospace')
            beta_text_obj._beta_text = True

        self.ax_plot.set_title(f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, A={current_params.nucleons})', fontsize=14)
        self.ax_plot.legend(loc='upper right', bbox_to_anchor=(-0.05, 1))
        self.fig.canvas.draw_idle()

    def _clear_beta_lines(self):
        self.line_fos_spherical.set_data([], [])
        self.line_fos_spherical_mirror.set_data([], [])
        self.line_fos_spherical_shifted_for_conversion.set_data([], [])
        self.line_fos_spherical_shifted_for_conversion_mirror.set_data([], [])
        self.line_beta.set_data([], [])
        self.line_beta_mirror.set_data([], [])
        self.line_beta_fitted.set_data([], [])
        self.line_beta_fitted_mirror.set_data([], [])

    def run(self):
        self.update_plot(None)
        plt.show(block=True)
