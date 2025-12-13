"""FoS Shape Plotter with Beta Deformation Fitting UI."""
from typing import Any, Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, Slider

from src.parameterizations.beta import BetaDeformationCalculator, ShapeComparisonMetrics
from src.parameterizations.fos import FoSParameters, FoSShapeCalculator
from src.utilities.converter import CylindricalToSphericalConverter

# Set backend
matplotlib.use('TkAgg')

# Module-level constants
BATCH_SIZE: Final[int] = 32  # Number of beta parameters to add per iteration
MAX_BETA: Final[int] = 1024  # Maximum number of beta parameters for fitting
RMSE_THRESHOLD: Final[float] = 0.2  # RMSE convergence threshold in fm
LINF_THRESHOLD: Final[float] = 0.5  # L-infinity convergence threshold in fm
SURFACE_DIFF_THRESHOLD: Final[float] = 0.5  # Surface area difference threshold in fm^2


class FoSShapePlotter:
    """Interactive FoS Shape Plotter with Beta Deformation Fitting."""

    def __init__(self) -> None:
        # --- CONFIGURATION ---
        self.n_calc: int = 7200  # High precision for physics/fitting
        self.n_plot: int = 360  # Sufficient for visual smoothness

        # Default parameters
        self.params = FoSParameters()

        # UI State
        self.show_text_info = True
        self.show_beta_approx = False
        self.updating = False
        self.slider_buttons: dict[str, dict[str, Button | Slider]] = {}

        # Plot elements
        self.fig: Figure | None = None
        self.ax_plot: Axes | None = None
        self.ax_text: Axes | None = None
        self.lines: dict[str, Line2D] = {}

        # Control elements
        # Sliders
        self.sl_z: Slider | None = None
        self.sl_n: Slider | None = None
        self.sl_c: Slider | None = None
        self.sl_a3: Slider | None = None
        self.sl_a4: Slider | None = None
        self.sl_a5: Slider | None = None
        self.sl_a6: Slider | None = None
        # Checkboxes and Buttons
        self.chk_text: CheckButtons | None = None
        self.chk_beta: CheckButtons | None = None
        self.btn_reset: Button | None = None
        self.btn_save: Button | None = None
        self.btn_print_beta: Button | None = None

        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self) -> None:
        """Creates the main figure and axes for plotting."""
        self.fig = plt.figure(figsize=(16.0, 8.0))
        self.ax_plot = self.fig.add_subplot(121)
        self.ax_text = self.fig.add_subplot(122)

        if self.ax_text is None or self.ax_plot is None:
            raise RuntimeError("Failed to create axes")

        self.ax_text.axis('off')

        plt.subplots_adjust(left=0.13, bottom=0.38, right=0.97, top=0.97, wspace=0.1)
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_xlabel('z (fm)')
        self.ax_plot.set_ylabel('ρ (fm)')

        # Initialize lines
        self.lines['fos'] = self.ax_plot.plot([], [], 'b-', label='FoS Shape', lw=2)[0]
        self.lines['fos_m'] = self.ax_plot.plot([], [], 'b-', lw=2)[0]
        self.lines['beta'] = self.ax_plot.plot([], [], 'r--', label='Beta Approx', lw=2, alpha=0.7)[0]
        self.lines['beta_m'] = self.ax_plot.plot([], [], 'r--', lw=2, alpha=0.7)[0]
        self.lines['ref_sphere'] = self.ax_plot.plot([], [], '--', color='gray', alpha=0.5)[0]

        self.ax_plot.legend(loc='upper right')

    def create_slider(self, name: str, y_pos: float, label: str, vmin: float, vmax: float, vinit: float, step: float) -> Slider:
        ax_dec = plt.axes((0.20, y_pos, 0.016, 0.024))
        ax_sl = plt.axes((0.25, y_pos, 0.5, 0.024))
        ax_inc = plt.axes((0.78, y_pos, 0.016, 0.024))

        slider = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=step)
        btn_dec = Button(ax_dec, '-')
        btn_inc = Button(ax_inc, '+')

        self.slider_buttons[name] = {'slider': slider, 'dec': btn_dec, 'inc': btn_inc}
        return slider

    def setup_controls(self) -> None:
        """Sets up sliders, buttons, and checkboxes."""
        y_start = 0.02
        spacing = 0.03

        # Sliders
        self.sl_z = self.create_slider('z', y_start, 'Z', 82, 120, 92, 1)
        self.sl_n = self.create_slider('n', y_start + spacing, 'N', 120, 180, 144, 1)
        self.sl_c = self.create_slider('c', y_start + 2 * spacing, 'c', 0.5, 3.0, 1.0, 0.01)
        self.sl_a3 = self.create_slider('a3', y_start + 3 * spacing, 'a3', -0.6, 0.6, 0.0, 0.01)
        self.sl_a4 = self.create_slider('a4', y_start + 4 * spacing, 'a4', -0.75, 0.75, 0.0, 0.01)
        self.sl_a5 = self.create_slider('a5', y_start + 5 * spacing, 'a5', -0.5, 0.5, 0.0, 0.01)
        self.sl_a6 = self.create_slider('a6', y_start + 6 * spacing, 'a6', -0.5, 0.5, 0.0, 0.01)

        # Checkboxes and Buttons
        self.chk_text = CheckButtons(plt.axes((0.82, 0.45, 0.08, 0.032)), ['Show Info'], [True])
        self.chk_beta = CheckButtons(plt.axes((0.82, 0.42, 0.08, 0.032)), ['Show Beta'], [False])
        self.btn_reset = Button(plt.axes((0.82, 0.37, 0.08, 0.032)), 'Reset')
        self.btn_save = Button(plt.axes((0.82, 0.32, 0.08, 0.032)), 'Save Plot')
        self.btn_print_beta = Button(plt.axes((0.82, 0.27, 0.08, 0.032)), 'Print Betas')

    def setup_event_handlers(self) -> None:
        """Sets up event handlers for sliders and buttons."""
        if self.sl_z is None: return

        sliders = [self.sl_z, self.sl_n, self.sl_c, self.sl_a3, self.sl_a4, self.sl_a5, self.sl_a6]
        for sl in sliders:
            if sl is not None:
                sl.on_changed(self.update_plot)

        for name, items in self.slider_buttons.items():
            slider = items['slider']
            btn_dec = items['dec']
            btn_inc = items['inc']

            # Button.on_clicked expects a callback.
            # Using default arg 's=slider' to capture the specific slider instance
            btn_dec.on_clicked(lambda _, s=slider: s.set_val(max(s.val - s.valstep, s.valmin)))
            btn_inc.on_clicked(lambda _, s=slider: s.set_val(min(s.val + s.valstep, s.valmax)))

        if self.chk_text: self.chk_text.on_clicked(self.toggle_text)
        if self.chk_beta: self.chk_beta.on_clicked(self.toggle_beta)
        if self.btn_reset: self.btn_reset.on_clicked(self.reset_values)
        if self.btn_save: self.btn_save.on_clicked(self.save_plot)
        if self.btn_print_beta: self.btn_print_beta.on_clicked(self.print_beta_parameters)

    def toggle_text(self, _: Any) -> None:
        self.show_text_info = not self.show_text_info
        self.update_plot(None)

    def toggle_beta(self, _: Any) -> None:
        self.show_beta_approx = not self.show_beta_approx
        self.update_plot(None)

    def _fit_beta_iteratively(
            self,
            theta: np.ndarray,
            r_original: np.ndarray,
            fos_spherical_surface: float,
            nucleons: int
    ) -> tuple[dict[int, float], int, np.ndarray, np.ndarray, ShapeComparisonMetrics, float, bool]:
        """Iteratively fit beta parameters until convergence criteria are met.

        Args:
            theta: Theta values for the original shape.
            r_original: r(theta) values for the original FoS shape.
            fos_spherical_surface: Surface area of FoS shape in spherical coordinates.
            nucleons: Number of nucleons (A).

        Returns:
            Tuple of (beta_parameters, l_max, theta_reconstructed, r_reconstructed, errors, surface_diff, converged).
        """

        l_max: int = BATCH_SIZE
        converged: bool = False
        beta_parameters: dict[int, float] = {}
        theta_reconstructed: np.ndarray = np.array([])
        r_reconstructed: np.ndarray = np.array([])
        errors: ShapeComparisonMetrics = {
            'rmse': float('inf'),
            'chi_squared': float('inf'),
            'chi_squared_reduced': float('inf'),
            'l_infinity': float('inf'), }
        surface_diff: float = np.inf

        while l_max <= MAX_BETA:
            beta_calculator = BetaDeformationCalculator(theta, r_original, nucleons)
            beta_parameters = beta_calculator.calculate_beta_parameters(l_max)
            theta_reconstructed, r_reconstructed = beta_calculator.reconstruct_shape(beta_parameters, self.n_calc)

            errors = beta_calculator.calculate_errors(r_original, r_reconstructed, n_params=l_max)
            beta_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_reconstructed, r_reconstructed)
            surface_diff = abs(beta_surface - fos_spherical_surface)

            rmse = errors['rmse']
            l_inf = errors['l_infinity']

            if rmse < RMSE_THRESHOLD and l_inf < LINF_THRESHOLD and surface_diff < SURFACE_DIFF_THRESHOLD:
                converged = True
                break

            if l_max >= MAX_BETA:
                break

            l_max += BATCH_SIZE

        return beta_parameters, l_max, theta_reconstructed, r_reconstructed, errors, surface_diff, converged

    def reset_values(self, _: Any) -> None:
        if self.sl_z is None or self.sl_n is None or self.sl_c is None:
            return

        self.updating = True
        self.sl_z.set_val(92)
        self.sl_n.set_val(144)
        self.sl_c.set_val(1.0)
        # Assuming other sliders exist if these do
        if self.sl_a3: self.sl_a3.set_val(0.0)
        if self.sl_a4: self.sl_a4.set_val(0.0)
        if self.sl_a5: self.sl_a5.set_val(0.0)
        if self.sl_a6: self.sl_a6.set_val(0.0)
        self.updating = False
        self.update_plot(None)

    def save_plot(self, _: Any) -> None:
        if self.sl_z is None or self.sl_n is None or self.fig is None:
            return

        file_name = f"fos_shape_Z{int(self.sl_z.val)}_N{int(self.sl_n.val)}_c{self.sl_c.val:.2f}_a3{self.sl_a3.val:.2f}_a4{self.sl_a4.val:.2f}_a5{self.sl_a5.val:.2f}_a6{self.sl_a6.val:.2f}.png"
        self.fig.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Saved {file_name}")

    def print_beta_parameters(self, _: Any) -> None:
        """Calculate and print beta parameters to the command line using iterative fitting."""
        calc = FoSShapeCalculator(self.params)
        z_fos_calc, rho_fos_calc = calc.calculate_shape(self.n_calc)

        z_work = z_fos_calc.copy()
        conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)

        shift = 0.0
        if not conv.is_unambiguously_convertible(self.n_calc):
            direction = -1.0 if self.params.z_sh >= 0 else 1.0
            max_shift = (float(np.max(z_fos_calc)) - float(np.min(z_fos_calc))) / 2.0
            while abs(shift) < max_shift:
                shift += direction * 0.1
                z_work = z_fos_calc + shift
                conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)
                if conv.is_unambiguously_convertible(self.n_calc):
                    break

        if conv.is_unambiguously_convertible(self.n_calc):
            theta_calc, r_fos_sph_calc = conv.convert_to_spherical(self.n_calc)
            sph_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_calc, r_fos_sph_calc)

            # Use iterative fitting
            betas, l_max, _, _, errors, surface_diff, converged = \
                self._fit_beta_iteratively(theta_calc, r_fos_sph_calc, sph_surface, self.params.nucleons)

            status = "Converged" if converged else "Max l reached"
            print("\n" + "=" * 50)
            print(f"Fitted Beta Parameters ({status})")
            print("=" * 50)
            print(f"Z={self.params.protons}, N={self.params.neutrons}, A={self.params.nucleons}")
            print(f"FoS: c={self.params.c_elongation:.4f}, a3={self.params.get_coefficient(3):.4f}, "
                  f"a4={self.params.get_coefficient(4):.4f}, a5={self.params.get_coefficient(5):.4f}, "
                  f"a6={self.params.get_coefficient(6):.4f}")
            print("-" * 50)
            print(f"l_max:     {l_max}")
            print(f"RMSE:      {errors['rmse']:.4f} fm")
            print(f"L_inf:     {errors['l_infinity']:.4f} fm")
            print(f"Surface Δ: {surface_diff:.4f} fm^2")
            print("-" * 50)
            for l, val in sorted(betas.items()):
                print(f"beta_{l:2d} = {val:+.6f}")
            print("=" * 50 + "\n")
        else:
            print("Shape is not convertible to spherical coordinates.")

    def update_plot(self, _: Any) -> None:
        """Updates the plot based on current slider values."""
        if self.updating:
            return

        # Ensure critical attributes are initialized
        if (self.sl_z is None or self.sl_n is None or self.sl_c is None or
                self.sl_a3 is None or self.sl_a4 is None or self.sl_a5 is None or
                self.sl_a6 is None or
                self.ax_plot is None or self.ax_text is None or self.fig is None):
            return

        # 1. Update Params
        self.params.protons = int(self.sl_z.val)
        self.params.neutrons = int(self.sl_n.val)
        self.params.c_elongation = self.sl_c.val
        self.params.set_coefficient(3, self.sl_a3.val)
        self.params.set_coefficient(4, self.sl_a4.val)
        self.params.set_coefficient(5, self.sl_a5.val)
        self.params.set_coefficient(6, self.sl_a6.val)

        # 2. Calculate FoS (HIGH PRECISION)
        calc = FoSShapeCalculator(self.params)
        z_fos_calc, rho_fos_calc = calc.calculate_shape(self.n_calc)

        # 3. Downsample for Plotting
        idx = np.linspace(0, self.n_calc - 1, self.n_plot, dtype=int)

        self.lines['fos'].set_data(z_fos_calc[idx], rho_fos_calc[idx])
        self.lines['fos_m'].set_data(z_fos_calc[idx], -rho_fos_calc[idx])

        # Update Reference Sphere
        theta_ref = np.linspace(0, 2 * np.pi, 180)
        self.lines['ref_sphere'].set_data(
            self.params.radius0 * np.cos(theta_ref),
            self.params.radius0 * np.sin(theta_ref)
        )

        # Calculate FoS volume and surface area (cylindrical)
        fos_volume = FoSShapeCalculator.calculate_volume(z_fos_calc, rho_fos_calc)
        fos_surface = FoSShapeCalculator.calculate_surface_area(z_fos_calc, rho_fos_calc)

        # Spherical conversion for volume/surface calculation
        beta_fit_text: str = ""
        metrics_text: str = ""

        # Always try to convert to spherical for volume/surface calculation
        z_work = z_fos_calc.copy()
        conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)

        shift = 0.0
        if not conv.is_unambiguously_convertible(self.n_calc):
            direction = -1.0 if self.params.z_sh >= 0 else 1.0
            max_shift = (float(np.max(z_fos_calc)) - float(np.min(z_fos_calc))) / 2.0
            while abs(shift) < max_shift:
                shift += direction * 0.1
                z_work = z_fos_calc + shift
                conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)
                if conv.is_unambiguously_convertible(self.n_calc):
                    break

        if conv.is_unambiguously_convertible(self.n_calc):
            theta_calc, r_fos_sph_calc = conv.convert_to_spherical(self.n_calc)

            # Calculate spherical volume/surface
            sph_volume = BetaDeformationCalculator.calculate_volume_spherical(theta_calc, r_fos_sph_calc)
            sph_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_calc, r_fos_sph_calc)
            spherical_text: str = (f"FoS Shape (spherical):\n"
                              f"  Volume:  {sph_volume:.2f} fm^3\n"
                              f"  Surface: {sph_surface:.2f} fm^2\n\n")

            if self.show_beta_approx:
                # Iterative Beta Fitting (adds betas in batches of 16 until convergence)
                betas, l_max, theta_rec_calc, r_rec_calc, errors, surface_diff, converged = \
                    self._fit_beta_iteratively(theta_calc, r_fos_sph_calc, sph_surface, self.params.nucleons)

                # Calculate beta fit volume/surface
                beta_volume = BetaDeformationCalculator.calculate_volume_spherical(theta_rec_calc, r_rec_calc)
                beta_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_rec_calc, r_rec_calc)
                beta_fit_text = (f"Beta Fit Shape (l_max={l_max}):\n"
                                 f"  Volume:  {beta_volume:.2f} fm^3\n"
                                 f"  Surface: {beta_surface:.2f} fm^2\n\n")

                # Build metrics text with convergence status
                status = "Converged" if converged else "Max l reached"
                metrics_text = (f"Fit Metrics (N={self.n_calc}, {status}):\n"
                                f"  l_max:       {l_max}\n"
                                f"  RMSE:        {errors['rmse']:.4f} fm\n"
                                f"  Chi^2:       {errors['chi_squared']:.4f}\n"
                                f"  Chi^2 (Red): {errors['chi_squared_reduced']:.6f}\n"
                                f"  L_inf:       {errors['l_infinity']:.4f} fm\n"
                                f"  Surface Δ:   {surface_diff:.4f} fm^2")

                # Reconstruct for Plotting (LOW PRECISION via Downsampling)
                theta_plot = theta_rec_calc[idx]
                r_rec_plot = r_rec_calc[idx]

                z_beta = r_rec_plot * np.cos(theta_plot) - shift
                rho_beta = r_rec_plot * np.sin(theta_plot)

                self.lines['beta'].set_data(z_beta, rho_beta)
                self.lines['beta_m'].set_data(z_beta, -rho_beta)
            else:
                self.lines['beta'].set_data([], [])
                self.lines['beta_m'].set_data([], [])
        else:
            spherical_text = "Shape not convertible to spherical.\n\n"
            self.lines['beta'].set_data([], [])
            self.lines['beta_m'].set_data([], [])

        # Auto-scale
        limit = max(np.max(np.abs(z_fos_calc)), np.max(rho_fos_calc)) * 1.2
        self.ax_plot.set_xlim(-limit, limit)
        self.ax_plot.set_ylim(-limit, limit)
        self.ax_plot.set_title(f"Z={self.params.protons} N={self.params.neutrons} (FoS)")

        # Update Text
        self.ax_text.clear()
        self.ax_text.axis('off')
        if self.show_text_info:
            info = (f"FoS Parameters:\n"
                    f"c={self.params.c_elongation:.3f}, q2={self.params.q2:.3f}\n"
                    f"a3={self.params.get_coefficient(3):.3f}, a4={self.params.get_coefficient(4):.3f}\n"
                    f"R0={self.params.radius0:.3f}\n\n"
                    f"Spherical Nucleus:\n"
                    f"  Volume:  {self.params.sphere_volume:.2f} fm^3\n"
                    f"  Surface: {self.params.sphere_surface_area:.2f} fm^2\n\n"
                    f"FoS Shape (cylindrical):\n"
                    f"  Volume:  {fos_volume:.2f} fm^3\n"
                    f"  Surface: {fos_surface:.2f} fm^2\n\n"
                    + spherical_text + beta_fit_text + metrics_text)

            self.ax_text.text(0, 1, info, va='top', fontfamily='monospace', fontsize=9)

        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """Starts the interactive plotter."""
        self.update_plot(None)
        plt.show()
