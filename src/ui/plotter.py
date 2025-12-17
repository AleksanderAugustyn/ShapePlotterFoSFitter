"""FoS Shape Plotter with Beta Deformation Fitting UI."""
from typing import Any, Callable, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons, Slider

from src.parameterizations.beta import (
    BetaDeformationCalculator,
    BetaFitResult,
    IterativeBetaFitter,
)
from src.parameterizations.fos import FoSParameters, FoSShapeCalculator
from src.utilities.converter import CylindricalToSphericalConverter

# Set backend
matplotlib.use('TkAgg')


class SliderControls(TypedDict):
    """Dictionary for slider controls."""
    slider: Slider
    dec: Button
    inc: Button


class FoSShapePlotter:
    """Interactive FoS Shape Plotter with Beta Deformation Fitting."""

    def __init__(self) -> None:
        # --- CONFIGURATION ---
        self.n_calc: int = 7200  # High precision for physics/fitting
        self.n_plot: int = 720  # Sufficient for visual smoothness

        # Default parameters
        self.params = FoSParameters()

        # Beta fitter instance
        self.beta_fitter = IterativeBetaFitter()

        # UI State
        self.show_text_info = True
        self.show_beta_approx = False
        self.updating = False
        self.slider_buttons: dict[str, SliderControls] = {}

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

        # Spherical nucleus cache (recalculated only when Z or N change)
        self._cached_sphere_volume: float | None = None
        self._cached_sphere_surface: float | None = None
        self._cached_Z: int = self.params.protons
        self._cached_N: int = self.params.neutrons

        # Beta fitting cache (for Print Betas button)
        self._last_beta_result: BetaFitResult | None = None

        # Warning text for invalid shapes
        self.warning_text: Text | None = None

        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

        # Initial button state: Print Betas grayed out when Show Beta is off
        if self.btn_print_beta:
            self.btn_print_beta.ax.set_alpha(0.3)

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
        self.lines['fos'] = self.ax_plot.plot([], [], 'b-', label='FoS Shape', lw=3)[0]
        self.lines['fos_m'] = self.ax_plot.plot([], [], 'b-', lw=3)[0]
        self.lines['fos_sph'] = self.ax_plot.plot([], [], 'g--', label='FoS (Shifted Frame)', lw=2, alpha=0.7)[0]
        self.lines['fos_sph_m'] = self.ax_plot.plot([], [], 'g--', lw=2, alpha=0.7)[0]
        self.lines['beta'] = self.ax_plot.plot([], [], 'r-', label='Beta Approx', lw=1.5, alpha=0.7)[0]
        self.lines['beta_m'] = self.ax_plot.plot([], [], 'r-', lw=1.5, alpha=0.7)[0]
        self.lines['ref_sphere'] = self.ax_plot.plot([], [], '--', color='gray', alpha=0.5)[0]
        self.lines['neck'] = self.ax_plot.plot([], [], 'm-', lw=2, alpha=0.8)[0]
        self.lines['neck_m'] = self.ax_plot.plot([], [], 'm-', lw=2, alpha=0.8)[0]
        self.lines['l_inf_marker'] = self.ax_plot.plot([], [], 'o-', color='orange', lw=2, markersize=6, alpha=0.9)[0]
        self.lines['l_inf_marker_m'] = self.ax_plot.plot([], [], 'o-', color='orange', lw=2, markersize=6, alpha=0.9)[0]

        self.ax_plot.legend(loc='upper right')

    def create_slider(self, name: str, y_pos: float, label: str,
                      vmin: float, vmax: float, vinit: float, step: float,
                      markers: list[float] | None = None) -> Slider:
        """
        Creates a slider with decrement/increment buttons and optional limit markers.

        Args:
            name: ID for the slider.
            y_pos: Vertical position on the figure.
            label: Text label.
            vmin: Minimum value.
            vmax: Maximum value.
            vinit: Initial value.
            step: Step size.
            markers: List of values to draw vertical red dotted lines at (practical limits).
        """
        ax_dec = plt.axes((0.20, y_pos, 0.016, 0.024))
        ax_sl = plt.axes((0.25, y_pos, 0.5, 0.024))
        ax_inc = plt.axes((0.80, y_pos, 0.016, 0.024))

        slider = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=step)
        btn_dec = Button(ax_dec, '-')
        btn_inc = Button(ax_inc, '+')

        # --- Visual Indicators ---
        # 1. Permanent mark for the initial value (reset point) - Thin Grey
        slider.ax.vlines(vinit, 0, 1, color='k', linestyle='-', alpha=0.3, linewidth=1)

        # 2. Markers for practical limits - Red Dotted
        if markers:
            slider.ax.vlines(markers, 0, 1, color='r', linestyle=':', alpha=0.7, linewidth=1.5)

        self.slider_buttons[name] = {'slider': slider, 'dec': btn_dec, 'inc': btn_inc}
        return slider

    def setup_controls(self) -> None:
        """Sets up sliders, buttons, and checkboxes."""
        y_start = 0.02
        spacing = 0.03

        # Sliders
        # Note: 'markers' argument adds the red dotted lines for practical limits

        self.sl_z = self.create_slider('z', y_start, 'Z', 82, 120, 92, 1)
        self.sl_n = self.create_slider('n', y_start + spacing, 'N', 120, 180, 144, 1)
        self.sl_c = self.create_slider('c', y_start + 2 * spacing, 'c', 0.5, 3.5, 1.0, 0.01, markers=[1.0, 3.0])
        self.sl_a3 = self.create_slider('a3', y_start + 3 * spacing, 'a3', -0.6, 0.6, 0.0, 0.01, markers=[0.0, 0.5])
        self.sl_a4 = self.create_slider('a4', y_start + 4 * spacing, 'a4', -0.75, 0.75, 0.0, 0.01, markers=[-0.2, 0.72])
        self.sl_a5 = self.create_slider('a5', y_start + 5 * spacing, 'a5', -0.5, 0.5, 0.0, 0.01, markers=[-0.2, 0.2])
        self.sl_a6 = self.create_slider('a6', y_start + 6 * spacing, 'a6', -0.5, 0.5, 0.0, 0.01, markers=[-0.2, 0.2])

        # Checkboxes and Buttons
        self.chk_text = CheckButtons(plt.axes((0.82, 0.45, 0.08, 0.032)), ['Show Info'], [True])
        self.chk_beta = CheckButtons(plt.axes((0.82, 0.42, 0.08, 0.032)), ['Fit Beta'], [False])
        self.btn_reset = Button(plt.axes((0.82, 0.37, 0.08, 0.032)), 'Reset')
        self.btn_save = Button(plt.axes((0.82, 0.32, 0.08, 0.032)), 'Save Plot')
        self.btn_print_beta = Button(plt.axes((0.82, 0.27, 0.08, 0.032)), 'Print Betas')

    @staticmethod
    def _make_decrement(slider: Slider) -> Callable[[Any], None]:
        def handler(_: Any) -> None:
            val: float = float(slider.val)
            step: float = float(slider.valstep)  # type: ignore[arg-type]
            vmin: float = float(slider.valmin)
            slider.set_val(max(val - step, vmin))

        return handler

    @staticmethod
    def _make_increment(slider: Slider) -> Callable[[Any], None]:
        def handler(_: Any) -> None:
            val: float = float(slider.val)
            step: float = float(slider.valstep)  # type: ignore[arg-type]
            vmax: float = float(slider.valmax)
            slider.set_val(min(val + step, vmax))

        return handler

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

            btn_dec.on_clicked(self._make_decrement(slider))
            btn_inc.on_clicked(self._make_increment(slider))

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
        # Update Print Betas button appearance based on state
        if self.btn_print_beta:
            self.btn_print_beta.ax.set_alpha(1.0 if self.show_beta_approx else 0.3)
            self.fig.canvas.draw_idle() if self.fig else None
        self.update_plot(None)

    def _get_sphere_properties(self) -> tuple[float | None, float | None]:
        """Return cached sphere volume/surface, recalculating only if Z or N changed."""
        if (self._cached_sphere_volume is None or
                self._cached_Z != self.params.protons or
                self._cached_N != self.params.neutrons):
            self._cached_sphere_volume = self.params.sphere_volume
            self._cached_sphere_surface = self.params.sphere_surface_area
            self._cached_Z = self.params.protons
            self._cached_N = self.params.neutrons

        volume = float(self._cached_sphere_volume) if self._cached_sphere_volume is not None else None
        surface = float(self._cached_sphere_surface) if self._cached_sphere_surface is not None else None
        return volume, surface

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
        if (self.sl_z is None or self.sl_n is None or self.sl_c is None or
                self.sl_a3 is None or self.sl_a4 is None or
                self.sl_a5 is None or self.sl_a6 is None or self.fig is None):
            return

        file_name = f"fos_shape_Z{int(self.sl_z.val)}_N{int(self.sl_n.val)}_c{self.sl_c.val:.2f}_a3{self.sl_a3.val:.2f}_a4{self.sl_a4.val:.2f}_a5{self.sl_a5.val:.2f}_a6{self.sl_a6.val:.2f}.png"
        self.fig.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Saved {file_name}")

    def print_beta_parameters(self, _: Any) -> None:
        """Print cached beta parameters to the command line.

        Requires Show Beta to be enabled to have cached values available.
        """
        if not self.show_beta_approx:
            print("Enable 'Show Beta' first to calculate beta parameters.")
            return

        if self._last_beta_result is None:
            print("No beta parameters available. Shape may not be convertible to spherical coordinates.")
            return

        # Read from the cached result
        result = self._last_beta_result
        betas = result.beta_parameters
        l_max = result.l_max
        errors = result.errors
        converged = result.converged

        status = "Converged" if converged else "Max l reached"
        print("\n" + "=" * 50)
        print(f"Fitted Beta Parameters ({status})")
        print("=" * 50)
        print(f"Z={self.params.protons}, N={self.params.neutrons}, A={self.params.nucleons}")
        print(f"FoS: c={self.params.c_elongation:.4f}, a3={self.params.get_coefficient(3):.4f}, "
              f"a4={self.params.get_coefficient(4):.4f}, a5={self.params.get_coefficient(5):.4f}, "
              f"a6={self.params.get_coefficient(6):.4f}")
        print("-" * 50)
        print(f"l_max:       {l_max}")
        if errors:
            print(f"RMSE ρ:      {errors['rmse_rho']:.4f} fm")
            print(f"L_inf ρ:     {errors['l_infinity_rho']:.4f} fm @ z={errors['l_infinity_z']:.2f} fm")
            print(f"Surface Δ:   {errors['surface_diff']:.4f} fm²")
        print("-" * 50)
        for l, val in sorted(betas.items()):
            print(f"beta_{l:2d} = {val:+.6f}")
        print("=" * 50 + "\n")

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
        z_fos_calc, rho_fos_calc, is_valid = calc.calculate_shape(self.n_calc)

        fos_volume, fos_surface, fos_com = calc.calculate_metrics_fast(self.n_calc)

        # 2a. Neck detection for shapes with pronounced necks (a4 + a6 >= 0.4)
        neck_rho: float | None = None
        neck_z: float | None = None
        fragment_ratio_text: str = ""
        a4 = self.params.get_coefficient(4)
        a6 = self.params.get_coefficient(6)
        if a4 + a6 >= 0.4:
            neck_result = calc.find_neck_thickness(self.n_calc)
            if neck_result is not None:
                neck_rho, neck_z = neck_result
                # Calculate fragment volumes
                vol_left, vol_right = calc.calculate_fragment_volumes(neck_z, self.n_calc)
                vol_heavier = max(vol_left, vol_right)
                vol_lighter = min(vol_left, vol_right)
                if vol_lighter > 1e-10:
                    vol_heavier = vol_heavier / self._cached_sphere_volume if self._cached_sphere_volume is not None else -1.0
                    vol_lighter = vol_lighter / self._cached_sphere_volume if self._cached_sphere_volume is not None else -1.0
                    fragment_ratio_text = f"  Frag vol ratio:  {vol_heavier:.2f}:{vol_lighter:.2f}\n"

        # --- Validity Check ---
        # Shape is invalid if rho^2 <= 0 at any interior point (volume/elongation not conserved)
        if not is_valid:
            if self.warning_text is None:
                self.warning_text = self.ax_plot.text(
                    0.5, 0.5, "SHAPE INVALID\n(ρ² ≤ 0 at interior)",
                    transform=self.ax_plot.transAxes,
                    color='red', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
                )
            self.warning_text.set_visible(True)
        else:
            if self.warning_text is not None:
                self.warning_text.set_visible(False)

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

        # Update Neck Line (vertical line from axis to neck surface)
        if neck_rho is not None and neck_z is not None:
            self.lines['neck'].set_data([neck_z, neck_z], [0, neck_rho])
            self.lines['neck_m'].set_data([neck_z, neck_z], [0, -neck_rho])
        else:
            self.lines['neck'].set_data([], [])
            self.lines['neck_m'].set_data([], [])

        # Initialize text sections
        spherical_text: str = ""
        conversion_metrics_text: str = ""
        beta_fit_text: str = ""
        beta_vs_fos_text: str = ""

        # Only convert to spherical and fit betas when Show Beta is enabled
        if self.show_beta_approx:
            print(f"Calculating spherical conversion and fitting beta parameters for current shape: "
                  f"Z={self.params.protons}, N={self.params.neutrons}, "
                  f"c={self.params.c_elongation:.2f}, a3={self.params.get_coefficient(3):.2f}, "
                  f"a4={self.params.get_coefficient(4):.2f}, a5={self.params.get_coefficient(5):.2f}, "
                  f"a6={self.params.get_coefficient(6):.2f}")

            # Use optimized shift finder for better beta fitting
            conv, shift, opt_metrics = CylindricalToSphericalConverter.find_optimal_beta_shift(
                z_fos_calc,
                rho_fos_calc,
                self.params.nucleons,
                z_sh=self.params.z_sh,
                n_check=self.n_calc,
                search_range=0.5,  # Search ±0.5 fm around base shift
                search_step=0.05,  # Test every 0.05 fm
                l_max_test=64  # Quick test with the first N parameters
            )

            if conv.is_unambiguously_convertible(self.n_calc):
                theta_calc, r_fos_sph_calc = conv.convert_to_spherical(self.n_calc)

                # Calculate spherical volume/surface/CoM
                sph_volume = BetaDeformationCalculator.calculate_volume_spherical(theta_calc, r_fos_sph_calc)
                sph_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_calc, r_fos_sph_calc)
                sph_com = BetaDeformationCalculator.calculate_center_of_mass_spherical(theta_calc, r_fos_sph_calc)
                spherical_text = (f"FoS Shape (spherical):\n"
                                  f"  Volume:  {sph_volume:.2f} fm^3\n"
                                  f"  Surface: {sph_surface:.2f} fm^2\n"
                                  f"  CoM z:   {sph_com:.2f} fm\n\n")

                # Plot the converted spherical shape in the SHIFTED frame where conversion happens
                # (don't subtract shift - show where the origin actually is for the spherical representation)
                theta_sph_plot = theta_calc[idx]
                r_sph_plot = r_fos_sph_calc[idx]
                z_sph = r_sph_plot * np.cos(theta_sph_plot)  # Shifted frame (origin at neck)
                rho_sph = r_sph_plot * np.sin(theta_sph_plot)
                self.lines['fos_sph'].set_data(z_sph, rho_sph)
                self.lines['fos_sph_m'].set_data(z_sph, -rho_sph)

                # Calculate conversion metrics using the new method
                conv_metrics = conv.calculate_round_trip_metrics(z_fos_calc, rho_fos_calc, shift)

                conversion_metrics_text = (f"Conversion Metrics (Cyl→Sph):\n"
                                           f"  RMSE:      {conv_metrics['rmse']:.4f} fm\n"
                                           f"  L_inf:     {conv_metrics['l_infinity']:.4f} fm\n"
                                           f"  Volume Δ:  {conv_metrics['volume_diff']:.4f} fm^3\n"
                                           f"  Surface Δ: {conv_metrics['surface_diff']:.4f} fm^2\n"
                                           f"  Z-shift:   {conv_metrics['z_shift']:.2f} fm\n\n")

                # Display shift optimization metrics
                if opt_metrics:
                    conversion_metrics_text += (f"Shift Optimization (l_max={opt_metrics['l_max_test']}):\n"
                                                f"  Optimal:     {opt_metrics['shift']:.2f} fm\n"
                                                f"  Surface Δ:   {opt_metrics['surface_diff']:.4f} fm²\n"
                                                f"  RMSE ρ:      {opt_metrics['rmse']:.4f} fm\n"
                                                f"  L_inf ρ:     {opt_metrics['l_infinity']:.4f} fm\n"
                                                f"  Combined:    {opt_metrics['combined_metric']:.4f}\n\n")

                # Iterative Beta Fitting using the dedicated fitter class
                # Pass a callback to flush UI events and prevent "Not Responding"
                def flush_events() -> None:
                    if self.fig is not None:
                        self.fig.canvas.flush_events()

                # Pass original FoS data for cylindrical comparison (consistent metrics)
                fit_result = self.beta_fitter.fit(
                    theta_calc, r_fos_sph_calc,
                    z_fos_calc, rho_fos_calc, shift,  # Original FoS data for comparison
                    fos_surface, self.params.nucleons, self.n_calc,
                    progress_callback=flush_events
                )

                # Cache beta fitting results for the Print Betas button
                self._last_beta_result = fit_result

                # Calculate beta fit volume/surface/CoM
                beta_volume = BetaDeformationCalculator.calculate_volume_spherical(
                    fit_result.theta_reconstructed, fit_result.r_reconstructed
                )
                beta_surface = BetaDeformationCalculator.calculate_surface_area_spherical(
                    fit_result.theta_reconstructed, fit_result.r_reconstructed
                )
                beta_com = BetaDeformationCalculator.calculate_center_of_mass_spherical(
                    fit_result.theta_reconstructed, fit_result.r_reconstructed
                )

                status = "Converged" if fit_result.converged else "Max l reached"
                beta_fit_text = (f"Beta Fit Shape (l_max={fit_result.l_max}, {status}):\n"
                                 f"  Volume:  {beta_volume:.2f} fm^3\n"
                                 f"  Surface: {beta_surface:.2f} fm^2\n"
                                 f"  CoM z:   {beta_com:.2f} fm\n\n")

                # Use the errors from fit_result (already calculated as cylindrical comparison)
                errors = fit_result.errors

                # Build the primary accuracy metrics text
                beta_vs_fos_text = (f"Beta vs Original FoS (Cylindrical):\n"
                                    f"  RMSE ρ:    {errors['rmse_rho']:.4f} fm\n"
                                    f"  L_inf ρ:   {errors['l_infinity_rho']:.4f} fm @ z={errors['l_infinity_z']:.2f} fm\n"
                                    f"  Volume Δ:  {abs(beta_volume - fos_volume):.4f} fm^3\n"
                                    f"  Surface Δ: {errors['surface_diff']:.4f} fm^2\n")

                # Reconstruct for Plotting (downsampled for efficiency)
                theta_plot = fit_result.theta_reconstructed[idx]
                r_rec_plot = fit_result.r_reconstructed[idx]

                z_beta = r_rec_plot * np.cos(theta_plot) - shift
                rho_beta = r_rec_plot * np.sin(theta_plot)

                self.lines['beta'].set_data(z_beta, rho_beta)
                self.lines['beta_m'].set_data(z_beta, -rho_beta)

                # Draw L_inf marker at the location of maximum error IN CYLINDRICAL COORDINATES
                # Need to get detailed comparison for the marker positions
                cyl_comparison = CylindricalToSphericalConverter.calculate_cylindrical_comparison(
                    fit_result.theta_reconstructed,
                    fit_result.r_reconstructed,
                    z_fos_calc,
                    rho_fos_calc,
                    shift
                )

                l_inf_z = cyl_comparison['l_infinity_z']
                rho_fos_at_linf = cyl_comparison['l_infinity_rho_fos']
                rho_beta_at_linf = cyl_comparison['l_infinity_rho_beta']

                # Draw connecting line with markers at the L_inf location
                self.lines['l_inf_marker'].set_data([l_inf_z, l_inf_z],
                                                    [rho_fos_at_linf, rho_beta_at_linf])
                self.lines['l_inf_marker_m'].set_data([l_inf_z, l_inf_z],
                                                      [-rho_fos_at_linf, -rho_beta_at_linf])
            else:
                spherical_text = "Shape not convertible to spherical.\n\n"
                self.lines['fos_sph'].set_data([], [])
                self.lines['fos_sph_m'].set_data([], [])
                self.lines['beta'].set_data([], [])
                self.lines['beta_m'].set_data([], [])
                self.lines['l_inf_marker'].set_data([], [])
                self.lines['l_inf_marker_m'].set_data([], [])
                # Clear cached beta params when conversion fails
                self._last_beta_result = None
        else:
            # Show Beta is OFF - clear spherical and beta lines and cache
            self.lines['fos_sph'].set_data([], [])
            self.lines['fos_sph_m'].set_data([], [])
            self.lines['beta'].set_data([], [])
            self.lines['beta_m'].set_data([], [])
            self.lines['l_inf_marker'].set_data([], [])
            self.lines['l_inf_marker_m'].set_data([], [])
            self._last_beta_result = None

        # Auto-scale
        limit = max(np.max(np.abs(z_fos_calc)), np.max(rho_fos_calc)) * 1.2
        self.ax_plot.set_xlim(-limit, limit)
        self.ax_plot.set_ylim(-limit, limit)
        self.ax_plot.set_title(f"Z={self.params.protons} N={self.params.neutrons} (FoS)")

        # Update Text
        self.ax_text.clear()
        self.ax_text.axis('off')
        if self.show_text_info:
            sphere_vol, sphere_surf = self._get_sphere_properties()

            # Build FoS cylindrical metrics text with optional neck info
            fos_cyl_text = (f"FoS Shape (cylindrical):\n"
                            f"  Volume:  {fos_volume:.2f} fm^3\n"
                            f"  Surface: {fos_surface:.2f} fm^2\n"
                            f"  CoM z:   {fos_com:.2f} fm\n")
            if neck_rho is not None:
                fos_cyl_text += f"  Neck ρ:  {neck_rho:.2f} fm\n"
                fos_cyl_text += fragment_ratio_text
            fos_cyl_text += "\n"

            info = (f"FoS Parameters:\n"
                    f"c={self.params.c_elongation:.3f}, q2={self.params.q2:.3f}\n"
                    f"a3={self.params.get_coefficient(3):.3f}, a4={self.params.get_coefficient(4):.3f}\n"
                    f"R0={self.params.radius0:.3f}\n\n"
                    f"Spherical Nucleus:\n"
                    f"  Volume:  {sphere_vol:.2f} fm^3\n"
                    f"  Surface: {sphere_surf:.2f} fm^2\n\n"
                    + fos_cyl_text
                    + spherical_text + conversion_metrics_text + beta_fit_text + beta_vs_fos_text)

            self.ax_text.text(0, 1, info, va='top', fontfamily='monospace', fontsize=9)

        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """Starts the interactive plotter."""
        self.update_plot(None)
        plt.show()
