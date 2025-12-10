import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, CheckButtons

from src.parameterizations.beta import BetaDeformationCalculator
from src.parameterizations.fos import FoSParameters, FoSShapeCalculator
from src.utilities.converter import CylindricalToSphericalConverter

# Set backend
matplotlib.use('TkAgg')


class FoSShapePlotter:
    """Interactive FoS Shape Plotter with Beta Deformation Fitting."""

    def __init__(self) -> None:
        # --- CONFIGURATION ---
        self.n_calc = 7200  # High precision for physics/fitting
        self.n_plot = 360  # Sufficient for visual smoothness

        # Default parameters
        self.params = FoSParameters()

        # UI State
        self.show_text_info = True
        self.show_beta_approx = True
        self.updating = False
        self.slider_buttons = {}

        # Plot elements
        self.fig = None
        self.ax_plot = None
        self.ax_text = None
        self.lines = {}

        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self) -> None:
        self.fig = plt.figure(figsize=(16.0, 8.0))
        self.ax_plot = self.fig.add_subplot(121)
        self.ax_text = self.fig.add_subplot(122)
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

    def create_slider(self, name, y_pos, label, vmin, vmax, vinit, step):
        ax_dec = plt.axes((0.20, y_pos, 0.016, 0.024))
        ax_sl = plt.axes((0.25, y_pos, 0.5, 0.024))
        ax_inc = plt.axes((0.78, y_pos, 0.016, 0.024))

        slider = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=step)
        btn_dec = Button(ax_dec, '-')
        btn_inc = Button(ax_inc, '+')

        self.slider_buttons[name] = {'slider': slider, 'dec': btn_dec, 'inc': btn_inc}
        return slider

    def setup_controls(self):
        y_start = 0.02
        spacing = 0.03

        self.sl_z = self.create_slider('z', y_start, 'Z', 82, 120, 92, 1)
        self.sl_n = self.create_slider('n', y_start + spacing, 'N', 120, 180, 144, 1)
        self.sl_c = self.create_slider('c', y_start + 2 * spacing, 'c', 0.5, 3.0, 1.0, 0.01)
        self.sl_a3 = self.create_slider('a3', y_start + 3 * spacing, 'a3', -0.6, 0.6, 0.0, 0.01)
        self.sl_a4 = self.create_slider('a4', y_start + 4 * spacing, 'a4', -0.75, 0.75, 0.0, 0.01)
        self.sl_a5 = self.create_slider('a5', y_start + 5 * spacing, 'a5', -0.5, 0.5, 0.0, 0.01)
        self.sl_a6 = self.create_slider('a6', y_start + 6 * spacing, 'a6', -0.5, 0.5, 0.0, 0.01)
        self.sl_max_beta = self.create_slider('mb', y_start + 7 * spacing, 'Max Beta', 1, 128, 12, 1)

        # Checkboxes and Buttons
        self.chk_text = CheckButtons(plt.axes((0.82, 0.45, 0.08, 0.032)), ['Show Info'], [True])
        self.chk_beta = CheckButtons(plt.axes((0.82, 0.42, 0.08, 0.032)), ['Show Beta'], [True])
        self.btn_reset = Button(plt.axes((0.82, 0.37, 0.08, 0.032)), 'Reset')
        self.btn_save = Button(plt.axes((0.82, 0.32, 0.08, 0.032)), 'Save Plot')

    def setup_event_handlers(self):
        sliders = [self.sl_z, self.sl_n, self.sl_c, self.sl_a3, self.sl_a4, self.sl_a5, self.sl_a6, self.sl_max_beta]
        for sl in sliders:
            sl.on_changed(self.update_plot)

        for name, items in self.slider_buttons.items():
            s = items['slider']
            items['dec'].on_clicked(lambda _, s=s: s.set_val(max(s.val - s.valstep, s.valmin)))
            items['inc'].on_clicked(lambda _, s=s: s.set_val(min(s.val + s.valstep, s.valmax)))

        self.chk_text.on_clicked(self.toggle_text)
        self.chk_beta.on_clicked(self.toggle_beta)
        self.btn_reset.on_clicked(self.reset_values)
        self.btn_save.on_clicked(self.save_plot)

    def toggle_text(self, _):
        self.show_text_info = not self.show_text_info
        self.update_plot(None)

    def toggle_beta(self, _):
        self.show_beta_approx = not self.show_beta_approx
        self.update_plot(None)

    def reset_values(self, _):
        self.updating = True
        self.sl_z.set_val(92)
        self.sl_n.set_val(144)
        self.sl_c.set_val(1.0)
        self.sl_a3.set_val(0.0)
        self.sl_a4.set_val(0.0)
        self.sl_a5.set_val(0.0)
        self.sl_a6.set_val(0.0)
        self.sl_max_beta.set_val(12)
        self.updating = False
        self.update_plot(None)

    def save_plot(self, _):
        fname = f"fos_shape_Z{int(self.sl_z.val)}_N{int(self.sl_n.val)}.png"
        self.fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved {fname}")

    def update_plot(self, _: None) -> None:
        if self.updating: return

        # 1. Update Params
        self.params.protons = int(self.sl_z.val)
        self.params.neutrons = int(self.sl_n.val)
        self.params.c_elongation = self.sl_c.val
        self.params.a3 = self.sl_a3.val
        self.params.a4 = self.sl_a4.val
        self.params.a5 = self.sl_a5.val
        self.params.a6 = self.sl_a6.val

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

        beta_text = ""
        metrics_text = ""

        if self.show_beta_approx:
            # 4. Conversion (HIGH PRECISION)
            z_work = z_fos_calc.copy()
            conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)

            # Shift Logic
            shift = 0.0
            if not conv.is_unambiguously_convertible(self.n_calc):
                direction = -1.0 if self.params.z_sh >= 0 else 1.0
                max_shift = (np.max(z_fos_calc) - np.min(z_fos_calc)) / 2.0
                while abs(shift) < max_shift:
                    shift += direction * 0.1
                    z_work = z_fos_calc + shift
                    conv = CylindricalToSphericalConverter(z_work, rho_fos_calc)
                    if conv.is_unambiguously_convertible(self.n_calc): break

            if conv.is_unambiguously_convertible(self.n_calc):
                theta_calc, r_fos_sph_calc = conv.convert_to_spherical(self.n_calc)

                # 5. Beta Calculation (HIGH PRECISION)
                beta_calc = BetaDeformationCalculator(theta_calc, r_fos_sph_calc, self.params.nucleons)
                l_max = int(self.sl_max_beta.val)
                betas = beta_calc.calculate_beta_parameters(l_max)

                # 6. Reconstruct for Error Metrics (HIGH PRECISION)
                theta_rec_calc, r_rec_calc = beta_calc.reconstruct_shape(betas, self.n_calc)

                # Pass n_params (l_max) to calculate Reduced Chi-Squared
                errors = beta_calc.calculate_errors(r_fos_sph_calc, r_rec_calc, n_params=l_max)

                # 7. Reconstruct for Plotting (LOW PRECISION via Downsampling)
                theta_plot = theta_rec_calc[idx]
                r_rec_plot = r_rec_calc[idx]

                z_beta = r_rec_plot * np.cos(theta_plot) - shift
                rho_beta = r_rec_plot * np.sin(theta_plot)

                self.lines['beta'].set_data(z_beta, rho_beta)
                self.lines['beta_m'].set_data(z_beta, -rho_beta)

                # Text Info
                beta_str = [f"β_{l}={v:+.4f}" for l, v in betas.items() if abs(v) > 0.001]
                beta_text = "Beta Params:\n" + "\n".join([", ".join(beta_str[i:i + 3]) for i in range(0, len(beta_str), 3)])
                metrics_text = (f"\nFit Metrics (N={self.n_calc}):\n"
                                f"RMSE: {errors['rmse']:.4f} fm\n"
                                f"Chi^2: {errors['chi_squared']:.4f}\n"
                                f"Chi^2 (Red): {errors['chi_squared_reduced']:.6f}\n"
                                f"L_inf: {errors['l_infinity']:.4f} fm")
            else:
                self.lines['beta'].set_data([], [])
                self.lines['beta_m'].set_data([], [])
                beta_text = "Shape not convertible to spherical."
        else:
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
                    f"a3={self.params.a3:.3f}, a4={self.params.a4:.3f}\n"
                    f"R0={self.params.radius0:.3f}\n\n" + beta_text + metrics_text)

            self.ax_text.text(0, 1, info, va='top', fontfamily='monospace', fontsize=9)

        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """Starts the interactive plotter."""
        self.update_plot(None)
        plt.show()
