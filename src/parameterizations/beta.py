"""Beta deformation parameterization and shape error calculations."""
from dataclasses import dataclass
from typing import Callable, Dict, Final, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.special import sph_harm_y


class ShapeComparisonMetrics(TypedDict):
    """Metrics for comparing original and reconstructed shapes."""
    rmse: float
    chi_squared: float  # Raw Chi-Squared
    chi_squared_reduced: float  # Reduced Chi-Squared (Chi^2 / DoF)
    l_infinity: float


@dataclass
class BetaFitResult:
    """Result of iterative beta fitting."""
    beta_parameters: Dict[int, float]
    l_max: int
    theta_reconstructed: np.ndarray
    r_reconstructed: np.ndarray
    errors: ShapeComparisonMetrics
    surface_diff: float
    converged: bool


# Fitting constants
BATCH_SIZE: Final[int] = 32  # Number of beta parameters to add per iteration
MAX_BETA: Final[int] = 512  # Maximum number of beta parameters for fitting
RMSE_THRESHOLD: Final[float] = 0.2  # RMSE convergence threshold in fm
LINF_THRESHOLD: Final[float] = 0.5  # L-infinity convergence threshold in fm
SURFACE_DIFF_THRESHOLD: Final[float] = 0.5  # Surface area difference threshold in fm^2


class BetaDeformationCalculator:
    """Calculates beta deformation parameters and shape errors."""

    def __init__(self, theta: np.ndarray, radius: np.ndarray, number_of_nucleons: int):
        self.theta = np.asarray(theta)
        self.r = np.asarray(radius)
        self.radius0 = 1.16 * number_of_nucleons ** (1 / 3)

        # Sort
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.r = self.r[sort_idx]
        self.sin_theta = np.sin(self.theta)
        self._ylm_cache: Dict[int, NDArray[np.float64]] = {}

        # Cache the denominator (constant for a given shape)
        # Denominator: ∫ Y_00 R sin(θ) dθ
        denominator_integrand = self.r * self._get_ylm(0) * self.sin_theta
        self._denominator: float = float(simpson(denominator_integrand, x=self.theta))

    def _get_ylm(self, l: int) -> NDArray[np.float64]:
        """Get cached Y_l0(θ) values."""
        if l not in self._ylm_cache:
            self._ylm_cache[l] = sph_harm_y(l, 0, self.theta, 0.0).real
        return self._ylm_cache[l]

    def calculate_beta_range(self, l_start: int, l_end: int) -> Dict[int, float]:
        """Calculates analytical beta parameters for a specific range of l."""
        beta = {}
        for l in range(l_start, l_end + 1):
            # Numerator: ∫ Y_l0 R sin(θ) dθ
            numerator_integrand = self.r * self._get_ylm(l) * self.sin_theta
            numerator = simpson(numerator_integrand, x=self.theta)
            beta[l] = (np.sqrt(4 * np.pi) * numerator / self._denominator if abs(self._denominator) > 1e-10 else 0.0)
        return beta

    def calculate_beta_parameters(self, l_max: int = 12) -> Dict[int, float]:
        """Calculates analytical beta parameters using cached denominator."""
        return self.calculate_beta_range(1, l_max)

    def reconstruct_shape(self, beta: Dict[int, float], n_theta: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstructs r(θ) from beta parameters with volume conservation."""
        theta_recon = np.linspace(0, np.pi, n_theta)
        r_pre = np.ones_like(theta_recon) * self.radius0

        for l, val in beta.items():
            ylm = sph_harm_y(l, 0, theta_recon, 0).real
            r_pre += self.radius0 * val * ylm

        # Volume fixing
        vol_int = r_pre ** 3 * np.sin(theta_recon) * 2 / 3 * np.pi
        vol_pre = float(simpson(vol_int, x=theta_recon))
        sphere_vol = (4 / 3) * np.pi * self.radius0 ** 3

        # Protect against division by zero if the volume is extremely small
        if vol_pre <= 1e-12:
            scale_factor = 1.0
        else:
            scale_factor = (sphere_vol / vol_pre) ** (1 / 3)

        return theta_recon, scale_factor * r_pre

    @staticmethod
    def calculate_volume_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate volume from spherical coordinates r(θ).

        V = (2π/3) ∫₀^π r(θ)³ sin(θ) dθ
        """
        integrand = r ** 3 * np.sin(theta)
        return float((2 * np.pi / 3) * simpson(integrand, x=theta))

    @staticmethod
    def calculate_surface_area_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate surface area from spherical coordinates r(θ).

        S = 2π ∫₀^π r(θ) sin(θ) √(r² + (dr/dθ)²) dθ
        """
        dr_dtheta = np.gradient(r, theta)
        integrand = r * np.sin(theta) * np.sqrt(r ** 2 + dr_dtheta ** 2)
        return float(2 * np.pi * simpson(integrand, x=theta))

    @staticmethod
    def calculate_center_of_mass_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate center of mass z-coordinate from spherical coordinates r(θ).

        z_cm = ∫ z * dV / V where z = r*cos(θ) and dV = (2π/3) r³ sin(θ) dθ
        z_cm = (2π/3) ∫ r⁴ cos(θ) sin(θ) dθ / V
        """
        volume = BetaDeformationCalculator.calculate_volume_spherical(theta, r)
        if volume < 1e-10:
            return 0.0
        integrand = r ** 4 * np.cos(theta) * np.sin(theta)
        return float((2 * np.pi / 3) * simpson(integrand, x=theta) / volume)

    @staticmethod
    def calculate_errors(r_original: np.ndarray,
                         r_reconstructed: np.ndarray,
                         n_params: int = 0) -> ShapeComparisonMetrics:
        """
        Calculates RMSE, Raw Chi-Squared, Reduced Chi-Squared, and L-infinity errors.

        Args:
            r_original: The reference radial values.
            r_reconstructed: The radial values from the beta expansion.
            n_params: Number of fitted parameters (used for Degrees of Freedom).
                      If 0, DoF = N_points.
        """
        diff = r_original - r_reconstructed
        squared_diff = diff ** 2

        # RMSE
        rmse = np.sqrt(np.mean(squared_diff))

        # L-infinity (Max absolute error)
        l_inf = np.max(np.abs(diff))

        # Raw Chi-Squared (Pearson-like): sum((Obs - Exp)^2 / Exp)
        # We use r_original as the expected value in the denominator for weighting.
        # Avoid division by zero by clamping the denominator.
        safe_r = np.where(r_original < 1e-10, 1e-10, r_original)
        chi_sq = np.sum(squared_diff / safe_r)

        # Reduced Chi-Squared: Chi^2 / DoF
        # Degrees of Freedom = N_points - N_parameters
        n_points = len(r_original)
        dof = max(1, n_points - n_params)
        chi_sq_red = chi_sq / dof

        return {
            "rmse": rmse,
            "chi_squared": chi_sq,
            "chi_squared_reduced": chi_sq_red,
            "l_infinity": l_inf
        }


class IterativeBetaFitter:
    """Iteratively fits beta parameters to a nuclear shape until convergence."""

    def __init__(
            self,
            batch_size: int = BATCH_SIZE,
            max_beta: int = MAX_BETA,
            rmse_threshold: float = RMSE_THRESHOLD,
            linf_threshold: float = LINF_THRESHOLD,
            surface_diff_threshold: float = SURFACE_DIFF_THRESHOLD
    ):
        """Initialize the fitter with convergence parameters.

        Args:
            batch_size: Number of beta parameters to add per iteration.
            max_beta: Maximum number of beta parameters for fitting.
            rmse_threshold: RMSE convergence threshold in fm.
            linf_threshold: L-infinity convergence threshold in fm.
            surface_diff_threshold: Surface area difference threshold in fm².
        """
        self.batch_size = batch_size
        self.max_beta = max_beta
        self.rmse_threshold = rmse_threshold
        self.linf_threshold = linf_threshold
        self.surface_diff_threshold = surface_diff_threshold

    def fit(
            self,
            theta: np.ndarray,
            r_original: np.ndarray,
            reference_surface: float,
            nucleons: int,
            n_points: int = 7200,
            progress_callback: Optional[Callable[[], None]] = None
    ) -> BetaFitResult:
        """Iteratively fit beta parameters until convergence criteria are met.

        Args:
            theta: Theta values for the original shape.
            r_original: r(θ) values for the original shape.
            reference_surface: Reference surface area for convergence check.
            nucleons: Number of nucleons (A).
            n_points: Number of points for shape reconstruction.
            progress_callback: Optional callback to invoke after each iteration (e.g., for UI event processing).

        Returns:
            BetaFitResult containing fitted parameters and convergence info.
        """
        # Initialize calculator once to reuse cached values and sort logic
        beta_calculator = BetaDeformationCalculator(theta, r_original, nucleons)

        l_max: int = 0
        converged: bool = False
        beta_parameters: Dict[int, float] = {}
        theta_reconstructed: np.ndarray = np.array([])
        r_reconstructed: np.ndarray = np.array([])
        errors: ShapeComparisonMetrics = {
            'rmse': float('inf'),
            'chi_squared': float('inf'),
            'chi_squared_reduced': float('inf'),
            'l_infinity': float('inf'),
        }
        surface_diff: float = float('inf')

        # Track previous values for stagnation detection
        prev_rmse: float = float('inf')
        prev_linf: float = float('inf')
        prev_surface_diff: float = float('inf')
        min_improvement: float = 0.0001

        while l_max < self.max_beta:
            # Determine batch range
            l_start = l_max + 1
            l_end = min(l_max + self.batch_size, self.max_beta)
            l_max = l_end

            print(f"Current l_max = {l_max}")

            # Calculate only new beta parameters for this batch
            new_betas = beta_calculator.calculate_beta_range(l_start, l_end)

            # Check for numerical instability (NaNs) in new parameters
            if any(np.isnan(val) for val in new_betas.values()):
                print(f"Numerical instability detected (NaN beta parameters) at l_range {l_start}-{l_end}. Stopping fit.")
                break

            # Tentatively update parameters
            test_betas = beta_parameters.copy()
            test_betas.update(new_betas)

            theta_reconstructed, r_reconstructed = beta_calculator.reconstruct_shape(
                test_betas, n_points
            )

            # Check for NaNs in reconstruction
            if np.isnan(r_reconstructed).any():
                print(f"Numerical instability detected (NaN in reconstruction) at l_max={l_end}. Stopping fit.")
                break

            # If valid, commit changes
            beta_parameters = test_betas

            errors = BetaDeformationCalculator.calculate_errors(
                r_original, r_reconstructed, n_params=l_max
            )
            beta_surface = BetaDeformationCalculator.calculate_surface_area_spherical(
                theta_reconstructed, r_reconstructed
            )
            surface_diff = abs(beta_surface - reference_surface)

            rmse = errors['rmse']
            l_inf = errors['l_infinity']

            if (rmse < self.rmse_threshold and
                    l_inf < self.linf_threshold and
                    surface_diff < self.surface_diff_threshold):
                converged = True
                break

            # Check for stagnation: if all metrics improved by less than min_improvement, stop
            rmse_improvement = prev_rmse - rmse
            linf_improvement = prev_linf - l_inf
            surface_improvement = prev_surface_diff - surface_diff

            if (rmse_improvement < min_improvement and
                    linf_improvement < min_improvement and
                    surface_improvement < min_improvement):
                print(f"Stagnation detected at l_max={l_max}: improvements below {min_improvement}")
                break

            # Update previous values for the next iteration
            prev_rmse = rmse
            prev_linf = l_inf
            prev_surface_diff = surface_diff

            print(f"RMSE: {rmse:.4f} fm, L-infinity: {l_inf:.4f} fm, Surface Diff: {surface_diff:.4f} fm²")

            # Allow UI to process events between iterations
            if progress_callback is not None:
                progress_callback()

        status = "converged" if converged else f"reached max l_max={l_max}"
        print(f"Beta fitting completed: {status}")

        return BetaFitResult(
            beta_parameters=beta_parameters,
            l_max=l_max,
            theta_reconstructed=theta_reconstructed,
            r_reconstructed=r_reconstructed,
            errors=errors,
            surface_diff=surface_diff,
            converged=converged
        )
