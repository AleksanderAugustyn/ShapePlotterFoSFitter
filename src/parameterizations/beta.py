"""Beta deformation parameterization and shape error calculations.

OPTIMIZED VERSION:
1. Uses Legendre polynomials directly instead of spherical harmonics (Y_l0 = sqrt((2l+1)/(4π)) * P_l(cos(θ))).
2. Uses NumPy broadcasting for beta parameter calculation and shape reconstruction.
3. Uses fast vectorized Simpson's rule implementation.
4. Analytical derivative for surface area calculation (more stable than np.gradient).
"""
from dataclasses import dataclass
from typing import Callable, Dict, Final, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre

FloatArray = NDArray[np.float64]


class ShapeComparisonMetrics(TypedDict):
    """Metrics for comparing original and reconstructed shapes in spherical coordinates."""
    rmse: float
    chi_squared: float  # Raw Chi-Squared
    chi_squared_reduced: float  # Reduced Chi-Squared (Chi^2 / DoF)
    l_infinity: float
    l_infinity_angle: float  # Angle (radians) where L_infinity occurs


class CylindricalComparisonMetrics(TypedDict):
    """Metrics comparing beta fit to original FoS shape in cylindrical coordinates.

    This is the primary accuracy metric - it measures how well the entire
    pipeline (FoS → spherical → beta → cylindrical) reproduces the original shape.
    """
    rmse_rho: float  # RMSE in ρ(z) in fm
    l_infinity_rho: float  # Maximum |Δρ| in fm
    l_infinity_z: float  # z-coordinate where L_inf occurs in fm
    surface_diff: float  # Absolute surface difference in fm²


@dataclass
class BetaFitResult:
    """Result of iterative beta fitting."""
    beta_parameters: Dict[int, float]
    l_max: int
    theta_reconstructed: np.ndarray
    r_reconstructed: np.ndarray
    errors: CylindricalComparisonMetrics  # Uses cylindrical comparison for consistency
    converged: bool


# Fitting constants
BATCH_SIZE: Final[int] = 32  # Number of beta parameters to add per iteration
MAX_BETA: Final[int] = 2 * 512  # Maximum number of beta parameters for fitting
RMSE_THRESHOLD: Final[float] = 0.2  # RMSE convergence threshold in fm
LINF_THRESHOLD: Final[float] = 0.4  # L-infinity convergence threshold in fm
SURFACE_DIFF_THRESHOLD: Final[float] = 0.5  # Surface area difference threshold in fm^2
SURFACE_DIFF_THRESHOLD_RELAXED: Final[float] = 4.0  # Relaxed surface area difference threshold in fm^2
RELAX_ITERATION_THRESHOLD: Final[int] = 16  # Iteration to relax the surface diff threshold
POLE_EXCLUSION_DEG: Final[float] = 0.0  # Exclude X° at each pole to avoid singularities


def _simpson_fast(y: FloatArray, x: FloatArray) -> float:
    """Fast implementation of Simpson's rule for strictly odd N (even intervals).

    Falls back to trapezoidal rule for even N.
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n % 2 == 0:
        # Trapezoidal fallback for even N
        return float(np.trapezoid(y, x))

    h = (x[-1] - x[0]) / (n - 1)
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return float(s * h / 3.0)


def _ylm_normalization(l: int | np.ndarray) -> float | np.ndarray:
    """Compute Y_l0 normalization factor: sqrt((2l+1)/(4π))."""
    return np.sqrt((2 * l + 1) / (4 * np.pi))


def _compute_ylm_matrix(l_values: np.ndarray, cos_theta: np.ndarray) -> FloatArray:
    """Compute Y_l0 matrix efficiently using Legendre polynomials.

    Y_l0(θ) = sqrt((2l+1)/(4π)) * P_l(cos(θ))

    Args:
        l_values: Array of l values, shape (n_l,)
        cos_theta: Array of cos(θ) values, shape (n_theta)

    Returns:
        Matrix of shape (n_l, n_theta) containing Y_l0 values.
    """
    n_l = len(l_values)
    n_theta = len(cos_theta)
    result = np.empty((n_l, n_theta), dtype=np.float64)

    # eval_legendre is very fast for scalar l with vector x
    for i, l in enumerate(l_values):
        norm = _ylm_normalization(l)
        result[i, :] = norm * eval_legendre(int(l), cos_theta)

    return result


class BetaDeformationCalculator:
    """Calculates beta deformation parameters and shape errors.

    OPTIMIZATIONS:
    - Uses Legendre polynomials directly (faster than spherical harmonics for m=0).
    - Pre-computes cos(θ) and sin(θ) once.
    - Uses vectorized integration via fast Simpson's rule.
    - Batch beta parameter calculation using matrix operations.
    """

    def __init__(self, theta: np.ndarray, radius: np.ndarray, number_of_nucleons: int,
                 pole_exclusion_deg: float = POLE_EXCLUSION_DEG):
        theta = np.asarray(theta, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)
        self.radius0 = 1.16 * number_of_nucleons ** (1 / 3)

        # Sort by theta
        sort_idx = np.argsort(theta)
        theta = theta[sort_idx]
        radius = radius[sort_idx]

        # Apply pole exclusion mask to avoid fitting singularities at θ=0 and θ=π
        pole_exclusion_rad = np.radians(pole_exclusion_deg)
        mask = (theta >= pole_exclusion_rad) & (theta <= np.pi - pole_exclusion_rad)
        self.theta = theta[mask]
        self.r = radius[mask]

        # Pre-compute trigonometric values once
        self.sin_theta = np.sin(self.theta)
        self.cos_theta = np.cos(self.theta)

        # Y_l0 cache for specific l values
        self._ylm_cache: Dict[int, FloatArray] = {}

        # Pre-compute the normalization denominator (constant for a given shape)
        y00 = _ylm_normalization(0) * np.ones_like(self.theta)  # P_0 = 1
        denominator_integrand = self.r * y00 * self.sin_theta
        self._denominator: float = float(_simpson_fast(denominator_integrand, self.theta))

    def _get_ylm(self, l: int) -> FloatArray:
        """Get cached Y_l0(θ) values using Legendre polynomials."""
        if l not in self._ylm_cache:
            norm = _ylm_normalization(l)
            self._ylm_cache[l] = norm * eval_legendre(l, self.cos_theta)
        return self._ylm_cache[l]

    def _build_ylm_matrix(self, l_start: int, l_end: int) -> FloatArray:
        """Build a matrix of Y_l0 values for l in [l_start, l_end].

        OPTIMIZED: Uses Legendre polynomials, which are much faster than sph_harm_y.

        Returns:
            Matrix of shape (l_end - l_start + 1, n_theta) where each row is Y_l0(θ).
        """
        l_values = np.arange(l_start, l_end + 1, dtype=np.int64)
        return _compute_ylm_matrix(l_values, self.cos_theta)

    def calculate_beta_range(self, l_start: int, l_end: int) -> Dict[int, float]:
        """Calculates analytical beta parameters for a specific range of l.

        OPTIMIZED: Uses matrix operations to compute all betas in the range simultaneously.
        """
        if l_end < l_start:
            return {}

        # Build Y_lm matrix: shape (n_l, n_theta)
        ylm_matrix = self._build_ylm_matrix(l_start, l_end)

        # Integrand matrix: Y_l0(θ) * R(θ) * sin(θ) for each l
        # Broadcasting: (n_l, n_theta) * (n_theta,) * (n_theta,) -> (n_l, n_theta)
        integrand_matrix = ylm_matrix * self.r * self.sin_theta

        # Vectorized integration using Simpson's rule for each row
        # For truly vectorized integration, we apply Simpson weights
        n_theta = len(self.theta)
        if n_theta % 2 == 1 and n_theta >= 3:
            # Simpson's rule weights
            h = (self.theta[-1] - self.theta[0]) / (n_theta - 1)
            weights = np.ones(n_theta, dtype=np.float64)
            weights[1:-1:2] = 4.0  # Odd indices
            weights[2:-1:2] = 2.0  # Even indices (except first and last)
            weights *= h / 3.0

            # Matrix-vector product: sum over theta dimension with weights
            numerators = integrand_matrix @ weights
        else:
            # Fallback: trapezoidal rule
            numerators = np.trapezoid(integrand_matrix, x=self.theta, axis=1)

        # Compute beta values
        sqrt_4pi = np.sqrt(4.0 * np.pi)
        if abs(self._denominator) > 1e-10:
            beta_values = sqrt_4pi * numerators / self._denominator
        else:
            beta_values = np.zeros(l_end - l_start + 1)

        # Build result dictionary
        return {l: float(beta_values[i]) for i, l in enumerate(range(l_start, l_end + 1))}

    def calculate_beta_parameters(self, l_max: int = 12) -> Dict[int, float]:
        """Calculates analytical beta parameters using cached denominator."""
        return self.calculate_beta_range(1, l_max)

    def reconstruct_shape(self, beta: Dict[int, float], n_theta: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstructs r(θ) from beta parameters with volume conservation.

        OPTIMIZED: Uses Legendre polynomials (much faster than spherical harmonics for m=0).
        """
        theta_recon = np.linspace(0, np.pi, n_theta, dtype=np.float64)

        if not beta:
            # No deformation - return sphere
            r_recon = np.ones(n_theta, dtype=np.float64) * self.radius0
            return theta_recon, r_recon

        # Pre-compute cos(θ) for Legendre polynomials
        cos_theta_recon = np.cos(theta_recon)

        # Extract l values and corresponding beta values as arrays
        l_values = np.array(sorted(beta.keys()), dtype=np.int64)
        beta_values = np.array([beta[l] for l in l_values], dtype=np.float64)

        # Compute Y_lm matrix using fast Legendre polynomials
        ylm_matrix = _compute_ylm_matrix(l_values, cos_theta_recon)

        # Compute deformation: sum_l beta_l * Y_l0(θ)
        # Matrix multiplication: (n_l,) @ (n_l, n_theta) -> (n_theta,)
        deformation = beta_values @ ylm_matrix

        # r(θ) = R0 * (1 + sum_l beta_l * Y_l0)
        r_pre = self.radius0 * (1.0 + deformation)

        # Volume fixing using fast Simpson
        sin_theta = np.sin(theta_recon)
        vol_integrand = r_pre ** 3 * sin_theta * (2.0 / 3.0) * np.pi
        vol_pre = _simpson_fast(vol_integrand, theta_recon)
        sphere_vol = (4.0 / 3.0) * np.pi * self.radius0 ** 3

        # Protect against division by zero
        if vol_pre <= 1e-12:
            scale_factor = 1.0
        else:
            scale_factor = (sphere_vol / vol_pre) ** (1.0 / 3.0)

        return theta_recon, scale_factor * r_pre

    @staticmethod
    def calculate_volume_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate volume from spherical coordinates r(θ).

        V = (2π/3) ∫₀^π r(θ)³ sin(θ) dθ

        Uses fast Simpson integration.
        """
        integrand = r ** 3 * np.sin(theta)
        return float((2.0 * np.pi / 3.0) * _simpson_fast(integrand, theta))

    @staticmethod
    def calculate_surface_area_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate surface area from spherical coordinates r(θ).

        S = 2π ∫₀^π r(θ) sin(θ) √(r² + (dr/dθ)²) dθ

        OPTIMIZED: Uses central differences with boundary handling for stable derivatives.
        """
        n = len(theta)
        if n < 2:
            return 0.0

        # Compute dr/dθ using central differences (more stable than np.gradient)
        # Interior points: central difference
        # Boundary points: forward/backward difference
        dr_dtheta = np.empty(n, dtype=np.float64)

        if n == 2:
            # Only two points - use a simple difference
            dr_dtheta[:] = (r[1] - r[0]) / (theta[1] - theta[0])
        else:
            # Central differences for interior
            dr_dtheta[1:-1] = (r[2:] - r[:-2]) / (theta[2:] - theta[:-2])

            # Forward difference at the start
            dr_dtheta[0] = (r[1] - r[0]) / (theta[1] - theta[0])

            # Backward difference at the end
            dr_dtheta[-1] = (r[-1] - r[-2]) / (theta[-1] - theta[-2])

        sin_theta = np.sin(theta)
        integrand = r * sin_theta * np.sqrt(r ** 2 + dr_dtheta ** 2)

        return float(2.0 * np.pi * _simpson_fast(integrand, theta))

    @staticmethod
    def calculate_center_of_mass_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate the center of mass z-coordinate from spherical coordinates r(θ).

        z_cm = ∫ z * dV / V where z = r*cos(θ)
        z_cm = (π/2) ∫ r⁴ cos(θ) sin(θ) dθ / V

        Uses fast Simpson integration.
        """
        volume = BetaDeformationCalculator.calculate_volume_spherical(theta, r)
        if volume < 1e-10:
            return 0.0

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        integrand = r ** 4 * cos_theta * sin_theta

        return float((np.pi / 2.0) * _simpson_fast(integrand, theta) / volume)

    @staticmethod
    def calculate_errors(r_original: np.ndarray,
                         r_reconstructed: np.ndarray,
                         theta: np.ndarray,
                         n_params: int = 0,
                         pole_exclusion_deg: float = POLE_EXCLUSION_DEG) -> ShapeComparisonMetrics:
        """
        Calculates RMSE, Raw Chi-Squared, Reduced Chi-Squared, and L-infinity errors.

        Args:
            r_original: The reference radial values.
            r_reconstructed: The radial values from the beta expansion.
            theta: The theta angle array (radians) corresponding to r values.
            n_params: Number of fitted parameters (used for Degrees of Freedom).
                      If 0, DoF = N_points.
            pole_exclusion_deg: Degrees to exclude at each pole (θ=0 and θ=π).
        """
        # Apply pole exclusion mask to avoid singularities at θ=0 and θ=π
        pole_exclusion_rad = np.radians(pole_exclusion_deg)
        mask = (theta >= pole_exclusion_rad) & (theta <= np.pi - pole_exclusion_rad)

        r_orig_masked = r_original[mask]
        r_recon_masked = r_reconstructed[mask]
        theta_masked = theta[mask]

        diff = r_orig_masked - r_recon_masked
        squared_diff = diff ** 2

        # RMSE
        rmse = float(np.sqrt(np.mean(squared_diff)))

        # L-infinity (Max absolute error) and angle where it occurs
        abs_diff = np.abs(diff)
        max_idx = np.argmax(abs_diff)
        l_inf = float(abs_diff[max_idx])
        l_inf_angle = float(theta_masked[max_idx])

        # Raw Chi-Squared (Pearson-like): sum((Obs - Exp)^2 / Exp)
        # We use r_original as the expected value in the denominator for weighting.
        # Avoid division by zero by clamping the denominator.
        safe_r = np.maximum(r_orig_masked, 1e-10)
        chi_sq = float(np.sum(squared_diff / safe_r))

        # Reduced Chi-Squared: Chi^2 / DoF
        # Degrees of Freedom = N_points - N_parameters
        n_points = len(r_orig_masked)
        dof = max(1, n_points - n_params)
        chi_sq_red = chi_sq / dof

        return {
            "rmse": rmse,
            "chi_squared": chi_sq,
            "chi_squared_reduced": chi_sq_red,
            "l_infinity": l_inf,
            "l_infinity_angle": l_inf_angle
        }


class IterativeBetaFitter:
    """Iteratively fits beta parameters to a nuclear shape until convergence.

    OPTIMIZATIONS:
    - Reuses BetaDeformationCalculator instance across iterations.
    - Only calculates new beta parameters for each batch (not full recalculation).
    - Uses optimized reconstruction and metric calculations.
    - Compares against original FoS shape in cylindrical coordinates for consistency.
    """

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
            rmse_threshold: RMSE convergence threshold in fm (cylindrical).
            linf_threshold: L-infinity convergence threshold in fm (cylindrical).
            surface_diff_threshold: Surface area difference threshold in fm².
        """
        self.batch_size = batch_size
        self.max_beta = max_beta
        self.rmse_threshold = rmse_threshold
        self.linf_threshold = linf_threshold
        self.surface_diff_threshold = surface_diff_threshold

    @staticmethod
    def _calculate_cylindrical_comparison(
            theta_beta: np.ndarray,
            r_beta: np.ndarray,
            z_original: np.ndarray,
            rho_original: np.ndarray,
            z_shift: float,
            reference_surface: float
    ) -> CylindricalComparisonMetrics:
        """Compare beta fit to original FoS shape in cylindrical coordinates.

        This is the PRIMARY accuracy metric - it measures how well the entire
        pipeline (FoS → spherical → beta → cylindrical) reproduces the original shape.

        Args:
            theta_beta: θ values from beta reconstruction.
            r_beta: r(θ) values from beta reconstruction.
            z_original: Original FoS z coordinates.
            rho_original: Original FoS ρ(z) values.
            z_shift: The z-shift applied during spherical conversion.
            reference_surface: Original FoS surface area for surface_diff calculation.

        Returns:
            CylindricalComparisonMetrics with RMSE, L∞ in ρ, and surface difference.
        """
        # Convert beta fit back to cylindrical coordinates
        z_beta = r_beta * np.cos(theta_beta) - z_shift
        rho_beta = r_beta * np.sin(theta_beta)

        # Create spline of original FoS for interpolation
        sort_idx = np.argsort(z_original)
        z_orig_sorted = z_original[sort_idx]
        rho_orig_sorted = rho_original[sort_idx]

        # Find valid z range for comparison (where both shapes exist)
        z_min_orig = float(z_orig_sorted[0])
        z_max_orig = float(z_orig_sorted[-1])
        z_min_beta = float(np.min(z_beta))
        z_max_beta = float(np.max(z_beta))

        z_min_compare = max(z_min_orig, z_min_beta)
        z_max_compare = min(z_max_orig, z_max_beta)

        # Create spline of original shape
        spline_original = CubicSpline(z_orig_sorted, rho_orig_sorted, bc_type='natural', extrapolate=False)

        # Mask for beta points within valid comparison range
        # Exclude very tips (within 0.1 fm of endpoints) to avoid spline boundary artifacts
        margin = 0.1
        valid_mask = (z_beta >= z_min_compare + margin) & (z_beta <= z_max_compare - margin)

        if not np.any(valid_mask):
            # Fallback: no valid comparison points
            return CylindricalComparisonMetrics(
                rmse_rho=float('inf'),
                l_infinity_rho=float('inf'),
                l_infinity_z=0.0,
                surface_diff=float('inf')
            )

        z_compare = z_beta[valid_mask]
        rho_beta_compare = rho_beta[valid_mask]

        # Evaluate original FoS at the beta shape's z positions
        rho_original_interp = spline_original(z_compare)
        rho_original_interp = np.maximum(rho_original_interp, 0.0)  # Clamp negatives

        # Calculate shape comparison metrics
        diff_rho = rho_original_interp - rho_beta_compare
        rmse_rho = float(np.sqrt(np.mean(diff_rho ** 2)))

        abs_diff = np.abs(diff_rho)
        l_inf_idx = int(np.argmax(abs_diff))
        l_infinity_rho = float(abs_diff[l_inf_idx])
        l_infinity_z = float(z_compare[l_inf_idx])

        # Calculate surface difference using spherical coordinates (reliable)
        beta_surface = BetaDeformationCalculator.calculate_surface_area_spherical(theta_beta, r_beta)
        surface_diff = abs(beta_surface - reference_surface)

        return CylindricalComparisonMetrics(
            rmse_rho=rmse_rho,
            l_infinity_rho=l_infinity_rho,
            l_infinity_z=l_infinity_z,
            surface_diff=surface_diff
        )

    def fit(
            self,
            theta: np.ndarray,
            r_original: np.ndarray,
            z_fos: np.ndarray,
            rho_fos: np.ndarray,
            z_shift: float,
            reference_surface: float,
            nucleons: int,
            n_points: int = 7200,
            progress_callback: Optional[Callable[[], None]] = None
    ) -> BetaFitResult:
        """Iteratively fit beta parameters until convergence criteria are met.

        Args:
            theta: Theta values for the spherical representation (converted from FoS).
            r_original: r(θ) values for the spherical representation.
            z_fos: Original FoS z coordinates (for cylindrical comparison).
            rho_fos: Original FoS ρ(z) values (for cylindrical comparison).
            z_shift: The z-shift applied during spherical conversion.
            reference_surface: Reference surface area (original FoS) for convergence check.
            nucleons: Number of nucleons (A).
            n_points: Number of points for shape reconstruction.
            progress_callback: Optional callback to invoke after each iteration.

        Returns:
            BetaFitResult containing fitted parameters and convergence info.
        """
        # Initialize calculator once to reuse cached values
        beta_calculator = BetaDeformationCalculator(theta, r_original, nucleons)

        l_max: int = 0
        converged: bool = False
        beta_parameters: Dict[int, float] = {}
        theta_reconstructed: np.ndarray = np.array([])
        r_reconstructed: np.ndarray = np.array([])
        errors: CylindricalComparisonMetrics = {
            'rmse_rho': float('inf'),
            'l_infinity_rho': float('inf'),
            'l_infinity_z': 0.0,
            'surface_diff': float('inf'),
        }

        # Track previous values for stagnation detection
        prev_rmse: float = float('inf')
        prev_linf: float = float('inf')
        prev_surface_diff: float = float('inf')
        min_improvement: float = 0.0001

        # Track iterations for threshold relaxation
        iteration_count: int = 0
        relaxed_threshold: bool = False

        while l_max < self.max_beta:
            iteration_count += 1

            # Relax the surface diff threshold after N iterations (N*32 parameters)
            if iteration_count == RELAX_ITERATION_THRESHOLD and not relaxed_threshold:
                self.surface_diff_threshold = SURFACE_DIFF_THRESHOLD_RELAXED
                relaxed_threshold = True
                print(f"Relaxing surface diff threshold to {SURFACE_DIFF_THRESHOLD_RELAXED:.1f} fm² after 10 iterations")

            # Determine batch range
            l_start = l_max + 1
            l_end = min(l_max + self.batch_size, self.max_beta)
            l_max = l_end

            print(f"Current l_max = {l_max}")

            # Calculate only new beta parameters for this batch (OPTIMIZED)
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

            # Calculate cylindrical comparison metrics (consistent with plotter display)
            errors = self._calculate_cylindrical_comparison(
                theta_reconstructed, r_reconstructed,
                z_fos, rho_fos, z_shift, reference_surface
            )

            rmse = errors['rmse_rho']
            l_inf = errors['l_infinity_rho']
            surface_diff = errors['surface_diff']

            if (rmse < self.rmse_threshold and
                    l_inf < self.linf_threshold and
                    surface_diff < self.surface_diff_threshold):
                converged = True
                break

            # Check for stagnation
            rmse_improvement = prev_rmse - rmse
            linf_improvement = prev_linf - l_inf
            surface_improvement = prev_surface_diff - surface_diff

            if (rmse_improvement < min_improvement and
                    linf_improvement < min_improvement and
                    surface_improvement < min_improvement):
                print(f"Stagnation detected at l_max={l_max}: improvements below {min_improvement}")
                break

            # Update previous values
            prev_rmse = rmse
            prev_linf = l_inf
            prev_surface_diff = surface_diff

            print(f"RMSE ρ: {rmse:.4f} fm, L_inf ρ: {l_inf:.4f} fm, Surface Diff: {surface_diff:.4f} fm²")

            # Allow UI to process events
            if progress_callback is not None:
                progress_callback()

        status = "converged" if converged else f"reached max l_max={l_max}"
        print(f"RMSE ρ: {errors['rmse_rho']:.4f} fm, L_inf ρ: {errors['l_infinity_rho']:.4f} fm, Surface Diff: {errors['surface_diff']:.4f} fm²")
        print(f"Beta fitting completed: {status}")

        return BetaFitResult(
            beta_parameters=beta_parameters,
            l_max=l_max,
            theta_reconstructed=theta_reconstructed,
            r_reconstructed=r_reconstructed,
            errors=errors,
            converged=converged
        )
