"""Module for converting nuclear shapes from cylindrical to spherical coordinates.

OPTIMIZED VERSION:
1. Vectorized Newton-Raphson root-finding for spherical conversion (no Python loops).
2. Fast vectorized Simpson's rule implementation.
3. Analytical/stable derivatives for surface area calculations.
4. Pre-computed interpolation coefficients for faster evaluation.
"""
from typing import Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

FloatArray = NDArray[np.float64]


class ConversionMetrics(TypedDict):
    """Metrics for cylindrical-to-spherical coordinate conversion accuracy."""
    rmse: float  # Root-mean-square error in fm
    l_infinity: float  # Maximum absolute error in fm
    volume_diff: float  # Absolute volume difference in fm³
    surface_diff: float  # Absolute surface area difference in fm²
    z_shift: float  # Applied z-shift to make shape star-convex in fm


def _simpson_fast(y: FloatArray, x: FloatArray) -> float:
    """Fast implementation of Simpson's rule for strictly odd N (even intervals).

    Falls back to trapezoidal rule for even N.
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n % 2 == 0:
        return float(np.trapezoid(y, x))

    h = (x[-1] - x[0]) / (n - 1)
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return float(s * h / 3.0)


class CylindricalToSphericalConverter:
    """Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ).

    OPTIMIZATIONS:
    - Uses CubicSpline for faster and more accurate interpolation with derivative access.
    - Vectorized Newton-Raphson for convert_to_spherical (processes all θ simultaneously).
    - Fast Simpson integration.
    """

    def __init__(self, z_points: np.ndarray, rho_points: np.ndarray):
        self.z_points = np.asarray(z_points, dtype=np.float64)
        self.rho_points = np.asarray(rho_points, dtype=np.float64)

        # Sort by z
        sort_idx = np.argsort(self.z_points)
        self.z_points = self.z_points[sort_idx]
        self.rho_points = self.rho_points[sort_idx]

        # Use CubicSpline for interpolation (faster evaluation, provides derivatives)
        # Boundary condition: natural spline (second derivative = 0 at boundaries)
        self._spline = CubicSpline(self.z_points, self.rho_points, bc_type='natural', extrapolate=False)

        # Valid z range (where rho > 0)
        positive_mask = self.rho_points > 0
        if np.any(positive_mask):
            positive_z = self.z_points[positive_mask]
            self.z_min: float = float(np.min(positive_z))
            self.z_max: float = float(np.max(positive_z))
        else:
            self.z_min = float(self.z_points[0])
            self.z_max = float(self.z_points[-1])

        # Cache r_max for initial guesses
        self._r_max = float(2.0 * max(abs(self.z_max), abs(self.z_min), np.max(self.rho_points)))

    def rho_of_z(self, z: FloatArray | float) -> FloatArray | float:
        """Returns ρ(z) using spline interpolation.

        Handles both scalar and array inputs efficiently.
        """
        z_arr = np.atleast_1d(z)
        result = np.zeros_like(z_arr, dtype=np.float64)

        # Only evaluate within a valid range
        valid_mask = (z_arr >= self.z_min) & (z_arr <= self.z_max)
        if np.any(valid_mask):
            result[valid_mask] = self._spline(z_arr[valid_mask])
            # Clamp negative values (can occur at boundaries due to spline oscillation)
            result = np.maximum(result, 0.0)

        if np.isscalar(z):
            return float(result[0])
        return result

    def rho_prime_of_z(self, z: FloatArray | float) -> FloatArray | float:
        """Returns dρ/dz using spline derivative.

        Handles both scalar and array inputs.
        """
        z_arr = np.atleast_1d(z)
        result = np.zeros_like(z_arr, dtype=np.float64)

        valid_mask = (z_arr >= self.z_min) & (z_arr <= self.z_max)
        if np.any(valid_mask):
            result[valid_mask] = self._spline(z_arr[valid_mask], 1)  # First derivative

        if np.isscalar(z):
            return float(result[0])
        return result

    def convert_to_spherical(self, n_theta: int = 180) -> Tuple[FloatArray, FloatArray]:
        """Converts the shape to spherical coordinates r(θ).

        Uses a fast hybrid approach:
        1. Geometry-aware initial guesses based on the shape extent
        2. Vectorized Newton-Raphson for fast convergence
        3. Bisection fallback for points that don't converge

        For each θ, solves: r·sin(θ) - ρ(r·cos(θ)) = 0
        """
        theta = np.linspace(0, np.pi, n_theta, dtype=np.float64)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Compute maximum possible r
        r_max = float(np.sqrt(self.z_points ** 2 + self.rho_points ** 2).max() * 1.5)

        # Initialize r with geometry-aware guesses
        r = np.zeros(n_theta, dtype=np.float64)
        r[0] = abs(self.z_max)
        r[-1] = abs(self.z_min)

        # For interior points, compute initial guess based on ray-shape intersection
        # At an angle θ, the ray z = r*cos(θ), ρ = r*sin(θ) should intersect the shape
        # Initial guess: blend between pole values and equatorial value
        rho_at_0 = max(float(self.rho_of_z(0.0)), 0.1)
        for i in range(1, n_theta - 1):
            # Smooth blend: at θ=0 use z_max, at θ=π use |z_min|, at θ=π/2 use ρ(0)
            # Using sin²/cos² weighting for smooth transition
            w_pole = abs(cos_theta[i])
            w_equator = sin_theta[i]

            if cos_theta[i] > 0:
                r_pole = abs(self.z_max)
            else:
                r_pole = abs(self.z_min)

            r[i] = w_pole * r_pole + w_equator * rho_at_0

        # Ensure positive initial values
        r = np.maximum(r, 0.1)

        # Newton-Raphson iteration (vectorized)
        interior = np.ones(n_theta, dtype=bool)
        interior[0] = interior[-1] = False

        for iteration in range(50):
            z = r * cos_theta

            # Evaluate ρ(z) and ρ'(z)
            in_range = (z >= self.z_min) & (z <= self.z_max)
            rho_vals = np.zeros_like(r)
            rho_prime = np.zeros_like(r)

            if np.any(in_range):
                rho_vals[in_range] = np.maximum(self._spline(z[in_range]), 0.0)
                rho_prime[in_range] = self._spline(z[in_range], 1)

            # f(r) = r·sin(θ) - ρ(z)
            f = r * sin_theta - rho_vals

            # f'(r) = sin(θ) - ρ'(z)·cos(θ)
            fp = sin_theta - rho_prime * cos_theta
            fp = np.where(np.abs(fp) < 1e-12, 1e-12, fp)

            # Newton step with damping
            delta = f / fp
            # Limit step size
            max_delta = 0.3 * r
            delta = np.clip(delta, -max_delta, max_delta)

            r_new = np.where(interior, r - delta, r)
            r_new = np.clip(r_new, 0.01, r_max)

            # Check convergence
            change = np.max(np.abs(r_new[interior] - r[interior]))
            r = r_new

            if change < 1e-10:
                break

        # Check for bad points and fix with bisection
        z_final = r * cos_theta
        in_range_final = (z_final >= self.z_min) & (z_final <= self.z_max)
        rho_final = np.zeros_like(r)
        if np.any(in_range_final):
            rho_final[in_range_final] = np.maximum(self._spline(z_final[in_range_final]), 0.0)

        residual = np.abs(r * sin_theta - rho_final)
        bad = interior & (residual > 0.01)

        if np.any(bad):
            for i in np.where(bad)[0]:
                r[i] = self._bisection_solve(theta[i], r_max)

        return theta, r

    def _bisection_solve(self, theta: float, r_max: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        """Solve for r at given theta using bisection."""
        if theta < 1e-10:
            return abs(self.z_max)
        if theta > np.pi - 1e-10:
            return abs(self.z_min)

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        def f(r: float) -> float:
            z = r * cos_t
            if z < self.z_min or z > self.z_max:
                return float(r * sin_t)
            return float(r * sin_t - max(float(self._spline(z)), 0.0))

        r_lo, r_hi = 0.0, r_max
        f_lo, f_hi = f(r_lo), f(r_hi)

        if f_lo * f_hi > 0:
            return r_lo if abs(f_lo) < abs(f_hi) else r_hi

        for _ in range(max_iter):
            r_mid = (r_lo + r_hi) / 2.0
            f_mid = f(r_mid)

            if abs(f_mid) < tol or (r_hi - r_lo) < tol:
                return r_mid

            if f_mid * f_lo < 0:
                r_hi, f_hi = r_mid, f_mid
            else:
                r_lo, f_lo = r_mid, f_mid

        return (r_lo + r_hi) / 2.0

    def is_unambiguously_convertible(self, n_points: int = 720, tolerance: float = 1e-9) -> bool:
        """Checks if the shape is star-shaped w.r.t origin (monotonic theta(z)).

        The condition is: z·ρ'(z) - ρ(z) ≤ 0 for all z in [z_min, z_max].

        OPTIMIZED: Fully vectorized check.
        """
        if self.z_min >= self.z_max:
            return True

        epsilon = 1e-6
        z_check = np.linspace(self.z_min + epsilon, self.z_max - epsilon, n_points)

        # Use spline for both ρ and ρ'
        rho_vals = self._spline(z_check)
        rho_prime = self._spline(z_check, 1)  # First derivative

        # Condition: z * ρ'(z) - ρ(z) <= 0
        test_values = z_check * rho_prime - rho_vals

        return not np.any(test_values > tolerance)

    def _find_neck_z_position(self, n_samples: int = 720) -> Optional[float]:
        """Find the z-coordinate of the neck center.

        The neck is the minimum of ρ(z) between two local maxima (fragment tops).

        Returns:
            The z-coordinate of the neck center, or None if no clear neck structure.
        """
        z_samples = np.linspace(self.z_min, self.z_max, n_samples)
        rho_samples = self._spline(z_samples)

        # Find local maxima using vectorized comparison
        # A point is a local max if it's greater than both neighbors
        is_local_max = np.zeros(n_samples, dtype=bool)
        is_local_max[1:-1] = (rho_samples[1:-1] > rho_samples[:-2]) & (rho_samples[1:-1] > rho_samples[2:])

        maxima_indices = np.where(is_local_max)[0]
        if len(maxima_indices) < 2:
            return None

        # Get the two largest maxima by ρ value
        maxima_rho_values = rho_samples[maxima_indices]
        top_two_local_indices = np.argsort(maxima_rho_values)[-2:]  # Two largest
        top_two = maxima_indices[top_two_local_indices]

        # Ensure left < right
        left_idx, right_idx = int(min(top_two)), int(max(top_two))

        # Find minimum ρ between these two maxima
        between_rho = rho_samples[left_idx:right_idx + 1]
        min_local_idx = np.argmin(between_rho)
        neck_idx = left_idx + min_local_idx

        return float(z_samples[neck_idx])

    def calculate_round_trip_metrics(
            self,
            z_original: np.ndarray,
            rho_original: np.ndarray,
            z_shift: float = 0.0
    ) -> ConversionMetrics:
        """Calculate metrics for the cylindrical → spherical → cylindrical round-trip.

        Args:
            z_original: Original z coordinates.
            rho_original: Original ρ(z) values.
            z_shift: The z-shift applied during conversion (for star-convexity).

        Returns:
            ConversionMetrics with RMSE, L∞, and volume/surface differences.
        """
        n_points = len(z_original)

        # Convert to spherical
        theta, r_spherical = self.convert_to_spherical(n_points)

        # Convert back to cylindrical (at the spherical grid points)
        z_roundtrip = r_spherical * np.cos(theta) - z_shift
        rho_roundtrip = r_spherical * np.sin(theta)

        # Calculate volumes using fast Simpson
        vol_original = float(_simpson_fast(np.pi * rho_original ** 2, z_original))
        vol_spherical = self._calculate_volume_spherical(theta, r_spherical)
        volume_diff = abs(vol_spherical - vol_original)

        # Calculate surface areas
        surf_original = self._calculate_surface_cylindrical(z_original, rho_original)
        surf_spherical = self._calculate_surface_spherical(theta, r_spherical)
        surface_diff = abs(surf_spherical - surf_original)

        # For shape comparison, evaluate original rho at the roundtrip z positions
        # This is more robust than trying to interpolate roundtrip back to the original grid
        rho_expected = np.zeros_like(z_roundtrip)
        in_range = (z_roundtrip >= self.z_min) & (z_roundtrip <= self.z_max)
        if np.any(in_range):
            rho_expected[in_range] = np.maximum(self._spline(z_roundtrip[in_range]), 0.0)

        # Compare: rho_roundtrip should equal rho_expected
        diff = rho_expected - rho_roundtrip
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        l_inf = float(np.max(np.abs(diff)))

        return ConversionMetrics(
            rmse=rmse,
            l_infinity=l_inf,
            volume_diff=volume_diff,
            surface_diff=surface_diff,
            z_shift=z_shift
        )

    @staticmethod
    def _calculate_volume_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate volume from spherical coordinates.

        V = (2π/3) ∫₀^π r(θ)³ sin(θ) dθ
        """
        integrand = r ** 3 * np.sin(theta)
        return float((2.0 * np.pi / 3.0) * _simpson_fast(integrand, theta))

    @staticmethod
    def _calculate_surface_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate surface area from spherical coordinates.

        S = 2π ∫₀^π r(θ) sin(θ) √(r² + (dr/dθ)²) dθ

        Uses central differences for stable derivative calculation.
        """
        n = len(theta)
        if n < 2:
            return 0.0

        # Compute dr/dθ using central differences
        dr_dtheta = np.empty(n, dtype=np.float64)

        if n == 2:
            dr_dtheta[:] = (r[1] - r[0]) / (theta[1] - theta[0])
        else:
            # Central differences for interior
            dr_dtheta[1:-1] = (r[2:] - r[:-2]) / (theta[2:] - theta[:-2])
            # Forward/backward at boundaries
            dr_dtheta[0] = (r[1] - r[0]) / (theta[1] - theta[0])
            dr_dtheta[-1] = (r[-1] - r[-2]) / (theta[-1] - theta[-2])

        sin_theta = np.sin(theta)
        integrand = r * sin_theta * np.sqrt(r ** 2 + dr_dtheta ** 2)

        return float(2.0 * np.pi * _simpson_fast(integrand, theta))

    @staticmethod
    def _calculate_surface_cylindrical(z: np.ndarray, rho: np.ndarray) -> float:
        """Calculate surface area from cylindrical coordinates.

        S = 2π ∫ ρ √(1 + (dρ/dz)²) dz

        Uses central differences for stable derivative calculation.
        """
        n = len(z)
        if n < 2:
            return 0.0

        # Compute dρ/dz using central differences
        d_rho_dz = np.empty(n, dtype=np.float64)

        if n == 2:
            d_rho_dz[:] = (rho[1] - rho[0]) / (z[1] - z[0])
        else:
            d_rho_dz[1:-1] = (rho[2:] - rho[:-2]) / (z[2:] - z[:-2])
            d_rho_dz[0] = (rho[1] - rho[0]) / (z[1] - z[0])
            d_rho_dz[-1] = (rho[-1] - rho[-2]) / (z[-1] - z[-2])

        integrand = 2.0 * np.pi * rho * np.sqrt(1.0 + d_rho_dz ** 2)

        return float(_simpson_fast(integrand, z))

    @staticmethod
    def find_star_convex_shift(
            z_points: np.ndarray,
            rho_points: np.ndarray,
            z_sh: float,
            n_check: int = 720,
            shift_step: float = 0.1
    ) -> Tuple['CylindricalToSphericalConverter', float]:
        """Find a z-shift that makes the shape star-convex (unambiguously convertible).

        Uses intelligent neck-detection to find optimal shift. Falls back to
        incremental shifting if no neck structure is found.

        Args:
            z_points: Original z coordinates.
            rho_points: Original ρ(z) values.
            z_sh: The shape's center-of-mass z-shift (from FoS parameters).
            n_check: Number of points for convertibility check.
            shift_step: Step size for fallback shift search in fm.

        Returns:
            Tuple of (converter with shifted z, total shift applied).
        """
        z_work = z_points.copy()
        conv = CylindricalToSphericalConverter(z_work, rho_points)

        if conv.is_unambiguously_convertible(n_check):
            return conv, 0.0

        # Try neck-centered shifting first
        z_neck = conv._find_neck_z_position(n_check)

        if z_neck is not None:
            # Try centering the neck at origin
            shift = -z_neck
            z_work = z_points + shift
            conv = CylindricalToSphericalConverter(z_work, rho_points)
            if conv.is_unambiguously_convertible(n_check):
                return conv, shift

            # Try adjustments around the neck position
            for delta in [0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2]:
                shift = -z_neck + delta
                z_work = z_points + shift
                conv = CylindricalToSphericalConverter(z_work, rho_points)
                if conv.is_unambiguously_convertible(n_check):
                    return conv, shift

        # Fallback: incremental shifting
        direction = -1.0 if z_sh >= 0 else 1.0
        max_shift = (float(np.max(z_points)) - float(np.min(z_points))) / 2.0
        shift = 0.0

        while abs(shift) < max_shift:
            shift += direction * shift_step
            z_work = z_points + shift
            conv = CylindricalToSphericalConverter(z_work, rho_points)
            if conv.is_unambiguously_convertible(n_check):
                return conv, shift

        # Return last attempt even if not convertible
        return conv, shift

    @staticmethod
    def find_optimal_beta_shift(
            z_points: np.ndarray,
            rho_points: np.ndarray,
            nucleons: int,
            z_sh: float = 0.0,
            n_check: int = 7200,
            search_range: float = 0.5,
            search_step: float = 0.05,
            l_max_test: int = 32
    ) -> Tuple['CylindricalToSphericalConverter', float, dict]:
        """Find optimal z-shift for beta parameter fitting using combined metric.

        Strategy:
        1. Find a star-convex shift using the existing method
        2. Search around that shift to minimize beta fit error
        3. Evaluate using a quick l_max_test fit
        4. Optimize combined metric: 1.0*surface_diff + 0.2*rmse + 0.5*L_inf

        Args:
            z_points: Original z coordinates.
            rho_points: Original ρ(z) values.
            nucleons: Number of nucleons (for beta calculation).
            z_sh: The shape's center-of-mass z-shift (from FoS parameters).
            n_check: Number of points for calculations.
            search_range: Range to search around star-convex shift (±fm).
            search_step: Step size for shift search in fm.
            l_max_test: Maximum l for quick beta fit evaluation.

        Returns:
            Tuple of (best converter, best shift, metrics_dict).
        """
        from src.parameterizations.beta import BetaDeformationCalculator

        # Step 1: Find base star-convex shift
        base_conv, base_shift = CylindricalToSphericalConverter.find_star_convex_shift(
            z_points, rho_points, z_sh, n_check
        )

        if not base_conv.is_unambiguously_convertible(n_check):
            print("Warning: Base shift is not star-convex. Results may be unreliable.")
            return base_conv, base_shift, {}

        # Step 2: Define search range
        shifts_to_test = np.arange(
            base_shift - search_range,
            base_shift + search_range + search_step / 2,
            search_step
        )

        best_shift = base_shift
        best_metric_value = float('inf')
        best_metrics = {}

        print(f"Optimizing z-shift: testing {len(shifts_to_test)} shifts around {base_shift:.2f} fm...")

        for i, shift in enumerate(shifts_to_test):
            # Apply shift
            z_shifted = z_points + shift
            conv = CylindricalToSphericalConverter(z_shifted, rho_points)

            # Check if still star-convex
            if not conv.is_unambiguously_convertible(n_check):
                continue

            # Convert to spherical
            theta, r_spherical = conv.convert_to_spherical(n_check)

            # Quick beta fit
            beta_calc = BetaDeformationCalculator(theta, r_spherical, nucleons)
            betas = beta_calc.calculate_beta_range(1, l_max_test)

            # Reconstruct
            theta_recon, r_recon = beta_calc.reconstruct_shape(betas, n_check)

            # Calculate reference surface
            ref_surface = CylindricalToSphericalConverter._calculate_surface_spherical(
                theta, r_spherical
            )

            # Calculate fit surface
            fit_surface = CylindricalToSphericalConverter._calculate_surface_spherical(
                theta_recon, r_recon
            )

            # Calculate metrics
            surface_diff = abs(fit_surface - ref_surface)

            # RMSE between original and reconstructed
            diff = r_spherical - r_recon
            rmse = float(np.sqrt(np.mean(diff ** 2)))

            # L-infinity (maximum absolute error)
            l_inf = float(np.max(np.abs(diff)))

            # Combined metric with specified weights
            # surface_diff: 1.0, rmse: 0.2, L_inf: 0.5
            metric_value = 1.0 * surface_diff + 0.2 * rmse + 0.5 * l_inf

            # Track best
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_shift = shift
                best_metrics = {
                    'surface_diff': surface_diff,
                    'rmse': rmse,
                    'l_infinity': l_inf,
                    'combined_metric': metric_value,
                    'shift': shift,
                    'l_max_test': l_max_test
                }

            # Progress indicator (every 5 shifts)
            if (i + 1) % 5 == 0 or (i + 1) == len(shifts_to_test):
                print(f"  Tested {i + 1}/{len(shifts_to_test)} shifts... "
                      f"Best: {best_shift:.2f} fm (metric: {best_metric_value:.4f})")

        # Step 3: Return the best converter
        z_best = z_points + best_shift
        best_conv = CylindricalToSphericalConverter(z_best, rho_points)

        print(f"Optimal shift found: {best_shift:.2f} fm (base was {base_shift:.2f} fm)")
        if best_metrics:
            print(f"  Surface Δ:   {best_metrics['surface_diff']:.4f} fm²")
            print(f"  RMSE:        {best_metrics['rmse']:.4f} fm")
            print(f"  L_inf:       {best_metrics['l_infinity']:.4f} fm")
            print(f"  Combined:    {best_metrics['combined_metric']:.4f}")

        return best_conv, best_shift, best_metrics
