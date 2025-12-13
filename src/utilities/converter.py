"""Module for converting nuclear shapes from cylindrical to spherical coordinates."""
from typing import Optional, Tuple, TypedDict

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import brentq, fsolve


class ConversionMetrics(TypedDict):
    """Metrics for cylindrical-to-spherical coordinate conversion accuracy."""
    rmse: float  # Root-mean-square error in fm
    l_infinity: float  # Maximum absolute error in fm
    volume_diff: float  # Absolute volume difference in fm³
    surface_diff: float  # Absolute surface area difference in fm²
    z_shift: float  # Applied z-shift to make shape star-convex in fm


class CylindricalToSphericalConverter:
    """Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ)."""

    def __init__(self, z_points: np.ndarray, rho_points: np.ndarray):
        self.z_points = np.asarray(z_points)
        self.rho_points = np.asarray(rho_points)

        # Sort by z
        sort_idx = np.argsort(self.z_points)
        self.z_points = self.z_points[sort_idx]
        self.rho_points = self.rho_points[sort_idx]

        self.rho_interp = interp1d(
            self.z_points, self.rho_points, kind='cubic', bounds_error=False, fill_value=0.0
        )
        self.z_min: float = np.min(self.z_points[self.rho_points > 0])
        self.z_max: float = np.max(self.z_points[self.rho_points > 0])

    def rho_of_z(self, z: float) -> float:
        """Returns ρ(z) using interpolation."""
        if z < self.z_min or z > self.z_max:
            return 0.0
        return float(self.rho_interp(z))

    def _solve_r_for_theta(self, theta: float, initial_guess: Optional[float] = None) -> float:
        """Solves for r given theta using root-finding."""
        if theta == 0:
            return abs(self.z_max)

        if theta == np.pi:
            return abs(self.z_min)

        if abs(theta - np.pi / 2) < 1e-10:
            return self.rho_of_z(0.0)

        def equation(r: float) -> float:
            """Equation to solve: r * sin(theta) - rho(z) = 0, where z = r * cos(theta)"""
            z: float = r * np.cos(theta)
            if z < self.z_min or z > self.z_max:
                return r

            return float(r * np.sin(theta) - self.rho_of_z(z))

        # Bracket and solve
        r_max = 2 * max(abs(self.z_max), abs(self.z_min), np.max(self.rho_points))
        try:
            if equation(0) * equation(r_max) < 0:
                r_solution = brentq(equation, 0, r_max)
            else:
                initial_guess = initial_guess or max(abs(self.z_max), abs(self.z_min))
                r_solution = fsolve(equation, initial_guess)[0]
        except (RuntimeError, ValueError):
            r_solution = 0.0

        return max(0.0, float(r_solution))

    def convert_to_spherical(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the shape to spherical coordinates r(θ)."""
        theta_points = np.linspace(0, np.pi, n_theta)
        r_points = np.array([self._solve_r_for_theta(t) for t in theta_points])
        return theta_points, r_points

    def is_unambiguously_convertible(self, n_points: int = 720, tolerance: float = 1e-9) -> bool:
        """Checks if the shape is star-shaped w.r.t origin (monotonic theta(z))."""
        if self.z_min >= self.z_max:
            return True

        epsilon = 1e-6
        z_check = np.linspace(self.z_min + epsilon, self.z_max - epsilon, n_points)
        h = 1e-6
        rho_vals = self.rho_interp(z_check)
        rho_prime = (self.rho_interp(z_check + h) - self.rho_interp(z_check - h)) / (2 * h)

        # Condition: z * rho'(z) - rho(z) <= 0
        test_values = z_check * rho_prime - rho_vals
        return not np.any(test_values > tolerance)

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
        # Convert to spherical
        n_points = len(z_original)
        theta, r_spherical = self.convert_to_spherical(n_points)

        # Convert back to cylindrical
        z_roundtrip = r_spherical * np.cos(theta) - z_shift
        rho_roundtrip = r_spherical * np.sin(theta)

        # Sort for interpolation
        sort_idx = np.argsort(z_roundtrip)
        z_rt_sorted = z_roundtrip[sort_idx]
        rho_rt_sorted = rho_roundtrip[sort_idx]

        # Remove duplicates for interpolation
        unique_mask = np.concatenate(([True], np.diff(z_rt_sorted) > 1e-12))
        z_rt_sorted = z_rt_sorted[unique_mask]
        rho_rt_sorted = rho_rt_sorted[unique_mask]

        # Interpolate roundtrip rho at original z points
        # Use fill_value=0.0 since rho should be 0 at the tips (beyond the roundtrip z range)
        rho_interp_func = interp1d(
            z_rt_sorted, rho_rt_sorted, kind='cubic', bounds_error=False, fill_value=0.0
        )
        rho_rt_at_z = rho_interp_func(z_original)

        # Calculate shape errors
        diff = rho_original - rho_rt_at_z
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        l_inf = float(np.max(np.abs(diff)))

        # Calculate volumes
        vol_original = float(simpson(np.pi * rho_original ** 2, x=z_original))
        vol_spherical = self._calculate_volume_spherical(theta, r_spherical)
        volume_diff = abs(vol_spherical - vol_original)

        # Calculate surface areas
        surf_original = self._calculate_surface_cylindrical(z_original, rho_original)
        surf_spherical = self._calculate_surface_spherical(theta, r_spherical)
        surface_diff = abs(surf_spherical - surf_original)

        return ConversionMetrics(
            rmse=rmse,
            l_infinity=l_inf,
            volume_diff=volume_diff,
            surface_diff=surface_diff,
            z_shift=z_shift
        )

    @staticmethod
    def _calculate_volume_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate volume from spherical coordinates."""
        integrand = r ** 3 * np.sin(theta)
        return float((2 * np.pi / 3) * simpson(integrand, x=theta))

    @staticmethod
    def _calculate_surface_spherical(theta: np.ndarray, r: np.ndarray) -> float:
        """Calculate surface area from spherical coordinates."""
        dr_dtheta = np.gradient(r, theta)
        integrand = r * np.sin(theta) * np.sqrt(r ** 2 + dr_dtheta ** 2)
        return float(2 * np.pi * simpson(integrand, x=theta))

    @staticmethod
    def _calculate_surface_cylindrical(z: np.ndarray, rho: np.ndarray) -> float:
        """Calculate surface area from cylindrical coordinates."""
        if len(z) < 2:
            return 0.0
        d_rho_dz = np.gradient(rho, z)
        integrand = 2 * np.pi * rho * np.sqrt(1 + d_rho_dz ** 2)
        return float(simpson(integrand, x=z))

    @staticmethod
    def find_star_convex_shift(
            z_points: np.ndarray,
            rho_points: np.ndarray,
            z_sh: float,
            n_check: int = 720,
            shift_step: float = 0.1
    ) -> Tuple['CylindricalToSphericalConverter', float]:
        """Find a z-shift that makes the shape star-convex (unambiguously convertible).

        Args:
            z_points: Original z coordinates.
            rho_points: Original ρ(z) values.
            z_sh: The shape's center-of-mass z-shift (from FoS parameters).
            n_check: Number of points for convertibility check.
            shift_step: Step size for shift search in fm.

        Returns:
            Tuple of (converter with shifted z, total shift applied).
        """
        z_work = z_points.copy()
        conv = CylindricalToSphericalConverter(z_work, rho_points)

        if conv.is_unambiguously_convertible(n_check):
            return conv, 0.0

        # Search direction: opposite to center-of-mass shift
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
