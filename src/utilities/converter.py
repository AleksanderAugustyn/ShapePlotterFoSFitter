from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, fsolve


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
        self.z_min = np.min(self.z_points[self.rho_points > 0])
        self.z_max = np.max(self.z_points[self.rho_points > 0])

    def rho_of_z(self, z: float) -> float:
        if z < self.z_min or z > self.z_max:
            return 0.0
        return float(self.rho_interp(z))

    def _solve_r_for_theta(self, theta: float, initial_guess: Optional[float] = None) -> float:
        if theta == 0: return abs(self.z_max)
        if theta == np.pi: return abs(self.z_min)
        if abs(theta - np.pi / 2) < 1e-10: return self.rho_of_z(0.0)

        def equation(r):
            z = r * np.cos(theta)
            if z < self.z_min or z > self.z_max: return r
            return r * np.sin(theta) - self.rho_of_z(z)

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

        return max(0.0, r_solution)

    def convert_to_spherical(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        theta_points = np.linspace(0, np.pi, n_theta)
        r_points = np.array([self._solve_r_for_theta(t) for t in theta_points])
        return theta_points, r_points

    def is_unambiguously_convertible(self, n_points: int = 720, tolerance: float = 1e-9) -> bool:
        """Checks if shape is star-shaped w.r.t origin (monotonic theta(z))."""
        if self.z_min >= self.z_max: return True
        epsilon = 1e-6
        z_check = np.linspace(self.z_min + epsilon, self.z_max - epsilon, n_points)
        h = 1e-6
        rho_vals = self.rho_interp(z_check)
        rho_prime = (self.rho_interp(z_check + h) - self.rho_interp(z_check - h)) / (2 * h)

        # Condition: z * rho'(z) - rho(z) <= 0
        test_values = z_check * rho_prime - rho_vals
        return not np.any(test_values > tolerance)
