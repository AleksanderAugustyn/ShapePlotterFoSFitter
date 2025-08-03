"""
Cylindrical to Spherical Coordinate Converter for Nuclear Shapes
Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ)
"""

from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, brentq


class CylindricalToSphericalConverter:
    """
    Converts nuclear shapes from cylindrical coordinates ρ(z) to spherical coordinates r(θ).

    In cylindrical coordinates:
    - z: axial coordinate
    - ρ: radial coordinate perpendicular to z-axis

    In spherical coordinates:
    - r: radial distance from origin
    - θ: polar angle from positive z-axis (0 to π)
    - φ: azimuthal angle (not used due to axial symmetry)

    Relationships:
    - z = r cos(θ)
    - ρ = r sin(θ)
    - r = √(z² + ρ²)
    """

    def __init__(self, z_points: np.ndarray, rho_points: np.ndarray):
        """
        Initialize the converter with shape data in cylindrical coordinates.

        Args:
            z_points: Array of z coordinates
            rho_points: Array of corresponding ρ values
        """
        # Store the original data
        self.z_points = np.asarray(z_points)
        self.rho_points = np.asarray(rho_points)

        # Ensure points are sorted by z
        sort_idx = np.argsort(self.z_points)
        self.z_points = self.z_points[sort_idx]
        self.rho_points = self.rho_points[sort_idx]

        # Create an interpolation function for ρ(z)
        self.rho_interp = interp1d(
            self.z_points,
            self.rho_points,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        # Store shape boundaries
        self.z_min = np.min(self.z_points[self.rho_points > 0])
        self.z_max = np.max(self.z_points[self.rho_points > 0])

    def rho_of_z(self, z: float) -> float:
        """
        Get the radial coordinate ρ for a given axial coordinate z.

        Args:
            z: Axial coordinate

        Returns:
            ρ: Radial coordinate at z
        """
        if z < self.z_min or z > self.z_max:
            return 0.0
        return float(self.rho_interp(z))

    def _solve_r_for_theta(self, theta: float, initial_guess: Optional[float] = None) -> float:
        """
        Solve for r given θ using the implicit equation:
        r sin(θ) = ρ(r cos(θ))

        Args:
            theta: Polar angle in radians (0 to π)
            initial_guess: Initial guess for r

        Returns:
            r: Radial distance at an angle θ
        """
        # Handle special cases
        if theta == 0:  # North Pole
            return abs(self.z_max)
        if theta == np.pi:  # South Pole
            return abs(self.z_min)
        if abs(theta - np.pi / 2) < 1e-10:  # Equator
            return self.rho_of_z(0.0)

        # For general θ, we need to solve: r sin(θ) = ρ(r cos(θ))
        def equation(r):
            z = r * np.cos(theta)
            # Check if z is within the valid range
            if z < self.z_min or z > self.z_max:
                return r  # This makes r = 0 a solution outside the shape
            return r * np.sin(theta) - self.rho_of_z(z)

        # Get a good initial guess
        if initial_guess is None:
            # Try to estimate based on the maximum extent
            z_at_theta = self.z_max * np.cos(theta) if theta < np.pi / 2 else self.z_min * np.cos(theta)
            if self.z_min <= z_at_theta <= self.z_max:
                rho_at_z = self.rho_of_z(z_at_theta)
                if np.sin(theta) > 1e-10:
                    initial_guess = rho_at_z / np.sin(theta)
                else:
                    initial_guess = abs(z_at_theta / np.cos(theta))
            else:
                initial_guess = max(abs(self.z_max), abs(self.z_min))

        # Try to bracket the root
        try:
            # Find bounds for the root
            r_min = 0
            r_max = 2 * max(abs(self.z_max), abs(self.z_min), np.max(self.rho_points))

            # Check if there's a sign change
            f_min = equation(r_min)
            f_max = equation(r_max)

            if f_min * f_max < 0:
                # Use Brent's method if we have a bracketed root
                r_solution = brentq(equation, r_min, r_max)
            else:
                # Fall back to solve
                result = fsolve(equation, initial_guess, full_output=True)
                r_solution = result[0][0]

                # Check if a solution is valid
                if result[2] != 1 or r_solution < 0:
                    r_solution = 0.0
        except (RuntimeError, ValueError):
            # If numerical methods fail (e.g., non-convergence), default to 0
            r_solution = 0.0

        return max(0.0, r_solution)  # Ensure non-negative

    def convert_to_spherical(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to spherical coordinates.

        Args:
            n_theta: Number of θ points (from 0 to π)

        Returns:
            theta_points: Array of polar angles
            r_points: Array of radial distances r(θ)
        """
        # Create theta array from 0 to π
        theta_points = np.linspace(0, np.pi, n_theta)

        # Calculate r for each θ using the implicit equation
        r_points = np.array([self._solve_r_for_theta(theta) for theta in theta_points])

        return theta_points, r_points

    def convert_to_cylindrical(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to cylindrical coordinates (z, rho) for 2D plotting.
        This gives the cross-section of the axially symmetric shape.

        Args:
            n_theta: Number of θ points

        Returns:
            z_points: Array of z coordinates (rcos(θ))
            rho_points: Array of radial coordinates ρ (rsin(θ))
        """
        theta_points, r_points = self.convert_to_spherical(n_theta)

        # Convert to Cartesian for 2D cross-section
        z_points = r_points * np.sin(theta_points)
        rho_points = r_points * np.cos(theta_points)

        return z_points, rho_points

    def convert_to_cartesian(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the shape to Cartesian coordinates (x, y) for 2D plotting.
        This gives the cross-section of the axially symmetric shape.

        Args:
            n_theta: Number of θ points

        Returns:
            x_points: Array of x coordinates
            y_points: Array of y coordinates
        """
        theta_points, r_points = self.convert_to_spherical(n_theta)

        # Convert to Cartesian for 2D cross-section
        x_points = r_points * np.cos(theta_points)
        y_points = r_points * np.sin(theta_points)

        return x_points, y_points

    def get_max_radius(self) -> float:
        """
        Get the maximum radial distance of the shape.

        Returns:
            Maximum radius
        """
        # Sample many angles to find maximum
        theta_sample = np.linspace(0, np.pi, 361)
        r_sample = np.array([self._solve_r_for_theta(theta) for theta in theta_sample])
        return np.max(r_sample)

    def get_shape_at_angle(self, theta: float) -> Tuple[float, float, float]:
        """
        Get the shape parameters at a specific angle θ.

        Args:
            theta: Polar angle in radians

        Returns:
            r: Radial distance
            z: Axial coordinate
            rho: Radial coordinate in a cylindrical system
        """
        r = self._solve_r_for_theta(theta)
        z = r * np.cos(theta)
        rho = r * np.sin(theta)
        return r, z, rho

    def validate_conversion(self, theta_converted: np.ndarray, r_converted: np.ndarray) -> dict:
        """
        Validate the conversion by comparing with original data points.

        Args:
            theta_converted: Array of theta values from conversion
            r_converted: Array of r values from conversion

        Returns:
            Dictionary with validation metrics
        """
        errors_z = []
        errors_rho = []

        # Create an interpolation function for the converted curve
        r_interp = interp1d(theta_converted, r_converted, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')

        # Check each original data point
        for i in range(len(self.z_points)):
            z_original = self.z_points[i]
            rho_original = self.rho_points[i]

            # Skip points at origin
            if abs(z_original) < 1e-10 and abs(rho_original) < 1e-10:
                continue

            # Convert the original point to spherical
            r_original = np.sqrt(z_original ** 2 + rho_original ** 2)
            if r_original > 0:
                theta_original = np.arccos(np.clip(z_original / r_original, -1, 1))

                # Get r from the converted curve at this theta
                r_from_conversion = r_interp(theta_original)

                # Convert back to cylindrical
                z_reconstructed = r_from_conversion * np.cos(theta_original)
                rho_reconstructed = r_from_conversion * np.sin(theta_original)

                # Calculate errors
                errors_z.append((z_original - z_reconstructed) ** 2)
                errors_rho.append((rho_original - rho_reconstructed) ** 2)

        # Combined RMSE using both z and rho errors
        total_error = np.array(errors_z) + np.array(errors_rho)

        return {
            'rmse_combined': np.sqrt(np.mean(total_error)) if len(total_error) > 0 else 0,
            'rmse_z': np.sqrt(np.mean(errors_z)) if errors_z else 0,
            'rmse_rho': np.sqrt(np.mean(errors_rho)) if errors_rho else 0,
            'max_error': np.sqrt(np.max(total_error)) if len(total_error) > 0 else 0,
            'n_valid_points': len(errors_z)
        }

    def is_unambiguously_convertible(self, n_points: int = 720, tolerance: float = 1e-9) -> bool:
        """
        Checks if the shape can be unambiguously converted to spherical coordinates.

        The conversion is unambiguous if the shape is "star-shaped" with respect
        to the origin. This is checked by verifying that the angle theta(z) is
        a monotonic function of z. This is equivalent to checking the sign of
        the expression `z * rho'(z) - rho(z)`. For a valid, star-shaped
        nuclear shape (with z=0 at the center), this expression should always
        be less than or equal to zero.

        Args:
            n_points: Number of points along the z-axis to check.
            tolerance: A small numerical tolerance for the check.

        Returns:
            True if the shape is unambiguously convertible, False otherwise.
        """
        # Create a set of z-points to check, avoiding the very ends where rho=0
        # and the derivative might be ill-defined.
        epsilon = 1e-6
        # Ensure z_min and z_max are valid before creating linspace
        if self.z_min >= self.z_max:
            return True  # Treat empty or point shapes as unambiguously convertible.

        z_check_points = np.linspace(self.z_min + epsilon, self.z_max - epsilon, n_points)

        # Use a small step for numerical differentiation
        h = 1e-6

        # Calculate rho and its derivative at the check points
        rho_vals = self.rho_interp(z_check_points)
        # Use central difference for the derivative
        rho_prime_vals = (self.rho_interp(z_check_points + h) - self.rho_interp(z_check_points - h)) / (2 * h)

        # Calculate the test expression: z * rho'(z) - rho(z)
        test_values = z_check_points * rho_prime_vals - rho_vals

        # The shape is unambiguously convertible if all test values are <= 0.
        # We allow for a small positive tolerance due to numerical precision.
        if np.any(test_values > tolerance):
            return False

        return True
