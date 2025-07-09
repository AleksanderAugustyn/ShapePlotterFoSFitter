"""
FoS Coordinate Transformer
Converts Fourier-over-Spheroid shapes from ρ(z) to R(θ) representation
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class FoSCoordinateTransformer:
    """
    Class for transforming FoS shape coordinates from cylindrical ρ(z) to spherical R(θ).
    
    For axially symmetric nuclear shapes, this class converts from the native
    FoS representation ρ(z) to the spherical coordinate representation R(θ)
    where θ is the polar angle measured from the positive z-axis.
    """

    def __init__(self, z_coords: np.ndarray, rho_coords: np.ndarray,
                 center_shift: float = 0.0):
        """
        Initialize the coordinate transformer.
        
        Args:
            z_coords: Array of z coordinates from FoS calculation
            rho_coords: Array of ρ coordinates from FoS calculation
            center_shift: Optional shift to apply to z coordinates (default: 0.0)
        """
        self.z_original = z_coords.copy()
        self.rho_original = rho_coords.copy()
        self.center_shift = center_shift

        # Apply center shift
        self.z_coords = z_coords - center_shift
        self.rho_coords = rho_coords.copy()

        # Filter out invalid points (where rho = 0)
        valid_mask = self.rho_coords > 1e-10
        self.z_coords = self.z_coords[valid_mask]
        self.rho_coords = self.rho_coords[valid_mask]

        # Ensure monotonic ordering for interpolation
        self._prepare_interpolation()

    def _prepare_interpolation(self):
        """Prepare interpolation functions for coordinate transformation."""
        if len(self.z_coords) < 2:
            raise ValueError("Need at least 2 valid coordinate points for transformation")

        # Sort by z coordinate to ensure monotonic interpolation
        sort_indices = np.argsort(self.z_coords)
        self.z_sorted = self.z_coords[sort_indices]
        self.rho_sorted = self.rho_coords[sort_indices]

        # Create interpolation function: ρ = f(z)
        self.rho_interp = interp1d(
            self.z_sorted, self.rho_sorted,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        # Find the extrema for transformation bounds
        self.z_min = np.min(self.z_sorted)
        self.z_max = np.max(self.z_sorted)
        self.rho_max = np.max(self.rho_sorted)

    def transform_to_spherical(self, n_theta: int = 180) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from ρ(z) to R(θ) representation.
        
        Args:
            n_theta: Number of angular points to calculate (default: 180)
            
        Returns:
            theta: Array of polar angles in radians [0, π]
            R: Array of radial distances R(θ)
        """
        # Create theta array from 0 to π
        theta = np.linspace(0, np.pi, n_theta)
        R = np.zeros_like(theta)

        for i, th in enumerate(theta):
            if th == 0:
                # At θ = 0 (north pole), R = |z_max|
                R[i] = abs(self.z_max)
            elif th == np.pi:
                # At θ = π (south pole), R = |z_min|
                R[i] = abs(self.z_min)
            else:
                # For general θ, find intersection with surface
                R[i] = self._find_radius_at_theta(th)

        return theta, R

    def _find_radius_at_theta(self, theta: float) -> float:
        """
        Find the radius R at a given polar angle θ.
        
        For a point on the surface in spherical coordinates (R, θ):
        z = R * cos(θ)
        ρ = R * sin(θ)
        
        We need to find R such that the point (z, ρ) lies on the FoS surface.
        
        Args:
            theta: Polar angle in radians
            
        Returns:
            R: Radial distance at angle θ
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        if abs(sin_theta) < 1e-10:
            # Very close to poles
            if cos_theta > 0:
                return abs(self.z_max)
            else:
                return abs(self.z_min)

        # For general angles, we need to solve: ρ_surface(z) = R * sin(θ)
        # where z = R * cos(θ)
        # This gives us: ρ_surface(R * cos(θ)) = R * sin(θ)

        def objective(R):
            z = R * cos_theta
            if z < self.z_min or z > self.z_max:
                return float('inf')

            rho_surface = self.rho_interp(z)
            rho_expected = R * sin_theta
            return abs(rho_surface - rho_expected)

        # Search for the optimal R
        # Start with a reasonable bound based on the maximum extent
        max_extent = max(abs(self.z_max), abs(self.z_min), self.rho_max)

        try:
            result = minimize_scalar(objective, bounds=(0, 2 * max_extent), method='bounded')
            return result.x if result.success else max_extent
        except:
            # Fallback: use simple geometric approximation
            z_est = max_extent * cos_theta
            if self.z_min <= z_est <= self.z_max:
                rho_est = self.rho_interp(z_est)
                return np.sqrt(z_est ** 2 + rho_est ** 2)
            else:
                return max_extent

    def get_cartesian_coordinates(self, n_theta: int = 180, n_phi: int = 360) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get full 3D Cartesian coordinates assuming axial symmetry.
        
        Args:
            n_theta: Number of polar angle points
            n_phi: Number of azimuthal angle points
            
        Returns:
            x, y, z: 3D coordinate arrays
        """
        theta, R = self.transform_to_spherical(n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)

        # Create meshgrid
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        R_MESH = np.tile(R[:, np.newaxis], (1, n_phi))

        # Convert to Cartesian
        x = R_MESH * np.sin(THETA) * np.cos(PHI)
        y = R_MESH * np.sin(THETA) * np.sin(PHI)
        z = R_MESH * np.cos(THETA)

        return x, y, z

    def get_surface_area(self, n_theta: int = 180) -> float:
        """
        Calculate the surface area of the nuclear shape.
        
        Args:
            n_theta: Number of angular points for integration
            
        Returns:
            Surface area in fm²
        """
        theta, R = self.transform_to_spherical(n_theta)

        # Calculate dR/dθ for surface area element
        dR_dtheta = np.gradient(R, theta)

        # Surface area element: dS = 2π R sqrt(R² + (dR/dθ)²) sin(θ) dθ
        integrand = 2 * np.pi * R * np.sqrt(R ** 2 + dR_dtheta ** 2) * np.sin(theta)

        # Integrate using trapezoidal rule
        surface_area = np.trapz(integrand, theta)

        return surface_area

    def get_moment_of_inertia(self, n_theta: int = 180) -> Tuple[float, float]:
        """
        Calculate moments of inertia about different axes.
        
        Args:
            n_theta: Number of angular points for integration
            
        Returns:
            I_parallel: Moment of inertia about symmetry axis (z-axis)
            I_perpendicular: Moment of inertia about perpendicular axis
        """
        theta, R = self.transform_to_spherical(n_theta)

        # For axially symmetric shapes:
        # I_parallel = ∫ ρ² * R² * sin³(θ) dθ (integrated over φ gives 2π)
        # I_perpendicular = ∫ (R² sin²(θ) + R² cos²(θ)/2) * R² * sin(θ) dθ

        # Assuming uniform density, we integrate R⁵ terms
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Parallel moment (about z-axis)
        I_parallel_integrand = 2 * np.pi * R ** 5 * sin_theta ** 3
        I_parallel = np.trapz(I_parallel_integrand, theta)

        # Perpendicular moment (about x or y axis)
        I_perp_integrand = 2 * np.pi * R ** 5 * sin_theta * (sin_theta ** 2 + cos_theta ** 2 / 2)
        I_perpendicular = np.trapz(I_perp_integrand, theta)

        return I_parallel, I_perpendicular

    def export_spherical_data(self, filename: str, n_theta: int = 180):
        """
        Export the spherical coordinate data to a file.
        
        Args:
            filename: Output filename
            n_theta: Number of angular points
        """
        theta, R = self.transform_to_spherical(n_theta)

        # Convert theta to degrees for easier interpretation
        theta_deg = np.degrees(theta)

        # Create output data
        data = np.column_stack((theta_deg, theta, R))

        # Save to file
        header = "Theta_deg\tTheta_rad\tR_fm\n"
        header += "# FoS shape in spherical coordinates\n"
        header += f"# {len(theta)} points from 0 to 180 degrees\n"

        np.savetxt(filename, data, delimiter='\t', header=header,
                   fmt=['%.2f', '%.6f', '%.6f'])
