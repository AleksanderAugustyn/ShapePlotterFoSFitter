"""
Beta Deformation Parameter Calculator
Calculates nuclear deformation parameters β_λ using spherical harmonic decomposition
"""

from typing import Dict, Tuple, Optional, TypedDict

import numpy as np
from scipy.integrate import simpson
from scipy.special import sph_harm_y


class FitResult(TypedDict):
    """Type definition for fit_beta_parameters_rmse return value."""
    beta_fitted: Dict[int, float]
    scaling_factor_fitted: float
    scaling_factor_volume: float
    rmse: float
    beta_analytical: Dict[int, float]

class BetaDeformationCalculator:
    """
    Calculate beta deformation parameters for nuclear shapes using spherical harmonics.

    The nuclear surface is expanded as:
    R(θ, φ) = c R₀ [1 + Σ_λμ β_λμ Y_λμ(θ, φ)]

    For axially symmetric shapes (m=0):
    R(θ) = c R₀ [1 + Σ_λ β_λ0 Y_λ0(θ)]
    where c is a volume fixing factor and R₀ is the radius of a sphere with the same volume as the nucleus.
    """

    def __init__(self, theta: np.ndarray, radius: np.ndarray, number_of_nucleons: int):
        """
        Initialize the calculator with shape data.

        Args:
            theta: Array of polar angles (0 to π)
            radius: Array of radial distances r(θ)
        """
        self.theta = np.asarray(theta)
        self.r = np.asarray(radius)
        self.r0: float = 1.16
        self.radius0: float = 1.16 * number_of_nucleons ** (1 / 3)

        # Ensure theta is sorted
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.r = self.r[sort_idx]

        # Precompute sin(theta) for integration
        self.sin_theta = np.sin(self.theta)

        # Cache for spherical harmonics
        self._ylm_cache: dict[Tuple[int, int], np.ndarray] = {}

    def _get_ylm(self, l: int, m: int = 0) -> np.ndarray:
        """
        Get spherical harmonic Y_lm(θ, φ) for m=0 (axial symmetry).

        For m=0, Y_l0 only depends on θ and is real.

        Args:
            l: Degree of spherical harmonic
            m: Order (default 0 for axial symmetry)

        Returns:
            Array of Y_l0(θ) values
        """
        key = (l, m)
        if key not in self._ylm_cache:
            # sph_harm_y uses (l, m, theta, phi) convention
            # For m=0, the result is independent of phi
            phi: float = 0.0
            self._ylm_cache[key] = sph_harm_y(l, m, self.theta, phi).real
        return self._ylm_cache[key]

    def calculate_beta_parameters(self, l_max: int = 12) -> Dict[int, float]:
        """
        Calculate β_λ0 deformation parameters.

        β_λ0 = sqrt(4π) * ∫ r(θ) Y_λ0(θ) sin(θ) dθ / ∫ r(θ) Y_00(θ) sin(θ) dθ

        Args:
            l_max: Maximum l value to calculate

        Returns:
            Dictionary of β_λ parameters
        """
        beta = {}

        # Calculate denominator: ∫ Y_00(θ) sin(θ) dθ
        # For l=0, Y_00 = 1/sqrt(4π)
        integrand_denominator = self.r * self._get_ylm(0, 0) * self.sin_theta
        denominator = simpson(integrand_denominator, x=self.theta)

        for l in range(1, l_max + 1):
            # Get spherical harmonic
            ylm = self._get_ylm(l, 0)

            # Calculate numerator: ∫ r(θ) Y_λ0(θ) sin(θ) dθ
            integrand_numerator = self.r * ylm * self.sin_theta
            numerator = simpson(integrand_numerator, x=self.theta)

            if abs(denominator) > 1e-10:
                beta[l] = np.sqrt(4 * np.pi) * numerator / denominator
            else:
                beta[l] = 0.0

        return beta

    @staticmethod
    def calculate_rms_deformation(beta: Dict[int, float], l_max: Optional[int] = None) -> float:
        """
        Calculate root-mean-square deformation.

        Δ_rms = sqrt(Σ_λ β_λ²)

        Args:
            beta: Dictionary of beta parameters
            l_max: Maximum l to include (None = all available)

        Returns:
            RMS deformation
        """
        if l_max is None:
            l_values = [l for l in beta.keys() if l > 0]
        else:
            l_values = [l for l in beta.keys() if 0 < l <= l_max]

        sum_squared: float = sum(beta[l] ** 2 for l in l_values)
        return np.sqrt(sum_squared)

    @staticmethod
    def calculate_surface_area_in_spherical_coordinates(radius: np.ndarray, theta: np.ndarray) -> float:
        """
        Calculate the surface area of an axially symmetric shape in spherical coordinates.

        A = 2π ∫ r(θ) sin(θ) √ (r(θ)² + (dr/dθ)²) dθ

        Args:
            radius: Array of radial distances r(θ)
            theta: Array of polar angles θ

        Returns:
            Surface area of the shape
        """
        # Calculate the derivative dr/dθ using finite differences
        d_r = np.diff(radius)
        d_theta = np.diff(theta)

        dr_dtheta = d_r / d_theta

        # Calculate the integrand
        integrand = 2 * np.pi * radius[:-1] * np.sin(theta[:-1]) * np.sqrt(radius[:-1] ** 2 + dr_dtheta ** 2)

        # Integrate using Simpson's rule
        surface_area: float = float(simpson(integrand, x=theta[:-1]))

        return surface_area

    @staticmethod
    def calculate_volume_in_spherical_coordinates(radius: np.ndarray, theta: np.ndarray) -> float:
        """
        Calculate the volume of the shape in spherical coordinates.

        V = ∫ r(θ)² sin(θ) dr dθ dφ = ∫ r(θ)² sin(θ) dr dθ * 2π

        Args:
            radius: Array of radial distances r(θ)
            theta: Array of polar angles θ

        Returns:
            Volume of the shape
        """
        # Volume integral in spherical coordinates
        integrand = radius ** 3 * np.sin(theta) * 2 / 3 * np.pi
        volume: float = float(simpson(integrand, x=theta))

        return volume

    @staticmethod
    def calculate_center_of_mass_in_spherical_coordinates(radius: np.ndarray, theta: np.ndarray) -> float:
        """
        Calculate the center of mass in spherical coordinates.

        CM = ∫ r(θ)² sin(θ) dr dθ dφ / ∫ r(θ) sin(θ) dr dθ dφ

        Args:
            radius: Array of radial distances r(θ)
            theta: Array of polar angles θ

        Returns:
            Center of mass in spherical coordinates
        """
        # Numerator: 2π ∫ r(θ)³ sin(θ) cos(θ) dr dθ = 1/2 π r(θ)⁴ ∫ sin(θ) cos(θ) dθ
        numerator: float = float(simpson(1 / 2 * np.pi * radius ** 4 * np.sin(theta) * np.cos(theta), x=theta))

        # Denominator: Volume integral = 2π ∫ r(θ)² sin(θ) dr dθ
        denominator: float = BetaDeformationCalculator.calculate_volume_in_spherical_coordinates(radius, theta)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def reconstruct_shape(self, beta: Dict[int, float], n_theta: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the shape from beta parameters.

        R(θ) = c R₀ [1 + Σ_λ β_λ Y_λ0(θ)]

        Args:
            beta: Dictionary of beta parameters
            n_theta: Number of theta points for reconstruction

        Returns:
            theta_reconstructed: Array of theta values
            radius_reconstructed: Array of reconstructed radii
        """
        theta_reconstructed = np.linspace(0, np.pi, n_theta)
        radius_pre_normalization = np.ones_like(theta_reconstructed) * self.radius0

        for l, beta_l in beta.items():
            if l == 0:
                continue  # Skip l=0 for shape reconstruction

            # Get Y_l0 at reconstruction points
            phi = 0
            ylm = sph_harm_y(l, 0, theta_reconstructed, phi).real

            # Add contribution
            radius_pre_normalization += self.radius0 * beta[l] * ylm

        # Calculate volume fixing factor
        volume_pre_normalization: float = self.calculate_volume_in_spherical_coordinates(radius_pre_normalization, theta_reconstructed)
        sphere_volume: float = (4 / 3) * np.pi * self.radius0 ** 3
        volume_fixing_factor: float = sphere_volume / volume_pre_normalization

        # Calculate radius fixing factor
        radius_fixing_factor: float = volume_fixing_factor ** (1 / 3)

        radius_reconstructed = radius_fixing_factor * radius_pre_normalization

        return theta_reconstructed, radius_reconstructed

    def _reconstruct_shape_with_scaling(self, beta_list: np.ndarray, scaling_factor: float,
                                        l_values: list, theta_eval: np.ndarray) -> np.ndarray:
        """
        Reconstruct shape with an explicit scaling factor (used for fitting).

        Args:
            beta_list: Array of beta values
            scaling_factor: Radius scaling factor
            l_values: List of l values corresponding to beta_list
            theta_eval: Theta values at which to evaluate

        Returns:
            Array of reconstructed radii
        """
        radius = np.ones_like(theta_eval) * self.radius0

        for i, (l, beta_l) in enumerate(zip(l_values, beta_list)):
            phi = 0
            ylm = sph_harm_y(l, 0, theta_eval, phi).real
            radius += self.radius0 * beta_l * ylm

        return scaling_factor * radius

    def fit_beta_parameters_rmse(self, l_max: int = 12) -> FitResult:
        """
        Fit β_λ0 deformation parameters by minimizing RMSE.

        This method treats the radius scaling factor as a free parameter
        during optimization, then compares with the volume-based scaling.

        Args:
            l_max: Maximum l value to fit

        Returns:
            Dictionary containing:
                - 'beta_fitted': Dictionary of fitted beta parameters
                - 'scaling_factor_fitted': Fitted radius scaling factor
                - 'scaling_factor_volume': Volume-based scaling factor
                - 'rmse': Root mean square error of the fit
                - 'beta_analytical': Analytical beta parameters for comparison
        """
        from scipy.optimize import minimize

        # Get an analytical solution as an initial guess
        beta_analytical = self.calculate_beta_parameters(l_max)

        # Prepare optimization
        l_values = list(range(1, l_max + 1))
        beta_initial = np.array([beta_analytical.get(l, 0.0) for l in l_values])

        # Add a scaling factor as the last parameter (start with 1.0)
        params_initial = np.append(beta_initial, 1.0)

        # Define objective function
        def objective(params):
            beta_values = params[:-1]
            scaling_factor = params[-1]

            # Reconstruct shape
            r_reconstructed = self._reconstruct_shape_with_scaling(
                beta_values, scaling_factor, l_values, self.theta
            )

            # Calculate RMSE
            rmse = np.sqrt(np.mean((r_reconstructed - self.r) ** 2))
            return rmse

        # Optimize
        result = minimize(objective, params_initial, method='L-BFGS-B')

        # Extract results
        beta_fitted_array = result.x[:-1]
        scaling_factor_fitted = result.x[-1]

        # Convert to dictionary
        beta_fitted = {l: beta for l, beta in zip(l_values, beta_fitted_array)}

        # Calculate what the scaling factor should be based on volume
        theta_test = np.linspace(0, np.pi, len(self.theta))
        radius_unnormalized = self._reconstruct_shape_with_scaling(
            beta_fitted_array, 1.0, l_values, theta_test
        )
        volume_unnormalized = self.calculate_volume_in_spherical_coordinates(
            radius_unnormalized, theta_test
        )
        sphere_volume = (4 / 3) * np.pi * self.radius0 ** 3
        volume_fixing_factor = sphere_volume / volume_unnormalized
        scaling_factor_volume = volume_fixing_factor ** (1 / 3)

        # Calculate final RMSE
        final_rmse = result.fun

        return {
            'beta_fitted': beta_fitted,
            'scaling_factor_fitted': scaling_factor_fitted,
            'scaling_factor_volume': scaling_factor_volume,
            'rmse': final_rmse,
            'beta_analytical': beta_analytical
        }
