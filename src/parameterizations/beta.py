from typing import Dict, Tuple, TypedDict

import numpy as np
from scipy.integrate import simpson
from scipy.special import sph_harm_y


class ShapeComparisonMetrics(TypedDict):
    rmse: float
    chi_squared: float  # Raw Chi-Squared
    chi_squared_reduced: float  # Reduced Chi-Squared (Chi^2 / DoF)
    l_infinity: float


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
        self._ylm_cache = {}

    def _get_ylm(self, l: int) -> np.ndarray:
        if l not in self._ylm_cache:
            self._ylm_cache[l] = sph_harm_y(l, 0, self.theta, 0.0).real
        return self._ylm_cache[l]

    def calculate_beta_parameters(self, l_max: int = 12) -> Dict[int, float]:
        """Calculates analytical beta parameters."""
        beta = {}
        # Denominator: ∫ Y_00 sin(θ) dθ (Always constant for l=0)
        denom_integrand = self.r * self._get_ylm(0) * self.sin_theta
        denominator = simpson(denom_integrand, x=self.theta)

        for l in range(1, l_max + 1):
            num_integrand = self.r * self._get_ylm(l) * self.sin_theta
            numerator = simpson(num_integrand, x=self.theta)
            beta[l] = np.sqrt(4 * np.pi) * numerator / denominator if abs(denominator) > 1e-10 else 0.0
        return beta

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

        scale_factor = (sphere_vol / vol_pre) ** (1 / 3)
        return theta_recon, scale_factor * r_pre

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
        # DoF = N_points - N_parameters
        n_points = len(r_original)
        dof = max(1, n_points - n_params)
        chi_sq_red = chi_sq / dof

        return {
            "rmse": rmse,
            "chi_squared": chi_sq,
            "chi_squared_reduced": chi_sq_red,
            "l_infinity": l_inf
        }
