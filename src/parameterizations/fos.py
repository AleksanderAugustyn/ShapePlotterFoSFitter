"""Module for calculating Fourier-over-Spheroid (FoS) nuclear shapes.

OPTIMIZED VERSION:
1. Uses NumPy broadcasting for shape generation (Vectorized).
2. Calculates derivatives analytically (Stable Surface Area).
3. Uses vectorized Simpson's rule (Fast Integration).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int = 92
    neutrons: int = 144
    c_elongation: float = 1.0
    coefficients: Dict[int, float] = field(default_factory=dict)
    r0_constant: float = 1.16

    # Cache for vectorized coefficients (Internal optimization state)
    _coeff_array_even: Optional[FloatArray] = field(init=False, repr=False, default=None)
    _coeff_array_odd: Optional[FloatArray] = field(init=False, repr=False, default=None)
    _k_indices: Optional[FloatArray] = field(init=False, repr=False, default=None)
    _last_coeff_hash: int = field(init=False, repr=False, default=0)

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons

    @property
    def q2(self) -> float:
        """Calculate the deformation parameter q2."""
        return self.c_elongation - 1.0 - 1.5 * self.get_coefficient(4)

    @property
    def radius0(self) -> float:
        """Calculate the radius of the spherical nucleus."""
        return float(self.r0_constant * (self.nucleons ** (1 / 3)))

    @property
    def z0(self) -> float:
        """Calculate the semi-axis along the symmetry axis."""
        return self.c_elongation * self.radius0

    @property
    def sphere_volume(self) -> float:
        """Volume of the spherical nucleus."""
        return (4.0 / 3.0) * np.pi * self.radius0 ** 3

    @property
    def sphere_surface_area(self) -> float:
        """Surface area of the spherical nucleus."""
        return 4.0 * np.pi * self.radius0 ** 2

    def get_coefficient(self, index: int) -> float:
        """Get coefficient a_index, returning 0.0 if not set."""
        return self.coefficients.get(index, 0.0)

    def set_coefficient(self, index: int, value: float) -> None:
        """Set coefficient a_index."""
        self.coefficients[index] = value
        # Invalidate cache so it recalculates on the next draw
        self._last_coeff_hash = 0

    @property
    def a2(self) -> float:
        """Volume constraint: a2 = sum_{n=2}^{inf} (-1)^n * a_{2n} / (2n-1)."""
        # This is kept for external access/verification.
        # The vectorized calculator computes this internally as well.
        result = 0.0
        for n in range(2, 100):  # Practical upper limit
            a_2n = self.get_coefficient(2 * n)
            if a_2n != 0.0:
                result += ((-1) ** n) * a_2n / (2 * n - 1)
        return result

    def _update_vectorized_coefficients(self) -> None:
        """Pre-calculates coefficient arrays for vectorized operations."""
        current_hash = hash(frozenset(self.coefficients.items()))
        if self._last_coeff_hash == current_hash and self._k_indices is not None:
            return

        # Determine the max index needed
        max_idx = 4
        if self.coefficients:
            max_idx = max(max_idx, max(self.coefficients.keys()))

        # We need k up to where 2k or 2k+1 covers max_idx
        k_max = (max_idx + 1) // 2 + 1

        k_vals = np.arange(1, k_max + 1, dtype=np.float64)
        a_even = np.zeros(k_max, dtype=np.float64)
        a_odd = np.zeros(k_max, dtype=np.float64)

        # Calculate a2 specifically for the vector array
        a2_val = 0.0
        for idx, val in self.coefficients.items():
            if idx > 2 and idx % 2 == 0:
                n = idx // 2
                a2_val += ((-1) ** n) * val / (2 * n - 1)

        # Populate arrays
        for i, k in enumerate(k_vals):
            # Even: a_{2k}
            if k == 1:
                a_even[i] = a2_val
            else:
                a_even[i] = self.get_coefficient(int(2 * k))

            # Odd: a_{2k+1}
            a_odd[i] = self.get_coefficient(int(2 * k + 1))

        self._k_indices = k_vals
        self._coeff_array_even = a_even
        self._coeff_array_odd = a_odd
        self._last_coeff_hash = current_hash

    @property
    def vectorized_data(self) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """Returns (k_values, a_even, a_odd) for vectorization."""
        self._update_vectorized_coefficients()
        assert self._k_indices is not None
        assert self._coeff_array_even is not None
        assert self._coeff_array_odd is not None
        return self._k_indices, self._coeff_array_even, self._coeff_array_odd

    @property
    def z_sh(self) -> float:
        """Shift to place the center of mass at origin."""
        result = 0.0
        for n in range(1, 50):
            a_odd = self.get_coefficient(2 * n + 1)
            if a_odd != 0.0:
                result += ((-1) ** (n + 1)) * a_odd / n
        return (3.0 / (2.0 * np.pi)) * self.z0 * result


class FoSShapeCalculator:
    """Optimized calculator for Fourier-over-Spheroid shapes."""

    def __init__(self, params: FoSParameters):
        self.params = params

    def _calculate_f_and_deriv(self, u: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Computes f(u) and f'(u) simultaneously using matrix broadcasting.

        f(u) = 1 - u^2 - sum(...)
        f'(u) = -2u - sum(...)
        """
        k, a_even, a_odd = self.params.vectorized_data

        # Broadcasting setup
        u_2d = u[np.newaxis, :]  # (1, N)
        k_2d = k[:, np.newaxis]  # (K, 1)
        a_even_2d = a_even[:, np.newaxis]
        a_odd_2d = a_odd[:, np.newaxis]

        pi = np.pi

        # Pre-calculate arguments
        arg_even = (2 * k_2d - 1) / 2.0 * pi * u_2d
        arg_odd = k_2d * pi * u_2d

        # Trig terms
        cos_even = np.cos(arg_even)
        sin_even = np.sin(arg_even)
        cos_odd = np.cos(arg_odd)
        sin_odd = np.sin(arg_odd)

        # --- f(u) Calculation ---
        sum_terms = np.sum(a_even_2d * cos_even + a_odd_2d * sin_odd, axis=0)
        f = 1.0 - u ** 2 - sum_terms

        # --- f'(u) Calculation (Analytical) ---
        c_even = (2 * k_2d - 1) / 2.0 * pi
        c_odd = k_2d * pi

        sum_deriv = np.sum(
            a_even_2d * (-sin_even * c_even) +
            a_odd_2d * (cos_odd * c_odd),
            axis=0
        )

        f_prime = -2.0 * u - sum_deriv

        return f, f_prime

    def calculate_shape(self, n_points: int = 720) -> Tuple[FloatArray, FloatArray, bool]:
        """Calculates z and rho using vectorized operations."""
        z0 = self.params.z0
        z_sh = self.params.z_sh

        z = np.linspace(-z0 + z_sh, z0 + z_sh, n_points)
        u = np.linspace(-1.0, 1.0, n_points)

        f_vals, _ = self._calculate_f_and_deriv(u)

        rho_squared = (self.params.radius0 ** 2 / self.params.c_elongation) * f_vals

        # Validity check (interior points > 0)
        is_valid = bool(np.all(rho_squared[1:-1] > 0))

        rho = np.sqrt(np.maximum(rho_squared, 0.0))

        return z, rho, is_valid

    @staticmethod
    def _simpson_fast(y: FloatArray, x: FloatArray) -> float:
        """Fast implementation of Simpson's rule for strictly odd N (even intervals)."""
        n = len(y)
        if n % 2 == 0:
            return float(np.trapezoid(y, x))

        h = (x[-1] - x[0]) / (n - 1)
        s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
        return float(s * h / 3.0)

    def calculate_metrics_fast(self, n_points: int = 7200) -> Tuple[float, float, float]:
        """
        Calculates Volume, Surface Area, and CoM (z) simultaneously.

        CORRECTION:
        Clamps invalid (negative rho^2) regions to zero to return physical metrics.
        This ensures we don't integrate 'negative volume' for invalid shapes.
        """
        z0 = self.params.z0
        z_sh = self.params.z_sh
        r0 = self.params.radius0
        c = self.params.c_elongation

        z = np.linspace(-z0 + z_sh, z0 + z_sh, n_points)
        u = np.linspace(-1.0, 1.0, n_points)

        f, f_prime = self._calculate_f_and_deriv(u)

        # --- CLAMPING FOR PHYSICAL VALIDITY ---
        # If f < 0, rho^2 < 0. Physically this is a vacuum (rho=0).
        # We must mask both f and f_prime to 0 in these regions to avoid integrating negative volume or ghost surface area.
        valid_mask = f > 0

        # Apply mask
        f_clamped = np.where(valid_mask, f, 0.0)
        f_prime_clamped = np.where(valid_mask, f_prime, 0.0)

        # Factor A where rho = A * sqrt(f)
        a2 = (r0 ** 2) / c

        # Volume Integrand
        # rho^2 = A^2 * f_clamped
        rho2 = a2 * f_clamped
        integrand_vol = np.pi * rho2
        volume = self._simpson_fast(integrand_vol, z)

        # CoM Integrand
        integrand_com = z * integrand_vol
        com_z = 0.0
        if volume > 1e-12:
            com_z = self._simpson_fast(integrand_com, z) / volume

        # Surface Integrand
        # term = rho * rho' = (A^2 / 2*z0) * f'
        # With clamped arrays, this is 0 in invalid regions.
        term_rho_rho_prime = (a2 / (2 * z0)) * f_prime_clamped

        # S = 2*pi * integral(sqrt(rho^2 + (rho*rho')^2))
        # Note: rho2 is A^2 * f (already squared rho)
        # We square the term_rho_rho_prime.
        integrand_surf = 2 * np.pi * np.sqrt(rho2 + term_rho_rho_prime ** 2)
        surface = self._simpson_fast(integrand_surf, z)

        return volume, surface, com_z

    # Static wrappers for backward compatibility with existing Plotter code if needed
    @staticmethod
    def calculate_volume(z: np.ndarray, rho: np.ndarray) -> float:
        return FoSShapeCalculator._simpson_fast(np.pi * rho ** 2, z)

    @staticmethod
    def calculate_surface_area(z: np.ndarray, rho: np.ndarray) -> float:
        d_rho_dz = np.gradient(rho, z)
        integrand = 2 * np.pi * rho * np.sqrt(1 + d_rho_dz ** 2)
        return FoSShapeCalculator._simpson_fast(integrand, z)

    @staticmethod
    def calculate_center_of_mass(z: np.ndarray, rho: np.ndarray) -> float:
        vol = FoSShapeCalculator.calculate_volume(z, rho)
        if vol < 1e-10: return 0.0
        return FoSShapeCalculator._simpson_fast(z * np.pi * rho ** 2, z) / vol
