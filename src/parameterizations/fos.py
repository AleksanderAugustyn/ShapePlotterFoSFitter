"""Module for calculating Fourier-over-Spheroid (FoS) nuclear shapes."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson

FloatArray = NDArray[np.float64]


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters.

    Coefficients are stored in a dictionary where keys are the index (3, 4, 5, 6, ...).
    a2 is computed from the volume constraint and should not be set directly.
    """
    protons: int = 92
    neutrons: int = 144
    c_elongation: float = 1.0
    coefficients: Dict[int, float] = field(default_factory=dict)
    r0_constant: float = 1.16

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

    def get_coefficient(self, index: int) -> float:
        """Get coefficient a_index, returning 0.0 if not set."""
        return self.coefficients.get(index, 0.0)

    def set_coefficient(self, index: int, value: float) -> None:
        """Set coefficient a_index."""
        self.coefficients[index] = value

    @property
    def a2(self) -> float:
        """Volume constraint: a2 = sum_{n=2}^{inf} (-1)^n * a_{2n} / (2n-1)"""
        result = 0.0
        for n in range(2, 100):  # Practical upper limit
            a_2n = self.get_coefficient(2 * n)
            if a_2n != 0.0:
                result += ((-1) ** n) * a_2n / (2 * n - 1)
        return result

    @property
    def sphere_volume(self) -> float:
        """Volume of the spherical nucleus."""
        return (4.0 / 3.0) * np.pi * self.radius0 ** 3

    @property
    def z_sh(self) -> float:
        """Shift to place the center of mass at origin.

        z_sh = (3 / 2 pi) * z0 * sum_{n=1}^{inf} (-1)^{n+1} * a_{2n+1} / n
        """
        result = 0.0
        for n in range(1, 100):  # Practical upper limit
            a_odd = self.get_coefficient(2 * n + 1)
            if a_odd != 0.0:
                result += ((-1) ** (n + 1)) * a_odd / n
        return (3.0 / (2.0 * np.pi)) * self.z0 * result


class FoSShapeCalculator:
    """Class for calculating Fourier-over-Spheroid shapes."""

    def __init__(self, params: FoSParameters):
        self.params = params

    def f_function(self, u: FloatArray) -> FloatArray:
        """Compute f(u) = 1 - u^2 - sum_{k=1}^{n} [a_{2k}*cos((2k-1)/2 * pi * u) + a_{2k+1}*sin(k * pi * u)]"""
        f: FloatArray = 1.0 - u ** 2.0
        sum_terms: FloatArray = np.zeros_like(u, dtype=float)

        # Find max index from coefficients (include a2, which is derived)
        max_even: int = 2  # Always include a2
        max_odd: int = 1
        for idx in self.params.coefficients.keys():
            if idx % 2 == 0:
                max_even = max(max_even, idx)
            else:
                max_odd = max(max_odd, idx)

        # Sum over k: a_{2k} terms (even) and a_{2k+1} terms (odd)
        k_max = max(max_even // 2, (max_odd - 1) // 2) + 1
        for k in range(1, k_max + 1):
            # Even term: a_{2k} * cos((2k-1)/2 * pi * u)
            a_even = self.params.a2 if k == 1 else self.params.get_coefficient(2 * k)
            if a_even != 0.0:
                sum_terms += a_even * np.cos((2 * k - 1) / 2.0 * np.pi * u)

            # Odd term: a_{2k+1} * sin(k * pi * u)
            a_odd = self.params.get_coefficient(2 * k + 1)
            if a_odd != 0.0:
                sum_terms += a_odd * np.sin(k * np.pi * u)

        result: FloatArray = np.asarray(f - sum_terms, dtype=float)

        return result

    def calculate_shape(self, n_points: int = 720) -> tuple[FloatArray, FloatArray]:
        """Returns: z: axial coordinates, rho: radial coordinates"""
        z_min = -self.params.z0 + self.params.z_sh
        z_max = self.params.z0 + self.params.z_sh
        z: FloatArray = np.linspace(z_min, z_max, n_points, dtype=float)

        u = (z - self.params.z_sh) / self.params.z0
        u = np.clip(u, -1.0, 1.0)

        f_vals = self.f_function(u)
        rho_squared = self.params.radius0 ** 2 / self.params.c_elongation * f_vals
        rho: FloatArray = np.asarray(np.sqrt(np.maximum(rho_squared, 0.0)), dtype=float)

        return z, rho

    @staticmethod
    def calculate_volume(z: np.ndarray, rho: np.ndarray) -> float:
        """Calculate volume in cylindrical coordinates."""
        return float(simpson(np.pi * rho ** 2, x=z))

    @staticmethod
    def calculate_surface_area(z: np.ndarray, rho: np.ndarray) -> float:
        """Calculate surface area in cylindrical coordinates."""
        if len(z) < 2:
            return 0.0

        d_rho_dz = np.gradient(rho, z)
        integrand = 2 * np.pi * rho * np.sqrt(1 + d_rho_dz ** 2)
        return float(simpson(integrand, x=z))
