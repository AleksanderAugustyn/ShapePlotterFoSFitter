"""
fos_parameterization.py
Contains the Data Class for parameters and the Calculator for the FoS shape geometry.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int = 92
    neutrons: int = 144
    c_elongation: float = 1.0  # elongation
    a3: float = 0.0  # reflection asymmetry
    a4: float = 0.0  # neck parameter
    a5: float = 0.0  # higher order parameter
    a6: float = 0.0  # higher order parameter
    r0_constant: float = 1.16  # Radius constant in fm
    max_beta: int = 12  # Maximum number of beta parameters used for fitting

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons

    @property
    def q2(self) -> float:
        """Entangled parameter calculated from c and a4: q2 = c - 1.0 - 1.5 * a4"""
        return self.c_elongation - 1.0 - 1.5 * self.a4

    @property
    def radius0(self) -> float:
        """Radius of a spherical nucleus with the same number of nucleons."""
        return self.r0_constant * (self.nucleons ** (1 / 3))

    @property
    def z0(self) -> float:
        """Half-length of nucleus."""
        return self.c_elongation * self.radius0

    @property
    def a2(self) -> float:
        """Volume conservation constraint: a2 = a4/3 - a6/5 + ..."""
        return self.a4 / 3.0 - self.a6 / 5.0

    @property
    def sphere_surface_area(self) -> float:
        """Surface area of a sphere with the same nucleon number."""
        return 4 * np.pi * self.radius0 ** 2

    @property
    def sphere_volume(self) -> float:
        """Volume of a sphere with the same nucleon number."""
        return (4 / 3) * np.pi * self.radius0 ** 3

    @property
    def z_sh(self) -> float:
        """Shift to place the center of mass at origin."""
        return 3.0 / (2.0 * np.pi) * self.z0 * (self.a3 - self.a5 / 2.0)


class FoSShapeCalculator:
    """Class for calculating Fourier-over-Spheroid shapes."""

    def __init__(self, params: FoSParameters):
        self.params = params

    def f_function(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the shape function f(u).
        """
        # Base spherical shape
        f: np.ndarray = 1.0 - u ** 2.0

        # Sum Fourier terms
        sum_terms: np.ndarray = np.zeros_like(u)

        # Add Fourier terms
        # k=1: a2 * cos((2 - 1) / 2 * π * u) + a3 sin(1 * π * u)
        sum_terms += self.params.a2 * np.cos((2.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a3 * np.sin(1.0 * np.pi * u)

        # k=2: a4 * cos((4 - 1) / 2 * π * u) + a5 sin(2πu)
        sum_terms += self.params.a4 * np.cos((4.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a5 * np.sin(2.0 * np.pi * u)

        # k=3: a6 * cos((6 - 1) / 2 * π * u)
        sum_terms += self.params.a6 * np.cos((6.0 - 1.0) / 2.0 * np.pi * u)

        return f - sum_terms

    def calculate_shape(self, n_points: int = 720):
        """
        Calculate the shape coordinates in (z, ρ) space.
        Returns: z: axial coordinates, rho: radial coordinates
        """
        # Calculate shape without a shift first
        z_min: float = -self.params.z0 + self.params.z_sh
        z_max: float = self.params.z0 + self.params.z_sh
        z = np.linspace(z_min, z_max, n_points)

        # Calculate normalized u with shift in z
        u = (z - self.params.z_sh) / self.params.z0

        # Ensure u is in [-1, 1]
        u = np.clip(u, -1.0, 1.0)

        # Calculate f(u)
        f_vals = self.f_function(u)

        # Calculate ρ² = R₀² / c * f(u)
        rho_squared = self.params.radius0 ** 2 / self.params.c_elongation * f_vals

        # Handle negative values (set to 0)
        rho_squared = np.maximum(rho_squared, 0)

        # Calculate ρ
        rho = np.sqrt(rho_squared)

        return z, rho

    @staticmethod
    def calculate_surface_area_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray):
        if len(z) < 2 or len(rho) < 2:
            return 0.0
        d_rho_dz = np.gradient(rho, z)
        integrand = 2 * np.pi * rho * np.sqrt(1 + d_rho_dz ** 2)
        surface_area: float = float(simpson(integrand, x=z))
        return surface_area

    @staticmethod
    def calculate_volume_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray) -> float:
        volume: float = float(simpson(np.pi * rho ** 2, x=z))
        return volume

    @staticmethod
    def calculate_center_of_mass_in_cylindrical_coordinates(z: np.ndarray, rho: np.ndarray) -> float:
        if len(z) < 2:
            return 0.0
        z_mid = (z[1:] + z[:-1]) / 2
        rho_mid = (rho[1:] + rho[:-1]) / 2
        numerator: float = float(simpson(z_mid * rho_mid ** 2, x=z[:-1]))
        denominator: float = float(simpson(rho_mid ** 2, x=z[:-1]))
        if denominator == 0:
            return 0.0
        return numerator / denominator
