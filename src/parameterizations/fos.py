from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson


@dataclass
class FoSParameters:
    """Class to store Fourier-over-Spheroid shape parameters."""
    protons: int = 92
    neutrons: int = 144
    c_elongation: float = 1.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    a6: float = 0.0
    r0_constant: float = 1.16

    @property
    def nucleons(self) -> int:
        return self.protons + self.neutrons

    @property
    def q2(self) -> float:
        return self.c_elongation - 1.0 - 1.5 * self.a4

    @property
    def radius0(self) -> float:
        return self.r0_constant * (self.nucleons ** (1 / 3))

    @property
    def z0(self) -> float:
        return self.c_elongation * self.radius0

    @property
    def a2(self) -> float:
        return self.a4 / 3.0 - self.a6 / 5.0

    @property
    def sphere_volume(self) -> float:
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
        f = 1.0 - u ** 2.0
        sum_terms = np.zeros_like(u)

        # Fourier terms
        sum_terms += self.params.a2 * np.cos((2.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a3 * np.sin(1.0 * np.pi * u)
        sum_terms += self.params.a4 * np.cos((4.0 - 1.0) / 2.0 * np.pi * u)
        sum_terms += self.params.a5 * np.sin(2.0 * np.pi * u)
        sum_terms += self.params.a6 * np.cos((6.0 - 1.0) / 2.0 * np.pi * u)

        return f - sum_terms

    def calculate_shape(self, n_points: int = 720):
        """Returns: z: axial coordinates, rho: radial coordinates"""
        z_min = -self.params.z0 + self.params.z_sh
        z_max = self.params.z0 + self.params.z_sh
        z = np.linspace(z_min, z_max, n_points)

        u = (z - self.params.z_sh) / self.params.z0
        u = np.clip(u, -1.0, 1.0)

        f_vals = self.f_function(u)
        rho_squared = self.params.radius0 ** 2 / self.params.c_elongation * f_vals
        rho = np.sqrt(np.maximum(rho_squared, 0))

        return z, rho

    @staticmethod
    def calculate_volume(z: np.ndarray, rho: np.ndarray) -> float:
        return float(simpson(np.pi * rho ** 2, x=z))

    @staticmethod
    def calculate_surface_area(z: np.ndarray, rho: np.ndarray) -> float:
        if len(z) < 2: return 0.0
        d_rho_dz = np.gradient(rho, z)
        integrand = 2 * np.pi * rho * np.sqrt(1 + d_rho_dz ** 2)
        return float(simpson(integrand, x=z))
