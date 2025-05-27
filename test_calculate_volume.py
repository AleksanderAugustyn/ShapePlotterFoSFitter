"""
Unit tests for the calculate_volume method in FoSShapeCalculator.
"""

import unittest

import numpy as np

from main import FoSParameters, FoSShapeCalculator


class TestCalculateVolume(unittest.TestCase):
    """Test cases for the calculate_volume method."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_params = FoSParameters(protons=92, neutrons=144)
        self.calculator = FoSShapeCalculator(self.default_params)

    def test_sphere_volume(self):
        """Test volume calculation for a perfect sphere (c=1, all a_i=0)."""
        sphere_params = FoSParameters(protons=92, neutrons=144, c=1.0)
        calculator = FoSShapeCalculator(sphere_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Allow 1% tolerance for numerical integration
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.01)

    def test_prolate_volume_conservation(self):
        """Test that prolate shape conserves volume."""
        prolate_params = FoSParameters(protons=92, neutrons=144, c=1.5)
        calculator = FoSShapeCalculator(prolate_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Volume should be conserved within 2% tolerance
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

    def test_oblate_volume_conservation(self):
        """Test that oblate shape conserves volume."""
        oblate_params = FoSParameters(protons=92, neutrons=144, c=0.7)
        calculator = FoSShapeCalculator(oblate_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Volume should be conserved within 2% tolerance
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

    def test_asymmetric_shape_volume(self):
        """Test volume calculation for asymmetric shapes."""
        asym_params = FoSParameters(protons=92, neutrons=144, c=1.2, a3=0.2)
        calculator = FoSShapeCalculator(asym_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Volume should be conserved within 3% tolerance for asymmetric shapes
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.03)

    def test_neck_parameter_volume(self):
        """Test volume calculation with neck parameter."""
        neck_params = FoSParameters(protons=92, neutrons=144, c=1.5, a4=0.3)
        calculator = FoSShapeCalculator(neck_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Volume should be conserved within 3% tolerance
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.03)

    def test_multiple_parameters_volume(self):
        """Test volume calculation with multiple non-zero parameters."""
        complex_params = FoSParameters(
            protons=92, neutrons=144, c=1.3, a3=0.1, a4=0.2, a5=0.05, a6=0.02
        )
        calculator = FoSShapeCalculator(complex_params)
        z, rho = calculator.calculate_shape()

        calculated_volume = calculator.calculate_volume(z, rho)
        expected_volume = calculator.calculate_sphere_volume()

        # Volume should be conserved within 5% tolerance for complex shapes
        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.05)

    def test_different_nucleus_sizes(self):
        """Test volume calculation for different nucleus sizes."""
        # Light nucleus
        light_params = FoSParameters(protons=26, neutrons=30, c=1.2)
        light_calculator = FoSShapeCalculator(light_params)
        z, rho = light_calculator.calculate_shape()

        calculated_volume = light_calculator.calculate_volume(z, rho)
        expected_volume = light_calculator.calculate_sphere_volume()

        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

        # Heavy nucleus
        heavy_params = FoSParameters(protons=110, neutrons=170, c=1.1)
        heavy_calculator = FoSShapeCalculator(heavy_params)
        z, rho = heavy_calculator.calculate_shape()

        calculated_volume = heavy_calculator.calculate_volume(z, rho)
        expected_volume = heavy_calculator.calculate_sphere_volume()

        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

    def test_empty_arrays(self):
        """Test volume calculation with empty arrays."""
        empty_z = np.array([])
        empty_rho = np.array([])

        volume = FoSShapeCalculator.calculate_volume(empty_z, empty_rho)
        self.assertEqual(volume, 0.0)

    def test_single_point(self):
        """Test volume calculation with single point."""
        z = np.array([0.0])
        rho = np.array([5.0])

        volume = FoSShapeCalculator.calculate_volume(z, rho)
        self.assertEqual(volume, 0.0)

    def test_zero_radius(self):
        """Test volume calculation with zero radius."""
        z = np.linspace(-5, 5, 100)
        rho = np.zeros_like(z)

        volume = FoSShapeCalculator.calculate_volume(z, rho)
        self.assertEqual(volume, 0.0)

    def test_constant_radius(self):
        """Test volume calculation with constant radius."""
        z = np.linspace(-5, 5, 1000)
        rho = np.full_like(z, 3.0)

        volume = FoSShapeCalculator.calculate_volume(z, rho)
        expected_volume = np.pi * 3.0 ** 2 * 10  # π * r² * length

        self.assertAlmostEqual(volume, expected_volume, delta=expected_volume * 0.01)

    def test_volume_positive(self):
        """Test that calculated volume is always positive."""
        for c in [0.5, 1.0, 1.5, 2.0]:
            for a4 in [0.0, 0.2, 0.4]:
                params = FoSParameters(protons=92, neutrons=144, c=c, a4=a4)
                calculator = FoSShapeCalculator(params)
                z, rho = calculator.calculate_shape()

                volume = calculator.calculate_volume(z, rho)
                self.assertGreater(volume, 0, f"Volume should be positive for c={c}, a4={a4}")


if __name__ == '__main__':
    unittest.main()
