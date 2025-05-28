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

    def test_q2_parameter_relationship(self):
        """Test that q2, c, and a4 maintain their relationship."""
        # Test case 1: q2 = 0.5, a4 = 0.2
        q2_val = 0.5  # Renamed to avoid conflict with instance variable
        a4_val = 0.2  # Renamed to avoid conflict with instance variable
        expected_c = q2_val + 1.0 + 1.5 * a4_val

        params = FoSParameters(protons=92, neutrons=144, c=expected_c, q2=q2_val, a4=a4_val)
        self.assertAlmostEqual(params.c, expected_c, places=6)

        # Test case 2: q2 = -0.3, a4 = 0.4
        q2_val = -0.3 # Renamed
        a4_val = 0.4  # Renamed
        expected_c = q2_val + 1.0 + 1.5 * a4_val

        params = FoSParameters(protons=92, neutrons=144, c=expected_c, q2=q2_val, a4=a4_val)
        self.assertAlmostEqual(params.c, expected_c, places=6)

    def test_q2_volume_conservation(self):
        """Test volume conservation with different q2 values."""
        for q2_val in [-0.5, 0.0, 0.5, 1.0]: # Renamed
            for a4_val in [0.0, 0.2, 0.4]: # Renamed
                c_val = q2_val + 1.0 + 1.5 * a4_val # Renamed
                params = FoSParameters(protons=92, neutrons=144, c=c_val, q2=q2_val, a4=a4_val)
                calculator = FoSShapeCalculator(params)
                z, rho = calculator.calculate_shape()

                calculated_volume = calculator.calculate_volume(z, rho)
                expected_volume = calculator.calculate_sphere_volume()

                # Volume should be conserved within 5% tolerance
                self.assertAlmostEqual(
                    calculated_volume, expected_volume,
                    delta=expected_volume * 0.05,
                    msg=f"Volume not conserved for q2={q2_val}, a4={a4_val}, c={c_val}"
                )

    def test_multiple_parameters_volume(self):
        """Test volume calculation with multiple non-zero parameters."""
        complex_params = FoSParameters(
            protons=92, neutrons=144, c=1.3, q2=0.1, a3=0.1, a4=0.2, a5=0.05, a6=0.02
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
        light_params = FoSParameters(protons=26, neutrons=30, c=1.2, q2=0.2)
        light_calculator = FoSShapeCalculator(light_params)
        z, rho = light_calculator.calculate_shape()

        calculated_volume = light_calculator.calculate_volume(z, rho)
        expected_volume = light_calculator.calculate_sphere_volume()

        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

        # Heavy nucleus
        heavy_params = FoSParameters(protons=110, neutrons=170, c=1.1, q2=0.1)
        heavy_calculator = FoSShapeCalculator(heavy_params)
        z, rho = heavy_calculator.calculate_shape()

        calculated_volume = heavy_calculator.calculate_volume(z, rho)
        expected_volume = heavy_calculator.calculate_sphere_volume()

        self.assertAlmostEqual(calculated_volume, expected_volume, delta=expected_volume * 0.02)

    def test_extreme_q2_values(self):
        """Test shape calculations with extreme q2 values."""
        # Test very negative q2
        params_neg = FoSParameters(protons=92, neutrons=144, c=0.5, q2=-0.5, a4=0.0)
        calculator_neg = FoSShapeCalculator(params_neg)
        z_neg, rho_neg = calculator_neg.calculate_shape()

        # Verify c = q2 + 1.0 + 1.5 * a4
        self.assertAlmostEqual(params_neg.c, -0.5 + 1.0 + 0.0, places=6)

        # Test very positive q2
        params_pos = FoSParameters(protons=92, neutrons=144, c=2.5, q2=1.5, a4=0.0)
        calculator_pos = FoSShapeCalculator(params_pos)
        z_pos, rho_pos = calculator_pos.calculate_shape()

        # Verify c = q2 + 1.0 + 1.5 * a4
        self.assertAlmostEqual(params_pos.c, 1.5 + 1.0 + 0.0, places=6)

        # Both should produce valid shapes
        self.assertTrue(np.all(rho_neg >= 0))
        self.assertTrue(np.all(rho_pos >= 0))

    def test_empty_arrays(self):
        """Test volume calculation with empty arrays."""
        empty_z = np.array([])
        empty_rho = np.array([])

        # Accessing calculate_volume statically
        volume = FoSShapeCalculator.calculate_volume(empty_z, empty_rho)
        self.assertEqual(volume, 0.0)

    def test_single_point(self):
        """Test volume calculation with single point."""
        z = np.array([0.0])
        rho = np.array([5.0])

        # Accessing calculate_volume statically
        volume = FoSShapeCalculator.calculate_volume(z, rho)
        self.assertEqual(volume, 0.0)

    def test_zero_radius(self):
        """Test volume calculation with zero radius."""
        z = np.linspace(-5, 5, 100)
        rho = np.zeros_like(z)
        
        # Accessing calculate_volume statically
        volume = FoSShapeCalculator.calculate_volume(z, rho)
        self.assertEqual(volume, 0.0)

    def test_constant_radius(self):
        """Test volume calculation with constant radius."""
        z = np.linspace(-5, 5, 1000)
        rho = np.full_like(z, 3.0)

        # Accessing calculate_volume statically
        volume = FoSShapeCalculator.calculate_volume(z, rho)
        expected_volume = np.pi * 3.0 ** 2 * 10  # π * r² * length

        self.assertAlmostEqual(volume, expected_volume, delta=expected_volume * 0.01)

    def test_volume_positive(self):
        """Test that calculated volume is always positive."""
        for c_val in [0.5, 1.0, 1.5, 2.0]: # Renamed
            for a4_val in [0.0, 0.2, 0.4]: # Renamed
                q2_val = c_val - 1.0 - 1.5 * a4_val  # Calculate q2 from c and a4
                params = FoSParameters(protons=92, neutrons=144, c=c_val, q2=q2_val, a4=a4_val)
                calculator = FoSShapeCalculator(params)
                z, rho = calculator.calculate_shape()

                volume = calculator.calculate_volume(z, rho)
                self.assertGreater(volume, 0, f"Volume should be positive for c={c_val}, q2={q2_val}, a4={a4_val}")


class TestFoSParameters(unittest.TestCase):
    def test_nucleons(self):
        params = FoSParameters(protons=50, neutrons=70)
        self.assertEqual(params.nucleons, 120)

    def test_R0(self):
        # Test with r0 = 1.2 for easier calculation if needed, though default is 1.16
        params_default_r0 = FoSParameters(protons=50, neutrons=70) # A = 120
        expected_R0_default = 1.16 * (120 ** (1/3))
        self.assertAlmostEqual(params_default_r0.radius0, expected_R0_default)

        params_custom_r0 = FoSParameters(protons=50, neutrons=70, r0_constant=1.2)  # A = 120
        expected_R0_custom = 1.2 * (120 ** (1/3))
        self.assertAlmostEqual(params_custom_r0.radius0, expected_R0_custom)

    def test_z0(self):
        params = FoSParameters(protons=50, neutrons=70, c=1.5, r0_constant=1.2)  # A=120
        expected_R0 = 1.2 * (120 ** (1/3))
        expected_z0 = 1.5 * expected_R0
        self.assertAlmostEqual(params.z0, expected_z0)

    def test_zsh(self):
        # z_sh = -3/(4π) z_0 (a_3 - a_5/2)
        # Case 1: a3 and a5 are zero
        params1 = FoSParameters(protons=92, neutrons=144, c=1.2, a3=0.0, a5=0.0)
        self.assertAlmostEqual(params1.zsh, 0.0)

        # Case 2: Non-zero a3, zero a5
        params2 = FoSParameters(protons=92, neutrons=144, c=1.2, a3=0.2, a5=0.0)
        expected_zsh2 = -3 / (4 * np.pi) * params2.z0 * 0.2
        self.assertAlmostEqual(params2.zsh, expected_zsh2)

        # Case 3: Zero a3, non-zero a5
        params3 = FoSParameters(protons=92, neutrons=144, c=1.2, a3=0.0, a5=0.1)
        expected_zsh3 = -3 / (4 * np.pi) * params3.z0 * (0.0 - 0.1 / 2)
        self.assertAlmostEqual(params3.zsh, expected_zsh3)
        
        # Case 4: Non-zero a3 and a5
        params4 = FoSParameters(protons=92, neutrons=144, c=1.2, a3=0.2, a5=0.1)
        expected_zsh4 = -3 / (4 * np.pi) * params4.z0 * (0.2 - 0.1 / 2)
        self.assertAlmostEqual(params4.zsh, expected_zsh4)

    def test_a2(self):
        # a2 = a4/3 - a6/5
        # Case 1: a4 and a6 are zero
        params1 = FoSParameters(protons=92, neutrons=144, a4=0.0, a6=0.0)
        self.assertAlmostEqual(params1.a2, 0.0)

        # Case 2: Non-zero a4, zero a6
        params2 = FoSParameters(protons=92, neutrons=144, a4=0.3, a6=0.0)
        expected_a2_2 = 0.3 / 3.0
        self.assertAlmostEqual(params2.a2, expected_a2_2)

        # Case 3: Zero a4, non-zero a6
        params3 = FoSParameters(protons=92, neutrons=144, a4=0.0, a6=0.25)
        expected_a2_3 = -0.25 / 5.0
        self.assertAlmostEqual(params3.a2, expected_a2_3)

        # Case 4: Non-zero a4 and a6
        params4 = FoSParameters(protons=92, neutrons=144, a4=0.3, a6=0.25)
        expected_a2_4 = (0.3 / 3.0) - (0.25 / 5.0)
        self.assertAlmostEqual(params4.a2, expected_a2_4)


class TestFFunction(unittest.TestCase):
    def setUp(self):
        self.default_params = FoSParameters(protons=92, neutrons=144) # A=236
        self.calculator = FoSShapeCalculator(self.default_params)
        self.u_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_f_function_basic_sphere(self):
        # For a sphere, a2=a3=a4=a5=a6=0. So f(u) = 1 - u^2
        # params are by default spherical (all a_i = 0, so a2=0)
        params_sphere = FoSParameters(protons=92, neutrons=144, c=1.0) # Ensure all a_i are zero
        calculator_sphere = FoSShapeCalculator(params_sphere)
        
        u = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected_f = 1 - u**2
        calculated_f = calculator_sphere.f_function(u)
        np.testing.assert_array_almost_equal(calculated_f, expected_f, decimal=6)

    def test_f_function_with_a2(self):
        # f(u) = 1 - u² - a2*cos(π/2 u)
        # For this test, let a4=0.3 so a2 = 0.3/3 = 0.1. a3,a5,a6=0.
        params = FoSParameters(protons=92, neutrons=144, c=1.0, a4=0.3) # a2 becomes 0.1
        calculator = FoSShapeCalculator(params)
        
        u_val = 0.5
        # a2 = params.a4 / 3 - params.a6 / 5 = 0.3/3 - 0/5 = 0.1
        a2_val = 0.1 
        expected_f = 1 - u_val**2 - a2_val * np.cos(0.5 * np.pi * u_val)
        calculated_f = calculator.f_function(np.array([u_val]))
        self.assertAlmostEqual(calculated_f[0], expected_f, places=6)

    def test_f_function_with_a3(self):
        # f(u) = 1 - u² - a3*sin(πu) (assuming a2,a4,a5,a6=0)
        params = FoSParameters(protons=92, neutrons=144, c=1.0, a3=0.2) # a2,a4,a5,a6=0
        calculator = FoSShapeCalculator(params)

        u_val = 0.5
        # a2 = 0 since a4 and a6 are 0
        expected_f = 1 - u_val**2 - params.a3 * np.sin(np.pi * u_val)
        calculated_f = calculator.f_function(np.array([u_val]))
        self.assertAlmostEqual(calculated_f[0], expected_f, places=6)

    def test_f_function_with_all_params(self):
        params = FoSParameters(protons=92, neutrons=144, c=1.0, 
                               a3=0.1, a4=0.15, a5=0.05, a6=0.075)
        # This implies a2 = a4/3 - a6/5 = 0.15/3 - 0.075/5 = 0.05 - 0.015 = 0.035
        calculator = FoSShapeCalculator(params)
        
        u_val = 0.25
        a2_val = params.a2 # Should be 0.035
        
        expected_f = 1 - u_val**2
        expected_f -= a2_val * np.cos(0.5 * np.pi * u_val)
        expected_f -= params.a3 * np.sin(np.pi * u_val)
        expected_f -= params.a4 * np.cos(1.5 * np.pi * u_val)
        expected_f -= params.a5 * np.sin(2 * np.pi * u_val)
        expected_f -= params.a6 * np.cos(2.5 * np.pi * u_val)
        
        calculated_f = calculator.f_function(np.array([u_val]))
        self.assertAlmostEqual(calculated_f[0], expected_f, places=6)

    def test_f_function_endpoints(self):
        # f(u) should be 0 at u = ±1 for a sphere (all a_i=0)
        # f(u) = 1 - u² - Σ[a_{2k} cos((k-1/2)πu) + a_{2k+1} sin(kπu)]
        # For u=1:
        # cos(π/2) = 0, sin(π) = 0
        # cos(3π/2) = 0, sin(2π) = 0
        # cos(5π/2) = 0
        # So f(1) = 1 - 1^2 - 0 = 0, regardless of a_i values.
        # For u=-1:
        # cos(-π/2)=0, sin(-π)=0
        # cos(-3π/2)=0, sin(-2π)=0
        # cos(-5π/2)=0
        # So f(-1) = 1 - (-1)^2 - 0 = 0, regardless of a_i values.

        params_complex = FoSParameters(protons=92, neutrons=144, c=1.0, 
                                       a3=0.1, a4=0.15, a5=0.05, a6=0.075)
        calculator_complex = FoSShapeCalculator(params_complex)
        
        u_endpoints = np.array([-1.0, 1.0])
        calculated_f_endpoints = calculator_complex.f_function(u_endpoints)
        np.testing.assert_array_almost_equal(calculated_f_endpoints, np.array([0.0, 0.0]), decimal=6)


class TestCalculateShape(unittest.TestCase):
    def setUp(self):
        self.default_params = FoSParameters(protons=92, neutrons=144) # A=236
        self.calculator = FoSShapeCalculator(self.default_params)

    def test_rho_non_negative(self):
        # Test with various parameters to ensure rho is never negative
        param_sets = [
            FoSParameters(protons=92, neutrons=144, c=1.0), # Sphere
            FoSParameters(protons=92, neutrons=144, c=1.5, a3=0.2), # Prolate, asymmetric
            FoSParameters(protons=92, neutrons=144, c=0.7, a4=-0.2), # Oblate, diamond-like
            FoSParameters(protons=92, neutrons=144, c=2.0, a4=0.5), # Two-center like
            FoSParameters(protons=92, neutrons=144, c=1.2, q2=0.0, a3=0.1, a4=0.1, a5=-0.1, a6=0.05) # Complex
        ]
        for params in param_sets:
            calculator = FoSShapeCalculator(params)
            z, rho = calculator.calculate_shape()
            self.assertTrue(np.all(rho >= -1e-9), f"Rho values should be non-negative for params: {params}") # Allow for tiny numerical errors

    def test_sphere_shape(self):
        # For a sphere (c=1, all a_i=0), rho^2(z) = R0^2 * c * (1 - u^2) = R0^2 * (1 - (z/R0)^2)
        # So, rho(z) = sqrt(R0^2 - z^2)
        # z_sh = 0 for sphere
        params_sphere = FoSParameters(protons=92, neutrons=144, c=1.0) # Ensure all a_i are effectively zero
        calculator_sphere = FoSShapeCalculator(params_sphere)
        R0 = params_sphere.radius0
        z0 = params_sphere.z0 # Should be R0 for c=1
        self.assertAlmostEqual(z0, R0, msg="z0 should be R0 for a sphere with c=1")

        z_coords, rho_coords = calculator_sphere.calculate_shape(n_points=100)

        # Check endpoints: rho should be close to 0 at z = +/- R0
        # z_coords are between -z0 and z0. For c=1, z0 = R0.
        self.assertAlmostEqual(rho_coords[0], 0.0, delta=1e-6, msg="Rho at -R0 should be 0 for sphere")
        self.assertAlmostEqual(rho_coords[-1], 0.0, delta=1e-6, msg="Rho at R0 should be 0 for sphere")

        # Check midpoint: rho should be R0 at z = 0
        # Need to find the point closest to z=0, as z_coords might not have exact 0
        closest_to_zero_idx = np.argmin(np.abs(z_coords))
        self.assertAlmostEqual(rho_coords[closest_to_zero_idx], R0, delta=R0*0.01, msg="Rho at z=0 should be R0 for sphere")
        
        # Verify a few points against the analytical solution rho(z) = sqrt(R0^2 - z^2)
        for idx in [len(z_coords)//4, len(z_coords)//2, 3*len(z_coords)//4]:
            z_val = z_coords[idx]
            if np.abs(z_val) < R0: # Avoid domain error for sqrt if z_val is slightly outside due to discretization
                 expected_rho_val = np.sqrt(max(0, R0**2 - z_val**2)) # max(0,...) for robustness
                 self.assertAlmostEqual(rho_coords[idx], expected_rho_val, delta=R0*0.02, # 2% tolerance
                                        msg=f"Mismatch for sphere at z={z_val:.2f}")

    def test_prolate_shape_qualitative(self):
        # For a prolate shape (c > 1), z0 > R0.
        params_prolate = FoSParameters(protons=92, neutrons=144, c=1.5) # a_i = 0
        calculator_prolate = FoSShapeCalculator(params_prolate)
        R0 = params_prolate.radius0
        z0 = params_prolate.z0
        self.assertGreater(z0, R0)

        z_coords, rho_coords = calculator_prolate.calculate_shape(n_points=100)
        
        expected_max_rho = R0 * np.sqrt(params_prolate.c)
        closest_to_zero_idx = np.argmin(np.abs(z_coords - params_prolate.zsh)) # zsh is 0 here
        self.assertAlmostEqual(rho_coords[closest_to_zero_idx], expected_max_rho, delta=expected_max_rho*0.01,
                               msg="Max rho for prolate spheroid incorrect.")
        
        # Length along z should be approx 2*z0
        self.assertAlmostEqual(z_coords[-1] - z_coords[0], 2 * z0, delta=z0*0.01)

    def test_oblate_shape_qualitative(self):
        # For an oblate shape (c < 1), z0 < R0.
        params_oblate = FoSParameters(protons=92, neutrons=144, c=0.7) # a_i = 0
        calculator_oblate = FoSShapeCalculator(params_oblate)
        R0 = params_oblate.radius0
        z0 = params_oblate.z0
        self.assertLess(z0, R0)

        z_coords, rho_coords = calculator_oblate.calculate_shape(n_points=100)
        
        expected_max_rho = R0 * np.sqrt(params_oblate.c)
        closest_to_zero_idx = np.argmin(np.abs(z_coords - params_oblate.zsh)) # zsh is 0 here
        self.assertAlmostEqual(rho_coords[closest_to_zero_idx], expected_max_rho, delta=expected_max_rho*0.01,
                               msg="Max rho for oblate spheroid incorrect.")
        
        # Length along z should be approx 2*z0
        self.assertAlmostEqual(z_coords[-1] - z_coords[0], 2 * z0, delta=z0*0.01)


class TestCalculateNormalizedShape(unittest.TestCase):
    def setUp(self):
        # Using a default set of parameters that are not a perfect sphere initially
        # to make volume normalization meaningful.
        self.params_deformed = FoSParameters(protons=92, neutrons=144, c=1.5, a3=0.1, a4=0.05) # A=236
        self.calculator_deformed = FoSShapeCalculator(self.params_deformed)

        self.params_sphere = FoSParameters(protons=92, neutrons=144, c=1.0) # A=236, perfect sphere
        self.calculator_sphere = FoSShapeCalculator(self.params_sphere)

    def test_normalization_reports(self):
        # Test that original_volume and sphere_volume are correctly reported
        _, _, reported_original_volume, reported_sphere_volume, _ = \
            self.calculator_deformed.calculate_normalized_shape()

        # Calculate them independently for verification
        z_orig, rho_orig = self.calculator_deformed.calculate_shape()
        expected_original_volume = self.calculator_deformed.calculate_volume(z_orig, rho_orig)
        expected_sphere_volume = self.calculator_deformed.calculate_sphere_volume()

        self.assertAlmostEqual(reported_original_volume, expected_original_volume, delta=expected_original_volume*0.001)
        self.assertAlmostEqual(reported_sphere_volume, expected_sphere_volume, delta=expected_sphere_volume*0.001)

    def test_scaling_factor_calculation(self):
        # Test that scaling_factor = (sphere_volume / original_volume) ** (1/3)
        _, _, original_volume, sphere_volume, scaling_factor = \
            self.calculator_deformed.calculate_normalized_shape()
        
        if original_volume == 0: # Avoid division by zero if shape calculation yields zero volume
            self.fail("Original volume is zero, cannot test scaling factor.")

        expected_scaling_factor = (sphere_volume / original_volume) ** (1/3)
        self.assertAlmostEqual(scaling_factor, expected_scaling_factor, places=6)

    def test_normalized_volume_conservation(self):
        # Test that the volume of the normalized shape is close to sphere_volume
        z_norm, rho_norm, _, sphere_volume, _ = \
            self.calculator_deformed.calculate_normalized_shape()

        # Calculate volume of the normalized shape
        # Need to use the static method calculate_volume or an instance of FoSShapeCalculator
        volume_of_normalized_shape = FoSShapeCalculator.calculate_volume(z_norm, rho_norm)

        # This should be very close to the sphere_volume
        # Tolerance here depends on integration accuracy and the effect of scaling.
        # Using a slightly larger delta than for direct volume comparisons.
        self.assertAlmostEqual(volume_of_normalized_shape, sphere_volume, delta=sphere_volume*0.001, # 0.1%
                             msg="Volume of normalized shape does not match sphere volume.")

    def test_normalization_on_perfect_sphere(self):
        # For a perfect sphere, original volume should be sphere volume, scaling factor should be ~1.0
        z_norm, rho_norm, original_volume, sphere_volume, scaling_factor = \
            self.calculator_sphere.calculate_normalized_shape()

        self.assertAlmostEqual(original_volume, sphere_volume, delta=sphere_volume*0.01, # 1% for initial calc
                             msg="Original volume of a sphere is not close to sphere_volume.")
        self.assertAlmostEqual(scaling_factor, 1.0, delta=1e-3, # Scaling factor should be very close to 1
                             msg="Scaling factor for a sphere is not close to 1.0.")

        volume_of_normalized_shape = FoSShapeCalculator.calculate_volume(z_norm, rho_norm)
        self.assertAlmostEqual(volume_of_normalized_shape, sphere_volume, delta=sphere_volume*0.001,
                             msg="Normalized volume of a sphere does not match sphere_volume.")
        
        # Also, z_norm, rho_norm should be very similar to z_orig, rho_orig for a sphere
        z_orig_sphere, rho_orig_sphere = self.calculator_sphere.calculate_shape()
        np.testing.assert_array_almost_equal(z_norm, z_orig_sphere, decimal=3)
        np.testing.assert_array_almost_equal(rho_norm, rho_orig_sphere, decimal=3)

    def test_normalized_shape_coordinates_scaling(self):
        # Test if z_normalized = z_orig * scaling_factor and rho_normalized = rho_orig * scaling_factor
        z_orig, rho_orig = self.calculator_deformed.calculate_shape()
        z_norm, rho_norm, _, _, scaling_factor = \
            self.calculator_deformed.calculate_normalized_shape()

        expected_z_norm = z_orig * scaling_factor
        expected_rho_norm = rho_orig * scaling_factor

        np.testing.assert_array_almost_equal(z_norm, expected_z_norm, decimal=6,
                                             err_msg="z_normalized is not correctly scaled from z_original.")
        np.testing.assert_array_almost_equal(rho_norm, expected_rho_norm, decimal=6,
                                             err_msg="rho_normalized is not correctly scaled from rho_original.")


class TestCalculateSphereVolume(unittest.TestCase):
    def test_calculate_sphere_volume_basic(self):
        # R0 = r0 * nucleons**(1/3)
        # V_sphere = (4/3) * pi * R0**3
        params = FoSParameters(protons=50, neutrons=70) # A = 120
        calculator = FoSShapeCalculator(params)

        expected_R0 = params.r0_constant * (120 ** (1 / 3))
        expected_volume = (4/3) * np.pi * (expected_R0**3)
        
        calculated_volume = calculator.calculate_sphere_volume()
        self.assertAlmostEqual(calculated_volume, expected_volume, places=6)

    def test_calculate_sphere_volume_different_r0(self):
        params = FoSParameters(protons=50, neutrons=70, r0_constant=1.2)  # A = 120
        calculator = FoSShapeCalculator(params)
        
        expected_R0 = 1.2 * (120**(1/3))
        expected_volume = (4/3) * np.pi * (expected_R0**3)
        
        calculated_volume = calculator.calculate_sphere_volume()
        self.assertAlmostEqual(calculated_volume, expected_volume, places=6)

    def test_calculate_sphere_volume_different_nucleon_number(self):
        params1 = FoSParameters(protons=20, neutrons=20) # A = 40
        calculator1 = FoSShapeCalculator(params1)
        expected_R0_1 = params1.r0_constant * (40 ** (1 / 3))
        expected_volume_1 = (4/3) * np.pi * (expected_R0_1**3)
        self.assertAlmostEqual(calculator1.calculate_sphere_volume(), expected_volume_1, places=6)

        params2 = FoSParameters(protons=100, neutrons=150) # A = 250
        calculator2 = FoSShapeCalculator(params2)
        expected_R0_2 = params2.r0_constant * (250 ** (1 / 3))
        expected_volume_2 = (4/3) * np.pi * (expected_R0_2**3)
        self.assertAlmostEqual(calculator2.calculate_sphere_volume(), expected_volume_2, places=6)


if __name__ == '__main__':
    unittest.main()
