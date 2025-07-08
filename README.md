# Nuclear Shape Plotter (FoS + Beta Harmonics)

This project implements an interactive plotter for visualizing nuclear shapes using two complementary parametrizations:

1. **Fourier-over-Spheroid (FoS)** parametrization, as described in Pomorski et al. (2023)
2. **Beta Spherical Harmonics** parametrization fitting

The application allows users to adjust FoS shape parameters and observe their effects on the nuclear shape in real-time, while simultaneously fitting and displaying the equivalent beta spherical harmonics representation. This dual approach
provides insights into the relationship between different nuclear shape parametrizations and ensures volume conservation through normalization.

## Features

- **Interactive FoS Shape Generation**: Real-time visualization of nuclear shapes using the Fourier-over-Spheroid parametrization
- **Beta Harmonics Fitting**: Automatic fitting of beta spherical harmonics (β₁, β₂, β₃, β₄, etc.) to the generated FoS shape
- **Dual Parametrization Display**: Side-by-side comparison of FoS and beta harmonics representations
- **Volume Conservation**: Ensures physical consistency through proper normalization
- **Preset Configurations**: Quick access to common nuclear shapes (spherical, prolate, oblate, pear-shaped, two-center)
- **Parameter Relationships**: Real-time display of parameter correlations and physical quantities
- **Export Capabilities**: Save plots and parameter sets for further analysis

## Running the Application

To run the application, ensure you have Python installed along with the following libraries:
- `numpy`
- `matplotlib`
- `scipy` (for optimization and fitting routines)

You can install them using pip:
```bash
pip install numpy matplotlib scipy
```

Then, execute the `ShapePlotterFoS.py` script:
```bash
python ShapePlotterFoSFitter.py
```

This will open an interactive window where you can adjust the parameters and observe both parametrizations.

## Fourier-over-Spheroid (FoS) Parameters

The FoS parametrization involves several parameters:

-   **Z**: Number of protons.
-   **N**: Number of neutrons.
-   **c**: Elongation parameter. Defines the overall elongation of the nucleus along the z-axis. `c = 1` corresponds to a spherical shape in the absence of other deformations.
- **q₂**: Entangled parameter related to elongation `c` and neck parameter `a₄` by the formula: `c = q₂ + 1.0 + 1.5 * a₄`. Adjusting `q₂` will affect `c` and vice versa, depending on `a₄`.
-   **a₃**: Reflection asymmetry parameter. Non-zero values introduce a pear-like shape (octuple deformation).
-   **a₄**: Neck parameter (hexadecapole deformation). Positive values create a neck in the middle of the nucleus, while negative values make it more diamond-like.
-   **a₅**: Higher-order reflection asymmetry parameter.
-   **a₆**: Higher-order neck/deformation parameter.

## Beta Spherical Harmonics Parameters

The beta harmonics parametrization uses deformation parameters:

- **β₂**: Quadrupole deformation parameter (prolate/oblate shapes)
- **β₃**: Octupole deformation parameter (reflection asymmetry)
- **β₄**: Hexadecapole deformation parameter (necking/anti-necking)
- **β₅, β₆, ...**: Higher-order deformation parameters

## Derived and Physical Quantities

-   **A**: Total number of nucleons (Z + N).
-   **R₀**: Radius of a sphere with the same volume as the nucleus, calculated as `r₀ * A^(1/3)`, where `r₀` is a radius constant (default 1.16 fm).
-   **z₀**: Half-length of the nucleus, `c * R₀`.
-   **z_sh**: Shift along the z-axis to place the center of mass at the origin, important for asymmetric shapes.
-   **a₂**: Parameter determined by volume conservation constraints, related to `a₄` and `a₆` (`a₂ = a₄/3 - a₆/5 + ...`).

## Parameter Correlation Analysis

The application provides real-time analysis of the relationship between FoS and beta harmonics parameters:

- **Fitting Quality**: R² correlation coefficient between FoS and fitted beta harmonics shapes
- **Parameter Mapping**: Automatic determination of equivalent β values for given FoS parameters
- **Convergence Metrics**: Visual indicators of fitting accuracy and stability
- **Cross-Validation**: Comparison of key physical quantities (volume, center of mass, moments) between parametrizations

## Usage Workflow

1. **Set Nuclear Properties**: Adjust Z (protons) and N (neutrons) for your nucleus of interest
2. **Configure FoS Shape**: Use sliders to modify FoS parameters (c, q₂, a₃, a₄, a₅, a₆)
3. **Observe Beta Fitting**: Watch as the application automatically fits beta harmonics to the FoS shape
4. **Compare Parametrizations**: Analyze the relationship between FoS and beta parameters
5. **Export Results**: Save plots and parameter correlations for documentation

## Preset Configurations

- **Sphere**: Perfectly spherical nucleus (c=1.0, all deformation parameters=0)
- **Prolate**: Elongated along the symmetry axis
- **Oblate**: Flattened along the symmetry axis
- **Pear-shaped**: Reflection asymmetric configuration
- **Two-center**: Highly elongated with pronounced necking

## Scientific Applications

This tool is valuable for:

- **Nuclear Structure Studies**: Understanding shape evolution in different mass regions
- **Fission Research**: Analyzing shape changes leading to scission
- **Parameter Conversion**: Translating between different theoretical frameworks
- **Model Validation**: Comparing predictions from different shape parametrizations
- **Educational Purposes**: Visualizing nuclear deformation concepts

The application dynamically updates both representations as you adjust parameters, providing immediate visual feedback and quantitative analysis of the relationship between these important nuclear shape parametrizations.