# Nuclear Shape Plotter (FoS + Beta Harmonics)

This project implements an interactive plotter for visualizing nuclear shapes using two complementary parametrizations:

1. **Fourier-over-Spheroid (FoS)** parametrization, as described in Pomorski et al. (2023)
2. **Beta Spherical Harmonics** parametrization fitting

The application allows users to adjust FoS shape parameters and observe their effects on the nuclear shape in real-time, while simultaneously fitting and displaying the equivalent beta spherical harmonics representation. This dual approach
provides insights into the relationship between different nuclear shape parametrizations and ensures volume conservation through normalization.

## Features

### Core Functionality

- **Interactive FoS Shape Generation**: Real-time visualization of nuclear shapes using the Fourier-over-Spheroid parametrization
- **Beta Harmonics Fitting**: Automatic fitting of beta spherical harmonics (β₁, β₂, β₃, β₄, etc.) to the generated FoS shape
- **Multi-Coordinate System Display**: Simultaneous visualization in cylindrical coordinates (original FoS), spherical coordinates (converted FoS), and reconstructed beta harmonics
- **Volume Conservation**: Ensures physical consistency through proper normalization across all representations
- **Advanced Parameter Controls**: Increment/decrement buttons and sliders for precise parameter adjustment

### Shape Analysis & Validation

- **Center of Mass Analysis**: Real-time calculation and display of center of mass positions for:
  - Theoretical shift prediction
  - FoS shape (cylindrical coordinates)
  - Spherical coordinate conversion
  - Beta harmonics reconstruction
- **Conversion Validation**: RMSE metrics for cylindrical-to-spherical coordinate conversion accuracy
- **Fitting Quality Assessment**: RMSE analysis of beta parametrization fitting quality
- **Volume Verification**: Cross-validation of volume conservation across all parametrizations

### Physical Quantities & Measurements

- **Shape Dimensions**: Automatic calculation of maximum extents, neck radius, and elongation ratios
- **Nuclear Physics Parameters**: Display of nuclear radius R₀, nucleon number effects, and shape evolution
- **Parameter Relationships**: Real-time analysis of entangled parameters (c, q₂, a₄ relationships)
- **Reference Visualization**: Dynamic reference sphere that updates with nucleon number changes

### User Interface Features

- **Preset Configurations**: Quick access to common nuclear shapes (spherical, prolate, oblate, pear-shaped, two-center)
- **Advanced Information Panel**: Comprehensive display of:
  - Parameter relationships and constraints
  - Volume and dimension analysis
  - Center of mass calculations
  - Fitting quality metrics
  - Significant beta parameters (>0.001 threshold)
- **Export Capabilities**: Save plots and parameter sets for further analysis
- **Interactive Controls**: Fine-tuned adjustment with both sliders and increment/decrement buttons

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

- **Z**: Number of protons.
- **N**: Number of neutrons.
- **c**: Elongation parameter. Defines the overall elongation of the nucleus along the z-axis. `c = 1` corresponds to a spherical shape in the absence of other deformations.
- **q₂**: Entangled parameter related to elongation `c` and neck parameter `a₄` by the formula: `c = q₂ + 1.0 + 1.5 * a₄`. Adjusting `q₂` will affect `c` and vice versa, depending on `a₄`.
- **a₃**: Reflection asymmetry parameter. Non-zero values introduce a pear-like shape (octupole deformation).
- **a₄**: Neck parameter (hexadecapole deformation). Positive values create a neck in the middle of the nucleus, while negative values make it more diamond-like.
- **a₅**: Higher-order reflection asymmetry parameter.
- **a₆**: Higher-order neck/deformation parameter.
- **Max Betas Used For Fit**: Controls the maximum number of beta harmonics (β₁ through β_max) used in the spherical harmonics fitting process.

## Beta Spherical Harmonics Parameters

The beta harmonics parametrization uses deformation parameters:

- **β₂**: Quadrupole deformation parameter (prolate/oblate shapes)
- **β₃**: Octupole deformation parameter (reflection asymmetry)
- **β₄**: Hexadecapole deformation parameter (necking/anti-necking)
- **β₅, β₆, ...**: Higher-order deformation parameters

The application automatically determines which beta parameters are significant (|β| > 0.001) and displays them in the information panel.

## Derived and Physical Quantities

### Automatically Calculated Parameters

- **A**: Total number of nucleons (Z + N).
- **R₀**: Radius of a sphere with the same volume as the nucleus, calculated as `r₀ * A^(1/3)`, where `r₀` is a radius constant (default 1.16 fm).
- **z₀**: Half-length of the nucleus, `c * R₀`.
- **z_sh**: Shift along the z-axis to place the center of mass at the origin, important for asymmetric shapes.
- **a₂**: Parameter determined by volume conservation constraints, related to `a₄` and `a₆` (`a₂ = a₄/3 - a₆/5 + ...`).

### Shape Analysis Metrics

- **Volume Calculations**: Comparison of volumes across FoS cylindrical, spherical conversion, and beta reconstruction
- **Center of Mass Positions**: Multiple CM calculations to verify theoretical predictions and coordinate transformations
- **Shape Dimensions**: Maximum z-extent, maximum ρ-extent, neck radius (for necked shapes)
- **Conversion Accuracy**: RMSE metrics for coordinate system transformations and parametrization fitting

## Parameter Correlation Analysis

The application provides real-time analysis of the relationship between FoS and beta harmonics parameters:

- **Entangled Parameter Tracking**: Automatic updates of c, q₂, and a₄ relationships
- **Fitting Quality Metrics**: RMSE analysis of both coordinate conversion and beta parametrization fitting
- **Volume Conservation Verification**: Cross-validation of volume preservation across all representations
- **Parameter Significance Analysis**: Identification and display of significant beta harmonics coefficients
- **Physical Consistency Checks**: Center of mass validation across different coordinate systems

## Advanced Coordinate System Handling

The application handles three coordinate representations:

1. **Cylindrical Coordinates (z, ρ)**: Original FoS parametrization output
2. **Spherical Coordinates (r, θ)**: Converted for beta harmonics analysis
3. **Beta Reconstruction**: Shape reconstructed from fitted beta parameters

Each representation is validated for consistency, with RMSE metrics displayed for conversion accuracy.

## Usage Workflow

1. **Set Nuclear Properties**: Adjust Z (protons) and N (neutrons) for your nucleus of interest
2. **Configure FoS Shape**: Use sliders and increment/decrement buttons to modify FoS parameters
3. **Monitor Beta Fitting**: Watch real-time fitting of beta harmonics with quality metrics
4. **Analyze Correlations**: Review parameter relationships and fitting accuracy in the information panel
5. **Validate Physics**: Check center of mass calculations and volume conservation
6. **Export Results**: Save plots and parameter correlations for documentation

## Preset Configurations

- **Sphere**: Perfectly spherical nucleus (c=1.0, all deformation parameters=0)
- **Prolate**: Elongated along the symmetry axis
- **Oblate**: Flattened along the symmetry axis
- **Pear-shaped**: Reflection asymmetric configuration
- **Two-center**: Highly elongated with pronounced necking

## Quality Control and Validation

The application includes comprehensive validation features:

- **Coordinate Conversion Validation**: RMSE tracking for cylindrical-to-spherical transformation
- **Beta Fitting Quality**: RMSE analysis of spherical harmonics reconstruction accuracy
- **Volume Conservation Check**: Verification that volume is preserved across all representations
- **Center of Mass Consistency**: Multiple CM calculations to ensure physical correctness
- **Parameter Relationship Verification**: Real-time validation of entangled parameter constraints

## Scientific Applications

This tool is valuable for:

- **Nuclear Structure Studies**: Understanding shape evolution in different mass regions with dual parametrization insight
- **Fission Research**: Analyzing shape changes leading to scission with real-time beta parameter tracking
- **Parameter Conversion**: Translating between FoS and beta harmonics frameworks with quality metrics
- **Model Validation**: Comparing predictions from different shape parametrizations with quantitative accuracy assessment
- **Coordinate System Analysis**: Understanding the relationship between cylindrical and spherical nuclear shape representations
- **Educational Purposes**: Visualizing nuclear deformation concepts with interactive parameter exploration

## Technical Features

- **Real-time Parameter Coupling**: Automatic handling of entangled parameters (c, q₂, a₄)
- **Multi-threaded Calculations**: Efficient computation of coordinate transformations and beta fitting
- **Robust Numerical Methods**: Error handling and validation for edge cases in shape calculations
- **Interactive Visualization**: Dynamic updating of all plot elements with parameter changes
- **Comprehensive Information Display**: Detailed parameter relationships, physical quantities, and fitting metrics

The application dynamically updates all representations as you adjust parameters, providing immediate visual feedback and quantitative analysis of the relationship between these important nuclear shape parametrizations, while ensuring
physical consistency through comprehensive validation metrics.