# Nuclear Shape Plotter (Fourier-over-Spheroid)

This project implements an interactive plotter for visualizing nuclear shapes using the Fourier-over-Spheroid (FoS) parametrization, as described in Pomorski et al. (2023). It allows users to adjust various shape parameters and observe their effects on the nuclear shape in real-time. The application ensures volume conservation by normalizing the calculated shapes.

## Running the Application

To run the application, ensure you have Python installed along with the following libraries:
- `numpy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib
```

Then, execute the `main.py` script:
```bash
python main.py
```

This will open an interactive window where you can adjust the parameters.

## Parameters

The FoS parametrization involves several parameters:

-   **Z**: Number of protons.
-   **N**: Number of neutrons.
-   **c**: Elongation parameter. Defines the overall elongation of the nucleus along the z-axis. `c = 1` corresponds to a spherical shape in the absence of other deformations.
-   **q₂**: Entangled parameter related to elongation `c` and neck parameter `a₄` by the formula: `c = q₂ + 1.0 + 1.5 * a₄`. Adjusting `q₂` will affect `c` and vice-versa, depending on `a₄`.
-   **a₃**: Reflection asymmetry parameter. Non-zero values introduce a pear-like shape (octuple deformation).
-   **a₄**: Neck parameter (hexadecapole deformation). Positive values create a neck in the middle of the nucleus, while negative values make it more diamond-like.
-   **a₅**: Higher-order reflection asymmetry parameter.
-   **a₆**: Higher-order neck/deformation parameter.

Derived parameters:
-   **A**: Total number of nucleons (Z + N).
-   **R₀**: Radius of a sphere with the same volume as the nucleus, calculated as `r₀ * A^(1/3)`, where `r₀` is a radius constant (default 1.16 fm).
-   **z₀**: Half-length of the nucleus, `c * R₀`.
-   **z_sh**: Shift along the z-axis to place the center of mass at the origin, important for asymmetric shapes.
-   **a₂**: Parameter determined by volume conservation constraints, related to `a₄` and `a₆` (`a₂ = a₄/3 - a₆/5 + ...`).

The application dynamically updates the shape and related physical quantities as you adjust these parameters using sliders. It also provides preset buttons for common nuclear shapes and an option to save the current plot.