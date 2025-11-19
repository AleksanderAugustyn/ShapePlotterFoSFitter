"""
Nuclear Shape Calculator and Plotter
Main entry point.
"""

import argparse
import os
import sys

from BatchProcessor import calculate_and_save_shape
from FoSPlotter import FoSShapePlotter


def create_parser():
    """Create a command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Nuclear Shape Calculator using Fourier-over-Spheroid Parametrization',
        epilog='If no arguments are provided, the interactive GUI will launch.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('Z', type=int, nargs='?', help='Number of protons')
    parser.add_argument('N', type=int, nargs='?', help='Number of neutrons')
    parser.add_argument('c_elongation', type=float, nargs='?', help='Elongation parameter c')
    parser.add_argument('a3', type=float, nargs='?', help='Reflection asymmetry parameter')
    parser.add_argument('a4', type=float, nargs='?', help='Neck parameter')
    parser.add_argument('a5', type=float, nargs='?', help='Higher order parameter')
    parser.add_argument('a6', type=float, nargs='?', help='Higher order parameter')
    parser.add_argument('number_of_points', type=int, nargs='?', help='Number of points (180-3600)')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for saved files (default: current directory)')
    return parser


def validate_parameters(z, n, c_elongation, a3, a4, a5, a6, number_of_points):
    """Validate input parameters."""
    errors = []
    if z <= 0: errors.append("Z must be positive")
    if n <= 0: errors.append("N must be positive")
    if c_elongation <= 0: errors.append("c_elongation must be positive")
    for param_name, param_value in [('a3', a3), ('a4', a4), ('a5', a5), ('a6', a6)]:
        if abs(param_value) >= 2: errors.append(f"{param_name} must have absolute value less than 2")
    if not (180 <= number_of_points <= 3600): errors.append("number_of_points must be between 180 and 3600")
    return errors


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Check if all required arguments are provided for batch mode
    if all(v is not None for v in [args.Z, args.N, args.c_elongation, args.a3,
                                   args.a4, args.a5, args.a6, args.number_of_points]):
        # Batch mode
        errors = validate_parameters(args.Z, args.N, args.c_elongation, args.a3,
                                     args.a4, args.a5, args.a6, args.number_of_points)

        if errors:
            print("Error: Invalid parameters:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir)
            except OSError as e:
                print(f"Error: Could not create output directory: {e}", file=sys.stderr)
                sys.exit(1)

        success = calculate_and_save_shape(
            args.Z, args.N, args.c_elongation, args.a3, args.a4, args.a5, args.a6,
            args.number_of_points, args.output_dir
        )

        if not success:
            print("Error: Failed to calculate or save shape", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    elif any(v is not None for v in [args.Z, args.N, args.c_elongation, args.a3,
                                     args.a4, args.a5, args.a6, args.number_of_points]):
        print("Error: All parameters must be provided for batch mode", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    else:
        # No arguments - run GUI mode
        plotter = FoSShapePlotter()
        plotter.run()


if __name__ == '__main__':
    main()
