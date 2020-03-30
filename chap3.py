#! /usr/bin/env python3

r"""
Notes from Chapter 3 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

from utils.solver import solver


__all__ = ['variable_coefficient']

IMGDIR = './img/chap3/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def variable_coefficient() -> None:
    """
    Generalisation: including a variable coefficient

    Notes
    ----------
    We now start to look at the generalisations u' = -a(t) u and
    u' = -a(t) u + b(t). Verification can no longer make use of an exact
    solution of the numerical problem, so we make use of manufactured
    solutions, for deriving an exact solution of the ODE problem, and then we
    can compute empirical convergence rates for the method and see if these
    coincide with the expected rates from theory.

    We start by considered the case where a depends on time

               u'(t) = -a(t) u(t)       t ∈ (0, T]     u(0) = I       (2)

    A Forward Euler scheme consists of evaluating (2) at t = tn, and
    approximating with a forward difference [Dₜ⁺ u]ⁿ.

                        (uⁿ⁺¹ - uⁿ) / Δt = -a(tn) uⁿ

    The Backward scheme becomes

                        (uⁿ - uⁿ⁻¹) / Δt = -a(tn) uⁿ

    The Crank-Nicolson scheme builds on sampling the ODE at t{n+½}. We can
    evaluate at t{n+½} and use an average u at times tn and t{n+1}

                 (uⁿ⁺¹ - uⁿ) / Δt = -a(t{n+½}) ½(uⁿ + uⁿ⁺¹)

    Alternatively, we can use an average for the product a u

               (uⁿ⁺¹ - uⁿ) / Δt = -½(a(tn) uⁿ + a(t{n+1}) uⁿ⁺¹)

    The θ-tule unifies the three mentioned schemes. One version is to have a
    evaluated at the weighted time point (1 - θ) tn + θ t{n+1}

        (uⁿ⁺¹ - uⁿ) / Δt = -a((1 - θ) tn + θ t{n+1})((1 - θ) uⁿ + θ uⁿ⁺¹)

    Another possibility is to apply a weighted average for the product a u

             (uⁿ⁺¹ - uⁿ) / Δt = -(1 - θ) a(tn) uⁿ + θ a(t{n+1})uⁿ⁺¹

    With the finite difference operator notation, the Forward Euler and
    Backward Euler can be summarised as

                              [Dₜ⁺ u = -a u]ⁿ
                              [Dₜ⁻ u = -a u]ⁿ

    The Crank-Nicolson and θ schemes depend on whether we evaluate a at the
    sample point for the ODE or if we use an average.

                            [Dₜ u = -a ūᵗ]ⁿ⁺¹⸍²
                            [Dₜ u = -ā ūᵗ]ⁿ⁺¹⸍²
                            [D̄ₜ u = -a ūᵗʼᶿ]ⁿ⁺ᶿ
                            [D̄ₜ u = -ā ūᵗʼᶿ]ⁿ⁺ᶿ

    """


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description=
        'Finite Difference Computing with Exponential Decay Models - Chapter 3'
    )
    parser.add_argument('functions', nargs='*', help=f'Choose from {__all__}')
    args = parser.parse_args()

    functions = args.functions if args.functions else __all__
    for f in functions:
        if f not in __all__:
            raise ValueError(f'Invalid function "{f}" (choose from {__all__})')
        print('------', f'\nRunning "{f}"')
        globals()[f]()
        print('------')


main.__doc__ = __doc__


if __name__ == "__main__":
    main()
