#! /usr/bin/env python3

r"""
Notes from Chapter 4 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = -a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)
               u'(t) = f(u, t)     t ∈ (0, T]     u(0) = I             (2)

"""

import numpy as np
from matplotlib import pyplot as plt

from utils.solver import compute_rates


__all__ = ['scaling']

IMGDIR = './img/chap4/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def scaling() -> None:
    """
    Scaling and dimensionless variables.

    Notes
    ----------
    Real applications of a model u' = -a u + b will often involve a lot of
    parameters in the expressions for a and b. It can be quite challenging to
    find relevant values for all parameters. In simple problems, however, it
    turns out that it is not always necessary to estimate all parameters
    because we can lump them into one or a few dimensionless numbers by using a
    very attractive technique called scaling. It simply means to stretch the u
    and t axis in the present problem - and suddenly all parameters in the
    problem are lumped into one parameter if b ≠ 0 and no parameter when b = 0!

    Scaling means we introduce a new function ū(ť), with

                       ū = (u - uᵐ) / uᶜ        ť = t / tᶜ

    where uᵐ is a characteristic value of u, uᶜ is a characteristic size of the
    range of u values, and tᶜ is a characteristic size of the range of t where
    u shows significant variation. Choosing uᵐ, uᶜ, and tᶜ is not always easy
    and is often an art in complicated problems. We just state one choice
    first:

                     uᶜ = I,      uᵐ = b / a,     tᶜ = 1 / a

    Inserting u = uᵐ + uᶜ ū and t = tᶜ ť in the problem u' = a u + b, assuming
    a and b are constants, results (after some algebra) in the scaled problem

                       dū/dť = -ū,          ū(0) = 1 - β

    where                       β = b / (I a)

    The parameter β is a dimensionless number. An important observation is that
    ū depends on ť and β. That is, only the special combination of b / (I a)
    matters, not what the individual values of b, a, and I are. The original
    unscaled function depends on t, b, a, and I. A second observation is
    striking: if b = 0, the scaled problem is independent of a and I! In
    practice, this means that we can perform a single numerical simulation of
    the scaled problem and recover the solution of any problems for any given a
    and I by stretching the axis in the plot: u = I ū and t = ť / a. For any
    b ≠ 0, we simulate the scaled problem for a few β values and recover the
    physical solution u by translating and stretching the t axis.

    In general, scaling combines the parameters in a problem to a set of
    dimensionless parameters. The number of dimensionless parameters is usually
    much smaller than the number of original parameters.

    This scaling breaks down if I = 0. In that case, we may choose uᵐ = 0,
    uᶜ = b / a, and tᶜ = 1 / b, resulting in the slightly different scaled
    problem

                       dū/dť = 1 - ū,          ū(0) = 0

    As with b = 0, the case I = 0 has a scaled problem with no physical
    parameters! It is common to drop the bars after scaling and write the
    scaled problem as u' = u, u(0) = 1 - β, or u' = 1 - u, u(0) = 0. Any
    implementation of the problem u' = -a u + b, u(0) = I can then be reused
    for the scaled problem by setting a = 1, b = 0, and I = 1 - β if I ≠ 0, or
    one sets a = 1, b = 1, and I = 0 when the physical I is zero.

    """
    from utils.solver import solver_chap2 as solver

    # Definitions
    # Solving y'(x) = λ y + α    x ∈ (0, X]    y(0) = I ≠ 0
    I = 1
    X = 4
    lamda = 0.5
    alpha = 2
    dx = 0.05
    Nx = int(X / dx)
    x = np.linspace(0, Nx * dx, Nx + 1)
    y_exact = lambda x: np.exp(x * lamda) * I + \
        ((-1 + np.exp(x * lamda)) * alpha) / lamda

    # Scaling
    #                  ỹ = (y + yᵐ) / yᶜ,      𝐱 = x / xᶜ
    #               yᶜ = I,     yᵐ = α / λ,       xᶜ = 1 / λ
    # therefore,
    #        dỹ/d𝐱 = ỹ,     𝐱 ∈ (0, λ T],     ỹ(0) = 1 + β = 1 + α / (I λ)
    #
    #                        y(x) = (ỹ(λ x) - β) * I
    beta = alpha / (I * lamda)
    t = lamda * x

    # Solve dỹ/d𝐱 = ỹ using Crank-Nicolson scheme
    u, _ = solver(I=1 + beta, a=-1, T=t[-1], dt=t[1] - t[0], theta=0.5)

    # Convert back to y
    y = (u - beta) * I
    print(STR_FMT.format('u', u))
    print(STR_FMT.format('y', y))

    # Calculate exact solution
    y_e = y_exact(x)

    # Assert that max deviation is below tolerance
    tol = 1E-2
    diff = np.max(np.abs(y_e - y))
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}')


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description=
        'Finite Difference Computing with Exponential Decay Models - Chapter 4'
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

