#! /usr/bin/env python3

r"""
Notes from Chapter 3 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = -a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

from utils.solver import solver_chap3 as solver


__all__ = ['generalisation', 'verification', 'convergence']

IMGDIR = './img/chap3/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def generalisation() -> None:
    """
    Generalisation: including a variable coefficient and a source term.

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

    A further extension of the model ODE is to include a source term b(t):

              u'(t) = a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I

    The time point where we sample the ODE determines where b(t) is evaluated.
    For the Crank-Nicolson scheme and the θ-rule, we have a choice of whether
    to evaluate a(t) and b(t) at the correct point, or use an average. The
    chosen strategy becomes particularly clear if we write up the schemes in
    operation notation:

                            [Dₜ⁺ u = -a u + b]ⁿ
                            [Dₜ⁻ u = -a u + b]ⁿ
                            [Dₜ u = -a ūᵗ + b]ⁿ⁺¹⸍²
                            [Dₜ u = -ā ū + b̄ᵗ]ⁿ⁺¹⸍²
                            [D̄ₜ u = -a ūᵗʼᶿ + b]ⁿ⁺ᶿ
                            [D̄ₜ u = -ā ū + b̄ᵗʼᶿ]ⁿ⁺ᶿ

    Deriving the 0-rule formula when averaging over a and b, we get

        (uⁿ⁺¹ - uⁿ) / Δt = 0(-aⁿ⁺¹ uⁿ⁺¹ + bⁿ⁺¹) + (1 - 0)(-aⁿ uⁿ + bⁿ)

    Solving for uⁿ⁺¹,

     uⁿ⁺¹ = ((1 - Δt(1 - 0)aⁿ) uⁿ + Δt(0 bⁿ⁺¹ + (1 - 0)bⁿ)) / (1 + Δt 0 aⁿ⁺¹)

    Here, we start by verifying a constant solution, where u = C. We choose any
    a(t) and set b(t) = a(t) C and I = C.

    """
    # Definitions
    u_const = 2.15
    I = u_const
    Nt = 4
    dt = 4
    theta = 0.4
    a = lambda t: 2.5 * (1 + t**3)
    b = lambda t: a(t) * u_const

    # Solve
    u, t = solver(I=I, a=a, b=b, T=Nt * dt, dt=dt, theta=theta)

    # Calculate exact solution
    u_exact = lambda t: u_const
    u_e = u_exact(t)

    # Assert that max deviation is below tolerance
    tol = 1E-14
    diff = np.max(np.abs(u_e - u))
    if diff > tol:
        raise AssertionError(f'Tolerance not reach, diff = {diff}')


def verification() -> None:
    """
    Verification via manufactured solutions.

    Notes
    ----------
    Following the idea above, we choose any formula as the exact solution,
    insert the formula in the ODE problem and fit the data a(t), b(t), and I to
    make the chosen formula fulfill the equation. This powerful technique for
    generating exact solutions is very useful for verification purposes and
    known as the method of manufactured solutions, often abbreviated MMS.

    One common choice of solution is a linear function in the independent
    variable(s). The rationale behind such a simple variation is that almost
    any relevant numerical solution method for differential equation problems
    is able to reproduce a linear function exactly to machine precision. The
    linear solution also makes some stronger demands to the numerical solution
    and the implementation than the constant solution one.

    We choose a linear solution u(t) = c t + d. From the initial condition, it
    follows that d = I. Inserting this u in the left-hand side of (1), we get

                            c = -a(t) u + b(t)

    Any function u = c t + I is then a correct solution if we choose

                          b(t) = c + a(t)(c t + I)

    Therefore, we must check that uⁿ = c a(tn)(c tn + I) fulfills the discrete
    equations. For these equations, it is convenient to compute the action of
    the difference operator on a linear function t:

                         [Dₜ⁺ t]ⁿ = (tⁿ⁺¹ - tⁿ) / Δt = 1
                         [Dₜ⁻ t]ⁿ = (tⁿ - tⁿ⁻¹) / Δt = 1
                         [Dₜ t]ⁿ = (tⁿ⁺¹⸍² - tⁿ⁻¹⸍²) / Δt
                                 = ((n + ½)Δt - (n - ½)Δt) / Δt = 1

    Clearly, all three difference approximations to the derivative are exact
    for u(t) = t or its mesh function counterpart uⁿ = tn. The difference
    equation in the Forward Euler scheme

                              [Dₜ⁺ u = -a u + b]ⁿ

    with aⁿ = a(tn), bⁿ = c + a(tn)(c tn + I), and uⁿ = ctn + I then results in

                  c = -a(tn)(c tn + I) + c + a(tn)(c tn + I) = c

    which is always fulfilled. Similar calculations can be done for the Forward
    Euler and Crank-Nicolson schemes. Therefore, we expect that uⁿ - uₑ(tn) = 0
    mathematically and |uⁿ - uₑ(tn)| less than a small number about the machine
    precision.

    """
    # Definitions
    I = 0.1
    T = 4
    dt = 0.1
    Nt = int(T / dt)
    theta = 0.4
    c = -0.5
    u_exact = lambda t: c * t + I
    a = lambda t: t**0.5
    b = lambda t: c + a(t) * u_exact(t)

    # Solve
    u, t = solver(I=I, a=a, b=b, T=Nt * dt, dt=dt, theta=theta)

    # Calculate exact solution
    u_e = u_exact(t)

    # Assert that max deviation is below tolerance
    tol = 1E-14
    diff = np.max(np.abs(u_e - u))
    if diff > tol:
        raise AssertionError(f'Tolerance not reach, diff = {diff}')


def convergence() -> None:
    """
    Computing convergence rates.

    Notes
    ----------
    We expect that the error E in the numerical solution is reduced if the mesh
    size Δt is decreased. More specifically, many numerical methods obey a
    power-law relation between E and Δt

                                E = C Δtʳ

    where C and r are (usually unknown) constant independent of Δt. The
    parameter r is known as the convergence rate. For example, if the
    convergence rate is 2, halving Δt reduces the error by a factor of 4.
    Diminishing Δt then has a greater impact on the error compared with methods
    that have r = 1. For a given value of r, we refer to the method as of r-th
    order.

    There are two alternative ways of estimating C and r based on a set of m
    simulations with corresponding pairs (Δtᵢ, Eᵢ), i = 0,...,m-1, and
    Δtᵢ < Δtᵢ₋₁.
    1) Take the logarithm of E = C Δtʳ and fit a straight line to the data
       points (Δtᵢ, Eᵢ).
    2) Consider two consecutive experiments (Δtᵢ, Eᵢ) and (Δtᵢ₋₁, Eᵢ₋₁).
       Dividing the equation Eᵢ₋₁ = C Δtʳᵢ₋₁ by Eᵢ = C Δtʳᵢ and solving for r

                      rᵢ₋₁ = ln(Eᵢ₋₁ / Eᵢ) / ln(Δtᵢ₋₁ / Δtᵢ)

    for i = 1,...,m-1. Note that we have introduced a subindex i - 1 on r
    because r estimated from a pair of experiments must be expected to change
    with i.

    The disadvantage of method 1 is that it may not be valid for the coarsest
    meshes (largest Δt values). Fitting a line to all the data points is then
    misleading. Method 2 computes the convergence rates for pairs of
    experiments and allows us to see if the sequence rᵢ converges to some
    values as i → m - 2. The final rₘ₋₂ can then be taken as the convergence
    rate.

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
