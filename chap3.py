#! /usr/bin/env python3

r"""
Notes from Chapter 3 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = -a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)

"""

from typing import List

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

from utils.solver import solver_chap3 as solver


__all__ = ['generalisation', 'verification', 'convergence', 'systems']

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

    Deriving the θ-rule formula when averaging over a and b, we get

        (uⁿ⁺¹ - uⁿ) / Δt = θ(-aⁿ⁺¹ uⁿ⁺¹ + bⁿ⁺¹) + (1 - θ)(-aⁿ uⁿ + bⁿ)

    Solving for uⁿ⁺¹,

     uⁿ⁺¹ = ((1 - Δt(1 - θ)aⁿ) uⁿ + Δt(θ bⁿ⁺¹ + (1 - θ)bⁿ)) / (1 + Δt θ aⁿ⁺¹)

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
        raise AssertionError(f'Tolerance not reached, diff = {diff}')


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
        raise AssertionError(f'Tolerance not reached, diff = {diff}')


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

    The strong practical application of computing convergence rates is for
    verification: wrong convergence rates point to errors in the code, and
    correct convergence rates provide strong support for a correct
    implementation. Bugs in the code can easily destroy the expected
    convergence rate.

    The example here used the manufactured solution uₑ(t) = sin(t) exp{-2t} and
    a(t) = t². This implies we must fit b as b(t) = u'(t) + a(t) u(t). We first
    compute with SymPy expressions and then convert the exact solution, a, and
    b to Python functions that we can use in subsequent numerical computing.

    """
    # Created manufactured solution with SymPy
    t = sym.Symbol('t')
    u_e_sym = sym.sin(t) * sym.exp(-2 * t)
    a_sym = t**2
    b_sym = sym.diff(u_e_sym, t) + a_sym * u_e_sym

    # Turn SymPy expressions into Python functions
    u_exact = sym.lambdify([t], u_e_sym, modules='numpy')
    a = sym.lambdify([t], a_sym, modules='numpy')
    b = sym.lambdify([t], b_sym, modules='numpy')

    def compute_rates(dt_values: List[float],
                      E_values: List[float]) -> List[float]:
        """
        Estimate the convergence rate.

        Parameters
        ----------
        dt_values : List of dt values.
        E_values : List of errors.

        Returns
        ----------
        r : Convergence rates.

        """
        m = len(dt_values)

        # Compute the convergence rates
        # rᵢ₋₁ = ln(Eᵢ₋₁ / Eᵢ) / ln(Δtᵢ₋₁ / Δtᵢ)
        r = [np.log(E_values[i - 1] / E_values[i]) /
             np.log(dt_values[i - 1] / dt_values[i])
             for i in range(1, m)]

        # Round to two d.p.
        r = [round(r_, 2) for r_ in r]
        return r

    T = 6
    I = u_exact(0)
    dt_values = [0.1 * 2**(-i) for i in range(7)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    th_dict = {0: ('Forward Euler', 'fe', 'r-s'),
               1: ('Backward Euler', 'be', 'g-v'),
               0.5: ('Crank-Nicolson', 'cn', 'b-^')}

    for theta in th_dict:
        print(f'{th_dict[theta][0]}')
        E_values = []
        for dt in dt_values:
            # Solve for mesh function
            u, t = solver(I=I, a=a, b=b, T=T, dt=dt, theta=theta)

            # Compute error
            u_e = u_exact(t)
            e = u_e - u
            E = np.sqrt(dt * np.sum(e**2))
            E_values.append(E)

        # Compute convergence rates
        r = compute_rates(dt_values, E_values)
        print(STR_FMT.format('r', f'{r}'))

        # Test final entry with expected convergence rate
        expected_rate = 2 if theta == 0.5 else 1
        tol = 0.1
        diff = np.abs(expected_rate - r[-1])
        if diff > tol:
            raise AssertionError(f'Tolerance not reached, diff = {diff}')


def systems() -> None:
    """
    Extension to systems of ODEs.

    Notes
    ----------
    Many ODE models involve more than one unknown function and more than one
    equation. Here is an example of two unknown functions u(t) and v(t)

                u' = a u + b v             v' = c u + d v

    for constants a, b, c, d. Applying the Forward Euler method to each
    equation results in a simple updating formula

                       uⁿ⁺¹ = uⁿ + Δt(a uⁿ + b vⁿ)
                       vⁿ⁺¹ = vⁿ + Δt(c uⁿ + d vⁿ)

    On the other hand, the Crank-Nicolson and Backward Euler schemes result in
    a 2x2 linear system for the new unknowns. The latter scheme becomes

                       uⁿ⁺¹ = uⁿ + Δt(a uⁿ⁺¹ + b vⁿ⁺¹)
                       vⁿ⁺¹ = vⁿ + Δt(c uⁿ⁺¹ + d vⁿ⁺¹)

    Collecting uⁿ⁺¹ as well as vⁿ⁺¹ on the left-hand side results in

                        (1 - Δt a)uⁿ⁺¹ + b vⁿ⁺¹ = uⁿ
                        c uⁿ⁺¹ + (1 - Δt d)vⁿ⁺¹ = vⁿ

    which is a system of two coupled, linear, algebraic equations in two
    unknowns. These equations can be solved algebraically resulting in explicit
    forms for uⁿ⁺¹ and vⁿ⁺¹ that can be directly implemented. For a system of
    ODEs with many equations and unknowns, one will express the coupled
    equations at each time level in matrix form and call software for numerical
    solution of linear systems of equations.

    Deriving the θ-rule formula we get

           (uⁿ⁺¹ - uⁿ) / Δt = θ(a uⁿ⁺¹ + b vⁿ⁺¹) + (1 - θ)(a uⁿ + b vⁿ)
           (vⁿ⁺¹ - vⁿ) / Δt = θ(c uⁿ⁺¹ + d vⁿ⁺¹) + (1 - θ)(c uⁿ + v uⁿ)

    Collecting uⁿ⁺¹ as well as vⁿ⁺¹ on the left-hand side results in

        (Δt a θ - 1) uⁿ⁺¹ + b Δt θ vⁿ⁺¹ = (a Δt(θ - 1) - 1)uⁿ + b Δt(θ - 1)vⁿ
        c Δt θ uⁿ⁺¹ + (d Δt θ - 1) vⁿ⁺¹ =  c Δt(θ - 1)uⁿ + (d Δt(θ - 1) - 1)vⁿ

    Which can be solved for uⁿ⁺¹ and vⁿ⁺¹ at each time step, n.

    """
    import scipy.linalg

    # Definitions
    I = 1
    T = 4
    dt = 0.1
    Nt = int(T / dt)
    theta = 0.5
    a = -1
    b = -2
    c = 3
    d = -1
    u_exact = lambda t: (1/3) * np.exp(-t) * \
        (3 * np.cos(np.sqrt(6) * t) - np.sqrt(6) * np.sin(np.sqrt(6) * t))
    v_exact = lambda t: (1/2) * np.exp(-t) * \
        (2 * np.cos(np.sqrt(6) * t) + np.sqrt(6) * np.sin(np.sqrt(6) * t))

    # Define mesh functions and points
    u = np.zeros(Nt + 1)
    v = np.zeros(Nt + 1)
    t = np.linspace(0, Nt * dt, Nt + 1)
    u[0] = I
    v[0] = I

    # Solve using difference equations
    # (Δt a θ - 1) uⁿ⁺¹ + b Δt θ vⁿ⁺¹ = (a Δt(θ - 1) - 1)uⁿ + b Δt(θ - 1)vⁿ
    # c Δt θ uⁿ⁺¹ + (d Δt θ - 1) vⁿ⁺¹ =  c Δt(θ - 1)uⁿ + (d Δt(θ - 1) - 1)vⁿ
    A = [[dt * a * theta - 1, b * dt * theta],
         [c * dt * theta, d * dt * theta - 1]]
    for n in range(Nt):
        B = [(a * dt * (theta - 1) - 1) * u[n] + b * dt * (theta - 1) * v[n],
             c * dt * (theta - 1) * u[n] + (d * dt * (theta - 1) - 1) * v[n]]
        u[n + 1], v[n + 1] = scipy.linalg.solve(A, B)
    print(STR_FMT.format('u', u))
    print(STR_FMT.format('v', v))

    # Compute error
    u_e = u_exact(t)
    e = u_e - u

    # Assert that max deviation is below tolerance
    tol = 1E-2
    diff = np.max(np.abs(u_e - u))
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}')


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
