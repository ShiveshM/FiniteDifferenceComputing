#! /usr/bin/env python3

r"""
Notes from Chapter 3 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = -a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)
               u'(t) = f(u, t)     t ∈ (0, T]     u(0) = I             (2)

"""

from copy import deepcopy
from functools import partial
from typing import List

import numpy as np
import sympy as sym
import scipy.linalg
import scipy.optimize
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from utils.solver import compute_rates
from utils.solver import solver_chap3 as solver


__all__ = ['generalisation', 'verification', 'convergence', 'systems',
           'generic_FO_ODE', 'bdf2', 'leapfrog', 'rk2', 'taylor_series',
           'adams_bashforth', 'rk4', 'compare_schemes']

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

    T = 6
    I = u_exact(0)
    dt_values = [0.1 * 2**(-i) for i in range(8)]
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


def generic_FO_ODE() -> None:
    """
    Generic first-order ODEs.

    Notes
    ----------
    We now turn the attention to general, nonlinear ODEs and systems of such
    ODEs. Our focus is on numerical methods that can be readily reused for
    discretisation of PDEs, and diffusion PDEs in particular.

    ODEs are commonly written in the generic form

                           u' = f(u, t),        u(0) = I

    where f(u, t) is some prescribed function. The unknown u may either be a
    scalar function of time t, or a vector values function of t in the case of
    a system of ODEs with m unknown components

                    u(t) = (u⁽⁰⁾(t), u⁽¹⁾(t), ..., u⁽ᵐ⁻¹⁾(t))

    In that case, the right-hand side is a vector-valued function with m
    components

            f(u, t) = (f⁽⁰⁾(u⁽⁰⁾(t), u⁽¹⁾(t), ..., u⁽ᵐ⁻¹⁾(t)),
                       f⁽¹⁾(u⁽⁰⁾(t), u⁽¹⁾(t), ..., u⁽ᵐ⁻¹⁾(t)),
                       ...
                       f⁽ᵐ⁻¹⁾(u⁽⁰⁾(t), u⁽¹⁾(t), ..., u⁽ᵐ⁻¹⁾(t)))

    Actually, any system of ODEs can be written in the form (2), but
    higher-order ODEs then need auxiliary unknown functions to enable
    conversion to a first-order system.

    The θ-rule scheme applied to u' = f(u, t) becomes

            (uⁿ⁺¹ - uⁿ) / Δt = θ f(uⁿ⁺¹, t{n+1}) + (1 - θ) f(uⁿ, tn})

    Bring the unknown uⁿ⁺¹ to the left-hand side and the known terms on the
    right-hand side gives

            uⁿ⁺¹ - Δt θ f(uⁿ⁺¹, t{n+1}) = uⁿ + Δt(1 - θ) f(uⁿ, tn)

    For a general f (not linear in u), this equation is nonlinear in the
    unknown uⁿ⁺¹ unless θ = 0. For a scalar ODE (m = 1), we have to solve a
    single nonlinear algebraic equation for uⁿ⁺¹, while for a system of ODEs,
    we get a system of coupled, nonlinear algebraic equations. Newton's method
    is a popular solution approach in both cases. Note that with the Forward
    Euler scheme (θ = 0), we do not have to deal with nonlinear equations,
    because in that case we have an explicit updating formula for uⁿ⁺¹. This is
    known as an explicit scheme. With θ ≠ 0, we have to solve (systems of)
    algebraic equations, and the scheme is said to be implicit.

    Here we demonstrate using the ODE u'(t) = u(t)² - 2, u(0) = 1. Then,

            uⁿ⁺¹ - Δt θ ((uⁿ⁺¹)² - 2) = uⁿ + Δt(1 - θ) ((uⁿ)² - 2)
          y = Δt θ (uⁿ⁺¹)² - uⁿ⁺¹ + Δt (uⁿ)² (1 - θ) + uⁿ - 2 Δt = 0

    The derivative of this can be used to solve for uⁿ⁺¹ using Newton's method

                        dy/duⁿ⁺¹ = 2 Δt θ uⁿ⁺¹ - 1

    """
    # Definitions
    I = 1
    T = 4
    dt = 0.1
    Nt = int(T / dt)
    theta = 0.5
    sqrt2 = np.sqrt(2)
    f_ut = lambda u, t: u(t)**2 - 2
    u_exact = lambda t: (2 + sqrt2 + (-2 + sqrt2) * np.exp(2 * sqrt2 * t)) / \
        (1 + sqrt2 + (-1 + sqrt2) * np.exp(2 * sqrt2 * t))

    # Define mesh functions and points
    u = np.zeros(Nt + 1)
    t = np.linspace(0, Nt * dt, Nt + 1)
    u[0] = I

    # Solve using difference equation
    # uⁿ⁺¹ - Δt θ f(uⁿ⁺¹, t{n+1}) = uⁿ + Δt(1 - θ) f(uⁿ, tn)
    # y = Δt θ (uⁿ⁺¹)² - uⁿ⁺¹ + Δt (uⁿ)² (1 - θ) + uⁿ - 2 Δt = 0
    y = lambda u1, u0: dt * theta * u1**2 - u1 + dt * u0**2 * (1 - theta) + \
        u0 - 2 * dt
    dy = lambda u1: 2 * dt * theta * u1 - 1
    for n in range(Nt):
        u[n + 1] = scipy.optimize.fsolve(
            partial(y, u0=u[n]), u[n], fprime=dy
        )[0]
    print(STR_FMT.format('u', u))

    # Compute error
    u_e = u_exact(t)
    e = u_e - u

    # Assert that max deviation is below tolerance
    tol = 1E-2
    diff = np.max(np.abs(u_e - u))
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}')


def bdf2() -> List[List[float]]:
    """
    An implicit 2-step backward scheme.

    Notes
    ----------
    The implicit backward method with 2 steps applies a three-level backward
    difference as approximation to u'(t).

                u'(t{n+1}) ≈ (3 uⁿ⁺¹ - 4 uⁿ + uⁿ⁻¹) / (2 Δt)

    which is an approximation of order Δt² to the first derivative. The
    resulting scheme for u' = f(u, t) reads

            uⁿ⁺¹ = (4/3) uⁿ - (1/3) uⁿ⁻¹ + (2/3) Δt f(uⁿ⁺¹, t{n+1})

    Higher-order versions of this scheme can be constructed by including more
    time levels. These schemes are known as Backward Differentiation Formulas
    (BDF), and this particular version is referred to as BDF2 and has second
    order convergence.

    Also note that this scheme is implicit and requires solution of nonlinear
    equations when f is nonlinear in u. The standard 1st-order Backward Euler
    method or Crank-Nicolson scheme can be used for the first step.

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u[0] = I

        # Solve for mesh function
        first_step = 'BackwardEuler'
        if first_step == 'CrankNicolson':
            # Crank-Nicolson 1. step
            u[1] = (1 - 0.5 * a * dt) / (1 + 0.5 * dt * a) * u[0]
        elif first_step == 'BackwardEuler':
            # Backward Euler 1. step
            u[1] = 1 / (1 + dt * a) * u[0]
        for n in range(1, Nt):
            u[n + 1] = (4 * u[n] - u[n - 1]) / (3 + 2 * dt * a)

        # Compute error
        u_e = u_exact(t)
        e = u_e - u
        E = np.sqrt(dt * np.sum(e**2))
        E_values.append(E)

        solutions.append(u)

    # Compute convergence rates
    r = compute_rates(dt_values, E_values)
    print(STR_FMT.format('r', f'{r}'))

    # Test final entry with expected convergence rate
    expected_rate = 2
    tol = 0.1
    diff = np.abs(expected_rate - r[-1])
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}!={r[-1]}')

    return solutions


def leapfrog() -> List[List[float]]:
    """
    Leapfrog schemes.

    Notes
    ----------
    The ordinary Leapfrog scheme
        The derivative of u at some point tn can be approximated by a central
        difference over two time steps,

                      u'(tn) ≈ (uⁿ⁺¹ - uⁿ⁻¹) / (2 Δt) = [D₂ₜ u]ⁿ

        which is an approximation of second order in Δt. The scheme can then be
        written as

                                 [D₂ₜ u = f(u, t)]ⁿ

        in the operator notation. Solving for uⁿ⁺¹ gives

                             uⁿ⁺¹ = uⁿ⁻¹ + 2 Δt f(uⁿ, tn)

        Observe that this is an explicit scheme, and that a nonlinear f (in u)
        is trivial to handle since it only involves the known uⁿ values. Some
        other scheme must be used as a starter to compute u¹, preferable the
        Forward Euler scheme since it is also explicit.

    The filtered Leapfrog scheme
        Unfortunately, the ordinary Leapfrog scheme may develop growing
        oscillations with time. A remedy for such undesired oscillation is to
        introduce a filtering technique.

        First, a standard Leapfrog step is taken, and then the previous uⁿ
        value is adjusted according to

                           uⁿ ← uⁿ + γ(uⁿ⁻¹ - 2 uⁿ + uⁿ⁺¹)

        The γ-terms will effectively damp oscillations in the solution,
        especially those with short wavelength (like point-to-point
        oscillations). A common choice of γ is 0.6 (a values used in the famous
        NCAR Climate Model).

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    gamma = 0.6
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_values = []
    E_filt_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u = np.zeros(Nt + 1)
        u_filt = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u[0] = I
        u_filt[0] = I

        # Solve for mesh function
        first_step = 'ForwardEuler'
        if first_step == 'CrankNicolson':
            # Crank-Nicolson 1. step
            u[1] = (1 - 0.5 * a * dt) / (1 + 0.5 * dt * a) * u[0]
        elif first_step == 'ForwardEuler':
            # Forward Euler 1. step
            u[1] = (1 - dt * a) * u[0]
        u_filt[1] = u[1]

        for n in range(1, Nt):
            # Ordinary
            u[n + 1] = u[n - 1] - 2 * dt * a * u[n]

            # Filtered
            u_filt[n + 1] = u_filt[n - 1] - 2 * dt * a * u_filt[n]
            u_filt[n] = u_filt[n] + gamma * (
                u_filt[n - 1] - 2 * u_filt[n] + u_filt[n + 1]
            )

        # Compute error
        u_e = u_exact(t)
        e = u_e - u
        E = np.sqrt(dt * np.sum(e**2))
        E_values.append(E)

        e_filt = u_e - u_filt
        E_filt = np.sqrt(dt * np.sum(e_filt**2))
        E_filt_values.append(E_filt)

        solutions.append(u_filt)


    # Compute convergence rates
    r = compute_rates(dt_values, E_values)
    print(STR_FMT.format('E_values', f'{E_values}'))
    print(STR_FMT.format('r', f'{r}'))
    r_filt = compute_rates(dt_values, E_filt_values)
    print(STR_FMT.format('E_filt_values', f'{E_filt_values}'))
    print(STR_FMT.format('r_filt', f'{r_filt}'))

    # Test final entry with expected convergence rate
    expected_rate = 2
    tol = 0.1
    diff = np.abs(expected_rate - r[-1])
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}!={r[-1]}')

    expected_rate = 1
    diff_filt = np.abs(expected_rate - r_filt[-1])
    if diff_filt > tol:
        raise AssertionError(
            f'Tolerance not reached, diff_filt = {diff_filt}!={r_filt[-1]}'
        )

    return solutions


def rk2() -> List[List[float]]:
    """
    The 2nd-order Runge-Kutta method.

    Notes
    ----------
    The two-step scheme

                            u* = uⁿ + Δt f(uⁿ, tn)
                    uⁿ⁺¹ = uⁿ + Δt ½ (f(uⁿ, tn) + f(u*, t{n+1}))

    essentially applied a Crank-Nicolson method to the ODE, but replaces the
    term f(uⁿ⁺¹, t{n+1}) by a prediction f(u*, t{n+1}) based on the Forward
    Euler step. This scheme is known as Huen's method, but it also a 2nd-order
    Runge-Kutta method. The scheme is explicit, and the error is expected to
    behave as Δt².

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u[0] = I

        # Solve for mesh function
        for n in range(Nt):
            u_star = u[n] - dt * a * u[n]
            u[n + 1] = u[n] - (1/2) * dt * a * (u[n] + u_star)

        # Compute error
        u_e = u_exact(t)
        e = u_e - u
        E = np.sqrt(dt * np.sum(e**2))
        E_values.append(E)

        solutions.append(u)

    # Compute convergence rates
    r = compute_rates(dt_values, E_values)
    print(STR_FMT.format('r', f'{r}'))

    # Test final entry with expected convergence rate
    expected_rate = 2
    tol = 0.1
    diff = np.abs(expected_rate - r[-1])
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}!={r[-1]}')

    return solutions


def taylor_series() -> List[List[float]]:
    """
    2nd-order Taylor-series method.

    Notes
    ----------
    One way to compute uⁿ⁺¹ given uⁿ is to use a Taylor polynomial. We may
    write up a polynomial of 2nd degree

                    uⁿ⁺¹ = uⁿ + u'(tn) Δt + ½ u''(tn) Δt²

    From the equation u' = f(u, t), it follows that the derivatives of u can be
    expressed in terms of f and its derivatives

                            u'(tn) = f(uⁿ, tn)
                    u''(tn) = ∂f/∂u(uⁿ, tn) u'(tn) + ∂f/∂t
                            = f(uⁿ, tn) ∂f/du(uⁿ, tn) + ∂f/dt

    resulting in the scheme

        uⁿ⁺¹ = uⁿ + f(uⁿ, tn) Δt + ½(f(uⁿ, tn) ∂f/du(uⁿ, tn) + ∂f/dt) Δt²

    More terms in the series could be included in the Taylor polynomial to
    obtain methods of higher order than 2.

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving f(t) = u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    f = lambda u, n: -a * u[n]
    dfdu = lambda u, n: -a
    dfdt = lambda u, n: 0
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u[0] = I

        # Solve for mesh function
        for n in range(Nt):
            u[n + 1] = u[n] + f(u, n) * dt + (1/2) * (
                f(u, n) * dfdu(u, t) + dfdt(u, t)
            ) * dt**2

        # Compute error
        u_e = u_exact(t)
        e = u_e - u
        E = np.sqrt(dt * np.sum(e**2))
        E_values.append(E)

        solutions.append(u)

    # Compute convergence rates
    r = compute_rates(dt_values, E_values)
    print(STR_FMT.format('r', f'{r}'))

    # Test final entry with expected convergence rate
    expected_rate = 2
    tol = 0.1
    diff = np.abs(expected_rate - r[-1])
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}!={r[-1]}')

    return solutions


def adams_bashforth() -> List[List[float]]:
    """
    The 2nd- and 3rd-order Adams-Bashforth schemes.

    Notes
    ----------
    The following method is known as the 2nd-order Adams-Bashforth scheme

                uⁿ⁺¹ = uⁿ + ½ Δt (3 f(uⁿ, tn) - f(uⁿ⁻¹, t{n-1}))

    This scheme is explicit and requires another one-step scheme to compute u¹
    (the Forward Euler of Huen's method, for instance). As the name implies,
    the error behaves as Δt².

    Another explicit scheme, involving four time levels, is the 3rd-order
    Adams-Bashforth scheme

  uⁿ⁺¹ = uⁿ + (1/12) Δt (23 f(uⁿ, tn) - 16 f(uⁿ⁻¹, t{n-1}) + 5 f(uⁿ⁻², t{n-2}))

    The numerical error is of order Δt³, and the scheme needs some method for
    computing u¹ and u².

    More general, higher-order Adams-Bashforth schemes (also called explicit
    Adams methods) compute uⁿ⁺¹ as a linear combination of f at k+1 previous
    time steps:

                      uⁿ⁺¹ = uⁿ + ∑ⱼ₌₀ᵏ βⱼ f(uⁿ⁻ʲ, t{n-j})

    where βⱼ are known coefficients.

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving f(t) = u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    f = lambda u, n: -a * u[n]
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_2_values = []
    E_3_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u_2 = np.zeros(Nt + 1)
        u_3 = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u_2[0] = I

        # Crank-Nicolson 1. and 2. step
        u_2[1] = (1 - (1/2) * a * dt) / (1 + (1/2) * dt * a) * u_2[0]
        u_2[2] = (1 - (1/2) * a * dt) / (1 + (1/2) * dt * a) * u_2[1]
        u_3[:3] = deepcopy(u_2[:3])

        # Solve for mesh function
        for n in range(2, Nt):
            u_2[n + 1] = u_2[n] + (1/2) * dt * (3 * f(u_2, n) - f(u_2, n - 1))
            u_3[n + 1] = u_3[n] + (1/12) * dt * (
                23 * f(u_3, n) - 16 * f(u_3, n - 1) + 5 * f(u_3, n - 2)
            )

        # Compute error
        u_e = u_exact(t)
        e = u_e - u_2
        E = np.sqrt(dt * np.sum(e**2))
        E_2_values.append(E)

        # Compute error
        e = u_e - u_3
        E = np.sqrt(dt * np.sum(e**2))
        E_3_values.append(E)

        solutions.append(u_3)

    # Compute convergence rates
    r_2 = compute_rates(dt_values, E_2_values)
    print(STR_FMT.format('r_2', f'{r_2}'))

    r_3 = compute_rates(dt_values, E_3_values)
    print(STR_FMT.format('r_3', f'{r_3}'))

    # Test final entry with expected convergence rate
    expected_rate = 2
    tol = 0.1
    diff = np.abs(expected_rate - r_2[-1])
    if diff > tol:
        raise AssertionError(
            f'Tolerance not reached, diff = {diff}!={r_2[-1]}'
        )

    # Test final entry with expected convergence rate
    expected_rate = 3
    tol = 0.1
    diff = np.abs(expected_rate - r_3[-1])
    if diff > tol:
        raise AssertionError(
            f'Tolerance not reached, diff = {diff}!={r_3[-1]}'
        )

    return solutions


def rk4() -> List[List[float]]:
    """
    The 4nd-order Runge-Kutta method.

    Notes
    ----------
    Perhaps the most widely used method to solve ODEs is the 4th-order
    Runge-Kutta method, often called RK4. Its derivation is a nice illustration
    of common numerical approximation strategies.

    The starting point is to integrate the ODE u' = f(u, t) from tn to t{n+1}

                u(t{n+1}) - u(tn) = ∫ₜₙᵗⁿ⁺¹ f(u(t), t) dt

    We want to compute u(t{n+1}) and regard u(tn) as known. The task is to find
    good approximations for the integral, since the integrand involves the
    unknown u between tn and t{n+1}.

    The integrand can be approximated by the famous Simpson's rule

            ∫ₜₙᵗⁿ⁺¹ f(u(t), t) dt ≈ Δt / 6 (fⁿ + 4 fⁿ⁺¹⸍² + fⁿ⁺¹)

    The problem is that we do not know fⁿ⁺¹⸍² and fⁿ⁺¹ as we know only uⁿ and
    hence fⁿ. The idea is to use various approximations for fⁿ⁺¹⸍² and fⁿ⁺¹
    based on well-known schemes for the ODE in the intervals [tn, t{n+½}] and
    [tn, t{n+1}]. we split the integral approximation into four terms

         ∫ₜₙᵗⁿ⁺¹ f(u(t), t) dt ≈ Δt / 6 (fⁿ + 2 𝒇ⁿ⁺¹⸍² + 2 𝕗ⁿ⁺¹⸍² + ḟⁿ⁺¹)

    where 𝒇ⁿ⁺¹⸍², 𝕗ⁿ⁺¹⸍², and ḟⁿ⁺¹ are approximations to fⁿ⁺¹⸍² and fⁿ⁺¹
    respectively, that can be based on already computed quantities. For 𝒇ⁿ⁺¹⸍²
    we can apply an approximation to uⁿ⁺¹⸍² using the Forward Euler method with
    step ½ Δt

                        𝒇ⁿ⁺¹⸍² = f(uⁿ + ½ Δt fⁿ, t{n+½})

    Since this gives us a prediction of fⁿ⁺¹⸍², we can for 𝕗ⁿ⁺¹⸍² try a
    Backward Euler method to approximate uⁿ⁺¹⸍²

                     𝕗ⁿ⁺¹⸍² = f(uⁿ + ½ Δt 𝒇ⁿ⁺¹⸍², t{n+½})

    With 𝕗ⁿ⁺¹⸍² as a hopefully good approximation to fⁿ⁺¹⸍², we can, for the
    final term ḟⁿ⁺¹, use a Crank-Nicolson method on [tn, t{n+1}] to approximate
    uⁿ⁺¹

                     ḟⁿ⁺¹⸍² = f(uⁿ + Δt 𝕗ⁿ⁺¹⸍², t{n+1})

    We have now used the Forward and Backward Euler methods as well as the
    Crank-Nicolson method in the context of Simpson's rule. The hope is that
    the combination of these methods yields an overall time-stepping scheme
    from tn to t{n+1} that is much more accurate that the 𝓞(Δt) and 𝓞(Δt²) of
    the individual steps. This in indeed true: the overall accuracy is 𝓞(Δt⁴)!

    To summarise, the 4th-order Runge-Kutta method becomes

               uⁿ⁺¹ = uⁿ + Δt / 6 (fⁿ + 2 𝒇ⁿ⁺¹⸍² + 2 𝕗ⁿ⁺¹⸍² + ḟⁿ⁺¹)

    where the quantities on the right-hand side are computed as defined above.
    Note that the scheme is fully explicit, so there is never any need to solve
    linear or nonlinear algebraic equations. However, the stability is
    conditional and depends on f. There is a whole range of implicit
    Runge-Kutta methods that are unconditionally stable, but require solution
    of algebraic equations involving f at each time step.

    Returns
    ----------
    solutions: Numerical solutions for 8 Δt values.

    """
    # Definitions
    # Solving f(t) = u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    f = lambda u, n: -a * u[n]
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]
    print(STR_FMT.format('dt_values', f'{dt_values}'))

    solutions = []
    E_values = []
    for dt in dt_values:
        # Define mesh functions and points
        Nt = int(T / dt)
        u = np.zeros(Nt + 1)
        t = np.linspace(0, Nt * dt, Nt + 1)
        u[0] = I

        # Solve for mesh function
        for n in range(Nt):
            f_fe = -a * (u[n] + (1/2) * dt * f(u, n))
            f_be = -a * (u[n] + (1/2) * dt * f_fe)
            f_cn = -a * (u[n] + dt * f_be)
            u[n + 1] = u[n] + (1/6) * dt * (
                f(u, n) + 2 * f_fe + 2 * f_be + f_cn
            )

        # Compute error
        u_e = u_exact(t)
        e = u_e - u
        E = np.sqrt(dt * np.sum(e**2))
        E_values.append(E)

        solutions.append(u)

    # Compute convergence rates
    r = compute_rates(dt_values, E_values)
    print(STR_FMT.format('r', f'{r}'))

    # Test final entry with expected convergence rate
    expected_rate = 4
    tol = 0.1
    diff = np.abs(expected_rate - r[-1])
    if diff > tol:
        raise AssertionError(f'Tolerance not reached, diff = {diff}!={r[-1]}')

    return solutions


def compare_schemes() -> None:
    """
    Compare schemes discussed in this chapter.
    """
    schemes = {'Filtered Leapfrog': leapfrog,
               '2nd-order Taylor-series': taylor_series,
               '3rd-order Adams-Bashforth': adams_bashforth,
               '2nd-order Runge-Kutta': rk2,
               '4th-order Runge-Kutta': rk4}

    # Definitions
    # Solving f(t) = u'(t) = -a * u(t)
    I = 1
    T = 4
    a = 3
    f = lambda u, n: -a * u[n]
    u_exact = lambda t: np.exp(-a * t)

    dt_values = [0.1 * 2**(-i) for i in range(8)]

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(
        r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$', y=0.95
    )

    axs = []
    gs = gridspec.GridSpec(2, 4)
    gs.update(hspace=0.3, wspace=0.3)
    for sc_idx, scheme in enumerate(schemes):
        # Get solution
        solutions = schemes[scheme]()

        for dt_idx, dt in enumerate(dt_values):
            Nt = int(T / dt)
            t = np.linspace(0, Nt * dt, Nt + 1)

            if sc_idx == 0:
                gssub = gs[dt_idx].subgridspec(
                    2, 1, height_ratios=(1, 1), hspace=0
                )
                ax1 = fig.add_subplot(gssub[0])
                ax2 = fig.add_subplot(gssub[1])
                axs.append([ax1, ax2])

                ax1.set_title('$\Delta t={:g}$'.format(dt))
                ax1.set_ylim(0, 1)
                ax1.set_xlim(0, T)
                ax1.set_ylabel('u')
                ax1.set_xticks([])
                ax1.set_yticks(ax1.get_yticks()[1:])

                ax2.grid(c='k', ls='--', alpha=0.3)
                ax2.set_yscale('log')
                ax2.set_xlim(0, T)
                ax2.set_xlabel('t')
                ax2.set_ylabel('log err')

                # Calculate exact solution
                t_e = np.linspace(0, T, 1001)
                u_e = u_exact(t_e)

                # Plot with black line
                ax1.plot(t_e, u_e, 'k-', label='exact')

            ax1, ax2 = axs[dt_idx]

            # Plot
            p = ax1.plot(t, solutions[dt_idx], ls='-', label=scheme)
            ax1.legend()

            # Calculate exact solution
            u_e = u_exact(t)
            err = np.abs(u_e - solutions[dt_idx])
            ax2.plot(t, err, ls='-', color=p[0].get_color())

    # Save figure
    fig.savefig(IMGDIR + f'comparison.png', bbox_inches='tight')


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
