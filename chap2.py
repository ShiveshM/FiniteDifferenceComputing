#! /usr/bin/env python3

r"""
Notes from Chapter 2 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
          u'(t) = -a u(t)     a > 0     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

from utils.solver import solver


__all__ = ['investigations', 'stability', 'visual_accuracy', 'amp_error',
           'global_error', 'integrated_error', 'truncation_error',
           'consistency_stability_convergence', 'model_errors', 'data_errors',
           'discretisation_errors', 'rounding_errors', 'exponential_growth',
           'explore_rounding_err']

IMGDIR = './img/chap2/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def investigations() -> None:
    """
    Experimental investigations of numerical instability.

    Notes
    ----------
    The characteristics shown in the produced figures can be summarised as
    follows:
    - The Backward Euler scheme gives a monotone solution in all cases, lying
      above the exact curve.
    - The Crank-Nicolson scheme gives the most accurate results, but for
      Δt = 1.25 the solution oscillates.
    - The Forward Euler scheme gives a growing, oscillating solution for
      Δt = 1.25; a decaying, oscillating solution for Δt = 0.75; strange
      solution uⁿ = 0 for n ≥ 1 when Δt = 0.5; and a solution seemingly as
      accurate as the one by Backward Euler scheme for Δt = 0.1, but the curve
      lies below the exact solution.

    Key questions
    - Under what circumstances, i.e., values of the input data I, a, and in Δt
      will Forward Euler and Crank-Nicolson schemes results in undesired
      oscillatory solutions?
    - How does Δt impact the error in the numerical solution?

    """
    I = 1
    a = 2
    T = 6

    th_dict = {0: ('Forward Euler', 'fe'), 1: ('Backward Euler', 'be'),
               0.5: ('Crank-Nicolson', 'cn')}

    for th in th_dict:
        fig, axs = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3}
        )
        for dt_idx, dt in enumerate((1.25, 0.75, 0.5, 0.1)):
            ax = axs.flat[dt_idx]
            ax.set_title('{}, dt={:g}'.format(th_dict[th][0], dt))
            ax.set_ylim(0, 1)
            ax.set_xlim(0, T)
            ax.set_xlabel('t')
            ax.set_ylabel('u')

            # Solve
            u, t = solver(I=I, a=a, T=T, dt=dt, theta=th)

            # Calculate exact solution
            u_exact = lambda t, I, a: I * np.exp(-a * t)
            t_e = np.linspace(0, T, 1001)
            u_e = u_exact(t_e, I, a)

            # Plot with red dashes w/ circles
            ax.plot(t, u, 'r--o', label='numerical')

            # Plot with blue line
            ax.plot(t_e, u_e, 'b-', label='exact')
            ax.legend()

        # Save figure
        fig.savefig(IMGDIR + f'{th_dict[th][1]}.png', bbox_inches='tight')


def stability() -> None:
    """
    Detailed experiments of instability and oscillations.

    Notes
    ----------
    We loop over values of I, a and Δt in our chosen model problem. For each
    experiments, we flag the solution as oscillatory if

                                   uⁿ > uⁿ⁻¹

    for some value of n, since we expect uⁿ to decay with n. Doing some
    experiments varying I, a and Δt, reveals that oscillations are independent
    of I, but they do depend on a and Δt. Based on this, we introduce a
    two-dimensional function B(a, Δt) which is 1 if oscillations occur and 0
    otherwise. We visualise B as a contour plot (lines for which B = const).
    The contour B = 0.5 corresponds to the borderline between oscillatory
    regions with B = 1 and monotone regions with B = 0 in the a, Δt plane.

    Starting with u⁰ = I, the simple recursion below can be applied repeatedly
    n times to obtain uⁿ,

                 uⁿ = I Aⁿ,     A = (1 - (1 - θ) a Δt) / (1 + θ a Δt)

    The factor A is called the amplification factor. Difference equations where
    all terms are linear in uⁿ⁺¹, uⁿ, and maybe uⁿ⁻¹, uⁿ⁻², etc., are called
    homogeneous, linear difference equations, and their solutions are generally
    of the form uⁿ = Aⁿ.

    This formula can explain everything we see in the figures produced here,
    but it also gives us a more general insight into the accuracy and stability
    properties of the three schemes.

    Since uⁿ is a factor A raise to an integer power n, we realise that A < 0
    will imply uⁿ < 0 for odd n and uⁿ > 0 for even n. That is, the solution
    oscillates between the mesh points. To avoid oscillations we require A > 0

                              (1 - θ) a Δt < 1

    Since A > 0 is a requirement for having a numerical solution with the basic
    property (monotonicity) as the exact solution, we may say that A > 0 is a
    stability criterion. Expressed in terms of Δt,

                             Δt < 1 / ((1 - θ) a)

    The Backward Euler scheme is always stable since A < 0 is impossible for
    θ = 1, while non-oscillating solutions for Forward Euler and Crank-Nicolson
    demand that Δt ≤ 1/a and Δt ≤ 2/a, respectively. The relation between Δt
    and a looks reasonable: a larger a means faster decay and hence a need for
    smaller time steps.

    For a decay process, we must also have |A| ≤ 1, which is fulfilled for all
    Δt if θ ≥ 1/2. Arbitrarily large values of u can be generated when |A| > 1
    and n is large enough - which is totally irrelevant to an ODE modeling a
    decay process! To avoid this situation, we must demand |A| ≤ 1 also for
    θ < 1/2, which implies

                              Δt ≤ 2 / (1 - 2 θ) a

    For example, Δt must not exceed 2/a when computing with the FE scheme.

    In summary,
    1) The Forward Euler method is a conditionally stable scheme because is
       requires Δt < 2/a for avoiding growing solutions and Δt < 1/a for
       avoiding oscillatory solutions.
    2) The Crank-Nicolsonis unconditionally stable with respect to growing
       solutions, while it is conditionally stable with the criterion Δt < 2/a
       for avoiding oscillatory solutions.
    3) The Backward Euler method is unconditionally stable with respect to
       growing and oscillatory solutions - any Δt will work.

    Much literature on ODEs speaks about L-stable and A-stable methods. In our
    case, A-stable methods ensures non-growing solutions, while L-stable
    methods also avoids oscillatory solutions.

    """
    I = 1
    a = np.linspace(0.01, 4, 22)
    dt = np.linspace(0.01, 2.5, 22)
    T = 6

    th_dict = {0: ('Forward Euler', 'fe'), 1: ('Backward Euler', 'be'),
               0.5: ('Crank-Nicolson', 'cn')}

    for th in th_dict:
        # Initialise B data structure
        B = np.zeros((len(a), len(dt)))

        # Solve for each a, Δt
        for a_idx, a_val in enumerate(a):
            for dt_idx, dt_val in enumerate(dt):
                u, _ = solver(I, a_val, T, dt_val, th)

                # Does u have the right monotone decay properties?
                is_monotone = True
                for n in range(1, len(u)):
                    if u[n] > u[n - 1]:
                        is_monotone = False
                        break
                B[a_idx, dt_idx] = 1. if is_monotone else 0.

        # Meshgrid
        a_grid, dt_grid = np.meshgrid(a, dt, indexing='ij')

        # Plot
        fig, ax = plt.subplots()
        ax.set_title('{} oscillatory region'.format(th_dict[th][0]))
        ax.set_xlabel('a')
        ax.set_ylabel('dt')
        ax.set_xlim(0, np.max(a))
        ax.set_ylim(0, np.max(dt))
        ax.grid(c='k', ls='--', alpha=0.3)
        B = B.reshape(len(a), len(dt))
        ax.contourf(a_grid, dt_grid, B, levels=[0., 0.1], hatches='XX',
                    colors='grey', alpha=0.3, lines='k')
        fig.savefig(IMGDIR + f'{th_dict[th][1]}_osc.png', bbox_inches='tight')


def visual_accuracy() -> None:
    """
    Visually examine how large numerical errors are.

    Notes
    ----------
    The exact solution reads u(t) = I e⁻ᵃᵗ, which can be rewritten as

                 uₑ(tn) = I exp{-a n Δt} = I (exp{-a n Δt})ⁿ

    From this, we see the exact amplification factor is

                              Aₑ = exp{-a n Δt}

    We saw that the exact and numerical amplification factors depend on a and
    Δt through the dimensionless product a Δt, which we will denote as
    p = a Δt, and view A and Aₑ as functions of p. It is common that the
    numerical performance of methods for solving ODEs and PDEs is governed by
    dimensionless parameters that combine mesh sizes with physical parameters.

    For any finite difference approximation, take the Forward Euler difference
    as an example, and any specific function, take u = exp{-a t}, we may
    introduce an error fraction

     E = [Dₜ⁺ u]ⁿ / u'(tn) = exp{-a(tn + Δt)) - exp{-a tn) / -a exp{-a tn} Δt
                           = (1 / (a Δt)) (1 - exp{-a Δt})

    and view E as a function of Δt. We expect that as Δt → 0, E → 1, while E
    may deviate significantly from unity for large Δt. How the error depends on
    Δt is best visualised in a graph with a logarithmic axis for Δt.

    """
    def calc_amp(p: float, theta: float):
        """
        Calculate the amplification factor for u'=-a*u.

        Parameters
        ----------
        p : a Δt
        theta : theta=0 corresponds to FE, theta=0.5 to CN and theta=1 to BE.

        Returns
        ----------
        amp : Amplification factor.

        """
        return (1 - (1 - theta) * p) / (1 + theta * p)

    th_dict = {0: ('Forward Euler', 'fe', 'r-s'),
               1: ('Backward Euler', 'be', 'g-v'),
               0.5: ('Crank-Nicolson', 'cn', 'b-^')}

    fig, ax = plt.subplots()
    ax.set_title('Amplification factors')
    ax.set_xlabel('$p=a\Delta t$')
    ax.set_ylabel('Amplification factor')
    ax.set_xlim(0, 3)
    ax.set_ylim(-2, 1)
    ax.grid(c='k', ls='--', alpha=0.3)

    # Mesh grid for p = a Δt
    p = np.linspace(0, 3, 20)

    for th in th_dict:
        # Calculate amplification factor
        amp = calc_amp(p, th)

        # Plot
        ax.plot(p, amp, th_dict[th][2], label=th_dict[th][0])

    # Exact solution
    amp_exact = np.exp(-p)
    ax.plot(p, amp_exact, 'k-o', label='exact')

    ax.legend(loc='lower left')
    fig.savefig(IMGDIR + 'amplification_factors.png', bbox_inches='tight')

    # Define difference approximations
    D_f = lambda u, dt, t: (u(t + dt) - u(t)) / dt
    D_b = lambda u, dt, t: (u(t) - u(t - dt)) / dt
    D_c = lambda u, dt, t: (u(t + dt) - u(t - dt)) / (2 * dt)

    # Define symbols
    a, t, dt, p = sym.symbols('a t dt p')

    # Define exact solution
    u = lambda t: sym.exp(-a * t)
    u_sym = sym.exp(-a * t)
    dudt = sym.diff(u_sym, t)

    # Setup figure
    fig, ax = plt.subplots()
    ax.set_title(r'$\frac{du(t)}{dt}=-a\cdot u(t)$')
    ax.set_xlabel('$p=a\Delta t$')
    ax.set_ylabel('E, error fraction')
    ax.set_xlim(1E-6, 1)
    ax.set_ylim(0.8, 1.2)
    ax.set_xscale('log')

    for name, op in zip(['FE', 'BE', 'CN'], [D_f, D_b, D_c]):
        E = op(u, dt, t) / dudt

        # Set p = a * dt
        E = sym.simplify(sym.expand(E).subs(a * dt, p))
        print(STR_FMT.format(f'{name} E = ', f'{E} ≈ {E.series(p, 0, 3)}'))

        # Convert to lambda expr
        f_E = sym.lambdify([p], E, modules='numpy')

        # Calculate E as a function of p
        p_values = np.logspace(-6, 1, 101)
        E_v = f_E(p_values)

        # Plot
        ax.plot(p_values, E_v, label=f'${name}: {sym.latex(E)}$')

    ax.legend()
    fig.savefig(IMGDIR + 'visual_accuracy.png', bbox_inches='tight')


def amp_error() -> None:
    """
    Series expansion of amplification factors.

    Notes
    ----------
    Next we would like to establish a formula for approximating the errors.
    Calculating the Taylor series for Aₑ can easily be done by hand, but the
    three versions of A for θ = 0, 1, ½ lead to more cumbersome calculations.

    Nowadays, analytical computations can benefit greatly by symbolic computer
    algebra software. The package SymPy represents a powerful computer algebra
    system.

    From the expressions below, we see that A - Aₑ ~ O(p²) for the Forward and
    Backward Euler schemes, while A - Aₑ ~ O(p³) for the Crank-Nicolson scheme.
    Since a is a given parameter and Δt is what we can vary, the error
    expressions are usually written in terms of Δt

                    A - Aₑ = O(Δt²)   Forward and Backward Euler
                    A - Aₑ = O(Δt³)   Crank-Nicolson

    We say that the Crank-Nicolson scheme has an error in the amplification
    factor of order Δt³, while the other two are of order Δt². That is, as we
    reduce Δt to obtain more accurate results, the Crank-Nicolson scheme
    reduces the error more efficiently than the other schemes.

    An alternative comparison of the schemes is provided by looking at the
    ratio A/Aₑ, or the error 1 - A/Aₑ in this ratio. The leading order terms
    have the same powers as in the analysis of A - Aₑ.

    """
    # Create p as a mathematical symbol with name 'p'
    p = sym.Symbol('p')

    # Create a mathematical expression with p
    A_e = sym.exp(-p)

    # Find the first 6 terms of the Taylor series of Aₑ
    print(STR_FMT.format('A_e.series(p, 0, 6)', A_e.series(p, 0, 6)))

    # Create analytical function for A, dependent on theta
    theta = sym.Symbol('theta')
    A = (1 - (1 - theta) * p) / (1 + theta * p)

    # To work with the Forward Euler scheme, we can substitue theta = 0
    A_fe = A.subs(theta, 0)
    print(STR_FMT.format('Forward Euler Amplification factor', A_fe))

    # Similar for Backward Euler and Crank-Nicolson schemes
    A_be = A.subs(theta, 1)
    half = sym.Rational(1, 2)
    A_cn = A.subs(theta, half)
    print(STR_FMT.format('Backward Euler Amplification factor', A_be))
    print(STR_FMT.format('Crank-Nicolson Amplification factor', A_cn))

    # Now we can compare the Taylor series expansions of the amplification
    fe_err = A_e.series(p, 0, 4) - A_fe.series(p, 0, 4)
    be_err = A_e.series(p, 0, 4) - A_be.series(p, 0, 4)
    cn_err = A_e.series(p, 0, 4) - A_cn.series(p, 0, 4)
    print(STR_FMT.format('Forward Euler Amplification error', fe_err))
    print(STR_FMT.format('Backward Euler Amplification error', be_err))
    print(STR_FMT.format('Crank-Nicolson Amplification error', cn_err))

    # Alternatively, use the ratio A/Aₑ, or the error 1 - A/Aₑ in this ratio
    fe_ratio = 1 - (A_fe / A_e).series(p, 0, 4)
    be_ratio = 1 - (A_be / A_e).series(p, 0, 4)
    cn_ratio = 1 - (A_cn / A_e).series(p, 0, 4)
    print(STR_FMT.format('Forward Euler Ratio error', fe_ratio))
    print(STR_FMT.format('Backward Euler Ratio error', be_ratio))
    print(STR_FMT.format('Crank-Nicolson Ratio error', cn_ratio))


def global_error() -> None:
    """
    The global error at a point.

    Notes
    ----------
    The error in the amplification factor reflects the error when progressing
    from time level tn to t{n-1} only. That is, we disregard the error already
    present in the solution at t{n-1}. The real error at a point, however,
    depends on the error development over all previous time steps.

                            eⁿ = uⁿ - uₑ(tn)

    This is known as the global error. We may look at uⁿ for some n and Taylor
    expand as functions of p to get a simple expression for the global error.

    Here we see that the global error for the Forward Euler and Backward Euler
    schemes is O(Δt) and for the Crank-Nicolson scheme it is O(Δt²).

    When the global error eⁿ → 0 as Δt → 0, we say the scheme is convergent. It
    means that the numerical solution approaches the exact solution as the mesh
    is refined, and this is a much desired property of a numerical method.

    """
    # Define amplification factor
    p, theta = sym.symbols('p theta')
    A = (1 - (1 - theta) * p) / (1 + theta * p)

    # Define mesh function along with the exact function
    n = sym.Symbol('n')
    u_e = sym.exp(-p * n)
    u_n = A**n

    # Define FE, BE, CN
    fe = u_n.subs(theta, 0)
    be = u_n.subs(theta, 1)
    half = sym.Rational(1, 2)
    cn = u_n.subs(theta, half)

    print(STR_FMT.format('Forward Euler', fe))
    print(STR_FMT.format('Backward Euler', be))
    print(STR_FMT.format('Crank-Nicolson', cn))

    # Now we can compute the global error
    fe_err = u_e.series(p, 0, 4) - fe.series(p, 0, 4)
    be_err = u_e.series(p, 0, 4) - be.series(p, 0, 4)
    cn_err = u_e.series(p, 0, 4) - cn.series(p, 0, 4)

    # Substitute back in a, t, and Δt and extract leading dt term
    a, t, dt = sym.symbols('a t dt')
    fe_err = fe_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)
    be_err = be_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)
    cn_err = cn_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)

    print(STR_FMT.format('Forward Euler global error', fe_err))
    print(STR_FMT.format('Backward Euler global error', be_err))
    print(STR_FMT.format('Crank-Nicolson global error', cn_err))


def integrated_error() -> None:
    """
    Study the norm of the numerical error by performing symbolic integration.

    Notes
    ----------
    The L² norm of the error can be computed by treating eⁿ as a function of t
    in SymPy and performing symbolic integration.

    In summary, both the global point-wise errors and their time-integrated
    versions show that
    - The Crank-Nicolson scheme is of second order in Δt.
    - The Forward Euler and Backward Euler schemes are of first order in Δt.

    """
    # Define amplification factor
    p, theta = sym.symbols('p theta')
    A = (1 - (1 - theta) * p) / (1 + theta * p)

    # Define mesh function along with the exact function
    n = sym.Symbol('n')
    u_e = sym.exp(-p * n)
    u_n = A**n

    # Define FE, BE, CN
    fe = u_n.subs(theta, 0)
    be = u_n.subs(theta, 1)
    half = sym.Rational(1, 2)
    cn = u_n.subs(theta, half)

    # Now we can compute the global error
    fe_err = u_e.series(p, 0, 4) - fe.series(p, 0, 4)
    be_err = u_e.series(p, 0, 4) - be.series(p, 0, 4)
    cn_err = u_e.series(p, 0, 4) - cn.series(p, 0, 4)

    # Substitute back in a, t, and Δt and extract leading dt term
    a, t, dt = sym.symbols('a t dt')
    fe_err = fe_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)
    be_err = be_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)
    cn_err = cn_err.subs('n', 't/dt').subs('p', 'a*dt').as_leading_term(dt)

    # Compute the L² norm of the error
    T = sym.Symbol('T')
    fe_err_L2 = sym.sqrt(sym.integrate(fe_err**2, (t, 0, T)))
    be_err_L2 = sym.sqrt(sym.integrate(be_err**2, (t, 0, T)))
    cn_err_L2 = sym.sqrt(sym.integrate(cn_err**2, (t, 0, T)))

    print(STR_FMT.format('Forward Euler global L2 error', fe_err_L2))
    print(STR_FMT.format('Backward Euler global L2 error', be_err_L2))
    print(STR_FMT.format('Crank-Nicolson global L2 error', cn_err_L2))


def truncation_error() -> None:
    """
    Truncation error.

    Notes
    ----------
    The truncation error is defined as the error in the difference equation
    that arises when inserting the exact solution. Contrary to many other error
    measures, e.g. the true global error eⁿ = uⁿ - uₑ(tn), the truncation error
    is a quantity that is easily computable.

    For the Forward Euler scheme, start with the difference equation

                               [Dₜ⁺ u = -a u]ⁿ

    The idea is to see how well the exact solution uₑ(t) fulfills this
    equation, since uₑ(t) in general will not obey the discrete equation. This
    error is called a residual, denoted by Rⁿ

                  Rⁿ = (uₑ(t{n+1}) - uₑ(tn)) / Δt + a uₑ(tn)

    The residual is defined at each mesh point and is therefore a mesh function
    with subscript n.

    The interesting feature of Rⁿ is to see how it depends on the
    discretisation parameter Δt, which we will compute using a Taylor expansion
    of uₑ around the point where the difference equation is supposed to hold,
    where t = tn. We have

            uₑ(t{n+1}) = uₑ(tn) + u'ₑ(tn) Δt + ½ u''ₑ(tn) (Δt)² + ...

    Therefore

                  Rⁿ = u'ₑ(tn) + ½ u''ₑ(tn) Δt + ... + a uₑ(tn)

                          Rⁿ = ½ u''ₑ(tn) Δt + O(Δt²)

    This Rⁿ is the truncation error, which for the Forward Euler is seen to be
    of first order in Δt.

    The Backward Euler schemes leads to

                             Rⁿ ≈ -½ u''ₑ(tn) Δt

    while the Crank-Nicolson scheme gives

                           Rⁿ ≈ (1/24) u'''ₑ(tn) Δt²

    The above expressions point out that the Forward and Backward Euler schemes
    are of first order, while Crank-Nicolson is of second order.

    """
    # Define the exact function
    a, t = sym.symbols('a t')
    u_e = sym.exp(-a * t)

    # Calculate truncation errors
    dt = sym.Symbol('dt')
    half = sym.Rational(1, 2)
    u_e_exp = u_e.subs('t', 't+dt').series(dt, 0, 4)
    Rn_fe = (u_e_exp - u_e) / dt + a * u_e
    Rn_be = (u_e - u_e.subs('t', 't-dt').series(dt, 0, 4)) / dt + a * u_e
    Rn_cn = (u_e_exp - u_e) / dt + a * half * (u_e + u_e_exp)

    # Get the leading dt term
    Rn_fe = Rn_fe.as_leading_term(dt)
    Rn_be = Rn_be.as_leading_term(dt)
    Rn_cn = Rn_cn.as_leading_term(dt)

    print(STR_FMT.format('Forward Euler truncation error', Rn_fe))
    print(STR_FMT.format('Backward Euler truncation error', Rn_be))
    print(STR_FMT.format('Crank-Nicolson truncation error', Rn_cn))


def consistency_stability_convergence() -> None:
    """
    Notes on consistency, stability, and convergence.

    Notes
    ----------
    Consistency
        Consistency means that the error in the difference equation, measured
        through the truncation error, goes to zero as Δt → 0. Since the
        truncation error tells how well the exact solution fulfills the
        difference equation, and the exact solution fulfills the differential
        equation, consistency ensures that the difference equation approaches
        the differential equation in the limit. Lack of consistency implies we
        actually solve some other differential equation in the limit Δt → 0
        that we aim at.

    Stability
        Stability means that the numerical solution exhibits the same
        qualitative properties as the exact solution. In the exponential decay
        model, the exact solution is monotone and decaying. An increasing
        numerical solution is not in accordance with the decaying nature of the
        exact solution and hence unstable. We can also say that an oscillating
        numerical solution lacks the property of monotonicity of the exact
        solution and is also unstable.

    Convergence
        Convergence implies that the global (true) error mesh function
        eⁿ = uⁿ - uₑ(tn) → 0 as Δt → 0. This is really what we want: the
        numerical solution gets as close to the exact solution as we request by
        having a sufficiently fine mesh. Convergence is hard to establish
        theoretically, except in quite simple problems like the present one. A
        major breakthrough in the understanding of numerical methods for
        differential equations came in 1956 when Lax and Richtmeyer established
        equivalence between convergence on one hand and consistency and
        stability on the other (the Lax equivalence theorem [1]). In practice
        it meant that one can first establish that a method is stable and
        consistent, and then it is automatically convergent. The result holds
        for linear problems only, and in the world of nonlinear differential
        equations, the relations between consistency, stability, and
        convergence are much more complicated.

    We have seen in the previous analysis that the Forward Euler, Backward
    Euler, and Crank-Nicolson schemes are convergent (eⁿ → 0), that they are
    consistent (Rⁿ → 0), and that they are stable under certain conditions on
    the size of Δt.

    [1]_ http://en.wikipedia.org/wiki/Lax_equivalence_theorem

    """
    pass


def model_errors() -> None:
    """
    Model errors.

    Notes
    ----------
    So far we have been concerned with one type of error, namely the
    discretisation error committed by replacing the differential equation
    problem by a recursive set of difference equations. There are, however,
    other types of errors that must be considered too. We classify into four
    groups

    1) model errors: how wrong is the ODE model?
    2) data errors: how wrong are the input parameters?
    3) discretisation errors: how wrong is the numerical method?
    4) rounding errors: how wrong is the computer arithmetics?

    Any mathematical model like u' = -a u, u(0) = I, is just an approximate
    description of a real-world phenomenon. How good this approximation is can
    be determined by comparing physical experiments with what the model
    predicts. This is the topic of validation. One difficulty with validation
    is that we need to estimate the parameters in the model, and this brings in
    data errors. Quantifying data errors is challenging, and a frequently used
    method is to tune the parameters in the model to make model predictions as
    close as possible to the experiments. Another difficulty is that the
    response in experiments also contains errors due to measurement techniques.

    Let us try and quantify model errors in a very simple example. Suppose a
    more accurate model has a as a function of time rather than a constant.
    Here we take a(t) as a simple linear function: (a + p t). The solution of

                      u' = -(a + p t) u,        u(0) = I

            is            u(t) = I exp{-t(a + ½ p t)}

    We can use SymPy to solve this instead and then make plots of the two
    models and the error for some values of p.

    """
    u = sym.Function('u')
    t, a, p, I = sym.symbols('t, a, p I', real=True)

    def ode(u, t, a, p):
        """Define ODE: u' = (a + p * t) * u and return residual."""
        return sym.diff(u, t) + (a + p * t) * u

    # Solve equation
    eq = ode(u(t), t, a, p)
    s = sym.dsolve(eq)
    print(STR_FMT.format('eq', eq))
    print(STR_FMT.format('s', s))

    # Grab the right hand side of the equality object
    u = s.rhs
    print(STR_FMT.format('u', u))

    # Substitute C1 to a defined symbol
    C1 = sym.Symbol('C1', real=True)
    u = u.subs('C1', C1)

    # Solve C1 for the initial condition
    eq = u.subs(t, 0) - I
    s = sym.solve(eq, C1)
    u = u.subs(C1, s[0])
    print(STR_FMT.format('s', s))
    print(STR_FMT.format('u', u))
    print(STR_FMT.format('sym.latex(u)', sym.latex(u)))

    # Consistency check: u must fulfill ODE and initial condition
    print('ODE is fulfilled', sym.simplify(ode(u, t, a, p)))
    print('u(0) - I', sym.simplify(u.subs(t, 0) - I))

    # Convert u expression to Python numerical function
    u_func = sym.lambdify([t, I, a, p], u, modules='numpy')
    help(u_func)

    # Define values to plot for
    p_values = [0.01, 0.1, 1]
    a = 1
    I = 1
    t = np.linspace(0, 4, 101)
    u = I * np.exp(-a * t)

    # Plotting
    fig1, axs1 = plt.subplots(
        ncols=len(p_values), figsize=(16, 5), gridspec_kw={'wspace': 0.3}
    )
    fig2, axs2 = plt.subplots()
    axs2.set_xlabel('t')
    axs2.set_ylabel('u_true(t) - u(t)')
    axs2.set_xlim(0, np.max(t))
    axs2.set_ylim(-0.16, 0.02)

    for idx, p in enumerate(p_values):
        # Calculate u(t)
        u_true = u_func(t, I, a, p)
        discrep = u_true - u

        # Plotting
        ax = axs1[idx]
        ax.set_title(f'p = {p:.2f}')
        ax.set_xlabel('t')
        ax.set_ylabel('u(t)')
        ax.set_xlim(0, np.max(t))
        ax.set_ylim(0, np.max(u))
        ax.plot(t, u, 'r-', label='model')
        ax.plot(t, u_true, 'b--', label='true model')
        ax.legend()

        axs2.plot(t, discrep, label=f'p = {p:.2f}')
    axs2.legend()

    # Save figures
    fig1.savefig(IMGDIR + f'model_error1.png', bbox_inches='tight')
    fig2.savefig(IMGDIR + f'model_error2.png', bbox_inches='tight')


def data_errors() -> None:
    """
    Data errors.

    Notes
    ----------
    By "data" we mean all input parameters to our model, in our case I and a.
    The values may contain errors, or at least uncertainty. Ideally, we may
    have samples of I and a and from these we can fit probability
    distributions. Assume that I turns out to be normally distributed with mean
    1 and std 0.2, while a is uni formally distributed in the interval
    [0.5, 1.5].

    How will the uncertainty in I and a propagate through the model
    u = I exp{-a t}? The answer can be easily answered using Monte Carlo
    simulation. For each combination of I and a sample, we compute the
    corresponding u value for selected values of t. Afterwards, we can for each
    selected t values, make a histogram of all the computed u values to see
    what the distribution of u values look like.

    u(t; I, a) becomes a stochastic variable for each t, and I and a are
    stochastic variables. The mean of the stochastic u(t; I, a) is not equal to
    u with mean values of the input dtaa, u(t; I=a=1), unless u is linear in I
    and a.

    Estimating statistical uncertainty in the input data and investigating how
    the uncertainty propagates to the uncertainty in the response of a
    differential equation model are key topics in the scientific field called
    uncertainty quantification, simply known as UQ. Monte Carlo simulation is a
    general and widely used tool to solve associated statistical problem. The
    accuracy of the Monte Carlo results increases with increasing number of
    samples N, typically the error behaves as N⁻¹⸍².

    """
    # Define number of trials and priors
    N = 10000
    I_values = np.random.normal(1, 0.2, N)
    a_values = np.random.uniform(0.5, 1.5, N)

    # Compute corresponding u values for some t values
    t_values = [0, 1, 3]
    u_values = [0] * len(t_values)
    u_mean = [0] * len(t_values)
    u_std = [0] * len(t_values)

    # Setup figure
    fig, axs = plt.subplots(
        ncols=len(t_values), figsize=(16, 5), gridspec_kw={'wspace': 0.2}
    )

    for idx, t in enumerate(t_values):
        # Setup axis
        ax = axs[idx]
        ax.set_title(f't = {t}')
        ax.set_xlabel('u')
        ax.set_ylabel('Entries (a.u.)')
        ax.set_xlim(0, 1.8)

        # Compute u samples according to I and a samples
        u_values[idx] = [I * np.exp(-a * t)
                         for I, a in zip(I_values, a_values)]
        u_mean[idx] = np.mean(u_values[idx])
        u_std[idx] = np.std(u_values[idx])

        # Plot
        ax.hist(
            u_values[idx], bins=30, range=(0, 1.8), density=True,
            facecolor='green', lw=0.7, edgecolor='k'
        )

    fig.savefig(IMGDIR + 'data_errors.png', bbox_inches='tight')


def discretisation_errors() -> None:
    """
    Discretisation errors.

    Notes
    ----------
    The errors implied by solving the differential equation problem by the
    θ-rule has been thoroughly analysed in the previous methods. Here are some
    plots of the error versus time for the Forward Euler, Backward Euler, and
    Crank-Nicolson schemes for decreasing values of Δt. Since the difference in
    magnitude between the errors in the CN scheme versus the FE and BN schemes
    grows significantly as Δt is reduced, the logarithm of the absolute values
    of the numerical error is plotted as a mesh function.

    """
    I = 1
    a = 1
    T = 4

    # Schemes
    th_dict = {0: ('Forward Euler', 'fe', 'r-s'),
               1: ('Backward Euler', 'be', 'g-v'),
               0.5: ('Crank-Nicolson', 'cn', 'b-^')}

    # dt values
    dt_values = [0.8, 0.4, 0.1, 0.01]

    # Setup figure
    fig, axs = plt.subplots(
        2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.3}
    )

    for dt_idx, dt in enumerate(dt_values):
        # Setup axes
        ax = axs.flat[dt_idx]
        ax.set_title(f'$\Delta t={dt}$')
        ax.set_xlim(0, T)
        ax.set_xlabel('t')
        ax.set_ylabel('log(abs(global error))')

        for th in th_dict:
            u, t = solver(I, a, T, dt, th)
            u_e = I * np.exp(-a * t)
            error = u_e - u

            # Exclude fist error entry as it is 0
            ax.plot(t[1:], np.log(np.abs(error[1:])), label=th_dict[th][1])

        ax.legend(loc='upper right')

    fig.savefig(IMGDIR + 'discretisation_errors.png', bbox_inches='tight')


def rounding_errors() -> None:
    """
    Rounding errors.

    Notes
    ----------
    Real numbers on a computer are represented by floating point numbers, which
    means that just a finite number of digits are stored and used. Therefore,
    the floating-point number is an approximation to the underlying real
    number. When doing arithmetics with floating-point numbers, there will be
    small approximation errors, called round-off errors or rounding errors,
    that may or may not accumulate in comprehensive computations.

    The typical level of rounding error from an arithmetic operation with the
    widely used 64 bit floating-point number is O(1E-16). The big question is
    if errors at this level accumulate in a given numerical algorithm.

    To investigate this, we can use the Python `Decimal` object in the
    `decimal` module that allows us to use as many digits in floating point
    numbers as we like. Here we take 1000 digits as the true answer.

    When computing with numbers around unity in size and doing Nₜ = 40 time
    steps, we typically get a rounding error of 1E{-d}, where d is the number
    of digits used. The effect of rounding errors may accumulate if we perform
    more operations, so increasing the number of time steps to 4000 gives a
    rounding error of 1E{-d+2}. Also if we compute with numbers that are much
    larger than unity, we lose accuracy due to rounding errors. For example,
    for the u values implied by I = 1000 and a = 100 (u ~ 1E3), the rounding
    errors increase to about 1E{-d+3}. A rough model for the size of rounding
    errors is 1E{-d+q+r}, where d is the number of digits, the number of time
    steps is of the order 1Eq time steps, and the size of the numbers in the
    arithmetic expressions are of order 1Er.

    We realise that rounding errors are at the lowest possible level if we
    scale the differential equation model, so the numbers entering the
    computations are of unity in size, and if we take a small number of steps.
    In general rounding errors are negligible in comparison with other errors
    in differential equation models.

    """
    import decimal
    from decimal import Decimal
    from typing import Tuple

    def solver_decimal(I: float, a: float, T: float, dt: float,
                       theta: float) -> Tuple[float]:
        """
        Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt.

        Parameters
        ----------
        I : Initial condition.
        a : Constant coefficient.
        T : Maximum time to compute to.
        dt : Step size.
        theta : theta=0 corresponds to FE, theta=0.5 to CN and theta=1 to BE.

        Returns
        ----------
        u : Mesh function.
        t : Mesh points.

        """
        # Initialise data structures
        I = Decimal(I)
        a = Decimal(a)
        T = Decimal(T)
        dt = Decimal(dt)
        theta = Decimal(theta)
        Nt = int(round(T / dt))      # Number of time intervals
        u = np.zeros(Nt + 1, dtype=np.object)      # Mesh function
        t = np.linspace(0, float(Nt * dt), Nt + 1) # Mesh points

        # Calculate mesh function using difference equation
        # (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a (θ uⁿ⁺¹ + (1 - θ) uⁿ)
        u[0] = I
        for n_idx in range(Nt):
            u[n_idx + 1] = (1 - (1 - theta) * a * dt) / \
                (1 + theta * a * dt) * u[n_idx]
        return u, t

    # Initialisation
    I = 1
    a = 1
    T = 4
    dt = 0.1
    digit_values = [4, 16, 64, 128]

    # Use 1000 digits for "exact" calculation
    decimal.getcontext().prec = 1000
    u_e, t = solver_decimal(I=I, a=a, T=T, dt=dt, theta=0.5)

    for digits in digit_values:
        decimal.getcontext().prec = digits
        u, t = solver_decimal(I=I, a=a, T=T, dt=dt, theta=0.5)
        error = u_e - u
        print(f'{digits:4} digits, {len(u) - 1} steps, max abs(error): ' \
              f'{np.max(np.abs(error)):.2E}')


def exponential_growth() -> None:
    """
    Explore θ-rule for exponential growth.

    Notes
    ----------
    Solve the ODE u' = -a u with a < 0 such that the ODE models exponential
    growth instead of decay. Investigate numerical artifacts and non-physical
    solution behaviour.

    """
    I = 1
    a = -1
    T = 2.5

    th_dict = {0: ('Forward Euler', 'fe', 'r-s'),
               1: ('Backward Euler', 'be', 'g-v'),
               0.5: ('Crank-Nicolson', 'cn', 'b-^')}

    for th in th_dict:
        fig, axs = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3}
        )
        fig.suptitle(
            r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$', y=0.95
        )
        for dt_idx, dt in enumerate((1.25, 0.75, 0.5, 0.1)):
            ax = axs.flat[dt_idx]
            ax.set_title('{}, dt={:g}'.format(th_dict[th][0], dt))
            ax.set_ylim(0, 15)
            ax.set_xlim(0, T)
            ax.set_xlabel('t')
            ax.set_ylabel('u')

            # Solve
            u, t = solver(I=I, a=a, T=T, dt=dt, theta=th)

            # Calculate exact solution
            u_exact = lambda t, I, a: I * np.exp(-a * t)
            t_e = np.linspace(0, T, 1001)
            u_e = u_exact(t_e, I, a)

            # Plot with red dashes w/ circles
            ax.plot(t, u, 'r--o', label='numerical')

            # Plot with blue line
            ax.plot(t_e, u_e, 'b-', label='exact')
            ax.legend()

        # Save figure
        fig.savefig(
            IMGDIR + f'/growth/{th_dict[th][1]}_growth.png',
            bbox_inches='tight'
        )

    def calc_amp(p: float, theta: float):
        """
        Calculate the amplification factor for u'=a*u.

        Parameters
        ----------
        p : -a Δt
        theta : theta=0 corresponds to FE, theta=0.5 to CN and theta=1 to BE.

        Returns
        ----------
        amp : Amplification factor.

        """
        return (1 + (1 - theta) * p) / (1 - theta * p)

    fig, ax = plt.subplots()
    ax.set_title(r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$')
    ax.set_xlabel('$p=-a\Delta t$')
    ax.set_ylabel('Amplification factor')
    ax.set_xlim(0, 3)
    ax.set_ylim(-20, 20)
    ax.grid(c='k', ls='--', alpha=0.3)

    # Mesh grid for p = -a Δt
    p = np.linspace(0, 3, 20)

    for th in th_dict:
        # Calculate amplification factor
        amp = calc_amp(p, th)

        # Plot
        ax.plot(p, amp, th_dict[th][2], label=th_dict[th][0])

    # Exact solution
    amp_exact = np.exp(p)
    ax.plot(p, amp_exact, 'k-o', label='exact')

    ax.legend(loc='lower left')
    fig.savefig(
        IMGDIR + '/growth/amplification_factors_growth.png',
        bbox_inches='tight'
    )

    I = 1
    a = np.linspace(0.01, 4, 22)
    dt = np.linspace(0.01, 2.5, 22)
    T = 6

    for th in th_dict:
        # Initialise B data structure
        B = np.zeros((len(a), len(dt)))

        # Solve for each a, Δt
        for a_idx, a_val in enumerate(a):
            for dt_idx, dt_val in enumerate(dt):
                u, t = solver(I, -a_val, T, dt_val, th)

                # Does u have the right monotone decay properties?
                is_monotone = True
                for n in range(1, len(u)):
                    if u[n] < u[n - 1]:
                        is_monotone = False
                        break
                B[a_idx, dt_idx] = 1. if is_monotone else 0.

        # Meshgrid
        a_grid, dt_grid = np.meshgrid(a, dt, indexing='ij')

        # Plot
        fig, ax = plt.subplots()
        ax.set_title('{} oscillatory region '.format(th_dict[th][0]) +
                     r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$')
        ax.set_xlabel('-a')
        ax.set_ylabel('dt')
        ax.set_xlim(0, np.max(a))
        ax.set_ylim(0, np.max(dt))
        ax.grid(c='k', ls='--', alpha=0.3)
        B = B.reshape(len(a), len(dt))
        ax.contourf(a_grid, dt_grid, B, levels=[0., 0.1], hatches='XX',
                    colors='grey', alpha=0.3, lines='k')
        fig.savefig(
            IMGDIR + f'/growth/{th_dict[th][1]}_osc_growth.png',
            bbox_inches='tight'
        )

    a = -1
    T = 4

    # dt values
    dt_values = [0.8, 0.4, 0.1, 0.01]

    # Setup figure
    fig, axs = plt.subplots(
        2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.3}
    )
    fig.suptitle(r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$', y=0.95)

    for dt_idx, dt in enumerate(dt_values):
        # Setup axes
        ax = axs.flat[dt_idx]
        ax.set_title(f'$\Delta t={dt}$')
        ax.set_xlim(0, T)
        ax.set_xlabel('t')
        ax.set_ylabel('log(abs(global error))')

        for th in th_dict:
            u, t = solver(I, a, T, dt, th)
            u_e = I * np.exp(-a * t)
            error = u_e - u

            # Exclude first error entry as it is 0
            ax.plot(t[1:], np.log(np.abs(error[1:])), label=th_dict[th][1])

        ax.legend(loc='upper right')

    fig.savefig(
        IMGDIR + '/growth/discretisation_errors_growth.png',
        bbox_inches='tight'
    )

    # Define difference approximations
    D_f = lambda u, dt, t: (u(t + dt) - u(t)) / dt
    D_b = lambda u, dt, t: (u(t) - u(t - dt)) / dt
    D_c = lambda u, dt, t: (u(t + dt) - u(t - dt)) / (2 * dt)

    # Define symbols
    a, t, dt, p = sym.symbols('a t dt p')

    # Define exact solution
    u = lambda t: sym.exp(a * t)
    u_sym = sym.exp(a * t)
    dudt = sym.diff(u_sym, t)

    # Setup figure
    fig, ax = plt.subplots()
    ax.set_title(r'$\frac{du(t)}{dt}=-a\cdot u(t)\:{\rm where}\:a<0$')
    ax.set_xlabel('$p=-a\Delta t$')
    ax.set_ylabel('E, error fraction')
    ax.set_xlim(1E-6, 1)
    ax.set_ylim(0.8, 1.2)
    ax.set_xscale('log')

    for name, op in zip(['FE', 'BE', 'CN'], [D_f, D_b, D_c]):
        E = op(u, dt, t) / dudt

        # Set p = -a * dt
        E = sym.simplify(sym.expand(E).subs(a * dt, p))
        print(STR_FMT.format(f'{name} E = ', f'{E} ≈ {E.series(p, 0, 3)}'))

        # Convert to lambda expr
        f_E = sym.lambdify([p], E, modules='numpy')

        # Calculate E as a function of p
        p_values = np.logspace(-6, 1, 101)
        E_v = f_E(p_values)

        # Plot
        ax.plot(p_values, E_v, label=f'${name}: {sym.latex(E)}$')

    ax.legend()
    fig.savefig(
        IMGDIR + '/growth/visual_accuracy_growth.png', bbox_inches='tight'
    )


def explore_rounding_err() -> None:
    """
    Exploring rounding errors in numerical calculus.

    Notes
    ----------
    a) Compute the absolute values of the errors in the numerical derivative of
       exp{-t} at t = ½ for forward difference, backward difference, and a
       centered difference, for Δt = 2^{-k}, k = 0,4,8,12,...,60. When do
       rounding errors destroy the accuracy?
    b) Compute the absolute values of the errors in the numerical approximation
       of ∫₀⁴ exp{-t} dt using the Trapezoidal and the Midpoint integration
       methods. Make a table of the errors for n = 2^k intervals, k =
       1,3,5,...,21. Is there any impact of the rounding errors?

    """
    from typing import Callable

    # Define difference approximations
    D_f = lambda u, dt, t: (u(t + dt) - u(t)) / dt
    D_b = lambda u, dt, t: (u(t) - u(t - dt)) / dt
    D_c = lambda u, dt, t: (u(t + dt) - u(t - dt)) / (2 * dt)

    # Definitions
    u = lambda t: np.exp(-t)
    dudt = u
    t = 0.5
    k_values = range(0, 63, 4)

    print('k    forward backward centered')
    for k in k_values:
        dt = 2**(-k)
        fwd_err = abs(dudt(t) - D_f(u, dt, t))
        bwd_err = abs(dudt(t) - D_b(u, dt, t))
        cen_err = abs(dudt(t) - D_c(u, dt, t))
        print(f'{k:3} {fwd_err:.2E} {bwd_err:.2E} {cen_err:.2E}')

    def trapezoidal(f: Callable[[float], float], a: float, b: float, n: int):
        """
        Trapezoidal rule for integral of f from a to b using n intervals.

        Parameters
        ----------
        f : Function to integrate
        a : Lower bound.
        b : Upper bound.
        n : Number of intervals.

        Returns
        ----------
        integral : Integral of f(x) from x=a to x=b.

        """
        h = float(b - a) / n
        I = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            I += f(a + i * h)
        return h * I

    def midpoint(f: Callable[[float], float], a: float, b: float, n: int):
        """
        Midpoint-rule for integral of f from a to b using n intervals.

        Parameters
        ----------
        f : Function to integrate
        a : Lower bound.
        b : Upper bound.
        n : Number of intervals.

        Returns
        ----------
        integral : Integral of f(x) from x=a to x=b.

        """
        h = float(b - a) / n
        I = 0
        for i in range(n):
            I += f(a + (i + 0.5) * h)
        return h * I

    # Definitions
    u = lambda t: np.exp(-t)
    U = lambda a, b: -u(b) - (-u(a))
    a = 0
    b = 4

    print('k  trapezoidal midpoint')
    for k in range(1, 23, 2):
        n = 2**k
        err_trap = abs(U(a, b) - trapezoidal(u, a, b, n))
        err_midp = abs(U(a, b) - midpoint(u, a, b, n))
        print(f'{k:3} {err_trap:.2E} {err_midp:.2E}')


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description=
        'Finite Difference Computing with Exponential Decay Models - Chapter 2'
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
