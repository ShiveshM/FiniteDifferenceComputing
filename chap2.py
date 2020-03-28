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
           'global_error', 'integrated_error', 'truncation_error']

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

    for th in th_dict.keys():
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

    for th in th_dict.keys():
        # Initialise B data structure
        B = np.zeros((len(a), len(dt)))

        # Solve for each a, Δt
        for a_idx, a_val in enumerate(a):
            for dt_idx, dt_val in enumerate(dt):
                u, t = solver(I, a_val, T, dt_val, th)

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

    """
    def calc_amp(p: float, theta: float):
        """Calculate the amplification factor for u'=-a*u.

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

    for th in th_dict.keys():
        # Calculate amplification factor
        amp = calc_amp(p, th)

        # Plot
        ax.plot(p, amp, th_dict[th][2], label=th_dict[th][0])

    # Exact solution
    amp_exact = np.exp(-p)
    ax.plot(p, amp_exact, 'k-o', label='exact')

    ax.legend(loc='lower left')
    fig.savefig(IMGDIR + 'amplification_factors.png', bbox_inches='tight')


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

    """


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


if __name__ == "__main__":
    main()
