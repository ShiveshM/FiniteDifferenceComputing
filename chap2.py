#! /usr/bin/env python3

r"""
Notes from Chapter 2 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
          u'(t) = -a u(t)     a > 0     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
from matplotlib import pyplot as plt

from utils.solver import solver


__all__ = ['investigations', 'stability']

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

    This formula can explain everything we see in the figures produced in
    `detailed_experiments`, but it also gives us a more general insight into
    the accuracy and stability properties of the three schemes.

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


def accuracy() -> None:
    """
    Quantitatively examine how large numerical errors are.

    Notes
    ----------
    While stability concerns the qualitative properties of the numerical
    solution, it remains to investigate the quantitative properties to see
    exactly how large the numerical errors are.

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
