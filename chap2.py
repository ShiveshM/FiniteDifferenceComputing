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


__all__ = ['investigations']

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
