#! /usr/bin/env python3

r"""
Notes from Chapter 4 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
           u'(t) = -a(t) u(t) + b(t)     t ∈ (0, T]     u(0) = I       (1)
               u'(t) = f(u, t)     t ∈ (0, T]     u(0) = I             (2)

"""

import numpy as np
import matplotlib.gridspec as gridspec
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
    from utils.solver import solver_chap4 as solver

    # Definitions
    # Solving y'(x) = λ y + α    x ∈ (0, X]    y(0) = I ≠ 0
    I = 1
    X = 4
    # lamda = -0.8
    # alpha = 0.5
    # ylims = (0.5, 1)
    lamda = 0.8
    alpha = 0.5
    ylims = (0, 40)
    y_exact = lambda x: np.exp(x * lamda) * I + \
        ((-1 + np.exp(x * lamda)) * alpha) / lamda
    u_exact = lambda I, t: I * np.exp(t)

    # Scaling
    #                  ỹ = (y + yᵐ) / yᶜ,      𝐱 = x / xᶜ
    #               yᶜ = I,     yᵐ = α / λ,       xᶜ = 1 / λ
    # therefore,
    #        dỹ/d𝐱 = ỹ,     𝐱 ∈ (0, λ T],     ỹ(0) = 1 + β = 1 + α / (I λ)
    #
    #                        y(x) = (ỹ(λ x) - β) * I
    beta = alpha / (I * lamda)
    ulims = list(map(lambda x: (x / I) + beta, ylims))

    th_dict = {0: ('Forward Euler', 'fe', 'r-s'),
               1: ('Backward Euler', 'be', 'g-v'),
               0.5: ('Crank-Nicolson', 'cn', 'b-^')}

    fig1 = plt.figure(figsize=(14, 10))
    fig3 = plt.figure(figsize=(14, 10))
    fig1.suptitle(
        r'$\frac{dy(x)}{dx}=\lambda y(x)+\alpha\:{\rm where}\:\lambda=' +
        f'{lamda:g}' + r',\alpha=' + f'{alpha:g}' + r'$', y=0.95
    )
    fig3.suptitle(
        r'$\frac{du(t)}{dt}=u(t)\:{\rm where}\:\beta=' +
        f'{beta:g}' + r'$', y=0.95
    )

    axs1 = []
    axs3 = []
    gs1 = gridspec.GridSpec(2, 2)
    gs3 = gridspec.GridSpec(2, 2)
    gs1.update(hspace=0.2, wspace=0.2)
    gs3.update(hspace=0.2, wspace=0.2)
    for th_idx, th in enumerate(th_dict):
        fig2, axs2 = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3}
        )
        for dx_idx, dx in enumerate((1.5, 1.25, 0.75, 0.1)):
            # Calculate mesh points
            Nx = int(X / dx)
            x = np.linspace(0, Nx * dx, Nx + 1)

            # Calculate scaled mesh points
            T = lamda * X
            t = lamda * x
            dt = lamda * dx

            if th_idx == 0:
                gssub1 = gs1[dx_idx].subgridspec(
                    2, 1, height_ratios=(2, 1), hspace=0
                )
                ax1 = fig1.add_subplot(gssub1[0])
                ax2 = fig1.add_subplot(gssub1[1])
                gssub2 = gs3[dx_idx].subgridspec(
                    2, 1, height_ratios=(2, 1), hspace=0
                )
                axs1.append([ax1, ax2])
                ax3 = fig3.add_subplot(gssub2[0])
                ax4 = fig3.add_subplot(gssub2[1])
                axs3.append([ax3, ax4])

                ax1.set_title('$\Delta x={:g}$'.format(dx))
                ax1.set_ylim(*ylims)
                ax1.set_xlim(0, X)
                ax1.set_ylabel('y(x)')
                ax1.set_xticks([])
                ax1.set_yticks(ax1.get_yticks()[1:])

                ax2.grid(c='k', ls='--', alpha=0.3)
                ax2.set_yscale('log')
                ax2.set_xlim(0, X)
                ax2.set_xlabel('x')
                ax2.set_ylabel('log err')

                ax3.set_title('$\Delta t={:g}$'.format(dt))
                ax3.set_ylim(*ulims)
                ax3.set_xlim(0, T)
                ax3.set_ylabel('u(t)')
                ax3.set_xticks([])
                ax3.set_yticks(ax3.get_yticks()[1:])

                ax4.grid(c='k', ls='--', alpha=0.3)
                ax4.set_yscale('log')
                ax4.set_xlim(0, T)
                ax4.set_xlabel('t')
                ax4.set_ylabel('log err')

                # Calculate exact solution
                x_e = np.linspace(0, X, 1001)
                y_e = y_exact(x_e)
                t_e = np.linspace(0, T, 1001)
                u_e = u_exact(1 + beta, t_e)

                # Plot with black line
                ax1.plot(x_e, y_e, 'k-', label='exact')
                ax3.plot(t_e, u_e, 'k-', label='exact')

            ax1, ax2 = axs1[dx_idx]
            ax3, ax4 = axs3[dx_idx]

            ax = axs2.flat[dx_idx]
            ax.set_title('{}, dx={:g}'.format(th_dict[th][0], dx))
            # ax.set_ylim(0, 15)
            ax.set_xlim(0, X)
            ax.set_xlabel(r'x')
            ax.set_ylabel(r'$y(x)$')

            # Solve dỹ/d𝐱 = ỹ using Crank-Nicolson scheme
            u = solver(I=1 + beta, a=-1, t=t, theta=th)

            # Convert back to y
            y = (u - beta) * I

            # Calculate exact solution
            y_e = y_exact(x)
            u_e = u_exact(1 + beta, t)

            # Plot
            ax.plot(x, y, 'r--o', label='numerical')
            ax1.plot(x, y, th_dict[th][2], label=th_dict[th][0])
            ax1.legend()
            ax3.plot(t, u, th_dict[th][2], label=th_dict[th][0])
            ax3.legend()

            # Plot exact solution
            ax.plot(x, y_e, 'b-', label='exact')
            ax.legend()
            err = np.abs(y_e - y)
            ax2.plot(x, err, th_dict[th][2])
            err = np.abs(u_e - u)
            ax4.plot(t, err, th_dict[th][2])

        # Save figure
        fig2.savefig(
            IMGDIR + f'{th_dict[th][1]}_growth.png',
            bbox_inches='tight'
        )

    # Save figure
    fig1.savefig(
        IMGDIR + f'comparison.png', bbox_inches='tight'
    )
    fig3.savefig(
        IMGDIR + f'comparison_scaled.png', bbox_inches='tight'
    )


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

