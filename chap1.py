#! /usr/bin/env python3

"""
Notes from Chapter 1 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
          u'(t) = -a u(t)     a > 0     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
from matplotlib import pyplot as plt


__all__ = ['forward_euler']

IMGDIR = './img/chap1/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def forward_euler() -> None:
    r"""
    Solve ODE (1) by the Forward Euler (FE) finite difference method.

    Notes
    ----------
    Solving an ODE like (1) by a finite difference method consists of the
    following four steps:
    1) discretising the domain,
    2) requiring fulfillment of the equation at discrete time points,
    3) replacing derivatives by finite differences,
    4) formulating a recursive algorithm.

    Step 1: Discretising the domain.
        We represent the time domain [0, T] by a finite number of points N_t+1,
        called a "mesh" or "grid".

                  0 = t0 < t1 < t2 < ... < t{N_t-1} < t{N_t} = T

        The goal is to find the solution u at the mesh points: u(tn) := uⁿ,
        n=0,1,...,N_t. More precisely, we let uⁿ be the numerical approximation
        to the exact solution u(tn) at t=tn.

        We say that the numerical approximation constitutes a mesh function,
        which is defined at discrete points in time. To compute the solution
        for some solution t ∈ [tn, tn+1], we can use an interpolation method,
        e.g. the linear interpolation formula

                u(t) ≈ uⁿ + (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) * (t - tn)

    Step 2: Fulfilling the equation at discrete time points.
        We relax the requirement that the ODE hold for all t ∈ (0, T] and
        require only that the ODE be fulfilled at discrete points in time. The
        mesh points are a natural (but not the only) choice of points. The
        original ODE is reduced to the following

             u'(tn) = -a u(tn)     a > 0     n = 0,...,N_t     u(0) = I

    Step 3: Replace derivative with finite differences.
        The next most essential step is to replace the derivative u' by the
        finite difference approximation. Here the forward difference
        approximation is

                      u'(tn) ≈ (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn)

        The name forward relates to the face that we use a value forward in
        time, uⁿ⁺¹, together with the value uⁿ at the point tn. Therefore,

              (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a uⁿ,     n=0,1,...,N_t-1

        This is often referred to as a finite difference scheme or more
        generally as the discrete equations of the problem. The fundamental
        feature of these equations is that they are algebraic and can hence be
        straightforwardly solved to produce the mesh function uⁿ.

    Step 4: Formulating a recursive algorithm
        The final step is to identify the computational algorithm to be
        implemented in a program. In general, starting from u⁰ = u(0) = I, uⁿ
        can be assumed known, and then we can easily solve for the unknown uⁿ⁺¹

                        uⁿ⁺¹ = uⁿ - a (t{n+1} - tn) uⁿ

        From a mathematical point of view, these type of equations are known as
        difference equations since they express how differences in the
        dependent variable, here u, evolve with n.

    Interpretation: We have computed some point values on the solution curve,
    and the question is how we reason about the next point. Since we know u and
    t at the most recently computed point, the differential equation gives us
    the slope of the solution u' = -a u. We can continue the solution curve
    along that slop. As soon as we have chosen the next point on this line, we
    have a new t and u value and can compute a new slope and continue this
    process.

    """
    # Define solver function
    def solver(I: float, a: float, T: float, dt: float):
        """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt using FE.

        Parameters
        ----------
        I : Initial condition.
        a : Constant coefficient.
        T : Time to compute to.
        dt : Step size.

        Returns
        ----------
        u : Mesh function.
        t : Mesh points.

        """
        # Initialise data structures
        Nt = int(round(T / dt)) # Number of time intervals
        u = np.zeros(Nt + 1)    # Mesh function
        t = np.linspace(0, T, Nt + 1) # Mesh points

        # Calculate mesh function using difference equation
        # uⁿ⁺¹ = uⁿ - a (t{n+1} - tn) uⁿ
        u[0] = I
        for n_idx in range(Nt):
            u[n_idx + 1] = (1 - a * dt) * u[n_idx]
        return u, t

    I = 1
    a = 2
    T = 8
    dt = 0.8
    u, t = solver(I=I, a=a, T=T, dt=dt)

    # Write out a table of t and u values:
    for idx, t_i in enumerate(t):
        print('t={0:6.3f} u={1:g}'.format(t_i, u[idx]))

    # Plot with red dashes w/ circles
    plt.plot(t, u, 'r--o', label='numerical')

    # Calculate exact solution
    u_exact = lambda t, I, a: I * np.exp(-a * t)
    t_e = np.linspace(0, T, 1001)
    u_e = u_exact(t_e, I, a)

    # Plot with blue line
    plt.plot(t_e, u_e, 'b-', label='exact')

    # Save figure
    plt.xlabel('t')
    plt.ylabel('u')
    plt.title('Forward Euler, dt={:g}'.format(dt))
    plt.legend()
    plt.savefig(IMGDIR + 'fe.png', bbox_inches='tight')


def main() -> None:
    """Main program, used when run as a script."""
    import argparse
    parser = argparse.ArgumentParser(
        description=
        'Finite Difference Computing with Exponential Decay Models - Chapter 1'
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
