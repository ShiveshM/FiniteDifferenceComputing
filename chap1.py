#! /usr/bin/env python3

r"""
Notes from Chapter 1 of Hans Petter Langtangen's book "Finite Difference
Computing with Exponential Decay Models".

Problem formulation: Using finite difference methods find u(t) such that:
          u'(t) = -a u(t)     a > 0     t ∈ (0, T]     u(0) = I       (1)

"""

import numpy as np
from matplotlib import pyplot as plt


__all__ = ['forward_euler', 'backward_euler', 'crank_nicolson', 'unifying']

IMGDIR = './img/chap1/'
"""Path to store images."""
STR_FMT = '{0}\n{1}\n'
"""String formatting for printing to standard output."""


def forward_euler() -> None:
    r"""
    Solve ODE (1) by the Forward Euler (FE) finite difference scheme.

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

    Derivation: The key tool for this is the Taylor series.

    f(x) ≈ f(a) + f'(a)(x - a) + ½f''(a)(x - a)² + ⅙f'''(a)(x - a)³ + ...
           + (1 / m!) dᵐf/dxᵐ(a)(x - a)ᵐ

    For a function of time, f(t), related to mesh spacing Δt,

    f(tn + Δt) ≈ f(tn) + f'(tn)Δt + ½f''(tn)Δt² + ⅙f'''(tn)Δt³ + ...
                 + (1 / m!) dᵐf/dtᵐ(tn)Δtᵐ

    Now, by rearranging for f'(tn)

    f'(tn) ≈ (f(tn + Δt) - f(tn)) / Δt - ½f''(tn)Δt - ⅙f'''(tn)Δt² - ...
             - (1 / m!) dᵐf/dtᵐ(tn)Δtᵐ⁻¹

    Now, in the limit Δt -> 0,

                    f'(tn) ≈ (f(tn + Δt) - f(tn) / Δt

    An interesting point is that we have a measure of the error as seen by the
    O(Δtᵐ) terms. A leading order, the error can be given as ½f''(tn)Δt.

    Compact notation: For a function u(t), a forward difference approximation
    is denoted by the Dₜ⁺ operator and written as

                 [Dₜ⁺ u]ⁿ = (uⁿ⁺¹ - uⁿ) / Δt    (≈ du(tn)/dt)

    This notation consists of an operator that approximates differentiation wrt
    an independent variable, here t. The operator is built on the symbol D,
    with the independent variable as subscript and a superscript denoting the
    type of difference. The superscript ⁺ indicates a forward difference. We
    place square brackets around the operator and the function it operates on
    and specify the mesh point, where the operator is acting, by a superscript
    after the closing bracket. In our compact notation, the Forward Euler
    scheme can be written as

                              [Dₜ⁺ u]ⁿ = -a uⁿ

    In difference equations, we often place the square brackets around the
    whole equation, to indicate which mesh point the equation applies, since
    each term by be approximated at the same point:

                              [Dₜ⁺ u = -a u]ⁿ

    """
    # Define solver function
    def solver(I: float, a: float, T: float, dt: float):
        """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt using FE.

        Parameters
        ----------
        I : Initial condition.
        a : Constant coefficient.
        T : Maximum time to compute to.
        dt : Step size.

        Returns
        ----------
        u : Mesh function.
        t : Mesh points.

        """
        # Initialise data structures
        Nt = int(T / dt)      # Number of time intervals
        u = np.zeros(Nt + 1)  # Mesh function
        t = np.arange(0, (Nt + 1) * dt, dt) # Mesh points

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

    # Write out a table of t and u values
    for idx, t_i in enumerate(t):
        print('t={0:6.3f} u={1:g}'.format(t_i, u[idx]))

    # Plot with red dashes w/ circles
    plt.figure()
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


def backward_euler() -> None:
    r"""
    Solve ODE (1) by the Backward Euler (BE) finite difference scheme.

    Notes
    ----------
    There are several choices of difference approximations in step 3 of the
    finite difference scheme presenting in `forward_euler`. Another alternative
    is

                      u'(tn) ≈ (uⁿ - uⁿ⁻¹) / (tn - t{n-1})

    Since this difference is going backward in time (t{n-1}) for information,
    it is known as a backward difference, also called Backward Euler
    difference. Inserting our equation yields

              (uⁿ - uⁿ⁻¹) / (tn - t{n-1}) = -a uⁿ,     n=1,...,N_t

    For direct similarity to the Forward Euler scheme, we replace n by n+1 and
    solve for the unknown value uⁿ⁺¹

            uⁿ⁺¹ = 1 / (1 + a (t{n+1} - tn)) * uⁿ,     n=0,...,N_t-1

    Derivation: Here we use the Taylor series around f(fn - Δt)

    f(tn - Δt) ≈ f(tn) - f'(tn)Δt + ½f''(tn)Δt² - ⅙f'''(tn)Δt³ + ...
                 + (1 / m!) dᵐf/dtᵐ(tn)Δtᵐ

    Solving with respect to f'(tn) gives

    f'(tn) ≈ (f(tn) - f(tn - Δt)) / Δt + ½f''(tn)Δt - ⅙f'''(tn)Δt² + ...
             - (1 / m!) dᵐf/dtᵐ(tn)Δtᵐ⁻¹

    Then term ½f''(tn)Δt can be taken as a simple measure of the approximation
    error.

    Compact notation: The backward difference reads

                 [Dₜ⁻ u]ⁿ = (uⁿ - uⁿ⁻¹) / Δt    (≈ du(tn)/dt)

    Note the subscript ⁻ denotes the backward difference. The Backward Euler
    scheme to our ODE can be written as

                              [Dₜ⁻ u = -a u]ⁿ

    """
    # Define solver function
    def solver(I: float, a: float, T: float, dt: float):
        """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt using BE.

        Parameters
        ----------
        I : Initial condition.
        a : Constant coefficient.
        T : Maximum time to compute to.
        dt : Step size.

        Returns
        ----------
        u : Mesh function.
        t : Mesh points.

        """
        # Initialise data structures
        Nt = int(T / dt)      # Number of time intervals
        u = np.zeros(Nt + 1)  # Mesh function
        t = np.arange(0, (Nt + 1) * dt, dt) # Mesh points

        # Calculate mesh function using difference equation
        # uⁿ⁺¹ = 1 / (1 + a (t{n+1} - tn)) * uⁿ
        u[0] = I
        for n_idx in range(Nt):
            u[n_idx + 1] = 1 / (1 + a * dt) * u[n_idx]
        return u, t

    I = 1
    a = 2
    T = 8
    dt = 0.8
    u, t = solver(I=I, a=a, T=T, dt=dt)

    # Write out a table of t and u values
    for idx, t_i in enumerate(t):
        print('t={0:6.3f} u={1:g}'.format(t_i, u[idx]))

    # Plot with red dashes w/ circles
    plt.figure()
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
    plt.title('Backward Euler, dt={:g}'.format(dt))
    plt.legend()
    plt.savefig(IMGDIR + 'be.png', bbox_inches='tight')


def crank_nicolson() -> None:
    r"""
    Solve ODE (1) by the Crank-Nicolson (CN) finite difference scheme.

    Notes
    ----------
    The finite difference schemes derived in `forward_euler` and
    `backward_euler` are both one-sided differences. Such one-sided differences
    are known to be less accurate than central (or midpoint) differences, where
    we use information both forward and backward in time. A natural next step
    is therefore to construct a central difference approximation.

    The central difference approximation to the derivative is sought at the
    point t{n+½} = ½(tn + t{n+1}). The approximation reads

                    u'(t{n + ½}) ≈ (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn)

    With this formula, it is natural to demand the ODE be fulfilled at the time
    points between the mesh points:

                    u'(t{n + ½}) = -a u(t{n + ½}),     n=0,...,N_t-1

    Combining these results results in the approximate discrete equation

            (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a uⁿ⁺¹⸍²,     n=0,1,...,N_t-1

    However, there is a fundamental problem with the right-hand side of the
    equation. We aim to compute uⁿ for integer n, which means that uⁿ⁺¹⸍² is
    not a quantity computed by our method. One possibility is to approximate
    uⁿ⁺¹⸍² as an arithmetic mean of the u values at the neighbouring mesh
    points

                            uⁿ⁺¹⸍² ≈ ½(uⁿ + uⁿ⁺¹)

    We then obtain the approximate discrete equation

        (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a ½(uⁿ + uⁿ⁺¹),     n=0,1,...,N_t-1

    There are three approximation steps leading to this formula. First, the ODE
    is only valid at discrete points. Second, the derivative is approximated by
    finite differences, and third, the value of u between mesh points is
    approximated by an arithmetic mean value. Despite one more approximation
    than for the Backward and Forward Euler schemes, the use of a centered
    difference leads to a more accurate method.

    To formulate a recursive method, we assume that uⁿ is already computed so
    that uⁿ⁺¹ is the unknown, which we can solve for:

            uⁿ⁺¹ = (1 - ½ a (t{n+1} - tn) / (1 + ½ a (t{n+1} - tn) * uⁿ

    Derivation: The centered difference approximates the derivative at
    tn + ½Δt. The Taylor expansions of f(tn) and f(t{n+1}) around tn + ½Δt are

    f(tn) ≈ f(tn + ½Δt) - f'(tn + ½Δt)½Δt + ½f''(tn + ½Δt)(½Δt)² -
            ⅙f'''(tn + ½Δt)(½Δt)³ + ... + (1 / m!) dᵐf/dtᵐ(tn + ½Δt)(½Δt)ᵐ
    f(t{n+1}) ≈ f(tn + ½Δt) + f'(tn + ½Δt)½Δt + ½f''(tn + ½Δt)(½Δt)² +
                ⅙f'''(tn + ½Δt)(½Δt)³ + ... + (1 / m!) dᵐf/dtᵐ(tn + ½Δt)(½Δt)ᵐ

    Subtracting the first from the second givesΔt)²

    f(t{n+1}) - f(tn) ≈ f'(tn + ½Δt)Δt + ⅓f'''(tn + ½Δt)(½Δt)³ + ...

    Solving with respect to f'(tn + ½Δt) results in

    f'(tn + ½Δt) ≈ (f(t{n+1}) - f(tn)) / Δt - (1/24)f''(tn + ½Δt)Δt² + ...

    The error measure goes like O(Δt²), which means the error here goes faster
    to zero compared to the forward and backward differences for small Δt.

    Compact notation: The centered difference operator notation reads

                 [Dₜ u]ⁿ = (uⁿ⁺¹⸍² - uⁿ⁻¹⸍²) / Δt    (≈ du(tn)/dt)

    Note here that no superscript implies a central differences. An averaging
    operator is also convenient to have:

                    [ūᵗ]ⁿ = ½(uⁿ⁻¹⸍² + uⁿ⁺¹⸍²) ≈ u(tn)

    The superscript ᵗ indicates that the average is taken along the time
    coordinate. The common average (uⁿ + uⁿ⁺¹⸍²) / 2 can now be expressed as
    [ūᵗ]ⁿ⁺¹⸍².

    Now the Crank-Nicolson scheme to our ODE can be written as

                            [Dₜ u = -a ūᵗ]ⁿ⁺¹⸍²

    """
    # Define solver function
    def solver(I: float, a: float, T: float, dt: float):
        """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt using BE.

        Parameters
        ----------
        I : Initial condition.
        a : Constant coefficient.
        T : Maximum time to compute to.
        dt : Step size.

        Returns
        ----------
        u : Mesh function.
        t : Mesh points.

        """
        # Initialise data structures
        Nt = int(T / dt)      # Number of time intervals
        u = np.zeros(Nt + 1)  # Mesh function
        t = np.arange(0, (Nt + 1) * dt, dt) # Mesh points

        # Calculate mesh function using difference equation
        # uⁿ⁺¹ = 1 / (1 + a (t{n+1} - tn)) * uⁿ
        u[0] = I
        for n_idx in range(Nt):
            u[n_idx + 1] = (1 - 0.5 * a * dt) / (1 + 0.5 * a * dt) * u[n_idx]
        return u, t

    I = 1
    a = 2
    T = 8
    dt = 0.8
    u, t = solver(I=I, a=a, T=T, dt=dt)

    # Write out a table of t and u values
    for idx, t_i in enumerate(t):
        print('t={0:6.3f} u={1:g}'.format(t_i, u[idx]))

    # Plot with red dashes w/ circles
    plt.figure()
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
    plt.title('Crank-Nicolson, dt={:g}'.format(dt))
    plt.legend()
    plt.savefig(IMGDIR + 'cn.png', bbox_inches='tight')


def unifying() -> None:
    r"""
    The Forward Euler, Backward Euler, and Crank-Nicolson schemes.

    Notes
    ----------
    The Forward Euler, Backward Euler, and Crank-Nicolson schemes can be
    formulated as one scheme with varying parameter θ:

          (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a (θ uⁿ⁺¹ + (1 - θ) uⁿ)

    Observe that
    - θ = 0 gives the Forward Euler scheme.
    - θ = 1 gives the Backward Euler scheme.
    - θ = ½ gives the Crank-Nicolson scheme.

    One may alternatively choose any other value of θ in [0, 1], but this is
    not so common since the accuracy and stability of the scheme do not improve
    compared to the values θ = 0, 1, ½. As before, uⁿ is considered known and
    uⁿ⁺¹ unknown, so we solve for the latter:

          uⁿ⁺¹ = (1 - (1 - θ) a (t{n+1} - tn)) / (1 + θ a (t{n+1} - tn)

    This is known as the θ-rule, or alternatively written as the "theta-rule".

    Compact notation: The θ-rule can be specified in operator notation by

                            [D̄ₜ u = -a ūᵗʼᶿ]ⁿ⁺ᶿ

    We define a new time difference

                    [D̄ₜ u]ⁿ⁺ᶿ = (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn)

    To be applied at the time point t{n+θ} ≈ θ tn + (1 - θ) t{n+1}. This
    weighted average gives rise to the weighted average operator

                  [ūᵗʼᶿ]ⁿ⁺ᶿ = (1 - θ) uⁿ + θ uⁿ⁺¹ ≈ u(t{n+θ})

    where θ ∈ [0, 1] as usual. Note that for θ = ½, we recover the standard
    centered difference and standard arithmetic mean. An alternative and
    perhaps clearer notation is

                  [Dₜ u]ⁿ⁺¹⸍² = θ [-a u]ⁿ⁺¹ + (1 - θ) [-a u]ⁿ

    """
    # Define solver function
    def solver(I: float, a: float, T: float, dt: float, theta: float):
        """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt using BE.

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
        Nt = int(T / dt)      # Number of time intervals
        u = np.zeros(Nt + 1)  # Mesh function
        t = np.arange(0, (Nt + 1) * dt, dt) # Mesh points

        # Calculate mesh function using difference equation
        # (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a (θ uⁿ⁺¹ + (1 - θ) uⁿ)
        u[0] = I
        for n_idx in range(Nt):
            u[n_idx + 1] = (1 - (1 - theta) * a * dt) / \
                (1 + theta * a * dt) * u[n_idx]
        return u, t

    I = 1
    a = 2
    T = 8
    dt = 0.8

    # Write out a table of t and u values for each theta
    for th in (0, 0.5, 1):
        u, t = solver(I=I, a=a, T=T, dt=dt, theta=th)
        print('theta = {:g}'.format(th))
        for idx, t_i in enumerate(t):
            print('t={0:6.3f} u={1:g}'.format(t_i, u[idx]))
        print('-----')


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
