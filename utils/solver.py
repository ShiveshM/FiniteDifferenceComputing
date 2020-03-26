"""
Solver functions

"""

import numpy as np


__all__ = ['solver']


def solver(I: float, a: float, T: float, dt: float, theta: float):
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt.

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
    t = np.linspace(0, Nt * dt, Nt + 1) # Mesh points

    # Calculate mesh function using difference equation
    # (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a (θ uⁿ⁺¹ + (1 - θ) uⁿ)
    u[0] = I
    for n_idx in range(Nt):
        u[n_idx + 1] = (1 - (1 - theta) * a * dt) / \
            (1 + theta * a * dt) * u[n_idx]
    return u, t
