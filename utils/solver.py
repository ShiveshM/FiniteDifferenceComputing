"""
Solver functions

"""

from typing import Callable, Tuple

import numpy as np


__all__ = ['solver_chap2', 'solver_chap3']


def solver_chap2(I: float, a: float, T: float, dt: float,
                 theta: float, c: float = 0) -> Tuple[float]:
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
    Nt = int(T / dt)      # Number of time intervals
    u = np.zeros(Nt + 1)  # Mesh function
    t = np.linspace(0, Nt * dt, Nt + 1) # Mesh points

    # Calculate mesh function using difference equation
    # (uⁿ⁺¹ - uⁿ) / (t{n+1} - tn) = -a (θ uⁿ⁺¹ + (1 - θ) uⁿ)
    u[0] = I
    for n in range(Nt):
        u[n + 1] = (1 - (1 - theta) * a * dt) / \
            (1 + theta * a * dt) * u[n] + \
            (c * dt) / (1 + theta * a * dt)
    return u, t


def solver_chap3(I: float, a: Callable[[float], float],
                 b: Callable[[float], float], T: float, dt: float,
                 theta: float) -> Tuple[float]:
    """
    Solve u'=-a(t)*u + b(t), u(0)=I, for t in (0,T] with steps of dt.

    Parameters
    ----------
    I : Initial condition.
    a : Function of t.
    b : Function of t.
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
    # (uⁿ⁺¹ - uⁿ) / Δt = 0(-aⁿ⁺¹ uⁿ⁺¹ + bⁿ⁺¹) + (1 - 0)(-aⁿ uⁿ + bⁿ)
    u[0] = I
    for n in range(Nt):
        u[n + 1] = ((1 - dt * (1 - theta) * a(t[n])) * u[n] + \
                    dt * (theta * b(t[n + 1]) + (1 - theta) * b(t[n]))) / \
            (1 + dt * theta * a(t[n + 1]))
    return u, t
