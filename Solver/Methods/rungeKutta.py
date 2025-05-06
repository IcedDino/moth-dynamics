import numpy as np
from numba import njit, prange

@njit
def solve(f, domain: np.ndarray, n: int, alpha: np.ndarray, params):

    """
    4th-order Runge-Kutta solver using Numba.

    Parameters:
        f (function): Function representing the system of ODEs (dy/dt = F(t, y, params)).
        domain (array): Packaged interval.
        n (int): Number of steps.
        alpha (array-like): Initial conditions.
        params (tuple, optional): Additional parameters to pass to function f.

    Returns:
        t_values (numpy array): Time values.
        y_values (numpy array): Solution values for each time step.
    """
    a, b = domain[0], domain[1]
    h = (b - a) / n  # Step size
    t_values = np.linspace(a, b, n + 1)
    y_values = np.zeros((n + 1, len(alpha)))
    y_values[0] = alpha

    for i in prange(n):
        t = t_values[i]
        y = y_values[i]

        k1 = h * f(t, y, params)
        k2 = h * f(t + h/2, y + k1/2, params)
        k3 = h * f(t + h/2, y + k2/2, params)
        k4 = h * f(t + h, y + k3, params)

        y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values