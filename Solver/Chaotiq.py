from Solver.Methods import rungeKutta

method_registry = {
    'RK4': rungeKutta.solve
}

def solver(method, f, domain, n, alpha, params):

    """
    Solves a given differential equation using the specified numerical method.

    Parameters:
    - method (str): The method to use for solving the differential equation.
                    Available options are 'RK4' for the 4th-order Runge-Kutta method.
    - params (tuple, optional): Additional parameters to pass to function f.

    Returns:
    - solution (list or array): The solution of the differential equation over the specified domain.

    Raises:
    - ValueError: If the specified method is not available in the method_registry
    """
    if method in method_registry:
        return method_registry[method](f, domain, n, alpha, params)
    else:
        raise ValueError('Method "{}" not available'.format(method))