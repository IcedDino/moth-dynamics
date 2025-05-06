import numpy as np
from numba import njit, float64
import numpy.typing as npt
from scipy.optimize import differential_evolution
from Solver import Chaotiq
from Plague.DataScrap.LoadData import load_insect_data
from Plague.DataScrap.LoadTableData import load_table
from Plague.Error.ErrorFunction import lsquare
from Plague.Wasps.DiadegmaFunction.Diadegma import diadegma_lookup
from Plague.Wasps.TrichogrammaFunction.Trichogramma import trich_lookup
# Data

data_filepath = '../../DataScrap/Data/InsectData_206.csv'
data_arrays = load_insect_data(file_path=data_filepath, from_excel=False, shift_time=7, min_day=0)

# Extract individual arrays from the dictionary
T = data_arrays['T']  # Sample days
E = data_arrays['E']  # Eggs per plant
L = data_arrays['L']  # Larvae per plant
P = data_arrays['P']  # Pupae per plant
Ti = data_arrays['Ti'] # Pesticide Applications

# Table 1
mE, mL, mP, mA, aE, aL, aP, mu, mT, mD = load_table()

#Sol domain
domain = np.array([T[0], T[-1]])

@njit
def g(t: float64, gamma: float64, lambd: float64) -> float64:
    result = 0.0
    for day in Ti:
        if t >= day:
            result += gamma * np.exp(-lambd * (t - day))
    return result

@njit
def f(t: float64, y: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

    r, gamma, lambd, sigma, aD, aT = params

    dEdt = mu * y[3] - mE * y[0] - aT * y[0] * y[5] - aE * y[0]
    dLdt = aE * y[0] - mL * y[1] - np.exp(-r * t) * g(t, gamma, lambd) * y[1] - aL * y[1]
    dPdt = aL * y[1] - mP * y[2] - aD * y[2] * y[4] - aP * y[2]
    dAdt = aP * y[2] - mA * y[3] + sigma
    dDdt = aD * y[4] * y[2] - mD * y[4] + diadegma_lookup(t) * 100
    dTdt = aT * y[5] * y[0] - mT * y[5] + trich_lookup(t)

    return np.array([dEdt, dLdt, dPdt, dAdt, dDdt, dTdt])

alpha = np.array([E[0], L[0], P[0], 0, 0, 0])

def objective(params):

    """Objective function with explicit parameters"""
    t_values, y_values = Chaotiq.solver('RK4', f, domain, 10000, alpha, params)

    #negative penalization
    penalty = np.sum(np.minimum(0, params))  # Adds penalty for negative params

    # Interpolate solution at data points
    interp_L = np.interp(T, t_values, y_values[:, 1])

    # Calculate error
    error = np.sum((L - interp_L) ** 2)
    return error - penalty * 1e5


bounds = [
    (0, 0.5),      # r - reasonable decay rate
    (0, 5),      # gamma - pesticide effect
    (0, 0.5),      # lambd - decay rate
    (0, 0.5),      # Sigma - Reasonable migration rate
    (0, 0.05),      # aD - reasonable Diadegma induced death rate
    (0, 0.5),      # aT - reasonable Trichogramma induced death rate
]

result = differential_evolution(objective, bounds, strategy='best1bin', tol=1e-10, maxiter=1000, disp=True)

end_params = result.x

# Graph options
folder_name = "Optimized_WaspsGraphs"
labels = ['E(t)', 'L(t)', 'P(t)', 'A(t)', 'D(t)', 'T(t)']
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Data to overlay (scatter points)
scatter_data = [(T, E), (T, L), (T, P)]
scatter_labels = ['E (Data)', 'L (Data)', 'P (Data)']

# Parameters dictionary (for annotation)
params = {
    'mE': mE, 'mL': mL, 'mP': mP, 'mA': mA,
    'aE': aE, 'aL': aL, 'aP': aP, 'mu': mu, 'mT': mT, 'mD': mD,
    'r': end_params[0],
    'gamma': end_params[1],
    'lambd': end_params[2],
    'Sigma': end_params[3],
    'aD': end_params[4],
    'aT': end_params[5],
}

# Call lsquare
lsquare(
    f=f,
    domain=domain,
    n=10000,
    alpha=alpha,
    E=E,
    L=L,
    P=P,
    T=T,
    folder_name=folder_name,
    labels=labels,
    colors=colors,
    scatter_data=scatter_data,
    scatter_labels=scatter_labels,
    params=params,
    end_params=end_params,
    data_filepath = data_filepath,
)



