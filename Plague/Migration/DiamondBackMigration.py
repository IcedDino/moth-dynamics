import numpy as np
from numba import njit, float64
import numpy.typing as npt
from scipy.optimize import differential_evolution
from Solver import Chaotiq
from Plague.DataScrap.LoadData import load_insect_data
from Plague.DataScrap.LoadTableData import load_table
from Plague.Error.ErrorFunction import lsquare
# Data

data_filepath = '../DataScrap/Data/InsectData_206.csv'
data_arrays = load_insect_data(file_path=data_filepath, from_excel=False, shift_time=7, min_day=0)

# Extract individual arrays from the dictionary
T = data_arrays['T']  # Sample days
E = data_arrays['E']  # Eggs per plant
L = data_arrays['L']  # Larvae per plant
P = data_arrays['P']  # Pupae per plant
Ti = data_arrays['Ti'] # Pesticide Applications

# Table 1
mE, mL, mP, mA, aE, aL, aP, mu, _, _ = load_table()

#Determined parameters
end_params = np.array([0.052583, 2.052609, 0.000000, 0.000000, 0.009451])

domain = np.array([0, 80])

alpha = np.array([E[0], L[0], P[0], 0])

@njit
def g(t: float64, gamma: float64, lambd: float64) -> float64:
    result = 0.0
    for day in Ti:
        if t >= day:
            result += gamma * np.exp(-lambd * (t - day))
    return result


@njit
def f(t: float64, y: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

    r, gamma, lambd, Beta, Sigma = params

    dEdt = mu * y[3] - mE * y[0] - Beta * y[0] - aE * y[0]
    dLdt = aE * y[0] - mL * y[1] - np.exp(-r * t) * g(t, gamma, lambd) * y[1] - aL * y[1]
    dPdt = aL * y[1] - mP * y[2] - Beta * y[2] - aP * y[2]
    dAdt = aP * y[2] - mA * y[3] + Sigma

    return np.array([dEdt, dLdt, dPdt, dAdt])


bounds = [
    (0, 1),      # r - reasonable decay rate
    (0, 5),      # gamma - pesticide effect
    (0, 1),      # lambd - decay rate
    (0, 1),      # beta - Wasp interaction coefficient
    (0, 1),      # Sigma - Reasonable migration rate
]

# Graph options
folder_name = "MigrationGraphs"
labels = ['E(t)', 'L(t)', 'P(t)', 'A(t)']
colors = ['b', 'g', 'r', 'c']

# Data to overlay (scatter points)
scatter_data = [(T, E), (T, L), (T, P)]
scatter_labels = ['E (Data)', 'L (Data)', 'P (Data)']

# Parameters dictionary (for annotation)
params = {
    'mE': mE, 'mL': mL, 'mP': mP, 'mA': mA,
    'aE': aE, 'aL': aL, 'aP': aP, 'mu': mu,
    'r': end_params[0],
    'gamma': end_params[1],
    'lambd': end_params[2],
    'Beta': end_params[3],
    'Sigma': end_params[4],
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



