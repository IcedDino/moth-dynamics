from Solver import Chaotiq
from Plague.Graphing.GenerateGraph import graph_sol, save_sol
from typing import Callable, Tuple, List, Dict, Optional
import numpy as np
import os
import numpy.typing as npt
import re

def lsquare(
    f: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    domain: Tuple[float, float],
    n: int,
    alpha: npt.NDArray[np.float64],
    E: npt.NDArray[np.float64],
    L: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    folder_name: str,
    labels: List[str],
    colors: List[str],
    scatter_data: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    scatter_labels: List[str],
    params: Dict[str, float],
    end_params: npt.NDArray[np.float64],
    data_filepath: Optional[str] = None  # <-- New parameter
) -> None:
    """
    Compute the least squares error between model predictions and data.

    Parameters:
    - f: Function defining the ODE system.
    - domain: (start_time, end_time).
    - n: Number of integration steps.
    - alpha: Initial condition array.
    - E, L, P: Experimental population data.
    - T: Time points for experimental data.
    - folder_name: Directory to save results.
    - labels, colors: Plot settings for each variable.
    - scatter_data: Experimental data to overlay on plots.
    - scatter_labels: Labels for the scatter data.
    - params: Dictionary of model parameters for annotation/saving.
    """

    t_values, y_values = Chaotiq.solver('RK4', f, domain, n, alpha, end_params)

    graph_sol(t_values, y_values, labels, colors, scatter_data, scatter_labels, params)
    save_sol(t_values, y_values, folder_name, labels, colors, scatter_data, scatter_labels, params, data_filepath)

    interp_E = np.interp(T, t_values, y_values[:, 0])
    interp_L = np.interp(T, t_values, y_values[:, 1])
    interp_P = np.interp(T, t_values, y_values[:, 2])

    # Compute RMSE for larvae only
    ls = np.sqrt(np.sum((L - interp_L) ** 2))
    print(f"Larval RMSE: {ls:.6f}")

    # Compute total RMSE across E, L, P
    eps = np.sqrt(np.sum((E - interp_E) ** 2 + (L - interp_L) ** 2 + (P - interp_P) ** 2))
    print(f"Total RMSE: {eps:.6f}")

    #Strip id
    file_id = ""
    if data_filepath:
        match = re.search(r'(\d+)(?=\D*$)', data_filepath)
        if match:
            file_id = f"_ID{match.group(1)}"

    # Save RMSE results to a text file in the same folder
    rmse_text = f"Larval RMSE: {ls:.6f}\nTotal RMSE: {eps:.6f}\n"
    rmse_path = os.path.join(folder_name, f"rmse{file_id}.txt")
    with open(rmse_path, "w") as f:
        f.write(rmse_text)
