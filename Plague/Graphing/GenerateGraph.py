import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Dict, Tuple
import os
import re


def graph_sol(
    t_values: npt.NDArray[np.float64],
    y_values: npt.NDArray[np.float64],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    scatter_data: Optional[List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = None,
    scatter_labels: Optional[List[str]] = None,
    params: Optional[Dict[str, float]] = None
) -> None:
    n_graphs = y_values.shape[1]
    n_cols = 2
    n_rows = (n_graphs + 1) // 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axs = axs.flat if n_graphs > 1 else [axs]

    default_labels = [f'Var{i}(t)' for i in range(n_graphs)]
    default_colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(n_graphs):
        ax = axs[i]
        lbl = labels[i] if labels else default_labels[i]
        clr = colors[i] if colors else default_colors[i % len(default_colors)]
        ax.plot(t_values, y_values[:, i], label=lbl, color=clr)

        if scatter_data and i < len(scatter_data):
            scatter_x, scatter_y = scatter_data[i]
            sc_label = scatter_labels[i] if scatter_labels else f'{lbl} (Data)'
            ax.scatter(scatter_x, scatter_y, color=clr, label=sc_label, marker='o', edgecolors='k')

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Population')
        ax.set_title(f'{lbl} population')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    # Parameter box
    if params:
        param_str = '\n'.join(f'{k}: {v:.6f}' for k, v in params.items())
        fig.text(0.82, 0.5, param_str, fontsize=10, va='center', ha='left', family='monospace')
        plt.tight_layout(rect=(0, 0, 0.8, 1))
    else:
        plt.tight_layout()

    plt.show()


def save_sol(
    t_values: npt.NDArray[np.float64],
    y_values: npt.NDArray[np.float64],
    folder_name: str = "Migration",
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    scatter_data: Optional[List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = None,
    scatter_labels: Optional[List[str]] = None,
    params: Optional[Dict[str, float]] = None,
    data_filepath: Optional[str] = None  # <-- New parameter
) -> None:
    os.makedirs(folder_name, exist_ok=True)

    n_graphs = y_values.shape[1]
    default_labels = [f'Var{i}(t)' for i in range(n_graphs)]
    default_colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Extract file ID from data_filepath if available
    file_id = ""
    if data_filepath:
        match = re.search(r'(\d+)(?=\D*$)', data_filepath)
        if match:
            file_id = f"_ID{match.group(1)}"

    for i in range(n_graphs):
        fig, ax = plt.subplots(figsize=(6, 4))
        lbl = labels[i] if labels else default_labels[i]
        clr = colors[i] if colors else default_colors[i % len(default_colors)]
        ax.plot(t_values, y_values[:, i], label=lbl, color=clr)

        if scatter_data and i < len(scatter_data):
            scatter_x, scatter_y = scatter_data[i]
            sc_label = scatter_labels[i] if scatter_labels else f'{lbl} (Data)'
            ax.scatter(scatter_x, scatter_y, color=clr, label=sc_label, marker='o', edgecolors='k')

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Population')
        ax.set_title(f'{lbl} population')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add file ID to graph filename
        base_name = lbl.replace("(t)", "").replace(" ", "_")
        graph_filename = f"{base_name}{file_id}_graph.svg"
        graph_path = os.path.join(folder_name, graph_filename)
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save parameters + file ID
    if params or data_filepath:
        lines = []
        if params:
            lines.extend(f'{k}: {v:.6f}' for k, v in params.items())

        if file_id:
            lines.append(f'DataFileID: {file_id[3:]}')  # strip "_ID" prefix for cleaner output

        param_file = os.path.join(folder_name, "parameters.txt")
        with open(param_file, "w") as f:
            f.write('\n'.join(lines))
            f.write('\n'.join(lines))