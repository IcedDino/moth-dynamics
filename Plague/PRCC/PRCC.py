import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, rankdata, pearsonr
from sklearn.linear_model import LinearRegression

# Import your existing models
from Plague.LevereBresnahan.LevereBresnahan import f as levere_f, end_params as levere_baseline
from Plague.Migration.DiamondBackMigration import f as diamond_f, end_params as diamond_baseline
from Plague.Wasps.Wasps import f as wasps_f, end_params as wasps_baseline
from Solver import Chaotiq


# PRCC Functions
def latin_hypercube_sampling(bounds, N):
    q = len(bounds)
    lhs = np.zeros((N, q))
    for i, (l, u) in enumerate(bounds):
        cut = np.linspace(0, 1, N + 1)
        u01 = np.random.uniform(low=cut[:-1], high=cut[1:], size=N)
        np.random.shuffle(u01)
        lhs[:, i] = l + u01 * (u - l)
    return lhs


def compute_model_outputs(param_matrix, model_func):
    return np.array([model_func(params) for params in param_matrix])


def partial_rank_correlation(L, Y):
    N, q = L.shape
    ranked_L = np.apply_along_axis(rankdata, 0, L)
    ranked_Y = rankdata(Y)

    prccs = []
    pvals = []

    for i in range(q):
        # Regress parameter i on all other parameters
        X_i = np.delete(ranked_L, i, axis=1)
        model_L = LinearRegression().fit(X_i, ranked_L[:, i])
        res_L = ranked_L[:, i] - model_L.predict(X_i)

        # Regress output on all parameters except i
        model_Y = LinearRegression().fit(X_i, ranked_Y)
        res_Y = ranked_Y - model_Y.predict(X_i)

        prcc, pval = pearsonr(res_L, res_Y)
        prccs.append(prcc)
        pvals.append(pval)

    return prccs, pvals


def run_prcc_analysis(bounds, model_func, N=1000, repeats=100):
    q = len(bounds)
    assert N > (4 * q) / 3, "N must be greater than 4/3*q for accuracy."

    all_prccs = []
    all_pvals = []

    for _ in range(repeats):
        L = latin_hypercube_sampling(bounds, N)
        Y = compute_model_outputs(L, model_func)
        prccs, pvals = partial_rank_correlation(L, Y)
        all_prccs.append(prccs)
        all_pvals.append(pvals)

    return np.array(all_prccs), np.array(all_pvals)


# Wrapper functions for model evaluations
def evaluate_levere_model(params):
    # Setup initial conditions from LevereBresnahan.py
    from Plague.LevereBresnahan.LevereBresnahan import domain, alpha, T, L

    alpha[3] = params[0]

    # Solve the ODE system
    t_values, y_values = Chaotiq.solver('RK4', levere_f, domain, 10000, alpha, params)

    # Calculate model fitness (sum of squared errors)
    interp_L = np.interp(T, t_values, y_values[:, 1])

    # Compute RMSE for larvae only
    ls = np.sqrt(np.sum((L - interp_L) ** 2))

    return ls


def evaluate_diamond_model(params):
    # Setup initial conditions from LevereBresnahan.py
    from Plague.Migration.DiamondBackMigration import domain, alpha, T, L

    # Solve the ODE system
    t_values, y_values = Chaotiq.solver('RK4', diamond_f, domain, 10000, alpha, params)

    # Calculate model fitness (sum of squared errors)
    interp_L = np.interp(T, t_values, y_values[:, 1])

    # Compute RMSE for larvae only
    ls = np.sqrt(np.sum((L - interp_L) ** 2))

    return ls


def evaluate_wasps_model(params):
    # Setup initial conditions from LevereBresnahan.py
    from Plague.Wasps.Wasps import domain, alpha, T, L

    # Solve the ODE system
    t_values, y_values = Chaotiq.solver('RK4', wasps_f, domain, 10000, alpha, params)

    # Calculate model fitness (sum of squared errors)
    interp_L = np.interp(T, t_values, y_values[:, 1])

    # Compute RMSE for larvae only
    ls = np.sqrt(np.sum((L - interp_L) ** 2))

    return ls

# Define parameter bounds for each model
levere_bounds = [(levere_baseline[0] * 0.2, levere_baseline[0] * 0.8),  # A0
                 (levere_baseline[1] * 0.2, levere_baseline[1] * 0.8),  # r
                 (levere_baseline[2] * 0.2, levere_baseline[2] * 0.8),  # gamma
                 (levere_baseline[3] * 0.2, levere_baseline[3] * 0.8),  # lambda
                 (levere_baseline[4] * 0.2, levere_baseline[4] * 0.8)]  # Beta

diamond_bounds = [(diamond_baseline[0] * 0.2, diamond_baseline[0] * 0.8),  # r
                  (diamond_baseline[1] * 0.2, diamond_baseline[1] * 0.8),  # gamma
                  (diamond_baseline[2] * 0.2 + 0.0000001, diamond_baseline[2] * 0.8 + 0.0001),  # lambda (adjusted to avoid 0)
                  (diamond_baseline[3] * 0.2 + 0.0000001, diamond_baseline[3] * 0.8 + 0.0001),  # Beta (adjusted to avoid 0)
                  (diamond_baseline[4] * 0.2, diamond_baseline[4] * 0.8)]  # Sigma

wasps_bounds = [(wasps_baseline[0] * 0.2, wasps_baseline[0] * 0.8),  # r
                (wasps_baseline[1] * 0.2, wasps_baseline[1] * 0.8),  # gamma
                (wasps_baseline[2] * 0.2 + 0.00000001, wasps_baseline[2] * 0.8 + 0.0001),  # lambda
                (wasps_baseline[3] * 0.2, wasps_baseline[3] * 0.8),  # sigma
                (wasps_baseline[4] * 0.2, wasps_baseline[4] * 0.8),  # aD
                (wasps_baseline[5] * 0.2, wasps_baseline[5] * 0.8)]  # aT


# Run PRCC analysis and plot results
def run_analysis_and_plot(model_name, param_bounds, model_func, param_names, N=1000, repeats=10):
    print(f"Running PRCC analysis for {model_name} model...")
    prcc_results, pval_results = run_prcc_analysis(param_bounds, model_func, N=N, repeats=repeats)

    avg_prccs = np.mean(prcc_results, axis=0)
    avg_pvals = np.mean(pval_results, axis=0)

    # Print results
    print(f"\n{model_name} PRCC Results:")
    print("Parameter\tPRCC\t\tp-value")
    print("-" * 40)
    for i, param in enumerate(param_names):
        print(f"{param}\t\t{avg_prccs[i]:.6f}\t{avg_pvals[i]:.6f}")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    colors = ['blue' if p < 0.05 else 'gray' for p in avg_pvals]
    bars = plt.bar(param_names, avg_prccs, color=colors)

    # Add significance asterisks
    for i, p in enumerate(avg_pvals):
        if p < 0.001:
            plt.text(i, avg_prccs[i] + 0.05 * np.sign(avg_prccs[i]), '***',
                     ha='center', va='center', fontweight='bold')
        elif p < 0.01:
            plt.text(i, avg_prccs[i] + 0.05 * np.sign(avg_prccs[i]), '**',
                     ha='center', va='center', fontweight='bold')
        elif p < 0.05:
            plt.text(i, avg_prccs[i] + 0.05 * np.sign(avg_prccs[i]), '*',
                     ha='center', va='center', fontweight='bold')

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.ylabel('PRCC Value')
    plt.title(f'Partial Rank Correlation Coefficients for {model_name} Model')
    plt.ylim(-1.1, 1.1)

    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='p < 0.05 (significant)'),
        Patch(facecolor='gray', edgecolor='gray', label='p ≥ 0.05 (not significant)'),
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(f'{model_name}_prcc_analysis.png', dpi=300)
    plt.show()

    return avg_prccs, avg_pvals


# Run analysis for each model
levere_param_names = ['A0', 'r', 'gamma', 'lambda', 'Beta']
levere_prccs, levere_pvals = run_analysis_and_plot(
    'Levere-Bresnahan', levere_bounds, evaluate_levere_model, levere_param_names
)

diamond_param_names = ['r', 'gamma', 'lambda', 'Beta', 'Sigma']
diamond_prccs, diamond_pvals = run_analysis_and_plot(
    'DiamondBack-Migration', diamond_bounds, evaluate_diamond_model, diamond_param_names
)

wasps_param_names = ['r', 'gamma', 'lambda', 'sigma', 'aD', 'aT']
wasps_prccs, wasps_pvals = run_analysis_and_plot(
    'Wasps', wasps_bounds, evaluate_wasps_model, wasps_param_names
)

# Create a combined plot for comparison of common parameters
plt.figure(figsize=(15, 8))

# Plot parameters that are common to all three models
common_params = ['r', 'gamma', 'lambda']
x = np.arange(len(common_params))
width = 0.25

# Get indices for common parameters in each model
levere_indices = [1, 2, 3]  # r, gamma, lambda in Levere model
diamond_indices = [0, 1, 2]  # r, gamma, lambda in Diamond model
wasps_indices = [0, 1, 2]  # r, gamma, lambda in Wasps model

plt.bar(x - width, [levere_prccs[i] for i in levere_indices],
        width, label='Levere-Bresnahan', color='skyblue')
plt.bar(x, [diamond_prccs[i] for i in diamond_indices],
        width, label='DiamondBack-Migration', color='orange')
plt.bar(x + width, [wasps_prccs[i] for i in wasps_indices],
        width, label='Wasps', color='green')

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.ylabel('PRCC Value')
plt.title('Comparison of Common Parameters Across Models')
plt.xticks(x, common_params)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig('combined_prcc_comparison.png', dpi=300)
plt.show()

# Print a summary comparing parameters across models
print("\nSummary of Common Parameters Across Models:")
print("Parameter\tLevere PRCC\tDiamond PRCC\tWasps PRCC")
print("-" * 60)
print(f"r\t\t{levere_prccs[1]:.6f}\t{diamond_prccs[0]:.6f}\t{wasps_prccs[0]:.6f}")
print(f"gamma\t\t{levere_prccs[2]:.6f}\t{diamond_prccs[1]:.6f}\t{wasps_prccs[1]:.6f}")
print(f"lambda\t\t{levere_prccs[3]:.6f}\t{diamond_prccs[2]:.6f}\t{wasps_prccs[2]:.6f}")