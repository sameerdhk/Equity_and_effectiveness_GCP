# -----------------------------------------------------------------------------
# 05_plot_lorenz_curves.py
#
# This script generates the final Lorenz curve figure for the manuscript
# (e.g., Figure 6). It uses the detailed, farmer-level profit data from the
# individual policy CSV files to compare the equity of profit distribution
# for farmers who are equipped for irrigation.
#
# The script calculates the Gini coefficient for each policy and displays it
# in the legend. The formatting is designed to be consistent with the other
# figures in the project.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm

# --- 2. Gini and Lorenz Helper Functions ---
def gini_coefficient(arr):
    """
    Gini coefficient (classical formula) with profits scaled by 1e4.
    Works with signed values but returns NaN if total profit <= 0.
    """
    arr = np.asarray(arr) * 1e4  # scale from 10^4 units to $ units
    sorted_arr = np.sort(arr)
    total = np.sum(sorted_arr)
    n = arr.size
    if total <= 0:
        return np.nan  # Gini undefined if total profit <= 0
    index = np.arange(1, n+1)
    rank_sum = np.sum((n + 1 - index) * sorted_arr)
    gini = (n + 1 - 2 * rank_sum / total) / n
    return gini


def lorenz_curve(values):
    """
    Lorenz curve coordinates with profits scaled by 1e4.
    Note: if profits include negatives, cumulative shares may dip below 0.
    """
    values = np.asarray(values) * 1e4  # scale from 10^4 units to $ units
    x = np.sort(values)
    cumvals = np.cumsum(x)
    cumvals = np.insert(cumvals, 0, 0)
    cumvals = cumvals / cumvals[-1] if cumvals[-1] != 0 else cumvals
    x_vals = np.linspace(0, 1, len(cumvals))
    return x_vals, cumvals

# --- 3. Set Up Paths and Plotting Parameters ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

DATA_DIR = PROJECT_ROOT / "outputs" / "data_for_figures"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

POLICY_STYLES = {
    "BAU":  {"color": "#808080", "linestyle": "-", "lw": 2, "zorder": 2},
    "UR":   {"color": "#C00000", "linestyle": "-", "lw": 2.5, "zorder": 4},
    "FB-I":   {"color": "#FF7F0E", "linestyle": "-", "lw": 2, "zorder": 2},
    "FB-II": {"color": "#B85C00", "linestyle": "--", "lw": 2.5, "zorder": 6},
    "PR-I": {"color": "#4169E1", "linestyle": ":", "lw": 2, "zorder": 5},
    "PR-II":{"color": "#00BFFF", "linestyle": "-", "lw": 2, "zorder": 2},
    "PWPR": {"color": "#DAA520", "linestyle": (0, (5, 5)), "lw": 3, "zorder": 5},
}
background_color = '#F0F0F0'
plt.rcParams['font.family'] = 'Arial'

# --- 4. Load and Prepare Data ---
print(f"Loading data from: {DATA_DIR}")
file_paths = list(DATA_DIR.glob("profit_distribution_*.csv"))

if not file_paths:
    raise FileNotFoundError(f"Data files not found in {DATA_DIR}. Please run '01_prepare_data_for_figures.py' first.")

# Read and combine all individual policy CSVs into one DataFrame
all_data = pd.concat([pd.read_csv(fp) for fp in tqdm(file_paths, desc="Loading data files")], ignore_index=True)

# Filter for only farmers equipped for irrigation
all_data_irrigated = all_data[all_data["field_type_rn"] == "optimize"].copy()

# Calculate average profit per farmer for each bootstrap run
avg_profit_irrigated = (
    all_data_irrigated.groupby(["Policy", "Bootstrap", "AgentID"])
    .agg(avg_profit=("profit", "mean"))
    .reset_index()
)

# --- 5. Generate Lorenz Curve Plot ---
print("Generating Lorenz Curve plot...")

fig = plt.figure(figsize=(10, 8), dpi=800)
ax = fig.add_subplot(111)

gini_summary = []
policy_order = list(POLICY_STYLES.keys())

for policy in policy_order:
    policy_df = avg_profit_irrigated[avg_profit_irrigated['Policy'] == policy]
    if policy_df.empty:
        continue
    
    # For Lorenz curve, we need the distribution of average profits across ALL bootstrap runs
    profits = policy_df["avg_profit"].dropna().values
    
    if len(profits) > 0:
        x, y = lorenz_curve(profits)
        gini = gini_coefficient(profits)
        gini_summary.append({"Policy": policy, "Gini Coefficient": round(gini, 3)})
        
        style = POLICY_STYLES.get(policy, {})
        ax.plot(x, y, label=f"{policy} (Gini: {gini:.3f})", **style)

# Plot lines for perfect equality and inequality
ax.plot([0, 1], [0, 1], 'k--', label="Perfect Equality (Gini: 0.0)")
ax.plot([0, 1, 1], [0, 0, 1], color='red', linestyle='--', label="Perfect Inequality (Gini: 1.0)")

# --- 6. Formatting and Saving ---
# ax.set_title("Average Yearly Profit Among Farmers Equipped for Irrigation", fontsize=20, fontweight='bold')
ax.set_xlabel("Cumulative Share of Farmers", fontsize=20)
ax.set_ylabel("Cumulative Share of Profit", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax.set_aspect('equal', adjustable='box')
ax.set_facecolor(background_color)

handles, labels = ax.get_legend_handles_labels()
order = {label.split(' ')[0]: i for i, label in enumerate(policy_order)}
handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: order.get(t[1].split(' ')[0], 99)))

fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15), fontsize=20)

plt.tight_layout()
save_path = FIGURES_DIR / "lorenz_curve.png"
plt.savefig(save_path, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to: {save_path}")

# --- 7. Print Gini Summary Table ---
if gini_summary:
    gini_df = pd.DataFrame(gini_summary).set_index('Policy').reindex(policy_order).reset_index()
    print("\n--- Gini Coefficient Summary ---")
    print(gini_df.to_string(index=False))