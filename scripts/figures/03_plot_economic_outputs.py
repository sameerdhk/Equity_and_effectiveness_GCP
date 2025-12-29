# -----------------------------------------------------------------------------
# 03_plot_economic_outputs.py
#
# This script generates the final economic outcomes figure (Figure ) for the 
# manuscript, based on the consolidated median data. It produces a two-panel 
# time-series plot showing:
#   1. Meadin of Average Profit Among Farmers
#   2. Median of Average Profit per Unit of Applied Water
#
# The script is designed to precisely replicate the formatting of the original
# plotting code to ensure publication-quality figures.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- 2. Set Up Plotting Parameters ---
# Set figure width based on manuscript standards
figwd = {"1.5": 7.5}

# Set default plot parameters for a consistent look
fontsize = 16
plt.rcParams['font.size'] = fontsize
plt.rcParams['font.family'] = 'Arial'

# Define consistent colors and styles for each policy
POLICY_STYLES = {
    "BAU":  {"color": "#808080", "linestyle": "-", "lw": 2, "zorder": 2},
    "UR":   {"color": "#C00000", "linestyle": "-", "lw": 2.5, "zorder": 4},
    "FB":   {"color": "#FF7F0E", "linestyle": "-", "lw": 2, "zorder": 2},
    "PR-I": {"color": "#4169E1", "linestyle": ":", "lw": 2, "zorder": 5},
    "PR-II":{"color": "#00BFFF", "linestyle": "-", "lw": 2, "zorder": 2},
    "R+PR": {"color": "#DAA520", "linestyle": (0, (5, 5)), "lw": 3, "zorder": 5},
}
background_color = '#F0F0F0'


# --- 3. Set Up File Paths and Load Data ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

DATA_PATH = PROJECT_ROOT / "outputs" / "data_for_figures" / "economic_outcomes.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the processed data
data = pd.read_csv(DATA_PATH)


# --- 4. Plotting Function ---
def plot_economic_outcomes(df, save_path=None):
    """
    Generates the Economic Outcomes figure (e.g., Figure 4 in manuscript).
    This function is adapted from the user's original code to ensure exact formatting.
    """
    fig = plt.figure(figsize=(figwd["1.5"], figwd["1.5"] * 0.9), dpi=600)
    axes = []
    locator = ticker.MaxNLocator(integer=True)
    ylabel_xloc = -0.12

    # Map the column names in the CSV to the plot keys and labels
    metrics = [
        {'key': 'avg_profit', 'ylabel': '(a) Average Profit\namong farmers\n($\$ 10^4$)'},
        {'key': 'profit_per_water', 'ylabel': '(b) Average Profit\nper applied water\n($\$ 10^4$ per cm)'},
    ]

    num_metrics = len(metrics)
    subplot_height = 1 / num_metrics - 0.0065

    for i, metric in enumerate(metrics):
        key, ylabel = metric['key'], metric['ylabel']
        position = [0, 1 - (i + 1) * (subplot_height + 0.03), 1, subplot_height]
        ax = fig.add_axes(position)
        axes.append(ax)
        ax.xaxis.set_major_locator(locator)
        ax.set_xlim([START_YEAR, END_YEAR])
        ax.set_facecolor(background_color)
        
        for policy_name, policy_df in df.groupby('Policy'):
            style = POLICY_STYLES.get(policy_name, {})
            ax.plot(policy_df['Year'], policy_df[key], **style, label=policy_name)

        ax.set_ylabel(ylabel, fontsize=24)
        ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
        ax.tick_params(axis='y', labelsize=22)
        
        if key != 'profit_per_water':
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Year', fontsize=24)
            ax.tick_params(axis='x', labelsize=22, rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    order = {label: i for i, label in enumerate(POLICY_STYLES.keys())}
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: order.get(t[1], 99)))
    
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15), fontsize=24)

    if save_path:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close(fig)

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    if DATA_PATH.exists():
        print(f"\nLoading data from {DATA_PATH}...")
        all_data = pd.read_csv(DATA_PATH)
        # Ensure the 'Policy' column is treated as a categorical type with the correct order
        all_data['Policy'] = pd.Categorical(all_data['Policy'], categories=POLICY_STYLES.keys(), ordered=True)
        all_data.sort_values(['Policy', 'Year'], inplace=True)

        START_YEAR = all_data['Year'].min()
        END_YEAR = all_data['Year'].max()

        # Plot Economic Outcomes
        print("Generating Economic Outcomes figure...")
        save_path_econ = FIGURES_DIR / "economic_outcomes.png"
        plot_economic_outcomes(all_data, save_path=save_path_econ)
        print(f"Figure saved to: {save_path_econ}")

    else:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run '01_prepare_data_for_figures.py' first.")