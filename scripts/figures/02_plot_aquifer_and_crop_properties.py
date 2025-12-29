# -----------------------------------------------------------------------------
# 02_plot_aquifer_and_crop_properties.py
#
# This script generates the final figures for the manuscript based on the
# consolidated median data created by the previous script. It produces two
# main figures:
#   1. Aquifer Properties (Figure XX): A time-series plot showing Saturated Thickness,
#      Water Level Change, and Total Withdrawal for all policies.
#   2. Crop Ratios (Figure XX): A time-series plot showing the proportion of land
#      allocated to each crop type for all policies.
#
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

# --- 2. Set Up Plotting Parameters ---
# Set figure width based on manuscript standards
figwd = {"1": 140 / 1.5 / 25.4, "1.5": 7.5, "2": 140 / 1.5 / 25.4 * 2, "3": 8.5}

# Set default plot parameters for a consistent look
fontsize = 16
plt.rcParams['font.size'] = fontsize
plt.rcParams['font.family'] = 'Arial'

# Define consistent colors and styles for each policy, matching the original script
# The keys are the short policy names from the 'Policy' column in the CSV
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

DATA_PATH = PROJECT_ROOT / "outputs" / "data_for_figures" / "aquifer_and_crop_properties.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the processed data
data = pd.read_csv(DATA_PATH)


# --- 4. Plotting Functions ---
def plot_aquifer_properties(df, save_path=None):
    """
    Generates the Aquifer Properties figure (e.g., Figure 2 in manuscript).
    This function is adapted from the user's original code to ensure exact formatting.
    """
    fig = plt.figure(figsize=(figwd["1.5"], figwd["1.5"] * 1.3), dpi=600)
    axes = []
    locator = ticker.MaxNLocator(integer=True)
    ylabel_xloc = -0.12

    metrics = [
        {'key': 'GW_st', 'ylabel': '(a) Saturated\nthickness (m)', 'interval': 0.1, 'y_start': 21.0, 'y_end': 24.8},
        {'key': 'GW_st_change', 'ylabel': '(b) Water Level\nchange (m)', 'interval': 0.05, 'y_start': -1, 'y_end': 1},
        {'key': 'withdrawal', 'ylabel': '(c) Withdrawal\n($10^4$ $m^3$)', 'interval': 1000, 'y_start': 0, 'y_end': 4000}
    ]

    num_metrics = len(metrics)
    subplot_height = 1 / num_metrics - 0.005

    for i, metric in enumerate(metrics):
        key, ylabel = metric['key'], metric['ylabel']
        position = [0, 1 - (i + 1) * (subplot_height + 0.025), 1, subplot_height]
        ax = fig.add_axes(position)
        axes.append(ax)
        ax.xaxis.set_major_locator(locator)
        ax.set_xlim([START_YEAR, END_YEAR])
        ax.set_facecolor(background_color)
        
        for policy_name, policy_df in df.groupby('Policy'):
            style = POLICY_STYLES.get(policy_name, {})
            ax.plot(policy_df['Year'], policy_df[key], **style, label=policy_name)
        
        if key == 'GW_st_change':
            ax.axhline(0, color='black', linestyle='--', lw=0.8)

        ax.set_ylabel(ylabel, fontsize=24)
        ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
        ax.tick_params(axis='y', labelsize=22)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
        
        if key != 'withdrawal':
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Year', fontsize=24)
            ax.tick_params(axis='x', labelsize=22, rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    order = {label: i for i, label in enumerate(POLICY_STYLES.keys())}
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: order.get(t[1], 99)))
    
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.4, 1.12), fontsize=24)
    plt.tight_layout()
    
    if save_path:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def plot_crop_ratios(df, save_path=None):
    """
    Generates the Crop Ratios figure (e.g., Figure 3 in manuscript).
    This function is adapted from the user's original code to ensure exact formatting.
    """
    fig = plt.figure(figsize=(figwd["1.5"], figwd["1.5"] * 1.6), dpi=600)
    axes = []
    locator = ticker.MaxNLocator(integer=True)
    ylabel_xloc = -0.12

    crops_to_plot = ["corn", "sorghum", "soybeans", "wheat", "fallow"]
    labels = ["(a) Corn", "(b) Sorghum", "(c) Soybeans", "(d) Wheat", "(e) Fallow"]
    
    for i, (crop, label) in enumerate(zip(crops_to_plot, labels)):
        position = [0, 1 - i / 4, 1, 1 / 4 - 0.025]
        ax = fig.add_axes(position)
        axes.append(ax)
        ax.set_xlim([START_YEAR, END_YEAR])
        ax.set_facecolor(background_color)
        
        for policy_name, policy_df in df.groupby('Policy'):
            style = POLICY_STYLES.get(policy_name, {})
            ax.plot(policy_df['Year'], policy_df[crop], **style, label=policy_name)
        
        ax.set_ylabel(label, fontsize=24)
        ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
        ax.tick_params(axis='y', labelsize=22)

        if crop != "fallow":
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=24)
            ax.xaxis.set_major_locator(locator)
            ax.tick_params(axis='x', labelsize=22, rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    order = {label: i for i, label in enumerate(POLICY_STYLES.keys())}
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: order.get(t[1], 99)))
    
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.4, 1.42), fontsize=24)

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

        # Plot Aquifer Properties
        print("Generating Aquifer Properties figure...")
        save_path_aq = FIGURES_DIR / "aquifer_properties.png"
        plot_aquifer_properties(all_data, save_path=save_path_aq)
        print(f"Figure saved to: {save_path_aq}")

        # Plot Crop Ratios
        print("\nGenerating Crop Ratios figure...")
        save_path_cr = FIGURES_DIR / "crop_ratios.png"
        plot_crop_ratios(all_data, save_path=save_path_cr)
        print(f"Figure saved to: {save_path_cr}")

    else:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run '01_prepare_data_for_figures.py' first.")