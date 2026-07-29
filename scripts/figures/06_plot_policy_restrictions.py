# -----------------------------------------------------------------------------
# 06_plot_policy_restrictions.py
#
# This script generates the summary figure for the optimal policy restriction
# values (e.g., Figure 2 in the manuscript). It processes the output files
# from five different policy run directories, extracting the final converged
# policy value (e.g., water limit, fee, cutoff) for each of the 500
# bootstrap scenarios.
#
# It then calculates the median, range (min/max), and coefficient of variation
# for each policy and creates a two-panel bar plot to visualize these
# distributions, consistent with the style of the other manuscript figures.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- 2. Set Up File Paths and Plotting Parameters ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_FOR_FIGURES_DIR = OUTPUTS_DIR / "data_for_figures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
os.makedirs(DATA_FOR_FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define paths for each policy's output directory
POLICY_PATHS = {
    "UR": OUTPUTS_DIR / "ur_runs",
    "FB-I": OUTPUTS_DIR / "fb_runs",
    "FB-II": OUTPUTS_DIR / "fb_cb_runs",
    "PR-I": OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
    "PWPR": OUTPUTS_DIR / "r_plus_pr_runs"
}

UR_RESULTS_PATH = OUTPUTS_DIR / "ur_runs" / "ur_optimal_water_limits.csv"

# Define consistent colors for each policy
COLORS = {
    "UR": "#C00000",
    "FB-I": "#FF7F0E",
    "FB-II": "#B85C00",
    "PR-I": "#4169E1",
    "PR-II": "#00BFFF",
    "PWPR": "#DAA520",
}
background_color = '#f4f4f4'
plt.rcParams['font.family'] = 'Arial'
fontsize = 22

# --- 3. Data Extraction and Calculation Functions ---
def get_policy_values(policy_name, directory):
    """Extracts policy values from filenames in a directory."""
    values = []
    if not directory.exists():
        print(f"Warning: Directory not found for {policy_name}: {directory}")
        return values
    
    patterns = {
        "UR": r"ur_b_\d+_wl_(\d+\.?\d*)\.xlsx",
        "FB-I": r"fb_b_\d+_pf_(\d+\.?\d*)\.xlsx",
        "FB-II": r"fb_cb_b_\d+_pf_(\d+\.?\d*)\.xlsx",
        "PR-I": r"pr1_b_\d+_co_(\d+\.?\d*)\.xlsx",
        "PR-II": r"pr2_b_\d+_sf_(\d+)\.xlsx",
    }
    pattern = re.compile(patterns[policy_name])
    
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            values.append(float(match.group(1)))
    return values

def calculate_r_plus_pr_stats(ur_results_path):
    """
    Performs the detailed calculation for the PWPR policy by first extracting
    UR limits directly from the UR run filenames.
    """
    if not ur_results_path.exists():
        print(f"Warning: UR results file not found at {ur_results_path}")
        return pd.DataFrame()
        
    num_farmers = 254
    pf_values = np.linspace(1.0, 0.7, num_farmers)
    pf_dict = {}
    total_shares = 0
    for i in range(num_farmers):
        seniority_id = i + 1
        priority_factor = round(pf_values[i], 4)
        unrestricted_num_shares = round(24 * priority_factor, 4)
        pf_dict[seniority_id] = {"unrestricted_num_shares": unrestricted_num_shares}
        total_shares += unrestricted_num_shares

    water_limits_df = pd.read_csv(ur_results_path)
    water_limits_df["inches_per_share"] = (water_limits_df["Final_Water_Limit_Inches"] * num_farmers) / total_shares

    bootstrap_limits = []
    for _, row in water_limits_df.iterrows():
        farmer_limits = [
            round(vals["unrestricted_num_shares"] * row["inches_per_share"], 4)
            for sid, vals in pf_dict.items()
        ]
        bootstrap_limits.append({
            "min_water_limit": min(farmer_limits),
            "median_water_limit": np.median(farmer_limits),
            "max_water_limit": max(farmer_limits)
        })
    return pd.DataFrame(bootstrap_limits)

def get_stats(values):
    """Calculates median, min, max, 95% CI, and CV for a list or pandas Series."""
    if len(values) < 2:
        return [np.nan] * 6
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    lower_ci = np.percentile(values, 2.5)
    upper_ci = np.percentile(values, 97.5)
    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    return median, min_val, max_val, lower_ci, upper_ci, cv

# --- 4. Main Data Processing ---
print("Processing results from all policy runs...")
# Conversion factors
INCH_TO_M = 0.0254
ACRE_FOOT_TO_M3 = 1233.48
HA_TO_M2 = 10000
FIELD_AREA_HA = 50
VOLUME_SCALE = 1e4

# Process Panel A policies
ur_vals = get_policy_values("UR", POLICY_PATHS["UR"])
ur_vol = [v * INCH_TO_M * (FIELD_AREA_HA * HA_TO_M2) / VOLUME_SCALE for v in ur_vals]

fb_vals = get_policy_values("FB-I", POLICY_PATHS["FB-I"])
fb_fee = [v / ACRE_FOOT_TO_M3 for v in fb_vals]

fb_cb_vals = get_policy_values("FB-II", POLICY_PATHS["FB-II"])
fb_cb_fee = [v / ACRE_FOOT_TO_M3 for v in fb_cb_vals]

pr1_vals = get_policy_values("PR-I", POLICY_PATHS["PR-I"])
pr1_co = [v * 1 for v in pr1_vals]

pr2_vals = get_policy_values("PR-II", POLICY_PATHS["PR-II"])

panel_a_stats = {
    "UR": get_stats(ur_vol),
    "PR-I": get_stats(pr1_co),
    "PR-II": get_stats(pr2_vals),
    "FB-I": get_stats(fb_fee),
    "FB-II": get_stats(fb_cb_fee),
}

# Process Panel B policy (PWPR)
r_plus_pr_df = calculate_r_plus_pr_stats(UR_RESULTS_PATH)
if r_plus_pr_df.empty:
    raise RuntimeError(
        f"PWPR stats could not be computed because UR results were missing or empty "
        f"(expected at {UR_RESULTS_PATH})."
    )
    
r_plus_pr_df_converted = r_plus_pr_df.copy()
for col in r_plus_pr_df.columns:
    r_plus_pr_df_converted[col] = r_plus_pr_df[col] * INCH_TO_M * (FIELD_AREA_HA * HA_TO_M2) / VOLUME_SCALE

panel_b_stats = {
    "Minimum Water Limit": get_stats(r_plus_pr_df_converted["min_water_limit"]),
    "Median Water Limit": get_stats(r_plus_pr_df_converted["median_water_limit"]),
    "Maximum Water Limit": get_stats(r_plus_pr_df_converted["max_water_limit"])
}

# --- 5. Save Summary Statistics to CSV ---
summary_data = []
for policy, (median, min_v, max_v, lower_ci, upper_ci, cv) in panel_a_stats.items():
    summary_data.append({"Policy": policy, "Metric": "Value", "Median": median, "Min": min_v, "Max": max_v, "95_CI_Lower": lower_ci, "95_CI_Upper": upper_ci, "CV": cv})
for metric, (median, min_v, max_v, lower_ci, upper_ci, cv) in panel_b_stats.items():
    summary_data.append({"Policy": "PWPR", "Metric": metric, "Median": median, "Min": min_v, "Max": max_v, "95_CI_Lower": lower_ci, "95_CI_Upper": upper_ci, "CV": cv})

summary_df = pd.DataFrame(summary_data)
summary_path = DATA_FOR_FIGURES_DIR / "policy_restrictions_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Summary statistics saved to: {summary_path}")

# --- 6. Plotting ---
print("Generating policy restrictions figure...")
fig = plt.figure(figsize=(16.5, 13), dpi=600)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.6)

# --- Panel A ---
ax_a = fig.add_subplot(gs[0])
gs_a = ax_a.get_subplotspec().subgridspec(1, 5, wspace=3.5)
ax_a.set_axis_off()
policies_a = ["UR", "PR-I", "PR-II", "FB-I", "FB-II"]

ylabels_a = [
    r"Individual Pumping Limit" + "\n" + r"($10^4\,m^3$)",
    r"Regional Pumping Limit" + "\n" + r"($10^4\,m^3$)",
    r"Number of Senior" + "\n" + r"Farmers Permitted" "\n" + r"to Pump (-)",
    r"Pumping Fee" + "\n" + r"($ per m³)",
    r"Puming Fee" + "\n" + r"($ per m³)",
]

# Use shared y-axis limits for FB-I and FB-II because both are pumping fees
fb_fee_values_for_ylim = []

for policy in ["FB-I", "FB-II"]:
    median, _, _, lower_ci, upper_ci, _ = panel_a_stats[policy]
    fb_fee_values_for_ylim.extend([lower_ci, upper_ci, median])

fb_fee_ymax = max(fb_fee_values_for_ylim) * 1.15
fb_fee_ylim = (0, fb_fee_ymax)

for i, (policy, ylabel) in enumerate(zip(policies_a, ylabels_a)):
    ax = fig.add_subplot(gs_a[i])
    median, _, _, lower_ci, upper_ci, cv = panel_a_stats[policy]
    ax.bar(0, median, width=0.5, color=COLORS[policy], align='center')
    ax.errorbar(0, median, yerr=[[median - lower_ci], [upper_ci - median]], fmt='o', color='black', capsize=8)
    ax.set_title(policy, fontsize=fontsize, pad=20)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_facecolor(background_color)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks([])
    
    if policy in ["FB-I", "FB-II"]:
        ax.set_ylim(fb_fee_ylim)
        
    # --- TEXT FORMATTING ---
    if policy == "UR":
        text_str = f"Median: {median:.2f} " + r"$\times 10^4\,m^3$" + f"\nCV: {cv:.2f}"
    elif policy == "PR-I":
            text_str = f"Median: {median:.0f} " + r"$\times 10^4\,m^3$" + f"\nCV: {cv:.2f}"
    elif policy == "PR-II":
        text_str = f"Median: {median:.0f}\nCV: {cv:.2f}"
    elif policy in ["FB-I", "FB-II"]:
        text_str = f"Median: ${median:.2f} per m³\nCV: {cv:.2f}"

    ax.text(0.5, -0.15, text_str, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize - 2)

ax_a.text(-0.13, 1.1, "(a)", transform=ax_a.transAxes, fontsize=fontsize, fontweight='bold')

# --- Panel B ---
ax_b = fig.add_subplot(gs[1])
gs_b = ax_b.get_subplotspec().subgridspec(1, 5, wspace=3.5)
ax_b.set_axis_off()
titles_b = ["Minimum Individual Pumping\nAllocation for PWPR", "Median Individual Pumping\nAllocation for PWPR", "Maximum Individual Pumping\nAllocation for PWPR"]
stats_b = list(panel_b_stats.values())

b_cols = [0, 2, 4]

for i, (title, stats) in enumerate(zip(titles_b, stats_b)):
    ax = fig.add_subplot(gs_b[b_cols[i]])
    median, _, _, lower_ci, upper_ci, cv = stats
    ax.bar(0, median, width=0.5, color=COLORS["PWPR"], align='center')
    ax.errorbar(0, median, yerr=[[median - lower_ci], [upper_ci - median]], fmt='o', color='black', capsize=8)
    ax.set_title(title, fontsize=fontsize, pad=20)
    ax.set_ylabel(r'Individual Pumping Allocation' + '\n' + r'($10^4\,m^3$)', fontsize=fontsize)
    ax.set_facecolor(background_color)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks([])
    # --- TEXT FORMATTING ---
    text_str = f"Median: {median:.2f} " + r"$\times 10^4\,m^3$" + f"\nCV: {cv:.2f}"
    ax.text(0.5, -0.15, text_str, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize - 2)

ax_b.text(-0.13, 1, "(b)", transform=ax_b.transAxes, fontsize=fontsize, fontweight='bold')

# --- 7. Save Figure ---
save_path = FIGURES_DIR / "policy_restrictions_summary.png"
plt.savefig(save_path, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to: {save_path}")