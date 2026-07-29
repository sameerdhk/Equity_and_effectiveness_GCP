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
    "FB-I":   {"color": "#FF7F0E", "linestyle": "-", "lw": 2, "zorder": 2},
    "FB-II": {"color": "#B85C00", "linestyle": "--", "lw": 2.5, "zorder": 6},
    "PR-I": {"color": "#4169E1", "linestyle": ":", "lw": 2, "zorder": 5},
    "PR-II":{"color": "#00BFFF", "linestyle": "-", "lw": 2, "zorder": 2},
    "PWPR": {"color": "#DAA520", "linestyle": (0, (5, 5)), "lw": 3, "zorder": 5},
}
background_color = '#F0F0F0'

# Choose which percentile/statistic to plot: p10, p25, p50, p75, or p90
STAT_NAME = "p50"
STAT_SUFFIX = f"_{STAT_NAME}"

def stat_col(base_name):
    return f"{base_name}{STAT_SUFFIX}"

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
        {'key': 'GW_st_change', 'ylabel': '(b) Water level\nchange (m)', 'interval': 0.05, 'y_start': -1, 'y_end': 1},
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
            plot_key = stat_col(key)
            ax.plot(policy_df['Year'], policy_df[plot_key], **style, label=policy_name)
        
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
    
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.45, 1.15), fontsize=24)
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
            plot_crop = stat_col(crop)
            ax.plot(policy_df['Year'], policy_df[plot_crop], **style, label=policy_name)
        
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
        required_cols = [
            stat_col("GW_st"),
            stat_col("GW_st_change"),
            stat_col("withdrawal"),
            stat_col("corn"),
            stat_col("sorghum"),
            stat_col("soybeans"),
            stat_col("wheat"),
            stat_col("fallow"),
        ]
        
        missing_cols = [c for c in required_cols if c not in all_data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected columns for STAT_NAME={STAT_NAME}: "
                + ", ".join(missing_cols)
            )
        # Ensure the 'Policy' column is treated as a categorical type with the correct order
        all_data['Policy'] = pd.Categorical(all_data['Policy'], categories=POLICY_STYLES.keys(), ordered=True)
        all_data.sort_values(['Policy', 'Year'], inplace=True)

        START_YEAR = all_data['Year'].min()
        END_YEAR = all_data['Year'].max()

        # Plot Aquifer Properties
        print("Generating Aquifer Properties figure...")
        save_path_aq = FIGURES_DIR / f"aquifer_properties_{STAT_NAME}.png"
        plot_aquifer_properties(all_data, save_path=save_path_aq)
        print(f"Figure saved to: {save_path_aq}")

        # Plot Crop Ratios
        print("\nGenerating Crop Ratios figure...")
        save_path_cr = FIGURES_DIR / f"crop_ratios_{STAT_NAME}.png"
        plot_crop_ratios(all_data, save_path=save_path_cr)
        print(f"Figure saved to: {save_path_cr}")

    else:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run '01_prepare_data_for_figures.py' first.")

#%% Error bars
# -----------------------------------------------------------------------------
#
# SI figure:
#   One 3x3 figure showing annual bootstrap uncertainty by policy.
#
# Layout:
#   BAU      UR       FB-I
#   FB-II    PR-I     PR-II
#   empty    PWPR     empty
#
# Each policy cell contains three stacked panels:
#   1. Saturated thickness
#   2. Water level change
#   3. Withdrawal
#
# Annual medians are shown as black dashed lines.
# Annual uncertainty is shown with 5th–95th percentile error bars.
# Each subplot has its own independent y-scale.
# -----------------------------------------------------------------------------

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1. Paths and plotting setup
# -----------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SI_FIGURES_DIR = FIGURES_DIR / "supplementary"
SI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

POLICY_PATHS = {
    "BAU":   OUTPUTS_DIR / "baseline_runs",
    "UR":    OUTPUTS_DIR / "ur_runs",
    "FB-I":  OUTPUTS_DIR / "fb_runs",
    "FB-II": OUTPUTS_DIR / "fb_cb_runs",
    "PR-I":  OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
    "PWPR":  OUTPUTS_DIR / "r_plus_pr_runs",
}

POLICIES_TO_PLOT = [
    "BAU",
    "UR",
    "FB-I",
    "FB-II",
    "PR-I",
    "PR-II",
    "PWPR",
]

POLICY_POSITIONS = {
    "BAU":   (0, 0),
    "UR":    (0, 1),
    "FB-I":  (0, 2),
    "FB-II": (1, 0),
    "PR-I":  (1, 1),
    "PR-II": (1, 2),
    "PWPR":  (2, 1),
}

POLICY_STYLES = {
    "BAU":   {"color": "#808080"},
    "UR":    {"color": "#C00000"},
    "FB-I":  {"color": "#FF7F0E"},
    "FB-II": {"color": "#B85C00"},
    "PR-I":  {"color": "#4169E1"},
    "PR-II": {"color": "#00BFFF"},
    "PWPR":  {"color": "#DAA520"},
}

INITIAL_GW_ST = 24.203292398301375

background_color = "#F0F0F0"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.rcParams["axes.linewidth"] = 1.0

# Set to 25 for quick testing. Set to None for all files.
MAX_FILES_PER_POLICY = None

# Annual percentile interval for error bars.
LOWER_PCTL = 5
UPPER_PCTL = 95

# Error-bar styling
MEDIAN_LW = 1.1
MEDIAN_LINESTYLE = "--"
ERRORBAR_LW = 0.8
ERRORBAR_CAPSIZE = 2.2
ERRORBAR_CAPTHICK = 0.8
MARKER_SIZE = 2.5

# Figure styling
# Same template, but with more height for readability
FIGSIZE = (18.0, 25.5)
DPI = 500
SAVE_DPI = 500

# This prevents rerunning the file from wiping the cache in Spyder,
# unless you clear/restart the kernel.
try:
    BOOTSTRAP_SYSTEM_CACHE
except NameError:
    BOOTSTRAP_SYSTEM_CACHE = None


# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def get_excel_files(directory):
    if not directory.exists():
        return []

    files = sorted(directory.glob("*.xlsx"))

    if MAX_FILES_PER_POLICY is not None:
        files = files[:MAX_FILES_PER_POLICY]

    return files


def load_system_bootstrap_data(policy_name, directory):
    """
    Load System sheets from all bootstrap outputs for a policy.
    """
    rows = []
    files = get_excel_files(directory)

    print(f"\nLoading System sheets for {policy_name}")
    print(f"  Directory: {directory}")
    print(f"  Files found: {len(files)}")

    for path in tqdm(files, desc=f"  Reading {policy_name}", leave=False):
        try:
            df = pd.read_excel(path, sheet_name="System")
        except Exception as e:
            print(f"  Warning: could not read System sheet from {path.name}: {e}")
            continue

        if "year" not in df.columns:
            df = df.reset_index().rename(columns={"index": "year"})

        df = df.sort_values("year").copy()
        df["Policy"] = policy_name
        df["BootstrapFile"] = path.name

        if "GW_st" in df.columns:
            df["GW_st_change"] = df["GW_st"].diff()
            mask_2002 = df["year"].eq(2002)
            df.loc[mask_2002, "GW_st_change"] = (
                df.loc[mask_2002, "GW_st"] - INITIAL_GW_ST
            )

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_all_policy_system_data(force_reload=False):
    """
    Load all policy System data once and cache it.

    If force_reload=False and data already exists in BOOTSTRAP_SYSTEM_CACHE,
    the script redraws the figure without rereading Excel files.
    """
    global BOOTSTRAP_SYSTEM_CACHE

    if BOOTSTRAP_SYSTEM_CACHE is not None and not force_reload:
        print("\nUsing cached System data. No Excel files reloaded.")
        return BOOTSTRAP_SYSTEM_CACHE

    policy_data = {}

    for policy in POLICIES_TO_PLOT:
        directory = POLICY_PATHS[policy]

        if not directory.exists():
            print(f"Warning: directory not found for {policy}: {directory}")
            continue

        df = load_system_bootstrap_data(policy, directory)

        if df.empty:
            print(f"Warning: no System data found for {policy}; skipping.")
            continue

        policy_data[policy] = df

    if not policy_data:
        raise RuntimeError("No System data found for any policy.")

    BOOTSTRAP_SYSTEM_CACHE = policy_data
    return BOOTSTRAP_SYSTEM_CACHE


def summarize_annual(df, year_col, key):
    """
    Annual median and percentile interval across bootstrap runs.
    """
    summary = (
        df.groupby(year_col)[key]
        .agg(
            median="median",
            lower=lambda x: np.nanpercentile(x, LOWER_PCTL),
            upper=lambda x: np.nanpercentile(x, UPPER_PCTL),
        )
        .reset_index()
        .sort_values(year_col)
    )

    summary["yerr_lower"] = summary["median"] - summary["lower"]
    summary["yerr_upper"] = summary["upper"] - summary["median"]

    return summary


def get_axis_limits_from_summary(summary, key_is_zero_relevant=False, pad_frac=0.06):
    """
    Independent y-axis limits based on percentile error bars.
    """
    vals = pd.concat([summary["lower"], summary["upper"]], ignore_index=True).dropna()

    if vals.empty:
        return None

    ymin = vals.min()
    ymax = vals.max()

    if key_is_zero_relevant:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad

def get_shared_axis_limits_for_metric(
    policy_data,
    metric,
    shared_policies,
    year_col="year",
    pad_frac=0.06,
):
    """
    Compute one shared y-axis limit for a metric across selected policies.

    Uses percentile error-bar bounds, not just medians.
    """
    all_vals = []

    key = metric["key"]

    for policy in shared_policies:
        if policy not in policy_data:
            continue

        df = policy_data[policy]
        summary = summarize_annual(df, year_col, key)

        all_vals.append(summary["lower"])
        all_vals.append(summary["upper"])

    if not all_vals:
        return None

    vals = pd.concat(all_vals, ignore_index=True).dropna()

    if vals.empty:
        return None

    ymin = vals.min()
    ymax = vals.max()

    if metric["zero_relevant"]:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad
# -----------------------------------------------------------------------------
# 3. Main plotting function
# -----------------------------------------------------------------------------
def plot_all_policy_bootstrap_aquifer_errorbars(force_reload=False):
    metrics = [
        {
            "key": "GW_st",
            "ylabel": "(a) Saturated\nthickness\n(m)",
            "zero_relevant": False,
        },
        {
            "key": "GW_st_change",
            "ylabel": "(b) Water\nlevel\nchange (m)",
            "zero_relevant": True,
        },
        {
            "key": "withdrawal",
            "ylabel": "(c) Withdrawal\n($10^4$ $m^3$)",
            "zero_relevant": True,
        },
    ]

    policy_data = load_all_policy_system_data(force_reload=force_reload)

    all_data = pd.concat(policy_data.values(), ignore_index=True)
    start_year = int(all_data["year"].min())
    end_year = int(all_data["year"].max())
    
    # Shared axes for all governance policies.
    # BAU keeps its own independent axis because it is the baseline.
    SHARED_AXIS_POLICIES = [
        "UR",
        "FB-I",
        "FB-II",
        "PR-I",
        "PR-II",
        "PWPR",
    ]
    
    shared_ylims = {
        metric["key"]: get_shared_axis_limits_for_metric(
            policy_data=policy_data,
            metric=metric,
            shared_policies=SHARED_AXIS_POLICIES,
            year_col="year",
        )
        for metric in metrics
    }

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    outer_gs = fig.add_gridspec(
        3,
        3,
        left=0.10,
        right=0.985,
        bottom=0.070,
        top=0.905,
        wspace=0.24,
        hspace=0.3,
    )

    x_ticks = list(range(start_year, end_year + 1, 4))
    if end_year not in x_ticks:
        x_ticks.append(end_year)

    for policy in POLICIES_TO_PLOT:
        if policy not in policy_data:
            continue

        row, col = POLICY_POSITIONS[policy]
        df = policy_data[policy]

        inner_gs = outer_gs[row, col].subgridspec(
            3,
            1,
            hspace=0.25,
        )

        color = POLICY_STYLES[policy]["color"]

        for m_idx, metric in enumerate(metrics):
            key = metric["key"]
            ylabel = metric["ylabel"]

            ax = fig.add_subplot(inner_gs[m_idx, 0])
            ax.set_facecolor(background_color)

            summary = summarize_annual(df, "year", key)

            yerr = np.vstack([
                summary["yerr_lower"].values,
                summary["yerr_upper"].values,
            ])

            # Annual median and percentile interval.
            ax.errorbar(
                summary["year"],
                summary["median"],
                yerr=yerr,
                color="black",
                ecolor=color,
                linestyle=MEDIAN_LINESTYLE,
                linewidth=MEDIAN_LW,
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor="black",
                markeredgecolor="black",
                elinewidth=ERRORBAR_LW,
                capsize=ERRORBAR_CAPSIZE,
                capthick=ERRORBAR_CAPTHICK,
                alpha=1.0,
                zorder=5,
            )

            if key == "GW_st_change":
                ax.axhline(
                    0,
                    color="black",
                    linestyle=":",
                    linewidth=0.9,
                    zorder=0,
                )

            if key == "withdrawal":
                ax.axhline(
                    1900,
                    color="black",
                    linestyle=":",
                    linewidth=0.9,
                    zorder=0,
                )

            ax.set_xlim(start_year - 0.5, end_year + 0.5)

            if policy == "BAU":
                # Keep BAU on its own axis.
                ylim = get_axis_limits_from_summary(
                    summary,
                    key_is_zero_relevant=metric["zero_relevant"],
                )
            else:
                # Use one shared axis for all governance policies.
                ylim = shared_ylims.get(key)
            
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.set_xticks(x_ticks)

            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.45,
                color="grey",
                alpha=0.65,
            )

            ax.tick_params(
                axis="both",
                labelsize=24,
                width=1.0,
                length=4,
                pad=4,
            )

            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            if m_idx == 0:
                ax.set_title(policy, fontsize=24, pad=8)

            ax.set_ylabel("")

            show_metric_label = (col == 0 and row < 2) or (policy == "PWPR")

            if show_metric_label:
                ax.text(
                    -0.3,
                    0.5,
                    ylabel,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=24,
                    ha="center",
                    va="center",
                    clip_on=False,
                )

            if m_idx < 2:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", labelsize=24, rotation=45)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")

            if m_idx == 2 and policy == "PWPR":
                ax.set_xlabel("Year", fontsize=24, labelpad=8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=MEDIAN_LINESTYLE,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=MEDIAN_LW,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=ERRORBAR_LW,
            label=f"{LOWER_PCTL}th–{UPPER_PCTL}th percentile interval",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.958),
        fontsize=24,
        handlelength=3.0,
        columnspacing=2.5,
    )

    save_path = SI_FIGURES_DIR / "aquifer_bootstrap_errorbars_all_policies.png"

    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"\nSaved figure to: {save_path}")


# -----------------------------------------------------------------------------
# 4. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Plotting aquifer bootstrap error bars for all policies")
    print("--------------------------------------------------")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output figure directory: {SI_FIGURES_DIR}")
    print(f"Policies: {', '.join(POLICIES_TO_PLOT)}")
    print("--------------------------------------------------")

    plot_all_policy_bootstrap_aquifer_errorbars(force_reload=False)

    print("\nDone.")