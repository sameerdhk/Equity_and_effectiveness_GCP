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
            plot_key = stat_col(key)
            ax.plot(policy_df['Year'], policy_df[plot_key], **style, label=policy_name)

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
    
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.45, 1.22), fontsize=24)

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
            "avg_profit_p50",
            "profit_per_water_p50",
        ]
        
        missing_cols = [c for c in required_cols if c not in all_data.columns]
        if missing_cols:
            raise ValueError(
                "Missing expected median columns from economic_outcomes.csv: "
                + ", ".join(missing_cols)
            )
        # Ensure the 'Policy' column is treated as a categorical type with the correct order
        all_data['Policy'] = pd.Categorical(all_data['Policy'], categories=POLICY_STYLES.keys(), ordered=True)
        all_data.sort_values(['Policy', 'Year'], inplace=True)

        START_YEAR = all_data['Year'].min()
        END_YEAR = all_data['Year'].max()

        # Plot Economic Outcomes
        print("Generating Economic Outcomes figure...")
        save_path_econ = FIGURES_DIR / f"economic_outcomes_{STAT_NAME}.png"
        plot_economic_outcomes(all_data, save_path=save_path_econ)
        print(f"Figure saved to: {save_path_econ}")

    else:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run '01_prepare_data_for_figures.py' first.")

#%% Error Bars
# -----------------------------------------------------------------------------
#
# SI figure:
#   One 3x3 figure showing annual bootstrap economic uncertainty by policy.
#
# Layout:
#   BAU      UR       FB-I
#   FB-II    PR-I     PR-II
#   empty    PWPR     empty
#
# Each policy cell contains two stacked panels:
#   1. Average profit among irrigation-equipped farmers
#   2. Average profit per unit of applied water
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
    "FB-I":    OUTPUTS_DIR / "fb_runs",
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
    "FB-I":    (0, 2),
    "FB-II": (1, 0),
    "PR-I":  (1, 1),
    "PR-II": (1, 2),
    "PWPR":  (2, 1),
}

POLICY_STYLES = {
    "BAU":   {"color": "#808080"},
    "UR":    {"color": "#C00000"},
    "FB-I":    {"color": "#FF7F0E"},
    "FB-II": {"color": "#B85C00"},
    "PR-I":  {"color": "#4169E1"},
    "PR-II": {"color": "#00BFFF"},
    "PWPR":  {"color": "#DAA520"},
}

background_color = "#F0F0F0"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 26
plt.rcParams["axes.linewidth"] = 1.0

# Set to 25 for quick testing. Set to None for all 500 files.
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
FIGSIZE = (18.0, 24.0)
DPI = 400
SAVE_DPI = 400

# This prevents rerunning the file from wiping the cache in Spyder,
# unless you clear/restart the kernel.
try:
    BOOTSTRAP_ECON_CACHE
except NameError:
    BOOTSTRAP_ECON_CACHE = None


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


def load_economic_bootstrap_data(policy_name, directory):
    """
    Load Farmers and Fields sheets from all bootstrap outputs for one policy.

    This matches the original economic calculation:
      - merge Farmers with Fields
      - keep field_type_rn == "optimize"
      - avg_profit = yearly mean profit
      - profit_per_water = yearly mean of profit / w
    """
    rows = []
    files = get_excel_files(directory)

    print(f"\nLoading economic outputs for {policy_name}")
    print(f"  Directory: {directory}")
    print(f"  Files found: {len(files)}")

    for path in tqdm(files, desc=f"  Reading {policy_name}", leave=False):
        try:
            df_farmers = pd.read_excel(path, sheet_name="Farmers")
            df_fields = pd.read_excel(path, sheet_name="Fields")
        except Exception as e:
            print(f"  Warning: could not read Farmers/Fields from {path.name}: {e}")
            continue

        required_farmer_cols = ["year", "Step", "AgentID", "profit"]
        required_field_cols = ["year", "Step", "AgentID", "field_type_rn", "w"]

        missing_farmer = [c for c in required_farmer_cols if c not in df_farmers.columns]
        missing_field = [c for c in required_field_cols if c not in df_fields.columns]

        if missing_farmer or missing_field:
            print(
                f"  Warning: skipping {path.name}; "
                f"missing farmer cols={missing_farmer}, field cols={missing_field}"
            )
            continue

        df_fields = df_fields.copy()
        df_farmers = df_farmers.copy()

        df_fields["AgentID_numeric"] = (
            df_fields["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
        )
        df_farmers["AgentID_numeric"] = (
            df_farmers["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
        )

        df_merged = (
            pd.merge(
                df_farmers,
                df_fields[["year", "Step", "AgentID_numeric", "field_type_rn", "w"]],
                on=["year", "Step", "AgentID_numeric"],
                how="left",
            )
            .set_index("year")
        )

        irrigators_df = df_merged[df_merged["field_type_rn"] == "optimize"].copy()

        if irrigators_df.empty:
            continue

        irrigators_df["profit_per_water"] = (
            irrigators_df["profit"] / irrigators_df["w"]
        ).replace([np.inf, -np.inf], 0)

        yearly_avg_profit = irrigators_df.groupby("year")["profit"].mean()
        yearly_profit_per_water = irrigators_df.groupby("year")["profit_per_water"].mean()

        run_df = pd.DataFrame(
            {
                "Year": yearly_avg_profit.index,
                "avg_profit": yearly_avg_profit.values,
                "profit_per_water": yearly_profit_per_water.values,
            }
        )

        run_df["Policy"] = policy_name
        run_df["BootstrapFile"] = path.name

        rows.append(run_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_all_policy_economic_data(force_reload=False):
    """
    Load all policy economic data once and cache it.

    If force_reload=False and data already exists in BOOTSTRAP_ECON_CACHE,
    the script redraws the figure without rereading Excel files.
    """
    global BOOTSTRAP_ECON_CACHE

    if BOOTSTRAP_ECON_CACHE is not None and not force_reload:
        print("\nUsing cached economic data. No Excel files reloaded.")
        return BOOTSTRAP_ECON_CACHE

    policy_data = {}

    for policy in POLICIES_TO_PLOT:
        directory = POLICY_PATHS[policy]

        if not directory.exists():
            print(f"Warning: directory not found for {policy}: {directory}")
            continue

        df = load_economic_bootstrap_data(policy, directory)

        if df.empty:
            print(f"Warning: no economic data found for {policy}; skipping.")
            continue

        policy_data[policy] = df

    if not policy_data:
        raise RuntimeError("No economic data found for any policy.")

    BOOTSTRAP_ECON_CACHE = policy_data
    return BOOTSTRAP_ECON_CACHE


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


def get_axis_limits_from_summary(summary, pad_frac=0.06):
    """
    Independent y-axis limits based on percentile error bars.
    """
    vals = pd.concat([summary["lower"], summary["upper"]], ignore_index=True).dropna()

    if vals.empty:
        return None

    ymin = vals.min()
    ymax = vals.max()

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad

def get_shared_axis_limits_for_metric(
    policy_data,
    metric,
    shared_policies,
    year_col="Year",
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

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad

# -----------------------------------------------------------------------------
# 3. Main plotting function
# -----------------------------------------------------------------------------
def plot_all_policy_bootstrap_economic_errorbars(force_reload=False):
    metrics = [
        {
            "key": "avg_profit",
            "ylabel": "(a) Average\nprofit among\nfarmers\n($\$ 10^4$)",
        },
        {
            "key": "profit_per_water",
            "ylabel": "(b) Average\nprofit per\napplied water\n($\$ 10^4$ per cm)",
        },
    ]

    policy_data = load_all_policy_economic_data(force_reload=force_reload)

    all_data = pd.concat(policy_data.values(), ignore_index=True)
    start_year = int(all_data["Year"].min())
    end_year = int(all_data["Year"].max())
    # Shared axes for all governance policies.
    # BAU keeps its own independent axis because it is the baseline.
    SHARED_AXIS_POLICIES = [
        "BAU",
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
            year_col="Year",
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
        hspace=0.30,
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
            2,
            1,
            hspace=0.25,
        )
        
        color = POLICY_STYLES[policy]["color"]

        for m_idx, metric in enumerate(metrics):
            key = metric["key"]
            ylabel = metric["ylabel"]

            ax = fig.add_subplot(inner_gs[m_idx, 0])
            ax.set_facecolor(background_color)

            summary = summarize_annual(df, "Year", key)

            yerr = np.vstack([
                summary["yerr_lower"].values,
                summary["yerr_upper"].values,
            ])

            # Annual median and percentile interval.
            ax.errorbar(
                summary["Year"],
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

            ax.set_xlim(start_year - 0.5, end_year + 0.5)

            if policy == None:
                # Keep BAU on its own axis.
                ylim = get_axis_limits_from_summary(summary)
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
                labelsize=26,
                width=1.0,
                length=4,
                pad=4,
            )
            
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            
            if m_idx == 0:
                ax.set_title(policy, fontsize=26, pad=8)

            ax.set_ylabel("")

            show_metric_label = (col == 0 and row < 2) or (policy == "PWPR")
            
            if show_metric_label:
                ax.text(
                    -0.35,
                    0.5,
                    ylabel,
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=26,
                    ha="center",
                    va="center",
                    clip_on=False,
                )
            if m_idx < 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", labelsize=26, rotation=45)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")

            if m_idx == 1 and policy == "PWPR":
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
        fontsize=26,
        handlelength=3.0,
        columnspacing=2.5,
    )

    save_path = SI_FIGURES_DIR / "economic_bootstrap_errorbars_all_policies.png"

    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"\nSaved figure to: {save_path}")


# -----------------------------------------------------------------------------
# 4. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Plotting economic bootstrap error bars for all policies")
    print("--------------------------------------------------")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output figure directory: {SI_FIGURES_DIR}")
    print(f"Policies: {', '.join(POLICIES_TO_PLOT)}")
    print("--------------------------------------------------")

    plot_all_policy_bootstrap_economic_errorbars(force_reload=False)

    print("\nDone.")
#%%Annual policy-caused cutoff diagnostics for PR-I and PR-II.
# -----------------------------------------------------------------------------
#
# SI / response figure:
#   Annual policy-caused cutoff diagnostics for PR-I and PR-II.
#
# Reads raw bootstrap Excel files directly and applies the same cutoff logic:
#   n_policy_cutoff = cutoff + partially cutoff + prior_appropriation
#
# Figure:
#   Two panels:
#      PR-I      PR-II
#
# Annual medians are shown as black dashed lines.
# Annual uncertainty is shown with percentile error bars.
#
# -----------------------------------------------------------------------------

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "PR-I":  OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
}

POLICIES_TO_PLOT = ["PR-I", "PR-II"]

POLICY_STYLES = {
    "PR-I":  {"color": "#4169E1"},
    "PR-II": {"color": "#00BFFF"},
}

background_color = "#F0F0F0"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# Set to 25 for quick testing. Set to None for all 500 files.
MAX_FILES_PER_POLICY = None

# Use 5/95 to match your new error-bar figures.
# Change to 10/90 if you want the same range as the old diagnostic CSV.
LOWER_PCTL = 5
UPPER_PCTL = 95

MEDIAN_LW = 1.1
MEDIAN_LINESTYLE = "--"
ERRORBAR_LW = 0.8
ERRORBAR_CAPSIZE = 2.2
ERRORBAR_CAPTHICK = 0.8
MARKER_SIZE = 2.5

FIGSIZE = (12.0, 4.8)
DPI = 500
SAVE_DPI = 500

SAVE_PATH = SI_FIGURES_DIR / "priority_policy_cutoff_errorbars_pr1_pr2.png"


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


def parse_bootstrap_id(filename):
    """
    Extract bootstrap number from filenames.

    This is intentionally flexible. It first looks for b_<number>,
    then falls back to the first number in the filename.
    """
    name = Path(filename).stem

    match = re.search(r"b[_-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))

    return None


def safe_read_excel(path, sheet_name):
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        print(f"  Warning: could not read {sheet_name} from {path.name}: {e}")
        return None


def add_agent_numeric_id(df):
    """
    Add AgentID_numeric from AgentID.

    Handles AgentID values like 'farmer_12', 'Agent_12', or numeric IDs.
    """
    df = df.copy()

    if "AgentID_numeric" in df.columns:
        return df

    if "AgentID" not in df.columns:
        raise ValueError("Expected AgentID column.")

    extracted = df["AgentID"].astype(str).str.extract(r"(\d+)$")[0]

    # If extraction fails for some rows, try direct numeric conversion.
    direct = pd.to_numeric(df["AgentID"], errors="coerce").astype("Int64")
    extracted_num = pd.to_numeric(extracted, errors="coerce").astype("Int64")

    df["AgentID_numeric"] = extracted_num.fillna(direct).astype("Int64")

    return df


def get_irrigation_equipped_status(fields):
    """
    Identify irrigation-equipped farmers from the Fields sheet.

    The same logic as your extraction code:
      irrigation_equipped = field_type_rn == "optimize"

    Returns one row per year-Step-AgentID_numeric.
    """
    required_cols = ["year", "Step", "AgentID", "field_type_rn"]
    missing = [c for c in required_cols if c not in fields.columns]
    if missing:
        raise ValueError(f"Fields sheet missing columns: {missing}")

    fields = add_agent_numeric_id(fields)

    out = fields[["year", "Step", "AgentID_numeric", "field_type_rn"]].copy()

    out["irrigation_equipped"] = (
        out["field_type_rn"]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("optimize")
    )

    # One row per farmer-year-step. If there are duplicates, keep equipped=True
    # if any field row indicates optimize.
    out = (
        out.groupby(["year", "Step", "AgentID_numeric"], as_index=False)
        .agg(
            field_type_rn=("field_type_rn", "first"),
            irrigation_equipped=("irrigation_equipped", "max"),
        )
    )

    return out


def summarize_annual(df, group_cols, value_cols):
    """
    Summarize raw bootstrap/year data with median and percentile bounds.
    """
    rows = []

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()

            if vals.empty:
                row[f"{col}_median"] = np.nan
                row[f"{col}_p{LOWER_PCTL}"] = np.nan
                row[f"{col}_p{UPPER_PCTL}"] = np.nan
            else:
                row[f"{col}_median"] = np.nanmedian(vals)
                row[f"{col}_p{LOWER_PCTL}"] = np.nanpercentile(vals, LOWER_PCTL)
                row[f"{col}_p{UPPER_PCTL}"] = np.nanpercentile(vals, UPPER_PCTL)

        rows.append(row)

    return pd.DataFrame(rows)


def get_axis_limits(summary, lower_col, upper_col, pad_frac=0.08):
    vals = pd.concat([summary[lower_col], summary[upper_col]], ignore_index=True)
    vals = pd.to_numeric(vals, errors="coerce").dropna()

    if vals.empty:
        return None

    ymin = min(vals.min(), 0)
    ymax = vals.max()

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return max(0, ymin - pad), ymax + pad


# -----------------------------------------------------------------------------
# 3. Extract cutoff diagnostics from raw Excel outputs
# -----------------------------------------------------------------------------
def load_priority_cutoff_raw_data():
    cutoff_rows = []

    for p_idx, policy in enumerate(POLICIES_TO_PLOT, start=1):
        directory = POLICY_PATHS[policy]

        print(f"\n[{p_idx}/{len(POLICIES_TO_PLOT)}] Cutoff diagnostics: {policy}")
        print(f"    Directory: {directory}")

        if not directory.exists():
            print(f"    Warning: directory not found. Skipping {policy}.")
            continue

        files = get_excel_files(directory)
        print(f"    Files found: {len(files)}")

        for path in tqdm(files, desc=f"    Reading {policy}", leave=False):
            bnum = parse_bootstrap_id(path.name)
            if bnum is None:
                print(f"    Warning: could not parse bootstrap id from {path.name}. Skipping.")
                continue

            farmers = safe_read_excel(path, "Farmers")
            fields = safe_read_excel(path, "Fields")

            if farmers is None or fields is None:
                continue

            if "zero_irrigation_reason" not in farmers.columns:
                print(f"    Warning: zero_irrigation_reason missing in {path.name}. Skipping.")
                continue

            try:
                farmers = add_agent_numeric_id(farmers)
                field_status = get_irrigation_equipped_status(fields)
            except Exception as e:
                print(f"    Warning: could not process IDs/status in {path.name}: {e}")
                continue

            required_farmer_cols = ["year", "Step", "AgentID_numeric", "zero_irrigation_reason"]
            missing_farmer_cols = [c for c in required_farmer_cols if c not in farmers.columns]
            if missing_farmer_cols:
                print(f"    Warning: missing farmer cols {missing_farmer_cols} in {path.name}. Skipping.")
                continue

            merged = farmers.merge(
                field_status[
                    ["year", "Step", "AgentID_numeric", "field_type_rn", "irrigation_equipped"]
                ],
                on=["year", "Step", "AgentID_numeric"],
                how="left",
            )

            # Focus on irrigation-equipped farmers. Always-rainfed farmers are not
            # junior irrigators being cut off by the policy.
            irrigators = merged[merged["irrigation_equipped"].fillna(False)].copy()

            if irrigators.empty:
                continue

            irrigators["zero_irrigation_reason"] = (
                irrigators["zero_irrigation_reason"]
                .fillna("NA")
                .astype(str)
                .str.strip()
            )

            for year, g in irrigators.groupby("year"):
                reason = g["zero_irrigation_reason"]
                n = len(g)

                n_cutoff = reason.eq("cutoff").sum()
                n_partial = reason.eq("partially cutoff").sum()
                n_prior = reason.eq("prior_appropriation").sum()

                # Combined policy-caused cutoff count.
                n_policy_cutoff = n_cutoff + n_partial + n_prior

                n_rainfed = reason.eq("rainfed").sum()
                n_optimization = reason.eq("optimization").sum()
                n_no_zero_reason = reason.eq("NA").sum()

                known = (
                    n_cutoff
                    + n_partial
                    + n_prior
                    + n_rainfed
                    + n_optimization
                    + n_no_zero_reason
                )
                n_other_reason = n - known

                cutoff_rows.append({
                    "Policy": policy,
                    "Bootstrap": bnum,
                    "Year": int(year),
                    "n_irrigation_equipped": n,

                    "n_cutoff": n_cutoff,
                    "n_partially_cutoff": n_partial,
                    "n_prior_appropriation": n_prior,
                    "n_policy_cutoff": n_policy_cutoff,

                    "n_rainfed_reason": n_rainfed,
                    "n_optimization_zero": n_optimization,
                    "n_no_zero_reason": n_no_zero_reason,
                    "n_other_reason": n_other_reason,

                    "share_cutoff": n_cutoff / n if n else np.nan,
                    "share_partially_cutoff": n_partial / n if n else np.nan,
                    "share_prior_appropriation": n_prior / n if n else np.nan,
                    "share_policy_cutoff": n_policy_cutoff / n if n else np.nan,

                    "share_rainfed_reason": n_rainfed / n if n else np.nan,
                    "share_optimization_zero": n_optimization / n if n else np.nan,
                    "share_no_zero_reason": n_no_zero_reason / n if n else np.nan,
                    "share_other_reason": n_other_reason / n if n else np.nan,
                })

    if not cutoff_rows:
        raise RuntimeError("No cutoff diagnostics processed.")

    return pd.DataFrame(cutoff_rows)


# -----------------------------------------------------------------------------
# 4. Plot cutoff diagnostics
# -----------------------------------------------------------------------------
def plot_priority_cutoff_errorbars():
    cutoff_all = load_priority_cutoff_raw_data()

    value_cols = [
        "n_irrigation_equipped",
        "n_cutoff",
        "n_partially_cutoff",
        "n_prior_appropriation",
        "n_policy_cutoff",
        "share_policy_cutoff",
    ]

    cutoff_summary = summarize_annual(
        cutoff_all,
        group_cols=["Policy", "Year"],
        value_cols=value_cols,
    )

    start_year = int(cutoff_summary["Year"].min())
    end_year = int(cutoff_summary["Year"].max())

    x_ticks = list(range(start_year, end_year + 1, 4))
    if start_year not in x_ticks:
        x_ticks.insert(0, start_year)
    if end_year not in x_ticks:
        x_ticks.append(end_year)

    value_base = "n_policy_cutoff"
    median_col = f"{value_base}_median"
    lower_col = f"{value_base}_p{LOWER_PCTL}"
    upper_col = f"{value_base}_p{UPPER_PCTL}"

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    gs = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.18,
        top=0.82,
        wspace=0.22,
    )

    for idx, policy in enumerate(POLICIES_TO_PLOT):
        df = cutoff_summary[cutoff_summary["Policy"].eq(policy)].copy()

        if df.empty:
            print(f"Warning: no cutoff diagnostics found for {policy}.")
            continue

        df = df.sort_values("Year")
        color = POLICY_STYLES[policy]["color"]

        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(background_color)

        y = pd.to_numeric(df[median_col], errors="coerce")
        lower = pd.to_numeric(df[lower_col], errors="coerce")
        upper = pd.to_numeric(df[upper_col], errors="coerce")

        yerr = np.vstack([
            y - lower,
            upper - y,
        ])

        ax.errorbar(
            df["Year"],
            y,
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

        ax.set_xlim(start_year - 0.5, end_year + 0.5)
        ax.set_xticks(x_ticks)

        ylim = get_axis_limits(df, lower_col, upper_col)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.45,
            color="grey",
            alpha=0.65,
        )

        ax.tick_params(axis="both", labelsize=18)
        ax.tick_params(axis="x", labelsize=18, rotation=45)

        ax.set_title(policy, fontsize=18, pad=7)
        ax.set_xlabel("Year", fontsize=18)

        if idx == 0:
            ax.set_ylabel(
                "Irrigation-equipped farmers\nrestricted from withdrawal",
                fontsize=18,
            )
        else:
            ax.set_ylabel("")

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
        bbox_to_anchor=(0.5, 0.965),
        fontsize=18,
    )

    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(SAVE_PATH, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"\nSaved figure to: {SAVE_PATH}")


# -----------------------------------------------------------------------------
# 5. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Plotting PR-I and PR-II cutoff diagnostics from raw Excel outputs")
    print("--------------------------------------------------")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output file: {SAVE_PATH}")
    print("--------------------------------------------------")

    plot_priority_cutoff_errorbars()

    print("\nDone.")