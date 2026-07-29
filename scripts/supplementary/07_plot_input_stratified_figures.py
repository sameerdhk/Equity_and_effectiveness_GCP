# -----------------------------------------------------------------------------
# 06_plot_input_stratified_figures.py
#
# Purpose:
#   Create four supplementary input-stratified figures comparing:
#
#       1. Full ensemble
#       2. Dry bootstraps
#       3. Wet bootstraps
#       4. Low market-return bootstraps
#       5. High market-return bootstraps
#
#   across the seven policy panels:
#
#       BAU, UR, FB, FB-CB, PR-I, PR-II, R+PR
#
# Figures:
#   1. Aquifer outcomes:
#        - Saturated thickness
#        - Water-level change
#        - Withdrawal
#
#   2. Economic outcomes:
#        - Average yearly profit
#        - Economic water productivity
#
#   3. Profit distributions:
#        - Horizontal boxplots of farmer-bootstrap average annual profit
#
#   4. Gini coefficients:
#        - Horizontal bars of Gini coefficients calculated from pooled
#          farmer-bootstrap average annual profits
#
# Inputs:
#   - inputs/supplementary/bootstrap_climate_market_groups.csv
#   - outputs/<policy>_runs/*.xlsx
#   - outputs/data_for_figures/profit_distribution_*.csv
#
# Cache files:
#   - inputs/supplementary/cache_aquifer_system_metrics_from_excel.csv
#   - inputs/supplementary/cache_annual_economic_metrics_for_stratified_figures.csv
#   - inputs/supplementary/cache_farmer_avg_profit_for_gini.csv
#
# Outputs:
#   Figures:
#   - outputs/figures/supplementary/si_stratified_aquifer_full_dry_wet_market.png
#   - outputs/figures/supplementary/si_stratified_economic_full_dry_wet_market.png
#   - outputs/figures/supplementary/si_stratified_profit_distribution_full_dry_wet_market.png
#   - outputs/figures/supplementary/si_stratified_gini_full_dry_wet_market.png
#
#   Summary CSVs:
#   - inputs/supplementary/annual_aquifer_summary_full_dry_wet_market.csv
#   - inputs/supplementary/annual_economic_summary_full_dry_wet_market.csv
#   - inputs/supplementary/profit_distribution_summary_full_dry_wet_market.csv
#   - inputs/supplementary/gini_summary_full_dry_wet_market.csv
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import io
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm


# =============================================================================
# 1. Paths
# =============================================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_FOR_FIGURES_DIR = OUTPUTS_DIR / "data_for_figures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SI_FIGURES_DIR = FIGURES_DIR / "supplementary"
SI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUPP_DIR = PROJECT_ROOT / "inputs" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

GROUPS_FILE = SUPP_DIR / "bootstrap_climate_market_groups.csv"

# Caches
AQUIFER_CACHE = SUPP_DIR / "cache_aquifer_system_metrics_from_excel.csv"
ECON_ANNUAL_CACHE = SUPP_DIR / "cache_annual_economic_metrics_for_stratified_figures.csv"
FARMER_PROFIT_CACHE = SUPP_DIR / "cache_farmer_avg_profit_for_gini.csv"

# Summary CSVs
AQUIFER_SUMMARY_OUT = SUPP_DIR / "annual_aquifer_summary_full_dry_wet_market.csv"
ECON_SUMMARY_OUT = SUPP_DIR / "annual_economic_summary_full_dry_wet_market.csv"
PROFIT_SUMMARY_OUT = SUPP_DIR / "profit_distribution_summary_full_dry_wet_market.csv"
GINI_SUMMARY_OUT = SUPP_DIR / "gini_summary_full_dry_wet_market.csv"

# Figures
AQUIFER_FIG_OUT = SI_FIGURES_DIR / "si_stratified_aquifer_full_dry_wet_market.png"
ECON_FIG_OUT = SI_FIGURES_DIR / "si_stratified_economic_full_dry_wet_market.png"
PROFIT_FIG_OUT = SI_FIGURES_DIR / "si_stratified_profit_distribution_full_dry_wet_market.png"
GINI_FIG_OUT = SI_FIGURES_DIR / "si_stratified_gini_full_dry_wet_market.png"


# =============================================================================
# 2. Configuration
# =============================================================================
YEAR_MIN, YEAR_MAX = 2002, 2022

# Same initial saturated thickness used in your aquifer plotting workflow
INITIAL_GW_ST = 24.203292398301375

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

# If True, force rebuilding even if caches exist.
# Leave these False unless you change underlying model output files.
REBUILD_AQUIFER_CACHE = False
REBUILD_ECON_CACHE = False
REBUILD_FARMER_PROFIT_CACHE = False

# Profit/Gini population.
# Use "irrigation_equipped" to match field_type_rn == "optimize".
# Use "all" only if you explicitly want all farmers.
PROFIT_POPULATION = "irrigation_equipped"

# If your existing economic-water-productivity figure uses a different unit,
# change this label only. The calculation here is profit / w, where profit is
# stored in $10^4 and w is applied water depth in cm.
EWP_YLABEL = "(b) Average\nprofit per\napplied water\n($\$ 10^4$ per cm)"

# Condition styles. Colors represent input conditions, not policies.
CONDITION_ORDER = [
    "Full ensemble",
    "Dry",
    "Wet",
    "Low return",
    "High return",
]

CONDITION_SHORT_LABELS = {
    "Full ensemble": "Full",
    "Dry": "Dry",
    "Wet": "Wet",
    "Low return": "Low return",
    "High return": "High return",
}

CONDITION_STYLES = {
    "Full ensemble": {"color": "#4D4D4D", "linestyle": "-",  "lw": 2.8, "alpha": 1.00},
    "Dry":           {"color": "#C44E52", "linestyle": "--", "lw": 2.4, "alpha": 0.95},
    "Wet":           {"color": "#4C72B0", "linestyle": "-.", "lw": 2.4, "alpha": 0.95},
    "Low return":    {"color": "#8172B2", "linestyle": ":",  "lw": 2.6, "alpha": 0.95},
    "High return":   {"color": "#55A868", "linestyle": "-",  "lw": 2.4, "alpha": 0.95},
}

background_color = "#F0F0F0"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["svg.fonttype"] = "none"

DPI = 500
SAVE_DPI = 500

# Match the settled aquifer template.
FIGSIZE_AQUIFER = (18, 26.5)

# Economic has only two panels per policy, so it should be shorter but still roomy.
FIGSIZE_ECON = (18, 24)

# Profit distribution and Gini have one panel per policy.
# They need enough height for condition labels.
FIGSIZE_SINGLE = (20, 18)

TITLE_FONTSIZE = 24
TICK_FONTSIZE = 24
YLABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 22
YEAR_LABEL_FONTSIZE = 24

SINGLE_TITLE_FONTSIZE = 24
SINGLE_XTICK_FONTSIZE = 22
SINGLE_YTICK_FONTSIZE = 22
SINGLE_XLABEL_FONTSIZE = 22
SINGLE_ANNOTATION_FONTSIZE = 17


# Axis-scaling options
# Use "bau_separate_shared_policy" for BAU separate + all governance policies shared.
# Use "independent" to restore the original behavior where every policy panel has its own axis.
AQUIFER_YLIM_MODE = "independent"
# Economic axis-scaling options
# panel (a): BAU separate, governance policies shared
ECON_PANEL_A_YLIM_MODE = "independent"

# panel (b): all policies share the same axis
ECON_PANEL_B_YLIM_MODE = "independent"

SHARED_AXIS_POLICIES = [
    "UR",
    "FB-I",
    "FB-II",
    "PR-I",
    "PR-II",
    "PWPR",
]

# =============================================================================
# 3. Helper functions
# =============================================================================
def norm_col(c: object) -> str:
    s = str(c).strip()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    return s.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    return df


def normalize_policy_label(x: object) -> str:
    s = str(x).strip()

    mapping = {
        "BAU": "BAU",
        "UR": "UR",

        "FB": "FB-I",
        "FB_I": "FB-I",
        "FB-I": "FB-I",

        "FB_CB": "FB-II",
        "FB-CB": "FB-II",
        "FB_II": "FB-II",
        "FB-II": "FB-II",

        "PR_I": "PR-I",
        "PR-I": "PR-I",
        "PRI": "PR-I",

        "PR_II": "PR-II",
        "PR-II": "PR-II",
        "PRII": "PR-II",

        "R_PR": "PWPR",
        "R+PR": "PWPR",
        "R_PLUS_PR": "PWPR",
        "PWPR": "PWPR",
    }

    return mapping.get(s.upper(), s)


def read_csv_autodelim(path: Path, nrows=None) -> pd.DataFrame:
    raw = path.read_bytes()
    head = raw[:2048].decode(errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","

    return pd.read_csv(io.BytesIO(raw), delimiter=delim, nrows=nrows)


def parse_bootstrap_from_name(fname: str) -> int | None:
    m = re.search(r"_b_(\d+)", fname)
    if m:
        return int(m.group(1))

    m = re.search(r"_([0-9]+)(?=\.[A-Za-z]+$)", fname)
    if m:
        return int(m.group(1))

    return None


def parse_policy_from_profit_file(fname: str) -> str | None:
    m = re.search(r"profit_distribution_(.+?)_b_\d+", fname)
    if not m:
        return None
    return normalize_policy_label(m.group(1))


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def list_excel_files(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []

    files = []
    for p in run_dir.iterdir():
        if p.name.startswith("~$"):
            continue
        if p.suffix.lower() in [".xlsx", ".xls"]:
            files.append(p)

    return sorted(files)


def gini_coefficient(values: pd.Series | np.ndarray) -> float:
    """
    Gini coefficient from pooled farmer-bootstrap average annual profit values.

    This follows the manuscript/Lorenz logic:
      1. calculate each farmer-bootstrap average annual profit;
      2. pool those values within each condition/policy;
      3. calculate one Gini from that pooled distribution.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan

    # Profit is stored in $10^4. Scaling does not change Gini.
    x = np.sort(x * 1e4)

    total = np.sum(x)
    n = x.size

    if total <= 0:
        return np.nan

    index = np.arange(1, n + 1)
    rank_sum = np.sum((n + 1 - index) * x)

    return float((n + 1 - 2 * rank_sum / total) / n)


def summarize_annual_median(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "median"])

    out = (
        df.groupby("year", as_index=False)[metric]
        .median()
        .rename(columns={metric: "median"})
        .sort_values("year")
    )

    out["year"] = out["year"].astype(int)
    return out


def get_axis_limits_from_values(values_list, zero_relevant=False, pad_frac=0.06):
    vals = []
    for v in values_list:
        vals.extend(pd.Series(v).dropna().tolist())

    if len(vals) == 0:
        return None

    ymin = min(vals)
    ymax = max(vals)

    if zero_relevant:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad

def get_shared_axis_limits_from_summary(
    summary_df,
    metric_key,
    policies,
    zero_relevant=False,
    pad_frac=0.06,
):
    """
    Compute one shared y-axis limit for one metric across selected policies.

    This is used to give UR, FB-I, FB-II, PR-I, PR-II, and PWPR
    a common axis while allowing BAU to keep its own axis.
    """
    vals = summary_df[
        summary_df["policy"].isin(policies)
        & summary_df["metric"].eq(metric_key)
    ]["median"].dropna()

    if vals.empty:
        return None

    ymin = vals.min()
    ymax = vals.max()

    if zero_relevant:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)

    if ymin == ymax:
        pad = 1.0
    else:
        pad = (ymax - ymin) * pad_frac

    return ymin - pad, ymax + pad

def style_axis(ax, tick_fontsize=TICK_FONTSIZE):
    ax.set_facecolor(background_color)

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
        labelsize=tick_fontsize,
        width=1.0,
        length=4,
        pad=4,
    )

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))


def format_year_axis(ax, show_tick_labels=False, show_xlabel=False):
    ax.set_xlim(YEAR_MIN, YEAR_MAX)

    x_ticks = [2002, 2006, 2010, 2014, 2018, 2022]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    if show_tick_labels:
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    if show_xlabel:
        ax.set_xlabel("Year", fontsize=YEAR_LABEL_FONTSIZE, labelpad=8)
    else:
        ax.set_xlabel("")
        
def make_outer_grid(fig):
    return fig.add_gridspec(
        3,
        3,
        left=0.10,
        right=0.985,
        bottom=0.070,
        top=0.905,
        wspace=0.24,
        hspace=0.28,
    )


def add_metric_label(ax, ylabel, x=-0.4):
    ax.text(
        x,
        0.5,
        ylabel,
        transform=ax.transAxes,
        rotation=90,
        fontsize=YLABEL_FONTSIZE,
        ha="center",
        va="center",
        clip_on=False,
    )


def set_policy_title(ax, policy):
    ax.set_title(policy, fontsize=TITLE_FONTSIZE, pad=8)


def add_condition_legend(fig, line_style=True):
    fig.legend(
        handles=condition_legend_handles(line_style=line_style),
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.958),
        fontsize=LEGEND_FONTSIZE,
        handlelength=3.0,
        columnspacing=1.8,
    )


def condition_legend_handles(line_style=True):
    handles = []

    for cond in CONDITION_ORDER:
        st = CONDITION_STYLES[cond]

        if line_style:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=st["lw"] + 0.5,
                    alpha=st["alpha"],
                    label=CONDITION_SHORT_LABELS[cond],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=st["color"],
                    linewidth=7,
                    alpha=0.70,
                    label=CONDITION_SHORT_LABELS[cond],
                )
            )

    return handles


def add_blank_outer_cells(fig, outer_gs):
    used = set(POLICY_POSITIONS.values())
    for r in range(3):
        for c in range(3):
            if (r, c) not in used:
                ax = fig.add_subplot(outer_gs[r, c])
                ax.axis("off")


def require_columns(df: pd.DataFrame, required: set[str], name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def make_outer_grid_single(fig):
    return fig.add_gridspec(
        3,
        3,
        left=0.08,
        right=0.985,
        bottom=0.070,
        top=0.905,
        wspace=0.35,
        hspace=0.28,
    )

# =============================================================================
# 4. Load bootstrap condition labels
# =============================================================================
if not GROUPS_FILE.exists():
    raise FileNotFoundError(
        f"Missing {GROUPS_FILE}. Run your bootstrap classification script first."
    )

groups = pd.read_csv(GROUPS_FILE)
groups = normalize_columns(groups)

require_columns(
    groups,
    {"bootstrap", "precip_group", "market_return_group"},
    GROUPS_FILE.name,
)

groups["bootstrap"] = pd.to_numeric(groups["bootstrap"], errors="coerce").astype(int)

all_bootstraps = set(groups["bootstrap"].tolist())
dry_bootstraps = set(groups.loc[groups["precip_group"] == "dry", "bootstrap"])
wet_bootstraps = set(groups.loc[groups["precip_group"] == "wet", "bootstrap"])
low_return_bootstraps = set(
    groups.loc[groups["market_return_group"] == "low_return", "bootstrap"]
)
high_return_bootstraps = set(
    groups.loc[groups["market_return_group"] == "high_return", "bootstrap"]
)

CONDITION_BOOTSTRAPS = {
    "Full ensemble": all_bootstraps,
    "Dry": dry_bootstraps,
    "Wet": wet_bootstraps,
    "Low return": low_return_bootstraps,
    "High return": high_return_bootstraps,
}

print("\nCondition bootstrap counts:")
for cond in CONDITION_ORDER:
    print(f"  {cond}: {len(CONDITION_BOOTSTRAPS[cond])} bootstraps")


# =============================================================================
# 5. Build or load aquifer cache
# =============================================================================
aquifer_cache_valid = False

if AQUIFER_CACHE.exists() and not REBUILD_AQUIFER_CACHE:
    try:
        aquifer_df = pd.read_csv(AQUIFER_CACHE)
        aquifer_df = normalize_columns(aquifer_df)

        require_columns(
            aquifer_df,
            {"policy", "bootstrap", "year", "gw_st", "gw_st_change", "withdrawal"},
            AQUIFER_CACHE.name,
        )

        aquifer_cache_valid = True
        print(f"\nUsing existing aquifer cache:\n  {AQUIFER_CACHE}")

    except Exception as e:
        print(f"\nAquifer cache exists but is not valid: {e}")
        print("Rebuilding aquifer cache from Excel System sheets...")

if not aquifer_cache_valid:
    print("\nBuilding aquifer cache from raw Excel System sheets...")

    aquifer_rows = []

    for policy in POLICIES_TO_PLOT:
        run_dir = POLICY_PATHS[policy]
        excel_files = list_excel_files(run_dir)

        if not excel_files:
            print(f"Skipping {policy}: no files found in {run_dir}")
            continue

        print(f"\nProcessing {policy}: {len(excel_files)} System sheets")

        for path in tqdm(excel_files, desc=f"{policy} System", leave=False):
            bootstrap = parse_bootstrap_from_name(path.name)

            if bootstrap is None:
                continue

            try:
                df = pd.read_excel(path, sheet_name="System")
            except Exception as e:
                print(f"  Warning: could not read {path.name}: {e}")
                continue

            year_col = find_col(df, ["year"])
            gw_col = find_col(df, ["GW_st", "saturated_thickness"])
            withdrawal_col = find_col(df, ["withdrawal", "withdrawals"])

            if year_col is None or gw_col is None or withdrawal_col is None:
                print(f"  Warning: missing required System columns in {path.name}")
                continue

            df = df.copy()
            df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
            df[gw_col] = pd.to_numeric(df[gw_col], errors="coerce")
            df[withdrawal_col] = pd.to_numeric(df[withdrawal_col], errors="coerce")

            df = (
                df[df[year_col].between(YEAR_MIN, YEAR_MAX)]
                .sort_values(year_col)
                .copy()
            )

            if df.empty:
                continue

            df["gw_st_change_calc"] = df[gw_col].diff()
            first_idx = df.index[0]
            df.loc[first_idx, "gw_st_change_calc"] = (
                df.loc[first_idx, gw_col] - INITIAL_GW_ST
            )

            out = pd.DataFrame(
                {
                    "policy": policy,
                    "bootstrap": int(bootstrap),
                    "year": df[year_col].astype(int),
                    "gw_st": df[gw_col],
                    "gw_st_change": df["gw_st_change_calc"],
                    "withdrawal": df[withdrawal_col],
                }
            )

            aquifer_rows.append(out)

    if not aquifer_rows:
        raise RuntimeError("No aquifer rows were created from Excel outputs.")

    aquifer_df = pd.concat(aquifer_rows, ignore_index=True)
    aquifer_df.to_csv(AQUIFER_CACHE, index=False)

    print(f"\nSaved aquifer cache:\n  {AQUIFER_CACHE}")

aquifer_df = normalize_columns(aquifer_df)
aquifer_df["policy"] = aquifer_df["policy"].map(normalize_policy_label)
aquifer_df["bootstrap"] = pd.to_numeric(aquifer_df["bootstrap"], errors="coerce").astype("Int64")
aquifer_df["year"] = pd.to_numeric(aquifer_df["year"], errors="coerce").astype("Int64")

for col in ["gw_st", "gw_st_change", "withdrawal"]:
    aquifer_df[col] = pd.to_numeric(aquifer_df[col], errors="coerce")

aquifer_df = aquifer_df[
    aquifer_df["policy"].isin(POLICIES_TO_PLOT)
    & aquifer_df["year"].between(YEAR_MIN, YEAR_MAX)
].copy()

print(f"Aquifer rows loaded: {len(aquifer_df):,}")


# =============================================================================
# 6. Build or load annual economic cache and farmer-profit cache
# =============================================================================
econ_cache_valid = False
farmer_cache_valid = False

if ECON_ANNUAL_CACHE.exists() and not REBUILD_ECON_CACHE:
    try:
        econ_df = pd.read_csv(ECON_ANNUAL_CACHE)
        econ_df = normalize_columns(econ_df)

        require_columns(
            econ_df,
            {"policy", "bootstrap", "year", "avg_profit", "profit_per_water"},
            ECON_ANNUAL_CACHE.name,
        )

        econ_cache_valid = True
        print(f"\nUsing existing annual economic cache:\n  {ECON_ANNUAL_CACHE}")

    except Exception as e:
        print(f"\nAnnual economic cache exists but is not valid: {e}")

if FARMER_PROFIT_CACHE.exists() and not REBUILD_FARMER_PROFIT_CACHE:
    try:
        farmer_profit_df = pd.read_csv(FARMER_PROFIT_CACHE)
        farmer_profit_df = normalize_columns(farmer_profit_df)

        require_columns(
            farmer_profit_df,
            {"policy", "bootstrap", "agentid", "avg_annual_profit_farmer"},
            FARMER_PROFIT_CACHE.name,
        )

        farmer_cache_valid = True
        print(f"\nUsing existing farmer-profit cache:\n  {FARMER_PROFIT_CACHE}")

    except Exception as e:
        print(f"\nFarmer-profit cache exists but is not valid: {e}")

need_profit_files = (not econ_cache_valid) or (not farmer_cache_valid)

if need_profit_files:
    print("\nBuilding missing economic/farmer-profit caches from profit_distribution files...")

    profit_files = sorted(DATA_FOR_FIGURES_DIR.glob("profit_distribution_*.csv"))

    if not profit_files:
        raise FileNotFoundError(
            f"No profit_distribution_*.csv files found in {DATA_FOR_FIGURES_DIR}"
        )

    econ_rows = []
    farmer_profit_rows = []

    for path in tqdm(profit_files, desc="Profit distribution files"):
        try:
            df = read_csv_autodelim(path)
            df = normalize_columns(df)

            if "policy" in df.columns and df["policy"].notna().any():
                policy = normalize_policy_label(df["policy"].dropna().iloc[0])
            else:
                policy = parse_policy_from_profit_file(path.name)

            if "bootstrap" in df.columns and df["bootstrap"].notna().any():
                bootstrap = int(pd.to_numeric(df["bootstrap"].dropna().iloc[0]))
            else:
                bootstrap = parse_bootstrap_from_name(path.name)

            if policy is None or bootstrap is None:
                continue

            required_cols = {"agentid", "year", "profit", "w", "field_type_rn"}
            if not required_cols.issubset(set(df.columns)):
                continue

            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
            df["w"] = pd.to_numeric(df["w"], errors="coerce")
            df["field_type_rn"] = df["field_type_rn"].astype(str).str.strip().str.lower()

            df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()

            if PROFIT_POPULATION == "irrigation_equipped":
                use_df = df[df["field_type_rn"] == "optimize"].copy()
            elif PROFIT_POPULATION == "all":
                use_df = df.copy()
            else:
                raise ValueError(
                    "PROFIT_POPULATION must be either 'irrigation_equipped' or 'all'."
                )

            if use_df.empty:
                continue

            # Economic water productivity:
            # profit is in $10^4, w is applied water depth in cm.
            use_df["profit_per_water"] = (use_df["profit"] / use_df["w"]).replace(
                [np.inf, -np.inf],
                0,
            )

            if not econ_cache_valid:
                annual_econ = (
                    use_df.groupby("year", as_index=False)
                    .agg(
                        avg_profit=("profit", "mean"),
                        profit_per_water=("profit_per_water", "mean"),
                    )
                )
                annual_econ["policy"] = policy
                annual_econ["bootstrap"] = int(bootstrap)

                econ_rows.append(
                    annual_econ[
                        ["policy", "bootstrap", "year", "avg_profit", "profit_per_water"]
                    ]
                )

            if not farmer_cache_valid:
                per_farmer = (
                    use_df.groupby("agentid", as_index=False)
                    .agg(avg_annual_profit_farmer=("profit", "mean"))
                )
                per_farmer["policy"] = policy
                per_farmer["bootstrap"] = int(bootstrap)

                farmer_profit_rows.append(
                    per_farmer[
                        ["policy", "bootstrap", "agentid", "avg_annual_profit_farmer"]
                    ]
                )

        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    if not econ_cache_valid:
        if not econ_rows:
            raise RuntimeError("No annual economic rows were created.")

        econ_df = pd.concat(econ_rows, ignore_index=True)
        econ_df.to_csv(ECON_ANNUAL_CACHE, index=False)
        print(f"\nSaved annual economic cache:\n  {ECON_ANNUAL_CACHE}")

    if not farmer_cache_valid:
        if not farmer_profit_rows:
            raise RuntimeError("No farmer-profit rows were created.")

        farmer_profit_df = pd.concat(farmer_profit_rows, ignore_index=True)
        farmer_profit_df.to_csv(FARMER_PROFIT_CACHE, index=False)
        print(f"\nSaved farmer-profit cache:\n  {FARMER_PROFIT_CACHE}")

# Standardize economic cache
econ_df = normalize_columns(econ_df)
econ_df["policy"] = econ_df["policy"].map(normalize_policy_label)
econ_df["bootstrap"] = pd.to_numeric(econ_df["bootstrap"], errors="coerce").astype("Int64")
econ_df["year"] = pd.to_numeric(econ_df["year"], errors="coerce").astype("Int64")
econ_df["avg_profit"] = pd.to_numeric(econ_df["avg_profit"], errors="coerce")
econ_df["profit_per_water"] = pd.to_numeric(econ_df["profit_per_water"], errors="coerce")

econ_df = econ_df[
    econ_df["policy"].isin(POLICIES_TO_PLOT)
    & econ_df["year"].between(YEAR_MIN, YEAR_MAX)
].copy()

# Standardize farmer-profit cache
farmer_profit_df = normalize_columns(farmer_profit_df)
farmer_profit_df["policy"] = farmer_profit_df["policy"].map(normalize_policy_label)
farmer_profit_df["bootstrap"] = pd.to_numeric(
    farmer_profit_df["bootstrap"],
    errors="coerce",
).astype("Int64")
farmer_profit_df["avg_annual_profit_farmer"] = pd.to_numeric(
    farmer_profit_df["avg_annual_profit_farmer"],
    errors="coerce",
)

farmer_profit_df = farmer_profit_df[
    farmer_profit_df["policy"].isin(POLICIES_TO_PLOT)
].copy()

print(f"Annual economic rows loaded: {len(econ_df):,}")
print(f"Farmer-profit rows loaded: {len(farmer_profit_df):,}")


# =============================================================================
# 7. Build summary CSVs
# =============================================================================
print("\nBuilding summary CSVs for figures...")

# Aquifer annual medians
aquifer_summary_rows = []

for policy in POLICIES_TO_PLOT:
    for condition in CONDITION_ORDER:
        bs = CONDITION_BOOTSTRAPS[condition]

        sub = aquifer_df[
            (aquifer_df["policy"] == policy)
            & (aquifer_df["bootstrap"].isin(bs))
        ].copy()

        for metric in ["gw_st", "gw_st_change", "withdrawal"]:
            ann = summarize_annual_median(sub[["year", metric]].dropna(), metric)
            ann["policy"] = policy
            ann["condition"] = condition
            ann["metric"] = metric
            ann["n_bootstraps"] = len(bs)
            aquifer_summary_rows.append(ann)

aquifer_summary = pd.concat(aquifer_summary_rows, ignore_index=True)
aquifer_summary.to_csv(AQUIFER_SUMMARY_OUT, index=False)

# Economic annual medians
econ_summary_rows = []

for policy in POLICIES_TO_PLOT:
    for condition in CONDITION_ORDER:
        bs = CONDITION_BOOTSTRAPS[condition]

        sub = econ_df[
            (econ_df["policy"] == policy)
            & (econ_df["bootstrap"].isin(bs))
        ].copy()

        for metric in ["avg_profit", "profit_per_water"]:
            ann = summarize_annual_median(sub[["year", metric]].dropna(), metric)
            ann["policy"] = policy
            ann["condition"] = condition
            ann["metric"] = metric
            ann["n_bootstraps"] = len(bs)
            econ_summary_rows.append(ann)

econ_summary = pd.concat(econ_summary_rows, ignore_index=True)
econ_summary.to_csv(ECON_SUMMARY_OUT, index=False)

# Profit distribution summary
profit_summary_rows = []

for policy in POLICIES_TO_PLOT:
    for condition in CONDITION_ORDER:
        bs = CONDITION_BOOTSTRAPS[condition]

        vals = farmer_profit_df[
            (farmer_profit_df["policy"] == policy)
            & (farmer_profit_df["bootstrap"].isin(bs))
        ]["avg_annual_profit_farmer"].dropna()

        profit_summary_rows.append(
            {
                "policy": policy,
                "condition": condition,
                "n_bootstraps": len(bs),
                "n_profit_points": len(vals),
                "median": vals.median() if len(vals) else np.nan,
                "p25": vals.quantile(0.25) if len(vals) else np.nan,
                "p75": vals.quantile(0.75) if len(vals) else np.nan,
                "p05": vals.quantile(0.05) if len(vals) else np.nan,
                "p95": vals.quantile(0.95) if len(vals) else np.nan,
            }
        )

profit_summary = pd.DataFrame(profit_summary_rows)
profit_summary.to_csv(PROFIT_SUMMARY_OUT, index=False)

# Gini summary
gini_summary_rows = []

for policy in POLICIES_TO_PLOT:
    for condition in CONDITION_ORDER:
        bs = CONDITION_BOOTSTRAPS[condition]

        vals = farmer_profit_df[
            (farmer_profit_df["policy"] == policy)
            & (farmer_profit_df["bootstrap"].isin(bs))
        ]["avg_annual_profit_farmer"].dropna()

        gini_summary_rows.append(
            {
                "policy": policy,
                "condition": condition,
                "n_bootstraps": len(bs),
                "n_profit_points": len(vals),
                "gini": gini_coefficient(vals),
            }
        )

gini_summary = pd.DataFrame(gini_summary_rows)
gini_summary.to_csv(GINI_SUMMARY_OUT, index=False)

print(f"Saved summary CSV: {AQUIFER_SUMMARY_OUT}")
print(f"Saved summary CSV: {ECON_SUMMARY_OUT}")
print(f"Saved summary CSV: {PROFIT_SUMMARY_OUT}")
print(f"Saved summary CSV: {GINI_SUMMARY_OUT}")


# =============================================================================
# 8. Figure 1: Aquifer outcomes
# =============================================================================
print("\nPlotting aquifer figure...")

fig = plt.figure(figsize=FIGSIZE_AQUIFER, dpi=DPI)

outer_gs = make_outer_grid(fig)

aquifer_metrics = [
    {
        "key": "gw_st",
        "ylabel": "(a) Saturated\nthickness\n(m)",
        "zero_relevant": False,
    },
    {
        "key": "gw_st_change",
        "ylabel": "(b) Water\nlevel\nchange (m)",
        "zero_relevant": True,
    },
    {
        "key": "withdrawal",
        "ylabel": "(c) Withdrawal\n($10^4$ m$^3$)",
        "zero_relevant": True,
    },
]

aquifer_shared_ylims = {
    metric["key"]: get_shared_axis_limits_from_summary(
        summary_df=aquifer_summary,
        metric_key=metric["key"],
        policies=SHARED_AXIS_POLICIES,
        zero_relevant=metric["zero_relevant"],
    )
    for metric in aquifer_metrics
}

for policy in POLICIES_TO_PLOT:
    row, col = POLICY_POSITIONS[policy]

    inner_gs = outer_gs[row, col].subgridspec(
        3,
        1,
        hspace=0.25,
    )

    for m_idx, metric in enumerate(aquifer_metrics):
        key = metric["key"]
        ylabel = metric["ylabel"]

        ax = fig.add_subplot(inner_gs[m_idx, 0])
        style_axis(ax)

        axis_values = []

        for condition in CONDITION_ORDER:
            st = CONDITION_STYLES[condition]

            sub = aquifer_summary[
                (aquifer_summary["policy"] == policy)
                & (aquifer_summary["condition"] == condition)
                & (aquifer_summary["metric"] == key)
            ].copy()

            if sub.empty:
                continue

            axis_values.append(sub["median"])

            ax.plot(
                sub["year"],
                sub["median"],
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["lw"],
                alpha=st["alpha"],
                zorder=5,
            )

        if key == "gw_st_change":
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

        if AQUIFER_YLIM_MODE == "bau_separate_shared_policy" and policy != "BAU":
            ylim = aquifer_shared_ylims.get(key)
        else:
            # Original behavior: each policy panel gets its own independent axis.
            ylim = get_axis_limits_from_values(
                axis_values,
                zero_relevant=metric["zero_relevant"],
            )
        
        if ylim is not None:
            ax.set_ylim(ylim)

        format_year_axis(
            ax,
            show_tick_labels=(m_idx == 2),
            show_xlabel=(m_idx == 2 and policy == "PWPR"),
        )

        style_axis(ax)
        
        if m_idx == 0:
            set_policy_title(ax, policy)

        show_metric_label = (col == 0 and row < 2) or (policy == "PWPR")

        if show_metric_label:
            add_metric_label(ax, ylabel, x=-0.35)

add_blank_outer_cells(fig, outer_gs)

add_condition_legend(fig, line_style=True)

plt.savefig(AQUIFER_FIG_OUT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print(f"Saved aquifer figure: {AQUIFER_FIG_OUT}")


# =============================================================================
# 9. Figure 2: Economic outcomes
# =============================================================================
print("\nPlotting economic figure...")

fig = plt.figure(figsize=FIGSIZE_ECON, dpi=DPI)
outer_gs = make_outer_grid(fig)

economic_metrics = [
    {
        "key": "avg_profit",
        "ylabel": "(a) Average\nprofit among\nfarmers\n($\$ 10^4$)",
        "zero_relevant": False,
    },
    {
        "key": "profit_per_water",
        "ylabel": EWP_YLABEL,
        "zero_relevant": False,
    },
]

# Shared y-limit for panel (a): governance policies only
econ_panel_a_shared_ylim = get_shared_axis_limits_from_summary(
    summary_df=econ_summary,
    metric_key="avg_profit",
    policies=SHARED_AXIS_POLICIES,
    zero_relevant=False,
)

# Shared y-limit for panel (b): all policies, including BAU
econ_panel_b_shared_ylim = get_shared_axis_limits_from_summary(
    summary_df=econ_summary,
    metric_key="profit_per_water",
    policies=POLICIES_TO_PLOT,
    zero_relevant=False,
)

for policy in POLICIES_TO_PLOT:
    row, col = POLICY_POSITIONS[policy]

    inner_gs = outer_gs[row, col].subgridspec(
        2,
        1,
        hspace=0.25,
    )

    for m_idx, metric in enumerate(economic_metrics):
        key = metric["key"]
        ylabel = metric["ylabel"]

        ax = fig.add_subplot(inner_gs[m_idx, 0])
        style_axis(ax)

        axis_values = []

        for condition in CONDITION_ORDER:
            st = CONDITION_STYLES[condition]

            sub = econ_summary[
                (econ_summary["policy"] == policy)
                & (econ_summary["condition"] == condition)
                & (econ_summary["metric"] == key)
            ].copy()

            if sub.empty:
                continue

            axis_values.append(sub["median"])

            ax.plot(
                sub["year"],
                sub["median"],
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["lw"],
                alpha=st["alpha"],
                zorder=5,
            )

        if key == "avg_profit":
            if ECON_PANEL_A_YLIM_MODE == "bau_separate_shared_policy":
                if policy == "BAU":
                    ylim = get_axis_limits_from_values(
                        axis_values,
                        zero_relevant=metric["zero_relevant"],
                    )
                else:
                    ylim = econ_panel_a_shared_ylim
            else:
                # original independent behavior
                ylim = get_axis_limits_from_values(
                    axis_values,
                    zero_relevant=metric["zero_relevant"],
                )
        
        elif key == "profit_per_water":
            if ECON_PANEL_B_YLIM_MODE == "all_shared":
                ylim = econ_panel_b_shared_ylim
            else:
                # original independent behavior
                ylim = get_axis_limits_from_values(
                    axis_values,
                    zero_relevant=metric["zero_relevant"],
                )
        
        if ylim is not None:
            ax.set_ylim(ylim)

        format_year_axis(
            ax,
            show_tick_labels=(m_idx == 1),
            show_xlabel=(m_idx == 1 and policy == "PWPR"),
        )

        style_axis(ax)
        
        if m_idx == 0:
            set_policy_title(ax, policy)

        show_metric_label = (col == 0 and row < 2) or (policy == "PWPR")

        if show_metric_label:
            add_metric_label(ax, ylabel, x=-0.38)

add_blank_outer_cells(fig, outer_gs)

add_condition_legend(fig, line_style=True)

plt.savefig(ECON_FIG_OUT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print(f"Saved economic figure: {ECON_FIG_OUT}")


# =============================================================================
# 10. Figure 3: Profit distributions, horizontal boxplots
# =============================================================================
print("\nPlotting profit-distribution figure...")

all_profit_vals = farmer_profit_df["avg_annual_profit_farmer"].dropna()

if all_profit_vals.empty:
    raise RuntimeError("No average annual farmer-profit values available.")

x_min = -2 #min(0, all_profit_vals.quantile(0.001) - 0.25)
x_max = 12#all_profit_vals.quantile(0.999) + 0.25

fig = plt.figure(figsize=FIGSIZE_SINGLE, dpi=DPI)
outer_gs = make_outer_grid_single(fig)

for policy in POLICIES_TO_PLOT:
    row, col = POLICY_POSITIONS[policy]
    ax = fig.add_subplot(outer_gs[row, col])
    style_axis(ax)

    box_data = []
    y_positions = np.arange(len(CONDITION_ORDER), 0, -1)

    for condition in CONDITION_ORDER:
        bs = CONDITION_BOOTSTRAPS[condition]

        vals = farmer_profit_df[
            (farmer_profit_df["policy"] == policy)
            & (farmer_profit_df["bootstrap"].isin(bs))
        ]["avg_annual_profit_farmer"].dropna().values

        if len(vals) == 0:
            vals = np.array([np.nan])

        box_data.append(vals)

    bp = ax.boxplot(
        box_data,
        vert=False,
        positions=y_positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(color="#666666", linewidth=1.5),
        capprops=dict(color="#666666", linewidth=1.5),
        boxprops=dict(linewidth=1.5),
    )

    for patch, condition in zip(bp["boxes"], CONDITION_ORDER):
        patch.set_facecolor(CONDITION_STYLES[condition]["color"])
        patch.set_alpha(0.58)
        patch.set_edgecolor("#444444")

    ax.set_xlim(x_min, x_max)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [CONDITION_SHORT_LABELS[c] for c in CONDITION_ORDER],
        fontsize=SINGLE_YTICK_FONTSIZE,
        color="black",
    )
    
    ax.set_title(policy, fontsize=SINGLE_TITLE_FONTSIZE, pad=8)
    ax.tick_params(axis="x", labelsize=SINGLE_XTICK_FONTSIZE, pad=4)
    
    if policy == "PWPR":
        ax.set_xlabel("Profit ($\$10^4$)", fontsize=SINGLE_XLABEL_FONTSIZE, labelpad=6)
    else:
        ax.set_xlabel("")

add_blank_outer_cells(fig, outer_gs)

add_condition_legend(fig, line_style=False)

plt.savefig(PROFIT_FIG_OUT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print(f"Saved profit-distribution figure: {PROFIT_FIG_OUT}")


# =============================================================================
# 11. Figure 4: Gini coefficients, horizontal bars
# =============================================================================
print("\nPlotting Gini figure...")

fig = plt.figure(figsize=FIGSIZE_SINGLE, dpi=DPI)
outer_gs = make_outer_grid_single(fig)

global_gini_max = gini_summary["gini"].max()
x_max = max(0.6, global_gini_max * 1.25)

for policy in POLICIES_TO_PLOT:
    row, col = POLICY_POSITIONS[policy]
    ax = fig.add_subplot(outer_gs[row, col])
    style_axis(ax)

    sub = gini_summary[gini_summary["policy"] == policy].copy()
    sub["condition"] = pd.Categorical(
        sub["condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )
    sub = sub.sort_values("condition")

    y_positions = np.arange(len(CONDITION_ORDER), 0, -1)
    values = sub["gini"].values

    ax.barh(
        y_positions,
        values,
        color=[CONDITION_STYLES[c]["color"] for c in CONDITION_ORDER],
        alpha=0.70,
        edgecolor="#444444",
        linewidth=0.9,
        height=0.58,
    )

    for y, val in zip(y_positions, values):
        if pd.notna(val):
            ax.text(
                val + 0.01,
                y,
                f"{val:.3f}",
                ha="left",
                va="center",
                fontsize=20,
                # fontweight="bold",
            )

    ax.set_xlim(0, x_max)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [CONDITION_SHORT_LABELS[c] for c in CONDITION_ORDER],
        fontsize=SINGLE_YTICK_FONTSIZE,
        color="black",
    )
    
    #ax.set_xlabel("Gini coefficient", fontsize=SINGLE_XLABEL_FONTSIZE, labelpad=6)
    ax.tick_params(axis="x", labelsize=SINGLE_XTICK_FONTSIZE, pad=4)
    ax.set_title(policy, fontsize=SINGLE_TITLE_FONTSIZE, pad=8)
    if policy == "PWPR":
        ax.set_xlabel("Gini Coefficient", fontsize=SINGLE_XLABEL_FONTSIZE, labelpad=6)
    else:
        ax.set_xlabel("")

add_blank_outer_cells(fig, outer_gs)

add_condition_legend(fig, line_style=False)

plt.savefig(GINI_FIG_OUT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print(f"Saved Gini figure: {GINI_FIG_OUT}")


# =============================================================================
# 12. Final printout
# =============================================================================
print("\nSaved figures:")
print(f"  {AQUIFER_FIG_OUT}")
print(f"  {ECON_FIG_OUT}")
print(f"  {PROFIT_FIG_OUT}")
print(f"  {GINI_FIG_OUT}")

print("\nSaved figure-value CSVs:")
print(f"  {AQUIFER_SUMMARY_OUT}")
print(f"  {ECON_SUMMARY_OUT}")
print(f"  {PROFIT_SUMMARY_OUT}")
print(f"  {GINI_SUMMARY_OUT}")

print("\nDone.")