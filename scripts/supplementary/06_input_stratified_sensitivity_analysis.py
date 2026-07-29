# -----------------------------------------------------------------------------
# 06_input_stratified_sensitivity_analysis.py
#
# Purpose:
#   Create an input-stratified supplementary sensitivity table using:
#
#   1. Bootstrap climate/market labels from:
#        inputs/supplementary/bootstrap_climate_market_groups.csv
#
#   2. Raw policy Excel outputs for aquifer metrics:
#        outputs/<policy>_runs/*.xlsx
#
#   3. Existing profit_distribution files for economic metrics and Gini:
#        outputs/data_for_figures/profit_distribution_*.csv
#
# Output:
#   inputs/supplementary/input_stratified_sensitivity_table.csv
#
# Cache files created by this script:
#   inputs/supplementary/cache_aquifer_system_metrics_from_excel.csv
#   inputs/supplementary/cache_economic_metrics_from_profit_distribution.csv
#   inputs/supplementary/cache_farmer_avg_profit_for_gini.csv
#
# Notes:
#   - This script does NOT rely on bootstrap_policy_outcomes_with_groups.csv.
#   - The full ensemble row is a reference across all 500 bootstraps.
#   - Dry/wet and low/high-return rows summarize subsets of 125 bootstraps.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
import io
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm


# ================================================================
# 1. Path automation
# ================================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_FOR_FIGURES_DIR = OUTPUTS_DIR / "data_for_figures"
SUPP_DIR = PROJECT_ROOT / "inputs" / "supplementary"

GROUPS_FILE = SUPP_DIR / "bootstrap_climate_market_groups.csv"

OUT_FILE = SUPP_DIR / "input_stratified_sensitivity_table.csv"

AQUIFER_CACHE = SUPP_DIR / "cache_aquifer_system_metrics_from_excel.csv"
ECON_CACHE = SUPP_DIR / "cache_economic_metrics_from_profit_distribution.csv"
GINI_CACHE = SUPP_DIR / "cache_farmer_avg_profit_for_gini.csv"

print(f"Project root: {PROJECT_ROOT}")
print(f"Bootstrap labels: {GROUPS_FILE}")
print(f"Output file: {OUT_FILE}")


# ================================================================
# 2. Configuration
# ================================================================
YEAR_MIN, YEAR_MAX = 2002, 2022

# Same initial saturated thickness used in your figure-prep workflow.
INITIAL_GW_ST = 24.203292398301375

POLICY_PATHS = {
    "BAU": OUTPUTS_DIR / "baseline_runs",
    "UR": OUTPUTS_DIR / "ur_runs",
    "FB": OUTPUTS_DIR / "fb_runs",
    "FB-CB": OUTPUTS_DIR / "fb_cb_runs",
    "PR-I": OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
    "R+PR": OUTPUTS_DIR / "r_plus_pr_runs",
}

POLICY_ORDER = ["BAU", "UR", "FB", "FB-CB", "PR-I", "PR-II", "R+PR"]

# Set these to True if you want to force rebuilding from source files.
REBUILD_AQUIFER_CACHE = False
REBUILD_ECON_GINI_CACHE = False


# ================================================================
# 3. Helper functions
# ================================================================
def read_csv_autodelim(path: Path, nrows=None) -> pd.DataFrame:
    """Read CSV with delimiter auto-detection."""
    raw = path.read_bytes()
    head = raw[:2048].decode(errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","

    return pd.read_csv(io.BytesIO(raw), delimiter=delim, nrows=nrows)


def norm_col(c: object) -> str:
    """Normalize column names."""
    s = str(c).strip()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    return s.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    return df


def normalize_policy_label(x: object) -> str:
    """Normalize policy labels."""
    s = str(x).strip()

    mapping = {
        "BAU": "BAU",
        "UR": "UR",
        "FB": "FB",
        "FB_CB": "FB-CB",
        "FB-CB": "FB-CB",
        "PR_I": "PR-I",
        "PR-I": "PR-I",
        "PRI": "PR-I",
        "PR_II": "PR-II",
        "PR-II": "PR-II",
        "PRII": "PR-II",
        "R_PR": "R+PR",
        "R+PR": "R+PR",
        "R_PLUS_PR": "R+PR",
    }

    return mapping.get(s.upper(), s)


def parse_bootstrap_from_name(fname: str) -> int | None:
    """Extract bootstrap number from filenames containing '_b_<number>'."""
    m = re.search(r"_b_(\d+)", fname)
    if m:
        return int(m.group(1))

    m = re.search(r"_([0-9]+)(?=\.[A-Za-z]+$)", fname)
    if m:
        return int(m.group(1))

    return None


def parse_policy_from_profit_file(fname: str) -> str | None:
    """Infer policy from profit_distribution_<policy>_b_<id>.csv."""
    m = re.search(r"profit_distribution_(.+?)_b_\d+", fname)
    if not m:
        return None

    return normalize_policy_label(m.group(1))


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column using normalized, case-insensitive matching."""
    lookup = {norm_col(c): c for c in df.columns}

    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]

    return None


def list_excel_files(run_dir: Path) -> list[Path]:
    """List valid Excel outputs, excluding temporary Excel lock files."""
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

    This matches the manuscript/Lorenz-curve logic:
      - calculate each farmer-bootstrap average annual profit first;
      - pool those values within the policy/group;
      - calculate one Gini.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan

    # Scaling does not change Gini, but profit is stored in 1e4 dollars.
    x = np.sort(x * 1e4)

    total = np.sum(x)
    n = x.size

    if total <= 0:
        return np.nan

    index = np.arange(1, n + 1)
    rank_sum = np.sum((n + 1 - index) * x)

    return float((n + 1 - 2 * rank_sum / total) / n)


# ================================================================
# 4. Load bootstrap climate/market groups
# ================================================================
if not GROUPS_FILE.exists():
    raise FileNotFoundError(
        f"Missing {GROUPS_FILE}. Run 05_classify_bootstrap_climate_market.py first."
    )

groups = pd.read_csv(GROUPS_FILE)
groups = normalize_columns(groups)

required_group_cols = {"bootstrap", "precip_group", "market_return_group"}
missing_group_cols = required_group_cols - set(groups.columns)
if missing_group_cols:
    raise ValueError(
        f"{GROUPS_FILE.name} is missing columns: {missing_group_cols}\n"
        f"Available columns: {list(groups.columns)}"
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

strata = [
    ("Full ensemble", "All", all_bootstraps),
    ("Precipitation", "Dry", dry_bootstraps),
    ("Precipitation", "Wet", wet_bootstraps),
    ("Market return", "Low return", low_return_bootstraps),
    ("Market return", "High return", high_return_bootstraps),
]

print("\nBootstrap strata:")
for stratification, group_name, bootstraps in strata:
    print(f"  {stratification} | {group_name}: {len(bootstraps)} bootstraps")


# ================================================================
# 5. Build/load aquifer cache from raw Excel System sheets
# ================================================================
if AQUIFER_CACHE.exists() and not REBUILD_AQUIFER_CACHE:
    print("\nUsing cached aquifer metrics:")
    print(f"  {AQUIFER_CACHE}")
    aquifer_df = pd.read_csv(AQUIFER_CACHE)
    aquifer_df = normalize_columns(aquifer_df)

else:
    print("\nBuilding aquifer cache from raw Excel System sheets...")

    aquifer_rows = []

    for policy in POLICY_ORDER:
        run_dir = POLICY_PATHS[policy]
        excel_files = list_excel_files(run_dir)

        if not excel_files:
            print(f"Skipping {policy}: no Excel files found in {run_dir}")
            continue

        print(f"\nProcessing System sheets for {policy}: {len(excel_files)} files")

        for path in tqdm(excel_files, desc=f"{policy} System", leave=False):
            bootstrap = parse_bootstrap_from_name(path.name)

            if bootstrap is None:
                continue

            try:
                sys_df = pd.read_excel(path, sheet_name="System")
                sys_df.columns = [str(c).strip() for c in sys_df.columns]

                year_col = find_col(sys_df, ["year", "Year"])
                gw_col = find_col(sys_df, ["GW_st", "gw_st", "saturated_thickness"])
                withdrawal_col = find_col(
                    sys_df,
                    ["withdrawal", "Withdrawal", "withdrawals", "Withdrawals"],
                )

                if year_col is None or gw_col is None or withdrawal_col is None:
                    raise ValueError(
                        f"Missing required System columns. Found: {list(sys_df.columns)}"
                    )

                sys_df[year_col] = pd.to_numeric(sys_df[year_col], errors="coerce")
                sys_df[gw_col] = pd.to_numeric(sys_df[gw_col], errors="coerce")
                sys_df[withdrawal_col] = pd.to_numeric(
                    sys_df[withdrawal_col], errors="coerce"
                )

                sys_df = (
                    sys_df[sys_df[year_col].between(YEAR_MIN, YEAR_MAX)]
                    .sort_values(year_col)
                    .copy()
                )

                if sys_df.empty:
                    continue

                # Annual saturated-thickness change.
                # First modeled year is relative to fixed initial saturated thickness.
                sys_df["GW_st_change"] = sys_df[gw_col].diff()
                first_idx = sys_df.index[0]
                sys_df.loc[first_idx, "GW_st_change"] = (
                    sys_df.loc[first_idx, gw_col] - INITIAL_GW_ST
                )

                out = pd.DataFrame(
                    {
                        "policy": policy,
                        "bootstrap": int(bootstrap),
                        "year": sys_df[year_col].astype(int),
                        "GW_st": sys_df[gw_col],
                        "GW_st_change": sys_df["GW_st_change"],
                        "withdrawal": sys_df[withdrawal_col],
                    }
                )

                aquifer_rows.append(out)

            except Exception as e:
                print(f"  Error reading System sheet from {path.name}: {e}")

    if not aquifer_rows:
        raise RuntimeError("No aquifer rows were created from Excel outputs.")

    aquifer_df = pd.concat(aquifer_rows, ignore_index=True)

    AQUIFER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    aquifer_df.to_csv(AQUIFER_CACHE, index=False)

    print("\nSaved aquifer cache:")
    print(f"  {AQUIFER_CACHE}")

# ================================================================
# After aquifer cache is either loaded or built
# ================================================================
aquifer_df = normalize_columns(aquifer_df)

aquifer_df["policy"] = aquifer_df["policy"].map(normalize_policy_label)
aquifer_df["bootstrap"] = pd.to_numeric(aquifer_df["bootstrap"], errors="coerce").astype("Int64")
aquifer_df["year"] = pd.to_numeric(aquifer_df["year"], errors="coerce")

for col in ["gw_st", "gw_st_change", "withdrawal"]:
    if col not in aquifer_df.columns:
        raise ValueError(
            f"Aquifer cache is missing required column '{col}'. "
            f"Available columns: {list(aquifer_df.columns)}"
        )
    aquifer_df[col] = pd.to_numeric(aquifer_df[col], errors="coerce")

aquifer_df = aquifer_df[
    aquifer_df["year"].between(YEAR_MIN, YEAR_MAX)
    & aquifer_df["policy"].isin(POLICY_ORDER)
].copy()

print("\nAquifer data loaded:")
print(f"  Rows: {len(aquifer_df):,}")
print(f"  Policy-bootstrap pairs: {aquifer_df[['policy', 'bootstrap']].drop_duplicates().shape[0]:,}")
print(f"  Policies: {sorted(aquifer_df['policy'].dropna().unique())}")


# ================================================================
# 6. Build/load economic and Gini caches from profit_distribution files
# ================================================================
if (
    ECON_CACHE.exists()
    and GINI_CACHE.exists()
    and not REBUILD_ECON_GINI_CACHE
):
    print("\nUsing cached economic and Gini inputs:")
    print(f"  {ECON_CACHE}")
    print(f"  {GINI_CACHE}")

    econ_df = pd.read_csv(ECON_CACHE)
    econ_df = normalize_columns(econ_df)

    farmer_profit_df = pd.read_csv(GINI_CACHE)
    farmer_profit_df = normalize_columns(farmer_profit_df)

else:
    print("\nBuilding economic and Gini caches from profit_distribution files...")

    profit_files = sorted(DATA_FOR_FIGURES_DIR.glob("profit_distribution_*.csv"))

    if not profit_files:
        raise FileNotFoundError(
            f"No profit_distribution_*.csv files found in {DATA_FOR_FIGURES_DIR}."
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
                print(f"Skipping {path.name}: could not infer policy/bootstrap.")
                continue

            required_cols = {"agentid", "year", "profit", "w", "field_type_rn"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"Skipping {path.name}: missing columns {missing}")
                continue

            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
            df["w"] = pd.to_numeric(df["w"], errors="coerce")
            df["field_type_rn"] = (
                df["field_type_rn"].astype(str).str.strip().str.lower()
            )

            df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()

            irrig = df[df["field_type_rn"] == "optimize"].copy()

            if irrig.empty:
                continue

            # Economic water productivity, matching plot_economic_output logic.
            irrig["profit_per_water"] = (irrig["profit"] / irrig["w"]).replace(
                [np.inf, -np.inf],
                0,
            )

            # Annual economic metrics for this policy-bootstrap.
            annual_econ = (
                irrig.groupby("year", as_index=False)
                .agg(
                    avg_profit=("profit", "mean"),
                    profit_per_water=("profit_per_water", "mean"),
                )
            )

            # Collapse to one policy-bootstrap value by averaging across years.
            # Later, the sensitivity table takes the median across bootstraps.
            econ_rows.append(
                {
                    "policy": policy,
                    "bootstrap": int(bootstrap),
                    "mean_annual_avg_profit": annual_econ["avg_profit"].mean(),
                    "mean_annual_profit_per_water": annual_econ[
                        "profit_per_water"
                    ].mean(),
                }
            )

            # Farmer-bootstrap average annual profit for Gini.
            per_farmer = (
                irrig.groupby("agentid", as_index=False)
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

    if not econ_rows:
        raise RuntimeError("No economic rows were created from profit_distribution files.")

    if not farmer_profit_rows:
        raise RuntimeError("No farmer-profit rows were created for Gini.")

    econ_df = pd.DataFrame(econ_rows)
    farmer_profit_df = pd.concat(farmer_profit_rows, ignore_index=True)

    ECON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    econ_df.to_csv(ECON_CACHE, index=False)
    farmer_profit_df.to_csv(GINI_CACHE, index=False)

    print("\nSaved economic/Gini caches:")
    print(f"  {ECON_CACHE}")
    print(f"  {GINI_CACHE}")

econ_df["policy"] = econ_df["policy"].map(normalize_policy_label)
econ_df["bootstrap"] = pd.to_numeric(econ_df["bootstrap"], errors="coerce").astype("Int64")

for col in ["mean_annual_avg_profit", "mean_annual_profit_per_water"]:
    econ_df[col] = pd.to_numeric(econ_df[col], errors="coerce")

econ_df = econ_df[econ_df["policy"].isin(POLICY_ORDER)].copy()

farmer_profit_df["policy"] = farmer_profit_df["policy"].map(normalize_policy_label)
farmer_profit_df["bootstrap"] = pd.to_numeric(
    farmer_profit_df["bootstrap"], errors="coerce"
).astype("Int64")
farmer_profit_df["avg_annual_profit_farmer"] = pd.to_numeric(
    farmer_profit_df["avg_annual_profit_farmer"], errors="coerce"
)

farmer_profit_df = farmer_profit_df[
    farmer_profit_df["policy"].isin(POLICY_ORDER)
].copy()

print("\nEconomic/Gini data loaded:")
print(f"  Economic policy-bootstrap rows: {len(econ_df):,}")
print(f"  Farmer-bootstrap profit rows for Gini: {len(farmer_profit_df):,}")
print(f"  Policies: {sorted(econ_df['policy'].dropna().unique())}")


# ================================================================
# 7. Summarize by policy and input stratum
# ================================================================
def summarize_policy_stratum(policy: str, bootstraps: set[int]) -> dict:
    # ------------------------------------------------------------
    # Aquifer metrics
    # ------------------------------------------------------------
    aq = aquifer_df[
        (aquifer_df["policy"] == policy)
        & (aquifer_df["bootstrap"].isin(bootstraps))
    ].copy()

    if aq.empty:
        n_aquifer_bootstraps = 0
        median_final_gw_st = np.nan
        median_mean_annual_gw_st_change = np.nan
        median_mean_annual_withdrawal = np.nan
    else:
        n_aquifer_bootstraps = aq["bootstrap"].nunique()

        final_rows = (
            aq.sort_values(["bootstrap", "year"])
            .groupby("bootstrap", as_index=False)
            .tail(1)
        )

        per_boot_aq = (
            aq.groupby("bootstrap", as_index=False)
            .agg(
                mean_annual_gw_st_change=("gw_st_change", "mean"),
                mean_annual_withdrawal=("withdrawal", "mean"),
            )
        )

        median_final_gw_st = final_rows["gw_st"].median()
        median_mean_annual_gw_st_change = per_boot_aq[
            "mean_annual_gw_st_change"
        ].median()
        median_mean_annual_withdrawal = per_boot_aq[
            "mean_annual_withdrawal"
        ].median()

    # ------------------------------------------------------------
    # Economic metrics
    # ------------------------------------------------------------
    ec = econ_df[
        (econ_df["policy"] == policy)
        & (econ_df["bootstrap"].isin(bootstraps))
    ].copy()

    if ec.empty:
        n_econ_bootstraps = 0
        median_mean_annual_avg_profit = np.nan
        median_mean_annual_profit_per_water = np.nan
    else:
        n_econ_bootstraps = ec["bootstrap"].nunique()
        median_mean_annual_avg_profit = ec["mean_annual_avg_profit"].median()
        median_mean_annual_profit_per_water = ec[
            "mean_annual_profit_per_water"
        ].median()

    # ------------------------------------------------------------
    # Gini coefficient
    # ------------------------------------------------------------
    fp = farmer_profit_df[
        (farmer_profit_df["policy"] == policy)
        & (farmer_profit_df["bootstrap"].isin(bootstraps))
    ].copy()

    n_profit_points_for_gini = len(fp)

    if fp.empty:
        gini = np.nan
    else:
        gini = gini_coefficient(fp["avg_annual_profit_farmer"])

    return {
        "n_aquifer_bootstraps": n_aquifer_bootstraps,
        "n_econ_bootstraps": n_econ_bootstraps,
        "n_profit_points_for_gini": n_profit_points_for_gini,
        "median_final_GW_st": median_final_gw_st,
        "median_mean_annual_GW_st_change": median_mean_annual_gw_st_change,
        "median_mean_annual_withdrawal": median_mean_annual_withdrawal,
        "median_mean_annual_avg_profit": median_mean_annual_avg_profit,
        "median_mean_annual_profit_per_water": median_mean_annual_profit_per_water,
        "gini_avg_annual_profit_irrigators": gini,
    }


summary_rows = []

for stratification, group_name, bootstraps in strata:
    for policy in POLICY_ORDER:
        metrics = summarize_policy_stratum(policy, bootstraps)

        summary_rows.append(
            {
                "stratification": stratification,
                "group": group_name,
                "n_bootstraps": len(bootstraps),
                "Policy": policy,
                **metrics,
            }
        )

summary = pd.DataFrame(summary_rows)

# Ordering
strat_order = {
    "Full ensemble": 0,
    "Precipitation": 1,
    "Market return": 2,
}

group_order = {
    "All": 0,
    "Dry": 1,
    "Wet": 2,
    "Low return": 3,
    "High return": 4,
}

policy_order = {p: i for i, p in enumerate(POLICY_ORDER)}

summary["strat_order"] = summary["stratification"].map(strat_order)
summary["group_order"] = summary["group"].map(group_order)
summary["policy_order"] = summary["Policy"].map(policy_order)

summary = (
    summary.sort_values(["strat_order", "group_order", "policy_order"])
    .drop(columns=["strat_order", "group_order", "policy_order"])
    .reset_index(drop=True)
)


# ================================================================
# 8. Save output
# ================================================================
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_FILE, index=False)

print("\n" + "-" * 80)
print(f"Saved input-stratified sensitivity table:")
print(f"  {OUT_FILE}")
print(f"Rows: {len(summary)}")
print("-" * 80)

preview_cols = [
    "stratification",
    "group",
    "n_bootstraps",
    "Policy",
    "median_final_GW_st",
    "median_mean_annual_GW_st_change",
    "median_mean_annual_withdrawal",
    "median_mean_annual_avg_profit",
    "median_mean_annual_profit_per_water",
    "gini_avg_annual_profit_irrigators",
]

print("\nPreview:")
print(summary[preview_cols].to_string(index=False))

print("\nDone.")