# -----------------------------------------------------------------------------
# 01_prepare_data_for_figures.py
#
# Consolidates model outputs into figure-ready CSV files.
#
# Outputs:
#   1. aquifer_and_crop_properties.csv
#      - Hydrology, crop shares, rainfed share
#      - Includes p10, p25, p50, p75, p90 columns
#
#   2. economic_outcomes.csv
#      - Profit, losses, irrigation-volume economic metrics
#      - Includes FB-II accounting columns
#      - Includes p10, p25, p50, p75, p90 columns
#
#   3. cutoff_diagnostics.csv
#      - PR-I and PR-II cutoff diagnostics from zero_irrigation_reason
#      - Includes p10, p25, p50, p75, p90 columns
#
#   4. profit_distribution_<policy>_b_<id>.csv
#      - Farmer/field-level data for distribution figures
# -----------------------------------------------------------------------------

import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1. Paths and configuration
# -----------------------------------------------------------------------------
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INPUTS_DIR = PROJECT_ROOT / "inputs"
DATA_FOR_FIGURES_DIR = OUTPUTS_DIR / "data_for_figures"
BOOTSTRAP_DATA_DIR = INPUTS_DIR / "bootstrap_samples"

DATA_FOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

POLICY_PATHS = {
    "BAU":   OUTPUTS_DIR / "baseline_runs",
    "UR":    OUTPUTS_DIR / "ur_runs",
    "FB-I":    OUTPUTS_DIR / "fb_runs",
    "FB-II": OUTPUTS_DIR / "fb_cb_runs",
    "PR-I":  OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
    "PWPR":  OUTPUTS_DIR / "r_plus_pr_runs",
}

POLICIES_TO_RUN = [
    "BAU",
    "UR",
    "FB-I",
    "FB-II",
    "PR-I",
    "PR-II",
    "PWPR",
]

RUN_AQUIFER_CROP = True
RUN_ECONOMIC = False
RUN_CUTOFF = True
RUN_PROFIT_DISTRIBUTION = True

CUTOFF_POLICIES = ["PR-I", "PR-II"]

INITIAL_GW_ST = 24.203292398301375

PERCENTILES = {
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
}

print("--------------------------------------------------")
print("Preparing data for figures")
print("--------------------------------------------------")
print(f"Project root: {PROJECT_ROOT}")
print(f"Outputs directory: {OUTPUTS_DIR}")
print(f"Data-for-figures directory: {DATA_FOR_FIGURES_DIR}")
print(f"Policies: {', '.join(POLICIES_TO_RUN)}")
print("--------------------------------------------------")


# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def parse_bootstrap_id(fname: str):
    """Extract bootstrap id from result filename."""
    m = re.search(r"_b_(\d+)", fname)
    if not m:
        return None
    return int(m.group(1))


def policy_slug(policy_name: str):
    """Create a safe policy name for filenames."""
    return re.sub(r"[^A-Za-z0-9]+", "_", policy_name)


def get_excel_files(directory: Path):
    """Return sorted Excel files in a directory."""
    if not directory.exists():
        return []
    return sorted([p for p in directory.glob("*.xlsx")])


def safe_read_excel(path: Path, sheet_name: str):
    """Read Excel sheet safely."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        print(f"    Warning: could not read {sheet_name} from {path.name}: {e}")
        return None


def add_agent_numeric_id(df: pd.DataFrame):
    """Add numeric AgentID parsed from text IDs."""
    df = df.copy()
    df["AgentID_numeric"] = (
        df["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
    )
    return df


def get_irrigation_equipped_status(fields: pd.DataFrame):
    """
    Returns one row per farmer-year indicating whether the farmer is
    irrigation-equipped, based on field_type_rn == optimize.
    """
    f = add_agent_numeric_id(fields)

    status = (
        f.groupby(["year", "Step", "AgentID_numeric"])["field_type_rn"]
        .apply(lambda x: "optimize" if (x == "optimize").any() else "rainfed")
        .reset_index()
    )

    status["irrigation_equipped"] = status["field_type_rn"].eq("optimize")
    return status


def summarize_with_percentiles(df: pd.DataFrame, group_cols, value_cols):
    """
    Summarize across bootstrap runs.

    Output columns are:
      variable_p10
      variable_p25
      variable_p50
      variable_p75
      variable_p90
    """
    rows = []

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        for col in value_cols:
            if col not in g.columns:
                continue

            vals = (
                pd.to_numeric(g[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if vals.empty:
                continue

            for suffix, q in PERCENTILES.items():
                row[f"{col}_{suffix}"] = vals.quantile(q)

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. Aquifer and crop properties
# -----------------------------------------------------------------------------
if RUN_AQUIFER_CROP:
    print("\n--- Processing aquifer and crop properties ---")
    
    system_rows = []
    
    for p_idx, policy in enumerate(POLICIES_TO_RUN, start=1):
        directory = POLICY_PATHS[policy]
        files = get_excel_files(directory)
    
        print(f"\n[{p_idx}/{len(POLICIES_TO_RUN)}] System data: {policy}")
        print(f"    Directory: {directory}")
        print(f"    Files found: {len(files)}")
    
        if not directory.exists():
            print(f"    Warning: directory not found. Skipping {policy}.")
            continue
    
        for f_idx, path in enumerate(tqdm(files, desc=f"    Reading {policy}", leave=False), start=1):
            bnum = parse_bootstrap_id(path.name)
            if bnum is None:
                continue
    
            df = safe_read_excel(path, "System")
            if df is None:
                continue
    
            if "year" not in df.columns:
                df = df.reset_index().rename(columns={"index": "year"})
    
            df = df.sort_values("year").copy()
            df["Policy"] = policy
            df["Bootstrap"] = bnum
    
            if "GW_st" in df.columns:
                df["GW_st_change"] = df["GW_st"].diff()
                mask_2002 = df["year"].eq(2002)
                df.loc[mask_2002, "GW_st_change"] = (
                    df.loc[mask_2002, "GW_st"] - INITIAL_GW_ST
                )
    
            system_rows.append(df)
    
    if system_rows:
        system_all = pd.concat(system_rows, ignore_index=True)
    
        # Keep this file focused on hydrology, crop shares, and land use.
        system_value_cols = [
            "GW_st",
            "GW_st_change",
            "withdrawal",
            "rainfed",
            "rainfed_rn",
            "corn",
            "sorghum",
            "soybeans",
            "wheat",
            "fallow",
        ]
        system_value_cols = [c for c in system_value_cols if c in system_all.columns]
    
        system_summary = summarize_with_percentiles(
            system_all,
            group_cols=["Policy", "year"],
            value_cols=system_value_cols,
        )
    
        system_summary = system_summary.rename(columns={"year": "Year"}).set_index("Year")
    
        out_path = DATA_FOR_FIGURES_DIR / "aquifer_and_crop_properties.csv"
        system_summary.to_csv(out_path)
    
        print(f"\nSaved: {out_path}")
    else:
        system_all = pd.DataFrame()
        print("\nNo system data processed.")
    
# If only rerunning economic outcomes, we still need System data for FB-II
# accounting variables in economic_outcomes.csv.
if RUN_ECONOMIC and "system_all" not in globals():
    print("\n--- Loading minimal System data for FB-II accounting ---")
    system_rows = []

    for policy in POLICIES_TO_RUN:
        directory = POLICY_PATHS[policy]
        files = get_excel_files(directory)

        for path in tqdm(files, desc=f"  System for econ {policy}", leave=False):
            bnum = parse_bootstrap_id(path.name)
            if bnum is None:
                continue

            df = safe_read_excel(path, "System")
            if df is None:
                continue

            if "year" not in df.columns:
                df = df.reset_index().rename(columns={"index": "year"})

            df["Policy"] = policy
            df["Bootstrap"] = bnum
            system_rows.append(df)

    system_all = pd.concat(system_rows, ignore_index=True) if system_rows else pd.DataFrame()

# -----------------------------------------------------------------------------
# 4. Economic outcomes
# -----------------------------------------------------------------------------
if RUN_ECONOMIC:
    print("\n--- Processing economic outcomes ---")
    
    econ_rows = []
    
    for p_idx, policy in enumerate(POLICIES_TO_RUN, start=1):
        directory = POLICY_PATHS[policy]
        files = get_excel_files(directory)
    
        print(f"\n[{p_idx}/{len(POLICIES_TO_RUN)}] Economic data: {policy}")
        print(f"    Directory: {directory}")
        print(f"    Files found: {len(files)}")
    
        if not directory.exists():
            print(f"    Warning: directory not found. Skipping {policy}.")
            continue
    
        for f_idx, path in enumerate(tqdm(files, desc=f"    Reading {policy}", leave=False), start=1):
            bnum = parse_bootstrap_id(path.name)
            if bnum is None:
                continue
    
            farmers = safe_read_excel(path, "Farmers")
            fields = safe_read_excel(path, "Fields")
    
            if farmers is None or fields is None:
                continue
    
            if "AgentID" not in farmers.columns or "AgentID" not in fields.columns:
                print(f"    Warning: AgentID missing in {path.name}. Skipping.")
                continue
    
            # Match the original economic-processing logic exactly:
            # merge Farmers with Fields and use field-level w as the applied-water denominator.
            fields["AgentID_numeric"] = (
                fields["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
            )
            farmers["AgentID_numeric"] = (
                farmers["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
            )
    
            required_field_cols = ["year", "Step", "AgentID_numeric", "field_type_rn", "w"]
            missing_field_cols = [c for c in required_field_cols if c not in fields.columns]
            if missing_field_cols:
                print(f"    Warning: missing field columns in {path.name}: {missing_field_cols}. Skipping.")
                continue
    
            merged = pd.merge(
                farmers,
                fields[required_field_cols],
                on=["year", "Step", "AgentID_numeric"],
                how="left",
            ).set_index("year")
    
            # Original definition: irrigation-equipped/optimized fields only.
            irrigators_df = merged[merged["field_type_rn"] == "optimize"].copy()
    
            if irrigators_df.empty:
                continue
    
            # Original basis: profit per field-level applied water w.
            irrigators_df["profit_per_water"] = (
                irrigators_df["profit"] / irrigators_df["w"]
            ).replace([np.inf, -np.inf], 0)
    
            yearly_avg_profit = irrigators_df.groupby("year")["profit"].mean()
            yearly_profit_per_water = irrigators_df.groupby("year")["profit_per_water"].mean()
            yearly_farmers_in_loss = irrigators_df.groupby("year").apply(
                lambda x: (x["profit"] <= 0).sum()
            )
    
            run_df = pd.DataFrame(
                {
                    "avg_profit": yearly_avg_profit,
                    "profit_per_water": yearly_profit_per_water,
                    "farmers_in_loss": yearly_farmers_in_loss,
                }
            ).reset_index()
    
            # Optional extra economic diagnostics, using the same filtered irrigator set.
            run_df_extra = pd.DataFrame(
                {
                    "year": yearly_avg_profit.index,
                    "share_farmers_in_loss": irrigators_df.groupby("year").apply(
                        lambda x: (x["profit"] <= 0).mean()
                    ).values,
                }
            )
            run_df = run_df.merge(run_df_extra, on="year", how="left")
    
            # Optional FB-II farmer-level diagnostics.
            if "cb_enrolled" in irrigators_df.columns:
                cb_df = irrigators_df.groupby("year").apply(
                    lambda x: pd.Series({
                        "cb_share_enrolled": x["cb_enrolled"].fillna(False).astype(bool).mean(),
                        "cb_mean_payout_enrolled": (
                            x.loc[
                                x["cb_enrolled"].fillna(False).astype(bool),
                                "cb_payout"
                            ].mean()
                            if "cb_payout" in x.columns
                            and x["cb_enrolled"].fillna(False).astype(bool).any()
                            else np.nan
                        ),
                    })
                ).reset_index()
    
                run_df = run_df.merge(cb_df, on="year", how="left")
    
            run_df["Policy"] = policy
            run_df["Bootstrap"] = bnum
            econ_rows.append(run_df)
    
    if econ_rows:
        econ_all = pd.concat(econ_rows, ignore_index=True)
    
        econ_value_cols = [
            "avg_profit",
            "profit_per_water",
            "farmers_in_loss",
            "share_farmers_in_loss",
            "cb_share_enrolled",
            "cb_mean_payout_enrolled",
        ]
        econ_value_cols = [c for c in econ_value_cols if c in econ_all.columns]
    
        econ_summary = summarize_with_percentiles(
            econ_all,
            group_cols=["Policy", "year"],
            value_cols=econ_value_cols,
        )
    
        # Add FB-II system-accounting variables into economic_outcomes.csv.
        # These come from the System sheet because they cannot be reconstructed
        # cleanly from the farmer-level profit-distribution files.
        cb_system_cols = [
            "cb_available_fund",
            "cb_fee_revenue_used_for_fund",
            "cb_carryover_used_for_fund",
            "cb_fee_revenue_current_year",
            "cb_unspent_balance",
            "cb_num_enrolled",
            "cb_total_payout",
        ]
    
        if not system_all.empty:
            cb_source = system_all[system_all["Policy"].eq("FB-II")].copy()
            cb_system_cols = [c for c in cb_system_cols if c in cb_source.columns]
    
            if cb_system_cols:
                cb_summary = summarize_with_percentiles(
                    cb_source,
                    group_cols=["Policy", "year"],
                    value_cols=cb_system_cols,
                )
    
                econ_summary = econ_summary.merge(
                    cb_summary,
                    on=["Policy", "year"],
                    how="left",
                )
    
        econ_summary = econ_summary.rename(columns={"year": "Year"}).set_index("Year")
    
        out_path = DATA_FOR_FIGURES_DIR / "economic_outcomes.csv"
        econ_summary.to_csv(out_path)
    
        print(f"\nSaved: {out_path}")
    else:
        print("\nNo economic data processed.")


# -----------------------------------------------------------------------------
# 5. PR-I and PR-II cutoff diagnostics
# -----------------------------------------------------------------------------
if RUN_CUTOFF:
    print("\n--- Processing PR-I and PR-II cutoff diagnostics ---")
    
    cutoff_rows = []
    
    for p_idx, policy in enumerate(CUTOFF_POLICIES, start=1):
        directory = POLICY_PATHS[policy]
        files = get_excel_files(directory)
    
        print(f"\n[{p_idx}/{len(CUTOFF_POLICIES)}] Cutoff diagnostics: {policy}")
        print(f"    Directory: {directory}")
        print(f"    Files found: {len(files)}")
    
        if not directory.exists():
            print(f"    Warning: directory not found. Skipping {policy}.")
            continue
    
        for f_idx, path in enumerate(tqdm(files, desc=f"    Reading {policy}", leave=False), start=1):
            bnum = parse_bootstrap_id(path.name)
            if bnum is None:
                continue
    
            farmers = safe_read_excel(path, "Farmers")
            fields = safe_read_excel(path, "Fields")
    
            if farmers is None or fields is None:
                continue
    
            if "zero_irrigation_reason" not in farmers.columns:
                print(f"    Warning: zero_irrigation_reason missing in {path.name}. Skipping.")
                continue
    
            farmers = add_agent_numeric_id(farmers)
            field_status = get_irrigation_equipped_status(fields)
    
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
                    "year": year,
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
    
    if cutoff_rows:
        cutoff_all = pd.DataFrame(cutoff_rows)
    
        cutoff_value_cols = [
            "n_irrigation_equipped",
            "n_cutoff",
            "n_partially_cutoff",
            "n_prior_appropriation",
            "n_policy_cutoff",
            "n_rainfed_reason",
            "n_optimization_zero",
            "n_no_zero_reason",
            "n_other_reason",
            "share_cutoff",
            "share_partially_cutoff",
            "share_prior_appropriation",
            "share_policy_cutoff",
            "share_rainfed_reason",
            "share_optimization_zero",
            "share_no_zero_reason",
            "share_other_reason",
        ]
    
        cutoff_summary = summarize_with_percentiles(
            cutoff_all,
            group_cols=["Policy", "year"],
            value_cols=cutoff_value_cols,
        )
    
        cutoff_summary = cutoff_summary.rename(columns={"year": "Year"}).set_index("Year")
    
        out_path = DATA_FOR_FIGURES_DIR / "cutoff_diagnostics.csv"
        cutoff_summary.to_csv(out_path)
    
        print(f"\nSaved: {out_path}")
    else:
        print("\nNo cutoff diagnostics processed.")


# -----------------------------------------------------------------------------
# 6. Profit distribution data, one CSV per policy-bootstrap
# -----------------------------------------------------------------------------
if RUN_PROFIT_DISTRIBUTION:
    print("\n--- Processing profit distribution data ---")
    
    def prepare_bootstrap_costs(bootstrap_folder: Path):
        """
        Loads raw bootstrap data and applies existing conversion factors.
        """
        bootstrap_pattern = bootstrap_folder / "bootstrapped_data_*.csv"
        results_dict = {}
    
        conversion_factor_1 = 50 / 0.404686 * 1e-4
        conversion_factor_2 = 50 * 1e-4
    
        files = sorted(glob(str(bootstrap_pattern)))
    
        print(f"Bootstrap cost files found: {len(files)}")
    
        for idx, path in enumerate(tqdm(files, desc="  Pre-calculating costs", leave=False), start=1):
            fname = os.path.basename(path)
            m = re.match(r"bootstrapped_data_(\d+)\.csv$", fname)
    
            if not m:
                continue
    
            bnum = int(m.group(1))
            bkey = f"b_{bnum}"
    
            df = pd.read_csv(path)
            df_sel = df[["Year", "Crop", "Variable Costs", "Fixed Costs", "Prices"]].copy()
    
            df_sel["Variable Costs"] *= conversion_factor_1
            df_sel["Fixed Costs"] *= conversion_factor_1
            df_sel["Prices"] *= conversion_factor_2
    
            results_dict[bkey] = df_sel
    
        return results_dict
    
    
    def process_profit_distribution(policy_name, directory, bootstrap_costs, out_dir):
        if not directory.exists():
            print(f"Warning: directory not found for {policy_name}: {directory}")
            return
    
        slug = policy_slug(policy_name)
        files = get_excel_files(directory)
    
        print(f"\nProfit distribution: {policy_name}")
        print(f"    Directory: {directory}")
        print(f"    Files found: {len(files)}")
    
        y_max = {
            "corn": 457.6316,
            "sorghum": 184.4876,
            "soybeans": 145.3771,
            "wheat": 130.3249,
        }
    
        completed = 0
    
        for f_idx, path in enumerate(tqdm(files, desc=f"    Processing {policy_name}", leave=False), start=1):
            bnum = parse_bootstrap_id(path.name)
    
            if bnum is None:
                continue
    
            bkey = f"b_{bnum}"
    
            farmers = safe_read_excel(path, "Farmers")
            fields = safe_read_excel(path, "Fields")
    
            if farmers is None or fields is None:
                continue
    
            farmer_cols = [
                "year",
                "Step",
                "AgentID",
                "yield_rate",
                "revenue",
                "energy_cost",
                "profit",
                "irr_depth",
                "irr_vol",
            ]
    
            optional_farmer_cols = [
                "pumping_fee",
                "cb_enrolled",
                "cb_payout",
                "cb_production_profit",
                "cb_selection_ref_profit",
                "cb_selection_ref_irr_vol",
                "cb_selection_ranking_score",
                "cb_payout_benchmark_used",
                "zero_irrigation_reason",
            ]
    
            for col in optional_farmer_cols:
                if col in farmers.columns and col not in farmer_cols:
                    farmer_cols.append(col)
    
            farmer_cols = [c for c in farmer_cols if c in farmers.columns]
            farmers_sel = farmers[farmer_cols].copy()
    
            field_cols = [
                "year",
                "Step",
                "AgentID",
                "field_type_rn",
                "w",
                "pumping rate",
                "crop",
                "field_area",
            ]
    
            field_cols = [c for c in field_cols if c in fields.columns]
            fields_sel = fields[field_cols].copy()
    
            farmers_sel = add_agent_numeric_id(farmers_sel)
            fields_sel = add_agent_numeric_id(fields_sel)
    
            merged = farmers_sel.merge(
                fields_sel,
                on=["year", "Step", "AgentID_numeric"],
                suffixes=("_farmer", "_field"),
                how="left",
            )
    
            merged = merged.drop(
                columns=[
                    c for c in ["AgentID_farmer", "AgentID_field"]
                    if c in merged.columns
                ],
                errors="ignore",
            ).rename(columns={"AgentID_numeric": "AgentID"})
    
            if "crop" in merged.columns:
                merged["crop"] = merged["crop"].astype(str).str.lower()
                merged["Y_max"] = merged["crop"].map(y_max)
                merged["Yield"] = merged["yield_rate"] * merged["Y_max"]
                merged.drop(columns=["Y_max"], inplace=True)
    
            costs_df = bootstrap_costs.get(bkey)
    
            if costs_df is not None and "crop" in merged.columns:
                merged = (
                    merged.merge(
                        costs_df,
                        left_on=["year", "crop"],
                        right_on=["Year", "Crop"],
                        how="left",
                    )
                    .drop(columns=["Year", "Crop"], errors="ignore")
                )
    
            merged["Policy"] = policy_name
            merged["Bootstrap"] = bnum
    
            out_path = out_dir / f"profit_distribution_{slug}_b_{bnum:03d}.csv"
            merged.to_csv(out_path, index=False)
    
            completed += 1
    
            if completed % 25 == 0 or completed == len(files):
                print(f"    {policy_name}: {completed}/{len(files)} profit files completed")
    
        print(f"    Completed {completed}/{len(files)} profit distribution files for {policy_name}")
    
    
    bootstrap_costs_data = prepare_bootstrap_costs(BOOTSTRAP_DATA_DIR)
    
    for p_idx, policy in enumerate(POLICIES_TO_RUN, start=1):
        print(f"\n[{p_idx}/{len(POLICIES_TO_RUN)}] Profit distribution policy: {policy}")
        process_profit_distribution(
            policy,
            POLICY_PATHS[policy],
            bootstrap_costs_data,
            DATA_FOR_FIGURES_DIR,
        )


print("\n--------------------------------------------------")
print("All data preparation for figures is complete.")
print("--------------------------------------------------")
print(f"Output directory: {DATA_FOR_FIGURES_DIR}")
print("Created/updated:")
print("  - aquifer_and_crop_properties.csv")
print("  - economic_outcomes.csv")
print("  - cutoff_diagnostics.csv")
print("  - profit_distribution_<policy>_b_<id>.csv")
print("--------------------------------------------------")