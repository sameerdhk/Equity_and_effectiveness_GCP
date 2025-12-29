# -----------------------------------------------------------------------------
# 01_prepare_data_for_figures.py
#
# This script consolidates results from all policy scenarios into clean CSV
# files suitable for generating figures. It creates:
#   1. aquifer_and_crop_properties.csv: median timeseries for hydrologic
#      and land-use variables.
#   2. economic_outcomes.csv: median timeseries for key economic indicators.
#   3. profit_distribution_<policy>_b_<id>.csv: farmer-level profit data for
#      every bootstrap run under every policy, used for distribution plots
#      (Lorenz curves, etc.). One CSV per (policy, bootstrap) to keep file
#      sizes manageable.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from glob import glob

# --- 2. Set Up File Paths ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INPUTS_DIR = PROJECT_ROOT / "inputs"
DATA_FOR_FIGURES_DIR = OUTPUTS_DIR / "data_for_figures"
os.makedirs(DATA_FOR_FIGURES_DIR, exist_ok=True)

POLICY_PATHS = {
    "BAU":   OUTPUTS_DIR / "baseline_runs",
    "UR":    OUTPUTS_DIR / "ur_runs",
    "FB":    OUTPUTS_DIR / "fb_runs",
    "PR-I":  OUTPUTS_DIR / "pr1_runs",
    "PR-II": OUTPUTS_DIR / "pr2_runs",
    "R+PR":  OUTPUTS_DIR / "r_plus_pr_runs",
}
BOOTSTRAP_DATA_DIR = INPUTS_DIR / "bootstrap_samples"

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Outputs dir: {OUTPUTS_DIR}")
print(f"Bootstrap data dir: {BOOTSTRAP_DATA_DIR}")

# --- 3. USER CONFIGURATION: Select which policies to process ---
POLICIES_TO_RUN = [
    "BAU",
    "UR",
    "FB",
    "PR-I",
    "PR-II",
    "R+PR",
]
print(f"\nProcessing the following policies: {', '.join(POLICIES_TO_RUN)}")

# --------------------------------------------------------------------
# 4. Aquifer and Crop Data Processing
# --------------------------------------------------------------------
print("\n--- Processing Aquifer and Crop Data ---")

def load_system_data(directory, sheet_name="System", start_year=2002):
    case_data = {}
    if not directory.exists():
        print(f"Warning: Directory not found, skipping: {directory}")
        return case_data
    for file in tqdm(os.listdir(directory), desc="  Loading files", leave=False):
        if file.endswith(".xlsx"):
            path = os.path.join(directory, file)
            df = pd.read_excel(path, sheet_name=sheet_name, index_col="year")
            if start_year in df.index:
                # Initial GW storage used for baseline difference
                initial_st = 24.203292398301375
                df["GW_st_change"] = df["GW_st"].diff().fillna(
                    df.loc[start_year, "GW_st"] - initial_st
                )
            case_data[file] = df
    return case_data

all_policy_medians_system = []
for policy_name in tqdm(POLICIES_TO_RUN, desc="Processing Policies (System)"):
    directory_path = POLICY_PATHS[policy_name]
    cases = load_system_data(directory_path)
    if cases:
        combined_df = pd.concat(cases.values())
        median_df = combined_df.groupby(combined_df.index).median()
        median_df["Policy"] = policy_name
        all_policy_medians_system.append(median_df)

if all_policy_medians_system:
    final_system_df = pd.concat(all_policy_medians_system)
    final_system_df.index.name = "Year"
    save_path_system = DATA_FOR_FIGURES_DIR / "aquifer_and_crop_properties.csv"
    final_system_df.to_csv(save_path_system)
    print(f"\nAquifer and crop data saved to: {save_path_system}")
else:
    print("\nNo aquifer/crop data processed; System CSV not created.")

# --------------------------------------------------------------------
# 5. Economic Data Processing
# --------------------------------------------------------------------
print("\n--- Processing Economic Data ---")

def load_economic_data(directory):
    """
    Loads and merges Farmers and Fields sheets from all Excel files in a directory.
    """
    case_data = {}
    if not directory.exists():
        print(f"Warning: Directory not found, skipping: {directory}")
        return case_data
    
    for file in tqdm(os.listdir(directory), desc="  Loading files", leave=False):
        if file.endswith(".xlsx"):
            path = os.path.join(directory, file)
            df_farmers = pd.read_excel(path, sheet_name="Farmers")
            df_fields = pd.read_excel(path, sheet_name="Fields")
            
            df_fields["AgentID_numeric"] = df_fields["AgentID"].str.extract(r"(\d+)$").astype(int)
            df_farmers["AgentID_numeric"] = df_farmers["AgentID"].str.extract(r"(\d+)$").astype(int)
            
            df_merged = (
                pd.merge(
                    df_farmers,
                    df_fields[["year", "Step", "AgentID_numeric", "field_type_rn", "w"]],
                    on=["year", "Step", "AgentID_numeric"],
                    how="left",
                )
                .set_index("year")
            )
            case_data[file] = df_merged
    return case_data

all_policy_medians_econ = []
for policy_name in tqdm(POLICIES_TO_RUN, desc="Processing Policies (Econ)"):
    directory_path = POLICY_PATHS[policy_name]
    cases = load_economic_data(directory_path)
    if not cases:
        continue

    all_runs_metrics = []
    for case_name, df in cases.items():
        irrigators_df = df[df["field_type_rn"] == "optimize"].copy()
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
        )
        all_runs_metrics.append(run_df)

    if all_runs_metrics:
        combined_runs_df = pd.concat(all_runs_metrics)
        median_df = combined_runs_df.groupby(combined_runs_df.index).median()
        median_df["Policy"] = policy_name
        all_policy_medians_econ.append(median_df)

if all_policy_medians_econ:
    final_econ_df = pd.concat(all_policy_medians_econ)
    final_econ_df.index.name = "Year"
    save_path_econ = DATA_FOR_FIGURES_DIR / "economic_outcomes.csv"
    final_econ_df.to_csv(save_path_econ)
    print(f"\nEconomic data saved to: {save_path_econ}")
else:
    print("\nNo economic data processed; Economic CSV not created.")

# --------------------------------------------------------------------
# 6. Profit Distribution Data (one CSV per policy *and* bootstrap)
# --------------------------------------------------------------------
print("\n--- Processing Profit Distribution Data ---")

def prepare_bootstrap_costs(bootstrap_folder: Path):
    """
    Loads all raw bootstrap data, applies conversion factors to economic
    variables, and returns a dictionary of processed DataFrames keyed by 'b_X'.
    """
    bootstrap_pattern = bootstrap_folder / "bootstrapped_data_*.csv"
    results_dict = {}
    
    conversion_factor_1 = 50 / 0.404686 * 1e-4
    conversion_factor_2 = 50 * 1e-4

    for path in tqdm(sorted(glob(str(bootstrap_pattern))), desc="  Pre-calculating costs", leave=False):
        fname = os.path.basename(path)
        m = re.match(r"bootstrapped_data_(\d+)\.csv$", fname)
        if not m:
            continue
        num = int(m.group(1))
        key = f"b_{num}"  # Use 'b_X' format to match Excel file parsing

        df = pd.read_csv(path)
        df_sel = df[["Year", "Crop", "Variable Costs", "Fixed Costs", "Prices"]].copy()
        df_sel["Variable Costs"] *= conversion_factor_1
        df_sel["Fixed Costs"]    *= conversion_factor_1
        df_sel["Prices"]         *= conversion_factor_2
        results_dict[key] = df_sel
    return results_dict

def parse_bootstrap_from_excel_name(fname: str):
    """Returns ('b_1', 1) or (None, None) if no match."""
    m = re.search(r"_b_(\d+)", fname)
    if not m:
        return None, None
    num = int(m.group(1))
    return f"b_{num}", num

def process_profit_distribution(policy_name: str,
                                directory: Path,
                                bootstrap_costs: dict,
                                out_dir: Path):
    """
    For a given policy, loops over all Excel result files, joins with bootstrap
    cost data, and writes ONE CSV PER BOOTSTRAP:
        profit_distribution_<policy_slug>_b_<id>.csv

    This avoids creating one enormous file per policy.
    """
    if not directory.exists():
        print(f"Warning: Directory not found, skipping: {directory}")
        return

    # Safe stub for filenames (Policy column still keeps original name)
    policy_slug = re.sub(r"[^A-Za-z0-9]+", "_", policy_name)

    # Crop-specific max yields (same as in your local script)
    y_max = {
        "corn":     457.6316,
        "sorghum":  184.4876,
        "soybeans": 145.3771,
        "wheat":    130.3249,
    }

    files = [f for f in os.listdir(directory) if re.search(r"\.xlsx?$", f, re.I)]
    if not files:
        print(f"  No Excel files found for policy {policy_name} in {directory}")
        return

    for fname in tqdm(files, desc=f"  Processing {policy_name}", leave=False):
        try:
            bkey, bnum = parse_bootstrap_from_excel_name(fname)
            if bkey is None:
                continue

            in_path = directory / fname
            df_farm = pd.read_excel(in_path, sheet_name="Farmers")
            df_fld  = pd.read_excel(in_path, sheet_name="Fields")

            # Base columns from Farmers
            farmers_cols = [
                "year", "Step", "AgentID", "yield_rate",
                "revenue", "energy_cost", "profit", "irr_depth",
            ]
            if policy_name == "FB":
                farmers_cols.append("pumping_fee")
            farmers_sel = df_farm[farmers_cols].copy()

            fields_sel = df_fld[
                ["year", "Step", "AgentID", "field_type_rn", "w", "pumping rate", "crop"]
            ].copy()

            farmers_sel["AgentID_numeric"] = (
                farmers_sel["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
            )
            fields_sel["AgentID_numeric"] = (
                fields_sel["AgentID"].astype(str).str.extract(r"(\d+)$").astype(int)
            )

            merged = (
                pd.merge(
                    farmers_sel,
                    fields_sel,
                    on=["year", "Step", "AgentID_numeric"],
                    suffixes=("_farmer", "_field"),
                )
                .drop(columns=["AgentID_farmer", "AgentID_field"])
                .rename(columns={"AgentID_numeric": "AgentID"})
            )

            merged["crop"] = merged["crop"].str.lower()
            merged["Y_max"] = merged["crop"].map(y_max)
            merged["Yield"] = merged["yield_rate"] * merged["Y_max"]
            merged.drop(columns=["Y_max"], inplace=True)

            costs_df = bootstrap_costs.get(bkey)
            if costs_df is None:
                print(f"    - No bootstrap costs found for {bkey}, skipping {fname}")
                continue

            df_out = pd.merge(
                merged,
                costs_df,
                left_on=["year", "crop"],
                right_on=["Year", "Crop"],
                how="left",
            ).drop(columns=["Year", "Crop"])

            df_out["Policy"] = policy_name
            df_out["Bootstrap"] = bnum

            # One CSV per (policy, bootstrap)
            out_path = out_dir / f"profit_distribution_{policy_slug}_b_{bnum:03d}.csv"
            df_out.to_csv(out_path, index=False)
            print(f"    Saved {policy_name} bootstrap {bnum} → {out_path.name} (rows={len(df_out)})")

        except Exception as e:
            print(f"    - Error on {fname}: {e}")

# --- Main execution block for profit distribution ---
bootstrap_costs_data = prepare_bootstrap_costs(BOOTSTRAP_DATA_DIR)

for policy in tqdm(POLICIES_TO_RUN, desc="Processing Policies (Profit Dist)"):
    policy_path = POLICY_PATHS[policy]
    process_profit_distribution(policy, policy_path, bootstrap_costs_data, DATA_FOR_FIGURES_DIR)

print("\n--------------------------------------------------")
print("All data preparation for figures is complete.")
print("--------------------------------------------------")