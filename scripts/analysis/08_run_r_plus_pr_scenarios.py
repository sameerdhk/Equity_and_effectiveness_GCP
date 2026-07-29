# -----------------------------------------------------------------------------
# 08_run_r_plus_pr_scenarios.py
#
# This script runs the final policy scenario: a hybrid of Priority-Based Rights
# and a Uniform Pumping Restriction (R + PR).
#
# This is NOT an iterative search. Instead, it performs a pre-calculation to
# assign a unique annual water limit to each farmer based on their seniority.
# The total water allocated across all farmers is equal to the sustainable
# limit determined in the UR scenarios.
#
# Steps:
#   1. Pre-calculates a "priority factor" and "shares" for each farmer.
#   2. Loads the results from the UR scenario runs to get the total sustainable
#      water limit for each bootstrap scenario.
#   3. Combines these to calculate a unique water limit for each farmer for
#      each bootstrap scenario.
#   4. For each scenario, it loads the universal input file, modifies the
#      'wr_depth' for each farmer, and then runs the simulation once.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import sys
import gc
import dill
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# --- 2. Set Up File Paths and Imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

PACKAGE_PATH = PROJECT_ROOT / "scripts" / "py_champ_package"
sys.path.insert(0, str(PACKAGE_PATH))

# This policy uses the standard UR model, as the policy logic is applied
# by modifying the inputs before the model is called.
from py_champ.models.gcp_model_ur import GCPModelUr

INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
UR_RESULTS_PATH = PROJECT_ROOT / "outputs" / "ur_runs" / "ur_optimal_water_limits.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "r_plus_pr_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading model inputs from: {INPUT_DIR}")
print(f"Loading UR water limits from: {UR_RESULTS_PATH}")
print(f"Saving R+PR model outputs to: {OUTPUT_DIR}")


# --- 3. User Configuration ---
START_SCENARIO = 1
END_SCENARIO = 500


# --- 4. Pre-calculation and Core Functions ---
def calculate_individual_water_limits(ur_results_path):
    """
    Calculates the individual water limit for each farmer for each bootstrap scenario.
    """
    num_farmers = 254 # Number of irrigating farmers
    pf_values = np.linspace(1.0, 0.7, num_farmers)
    pf_dict = {}
    total_shares = 0
    for i in range(num_farmers):
        seniority_id = i + 1
        priority_factor = round(pf_values[i], 4)
        unrestricted_num_shares = round(24 * priority_factor, 4)
        pf_dict[seniority_id] = {"unrestricted_num_shares": unrestricted_num_shares}
        total_shares += unrestricted_num_shares

    # Load the total water limits from the UR scenario results
    water_limits_df = pd.read_csv(ur_results_path)
    water_limits_df["total_water_limit_inches"] = water_limits_df["Final_Water_Limit_Inches"] * num_farmers
    water_limits_df["inches_per_share"] = water_limits_df["total_water_limit_inches"] / total_shares

    # Create the final nested dictionary: {scenario_name: {seniority_id: limit}}
    nested_water_limits = {}
    for _, row in water_limits_df.iterrows():
        scenario_name = row["Scenario"]
        inches_per_share = row["inches_per_share"]
        farmer_limits = {}
        for seniority_id, vals in pf_dict.items():
            water_limit = round(vals["unrestricted_num_shares"] * inches_per_share, 4)
            farmer_limits[seniority_id] = water_limit
        nested_water_limits[scenario_name] = farmer_limits
        
    return nested_water_limits

def run_r_plus_pr_scenario(scenario_name, scenario_path, output_dir, individual_limits):
    """
    Loads inputs, applies individual water rights, runs the simulation, and saves results.
    """
    m = None
    try:
        with open(scenario_path, "rb") as f:
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = dill.load(f)

        # --- CRITICAL: Apply individual water rights to each farmer ---
        scenario_limits = individual_limits.get(scenario_name)
        if not scenario_limits:
            print(f"  Warning: No water limit data found for {scenario_name}. Skipping.")
            return

        in2cm = 2.54
        for bid, b_dict in behaviors_dict.items():
            seniority_id = int(b_dict['seniority_id'])
            # Rainfed farmers (seniority > 999) get no water right
            if seniority_id < 1000:
                water_limit_inches = scenario_limits.get(seniority_id, 0)
                b_dict["water_rights"]["wr_yr"]["wr_depth"] = water_limit_inches * in2cm

        # --- Initialize and Run the Model ---
        m = GCPModelUr(
        # General model settings
        crop_options=["corn", "sorghum", "soybeans", "wheat", "fallow"],
        tech_options=["center pivot LEPA"],
        area_split=1,
        init_year=2001,
        end_year=2022,
        seed=3,
        # Dictionaries from input file
        aquifers_dict=aquifers_dict,
        fields_dict=fields_dict,
        wells_dict=wells_dict,
        finances_dict=finances_dict,
        behaviors_dict=behaviors_dict,
        shared_config=shared_config,
        # Step-wise data from input file
        prec_aw_step=prec_aw_step,
        prec_aw_rolling_step=prec_aw_rolling_step,
        rolling_precipitaion_average=True,
        crop_price_step=crop_price_step,
        crop_price_avg_step=crop_price_avg_step,
        crop_variable_cost_step=crop_variable_cost_step,
        crop_variable_cost_avg_step=crop_variable_cost_avg_step,
        crop_fixed_cost_step=crop_fixed_cost_step,
        crop_fixed_cost_avg_step=crop_fixed_cost_avg_step,
        # Other options
        show_step=True
        )

        for _ in range(21):
            m.step()
        m.end()

        # --- Collect and Save Results ---
        df_farmers, df_fields, df_wells, df_aquifers = GCPModelUr.get_dfs(m)
        df_sys = GCPModelUr.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

        output_excel_path = output_dir / f'r_plus_pr_{scenario_name}.xlsx'
        with pd.ExcelWriter(output_excel_path) as writer:
            df_farmers.to_excel(writer, sheet_name='Farmers', index=True)
            df_fields.to_excel(writer, sheet_name='Fields', index=True)
            df_wells.to_excel(writer, sheet_name='Wells', index=True)
            df_aquifers.to_excel(writer, sheet_name='Aquifers', index=True)
            df_sys.to_excel(writer, sheet_name='System', index=True)
        
        print(f"  Scenario {scenario_name} completed successfully.")

    except Exception as e:
        print(f"--- ERROR in scenario {scenario_name}: {e} ---")
    finally:
        if m is not None:
            del m
        gc.collect()


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Pre-calculate all individual water limits before starting the simulations
    print("Pre-calculating individual water limits for all scenarios...")
    individual_water_limits = calculate_individual_water_limits(UR_RESULTS_PATH)
    print("...calculations complete.")
    
    parser = argparse.ArgumentParser(description="Run priority-based II scenario(s)")
    parser.add_argument("--id", type=int, help="If provided, run exactly this bootstrap id (e.g., 1..500)")
    args = parser.parse_args()
    if args.id is not None:
        START_SCENARIO = END_SCENARIO = int(args.id)

    print(f"\nStarting R + PR model runs for scenarios {START_SCENARIO} through {END_SCENARIO}...")
    for i in tqdm(range(START_SCENARIO, END_SCENARIO + 1), desc="Overall Progress"):
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"

        if scenario_path.exists():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            output_exists = any(f.startswith(f"r_plus_pr_{scenario_name}") for f in os.listdir(OUTPUT_DIR))
            if output_exists:
                print(f"Result for {scenario_name} already exists. Skipping.")
                continue
            run_r_plus_pr_scenario(scenario_name, scenario_path, OUTPUT_DIR, individual_water_limits)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")

    print("\nAll specified scenarios have been processed.")