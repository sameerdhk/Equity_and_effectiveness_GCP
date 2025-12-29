# -----------------------------------------------------------------------------
# 05_run_fb_scenarios.py
#
# This script runs the Fee-Based (FB) policy scenarios. For each bootstrap
# realization, it iteratively searches for the optimal pumping fee ($/ac-ft)
# that results in the target aquifer withdrawal level.
#
# Steps:
#   1. Sets up file paths and imports the necessary model class.
#   2. Configures the range of scenarios to run.
#   3. Defines a `withdrawal_difference` helper function that runs a full
#      simulation for a given fee and returns the deviation from the target.
#   4. Defines the `run_fb_scenario` function which contains the core logic:
#      a. It turns OFF the volumetric water right for the scenario.
#      b. It uses a bisection/secant search method to find the optimal fee by
#         repeatedly calling the `withdrawal_difference` function.
#      c. Once the optimal fee is found, it runs the simulation one final time
#         to save the detailed Excel results.
#   5. The main execution block iterates through scenarios and calls the
#      main function.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import sys
import gc
import dill
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import copy

# --- 2. Set Up File Paths and Imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

PACKAGE_PATH = PROJECT_ROOT / "scripts" / "parent_package"
sys.path.insert(0, str(PACKAGE_PATH))

from py_champ.models.gcp_model_fb import GCPModelFb

INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fb_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading model inputs from: {INPUT_DIR}")
print(f"Saving model outputs to: {OUTPUT_DIR}")


# --- 3. User Configuration ---
START_SCENARIO = 1
END_SCENARIO = 500
ACRE_FOOT_TO_M_HA = 0.123348 # Conversion factor from acre-foot to meter-hectare
TARGET_WITHDRAWAL = 1900


# --- 4. Define the Core Model Execution Function ---
def run_fb_scenario(scenario_name, scenario_path, output_dir):
    """
    Finds the optimal pumping fee for a single scenario by iteratively
    running simulations.
    """
    m = None
    try:
        # --- 4a. Define the Objective Function for the Search ---
        # This nested function runs a full simulation for a given fee
        def withdrawal_difference(fee, *args):
            # Unpack the static arguments
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = copy.deepcopy(args)
            
            # ensure water right is OFF for FB
            for bid in behaviors_dict:
                behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False

            # Apply the current fee to the shared config
            # Fee is in $/ac-ft, model expects in $1e4/m-ha
            shared_config["finance"]["pumping_fee"] = fee * 1 / (ACRE_FOOT_TO_M_HA * 1e4)

            # Initialize and run the model
            model = GCPModelFb(
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
                model.step()
            model.end()

            # Calculate and return the difference from the target
            _, _, _, df_aquifers = GCPModelFb.get_dfs(model)
            current_withdrawal = df_aquifers.loc[2002:2022, 'withdrawal'].mean()
            del model, df_aquifers
            gc.collect()
            return current_withdrawal - TARGET_WITHDRAWAL

        # --- 4b. Load Base Scenario Data ---
        with open(scenario_path, "rb") as f:
            model_inputs = dill.load(f)
        
        # Turn OFF the water right for the fee-based scenario
        behaviors_dict = model_inputs[4]
        for bid in behaviors_dict:
            behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False
        
        # --- 4c. Find Optimal Fee using Secant/Bisection Method ---
        fee_min, fee_max = 100.0, 500.0  # Initial bounds for the fee search
        tolerance = 0.005 * TARGET_WITHDRAWAL
        
        # This search method finds the fee where withdrawal_difference(fee) is close to zero
        optimal_fee = custom_bisect(
            withdrawal_difference, fee_min, fee_max, 
            withdrawal_tolerance=tolerance, max_iterations=50, f_args=model_inputs
        )
        final_pumping_fee = round(optimal_fee, 2)
        print(f"  SUCCESS: Converged at pumping fee: ${final_pumping_fee}/ac-ft.")
        
        # --- 4d. Run Final Simulation and Save ---
        # Reload original data and apply the final optimal fee
        with open(scenario_path, "rb") as f:
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = dill.load(f)
        
        for bid in behaviors_dict:
            behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False
        shared_config["finance"]["pumping_fee"] = final_pumping_fee * 1 / (ACRE_FOOT_TO_M_HA * 1e4)

        m = GCPModelFb(
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

        df_farmers, df_fields, df_wells, df_aquifers = GCPModelFb.get_dfs(m)
        df_sys = GCPModelFb.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

        output_excel_path = output_dir / f'fb_{scenario_name}_pf_{final_pumping_fee}.xlsx'
        with pd.ExcelWriter(output_excel_path) as writer:
            df_farmers.to_excel(writer, sheet_name='Farmers', index=True)
            df_fields.to_excel(writer, sheet_name='Fields', index=True)
            df_wells.to_excel(writer, sheet_name='Wells', index=True)
            df_aquifers.to_excel(writer, sheet_name='Aquifers', index=True)
            df_sys.to_excel(writer, sheet_name='System', index=True)

        return scenario_name, final_pumping_fee

    except Exception as e:
        print(f"--- ERROR in scenario {scenario_name}: {e} ---")
        return scenario_name, None
    finally:
        if m is not None:
            del m
        gc.collect()


def custom_bisect(f, a, b, withdrawal_tolerance, max_iterations, f_args):
    """A custom secant/bisection root-finding method."""
    fa = f(a, *f_args)
    fb = f(b, *f_args)
    if np.sign(fa) == np.sign(fb):
        print("Warning: Root may not be bracketed. Search may fail.")

    for iteration in range(1, max_iterations + 1):
        if fb - fa == 0:
            c = (a + b) / 2.0
        else:
            c = b - fb * (b - a) / (fb - fa)
        
        # Ensure c stays within bounds
        if not (a < c < b):
             c = (a + b) / 2.0

        fc = f(c, *f_args)
        print(f"  Iteration {iteration}: Fee=${c:.2f}, Withdrawal Diff={fc:.2f}")

        if abs(fc) <= withdrawal_tolerance:
            return c
        
        if np.sign(fa) != np.sign(fc):
            b, fb = c, fc
        else:
            a, fa = c, fc
            
    print("Maximum iterations reached.")
    return (a + b) / 2.0

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fee-based scenario(s)")
    parser.add_argument("--id", type=int, help="If provided, run exactly this bootstrap id (e.g., 1..500)")
    args = parser.parse_args()
    if args.id is not None:
        START_SCENARIO = END_SCENARIO = int(args.id)
    
    print(f"\nStarting Fee-Based model runs for scenarios {START_SCENARIO} through {END_SCENARIO}...")
    for i in tqdm(range(START_SCENARIO, END_SCENARIO + 1), desc="Overall Progress"):
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"
        
        if scenario_path.exists():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            output_exists = any(f.startswith(f"fb_{scenario_name}_pf_") for f in os.listdir(OUTPUT_DIR))
            if output_exists:
                print(f"Result for {scenario_name} already exists. Skipping.")
                continue
            run_fb_scenario(scenario_name, scenario_path, OUTPUT_DIR)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")

    print("\nAll specified scenarios have been processed.")