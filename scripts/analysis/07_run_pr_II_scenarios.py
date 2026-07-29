# -----------------------------------------------------------------------------
# 07_run_pr_II_scenarios.py
#
# This script runs the Priority-Based Pumping II (PR-II) policy scenarios.
# For each bootstrap realization, it iteratively searches for the optimal
# number of senior farmers allowed to irrigate to meet the target aquifer
# withdrawal level. This requires a specialized model class (GCPModelPr2).
#
# Steps:
#   1. Sets up file paths and imports the necessary (future) model class.
#   2. Configures the range of scenarios to run.
#   3. Defines the core `run_pr2_scenario` function which uses a custom
#      bisection/secant method to find the optimal number of senior farmers.
#   4. Once converged, it saves the final detailed results to an Excel file.
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
import copy

# --- 2. Set Up File Paths and Imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

PACKAGE_PATH = PROJECT_ROOT / "scripts" / "py_champ_package"
sys.path.insert(0, str(PACKAGE_PATH))

# This script requires a new, specialized model class for this policy
from py_champ.models.gcp_model_pr2 import GCPModelPr2

INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pr2_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading model inputs from: {INPUT_DIR}")
print(f"Saving PR-II model outputs to: {OUTPUT_DIR}")


# --- 3. User Configuration ---
START_SCENARIO = 1
END_SCENARIO = 500
TARGET_WITHDRAWAL = 1900


# --- 4. Define the Core Model Execution Function ---
def run_pr2_scenario(scenario_name, scenario_path, output_dir):
    """
    Finds the optimal number of senior farmers for a single scenario by
    iteratively running simulations.
    """
    m = None
    try:
        # --- 4a. Define the Objective Function for the Search ---
        def withdrawal_difference(senior_farmers_number, *args):
            # The model needs an integer number of farmers
            senior_farmers_number = int(senior_farmers_number)
            
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = copy.deepcopy(args)
            
            # ensure water right is OFF for FB
            for bid in behaviors_dict:
                behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False

            model = GCPModelPr2(
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
            show_step=True,
            # Senior farmers getting water
            senior_farmers_number=senior_farmers_number,
            )
            for _ in range(21):
                model.step()
            model.end()

            _, _, _, df_aquifers = GCPModelPr2.get_dfs(model)
            current_withdrawal = df_aquifers.loc[2002:2022, 'withdrawal'].mean()
            del model, df_aquifers
            gc.collect()
            return current_withdrawal - TARGET_WITHDRAWAL

        # --- 4b. Load Base Scenario Data ---
        with open(scenario_path, "rb") as f:
            model_inputs = dill.load(f)
            
        # Turn OFF the water right for the priority-based scenario
        behaviors_dict = model_inputs[4]
        for bid in behaviors_dict:
            behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False

        # --- 4c. Find Optimal Number of Farmers ---
        n_farmers_min, n_farmers_max = 50, 250 # Min/max possible irrigators
        withdrawal_tolerance = 0.005 * TARGET_WITHDRAWAL
        cutoff_tolerance = 1 # We are looking for an integer
        max_iterations = 50

        solution = custom_bisect(
            withdrawal_difference, n_farmers_min, n_farmers_max,
            withdrawal_tolerance=withdrawal_tolerance,
            cutoff_tolerance=cutoff_tolerance,
            max_iterations=max_iterations, f_args=model_inputs
        )
        
        if solution is None:
             print(f"  FAILURE: Search failed for scenario {scenario_name}.")
             return

        final_senior_farmers_number = int(round(solution))
        print(f"  SUCCESS: Converged at {final_senior_farmers_number} senior farmers.")

        # --- 4d. Run Final Simulation and Save ---
        with open(scenario_path, "rb") as f:
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = dill.load(f)
        
        for bid in behaviors_dict:
            behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False

        m = GCPModelPr2(
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
            show_step=True,
            # Senior farmers getting water
            senior_farmers_number=final_senior_farmers_number,
            )
        
        for _ in range(21):
            m.step()
        m.end()

        df_farmers, df_fields, df_wells, df_aquifers = GCPModelPr2.get_dfs(m)
        df_sys = GCPModelPr2.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

        output_excel_path = output_dir / f'pr2_{scenario_name}_sf_{final_senior_farmers_number}.xlsx'
        with pd.ExcelWriter(output_excel_path) as writer:
            df_farmers.to_excel(writer, sheet_name='Farmers', index=True)
            df_fields.to_excel(writer, sheet_name='Fields', index=True)
            df_wells.to_excel(writer, sheet_name='Wells', index=True)
            df_aquifers.to_excel(writer, sheet_name='Aquifers', index=True)
            df_sys.to_excel(writer, sheet_name='System', index=True)

    except Exception as e:
        print(f"--- ERROR in scenario {scenario_name}: {e} ---")
    finally:
        if m is not None:
            del m
        gc.collect()

def custom_bisect(f, a, b, withdrawal_tolerance, cutoff_tolerance, max_iterations, f_args):
    """A custom secant/bisection root-finding method for integer values."""
    fa = f(a, *f_args)
    fb = f(b, *f_args)
    if np.sign(fa) == np.sign(fb):
        print("Warning: Root may not be bracketed. Search may fail.")

    prev_c = None
    for iteration in range(1, max_iterations + 1):
        if fb - fa == 0:
            c = int((a + b) / 2.0)
        else:
            c = b - fb * (b - a) / (fb - fa)
        
        c = int(round(c)) # Ensure c is an integer
        if not (a <= c <= b):
             c = int((a + b) / 2.0)

        if c == prev_c:
            print(f"  Converged to the same integer value of c={c}.")
            return c
        
        prev_c = c
        fc = f(c, *f_args)
        print(f"  Iteration {iteration}: Senior Farmers={c}, Withdrawal Diff={fc:.2f}")

        if abs(fc) <= withdrawal_tolerance:
            return c
        if abs(b - a) <= cutoff_tolerance: # Check if interval is 1 or 0
            return c
        
        if np.sign(fa) != np.sign(fc):
            b, fb = c, fc
        else:
            a, fa = c, fc
            
    print("Maximum iterations reached.")
    return None

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run priority-based II scenario(s)")
    parser.add_argument("--id", type=int, help="If provided, run exactly this bootstrap id (e.g., 1..500)")
    args = parser.parse_args()
    if args.id is not None:
        START_SCENARIO = END_SCENARIO = int(args.id)
        
    print(f"\nStarting Priority Pumping II model runs for scenarios {START_SCENARIO} through {END_SCENARIO}...")

    for i in tqdm(range(START_SCENARIO, END_SCENARIO + 1), desc="Overall Progress"):
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"

        if scenario_path.exists():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            output_exists = any(f.startswith(f"pr2_{scenario_name}_sf_") for f in os.listdir(OUTPUT_DIR))
            if output_exists:
                print(f"Result for {scenario_name} already exists. Skipping.")
                continue
            run_pr2_scenario(scenario_name, scenario_path, OUTPUT_DIR)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")

    print("\nAll specified scenarios have been processed.")