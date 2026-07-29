# -----------------------------------------------------------------------------
# 04_run_ur_scenarios.py
#
# This script runs the Uniform Pumping Restrctions (UR) policy scenarios. For each bootstrap
# realization, it iteratively searches for the optimal water withdrawal limit (inches))
# that results in the target aquifer withdrawal level.
#
# Steps:
#   1. Sets up file paths and imports the necessary model class.
#   2. Configures the range of scenarios to run.
#   3. Defines a `withdrawal_difference` helper function that runs a full
#      simulation for a given fee and returns the deviation from the target.
#   4. Defines the `run_ur_scenario` function which contains the core logic:
#      a. It uses a bisection/secant search method to find the optimal fee by
#         repeatedly calling the `withdrawal_difference` function.
#      b. Once the optimal withdrawal limit is found, it runs the simulation one final time
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
    # Fallback for interactive environments like Spyder
    print("Could not set PROJECT_ROOT automatically. Using current working directory.")
    PROJECT_ROOT = Path(os.getcwd())

PACKAGE_PATH = PROJECT_ROOT / "scripts" / "py_champ_package"
sys.path.insert(0, str(PACKAGE_PATH))

from py_champ.models.gcp_model_ur import GCPModelUr

# Use the new input directory for shared baseline and UR inputs
INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ur_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading model inputs from: {INPUT_DIR}")
print(f"Saving model outputs to: {OUTPUT_DIR}")


# --- 3. User Configuration ---
# Set the range of bootstrap scenarios you want to run.
# For example, to run scenarios 1 through 10, set START_SCENARIO = 1 and END_SCENARIO = 10.
# To run only scenario 5, set both to 5.
START_SCENARIO = 1
END_SCENARIO = 500 # Set to the total number of scenarios you have (e.g., 100)
TARGET_WITHDRAWAL = 1900

# --- 4. Define the Core Model Execution Function ---
def run_ur_scenario(scenario_name, scenario_path, output_dir):
    """
    Finds the optimal water limit for a single scenario by iteratively
    running simulations.
    """
    m = None
    try:
        # --- 4a. Define the Objective Function for the Search ---
        def withdrawal_difference(water_limit, *args):
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = copy.deepcopy(args)        

            # This is the core of the UR policy scenario
            for bid in behaviors_dict:
                # The model expects water rights in cm, so convert from inches
                behaviors_dict[bid]["water_rights"]["wr_yr"]["wr_depth"] = water_limit * 2.54
              
            # --- Initialize and Run the Model ---
            model = GCPModelUr(
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

            _, _, _, df_aquifers = GCPModelUr.get_dfs(model)
            current_withdrawal = df_aquifers.loc[2002:2022, 'withdrawal'].mean()
            del model, df_aquifers
            gc.collect()
            return current_withdrawal - TARGET_WITHDRAWAL

        # --- 4b. Load Base Scenario Data ---
        with open(scenario_path, "rb") as f:
            model_inputs = dill.load(f)
            

        # --- 4c. Find Optimal water limit using Secant/Bisection Method ---
        limit_min, limit_max = 4, 12
        withdrawal_tolerance = 0.005 * TARGET_WITHDRAWAL
        max_iterations = 50

        solution = custom_bisect(
            withdrawal_difference, limit_min, limit_max,
            withdrawal_tolerance=withdrawal_tolerance,
            max_iterations=max_iterations, f_args=model_inputs
        )

        if solution is None:
             print(f"  FAILURE: Search failed for scenario {scenario_name}.")
             return

        final_water_limit = round(solution, 2)
        print(f"  SUCCESS: Converged at water limit: {final_water_limit} m-ha.")

        # --- 4d. Run Final Simulation and Save ---
        with open(scenario_path, "rb") as f:
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = dill.load(f)
            
        for bid in behaviors_dict:
            # The model expects water rights in cm, so convert from inches
            behaviors_dict[bid]["water_rights"]["wr_yr"]["wr_depth"] = final_water_limit * 2.54
            
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

        df_farmers, df_fields, df_wells, df_aquifers = GCPModelUr.get_dfs(m)
        df_sys = GCPModelUr.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

        output_excel_path = output_dir / f'ur_{scenario_name}_wl_{final_water_limit}.xlsx'
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


def custom_bisect(f, a, b, withdrawal_tolerance, max_iterations, f_args):
    """A custom secant/bisection root-finding method."""
    fa = f(a, *f_args)
    fb = f(b, *f_args)
    if np.sign(fa) == np.sign(fb):
        print("Warning: Root may not be bracketed (f(a) and f(b) have same sign). Search may fail.")

    for iteration in range(1, max_iterations + 1):
        if fb - fa == 0:
            c = (a + b) / 2.0
        else:
            c = b - fb * (b - a) / (fb - fa)
        
        if not (a < c < b):
             c = (a + b) / 2.0

        fc = f(c, *f_args)
        print(f"  Iteration {iteration}: Water limit={c:.2f}, Withdrawal Diff={fc:.2f}")

        if abs(fc) <= withdrawal_tolerance:
            return c
        
        if np.sign(fa) != np.sign(fc):
            b, fb = c, fc
        else:
            a, fa = c, fc
            
    print("Maximum iterations reached.")
    return None

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # NEW: allow a single-id override (for Slurm arrays)
    parser = argparse.ArgumentParser(description="Run baseline scenario(s)")
    parser.add_argument("--id", type=int, help="If provided, run exactly this bootstrap id (e.g., 1..500)")
    args = parser.parse_args()
    if args.id is not None:
        START_SCENARIO = END_SCENARIO = int(args.id)
        
    print(f"\nStarting Uniform Restriction model runs for scenarios {START_SCENARIO} through {END_SCENARIO}...")

    for i in tqdm(range(START_SCENARIO, END_SCENARIO + 1), desc="Overall Progress"):
        # Use the new file naming convention
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"

        if scenario_path.exists():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            
            # Check if an output file for this scenario already exists to allow skipping
            output_exists = any(f.startswith(f"ur_{scenario_name}_wl_") for f in os.listdir(OUTPUT_DIR))
            
            if output_exists:
                print(f"Result for {scenario_name} already exists. Skipping.")
                continue

            run_ur_scenario(scenario_name, scenario_path, OUTPUT_DIR)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")

    print("\nAll specified scenarios have been processed.")
#%%######################### WARNING #####################################
#This script is designed to be run after all the uniform restriction (UR)
# scenarios have completed. It scans the output directory for the generated
# Excel files, parses the filenames to extract the final optimal water limit
# for each scenario, and compiles them into a single summary CSV file.
#
# Steps:
#   1. Sets up the path to the output directory.
#   2. Scans the directory for all relevant .xlsx result files.
#   3. Uses regular expressions to parse the scenario name and water limit
#      from each filename.
#   4. Aggregates the data into a pandas DataFrame.
#   5. Saves the final summary to 'ur_optimal_water_limits.csv'.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
from pathlib import Path
import re

# --- 2. Set Up File Paths ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for interactive environments like Spyder
    PROJECT_ROOT = Path(os.getcwd())

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ur_runs"
print(f"Scanning for results in: {OUTPUT_DIR}")

# --- 3. Extract Results from Filenames ---
results_list = []
# Define a regular expression to capture the scenario name and the water limit
# It looks for files like 'ur_b_1_wl_8.13.xlsx'
filename_pattern = re.compile(r"ur_(b_\d+)_wl_(\d+\.\d+)\.xlsx")

# Check if the output directory exists
if OUTPUT_DIR.exists():
    # Loop through all files in the output directory
    for filename in os.listdir(OUTPUT_DIR):
        # Try to match the filename against our pattern
        match = filename_pattern.match(filename)
        if match:
            # If it matches, extract the scenario name and the water limit
            scenario_name = match.group(1)
            water_limit = float(match.group(2))
            
            # Append the extracted data to our results list
            results_list.append({
                'Scenario': scenario_name,
                'Final_Water_Limit_Inches': water_limit
            })
else:
    print(f"Error: Output directory not found at {OUTPUT_DIR}")

# --- 4. Create and Save Summary DataFrame ---
if results_list:
    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results_list)
    results_df.sort_values(by="Scenario", inplace=True)
    
    # Define the path for the final summary CSV file
    results_csv_path = OUTPUT_DIR / "ur_optimal_water_limits.csv"
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\nSuccessfully created summary file with {len(results_df)} entries.")
    print(f"Results saved to: {results_csv_path}")
else:
    print("\nNo result files found to summarize.")