# -----------------------------------------------------------------------------
# 03_run_bs_scenario.py
#
# This script runs the agent-based model for a specified range of baseline (bs)
# scenarios. It iterates through the pre-generated input pickle files,
# initializes the model for each, runs the simulation, and saves the results
# to multi-sheet Excel files.
#
# This script is designed to be run sequentially, which is more stable than
# parallel processing for memory-intensive simulations.
#
# Steps:
#   1. Sets up file paths and imports the necessary model class.
#   2. Sets user-configurable parameters for which scenarios to run.
#   3. Defines the `run_scenario` function, which handles loading data,
#      initializing the GCPModelUr, running the simulation, and saving results.
#   4. The main execution block loops through the specified range of scenarios,
#      calling the `run_scenario` function for each.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import sys
import gc
import dill
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# --- 2. Set Up File Paths and Imports ---
# This block adds the 'parent_package' directory to the Python path.
# This is a robust way to ensure that the script can find and import the
# custom model class (GCPModelUr) from its location in the repository.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for interactive environments like Spyder
    print("Could not set PROJECT_ROOT automatically. Using current working directory.")
    PROJECT_ROOT = Path(os.getcwd())

# Add the parent package to the system path to allow for direct import
PACKAGE_PATH = PROJECT_ROOT / "scripts" / "parent_package"
sys.path.insert(0, str(PACKAGE_PATH))

# Import the custom model class from its new location
from py_champ.models.gcp_model_ur import GCPModelUr
# from py_champ.components.behavior import Behavior

# Define input and output directories relative to the project root
INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_runs"

# Create the output directory if it doesn't exist
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


# --- 4. Define the Core Model Execution Function ---
def run_scenario(scenario_name, scenario_path, output_dir):
    """
    Loads, runs, and saves a single model scenario.

    Args:
        scenario_name (str): The identifier for the scenario (e.g., 'b_1').
        scenario_path (Path): The full path to the input pickle file.
        output_dir (Path): The directory where output Excel files will be saved.
    """
    # Define variables as None initially to ensure they exist for the 'finally' block
    m = None
    df_farmers, df_fields, df_wells, df_aquifers, df_sys = (None,) * 5
    
    try:
        # Define the expected output file path
        output_excel_file = output_dir / f'baseline_{scenario_name}.xlsx'

        # Skip the run if the output file already exists
        if output_excel_file.exists():
            print(f"Result for {scenario_name} already exists. Skipping.")
            return

        # Load the input data for the scenario
        with open(scenario_path, "rb") as f:
            (
                aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
                prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
                crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
                crop_variable_cost_step, shared_config
            ) = dill.load(f)

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

        # Run the model for 21 steps (from 2002 to 2022)
        for _ in range(21):
            m.step()
        m.end()

        # --- Collect and Save Results ---
        df_farmers, df_fields, df_wells, df_aquifers = GCPModelUr.get_dfs(m)
        df_sys = GCPModelUr.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

        with pd.ExcelWriter(output_excel_file) as writer:
            df_farmers.to_excel(writer, sheet_name='Farmers', index=True)
            df_fields.to_excel(writer, sheet_name='Fields', index=True)
            df_wells.to_excel(writer, sheet_name='Wells', index=True)
            df_aquifers.to_excel(writer, sheet_name='Aquifers', index=True)
            df_sys.to_excel(writer, sheet_name='System', index=True)

        print(f"Scenario {scenario_name} completed successfully.")

    except Exception as e:
        print(f"--- ERROR in scenario {scenario_name}: {e} ---")

    finally:
        # --- Clean up memory ---
        # This block checks if each variable was created before trying to delete it,
        # preventing the UnboundLocalError.
        if 'm' in locals() and m is not None:
            del m
        if 'df_farmers' in locals() and df_farmers is not None:
            del df_farmers, df_fields, df_wells, df_aquifers, df_sys
        
        # Clean up loaded input dictionaries
        if 'aquifers_dict' in locals():
            del aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict
            del prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step
            del crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step
            del crop_variable_cost_step, shared_config
        
        # Force Python's garbage collector to run
        gc.collect()


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # NEW: allow a single-id override (for Slurm arrays)
    parser = argparse.ArgumentParser(description="Run baseline scenario(s)")
    parser.add_argument("--id", type=int, help="If provided, run exactly this bootstrap id (e.g., 1..500)")
    args = parser.parse_args()
    if args.id is not None:
        START_SCENARIO = END_SCENARIO = int(args.id)

    print(f"\nStarting model runs for scenarios {START_SCENARIO} through {END_SCENARIO}...")
    for i in tqdm(range(START_SCENARIO, END_SCENARIO + 1), desc="Overall Progress"):
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"
        if scenario_path.exists():
            print(f"\n--- Running scenario: {scenario_name} ---")
            run_scenario(scenario_name, scenario_path, OUTPUT_DIR)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")
    print("\nAll specified scenarios have been processed.")