# -----------------------------------------------------------------------------
# 02_prepare_model_inputs.py
#
# This is a unified script that processes each bootstrapped climate and market
# scenario to create universal input files suitable for ALL policy scenarios
# (Baseline, Uniform Restriction, Fee-Based, Priority-Based, and Combined).
#
# Steps:
#   1. Sets up robust file paths to automatically locate input/output dirs.
#   2. Iterates through each of the 500 bootstrapped CSV files.
#   3. For each scenario, it processes precipitation, price, and cost data
#      into time-series and rolling average dictionaries.
#   4. Loads static farm and aquifer data (e.g., grid info, well depths).
#   5. Constructs the five main PyCHAMP input dictionaries: `aquifers_dict`,
#      `fields_dict`, `wells_dict`, `finances_dict`, and `behaviors_dict`.
#   6. Assembles these dictionaries and a `shared_config` into a final
#      tuple.
#   7. Saves this tuple as a compressed pickle file (`.pkl`) for each
#      bootstrap scenario.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import dill
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path


# --- 2. Set Up File Paths Automatically ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    print("Could not set PROJECT_ROOT automatically. Using current working directory.")
    PROJECT_ROOT = Path(os.getcwd())

# Define input and a single, unified output directory
BOOTSTRAP_DATA_DIR = PROJECT_ROOT / "inputs" / "bootstrap_samples"
RAW_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading bootstrapped data from: {BOOTSTRAP_DATA_DIR}")
print(f"Saving universal model inputs to: {OUTPUT_DIR}")


# --- 3. Define Main Parameters ---
INIT_YEAR = 2001
CROP_OPTIONS = ["corn", "sorghum", "soybeans", "wheat", "fallow"]
SELECTED_GRIDMET_GRIDS = [
    'grid37', 'grid9', 'grid26', 'grid27', 'grid17', 'grid29', 'grid10',
    'grid7', 'grid4', 'grid5', 'grid34', 'grid8', 'grid24', 'grid38', 'grid23',
    'grid19', 'grid14', 'grid43', 'grid12', 'grid42', 'grid28', 'grid18',
    'grid32', 'grid21', 'grid22', 'grid33', 'grid13'
]


# --- 4. Main Processing Loop ---
bootstrapped_files = list(BOOTSTRAP_DATA_DIR.glob("bootstrapped_data_*.csv"))
print(f"\nFound {len(bootstrapped_files)} bootstrapped scenarios to process.")

for file_path in tqdm(bootstrapped_files, desc="Processing Scenarios"):
    # --- 4a. Load and Prepare Data ---
    num_str = file_path.stem.split('_')[-1]
    scenario = f"b_{num_str}"

    bootstrapped_data = pd.read_csv(file_path)
    bootstrapped_data.set_index('Year', inplace=True)
    if 'Year Sampled From' in bootstrapped_data.columns:
        bootstrapped_data.drop(['Year Sampled From', 'Average_Precipitation'], axis=1, inplace=True)

    # --- 4b. Process Climate and Economic Data (Identical for all policies) ---
    precip_df = bootstrapped_data[['Crop'] + SELECTED_GRIDMET_GRIDS]
    prec_aw_step, prec_aw_rolling_step = {}, {}
    for grid in SELECTED_GRIDMET_GRIDS:
        pivot_df = precip_df.pivot(columns='Crop', values=grid)
        pivot_df['fallow'] = 0
        prec_aw_step[grid] = pivot_df[pivot_df.index >= INIT_YEAR].to_dict(orient='index')
        rolling_data = pivot_df.shift(1).rolling(window=5, min_periods=1).mean().round(2)
        rolling_data['fallow'] = 0
        prec_aw_rolling_step[grid] = rolling_data[rolling_data.index >= INIT_YEAR].to_dict(orient='index')

    prices_costs_dict = {}
    metrics = [
        ("Prices", "crop_price_step", "crop_price_avg_step"),
        ("Variable Costs", "crop_variable_cost_step", "crop_variable_cost_avg_step"),
        ("Fixed Costs", "crop_fixed_cost_step", "crop_fixed_cost_avg_step")
    ]
    for cost_col, step_key, avg_step_key in metrics:
        cost_pivot = bootstrapped_data.pivot(columns='Crop', values=cost_col)
        cost_pivot['fallow'] = 0

        # Current year economic data
        step_data = cost_pivot[cost_pivot.index >= INIT_YEAR].round(3)
        prices_costs_dict[step_key] = {"finance": step_data.T.to_dict()}

        # 5-year rolling average economic data
        rolling_data = cost_pivot.shift(1).rolling(window=5, min_periods=1).mean()
        rolling_data['fallow'] = 0
        avg_data = rolling_data[rolling_data.index >= INIT_YEAR].round(3)
        prices_costs_dict[avg_step_key] = {"finance": avg_data.T.to_dict()}

    crop_price_step = prices_costs_dict['crop_price_step']
    crop_price_avg_step = prices_costs_dict['crop_price_avg_step']
    crop_variable_cost_step = prices_costs_dict['crop_variable_cost_step']
    crop_variable_cost_avg_step = prices_costs_dict['crop_variable_cost_avg_step']
    crop_fixed_cost_step = prices_costs_dict['crop_fixed_cost_step']
    crop_fixed_cost_avg_step = prices_costs_dict['crop_fixed_cost_avg_step']

    # --- 4c. Load and Prepare Static Farm Data ---
    sd6_grid_info = pd.read_csv(RAW_DATA_DIR / "SD6_grid_info.csv")
    selected_sd6_grids = sd6_grid_info[sd6_grid_info['other_freq'] <= 4].reset_index()
    fnum = selected_sd6_grids.shape[0]
    selected_sd6_grids["fid"] = [f"f{i+1}" for i in range(fnum)]
    selected_sd6_grids["wid"] = [f"w{i+1}" for i in range(fnum)]
    selected_sd6_grids["bid"] = [f"b{i+1}" for i in range(fnum)]
    selected_sd6_grids["aqid"] = "sd6"

    # --- 4d. Construct Universal PyCHAMP Input Dictionaries ---
    # Aquifer Inputs
    aquifers_dict = {
        "sd6": {
            "aq_a": 0.0003310, "aq_b": 0.6286, "area": None, "sy": None,
            "init": {
                "st": sd6_grid_info[f"st_m_{INIT_YEAR}"].mean(),
                "dwl": -0.6
            }
        }
    }

    # Field Inputs
    fields_dict = {}
    # Use a pseudo-initial year for setting initial crop types
    pseudo_init_crop_year = 2006
    crop_pool = [c for c in selected_sd6_grids[f'Crop{pseudo_init_crop_year}'] if c in CROP_OPTIONS]
    for _, row in selected_sd6_grids.iterrows():
        fid = row['fid']
        init_crop = row[f'Crop{pseudo_init_crop_year}']
        if init_crop not in CROP_OPTIONS:
            init_crop = np.random.choice(crop_pool) # Assign a random valid crop if initial is invalid

        # Determine if a field is rainfed or can be irrigated
        field_type = "rainfed" if row['irr_freq'] <= 0 else "optimize"
        # Also designate fields as rainfed if they ceased irrigation after LEMA
        years_post_lema = [f'Irr{year}' for year in range(2013, 2021)]
        if all(row[year] == 0 for year in years_post_lema):
            field_type = "rainfed"

        fields_dict[fid] = {
            "field_area": 50., "water_yield_curves": None, "tech_pumping_rate_coefs": None,
            "prec_aw_id": row['gridmet_id'],
            "init": {"tech": "center pivot LEPA", "crop": init_crop, "field_type": field_type},
            # Additional metadata for analysis
            "truncated_normal_pars": None, 'irr_freq': row['irr_ratio'],
            "irr_freq_number": row['irr_freq'], "lat": row['lat'], "lon": row['lon'],
            "y": row['Y'], "x": row['X'], "field_type_rn": field_type,
            "field_type": field_type
        }
        
    # Well Inputs
    wells_dict = {}
    selected_sd6_grids["well_st"] = selected_sd6_grids[f'wl_ele_m_{INIT_YEAR}'] - selected_sd6_grids['well_depth_ele']
    for _, row in selected_sd6_grids.iterrows():
        wid = row['wid']
        wells_dict[wid] = {
            "r": None, "k": row['well_k'], "sy": row['well_sy'], "rho": None, "g": None,
            "eff_pump": None, "eff_well": None, "aquifer_id": row['aqid'], "pumping_capacity": None,
            "init": {
                "l_wt": row[f'wl_depth_m_{INIT_YEAR}'],
                "st": row['well_st'],
                "pumping_days": 90
            }
        }
        
    # Finance Inputs    
    finances_dict = {"finance": {
            "pumping_fee": None, "energy_price": None, "crop_price": {},
            "crop_fixed_cost": {}, "crop_variable_cost": {}, "crop_price_avg": {},
            "crop_fixed_cost_avg": {}, "crop_variable_cost_avg": {},
            "irr_tech_operational_cost": {}, "irr_tech_change_cost": {},
            "crop_change_cost": {}
        }
    }
    
    # Behavior Inputs
    # --- 4e. Add Seniority IDs and Construct Behavior Dictionary ---
    # This section is critical for priority-based scenarios
    num_optimize = sum(1 for fid, info in fields_dict.items() if info['init']['field_type'] == 'optimize')
    num_rainfed = sum(1 for fid, info in fields_dict.items() if info['init']['field_type'] == 'rainfed')
    
    optimize_ids = list(range(1, num_optimize + 1))
    rainfed_ids = list(range(1000, 1000 + num_rainfed))
    
    random.seed(3) # for reproducibility
    random.shuffle(optimize_ids)
    
    optimize_index, rainfed_index = 0, 0
    
    behaviors_dict = {}
    in2cm = 2.54
    for _, row in selected_sd6_grids.iterrows():
        bid = row['bid']
        fid = row['fid']
        field_type = fields_dict[fid]['field_type']
        
        if field_type == "optimize":
            seniority_id = optimize_ids[optimize_index]
            optimize_index += 1
        else: # rainfed
            seniority_id = rainfed_ids[rainfed_index]
            rainfed_index += 1
            
        behaviors_dict[bid] = {
            "seniority_id": str(seniority_id),
            "field_ids": [row["fid"]], "well_ids": [row["wid"]], "finance_id": "finance",
            "behavior_ids_in_network": [], # Set to empty list as social network is not used
            "decision_making": {
                "target": "profit", "horizon": 1, "n_dwl": 5, "keep_gp_model": False,
                "keep_gp_output": False, "display_summary": False, "display_report": False
            },
            "water_rights": {
                "wr_yr": {
                    "wr_depth": 50 * in2cm, "applied_field_ids": [row["fid"]],
                    "time_window": 1, "remaining_tw": None, "remaining_wr": None,
                    "tail_method": "proportion", "status": True
                },
            },
            "gurobi": {}
        }
    
    # Shared Configuration
    shared_config = {
        "field": {
            "water_yield_curves": {
                'corn': [457.6316, 79.4827, -3.0517, 5.5043, -1.482, 0.0193],
                'sorghum': [184.4876, 61.8959, -1.8133, 3.252, -0.458, 0.6327],
                'soybeans': [145.3771, 72.9901, -2.5114, 4.4496, -0.9709, 0.0081],
                'wheat': [130.3249, 78.3709, -2.026, 3.2315, -0.2886, 0.2867],
                'fallow': [0.0, 100.0, 0, 0, 0.0, 0]
            },
            "tech_pumping_rate_coefs": {
                "center pivot": [0.0051, 0.268744, 28.12],
                "center pivot LEPA": [0.0058, 0.212206, 12.65]
            },
        },
        "well": {"r": 0.4064 / 2, "rho": 1000., "g": 9.8016, "eff_pump": 0.77, "eff_well": 0.5},
        "finance": {
            "pumping_fee": None, "energy_price": 2777.777778,
            "crop_price_avg": {c: crop_price_avg_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "crop_price": {c: crop_price_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "crop_variable_cost_avg": {c: crop_variable_cost_avg_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "crop_variable_cost": {c: crop_variable_cost_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "crop_fixed_cost_avg": {c: crop_fixed_cost_avg_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "crop_fixed_cost": {c: crop_fixed_cost_step['finance'][INIT_YEAR][c] for c in CROP_OPTIONS},
            "irr_tech_operational_cost": {"center pivot LEPA": 1.876},
            "irr_tech_change_cost": {}, "crop_change_cost": {}
        },
        "behavior": {"gurobi": {"LogToConsole": 0, "Presolve": -1}}
    }

    # --- 4f. Assemble and Save Final Inputs ---
    inputs = (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
              prec_aw_step, prec_aw_rolling_step, crop_price_avg_step, crop_price_step,
              crop_fixed_cost_avg_step, crop_fixed_cost_step, crop_variable_cost_avg_step,
              crop_variable_cost_step, shared_config)

    output_filename = OUTPUT_DIR / f"{scenario}.pkl"
    with open(output_filename, "wb") as f:
        dill.dump(inputs, f)

print(f"\nProcessing complete. All universal pickle files saved to: {OUTPUT_DIR}")