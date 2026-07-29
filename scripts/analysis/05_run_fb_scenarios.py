# -----------------------------------------------------------------------------
# 05_run_fb_scenarios.py
#
# Runs Fee-Based (FB) scenarios and can also run the
# Fee-Based + Cash-for-Blue (FB_CB) 
#
# Key idea:
#   POLICY_MODE = "FB"     -> original fee-based policy
#   POLICY_MODE = "FB_CB"  -> fee-based policy with cash-for-blue logic
#
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import sys
import gc
import dill
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 2. Set Up File Paths and Imports ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

PACKAGE_PATH = PROJECT_ROOT / "scripts" / "py_champ_package"
sys.path.insert(0, str(PACKAGE_PATH))

from py_champ.models.gcp_model_fb import GCPModelFb

INPUT_DIR = PROJECT_ROOT / "inputs" / "model_inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Loading model inputs from: {INPUT_DIR}")


# --- 3. User Configuration ---
START_SCENARIO = 1
END_SCENARIO = 500

# Main knob. Can also be overridden from command line with --policy_mode.
POLICY_MODE = "FB"  # options: "FB", "FB_CB"

ACRE_FOOT_TO_M_HA = 0.123348  # acre-foot to m-ha
TARGET_WITHDRAWAL = 1900

# Fee-search controls
FEE_MIN = 0.0
FEE_MAX = 500.0
MAX_SEARCH_ITERATIONS = 25
WITHDRAWAL_TOLERANCE_FRAC = 0.005

# Cash-for-blue settings. These are placeholders for the next code steps.
# The model will start using these after we update gcp_model_fb.py / behavior.py.
CASH_FOR_BLUE_CONFIG = {
    "enabled": False,
    "first_enrollment_year": 2003,
    "fund_lag_years": 1,
    "ranking_metric": "profit_per_irr_vol",
    "ranking_order": "ascending",  # lowest return per irrigation volume first
    "rainfed_benchmark": "mean_always_rainfed_profit_previous_year",
    "payout_rule": "reference_profit_minus_rainfed_benchmark",
    "allow_partial_enrollment": False,
    "carryover_unused_funds": True,
    "allow_zero_payout_enrollment": True,
    "zero_payout_rule": "nonpositive_payout", #negative_ref_profit_only - other option
}


def get_output_dir(policy_mode: str) -> Path:
    """Return the output directory for the selected policy mode."""
    policy_mode = policy_mode.upper()
    if policy_mode == "FB":
        output_dir = OUTPUT_ROOT / "fb_runs"
    elif policy_mode == "FB_CB":
        output_dir = OUTPUT_ROOT / "fb_cb_runs"
    else:
        raise ValueError("policy_mode must be 'FB' or 'FB_CB'.")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def fee_to_model_units(fee_acft: float) -> float:
    """
    Convert fee from $/acre-foot to model units.

    Model expects pumping_fee in 1e4 $ / m-ha.
    """
    return fee_acft / (ACRE_FOOT_TO_M_HA * 1e4)


def turn_off_water_rights(behaviors_dict: dict) -> None:
    """Turn off volumetric water rights for fee-based scenarios."""
    for bid in behaviors_dict:
        behaviors_dict[bid]["water_rights"]["wr_yr"]["status"] = False


def unpack_model_inputs(model_inputs):
    """Unpack model input tuple using the existing file structure."""
    return model_inputs


def build_model(
    model_inputs,
    fee_acft: float,
    policy_mode: str,
    cash_for_blue_config: dict,
    show_step: bool = True,
):
    """Build a GCPModelFb instance for either FB or FB_CB."""
    (
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        prec_aw_rolling_step,
        crop_price_avg_step,
        crop_price_step,
        crop_fixed_cost_avg_step,
        crop_fixed_cost_step,
        crop_variable_cost_avg_step,
        crop_variable_cost_step,
        shared_config,
    ) = unpack_model_inputs(copy.deepcopy(model_inputs))

    turn_off_water_rights(behaviors_dict)
    shared_config["finance"]["pumping_fee"] = fee_to_model_units(fee_acft)

    # Make an explicit config copy so one simulation cannot mutate another.
    cb_config = copy.deepcopy(cash_for_blue_config)
    cb_config["enabled"] = policy_mode.upper() == "FB_CB"

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
        # Policy extension knobs. These will be used after model updates.
        policy_mode=policy_mode.upper(),
        cash_for_blue_config=cb_config,
        # Other options
        show_step=show_step,
    )
    return model


# --- 4. Define the Core Model Execution Function ---
def run_fee_scenario(scenario_name, scenario_path, output_dir, policy_mode="FB"):
    """
    Finds the optimal fixed pumping fee for one scenario by iteratively running
    simulations, then saves the detailed final run.
    """
    m = None
    policy_mode = policy_mode.upper()
    output_prefix = "fb_cb" if policy_mode == "FB_CB" else "fb"

    try:
        # --- 4a. Load Base Scenario Data ---
        with open(scenario_path, "rb") as f:
            model_inputs = dill.load(f)

        # --- 4b. Define Objective Function for Fee Search ---
        def withdrawal_difference(fee, *args):
            model_inputs_static = args[0]
            model = build_model(
                model_inputs=model_inputs_static,
                fee_acft=fee,
                policy_mode=policy_mode,
                cash_for_blue_config=CASH_FOR_BLUE_CONFIG,
                show_step=True,
            )

            for _ in range(21):
                model.step()
            model.end()

            _, _, _, df_aquifers = GCPModelFb.get_dfs(model)
            current_withdrawal = df_aquifers.loc[2002:2022, "withdrawal"].mean()

            del model, df_aquifers
            gc.collect()
            return current_withdrawal - TARGET_WITHDRAWAL

        # --- 4c. Find Optimal Fee ---
        tolerance = WITHDRAWAL_TOLERANCE_FRAC * TARGET_WITHDRAWAL
        optimal_fee = custom_bisect(
            withdrawal_difference,
            FEE_MIN,
            FEE_MAX,
            withdrawal_tolerance=tolerance,
            max_iterations=MAX_SEARCH_ITERATIONS,
            f_args=(model_inputs,),
        )
        final_pumping_fee = round(optimal_fee, 2)
        print(
            f"  SUCCESS: {policy_mode} converged at pumping fee: "
            f"${final_pumping_fee}/ac-ft."
        )

        # --- 4d. Run Final Simulation and Save ---
        m = build_model(
            model_inputs=model_inputs,
            fee_acft=final_pumping_fee,
            policy_mode=policy_mode,
            cash_for_blue_config=CASH_FOR_BLUE_CONFIG,
            show_step=True,
        )

        for _ in range(21):
            m.step()
        m.end()

        df_farmers, df_fields, df_wells, df_aquifers = GCPModelFb.get_dfs(m)
        df_sys = GCPModelFb.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)
        
        final_avg_withdrawal = df_aquifers.loc[2002:2022, "withdrawal"].mean()
        final_diff = final_avg_withdrawal - TARGET_WITHDRAWAL
        
        print(
            f"  FINAL CHECK: fee=${final_pumping_fee:.2f}, "
            f"avg_withdrawal={final_avg_withdrawal:.2f}, "
            f"target={TARGET_WITHDRAWAL:.2f}, "
            f"diff={final_diff:.2f}"
        )
        
        if abs(final_diff) > tolerance:
            print(
                "  WARNING: Final run did not meet withdrawal tolerance. "
                "Output is being saved but should be flagged."
            )

        output_excel_path = output_dir / f"{output_prefix}_{scenario_name}_pf_{final_pumping_fee}.xlsx"
        with pd.ExcelWriter(output_excel_path) as writer:
            df_farmers.to_excel(writer, sheet_name="Farmers", index=True)
            df_fields.to_excel(writer, sheet_name="Fields", index=True)
            df_wells.to_excel(writer, sheet_name="Wells", index=True)
            df_aquifers.to_excel(writer, sheet_name="Aquifers", index=True)
            df_sys.to_excel(writer, sheet_name="System", index=True)

        return scenario_name, final_pumping_fee

    except Exception as e:
        print(f"--- ERROR in scenario {scenario_name}: {e} ---")
        return scenario_name, None
    finally:
        if m is not None:
            del m
        gc.collect()


def custom_bisect(f, a, b, withdrawal_tolerance, max_iterations, f_args):
    """
    Bracket-safe fee search.

    f(fee) = avg_withdrawal - TARGET_WITHDRAWAL

    Positive f means withdrawal is too high and fee should increase.
    Negative f means withdrawal is too low and fee should decrease.
    """
    fa = f(a, *f_args)
    fb = f(b, *f_args)

    print(f"  Bracket check: fee=${a:.2f}, diff={fa:.2f}", flush=True)
    print(f"  Bracket check: fee=${b:.2f}, diff={fb:.2f}", flush=True)

    if abs(fa) <= withdrawal_tolerance:
        print(f"  Lower bound is within tolerance. Using fee=${a:.2f}.", flush=True)
        return a

    if abs(fb) <= withdrawal_tolerance:
        print(f"  Upper bound is within tolerance. Using fee=${b:.2f}.", flush=True)
        return b

    # Both negative means even the minimum fee is too restrictive.
    # The target is below the lower bound.
    if fa < 0 and fb < 0:
        print(
            "  WARNING: Target is not bracketed. "
            "Withdrawal is below target even at the minimum fee. "
            f"Returning lower bound fee=${a:.2f}.",
            flush=True,
        )
        return a

    # Both positive means even the maximum fee is not restrictive enough.
    # The target is above the upper bound.
    if fa > 0 and fb > 0:
        print(
            "  WARNING: Target is not bracketed. "
            "Withdrawal is above target even at the maximum fee. "
            f"Returning upper bound fee=${b:.2f}.",
            flush=True,
        )
        return b

    for iteration in range(1, max_iterations + 1):
        c = (a + b) / 2.0
        fc = f(c, *f_args)

        print(f"  Iteration {iteration}: Fee=${c:.2f}, Withdrawal Diff={fc:.2f}", flush=True)

        if abs(fc) <= withdrawal_tolerance:
            return c

        if np.sign(fa) != np.sign(fc):
            b, fb = c, fc
        else:
            a, fa = c, fc

    print("Maximum iterations reached.", flush=True)
    return (a + b) / 2.0


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fee-based scenario(s)")
    parser.add_argument("--id", type=int, help="Run exactly this bootstrap id, e.g., 1..500")
    parser.add_argument(
        "--policy_mode",
        type=str,
        choices=["FB", "FB_CB"],
        default=POLICY_MODE,
        help="Policy mode to run: FB or FB_CB",
    )
    parser.add_argument("--start", type=int, default=START_SCENARIO, help="First bootstrap id")
    parser.add_argument("--end", type=int, default=END_SCENARIO, help="Last bootstrap id")
    args = parser.parse_args()

    policy_mode = args.policy_mode.upper()
    output_dir = get_output_dir(policy_mode)

    start_scenario = args.start
    end_scenario = args.end
    if args.id is not None:
        start_scenario = end_scenario = int(args.id)

    output_prefix = "fb_cb" if policy_mode == "FB_CB" else "fb"

    print(f"Saving model outputs to: {output_dir}")
    print(
        f"\nStarting {policy_mode} model runs for scenarios "
        f"{start_scenario} through {end_scenario}..."
    )

    for i in tqdm(range(start_scenario, end_scenario + 1), desc="Overall Progress"):
        scenario_name = f"b_{i}"
        scenario_path = INPUT_DIR / f"{scenario_name}.pkl"

        if scenario_path.exists():
            print(f"\n--- Processing scenario: {scenario_name} ---")
            output_exists = any(
                f.startswith(f"{output_prefix}_{scenario_name}_pf_")
                for f in os.listdir(output_dir)
            )
            if output_exists:
                print(f"Result for {scenario_name} already exists. Skipping.")
                continue
            run_fee_scenario(scenario_name, scenario_path, output_dir, policy_mode=policy_mode)
        else:
            print(f"\nWarning: Input file not found for scenario {scenario_name}. Skipping.")

    print("\nAll specified scenarios have been processed.")
