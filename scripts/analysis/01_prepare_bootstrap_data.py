# -----------------------------------------------------------------------------
# 01_prepare_bootstrap_data.py
#
# This script prepares climate and economic data for bootstrapping.
# It automatically detects the project's root directory to ensure that
# file paths work on any computer, regardless of the IDE's current
# working directory.
#
# Steps:
#   1. Defines project file paths relative to the script's location.
#   2. Calculates growing season precipitation for each crop.
#   3. Processes and aggregates annual crop costs and prices.
#   4. Merges all data into a single dataset.
#   5. Generates 500 bootstrap samples from the merged data.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pathlib import Path # Modern library for handling file paths

# --- 2. Set Up File Paths Automatically ---

# It finds the path to this script and then goes up two levels (from
# scripts/analysis/ to the main project root).
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # If running interactively (e.g., in a notebook), you might need to
    # set the path manually. For Spyder, running the file should work.
    print("Could not set PROJECT_ROOT automatically. Using current working directory.")
    PROJECT_ROOT = Path(os.getcwd())


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "inputs" / "bootstrap_samples"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file paths
PRECIP_FILE = DATA_DIR / 'prec_D_cm_1996_2023.csv'
PRICES_FILE = DATA_DIR / "prices_raw_for_bootstrap.csv"
COSTS_FILE = DATA_DIR / "costs_raw_for_bootstrap.csv"

print(f"Project Root Directory set to: {PROJECT_ROOT}")
print(f"Looking for data in: {DATA_DIR}")
print(f"Saving results to: {OUTPUT_DIR}")


# --- 3. Define Helper Functions ---

def calculate_growing_season_precip(precip_df, growing_seasons, start_year, end_year, crop_options):
    """
    Calculates the total precipitation for each crop's growing season over a
    specified period.
    """
    if precip_df.index[-1].year < end_year + 1:
        raise ValueError(f"Precipitation data must extend to at least {end_year + 1}.")

    df_list = []
    for crop in crop_options:
        start_str, end_str = growing_seasons[crop]
        s_date = pd.to_datetime(start_str, format="%m/%d")
        e_date = pd.to_datetime(end_str, format="%m/%d")
        month_offset = s_date.month - 1
        s_adjusted = s_date - relativedelta(months=month_offset)
        e_adjusted = e_date - relativedelta(months=month_offset)
        precip_copy = precip_df.copy()
        precip_copy.index = [i - relativedelta(months=month_offset) for i in precip_copy.index]
        mask = pd.Series(precip_copy.index.strftime('%m%d').astype(int)).between(
            int(s_adjusted.strftime('%m%d')),
            int(e_adjusted.strftime('%m%d'))
        ).tolist()
        yearly_precip = precip_copy[mask].resample("YS").sum()[str(start_year):str(end_year)]
        yearly_precip.index = yearly_precip.index.year
        yearly_precip['Crop'] = crop
        df_list.append(yearly_precip)
    total_precip_df = pd.concat(df_list)
    return total_precip_df

# --- 4. Set Up Parameters ---
CROP_OPTIONS = ["corn", "sorghum", "soybeans", "wheat"]
GROWING_SEASONS = {
    "corn": ["5/1", "10/3"],
    "sorghum": ["6/2", "11/3"],
    "soybeans": ["6/2", "10/15"],
    "wheat": ["10/3", "6/27"]
}
ANALYSIS_START_YEAR = 1996
ANALYSIS_END_YEAR = 2022
BOOTSTRAP_SAMPLES = 500
RANDOM_SEED = 123

# --- 5. Load and Preprocess Input Data ---
print("\nLoading and preprocessing data...")
precip_data = pd.read_csv(PRECIP_FILE, index_col=0, parse_dates=True)
costs_data = pd.read_csv(COSTS_FILE)
prices_data = pd.read_csv(PRICES_FILE)

# --- 6. Process Precipitation Data ---
print("Calculating growing season precipitation...")
precip_totals = calculate_growing_season_precip(
    precip_data, GROWING_SEASONS, ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, CROP_OPTIONS
)
grid_columns = [col for col in precip_totals.columns if 'grid' in col]
precip_totals['Average_Precipitation'] = precip_totals[grid_columns].mean(axis=1)
precip_totals.reset_index(inplace=True)
precip_totals.rename(columns={'index': 'Year'}, inplace=True)

# --- 7. Process Economic Data ---
print("Processing economic data...")
avg_costs = costs_data.groupby(['Year', 'Crop']).mean(numeric_only=True).reset_index()
avg_costs['Variable Costs'] = avg_costs['seed'] + avg_costs['fertilizers'] + avg_costs['crop insurance']
avg_costs['Fixed Costs'] = avg_costs['depreciation']
avg_prices = prices_data.groupby(['Year', 'Crop']).mean(numeric_only=True).reset_index()
avg_prices.rename(columns={'Gross Income': 'Prices'}, inplace=True)

# --- 8. Merge All Data Sources ---
print("Merging precipitation and economic data...")
merged_data = pd.merge(
    precip_totals,
    avg_costs[['Year', 'Crop', 'Variable Costs', 'Fixed Costs']],
    on=['Year', 'Crop'], how='left'
)
merged_data = pd.merge(
    merged_data,
    avg_prices[['Year', 'Crop', 'Prices']],
    on=['Year', 'Crop'], how='left'
)
merged_data.set_index('Year', inplace=True)

# --- 9. Perform Bootstrapping ---
print(f"Generating {BOOTSTRAP_SAMPLES} bootstrap samples...")
np.random.seed(RANDOM_SEED)
study_years = merged_data.index.unique()

for i in range(1, BOOTSTRAP_SAMPLES + 1):
    bootstrapped_rows = []
    sampled_source_years = np.random.choice(study_years, size=len(study_years), replace=True)

    for target_year, source_year in zip(study_years, sampled_source_years):
        source_data = merged_data.loc[source_year].copy()
        source_data['Year'] = target_year
        source_data['Year Sampled From'] = source_year
        bootstrapped_rows.append(source_data)

    bootstrapped_df = pd.concat(bootstrapped_rows).reset_index(drop=True)
    cols = ['Year', 'Crop', 'Year Sampled From'] + [c for c in bootstrapped_df.columns if c not in ['Year', 'Crop', 'Year Sampled From']]
    bootstrapped_df = bootstrapped_df[cols]

    file_path = OUTPUT_DIR / f'bootstrapped_data_{i}.csv'
    bootstrapped_df.to_csv(file_path, index=False)

    if i % 10 == 0:
        print(f"  ...saved bootstrap sample {i}/{BOOTSTRAP_SAMPLES}")

print(f"\nBootstrap sampling complete. All files saved to: {OUTPUT_DIR}")

