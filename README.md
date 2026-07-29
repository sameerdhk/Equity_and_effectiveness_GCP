# Equity and Effectiveness of Groundwater Governance Policies

his repository contains the model code, analysis scripts, and small reference datasets supporting the study **“Equity and Effectiveness of Groundwater Governance Policies.”** The workflow compares six groundwater-governance policies—Uniform Pumping Restriction (UR), Fee-Based Pumping I (FB-I), Fee-Based Pumping II (FB-II), Priority-Based Pumping I (PR-I), Priority-Based Pumping II (PR-II), and Priority-Weighted Pumping Restriction (PWPR)—alongside a business-as-usual (BAU) baseline across 500 bootstrap climate–market realizations. Large generated model inputs and outputs are archived separately on Zenodo.

---

## Prerequisites

- Python 3.10+
- R (for generating the profit distribution plots)
- An academic license for the Gurobi optimizer

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/sameerdhk/Equity_and_effectiveness_GCP.git
cd Equity_and_effectiveness_GCP
```

2. Install the PyCHAMP package

The core agent-based model is contained within the `py_champ_package` folder. Install it in editable mode from the `scripts/py_champ_package` directory, which will also install all required Python dependencies listed in `pyproject.toml`.

```bash
pip install -e scripts/py_champ_packag
```

For more information on PyCHAMP, see the [PyCHAMP user manual](https://dises-pychamp.readthedocs.io/en/latest/index.html) and [“PyCHAMP: A crop-hydrological-agent modeling platform for groundwater management”](https://www.sciencedirect.com/science/article/pii/S1364815224002482).

---

## Repository naming conventions

Some internal code and folder names predate the final manuscript terminology:

- `FB` and `fb_runs` correspond to Fee-Based Pumping I (FB-I).
- `FB_CB` and `fb_cb_runs` correspond to Fee-Based Pumping II (FB-II), which recycles pumping-fee revenue through groundwater conservation payments.
- `R+PR` and `r_plus_pr_runs` correspond to Priority-Weighted Pumping Restriction (PWPR).

The final manuscript terminology is used in figures, tables, and reported results.

## Repository Structure

```bash
├── data/                                      # Source data used for bootstrapping and model-input preparation.
│   ├── costs_raw_for_bootstrap.csv            # Crop production-cost data.
│   ├── prec_D_cm_1996_2023.csv                # Crop- and season-specific precipitation data.
│   ├── prices_raw_for_bootstrap.csv           # Crop-price data.
│   └── SD6_grid_info.csv                      # Spatial, hydrogeologic, and historical land-use data.
├── inputs/                                    # Processed inputs required for model runs and supplementary analyses.
│   ├── bootstrap_samples/                     # 500 bootstrapped climate–market realizations.
│   ├── model_inputs/                          # Model-ready .pkl files for the 500 realizations.
│   └── supplementary/                         # Intermediate inputs used in supplementary analyses.
├── scripts/                                   # Executable scripts and model source code.
│   ├── analysis/                              # Data preparation and policy-simulation scripts, including HPC-compatible runs.
│   │   ├── 01_prepare_bootstrap_data.py       # Creates 500 bootstrapped climate–market realizations.
│   │   ├── 02_prepare_model_inputs.py         # Generates model-ready .pkl inputs for all policy simulations.
│   │   ├── 03_run_bs_scenario.py              # Runs the Business-as-Usual (BAU) simulations.
│   │   ├── 04_run_ur_scenarios.py             # Calibrates and runs Uniform Pumping Restriction (UR).
│   │   ├── 05_run_fb_scenarios.py             # Calibrates and runs Fee-Based Pumping I and II (FB-I and FB-II).
│   │   ├── 06_run_pr_I_scenarios.py           # Calibrates and runs Priority-Based Pumping I (PR-I).
│   │   ├── 07_run_pr_II_scenarios.py          # Calibrates and runs Priority-Based Pumping II (PR-II).
│   │   └── 08_run_r_plus_pr_scenarios.py      # Redistributes the UR regional budget by priority and runs PWPR.
│   ├── figures/                               # Scripts for preparing data and generating manuscript figures.
│   │   ├── 01_prepare_data_for_figures.py     # Aggregates model outputs into plotting datasets.
│   │   ├── 02_plot_aquifer_and_crop_properties.py # Generates aquifer and crop-ratio time-series figures.
│   │   ├── 03_plot_economic_outputs.py        # Generates profit and economic-water-productivity figures.
│   │   ├── 04_plot_profit_distribution.R      # Generates farmer profit-distribution figures.
│   │   ├── 05_plot_lorenz_curves.py           # Generates Lorenz curves and calculates Gini coefficients.
│   │   └── 06_plot_policy_restrictions.py     # Generates policy-restriction distributions.
│   ├── py_champ_package/                      # Core PyCHAMP agent-based model source code.
│   │   ├── py_champ/
│   │   │   ├── components/                    # Behavior, field, well, finance, aquifer, and optimization components.
│   │   │   ├── models/                        # Policy-specific model classes.
│   │   │   └── utility/                       # Scheduling, indicators, and supporting utilities.
│   │   └── pyproject.toml                     # Package configuration and Python dependencies.
│   └── supplementary/                         # Scripts for supplementary economic and input-stratified analyses.
│       ├── 01_extract_bootstrap_prices.py     # Extracts crop prices used in the FB-I marginal-revenue analysis.
│       ├── 02_energy_volume_relationship.py   # Evaluates the relationship between pumped volume and energy cost.
│       ├── 03_analyze_marginal_revenue_cost.py # Compares marginal irrigation revenue and pumping costs under FB-I.
│       ├── 04_analyze_marginal_yield_improvement.py # Calculates marginal yield improvement at farmer-year operating points.
│       ├── 05_classify_bootstrap_climate_market.py # Classifies bootstrap realizations by climate and market conditions.
│       ├── 06_input_stratified_sensitivity_analysis.py # Summarizes policy outcomes across climate and market strata.
│       └── 07_plot_input_stratified_figures.py # Generates input-stratified aquifer, economic, and equity figures.
├── outputs/                                   # Generated model results and figure-analysis products.
│   ├── baseline_runs/                         # BAU simulation results.
│   ├── fb_runs/                               # FB-I simulation results.
│   ├── fb_cb_runs/                            # FB-II simulation results with fee-revenue recycling.
│   ├── pr1_runs/                              # PR-I simulation results.
│   ├── pr2_runs/                              # PR-II simulation results.
│   ├── r_plus_pr_runs/                        # PWPR simulation results.
│   ├── ur_runs/                               # UR simulation results.
│   ├── data_for_figures/                      # Processed CSV files used for plotting.
│   └── figures/                               # Final manuscript figures.
│       └── supplementary/                     # Final Supporting Information figures.
├── .gitignore                                 # Excludes large generated inputs and outputs while retaining folder structure.
├── LICENSE                                    # Repository license.
└── README.md                                  # Project description and reproduction instructions.
```
---

## Reproducing the Results

The entire analysis workflow is managed through the scripts in `scripts/analysis/` and `scripts/figures/`. Run them in the following order from the project root: `GW_governance_policies`.

### Step 1: Prepare Bootstrap Data

Read the raw climate and market data and creates 500 unique bootstrap samples.

```bash
python scripts/analysis/01_prepare_bootstrap_data.py
```

### Step 2: Prepare Universal Model Inputs

Use the bootstrap samples and static data to generate the final, model-ready .pkl input files for all 500 scenarios.
```bash
python scripts/analysis/02_prepare_model_inputs.py
```
### Step 3: Run the Policy Scenarios

Run each of the following scripts sequentially. Each script will run the 500 bootstrap scenarios for its respective policy. These codes are also tailored for High-Performance Computing (HPC) runs. 
```bash
python scripts/analysis/03_run_bs_scenarios.py
python scripts/analysis/04_run_ur_scenarios.py
python scripts/analysis/05_run_fb_scenarios.py --policy_mode FB
python scripts/analysis/05_run_fb_scenarios.py --policy_mode FB_CB
python scripts/analysis/06_run_pr_I_scenarios.py
python scripts/analysis/07_run_pr_II_scenarios.py
python scripts/analysis/08_run_r_plus_pr_scenarios.py
```
### Step 4: Prepare Data for Figures

Aggregate all results from the outputs directory and creates the final CSV data files used for plotting.
```bash
python scripts/figures/01_prepare_data_for_figures.py
```

### Step 5: Generate Manuscript Figures

Generate the final figures and saves them to the outputs/figures directory.
```bash
# Aquifer & Crop Properties Figure
python scripts/figures/02_plot_aquifer_and_crop_properties.py

# Economic Outcomes Figure
python scripts/figures/03_plot_economic_outputs.py

# Profit Distribution Figure (requires R)
Rscript scripts/figures/04_plot_profit_distribution.R

# Lorenz Curve Figure
python scripts/figures/05_plot_lorenz_curves.py

# Policy Restrictions Summary Figure
python scripts/figures/06_plot_policy_restrictions.py
```

### Step 6: Run Supplementary Analysis

Perform supplementary checks on the economics of FB and yield sensitivity across all policies.
```bash
# Extract bootstrap prices for the FB scenario
python scripts/supplementary/01_extract_bootstrap_prices.py

# Verify linearity between pumped volume and energy costs
python scripts/supplementary/02_energy_volume_relationship.py

# Analyze marginal revenue vs. marginal pumping costs (FB scenario)
python scripts/supplementary/03_analyze_marginal_revenue_cost.py

# Compute marginal yield improvements across all policies
python scripts/supplementary/04_analyze_marginal_yield_improvement.py

# Classify bootstrap realizations by climate and market conditions
python scripts/supplementary/05_classify_bootstrap_climate_market.py

# Summarize policy outcomes across climate and market strata
python scripts/supplementary/06_input_stratified_sensitivity_analysis.py

# Plot input-stratified aquifer, economic, and equity outcomes
python scripts/supplementary/07_plot_input_stratified_figures.py
```

## Large model artifacts 
This repository contains the full analysis and model workflow (scripts, package code, and small reference datasets). The large `inputs/` and `outputs/` directories are **not tracked in GitHub** due to size constraints.

All `inputs/` and `outputs/` can be generated from scratch using the scripts in `scripts/` (see `scripts/analysis/` for input preparation and scenario runs). For transparency and verification, we provide a **reference snapshot** of the generated artifacts on Zenodo:

- **Zenodo archive:** `model_inputs_output.zip`  
- **link:** *https://doi.org/10.5281/zenodo.18089224*

This Zenodo snapshot can be used in two ways:

1. **Verification:** Run the workflow locally (or on HPC) to generate `inputs/` and `outputs/`, then compare against the Zenodo snapshot to verify reproducibility.
2. **Convenience:** Download the snapshot to inspect run-ready inputs, policy outputs, and figure products without running the full pipeline.

**To use the Zenodo snapshot directly:** download `model_inputs_output.zip` and unzip it into the repository root (the same folder as `README.md`), **preserving paths** as shown in the repository structure.

---
## Citation

If you use this code or data in your research, please cite:

Dhakal, S., Lin, C.-Y., & Marston, L. (2025). Equity and Effectiveness of Groundwater Governance Policies. [Journal Name].
```bash
@article{dhakal2025equity,
  title={Equity and Effectiveness of Groundwater Governance Policies},
  author={Dhakal, Sameer and Lin, Chung-Yi and Marston, Landon},
  journal={Journal Name},
  year={2025}
}
```

---
## License

This project is licensed under the terms specified in the LICENSE file.

---
## Acknowledgments

This research was supported by the National Science Foundation under Grant No. RISE-2108196 (“DISES: Toward resilient and adaptive community-driven management of groundwater-dependent agricultural systems”) and by the Foundation for Food and Agriculture Research under Grant No. FF-NIA19-0000000084. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors alone and do not necessarily reflect the views of the National Science Foundation or the Foundation for Food and Agriculture Research.
