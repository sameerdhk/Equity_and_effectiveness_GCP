# Equity and Effectiveness of Groundwater Governance Policies

This repository contains the complete codebase and data for the research paper **“Equity and Effectiveness of Groundwater Governance Policies.”** The project uses an agent-based model (ABM) to systematically compare five groundwater governance approaches—uniform pumping restrictions, volumetric pumping fees, two priority-based allocation systems, and a combined pumping restriction–priority system—to assess their impacts on aquifer sustainability and the distribution of farm-level profits.

---

## Prerequisites

- Python 3.10+
- R (for generating the profit distribution plots)
- An academic license for the Gurobi optimizer

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/sameerdhk/GW_governance_policies.git
cd GW_governance_policies
```

2. Install the PyCHAMP package

The core agent-based model is contained within the `py_champ_package` folder. Install it in editable mode from the `scripts/py_champ_package` directory, which will also install all required Python dependencies listed in `pyproject.toml`.

```bash
pip install -e scripts/py_champ_package
```

For more information on PyCHAMP, see the [PyCHAMP user manual](https://dises-pychamp.readthedocs.io/en/latest/index.html) and [“PyCHAMP: A crop-hydrological-agent modeling platform for groundwater management”](https://www.sciencedirect.com/science/article/pii/S1364815224002482).

---

## Repository Structure

```bash
├── data/                             # Input data files for bootstrapping and creation of model inputs.
│   ├── costs_raw_for_bootstrap.csv   # Crop cost data
│   ├── prec_D_cm_1996_2023.csv       # Precipitation data
│   ├── prices_raw_for_bootstrap.csv  # Crop price data
│   └── SD6_grid_info.csv             # Spatial, hydrogeologic, and historical land-use data.
├── inputs/                           # Processed data required for creating final model inputs and running models.
│   ├── bootstrap_samples/            # 500 bootstrapped climate/market scenarios.
│   └── model_inputs/                 # Final 500 .pkl files ready for model runs.
│   └── supplementary/                # Crop-specific prices required for marginal revenue analysis under FB (supplementary).
├── scripts/                          # All executable scripts for the analysis.
│   ├── analysis/                     # Scripts for data prep and running model scenarios. These scripts are also tailored for High-Performance Computing (HPC) runs.
│   │   ├── 01_prepare_bootstrap_data.py        # Creates 500 bootstrap samples of climate/market data.
│   │   ├── 02_prepare_all_inputs.py            # Generates universal .pkl input files for all policy model runs.
│   │   ├── 03_run_bs_scenarios.py              # Runs the Baseline (BAU) simulations.
│   │   ├── 04_run_ur_scenarios.py              # Iteratively finds the optimal water limit for aquifer stabilization and runs the Uniform Restriction (UR) policy.
│   │   ├── 05_run_fb_scenarios.py              # Iteratively finds the optimal volumetric fee for aquifer stabilization and runs the Fee-Based (FB) policy.
│   │   ├── 06_run_pr_I_scenarios.py            # Iteratively finds the optimal regional withdrawal threshold for aquifer stabilization and runs the Priority-Based I (PR-I) policy.
│   │   ├── 07_run_pr_II_scenarios.py           # Iteratively finds the optimal number of senior farmers getting water for aquifer stabilization and runs the Priority-Based II (PR-II) policy.
│   │   └── 08_run_pr_upr_scenarios.py          # Calculates the farmer-level water limit scaled by priority factors and runs the combined R+PR policy.
│   ├── figures/                      # Scripts for generating manuscript figures.
│   │   ├── 01_prepare_data_for_figures.py         # Aggregates all required model outputs into summary CSVs for plotting.
│   │   ├── 02_plot_aquifer_and_crop_properties.py # Generates the aquifer properties (withdrawal, saturated thickness, and change in water level) and crop-ratio time-series figure.
│   │   ├── 03_plot_economic_outputs.py            # Generates the average profit and water-use efficiency time-series figure.
│   │   ├── 04_plot_profit_distribution.R          # Generates the profit distribution ridgeline/boxplot figure (R script).
│   │   ├── 05_plot_lorenz_curves.py               # Generates the Lorenz curve and calculates the Gini coefficient for each policy.
│   │   └── 06_plot_policy_restrictions.py         # Generates the distribution of policy-specific restrictions across bootstraps.
│   ├── py_champ_package/             # The core agent-based model source code.
│   │   ├── py_champ/
│   │   │   ├── components/           # PyCHAMP components (Behavior, Field, Well, Finance, Aquifer, Optimization).
│   │   │   ├── models/               # Main MESA model classes for each policy.
│   │   │   └── utility/              # Helper functions (scheduler, indicators).
│   │   └── pyproject.toml            # Setup/installation file for the PyCHAMP package configuration.
│   └── supplementary/                # Scripts for supplementary analysis of marginal revenue and marginal pumping costs under the FB scenario, and marginal yield improvements across all policies.
│       ├── 01_extract_bootstrap_prices.py           # Extracts crop-specific prices under the FB scenario for each bootstrap run.
│       ├── 02_energy_volume_relationship.py         # Verifies the linearity between the volume of pumped water and energy costs.
│       ├── 03_analyze_marginal_revenue_cost.py      # Analyzes the equality of marginal revenue and marginal pumping costs under the FB scenario.
│       └── 04_analyze_marginal_yield_improvement.py # Computes marginal yield improvement at each farmer-year operating point using quadratic water–yield curves.
├── outputs/                          # All outputs from the model runs and analysis.
│   ├── baseline_runs/                # Excel results for the Baseline (BAU) scenario.
│   ├── fb_runs/                      # Excel results for the Fee-Based (FB) scenario.
│   ├── pr1_runs/                     # Excel results for the Priority-Based I (PR-I) scenario.
│   ├── pr2_runs/                     # Excel results for the Priority-Based II (PR-II) scenario.
│   ├── r_plus_pr_runs/               # Excel results for the R+PR scenario.
│   ├── ur_runs/                      # Excel results for the Uniform Restriction (UR) scenario.
│   ├── data_for_figures/             # Data (csv) cleaned for figure generation.
│   └── figures/                      # Final manuscript figures (.png).
│       └── supplementary/            # Final supplementary information figures.
└── README.md
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
python scripts/analysis/05_run_fb_scenarios.py
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
