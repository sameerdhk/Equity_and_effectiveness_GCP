# -----------------------------------------------------------------------------
# 02_energy_volume_relationship.py
#
# This script evaluates the "Zero-Intercept" hypothesis for the
# Energy-Volume relationship. It tests whether the Marginal Cost (MC) of
# pumping can be accurately approximated as a simple ratio of Total Cost to
# Total Volume (MC = C/V), rather than requiring a regression with an intercept.
#
# Steps:
#   1. Loads Fee-Based (FB) profit distribution data.
#   2. Fits a Global Linear Model forcing the intercept to zero (E = b * V).
#   3. Compares the Regression Slope (b) against the volumetric average
#      Marginal Cost (ΣC / ΣV) to ensure the estimators converge.
#   4. Performs this validation iteratively across all bootstraps to quantify
#      the goodness-of-fit (R²) distribution.
#   5. Generates a 2x2 diagnostic panel:
#       (a) Global Hexbin plot overlaid with the zero-intercept fit.
#       (b) Residual analysis to detect non-linear structural bias.
#       (c/d) Histograms and Boxplots of R² to prove consistency across
#             climate scenarios.
# -----------------------------------------------------------------------------

# ------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm
import re
import statsmodels.api as sm

# ------------------------------------------------------------
# 2. Paths and constants
# ------------------------------------------------------------
# Auto-resolve paths relative to this script's location
try:
    # Assuming script is in: root/scripts/analysis/ or similar depth
    # Adjust .parents[X] based on exactly where you save this file
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parents[1] # Go up to project root
except NameError:
    # Fallback for interactive/notebook usage
    ROOT_DIR = Path.cwd()

# Point to the directory containing the many CSV files
RESULTS_DIR = ROOT_DIR / "outputs" / "data_for_figures"
OUT_DIR = ROOT_DIR / "outputs" / "figures" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA_HECTARES_PER_FARM = 50.0
AREA_M2_PER_FARM = AREA_HECTARES_PER_FARM * 10_000.0
IRRIGATOR_MIN_DEPTH_CM = 1.0 

# ------------------------------------------------------------
# 3. Load and prepare data (Multiple Files Logic)
# ------------------------------------------------------------
print(f"Scanning for FB results in: {RESULTS_DIR}")

# 1. Define helper to parse Bootstrap ID from filename (e.g., ..._b_001.csv)
def parse_bootstrap_id_from_name(fname: str):
    m = re.search(r"_b_(\d+)", fname)
    return int(m.group(1)) if m else None

# 2. Find all matching files
fb_files = sorted(RESULTS_DIR.glob("profit_distribution_FB_I_b_*.csv"))

if not fb_files:
    raise RuntimeError(f"No FB-I profit distribution files found in {RESULTS_DIR}")

print(f"Found {len(fb_files)} files. Loading...")

# 3. Load and combine
frames = []
for f in fb_files:
    # Standard pd.read_csv is sufficient since we generated these files ourselves
    temp_df = pd.read_csv(f)

    # Ensure Bootstrap column exists; if not, parse from filename
    if "Bootstrap" not in temp_df.columns:
        bs_id = parse_bootstrap_id_from_name(f.name)
        if bs_id is None:
            print(f"Warning: Could not determine Bootstrap ID for {f.name}, skipping.")
            continue
        temp_df["Bootstrap"] = bs_id
    
    frames.append(temp_df)

# Concatenate all frames into the main DataFrame 'df'
df = pd.concat(frames, ignore_index=True)
print(f"Total records loaded: {len(df)}")

# ------------------------------------------------------------
# Data Cleaning (Continue with existing logic)
# ------------------------------------------------------------
required_cols = ["Bootstrap", "irr_depth", "energy_cost"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in combined data: {missing}")

df["irr_depth"] = pd.to_numeric(df["irr_depth"], errors="coerce")
df["energy_cost"] = pd.to_numeric(df["energy_cost"], errors="coerce")

# Filter to irrigators
df = df[(df["irr_depth"] >= IRRIGATOR_MIN_DEPTH_CM)].copy()
df = df.dropna(subset=["irr_depth", "energy_cost", "Bootstrap"])

# Compute pumped volume and energy cost in dollars
df["volume_m3"] = (df["irr_depth"] / 100.0) * AREA_M2_PER_FARM
df["energy_dollars"] = df["energy_cost"] * 1.0e4

# Drop any zero-volume rows
df = df[df["volume_m3"] > 0].copy()

# ------------------------------------------------------------
# 4. Global zero-intercept regression (E = b * V)
# ------------------------------------------------------------
X_global = df[["volume_m3"]].to_numpy()
y_global = df["energy_dollars"].to_numpy()

reg_global = LinearRegression(fit_intercept=False)
reg_global.fit(X_global, y_global)
slope_global = reg_global.coef_[0]
yhat_global = reg_global.predict(X_global)
r2_global = reg_global.score(X_global, y_global)

# Also compute global MC = ΣC / ΣV for comparison (should be ~ slope_global)
total_cost = y_global.sum()
total_vol = X_global.sum()
mc_global = total_cost / total_vol

print("\n=== Global zero-intercept regression E = b V ===")
print(f"Slope b (regression MC):     {slope_global:.6f} $/m³")
print(f"Global MC (ΣC / ΣV):         {mc_global:.6f} $/m³")
print(f"Difference |b - ΣC/ΣV|:      {abs(slope_global - mc_global):.6e}")
print(f"R² (global fit):             {r2_global:.4f}")
print(f"Number of observations:      {len(df)}")

# Global residuals
residuals_global = y_global - yhat_global

# ------------------------------------------------------------
# 4b. Unrestricted regression to test for a nonzero intercept
#     E = alpha + beta * V
# ------------------------------------------------------------
V = df["volume_m3"].to_numpy()
E = df["energy_dollars"].to_numpy()

# Add a constant so the intercept is estimated rather than forced to zero
X_unrestricted = sm.add_constant(V)

# Cluster-robust standard errors account for observations belonging
# to the same bootstrap realization
reg_unrestricted = sm.OLS(E, X_unrestricted).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Bootstrap"].to_numpy()}
)

intercept_free = reg_unrestricted.params[0]
slope_free = reg_unrestricted.params[1]

ci = reg_unrestricted.conf_int(alpha=0.05)
intercept_ci_low, intercept_ci_high = ci[0]
slope_ci_low, slope_ci_high = ci[1]

intercept_p = reg_unrestricted.pvalues[0]
slope_p = reg_unrestricted.pvalues[1]

print("\n=== Unrestricted regression E = alpha + beta V ===")
print(f"Intercept alpha:             ${intercept_free:.6f}")
print(
    f"95% CI for intercept:        "
    f"(${intercept_ci_low:.6f}, ${intercept_ci_high:.6f})"
)
print(f"Test H0: intercept = 0:      p = {intercept_p:.6f}")
print(f"Slope beta:                  {slope_free:.6f} $/m³")
print(
    f"95% CI for slope:            "
    f"({slope_ci_low:.6f}, {slope_ci_high:.6f}) $/m³"
)

print("\nObserved pumping-volume range:")
print(f"Minimum: {V.min():,.2f} m³")
print(f"Maximum: {V.max():,.2f} m³")

# ------------------------------------------------------------
# 5. Per-bootstrap zero-intercept regressions
# ------------------------------------------------------------
summary = []

for bs_id, sub in df.groupby("Bootstrap"):
    sub = sub.dropna(subset=["volume_m3", "energy_dollars"])
    if len(sub) < 5:
        continue

    Xb = sub[["volume_m3"]].to_numpy()
    yb = sub["energy_dollars"].to_numpy()

    # skip if essentially no variation
    if np.allclose(Xb, Xb[0]) or np.allclose(yb, yb[0]):
        continue

    reg_b = LinearRegression(fit_intercept=False)
    reg_b.fit(Xb, yb)
    slope_b = reg_b.coef_[0]
    yhat_b = reg_b.predict(Xb)
    r2_b = reg_b.score(Xb, yb)

    # bootstrap-level volumetric MC
    mc_b = yb.sum() / Xb.sum()

    summary.append(
        {
            "Bootstrap": bs_id,
            "slope": slope_b,
            "r2": r2_b,
            "mc_volumetric": mc_b,
            "abs_diff_slope_minus_mc": abs(slope_b - mc_b),
            "n_obs": len(sub),
        }
    )

summary_df = pd.DataFrame(summary).sort_values("Bootstrap")
# summary_csv = OUT_DIR / "energy_volume_summary_zero_intercept.csv"
# summary_df.to_csv(summary_csv, index=False)
# print("\nPer-bootstrap summary written to:", summary_csv)

# ------------------------------------------------------------
# 6. 2×2 diagnostic figure
# ------------------------------------------------------------
sns.set_style("whitegrid")

# --- CONFIG: FONT SIZES ---
LABEL_FS = 16  # Font size for Axis Labels (x and y)
TICK_FS = 14   # Font size for Tick Marks (numbers)
# --------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=600)

# ---- (a) All points + global zero-intercept regression ----
ax = axes[0, 0]
# Use hexbin or scatter; hexbin is nicer for huge N.
hb = ax.hexbin(
    df["volume_m3"],
    df["energy_dollars"],
    gridsize=50,
    cmap="Blues",
    mincnt=1,
    norm=LogNorm(),
)
# Update colorbar font sizes
cb = fig.colorbar(hb, ax=ax)
cb.set_label("Count", fontsize=LABEL_FS)
cb.ax.tick_params(labelsize=TICK_FS)

x_line = np.linspace(df["volume_m3"].min(), df["volume_m3"].max(), 200).reshape(-1, 1)
y_line = reg_global.predict(x_line)

ax.plot(
    x_line,
    y_line,
    color="red",
    lw=2,
    label=f"E = {slope_global:.4f}·V\nR² = {r2_global:.3f}",
)
ax.set_xlabel("Pumped Volume (m³)", fontsize=LABEL_FS)
ax.set_ylabel("Energy Cost ($)", fontsize=LABEL_FS)
ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax.legend(loc="lower right", frameon=False, fontsize=TICK_FS)
ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- (b) Residuals vs fitted (global) ----
ax = axes[0, 1]
ax.scatter(yhat_global, residuals_global, s=5, alpha=0.3)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Fitted Energy Cost ($)", fontsize=LABEL_FS)
ax.set_ylabel("Residual (Actual − Fitted)", fontsize=LABEL_FS)
ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- (c) Histogram of R² across bootstraps ----
ax = axes[1, 0]
sns.histplot(summary_df["r2"], bins=20, kde=True, color="teal", edgecolor="black", ax=ax)
ax.set_xlabel("Linearly fitted R² for each bootstrap", fontsize=LABEL_FS)
ax.set_ylabel("Number of bootstraps", fontsize=LABEL_FS)
ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax.text(0.02, 0.95, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# ---- (d) Boxplot of R² across bootstraps ----
ax = axes[1, 1]
ax.boxplot(summary_df["r2"].values, vert=True, showfliers=False)
ax.set_xticks([1])
# Fontsize for the specific boxplot label
ax.set_xticklabels(["Box Plot for R² across Bootstraps"], fontsize=TICK_FS)
ax.set_ylabel("R²", fontsize=LABEL_FS)
ax.tick_params(axis='y', which='major', labelsize=TICK_FS)
ax.text(0.02, 0.95, "(d)", transform=ax.transAxes, fontsize=14, fontweight="bold")

plt.tight_layout()
fig_path = OUT_DIR / "figS8_energy_volume_relationship.png"
plt.savefig(fig_path, dpi=600, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 7. Console summary of per-bootstrap stats
# ------------------------------------------------------------
def q(series, p):
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.quantile(arr, p) if arr.size > 0 else np.nan

print("\n--- Per-bootstrap statistics (zero-intercept) ---")
print(f"Bootstraps analyzed: {len(summary_df)}")
print(f"Median R²: {summary_df['r2'].median():.3f}")
print(f"IQR R²: ({q(summary_df['r2'],0.25):.3f}, {q(summary_df['r2'],0.75):.3f})")
print(f"Median |slope − ΣC/ΣV|: {summary_df['abs_diff_slope_minus_mc'].median():.4e}")