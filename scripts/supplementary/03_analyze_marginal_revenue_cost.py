# -----------------------------------------------------------------------------
# 03_analyze_marginal_revenue_cost.py
#
# This script performs a supplementary economic analysis of the Fee-Based (FB)
# policy. It verifies whether the marginal cost of extracting a unit of
# groundwater is equal to the marginal revenue obtained from it.
#
# This analysis recalculates the theoretical
# marginal product of water based on the crop production functions and
# observed irrigation depths to ensure the fee induced the correct behavioral
# response (MR = MC).
#
# Steps:
#   1. Loads crop prices and the processed profit distribution data for the
#      Fee-Based (FB) policy.
#   2. Reconstructs the marginal product of water (bushels/cm) for each
#      farmer based on their specific crop and irrigation depth.
#   3. Calculates the Marginal Revenue (Price * MP) and Marginal Cost
#      (Energy Cost + Fee) per cubic meter of water.
#   4. Aggregates these metrics volumetrically to the bootstrap level.
#   5. Generates supplementary figures (scatter plots, boxplots, histograms)
#      comparing MR and MC to visualize economic efficiency.
# -----------------------------------------------------------------------------

# ================================================================
# Imports
# ================================================================
from pathlib import Path
import re
import io, csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ================================================================
# Paths (auto-resolve from script location)
# ================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# repo root: supplementary_analysis -> analysis -> scripts -> root
ROOT_DIR = SCRIPT_DIR.parents[1]

RESULTS_DIR = ROOT_DIR / "outputs" / "data_for_figures"
PRICES_CSV = ROOT_DIR / "inputs" / "supplementary" / "bootstrap_prices_2002_2022.csv"

FIG_DIR = ROOT_DIR / "outputs" / "figures" / "supplementary"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Configuration
# ================================================================
AREA_HA = 50.0
AREA_M2 = AREA_HA * 10_000.0
IRRIGATOR_MIN_DEPTH_CM = 1.0

# Production-function parameters per crop: [Y_max, AW_max, a, b, c, y_min]
CROP_PARAMS = {
    "corn":     [457.6316, 79.4827, -3.0517, 5.5043, -1.4820, 0.0193],
    "sorghum":  [184.4876, 61.8959, -1.8133, 3.2520, -0.4580, 0.6327],
    "soybeans": [145.3771, 72.9901, -2.5114, 4.4496, -0.9709, 0.0081],
    "wheat":    [130.3249, 78.3709, -2.0260, 3.2315, -0.2886, 0.2867],
}

# ================================================================
# Helper functions
# ================================================================
def read_csv_autodelim(path: Path) -> pd.DataFrame:
    """Read CSV with a simple delimiter sniff (comma/semicolon/tab/pipe)."""
    raw = path.read_bytes()
    head = raw[:2048].decode(errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","
    return pd.read_csv(io.BytesIO(raw), delimiter=delim)


def dYdAW_bushels_per_cm(AW_cm, crop):
    """Marginal product of water (bushels per cm) for the 50 ha farm."""
    if pd.isna(AW_cm) or crop not in CROP_PARAMS:
        return np.nan
    Y_max, AW_max, a, b, c, y_min = CROP_PARAMS[crop]
    aw_norm = AW_cm / AW_max
    y_norm = a * aw_norm**2 + b * aw_norm + c
    if y_norm < y_min:
        return 0.0
    dy_daw = 2 * a * aw_norm + b
    return (Y_max * 1e-4 * AREA_M2) * (dy_daw / AW_max)


def MR_per_m3(AW_cm, crop, price):
    """Marginal revenue ($/m³) at the operating point."""
    if pd.isna(price):
        return np.nan
    dY = dYdAW_bushels_per_cm(AW_cm, crop)
    return price * dY * (100.0 / AREA_M2)


def parse_bootstrap_id_from_name(fname: str) -> int | None:
    """Fallback: extract bootstrap id from '..._b_092.csv' if needed."""
    m = re.search(r"FB_b_(\d+)\.csv$", fname)
    return int(m.group(1)) if m else None

print(f"Reading prices from: {PRICES_CSV}")


# ================================================================
# Load prices
# ================================================================
prices = read_csv_autodelim(PRICES_CSV)
prices["crop"] = prices["crop"].astype(str).str.strip().str.lower()

# ================================================================
# Load FB-I files only
# ================================================================
FB_I_PATTERN = "profit_distribution_FB_I_b_*.csv"

fb_files = sorted(RESULTS_DIR.glob(FB_I_PATTERN))

print(f"Found {len(fb_files)} FB-I files in {RESULTS_DIR}")

if not fb_files:
    raise RuntimeError(
        f"No FB-I files found using pattern: {FB_I_PATTERN}"
    )

if any("FB_II" in f.name for f in fb_files):
    raise RuntimeError("FB-II files were mistakenly included.")

print("First three files loaded:")
for f in fb_files[:3]:
    print(f"  {f.name}")

print("Last three files loaded:")
for f in fb_files[-3:]:
    print(f"  {f.name}")

frames = []
for f in fb_files:
    df = read_csv_autodelim(f)

    # Ensure Bootstrap column exists; if not, parse from filename
    if "Bootstrap" not in df.columns:
        bs_id = parse_bootstrap_id_from_name(f.name)
        if bs_id is None:
            raise ValueError(f"Cannot infer Bootstrap from filename: {f}")
        df["Bootstrap"] = bs_id

    frames.append(df)

results = pd.concat(frames, ignore_index=True)

# Normalize crops
results["crop"] = results["crop"].astype(str).str.strip().str.lower()

# Ensure key numeric fields are numeric
for col in ["irr_depth", "pumping_fee", "energy_cost", "w", "year"]:
    if col in results.columns:
        results[col] = pd.to_numeric(results[col], errors="coerce")

# Filter irrigators
results = results[results["irr_depth"] >= IRRIGATOR_MIN_DEPTH_CM].copy()

# Merge prices by (Bootstrap, year, crop)
if "price_per_unit" not in prices.columns:
    raise ValueError("PRICES CSV must contain 'price_per_unit'")
results = results.merge(
    prices[["Bootstrap", "year", "crop", "price_per_unit"]],
    on=["Bootstrap", "year", "crop"],
    how="left",
)

# ================================================================
# Row-level MR and MC
# ================================================================
results["fee_dollars"] = results["pumping_fee"] * 1e4
results["energy_dollars"] = results["energy_cost"] * 1e4
results["volume_m3"] = (results["irr_depth"] / 100.0) * AREA_M2

results["MR_per_m3"] = results.apply(
    lambda r: MR_per_m3(r["w"], r["crop"], r["price_per_unit"]), axis=1
)
results["MC_per_m3"] = (results["fee_dollars"] + results["energy_dollars"]) / results["volume_m3"]


# ================================================================
# Bootstrap × Crop volumetric aggregation
# ================================================================
def summarize_bootstrap_crop(df: pd.DataFrame) -> pd.Series:
    V = df["volume_m3"].sum()
    if V <= 0:
        return pd.Series(
            {
                "sum_volume_m3": 0.0,
                "MC_per_m3": np.nan,
                "MR_per_m3": np.nan,
                "n_rows": len(df),
            }
        )

    cost_sum = (df["fee_dollars"] + df["energy_dollars"]).sum()
    MC = cost_sum / V  # ratio of sums
    MR = np.average(df["MR_per_m3"], weights=df["volume_m3"])  # volume-weighted mean

    return pd.Series(
        {
            "sum_volume_m3": V,
            "MC_per_m3": MC,
            "MR_per_m3": MR,
            "n_rows": len(df),
        }
    )

by_bootstrap_crop = (
    results.groupby(["Bootstrap", "crop"])
    .apply(summarize_bootstrap_crop)
    .reset_index()
)

# ================================================================
# Volume-weighted means across crops within each bootstrap
# ================================================================
rows = []
for bs, sub in by_bootstrap_crop.groupby("Bootstrap"):
    vols = sub["sum_volume_m3"].to_numpy()
    if vols.sum() > 0:
        w = vols / vols.sum()
    else:
        w = np.ones_like(vols) / len(vols)

    MRw = np.sum(sub["MR_per_m3"] * w)
    MCw = np.sum(sub["MC_per_m3"] * w)

    rows.append(
        {
            "Bootstrap": bs,
            "MR_weighted_mean_per_m3": MRw,
            "MC_weighted_mean_per_m3": MCw,
            "gap_MR_minus_MC": MRw - MCw,
            "n_crops": len(sub),
        }
    )

by_bootstrap_weighted = pd.DataFrame(rows)


# ================================================================
# Plotting
# ================================================================
plt.style.use("seaborn-v0_8-whitegrid")

# Setup figure with 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

# ----------------------------------------------------------------
# Panel (a): MR vs MC Scatter (Volume-Weighted)
# ----------------------------------------------------------------
ax = axes[0]

# Prepare Data
x = by_bootstrap_weighted["MC_weighted_mean_per_m3"]
y = by_bootstrap_weighted["MR_weighted_mean_per_m3"]
mask = x.notna() & y.notna()
X = x[mask].to_numpy().reshape(-1, 1)
Y = y[mask].to_numpy()

# ------------------------------------------------------------
# Statistical inference for the MR–MC regression
# ------------------------------------------------------------
# X currently has shape (n, 1); flatten it before adding a constant.
X_inference = sm.add_constant(X.ravel())

# HC3 provides heteroskedasticity-robust standard errors.
ols_model = sm.OLS(Y, X_inference).fit(cov_type="HC3")

slope_inference = ols_model.params[1]
intercept_inference = ols_model.params[0]

slope_ci_low, slope_ci_high = ols_model.conf_int(alpha=0.05)[1]

# H0: slope = 1
slope_test = ols_model.t_test("x1 = 1")

# Joint identity-line test:
# H0: intercept = 0 and slope = 1
identity_test = ols_model.f_test("const = 0, x1 = 1")

print("\nRegression inference:")
print(f"  Intercept: {intercept_inference:.6f}")
print(f"  Slope: {slope_inference:.6f}")
print(
    f"  95% CI for slope: "
    f"({slope_ci_low:.6f}, {slope_ci_high:.6f})"
)
print(f"  Test H0: slope = 1, p = {float(slope_test.pvalue):.6f}")
print(
    "  Joint test H0: intercept = 0 and slope = 1, "
    f"p = {float(identity_test.pvalue):.6f}"
)

# Linear Regression
reg = LinearRegression().fit(X, Y)
r2 = reg.score(X, Y)
slope = reg.coef_[0]
intercept = reg.intercept_

# Plotting
ax.scatter(X, Y, s=30, alpha=0.7, label="Bootstraps", color="steelblue")

# Regression Line
x_line = np.linspace(X.min(), X.max(), 100)
y_pred = reg.predict(x_line.reshape(-1, 1))
ax.plot(
    x_line, 
    y_pred, 
    "r-", 
    lw=2, 
    label=f"y={slope:.2f}x+{intercept:.2f}, R²={r2:.2f}"
)

# 45-degree Reference Line
ax.plot(
    [x_line.min(), x_line.max()],
    [x_line.min(), x_line.max()],
    "k--",
    lw=1,
    label="45° line"
)

# Formatting Panel (a)
ax.set_xlabel("Marginal Cost, MC ($/m³)")
ax.set_ylabel("Marginal Revenue, MR ($/m³)")
# ax.set_title("Bootstrap-level MR vs MC (volume-weighted)")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.1, 1.05, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold")

# ----------------------------------------------------------------
# Panel (b): Boxplot of MR, MC, and Gap
# ----------------------------------------------------------------
ax = axes[1]

# Prepare Data
box_data = [
    by_bootstrap_weighted["MR_weighted_mean_per_m3"].dropna(),
    by_bootstrap_weighted["MC_weighted_mean_per_m3"].dropna(),
    by_bootstrap_weighted["gap_MR_minus_MC"].dropna(),
]

# Plotting
ax.boxplot(
    box_data, 
    tick_labels=["MR", "MC", "MR − MC"], 
    showfliers=False,
    widths=0.5,
    patch_artist=False,  # Keep simple black/white style or set True for color
    medianprops=dict(color="darkorange", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)

# Formatting Panel (b)
ax.set_ylabel("$/m³")
# ax.set_title("Distribution across bootstraps")
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.text(-0.1, 1.05, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold")

# ----------------------------------------------------------------
# Final Layout and Save
# ----------------------------------------------------------------
plt.tight_layout()
save_path = FIG_DIR / "figS9_MR_MC.png"
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"Saved combined figure to: {save_path}")

# # ---------- 3. MR vs MC by crop (Bootstrap×Crop means) ----------
# plt.figure(figsize=(6, 5), dpi=600)
# for crop_name, sub in by_bootstrap_crop.groupby("crop"):
#     mx, mr = sub["MC_per_m3"], sub["MR_per_m3"]
#     plt.scatter(mx, mr, s=25, alpha=0.6, label=crop_name)
#     if len(sub) > 3:
#         reg = LinearRegression().fit(mx.values.reshape(-1, 1), mr.values)
#         r2 = reg.score(mx.values.reshape(-1, 1), mr.values)
#         plt.plot(mx, reg.predict(mx.values.reshape(-1, 1)), lw=1, label=f"{crop_name} (R²={r2:.2f})")

# plt.plot([0, max(mx.max(), mr.max())], [0, max(mx.max(), mr.max())], "k--", lw=1)
# plt.xlabel("MC ($/m³)")
# plt.ylabel("MR ($/m³)")
# plt.title("MR vs MC by Crop (Bootstrap×Crop means)")
# plt.legend(frameon=False)
# plt.savefig(FIG_DIR / "MR_vs_MC_by_crop.png", dpi=600)
# plt.close()

# # ---------- 4. Histogram of MR − MC gap ----------
# plt.figure(figsize=(6, 5), dpi=600)
# plt.hist(by_bootstrap_weighted["gap_MR_minus_MC"].dropna(), bins=20, color="gray")
# plt.xlabel("MR − MC ($/m³)")
# plt.ylabel("Count (bootstraps)")
# plt.title("Distribution of MR − MC gap (volume-weighted)")
# plt.savefig(FIG_DIR / "hist_gap.png", dpi=600)
# plt.close()


# ================================================================
# Summary statistics
# ================================================================
def q(arr, p):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.nan if arr.size == 0 else np.quantile(arr, p)


mr = by_bootstrap_weighted["MR_weighted_mean_per_m3"]
mc = by_bootstrap_weighted["MC_weighted_mean_per_m3"]
gap = by_bootstrap_weighted["gap_MR_minus_MC"]

print("\nSummary across bootstraps (volume-weighted):")
print(f"  MR median = {np.nanmedian(mr):.4g}, IQR = ({q(mr, 0.25):.4g}, {q(mr, 0.75):.4g})")
print(f"  MC median = {np.nanmedian(mc):.4g}, IQR = ({q(mc, 0.25):.4g}, {q(mc, 0.75):.4g})")
print(f"  Gap median = {np.nanmedian(gap):.4g}, IQR = ({q(gap, 0.25):.4g}, {q(gap, 0.75):.4g})")

