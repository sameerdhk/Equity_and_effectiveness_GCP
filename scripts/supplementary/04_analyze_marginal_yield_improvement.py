# -----------------------------------------------------------------------------
# 04_analyze_marginal_yield_improvement.py
#
# Purpose (Option B; no prices, no costs)
#   Compute ex post marginal yield improvement (MPw = dY/dw) at each farmer-year
#   operating point using quadratic crop water–yield curves, then aggregate
#   following the a similar two-step structure as SI (S1.3):
#
#   (1) Filter: keep only "optimize" observations using column 'field_type_rn'
#       (these are irrigation-equipped fields), NOT irr_depth > 1.
#
#   (2) Weighting: replace irrigation-volume weighting with TOTAL APPLIED WATER
#       weighting, where applied water is 'w' (cm rainfall+irrigation).
#       That means the weight per observation is:
#           A_{i} = total applied water volume (m^3) = (w_i / 100) * AREA_M2
#
# Outputs
#   - outputs/data_for_figures/MPw_by_bootstrap_crop_policy_appliedW.csv
#   - outputs/data_for_figures/MPw_by_bootstrap_policy_appliedW.csv
#   - outputs/figures/supplementary/figSx_MPw_bu_per_cm_by_policy_appliedW.png
#
# Required columns in profit_distribution files
#   - crop
#   - w               (cm) total applied water = rainfall + irrigation
#   - field_type_rn   (string; must contain 'optimize' for irrigation-equipped)
#   - Policy          (optional; inferred from filename if absent)
#   - Bootstrap       (optional; inferred from filename if absent)
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
import io
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths (auto-resolve like your other scripts)
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

RESULTS_DIR = ROOT_DIR / "outputs" / "data_for_figures"
FIG_DIR = ROOT_DIR / "outputs" / "figures" / "supplementary"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_BC = RESULTS_DIR / "MPw_by_bootstrap_crop_policy_appliedW.csv"
OUT_CSV_B = RESULTS_DIR / "MPw_by_bootstrap_policy_appliedW.csv"
OUT_FIG = FIG_DIR / "figSx_MPw_bu_per_cm_by_policy_appliedW.png"


# =============================================================================
# Configuration
# =============================================================================
AREA_HA = 50.0
AREA_M2 = AREA_HA * 10_000.0

# Policy shorthands in your dataset
CANON_POLICIES = ["BAU", "UR", "FB", "PR_I", "PR_II", "R_PR"]

# Production-function parameters per crop: [Y_max, W_max, a, b, c, y_min]
# Y_max units: [1e-4 bu/m^2] (your clarification)
CROP_PARAMS = {
    "corn":     [457.6316, 79.4827, -3.0517, 5.5043, -1.4820, 0.0193],
    "sorghum":  [184.4876, 61.8959, -1.8133, 3.2520, -0.4580, 0.6327],
    "soybeans": [145.3771, 72.9901, -2.5114, 4.4496, -0.9709, 0.0081],
    "wheat":    [130.3249, 78.3709, -2.0260, 3.2315, -0.2886, 0.2867],
}


# =============================================================================
# Helpers
# =============================================================================
def read_csv_autodelim(path: Path) -> pd.DataFrame:
    """Read CSV with delimiter sniff (comma/semicolon/tab/pipe)."""
    raw = path.read_bytes()
    head = raw[:2048].decode(errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","
    return pd.read_csv(io.BytesIO(raw), delimiter=delim)


def normalize_crop(x) -> str:
    return str(x).strip().lower()


def normalize_policy(x) -> str:
    """Standardize common variants to your canonical shorthands."""
    t = str(x).strip().upper().replace(" ", "")
    t = t.replace("PR-I", "PR_I").replace("PR1", "PR_I").replace("PRI", "PR_I")
    t = t.replace("PR-II", "PR_II").replace("PR2", "PR_II").replace("PRII", "PR_II")
    t = t.replace("R+PR", "R_PR").replace("RPR", "R_PR").replace("R-PR", "R_PR")
    return t


def parse_bootstrap_from_filename(fname: str) -> int | None:
    m = re.search(r"_b_(\d+)", fname, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def infer_policy_from_filename(fname: str) -> str | None:
    # profit_distribution_<POLICY>_b_###.csv
    m = re.search(r"profit_distribution_(.+?)_b_\d+", fname, flags=re.IGNORECASE)
    return normalize_policy(m.group(1)) if m else None


def applied_water_volume_m3_from_w_cm(w_cm: float) -> float:
    """Convert total applied water depth (cm) over 50 ha to volume (m^3)."""
    if pd.isna(w_cm):
        return np.nan
    return (w_cm / 100.0) * AREA_M2


def dYdW_bushels_per_cm_farm(W_cm: float, crop: str) -> float:
    """
    Marginal product of total applied water (w) at operating point:
      MPw = dY/dw in bu/cm for the whole 50-ha farm.

    Uses normalized quadratic:
      y_norm = a*w_norm^2 + b*w_norm + c
      w_norm = W/W_max
      dy_norm/dw_norm = 2a*w_norm + b
      dY/dW = (Ymax_bushels_farm) * (dy_norm / W_max)

    Y_max is in [1e-4 bu/m^2]; convert to farm bushels by multiplying by 1e-4*A.
    """
    if pd.isna(W_cm) or crop not in CROP_PARAMS:
        return np.nan

    Y_max, W_max, a, b, c0, y_min = CROP_PARAMS[crop]

    w_norm = W_cm / W_max
    y_norm = a * w_norm**2 + b * w_norm + c0

    # Mirror floor behavior
    if y_norm < y_min:
        return 0.0

    dy_norm = 2.0 * a * w_norm + b

    # Convert Y_max to total bushels on the 50-ha field
    Ymax_bushels_farm = Y_max * 1e-4 * AREA_M2

    return Ymax_bushels_farm * (dy_norm / W_max)


def q(arr, p):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.nan if arr.size == 0 else np.quantile(arr, p)


# =============================================================================
# Load profit_distribution files
# =============================================================================
print(f"Reading profit distribution files from:\n  {RESULTS_DIR}")

files = sorted(RESULTS_DIR.glob("profit_distribution_*_b_*.csv"))
if not files:
    raise RuntimeError(f"No files found matching profit_distribution_*_b_*.csv in {RESULTS_DIR}")

frames = []
for f in files:
    df = read_csv_autodelim(f)

    # Policy
    if "Policy" in df.columns:
        df["Policy"] = df["Policy"].map(normalize_policy)
    else:
        pol = infer_policy_from_filename(f.name)
        if pol is None:
            raise ValueError(f"Cannot infer Policy from filename: {f.name}")
        df["Policy"] = pol

    # Bootstrap
    if "Bootstrap" not in df.columns:
        bs = parse_bootstrap_from_filename(f.name)
        if bs is None:
            raise ValueError(f"Cannot infer Bootstrap from filename: {f.name}")
        df["Bootstrap"] = bs

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# =============================================================================
# Validate & clean
# =============================================================================
required = ["crop", "w", "field_type_rn", "Policy", "Bootstrap"]
missing = [c for c in required if c not in data.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}. Found: {list(data.columns)}")

data["crop"] = data["crop"].map(normalize_crop)
data["Policy"] = data["Policy"].map(normalize_policy)

data["w"] = pd.to_numeric(data["w"], errors="coerce")
data["Bootstrap"] = pd.to_numeric(data["Bootstrap"], errors="coerce")
data["field_type_rn"] = data["field_type_rn"].astype(str).str.strip().str.lower()

# Keep only your canonical policies
data = data[data["Policy"].isin(CANON_POLICIES)].copy()

# (1) Filter to irrigation-equipped fields only
data = data[data["field_type_rn"] == "optimize"].copy()

# (2) Weight by TOTAL applied water (w)
data["appliedW_m3"] = data["w"].apply(applied_water_volume_m3_from_w_cm)

# =============================================================================
# Row-level MPw (bu/cm)
# =============================================================================
data["MPw_bu_per_cm"] = data.apply(lambda r: dYdW_bushels_per_cm_farm(r["w"], r["crop"]), axis=1)

# Keep negative MPw values if they occur (over-application beyond peak).
# If you prefer to drop negatives:
# data.loc[data["MPw_bu_per_cm"] < 0, "MPw_bu_per_cm"] = np.nan


# =============================================================================
# Step 1: Policy × Bootstrap × Crop total-applied-water-weighted means
#         (same structure as Eq. 12, but weights are appliedW instead of irrigation V)
# =============================================================================
def summarize_policy_bootstrap_crop(df: pd.DataFrame) -> pd.Series:
    Wtot = df["appliedW_m3"].sum()
    if not np.isfinite(Wtot) or Wtot <= 0:
        return pd.Series({"sum_appliedW_m3": 0.0, "MPw_bar_bu_per_cm": np.nan, "n_rows": len(df)})

    MPw_bar = np.average(df["MPw_bu_per_cm"], weights=df["appliedW_m3"])
    return pd.Series({"sum_appliedW_m3": Wtot, "MPw_bar_bu_per_cm": MPw_bar, "n_rows": len(df)})

by_policy_bs_crop = (
    data.groupby(["Policy", "Bootstrap", "crop"])
        .apply(summarize_policy_bootstrap_crop)
        .reset_index()
)

# by_policy_bs_crop.to_csv(OUT_CSV_BC, index=False)
# print(f"\nSaved Policy×Bootstrap×Crop table:\n  {OUT_CSV_BC}")


# =============================================================================
# Step 2: Crop-share weights within each (Policy, Bootstrap) and weighted sum
#         (same structure as Eqs. 13–14)
# =============================================================================
rows = []
for (pol, bs), sub in by_policy_bs_crop.groupby(["Policy", "Bootstrap"]):
    totalW = sub["sum_appliedW_m3"].sum()
    if not np.isfinite(totalW) or totalW <= 0:
        rows.append(
            {
                "Policy": pol,
                "Bootstrap": bs,
                "MPw_weighted_bu_per_cm": np.nan,
                "total_appliedW_m3": 0.0,
                "n_crops": len(sub),
            }
        )
        continue

    w_bc = sub["sum_appliedW_m3"] / totalW
    MPw_w = np.sum(w_bc * sub["MPw_bar_bu_per_cm"])

    rows.append(
        {
            "Policy": pol,
            "Bootstrap": bs,
            "MPw_weighted_bu_per_cm": MPw_w,
            "total_appliedW_m3": totalW,
            "n_crops": len(sub),
        }
    )

by_policy_bs = pd.DataFrame(rows)
# by_policy_bs.to_csv(OUT_CSV_B, index=False)
# print(f"\nSaved Policy×Bootstrap table:\n  {OUT_CSV_B}")


# =============================================================================
# Summary stats (across bootstraps)
# =============================================================================
print("\nSummary across bootstraps (MPw_weighted_bu_per_cm; weights=total applied water):")
for pol, sub in by_policy_bs.groupby("Policy"):
    mp = sub["MPw_weighted_bu_per_cm"].to_numpy()
    print(
        f"  {pol:5s} | median={np.nanmedian(mp):.4g}, "
        f"IQR=({q(mp,0.25):.4g},{q(mp,0.75):.4g}), n_boot={len(sub)}"
    )


# =============================================================================
# Figure: Boxplot across bootstraps (bu/cm)
# =============================================================================
plt.style.use("seaborn-v0_8-whitegrid")

policy_order = [p for p in CANON_POLICIES if p in by_policy_bs["Policy"].unique()]
box_data = [
    by_policy_bs.loc[by_policy_bs["Policy"] == p, "MPw_weighted_bu_per_cm"].dropna()
    for p in policy_order
]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=600)
ax.boxplot(
    box_data,
    tick_labels=policy_order,
    showfliers=False,
    widths=0.55,
    medianprops=dict(color="darkorange", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
ax.set_xlabel("Policy")
ax.set_ylabel("Applied-water-weighted marginal yield improvement, MPw (bu / cm)")
ax.set_title("Ex post marginal yield improvement at operating point")
ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

fig.tight_layout()
# fig.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
# plt.close(fig)

print(f"\nSaved figure:\n  {OUT_FIG}")
print("\nDone.")

# =============================================================================
# Diagnostic: Where each policy operates on the water–yield curve (aw = w / AW_max)
# =============================================================================

# 1) Compute aw ratio (crop-specific)
def get_aw_max(crop: str) -> float:
    if crop not in CROP_PARAMS:
        return np.nan
    return float(CROP_PARAMS[crop][1])  # AW_max

data["AW_max"] = data["crop"].apply(get_aw_max)
data["aw_ratio"] = data["w"] / data["AW_max"]

# Optional: flag out-of-range aw (should generally be within [0, 1] if w is bounded)
data["aw_out_of_range"] = (data["aw_ratio"] < 0) | (data["aw_ratio"] > 1.05)

# out_diag_csv = RESULTS_DIR / "aw_ratio_rowlevel_policy_crop.csv"
# data[["Policy", "Bootstrap", "crop", "w", "AW_max", "aw_ratio", "appliedW_m3", "aw_out_of_range"]].to_csv(
#     out_diag_csv, index=False
# )
# print(f"\nSaved row-level aw diagnostics:\n  {out_diag_csv}")

# 2) Bootstrap × Crop × Policy applied-water–weighted mean aw
def summarize_aw_policy_bootstrap_crop(df: pd.DataFrame) -> pd.Series:
    Wtot = df["appliedW_m3"].sum()
    if not np.isfinite(Wtot) or Wtot <= 0:
        return pd.Series({"sum_appliedW_m3": 0.0, "aw_bar": np.nan, "n_rows": len(df)})

    aw_bar = np.average(df["aw_ratio"], weights=df["appliedW_m3"])
    return pd.Series({"sum_appliedW_m3": Wtot, "aw_bar": aw_bar, "n_rows": len(df)})

aw_by_policy_bs_crop = (
    data.groupby(["Policy", "Bootstrap", "crop"])
        .apply(summarize_aw_policy_bootstrap_crop)
        .reset_index()
)

# out_aw_bc = RESULTS_DIR / "aw_ratio_by_bootstrap_crop_policy_appliedW.csv"
# aw_by_policy_bs_crop.to_csv(out_aw_bc, index=False)
# print(f"Saved Bootstrap×Crop aw table:\n  {out_aw_bc}")

# 3) Bootstrap-level aw (across crops), same crop-share weighting structure
rows_aw = []
for (pol, bs), sub in aw_by_policy_bs_crop.groupby(["Policy", "Bootstrap"]):
    totalW = sub["sum_appliedW_m3"].sum()
    if not np.isfinite(totalW) or totalW <= 0:
        rows_aw.append({"Policy": pol, "Bootstrap": bs, "aw_weighted": np.nan, "total_appliedW_m3": 0.0})
        continue

    w_bc = sub["sum_appliedW_m3"] / totalW
    aw_w = np.sum(w_bc * sub["aw_bar"])

    rows_aw.append({"Policy": pol, "Bootstrap": bs, "aw_weighted": aw_w, "total_appliedW_m3": totalW})

aw_by_policy_bs = pd.DataFrame(rows_aw)

# out_aw_b = RESULTS_DIR / "aw_ratio_by_bootstrap_policy_appliedW.csv"
# aw_by_policy_bs.to_csv(out_aw_b, index=False)
# print(f"Saved bootstrap-level aw table:\n  {out_aw_b}")

# 4) Figure: Boxplot of bootstrap-level aw by policy
plt.style.use("seaborn-v0_8-whitegrid")

policy_order = [p for p in CANON_POLICIES if p in aw_by_policy_bs["Policy"].unique()]
box_data_aw = [
    aw_by_policy_bs.loc[aw_by_policy_bs["Policy"] == p, "aw_weighted"].dropna()
    for p in policy_order
]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=600)
ax.boxplot(
    box_data_aw,
    tick_labels=policy_order,
    showfliers=False,
    widths=0.55,
    medianprops=dict(color="darkorange", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
ax.set_xlabel("Policy")
ax.set_ylabel("Applied-water–weighted operating point, aw = w / AW_max (ratio)")
ax.set_title("Where policies operate on the water–yield curve")
ax.axhline(1.0, linestyle="--", linewidth=1.0, color="k")
ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

fig.tight_layout()
# out_aw_fig = FIG_DIR / "figSx_aw_ratio_by_policy_appliedW.png"
# fig.savefig(out_aw_fig, dpi=600, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved figure:\n  {out_aw_fig}")

# 5) Quick validity checks (printed)
#    A) Too many out-of-range aw suggests w/AW_max mismatch or overwatering
oor = data["aw_out_of_range"].mean()
print(f"\naw out-of-range share (aw<0 or aw>1.05): {oor:.3%}")

#    B) Correlation check: higher aw should generally correspond to lower MPw (concave curves)
tmp = data[["aw_ratio", "MPw_bu_per_cm"]].dropna()
if len(tmp) > 10:
    corr = np.corrcoef(tmp["aw_ratio"].to_numpy(), tmp["MPw_bu_per_cm"].to_numpy())[0, 1]
    print(f"Row-level correlation corr(aw, MPw): {corr:.3f}  (should be negative if concave)")
else:
    print("Not enough non-NA rows to compute corr(aw, MPw).")
