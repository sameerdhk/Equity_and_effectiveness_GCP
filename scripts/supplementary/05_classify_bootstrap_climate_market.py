# -----------------------------------------------------------------------------
# 05_classify_bootstrap_climate_market.py
#
# Purpose:
#   Classify each bootstrap realization by exogenous climate and market
#   conditions for supplementary robustness/sensitivity analysis.
#
# Input:
#   inputs/bootstrap_samples/bootstrapped_data_*.csv
#
# Output:
#   inputs/supplementary/bootstrap_climate_market_groups.csv
#
# Output columns:
#   Bootstrap
#   precip_index
#   precip_group          dry / middle / wet
#   market_return_index
#   market_return_group   low_return / middle / high_return
#
# Notes:
#   - This script is input-side only. It does not use ABM outcomes.
#   - The precipitation index is based on crop-standardized average precipitation.
#   - The market-return index is a reference net-return proxy based on:
#         revenue at crop-specific maximum yield - fixed cost - variable cost
#   - This is not simulated ABM profit because it does not include irrigation
#     decisions, pumping energy costs, policy fees, technology costs, or crop-
#     change costs.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import re
import io
import csv

import numpy as np
import pandas as pd


# ================================================================
# 1. Path automation
# ================================================================
try:
    # Assumes:
    #   ROOT/scripts/supplementary/04_classify_bootstrap_climate_market.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

IN_DIR = PROJECT_ROOT / "inputs" / "bootstrap_samples"
OUT_FILE = PROJECT_ROOT / "inputs" / "supplementary" / "bootstrap_climate_market_groups.csv"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Scanning Input Dir: {IN_DIR}")
print(f"Target Output File: {OUT_FILE}")


# ================================================================
# 2. Configuration
# ================================================================
PATTERNS = ["bootstrapped_data_*.csv", "bootstrapped_data_*.scsv"]

YEAR_MIN, YEAR_MAX = 2002, 2022

REQUIRED_COLS = {
    "year",
    "crop",
    "average_precipitation",
    "variable costs",
    "fixed costs",
    "prices",
}

# Crop production-function parameters:
# [Y_max, AW_max, a, b, c, y_min]
#
# These are the same values used in your marginal revenue / marginal cost script.
# Y_max is interpreted as bu/ha, consistent with the Field module's yield
# calculation:
#   y = normalized_yield * Y_max * field_area_ha * 1e-4
CROP_PARAMS = {
    "corn":     [457.6316, 79.4827, -3.0517, 5.5043, -1.4820, 0.0193],
    "sorghum":  [184.4876, 61.8959, -1.8133, 3.2520, -0.4580, 0.6327],
    "soybeans": [145.3771, 72.9901, -2.5114, 4.4496, -0.9709, 0.0081],
    "wheat":    [130.3249, 78.3709, -2.0260, 3.2315, -0.2886, 0.2867],
}

FIELD_AREA_HA = 50.0
HA_TO_AC = 2.471053814671653
FIELD_AREA_AC = FIELD_AREA_HA * HA_TO_AC


# ================================================================
# 3. Helper functions
# ================================================================
def infer_bootstrap_from_name(path: Path) -> str | None:
    """
    Infer bootstrap ID from filenames such as:
        bootstrapped_data_001.csv
        bootstrapped_data_1.csv
    """
    m = re.search(r"_([0-9]+)(?=\.[A-Za-z]+$)", path.name)
    return m.group(1) if m else None


def read_with_sniffer(path: Path) -> pd.DataFrame:
    """Read CSV/SCSV with delimiter auto-detection."""
    raw = path.read_bytes()
    head = raw[:2048].decode(errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","
    return pd.read_csv(io.BytesIO(raw), delimiter=delim)


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names."""
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


def normalize_crop(x: object) -> str:
    return str(x).strip().lower()


def zscore_by_crop(
    df: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Standardize a variable within each crop:

        z = (value - crop_mean) / crop_std

    This prevents crops with naturally larger precipitation seasons or larger
    dollar returns from dominating the bootstrap-level index.
    """
    df = df.copy()

    crop_mean = df.groupby("crop")[value_col].transform("mean")
    crop_std = df.groupby("crop")[value_col].transform("std")

    df[out_col] = np.where(
        crop_std > 0,
        (df[value_col] - crop_mean) / crop_std,
        0.0,
    )

    return df


def assign_quartile_group(
    values: pd.Series,
    low_label: str,
    high_label: str,
    middle_label: str = "middle",
) -> pd.Series:
    """
    Assign bottom quartile, middle 50%, and top quartile labels.
    """
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    return pd.Series(
        np.where(
            values <= q25,
            low_label,
            np.where(values >= q75, high_label, middle_label),
        ),
        index=values.index,
    )


def reference_market_return_1e4(row: pd.Series) -> float:
    """
    Calculate a crop-specific reference net-return proxy in the same broad
    monetary unit used by the ABM finance module: 1e4 dollars per 50-ha field.

    This mirrors the revenue - fixed cost - variable cost part of the finance
    calculation, but it is not actual simulated profit.

    Formula:

        reference return =
            price * Y_max * field_area_ha * 1e-4
            - variable_cost * field_area_ac * 1e-4
            - fixed_cost * field_area_ac * 1e-4

    where:
        price is $/bu
        Y_max is bu/ha
        fixed and variable costs are $/acre
        field area is fixed at 50 ha
    """
    crop = row["crop"]

    if crop not in CROP_PARAMS:
        return np.nan

    y_max_bu_per_ha = CROP_PARAMS[crop][0]

    revenue_1e4 = (
        row["prices"]
        * y_max_bu_per_ha
        * FIELD_AREA_HA
        * 1e-4
    )

    variable_cost_1e4 = row["variable costs"] * FIELD_AREA_AC * 1e-4
    fixed_cost_1e4 = row["fixed costs"] * FIELD_AREA_AC * 1e-4

    return revenue_1e4 - variable_cost_1e4 - fixed_cost_1e4


# ================================================================
# 4. Main execution
# ================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Gather files
    # ------------------------------------------------------------
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {IN_DIR}")

    files = []
    for pat in PATTERNS:
        files.extend(sorted(IN_DIR.glob(pat)))

    if not files:
        raise FileNotFoundError(f"No bootstrap files found in {IN_DIR} matching {PATTERNS}")

    print(f"Found {len(files)} bootstrap files. Processing...")

    # ------------------------------------------------------------
    # Read and stack bootstrap inputs
    # ------------------------------------------------------------
    frames = []

    for path in files:
        try:
            df = read_with_sniffer(path)
            df = normalize_headers(df)

            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                print(f"Skipping {path.name}: missing columns {missing}")
                continue

            bootstrap_id = infer_bootstrap_from_name(path)
            if bootstrap_id is None:
                raise ValueError(f"Cannot infer bootstrap id from filename: {path.name}")

            keep_cols = [
                "year",
                "crop",
                "average_precipitation",
                "variable costs",
                "fixed costs",
                "prices",
            ]

            df = df[keep_cols].copy()
            df["Bootstrap"] = bootstrap_id
            df["crop"] = df["crop"].map(normalize_crop)

            numeric_cols = [
                "year",
                "average_precipitation",
                "variable costs",
                "fixed costs",
                "prices",
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()

            if not df.empty:
                frames.append(df)

        except Exception as e:
            print(f"Error reading {path.name}: {e}")

    if not frames:
        raise ValueError("No usable bootstrap rows remained after filtering.")

    all_inputs = pd.concat(frames, ignore_index=True)

    print(f"Total rows after filtering {YEAR_MIN}-{YEAR_MAX}: {len(all_inputs):,}")
    print(f"Bootstrap count: {all_inputs['Bootstrap'].nunique()}")
    print(f"Crops found: {sorted(all_inputs['crop'].dropna().unique())}")

    # ------------------------------------------------------------
    # Build precipitation index
    # ------------------------------------------------------------
    all_inputs = zscore_by_crop(
        all_inputs,
        value_col="average_precipitation",
        out_col="precip_z",
    )

    precip_index = (
        all_inputs
        .groupby("Bootstrap", as_index=False)["precip_z"]
        .mean()
        .rename(columns={"precip_z": "precip_index"})
    )

    # ------------------------------------------------------------
    # Build market-return index
    # ------------------------------------------------------------
    all_inputs["reference_market_return_1e4"] = all_inputs.apply(
        reference_market_return_1e4,
        axis=1,
    )

    missing_return = all_inputs["reference_market_return_1e4"].isna().sum()
    if missing_return > 0:
        missing_crops = sorted(
            all_inputs.loc[
                all_inputs["reference_market_return_1e4"].isna(),
                "crop",
            ].dropna().unique()
        )
        raise ValueError(
            f"Reference market return could not be calculated for {missing_return} rows. "
            f"Missing crop parameters for: {missing_crops}"
        )

    all_inputs = zscore_by_crop(
        all_inputs,
        value_col="reference_market_return_1e4",
        out_col="market_return_z",
    )

    market_index = (
        all_inputs
        .groupby("Bootstrap", as_index=False)["market_return_z"]
        .mean()
        .rename(columns={"market_return_z": "market_return_index"})
    )

    # ------------------------------------------------------------
    # Combine and assign groups
    # ------------------------------------------------------------
    groups = precip_index.merge(market_index, on="Bootstrap", how="inner")

    groups["precip_group"] = assign_quartile_group(
        groups["precip_index"],
        low_label="dry",
        high_label="wet",
    )

    groups["market_return_group"] = assign_quartile_group(
        groups["market_return_index"],
        low_label="low_return",
        high_label="high_return",
    )

    groups["precip_percentile"] = groups["precip_index"].rank(pct=True)
    groups["market_return_percentile"] = groups["market_return_index"].rank(pct=True)

    # Normalize Bootstrap as string to match your existing supplementary files.
    groups["Bootstrap"] = groups["Bootstrap"].astype(str)

    groups = groups.sort_values(
        by="Bootstrap",
        key=lambda s: s.astype(int),
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUT_FILE, index=False)

    print("-" * 70)
    print(f"Successfully wrote: {OUT_FILE}")
    print(f"Rows: {len(groups)}")
    print("-" * 70)

    print("\nPrecipitation group counts:")
    print(groups["precip_group"].value_counts().sort_index())

    print("\nMarket-return group counts:")
    print(groups["market_return_group"].value_counts().sort_index())

    print("\nPreview:")
    print(groups.head(10).to_string(index=False))

    print("\nPrecipitation index summary:")
    print(groups["precip_index"].describe())

    print("\nMarket-return index summary:")
    print(groups["market_return_index"].describe())

    print("\nDone.")