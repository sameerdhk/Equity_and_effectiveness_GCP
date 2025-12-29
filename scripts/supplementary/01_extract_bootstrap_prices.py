# -----------------------------------------------------------------------------
# 01_extract_bootstrap_prices.py
#
# This script consolidates raw crop price data from individual bootstrap
# sample files into a single, standardized CSV file for downstream economic
# analysis (specifically for Marginal Revenue calculations).
#
# It handles:
#   1. Directory scanning for all bootstrap files.
#   2. robust CSV parsing (detecting delimiters automatically).
#   3. Metadata extraction (inferring Bootstrap ID from filenames).
#   4. Standardization (normalizing column names, filtering years 2002-2022).
#   5. Deduplication (averaging prices if duplicate keys exist).
#
# Output:
#   - A single CSV containing columns: [Bootstrap, year, crop, price_per_unit]
# -----------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import re, io, csv

# ================================================================
# 1. Path Automation
# ================================================================
try:
    # Try to resolve the path relative to this script's location
    # Assumes structure: [ROOT]/scripts/supplementary/01_extract_bootstrap_prices.py
    # .parent    -> scripts/supplementary
    # .parents[1] -> scripts
    # .parents[2] -> [ROOT]
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for interactive sessions (e.g., Jupyter/Repl)
    PROJECT_ROOT = Path.cwd()

# Define input/output paths relative to the Project Root
IN_DIR = PROJECT_ROOT / "inputs" / "bootstrap_samples"
OUT_FILE = PROJECT_ROOT / "inputs" / "supplementary" / "bootstrap_prices_2002_2022.csv"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Scanning Input Dir: {IN_DIR}")
print(f"Target Output File: {OUT_FILE}")

# ================================================================
# 2. Configuration
# ================================================================
PATTERNS = ["bootstrapped_data_*.csv", "bootstrapped_data_*.scsv"]
YEAR_MIN, YEAR_MAX = 2002, 2022
REQUIRED_COLS = {"year", "crop", "prices"}  # expected in each file (case-insensitive)

# Optional alternate header mappings -> map to our expected lower-case names
COLMAP_FALLBACK = {
    # "yr": "year",
    # "commodity": "crop",
    # "price": "prices",
}

# ================================================================
# 3. Helper Functions
# ================================================================
def infer_bootstrap_from_name(path: Path) -> str | None:
    # looks for trailing _<digits> before the extension
    # e.g., bootstrapped_data_001.csv -> 001
    m = re.search(r"_([0-9]+)(?=\.[A-Za-z]+$)", path.name)
    return m.group(1) if m else None

def read_with_sniffer(p: Path) -> pd.DataFrame:
    """Read CSV/SCSV with delimiter auto-detection (comma/semicolon/tab/pipe)."""
    raw = p.read_bytes()
    # Read the first 2KB to sniff the dialect
    head = raw[:2048].decode(errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","  # fallback
    return pd.read_csv(io.BytesIO(raw), delimiter=delim)

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # lower-case & strip columns
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # apply any fallback renames -> "prices", "year", "crop"
    for src, dst in COLMAP_FALLBACK.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["year", "crop", "prices", "bootstrap"] if c in df.columns]
    df = df[keep].copy()
    if "crop" in df.columns:
        df["crop"] = df["crop"].astype(str).str.strip().str.lower()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "prices" in df.columns:
        df["prices"] = pd.to_numeric(df["prices"], errors="coerce")
    return df

# ================================================================
# 4. Main Execution
# ================================================================
if __name__ == "__main__":
    
    # --- Gather Files ---
    files = []
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {IN_DIR}")
        
    for pat in PATTERNS:
        files.extend(sorted(IN_DIR.glob(pat)))

    if not files:
        raise FileNotFoundError(f"No files found in {IN_DIR} matching {PATTERNS}")

    print(f"Found {len(files)} files. Processing...")

    # --- Stack Data ---
    frames = []
    for p in files:
        try:
            df = read_with_sniffer(p)
            df = normalize_headers(df)
            
            # Ensure required columns exist in some form
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                print(f"Skipping {p.name}: missing cols {missing}")
                continue

            df = standardize(df)

            # Ensure Bootstrap present; infer if missing/empty
            if "bootstrap" not in df.columns or df["bootstrap"].isna().all():
                bid = infer_bootstrap_from_name(p)
                if bid is None:
                    raise ValueError(f"Cannot infer Bootstrap id from filename: {p.name}")
                df["bootstrap"] = bid
            else:
                # normalize type to string
                df["bootstrap"] = df["bootstrap"].astype(str).str.strip()

            # Filter year window + non-null price
            df = df[(df["year"].between(YEAR_MIN, YEAR_MAX)) & df["prices"].notna()].copy()
            
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

    if not frames:
        raise ValueError("After filtering, no rows remained. Check year range and column names.")

    prices_all = pd.concat(frames, ignore_index=True)

    # --- Deduplicate ---
    # If duplicate (Bootstrap, year, crop) exist, average them
    dup_key = ["bootstrap", "year", "crop"]
    if prices_all.duplicated(dup_key).any():
        print("Duplicate entries found. Averaging prices...")
        prices_all = prices_all.groupby(dup_key, as_index=False, dropna=False)["prices"].mean()

    # --- Final Tidy Schema ---
    prices_all = prices_all.rename(columns={"prices": "price_per_unit", "bootstrap": "Bootstrap"})
    prices_all["Bootstrap"] = prices_all["Bootstrap"].astype(str)
    prices_all["year"] = prices_all["year"].astype(int)

    # --- Optional Validation ---
    issues = []
    for (b, c), g in prices_all.groupby(["Bootstrap", "crop"]):
        yrs = set(g["year"].tolist())
        missing = [y for y in range(YEAR_MIN, YEAR_MAX + 1) if y not in yrs]
        if missing:
            issues.append((b, c, missing))
    
    if issues:
        print("Note: Some (Bootstrap, crop) pairs are missing years (showing first 8):")
        for row in issues[:8]:
            print(f"  Bootstrap {row[0]}, Crop {row[1]}: Missing {len(row[2])} years")

    # --- Save ---
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices_all.to_csv(OUT_FILE, index=False)
    
    print("-" * 60)
    print(f"Successfully wrote: {OUT_FILE}")
    print(f"Total Rows: {len(prices_all)}")
    print(f"Unique Combinations: {prices_all.drop_duplicates(['Bootstrap','year','crop']).shape[0]}")
    print("-" * 60)