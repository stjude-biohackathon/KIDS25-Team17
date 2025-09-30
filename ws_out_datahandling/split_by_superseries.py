# split_by_superseries.py
import argparse, os, re
import pandas as pd

def sanitize_sheet_name(name: str, used: set) -> str:
    # Excel sheet name rules
    name = str(name) if pd.notna(name) and str(name).strip() else "NA"
    bad = r'[:\\/?*\[\]]'
    safe = re.sub(bad, "_", name)
    safe = safe[:31] or "NA"
    base = safe
    i = 1
    while safe in used:
        suffix = f"_{i}"
        safe = (base[:31-len(suffix)] + suffix) if len(base) + len(suffix) > 31 else base + suffix
        i += 1
    used.add(safe)
    return safe

def sanitize_filename(name: str) -> str:
    name = str(name) if pd.notna(name) and str(name).strip() else "NA"
    safe = re.sub(r'[^A-Za-z0-9._-]+', "_", name).strip("_")
    return safe or "NA"

def main(input_csv, excel_out, csv_dir=None):
    df = pd.read_csv(input_csv)
    # If SuperSeries has blanks, fall back to Series
    if "SuperSeries" not in df.columns:
        raise ValueError("Column 'SuperSeries' not found.")
    if "Series" in df.columns:
        df["SuperSeries"] = df["SuperSeries"].fillna(df["Series"])
    else:
        df["SuperSeries"] = df["SuperSeries"].fillna("NA")

    # Sort for nicer grouping
    df = df.sort_values(["SuperSeries", "Series"] if "Series" in df.columns else ["SuperSeries"]).reset_index(drop=True)

    # Summary counts
    counts = df["SuperSeries"].value_counts(dropna=False).rename_axis("SuperSeries").reset_index(name="Rows")

    # Write Excel
    os.makedirs(os.path.dirname(excel_out) or ".", exist_ok=True)
    used_names = set()
    with pd.ExcelWriter(excel_out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        counts.to_excel(writer, sheet_name="Summary", index=False)

        for ss, sub in df.groupby("SuperSeries", dropna=False):
            sheet = sanitize_sheet_name(ss, used_names)
            sub.to_excel(writer, sheet_name=sheet, index=False)

    # Optional per-group CSVs
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        for ss, sub in df.groupby("SuperSeries", dropna=False):
            fname = sanitize_filename(ss) + ".csv"
            sub.to_csv(os.path.join(csv_dir, fname), index=False)

    print(f"Excel written: {excel_out}")
    if csv_dir:
        print(f"Per-SuperSeries CSVs in: {csv_dir}")
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split GEO CSV by SuperSeries into Excel sheets and optional per-group CSVs.")
    p.add_argument("--input", required=True, help="Path to geo_webscrap.csv")
    p.add_argument("--excel", required=True, help="Output Excel path, e.g., geo_by_superseries.xlsx")
    p.add_argument("--csv-dir", default=None, help="Optional directory to write one CSV per SuperSeries")
    args = p.parse_args()
    main(args.input, args.excel, args.csv_dir)

