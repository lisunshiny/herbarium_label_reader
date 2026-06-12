import os
import re
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

# Scan experiments for *_comparison.csv, group by experiment name excluding the grounding flag,
# pair runs with preprocessors.grounding_dino.enabled=True vs ...=False, then
# for each pair find images where species was correct without grounding but incorrect with grounding,
# and save a CSV listing those cases.

def find_comparison_files(root_dir, required_substrings=None):
    comps = []
    if required_substrings is None:
        required_substrings = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn == "extracted_data_comparison.csv":
                full = os.path.join(dirpath, fn)
                # Only include files whose path contains all required substrings
                if all(substr in full for substr in required_substrings):
                    comps.append(full)
    return comps

def detect_grounding_flag(path):
    # Search the path for the specific substrings
    if "preprocessors.grounding_dino.enabled=True" in path:
        return True
    if "preprocessors.grounding_dino.enabled=False" in path:
        return False
    # Not found in path: return None
    return None

def pair_key_from_path(path):
    # Use the folder name (or entire path string) and remove the grounding flag token to make a pairing key.
    # Remove both possible tokens from the folder name(s).
    # Work on the full path but keep only a relative suffix (last 4 path parts) to keep keys manageable.
    parts = path.replace("\\", "/").split("/")
    # keep last up to 4 parts to form a key
    suffix = "/".join(parts[-4:])
    key = re.sub(r'preprocessors\.grounding_dino\.enabled=(?:True|False)', '', suffix)
    # collapse repeated separators and trim
    key = re.sub(r'[_\-\s]+', '_', key).strip("_")
    return key

def find_column(df_cols, *tokens):
    # find first column name containing all tokens (case-insensitive), tokens may be "extracted","Species"
    for col in df_cols:
        low = col.lower()
        if all(tok.lower() in low for tok in tokens):
            return col
    return None

def species_correct(extracted: str, gt: str) -> bool:
    # mimic compare_species_name with equality: compare first two words, case-insensitive
    try:
        ex = " ".join(str(extracted).split()[:2]).strip().lower()
        g = " ".join(str(gt).split()[:2]).strip().lower()
        return ex == g and ex != ""
    except Exception:
        return False

def analyze_pair(no_grounding_csv, grounding_csv):
    df_no = pd.read_csv(no_grounding_csv, keep_default_na=False)
    df_yes = pd.read_csv(grounding_csv, keep_default_na=False)

    # find source_image column (should be present)
    src_col_no = "source_image"
    src_col_yes = "source_image"

    # find extracted / ground_truth species columns (robust to underscores/spaces)
    ex_col_no = find_column(df_no.columns, "extracted", "species")
    gt_col_no = find_column(df_no.columns, "ground_truth", "species")
    ex_col_yes = find_column(df_yes.columns, "extracted", "species")
    gt_col_yes = find_column(df_yes.columns, "ground_truth", "species")

    if ex_col_no is None or gt_col_no is None or ex_col_yes is None or gt_col_yes is None:
        # If we couldn't find expected columns, return empty
        return pd.DataFrame(), {"error": "missing_species_columns", "no_cols": list(df_no.columns), "yes_cols": list(df_yes.columns)}

    # align by source_image
    df_no = df_no.rename(columns={src_col_no: "source_image", ex_col_no: "extracted_no", gt_col_no: "ground_truth"})
    df_yes = df_yes.rename(columns={src_col_yes: "source_image", ex_col_yes: "extracted_yes", gt_col_yes: "ground_truth_yes"})

    # Use ground_truth from the no-grounding file if available, else from yes file
    if "ground_truth" not in df_no.columns:
        # try to find in df_yes and copy
        gt_from_yes = find_column(df_yes.columns, "ground_truth", "species")
        if gt_from_yes:
            df_no["ground_truth"] = df_yes[gt_from_yes]
        else:
            df_no["ground_truth"] = ""

    # ensure columns exist
    if "extracted_no" not in df_no.columns:
        df_no["extracted_no"] = ""
    if "extracted_yes" not in df_yes.columns:
        df_yes["extracted_yes"] = ""

    merged = pd.merge(df_no[["source_image", "extracted_no", "ground_truth"]],
                      df_yes[["source_image", "extracted_yes"]], on="source_image", how="inner")

    # compute correctness
    merged["correct_no"] = merged.apply(lambda r: species_correct(r["extracted_no"], r["ground_truth"]), axis=1)
    merged["correct_yes"] = merged.apply(lambda r: species_correct(r["extracted_yes"], r["ground_truth"]), axis=1)

    # cases where no-grounding correct and grounding incorrect
    problem_cases = merged[(merged["correct_no"] == True) & (merged["correct_yes"] == False)].copy()

    return problem_cases, {"total_common": len(merged), "problems": len(problem_cases)}

def main(root_dir, output_csv=None):
    # filter experiments by these substrings (must all be present in path)
    required_substrings = ["batch_size=20", "data.handwritten.txt", "img_max_size=2048", "gemini"]

    files = find_comparison_files(root_dir, required_substrings=required_substrings)
    if not files:
        print("No comparison CSVs found under", root_dir, "matching filters:", required_substrings)
        return

    groups = defaultdict(dict)
    file_info = {}

    for p in files:
        flag = detect_grounding_flag(p)
        key = pair_key_from_path(p)
        if flag is True:
            groups[key]["grounding_true"] = p
        elif flag is False:
            groups[key]["grounding_false"] = p
        else:
            # store but mark unknown
            groups[key].setdefault("other", []).append(p)
        file_info[p] = {"flag": flag, "key": key}

    all_problem_rows = []
    summary = []

    for key, d in groups.items():
        if "grounding_true" in d and "grounding_false" in d:
            yes = d["grounding_true"]
            no = d["grounding_false"]
            problems_df, info = analyze_pair(no, yes)
            if isinstance(info, dict) and info.get("error"):
                summary.append({"pair_key": key, "no_file": no, "yes_file": yes, "status": "error", "info": info})
                continue
            # annotate rows
            if not problems_df.empty:
                problems_df["pair_key"] = key
                problems_df["no_file"] = no
                problems_df["yes_file"] = yes
                all_problem_rows.append(problems_df)
            summary.append({"pair_key": key, "no_file": no, "yes_file": yes, "total_common": info["total_common"], "problems": info["problems"]})
        else:
            # not a full pair
            summary.append({"pair_key": key, "status": "unpaired", "has_true": "grounding_true" in d, "has_false": "grounding_false" in d, "other_files": d.get("other", [])})

    if all_problem_rows:
        combined = pd.concat(all_problem_rows, ignore_index=True, sort=False)
        # select and rename columns for clarity
        out_cols = ["pair_key", "source_image", "ground_truth", "extracted_no", "extracted_yes", "no_file", "yes_file"]
        for c in out_cols:
            if c not in combined.columns:
                combined[c] = ""
        combined = combined[out_cols]
        out_path = output_csv or os.path.join(root_dir, "grounding_dino_problems.csv")
        combined.to_csv(out_path, index=False)
        print(f"Saved {len(combined)} problem cases to {out_path}")
    else:
        print("No problem cases found (no instances where no-grounding was correct and grounding incorrect).")

    # Write summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(root_dir, "grounding_dino_summary.csv") if output_csv is None else output_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Summary written to", summary_path)
    print(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experiments with and without grounding DINO using *_comparison.csv files.")
    parser.add_argument("experiments_root", help="Root folder containing experiment folders.")
    parser.add_argument("--output_csv", default=None, help="Path to save the problem cases CSV (optional).")
    args = parser.parse_args()
    main(args.experiments_root, args.output_csv)
