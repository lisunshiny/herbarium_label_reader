import os
import re
import argparse
import numpy as np
import pandas as pd
import Levenshtein as lev
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from typing import Callable, Dict
from difflib import SequenceMatcher
from functools import partial

# add safe import for linregress
try:
    from scipy.stats import linregress, pearsonr
except Exception:
    linregress = None
    pearsonr = None

regression_mode = "linear2" # "glm" # or "linear" "interaction"
compute_SEM = True

# Custom scaled logit to values -1/1 for glm
class ScaledLogit(sm.families.links.Link):
    def _clean(self, p):
        """
        Clip logistic values to range (eps, 1-eps)

        Parameters
        ----------
        p : array_like
            Probabilities

        Returns
        -------
        pclip : ndarray
            Clipped probabilities
        """
        return np.clip(p, -1 + 1e-7, 1. - 1e-7)

    def __call__(self, p):
        p = self._clean(p)
        #return np.log(p / (1. - p))
        return np.log(2. / (p + 1.) - 1.)

    def inverse(self, z):
        z = np.asarray(z)
        t = np.exp(-z)
        return 2 * (1. / (1. + t) - 0.5)

    def deriv(self, p):
        p = self._clean(p)
        #return 1. / (p * (1 - p))
        return 2 / (p ** 2 - 1)

    def inverse_deriv(self, z):
        t = np.exp(z)
        return 2 * t / (1 + t) ** 2

try:
    from sentence_transformers import SentenceTransformer

    phrase_embedder = SentenceTransformer("all-mpnet-base-v2")
except Exception:
    phrase_embedder = None

def semantic_similarity(a: str | list, b: str | list) -> float:
    if phrase_embedder is None:
        raise RuntimeError("SentenceTransformer not available for semantic similarity computation.")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    single = False

    assert type(a) == type(b)

    if isinstance(a, str):
        a = [a]
        single = True
    if isinstance(b, str):
        b = [b]
        single = True

    emb_a = phrase_embedder.encode(a)
    emb_b = phrase_embedder.encode(b)

    # Compute cosine similarity
    cos_sim = phrase_embedder.similarity_pairwise(emb_a, emb_b)

    if single:
        assert len(cos_sim) == 1

        cos_sim = cos_sim[0]

    return cos_sim.numpy()

def default_string_distance(a: str, b: str) -> float:
    # Returns similarity ratio (1.0 = identical, 0.0 = completely different)
    return SequenceMatcher(None, str(a), str(b)).ratio()

def equal(a: str, b: str):
    return a == b

def substring_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Returns True if any substring of b matches a with similarity above threshold.
    """
    if not a or not b:
        return False
    a = str(a)
    b = str(b)
    max_ratio = 0.0
    len_a = len(a)

    for i in range(len(b) - len_a + 1):
        sub = b[i:i + len_a]
        ratio = SequenceMatcher(None, a, sub).ratio()

        if ratio > max_ratio:
            max_ratio = ratio

            if max_ratio >= threshold:
                return True

    return max_ratio >= threshold

def substring_match_levenshtein(a: str, b: str, threshold: int=1) -> int:
    """
    Returns the minimal Levenshtein distance between 'a' and any substring of 'b' of length len(a).
    If 'a' or 'b' is empty, returns len(a) (i.e., the cost to insert all of 'a').
    """
    if not a or not b:
        return len(a) if a else 0
    a = str(a)
    b = str(b)
    len_a = len(a)
    min_dist = None

    for i in range(len(b) - len_a + 1):
        sub = b[i:i + len_a]
        dist = lev.distance(a, sub)
        if min_dist is None or dist < min_dist:
            min_dist = dist

    # If b is shorter than a, compare directly
    if min_dist is None:
        min_dist = lev.distance(a, b)

    return min_dist <= threshold

def compare_species_name(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Compare only the two first words of the species name
    extracted = " ".join(extracted.split(" ")[:2]) if extracted else ""
    gt = " ".join(gt.split(" ")[:2]) if gt else ""

    return dist_fn(extracted, gt)

def compare_collection_date(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Only compare the year
    extracted_year = str(extracted).split("-")[0] # "2023-08-17" -> "2023"
    gt_year = str(gt).split(" ")[-1] # "17. August 1951" -> "1951"

    return dist_fn(extracted_year, gt_year)

def compare_collectors_name(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Only compare the last name
    # Get the last word as last name
    extracted_last_name = extracted.split(" ")[-1] if extracted else ""

    # Get the last word as last name
    gt_last_name = gt.split(" ")[-1] if gt else ""

    return dist_fn(extracted_last_name, gt_last_name)

def compare_location(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Compare complete location strings, but format gt similarly to extracted

    return dist_fn(extracted, gt)


metric_fns_sim = {
    "Species name": compare_species_name,
    "Collection date": compare_collection_date,
    "Collector's name": compare_collectors_name,
    "Location/Place": compare_location,
    "Location/Description": compare_location,
}

metric_fns_lev = {
    "Species name": partial(compare_species_name, dist_fn=lev.distance),
    "Collection date": partial(compare_collection_date, dist_fn=lev.distance),
    "Collector's name": partial(compare_collectors_name, dist_fn=lev.distance),
    "Location/Place": partial(compare_location, dist_fn=lev.distance),
    "Location/Description": partial(compare_location, dist_fn=lev.distance),
}

metric_fns_match = {
    "Species name": partial(compare_species_name, dist_fn=equal),
    "Collection date": partial(compare_collection_date, dist_fn=equal),
    "Collector's name": partial(compare_collectors_name, dist_fn=equal),
    "Location/Place": partial(compare_location, dist_fn=equal),
    "Location/Description": partial(compare_location, dist_fn=equal),
}

metric_fns_semsim = {
    "Location": semantic_similarity,
    "Location/Place": semantic_similarity,
    "Location/Description": semantic_similarity,
}

def plot_per_category_regressions(
    metric_name: str,
    metric_values: Dict,
    years: pd.Series,
    output_dir: str,
    base_name: str,
    separate_plots: bool = False,
):
    """
    Plot per-category regressions (linear, linear2, glm modes).

    Parameters
    ----------
    metric_name : str
        Name of metric (e.g., "MSS", "MLD", "MMD", "SEM")
    metric_values : dict
        Dict with metric values per category
    years : pd.Series
        Collection years
    output_dir : str
        Output directory for plots
    base_name : str
        Base filename for saving
    separate_plots : bool
        If True, save each category to a separate PDF; else combine all in one plot

    Returns
    -------
    correlation_results : list
        List of dicts with correlation stats per category
    """
    correlation_results = []

    if separate_plots:
        # One plot per category
        for col in metric_values[metric_name].keys():
            ys = metric_values[metric_name][col]
            yrs = np.array(years.values, dtype=float)

            mask = ~np.isnan(yrs)
            if mask.sum() == 0:
                continue

            x = yrs[mask]
            y = ys[mask]

            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            plt.figure(figsize=(8, 5))
            plt.scatter(x_sorted, y_sorted, marker='o', s=30, alpha=0.8, label=col)

            corr_coef = np.nan
            corr_pvalue = np.nan
            r2_value = np.nan
            fstat = np.nan
            slope = np.nan

            if x_sorted.size >= 2 and np.unique(x_sorted).size >= 2:
                if pearsonr is not None:
                    corr_coef, corr_pvalue = pearsonr(x_sorted, y_sorted)
                    r2_value = corr_coef ** 2

                append_str = ""
                pvalue = np.nan

                if regression_mode == "linear":
                    lr = linregress(x_sorted, y_sorted)
                    slope, intercept, rvalue, pvalue, stderr = lr.slope, lr.intercept, lr.rvalue, lr.pvalue, lr.stderr
                    y_pred = slope * x_sorted + intercept
                    append_str = f"(p={pvalue:.3f})"

                elif regression_mode == "linear2":
                    x_sorted_ols = sm.add_constant(x_sorted)
                    mdl = sm.OLS(y_sorted, x_sorted_ols)
                    results = mdl.fit()

                    print(metric_name + " - " + col)
                    print(results.summary())

                    y_pred = results.predict(x_sorted_ols)
                    pvalue = results.pvalues[1]
                    r2 = results.rsquared
                    coef = results.params[1]
                    fstat = results.fvalue
                    slope = coef

                    append_str = f"(R²={r2:.3f}, p={pvalue:.3f}, slope={coef:.4f}, F={fstat:.3f})"

                elif regression_mode == "glm":
                    x_sorted_ols = sm.add_constant(x_sorted)
                    mdl = sm.GLM(y_sorted, x_sorted_ols, family=sm.families.Binomial(link=ScaledLogit()))
                    results = mdl.fit(maxiter=100)

                    print(metric_name + " - " + col)
                    print(results.summary())

                    y_pred = results.predict(x_sorted_ols)
                    pvalue = results.pvalues[1]
                    append_str = f"(p={pvalue:.3f})"

                if pvalue < 0.05:
                    label = "Trend line" #f"{col} {append_str}" if append_str else f"{col}"
                    plt.plot(x_sorted, y_pred, linestyle='--', linewidth=1.5, label=label)

                plt.xlabel("Collection Year")
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                out_path = os.path.join(output_dir, f"{base_name}_{metric_name}_{col.replace('/', '_')}.pdf")
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()

            correlation_results.append({
                "metric": metric_name,
                "category": col,
                "slope": slope,
                "pearson_correlation": corr_coef,
                "p_value": corr_pvalue,
                "r_squared": r2_value,
                "f_statistic": fstat,
                "n_samples": x_sorted.size,
            })
    else:
        # All categories in one plot
        plt.figure(figsize=(8, 5))

        for col in metric_values[metric_name].keys():
            ys = metric_values[metric_name][col]
            yrs = np.array(years.values, dtype=float)

            mask = ~np.isnan(yrs)
            if mask.sum() == 0:
                continue

            x = yrs[mask]
            y = ys[mask]

            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            plot = plt.scatter(x_sorted, y_sorted, marker='o', s=30, alpha=0.8, label=col)
            plot_color = plot.get_facecolor()

            corr_coef = np.nan
            corr_pvalue = np.nan
            r2_value = np.nan
            fstat = np.nan
            slope = np.nan

            if x_sorted.size >= 2 and np.unique(x_sorted).size >= 2:
                if pearsonr is not None:
                    corr_coef, corr_pvalue = pearsonr(x_sorted, y_sorted)
                    r2_value = corr_coef ** 2

                append_str = ""
                pvalue = np.nan

                if regression_mode == "linear":
                    lr = linregress(x_sorted, y_sorted)
                    slope, intercept, rvalue, pvalue, stderr = lr.slope, lr.intercept, lr.rvalue, lr.pvalue, lr.stderr
                    y_pred = slope * x_sorted + intercept
                    append_str = f"(p={pvalue:.3f})"

                elif regression_mode == "linear2":
                    x_sorted_ols = sm.add_constant(x_sorted)
                    mdl = sm.OLS(y_sorted, x_sorted_ols)
                    results = mdl.fit()

                    print(metric_name + " - " + col)
                    print(results.summary())

                    y_pred = results.predict(x_sorted_ols)
                    pvalue = results.pvalues[1]
                    r2 = results.rsquared
                    coef = results.params[1]
                    fstat = results.fvalue
                    slope = coef

                    append_str = f"(R²={r2:.3f}, p={pvalue:.3f}, slope={coef:.4f}, F={fstat:.3f})"

                elif regression_mode == "glm":
                    x_sorted_ols = sm.add_constant(x_sorted)
                    mdl = sm.GLM(y_sorted, x_sorted_ols, family=sm.families.Binomial(link=ScaledLogit()))
                    results = mdl.fit(maxiter=100)

                    print(metric_name + " - " + col)
                    print(results.summary())

                    y_pred = results.predict(x_sorted_ols)
                    pvalue = results.pvalues[1]
                    append_str = f"(p={pvalue:.3f})"

                if pvalue < 0.05:
                    label = f"{col} {append_str}" if append_str else f"{col}"
                    plt.plot(x_sorted, y_pred, linestyle='--', linewidth=1.5, label=label, color=plot_color)

            correlation_results.append({
                "metric": metric_name,
                "category": col,
                "slope": slope,
                "pearson_correlation": corr_coef,
                "p_value": corr_pvalue,
                "r_squared": r2_value,
                "f_statistic": fstat,
                "n_samples": x_sorted.size,
            })

        plt.xlabel("Collection Year")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(output_dir, f"{base_name}_{metric_name}.pdf")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    return correlation_results


def plot_pooled_interaction_model(
    metric_name: str,
    metric_values: Dict,
    years: pd.Series,
    output_dir: str,
    base_name: str,
    separate_plots: bool = False,
):
    """
    Plot pooled interaction model: value ~ year * C(category).

    Parameters
    ----------
    metric_name : str
        Name of metric
    metric_values : dict
        Dict with metric values per category
    years : pd.Series
        Collection years
    output_dir : str
        Output directory for plots
    base_name : str
        Base filename for saving
    separate_plots : bool
        If True, save each category to a separate PDF; else combine all in one plot
    """
    # Build long-form dataframe combining all categories for this metric
    rows = []
    plot_colors = {}

    for col in metric_values[metric_name].keys():
        ys = metric_values[metric_name][col]
        yrs = np.array(years.values, dtype=float)
        mask = ~np.isnan(yrs)

        if mask.sum() == 0:
            continue

        x = yrs[mask]
        y = ys[mask]

        for xi, yi in zip(x, y):
            rows.append({"year": float(xi), "value": float(yi), "category": str(col)})

    if len(rows) < 3:
        return

    pooled_df = pd.DataFrame(rows)

    # Ensure there is more than one category and more than one unique year
    if pooled_df["category"].nunique() < 2 or pooled_df["year"].nunique() < 2:
        return

    formula = "value ~ year * C(category)"
    mdl = smf.ols(formula, data=pooled_df)
    res = mdl.fit()

    # save summary
    summary_out = os.path.join(output_dir, f"{base_name}_{metric_name}_pooled_model.txt")
    with open(summary_out, "w") as fh:
        fh.write(res.summary().as_text())
    print(f"Pooled model saved: {summary_out}")

    # Title: R² and F-statistic
    r2 = getattr(res, "rsquared", np.nan)
    fstat = getattr(res, "fvalue", np.nan)
    title = f"{metric_name} (R²={r2:.3f}, F={fstat:.3f})"

    # If model significant (or f_pval not available) plot predicted curves per category
    f_pval = getattr(res, "f_pvalue", None)
    try:
        f_pval = float(f_pval) if f_pval is not None else None
    except Exception:
        f_pval = None

    if f_pval is None or f_pval < 0.05:
        yr_min, yr_max = int(pooled_df["year"].min()), int(pooled_df["year"].max())
        year_grid = np.linspace(yr_min, yr_max, 100)

        param_names = res.params.index.tolist()

        for cat in pooled_df["category"].unique():
            if separate_plots:
                plt.figure(figsize=(8, 5))

                # plot scatter points for this category
                cat_data = pooled_df[pooled_df["category"] == cat]
                plt.scatter(cat_data["year"], cat_data["value"], marker='o', s=30, alpha=0.8, label=cat)
            else:
                if cat == pooled_df["category"].unique()[0]:
                    plt.figure(figsize=(8, 5))

            df_pred = pd.DataFrame({"year": year_grid, "category": [cat] * len(year_grid)})
            y_pred = res.predict(df_pred)

            # compute slope for this category: coef(year) + coef(interaction) if present
            slope = float(res.params.get("year", 0.0))
            interaction_name = None
            for cand in (f"year:C(category)[T.{cat}]", f"C(category)[T.{cat}]:year"):
                if cand in param_names:
                    interaction_name = cand
                    break
            if interaction_name:
                slope += float(res.params.get(interaction_name, 0.0))

            # compute p-value for the slope via t_test on the linear combination
            contrast = np.zeros(len(param_names))
            if "year" in param_names:
                contrast[param_names.index("year")] = 1.0
            if interaction_name:
                contrast[param_names.index(interaction_name)] = 1.0
            try:
                ttest = res.t_test(contrast)
                slope_p = float(np.squeeze(ttest.pvalue))
            except Exception:
                slope_p = np.nan

            label = f"{cat} (slope={slope:.4f}, p={slope_p:.3f})"
            plt.plot(year_grid, y_pred, linestyle='-', linewidth=1.5, label=label, alpha=0.9)

            if separate_plots:
                plt.xlabel("Collection Year")
                plt.ylabel(metric_name)
                plt.title(title)
                plt.legend()
                plt.grid(True)
                out_path = os.path.join(output_dir, f"{base_name}_{metric_name}_{cat}.pdf")
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()

        if not separate_plots:
            plt.xlabel("Collection Year")
            plt.ylabel(metric_name)
            plt.title(title)
            plt.legend()
            plt.grid(True)
            out_path = os.path.join(output_dir, f"{base_name}_{metric_name}.pdf")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()


# New helper: build per-row metrics and save plots of metrics over ground-truth years
def plot_metrics_over_years(comparison_df: pd.DataFrame, extracted_csv: str, output_dir: str,
                            metric_fns_sim: Dict[str, Callable],
                            metric_fns_lev: Dict[str, Callable],
                            metric_fns_match: Dict[str, Callable],
                            metric_fns_semsim: Dict[str, Callable],
                            separate_plots: bool = False,):
    """
    Plot metrics over collection year with optional per-category plots.

    Parameters
    ----------
    separate_plots : bool
        If True, save each category to separate PDF files
    """
    # Helper to extract year using same logic as compare_collection_date: take last token after splitting by space
    def extract_year(s):
        if not s or pd.isna(s):
            return np.nan
        try:
            gt_year_token = str(s).split(" ")[-1]
            return int(gt_year_token)
        except Exception:
            return np.nan

    categories = set(list(metric_fns_sim.keys()) + list(metric_fns_semsim.keys()))

    years = comparison_df.get("ground_truth_Collection date", pd.Series([""] * len(comparison_df))).apply(extract_year)

    metric_values = {"MSS": {}, "MLD": {}, "MMD": {}, "SEM": {}}

    for col in categories:
        ex_col = f"extracted_{col}"
        gt_col = f"ground_truth_{col}"
        ex_vals = comparison_df[ex_col].astype(str).tolist()
        gt_vals = comparison_df[gt_col].astype(str).tolist()

        if col in metric_fns_sim:
            sim_list = [metric_fns_sim[col](e, g) for e, g in zip(ex_vals, gt_vals)]
            lev_list = [metric_fns_lev[col](e, g) for e, g in zip(ex_vals, gt_vals)]
            match_list = [int(bool(metric_fns_match[col](e, g))) for e, g in zip(ex_vals, gt_vals)]

            metric_values["MSS"][col] = np.array(sim_list, dtype=float)
            metric_values["MLD"][col] = np.array(lev_list, dtype=float)
            metric_values["MMD"][col] = np.array(match_list, dtype=float)

        if col in metric_fns_semsim and compute_SEM:
            #semsim_list = [metric_fns_semsim[col](e, g) for e, g in zip(ex_vals, gt_vals)]
            metric_values["SEM"][col] = metric_fns_semsim[col](ex_vals, gt_vals)

    # Ensure out dir exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    base_name = os.path.splitext(os.path.basename(extracted_csv))[0]

    # Collect correlation results for all metrics and categories
    correlation_results = []

    for metric_name in ["MSS", "MLD", "MMD"] + (["SEM"] if compute_SEM else []):

        if regression_mode == "interaction":
            plot_pooled_interaction_model(
                metric_name,
                metric_values,
                years,
                output_dir,
                base_name,
                separate_plots=separate_plots
            )
        else:
            corr_results = plot_per_category_regressions(
                metric_name,
                metric_values,
                years,
                output_dir,
                base_name,
                separate_plots=separate_plots
            )
            correlation_results.extend(corr_results)

    # Save correlation results to CSV
    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        corr_out_path = os.path.join(output_dir, f"{base_name}_correlations.csv")
        corr_df.to_csv(corr_out_path, index=False)
        corr_df.to_latex(corr_out_path.replace(".csv", ".tex"), index=False)
        print(f"Correlation results saved to {corr_out_path}")
        print(corr_df)

def compare_tables(
    extracted_csv: str,
    ground_truth_csv: str,
    column_map: Dict[str, str],
    output_dir: str = None,
):
    # Load CSVs
    extracted = pd.read_csv(extracted_csv, keep_default_na=False)
    ground_truth = pd.read_csv(ground_truth_csv, keep_default_na=False)

    # Index both tables by source_image
    extracted = extracted.set_index("source_image")
    ground_truth = ground_truth.set_index(column_map["source_image"])

    column_map = column_map.copy()

    del column_map["source_image"]  # Remove source_image from mapping

    # Filter columns
    extracted_cols = list(column_map.keys())
    ground_truth_cols = list(column_map.values())

    # Only keep rows present in both
    common_images = extracted.index.intersection(ground_truth.index)
    for col in extracted_cols:
        if col not in extracted.columns:
            extracted[col] = ""
    extracted = extracted.loc[common_images, extracted_cols]
    ground_truth = ground_truth.loc[common_images, ground_truth_cols]
    ground_truth.columns = extracted_cols  # Rename for comparison

    # Ensure both dataframes have the same source_image items
    missing_in_extracted = ground_truth.index.difference(extracted.index)

    # Add missing rows with empty strings
    for idx in missing_in_extracted:
        extracted.loc[idx] = [""] * len(extracted.columns)

    # Sort both dataframes to align rows
    extracted = extracted.sort_index()
    ground_truth = ground_truth.sort_index()

    if extracted.shape != ground_truth.shape:
        print(f"Warning: Dataframes have different shapes after alignment. {extracted.shape} vs {ground_truth.shape}")
        # Align by index, filling missing values with empty strings
        try:
            extracted = extracted.reindex(ground_truth.index, fill_value="")
            ground_truth = ground_truth.reindex(extracted.index, fill_value="")
        except Exception as e:
            print(f"Error aligning dataframes: {e}")
            return pd.Series()

    # Check if both indices are now identical
    assert all(extracted.index == ground_truth.index), "Indices do not match after alignment."
    assert all(s1 == s2 for s1, s2 in zip(extracted.shape, ground_truth.shape)), f"The dataframe shapes do not match. {(extracted.shape, ground_truth.shape)}"

    # Split "Location" columns into two columns based on a delimiter (e.g., ",")
    if "Location" in extracted.columns and "Location" in ground_truth.columns:
        extracted[["Location/Place", "Location/Description"]] = extracted["Location"].str.split(", ", n=1, expand=True)
        ground_truth[["Location/Place", "Location/Description"]] = ground_truth["Location"].str.split(" / ", n=1, expand=True)

        extracted[extracted.isnull()] = ""
        ground_truth[ground_truth.isnull()] = ""

    print(extracted)
    print(ground_truth)

    # Compute metrics
    results = {}

    sim_cols = []
    lev_cols = []
    semsim_cols = []

    for col in metric_fns_sim:
        results[f"{col}_similarity"] = extracted[col].combine(ground_truth[col], metric_fns_sim[col]).mean()
        results[f"{col}_levenshtein"] = extracted[col].combine(ground_truth[col], metric_fns_lev[col]).mean()
        results[f"{col}_match"] = extracted[col].combine(ground_truth[col], metric_fns_match[col]).mean()

        sim_cols.append(f"{col}_similarity")
        lev_cols.append(f"{col}_levenshtein")

    for col in metric_fns_semsim:
        results[f"{col}_semantic_similarity"] = extracted[col].combine(ground_truth[col], metric_fns_semsim[col]).mean()
        semsim_cols.append(f"{col}_semantic_similarity")

    results["loc_contained"] = np.mean([el in glp or el in gld for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained"] = np.mean([substring_match(el, glp + " " + gld) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev1"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 1) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev2"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 2) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev3"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 3) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])

    df_results = pd.Series(results)
    df_results["mean_similarity"] = df_results[sim_cols].mean()
    df_results["mean_levenshtein"] = df_results[lev_cols].mean()

    # Write side-by-side comparison to a CSV file for this extraction
    comparison_df = pd.DataFrame({
        "source_image": extracted.index,
    })
    for col in extracted.columns:
        comparison_df[f"extracted_{col}"] = extracted[col].values
        comparison_df[f"ground_truth_{col}"] = ground_truth[col].values

    # Save to file (filename based on extracted_csv)
    comparison_output = extracted_csv.replace(".csv", "_comparison.csv")
    comparison_df.to_csv(comparison_output, index=False)

    # Determine output directory for plots
    if output_dir:
        out_dir = output_dir if output_dir != "" else "."
    else:
        out_dir = os.path.dirname(comparison_output) or "."

    # --- New: extract rows where the first two words of species differ from ground truth ---
    def _find_col(df, *tokens):
        for c in df.columns:
            low = c.lower()
            if all(tok.lower() in low for tok in tokens):
                return c
        return None

    ex_species_col = _find_col(comparison_df, "extracted", "species")
    gt_species_col = _find_col(comparison_df, "ground_truth", "species")

    if ex_species_col and gt_species_col:
        def _first_two(s):
            return " ".join(str(s).split()[:2]).strip()

        comparison_df["extracted_species_2"] = comparison_df[ex_species_col].apply(_first_two)
        comparison_df["ground_truth_species_2"] = comparison_df[gt_species_col].apply(_first_two)

        diff_mask = comparison_df["extracted_species_2"] != comparison_df["ground_truth_species_2"]
        species_diff = comparison_df.loc[diff_mask, ["source_image", ex_species_col, gt_species_col, "extracted_species_2", "ground_truth_species_2"]].copy()
        species_diff.columns = ["source_image", "extracted_full", "ground_truth_full", "extracted_first_two", "ground_truth_first_two"]

        base_name = os.path.splitext(os.path.basename(extracted_csv))[0]

        species_diff_out = os.path.join(out_dir, f"{base_name}_species_differences.csv")
        species_diff.to_csv(species_diff_out, index=False)
        print(f"Species differences saved: {len(species_diff)} rows -> {species_diff_out}")
    else:
        print("Could not find extracted/ground_truth species columns in comparison dataframe; skipping species-differences extraction.")

    # Replace inline plotting with call to helper
    plot_metrics_over_years(
        comparison_df,
        extracted_csv,
        out_dir,
        metric_fns_sim,
        metric_fns_lev,
        metric_fns_match,
        metric_fns_semsim,
        separate_plots=True,
    )

    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extracted herbarium data with ground truth.")
    parser.add_argument("--extracted_csv", required=True, nargs="+", help="Path to extracted data CSV.")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--output_csv", default=None, help="Path to save comparison results.")
    args = parser.parse_args()

    # Parse column_map argument
    column_map = {
        "source_image": "Bildname",
        "Species name": "Spezies Label",
        "Collection date": "Sammeldatum",
        "Collector's name": "Sammler",
        "Country/State": "Geographische_Zuordnung",
        "Location": "Fundort Label",
        "Region": "Naturraum",
        "Notes": "Bemerkung_zur_Pflanze",
    }

    metric_series = {}

    for csv_file in args.extracted_csv:
        out_dir_for_plots = os.path.dirname(csv_file) or "."

        s = compare_tables(
            csv_file,
            args.ground_truth_csv,
            column_map,
            output_dir=out_dir_for_plots,
        )
        metric_series[csv_file] = s

    df_results = pd.DataFrame(metric_series).T

    print(df_results)

    if args.output_csv:
        df_results.to_csv(args.output_csv, index=True)
