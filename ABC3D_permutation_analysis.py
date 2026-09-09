# -*- coding: utf-8 -*-
"""
ABC3D permutation-based multivariate analysis
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix

INPUT_FILE = Path("./results/ABC3D_features.csv")
OUT_DIR = Path("./results/permutation_analysis")

N_PERMUTATIONS = 9999
RANDOM_SEED = 12345
DPI = 600

ANALYSIS_FEATURES = [
    "shannon_entropy",
    "renyi_entropy_a2",
    "boxcount_D",
    "lacunarity_mean",
    "glcm_asm_mean",
    "glcm_contrast_mean",
    "glcm_correlation_mean",
    "glcm_dissimilarity_mean",
    "glcm_homogeneity_mean",
    "LLL_rel_energy",
    "LLH_rel_energy",
    "LHL_rel_energy",
    "LHH_rel_energy",
    "HLL_rel_energy",
    "HLH_rel_energy",
    "HHL_rel_energy",
    "HHH_rel_energy",
]

EXPECTED_STRAINS = ["BW25113", "amiA", "ompR", "ydgD"]
EXPECTED_MEDIA = ["LB", "M9"]


def load_input_table(path):
    """Load the ABC3D feature table from CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input feature table not found: {path}\n"
            "Set INPUT_FILE to the ABC3D feature table to analyse."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported input format '{path.suffix}'. "
            "ABC3D analysis requires a CSV file."
        )

    return pd.read_csv(path)


def prepare_input_table(df):
    """Validate explicit metadata and the exact manuscript feature set."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    aliases = {}
    if "strain" in df.columns and "Strain" not in df.columns:
        aliases["strain"] = "Strain"
    if "medium" in df.columns and "Medium" not in df.columns:
        aliases["medium"] = "Medium"
    if "filename" in df.columns and "Sample" not in df.columns:
        aliases["filename"] = "Sample"
    df = df.rename(columns=aliases)

    missing = {"Strain", "Medium"} - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required metadata column(s): " + ", ".join(sorted(missing))
        )

    for col in ["Strain", "Medium"]:
        if df[col].isna().any():
            rows = (df.index[df[col].isna()] + 2).tolist()
            raise ValueError(f"Blank {col} value(s) at input row(s): {rows}")
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            rows = (df.index[df[col].eq("")] + 2).tolist()
            raise ValueError(f"Blank {col} value(s) at input row(s): {rows}")

    unexpected_strains = sorted(set(df["Strain"]) - set(EXPECTED_STRAINS))
    unexpected_media = sorted(set(df["Medium"]) - set(EXPECTED_MEDIA))
    if unexpected_strains:
        raise ValueError(
            "Unexpected Strain value(s): " + ", ".join(unexpected_strains)
        )
    if unexpected_media:
        raise ValueError(
            "Unexpected Medium value(s): " + ", ".join(unexpected_media)
        )

    missing_features = [c for c in ANALYSIS_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing required analysis feature(s): " + ", ".join(missing_features)
        )

    bad = df[ANALYSIS_FEATURES].isna().any(axis=1)
    if bad.any():
        rows = (df.index[bad] + 2).tolist()
        raise ValueError(
            "Missing analysis feature value(s) at input row(s): "
            + ", ".join(map(str, rows))
        )

    if "Sample" not in df.columns:
        df["Sample"] = [f"sample_{i+1:03d}" for i in range(len(df))]

    return df


def fit_rss(Y, X):
    """
    Fit multivariate least squares and return residual sum of squares
    across all response dimensions.
    """
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    rss = float(np.sum(resid ** 2))
    return rss, resid, X @ beta


def pseudo_f_from_models(Y, X_full, X_reduced):
    """
    Partial multivariate pseudo-F from nested linear models.
    """
    rss_full, _, _ = fit_rss(Y, X_full)
    rss_reduced, _, _ = fit_rss(Y, X_reduced)

    rank_full = np.linalg.matrix_rank(X_full)
    rank_reduced = np.linalg.matrix_rank(X_reduced)

    df_term = rank_full - rank_reduced
    df_resid = Y.shape[0] - rank_full

    ss_term = rss_reduced - rss_full

    if ss_term < 0 and abs(ss_term) < 1e-10:
        ss_term = 0.0

    ms_term = ss_term / df_term
    ms_resid = rss_full / df_resid

    F = ms_term / ms_resid

    return {
        "F": float(F),
        "SS_term": float(ss_term),
        "RSS_full": float(rss_full),
        "df_term": int(df_term),
        "df_resid": int(df_resid),
    }


def freedman_lane_test(Y, X_full, X_reduced, n_perm, rng):
    """
    Freedman-Lane residual permutation test for one model term.

    1. Fit reduced model.
    2. Permute rows of reduced-model residuals.
    3. Add permuted residuals to reduced-model fitted values.
    4. Refit full and reduced models.
    5. Compare pseudo-F with observed pseudo-F.
    """
    observed = pseudo_f_from_models(Y, X_full, X_reduced)

    _, reduced_resid, reduced_fitted = fit_rss(Y, X_reduced)

    null_F = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        Y_perm = reduced_fitted + reduced_resid[perm, :]

        perm_result = pseudo_f_from_models(
            Y_perm,
            X_full,
            X_reduced
        )
        null_F[i] = perm_result["F"]

    p_value = (1 + np.sum(null_F >= observed["F"])) / (n_perm + 1)

    observed["p_perm"] = float(p_value)

    return observed, null_F


def save_null_plot(null_values, observed_F, p_value, title, filename_base):
    plt.figure(figsize=(6.4, 4.6))
    plt.hist(null_values, bins=40, edgecolor="black", linewidth=0.5)
    plt.axvline(
        observed_F,
        linewidth=2,
        linestyle="--",
        label=f"Observed pseudo-F = {observed_F:.3f}"
    )
    plt.xlabel("Pseudo-F under null permutation")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(frameon=False)
    plt.text(
        0.98, 0.95,
        f"Permutation P = {p_value:.4g}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top"
    )
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / (filename_base + ".png"),
        dpi=DPI,
        bbox_inches="tight"
    )
    plt.close()



# =========================
# LOAD AND VALIDATE DATA
# =========================

OUT_DIR.mkdir(parents=True, exist_ok=True)
df_clean = prepare_input_table(load_input_table(INPUT_FILE))

counts = (
    df_clean.groupby(["Strain", "Medium"])
    .size()
    .unstack(fill_value=0)
)
print("\nGroup counts:")
print(counts)

expected_groups = {
    (strain, medium)
    for strain in EXPECTED_STRAINS
    for medium in EXPECTED_MEDIA
}
observed_groups = set(
    map(tuple, df_clean[["Strain", "Medium"]].drop_duplicates().to_numpy())
)
missing_groups = expected_groups - observed_groups
if missing_groups:
    raise ValueError(
        "Missing strain × medium group(s): "
        + ", ".join(f"{s}/{m}" for s, m in sorted(missing_groups))
    )


# =========================
# STANDARDISE FEATURES
# =========================

feature_cols = list(ANALYSIS_FEATURES)
Y = StandardScaler().fit_transform(
    df_clean[feature_cols].to_numpy(dtype=float)
)

print(f"\nSamples analysed: {Y.shape[0]}")
print(f"Features analysed: {Y.shape[1]}")

standardised_df = pd.DataFrame(Y, columns=feature_cols, index=df_clean.index)
audit = pd.concat(
    [
        df_clean[["Sample", "Strain", "Medium"]].reset_index(drop=True),
        standardised_df.reset_index(drop=True),
    ],
    axis=1,
)
audit.to_csv(OUT_DIR / "PERMANOVA_standardised_input.csv", index=False)


# =========================
# FACTORIAL DESIGN MATRIX
# =========================

meta = df_clean[["Strain", "Medium"]].copy()
meta["Strain"] = pd.Categorical(meta["Strain"], categories=EXPECTED_STRAINS)
meta["Medium"] = pd.Categorical(meta["Medium"], categories=EXPECTED_MEDIA)

design = dmatrix(
    "1 + C(Medium, Sum) * C(Strain, Sum)",
    data=meta,
    return_type="dataframe",
)
X_full = design.to_numpy(dtype=float)

term_slices = design.design_info.term_name_slices
medium_term = "C(Medium, Sum)"
strain_term = "C(Strain, Sum)"
interaction_term = "C(Medium, Sum):C(Strain, Sum)"

for term in [medium_term, strain_term, interaction_term]:
    if term not in term_slices:
        raise RuntimeError(f"Could not find design-matrix term: {term}")


def reduced_matrix_without_term(term_name):
    sl = term_slices[term_name]
    cols_to_remove = set(range(sl.start, sl.stop))
    keep = [i for i in range(X_full.shape[1]) if i not in cols_to_remove]
    return X_full[:, keep]


# =========================
# OBSERVED + PERMUTATION TESTS
# =========================

rng = np.random.default_rng(RANDOM_SEED)

tests = [
    ("Medium", medium_term, "Permutation_Medium"),
    ("Strain", strain_term, "Permutation_Strain"),
    ("Medium × Strain", interaction_term, "Permutation_Interaction"),
]

Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
SS_total = float(np.sum(Y_centered ** 2))

results = []
null_rows = []

for display_term, design_term, plot_name in tests:
    print(f"\nRunning {display_term}: {N_PERMUTATIONS} permutations...")

    X_reduced = reduced_matrix_without_term(design_term)
    obs, null_F = freedman_lane_test(
        Y, X_full, X_reduced, N_PERMUTATIONS, rng
    )

    R2 = obs["SS_term"] / SS_total

    results.append({
        "term": display_term,
        "pseudo_F": obs["F"],
        "df_term": obs["df_term"],
        "df_resid": obs["df_resid"],
        "SS_term": obs["SS_term"],
        "R2_total": R2,
        "permutation_p": obs["p_perm"],
        "n_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
    })

    for i, fval in enumerate(null_F, start=1):
        null_rows.append({
            "term": display_term,
            "permutation": i,
            "pseudo_F_null": fval,
        })

    save_null_plot(
        null_F,
        obs["F"],
        obs["p_perm"],
        title=f"{display_term}: permutation null distribution",
        filename_base=plot_name,
    )


# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "PERMANOVA_results.csv", index=False)

pd.DataFrame(null_rows).to_csv(
    OUT_DIR / "PERMANOVA_null_distributions.csv",
    index=False,
)

print("\n========================================")
print("PERMUTATION RESULTS")
print("========================================")
print(
    results_df[
        ["term", "pseudo_F", "df_term", "df_resid", "R2_total", "permutation_p"]
    ].to_string(index=False)
)

print("\nAll outputs saved to:")
print(OUT_DIR)
