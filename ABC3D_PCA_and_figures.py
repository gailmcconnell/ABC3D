# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:09:28 2026

@author: Gail McConnell
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.builtins import Q

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = Path("./results/ABC3D_features.csv")
OUT_DIR = Path("./results/PCA_and_figures")
DPI = 600

PCA_FEATURES = [
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

# =========================
# Helper functions
# =========================
def ensure_outdir(path: Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def savefig(name_base: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name_base}.pdf")
    plt.savefig(OUT_DIR / f"{name_base}.png", dpi=DPI)
    plt.close()

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; returns q-values aligned to pvals."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out_masked = np.empty(m, dtype=float)
    out_masked[order] = q
    out[mask] = out_masked
    return out

def load_input_table(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input feature table not found: {path}\n"
            "Set INPUT_FILE to the feature table to analyse."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported input format '{path.suffix}'. ABC3D analysis requires a CSV file."
        )

    return pd.read_csv(path)


def prepare_input_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate experimental metadata supplied explicitly in the feature table.

    Strain and medium are never inferred from filenames or sample names.
    """
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Accept either lower-case columns produced by ABC3D.py or conventional
    # capitalised column names used in existing analysis tables.
    aliases = {}
    if "strain" in df.columns and "Strain" not in df.columns:
        aliases["strain"] = "Strain"
    if "medium" in df.columns and "Medium" not in df.columns:
        aliases["medium"] = "Medium"
    if "filename" in df.columns and "Sample" not in df.columns:
        aliases["filename"] = "Sample"
    if "replicate" in df.columns and "Replicate" not in df.columns:
        aliases["replicate"] = "Replicate"
    df = df.rename(columns=aliases)

    required = {"Strain", "Medium"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Input feature table is missing required metadata column(s): "
            + ", ".join(sorted(missing))
            + ". Strain and Medium must be supplied explicitly."
        )

    for col in ["Strain", "Medium"]:
        if df[col].isna().any():
            bad = (df.index[df[col].isna()] + 2).tolist()
            raise ValueError(f"Blank {col} value(s) at input row(s): {bad}")
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            bad = (df.index[df[col].eq("")] + 2).tolist()
            raise ValueError(f"Blank {col} value(s) at input row(s): {bad}")

    # A sample identifier is useful for output traceability but is not used
    # to determine experimental group membership.
    if "Sample" not in df.columns:
        df["Sample"] = [f"sample_{i+1:03d}" for i in range(len(df))]

    if "Replicate" not in df.columns:
        df["Replicate"] = np.nan

    return df


# =========================
# Load and validate data
# =========================
ensure_outdir(OUT_DIR)
df = prepare_input_table(load_input_table(INPUT_FILE))

# Use the predefined analysis feature set rather than all numeric output columns.
missing_features = [c for c in PCA_FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(
        "Input feature table is missing required PCA feature column(s): "
        + ", ".join(missing_features)
    )

feature_cols = list(PCA_FEATURES)

# Stop rather than silently changing the analysis if a feature contains
# missing values.
missing_feature_rows = df[feature_cols].isna().any(axis=1)
if missing_feature_rows.any():
    bad = (df.index[missing_feature_rows] + 2).tolist()
    raise ValueError(
        "Missing value(s) found in PCA feature columns at input row(s): "
        + ", ".join(map(str, bad))
    )

df_clean = df.copy()

# Save the exact validated input used for analysis.
df_clean.to_csv(OUT_DIR / "validated_input.csv", index=False)

# =========================
# PCA of the standardized feature set
# =========================
X = df_clean[feature_cols].to_numpy(dtype=float)
Xz = StandardScaler().fit_transform(X)

pca = PCA()
scores = pca.fit_transform(Xz)
expl = pca.explained_variance_ratio_

df_clean["PC1"] = scores[:, 0]
df_clean["PC2"] = scores[:, 1]

print(f"Explained variance: PC1={expl[0]*100:.2f}%, PC2={expl[1]*100:.2f}%")
print(f"Combined PC1+PC2={100*(expl[0]+expl[1]):.2f}%")
print(f"PCA feature count: {len(feature_cols)}")

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
)

# Save core outputs
df_clean[["Sample", "Strain", "Medium", "Replicate", "PC1", "PC2"]].to_csv(
    OUT_DIR / "PC_scores.csv", index=False
)
loadings[["PC1", "PC2"]].to_csv(OUT_DIR / "PC_loadings_PC1_PC2.csv")

# =========================
# Two-way ANOVA on PC1 and PC2
# =========================
m1 = smf.ols("PC1 ~ C(Medium) * C(Strain)", data=df_clean).fit()
a1 = sm.stats.anova_lm(m1, typ=2)
m2 = smf.ols("PC2 ~ C(Medium) * C(Strain)", data=df_clean).fit()
a2 = sm.stats.anova_lm(m2, typ=2)

a1.to_csv(OUT_DIR / "ANOVA_PC1.csv")
a2.to_csv(OUT_DIR / "ANOVA_PC2.csv")

print("\nANOVA PC1:\n", a1)
print("\nANOVA PC2:\n", a2)

# =========================
# Matplotlib style
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42,
})

# =========================
# FIGURE 4: PCA scatter
# =========================

strain_order = ["BW25113", "amiA", "ompR", "ydgD"]

colour_map = {
    "BW25113": "#FF9900",  # orange
    "amiA":    "#0000FF",  # blue
    "ompR":    "#008000",  # green
    "ydgD":    "#800080",  # purple
}

marker_map = {
    "BW25113": "o",
    "amiA":    "s",
    "ompR":    "^",
    "ydgD":    "D",
}

display_name = {
    "BW25113": "BW25113",
    "amiA": r"$\Delta amiA$",
    "ompR": r"$\Delta ompR$",
    "ydgD": r"$\Delta ydgD$",
}


def add_covariance_ellipse(ax, x, y, colour, medium, n_std=2.0):
    """Draw a covariance ellipse representing approximately 2 SD dispersion."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return

    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    vals = np.maximum(vals, 0)
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    if medium == "LB":
        ellipse = Ellipse(
            (np.mean(x), np.mean(y)),
            width=width,
            height=height,
            angle=angle,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.22,
            linewidth=2.0,
            linestyle="-",
            zorder=1,
        )
    else:
        ellipse = Ellipse(
            (np.mean(x), np.mean(y)),
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=colour,
            linewidth=2.0,
            linestyle="--",
            zorder=1,
        )

    ax.add_patch(ellipse)


fig, ax = plt.subplots(figsize=(8.0, 5.6))

for strain in strain_order:
    if strain not in df_clean["Strain"].unique():
        continue

    colour = colour_map[strain]
    marker = marker_map[strain]

    for medium in ["LB", "M9"]:
        sub = df_clean[
            (df_clean["Strain"] == strain) &
            (df_clean["Medium"] == medium)
        ]

        if len(sub) == 0:
            continue

        add_covariance_ellipse(
            ax,
            sub["PC1"].to_numpy(),
            sub["PC2"].to_numpy(),
            colour,
            medium,
            n_std=2.0,
        )

        if medium == "LB":
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                marker=marker,
                s=70,
                facecolors=colour,
                edgecolors=colour,
                linewidths=1.2,
                zorder=3,
            )
        else:
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                marker=marker,
                s=70,
                facecolors="white",
                edgecolors=colour,
                linewidths=1.7,
                zorder=3,
            )

legend_handles = []
for strain in strain_order:
    colour = colour_map[strain]
    marker = marker_map[strain]
    name = display_name[strain]

    legend_handles.append(
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor=colour,
            markeredgecolor=colour,
            markeredgewidth=1.2,
            markersize=8,
            label=f"{name} – LB",
        )
    )
    legend_handles.append(
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=1.5,
            markersize=8,
            label=f"{name} – M9",
        )
    )

ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
ax.legend(
    handles=legend_handles,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

plt.tight_layout()
plt.savefig(OUT_DIR / "Figure4_PCA.pdf")
plt.savefig(OUT_DIR / "Figure4_PCA.png", dpi=DPI)
plt.close()

# =========================
# FIGURE 5: Interaction plots (mean ± SEM)
# =========================
summary = (
    df_clean.groupby(["Strain", "Medium"])
    .agg(
        PC1_mean=("PC1", "mean"),
        PC1_sem=("PC1", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        PC2_mean=("PC2", "mean"),
        PC2_sem=("PC2", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

for ax, mean_col, sem_col, ylabel, panel in [
    (axes[0], "PC1_mean", "PC1_sem", "Mean PC1 ± SEM", "A"),
    (axes[1], "PC2_mean", "PC2_sem", "Mean PC2 ± SEM", "B"),
]:
    for strain in strain_order:
        sub = summary[summary["Strain"] == strain].copy()
        if sub.empty:
            continue

        sub["Medium"] = pd.Categorical(
            sub["Medium"],
            categories=["LB", "M9"],
            ordered=True,
        )
        sub = sub.sort_values("Medium")
        sub = sub[sub["Medium"].isin(["LB", "M9"])]

        ax.errorbar(
            sub["Medium"].astype(str),
            sub[mean_col],
            yerr=sub[sem_col],
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=3,
            color=colour_map[strain],
            label=display_name[strain],
        )

    ax.set_xlabel("Medium")
    ax.set_ylabel(ylabel)
    ax.text(
        -0.08, 1.03, panel,
        transform=ax.transAxes,
        fontsize=16,
        va="bottom",
        ha="left",
    )
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

plt.tight_layout()
plt.savefig(OUT_DIR / "Figure5_StrainxMedium.pdf")
plt.savefig(OUT_DIR / "Figure5_StrainxMedium.png", dpi=DPI)
plt.close()

def format_feature_label(name: str) -> str:
    """Convert internal feature names to concise display labels."""
    labels = {
        "shannon_entropy": "Shannon",
        "renyi_entropy_a2": "Renyi",
        "boxcount_D": "RDBC",
        "lacunarity_mean": "Lacunarity",
        "glcm_asm_mean": "GLCM ASM",
        "glcm_contrast_mean": "GLCM contrast",
        "glcm_correlation_mean": "GLCM correlation",
        "glcm_dissimilarity_mean": "GLCM dissimilarity",
        "glcm_homogeneity_mean": "GLCM homogeneity",
        "LLL_rel_energy": "LLL relative energy",
        "LLH_rel_energy": "LLH relative energy",
        "LHL_rel_energy": "LHL relative energy",
        "LHH_rel_energy": "LHH relative energy",
        "HLL_rel_energy": "HLL relative energy",
        "HLH_rel_energy": "HLH relative energy",
        "HHL_rel_energy": "HHL relative energy",
        "HHH_rel_energy": "HHH relative energy",
    }
    return labels.get(name, str(name).replace("_", " "))


# =========================
# FIGURE 6: Principal-component loadings
# =========================
# Display all 17 analysis features ordered by loading.

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))

for ax, pc, panel in [
    (axes[0], "PC1", "A"),
    (axes[1], "PC2", "B"),
]:
    vals = loadings[pc].sort_values()
    labels = [format_feature_label(v) for v in vals.index]
    y = np.arange(len(vals))

    ax.barh(y, vals.values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{pc} loading")
    ax.text(
        -0.06,
        1.03,
        panel,
        transform=ax.transAxes,
        fontsize=16,
        va="bottom",
        ha="left",
    )

plt.tight_layout()
plt.savefig(OUT_DIR / "Figure6_Loadings.pdf")
plt.savefig(OUT_DIR / "Figure6_Loadings.png", dpi=DPI)
plt.close()

# =========================
# FIGURE 7: Heatmap of mean z-scored features by Strain × Medium
# =========================
Z = pd.DataFrame(Xz, columns=feature_cols, index=df_clean.index)

group_means = (
    Z.join(df_clean[["Strain", "Medium"]])
    .groupby(["Strain", "Medium"])
    .mean()
)

# Apply the predefined experimental-group order.
desired_index = [
    ("BW25113", "LB"),
    ("BW25113", "M9"),
    ("amiA", "LB"),
    ("amiA", "M9"),
    ("ompR", "LB"),
    ("ompR", "M9"),
    ("ydgD", "LB"),
    ("ydgD", "M9"),
]
existing_index = [idx for idx in desired_index if idx in group_means.index]
group_means = group_means.loc[existing_index]

# Order features by absolute PC1 loading.
ordered_cols = loadings["PC1"].abs().sort_values(ascending=False).index.tolist()
group_means = group_means[ordered_cols]

# Format feature labels for display.
feature_labels = [format_feature_label(c) for c in group_means.columns]

# Format experimental-group labels for display.
row_display = {
    "BW25113": "BW25113",
    "amiA": r"$\Delta amiA$",
    "ompR": r"$\Delta ompR$",
    "ydgD": r"$\Delta ydgD$",
}
yticklabels = [
    f"{row_display[strain]} – {medium}"
    for strain, medium in group_means.index
]

fig, ax = plt.subplots(figsize=(12.5, 4.8))
im = ax.imshow(
    group_means.to_numpy(),
    aspect="auto",
    interpolation="nearest",
)

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Mean z-score")

ax.set_yticks(range(len(yticklabels)))
ax.set_yticklabels(yticklabels)

ax.set_xticks(range(len(feature_labels)))
ax.set_xticklabels(feature_labels, rotation=90)

plt.tight_layout()
plt.savefig(OUT_DIR / "Figure7_Heatmap.pdf")
plt.savefig(OUT_DIR / "Figure7_Heatmap.png", dpi=DPI)
plt.close()

# Save the displayed heatmap matrix using the formatted feature labels.
heatmap_export = group_means.copy()
heatmap_export.columns = feature_labels
heatmap_export.index = yticklabels
heatmap_export.to_csv(OUT_DIR / "Figure7_heatmap_matrix_groupmeans_z.csv")

# =========================
# SUPPLEMENT: Per-feature two-way ANOVA + BH-FDR
# =========================
anova_rows = []

for feat in feature_cols:
    tmp = df_clean[["Medium", "Strain", feat]].dropna().copy()
    if tmp[feat].nunique() <= 1:
        continue

    # Use Q(feat) to safely handle any special characters in column names
    model = smf.ols(f"Q('{feat}') ~ C(Medium) * C(Strain)", data=tmp).fit()
    an = sm.stats.anova_lm(model, typ=2)

    def get_val(term, col):
        return float(an.loc[term, col]) if term in an.index else np.nan

    anova_rows.append({
        "feature": feat,
        "F_medium": get_val("C(Medium)", "F"),
        "p_medium": get_val("C(Medium)", "PR(>F)"),
        "F_strain": get_val("C(Strain)", "F"),
        "p_strain": get_val("C(Strain)", "PR(>F)"),
        "F_interaction": get_val("C(Medium):C(Strain)", "F"),
        "p_interaction": get_val("C(Medium):C(Strain)", "PR(>F)"),
    })

anova_feat = pd.DataFrame(anova_rows)

anova_feat["q_medium"] = bh_fdr(anova_feat["p_medium"].to_numpy())
anova_feat["q_strain"] = bh_fdr(anova_feat["p_strain"].to_numpy())
anova_feat["q_interaction"] = bh_fdr(anova_feat["p_interaction"].to_numpy())

anova_feat.to_csv(OUT_DIR / "Supp_feature_ANOVA_FDR.csv", index=False)

# Additional tables ranked by adjusted p-value
anova_feat.sort_values("q_medium").head(30).to_csv(OUT_DIR / "Supp_top30_medium_effect.csv", index=False)
anova_feat.sort_values("q_interaction").head(30).to_csv(OUT_DIR / "Supp_top30_interaction_effect.csv", index=False)

print("\nSignificant features at FDR q<0.05:")
print("  Medium:", int((anova_feat["q_medium"] < 0.05).sum()))
print("  Strain:", int((anova_feat["q_strain"] < 0.05).sum()))
print("  Interaction:", int((anova_feat["q_interaction"] < 0.05).sum()))

print("\nAll outputs saved to:", OUT_DIR)
