# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:09:28 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

# Creates:
#   Fig2: PCA scatter (PC1 vs PC2)
#   Fig3: Interaction plots for PC1 and PC2 (mean ± SEM)
#   Fig4: PC1/PC2 loading barplots (top ± and top absolute)
#   Fig5: Heatmap of mean z-scored features by Strain×Medium
#   Supplement: per-feature two-way ANOVA + BH-FDR tables
#
# INPUT: For PCA.xlsx (must contain a sample/name column like "Type" or "Sample")
# OUTPUT: PDFs + 600dpi PNGs + CSV tables in OUT_DIR

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from patsy.builtins import Q

# =========================
# USER SETTINGS - AMEND TO ACCOMMODATE APPROPRIATE INPUT AND OUTPUT DIRECTORIES
# =========================
IN_XLSX = r"C:\Users\For PCA.xlsx"
OUT_DIR = r"C:\Users\paper_figures"
DPI = 600

# If your file uses different naming, adjust these parsers:
STRAIN_KEYS = ["amiA", "ompR", "ydgD"]  # otherwise BW25113
MEDIUM_KEYS = ["LB", "M9"]

# If you want to exclude some numeric columns from PCA (e.g., IDs), put them here:
EXCLUDE_NUMERIC_COLS = set([
    # "some_id_col",
])

# =========================
# Helper functions
# =========================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def savefig(name_base: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name_base}.pdf"))
    plt.savefig(os.path.join(OUT_DIR, f"{name_base}.png"), dpi=DPI)
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

def parse_medium(sample: str) -> str:
    s = str(sample)
    # robust: look for token boundaries or underscores
    if "_LB_" in s or re.search(r"(^|[_\-\s])LB([_\-\s]|$)", s):
        return "LB"
    if "_M9_" in s or re.search(r"(^|[_\-\s])M9([_\-\s]|$)", s):
        return "M9"
    # fallback: substring
    if "LB" in s:
        return "LB"
    if "M9" in s:
        return "M9"
    return "UNK"

def parse_strain(sample: str) -> str:
    s = str(sample)
    for k in STRAIN_KEYS:
        if k in s:
            return k
    return "BW25113"

def parse_replicate(sample: str):
    s = str(sample)
    # Try common patterns: "_1", "stack1", etc.
    m = re.search(r"(?:stack|rep|r)(\d+)$", s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"[_\-](\d+)$", s)
    if m:
        return m.group(1)
    return None

# =========================
# Load data
# =========================
ensure_outdir(OUT_DIR)
df = pd.read_excel(IN_XLSX)

# Identify the sample/name column
if "Sample" in df.columns:
    sample_col = "Sample"
elif "Type" in df.columns:
    sample_col = "Type"
elif "Unnamed: 0" in df.columns:
    sample_col = "Unnamed: 0"
else:
    sample_col = df.columns[0]

df = df.rename(columns={sample_col: "Sample"}).copy()
df.columns = df.columns.str.replace(" ", "_")

# Add factors
df["Medium"] = df["Sample"].apply(parse_medium)
df["Strain"] = df["Sample"].apply(parse_strain)
df["Replicate"] = df["Sample"].apply(parse_replicate)

# Numeric feature columns
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c not in EXCLUDE_NUMERIC_COLS]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found in the input file.")

# Drop rows with missing values in any feature (paper-friendly default)
df_clean = df.dropna(subset=feature_cols).copy()

# Save the cleaned input for traceability
df_clean.to_csv(os.path.join(OUT_DIR, "input_cleaned.csv"), index=False)

# =========================
# PCA (full features, standardized)
# =========================
X = df_clean[feature_cols].to_numpy(dtype=float)
Xz = StandardScaler().fit_transform(X)

pca = PCA()
scores = pca.fit_transform(Xz)
expl = pca.explained_variance_ratio_

df_clean["PC1"] = scores[:, 0]
df_clean["PC2"] = scores[:, 1]

print(f"Explained variance: PC1={expl[0]*100:.2f}%, PC2={expl[1]*100:.2f}%")

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(len(feature_cols))]
)

# Save core outputs
df_clean[["Sample", "Strain", "Medium", "Replicate", "PC1", "PC2"]].to_csv(
    os.path.join(OUT_DIR, "PC_scores.csv"), index=False
)
loadings[["PC1", "PC2"]].to_csv(os.path.join(OUT_DIR, "PC_loadings_PC1_PC2.csv"))

# =========================
# Two-way ANOVA on PC1 and PC2
# =========================
m1 = smf.ols("PC1 ~ C(Medium) * C(Strain)", data=df_clean).fit()
a1 = sm.stats.anova_lm(m1, typ=2)
m2 = smf.ols("PC2 ~ C(Medium) * C(Strain)", data=df_clean).fit()
a2 = sm.stats.anova_lm(m2, typ=2)

a1.to_csv(os.path.join(OUT_DIR, "ANOVA_PC1.csv"))
a2.to_csv(os.path.join(OUT_DIR, "ANOVA_PC2.csv"))

print("\nANOVA PC1:\n", a1)
print("\nANOVA PC2:\n", a2)

# =========================
# Matplotlib style (journal-friendly)
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
# FIGURE 2: PCA scatter
# =========================
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# 4 strains x 2 media = 8 combos
strain_order = ["BW25113", "amiA", "ompR", "ydgD"]
medium_order = ["LB", "M9"]

# Marker SHAPE per strain (edit if you want different shapes)
strain_marker_map = {
    "BW25113": "o",  # circle
    "amiA": "s",     # square
    "ompR": "^",     # triangle up
    "ydgD": "D",     # diamond
}

# Colour per strain (edit to your preference)
strain_color_map = {
    "BW25113": "orange",
    "amiA": "blue",
    "ompR": "green",
    "ydgD": "purple",
}

def medium_facecolor(medium: str, strain_color: str):
    # LB filled, M9 hollow (matches your example)
    if medium == "LB":
        return strain_color
    if medium == "M9":
        return "none"
    return "none"

def add_confidence_ellipse(ax, x, y, n_std=2.0, edgecolor="black",
                           facecolor="none", alpha=0.12, linewidth=1.6,
                           linestyle="-", zorder=1):
    """
    Covariance-based confidence ellipse for points (x, y).
    n_std=2 is a common visual convention (~95% if roughly normal).
    Only draws if there are >=3 points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return None

    cov = np.cov(x, y)
    if not np.isfinite(cov).all():
        return None

    # Eigen-decomposition -> orientation and axis lengths
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Degenerate covariance -> skip
    if np.any(vals <= 0):
        return None

    width = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ell = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(ell)
    return ell

fig, ax = plt.subplots(figsize=(7.6, 5.2))

# Plot points + ellipses (only for combos that exist in data)
for strain in strain_order:
    for medium in medium_order:
        sub = df_clean[(df_clean["Strain"] == strain) & (df_clean["Medium"] == medium)]
        if len(sub) == 0:
            continue

        col = strain_color_map.get(strain, "black")
        mk = strain_marker_map.get(strain, "o")
        fc = medium_facecolor(medium, col)

        # points
        ax.scatter(
            sub["PC1"], sub["PC2"],
            marker=mk,
            s=55,
            facecolors=fc,
            edgecolors=col,
            linewidths=1.2,
            zorder=3,
        )

        # bounding ovals:
        # LB: solid + light fill; M9: dashed outline only
        if medium == "LB":
            add_confidence_ellipse(
                ax,
                sub["PC1"].to_numpy(),
                sub["PC2"].to_numpy(),
                n_std=2.0,
                edgecolor=col,
                facecolor=col,
                alpha=0.12,
                linewidth=1.6,
                linestyle="-",
                zorder=2,
            )
        else:  # M9
            add_confidence_ellipse(
                ax,
                sub["PC1"].to_numpy(),
                sub["PC2"].to_numpy(),
                n_std=2.0,
                edgecolor=col,
                facecolor="none",
                alpha=1.0,
                linewidth=1.6,
                linestyle="--",
                zorder=2,
            )

# Build a legend with ALL 8 combinations (even if some combos have no data)
legend_handles = []
legend_labels = []
for strain in strain_order:
    col = strain_color_map.get(strain, "black")
    mk = strain_marker_map.get(strain, "o")
    for medium in medium_order:
        fc = medium_facecolor(medium, col)
        h = Line2D(
            [0], [0],
            marker=mk,
            linestyle="None",
            markerfacecolor=fc,
            markeredgecolor=col,
            markeredgewidth=1.2,
            markersize=7,
        )
        legend_handles.append(h)
        legend_labels.append(f"{strain} - {medium}")

ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
ax.set_title("Full-feature PCA (standardized): PC1 vs PC2")
ax.legend(
    legend_handles, legend_labels,
    bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
)

savefig("Fig2_PCA_PC1_PC2_17Feb26")

# =========================
# FIGURE 3: Interaction plots (mean ± SEM)
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

def interaction_plot(mean_col, sem_col, ylabel, title, fname):
    plt.figure(figsize=(6.2, 4.0))
    for strain in sorted(summary["Strain"].unique()):
        sub = summary[summary["Strain"] == strain].copy()
        sub["Medium"] = pd.Categorical(sub["Medium"], categories=["LB", "M9"], ordered=True)
        sub = sub.sort_values("Medium")
        sub = sub[sub["Medium"].isin(["LB", "M9"])]
        if len(sub) == 0:
            continue
        plt.errorbar(sub["Medium"].astype(str), sub[mean_col], yerr=sub[sem_col],
                     marker="o", capsize=4, label=strain)
    plt.xlabel("Medium")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fname)

interaction_plot("PC1_mean", "PC1_sem", "Mean PC1 ± SEM", "Interaction Plot: PC1 (full features)", "Fig3_PC1_interaction_17Feb26")
interaction_plot("PC2_mean", "PC2_sem", "Mean PC2 ± SEM", "Interaction Plot: PC2 (full features)", "Fig3_PC2_interaction_17Feb26")

# =========================
# FIGURE 4: Loadings plots (PC1 & PC2)
# =========================
def plot_top_loadings(pc: str, top_n: int = 10):
    s = loadings[pc].copy()
    top_pos = s.sort_values(ascending=False).head(top_n)
    top_neg = s.sort_values(ascending=True).head(top_n)
    plot_s = pd.concat([top_neg, top_pos]).sort_values()

    plt.figure(figsize=(7.2, 5.4))
    plt.barh(plot_s.index, plot_s.values)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"{pc} loading")
    plt.title(f"Top ±{top_n} feature loadings ({pc})")
    savefig(f"Fig4_{pc}_top_pm{top_n}")

plot_top_loadings("PC1", top_n=10)
plot_top_loadings("PC2", top_n=10)

def plot_top_abs_loadings(pc: str, top_n: int = 15):
    s = loadings[pc].copy()
    top = s.abs().sort_values(ascending=False).head(top_n).index
    vals = s.loc[top].sort_values()

    plt.figure(figsize=(7.2, 5.8))
    plt.barh(vals.index, vals.values)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"{pc} loading")
    plt.title(f"Top {top_n} absolute loadings ({pc})")
    savefig(f"17Feb26Supp_{pc}_absloadings_top{top_n}")

plot_top_abs_loadings("PC1", top_n=15)
plot_top_abs_loadings("PC2", top_n=15)

# =========================
# FIGURE 5: Heatmap of mean z-scored features by strain × medium
# =========================
Z = pd.DataFrame(Xz, columns=feature_cols, index=df_clean.index)

group_means = (
    Z.join(df_clean[["Strain", "Medium"]])
    .groupby(["Strain", "Medium"])
    .mean()
)

# Order features by absolute PC1 loading (makes heatmap easier to read)
ordered_cols = loadings["PC1"].abs().sort_values(ascending=False).index.tolist()
group_means = group_means[ordered_cols]

plt.figure(figsize=(12.5, 4.8))
im = plt.imshow(group_means.to_numpy(), aspect="auto", interpolation="nearest")
plt.colorbar(im, label="Mean z-score")

yticklabels = [f"{s}-{m}" for (s, m) in group_means.index]
plt.yticks(range(len(yticklabels)), yticklabels)

# show ~25 x tick labels max
step = max(1, len(group_means.columns) // 25)
xticks = list(range(0, len(group_means.columns), step))
plt.xticks(xticks, [group_means.columns[i] for i in xticks], rotation=90)

plt.title("Feature heatmap (mean z-score) by Strain × Medium")
savefig("Fig5_feature_heatmap_groupmeans_17Feb26")

group_means.to_csv(os.path.join(OUT_DIR, "Fig5_heatmap_matrix_groupmeans_z.csv"))

# =========================
# SUPPLEMENT: Per-feature two-way ANOVA + BH-FDR
# =========================
anova_rows = []

for feat in feature_cols:
    tmp = df_clean[["Medium", "Strain", feat]].dropna().copy()

    # Skip if not enough variance
    if tmp[feat].nunique() < 2:
        continue

    try:
        model = smf.ols(f"Q('{feat}') ~ C(Medium) * C(Strain)", data=tmp).fit()
        an = sm.stats.anova_lm(model, typ=2)
        for term in ["C(Medium)", "C(Strain)", "C(Medium):C(Strain)"]:
            if term in an.index:
                anova_rows.append({
                    "Feature": feat,
                    "Term": term,
                    "F": an.loc[term, "F"],
                    "p": an.loc[term, "PR(>F)"],
                })
    except Exception as e:
        print(f"ANOVA failed for {feat}: {e}")

anova_df = pd.DataFrame(anova_rows)
if len(anova_df) > 0:
    # BH-FDR per term (common reporting)
    anova_df["q"] = np.nan
    for term in anova_df["Term"].unique():
        idx = anova_df["Term"] == term
        anova_df.loc[idx, "q"] = bh_fdr(anova_df.loc[idx, "p"].to_numpy())

    anova_df = anova_df.sort_values(["Term", "q", "p"])
    anova_df.to_csv(os.path.join(OUT_DIR, "Supp_feature_ANOVA_BH_FDR.csv"), index=False)
else:
    print("No per-feature ANOVA results generated (possibly no valid features).")
