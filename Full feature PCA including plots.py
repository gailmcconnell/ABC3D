# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:26:35 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
Principal Component Analysis of ABC3D - Analysis of Biofilm Complexity in 3D
"""

# -*- coding: utf-8 -*-
"""
Full-feature PCA + 2-way ANOVA on PC scores (Medium, Strain, Interaction)

Assumes your Excel file has a first column with sample IDs like:
  BW25113_LB_1
  BW25113_M9_ompR_3
etc.
and the remaining columns are numeric features (GLCM, wavelets, entropy, etc.)

Outputs:
- Explained variance for PCs
- Loadings for PC1/PC2
- Two-way ANOVA tables for PC1 and PC2
- PCA scatter (PC1 vs PC2) colored by Strain, marker by Medium
- Interaction plots for PC1 and PC2 (means ± SEM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import statsmodels.api as sm
import statsmodels.formula.api as smf


# -------------------------
# 1) Load data
# -------------------------
file_path = r"C:\Users\acp98112\OneDrive - University of Strathclyde\Documents\Research\ABC3D\Keio data results\For PCA.xlsx"
df = pd.read_excel(file_path)

# Rename first column to Sample if needed
if "Sample" not in df.columns:
    df = df.rename(columns={df.columns[0]: "Sample"})

# Clean column names (spaces -> underscores)
df.columns = df.columns.str.replace(" ", "_")


# -------------------------
# 2) Parse factors: Medium + Strain
# -------------------------
df["Medium"] = df["Sample"].astype(str).apply(lambda x: "LB" if "_LB_" in x else "M9")

def extract_strain(sample: str) -> str:
    sample = str(sample)
    if "amiA" in sample:
        return "amiA"
    if "ompR" in sample:
        return "ompR"
    if "ydgD" in sample:
        return "ydgD"
    return "WT"

df["Strain"] = df["Sample"].astype(str).apply(extract_strain)

# Set an explicit order (optional, but helps consistency)
df["Strain"] = pd.Categorical(df["Strain"], categories=["WT", "amiA", "ompR", "ydgD"], ordered=True)
df["Medium"] = pd.Categorical(df["Medium"], categories=["LB", "M9"], ordered=True)


# -------------------------
# 3) Select all numeric feature columns
# -------------------------
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found. Check the Excel sheet.")

# Drop rows that are completely empty across features
df = df.dropna(how="all", subset=feature_cols).copy()

# Fill remaining missing feature values with column means (robust for partial missingness)
df[feature_cols] = df[feature_cols].apply(lambda c: c.fillna(c.mean()))

# Sanity checks
print("N samples:", len(df))
print("N features:", len(feature_cols))
print("Any NaNs left in features?", df[feature_cols].isna().any().any())


# -------------------------
# 4) Standardise + PCA
# -------------------------
X = df[feature_cols].to_numpy(dtype=float)

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA()  # keep all PCs
scores_all = pca.fit_transform(Xz)

expl = pca.explained_variance_ratio_
print("\nExplained variance (first 10 PCs):")
for i in range(min(10, len(expl))):
    print(f"PC{i+1}: {expl[i]*100:.2f}%")

# Keep PC1 and PC2 for plots + ANOVA
df["PC1"] = scores_all[:, 0]
df["PC2"] = scores_all[:, 1]

# Loadings for interpretation (PC1/PC2)
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(len(feature_cols))]
)
print("\nTop loadings by absolute value (PC1):")
print(loadings["PC1"].abs().sort_values(ascending=False).head(15))
print("\nTop loadings by absolute value (PC2):")
print(loadings["PC2"].abs().sort_values(ascending=False).head(15))


# -------------------------
# 5) Two-way ANOVA on PC1 and PC2
# -------------------------
m_pc1 = smf.ols("PC1 ~ C(Medium) * C(Strain)", data=df).fit()
anova_pc1 = sm.stats.anova_lm(m_pc1, typ=2)

m_pc2 = smf.ols("PC2 ~ C(Medium) * C(Strain)", data=df).fit()
anova_pc2 = sm.stats.anova_lm(m_pc2, typ=2)

print("\nTwo-way ANOVA on PC1:")
print(anova_pc1)
print("\nTwo-way ANOVA on PC2:")
print(anova_pc2)


# -------------------------
# 6) PCA scatter: PC1 vs PC2
#    colour = Strain, marker = Medium
# -------------------------
marker_map = {"LB": "o", "M9": "s"}

plt.figure(figsize=(7, 5))
for strain in df["Strain"].cat.categories:
    for medium in df["Medium"].cat.categories:
        sub = df[(df["Strain"] == strain) & (df["Medium"] == medium)]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            marker=marker_map[str(medium)],
            label=f"{strain}-{medium}"
        )

plt.xlabel(f"PC1 ({expl[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({expl[1]*100:.1f}%)")
plt.title("Full-feature PCA (standardized): PC1 vs PC2")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# -------------------------
# 7) Interaction plots (means ± SEM)
# -------------------------
def mean_sem(x: pd.Series) -> tuple[float, float]:
    x = x.dropna().to_numpy()
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

summary = (
    df.groupby(["Strain", "Medium"])
      .agg(PC1_mean=("PC1", "mean"),
           PC1_sem=("PC1", lambda s: s.std(ddof=1)/np.sqrt(len(s))),
           PC2_mean=("PC2", "mean"),
           PC2_sem=("PC2", lambda s: s.std(ddof=1)/np.sqrt(len(s))))
      .reset_index()
)

# PC1 interaction
plt.figure(figsize=(6, 4))
for strain in df["Strain"].cat.categories:
    sub = summary[summary["Strain"] == strain].sort_values("Medium")
    plt.errorbar(sub["Medium"].astype(str), sub["PC1_mean"], yerr=sub["PC1_sem"], marker="o", capsize=4, label=strain)

plt.xlabel("Medium")
plt.ylabel("Mean PC1 ± SEM")
plt.title("Interaction Plot: PC1 (full features)")
plt.legend()
plt.tight_layout()
plt.show()

# PC2 interaction
plt.figure(figsize=(6, 4))
for strain in df["Strain"].cat.categories:
    sub = summary[summary["Strain"] == strain].sort_values("Medium")
    plt.errorbar(sub["Medium"].astype(str), sub["PC2_mean"], yerr=sub["PC2_sem"], marker="o", capsize=4, label=strain)

plt.xlabel("Medium")
plt.ylabel("Mean PC2 ± SEM")
plt.title("Interaction Plot: PC2 (full features)")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# 8) Save outputs
# -------------------------
out_dir = r"C:\Users\acp98112\OneDrive - University of Strathclyde\Documents\Research\ABC3D"
df_out = df[["Sample", "Strain", "Medium", "PC1", "PC2"]].copy()
df_out.to_csv(out_dir + r"\full_feature_pca_scores.csv", index=False)

loadings[["PC1", "PC2"]].to_csv(out_dir + r"\full_feature_pca_loadings_PC1_PC2.csv")
print("\nSaved:")
print(out_dir + r"\full_feature_pca_scores.csv")
print(out_dir + r"\full_feature_pca_loadings_PC1_PC2.csv")
