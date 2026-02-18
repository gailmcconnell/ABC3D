# ABC3D
Analysis of Biofilm Complexity in 3D - a Python framework for extraction of fractal, textural, and statistical descriptors of multidimensional image datasets


It extracts fractal, textural, statistical, and wavelet-based descriptors from volumetric OME-TIFF image stacks and exports analysis-ready feature matrices for downstream multivariate analysis.

---

## Features

ABC3D computes:

- 3D box-counting fractal dimension (RDBC)
- Lacunarity across dyadic spatial scales
- Shannon entropy
- Rényi entropy (α = 2)
- 2.5D grey-level co-occurrence matrix (GLCM) features:
  - Angular second moment
  - Contrast
  - Correlation
  - Dissimilarity
  - Energy
  - Homogeneity
- 3D discrete wavelet transform (db2) energy features:
  - LLL, LLH, LHL, LHH
  - HLL, HLH, HHL, HHH
- Biomass occupancy metrics
- PCA-ready CSV export

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ABC3D-biofilm-analysis.git
cd ABC3D-biofilm-analysis
Install dependencies:

pip install -r requirements.txt
Or using conda:

conda env create -f environment.yml
conda activate abc3d
