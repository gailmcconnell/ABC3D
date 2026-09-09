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
- 3D discrete relative wavelet transform (db2) energy features:
  - LLL, LLH, LHL, LHH
  - HLL, HLH, HHL, HHH
- Biomass occupancy metrics
- PCA-ready CSV export

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gailmcconnell/ABC3D.git
cd ABC3D
Install dependencies:

pip install -r requirements.txt
Or using conda:

conda env create -f environment.yml
conda activate abc3d

--

Reproducibility statement

ABC3D analyses are deterministic.

- Fixed random seed for Otsu subsampling
- No stochastic feature extraction steps
- Version logging included
- Example dataset provided
- Example output included

Results from the manuscript can be reproduced using the provided scripts.

## Metadata

ABC3D uses an explicit metadata file rather than inferring experimental labels from image filenames.

The required columns are:

- `filename`
- `strain`
- `medium`
- `relative_path`

Each row corresponds to one 3D TIFF image stack.

For example:

```text
filename,strain,medium,relative_path
BW25113_5um_stack1.tif,BW25113,LB,LB/BW25113/BW25113_5um_stack1.tif

```markdown
## Running ABC3D

Place the image data and `metadata.csv` inside the data directory.

Run:

```bash
python ABC3D.py

Results are written to results/ABC3D_features.csv
