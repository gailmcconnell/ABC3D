# -*- coding: utf-8 -*-
"""
Created on Wed Sept 9 09:29:00 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
ABC3D - Analysis of Biofilm Complexity in 3D
Metadata-driven batch analysis with relative wavelet energy
"""

from pathlib import Path
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.stats import kurtosis
from scipy import fft as spfft
import pywt


# =========================
# USER SETTINGS
# =========================
RUN_MODE = "file"     # "file" | "synthetic" | "both"

# Root folder containing metadata.csv and the LB/M9 subfolders.
# For the public repository, use a relative path such as Path("./data").
DATA_DIR = Path("./data")
METADATA_FILE = DATA_DIR / "metadata.csv"
OUT_DIR = Path("./results")

INVERT = False
THRESHOLD = None
MIN_OBJECT_SIZE = 0

ENTROPY_BINS = 256
RENYI_ALPHA = 2.0
GLCM_LEVELS = 32
GLCM_MIN_PIXELS_PER_SLICE = 500

# FFT controls
FFT_HF_CUT = 0.25
FFT_CROP_MARGIN = 8

EDGE_PERCENTILE = 90.0
WAVELET = "db2"

SAVE_CSV = True


# ---------------- I/O ----------------
def load_volume_as_zyx(path: Path, t_index: int = 0, c_index: int = 0) -> np.ndarray:
    arr = np.asarray(tiff.imread(str(path)))

    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        raise ValueError(f"{path.name}: 2D image; need 3D stack.")
    if arr.ndim == 5:
        vol = arr[t_index, c_index]
        if vol.ndim != 3:
            raise ValueError(f"{path.name}: after selecting T,C got {vol.shape}, expected 3D.")
        return np.asarray(vol, dtype=np.float32)
    if arr.ndim == 4:
        candidate = arr[0]
        if candidate.ndim == 3:
            return np.asarray(candidate, dtype=np.float32)

        shape = arr.shape
        dims = list(range(4))
        sorted_dims = sorted(dims, key=lambda i: shape[i], reverse=True)
        y_dim, x_dim = sorted_dims[0], sorted_dims[1]
        remaining = [d for d in dims if d not in (y_dim, x_dim)]
        z_dim = remaining[int(np.argmax([shape[d] for d in remaining]))]
        other_dim = remaining[int(np.argmin([shape[d] for d in remaining]))]

        slicer = [slice(None)] * 4
        slicer[other_dim] = 0
        reduced = arr[tuple(slicer)]
        kept_dims = [d for d in dims if d != other_dim]
        pos = {orig_d: i for i, orig_d in enumerate(kept_dims)}
        vol = np.moveaxis(reduced, [pos[z_dim], pos[y_dim], pos[x_dim]], [0, 1, 2])

        if vol.ndim != 3:
            raise ValueError(f"{path.name}: could not reduce to 3D cleanly, got {vol.shape}")
        return np.asarray(vol, dtype=np.float32)

    raise ValueError(f"{path.name}: unsupported ndim={arr.ndim}, shape={arr.shape}")


# ---------------- ROI mask ----------------
def make_binary_mask(vol: np.ndarray, threshold=None, invert=False, min_object_size=0) -> np.ndarray:
    if vol.dtype != np.float32:
        v = vol.astype(np.float32, copy=False)
    else:
        v = vol

    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax <= vmin:
        mask = (v == 0) if invert else (v > 0)
        mask = mask.astype(bool, copy=False)
        if min_object_size and min_object_size > 0:
            mask = remove_small_objects(mask, min_size=int(min_object_size))
        return mask

    if threshold is None:
        rng = np.random.default_rng(0)
        n_total = v.size
        n_samp = min(1_000_000, n_total)
        if n_samp < n_total:
            idx = rng.choice(n_total, size=n_samp, replace=False)
            sample = v.reshape(-1)[idx]
        else:
            sample = v.reshape(-1)
        sample = sample[np.isfinite(sample)]
        thr = threshold_otsu(sample) if sample.size else (vmin + vmax) / 2.0
    else:
        thr = float(threshold)

    mask = (v < thr) if invert else (v > thr)
    mask = mask.astype(bool, copy=False)

    if min_object_size and min_object_size > 0:
        mask = remove_small_objects(mask, min_size=int(min_object_size))

    return mask


# ---------------- Helpers ----------------
def _safe_mean(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

def _safe_std(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=0)) if a.size else float("nan")

def _safe_median(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


# ---------------- 3D box counting ----------------
def pad_to_multiple(arr: np.ndarray, k: int) -> np.ndarray:
    z, y, x = arr.shape
    pad_z = (k - (z % k)) % k
    pad_y = (k - (y % k)) % k
    pad_x = (k - (x % k)) % k
    if pad_z == pad_y == pad_x == 0:
        return arr
    return np.pad(arr, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)

def boxcount_3d(mask: np.ndarray, box_size: int) -> int:
    """
    Memory-safe 3D box counting:
    - No np.pad (avoids allocating big padded arrays)
    - No uint8 full-copy unless needed (works on bool directly)
    - Uses reshape+max on the largest cropped region that fits exactly,
      then handles leftover edge slabs with a small loop.
    """
    m = mask  # bool
    z, y, x = m.shape
    bs = int(box_size)

    z0 = (z // bs) * bs
    y0 = (y // bs) * bs
    x0 = (x // bs) * bs

    count = 0

    # Main block (exact multiple region)
    if z0 > 0 and y0 > 0 and x0 > 0:
        core = m[:z0, :y0, :x0]
        bz, by, bx = z0 // bs, y0 // bs, x0 // bs
        blocks = core.reshape(bz, bs, by, bs, bx, bs)
        occupied = blocks.any(axis=(1, 3, 5))
        count += int(np.count_nonzero(occupied))

    # Handle leftover slabs (edges) without allocating huge pads
    # z remainder
    if z0 < z:
        slab = m[z0:, :y0, :x0]
        if slab.size:
            bz = 1
            by, bx = y0 // bs, x0 // bs
            blocks = slab.reshape(bz, slab.shape[0], by, bs, bx, bs)
            occupied = blocks.any(axis=(1, 3, 5))
            count += int(np.count_nonzero(occupied))

    # y remainder
    if y0 < y:
        slab = m[:z0, y0:, :x0]
        if slab.size:
            bz, bx = z0 // bs, x0 // bs
            blocks = slab.reshape(bz, bs, 1, slab.shape[1], bx, bs)
            occupied = blocks.any(axis=(1, 3, 5))
            count += int(np.count_nonzero(occupied))

    # x remainder
    if x0 < x:
        slab = m[:z0, :y0, x0:]
        if slab.size:
            bz, by = z0 // bs, y0 // bs
            blocks = slab.reshape(bz, bs, by, bs, 1, slab.shape[2])
            occupied = blocks.any(axis=(1, 3, 5))
            count += int(np.count_nonzero(occupied))

    # Corner remainders (z&y, z&x, y&x, z&y&x)
    if (z0 < z and y0 < y and x0 > 0 and m[z0:, y0:, :x0].any()):
        count += 1
    if (z0 < z and x0 < x and y0 > 0 and m[z0:, :y0, x0:].any()):
        count += 1
    if (y0 < y and x0 < x and z0 > 0 and m[:z0, y0:, x0:].any()):
        count += 1
    if (z0 < z and y0 < y and x0 < x and m[z0:, y0:, x0:].any()):
        count += 1

    return count


def boxcount_dimension_3d(mask: np.ndarray, min_box=2, max_box=None) -> Dict[str, Any]:
    z, y, x = mask.shape
    smallest = min(z, y, x)
    if smallest < 2:
        return {"D": float("nan"), "r2": float("nan")}

    if max_box is None:
        max_box = 2 ** int(math.floor(math.log2(smallest)))

    s = 2 ** int(math.ceil(math.log2(max(min_box, 1))))
    sizes = []
    while s <= max_box:
        sizes.append(s)
        s *= 2

    sizes = np.array(sizes, dtype=float)
    counts = np.array([boxcount_3d(mask, int(bs)) for bs in sizes], dtype=float)
    valid = counts > 0
    sizes_v = sizes[valid]
    counts_v = counts[valid]
    if sizes_v.size < 2:
        return {"D": float("nan"), "r2": float("nan")}

    xlog = np.log(1.0 / sizes_v)
    ylog = np.log(counts_v)
    slope, intercept = np.polyfit(xlog, ylog, 1)
    ypred = slope * xlog + intercept
    ss_res = float(np.sum((ylog - ypred) ** 2))
    ss_tot = float(np.sum((ylog - np.mean(ylog)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"D": float(slope), "r2": r2}

def boxcount_dimension_3d_custom_sizes(mask: np.ndarray, sizes: List[int]) -> Dict[str, Any]:
    sizes = np.array(sorted(set(int(s) for s in sizes)), dtype=float)
    counts = np.array([boxcount_3d(mask, int(bs)) for bs in sizes], dtype=float)
    valid = counts > 0
    sizes_v = sizes[valid]
    counts_v = counts[valid]
    if sizes_v.size < 2:
        return {"D": float("nan"), "r2": float("nan")}
    xlog = np.log(1.0 / sizes_v)
    ylog = np.log(counts_v)
    slope, intercept = np.polyfit(xlog, ylog, 1)
    ypred = slope * xlog + intercept
    ss_res = float(np.sum((ylog - ypred) ** 2))
    ss_tot = float(np.sum((ylog - np.mean(ylog)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"D": float(slope), "r2": r2}


# ---------------- 3D lacunarity ----------------
def lacunarity_3d_nonoverlap(mask: np.ndarray, min_box=2, max_box=None) -> Dict[str, Any]:
    z, y, x = mask.shape
    smallest = min(z, y, x)
    if smallest < 2:
        return {"lac_mean": float("nan"), "lac_slope": float("nan")}

    if max_box is None:
        max_box = 2 ** int(math.floor(math.log2(smallest)))

    s = 2 ** int(math.ceil(math.log2(max(min_box, 1))))
    lacs = []
    m0 = mask.astype(np.uint8)

    while s <= max_box:
        m = pad_to_multiple(m0, s)
        Z, Y, X = m.shape
        bz, by, bx = Z // s, Y // s, X // s
        blocks = m.reshape(bz, s, by, s, bx, s)
        masses = blocks.sum(axis=(1, 3, 5)).astype(np.float64).ravel()
        mu = masses.mean()
        if mu <= 0:
            lac = float("nan")
        else:
            lac = float(masses.var(ddof=0) / (mu * mu) + 1.0)
        lacs.append(lac)
        s *= 2

    lac_mean = float(np.nanmean(lacs)) if np.any(np.isfinite(lacs)) else float("nan")
    finite = np.isfinite(lacs) & (np.array(lacs) > 0)
    lac_slope = float("nan")
    if np.count_nonzero(finite) >= 2:
        sizes = 2 ** np.arange(int(math.ceil(math.log2(max(min_box, 1)))),
                               int(math.floor(math.log2(smallest))) + 1)
        lx = np.log(sizes[finite])
        ly = np.log(np.array(lacs)[finite])
        lac_slope = float(np.polyfit(lx, ly, 1)[0])

    return {"lac_mean": lac_mean, "lac_slope": lac_slope}


# ---------------- Entropies ----------------
def histogram_probs(values: np.ndarray, n_bins: int, vmin=None, vmax=None) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float)
    if vmin is None:
        vmin = float(vals.min())
    if vmax is None:
        vmax = float(vals.max())
    if vmax <= vmin:
        return np.array([1.0], dtype=float)
    hist, _ = np.histogram(vals, bins=n_bins, range=(vmin, vmax), density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.array([], dtype=float)
    p /= s
    return p[p > 0]

def shannon_entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p))) if p.size else float("nan")

def renyi_entropy(p: np.ndarray, alpha: float = 2.0) -> float:
    if p.size == 0:
        return float("nan")
    if alpha == 1.0:
        return shannon_entropy(p)
    return float((1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha)))


# ---------------- FFT energy (MEMORY-SAFE) ----------------
def _roi_bbox_zyx(mask: np.ndarray, margin: int = 0) -> Tuple[slice, slice, slice]:
    zz, yy, xx = np.where(mask)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1

    if margin and margin > 0:
        z0 = max(0, z0 - margin); z1 = min(mask.shape[0], z1 + margin)
        y0 = max(0, y0 - margin); y1 = min(mask.shape[1], y1 + margin)
        x0 = max(0, x0 - margin); x1 = min(mask.shape[2], x1 + margin)

    return slice(z0, z1), slice(y0, y1), slice(x0, x1)

def _wrap_slices(n: int, r: int) -> List[slice]:
    # For unshifted FFT axes (z,y): low freq lives at start and end (wrap-around)
    if r <= 0:
        return [slice(0, 1)]
    # include [0:r] and [n-r:n]
    return [slice(0, r + 1), slice(n - r, n)]

def fft_energy_features_3d(vol: np.ndarray, roi: np.ndarray, hf_cut: float = 0.25, crop_margin: int = 8) -> dict:
    """Calculate memory-efficient 3D FFT energy features within the ROI."""
    if np.count_nonzero(roi) == 0:
        return {"fft_total_energy": float("nan"), "fft_highfreq_energy_ratio": float("nan"),
                "fft_used_shape_z": 0, "fft_used_shape_y": 0, "fft_used_shape_x": 0}

    # --- crop for FFT only ---
    if crop_margin is None:
        crop_margin = 0
    zsl, ysl, xsl = _roi_bbox_zyx(roi, margin=int(crop_margin))
    v = vol[zsl, ysl, xsl].astype(np.float32, copy=False)
    r = roi[zsl, ysl, xsl]

    # mean-center inside ROI
    mu = float(v[r].mean())
    x = np.zeros_like(v, dtype=np.float32)
    x[r] = v[r] - mu

    # Hann window (separable, no 3D array)
    Z, Y, X = x.shape
    if Z > 1:
        x *= np.hanning(Z).astype(np.float32)[:, None, None]
    if Y > 1:
        x *= np.hanning(Y).astype(np.float32)[None, :, None]
    if X > 1:
        x *= np.hanning(X).astype(np.float32)[None, None, :]

    # Real FFT (keeps float32->complex64 in scipy.fft)
    F = spfft.rfftn(x, workers=-1)
    # Power without np.abs() temp
    P = (F.real * F.real + F.imag * F.imag).astype(np.float32, copy=False)

    total = float(P.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 1e-12:
        return {"fft_total_energy": total, "fft_highfreq_energy_ratio": 0.0,
                "fft_used_shape_z": int(Z), "fft_used_shape_y": int(Y), "fft_used_shape_x": int(X)}

    # Low-frequency region size (unshifted):
    # z,y wrap; x is rfft so only positive freqs stored at start.
    rz = max(1, int(np.floor(hf_cut * (Z / 2))))
    ry = max(1, int(np.floor(hf_cut * (Y / 2))))
    # rfft last axis is Xr = X//2 + 1 (Nyquist included)
    Xr = P.shape[2]
    rx = max(1, int(np.floor(hf_cut * (Xr - 1))))

    z_slices = _wrap_slices(Z, rz)
    y_slices = _wrap_slices(Y, ry)
    x_slice = slice(0, rx + 1)

    low_energy = 0.0
    for zs in z_slices:
        for ys in y_slices:
            low_energy += float(P[zs, ys, x_slice].sum(dtype=np.float64))

    high_energy = total - low_energy
    ratio = float(high_energy / total)

    return {"fft_total_energy": total,
            "fft_highfreq_energy_ratio": ratio,
            "fft_used_shape_z": int(Z), "fft_used_shape_y": int(Y), "fft_used_shape_x": int(X)}


# ---------------- 2.5D masked GLCM ----------------
def quantize_to_levels(values: np.ndarray, n_levels: int, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.uint8)
    x = (values - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return np.floor(x * (n_levels - 1) + 1e-9).astype(np.uint8)

def glcm_2d_masked_features(img2d, mask2d, n_levels, offsets, vmin, vmax) -> Dict[str, float]:
    if np.count_nonzero(mask2d) == 0:
        return {k: float("nan") for k in ["glcm_asm", "glcm_contrast", "glcm_correlation",
                                         "glcm_dissimilarity", "glcm_energy", "glcm_homogeneity"]}

    q = quantize_to_levels(img2d.astype(np.float32), n_levels, vmin, vmax)

    I, J = np.meshgrid(np.arange(n_levels), np.arange(n_levels), indexing="ij")
    diff = np.abs(I - J)
    diff2 = (I - J) ** 2

    acc = {k: [] for k in ["glcm_asm", "glcm_contrast", "glcm_correlation",
                           "glcm_dissimilarity", "glcm_energy", "glcm_homogeneity"]}

    for dy, dx in offsets:
        y0 = max(0, -dy)
        y1 = min(q.shape[0], q.shape[0] - dy)
        x0 = max(0, -dx)
        x1 = min(q.shape[1], q.shape[1] - dx)

        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        m = mask2d[y0:y1, x0:x1] & mask2d[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        if np.count_nonzero(m) < 10:
            continue

        aa = a[m].astype(np.int32)
        bb = b[m].astype(np.int32)

        Pm = np.zeros((n_levels, n_levels), dtype=np.float64)
        np.add.at(Pm, (aa, bb), 1.0)
        s = Pm.sum()
        if s <= 0:
            continue
        Pm /= s

        asm = float(np.sum(Pm ** 2))
        energy = float(np.sqrt(asm))
        contrast = float(np.sum(diff2 * Pm))
        dissim = float(np.sum(diff * Pm))
        homog = float(np.sum(Pm / (1.0 + diff)))

        pi = Pm.sum(axis=1)
        pj = Pm.sum(axis=0)
        mi = float(np.sum(np.arange(n_levels) * pi))
        mj = float(np.sum(np.arange(n_levels) * pj))
        si = float(np.sqrt(np.sum(((np.arange(n_levels) - mi) ** 2) * pi)))
        sj = float(np.sqrt(np.sum(((np.arange(n_levels) - mj) ** 2) * pj)))
        corr = float(np.sum(((I - mi) * (J - mj) * Pm)) / (si * sj)) if (si > 0 and sj > 0) else float("nan")

        acc["glcm_asm"].append(asm)
        acc["glcm_energy"].append(energy)
        acc["glcm_contrast"].append(contrast)
        acc["glcm_dissimilarity"].append(dissim)
        acc["glcm_homogeneity"].append(homog)
        acc["glcm_correlation"].append(corr)

    return {k: _safe_mean(v) for k, v in acc.items()}

def glcm_25d_features(vol: np.ndarray, roi: np.ndarray, n_levels=32, min_pixels=500) -> Dict[str, float]:
    if np.count_nonzero(roi) == 0:
        return {f"{k}_{stat}": float("nan") for k in
                ["glcm_asm", "glcm_contrast", "glcm_correlation", "glcm_dissimilarity", "glcm_energy", "glcm_homogeneity"]
                for stat in ["mean", "median", "std"]}

    vals = vol[roi].astype(np.float32)
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.0))
    if vmax <= vmin:
        vmin = float(vals.min())
        vmax = float(vals.max())

    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    per = {k: [] for k in ["glcm_asm", "glcm_contrast", "glcm_correlation",
                           "glcm_dissimilarity", "glcm_energy", "glcm_homogeneity"]}

    for z in range(vol.shape[0]):
        m2 = roi[z]
        if int(m2.sum()) < min_pixels:
            continue
        f = glcm_2d_masked_features(vol[z], m2, n_levels, offsets, vmin, vmax)
        for k in per:
            per[k].append(f[k])

    out = {}
    for k, arr in per.items():
        a = np.array(arr, dtype=float)
        out[f"{k}_mean"] = _safe_mean(a)
        out[f"{k}_median"] = _safe_median(a)
        out[f"{k}_std"] = _safe_std(a)
    return out


# ---------------- Relative wavelet energies ----------------
def wavelet_energies_3d(vol: np.ndarray, roi: np.ndarray, wavelet: str = "db2") -> Dict[str, float]:
    """
    Calculate single-level 3D relative wavelet sub-band energies.

    Intensities are mean-centred within the ROI and voxels outside the ROI
    are set to zero before the 3D discrete wavelet transform. For each of
    the eight 3D sub-bands, energy is calculated as the sum of squared
    coefficients and then expressed as a fraction of total wavelet energy
    across all eight sub-bands.

    Relative energies therefore sum to 1 (within floating-point precision)
    for non-degenerate inputs. Only the eight primary sub-band measurements
    are returned; the previous aggregate LL/LH/HL/HH variables are omitted
    because they were mathematically derived from these same sub-bands.
    """
    names = [
        "LLL_rel_energy", "LLH_rel_energy",
        "LHL_rel_energy", "LHH_rel_energy",
        "HLL_rel_energy", "HLH_rel_energy",
        "HHL_rel_energy", "HHH_rel_energy",
    ]

    if np.count_nonzero(roi) == 0:
        return {k: float("nan") for k in names}

    # Mean-centre intensities inside the ROI and zero everything outside it.
    v = vol.astype(np.float32).copy()
    mu = float(v[roi].mean())
    v = np.where(roi, v - mu, 0.0).astype(np.float32)

    # Single-level 3D discrete wavelet transform.
    coeffs = pywt.dwtn(v, wavelet=wavelet, axes=(0, 1, 2))

    name_map = {
        "aaa": "LLL_rel_energy",
        "aad": "LLH_rel_energy",
        "ada": "LHL_rel_energy",
        "add": "LHH_rel_energy",
        "daa": "HLL_rel_energy",
        "dad": "HLH_rel_energy",
        "dda": "HHL_rel_energy",
        "ddd": "HHH_rel_energy",
    }

    # Absolute energy in each sub-band.
    raw_energies = {
        name_map[k]: float(np.sum(np.asarray(arr, dtype=np.float64) ** 2))
        for k, arr in coeffs.items()
    }

    # Ensure all expected bands are represented.
    for k in names:
        raw_energies.setdefault(k, 0.0)

    total_energy = float(sum(raw_energies[k] for k in names))

    if total_energy <= 0.0 or not np.isfinite(total_energy):
        return {k: float("nan") for k in names}

    relative_energies = {
        k: float(raw_energies[k] / total_energy)
        for k in names
    }

    return relative_energies


# ---------------- Feature extraction ----------------
def extract_features(vol: np.ndarray, roi: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["shape_z"], out["shape_y"], out["shape_x"] = map(int, vol.shape)
    out["foreground_voxels"] = int(roi.sum())
    out["foreground_fraction"] = float(out["foreground_voxels"] / roi.size) if roi.size else float("nan")
    out["intensity_min"] = float(np.min(vol))
    out["intensity_max"] = float(np.max(vol))

    roi_vals = vol[roi].astype(np.float32)
    if roi_vals.size == 0:
        out["intensity_mean"] = out["intensity_std"] = out["intensity_kurtosis"] = float("nan")
        out["shannon_entropy"] = out["renyi_entropy_a2"] = float("nan")
    else:
        out["intensity_mean"] = float(roi_vals.mean())
        out["intensity_std"] = float(roi_vals.std(ddof=0))
        if roi_vals.size < 4 or float(out["intensity_std"]) < 1e-6:
            out["intensity_kurtosis"] = float("nan")
        else:
            out["intensity_kurtosis"] = float(kurtosis(roi_vals, fisher=False, bias=False))

        p = histogram_probs(roi_vals, n_bins=ENTROPY_BINS, vmin=float(roi_vals.min()), vmax=float(roi_vals.max()))
        out["shannon_entropy"] = shannon_entropy(p)
        out["renyi_entropy_a2"] = renyi_entropy(p, alpha=RENYI_ALPHA)

    bc = boxcount_dimension_3d(roi, min_box=2)
    out["boxcount_D"] = float(bc["D"])
    out["boxcount_r2"] = float(bc["r2"])

    lac = lacunarity_3d_nonoverlap(roi, min_box=2)
    out["lacunarity_mean"] = float(lac["lac_mean"])
    out["lacunarity_slope"] = float(lac["lac_slope"])

    # FFT (cropped, rFFT, no shift)
    out.update(fft_energy_features_3d(vol, roi, hf_cut=FFT_HF_CUT, crop_margin=FFT_CROP_MARGIN))

    out.update(glcm_25d_features(vol, roi, n_levels=GLCM_LEVELS, min_pixels=GLCM_MIN_PIXELS_PER_SLICE))
    out.update(wavelet_energies_3d(vol, roi, wavelet=WAVELET))
    return out


# ---------------- Synthetic suite ----------------
def synthetic_suite_table() -> pd.DataFrame:
    def white(shape=(64, 64, 64)):
        return np.full(shape, 1.0, dtype=np.float32)

    def black(shape=(64, 64, 64)):
        return np.zeros(shape, dtype=np.float32)

    def noise(shape=(64, 64, 64), seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=shape).astype(np.float32)

    def full(shape=(64, 64, 64)):
        return np.ones(shape, dtype=bool)

    def empty(shape=(64, 64, 64)):
        return np.zeros(shape, dtype=bool)

    tests = [
        ("WHITE", white(), np.ones((64, 64, 64), dtype=bool)),
        ("BLACK", black(), np.zeros((64, 64, 64), dtype=bool)),
        ("NOISE", noise(seed=123), np.ones((64, 64, 64), dtype=bool)),
        ("FULL_MASK", (full().astype(np.float32) + 0.2 * noise(seed=1)), full()),
        ("EMPTY_MASK", np.zeros((64, 64, 64), dtype=np.float32), empty()),
    ]

    def line(shape=(128, 128, 128)):
        m = np.zeros(shape, dtype=bool)
        m[:, shape[1] // 2, shape[2] // 2] = True
        return m

    def plane(shape=(128, 128, 128), thickness=1):
        m = np.zeros(shape, dtype=bool)
        z0 = shape[0] // 2
        z_start = max(0, z0 - thickness // 2)
        z_end = min(shape[0], z_start + thickness)
        m[z_start:z_end, :, :] = True
        return m

    def sphere(shape=(128, 128, 128), radius=45):
        zz, yy, xx = np.indices(shape)
        cz, cy, cx = [(s - 1) / 2 for s in shape]
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        return dist2 <= radius**2

    def menger(iterations=3):
        n = 3 ** iterations
        cube = np.ones((n, n, n), dtype=bool)

        def carve(block, x0, y0, z0, size):
            step = size // 3
            if step < 1:
                return
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if (i == 1) + (j == 1) + (k == 1) >= 2:
                            xi = x0 + i * step
                            yj = y0 + j * step
                            zk = z0 + k * step
                            block[xi:xi + step, yj:yj + step, zk:zk + step] = False
                        else:
                            carve(block, x0 + i * step, y0 + j * step, z0 + k * step, step)

        carve(cube, 0, 0, 0, n)
        return cube

    rng = np.random.default_rng(999)

    for name, roi in [
        ("LINE", line(),),
        ("PLANE", plane(thickness=1),),
        ("SPHERE", sphere(radius=45),),
        ("MENGER", menger(3),),
    ]:
        vol = np.zeros_like(roi, dtype=np.float32)
        if roi.sum() > 0:
            vol[roi] = 1.0 + 0.2 * rng.normal(size=int(roi.sum())).astype(np.float32)
        tests.append((name, vol, roi))

    rows = []
    for name, vol, roi in tests:
        feats = extract_features(vol, roi)

        if name == "MENGER":
            bc3 = boxcount_dimension_3d_custom_sizes(roi, [1, 3, 9, 27])
            feats["boxcount_D"] = bc3["D"]
            feats["boxcount_r2"] = bc3["r2"]

        feats["type"] = name
        rows.append(feats)

    df = pd.DataFrame(rows)
    cols = ["type"] + [c for c in df.columns if c != "type"]
    return df[cols]


# ---------------- Metadata-driven file mode ----------------
REQUIRED_METADATA_COLUMNS = {"filename", "strain", "medium", "relative_path"}

def load_and_validate_metadata(metadata_path: Path, data_dir: Path) -> pd.DataFrame:
    """
    Load explicit experimental metadata and validate all image paths and labels.

    Required columns:
        filename, strain, medium, relative_path

    The analysis does not infer strain or medium from filenames.
    """
    metadata_path = Path(metadata_path)
    data_dir = Path(data_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    missing_cols = REQUIRED_METADATA_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(
            "Metadata file is missing required column(s): "
            + ", ".join(sorted(missing_cols))
        )

    if df.empty:
        raise ValueError("Metadata file contains no image records.")

    # Normalise text fields and reject missing values.
    for col in ["filename", "strain", "medium", "relative_path"]:
        if df[col].isna().any():
            bad_rows = (df.index[df[col].isna()] + 2).tolist()
            raise ValueError(
                f"Blank value(s) found in metadata column '{col}' "
                f"at CSV row(s): {bad_rows}"
            )
        df[col] = df[col].astype(str).str.strip()
        blank = df[col].eq("")
        if blank.any():
            bad_rows = (df.index[blank] + 2).tolist()
            raise ValueError(
                f"Blank value(s) found in metadata column '{col}' "
                f"at CSV row(s): {bad_rows}"
            )

    # Each image path must occur only once.
    duplicated = df["relative_path"].duplicated(keep=False)
    if duplicated.any():
        dupes = df.loc[duplicated, "relative_path"].tolist()
        raise ValueError(
            "Duplicate relative_path entries found in metadata: "
            + "; ".join(dupes)
        )

    # Confirm that metadata filename agrees with the filename at the end of the path.
    path_name_mismatch = [
        (row.filename, row.relative_path)
        for row in df.itertuples(index=False)
        if Path(row.relative_path).name != row.filename
    ]
    if path_name_mismatch:
        details = "\n".join(
            f"  filename='{fn}' but relative_path='{rp}'"
            for fn, rp in path_name_mismatch
        )
        raise ValueError(
            "Metadata filename does not match the file named in relative_path:\n"
            + details
        )

    # Confirm that every listed image exists and is a TIFF stack.
    missing_files = []
    invalid_extensions = []
    for row in df.itertuples(index=False):
        image_path = data_dir / Path(row.relative_path)
        if image_path.suffix.lower() not in {".tif", ".tiff"}:
            invalid_extensions.append(str(row.relative_path))
        if not image_path.exists():
            missing_files.append(str(row.relative_path))

    if invalid_extensions:
        raise ValueError(
            "Metadata contains non-TIFF file(s): "
            + "; ".join(invalid_extensions)
        )

    if missing_files:
        raise FileNotFoundError(
            "The following image file(s) listed in metadata were not found:\n"
            + "\n".join(f"  {p}" for p in missing_files)
        )

    # Report replicate counts without imposing study-specific labels.
    counts = (
        df.groupby(["strain", "medium"], dropna=False)
          .size()
          .reset_index(name="n")
          .sort_values(["medium", "strain"])
    )

    print("\nMetadata validation passed.")
    print(f"Images listed: {len(df)}")
    print("\nReplicate counts:")
    print(counts.to_string(index=False))

    return df


def run_file_mode() -> pd.DataFrame:
    metadata = load_and_validate_metadata(METADATA_FILE, DATA_DIR)
    rows = []

    total = len(metadata)

    for i, rec in enumerate(metadata.itertuples(index=False), start=1):
        image_path = DATA_DIR / Path(rec.relative_path)

        print(
            f"\n[{i}/{total}] Processing: {rec.relative_path} "
            f"(strain={rec.strain}, medium={rec.medium})"
        )

        vol = load_volume_as_zyx(image_path)
        roi = make_binary_mask(
            vol,
            threshold=THRESHOLD,
            invert=INVERT,
            min_object_size=MIN_OBJECT_SIZE,
        )

        if np.count_nonzero(roi) == 0:
            raise ValueError(
                f"{rec.relative_path}: segmentation produced an empty ROI. "
                "Check the image and threshold settings."
            )

        feats = extract_features(vol, roi)

        # Explicit metadata are carried into the output unchanged.
        row = {
            "filename": rec.filename,
            "strain": rec.strain,
            "medium": rec.medium,
            "relative_path": rec.relative_path,
        }

        # Preserve any additional metadata columns supplied by the user.
        for col in metadata.columns:
            if col not in row:
                row[col] = getattr(rec, col)

        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------- Main ----------------
if __name__ == "__main__":
    df_file = None
    df_syn = None

    if RUN_MODE in {"file", "both"}:
        df_file = run_file_mode()
        print("\n=== FILE MODE COMPLETE ===")

        preview_cols = [
            "filename", "strain", "medium",
            "boxcount_D", "boxcount_r2",
            "foreground_fraction",
            "fft_highfreq_energy_ratio",
        ]
        preview_cols = [c for c in preview_cols if c in df_file.columns]
        print(df_file[preview_cols].to_string(index=False))

    if RUN_MODE in {"synthetic", "both"}:
        df_syn = synthetic_suite_table()
        print("\n=== SYNTHETIC MODE ===")
        print(df_syn[[
            "type", "boxcount_D", "boxcount_r2", "lacunarity_mean",
            "shannon_entropy", "renyi_entropy_a2",
            "fft_highfreq_energy_ratio",
        ]].to_string(index=False))

    if SAVE_CSV:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        if df_file is not None:
            out_path = OUT_DIR / "ABC3D_features.csv"
            df_file.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}")

        if df_syn is not None:
            out_path = OUT_DIR / "features_synthetic.csv"
            df_syn.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

