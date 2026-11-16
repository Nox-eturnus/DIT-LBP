"""Extract every compared descriptor from a prepared 4-light stack."""
from __future__ import annotations

import time

import numpy as np

from .data import TILTS, to_patches
from .features import (U2, diff_feat, dit_gated_maps, dit_hist, dit_maps, lbp_codes, lbp_hist,
                       ldb_hist, noise_sigma)

METHODS = {
    # name: (images used, short description)
    "LBP (1 image)": (1, "uniform LBP^u2 P=8 R=1 histogram, light from tilt 0 only (59-d)"),
    "LBP (4 images)": (4, "concatenated LBP^u2 histograms of the four lights (236-d)"),
    "LDB (1 image)": (1, "dense Local Difference Binary bit rates, tilt 0 only (1386-d)"),
    "LDB (4 images)": (4, "concatenated dense LDB of the four lights (5544-d)"),
    "Image difference (4 images)": (4, "histograms/moments of (I_a-I_b)/(I_a+I_b) for 0/180, 90/270 (38-d)"),
    "DIT-LBP (proposed)": (4, "81-bin joint histogram of opposite-light LBP bit-flip counts, R=1"),
    "DIT-LBP multi-radius": (4, "DIT-LBP at R=1,2,3 concatenated (243-d), extension"),
    "DIT-LBP + LBP (1 image)": (4, "DIT-LBP concatenated with single-image LBP (140-d), extension"),
    "DIT-LBP noise-gated": (4, "DIT-LBP counting only bits with |n-c| >= 2*sqrt(2)*sigma in both images (81-d), extension"),
}
RADII = (1, 2, 3)
N_U2 = 59


def _hist(a: np.ndarray, n: int) -> np.ndarray:
    h = np.bincount(a.astype(np.int64).ravel(), minlength=n).astype(float)
    return h / h.sum()


def extract_stack(stack: dict[int, np.ndarray], patch: int = 128, ldb: bool = True):
    """Return (features: dict method -> (n_patches, d) array,
    flips: (n_patches, 4) mean flips h, v, gated h, gated v)."""
    dit_by_r, u2 = {}, None
    for R in RADII:
        codes = {t: lbp_codes(stack[t], 8, R) for t in TILTS}
        dit_by_r[R] = dit_maps(codes)
        if R == 1:
            u2 = {t: U2[codes[t]] for t in TILTS}
    gh, gv = dit_gated_maps(stack, sigmas={t: noise_sigma(stack[t]) for t in TILTS})
    gated_p = (to_patches(gh, patch), to_patches(gv, patch))
    u2_p = {t: to_patches(u2[t], patch) for t in TILTS}
    img_p = {t: to_patches(stack[t], patch) for t in TILTS}
    dit_p = {R: (to_patches(h, patch), to_patches(v, patch)) for R, (h, v) in dit_by_r.items()}
    n = len(u2_p[0])
    feats = {m: [] for m in METHODS}
    flips = []
    for i in range(n):
        lbp1 = _hist(u2_p[0][i], N_U2)
        lbp4 = np.concatenate([_hist(u2_p[t][i], N_U2) for t in TILTS])
        dits = [dit_hist(dit_p[R][0][i], dit_p[R][1][i]) for R in RADII]
        feats["LBP (1 image)"].append(lbp1)
        feats["LBP (4 images)"].append(lbp4)
        if ldb:
            l = {t: ldb_hist(img_p[t][i]) for t in TILTS}
            feats["LDB (1 image)"].append(l[0])
            feats["LDB (4 images)"].append(np.concatenate([l[t] for t in TILTS]))
        feats["Image difference (4 images)"].append(diff_feat({t: img_p[t][i] for t in TILTS}))
        feats["DIT-LBP (proposed)"].append(dits[0])
        feats["DIT-LBP multi-radius"].append(np.concatenate(dits))
        feats["DIT-LBP + LBP (1 image)"].append(np.concatenate([dits[0], lbp1]))
        feats["DIT-LBP noise-gated"].append(dit_hist(gated_p[0][i], gated_p[1][i]))
        flips.append([dit_p[1][0][i].mean(), dit_p[1][1][i].mean(),
                      gated_p[0][i].mean(), gated_p[1][i].mean()])
    feats = {m: np.asarray(v) for m, v in feats.items() if v}
    return feats, np.asarray(flips)


# ----------------------------------------------------------------------------
# Per-sample timing, each descriptor computed from scratch on one patch stack
# ----------------------------------------------------------------------------
def _single(method: str, s: dict[int, np.ndarray]) -> np.ndarray:
    if method == "LBP (1 image)":
        return lbp_hist(s[0])
    if method == "LBP (4 images)":
        return np.concatenate([lbp_hist(s[t]) for t in TILTS])
    if method == "LDB (1 image)":
        return ldb_hist(s[0])
    if method == "LDB (4 images)":
        return np.concatenate([ldb_hist(s[t]) for t in TILTS])
    if method == "Image difference (4 images)":
        return diff_feat(s)
    if method == "DIT-LBP (proposed)":
        return dit_hist(*dit_maps({t: lbp_codes(s[t]) for t in TILTS}))
    if method == "DIT-LBP multi-radius":
        return np.concatenate([dit_hist(*dit_maps({t: lbp_codes(s[t], 8, R) for t in TILTS})) for R in RADII])
    if method == "DIT-LBP + LBP (1 image)":
        return np.concatenate([dit_hist(*dit_maps({t: lbp_codes(s[t]) for t in TILTS})), lbp_hist(s[0])])
    if method == "DIT-LBP noise-gated":
        return dit_hist(*dit_gated_maps(s))
    raise KeyError(method)


def time_methods(stacks: list[dict[int, np.ndarray]], repeats: int = 3) -> dict[str, float]:
    """Median milliseconds per patch stack for each method."""
    out = {}
    for m in METHODS:
        _single(m, stacks[0])  # warm-up
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for s in stacks:
                _single(m, s)
            ts.append((time.perf_counter() - t0) / len(stacks) * 1e3)
        out[m] = float(np.median(ts))
    return out
