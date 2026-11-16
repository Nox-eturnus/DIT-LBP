"""Texture descriptors: LBP, LDB, image-difference, and the proposed DIT-LBP.

All functions take float images (2-D numpy arrays). Light-direction stacks are
dicts keyed by tilt angle in degrees: {0, 90, 180, 270}; 0/180 and 90/270 are
the two opposite-illumination pairs.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

# Number of set bits for every 8-bit value.
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
PAIRS = ((0, 180), (90, 270))


# ----------------------------------------------------------------------------
# LBP
# ----------------------------------------------------------------------------
# 8 neighbours in circular order at integer offsets (dy, dx), scaled by R.
_NEIGH = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))


def lbp_codes(img: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """Raw 8-bit LBP codes (Ojala 3x3 operator, generalised to radius R).

    Neighbours are sampled at integer offsets, without interpolation, so the code
    is exactly invariant to any monotonic grey-level change (exposure, gamma).
    Bit k is set when neighbour k >= centre. Borders use edge replication.
    """
    if P != 8:
        raise ValueError("only P=8 is implemented")
    R = int(R)
    pad = np.pad(img, R, mode="edge")
    H, W = img.shape
    c = pad[R:R + H, R:R + W]
    code = np.zeros((H, W), dtype=np.uint8)
    for k, (dy, dx) in enumerate(_NEIGH):
        n = pad[R + dy * R:R + dy * R + H, R + dx * R:R + dx * R + W]
        code |= (n >= c).astype(np.uint8) << k
    return code


def _u2_table() -> np.ndarray:
    """Map 256 codes to 59 labels: 58 uniform patterns (<=2 circular transitions) + 1."""
    table = np.full(256, 58, dtype=np.uint8)
    label = 0
    for v in range(256):
        bits = [(v >> k) & 1 for k in range(8)]
        if sum(bits[k] != bits[(k + 1) % 8] for k in range(8)) <= 2:
            table[v] = label
            label += 1
    assert label == 58
    return table


U2 = _u2_table()


def lbp_u2(img: np.ndarray, R: int = 1) -> np.ndarray:
    """Uniform-pattern label map (0..58)."""
    return U2[lbp_codes(img, 8, R)]


def lbp_hist(img: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """Standard uniform LBP histogram (LBP^{u2}_{8,R}, 59 bins)."""
    h = np.bincount(lbp_u2(img, R).ravel(), minlength=59).astype(float)
    return h / h.sum()


# ----------------------------------------------------------------------------
# DIT-LBP (proposed)
# ----------------------------------------------------------------------------
def flip_count(code_a: np.ndarray, code_b: np.ndarray) -> np.ndarray:
    """Number of LBP bits that differ between two aligned code maps (XOR + popcount)."""
    return POPCOUNT8[np.bitwise_xor(code_a, code_b)]


def dit_maps(codes: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal (0 vs 180) and vertical (90 vs 270) flip-count maps, values 0..8."""
    return flip_count(codes[0], codes[180]), flip_count(codes[90], codes[270])


def dit_hist(h_map: np.ndarray, v_map: np.ndarray, P: int = 8) -> np.ndarray:
    """81-bin joint histogram of (horizontal, vertical) flip counts, L1-normalised."""
    k = P + 1
    idx = h_map.astype(np.int64) * k + v_map.astype(np.int64)
    h = np.bincount(idx.ravel(), minlength=k * k).astype(float)
    return h / h.sum()


def dit_lbp(stack: dict[int, np.ndarray], P: int = 8, R: float = 1.0) -> np.ndarray:
    """DIT-LBP descriptor of one aligned 4-light stack."""
    codes = {t: lbp_codes(stack[t], P, R) for t in (0, 90, 180, 270)}
    return dit_hist(*dit_maps(codes), P=P)


# ----------------------------------------------------------------------------
# Noise-gated DIT-LBP (extension, added after the flat-card control showed that
# pure sensor noise produces ~4 random flips per pixel on textureless regions)
# ----------------------------------------------------------------------------
def noise_sigma(img: np.ndarray) -> float:
    """Robust noise std from the median absolute Laplacian-like residual."""
    r = img[1:-1, 1:-1] - 0.25 * (img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:])
    return float(np.median(np.abs(r - np.median(r))) / 0.6745 / np.sqrt(1.25))


def lbp_codes_conf(img: np.ndarray, tau: float, R: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """LBP codes plus a mask of 'confident' bits, |neighbour - centre| >= tau.

    Uses the dead zone of Local Ternary Patterns (Tan & Triggs, 2010) to mark bits
    whose sign is decided by noise rather than by the surface.
    """
    R = int(R)
    pad = np.pad(img, R, mode="edge")
    H, W = img.shape
    c = pad[R:R + H, R:R + W]
    code = np.zeros((H, W), dtype=np.uint8)
    conf = np.zeros((H, W), dtype=np.uint8)
    for k, (dy, dx) in enumerate(_NEIGH):
        d = pad[R + dy * R:R + dy * R + H, R + dx * R:R + dx * R + W] - c
        code |= (d >= 0).astype(np.uint8) << k
        conf |= (np.abs(d) >= tau).astype(np.uint8) << k
    return code, conf


def dit_gated_maps(stack: dict[int, np.ndarray], k: float = 2.0, R: int = 1,
                   sigmas: dict[int, float] | None = None):
    """Flip counts that only count bits confident in both images of a pair.

    tau = k * sqrt(2) * sigma_noise (std of a difference of two noisy pixels).
    """
    cc = {}
    for t, a in stack.items():
        s = sigmas[t] if sigmas else noise_sigma(a)
        cc[t] = lbp_codes_conf(a, k * np.sqrt(2) * s, R)
    out = []
    for a, b in PAIRS:
        (ca, ma), (cb, mb) = cc[a], cc[b]
        out.append(POPCOUNT8[np.bitwise_xor(ca, cb) & ma & mb])
    return tuple(out)


# ----------------------------------------------------------------------------
# LDB: Local Difference Binary (Yang & Cheng, ISMAR 2012; TPAMI 2014)
# ----------------------------------------------------------------------------
def _cell_bounds(size: int, n: int) -> np.ndarray:
    return np.round(np.linspace(0, size, n + 1)).astype(int)


def _box_sums(ii: np.ndarray, y0, y1, x0, x1):
    """Sum over [y0,y1)x[x0,x1) from an integral image padded with a zero row/col."""
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def ldb_hist(img: np.ndarray, win: int = 32, stride: int = 16,
             grids: tuple[int, ...] = (2, 3, 4, 5)) -> np.ndarray:
    """Dense LDB texture descriptor.

    LDB splits a window into n x n cells (several n), computes for each cell its
    mean intensity and mean x/y gradient, and emits one bit per cell pair and per
    quantity: bit = f(cell_i) > f(cell_j). Here LDB is evaluated densely on
    win x win windows and the descriptor is the firing rate of every bit over all
    windows (1386 values for grids 2..5), which turns the keypoint descriptor into
    a texture descriptor comparable to an LBP histogram.
    """
    gy, gx = np.gradient(img.astype(float))
    chans = [img.astype(float), gx, gy]
    iis = [np.pad(c.cumsum(0).cumsum(1), ((1, 0), (1, 0))) for c in chans]
    H, W = img.shape
    ys = np.arange(0, H - win + 1, stride)
    xs = np.arange(0, W - win + 1, stride)
    oy, ox = np.meshgrid(ys, xs, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    rates = []
    for n in grids:
        b = _cell_bounds(win, n)
        cells = []  # (n*n, 3, n_windows)
        for i in range(n):
            for j in range(n):
                y0, y1, x0, x1 = oy + b[i], oy + b[i + 1], ox + b[j], ox + b[j + 1]
                area = (b[i + 1] - b[i]) * (b[j + 1] - b[j])
                cells.append([_box_sums(ii, y0, y1, x0, x1) / area for ii in iis])
        cells = np.asarray(cells)
        a, c = np.array(list(combinations(range(n * n), 2))).T
        bits = cells[a] > cells[c]  # (pairs, 3, n_windows)
        rates.append(bits.mean(axis=2).ravel())
    return np.concatenate(rates)


# ----------------------------------------------------------------------------
# Simple image-difference baseline
# ----------------------------------------------------------------------------
def diff_feat(stack: dict[int, np.ndarray], bins: int = 16, eps: float = 1e-6) -> np.ndarray:
    """Histogram + moments of normalised opposite-light differences (a, b)/(a+b).

    This is the cheap photometric-stereo cue: on a relief, opposite lights brighten
    opposite slopes, so (I_0 - I_180)/(I_0 + I_180) tracks surface slope.
    """
    out = []
    for a, b in PAIRS:
        d = (stack[a] - stack[b]) / (stack[a] + stack[b] + eps)
        d = np.clip(d, -1, 1)
        h, _ = np.histogram(d, bins=bins, range=(-1, 1))
        h = h / h.sum()
        ad = np.abs(d)
        out.append(np.concatenate([h, [ad.mean(), d.std(), np.percentile(ad, 90)]]))
    return np.concatenate(out)
