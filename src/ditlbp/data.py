"""PhoTex loading, photometric normalisation, alignment and flat-print rendering.

Classes
-------
raised  : a real PhoTex surface photographed under the four opposite lights.
printed : a flat matte print of a photograph of that same surface, observed on
          the same rig. It is rendered, not photographed: the print reflectance
          is the surface photo under one light (shadows baked in), and each
          observation is  reflectance x measured light field of the rig
          (the PhoTex `reg` white card under that light) + sensor noise
          measured on the same card. So a single printed image is, by design,
          very close to a single raised image; only the change across lights
          differs. This is the "photo of a texture vs the texture" problem.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, shift as nd_shift
from skimage.registration import phase_cross_correlation

TILTS = (0, 90, 180, 270)
NAME_RE = re.compile(r"^(\w+)\.(\w+)\.(\d+)\.(\d+)\.(\d+)\.tif$")
_tl = logging.getLogger("tifffile")
_tl.setLevel(logging.CRITICAL)
_tl.propagate = False


@dataclass(frozen=True, order=True)
class QuadKey:
    texture: str
    investigation: str
    rotation: int
    slant: int

    def files(self, root: Path) -> dict[int, Path]:
        stem = f"{self.investigation}.{self.texture}.{self.rotation}.{self.slant}"
        return {t: root / self.texture / f"{stem}.{t}.tif" for t in TILTS}


def find_quads(root: Path, exclude=("reg",)) -> list[QuadKey]:
    keys = set()
    for p in sorted(root.glob("*/*.tif")):
        m = NAME_RE.match(p.name)
        if m and p.parent.name not in exclude:
            inv, tex, rot, slant, _ = m.groups()
            keys.add(QuadKey(tex, inv, int(rot), int(slant)))
    return sorted(k for k in keys if all(f.exists() for f in k.files(root).values()))


def read_tif(path: Path) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tifffile.imread(path).astype(np.float32)


def load_quad(root: Path, key: QuadKey) -> dict[int, np.ndarray]:
    return {t: read_tif(p) for t, p in key.files(root).items()}


# ----------------------------------------------------------------------------
# Rig calibration from the flat white card ("reg")
# ----------------------------------------------------------------------------
@dataclass
class Rig:
    """Measured light field F_k(x) (mean 1) and noise sigma per tilt and slant."""
    field: dict[int, dict[int, np.ndarray]]
    sigma: dict[int, dict[int, float]]
    level: dict[int, dict[int, float]]
    reg_raw: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)

    def for_slant(self, slant: int):
        s = min(self.field, key=lambda k: abs(k - slant))
        return self.field[s], self.sigma[s], self.level[s]


def build_rig(root: Path) -> Rig:
    from skimage.restoration import estimate_sigma

    fields, sig, lev, raw = {}, {}, {}, {}
    for p in sorted((root / "reg").glob("*.tif")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        slant, tilt = int(m.group(4)), int(m.group(5))
        if tilt not in TILTS:
            continue
        a = read_tif(p)
        f = gaussian_filter(a, 25)
        fields.setdefault(slant, {})[tilt] = f / f.mean()
        sig.setdefault(slant, {})[tilt] = float(estimate_sigma(a))
        lev.setdefault(slant, {})[tilt] = float(a.mean())
        raw.setdefault(slant, {})[tilt] = a
    fields = {s: v for s, v in fields.items() if len(v) == 4}
    if not fields:
        raise RuntimeError("no complete reg quad found; run scripts/download_photex.py")
    return Rig(fields, {s: sig[s] for s in fields}, {s: lev[s] for s in fields},
               {s: raw[s] for s in fields})


# ----------------------------------------------------------------------------
# Normalisation and alignment
# ----------------------------------------------------------------------------
def flat_field(stack: dict[int, np.ndarray], F: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Divide out the rig's light fall-off and normalise each image to mean 1."""
    out = {}
    for t, a in stack.items():
        b = a / F[t]
        out[t] = b / b.mean()
    return out


def estimate_shifts(stack: dict[int, np.ndarray], ref: int = 0) -> dict[int, np.ndarray]:
    """Sub-pixel translation of every image relative to `ref` (phase correlation on
    high-passed images, which are less affected by the shading change)."""
    hp = {t: a - gaussian_filter(a, 8) for t, a in stack.items()}
    out = {}
    for t in TILTS:
        if t == ref:
            out[t] = np.zeros(2)
        else:
            s, _, _ = phase_cross_correlation(hp[ref], hp[t], upsample_factor=10, normalization=None)
            out[t] = s
    return out


def align(stack: dict[int, np.ndarray], max_shift: float = 4.0) -> tuple[dict, dict]:
    shifts = estimate_shifts(stack)
    out = {}
    for t, a in stack.items():
        s = shifts[t]
        out[t] = a if np.abs(s).max() < 0.25 or np.abs(s).max() > max_shift else nd_shift(a, s, order=1, mode="nearest")
    return out, shifts


def crop_downsample(a: np.ndarray, size: int = 1024, factor: int = 2) -> np.ndarray:
    H, W = a.shape
    y0, x0 = (H - size) // 2, (W - size) // 2
    c = a[y0:y0 + size, x0:x0 + size]
    return c.reshape(size // factor, factor, size // factor, factor).mean(axis=(1, 3))


# ----------------------------------------------------------------------------
# Flat-print rendering and stress conditions
# ----------------------------------------------------------------------------
CONDITIONS = ("ideal", "realistic", "glossy", "exposure", "misaligned",
              "relief_0.30", "relief_0.10", "relief_0.03")


def relief_alpha(condition: str) -> float:
    return float(condition.split("_")[1]) if condition.startswith("relief_") else 1.0


def shallow(raw: dict[int, np.ndarray], F: dict[int, np.ndarray], alpha: float) -> dict:
    """Approximate a shallower version of the same surface.

    For a near-Lambertian surface with small slopes, I_k ~ albedo * (a + b * grad(h) . l_k),
    so the light-dependent part of each image scales with relief height. We keep the
    mean over the four lights (albedo + constant shading) and shrink each image's
    deviation from it by `alpha`, in the flat-fielded domain. Noise is added later,
    so shallow surfaces have proportionally lower signal-to-noise.
    """
    if alpha == 1.0:
        return raw
    r = {t: raw[t] / F[t] for t in TILTS}
    M = sum(r.values()) / 4
    return {t: ((M + alpha * (r[t] - M)) * F[t]).astype(np.float32) for t in TILTS}


def render_print(raw: dict[int, np.ndarray], F: dict[int, np.ndarray], sigma: dict[int, float],
                 rng: np.random.Generator, quality: str = "realistic",
                 glossy: bool = False) -> tuple[dict[int, np.ndarray], int]:
    """Raw-sensor images of a flat print of `raw[src]` observed under each light.

    The source photo is flat-fielded (a photographer would correct fall-off),
    optionally degraded by printing (dot-gain blur, paper grain, tone curve),
    then re-lit by the measured light field F_k with exposure matched to the real
    surface. Sensor noise is NOT added here; `observe` adds the same fresh noise
    to both classes.
    """
    src = int(rng.choice(TILTS))
    R = raw[src] / F[src]
    R = R / R.mean()
    if quality == "realistic":
        R = gaussian_filter(R, 1.0)                                   # dot gain
        R = R * (1 + 0.02 * gaussian_filter(rng.standard_normal(R.shape), 1.5) / 0.19)  # paper grain ~2%
        R = np.clip(R, 1e-3, None) ** 0.85                            # print tone curve
        R = R / R.mean()
    H, W = R.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = {}
    for t in TILTS:
        img = R * F[t] * raw[t].mean()
        if glossy:
            # broad specular sheen of a glossy print, displaced along the light azimuth
            th = np.deg2rad(t)
            cy, cx = H / 2 - 0.3 * H * np.sin(th), W / 2 + 0.3 * W * np.cos(th)
            w = 0.35 * W
            amp = rng.uniform(0.2, 0.6) * raw[t].mean()
            img = img + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * w * w))
        out[t] = img.astype(np.float32)
    return out, src


def observe(raw: dict[int, np.ndarray], sigma: dict[int, float], rng: np.random.Generator) -> dict:
    """Second, independent sensor read for both classes (keeps noise levels matched)."""
    return {t: a + rng.normal(0, sigma[t], a.shape).astype(np.float32) for t, a in raw.items()}


def perturb(stack: dict[int, np.ndarray], condition: str, rng: np.random.Generator) -> dict:
    """Post-normalisation perturbations applied identically to both classes."""
    if condition == "exposure":
        out = {}
        for t, a in stack.items():
            g, gam = rng.uniform(0.6, 1.4), rng.uniform(0.8, 1.25)
            out[t] = g * np.clip(a, 1e-4, None) ** gam
        return out
    if condition == "misaligned":
        out = dict(stack)
        for t in (180, 270):
            dy, dx = rng.integers(1, 3, size=2) * rng.choice([-1, 1], size=2)
            out[t] = np.roll(stack[t], (int(dy), int(dx)), axis=(0, 1))
        return out
    return stack


def prepare(raw: dict[int, np.ndarray], F, condition: str, rng, do_align: bool = True):
    """Full pipeline: flat-field -> align -> crop/downsample -> condition perturbation."""
    ff = flat_field(raw, F)
    shifts = {t: np.zeros(2) for t in TILTS}
    if do_align and condition != "misaligned":
        ff, shifts = align(ff)
    small = {t: crop_downsample(a) for t, a in ff.items()}
    return perturb(small, condition, rng), shifts


def to_patches(a: np.ndarray, size: int = 128) -> list[np.ndarray]:
    H, W = a.shape
    return [a[y:y + size, x:x + size] for y in range(0, H - size + 1, size)
            for x in range(0, W - size + 1, size)]
