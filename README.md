# DIT-LBP: Directional Illumination Transition Patterns for Physical Surface Texture Characterization

A proposed descriptor (research prototype, not an established algorithm) that measures **how
local binary patterns change when the light direction is reversed**, and uses that change to
tell a physically raised texture from a flat printed copy of it. It is compared against
standard **LBP** and **LDB** (Local Difference Binary), each used on one image and on all
four images, and against a simple image-difference baseline.

## Idea

Single-image LBP sees a printed photograph of a rough surface and the surface itself as
almost the same texture. Under a moving light they differ: on a relief, shading and shadows
move with the light, so the LBP code of a pixel changes; on a flat matte print the pattern is
fixed and only the overall brightness changes. LBP codes are invariant to monotonic
brightness changes, so the code of a print pixel should stay the same.

For each aligned pixel, with lights from tilt 0°, 90°, 180°, 270° (camera fixed):

```
h = popcount( LBP(I_0)  XOR LBP(I_180) )   # 0..8 changed bits, horizontal pair
v = popcount( LBP(I_90) XOR LBP(I_270) )   # 0..8 changed bits, vertical pair
DIT-LBP = normalised joint histogram of (h, v)   -> 9 x 9 = 81 bins
```

Example from the proposal: `11001010 XOR 10001110 = 01000100` → 2 changed bits (this is a unit test).

LBP here is the classic integer-offset 8-neighbour operator (no interpolation), so codes are
exactly invariant to any monotonic grey-level change (exposure, gamma). The skimage
interpolated LBP is **not**: a gamma change alone flips diagonal bits. The tests caught this.

## Dataset: PhoTex (real surfaces, real moving light)

[PhoTex](https://www.macs.hw.ac.uk/texturelab/resources/databases/photex/) (Texturelab,
Heriot-Watt University) photographs rough surfaces with a fixed camera while the light moves,
at many slants (zenith) and tilts (azimuth). Images are 1280×1024, 16-bit. Tilts 0/90/180/270
are exactly the four opposite light directions this method needs. `scripts/download_photex.py`
fetches, for each texture, up to two complete four-light sets at slant 45° and 60°.

The rig also imaged a flat, featureless white card (`reg`) under the same lights. It is used
as the **measured light field** of the rig (flat-field correction and print rendering) and the
**sensor-noise level**, and it is the one real flat surface in the evaluation.

**Run used for the results below:** 20 textures / 35 four-light sets (the download from the
Heriot-Watt server is slow, about 12 MB/min, and was stopped there because of time), 1 rendering seed.
Running `download_photex.py` to completion gives 29 textures. Only the 45° white-card set was downloaded, so the 60° sets are flat-fielded with the 45° light field (an approximation).

### The "printed" class is rendered, not photographed

PhoTex has no printed copies of its textures, so the flat class is semi-synthetic, built
from measured quantities:

- print reflectance = a photograph of the same real surface under one random light
  (shadows baked in), flat-field corrected;
- "realistic" print: dot-gain blur (σ = 1 px), 2% paper grain, tone curve γ = 0.85;
  "ideal" print: none of these;
- each observation = reflectance × the **measured** light field of the rig for that light
  direction (white card) × matched exposure;
- both classes then get the same fresh sensor noise, with σ measured on the white card
  (≈ 0.8% of signal).

So a single printed image is, by construction, very close to a single raised image. This is
the "photo of a texture vs the texture" problem. It is a controlled stand-in for real prints,
and **the main limitation of this study**: a real printed-vs-raised capture (smartphone +
movable LED, as in the proposal) is the necessary next step.

### Conditions (applied identically to both classes unless stated)

| Condition | What changes |
|---|---|
| ideal | ideal print (no print artefacts) |
| realistic | realistic print (blur, grain, tone curve) |
| glossy | realistic + a broad specular sheen on the print that moves with the light |
| exposure | per-image random gain 0.6–1.4 and gamma 0.8–1.25 |
| misaligned | images under 180° and 270° shifted by 1–2 px (no registration) |
| relief_0.30 / 0.10 / 0.03 | relief made shallower: the light-dependent part of each image is scaled by α, same noise; ideal prints |

## Methods compared

| Method | Images | Dims |
|---|---|---|
| LBP (1 image) | 1 | 59 (LBP^u2 8,1) |
| LBP (4 images) | 4 | 236 (concatenated) |
| LDB (1 image) | 1 | 1386 (dense LDB, grids 2–5, bit firing rates) |
| LDB (4 images) | 4 | 5544 |
| Image difference (4 images) | 4 | 38 (histograms/moments of (I_a−I_b)/(I_a+I_b)) |
| **DIT-LBP (proposed)** | 4 | 81 |
| DIT-LBP multi-radius (extension) | 4 | 243 (R = 1, 2, 3) |
| DIT-LBP + LBP (1 image) (extension) | 4 | 140 |
| DIT-LBP noise-gated (extension) | 4 | 81, flips counted only for bits with \|n−c\| ≥ 2√2·σ̂ in both images (LTP-style dead zone) |

LDB follows Yang & Cheng (ISMAR 2012 / TPAMI 2014): a window is split into n×n cells
(n = 2..5); for every cell pair and for mean intensity, mean d/dx and mean d/dy one bit is
emitted. It is a keypoint descriptor, so here it is evaluated densely on 32×32 windows
(stride 16) and the descriptor is the firing rate of each of the 1386 bits.

**Protocol.** 512×512 working images (centre crop, 2× binning) → 16 patches of 128×128 per
image set. Classifier: RBF-SVM (C = 10) on √-scaled, standardised features, identical for
every method. **Grouped 5-fold cross-validation by texture**: a texture and its print are
never split across train and test. Metrics: accuracy, macro F1, and feature-extraction time per patch set.

## Results

Evaluation in progress; results/tables.md and results/figures/ will be added when the run finishes.

## Reproduce

```bash
conda env create -f environment.yml
bash run.sh scripts/download_photex.py --out data/raw --slants 45 60 --max-quads 2
bash run.sh -m pytest -q tests
bash run.sh scripts/run_experiment.py --seeds 0 1 2
bash run.sh scripts/make_figures.py
```

`run.sh` puts the conda env's DLL folders on PATH (on Windows, without it scipy's BLAS can
crash with 0xc06d007f) and adds `src/` to `PYTHONPATH`.

## Layout

```
src/ditlbp/features.py   LBP, DIT-LBP (+ noise-gated), LDB, image-difference descriptors
src/ditlbp/data.py       PhoTex loading, flat-field, print rendering, stress conditions
src/ditlbp/extract.py    all descriptors per patch, timing
scripts/                 download, experiment, figures
tests/                   unit tests (incl. the XOR example)
results/                 CSVs, tables.md, figures/
```

## Novelty note

Targeted searches did not find this exact combination (pixelwise opposite-illumination LBP
bit-transition counts, 81-bin joint histogram, printed-vs-raised classification). Related
work exists: LBP under varying illumination direction, photometric stereo, and
presentation-attack (print) detection. A full search of Google Scholar, IEEE Xplore, Scopus
and patents is still needed before any first-of-its-kind claim.

## Data licence

PhoTex images are © Texturelab, Heriot-Watt University. They are not redistributed here; the
download script fetches them from the original server.
