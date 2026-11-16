"""Figures and markdown tables from results/ (run after run_experiment.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ditlbp.data import (TILTS, build_rig, find_quads, load_quad, observe, prepare,  # noqa: E402
                         render_print)
from ditlbp.features import dit_gated_maps, dit_maps, lbp_codes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, RAW = ROOT / "results", ROOT / "results" / "figures", ROOT / "data" / "raw"
FIG.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc3f0", "#2a78d6", "#0d3a73"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT,
                     "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["LBP (1 image)", "LBP (4 images)", "LDB (1 image)", "LDB (4 images)",
         "Image difference (4 images)", "DIT-LBP (proposed)", "DIT-LBP multi-radius",
         "DIT-LBP + LBP (1 image)", "DIT-LBP noise-gated"]
COND_ORDER = ["ideal", "realistic", "glossy", "exposure", "misaligned",
              "relief_0.30", "relief_0.10", "relief_0.03"]


def summary(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_csv(RES / "results_per_seed.csv")
    r = r[r.protocol == protocol]
    g = r.groupby(["method", "test"]).accuracy
    m, s = g.mean().unstack(), g.std().unstack()
    cols = [c for c in COND_ORDER if c in m.columns]
    rows = [x for x in ORDER if x in m.index]
    return m.loc[rows, cols], s.loc[rows, cols]


def heatmap(m: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(1.15 * m.shape[1] + 3.2, 0.42 * m.shape[0] + 1.3))
    im = ax.imshow(m.values, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.82 else TXT)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]), m.index)
    for lab in ax.get_yticklabels():
        if "proposed" in lab.get_text():
            lab.set_fontweight("bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("accuracy (chance = 0.5)")
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def relief_plot(m: pd.DataFrame, s: pd.DataFrame):
    alphas = {"ideal": 1.0, "relief_0.30": 0.3, "relief_0.10": 0.1, "relief_0.03": 0.03}
    cols = [c for c in alphas if c in m.columns]
    if len(cols) < 2:
        return
    show = ["LBP (1 image)", "LBP (4 images)", "LDB (4 images)", "Image difference (4 images)",
            "DIT-LBP (proposed)", "DIT-LBP noise-gated"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.array([alphas[c] for c in cols])
    for k, meth in enumerate(show):
        if meth not in m.index:
            continue
        y, e = m.loc[meth, cols].values, s.loc[meth, cols].values
        ax.plot(x, y, color=SERIES[k], lw=2, marker="o", ms=6, label=meth,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        if np.all(np.isfinite(e)):
            ax.fill_between(x, y - e, y + e, color=SERIES[k], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.invert_xaxis()
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color=TXT2, lw=1, ls=":")
    ax.text(x.min(), 0.505, "chance", color=TXT2, fontsize=8, ha="right", va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlabel("relative relief depth  (1 = real PhoTex surface)")
    ax.set_ylabel("accuracy, raised vs flat print")
    ax.set_title("Accuracy as the relief gets shallower (ideal prints, same sensor noise)", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_relief_sweep.png", dpi=160)
    plt.close(fig)


def example_figure():
    rig = build_rig(RAW)
    key = find_quads(RAW)[0]
    raw = load_quad(RAW, key)
    F, sig, _ = rig.for_slant(key.slant)
    rng = np.random.default_rng(7)
    pr, src = render_print(raw, F, sig, rng)
    flipcm = LinearSegmentedColormap.from_list("flip", ["#ffffff", "#9cc3f0", "#2a78d6", "#0d3a73"], N=9)
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.8))
    for r, (name, stack_raw) in enumerate((("raised (real surface)", raw),
                                           (f"flat print (photo under tilt {src})", pr))):
        st, _ = prepare(observe(stack_raw, sig, rng), F, "realistic", rng, do_align=False)
        crop = (slice(0, 192), slice(0, 192))
        for c, t in enumerate(TILTS):
            a = st[t][crop]
            axes[r, c].imshow(a, cmap="gray", vmin=np.percentile(a, 1), vmax=np.percentile(a, 99))
            axes[r, c].set_title(f"light tilt {t}°", fontsize=9)
        h, v = dit_maps({t: lbp_codes(st[t]) for t in TILTS})
        for c, (mp, lab) in enumerate(((h, "flips 0° vs 180°"), (v, "flips 90° vs 270°"))):
            im = axes[r, 4 + c].imshow(mp[crop], cmap=flipcm, vmin=-0.5, vmax=8.5)
            axes[r, 4 + c].set_title(f"{lab}  (mean {mp.mean():.2f})", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=10, color=TXT)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        a.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axes[:, 4:], fraction=0.04, ticks=range(9))
    cb.set_label("changed LBP bits (of 8)")
    cb.outline.set_visible(False)
    fig.suptitle(f"PhoTex texture '{key.texture}', slant {key.slant}°: four lights and DIT flip maps",
                 x=0.01, ha="left", color=TXT)
    fig.savefig(FIG / "fig1_example.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def dit_hist_figure():
    z = np.load(RES / "features.npz")
    meta = pd.read_csv(RES / "meta.csv")
    X = z["DIT-LBP (proposed)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (lab, name) in zip(axes, ((0, "raised"), (1, "flat print"))):
        sel = ((meta.condition == "realistic") & (meta.label == lab)).values
        H = X[sel].mean(0).reshape(9, 9)
        im = ax.imshow(np.log10(H + 1e-5), cmap=SEQ, vmin=-5, vmax=0, origin="lower")
        ax.set_title(f"{name}: mean 81-bin DIT-LBP histogram", fontsize=10, loc="left")
        ax.set_xlabel("vertical flips (90° vs 270°)")
        ax.set_ylabel("horizontal flips (0° vs 180°)")
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
    cb = fig.colorbar(im, ax=axes, fraction=0.03)
    cb.set_label("log10 fraction of pixels")
    cb.outline.set_visible(False)
    fig.savefig(FIG / "fig4_dit_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def md_table(m: pd.DataFrame, s: pd.DataFrame) -> str:
    head = "| Method | " + " | ".join(m.columns) + " |\n|---|" + "---|" * m.shape[1] + "\n"
    lines = []
    for meth in m.index:
        cells = []
        best = m.max(axis=0)
        for c in m.columns:
            txt = f"{m.loc[meth, c]:.3f} ± {s.loc[meth, c]:.3f}"
            cells.append(f"**{txt}**" if np.isclose(m.loc[meth, c], best[c]) else txt)
        lines.append(f"| {meth} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def main():
    m, s = summary("within")
    heatmap(m, "Within-condition accuracy (grouped 5-fold CV by texture, seed 0)",
            FIG / "fig2_accuracy_within.png")
    relief_plot(m, s)
    mc, sc = summary("cross")
    heatmap(mc, "Trained on 'realistic', tested on other conditions (held-out textures)",
            FIG / "fig5_accuracy_cross.png")
    example_figure()
    dit_hist_figure()
    r = pd.read_csv(RES / "results_per_seed.csv")
    f1 = r[r.protocol == "within"].groupby(["method", "test"]).f1.mean().unstack()
    tm = pd.read_csv(RES / "timing.csv", index_col=0).iloc[:, 0]
    with open(RES / "tables.md", "w", encoding="utf-8") as fh:
        fh.write("## Within-condition accuracy (mean ± std over 3 seeds)\n\n" + md_table(m, s))
        fh.write("\n## Macro F1, within-condition (mean over seeds)\n\n")
        f1 = f1.loc[m.index, m.columns]
        fh.write(md_table(f1, f1 * 0))
        fh.write("\n## Trained on realistic, tested on stress conditions (accuracy)\n\n" + md_table(mc, sc))
        fh.write("\n## Feature extraction time per 128x128 patch stack (ms, median)\n\n| Method | ms |\n|---|---|\n")
        for k in ORDER:
            if k in tm.index:
                fh.write(f"| {k} | {tm[k]:.2f} |\n")
        fh.write("\n## Mean LBP bit flips per pixel (of 8)\n\n")
        fh.write(pd.read_csv(RES / "mean_flips.csv").to_markdown(index=False) + "\n")
        fh.write("\n## Real flat control: PhoTex white card ('reg')\n\n")
        fh.write(pd.read_csv(RES / "reg_control.csv").round(3).to_markdown(index=False) + "\n")
    print("figures and tables written")


if __name__ == "__main__":
    main()
