"""Raised-vs-printed experiment on PhoTex: DIT-LBP vs LBP, LDB and image differences.

Stage 1 (features): for every PhoTex quad, seed and condition, render the flat
print, observe both classes, normalise, cut 128x128 patches and extract all
descriptors. Cached to results/features.npz.
Stage 2 (evaluation): grouped 5-fold CV by texture (a texture and its print are
never split across train and test), RBF-SVM on sqrt-scaled features.

Usage:
    bash run.sh scripts/run_experiment.py            # everything
    bash run.sh scripts/run_experiment.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from ditlbp.data import (CONDITIONS, TILTS, build_rig, find_quads, load_quad, observe,
                         prepare, relief_alpha, render_print, shallow, to_patches)
from ditlbp.extract import METHODS, extract_stack, time_methods

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def process_quad(key, rig, seeds, conditions, align: bool):
    raw_full = load_quad(RAW, key)
    F, sig, _ = rig.for_slant(key.slant)
    rows, feats, flips, shift_log = [], {m: [] for m in METHODS}, [], []
    for seed in seeds:
        for cond in conditions:
            rng = rng_for(seed, key, cond)
            raw = shallow(raw_full, F, relief_alpha(cond))
            printed, src = render_print(raw, F, sig, rng,
                                        quality="ideal" if cond == "ideal" or cond.startswith("relief") else "realistic",
                                        glossy=cond == "glossy")
            for label, r in ((0, raw), (1, printed)):
                stack, shifts = prepare(observe(r, sig, rng), F, cond, rng, do_align=align)
                if label == 0 and seed == seeds[0] and cond == "realistic":
                    shift_log.append({"texture": key.texture, "slant": key.slant,
                                      **{f"shift_{t}": float(np.abs(shifts[t]).max()) for t in TILTS}})
                f, fl = extract_stack(stack)
                for m in METHODS:
                    feats[m].append(f[m])
                flips.append(fl)
                for p in range(len(fl)):
                    rows.append(dict(texture=key.texture, inv=key.investigation, rot=key.rotation,
                                     slant=key.slant, seed=seed, condition=cond, label=label,
                                     src_tilt=src if label else -1, patch=p))
    return rows, {m: np.concatenate(v) for m, v in feats.items()}, np.concatenate(flips), shift_log


def build_features(seeds, conditions, align, n_jobs):
    rig = build_rig(RAW)
    quads = find_quads(RAW)
    print(f"{len(quads)} quads from {len({q.texture for q in quads})} textures; "
          f"rig slants {sorted(rig.field)}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_quad)(q, rig, seeds, conditions, align) for q in quads)
    print(f"features in {time.time() - t0:.0f}s", flush=True)
    meta = pd.DataFrame([r for rows, *_ in res for r in rows])
    X = {m: np.concatenate([f[m] for _, f, _, _ in res]) for m in METHODS}
    flips = np.concatenate([fl for _, _, fl, _ in res])
    shifts = pd.DataFrame([s for *_, sl in res for s in sl])
    meta["flip_h"], meta["flip_v"], meta["gflip_h"], meta["gflip_v"] = flips.T
    OUT.mkdir(exist_ok=True)
    meta.to_csv(OUT / "meta.csv", index=False)
    shifts.to_csv(OUT / "alignment_shifts.csv", index=False)
    np.savez_compressed(OUT / "features.npz", **{m: X[m].astype(np.float32) for m in METHODS})
    return meta, X, rig


def load_features():
    meta = pd.read_csv(OUT / "meta.csv")
    z = np.load(OUT / "features.npz")
    return meta, {m: z[m] for m in METHODS}


def make_clf():
    return make_pipeline(FunctionTransformer(np.sqrt), StandardScaler(),
                         SVC(C=10, gamma="scale", kernel="rbf"))


def cv_eval(meta, X, train_cond, test_cond, seed, n_splits=5):
    tr_rows = (meta.condition == train_cond) & (meta.seed == seed)
    te_rows = (meta.condition == test_cond) & (meta.seed == seed)
    textures = np.array(sorted(meta.texture.unique()))
    out = {}
    for m, Xm in X.items():
        y_true, y_pred = [], []
        for tr_tex, te_tex in GroupKFold(n_splits).split(textures, groups=textures):
            tr = tr_rows & meta.texture.isin(textures[tr_tex])
            te = te_rows & meta.texture.isin(textures[te_tex])
            clf = make_clf().fit(Xm[tr.values], meta.label[tr].values)
            y_pred.append(clf.predict(Xm[te.values]))
            y_true.append(meta.label[te].values)
        yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
        out[m] = dict(accuracy=accuracy_score(yt, yp), f1=f1_score(yt, yp, average="macro"))
    return out


def evaluate(meta, X, seeds, conditions):
    recs = []
    for seed in seeds:
        for cond in conditions:
            for m, r in cv_eval(meta, X, cond, cond, seed).items():
                recs.append(dict(protocol="within", train=cond, test=cond, seed=seed, method=m, **r))
            if cond != "realistic":
                for m, r in cv_eval(meta, X, "realistic", cond, seed).items():
                    recs.append(dict(protocol="cross", train="realistic", test=cond, seed=seed, method=m, **r))
        print(f"seed {seed} evaluated", flush=True)
    return pd.DataFrame(recs)


def reg_control(rig, meta, X, seed):
    """Real flat surface (PhoTex white card): fraction of patches called 'printed'
    by models trained on all realistic data, plus its mean flip counts."""
    from ditlbp.extract import extract_stack
    rng = rng_for("reg", seed)
    recs = []
    for slant, raw in rig.reg_raw.items():
        F, sig, _ = rig.for_slant(slant)
        stack, _ = prepare(observe(raw, sig, rng), F, "realistic", rng, do_align=False)
        f, fl = extract_stack(stack)
        tr = ((meta.condition == "realistic") & (meta.seed == seed)).values
        for m in METHODS:
            clf = make_clf().fit(X[m][tr], meta.label[tr].values)
            recs.append(dict(slant=slant, method=m, frac_called_flat=float(clf.predict(f[m]).mean()),
                             flip_h=float(fl[:, 0].mean()), flip_v=float(fl[:, 1].mean()),
                             gflip_h=float(fl[:, 2].mean()), gflip_v=float(fl[:, 3].mean())))
    return pd.DataFrame(recs)


def timing(rig, n=60):
    quads = find_quads(RAW)
    raw = load_quad(RAW, quads[0])
    F, sig, _ = rig.for_slant(quads[0].slant)
    stack, _ = prepare(observe(raw, sig, rng_for("t")), F, "realistic", rng_for("t"))
    pats = {t: to_patches(stack[t]) for t in TILTS}
    stacks = [{t: pats[t][i % len(pats[0])] for t in TILTS} for i in range(n)]
    return time_methods(stacks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--align", action="store_true",
                    help="phase-correlation alignment; off by default because shading "
                         "changes bias the estimated shift on relief surfaces")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    if args.eval_only:
        meta, X = load_features()
        rig = build_rig(RAW)
    else:
        meta, X, rig = build_features(args.seeds, args.conditions, args.align, args.jobs)

    res = evaluate(meta, X, sorted(meta.seed.unique()), [c for c in CONDITIONS if c in set(meta.condition)])
    res.to_csv(OUT / "results_per_seed.csv", index=False)
    summ = (res.groupby(["protocol", "train", "test", "method"])[["accuracy", "f1"]]
              .agg(["mean", "std"]).round(4))
    summ.to_csv(OUT / "results_summary.csv")

    tm = timing(rig)
    pd.Series(tm, name="ms_per_patch_stack").to_csv(OUT / "timing.csv")
    ctrl = reg_control(rig, meta, X, seed=int(meta.seed.min()))
    ctrl.to_csv(OUT / "reg_control.csv", index=False)

    flips = meta.groupby(["condition", "label"])[["flip_h", "flip_v", "gflip_h", "gflip_v"]].mean().round(3)
    flips.to_csv(OUT / "mean_flips.csv")
    with open(OUT / "methods.json", "w") as fh:
        json.dump({m: {"images": k, "description": d} for m, (k, d) in METHODS.items()}, fh, indent=2)
    pd.set_option("display.width", 200)
    print(summ.xs("within", level="protocol"))
    print(tm)
    print(flips)
    print(ctrl)


if __name__ == "__main__":
    main()
