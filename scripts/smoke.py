import time, numpy as np
from pathlib import Path
from ditlbp.data import *
from ditlbp.extract import extract_stack
RAW = Path("data/raw")
rig = build_rig(RAW); print("rig slants", sorted(rig.field), {s: {t: round(v) for t, v in rig.sigma[s].items()} for s in rig.sigma})
for key in find_quads(RAW)[:4]:
    raw = load_quad(RAW, key); F, sig, _ = rig.for_slant(key.slant)
    sh = estimate_shifts(flat_field(raw, F))
    rng = np.random.default_rng(0)
    pr, src = render_print(raw, F, sig, rng)
    t0 = time.time()
    for name, r in (("raised", raw), ("printed", pr)):
        st, _ = prepare(observe(r, sig, rng), F, "realistic", rng, do_align=False)
        f, fl = extract_stack(st)
        print(key.texture, key.slant, name, "flips h/v", fl.mean(0).round(2), "dims", {m[:12]: v.shape[1] for m, v in f.items()} if name == "raised" else "")
    print("  shifts", {t: s.round(2).tolist() for t, s in sh.items()}, f"{time.time()-t0:.1f}s")
