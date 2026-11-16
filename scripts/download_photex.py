"""Download the PhoTex opposite-illumination subset used by this project.

PhoTex (Texturelab, Heriot-Watt University) images rough surfaces with a fixed
camera while the light source moves. File names follow
    [investigation].[texture].[rotation].[slant].[tilt].tif
where slant is the light's zenith angle and tilt its azimuth.

For each texture we keep complete "quads": the four images at tilt
0/90/180/270 deg for one (investigation, rotation, slant). These are the four
opposite light directions DIT-LBP needs (0 vs 180 and 90 vs 270).

Usage:
    python scripts/download_photex.py --out data/raw --slants 45 60 --max-quads 2
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE = "http://www.macs.hw.ac.uk/texturelab/files/resources/Photex"
# Grayscale texture directories with tilt 0/90/180/270 quads (checked 2026-11-16).
TEXTURES = [
    "acc", "acd", "ace", "acf", "acg", "ach", "aci", "acj", "ack", "acl", "acm",
    "ada", "adb", "adc", "add", "ade", "adf", "adg", "adh", "adj", "adk", "adl",
    "adm", "afa", "afb", "afc", "afd", "afe", "afg",
]
# Flat registration target imaged on the same rig: a real flat control.
CONTROLS = ["reg"]
TILTS = (0, 90, 180, 270)
NAME_RE = re.compile(r"^(\w+)\.(\w+)\.(\d+)\.(\d+)\.(\d+)\.tif$")


def list_dir(session: requests.Session, d: str, retries: int = 5) -> list[str]:
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/{d}/", timeout=60)
            r.raise_for_status()
            return re.findall(r'href="([^"?/][^"]*\.tif)"', r.text)
        except requests.RequestException as e:  # pragma: no cover - network
            print(f"  list retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5)
    raise RuntimeError(f"cannot list {d}")


def select_quads(files: list[str], slants: list[int], max_quads: int):
    groups = defaultdict(dict)
    for f in files:
        m = NAME_RE.match(f)
        if not m:
            continue  # skips colour-channel files such as *_RED.tif
        inv, tex, rot, slant, tilt = m.groups()
        if int(slant) in slants and int(tilt) in TILTS:
            groups[(inv, tex, int(rot), int(slant))][int(tilt)] = f
    full = [(k, v) for k, v in groups.items() if len(v) == 4]
    # Prefer rotation 0, then spread over the requested slants.
    full.sort(key=lambda kv: (kv[0][2] != 0, slants.index(kv[0][3]), kv[0][2], kv[0][0]))
    chosen, seen_slant = [], defaultdict(int)
    for k, v in full:
        if len(chosen) >= max_quads:
            break
        if seen_slant[k[3]] >= max(1, max_quads // len(slants)) and len(chosen) < max_quads - 1:
            continue
        chosen.append((k, v))
        seen_slant[k[3]] += 1
    return chosen


def fetch(session, url: str, dest: Path, retries: int = 6) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=180)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except requests.RequestException as e:  # pragma: no cover - network
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"failed: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--slants", type=int, nargs="+", default=[45, 60])
    ap.add_argument("--max-quads", type=int, default=2)
    ap.add_argument("--textures", nargs="+", default=TEXTURES + CONTROLS)
    args = ap.parse_args()
    out = Path(args.out)
    s = requests.Session()
    s.headers["User-Agent"] = "dit-lbp-research-download/0.1"
    total = 0
    for d in args.textures:
        quads = select_quads(list_dir(s, d), args.slants, args.max_quads)
        if not quads:
            print(f"{d}: no complete quad")
            continue
        (out / d).mkdir(parents=True, exist_ok=True)
        for key, files in quads:
            for tilt in TILTS:
                fetch(s, f"{BASE}/{d}/{files[tilt]}", out / d / files[tilt])
                total += 1
        print(f"{d}: {[k for k, _ in quads]}", flush=True)
    print(f"done, {total} files")


if __name__ == "__main__":
    main()
