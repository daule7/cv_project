"""Step 5a: final CSV (HMM = main method) and possession summary."""
import shutil

import numpy as np
import pandas as pd

from . import config as C
from . import detect


def spells(lab):
    """Maximal runs of H or A: list of (start, stop, label)."""
    out = []
    if not len(lab):
        return out
    br = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.r_[0, br, len(lab)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if lab[a] in ("H", "A"):
            out.append((a, b, lab[a]))
    return out


def run(start, end, method="hmm", log=print):
    src = C.OUT / f"possession_{method}.csv"
    shutil.copyfile(src, C.OUT / "possession.csv")
    lab = pd.read_csv(C.OUT / "possession.csv").label.values.astype(str)
    proc = detect.processed_frames(start, end)
    lines = [f"Possession summary (method: {method}; frames {proc.min()}..{proc.max()}, "
             f"processed {len(proc)} frames)", ""]
    tot = {"H": 0, "A": 0, "D": 0}
    all_sp = []
    for i, (a, b) in enumerate(C.HALVES):
        p = proc[(proc >= a) & (proc <= b)]
        if not len(p):
            continue
        seg = lab[p.min():p.max() + 1]
        c = {k: int((seg == k).sum()) for k in "HAD"}
        n = sum(c.values())
        for k in c:
            tot[k] += c[k]
        sp = spells(seg)
        all_sp += sp
        ha = c["H"] + c["A"]
        lines.append(f"Half {i + 1} (frames {p.min()}..{p.max()}, {n} frames):")
        lines.append(f"  H {c['H'] / n:6.1%}   A {c['A'] / n:6.1%}   D {c['D'] / n:6.1%}")
        lines.append(f"  possession H/(H+A) = {c['H'] / ha:5.1%}   A/(H+A) = {c['A'] / ha:5.1%}")
        lines.append(f"  possessions: {len(sp)} (H {sum(s[2] == 'H' for s in sp)}, "
                     f"A {sum(s[2] == 'A' for s in sp)}), mean length "
                     f"{np.mean([s[1] - s[0] for s in sp]) / C.FPS:.1f} s")
        lines.append("")
    n = sum(tot.values())
    ha = tot["H"] + tot["A"]
    lines.append(f"Match ({n} frames):")
    lines.append(f"  H {tot['H'] / n:6.1%}   A {tot['A'] / n:6.1%}   D {tot['D'] / n:6.1%}")
    lines.append(f"  possession H/(H+A) = {tot['H'] / ha:5.1%}   A/(H+A) = {tot['A'] / ha:5.1%}")
    lines.append(f"  possessions: {len(all_sp)} (H {sum(s[2] == 'H' for s in all_sp)}, "
                 f"A {sum(s[2] == 'A' for s in all_sp)}), mean length "
                 f"{np.mean([s[1] - s[0] for s in all_sp]) / C.FPS:.1f} s")
    text = "\n".join(lines) + "\n"
    (C.OUT / "summary.txt").write_text(text)
    log(text)
    return text
