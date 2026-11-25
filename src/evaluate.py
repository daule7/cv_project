"""Step 6: evaluation against correct.csv with the official Metrica scorer (src/vendor).

Only frames that were actually processed (and are graded in correct.csv) are scored: the ground
truth is set to X elsewhere, so a partial run is evaluated on its own frames only.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import detect
from .vendor import possession_metrics as pm


def _fmt_cm(cm):
    labs = ["H", "A", "D", "X"]
    s = "| GT \\ pred | " + " | ".join(labs) + " |\n|---|" + "---|" * 4 + "\n"
    for i in range(3):
        s += f"| {labs[i]} | " + " | ".join(str(v) for v in cm[i]) + " |\n"
    return s


def run(start, end, log=print):
    gt_df = pm.read_label_csv(C.GT_CSV)
    gt = pm.label_codes(gt_df)
    proc = detect.processed_frames(start, end)
    lo, hi = int(proc.min()), int(proc.max())
    mask = np.zeros(len(gt), bool)
    for a, b in C.HALVES:   # every frame between first and last processed frame of each half
        p = proc[(proc >= a) & (proc <= b)]
        if len(p):
            mask[p.min():p.max() + 1] = True
    gt_eval = gt.copy()
    gt_eval[~mask] = pm.X
    n_graded = int((gt_eval != pm.X).sum())

    rows, details = [], {}
    for name in ("baseline", "hmm"):
        pdf = pm.read_label_csv(C.OUT / f"possession_{name}.csv")
        err = pm.validate_submission(pdf, len(gt))
        if err:
            raise ValueError(f"{name}: {err}")
        pred = pm.label_codes(pdf)
        r = pm.evaluate(gt_eval, pred)
        details[name] = r
        rows.append((name, r))

    ball = pd.read_parquet(C.OUT / "ball.parquet")
    ball = ball[ball.frame.isin(np.flatnonzero(mask))]
    gtb = gt[ball.frame.values]
    ball_raw = ball.raw_any.mean()
    ball_det = (ball.src == 1).mean()
    ball_any = (ball.src > 0).mean()
    alive = gtb != pm.D

    md = [f"# Evaluation vs correct.csv\n",
          f"Scored frames: {n_graded} (frames {lo}..{hi} that are graded; processed frames only).",
          f"Scorer: official `evaluation/possession_metrics.py` from "
          f"github.com/metrica-sports/more-than-a-hack-2027 (copied to src/vendor/).\n",
          "| method | accuracy | macro-F1 | spellF1 (mean IoU .3-.7) | spellF1@.5 | F1 H | F1 A | "
          "F1 D | alive/dead F1 | team acc (both alive) | home share pred / GT | spells pred / GT "
          "| mean spell s pred / GT |",
          "|---|" + "---|" * 12]
    for name, r in rows:
        pc = r["per_class"]
        md.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['spell_f1_mean']:.3f} "
                  f"| {r['spell_f1_at_headline']:.3f} | {pc['H']['f1']:.3f} | {pc['A']['f1']:.3f} "
                  f"| {pc['D']['f1']:.3f} | {r['alive_dead_f1']:.3f} | {r['team_acc_both_alive']:.3f} "
                  f"| {r['pred_home_share']:.3f} / {r['gt_home_share']:.3f} "
                  f"| {r['n_pred_spells']} / {r['n_gt_spells']} "
                  f"| {r['mean_spell_duration_s']:.1f} / {r['gt_mean_spell_duration_s']:.1f} |")
    for name, r in rows:
        md.append(f"\n## Confusion matrix — {name} (rows = GT, cols = prediction)\n")
        md.append(_fmt_cm(np.array(r["confusion"])))
        md.append("spellF1 per IoU: " + ", ".join(
            f"{t}: {v['f1']:.3f}" for t, v in r["spell_f1"].items()))
    md.append("\n## Ball visibility on scored frames\n")
    md.append(f"- raw detector, ball conf>=0.25 anywhere in frame: {ball_raw:.3f}")
    md.append(f"- after filtering (accepted detections): {ball_det:.3f} "
              f"(GT alive frames {(ball.src == 1)[alive].mean():.3f}, GT dead "
              f"{(ball.src == 1)[~alive].mean():.3f})")
    md.append(f"- with position incl. interpolation of gaps <= 0.5 s: {ball_any:.3f}")
    text = "\n".join(md) + "\n"
    (C.OUT / "eval.md").write_text(text)
    log(text)
    return details
