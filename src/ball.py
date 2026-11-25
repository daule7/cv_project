"""Step 3: ball track.

1. Candidates: ball detections with conf >= BALL_MIN_CONF whose centre is on the grass mask and not
   below the near touchline (bench / technical area / stands are rejected).
2. Temporal selection with a Viterbi pass over {candidates of the frame} + {no ball}: moving from
   one candidate to the next costs a small motion term and is forbidden beyond BALL_MAX_STEP_PX per
   frame (unrealistic jump); a new track can only start from a detection with conf >=
   BALL_START_CONF. This removes isolated false positives (lines, heads, socks).
3. Segments shorter than BALL_MIN_SEGMENT frames are dropped; gaps up to 0.5 s are filled by linear
   interpolation, longer gaps stay empty.
Also records, per frame, whether a confident ball was seen OFF the pitch (cue for dead ball).
Output: out/ball.parquet with one row per processed frame.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import detect

NONE_COST = -np.log(0.15)   # unary cost of "no ball" (a candidate with conf > 0.15 is cheaper)
START_COST = 1.0
END_COST = 1.0
MOTION_SCALE = 30.0         # px/frame; motion cost = 0.5 * (d / (MOTION_SCALE*dt))^2


def viterbi_select(frames, cands):
    """frames: sorted processed frames; cands: dict frame -> array (n,3) [x, y, conf]."""
    n = len(frames)
    back = []
    prev_cost = np.array([0.0])          # state 0 = none
    prev_xy = np.zeros((0, 2))
    prev_f = None
    for t, f in enumerate(frames):
        c = cands.get(f, np.zeros((0, 3)))
        k = len(c)
        unary = np.concatenate([[NONE_COST], -np.log(np.clip(c[:, 2], 1e-3, 1))])
        dt = 1 if prev_f is None else max(1, f - prev_f)
        m = len(prev_cost)
        trans = np.full((m, k + 1), np.inf)
        trans[:, 0] = 0.0                       # anything -> none
        trans[1:, 0] = END_COST
        start_ok = c[:, 2] >= C.BALL_START_CONF if k else np.zeros(0, bool)
        trans[0, 1:] = np.where(start_ok, START_COST, np.inf)
        if m > 1 and k:
            d = np.linalg.norm(prev_xy[:, None, :] - c[None, :, :2], axis=2)
            mc = 0.5 * (d / (MOTION_SCALE * dt)) ** 2
            mc[d > C.BALL_MAX_STEP_PX * dt] = np.inf
            # allow a fresh start even when a track exists (jump through "none" is implicit)
            trans[1:, 1:] = mc
        tot = prev_cost[:, None] + trans
        arg = np.argmin(tot, axis=0)
        prev_cost = tot[arg, np.arange(k + 1)] + unary
        back.append(arg)
        prev_xy = c[:, :2]
        prev_f = f
    # backtrack
    sel = np.zeros(n, int)
    s = int(np.argmin(prev_cost))
    for t in range(n - 1, -1, -1):
        sel[t] = s
        s = int(back[t][s])
    return sel   # 0 = none, i>0 = candidate i-1


def run(start, end, log=print):
    det = detect.load(start, end)
    frames = detect.processed_frames(start, end)
    B = det[det.cls == C.CLS_BALL]
    raw_any = set(B[B.conf >= 0.25].frame)
    cand = B[(B.conf >= C.BALL_MIN_CONF) & B.in_field]
    off = B[(B.conf >= 0.4) & ~B.in_field]
    cx = ((cand.x1 + cand.x2) / 2).values
    cy = ((cand.y1 + cand.y2) / 2).values
    arr = np.stack([cx, cy, cand.conf.values], 1)
    cands = {}
    for f, idx in cand.groupby("frame").indices.items():
        a = arr[idx]
        cands[int(f)] = a[np.argsort(-a[:, 2])]

    out = []
    for a, b in C.HALVES:   # never link across halftime
        fr = frames[(frames >= a) & (frames <= b)]
        if not len(fr):
            continue
        sel = viterbi_select(fr, cands)
        xy = np.full((len(fr), 2), np.nan)
        cf = np.zeros(len(fr))
        for i, (f, s) in enumerate(zip(fr, sel)):
            if s > 0:
                xy[i] = cands[int(f)][s - 1, :2]
                cf[i] = cands[int(f)][s - 1, 2]
        det_mask = ~np.isnan(xy[:, 0])
        # drop very short segments
        seg_id = np.cumsum(np.r_[True, det_mask[1:] != det_mask[:-1]])
        for sid in np.unique(seg_id[det_mask]):
            idx = np.flatnonzero(seg_id == sid)
            if len(idx) < C.BALL_MIN_SEGMENT:
                xy[idx] = np.nan
                cf[idx] = 0
        det_mask = ~np.isnan(xy[:, 0])
        src = det_mask.astype(np.int8)                # 1 = detected
        # interpolate short gaps (in frames, not processed steps)
        known = np.flatnonzero(det_mask)
        for i0, i1 in zip(known[:-1], known[1:]):
            if i1 - i0 > 1 and fr[i1] - fr[i0] <= C.BALL_INTERP_MAX + 1:
                w = (fr[i0 + 1:i1] - fr[i0]) / (fr[i1] - fr[i0])
                xy[i0 + 1:i1] = xy[i0] + w[:, None] * (xy[i1] - xy[i0])
                src[i0 + 1:i1] = 2                    # 2 = interpolated
        out.append(pd.DataFrame({"frame": fr, "bx": xy[:, 0], "by": xy[:, 1], "bconf": cf,
                                 "src": src, "off_pitch": np.isin(fr, off.frame.values),
                                 "raw_any": np.isin(fr, list(raw_any))}))
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(C.OUT / "ball.parquet", index=False)
    n = len(df)
    log(f"ball: {n} frames; raw detection conf>=0.25 in {df.raw_any.mean():.3f}; "
        f"after filtering detected {(df.src == 1).mean():.3f}, +interpolated "
        f"{(df.src == 2).mean():.3f}, total with position {(df.src > 0).mean():.3f}")
    return df
