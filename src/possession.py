"""Step 4: possession per frame, two methods.

Features per processed frame (out/features.parquet):
  dH, dA   distance ball -> nearest H / A player foot point (bottom-centre of the box), divided by
           that player's box height (no homography, camera moves: height is the local scale).
  bv       ball speed in player heights per frame, after subtracting camera motion.
  cam      camera motion = median displacement of all tracked persons between processed frames.
  pv       median camera-compensated player speed in heights/s, smoothed over 2 s.
  miss_s   seconds since the ball was last seen.
(a) baseline: team of the player nearest to the ball (normalised distance), no ball -> D, then a
    majority vote in a +-50-frame window.
(b) HMM with states H/A/D, Viterbi per half; emissions described in emissions().
"""
import numpy as np
import pandas as pd

from . import config as C
from . import detect

LAB = np.array(["H", "A", "D"])
H_, A_, D_ = 0, 1, 2


def features(start, end, log=print):
    P = pd.read_parquet(C.OUT / "persons.parquet")
    P = P[(P.frame >= start) & (P.frame <= end)]
    ball = pd.read_parquet(C.OUT / "ball.parquet")
    ball = ball[(ball.frame >= start) & (ball.frame <= end)].reset_index(drop=True)
    frames = ball.frame.values
    pos = pd.Series(np.arange(len(frames)), index=frames)

    P = P.assign(fx=(P.x1 + P.x2) / 2, fy=P.y2, h=P.y2 - P.y1)
    P = P[P.frame.isin(frames)]
    P = P.assign(i=P.frame.map(pos).values)
    onf = P[P.in_field]

    # ---- camera motion & player speed --------------------------------------------------------
    T = onf[onf.tid >= 0][["i", "frame", "tid", "fx", "fy", "h"]]
    prev = T.assign(i=T.i + 1)
    mm = T.merge(prev, on=["i", "tid"], suffixes=("", "_p"))
    mm = mm[mm.frame_p >= mm.frame - 8]                  # only truly consecutive samples
    mm["dx"] = mm.fx - mm.fx_p
    mm["dy"] = mm.fy - mm.fy_p
    cam = mm.groupby("i")[["dx", "dy"]].median()
    mm = mm.join(cam, on="i", rsuffix="_cam")
    dtf = (mm.frame - mm.frame_p).clip(lower=1)
    mm["sp"] = np.hypot(mm.dx - mm.dx_cam, mm.dy - mm.dy_cam) / mm.h / dtf * C.FPS
    pv = mm.groupby("i").sp.median().reindex(np.arange(len(frames)))
    win = max(1, int(round(C.SLOW_WINDOW_S * C.FPS / max(1, np.median(np.diff(frames))))))
    pv = pv.interpolate(limit_direction="both").rolling(win, center=True, min_periods=1).mean()
    camx = cam.dx.reindex(np.arange(len(frames))).fillna(0).values
    camy = cam.dy.reindex(np.arange(len(frames))).fillna(0).values

    # ---- distances ball -> players ---------------------------------------------------------
    hmed = onf.groupby("i").h.median().reindex(np.arange(len(frames)))
    hmed = hmed.interpolate(limit_direction="both").fillna(45).values
    team = onf[onf.team >= 0]
    order = np.argsort(team.i.values, kind="stable")
    ti = team.i.values[order]
    tx, ty, th, tt = (team.fx.values[order], team.fy.values[order], team.h.values[order],
                      team.team.values[order])
    starts = np.searchsorted(ti, np.arange(len(frames)))
    ends = np.searchsorted(ti, np.arange(len(frames)), side="right")
    bx, by = ball.bx.values, ball.by.values
    dH = np.full(len(frames), np.inf)
    dA = np.full(len(frames), np.inf)
    dAny = np.full(len(frames), np.inf)
    for k in np.flatnonzero(~np.isnan(bx)):
        s, e = starts[k], ends[k]
        if s == e:
            continue
        d = np.hypot(tx[s:e] - bx[k], ty[s:e] - by[k]) / np.maximum(th[s:e], 10)
        t = tt[s:e]
        if (t == 0).any():
            dH[k] = d[t == 0].min()
        if (t == 1).any():
            dA[k] = d[t == 1].min()
        dAny[k] = d.min()
    # nearest of any person (incl. referees) for the "nobody near" cue
    allp = onf
    o2 = np.argsort(allp.i.values, kind="stable")
    ai = allp.i.values[o2]
    ax, ay, ah = allp.fx.values[o2], allp.fy.values[o2], allp.h.values[o2]
    s2 = np.searchsorted(ai, np.arange(len(frames)))
    e2 = np.searchsorted(ai, np.arange(len(frames)), side="right")
    dNear = np.full(len(frames), np.inf)
    for k in np.flatnonzero(~np.isnan(bx)):
        if s2[k] < e2[k]:
            dNear[k] = (np.hypot(ax[s2[k]:e2[k]] - bx[k], ay[s2[k]:e2[k]] - by[k]) /
                        np.maximum(ah[s2[k]:e2[k]], 10)).min()

    # ---- ball speed --------------------------------------------------------------------------
    dfr = np.r_[1, np.diff(frames)].clip(min=1)
    vx = np.r_[np.nan, np.diff(bx)] - camx
    vy = np.r_[np.nan, np.diff(by)] - camy
    bv = np.hypot(vx, vy) / hmed / dfr
    bv[np.r_[True, np.diff(frames) > 8]] = np.nan
    bv = pd.Series(bv).rolling(3, center=True, min_periods=1).median().values

    # ---- time since ball last seen -----------------------------------------------------------
    seen = ~np.isnan(bx)
    miss = np.zeros(len(frames))
    last = None
    for k in range(len(frames)):
        if seen[k]:
            last = frames[k]
        miss[k] = (frames[k] - last) / C.FPS if last is not None else 99.0
        if k and C.half_of(frames[k]) != C.half_of(frames[k - 1]):
            last = frames[k] if seen[k] else None
            miss[k] = 0 if seen[k] else 99.0

    F = pd.DataFrame({"frame": frames, "half": [C.half_of(f) for f in frames],
                      "bx": bx, "by": by, "src": ball.src.values, "off_pitch": ball.off_pitch.values,
                      "dH": dH, "dA": dA, "dNear": dNear, "bv": bv, "pv": pv.values,
                      "camx": camx, "camy": camy, "hmed": hmed, "miss_s": miss})
    F.to_parquet(C.OUT / "features.parquet", index=False)
    log(f"features: {len(F)} frames; median player speed {np.nanmedian(F.pv):.2f} h/s "
        f"(p10 {np.nanpercentile(F.pv, 10):.2f}, p90 {np.nanpercentile(F.pv, 90):.2f}); "
        f"median |camera shift| {np.median(np.hypot(camx, camy)):.1f} px/frame")
    return F


# ---- (a) baseline ------------------------------------------------------------------------------
def baseline(F):
    raw = np.full(len(F), D_)
    has = np.isfinite(F.dH.values) | np.isfinite(F.dA.values)
    raw[has] = np.where(F.dH.values[has] <= F.dA.values[has], H_, A_)
    out = np.empty(len(F), int)
    step = max(1, int(np.median(np.diff(F.frame.values)))) if len(F) > 1 else 1
    w = max(1, C.BASELINE_WINDOW // step)
    for hf in F.half.unique():
        idx = np.flatnonzero(F.half.values == hf)
        r = raw[idx]
        oh = np.stack([r == k for k in range(3)], 1).astype(int)
        cs = np.vstack([np.zeros((1, 3), int), np.cumsum(oh, 0)])
        lo = np.clip(np.arange(len(r)) - w, 0, len(r))
        hi = np.clip(np.arange(len(r)) + w + 1, 0, len(r))
        cnt = cs[hi] - cs[lo]
        # ties broken in favour of the raw label of the frame itself
        cnt = cnt + 0.1 * oh
        out[idx] = np.argmax(cnt, 1)
    return out


# ---- (b) HMM -----------------------------------------------------------------------------------
def emissions(F):
    """Log-likelihoods (n,3). Constants from config, set a priori."""
    n = len(F)
    L = np.zeros((n, 3))
    has = (F.src.values > 0)
    bv = np.nan_to_num(F.bv.values, nan=0.05)
    slow = 1.0 / (1.0 + bv / C.BALL_SLOW_V0)            # 1 for a still ball, 0.5 at v0
    for k, col in ((H_, "dH"), (A_, "dA")):
        d = F[col].values
        c = np.exp(-np.square(np.where(np.isfinite(d), d, 99)) / (2 * C.CONTROL_SIGMA ** 2))
        L[:, k] += np.where(has, np.log1p(C.CONTROL_ALPHA * c * slow), 0.0)
    # D cues
    static_alone = has & (bv < C.STATIC_BALL_V) & (F.dNear.values > C.NOBODY_NEAR_H)
    L[:, D_] += C.W_D_STATIC * static_alone
    miss = F.miss_s.values
    L[:, D_] += C.W_D_MISSING * np.clip((miss - C.BALL_MISSING_D_S) / C.BALL_MISSING_D_S, 0, 1)
    off = F.off_pitch.values & (miss > 1.0)
    L[:, D_] += C.W_D_OUTFIELD * off
    # player speed relative to the median of its half (removes zoom/perspective bias)
    pv = F.pv.values / F.groupby("half").pv.transform("median").values
    f_slow = np.clip((C.PLAYER_FAST_REL - pv) / (C.PLAYER_FAST_REL - C.PLAYER_SLOW_REL), 0, 1)
    f_slow = np.nan_to_num(f_slow, nan=0.5)
    L[:, D_] += C.W_D_SLOW * f_slow
    L[:, [H_, A_]] += (C.W_ALIVE_FAST * (1 - f_slow))[:, None]
    # consecutive frames are strongly correlated: count ~one independent observation per
    # EMISSION_CORR_S seconds instead of one per frame
    step = max(1, np.median(np.diff(F.frame.values))) if len(F) > 1 else 1
    return L * min(1.0, step / (C.EMISSION_CORR_S * C.FPS))


def transition(dt_frames):
    pl = min(0.5, dt_frames / (C.POSS_MEAN_S * C.FPS))     # leave a possession
    sl = min(0.5, dt_frames / (C.STOP_MEAN_S * C.FPS))     # leave a stoppage
    T = np.array([[1 - pl, pl * C.P_TURNOVER, pl * (1 - C.P_TURNOVER)],
                  [pl * C.P_TURNOVER, 1 - pl, pl * (1 - C.P_TURNOVER)],
                  [sl / 2, sl / 2, 1 - sl]])
    return np.log(T)


def viterbi(L, frames):
    n = len(L)
    back = np.zeros((n, 3), np.int8)
    delta = np.log(np.full(3, 1 / 3)) + L[0]
    cache = {}
    for t in range(1, n):
        dt = int(frames[t] - frames[t - 1])
        if dt not in cache:
            cache[dt] = transition(dt)
        s = delta[:, None] + cache[dt]
        back[t] = np.argmax(s, 0)
        delta = s[back[t], np.arange(3)] + L[t]
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hmm(F):
    L = emissions(F)
    out = np.empty(len(F), int)
    for hf in F.half.unique():
        idx = np.flatnonzero(F.half.values == hf)
        out[idx] = viterbi(L[idx], F.frame.values[idx])
    return out


def expand(F, lab):
    """Processed-frame labels -> every frame of the video (nearest processed frame within half);
    frames outside the graded halves get X."""
    full = np.full(C.N_FRAMES, "X", dtype=object)
    fr = F.frame.values
    for a, b in C.HALVES:
        m = (fr >= a) & (fr <= b)
        if not m.any():
            continue
        f, l = fr[m], LAB[lab[m]]
        lo, hi = f[0], f[-1]
        allf = np.arange(lo, hi + 1)
        j = np.clip(np.searchsorted(f, allf), 1, len(f) - 1)
        left_closer = (allf - f[j - 1]) <= (f[j] - allf)
        nearest = np.where(left_closer, j - 1, j)
        nearest[allf <= f[0]] = 0
        full[lo:hi + 1] = l[nearest]
    return full


def run(start, end, log=print):
    F = features(start, end, log)
    b = baseline(F)
    h = hmm(F)
    res = pd.DataFrame({"frame": F.frame, "baseline": LAB[b], "hmm": LAB[h]})
    res.to_parquet(C.OUT / "labels_processed.parquet", index=False)
    for name, lab in (("baseline", b), ("hmm", h)):
        full = expand(F, lab)
        pd.DataFrame({"frame": np.arange(C.N_FRAMES), "label": full}).to_csv(
            C.OUT / f"possession_{name}.csv", index=False)
        log(f"possession[{name}]: H {np.mean(lab == H_):.3f} A {np.mean(lab == A_):.3f} "
            f"D {np.mean(lab == D_):.3f} on processed frames")
    return res
