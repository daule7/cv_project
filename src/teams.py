"""Step 2: team assignment.

* Field players: torso colour in Lab (computed in step 1 from the upper half of the box with grass
  pixels removed) -> KMeans(k=2) on a sample from the whole processed range -> per-track majority
  vote. Darker cluster (lower L*) = H (dark blue/garnet), lighter = A.
* Referees: excluded by class.
* Goalkeepers: not clustered by colour. In every half we find which image side each team defends
  (the team whose outfield players sit further left on average defends the left goal), then a
  goalkeeper track that stays left of the outfield players belongs to the team defending the left.
Output: out/persons.parquet (all person detections + team column: 0=H, 1=A, -1=none/referee).
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config as C
from . import detect

TEAM_H, TEAM_A, NONE = 0, 1, -1


def run(start, end, log=print):
    det = detect.load(start, end)
    P = det[det.cls != C.CLS_BALL].copy()
    P["h"] = P.y2 - P.y1
    P["fx"] = (P.x1 + P.x2) / 2
    P["half"] = np.where(P.frame <= C.HALVES[0][1], 0, 1)
    P["team"] = NONE

    # ---- class vote per track: a track that is often called referee/goalkeeper is one -------
    tracked = P[P.tid >= 0]
    frac = pd.crosstab(tracked.tid, tracked.cls, normalize="index")
    for c in (C.CLS_REF, C.CLS_GK):
        if c not in frac:
            frac[c] = 0.0
    is_ref = frac.index[frac[C.CLS_REF] >= 0.3]
    is_gk = frac.index[(frac[C.CLS_GK] >= 0.5) & (frac[C.CLS_REF] < 0.3)]
    P["cls_raw"] = P.cls
    P.loc[P.tid.isin(is_ref), "cls"] = C.CLS_REF
    P.loc[P.tid.isin(is_gk), "cls"] = C.CLS_GK
    log(f"teams: class vote: {len(is_ref)} tracks -> referee, {len(is_gk)} -> goalkeeper; "
        f"{int((P.cls != P.cls_raw).sum())} detections relabelled")

    # ---- KMeans on field players -----------------------------------------------------------
    pl = P[(P.cls == C.CLS_PLAYER)]
    ok = pl[(pl.conf >= C.TEAM_MIN_CONF) & (pl.h >= C.TEAM_MIN_HEIGHT) & pl.in_field &
            pl.L.notna()]
    rng = np.random.default_rng(0)
    samp = ok.iloc[rng.choice(len(ok), min(C.TEAM_SAMPLE, len(ok)), replace=False)]
    X = samp[["L", "a", "b"]].values.astype(np.float64)
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    dark = int(np.argmin(km.cluster_centers_[:, 0]))
    to_team = {dark: TEAM_H, 1 - dark: TEAM_A}
    log(f"teams: KMeans on {len(X)} player crops; centres Lab = "
        f"{np.round(km.cluster_centers_, 1).tolist()}; dark cluster {dark} -> H")

    valid = pl.L.notna()
    det_team = np.full(len(pl), NONE)
    det_team[valid.values] = [to_team[c] for c in km.predict(pl.loc[valid, ["L", "a", "b"]].values.astype(np.float64))]
    pl = pl.assign(dteam=det_team)
    # distance margin -> confidence (for reporting)
    d = km.transform(X)
    margin = np.abs(d[:, 0] - d[:, 1]) / (d.sum(1) + 1e-9)
    log(f"teams: sample cluster sizes H={int((km.labels_ == dark).sum())} "
        f"A={int((km.labels_ != dark).sum())}; median relative margin {np.median(margin):.2f}")

    # per-track vote
    tr = pl[(pl.tid >= 0) & (pl.dteam >= 0)].groupby("tid").dteam.agg(["mean", "size"])
    tr["team"] = np.where(tr["mean"] < 0.5, TEAM_H, TEAM_A)
    purity = np.maximum(tr["mean"], 1 - tr["mean"])
    log(f"teams: {len(tr)} player tracks, frame-weighted vote purity "
        f"{np.average(purity, weights=tr['size']):.3f}")
    # Sliding-window vote inside each track (+-0.5 s) instead of one vote per whole track:
    # ByteTrack sometimes passes an id between players of different teams; a whole-track vote
    # would then paint one of them with the wrong colour for the entire track.
    team_of = tr["team"].to_dict()
    t = pl[pl.tid >= 0].sort_values(["tid", "frame"])
    v = t.dteam.where(t.dteam >= 0).astype(float)
    roll = v.groupby(t.tid.values).transform(
        lambda x: x.rolling(int(C.FPS) + 1, center=True, min_periods=1).mean())
    roll = roll.fillna(t.tid.map(tr["mean"]))
    pl_team = pl.dteam.copy()
    pl_team.loc[t.index] = np.where(roll.isna(), -1, np.where(roll < 0.5, TEAM_H, TEAM_A))
    P.loc[pl.index, "team"] = pl_team.values
    changed = (pl_team.loc[t.index].values != t.tid.map(team_of).values).mean()
    log(f"teams: windowed vote differs from whole-track vote on {changed:.3f} of tracked detections")

    # ---- goalkeepers -----------------------------------------------------------------------
    field = P[(P.cls == C.CLS_PLAYER) & P.in_field & (P.team >= 0)]
    gk = P[P.cls == C.CLS_GK]
    sides = {}
    for hf in (0, 1):
        f = field[field.half == hf]
        if not len(f):
            continue
        m = f.groupby(["frame", "team"]).fx.mean().unstack()
        m = m.dropna()
        diff = (m[TEAM_H] - m[TEAM_A]).mean()      # <0: H sits left -> H defends left goal
        left_team = TEAM_H if diff < 0 else TEAM_A
        sides[hf] = left_team
        log(f"teams: half {hf + 1}: mean x(H)-x(A) = {diff:.1f}px over {len(m)} frames -> "
            f"{'H' if left_team == TEAM_H else 'A'} defends the left goal "
            f"(frames with H left: {(m[TEAM_H] < m[TEAM_A]).mean():.2f})")
    med = field.groupby("frame").fx.median()
    g = gk.assign(med=gk.frame.map(med)).dropna(subset=["med"])
    g = g.assign(left=(g.fx < g.med).astype(int))
    gkey = np.where(g.tid >= 0, g.tid, -(g.index.values + 1))   # untracked: per detection
    vote = g.assign(key=gkey).groupby("key").agg(left=("left", "mean"), half=("half", "first"))
    gteam = {}
    for key, row in vote.iterrows():
        lt = sides.get(int(row.half), TEAM_H)
        gteam[key] = lt if row.left >= 0.5 else 1 - lt
    P.loc[g.index, "team"] = [gteam[k] for k in gkey]
    log(f"teams: goalkeepers: {len(vote)} tracks/detections assigned")

    out = P[["frame", "cls", "conf", "x1", "y1", "x2", "y2", "tid", "in_field", "team"]]
    out.to_parquet(C.OUT / "persons.parquet", index=False)
    (C.OUT / "teams.json").write_text(json.dumps({
        "centers_lab": km.cluster_centers_.tolist(), "dark_cluster": dark,
        "left_team_by_half": {str(k): int(v) for k, v in sides.items()}}, indent=1))
    n = P[P.cls != C.CLS_REF]
    log(f"teams: assigned {int((n.team == TEAM_H).sum())} H / {int((n.team == TEAM_A).sum())} A "
        f"person detections; unassigned {int((n.team < 0).sum())}")
    return out
