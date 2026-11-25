"""Figures for the PDF report (out/report/fig_*.png). Everything is computed from out/ files."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .vendor import possession_metrics as pm

OUT = C.OUT / "report"
COL = {"H": "#2a78d6", "A": "#eb6834", "D": "#b4b2ab", "X": "#e8e6e1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e0da"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def load():
    gt = pd.read_csv(C.GT_CSV).label.values.astype(str)
    hmm = pd.read_csv(C.OUT / "possession_hmm.csv").label.values.astype(str)
    base = pd.read_csv(C.OUT / "possession_baseline.csv").label.values.astype(str)
    return gt, hmm, base


def shares(lab, a, b):
    s = lab[a:b + 1]
    s = s[s != "X"]
    return {k: float((s == k).mean()) for k in "HAD"}


def fig_shares(gt, hmm, base):
    rows = []
    for name, (a, b) in (("1-й тайм", C.HALVES[0]), ("2-й тайм", C.HALVES[1]),
                         ("Матч", (C.HALVES[0][0], C.HALVES[1][1]))):
        for src, lab in (("эталон", gt), ("HMM", hmm), ("baseline", base)):
            rows.append((name, src, shares(lab, a, b)))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = 0
    yt, yl = [], []
    for i, (name, src, s) in enumerate(rows):
        left = 0
        for k in "HAD":
            ax.barh(y, s[k] * 100, left=left, color=COL[k], height=0.72,
                    edgecolor="white", linewidth=1.5)
            if s[k] > 0.06:
                ax.text(left + s[k] * 50, y, f"{s[k] * 100:.0f}%", ha="center", va="center",
                        color="white" if k != "D" else INK, fontsize=8.5)
            left += s[k] * 100
        yt.append(y)
        yl.append(f"{name} · {src}")
        y += 1 if i % 3 != 2 else 1.6
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("доля оцениваемых кадров, %")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in "HAD"]
    ax.legend(handles, ["H хозяева", "A гости", "D мяч вне игры"], ncol=3, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / "fig_shares.png")
    plt.close(fig)


def fig_confusion(gt, hmm, base):
    g = pm.label_codes(gt)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, (name, lab) in zip(axes, (("baseline", base), ("HMM", hmm))):
        cm = pm.confusion(g, pm.label_codes(lab))[:3, :3].astype(float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n{int(cm[i, j]):,}".replace(",", " "),
                        ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["H", "A", "D"])
        ax.set_yticks(range(3), ["H", "A", "D"])
        ax.set_xlabel("предсказание")
        ax.set_ylabel("эталон")
        ax.set_title(name)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Матрица ошибок по кадрам (доля строки и число кадров), весь матч",
                 fontsize=10, color=INK2, y=1.02)
    fig.savefig(OUT / "fig_confusion.png")
    plt.close(fig)


def fig_timeline(gt, hmm, base, a, b, fname, title):
    fig, ax = plt.subplots(figsize=(8, 1.9))
    rows = (("эталон", gt), ("HMM", hmm), ("baseline", base))
    cmap = {"H": 0, "A": 1, "D": 2, "X": 3}
    from matplotlib.colors import ListedColormap
    cm = ListedColormap([COL["H"], COL["A"], COL["D"], COL["X"]])
    img = np.array([[cmap[x] for x in lab[a:b + 1]] for _, lab in rows])
    t = (np.arange(a, b + 2) - a) / C.FPS / 60
    ax.pcolormesh(t, np.arange(4), img, cmap=cm, vmin=0, vmax=3, shading="flat")
    for yy in (1, 2):
        ax.axhline(yy, color="white", lw=2)
    ax.set_yticks([0.5, 1.5, 2.5], [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"минуты от кадра {a}")
    ax.set_title(title, loc="left")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in "HAD"]
    ax.legend(handles, ["H", "A", "D"], ncol=3, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_spells(gt, hmm, base):
    runs = pm.valid_runs(pm.label_codes(gt))
    fig, ax = plt.subplots(figsize=(8, 3))
    bins = np.r_[0, 2, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90, 300]
    width = 0.27
    x = np.arange(len(bins) - 1)
    for i, (name, lab, col) in enumerate((("эталон", gt, INK2), ("HMM", hmm, COL["H"]),
                                          ("baseline", base, COL["A"]))):
        sp = pm.possession_spells(pm.label_codes(lab), runs)
        d = np.array([s[1] - s[0] for s in sp]) / C.FPS
        h, _ = np.histogram(d, bins)
        ax.bar(x + (i - 1) * width, h, width=width - 0.03, color=col,
               label=f"{name}: {len(d)} владений, медиана {np.median(d):.1f} с")
    ax.set_xticks(x, [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)], fontsize=8)
    ax.set_xlabel("длительность владения, с")
    ax.set_ylabel("число владений")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(OUT / "fig_spells.png")
    plt.close(fig)


def fig_features(gt, a, b):
    F = pd.read_parquet(C.OUT / "features.parquet")
    F = F[(F.frame >= a) & (F.frame <= b)]
    med = pd.read_parquet(C.OUT / "features.parquet").groupby("half").pv.median()
    t = (F.frame.values - a) / C.FPS / 60
    g = gt[F.frame.values]
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.2), sharex=True)
    for ax in axes:
        dead = g == "D"
        edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color=COL["D"], alpha=0.35, lw=0)
    rel = F.pv.values / med.loc[F.half.values].values
    axes[0].plot(t, rel, color=INK, lw=1)
    axes[0].axhline(C.PLAYER_SLOW_REL, color=INK2, lw=0.8, ls="--")
    axes[0].axhline(C.PLAYER_FAST_REL, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("скорость игроков\n/ медиана тайма")
    axes[0].set_title("Признаки на фрагменте 1-го тайма; серая заливка = D по эталону", loc="left")
    dH = np.clip(F.dH.values, 0, 8)
    dA = np.clip(F.dA.values, 0, 8)
    axes[1].plot(t, dH, color=COL["H"], lw=0.9, label="до ближайшего H")
    axes[1].plot(t, dA, color=COL["A"], lw=0.9, label="до ближайшего A")
    axes[1].set_ylabel("мяч → игрок,\nвысот рамки")
    axes[1].set_ylim(0, 8)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[2].plot(t, F.src.values, color=INK, lw=0.8, drawstyle="steps-post")
    axes[2].set_yticks([0, 1, 2], ["нет", "детекция", "интерп."])
    axes[2].set_ylabel("мяч")
    axes[2].set_xlabel(f"минуты от кадра {a}")
    fig.align_ylabels(axes)
    fig.savefig(OUT / "fig_features.png")
    plt.close(fig)


def fig_versions():
    data = [("v1 baseline\n(проверка)", 0.609, 0.583, 0.170),
            ("v1 HMM\n(проверка)", 0.594, 0.584, 0.120),
            ("v2 baseline\n(проверка)", 0.670, 0.629, 0.282),
            ("v2 HMM\n(проверка)", 0.729, 0.705, 0.365),
            ("baseline\n(весь матч)", 0.651, 0.645, 0.256),
            ("HMM\n(весь матч)", 0.712, 0.701, 0.242)]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(data))
    w = 0.27
    for i, (lbl, col) in enumerate((("accuracy", "#9aa0a6"), ("macro-F1", COL["H"]),
                                    ("spellF1", COL["A"]))):
        v = [d[i + 1] for d in data]
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=col, label=lbl)
        for xx, vv in zip(x, v):
            ax.text(xx + (i - 1) * w, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [d[0] for d in data], fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
    fig.savefig(OUT / "fig_versions.png")
    plt.close(fig)


def fig_detector():
    data = [("полный кадр 1280", 58.1, 0.853, 0.617, 0.147),
            ("полный кадр 1920", 53.4, 0.903, 0.617, 0.153),
            ("тайлы 2×2", 19.2, 0.890, 0.613, 0.197),
            ("тайлы 3×2", 13.8, 0.917, 0.707, 0.330)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(data))
    h = 0.26
    for i, (lbl, col) in enumerate((("conf ≥ 0.1", "#9fc4ef"), ("conf ≥ 0.25", COL["H"]),
                                    ("conf ≥ 0.5", "#174a8a"))):
        v = [d[i + 2] for d in data]
        ax.barh(y + (i - 1) * h, v, h - 0.03, color=col, label=lbl)
    ax.set_yticks(y, [d[0] for d in data])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("доля кадров, где найден мяч")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax = axes[1]
    ax.barh(y, [d[1] for d in data], color="#9aa0a6", height=0.6)
    for yy, d in zip(y, data):
        ax.text(d[1] + 1, yy, f"{d[1]:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [""] * len(data))
    ax.invert_yaxis()
    ax.set_xlabel("кадров/с (fp16, mps)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_detector.png")
    plt.close(fig)


def stats(gt, hmm, base):
    """Numbers quoted in the report text that are not in eval.md."""
    s = {}
    ball = pd.read_parquet(C.OUT / "ball.parquet")
    for i, (a, b) in enumerate(C.HALVES):
        m = (ball.frame >= a) & (ball.frame <= b)
        s[f"ball_det_h{i + 1}"] = float((ball.src[m] == 1).mean())
    det = []
    for name, lab in (("hmm", hmm), ("baseline", base)):
        r = pm.evaluate(pm.label_codes(gt), pm.label_codes(lab))
        s[name] = {k: r[k] for k in ("accuracy", "macro_f1", "spell_f1_mean", "kappa",
                                     "alive_dead_f1", "team_acc_both_alive",
                                     "pred_home_share", "gt_home_share", "fragmentation",
                                     "n_pred_segments", "n_gt_segments")}
        s[name]["trans12"] = {k: r["transitions"][12][k] for k in ("precision", "recall", "f1")}
        s[name]["trans25"] = {k: r["transitions"][25][k] for k in ("precision", "recall", "f1")}
        s[name]["spell_per_iou"] = {str(k): v["f1"] for k, v in r["spell_f1"].items()}
        s[name]["per_class"] = r["per_class"]
    P = pd.read_parquet(C.OUT / "persons.parquet", columns=["frame", "cls"])
    per = P.groupby("frame").size()
    s["persons_per_frame"] = float(per.mean())
    s["n_det_rows"] = int(len(P))
    (OUT / "stats.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    return s


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    gt, hmm, base = load()
    fig_shares(gt, hmm, base)
    fig_confusion(gt, hmm, base)
    fig_timeline(gt, hmm, base, 11185, 11185 + 10 * 60 * 25, "fig_timeline1.png",
                 "Первые 10 минут 1-го тайма")
    fig_timeline(gt, hmm, base, 150000, 150000 + 10 * 60 * 25, "fig_timeline2.png",
                 "10 минут 2-го тайма (кадры 150000–165000)")
    fig_spells(gt, hmm, base)
    fig_features(gt, 11185 + 3 * 60 * 25, 11185 + 9 * 60 * 25)
    fig_versions()
    fig_detector()
    return stats(gt, hmm, base)


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False)[:3000])
