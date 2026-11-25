"""Step 5b: visualisation video for any frame range (default: first 2.5 min of the 1st half)."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import config as C

FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
COL = {0: (235, 140, 30), 1: (60, 60, 230), "ref": (40, 220, 240), -1: (200, 200, 200)}  # BGR
LCOL = {"H": (235, 140, 30), "A": (60, 60, 230), "D": (150, 150, 150), "X": (60, 60, 60)}
NAME = {"H": "H — хозяева", "A": "A — гости", "D": "D — мяч вне игры", "X": "—"}


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _blend_rect(img, x0, y0, x1, y1, color=(20, 20, 20), alpha=0.55):
    sub = img[y0:y1, x0:x1]
    over = np.full_like(sub, color)
    cv2.addWeighted(over, alpha, sub, 1 - alpha, 0, dst=sub)


def _text(img, xy, s, font, fill=(255, 255, 255)):
    """Draw unicode text with PIL onto a BGR image region (only the text bbox is converted)."""
    l, t, r, b = font.getbbox(s)
    x, y = xy
    w, h = r + 4, b + 4
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    roi = img[y:y1, x:x1]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((0, 0), s, font=font, fill=fill[::-1])
    img[y:y1, x:x1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render(start, end, out_path=None, log=print):
    out_path = out_path or C.OUT / "demo.mp4"
    lab = pd.read_csv(C.OUT / "possession.csv").label.values.astype(str)
    P = pd.read_parquet(C.OUT / "persons.parquet")
    P = P[(P.frame >= start) & (P.frame <= end)]
    ball = pd.read_parquet(C.OUT / "ball.parquet").set_index("frame")
    groups = P.groupby("frame").indices
    Pv = P[["x1", "y1", "x2", "y2", "tid", "team", "cls"]].values

    f_big = _font(FONT_B, 34)
    f_mid = _font(FONT, 22)
    f_small = _font(FONT, 14)

    W, Hh = C.W, C.H
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), C.FPS, (W, Hh))
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), C.FPS, (W, Hh))
    cap = cv2.VideoCapture(str(C.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # timeline strip (precomputed)
    tl_x0, tl_x1, tl_y0, tl_h = 40, W - 40, Hh - 58, 18
    n = end - start + 1
    strip = np.zeros((tl_h, tl_x1 - tl_x0, 3), np.uint8)
    for x in range(tl_x1 - tl_x0):
        f = start + int(x / (tl_x1 - tl_x0) * n)
        strip[:, x] = LCOL[lab[f]]
    cum = {"H": 0, "A": 0, "D": 0}

    for f in range(start, end + 1):
        ok, img = cap.read()
        if not ok:
            break
        cur = lab[f]
        if cur in cum:
            cum[cur] += 1
        # persons
        for x1, y1, x2, y2, tid, team, cls in Pv[groups.get(f, [])]:
            col = COL["ref"] if cls == C.CLS_REF else COL[int(team)]
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, col, 1, cv2.LINE_AA)
            if tid >= 0:
                cv2.putText(img, str(int(tid) % 100000), (int(x1), int(y1) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        # ball
        if f in ball.index:
            b = ball.loc[f]
            if b.src > 0:
                c = (int(b.bx), int(b.by))
                colb = (0, 255, 255) if b.src == 1 else (0, 180, 255)
                cv2.circle(img, c, 11, colb, 2, cv2.LINE_AA)
                cv2.circle(img, c, 2, colb, -1, cv2.LINE_AA)
        # top plaque
        _blend_rect(img, W // 2 - 260, 12, W // 2 + 260, 72, (20, 20, 20), 0.6)
        cv2.rectangle(img, (W // 2 - 260, 12), (W // 2 - 248, 72), LCOL[cur], -1)
        _text(img, (W // 2 - 232, 20), f"владеет: {NAME[cur]}", f_big)
        # bottom: timeline + percentages
        _blend_rect(img, 20, Hh - 112, W - 20, Hh - 18, (20, 20, 20), 0.6)
        img[tl_y0:tl_y0 + tl_h, tl_x0:tl_x1] = (
            0.85 * strip + 0.15 * img[tl_y0:tl_y0 + tl_h, tl_x0:tl_x1]).astype(np.uint8)
        cx = tl_x0 + int((f - start) / n * (tl_x1 - tl_x0))
        cv2.line(img, (cx, tl_y0 - 6), (cx, tl_y0 + tl_h + 6), (255, 255, 255), 2, cv2.LINE_AA)
        tot = sum(cum.values()) or 1
        ha = (cum["H"] + cum["A"]) or 1
        s = (f"с начала фрагмента:  H {cum['H'] / tot:.0%}  ·  A {cum['A'] / tot:.0%}  ·  "
             f"D {cum['D'] / tot:.0%}      владение  H {cum['H'] / ha:.0%} : {cum['A'] / ha:.0%} A")
        _text(img, (tl_x0, Hh - 104), s, f_mid)
        mm, ss = divmod((f - C.HALVES[0 if f <= C.HALVES[0][1] else 1][0]) / C.FPS, 60)
        _text(img, (W - 300, Hh - 104), f"кадр {f}   {int(mm):02d}:{int(ss):02d} тайма", f_mid)
        # legend
        lx, ly = 130, 66
        _blend_rect(img, lx - 10, ly, lx + 190, ly + 92, (20, 20, 20), 0.5)
        for i, (k, t) in enumerate(((0, "H хозяева"), (1, "A гости"), ("ref", "судьи"))):
            cv2.rectangle(img, (lx, ly + 10 + 26 * i), (lx + 14, ly + 24 + 26 * i), COL[k], -1)
            _text(img, (lx + 24, ly + 7 + 26 * i), t, f_mid)
        vw.write(img)
        if (f - start) % 500 == 0:
            log(f"demo: frame {f} ({f - start}/{n})")
    vw.release()
    log(f"demo: wrote {out_path}")
