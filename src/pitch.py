"""Near touchline detection: the long, near-horizontal white line at the bottom of the grass.

Everything below it (bench, technical area, ball boys) is off the pitch. Returns (y_at_x0, y_at_xW)
in full-resolution pixels or None when the line is not visible.
"""
import cv2
import numpy as np

from . import config as C


def near_touchline(frame, grass_small):
    s = 2
    h, w = C.H // s, C.W // s
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] >= 135) & (hsv[..., 1] <= 85)).astype(np.uint8)
    grass = cv2.resize(grass_small, (w, h), interpolation=cv2.INTER_NEAREST)
    white &= grass
    white[: int(0.45 * h)] = 0
    lines = cv2.HoughLinesP(white * 255, 1, np.pi / 360, threshold=80,
                            minLineLength=int(0.20 * w), maxLineGap=25)
    if lines is None:
        return None
    best = None
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        k = (y2 - y1) / (x2 - x1)
        if abs(k) > 0.25:
            continue
        b = y1 - k * x1
        cands = [(x1, y1, x2, y2)]
        # merge collinear segments to measure total horizontal extent
        for X1, Y1, X2, Y2 in lines[:, 0]:
            if abs((Y1 - (k * X1 + b))) < 4 and abs((Y2 - (k * X2 + b))) < 4:
                cands.append((X1, Y1, X2, Y2))
        xs = [c[0] for c in cands] + [c[2] for c in cands]
        extent = max(xs) - min(xs)
        if extent < 0.4 * w:
            continue
        ymid = k * w / 2 + b
        if best is None or ymid > best[0]:
            best = (ymid, k, b)
    if best is None:
        return None
    _, k, b = best
    return float(b * s), float((k * w + b) * s)


def below_line(x, y, tl, margin=6.0):
    """True where point (x,y) lies below the touchline (off pitch, near side)."""
    if tl is None or np.isnan(tl[0]):
        return np.zeros_like(np.asarray(x, float), bool)
    y0, y1 = tl
    yl = y0 + (y1 - y0) * np.asarray(x, float) / C.W
    return np.asarray(y, float) > yl + margin
