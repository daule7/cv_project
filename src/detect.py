"""Step 1: YOLOv8 detection + ByteTrack tracking, cached in parquet chunks (resumable).

Each chunk file out/detections/chunk_XXXXXX.parquet covers grid frames [k*CHUNK, (k+1)*CHUNK)
intersected with the requested range and the graded halves. A sidecar .json records exactly which
frames were processed, so a later run skips complete chunks and resumes after an interruption.

Per detection we store: frame, cls, conf, box, track id (persons), colour feature of the torso in
Lab (persons), and whether the box centre lies on the grass mask (all).
"""
import json
import logging
import queue
import threading
import time
import warnings

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .pitch import near_touchline, below_line

warnings.filterwarnings("ignore")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def field_mask(frame):
    """Grass mask at 1/FIELD_MASK_SCALE resolution: largest green component with holes filled."""
    s = C.FIELD_MASK_SCALE
    small = cv2.resize(frame, (C.W // s, C.H // s), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] >= C.GREEN_H[0]) & (hsv[..., 0] <= C.GREEN_H[1]) &
         (hsv[..., 1] >= C.GREEN_S_MIN) & (hsv[..., 2] >= C.GREEN_V_MIN)).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, 8)
    if n <= 1:
        return np.zeros_like(g)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    # fill holes (players, lines, ball) = everything not reachable from the border outside the mask
    inv = (1 - m).astype(np.uint8)
    ff = inv.copy()
    h, w = ff.shape
    border = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[y, x]:
            cv2.floodFill(ff, border, (x, y), 2)
    m[(inv == 1) & (ff != 2)] = 1
    return m


def torso_lab(frame, box):
    """Median Lab colour of non-grass pixels in the torso area (rows 10-50% of the box)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    ya, yb = int(y1 + 0.10 * h), int(y1 + 0.50 * h)
    xa, ya = max(xa, 0), max(ya, 0)
    xb, yb = min(xb, C.W), min(yb, C.H)
    if xb - xa < 2 or yb - ya < 2:
        return (np.nan, np.nan, np.nan)
    crop = frame[ya:yb, xa:xb]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = ((hsv[..., 0] >= C.GREEN_H[0]) & (hsv[..., 0] <= C.GREEN_H[1]) &
             (hsv[..., 1] >= C.GREEN_S_MIN) & (hsv[..., 2] >= C.GREEN_V_MIN))
    keep = ~grass
    if keep.sum() < 8:
        return (np.nan, np.nan, np.nan)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(np.float32)
    return tuple(np.median(lab, axis=0))


def frames_to_process(start, end, stride):
    """Requested frames: within [start,end], inside graded halves, every `stride`-th from half start."""
    out = []
    for a, b in C.HALVES:
        lo, hi = max(a, start), min(b, end)
        if lo > hi:
            continue
        first = a + ((lo - a + stride - 1) // stride) * stride
        out.append(np.arange(first, hi + 1, stride))
    return np.concatenate(out) if out else np.zeros(0, int)


def chunk_path(k):
    return C.DET_DIR / f"chunk_{k:06d}.parquet"


def chunk_done(k, frames):
    meta = C.DET_DIR / f"chunk_{k:06d}.json"
    if not meta.exists() or not chunk_path(k).exists():
        return False
    done = set(json.loads(meta.read_text())["frames"])
    return set(int(f) for f in frames) <= done


class Reader(threading.Thread):
    """Decodes the needed frames in a background thread (sequential read, seek only on gaps)."""

    def __init__(self, frames, maxsize=64):
        super().__init__(daemon=True)
        self.frames = frames
        self.q = queue.Queue(maxsize)

    def run(self):
        cap = cv2.VideoCapture(str(C.VIDEO))
        pos = -1
        for f in self.frames:
            f = int(f)
            if f != pos + 1:
                if 0 < f - pos - 1 <= 8:          # small gap (stride): grab is cheaper than seek
                    for _ in range(f - pos - 1):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            pos = f
            self.q.put((f, img if ok else None))
        self.q.put(None)


def make_tracker(stride):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    cfg["track_buffer"] = max(10, int(30 / stride))
    return BYTETracker(IterableSimpleNamespace(**cfg))


def run(start, end, stride=1, log=print):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)

    C.DET_DIR.mkdir(parents=True, exist_ok=True)
    frames = frames_to_process(start, end, stride)
    if not len(frames):
        log("detect: nothing to do")
        return
    keys = frames // C.CHUNK
    todo = [k for k in np.unique(keys) if not chunk_done(k, frames[keys == k])]
    total = int(sum((keys == k).sum() for k in todo))
    log(f"detect: {len(frames)} frames requested, {total} to process in {len(todo)} chunks "
        f"(stride={stride})")
    if not todo:
        return

    device = pick_device()
    model = YOLO(str(C.MODEL))
    log(f"detect: device={device} imgsz={C.IMGSZ} half={C.HALF}")
    proc = np.concatenate([frames[keys == k] for k in todo])
    reader = Reader(proc)
    reader.start()

    tracker, last_f, session = None, None, None
    rows, frame_rows, cur_k, done_frames = [], [], None, []
    t0, n_done = time.time(), 0

    def flush():
        if cur_k is None:
            return
        df = pd.DataFrame(rows, columns=["frame", "cls", "conf", "x1", "y1", "x2", "y2", "tid",
                                         "L", "a", "b", "in_field"])
        df = df.astype({"frame": np.int32, "cls": np.int8, "conf": np.float32, "x1": np.float32,
                        "y1": np.float32, "x2": np.float32, "y2": np.float32, "tid": np.int64,
                        "L": np.float32, "a": np.float32, "b": np.float32, "in_field": bool})
        meta = C.DET_DIR / f"chunk_{cur_k:06d}.json"
        fdf = pd.DataFrame(frame_rows, columns=["frame", "tl0", "tl1"]).astype(
            {"frame": np.int32, "tl0": np.float32, "tl1": np.float32})
        fpath = C.DET_DIR / f"frames_{cur_k:06d}.parquet"
        if fpath.exists() and meta.exists():
            prevf = pd.read_parquet(fpath)
            fdf = pd.concat([prevf[~prevf.frame.isin(done_frames)], fdf]).sort_values("frame")
        fdf.to_parquet(fpath, index=False)
        old = set()
        if meta.exists() and chunk_path(cur_k).exists():
            prev = pd.read_parquet(chunk_path(cur_k))
            old = set(json.loads(meta.read_text())["frames"])
            prev = prev[~prev.frame.isin(done_frames)]
            df = pd.concat([prev, df], ignore_index=True).sort_values("frame", kind="stable")
        tmp = chunk_path(cur_k).with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(chunk_path(cur_k))
        meta.write_text(json.dumps({"frames": sorted(old | set(int(f) for f in done_frames)),
                                    "stride": stride}))

    while True:
        item = reader.q.get()
        if item is None:
            break
        f, img = item
        k = f // C.CHUNK
        if k != cur_k:
            flush()
            rows, frame_rows, cur_k, done_frames = [], [], k, []
        if img is None:
            log(f"detect: WARNING failed to read frame {f}")
            frame_rows.append((f, np.nan, np.nan))
            done_frames.append(f)
            continue
        # tracker session restarts on any discontinuity (resume, halftime, cached chunk skipped)
        if last_f is None or f - last_f != stride:
            tracker, session = make_tracker(stride), f
        last_f = f

        r = model.predict(img, imgsz=C.IMGSZ, conf=C.DET_CONF, iou=0.5, half=C.HALF,
                          device=device, verbose=False)[0]
        boxes = r.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        conf = boxes.conf
        xyxy = boxes.xyxy
        fm = field_mask(img)
        tl = near_touchline(img, fm)
        frame_rows.append((f, np.nan, np.nan) if tl is None else (f, tl[0], tl[1]))
        s = C.FIELD_MASK_SCALE

        def on_field(b):
            x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            cx = int(np.clip(x / s, 0, fm.shape[1] - 1))
            cy = int(np.clip(y / s, 0, fm.shape[0] - 1))
            return bool(fm[cy, cx]) and not bool(below_line(x, y, tl, 6.0))

        def on_field_feet(b):
            x, y = (b[0] + b[2]) / 2, b[3]
            cx = int(np.clip(x / s, 0, fm.shape[1] - 1))
            cy = int(np.clip(y / s - 1, 0, fm.shape[0] - 1))
            return bool(fm[cy, cx]) and not bool(below_line(x, y, tl, 15.0))

        # persons -> tracker
        pmask = (cls != C.CLS_BALL) & (conf >= C.PERSON_MIN_CONF)
        tracks = tracker.update(boxes[pmask], img) if pmask.any() else np.zeros((0, 8))
        pidx = np.flatnonzero(pmask)
        tid_of = {}
        for t in tracks:
            tid_of[int(pidx[int(t[7])])] = session * 100000 + int(t[4])
        for i in pidx:
            if conf[i] < C.PERSON_MIN_CONF:
                continue
            Lab = torso_lab(img, xyxy[i]) if cls[i] != C.CLS_REF else (np.nan,) * 3
            rows.append((f, cls[i], conf[i], *xyxy[i], tid_of.get(int(i), -1), *Lab,
                         on_field_feet(xyxy[i])))
        # ball candidates
        bidx = np.flatnonzero(cls == C.CLS_BALL)
        bidx = bidx[np.argsort(-conf[bidx])][:C.BALL_TOPK]
        for i in bidx:
            rows.append((f, cls[i], conf[i], *xyxy[i], -1, np.nan, np.nan, np.nan,
                         on_field(xyxy[i])))
        done_frames.append(f)

        n_done += 1
        if n_done % 500 == 0 or n_done == total:
            el = time.time() - t0
            fps = n_done / el
            log(f"detect: {n_done}/{total} frames ({100 * n_done / total:.1f}%), "
                f"{fps:.1f} fps, frame {f}, ETA {(total - n_done) / fps / 60:.1f} min")
    flush()
    log(f"detect: finished {n_done} frames in {(time.time() - t0) / 60:.1f} min")


def load(start=0, end=C.N_FRAMES - 1):
    """Load cached detections for frames in [start, end] (only chunks that exist)."""
    dfs = []
    for k in range(start // C.CHUNK, end // C.CHUNK + 1):
        p = chunk_path(k)
        if p.exists():
            d = pd.read_parquet(p)
            dfs.append(d[(d.frame >= start) & (d.frame <= end)])
    if not dfs:
        raise FileNotFoundError("no detections cached for this range; run --stage detect first")
    return pd.concat(dfs, ignore_index=True)


def load_frames(start=0, end=C.N_FRAMES - 1):
    dfs = []
    for k in range(start // C.CHUNK, end // C.CHUNK + 1):
        p = C.DET_DIR / f"frames_{k:06d}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            dfs.append(d[(d.frame >= start) & (d.frame <= end)])
    return pd.concat(dfs, ignore_index=True)


def processed_frames(start=0, end=C.N_FRAMES - 1):
    out = []
    for k in range(start // C.CHUNK, end // C.CHUNK + 1):
        meta = C.DET_DIR / f"chunk_{k:06d}.json"
        if meta.exists():
            out.extend(f for f in json.loads(meta.read_text())["frames"] if start <= f <= end)
    return np.array(sorted(out), dtype=int)
