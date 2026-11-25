"""Paths and constants shared by all pipeline steps.

All thresholds here were set a priori from common sense / football facts, NOT tuned on correct.csv.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "video.mp4"
MODEL = ROOT / "football_yolov8.pt"
GT_CSV = ROOT / "correct.csv"
OUT = ROOT / "out"
DET_DIR = OUT / "detections"

FPS = 25.0
N_FRAMES = 199319
W, H = 1920, 1080
# Graded frames (inclusive ranges), 0-based.
HALVES = [(11185, 90651), (110808, 190197)]

# ---- detector -------------------------------------------------------------------------------
# Full frame at native resolution (imgsz=1920, no downscaling). Chosen from the 300-frame study:
# same ball-found rate at conf>=0.25 as 2x2 tiling but ~3x faster; 3x2 tiling +9pp but ~4x slower.
IMGSZ = 1920
HALF = True
DET_CONF = 0.05            # keep low-confidence candidates; ball step filters them temporally
PERSON_MIN_CONF = 0.10     # persons below this are dropped entirely
BALL_TOPK = 5              # keep up to K ball candidates per frame
CHUNK = 1000               # frames per cache file (aligned grid)
CLS_BALL, CLS_GK, CLS_PLAYER, CLS_REF = 0, 1, 2, 3

# ---- grass / field mask ---------------------------------------------------------------------
# OpenCV HSV: H in [0,180]. Grass green ~ 35..85.
GREEN_H = (30, 90)
GREEN_S_MIN = 50
GREEN_V_MIN = 40
FIELD_MASK_SCALE = 4       # mask computed at 1/4 resolution

# ---- teams ----------------------------------------------------------------------------------
TEAM_SAMPLE = 60000        # detections sampled for KMeans
TEAM_MIN_HEIGHT = 25       # px, smaller boxes give unreliable colour
TEAM_MIN_CONF = 0.5

# ---- ball -----------------------------------------------------------------------------------
BALL_MIN_CONF = 0.10       # candidates below this are ignored
BALL_START_CONF = 0.30     # a new ball track segment must start from a confident detection
# Max plausible ball speed: ~35 m/s shot; a player (~1.8 m) is ~45 px tall at mid-field, so
# 35 m/s ~ 0.8 player heights per frame ~ 35-60 px/frame. Use 80 px/frame + camera motion margin.
BALL_MAX_STEP_PX = 80.0
BALL_INTERP_MAX = int(0.5 * FPS)   # fill gaps up to 0.5 s
BALL_MIN_SEGMENT = 3       # isolated detection runs shorter than this (frames) are dropped

# ---- possession -----------------------------------------------------------------------------
BASELINE_WINDOW = 50       # +-50 frames majority vote
CONTROL_SIGMA = 1.0        # distance (in player heights) scale of the control kernel
BALL_SLOW_V0 = 0.15        # ball speed (player heights / frame) at which control evidence halves
CONTROL_ALPHA = 4.0        # strength of team evidence
POSS_MEAN_S = 12.0         # mean possession length
STOP_MEAN_S = 25.0         # mean stoppage length
P_TURNOVER = 0.6           # share of possession ends that are turnovers (rest -> D)
BALL_MISSING_D_S = 3.0     # ball unseen longer than this -> evidence for D
STATIC_BALL_V = 0.02       # heights/frame: ball considered static
NOBODY_NEAR_H = 3.0        # nobody within 3 heights of the ball
# v1 used absolute thresholds 0.5 / 1.0 heights/s; on unlabeled data the match median turned out
# to be 0.59 h/s (p90 0.90), so "fast" was almost never reached. v2: relative to the half median.
PLAYER_SLOW_REL = 0.75     # below 75% of the half's median speed: play looks stopped
PLAYER_FAST_REL = 1.15     # above 115%: play looks alive
EMISSION_CORR_S = 0.2      # one independent observation per 0.2 s (v1 effectively used 0.04 s)
SLOW_WINDOW_S = 2.0        # smoothing window for player speed
# log-likelihood weights of D cues
W_D_OUTFIELD = 2.0
W_D_STATIC = 1.0
W_D_MISSING = 1.0
W_D_SLOW = 1.0
W_ALIVE_FAST = 0.5

DEMO_RANGE = (11185, 11185 + int(150 * FPS))   # first 2.5 minutes of the 1st half


def graded_mask():
    import numpy as np
    m = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        m[a:b + 1] = True
    return m


def half_of(frame):
    for i, (a, b) in enumerate(HALVES):
        if a <= frame <= b:
            return i
    return -1
