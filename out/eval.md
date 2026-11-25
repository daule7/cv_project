# Evaluation vs correct.csv

Scored frames: 158857 (frames 11185..190197 that are graded; processed frames only).
Scorer: official `evaluation/possession_metrics.py` from github.com/metrica-sports/more-than-a-hack-2027 (copied to src/vendor/).

| method | accuracy | macro-F1 | spellF1 (mean IoU .3-.7) | spellF1@.5 | F1 H | F1 A | F1 D | alive/dead F1 | team acc (both alive) | home share pred / GT | spells pred / GT | mean spell s pred / GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.651 | 0.645 | 0.256 | 0.250 | 0.710 | 0.564 | 0.660 | 0.798 | 0.810 | 0.543 / 0.594 | 661 / 323 | 6.9 / 10.7 |
| hmm | 0.712 | 0.701 | 0.242 | 0.238 | 0.734 | 0.620 | 0.750 | 0.846 | 0.813 | 0.598 / 0.594 | 197 / 323 | 22.5 / 10.7 |

## Confusion matrix — baseline (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 39985 | 8613 | 2503 | 0 |
| A | 6547 | 24472 | 3956 | 0 |
| D | 15073 | 18684 | 39024 | 0 |

spellF1 per IoU: 0.3: 0.388, 0.4: 0.323, 0.5: 0.250, 0.6: 0.183, 0.7: 0.136

## Confusion matrix — hmm (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 43013 | 7152 | 936 | 0 |
| A | 8444 | 24630 | 1901 | 0 |
| D | 14662 | 12717 | 45402 | 0 |

spellF1 per IoU: 0.3: 0.408, 0.4: 0.312, 0.5: 0.238, 0.6: 0.150, 0.7: 0.100

## Ball visibility on scored frames

- raw detector, ball conf>=0.25 anywhere in frame: 0.662
- after filtering (accepted detections): 0.619 (GT alive frames 0.768, GT dead 0.443)
- with position incl. interpolation of gaps <= 0.5 s: 0.704
