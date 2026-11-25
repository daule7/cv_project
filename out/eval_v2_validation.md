# Evaluation vs correct.csv

Scored frames: 30000 (frames 11185..125807 that are graded; processed frames only).
Scorer: official `evaluation/possession_metrics.py` from github.com/metrica-sports/more-than-a-hack-2027 (copied to src/vendor/).

| method | accuracy | macro-F1 | spellF1 (mean IoU .3-.7) | spellF1@.5 | F1 H | F1 A | F1 D | alive/dead F1 | team acc (both alive) | home share pred / GT | spells pred / GT | mean spell s pred / GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.670 | 0.629 | 0.282 | 0.273 | 0.763 | 0.619 | 0.507 | 0.862 | 0.830 | 0.668 / 0.674 | 127 / 56 | 8.1 / 15.0 |
| hmm | 0.729 | 0.705 | 0.365 | 0.392 | 0.789 | 0.635 | 0.689 | 0.902 | 0.822 | 0.699 / 0.674 | 46 / 56 | 21.3 / 15.0 |

## Confusion matrix — baseline (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 11996 | 1816 | 380 | 0 |
| A | 1620 | 4775 | 471 | 0 |
| D | 3641 | 1978 | 3323 | 0 |

spellF1 per IoU: 0.3: 0.404, 0.4: 0.372, 0.5: 0.273, 0.6: 0.197, 0.7: 0.164

## Confusion matrix — hmm (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 12357 | 1657 | 178 | 0 |
| A | 1998 | 4527 | 341 | 0 |
| D | 2766 | 1203 | 4973 | 0 |

spellF1 per IoU: 0.3: 0.510, 0.4: 0.431, 0.5: 0.392, 0.6: 0.275, 0.7: 0.216

## Ball visibility on scored frames

- raw detector, ball conf>=0.25 anywhere in frame: 0.777
- after filtering (accepted detections): 0.729 (GT alive frames 0.801, GT dead 0.558)
- with position incl. interpolation of gaps <= 0.5 s: 0.833
