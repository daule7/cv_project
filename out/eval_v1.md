# Evaluation vs correct.csv

Scored frames: 30000 (frames 11185..125807 that are graded; processed frames only).
Scorer: official `evaluation/possession_metrics.py` from github.com/metrica-sports/more-than-a-hack-2027 (copied to src/vendor/).

| method | accuracy | macro-F1 | spellF1 (mean IoU .3-.7) | spellF1@.5 | F1 H | F1 A | F1 D | alive/dead F1 | team acc (both alive) | home share pred / GT | spells pred / GT | mean spell s pred / GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.609 | 0.583 | 0.170 | 0.146 | 0.686 | 0.546 | 0.515 | 0.864 | 0.736 | 0.610 / 0.674 | 149 / 56 | 6.9 / 15.0 |
| hmm | 0.594 | 0.584 | 0.120 | 0.097 | 0.622 | 0.516 | 0.614 | 0.806 | 0.725 | 0.603 / 0.674 | 150 / 56 | 5.0 / 15.0 |

## Confusion matrix — baseline (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 10270 | 3541 | 381 | 0 |
| A | 1800 | 4625 | 441 | 0 |
| D | 3660 | 1898 | 3384 | 0 |

spellF1 per IoU: 0.3: 0.273, 0.4: 0.234, 0.5: 0.146, 0.6: 0.117, 0.7: 0.078

## Confusion matrix — hmm (rows = GT, cols = prediction)

| GT \ pred | H | A | D | X |
|---|---|---|---|---|
| H | 7951 | 2716 | 3525 | 0 |
| A | 1709 | 3702 | 1455 | 0 |
| D | 1702 | 1075 | 6165 | 0 |

spellF1 per IoU: 0.3: 0.262, 0.4: 0.165, 0.5: 0.097, 0.6: 0.039, 0.7: 0.039

## Ball visibility on scored frames

- raw detector, ball conf>=0.25 anywhere in frame: 0.777
- after filtering (accepted detections): 0.729 (GT alive frames 0.801, GT dead 0.558)
- with position incl. interpolation of gaps <= 0.5 s: 0.833
