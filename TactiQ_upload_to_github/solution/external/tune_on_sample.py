"""Choose the pipeline's decoding parameters on Metrica's open sample games (other matches).

    python solution/external/fetch_metrica_sample.py
    python solution/external/tune_on_sample.py

Runs the unchanged pipeline on Sample Games 1 and 2 (no Smart Tagging exists for them), scores it
with the competition's own metric code (evaluation/possession_metrics.py) against possession
labels derived from those games' human event data, and grid-searches the parameters that govern
segmentation. Nothing here reads BAR-ATH labels of any kind.

Writes solution/external/tuning_results.csv and prints the best setting by mean spellF1 over the
games. Copy the winner into the constants at the top of solution/pipeline.py.
"""

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)
ROOT = os.path.dirname(SOLUTION)
sys.path.insert(0, HERE)
sys.path.insert(0, SOLUTION)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import metrica_sample as M  # noqa: E402
import pipeline as P  # noqa: E402
import possession_metrics as pm  # noqa: E402

# DEAD_MEAN_SEGMENT_S was also searched over 8 / 15 / 25 s and moved spellF1 by < 0.002, so it is
# held at its prior here.
GRID = {
    "SWITCH_COST": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
    "MIN_DEAD_S": [2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
    "DEAD_MEAN_SEGMENT_S": [15.0],
}
OUT = os.path.join(HERE, "tuning_results.csv")


def score(gt, pred):
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    spells = pm.spell_scores(g, p, runs)
    return {"spellF1": spells["mean_f1"],
            "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(g, p))),
            "n_pred": spells["n_pred"], "n_gt": spells["n_gt"]}


def main():
    games = []
    for g in M.GAMES:
        trk, blocks, events = M.load_game(g)
        gt = M.labels_from_events(events, blocks, trk["n_frames"])
        games.append((g, trk, blocks, gt, P.prepare(trk, blocks)))
        print("game {} prepared".format(g))

    defaults = {k: getattr(P, k) for k in GRID}
    rows = []
    t0 = time.time()
    for values in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, values))
        for k, v in setting.items():
            setattr(P, k, v)
        row = dict(setting)
        for g, trk, blocks, gt, prepared in games:
            s = score(gt, P.predict_tracking(trk, P.NO_TAGS, blocks, prepared=prepared))
            for k, v in s.items():
                row["g{}_{}".format(g, k)] = v
        row["spellF1"] = np.mean([row["g{}_spellF1".format(g)] for g in M.GAMES])
        row["macroF1"] = np.mean([row["g{}_macroF1".format(g)] for g in M.GAMES])
        rows.append(row)
        print("{}  spellF1 {:.4f}  macroF1 {:.4f}  ({:.0f}s)".format(
            setting, row["spellF1"], row["macroF1"], time.time() - t0), flush=True)
    for k, v in defaults.items():
        setattr(P, k, v)

    table = pd.DataFrame(rows).sort_values(["spellF1", "macroF1"], ascending=False)
    table.to_csv(OUT, index=False)
    print("\nbest settings by mean spellF1 over the sample games:")
    print(table.head(10).to_string(index=False))
    print("\ncurrent defaults {}:".format(defaults))
    cur = table
    for k, v in defaults.items():
        cur = cur[np.isclose(cur[k], v)]
    print(cur.to_string(index=False) if len(cur) else "  (not on the grid)")


if __name__ == "__main__":
    main()
