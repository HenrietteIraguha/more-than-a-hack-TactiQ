# TactiQ — More Than a Hack 2027
 
Our submission for the possession-detection challenge: one label per frame for all 199,319
frames of FC Barcelona vs Athletic Club, LaLiga 2024/25, Matchday 2 — `H` (Barcelona),
`A` (Athletic Club), `D` (dead ball) or `X` (outside the graded blocks).
 
**Technical document:** [`TactiQ.pdf`](TactiQ.pdf)
 
## Reproducing our submission
 
```bash
python -m venv venv
source venv/Scripts/activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```
 
Unpack the match data into `data/` as described in [`data/README.md`](data/README.md). The data
is not redistributable and is not included in this repository.
 
```bash
python starter/our_solution.py      # writes evaluation/results/TactiQ.csv
```
 
That single command regenerates the exact file we submitted. It takes a couple of minutes,
almost all of it reading the tracking file.
 
## Checking the output
 
```bash
python evaluation/validate_submission.py evaluation/results/TactiQ.csv
python starter/check_output.py --file evaluation/results/TactiQ.csv
```
 
`starter/check_output.py` is our own whole-match check, and it uses no ground truth. It asks two
questions: does the output look like real football in its distribution of stoppages and
possession lengths, and does it agree with the Smart Tagging set-piece labels — which
`starter/our_solution.py` deliberately never reads, so the check cannot agree with itself.
 
It works on any labels file, including the baselines shipped with the challenge:
 
```bash
python starter/check_output.py --file evaluation/results/baseline_smooth125.csv
```
 
## How it works, briefly
 
The ball is missing from the tracking on 43% of frames, and missing most during stoppages. The
players are tracked continuously. So the structure of the answer is built from player movement,
and the ball is used only where it holds up.
 
1. Drop tracks that never move — objects beside the goal, not players
2. Interpolate ball gaps under half a second; leave longer gaps empty
3. Dead ball from the players: 75% below walking pace, too few tracked, or ball over a line
4. Restarting team from the Smart Tagging phase tags, else the player nearest the ball
5. Possession inside each stretch of play: a Viterbi decode with a penalty per change of hands
6. Remove stoppages the ball was clearly played through — a guarded final pass
Full reasoning, what we tried and dropped, and the limitations are in [`TactiQ.pdf`](TactiQ.pdf).
 
## What is ours in this repository
 
| File | |
|---|---|
| `starter/our_solution.py` | the solution |
| `starter/check_output.py` | our independent check |
| `TactiQ.pdf` | technical document |
| `README.md` | this section |
 
Everything else is the organisers' original repository, unchanged. Parameters live in one block
at the top of `starter/our_solution.py`, and the newer rules can each be switched off from the
command line, so any claim in the technical document can be tested by disabling that rule and
re-running.
 
No manually created labels for this match were used at any stage, and no parameter is tied to a
specific frame of it.
 
---
 
# ↓ Original challenge repository README
 
Everything below this line is the organisers' documentation, unchanged.
 
---
# Unpack the match data here

This folder is empty on purpose. The match assets are gated — see [DATA.md](../DATA.md) for how to
enrol, accept the terms, and get the download link.

Unpack the bundle here, keeping the names it arrives with:

```
data/
├─ tactical/                                  the evaluation reference
│  ├─ BAR-ATH_tactical.mp4                    row i of a submission is frame i of this
│  ├─ BAR-ATH_tactical_atd.xml                tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt                tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv         one 3x3 matrix per frame
│  └─ BAR-ATH_tactical_smart_tagging.xml      638 tactical phases and set pieces
└─ broadcast/                                 optional, NOT synchronised to tactical
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

One folder per recording, because the question that matters most about any of these files is which
timeline it is on. `games.py` derives every path, so nothing else in the repository spells one out:

```python
from games import asset_paths
asset_paths()                      # tactical, the default
asset_paths(source="broadcast")
```

If a filename ever changes, `games.py` is the only edit.

Prefer to keep the data somewhere else? Point `MTAH_DATA_DIR` at it and nothing else changes:

```bash
export MTAH_DATA_DIR=/mnt/big-disk/mtah2027       # macOS / Linux
$env:MTAH_DATA_DIR = "D:\mtah2027"                # Windows PowerShell
```

## Nothing in here is committed

`.gitignore` keeps everything except this README out of git. That is not only tidiness — the terms
you accepted cover redistribution, and a public fork with the match video in it is redistribution.
Your own submission repository has to be public too, so check the same thing there.

## You do not need any of this to run the scorer

The evaluation reads CSVs only. `python evaluation/run_evaluation.py` works on a fresh clone with
this folder empty, and scores the four baselines that ship with the repository.
