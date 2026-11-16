# Technical document — Possession HMM

More than a Hack 2027, Phase 1: ball possession for every frame of BAR-ATH.

Team: **TactiQ** · Members: **<names>**

The open sample data is by Metrica Sports (https://github.com/metrica-sports/sample-data).

## 1. Approach

The ranking metric scores whole possessions, so the main risk is **flicker**: per-frame decisions
that split one possession into many. A per-frame classifier is therefore the wrong shape. We use two
temporal models, each decoded with Viterbi, so a state change has to be paid for with sustained
evidence.

### Stage 1: ball alive or dead

The question is whether the game is being played at all. The clearest signal is collective pace:
in a stoppage everyone walks, while in live play somebody is always running.

Features, per frame and smoothed over about 1 s:

- log mean speed of the 3 fastest tracked players ("pace of the game")
- log mean speed of all tracked players
- log ball speed (missing where the ball is not seen; a missing feature contributes nothing)

A **two-state diagonal-Gaussian HMM** is fitted with Baum-Welch on the whole match, at 5 fps, with
no labels. The state whose players move faster is "alive". The fitted means are 4.4 vs 1.7 m/s for
the top-3 speed and 2.4 vs 1.0 m/s for the mean speed. Decoding runs at 25 fps with a prior mean
stretch length of 15 s. Emissions are tempered by 0.2 because the smoothed features are strongly
autocorrelated.

Hard evidence and boundary snapping are layered on top of the HMM:

- **A ball seen more than 0.5 m outside a line is dead**, for as long as it is seen there.
- **Alive→dead** is moved to the first out-of-play sighting within −2 s / +1 s, if there is one.
  That frame is when the ball crossed the line.
- **Dead→alive** is moved to the restart within ±2 s. A restart is either a struck ball (in bounds,
  below 2 m/s for 0.4 s, then above 5 m/s) or a throw-in (back in play after being seen out).
- **A half begins with its kickoff.** Everything before the Smart Tagging `Kick Off (Start)` of a
  half is dead, and the kick is then snapped. This fixes the 2nd half, which is graded about 2 min
  before it kicks off.
- Dead stretches under 8 s with no out-of-play sighting are discarded as slow live phases (slow
  build-up play, typically). A foul or offside stoppage almost always lasts longer, because the
  ball has to be placed. The 8 s was chosen on other games (section 3).
  Live stretches under 1 s between two stoppages are also discarded.

### Stage 2: which team, inside each live stretch

Inside each live stretch, a **two-state (H/A) Viterbi** runs over per-frame control evidence. The
ball position is linearly interpolated across dropouts of up to 0.5 s.

- **Fake dots are ignored.** A track that stays inside a 1 m box for a whole 25-second block is an
  object, not a player: the tracking has such dots, e.g. one standing still beside a goal. A team
  member found this while inspecting the data. It changes only 22 frames on BAR-ATH, and the sample
  games have no such tracks, so its effect could not be measured there.
- `d_H` and `d_A` are the distances from the ball to the nearest tracked player of each team, in
  metres. Coordinates are scaled to 105.3 × 68 m so distances are isotropic.
- The per-frame log-likelihood ratio is `0.35 · w · clip(d_A − d_H, −1, 1)`. The proximity weight
  `w` is 1 when the nearest player is within 1 m and falls linearly to 0 at 3 m. A ball in flight or
  loose gives no evidence.
- A change of possession costs a log-penalty of 3, which is about 0.35 s of clear control by the
  other team (several frames of their player on the ball). A brushed touch or a ball passing a
  player cannot pay it. This is the challenge's definition of *controlled* possession, written as
  a cost. The value was chosen on other games (section 3).
- Where the ball gives no evidence (not seen, or nobody near it), the current possession continues.
  That matches the rule "a contested or loose ball stays with the last team that had it".
- The **Smart Tagging** gives a weak prior of 0.0075 per frame, used only on frames with no ball
  evidence. It keeps the same ratio to the switch cost as the original a-priori values (0.02 / 8),
  because the sample games used for tuning have no Smart Tagging. Phase codes are mapped to the team in possession: `BUILD UP`, `PROGRESSION`,
  `BALL IN …`, `CREATING CHANCES`, `LONG BALL`, `PLAYERS IN THE BOX` and `ATTACKING TRANSITION` go to
  the tagged team; `DEFENDING IN …` and `DEFENSIVE TRANSITION` go to its opponent. Overlapping tags
  are reduced to their **net sign**. Without that, five correlated tags on one attack outvoted the
  ball itself and delayed transitions by several seconds.
- A `SET PIECES` tag with a team near a restart sets the start state of the live stretch it opens
  (log-odds ±3). Throw-ins are tagged `N/A`, so for them the geometry decides.

The output is `H`/`A`/`D` on graded frames and `X` outside the two halves. We never abstain on
graded frames. Under spellF1 an `X` only repeats the previous label, which the Viterbi already does
where it has no evidence.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | **used**: player positions and speeds, ball position, team of each track slot |
| tactical Smart Tagging | **used**: weak team prior, set-piece team at restarts, kickoff of each half |
| tactical homography | not used. The ATD is already in pitch coordinates and no video is decoded |
| tactical video | not used, beyond watching it to understand the data |
| broadcast video / ATD / homography | not used. They are not synchronised to the tactical timeline, and the tactical ATD covers every frame |
| ball visibility as a dead-ball feature | **tried and dropped**. The tracker loses the ball in crowded live play as often as in stoppages, so the HMM learned "invisible = dead" |
| stacked Smart Tagging votes | **tried and dropped** in favour of one net vote per frame (see above) |

## 3. Training and tuning data

**No labels of the provided game are used**, in training, tuning or validation.

**Unsupervised on BAR-ATH.** The HMM's Gaussian means and variances are estimated by Baum-Welch on
the provided tracking, without labels.

**Two parameters were chosen on other games.** These are `SWITCH_COST` and `MIN_DEAD_S`.

| | |
|---|---|
| dataset | Metrica Sports open sample data, Sample Games 1 and 2: https://github.com/metrica-sports/sample-data |
| what it is | two anonymised professional matches, not BAR-ATH: tracking at 25 fps plus event data (1,745 and 1,935 events) |
| who labelled it | the event data was produced by Metrica Sports and published openly; we added no labels of our own |
| how labels were made | `solution/external/metrica_sample.py` converts the events mechanically to the challenge's `H`/`A`/`D` per frame. Dead from a stoppage trigger to the next restart: BALL OUT, FAULT RECEIVED, SHOT…GOAL/OUT, BALL LOST by OFFSIDE/HAND BALL/END HALF, then SET PIECE or GOAL KICK. Otherwise the team of the latest on-ball event (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST); a CHALLENGE never moves possession. This yields 368 and 343 possessions and about 40% dead ball |
| how it was used | `solution/external/tune_on_sample.py` runs the unchanged pipeline on both games (no Smart Tagging exists for them) and scores it with the competition's own `evaluation/possession_metrics.py`. It grid-searches `SWITCH_COST` × `MIN_DEAD_S`; the full table is in `solution/external/tuning_results.csv` |

The sample tracking loses the ball on about 40% of frames, close to BAR-ATH's 41%. It has no speed
channel, so speeds are computed from positions.

Result, as the mean over the two games:

| setting | spellF1 | macroF1 | possessions (truth 368 / 343) |
|---|---|---|---|
| a-priori values (switch 8, min dead 4 s) | 0.453 | 0.783 | 230 / 253 |
| **chosen (switch 3, min dead 8 s)** | **0.525** | 0.791 | 294 / 280 |

The optimum is broad: switch costs of 2.5–3.5 and minimum stoppages of 6–10 s all land within
about 0.01. The improvement holds on both games (+0.087 and +0.057).

**The public 5-minute ground truth was used only as the rules allow**: to check the pipeline
(frame alignment, file format, what the labels mean) and to diagnose structural failures. Those
diagnoses led to dropping ball visibility, using one net tag vote and the kickoff rule, each
justified on football grounds above. No parameter was chosen on it. After tuning on the sample
games, the public window moved from spellF1 0.336 to 0.447 and from 8 to 17 possessions (truth:
17). That is an independent confirmation, not the selection criterion.

### An earlier version, compared on the same games

The team's first solution (`starter/our_solution.py`, kept in this repository for reference)
used a walking-share rule for dead balls and a Viterbi over "which team is nearer, within 4 m".
On the two sample games it scored spellF1 0.415. Its dead-ball detection was as good as the HMM's
(F1 0.815 vs 0.810). Its weakness was the possession evidence: counting the *nearer* team, rather
than a player *in control*, let a defender running close to the ball push possession. Changing the
evidence to the in-control form above raised it to 0.470. Snapping stoppage edges to the line
crossing and the restart kick, and the dead-ball HMM, account for the rest of the gap to 0.525.

## 4. Possession rules implemented

- **Ball in play**: the unsupervised pace HMM, plus out-of-play sightings, restart kicks and
  kickoffs. A dead ball is always `D`, and its "owner" is ignored, as the challenge asks.
- **Change of possession**: requires sustained control by the other team, about 1 s of their
  player being nearest and close to the ball. The new possession starts at the first frame of that
  control.
- **Loose balls, challenges and aerial duels**: give no or contradictory evidence, so possession
  stays with the last team.
- **Restarts**: the possession after a stoppage starts at the restart kick, with the team taken
  from the set-piece tag where one exists, otherwise from who touches the ball first.

## 5. Limitations and what we would do next

- **Domain gap in tuning.** The two parameters were tuned on two other games whose tracking was
  produced differently from the BAR-ATH ATD (fewer identity switches, no native speed channel).
  The full match now yields 314 possessions (mean 11.0 s); short interceptions that are lost
  immediately are still partly merged. More labelled games would make the choice firmer.
- **Tracking noise in team assignment.** In places the nearest tracked player is on the "wrong"
  team for seconds at a time, which looks like a team or identity error in the automatic tracking.
  Track-level smoothing of team membership (majority vote per tracklet, plus shirt colour from the
  video) would address it.
- **The ball is unseen on 41% of graded frames.** We only interpolate dropouts of up to 0.5 s.
  Ball-free proxies could fill longer gaps: which team is pressing, team shape and direction of
  movement, or a learned model trained on tracking from other games.
- **Foul stoppages with the ball in play** depend only on the pace HMM, so slow live phases and
  quick restarts are where stage 1 errs.
- **Ball height is unknown**, so a high ball passing over a player looks like a touch. The switch
  cost absorbs most of these.

## 6. Reproduction

```bash
pip install -r solution/requirements.txt
```

To repeat the tuning on the open sample games (optional, about 10 minutes):

```bash
python solution/external/fetch_metrica_sample.py
```

```bash
python solution/external/tune_on_sample.py
```

To produce the submission:

```bash
python solution/pipeline.py --name <team_name>
```

```bash
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Running the pipeline is deterministic and takes about 15 s once the ATD cache exists (30 s on the
first run). The match data is not in the repository. `data/` and `solution/.cache/` are gitignored.
