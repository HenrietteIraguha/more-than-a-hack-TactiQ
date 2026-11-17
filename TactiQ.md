# TactiQ — Technical document

More Than a Hack 2027 · FC Barcelona vs Athletic Club

## 01 Project objective

Label each of 199,319 frames as Barcelona, Athletic Club, or stopped, from 46 player tracks and one ball track at 25 fps, without using video in the prediction pipeline. Because spellF1 scores complete possession spells rather than frames, avoiding false turnovers and false stoppages was central to the design: a stoppage invented inside a real possession destroys that possession and creates two more.

## 02 What we found

- **The ball is unusable for restart detection on this dataset.** We tried three independent methods and measured all three; every one failed for the same cause. Ball coordinates are missing on 43% of frames, and missing *most* during stoppages — precisely when restarts happen.
- **Stoppages are half the answer, and the hardest half.** The organisers' baseline puts the ball in play 77.7% of the match, where football runs at roughly 55%. We brought that to 51.5% and raised set-piece coverage from 39 of 58 to 57 of 58, but we still over-detect stoppages and can quantify by how much.
- **Building from player movement rather than ball position is what separates this from the baseline.** The players are tracked continuously; the ball is not. Everything below follows from that.

## 03 Approach

**The ball is unreliable; the players are not.** Ball coordinates are missing in **43% of frames**, and most missing during stoppages. The players are tracked continuously. So the structure of the answer is built from player movement, and the ball is used only where it holds up. Each problem below came from studying the tactical video and the published labels.

**Problem: the nearest player often isn't the one in possession.** At an interception seven seconds in, the Barcelona player who won the ball was never the closest track. A pressing defender is as close as the carrier. *What we did:* a team must be clearly nearer **and stay nearer** for about 0.5 s before possession changes.

**Problem: the ball vanishes.** *What we did:* gaps under 0.5 s are interpolated along the ball's path; longer gaps produce no evidence and possession stays where it was. Above 10 m/s the ball is in flight — faster than any player can run — so those frames give no evidence either.

**Problem: the program didn't know play had stopped.** The ball is no guide: at an offside it sat still on the pitch, and while being handed back it moved although play was dead. *What we did:* use the players. Within a second of a whistle nearly all 22 slow to a walk, so 75% of tracked players below walking pace (2 m/s), averaged over one second, means stopped. Two more signals: fewer than eight tracked players means play cannot be underway, and a ball fully over a line for one second is dead by the laws of the game.

**Problem: one track wasn't a player.** A point stood motionless beside Barcelona's goal, identical before kickoff and during play. *What we did:* exclude any track confined to a one-metre area for 25 seconds.

**Problem: possession flickered.** A third of possessions lasted under three seconds. *What we did:* stop deciding frame by frame. Take a whole stretch of play, score every possible sequence of H and A across it, charge a fixed penalty for each change of hands, and keep the sequence that scores best (a Viterbi decode). Possession then changes only where the evidence outweighs the penalty: a wobble does not pay for itself, a real interception does. Median possession is now 9.2 s.

**Problem: calm passing looks like a whistle.** When a team keeps the ball at the back with nobody closing them down, every player is strolling — which is exactly the pattern the stoppage rule is looking for. So the program marked play as stopped in the middle of a possession that was still running. That is the most costly error available to us: an invented stoppage does not just mislabel those frames, it cuts one real possession into two and loses the real one.

*What we did:* a correction step that runs at the very end, after the labels are finished. It can only turn a stoppage back into possession, never the other way round, so nothing the decoder already decided can be disturbed. A stoppage is only removed when all three of these hold: the same team has the ball immediately before and after it, the ball never crossed a line during it, and the ball was visible and moving for more than half of it. Nine stoppages were removed this way, and set-piece coverage did not drop.

## 04 Possession rules

- **Loose ball** — ball missing, nobody near, or in flight: no evidence produced, possession unchanged.
- **Challenges** — proximity alone never decides a duel; the other team must be clearly nearer *and stay nearer* for about 0.5 s.
- **Near misses** — at 0:16 a Barcelona player almost intercepted a long pass and fell instead; the reference labels stay with Athletic, and so do we.
- **Restarts** — the laws place opponents 9.15 m back, 2 m at a throw-in, so the player over the ball is the taker. See 05 for why this was rarely applicable.
- **Ball out of play** — dead the moment it fully crosses a line. A matter of the laws, not inference, so no other rule may override it.

## 05 Assets used, and dropped

| Asset | Use |
| --- | --- |
| Automatic tracking data | Primary input |
| Smart Tagging — 575 phase tags | Restarting team, at 114 of 144 restarts |
| Smart Tagging — 59 set-piece tags | **Never read by the predictor.** Reserved for independent checks |
| Tactical video | To understand the label definitions |
| Published five-minute labels | Understanding the task, never to tune or select versions |
| Broadcast video, homography | Not used |
| Scorer and scoreboard (`evaluation/run_evaluation.py`) | Run early to learn how spellF1 scores whole possession spells rather than frames, and what the shipped baselines achieve. That shaped the design and led us to write our own check. |

**Why the set-piece tags are excluded.** Our checking tool marks our output against them. If the predictor read them too, it would be assessed against information it had copied — agreement every time, meaning nothing. For the same reason we never tuned towards the tagging: it is automatic and carries its own errors, so agreement above its own accuracy would mean reproducing them.

**Three restart methods tried and dropped.** All three read the restart from the ball, and all three failed for the same reason: the ball is missing on 43% of frames, and missing most during stoppages — which is exactly when restarts happen. We measured each one rather than abandoning it on a hunch:

- A stationary ball starting to move: found the taker in **49 of 143** restarts.
- The ball resting on a marked location: found one in **25 of 144**, type correct **17 of 48**, and **6** false locations.
- Ball contacts, from a sudden change of direction or speed: only **292 in the whole match**, where a professional match has well over a thousand.

We use the Smart Tagging phase tags instead, accounting for the four defensive categories where the tag names the team *without* the ball.

**One stoppage fix rejected.** Letting ball movement cancel any stoppage removed 15, but deleted a genuine throw-in. Rebuilt with the guards in 03. **Two ideas declined** from a teammate's variant: forcing a fixed block of frames to stopped (a constant tied to specific frames, which the rules forbid), and merging all possessions under two seconds (the match has genuine one-second possessions).

## 06 Results, and how we judged changes

| Measure | Organisers' baseline | TactiQ |
| --- | --- | --- |
| Ball in play | 77.7% | **51.5%** |
| Detected stoppages | 79 | **136** |
| Possession spells | 398 | **235** |
| Median possession | 8.1 s | **9.2 s** |
| Possessions under 1 second | 18% | **1%** |
| Possessions under 3 seconds | 34% | **19%** |
| Set pieces preceded by stoppage | 39 of 58 | **57 of 58** |
| Throw-ins preceded by stoppage | 13 of 26 | **26 of 26** |

**How this comparison was made.** We ran our own checking script, `starter/check_output.py`, over two files of labels: the organisers' strongest shipped baseline, `baseline_smooth125.csv`, and ours. The script opens only the labels it is given and the Smart Tagging — it never reads the ground truth — so neither column carries any information from the answer key, and anyone can reproduce both with one command each. Read that way, the baseline keeps the ball in play for 77.7% of the match, far outside anything real football produces, and misses half the throw-ins. That is what made stoppage detection the first thing we attacked.

These figures are descriptive, not an accuracy score.

Since the published five minutes cannot be used to choose between versions, we judged every change on an independent whole-match check instead: does the output resemble real football in its distribution of stoppages and possession lengths, and does it agree with the set-piece tags the predictor never sees. Acceptance criteria were agreed **before** each change was implemented. Three proposed changes failed those criteria and were discarded.

## 07 Limitations and next steps

**Uncertain restarts.** Some remain a guess, and our only independent measure covers 18 of 58 set pieces, since the rest carry no team label — including every throw-in. On that sample we cannot separate a good restart rule from a bad one, which is why several experiments were inconclusive rather than clearly wrong.

**Goal celebrations** are read as live play: the scoring team sprints while the conceding team walks, so the walking proportion stays below threshold.

**One penalty for all situations** — a thirty-second build-up and a two-second clearance cost the same, which likely smooths away genuine quick turnovers.

**Next steps.** Jointly decode the three states — Barcelona, Athletic, stopped — instead of deciding stoppages first and possession second, so an error in the first stage can be repaired by the second. Then: detect celebrations from **per-team** walking rates, since a goal is the one moment the teams move in opposite ways; and identify restarts from player positioning alone, available in every frame, since every ball-based method failed.

## 08 Reproduce the results

**Install:** `pip install -r requirements.txt`

**Run:** `python starter/our_solution.py`

**Check:** `python starter/check_output.py`

**Output:** `evaluation/results/TactiQ.csv`

The run command regenerates the submitted file exactly from the provided data bundle. All parameters sit in one block at the top of `starter/our_solution.py` and each newer rule can be disabled from the command line, so any claim here can be verified by switching that rule off and re-running. No manually created labels for this match were used at any stage, and no parameter is tied to a specific frame of it.