"""Watch the model work: an animated pitch for any stretch of the match, as a local HTML page.

    python solution/visualize.py --name TactiQ --at 12:30
    python solution/visualize.py --name TactiQ --at 52:00 --seconds 180
    python solution/visualize.py --name TactiQ --at 50:00 --half 1     # 1st-half added time
    python solution/visualize.py --name TactiQ --all                   # whole match, 10-min pages

--at is the match clock (0:00 = first-half kickoff, 45:00 = second-half kickoff); --frame takes a
video frame instead. The page shows the tracked players (H blue, A red), the ball, the model's
label for every frame, a timeline of the model's labels over the whole match (click it to jump to
another minute: re-run the script with that --at), and -- inside the public 5-minute window -- the
public ground truth underneath for comparison.

The page is written to viz/ and opens in your browser. It embeds frame-level tracking, so it is
gitignored and must never be published or shared: the match data is not redistributable.
"""

import argparse
import json
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from data_io import FPS, PITCH_LEN, PITCH_WID, load_tracking  # noqa: E402
from games import half_video_blocks  # noqa: E402

KICKOFF_1, KICKOFF_2 = 11185, 113742   # video frames of the two kickoffs (DATA.md, FAQ.md)
OUT_DIR = os.path.join(ROOT, "viz")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def clock_to_frame(text, half=None):
    """Match clock -> video frame. The 1st half runs past 45:00 (about 53 minutes of video), so
    a clock between 45:00 and 53:00 is ambiguous: pass half=1 or half=2. Default: <45:00 is the
    1st half, otherwise the 2nd."""
    m, s = text.split(":")
    secs = int(m) * 60 + int(s)
    if half is None:
        half = 1 if secs < 45 * 60 else 2
    if half == 1:
        return KICKOFF_1 + secs * FPS
    return KICKOFF_2 + (secs - 45 * 60) * FPS


def half_of(f):
    return 1 if f < KICKOFF_2 - 2 * 60 * FPS else 2


def frame_to_clock(f):
    secs = (f - KICKOFF_1) / FPS if half_of(f) == 1 else 45 * 60 + (f - KICKOFF_2) / FPS
    secs = max(secs, 0)
    return "{}:{:02d}".format(int(secs // 60), int(secs % 60))


def whole_match_timeline(labels):
    """The model's labels over the whole match, one cell per second of each half."""
    timeline = []
    for half, first, last in half_video_blocks("BAR-ATH"):
        cells = []
        for s in range(first, last + 1, FPS):
            vals, counts = np.unique(labels[s:s + FPS], return_counts=True)
            cells.append(str(vals[np.argmax(counts)]))
        timeline.append({"half": half, "first": first, "cells": "".join(cells)})
    return timeline


def page_path(name, start):
    return os.path.join(OUT_DIR, "{}_{}H_{}.html".format(
        name, half_of(start), frame_to_clock(start).replace(":", "m")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession_hmm", help="submission to show")
    parser.add_argument("--at", default="0:00", help="match clock to start at, e.g. 12:30")
    parser.add_argument("--half", type=int, choices=(1, 2),
                        help="which half --at is in (needed for 45:00-53:00, when both exist)")
    parser.add_argument("--frame", type=int, help="video frame to start at (overrides --at)")
    parser.add_argument("--seconds", type=int, default=120, help="how long a stretch to show")
    parser.add_argument("--all", action="store_true",
                        help="the whole match as 10-minute pages plus an index page")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    sub_path = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    if not os.path.isfile(sub_path):
        sys.exit("no submission at {} -- run: python solution/pipeline.py --name {}".format(
            sub_path, args.name))
    labels = pd.read_csv(sub_path)["label"].astype(str).to_numpy()
    gt = pd.read_csv(GT_PUBLIC)["label"].astype(str).to_numpy()
    trk = load_tracking()
    timeline = whole_match_timeline(labels)
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    if args.all:
        out = build_all(trk, labels, gt, timeline, args.name)
    else:
        start = (args.frame if args.frame is not None
                 else clock_to_frame(args.at, args.half))
        stop = min(start + args.seconds * FPS, len(labels))
        out = build_page(trk, labels, gt, timeline, args.name, start, stop)
    if not args.no_open:
        webbrowser.open("file:///" + out.replace("\\", "/"))


def build_all(trk, labels, gt, timeline, name, chunk_s=600):
    """10-minute pages covering both halves, and an index page linking them."""
    entries = []
    for half, first, last in half_video_blocks("BAR-ATH"):
        # The 2nd half is graded from ~3 min before its kickoff (all dead ball); start at kickoff.
        first = max(first, KICKOFF_1 if half == 1 else KICKOFF_2)
        for start in range(first, last + 1, chunk_s * FPS):
            stop = min(start + chunk_s * FPS, last + 1)
            path = build_page(trk, labels, gt, timeline, name, start, stop)
            seg = labels[start:stop]
            alive = np.isin(seg, ["H", "A"])
            entries.append({
                "half": half, "file": os.path.basename(path),
                "from": frame_to_clock(start), "to": frame_to_clock(stop),
                "H": float(np.mean(seg == "H")), "A": float(np.mean(seg == "A")),
                "D": float(np.mean(seg == "D")),
                "hshare": float(np.sum(seg == "H") / max(alive.sum(), 1)),
                "gt": bool(np.any(gt[start:stop] != "X")),
            })
    out = os.path.join(OUT_DIR, "{}_index.html".format(name))
    g = np.isin(labels, ["H", "A", "D"])
    alive = np.isin(labels, ["H", "A"])
    summary = {"name": name, "pages": entries,
               "hshare": float(np.sum(labels == "H") / alive.sum()),
               "dead": float(np.sum(labels == "D") / g.sum()),
               "poss": count_possessions(labels)}
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(INDEX.replace("__DATA__", json.dumps(summary, separators=(",", ":"))))
    print("wrote", os.path.relpath(out, ROOT), "-- open this one")
    return out


def build_page(trk, labels, gt, timeline, name, start, stop):
    team = trk["team"].tolist()

    def r(a):
        return [None if np.isnan(v) else round(float(v), 1) for v in a]

    frames = []
    for f in range(start, stop):
        players = [[i, x, y] for i, (x, y) in enumerate(zip(r(trk["px"][f]), r(trk["py"][f])))
                   if x is not None]
        bx, by = r([trk["bx"][f], trk["by"][f]])
        frames.append({"p": [[t[1], t[2], team[t[0]]] for t in players],
                       "b": None if bx is None else [bx, by],
                       "l": labels[f], "g": gt[f] if gt[f] != "X" else None})

    payload = {"frames": frames, "start": start, "fps": FPS, "pitch": [PITCH_LEN, PITCH_WID],
               "timeline": timeline, "name": name, "window": [start, stop],
               "kick": [KICKOFF_1, KICKOFF_2], "half": half_of(start),
               "index": "{}_index.html".format(name)}
    out = page_path(name, start)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    print("wrote", os.path.relpath(out, ROOT), "-- half {}, {} to {}".format(
        half_of(start), frame_to_clock(start), frame_to_clock(stop)))
    return out


def count_possessions(labels):
    n = 0
    for _, first, last in half_video_blocks("BAR-ATH"):
        seg = labels[first:last + 1]
        alive = np.isin(seg, ["H", "A"])
        starts = alive & np.concatenate(([True], seg[1:] != seg[:-1]))
        n += int(starts.sum())
    return n


INDEX = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whole match viewer</title>
<style>
:root{--bg:#0f1418;--panel:#182027;--ink:#e8edf1;--mute:#8a98a5;--H:#3b82f6;--A:#ef4444;--D:#9ca3af}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:20px 0 8px}
.sub{color:var(--mute);margin:0 0 12px}.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.tile{background:var(--panel);border-radius:8px;padding:10px 14px;min-width:160px}.tile b{display:block;font-size:22px}
a.row{display:grid;grid-template-columns:120px 1fr 150px;gap:12px;align-items:center;background:var(--panel);
border-radius:8px;padding:10px 14px;margin:6px 0;color:var(--ink);text-decoration:none}a.row:hover{outline:1px solid var(--mute)}
.bar{display:flex;height:14px;border-radius:4px;overflow:hidden}.bar div{height:100%}
.tag{font-size:12px;color:#111;background:#fde047;border-radius:4px;padding:1px 6px;margin-left:6px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}
@media (max-width:600px){a.row{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Whole match: who had the ball</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<p class="sub">Each row is 10 minutes of video. Click one to watch it. The bar shows how that stretch splits between
<span class="legend"><span style="background:var(--H)"></span>Barcelona<span style="background:var(--A)"></span>Athletic<span style="background:var(--D)"></span>stopped</span></p>
<div id="list"></div>
</main><script>
const D=__DATA__;
document.getElementById('sub').textContent='The program\'s answers for submission "'+D.name+'", both halves. Only the first 5 minutes have official answers to compare with.';
const pct=x=>Math.round(100*x)+'%';
const t=document.getElementById('tiles');
for(const [k,v] of [['Barcelona had the ball',pct(D.hshare)+' of play'],['Athletic had the ball',pct(1-D.hshare)+' of play'],
  ['Game stopped',pct(D.dead)+' of the time'],['Possessions',D.poss]]){
  const d=document.createElement('div');d.className='tile';d.innerHTML='<b></b><span></span>';d.querySelector('b').textContent=v;d.querySelector('span').textContent=k;t.appendChild(d);}
const list=document.getElementById('list');let half=0;
for(const p of D.pages){
  if(p.half!==half){half=p.half;const h=document.createElement('h2');h.textContent=half===1?'1st half':'2nd half';list.appendChild(h);}
  const a=document.createElement('a');a.className='row';a.href=p.file;
  const when=document.createElement('div');when.textContent=p.from+' – '+p.to;
  if(p.gt){const g=document.createElement('span');g.className='tag';g.textContent='has answers';when.appendChild(g);}
  const bar=document.createElement('div');bar.className='bar';
  for(const k of['H','A','D']){const s=document.createElement('div');s.style.width=(100*p[k])+'%';s.style.background='var(--'+k+')';bar.appendChild(s);}
  const share=document.createElement('div');share.style.color='var(--mute)';share.textContent='Barça '+pct(p.hshare)+' · Athletic '+pct(1-p.hshare);
  a.append(when,bar,share);list.appendChild(a);}
</script></body></html>
"""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Possession viewer</title>
<style>
:root{--bg:#0f1418;--panel:#182027;--ink:#e8edf1;--mute:#8a98a5;--grass:#1f5a32;--line:#d9e7dc;
--H:#3b82f6;--A:#ef4444;--D:#9ca3af;--X:#2b3540}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 12px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}
button{background:var(--panel);color:var(--ink);border:1px solid #2c3944;border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--mute)}
canvas{display:block;width:100%;height:auto;border-radius:8px;background:var(--grass)}
.badge{font-weight:700;padding:4px 10px;border-radius:6px;min-width:150px;text-align:center}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}
.strip{display:flex;height:16px;border-radius:4px;overflow:hidden;margin-top:4px;cursor:pointer}
.strip div{flex:1}.lab{color:var(--mute);font-size:12px;margin-top:10px}
input[type=range]{flex:1;min-width:200px}
#tl{position:relative}#cursor{position:absolute;top:0;width:2px;height:100%;background:#fff;pointer-events:none}
</style></head><body><main>
<h1>Possession viewer <a id="back" style="font-size:13px;color:#8a98a5;margin-left:10px">← whole match</a></h1>
<p class="sub" id="sub"></p>
<canvas id="pitch" width="1050" height="680"></canvas>
<div class="bar">
  <button id="play">Pause</button>
  <button data-s="0.5">0.5×</button><button data-s="1">1×</button><button data-s="2">2×</button><button data-s="4">4×</button>
  <input type="range" id="seek" min="0" value="0">
  <span id="clock"></span>
</div>
<div class="bar">
  <div>Model says: <span class="badge" id="model"></span></div>
  <div id="gtwrap">Public ground truth: <span class="badge" id="gt"></span></div>
</div>
<div class="lab">This stretch — model labels<span class="legend"><span style="background:var(--H)"></span>Barcelona (H)<span style="background:var(--A)"></span>Athletic (A)<span style="background:var(--D)"></span>dead ball (D)</span></div>
<div class="strip" id="local"></div>
<div class="lab" id="gtlab">This stretch — public ground truth</div>
<div class="strip" id="localgt"></div>
<div class="lab">Whole match — model labels (white line = this stretch). Click to see the command for that minute.</div>
<div id="tl"></div>
<p class="sub" id="jump"></p>
</main>
<script>
const D=__DATA__;
const C={H:getComputedStyle(document.documentElement).getPropertyValue('--H').trim(),
         A:getComputedStyle(document.documentElement).getPropertyValue('--A').trim(),
         D:getComputedStyle(document.documentElement).getPropertyValue('--D').trim(),
         X:getComputedStyle(document.documentElement).getPropertyValue('--X').trim()};
const NAMES={H:'Barcelona (H)',A:'Athletic (A)',D:'Dead ball (D)',X:'No answer (X)'};
const cv=document.getElementById('pitch'),ctx=cv.getContext('2d');
const [L,W]=D.pitch,S=cv.width/L,PAD=0;
const seek=document.getElementById('seek');seek.max=D.frames.length-1;
function clock(f){let s=f<D.kick[1]-3000?(f-D.kick[0])/D.fps:2700+(f-D.kick[1])/D.fps;s=Math.max(0,s);
  return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');}
document.getElementById('sub').textContent='Submission "'+D.name+'" · match clock '+clock(D.window[0])+' – '+clock(D.window[1])+' · video frames '+D.window[0]+'–'+(D.window[1]-1);
function X(x){return x*S}function Y(y){return y*S}
function drawPitch(){ctx.fillStyle='#1f5a32';ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='rgba(217,231,220,.8)';ctx.lineWidth=2;
  ctx.strokeRect(X(0),Y(0),X(L),Y(W));ctx.beginPath();ctx.moveTo(X(L/2),0);ctx.lineTo(X(L/2),Y(W));ctx.stroke();
  ctx.beginPath();ctx.arc(X(L/2),Y(W/2),X(9.15),0,7);ctx.stroke();
  for(const s of[0,1]){const x0=s?L-16.5:0;ctx.strokeRect(X(x0),Y(W/2-20.16),X(16.5),Y(40.32));
    const x1=s?L-5.5:0;ctx.strokeRect(X(x1),Y(W/2-9.16),X(5.5),Y(18.32));}}
function badge(el,l){el.textContent=l?NAMES[l]:'—';el.style.background=l?C[l]:'transparent';el.style.color=l==='D'?'#111':'#fff';}
function draw(i){const fr=D.frames[i];drawPitch();
  for(const [x,y,t] of fr.p){ctx.beginPath();ctx.arc(X(x),Y(y),7,0,7);ctx.fillStyle=t===0?C.H:C.A;ctx.fill();
    ctx.lineWidth=1.5;ctx.strokeStyle='#fff';ctx.stroke();}
  if(fr.b){ctx.beginPath();ctx.arc(X(fr.b[0]),Y(fr.b[1]),5,0,7);ctx.fillStyle='#fde047';ctx.fill();ctx.strokeStyle='#111';ctx.lineWidth=1.5;ctx.stroke();}
  else{ctx.fillStyle='rgba(255,255,255,.85)';ctx.font='bold 16px system-ui';ctx.fillText('ball not tracked on this frame',14,26);}
  badge(document.getElementById('model'),fr.l);
  const g=fr.g;document.getElementById('gtwrap').style.display=hasGT?'':'none';badge(document.getElementById('gt'),g);
  document.getElementById('clock').textContent=clock(D.start+i)+'  (frame '+(D.start+i)+')';
  seek.value=i;cur.style.left=(100*i/D.frames.length)+'%';curg.style.left=cur.style.left;}
function strip(el,key){const n=D.frames.length,step=Math.max(1,Math.floor(n/600));
  for(let i=0;i<n;i+=step){const d=document.createElement('div');const v=D.frames[i][key];d.style.background=v?C[v]:C.X;el.appendChild(d);}
  el.style.position='relative';const c=document.createElement('div');c.className='cur';
  c.style.cssText='position:absolute;top:0;width:2px;height:100%;background:#fff';el.appendChild(c);
  el.onclick=e=>{const r=el.getBoundingClientRect();i=Math.floor((e.clientX-r.left)/r.width*n);draw(i);};return c;}
const hasGT=D.frames.some(f=>f.g);
const cur=strip(document.getElementById('local'),'l');
const curg=hasGT?strip(document.getElementById('localgt'),'g'):document.createElement('div');
if(!hasGT){document.getElementById('gtlab').style.display='none';document.getElementById('localgt').style.display='none';}
const tl=document.getElementById('tl');
for(const h of D.timeline){const lab=document.createElement('div');lab.className='lab';lab.textContent=h.half===1?'1st half':'2nd half';tl.appendChild(lab);
  const s=document.createElement('div');s.className='strip';s.style.position='relative';
  const step=Math.max(1,Math.floor(h.cells.length/700));
  for(let k=0;k<h.cells.length;k+=step){const d=document.createElement('div');d.style.background=C[h.cells[k]]||C.X;s.appendChild(d);}
  const a=(D.window[0]-h.first)/D.fps/h.cells.length,b=(D.window[1]-h.first)/D.fps/h.cells.length;
  if(b>0&&a<1){const m=document.createElement('div');m.style.cssText='position:absolute;top:-2px;height:20px;border:2px solid #fff;border-radius:3px;left:'+(Math.max(0,a)*100)+'%;width:'+(Math.max(.4,(Math.min(1,b)-Math.max(0,a))*100))+'%';s.appendChild(m);}
  s.onclick=e=>{const r=s.getBoundingClientRect();const f=h.first+Math.floor((e.clientX-r.left)/r.width*h.cells.length)*D.fps;
    document.getElementById('jump').textContent='To watch from '+clock(f)+' ('+(h.half===1?'1st':'2nd')+' half), run:  python solution/visualize.py --name '+D.name+' --at '+clock(f)+' --half '+h.half;};
  tl.appendChild(s);}
let i=0,playing=true,speed=1,last=performance.now(),acc=0;
function tick(t){const dt=(t-last)/1000;last=t;if(playing){acc+=dt*D.fps*speed;while(acc>=1){acc-=1;i++;if(i>=D.frames.length)i=0;}draw(i);}requestAnimationFrame(tick);}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play';};
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>speed=+b.dataset.s);
seek.oninput=()=>{i=+seek.value;draw(i);};
document.getElementById('back').href=D.index;
draw(0);requestAnimationFrame(tick);
</script></body></html>
"""

if __name__ == "__main__":
    main()
