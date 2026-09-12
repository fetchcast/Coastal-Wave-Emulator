#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_epoch_report.py
=====================
For every completed benchmark run: completed epochs, whether early stopping
fired, the estimated number of optimizer updates, the direction transform the
autocorrect selected, the split used, and the test RMSE, joined with the
configuration (width, depth, modes_x x modes_y). Reads only existing logs and
summaries; nothing is modified.

Why: the recipe early-stops with patience 3, so "30 epochs" is a maximum.
If large configurations stop earlier than small ones, fewer updates is a
confound for the size-vs-error pattern and must be reported.

Usage (from ~/swan):  python3 bench_epoch_report.py [--csv out.csv]
"""
import argparse, csv, glob, json, math, re, sys
from pathlib import Path

ROOT = Path("/home/jovyan/swan/runs/v2_focused_all")


def parse_log(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    batches = {int(k): int(n) for k, n in re.findall(r"Epoch (\d+): 100%\|[^|]*\|\s*(\d+)/\2\s*\[", txt)}
    done = len(set(int(k) for k in re.findall(r"^Ep(\d+) Train ", txt, re.M)))
    es = re.search(r"\[EARLY STOP\].*?Stop at ep (\d+)", txt)
    best = [int(k) for k in re.findall(r"\[CHECKPOINT\] NEW BEST ep (\d+)", txt)]
    tr = re.search(r"chosen (reflection theta'=\S+|rotation \S+ deg|rotation [+-]?\d+°|[+-]?\d+°)", txt)
    split = re.search(r"\[split-ok\] (bh=\d+, q=\d+, emb=\d+)", txt)
    return dict(completed=done, early_stop_ep=int(es.group(1)) if es else None,
                best_ep=max(best) if best else None, batches=batches,
                transform=tr.group(1) if tr else None, split=split.group(1) if split else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=None); a = ap.parse_args()
    rows = []
    for sp in sorted(glob.glob(str(ROOT / "*" / "run_summary.json"))):
        rd = Path(sp).parent
        s = json.loads(Path(sp).read_text(encoding="utf-8"))
        if s.get("failed"): continue
        logs = glob.glob(str(rd / "*_stdout.log"))
        p = parse_log(logs[0]) if logs else dict(completed=None, early_stop_ep=None, best_ep=None, batches={}, transform=None, split=None)
        hp, m = s.get("hyperparams", {}), s.get("legacy_metrics", {})
        acc = int(hp.get("acc_steps", 1))
        upd = sum(math.ceil(n / acc) for n in p["batches"].values()) if p["batches"] else None
        rows.append(dict(model=s["model"], stage=s.get("stage"), config=s["config_id"], seed=s.get("seed"),
                         w=hp.get("fno_width") or hp.get("hidden_dim"), d=hp.get("fno_depth"),
                         modes=f"{hp.get('modes_x')}x{hp.get('modes_y')}" if hp.get("modes_x") else "",
                         completed=p["completed"], early_stop_ep=p["early_stop_ep"], best_ep=p["best_ep"],
                         est_updates=upd, transform=p["transform"], split=p["split"], rmse=m.get("rmse_m")))
    if not rows: sys.exit("no completed runs found")
    print(f"{'model':11s}{'config':24s}{'w':>5s}{'d':>3s}{'modes':>7s}{'done':>5s}{'ES@':>4s}{'best':>5s}{'est_upd':>8s}{'rmse':>8s}  transform")
    for r in sorted(rows, key=lambda r: (r["model"], r["stage"] or "", r["w"] or 0, r["d"] or 0, r["config"])):
        f = lambda v, w: (f"{v:>{w}}" if v is not None else f"{'-':>{w}}")
        print(f"{r['model']:11s}{r['config']:24s}{f(r['w'],5)}{f(r['d'],3)}{r['modes']:>7s}{f(r['completed'],5)}"
              f"{f(r['early_stop_ep'],4)}{f(r['best_ep'],5)}{f(r['est_updates'],8)}"
              f"{(r['rmse'] if r['rmse'] is not None else float('nan')):8.4f}  {r['transform']}")
    n_es = sum(1 for r in rows if r["early_stop_ep"])
    print(f"\n{len(rows)} runs; {n_es} early-stopped; transforms seen: {sorted(set(str(r['transform']) for r in rows))}; "
          f"splits seen: {sorted(set(str(r['split']) for r in rows))}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
