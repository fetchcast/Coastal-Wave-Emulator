#!/usr/bin/env python3
"""Aggregate BND leakage/control test results into one table and figure.

After running benchmark_inference_full_fixed.py with several --bnd-corrupt modes,
this script gathers:
  - clean bndON      (from _inference_metrics/per_run/*.npz, use_bnd=on)
  - clean bndOFF     (same, use_bnd=off)                 [reference floor]
  - each corrupt mode (from _inference_metrics/bnd_control_<mode>/per_run/*.npz)

and reports, per model, the seed-42 Hs/Tm/Dir under each condition, plus the
"BND value recovered" fraction:

    recovered = (err_corrupt - err_bndON) / (err_bndOFF - err_bndON)

Interpretation for a genuine boundary signal (no shortcut leakage):
  - recovered ~ 1.0  : corrupt BND is as useless as no BND (model relied on
                       correct temporal alignment -> BND carried real physics)
  - recovered ~ 0.0  : corrupt BND is as good as correct BND (model ignored the
                       time alignment; BND acted as a static prior only)
  - recovered  > 1.0 : corrupt BND is WORSE than no BND (misleading), which is
                       still consistent with genuine use, not leakage
A shortcut-leakage red flag would instead look like: bndON far below bndOFF AND
recovered ~ 0 for shuffle/shift (model reproduced target from a BND channel that
trivially encodes it), together with unphysical pred-vs-true structure.

Usage:
  python bnd_leakage_report.py --root ~/swan/runs/focused_rerun_all_onefile/_inference_metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_rows(per_run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    if not per_run_dir.is_dir():
        return rows
    for npz in sorted(per_run_dir.glob("*.npz")):
        try:
            z = np.load(npz, allow_pickle=False)
            rows.append(json.loads(str(z["row"])))
        except Exception as exc:
            print(f"[warn] could not read {npz.name}: {exc}")
    return rows


def seed42_on(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Map model -> metrics for the seed-42 multiseed bndON run."""
    out = {}
    for r in rows:
        if r.get("stage") == "stage2_multiseed" and int(r.get("seed", -1)) == 42 \
           and str(r.get("use_bnd", "on")).lower() == "on":
            out[r["model"]] = r
    return out


def bndoff_floor(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Map model -> metrics for the bndOFF ablation run (seed 42)."""
    out = {}
    for r in rows:
        if r.get("stage") == "stage2_bndoff" or str(r.get("use_bnd", "")).lower() == "off":
            out[r["model"]] = r
    return out


METRICS = [("hs_rmse_stepmean", "Hs_RMSE_m"),
           ("tm_rmse_stepmean", "Tm_RMSE_s"),
           ("dir_crmse_stepmean", "Dir_cRMSE_deg")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="the _inference_metrics directory")
    ap.add_argument("--modes", default="shuffle,shift,wrongyear",
                    help="comma-separated corruption modes to include if present")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    clean_rows = load_rows(root / "per_run")
    on = seed42_on(clean_rows)
    off = bndoff_floor(clean_rows)
    if not on:
        print(f"[FATAL] no clean bndON seed-42 runs under {root/'per_run'}")
        return 1

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    corrupt: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in modes:
        d = root / f"bnd_control_{m}" / "per_run"
        rws = load_rows(d)
        cm = {}
        for r in rws:
            if int(r.get("seed", -1)) == 42 and str(r.get("use_bnd", "on")).lower() == "on":
                cm[r["model"]] = r
        if cm:
            corrupt[m] = cm
            print(f"[load] mode '{m}': {len(cm)} models")
        else:
            print(f"[skip] mode '{m}': no runs found under {d}")

    # Build long table
    import csv
    out_csv = root / "bnd_leakage_summary.csv"
    models = sorted(on.keys())
    header = ["model", "metric", "bndON", "bndOFF"]
    for m in corrupt:
        header += [f"corrupt_{m}", f"recovered_{m}"]
    lines = []
    for model in models:
        for col, label in METRICS:
            e_on = float(on[model].get(col, float("nan")))
            e_off = float(off.get(model, {}).get(col, float("nan")))
            row = {"model": model, "metric": label, "bndON": e_on, "bndOFF": e_off}
            denom = e_off - e_on
            for m in corrupt:
                e_c = float(corrupt[m].get(model, {}).get(col, float("nan")))
                row[f"corrupt_{m}"] = e_c
                rec = (e_c - e_on) / denom if np.isfinite(denom) and abs(denom) > 1e-9 else float("nan")
                row[f"recovered_{m}"] = rec
            lines.append(row)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in lines:
            w.writerow(r)
    print(f"\n[table] wrote {out_csv}")

    # Console summary for Hs
    print("\n=== BND leakage test — Hs RMSE (seed 42) ===")
    print(f"{'model':15s}{'bndON':>9s}{'bndOFF':>9s}" + "".join(f"{'c_'+m:>10s}{'rec_'+m:>9s}" for m in corrupt))
    for model in models:
        e_on = float(on[model].get("hs_rmse_stepmean", float("nan")))
        e_off = float(off.get(model, {}).get("hs_rmse_stepmean", float("nan")))
        s = f"{model:15s}{e_on:9.4f}{e_off:9.4f}"
        denom = e_off - e_on
        for m in corrupt:
            e_c = float(corrupt[m].get(model, {}).get("hs_rmse_stepmean", float("nan")))
            rec = (e_c - e_on) / denom if np.isfinite(denom) and abs(denom) > 1e-9 else float("nan")
            s += f"{e_c:10.4f}{rec:9.2f}"
        print(s)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = root / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        order = sorted(on.keys(), key=lambda k: float(on[k].get("hs_rmse_stepmean", 9)))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        x = np.arange(len(order))
        conditions = ["bndON"] + [f"corrupt_{m}" for m in corrupt] + ["bndOFF"]
        width = 0.8 / len(conditions)
        for ax, (col, label) in zip(axes, METRICS):
            for j, cond in enumerate(conditions):
                vals = []
                for model in order:
                    if cond == "bndON":
                        v = on[model].get(col, np.nan)
                    elif cond == "bndOFF":
                        v = off.get(model, {}).get(col, np.nan)
                    else:
                        mm = cond.replace("corrupt_", "")
                        v = corrupt[mm].get(model, {}).get(col, np.nan)
                    vals.append(float(v))
                ax.bar(x + j * width - 0.4 + width / 2, vals, width, label=cond)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(label.replace("_", " "))
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle("BND leakage/control test: correct vs. corrupted boundary features (seed 42)")
        fig.tight_layout()
        fig.savefig(figs / "fig_bnd_leakage.png", dpi=200)
        plt.close(fig)
        print(f"[fig] wrote {figs/'fig_bnd_leakage.png'}")
    except Exception as exc:
        print(f"[fig] failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
