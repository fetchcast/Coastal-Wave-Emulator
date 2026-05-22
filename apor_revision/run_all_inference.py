# -*- coding: utf-8 -*-
"""
run_all_inference.py
====================
Run inference for all eight APOR revision experiments (E01-E08) in
sequence, each through the SAME pipeline (inference_ablation_v5_3.py),
so the resulting ablation tables come from one code base.

What this script does
---------------------
For each experiment it builds the correct command line for
inference_ablation_v5_3.py, with the right --variant, --split, and
--use_bnd, then runs it as a subprocess and waits for it to finish
before starting the next one.

The per-experiment settings below were read directly off the checkpoint
filenames you provided:

  E01_main          full           block   bnd on
  E02_convlstm_only convlstm_only  block   bnd on
  E03_unetpp_stack  unetpp_stack   block   bnd on
  E04_chrono_2019tr full           chrono  bnd on   (train 2019, test 2020)
  E05_bnd_off       full           block   bnd off
  E06_seed7         full           block   bnd on
  E07_seed1337      full           block   bnd on
  E08_seed2024      full           block   bnd on

Only E04 uses a different split. Every other run is block-stratified.
E05 is the only BND-off run (6 input channels); all others are BND-on
(10 input channels).

How to use
----------
1. Edit the PATHS section below so it points at your local files.
2. Make sure these three modules sit next to inference_ablation_v5_3.py:
       model_architectures.py
       revision_patches.py
       (and the project modules it already needs: bnd_features.py,
        boundspec_segments.py, etc.)
3. Run:
       python run_all_inference.py
   or run a subset:
       python run_all_inference.py --only E01_main E04_chrono_2019tr
   or skip ones you have already done:
       python run_all_inference.py --skip E05_bnd_off

Each experiment writes into its own timestamped folder under
./outputs/ (the --tag goes into the folder name), so runs never
overwrite each other.

Notes
-----
* This script does NOT train anything. It only runs inference on
  checkpoints that already exist.
* If a run fails, this script prints the error and, by default, stops.
  Pass --keep-going to continue with the remaining experiments instead.
* E04's checkpoint was trained on a chronological split. The inference
  MUST use --split chrono_2019_train_2020_test for E04, because the
  train indices drive both the normalization statistics and the
  train-only boundary-direction rotation. Using the block split for E04
  would silently produce wrong numbers.
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime


# ===========================================================================
# PATHS  --  EDIT THIS SECTION FOR YOUR MACHINE
# ===========================================================================
# The Python interpreter to use. sys.executable = the same Python you
# launched this script with, which is usually what you want.
PYTHON = sys.executable

# The inference script (the patched V5.3 ablation version).
INFERENCE_SCRIPT = "inference_ablation_v5_3.py"

DATA_PATH = r"C:\DELFT3DFM\South_Korea_emulator_2020_ST6_bnd_test\wave\wavm-Waves_2019_2020_final.nc"
STATION_ROOT = r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET"
WEIGHTS_DIR = r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET"

# Fixed settings shared by every run (these match the training config).
SEQ_LENGTH = 12
FEAT = ["32", "64", "128", "256", "512"]   # passed as separate argv tokens


# ===========================================================================
# EXPERIMENT TABLE
# ===========================================================================
# Each entry:
#   tag       : short name, also used for the output folder prefix
#   weights   : checkpoint filename (inside WEIGHTS_DIR)
#   variant   : full | convlstm_only | unetpp_stack | unetpp_only
#   split     : block | chrono_2019_train_2020_test | chrono_2020_train_2019_test
#   use_bnd   : on | off
#
# The weight filenames are exactly the ones you listed. If your files are
# named slightly differently (for example weights_*.pth instead of
# ckpt_*.pth), update the 'weights' field here.
EXPERIMENTS = [
    {
        "tag":     "E01_main",
        "weights": "ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth",
        "variant": "full",
        "split":   "block",
        "use_bnd": "on",
    },
    {
        "tag":     "E02_convlstm_only",
        "weights": "ckpt_E02_convlstm_only_convlstm_only_block_seed42_bndtrainonly_usebndon_best_ema.pth",
        "variant": "convlstm_only",
        "split":   "block",
        "use_bnd": "on",
    },
    {
        "tag":     "E03_unetpp_stack",
        "weights": "ckpt_E03_unetpp_stack_unetpp_stack_block_seed42_bndtrainonly_usebndon_best_ema.pth",
        "variant": "unetpp_stack",
        "split":   "block",
        "use_bnd": "on",
    },
    {
        "tag":     "E04_chrono_2019tr",
        "weights": "ckpt_E04_chrono_2019tr_full_chrono_2019_train_2020_test_seed42_bndtrainonly_usebndon_best_ema.pth",
        "variant": "full",
        "split":   "chrono_2019_train_2020_test",   # train 2019, test 2020
        "use_bnd": "on",
    },
    {
        "tag":     "E05_bnd_off",
        "weights": "ckpt_E05_bnd_off_full_block_seed42_bndtrainonly_usebndoff_best_ema.pth",
        "variant": "full",
        "split":   "block",
        "use_bnd": "off",
    },
    {
        "tag":     "E06_seed7",
        "weights": "ckpt_E06_seed7_full_block_seed7_bndtrainonly_usebndon_best_ema.pth",
        "variant": "full",
        "split":   "block",
        "use_bnd": "on",
    },
    {
        "tag":     "E07_seed1337",
        "weights": "ckpt_E07_seed1337_full_block_seed1337_bndtrainonly_usebndon_best_ema.pth",
        "variant": "full",
        "split":   "block",
        "use_bnd": "on",
    },
    {
        "tag":     "E08_seed2024",
        "weights": "ckpt_E08_seed2024_full_block_seed2024_bndtrainonly_usebndon_best_ema.pth",
        "variant": "full",
        "split":   "block",
        "use_bnd": "on",
    },
]


# ===========================================================================
# Runner
# ===========================================================================
def build_command(exp):
    """Build the argv list for one experiment."""
    weights_path = os.path.join(WEIGHTS_DIR, exp["weights"])
    cmd = [
        PYTHON, "-u", INFERENCE_SCRIPT,
        "--data_path",    DATA_PATH,
        "--weights",      weights_path,
        "--station_root", STATION_ROOT,
        "--variant",      exp["variant"],
        "--split",        exp["split"],
        "--use_bnd",      exp["use_bnd"],
        "--feat",         *FEAT,
        "--seq_length",   str(SEQ_LENGTH),
        "--tag",          exp["tag"],
    ]
    return cmd, weights_path


def preflight_checks(experiments):
    """Check that the script, data, and every checkpoint exist before
    starting. It is much better to fail here than three experiments in."""
    problems = []
    if not os.path.isfile(INFERENCE_SCRIPT):
        problems.append(f"inference script not found: {INFERENCE_SCRIPT}")
    if not os.path.isfile(DATA_PATH):
        problems.append(f"data file not found: {DATA_PATH}")
    if not os.path.isdir(STATION_ROOT):
        problems.append(f"station folder not found: {STATION_ROOT} "
                        f"(station tables will be NaN, ablation table still OK)")
    for exp in experiments:
        wp = os.path.join(WEIGHTS_DIR, exp["weights"])
        if not os.path.isfile(wp):
            problems.append(f"[{exp['tag']}] checkpoint not found: {wp}")
        else:
            size = os.path.getsize(wp)
            # A real checkpoint for this model is on the order of MB.
            # A file of a few hundred bytes is almost certainly broken.
            if size < 100_000:
                problems.append(
                    f"[{exp['tag']}] checkpoint is suspiciously small "
                    f"({size} bytes): {wp}  -- this is probably not a valid "
                    f"weight file; re-check or re-train this experiment.")
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Run E01-E08 inference in sequence through "
                    "inference_ablation_v5_3.py")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these tags (e.g. --only E01_main E04_chrono_2019tr)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip these tags (e.g. --skip E05_bnd_off)")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue with remaining experiments even if one fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would run, then exit")
    args = parser.parse_args()

    # Decide which experiments to run.
    selected = EXPERIMENTS
    if args.only:
        only = set(args.only)
        selected = [e for e in EXPERIMENTS if e["tag"] in only]
        missing = only - {e["tag"] for e in EXPERIMENTS}
        if missing:
            print(f"[warn] --only listed unknown tags, ignored: {sorted(missing)}")
    if args.skip:
        skip = set(args.skip)
        selected = [e for e in selected if e["tag"] not in skip]

    if not selected:
        print("[error] no experiments selected.")
        sys.exit(1)

    print("=" * 70)
    print(f"Planned runs ({len(selected)}):")
    for e in selected:
        print(f"  {e['tag']:22s} variant={e['variant']:14s} "
              f"split={e['split']:30s} bnd={e['use_bnd']}")
    print("=" * 70)

    # Preflight.
    problems = preflight_checks(selected)
    if problems:
        print("\n[preflight] problems found:")
        for p in problems:
            print("  -", p)
        # A missing station folder is only a warning; a missing or broken
        # checkpoint or data file is fatal.
        fatal = [p for p in problems if "station folder" not in p]
        if fatal and not args.dry_run:
            print("\n[preflight] fatal problems above. Fix them, then re-run. "
                  "(Use --dry-run to print commands without running.)")
            sys.exit(1)
        print()

    if args.dry_run:
        print("[dry-run] commands that would be executed:\n")
        for e in selected:
            cmd, _ = build_command(e)
            print("  " + " ".join(cmd))
        print("\n[dry-run] nothing was executed.")
        return

    # Run each experiment in sequence.
    results = []
    t_all = time.time()
    for i, exp in enumerate(selected, 1):
        cmd, weights_path = build_command(exp)
        print("\n" + "=" * 70)
        print(f"[{i}/{len(selected)}] {exp['tag']}  "
              f"({datetime.now().strftime('%H:%M:%S')})")
        print("  " + " ".join(cmd))
        print("=" * 70, flush=True)

        t0 = time.time()
        proc = subprocess.run(cmd)
        dt = time.time() - t0

        ok = (proc.returncode == 0)
        results.append((exp["tag"], ok, dt, proc.returncode))
        status = "OK" if ok else f"FAILED (exit {proc.returncode})"
        print(f"\n[{i}/{len(selected)}] {exp['tag']}: {status}  "
              f"({dt/60.0:.1f} min)", flush=True)

        if not ok and not args.keep_going:
            print(f"\n[stop] {exp['tag']} failed and --keep-going was not set. "
                  f"Stopping. Already-finished runs are saved.")
            break

    # Summary.
    total = time.time() - t_all
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for tag, ok, dt, rc in results:
        status = "OK    " if ok else f"FAIL({rc})"
        print(f"  {status}  {tag:24s}  {dt/60.0:6.1f} min")
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    print("-" * 70)
    print(f"  {n_ok}/{len(results)} succeeded   total {total/60.0:.1f} min")
    print("=" * 70)

    # Non-zero exit if anything failed, so a wrapping shell script can tell.
    if any(not ok for _, ok, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
