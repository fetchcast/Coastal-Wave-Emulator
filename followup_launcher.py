#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
followup_launcher.py  (v3)
==========================
Follow-up experiments on the size-dependent generalization result, isolated
from the benchmark. Groups (one initialization seed; sensitivity, not
confirmation):

  R  weight-decay sensitivity: large FFNO/FNO configurations at wd 1e-3 and
     1e-2, plus the control configurations at 1e-2.
  D  data scaling: control and large configurations (same modes 48x48) on
     25 % and 50 % of the training blocks, val/test unchanged, epochs scaled
     by 1/f. The number of optimizer updates is ESTIMATED from the training
     progress bars and reported next to the benchmark run's estimate.
  L  learning-rate sensitivity: the two largest configurations at 5e-5 and
     2.5e-5.
  B  schedule length: the two largest configurations with a 60-epoch
     OneCycle schedule. In the benchmark every FFNO run and most FNO runs
     reached their best validation loss at the final epoch, so a longer
     schedule tests whether the size gap is a budget effect.

Control configuration: the searched minimum width and depth (w128, d4) with
the modes matched to the large configurations (48x48). It was chosen after
the benchmark results were seen; it is not an a-priori selection.

Isolation: train_followup.py (copy of train.py with results_root, log_file and
legacy_train_script redirected) and the patched legacy copy from
patch_train_fraction.py. Benchmark files and folders are never written.

Usage (from ~/swan):
    python3 followup_launcher.py --plan
    python3 followup_launcher.py --run --gpus 0,1,2,3
    python3 followup_launcher.py --analyze
Exit code is non-zero if any job failed or produced no valid, verified run.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SWAN = Path("/home/jovyan/swan")
TRAIN_PY = SWAN / "train.py"
TRAIN_FOLLOWUP = SWAN / "train_followup.py"
LEGACY = SWAN / "UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py"
LEGACY_FOLLOWUP = LEGACY.with_name(LEGACY.stem + "_followup.py")
BENCH_ROOT = SWAN / "runs" / "v2_focused_all"
RESULTS_ROOT = SWAN / "runs" / "v2_followup"
JOBS_DIR = RESULTS_ROOT / "_jobs"
LOG_DIR = RESULTS_ROOT / "_launcher_logs"
STALE_DIR = RESULTS_ROOT / "_stale"
LOCK = RESULTS_ROOT / "_launcher.lock"
STAGE = "followup"
BENCH_STAGE = "focused_all_onefile"
SEED = 42
BASE_EPOCHS = 30
MASTER_PORT_BASE = 29800

CONTROL = {"ffno": (128, 4, 48, 48), "fno": (128, 4, 48, 48)}
LARGE = {"ffno": [(512, 6, 48, 48), (384, 6, 48, 48)],
         "fno":  [(384, 8, 48, 48)]}


# ------------------------------------------------------------------ plan
def batch_rule(model, width, depth, mx, my):
    """Same batch/accumulation rule as operator_jobs() in train.py."""
    m = max(mx, my)
    if model == "fno":
        if width >= 384 or depth >= 8 or m >= 64: return 1, 4
        if width >= 256 or m >= 48: return 2, 2
        return 4, 1
    if width >= 512 or depth >= 8 or m >= 64: return 1, 4
    if width >= 384 or m >= 48: return 2, 2
    return 4, 1


def make_job(model, width, depth, mx, my, wd, lr, fraction, tag, epochs=None):
    bs, acc = batch_rule(model, width, depth, mx, my)
    return {
        "model": model, "config_id": f"{model}_{tag}_w{width}d{depth}m{mx}x{my}",
        "stage": STAGE, "seed": SEED, "use_bnd": "on",
        # maximum epochs (early stopping, patience 3, is part of the recipe);
        # the OneCycle schedule spans this number, so it is a schedule length too
        "epochs": int(epochs if epochs is not None else round(BASE_EPOCHS / fraction)),
        "train_fraction": fraction,
        "bnd_dir_transform": "refl+270",                    # fixed convention; train-only search printed for the record
        "hyperparams": {"seq_length": 12, "hidden_dim": 256,
                        "fno_width": width, "fno_depth": depth, "modes_x": mx, "modes_y": my,
                        "max_lr": lr, "weight_decay": wd, "batch_size": bs, "acc_steps": acc},
    }


def build_plan():
    jobs = []
    for model, cfgs in LARGE.items():                                   # R
        for (w, d, mx, my) in cfgs:
            for wd in (1e-3, 1e-2):
                jobs.append(make_job(model, w, d, mx, my, wd, 1e-4, 1.0, f"reg{wd:.0e}"))
    for model, (w, d, mx, my) in CONTROL.items():
        jobs.append(make_job(model, w, d, mx, my, 1e-2, 1e-4, 1.0, "reg1e-02"))
    for model in ("ffno", "fno"):                                       # D
        for (w, d, mx, my) in (CONTROL[model], LARGE[model][0]):
            for frac in (0.25, 0.5):
                jobs.append(make_job(model, w, d, mx, my, 1e-4, 1e-4, frac, f"frac{int(frac*100):03d}"))
    for model in ("ffno", "fno"):                                       # L
        w, d, mx, my = LARGE[model][0]
        for lr in (5e-5, 2.5e-5):
            jobs.append(make_job(model, w, d, mx, my, 1e-4, lr, 1.0, f"lr{lr:.1e}"))
    for model in ("ffno", "fno"):                                       # B: longer schedule
        w, d, mx, my = LARGE[model][0]
        jobs.append(make_job(model, w, d, mx, my, 1e-4, 1e-4, 1.0, "ep060", epochs=60))
    ids = [j["config_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate config ids"
    return jobs


def run_name_of(job):
    hp = job["hyperparams"]
    return (f"{job['stage']}_{job['model']}_{job['config_id']}_seed{job['seed']}_"
            f"seq{hp['seq_length']}_lr{hp['max_lr']:.0e}_wd{hp['weight_decay']:.0e}")


# ------------------------------------------------------- isolation files
def _write_if_same_or_absent(path, content, overwrite):
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return "reused"
        if not overwrite:
            sys.exit(f"[STOP] {path.name} exists with different content; re-run with --overwrite if intended.")
    path.write_text(content, encoding="utf-8")
    return "written"


def ensure_train_copy(overwrite):
    src = TRAIN_PY.read_text(encoding="utf-8")
    subs = {
        '"results_root": "/home/jovyan/swan/runs/v2_focused_all"': f'"results_root": "{RESULTS_ROOT}"',
        '"log_file": "/home/jovyan/swan/v2_focused_all.log"': f'"log_file": "{SWAN}/v2_followup.log"',
        f'"legacy_train_script": "{LEGACY}"': f'"legacy_train_script": "{LEGACY_FOLLOWUP}"',
    }
    for old, new in subs.items():
        if src.count(old) != 1:
            sys.exit(f"[STOP] expected exactly one match in train.py for: {old}")
        src = src.replace(old, new)
    print(f"[{_write_if_same_or_absent(TRAIN_FOLLOWUP, src, overwrite)}] {TRAIN_FOLLOWUP.name}")


def check_legacy_copy():
    if not LEGACY_FOLLOWUP.exists():
        sys.exit("[STOP] legacy follow-up copy missing; run patch_train_fraction.py first")
    if "SWAN_TRAIN_FRACTION" not in LEGACY_FOLLOWUP.read_text(encoding="utf-8"):
        sys.exit("[STOP] legacy follow-up copy is not patched; run patch_train_fraction.py")
    if "SWAN_TRAIN_FRACTION" in LEGACY.read_text(encoding="utf-8"):
        sys.exit("[STOP] the ORIGINAL legacy script contains the hook; restore it before continuing")


def write_jobs(jobs, overwrite):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    code_sha = hashlib.sha256(LEGACY_FOLLOWUP.read_bytes()).hexdigest()
    paths = []
    for j in jobs:
        j["legacy_followup_sha256"] = code_sha          # code version the job must run with
        p = JOBS_DIR / f"{j['config_id']}_seed{j['seed']}.json"
        _write_if_same_or_absent(p, json.dumps(j, indent=2), overwrite)
        paths.append(p)
    return paths


# ------------------------------------------------------------ log parsing
def latest_launcher_log(job):
    logs = sorted(LOG_DIR.glob(f"{job['config_id']}.*.log"))
    return logs[-1] if logs else None


def parse_training_log(path):
    """Measured quantities from a worker's captured stdout.

    completed_epochs : number of 'Ep{k} Train ...' summary lines (one per finished epoch)
    train_batches    : per-epoch batch count from the 'Epoch k: 100%|...| N/N' bar
                       (the evaluation bar is labelled 'Evaluate' and is excluded)
    n_tr / n_va / n_te, kept_blocks, actual_fraction : from the split messages
    """
    out = dict(completed_epochs=0, train_batches={}, n_tr=None, n_va=None, n_te=None,
               kept_blocks=None, actual_fraction=None, split_fraction_line=False,
               early_stop=False, forced_transform=None, search_pick=None, split_tag=None)
    if path is None or not Path(path).exists():
        return out
    txt = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    m = re.search(r"tr/va/te=(\d+)/(\d+)/(\d+)", txt)
    if m:
        out["n_tr"], out["n_va"], out["n_te"] = (int(m.group(i)) for i in (1, 2, 3))
    m = re.search(r"\[split-fraction\] SWAN_TRAIN_FRACTION=[\d.]+: training blocks (\d+) -> (\d+) \(actual ([\d.]+)\)", txt)
    if m:
        out["split_fraction_line"] = True
        out["kept_blocks"] = int(m.group(2)); out["actual_fraction"] = float(m.group(3))
    for k, n in re.findall(r"Epoch (\d+): 100%\|[^|]*\|\s*(\d+)/\2\s*\[", txt):
        out["train_batches"][int(k)] = int(n)
    out["completed_epochs"] = len(set(int(k) for k in re.findall(r"^Ep(\d+) Train ", txt, re.M)))
    out["early_stop"] = "[EARLY STOP]" in txt
    m = re.search(r"\[split-ok\] (bh=\d+, q=\d+, emb=\d+)", txt)
    if m: out["split_tag"] = m.group(1)
    m = re.search(r"dir transform FORCED to (\S+) \(train-only search would pick (\S+)\)", txt)
    if m: out["forced_transform"], out["search_pick"] = m.group(1), m.group(2)
    return out


def estimated_updates(parsed, acc_steps):
    """Sum over epochs of ceil(batches/acc): the legacy loop steps every
    acc_steps batches and flushes the remainder at the end of the epoch.
    AMP-skipped steps are not visible in the logs, hence 'estimated'."""
    if not parsed["train_batches"]:
        return None
    return sum(math.ceil(n / acc_steps) for n in parsed["train_batches"].values())


# --------------------------------------------------------- verification
def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def hp_equal(a, b):
    for k, v in b.items():
        if k not in a: return False
        if isinstance(v, float):
            if not math.isclose(float(a[k]), v, rel_tol=1e-9): return False
        elif a[k] != v:
            return False
    return True


def verify_run(job):
    """Return (ok, reasons). A run is valid only if the summary, the manifest
    and the captured log all describe exactly the requested job."""
    reasons = []
    rd = RESULTS_ROOT / run_name_of(job)
    s = load_json(rd / "run_summary.json"); mf = load_json(rd / "run_manifest.json")
    if s is None: return False, ["no run_summary.json"]
    if mf is None: reasons.append("no run_manifest.json")
    if s.get("failed", False): reasons.append("summary marked failed")
    if s.get("model") != job["model"]: reasons.append("model mismatch")
    if s.get("stage") != STAGE: reasons.append("stage mismatch")
    if s.get("config_id") != job["config_id"] or int(s.get("seed", -1)) != job["seed"]:
        reasons.append("config_id/seed mismatch")
    if not hp_equal(s.get("hyperparams", {}), job["hyperparams"]): reasons.append("hyperparams mismatch")
    m = s.get("legacy_metrics", {})
    if m.get("rmse_m") is None or not math.isfinite(float(m["rmse_m"])): reasons.append("no finite rmse_m")
    if mf:
        mj = mf.get("job", {})
        if mj.get("use_bnd", "on") != "on": reasons.append("manifest use_bnd != on")
        if int(mj.get("epochs", -1)) != job["epochs"]: reasons.append("manifest epochs mismatch")
        if mj.get("config_id") != job["config_id"]: reasons.append("manifest config_id mismatch")
        if Path(mf.get("resolved_legacy_train_script", "")).name != LEGACY_FOLLOWUP.name:
            reasons.append("manifest legacy script is not the follow-up copy")
    p = parse_training_log(latest_launcher_log(job))
    if p["completed_epochs"] == 0:
        reasons.append("no completed epochs in log")
    elif p["completed_epochs"] < job["epochs"] and not p["early_stop"]:
        reasons.append(f"completed epochs {p['completed_epochs']} < requested {job['epochs']} without an early-stop line")
    elif p["completed_epochs"] > job["epochs"]:
        reasons.append(f"completed epochs {p['completed_epochs']} > requested {job['epochs']}")
    if p["forced_transform"] != job.get("bnd_dir_transform"):
        reasons.append(f"direction transform in log is {p['forced_transform']}, expected {job.get('bnd_dir_transform')}")
    if p["split_tag"] != "bh=168, q=5, emb=12":
        reasons.append(f"split is {p['split_tag']}, expected bh=168, q=5, emb=12")
    want_sha = job.get("legacy_followup_sha256")
    if want_sha and LEGACY_FOLLOWUP.exists():
        if hashlib.sha256(LEGACY_FOLLOWUP.read_bytes()).hexdigest() != want_sha:
            reasons.append("legacy follow-up copy changed since the job was written")
    f = job["train_fraction"]
    if f < 1.0:
        if not p["split_fraction_line"]: reasons.append("no split-fraction line in log")
        elif abs(p["actual_fraction"] - f) > 0.06:
            reasons.append(f"actual fraction {p['actual_fraction']} far from requested {f}")
    else:
        if p["split_fraction_line"]: reasons.append("split-fraction applied to a 100% run")
    return (not reasons), reasons


def job_state(job):
    rd = RESULTS_ROOT / run_name_of(job)
    if (rd / "run_summary.json").exists():
        return "done" if verify_run(job)[0] else "stale"
    return "stale" if rd.exists() else "new"


def move_stale(job):
    rd = RESULTS_ROOT / run_name_of(job)
    STALE_DIR.mkdir(parents=True, exist_ok=True)
    dst = STALE_DIR / f"{rd.name}.{time.strftime('%Y%m%d_%H%M%S')}"
    rd.rename(dst)
    print(f"[stale] {rd.name} -> _stale/{dst.name} (no verified result; will restart from scratch)")


# ------------------------------------------------------------ GPU checks
def busy_gpus():
    """GPU indices with any compute process. Raises if nvidia-smi is unusable."""
    idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=30).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True, timeout=30).stdout
    uuid2idx = {}
    for line in idx.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2: uuid2idx[parts[1]] = int(parts[0])
    busy = set()
    for line in apps.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] in uuid2idx: busy.add(uuid2idx[parts[0]])
    return busy


def gpu_is_free(g, allow_busy):
    if allow_busy: return True
    try:
        return g not in busy_gpus()
    except Exception as exc:
        print(f"[hold] nvidia-smi unavailable ({exc}); not launching on GPU {g}")
        return False


def _omp_threads():
    m = re.search(r'"omp_num_threads":\s*(\d+)', TRAIN_PY.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else 4


# ------------------------------------------------------------ pool
def run_pool(jobs, paths, gpus, allow_busy):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "_error_files").mkdir(parents=True, exist_ok=True)
    if not allow_busy:
        try:
            b = busy_gpus() & set(gpus)
        except Exception as exc:
            sys.exit(f"[STOP] cannot query GPUs ({exc}); pass --allow-busy only if you are sure they are free")
        if b: sys.exit(f"[STOP] GPUs {sorted(b)} have running compute processes")

    queue, skipped = [], 0
    for j, p in zip(jobs, paths):
        st = job_state(j)
        if st == "done": skipped += 1
        else:
            if st == "stale": move_stale(j)
            queue.append((j, p))
    print(f"[pool] {len(queue)} to run on GPUs {gpus}; {skipped} already verified complete")

    running, failed, port = {}, [], MASTER_PORT_BASE

    def _kill_all():
        for g, (proc, job, log) in list(running.items()):
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        time.sleep(5)
        for g, (proc, job, log) in list(running.items()):
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            log.close()
        running.clear()

    def _on_signal(signum, frame):
        print(f"\n[signal] {signum}: terminating {len(running)} worker process group(s)")
        _kill_all(); sys.exit(130)
    signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)

    try:
        while queue or running:
            for g in list(gpus):
                if g in running or not queue: continue
                if not gpu_is_free(g, allow_busy): continue      # re-checked at every assignment
                job, path = queue.pop(0)
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(g)
                env["SWAN_TRAIN_FRACTION"] = str(job["train_fraction"])
                env["SWAN_BND_DIR_TRANSFORM"] = job["bnd_dir_transform"]
                env["OMP_NUM_THREADS"] = str(_omp_threads())
                env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
                env["TORCHELASTIC_ERROR_FILE"] = str(RESULTS_ROOT / "_error_files" / f"{path.stem}_port{port}.json")
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
                       "--master_port", str(port), str(TRAIN_FOLLOWUP), "--worker", "--job_file", str(path)]
                port += 1
                log = open(LOG_DIR / f"{job['config_id']}.{time.strftime('%Y%m%d_%H%M%S')}.log", "w")
                log.write(f"# {time.strftime('%F %T')} gpu={g} frac={job['train_fraction']} epochs={job['epochs']}\n# {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, env=env, cwd=str(SWAN), stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[g] = (proc, job, log)
                print(f"[launch] gpu={g} {job['config_id']} frac={job['train_fraction']} epochs={job['epochs']}")
            for g, (proc, job, log) in list(running.items()):
                rc = proc.poll()
                if rc is None: continue
                log.close(); del running[g]
                ok, why = verify_run(job)
                if rc == 0 and ok:
                    print(f"[done] gpu={g} {job['config_id']}")
                else:
                    reason = f"rc={rc}" if rc != 0 else "; ".join(why)
                    print(f"[FAILED] gpu={g} {job['config_id']} ({reason})")
                    failed.append((job["config_id"], reason))
            time.sleep(30)
    finally:
        if running:
            print("[cleanup] terminating remaining workers"); _kill_all()

    print(f"\n[pool] finished: {len(jobs) - skipped - len(failed)} succeeded, {len(failed)} failed, {skipped} skipped")
    for cid, why in failed: print(f"  failed: {cid}: {why}")
    return 1 if failed else 0


# ------------------------------------------------------------ analysis
def benchmark_baseline(job):
    """The benchmark run with the same architecture and recipe: stage focused_all_onefile,
    BND on, seed 42, 30 epochs, wd 1e-4, lr 1e-4, same batch/acc. Returns
    (rmse, config_id, run_dir, note); ambiguous or missing matches return None."""
    hp = job["hyperparams"]
    want = dict(hp); want["weight_decay"] = 1e-4; want["max_lr"] = 1e-4
    hits = []
    for sp in glob.glob(str(BENCH_ROOT / "*" / "run_summary.json")):
        s = load_json(sp); mf = load_json(Path(sp).parent / "run_manifest.json")
        if not s or s.get("failed", False) or s.get("model") != job["model"]: continue
        if s.get("stage") != BENCH_STAGE or int(s.get("seed", -1)) != SEED: continue
        if not hp_equal(s.get("hyperparams", {}), want): continue
        mj = (mf or {}).get("job", {})
        if mj.get("use_bnd", "on") != "on" or int(mj.get("epochs", BASE_EPOCHS)) != BASE_EPOCHS: continue
        if Path((mf or {}).get("resolved_legacy_train_script", LEGACY.name)).name != LEGACY.name: continue
        hits.append((s["legacy_metrics"].get("rmse_m"), s["config_id"], Path(sp).parent))
    if len(hits) == 1: return hits[0] + ("",)
    if not hits: return None, None, None, "no matching benchmark run"
    return None, None, None, "ambiguous: " + ", ".join(h[1] for h in hits)


def bench_updates(run_dir, acc_steps):
    p = parse_training_log(next(iter(glob.glob(str(run_dir / "*_stdout.log"))), None))
    return estimated_updates(p, acc_steps), p["n_tr"]


def analyze():
    rows = []
    for j in build_plan():
        rd = RESULTS_ROOT / run_name_of(j)
        s = load_json(rd / "run_summary.json")
        if s is None: continue
        ok, why = verify_run(j)
        hp = j["hyperparams"]; p = parse_training_log(latest_launcher_log(j))
        b_rmse, b_id, b_dir, note = benchmark_baseline(j)
        b_upd, b_ntr = bench_updates(b_dir, hp["acc_steps"]) if b_dir else (None, None)
        rows.append(dict(cid=j["config_id"], ok=ok, why=why, model=j["model"], mx=hp["modes_x"], my=hp["modes_y"],
                         wd=hp["weight_decay"], lr=hp["max_lr"], frac=j["train_fraction"],
                         ep=p["completed_epochs"], es=p["early_stop"], pick=p["search_pick"],
                         n_tr=p["n_tr"], est_upd=estimated_updates(p, hp["acc_steps"]),
                         rmse=s.get("legacy_metrics", {}).get("rmse_m"), b_rmse=b_rmse, b_id=b_id,
                         b_upd=b_upd, b_ntr=b_ntr, note=note))
    if not rows: print("no follow-up runs with a summary yet"); return
    fmt = lambda v, w, f: (f"{v:{w}{f}}" if v is not None else f"{'-':>{w}}")
    print(f"{'config':30s}{'ok':>3s}{'modes':>7s}{'wd':>7s}{'lr':>8s}{'frac':>5s}{'ep':>4s}{'n_tr':>6s}"
          f"{'est_upd':>8s}{'rmse':>8s}{'bench':>8s}{'b_upd':>7s}{'b_ntr':>6s}{'delta':>8s}")
    for r in sorted(rows, key=lambda r: (r["model"], r["cid"])):
        delta = (r["rmse"] - r["b_rmse"]) if (r["rmse"] is not None and r["b_rmse"] is not None) else None
        print(f"{r['cid']:30s}{'Y' if r['ok'] else 'N':>3s}{str(r['mx'])+'x'+str(r['my']):>7s}{r['wd']:7.0e}{r['lr']:8.1e}"
              f"{r['frac']:5.2f}{r['ep']:4d}{fmt(r['n_tr'],6,'d')}{fmt(r['est_upd'],8,'d')}{fmt(r['rmse'],8,'.4f')}"
              f"{fmt(r['b_rmse'],8,'.4f')}{fmt(r['b_upd'],7,'d')}{fmt(r['b_ntr'],6,'d')}{fmt(delta,8,'.4f')}")
        print(f"{'':30s}   epochs completed {r['ep']}{' (early stop)' if r['es'] else ''}; train-only search would pick {r['pick']}")
        if not r["ok"]: print(f"{'':30s}   invalid: {'; '.join(r['why'])}")
        if r["note"]: print(f"{'':30s}   baseline: {r['note']}")
        elif r["b_id"]: print(f"{'':30s}   baseline: {r['b_id']}")
    print("\nEpochs are a maximum: the recipe early-stops (patience 3 on the EMA validation loss)."
          "\nEvaluated weights are the final-epoch raw weights (see benchmark_inference for best-checkpoint re-evaluation)."
          "\nest_upd / b_upd: ESTIMATED optimizer updates = sum over COMPLETED epochs of ceil(train batches / acc_steps),"
          "\n  read from the captured training progress bars; AMP-skipped steps are not visible."
          "\nbench: benchmark run with the same architecture, BND on, seed 42, 30 epochs, wd 1e-4, lr 1e-4."
          "\ndelta: follow-up minus bench. Rows marked N did not pass verification and must not be used."
          "\nOne seed: read all of this as sensitivity, not as confirmation.")


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--gpus", default=""); ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze: analyze(); return
    jobs = build_plan()
    if not a.run:
        print(f"{len(jobs)} jobs (seed {SEED}) -> {RESULTS_ROOT}")
        for j in jobs:
            hp = j["hyperparams"]
            print(f"  {j['config_id']:30s} wd={hp['weight_decay']:.0e} lr={hp['max_lr']:.1e} "
                  f"frac={j['train_fraction']:.2f} epochs={j['epochs']:3d} bs={hp['batch_size']} acc={hp['acc_steps']}")
        return
    gpus = [int(x) for x in a.gpus.split(",") if x.strip()]
    if not gpus: sys.exit("[STOP] --gpus is empty; give at least one GPU id")
    check_legacy_copy()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    lock_fh = open(LOCK, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"[STOP] another launcher holds {LOCK}")
    lock_fh.seek(0); lock_fh.truncate(); lock_fh.write(str(os.getpid())); lock_fh.flush()
    try:
        ensure_train_copy(a.overwrite)
        paths = write_jobs(jobs, a.overwrite)
        rc = run_pool(jobs, paths, gpus, a.allow_busy)
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN); lock_fh.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
