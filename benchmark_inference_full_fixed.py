#!/usr/bin/env python3
"""Stage-2 benchmark inference: per-variable physical metrics (Hs, Tm, Dir) and figures.

Strategy
--------
- Architectures are rebuilt from train.py's MODEL_REGISTRY (exactly the classes
  that were trained) and verified by strict state-dict loading. The number of
  input channels is inferred by trying candidates until the checkpoint loads.
- The data pipeline is NOT reimplemented. The legacy training module is imported
  and probed for a test-loader builder. A dataset of length TEST_LEN (1980) is
  used as the fingerprint of the test split.
- Self-validation: for every run, the recomputed Hs RMSE (per-timestep mean over
  ocean pixels, physical meters) is compared against the rmse_m stored in the
  run's run_summary.json. A mismatch means the data pipeline is not aligned and
  the numbers must not be used.

Conventions (mirroring the legacy evaluation code around line 1076-1105)
------------------------------------------------------------------------
- Output channels: [Hs, Tm, sin(theta), cos(theta)].
- Ocean mask: kcs_map > 0 AND finite(pred) AND finite(true), applied per sample.
- Legacy metric convention: per-timestep RMSE/MAE over ocean pixels, then the
  mean over timesteps ("stepmean"). Pooled metrics over all pixels x steps are
  also reported ("pooled") because the paper should state which one it uses.
- Hs and Tm are denormalized with global min-max parameters. Direction is
  reconstructed from sin/cos with atan2; the absolute angular convention cancels
  in pred-true differences as long as both use the same transform.

Usage
-----
  python benchmark_inference.py --probe                  # inspect legacy module
  python benchmark_inference.py --models ffno --limit-steps 20   # smoke test
  python benchmark_inference.py --models ffno            # full single model + validation
  python benchmark_inference.py                          # all 40 runs + figures
  python benchmark_inference.py --figures-only           # re-plot from saved outputs
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Torch is imported lazily inside main() paths so that --figures-only can run
# on a machine without a GPU stack. Module-level import kept for type usage.
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Constants and fallbacks
# ----------------------------------------------------------------------------

TEST_LEN_DEFAULT = 1980  # fingerprint of the test split (tr/va/te = 9770/1980/1980)

# Fallback normalization ranges, confirmed from training logs:
#   "Hs range: (0.0, 5.0370001792907715)   Tm range: (0.0, 15.0)"
FALLBACK_NORM = {"hs": (0.0, 5.0370001792907715), "tm": (0.0, 15.0)}

IN_CH_CANDIDATES = [9, 6, 5, 7, 8, 10, 12, 4, 11, 3]

# Hyperparameter keys that are training-only and never constructor kwargs.
HP_DROP = {"max_lr", "weight_decay", "batch_size", "acc_steps"}

# Per-model alias maps: hp-name -> constructor-kwarg-name. Identity is always
# tried first; aliases are fallbacks verified by strict state-dict loading.
HP_ALIASES: Dict[str, List[Dict[str, str]]] = {
    # Names in run_manifest.json -> names expected by the model constructors in train.py.
    # Identity is still tried, but these aliases are needed for configs whose search-space
    # names differ from constructor names.
    "fno": [{"fno_width": "width", "fno_depth": "depth"}],
    "ffno": [{"fno_width": "width", "fno_depth": "depth"}],
    "tno": [{"fno_width": "width", "fno_depth": "depth"}],
    "u_ffno": [{}],
    "convlstm": [{}],
    "unet_lstm": [{"unet_feat": "feat"}, {}],

    # FIXED from train.py:
    # ConvNeXtLSTMEmulator expects dims/depths/lstm_hidden. The manifest stores
    # convnext_dims/convnext_depths/lstm_hidden. Do NOT map lstm_hidden to hidden_dim.
    "convnext_lstm": [{"convnext_dims": "dims", "convnext_depths": "depths"}, {}],

    # SwinUNetEmulator expects depths/num_heads. The manifest stores
    # swin_depths/swin_num_heads. embed_dim/window_size/patch_size already match.
    "swin": [{"swin_depths": "depths", "swin_num_heads": "num_heads"}, {}],

    # ViTEmulator expects depth/num_heads. The manifest stores vit_depth/vit_heads.
    "vit": [{"vit_depth": "depth", "vit_heads": "num_heads"}, {}],

    # ConvSwinUNetEmulator constructor names already match the manifest keys.
    "conv_swin": [{}],
}

SEASTATE_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 999.0])

HS_MIN_FOR_DIR = 0.5  # meters; extra direction metric restricted to true Hs >= 0.5


# ----------------------------------------------------------------------------
# Module loading
# ----------------------------------------------------------------------------

def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def assert_main_guard(path: Path) -> None:
    """Refuse to import a script that would launch its main() on import."""
    src = path.read_text(encoding="utf-8", errors="replace")
    if "__name__" not in src or "__main__" not in src:
        raise RuntimeError(
            f"{path} has no `if __name__ == '__main__'` guard; importing it would "
            "execute its top-level entry point. Add the guard before running this script."
        )


# ----------------------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------------------

def discover_runs(results_root: Path, stages: List[str], models: Optional[List[str]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.name.startswith(s) for s in stages):
            continue
        manifest_path = d / "run_manifest.json"
        if not manifest_path.exists():
            print(f"[warn] no run_manifest.json in {d.name}; skipped")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        job = manifest.get("job", {})
        model = job.get("model")
        if models and model not in models:
            continue
        weights = sorted(d.glob("*model_weights*.pth"))
        if not weights:
            print(f"[warn] no weight file in {d.name}; skipped")
            continue
        if len(weights) > 1:
            print(f"[warn] {len(weights)} weight files in {d.name}; using {weights[-1].name}")
        summary = {}
        sp = d / "run_summary.json"
        if sp.exists():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                pass
        runs.append({
            "run_dir": d,
            "run_name": d.name,
            "job": job,
            "model": model,
            "stage": job.get("stage", ""),
            "seed": int(job.get("seed", -1)),
            "config_id": job.get("config_id", ""),
            "use_bnd": str(job.get("use_bnd", "on")),
            "weight_path": weights[-1],
            "summary": summary,
        })
    return runs


def summary_reference_hs_rmse(summary: Dict[str, Any]) -> Optional[float]:
    """Extract the stored physical Hs RMSE (rmse_m) from run_summary.json."""
    lm = summary.get("legacy_metrics") or {}
    for key in ("rmse_m", "rmse_hs", "rmse"):
        v = lm.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def summary_efficiency(summary: Dict[str, Any]) -> Dict[str, float]:
    sb = summary.get("synthetic_benchmark") or {}
    def pick(*names):
        for n in names:
            if n in sb and sb[n] is not None:
                try:
                    return float(sb[n])
                except (TypeError, ValueError):
                    pass
        return float("nan")
    return {
        "latency_ms": pick("mean_ms", "latency_ms", "mean_latency_ms"),
        "params_M": pick("params_M", "params_m", "n_params_M"),
        "gpu_mem_GB": pick("gpu_mem_GB", "gpu_mem_gb", "peak_mem_GB"),
    }


# ----------------------------------------------------------------------------
# Model reconstruction
# ----------------------------------------------------------------------------

def collect_ctor_params(cls) -> set:
    """Union of named __init__ parameters over the MRO (handles **kwargs chains)."""
    params: set = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                params.add(pname)
    return params


def clean_hp(hp: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in hp.items():
        if k in HP_DROP or v is None:
            continue
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        if isinstance(v, float) and float(v).is_integer() and k not in ("max_lr", "weight_decay"):
            v = int(v)
        out[k] = v
    return out


def train_style_kwargs(model: str, hp: Dict[str, Any], in_ch: int, seq_len: int) -> Dict[str, Any]:
    """Build constructor kwargs using the same mapping logic as train.py.

    This is the source-of-truth path for strict checkpoint reconstruction. The
    generic alias fallback below remains useful for older checkpoints, but the
    first attempt should mirror LegacyCompatibleBenchmarkModel / benchmark_model
    in train.py.
    """
    kw: Dict[str, Any] = {"input_channels": in_ch, "output_channels": 4, "seq_length": seq_len}

    if model in {"fno", "ffno"}:
        kw.update({
            "width": int(hp.get("fno_width", hp.get("width", 64))),
            "depth": int(hp.get("fno_depth", hp.get("depth", 4))),
            "modes_x": int(hp.get("modes_x", 24)),
            "modes_y": int(hp.get("modes_y", 24)),
        })
    elif model == "tno":
        kw.update({
            "width": int(hp.get("width", hp.get("hidden_dim", 64))),
            "depth": int(hp.get("depth", hp.get("fno_depth", 4))),
            "modes_x": int(hp.get("modes_x", 24)),
            "modes_y": int(hp.get("modes_y", 24)),
            "modes_t": int(hp.get("modes_t", 6)),
            "use_checkpoint": bool(hp.get("use_checkpoint", False)),
        })
    elif model == "u_ffno":
        kw.update({
            "unet_feat": hp.get("unet_feat", hp.get("feat", [64, 128, 256, 512, 1024])),
            "hidden_dim": int(hp.get("hidden_dim", 256)),
            "fno_width": int(hp.get("fno_width", hp.get("width", 256))),
            "fno_depth": int(hp.get("fno_depth", hp.get("depth", 4))),
            "modes_x": int(hp.get("modes_x", 16)),
            "modes_y": int(hp.get("modes_y", 16)),
        })
    elif model == "unet_lstm":
        kw.update({
            "hidden_dim": int(hp.get("hidden_dim", 128)),
            "feat": hp.get("unet_feat", hp.get("feat", [32, 64, 128, 256, 512])),
        })
    elif model == "convlstm":
        kw.update({
            "hidden_dim": int(hp.get("hidden_dim", hp.get("width", 128))),
            "width": int(hp.get("width", hp.get("hidden_dim", 128))),
            "depth": int(hp.get("depth", hp.get("fno_depth", 2))),
        })
    elif model == "swin":
        kw.update({
            "embed_dim": int(hp.get("embed_dim", hp.get("width", 96))),
            "depths": hp.get("swin_depths", hp.get("depths", [2, 2, 6, 2])),
            "num_heads": hp.get("swin_num_heads", hp.get("num_heads", [3, 6, 12, 24])),
            "window_size": int(hp.get("window_size", 8)),
            "patch_size": int(hp.get("patch_size", 4)),
            "use_checkpoint": bool(hp.get("use_checkpoint", False)),
        })
    elif model == "vit":
        kw.update({
            "embed_dim": int(hp.get("embed_dim", 384)),
            "depth": int(hp.get("vit_depth", hp.get("depth", 8))),
            "num_heads": int(hp.get("vit_heads", hp.get("num_heads", 6))),
            "patch_size": int(hp.get("patch_size", 16)),
            "use_checkpoint": bool(hp.get("use_checkpoint", False)),
        })
    elif model == "convnext_lstm":
        kw.update({
            "dims": hp.get("convnext_dims", hp.get("dims", [64, 128, 256])),
            "depths": hp.get("convnext_depths", hp.get("depths", [2, 2, 2])),
            "lstm_hidden": int(hp.get("lstm_hidden", 256)),
            "use_checkpoint": bool(hp.get("use_checkpoint", False)),
        })
    elif model == "conv_swin":
        kw.update({
            "base_width": int(hp.get("base_width", 48)),
            "swin_dim": int(hp.get("swin_dim", 256)),
            "swin_depth": int(hp.get("swin_depth", 4)),
            "swin_heads": int(hp.get("swin_heads", 8)),
            "window_size": int(hp.get("window_size", 8)),
            "use_checkpoint": bool(hp.get("use_checkpoint", False)),
        })
    return kw


def kwargs_variants(cls, model: str, hp: Dict[str, Any], in_ch: int, seq_len: int) -> Iterator[Dict[str, Any]]:
    sigp = collect_ctor_params(cls)
    base = {"input_channels": in_ch, "output_channels": 4, "seq_length": seq_len}
    seen: set = set()

    # First try the exact train.py-style mapping. This fixes ConvNeXt-LSTM and
    # Swin checkpoints whose manifest keys differ from constructor keys.
    primary = {k: v for k, v in train_style_kwargs(model, hp, in_ch, seq_len).items() if k in sigp}
    key = tuple(sorted((k, repr(v)) for k, v in primary.items()))
    seen.add(key)
    yield primary

    # Fallbacks: identity and explicit aliases. These are verified by strict
    # state-dict loading, so harmless duplicates are removed.
    alias_list = [{}] + [a for a in HP_ALIASES.get(model, []) if a]
    for alias in alias_list:
        mapped = {alias.get(k, k): v for k, v in hp.items()}
        kw = dict(base)
        kw.update({k: v for k, v in mapped.items() if k in sigp})
        key = tuple(sorted((k, repr(v)) for k, v in kw.items()))
        if key in seen:
            continue
        seen.add(key)
        yield kw


def normalize_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and not all(torch.is_tensor(v) for v in obj.values()):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break
    sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd):
        # Checkpoint saved from the LegacyCompatibleBenchmarkModel wrapper.
        sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    else:
        sd = {k: v for k, v in sd.items() if not k.startswith("log_vars")}
    return sd


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except (TypeError, Exception):
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return normalize_state_dict(obj)


def build_and_load_model(tm, model_name: str, hp: Dict[str, Any], weight_path: Path, seq_len: int):
    """Try (input_channels x kwargs-variant) until strict loading succeeds."""
    sd = load_checkpoint(weight_path)
    cls = tm.MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise KeyError(f"Model '{model_name}' not in train.py registry: {sorted(tm.MODEL_REGISTRY)}")
    hp_c = clean_hp(hp)
    errors: List[str] = []
    for in_ch in IN_CH_CANDIDATES:
        for kw in kwargs_variants(cls, model_name, hp_c, in_ch, seq_len):
            try:
                model = tm.create_model(model_name, **kw)
            except Exception as exc:
                errors.append(f"ctor in_ch={in_ch} kw={sorted(kw)} -> {type(exc).__name__}: {exc}")
                continue
            try:
                model.load_state_dict(sd, strict=True)
            except Exception as exc:
                errors.append(f"load in_ch={in_ch} kw={sorted(kw)} -> {str(exc)[:160]}")
                del model
                continue
            return model, in_ch, kw
    msg = "\n  ".join(errors[-12:])
    raise RuntimeError(
        f"Could not reconstruct '{model_name}' to match checkpoint {weight_path.name}.\n"
        f"Last attempts:\n  {msg}"
    )


# ----------------------------------------------------------------------------
# Legacy data pipeline adapter
# ----------------------------------------------------------------------------

BUILDER_CANDIDATES = [
    "build_test_loader", "make_test_loader", "get_test_loader",
    "build_loaders", "make_loaders", "get_loaders", "get_dataloaders",
    "make_datasets", "build_datasets", "prepare_data", "prepare_loaders",
    "load_data", "build_data", "make_data", "create_loaders", "create_datasets",
]


def set_bnd_env(tm) -> None:
    """Export boundary-feature directories so the legacy module finds them.

    This mirrors what the coordinator does. Without these, the legacy module can
    silently fall back to bndOFF inputs (the historical bug).
    """
    cfg = getattr(tm, "CONFIG", {})
    pairs = [
        ("SWAN_BND_DIR_2019", cfg.get("bnd_dir_2019")),
        ("SWAN_BND_DIR_2020", cfg.get("bnd_dir_2020")),
        ("SWAN_STATION_ROOT", cfg.get("station_root") or cfg.get("station_csv_root")),
    ]
    for env, val in pairs:
        if val and not os.environ.get(env):
            os.environ[env] = str(val)
            print(f"[env] {env}={val}")


def probe_legacy(legacy) -> None:
    """Print public callables/classes of the legacy module for adapter mapping."""
    print(f"=== Probe of {getattr(legacy, '__file__', '?')} ===")
    items = sorted(vars(legacy).items())
    for name, obj in items:
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj):
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            mark = "  <== builder candidate" if name in BUILDER_CANDIDATES else ""
            print(f"  def {name}{sig}{mark}")
        elif inspect.isclass(obj):
            print(f"  class {name}")
    print("\nIf no builder candidate matches, rerun with:")
    print("  --adapter '<function_name>'   (a function returning/creating the loaders)")


def _call_variants(data_path: str, use_bnd: str):
    return [
        ((data_path,), {}),
        ((data_path,), {"use_bnd": use_bnd}),
        ((), {"data_path": data_path, "use_bnd": use_bnd}),
        ((), {"data_path": data_path}),
        ((data_path, use_bnd), {}),
        ((), {}),
    ]


def _is_dataset_like(obj: Any, test_len: int) -> bool:
    if isinstance(obj, (str, bytes, dict, list, tuple, set, np.ndarray)):
        return False
    if torch.is_tensor(obj) or isinstance(obj, DataLoader):
        return False
    if not (hasattr(obj, "__getitem__") and hasattr(obj, "__len__")):
        return False
    try:
        return len(obj) == test_len
    except Exception:
        return False


def _hunt(container: Any, test_len: int, hint: str, found: List[Tuple[str, Any]], depth: int = 0) -> None:
    if container is None or depth > 2:
        return
    if isinstance(container, DataLoader):
        try:
            if len(container.dataset) == test_len:
                found.append((hint, container))
        except Exception:
            pass
        return
    if _is_dataset_like(container, test_len):
        found.append((hint, container))
        return
    if isinstance(container, dict):
        for k, v in container.items():
            _hunt(v, test_len, f"{hint}.{k}", found, depth + 1)
    elif isinstance(container, (list, tuple)):
        for i, v in enumerate(container):
            _hunt(v, test_len, f"{hint}[{i}]", found, depth + 1)


def _robust_block_split_for_benchmark(legacy, wave_data_for_split: np.ndarray, seq_length: int,
                                      train_ratio: float = 0.70, val_ratio: float = 0.15,
                                      test_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Mirror the robust split cascade inside the legacy wrapper()."""
    T = wave_data_for_split.shape[0]
    N = T - int(seq_length)
    block_hours_list = [168, 96, 72, 48, 336]
    q_list = [5, 4, 3, 2]
    embargo_list = [seq_length, max(seq_length // 2, 1), 0]
    for bh in block_hours_list:
        for q in q_list:
            for emb in embargo_list:
                try:
                    idx_tr, idx_va, idx_te = legacy.make_block_stratified_split(
                        wave_data_for_split,
                        seq_length,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        block_hours=bh,
                        q=q,
                        seed=42,
                        embargo_hours=emb,
                    )
                except Exception:
                    continue
                if len(idx_tr) > 0 and len(idx_va) > 0 and len(idx_te) > 0:
                    tag = f"block(bh={bh},q={q},emb={emb})"
                    print(f"[adapter-split] {tag} -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
                    return idx_tr, idx_va, idx_te, tag

    rng = np.random.default_rng(42)
    all_idx = np.arange(0, N, dtype=int)
    rng.shuffle(all_idx)
    n_tr = max(1, int(round(N * 0.70)))
    n_va = max(1, int(round(N * 0.15)))
    n_te = N - n_tr - n_va
    if n_te <= 0:
        n_te = 1
        n_va = max(1, n_va - 1)
    idx_tr = np.sort(all_idx[:n_tr])
    idx_va = np.sort(all_idx[n_tr:n_tr + n_va])
    idx_te = np.sort(all_idx[n_tr + n_va:n_tr + n_va + n_te])
    print(f"[adapter-split] random-fallback -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
    return idx_tr, idx_va, idx_te, "random-fallback"


def _merge_seg_series_dicts_for_benchmark(*dicts):
    """Fallback compatible with legacy.merge_seg_series_dicts()."""
    import pandas as pd
    out = {}
    all_names = set().union(*[d.keys() for d in dicts if d])
    for name in all_names:
        dfs = [d[name] for d in dicts if d and name in d]
        if not dfs:
            continue
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[name] = df
    return out


def _auto_align_bnd_dir_for_benchmark(bnd_feat: np.ndarray, ds_sim, kcs2d: np.ndarray) -> Tuple[float, Dict[float, float]]:
    """Mirror the BND direction convention harmonization inside wrapper()."""
    if "dir" not in ds_sim:
        return 0.0, {}
    T = bnd_feat.shape[0]
    rad = np.deg2rad(ds_sim["dir"].values[:T])
    tsin = np.sin(rad).astype(np.float32)
    tcos = np.cos(rad).astype(np.float32)
    sin_idx, cos_idx = 2, 3
    for try_s, try_c in [(2, 3), (3, 2)]:
        try:
            smin = np.nanmin(bnd_feat[:, try_s])
            cmin = np.nanmin(bnd_feat[:, try_c])
        except Exception:
            continue
        if (smin < -0.1) and (cmin < -0.1):
            sin_idx, cos_idx = try_s, try_c
            break
    sin_b = bnd_feat[:, sin_idx]
    cos_b = bnd_feat[:, cos_idx]
    mask = np.asarray(kcs2d > 0, dtype=bool)

    def _transform(deg: float, reflect: bool):
        """Return (sin, cos) of the transformed BND direction.

        rotation  : theta' = theta + deg
        reflection: theta' = deg - theta   (Cartesian <-> nautical is the
                    reflection with deg = 270)
        """
        r = np.deg2rad(deg)
        if not reflect:
            sin_r = sin_b * np.cos(r) + cos_b * np.sin(r)
            cos_r = cos_b * np.cos(r) - sin_b * np.sin(r)
        else:
            sin_r = np.sin(r) * cos_b - np.cos(r) * sin_b
            cos_r = np.cos(r) * cos_b + np.sin(r) * sin_b
        return sin_r, cos_r

    def _score(deg: float, reflect: bool) -> float:
        sin_r, cos_r = _transform(deg, reflect)
        v = sin_r * tsin + cos_r * tcos
        return float(np.nanmean(v[:, mask]))

    # Four rotations and four reflections cover every axis-aligned convention
    # change between compass-from, compass-to, and Cartesian angles.
    candidates = [(0.0, False), (90.0, False), (-90.0, False), (180.0, False),
                  (0.0, True), (90.0, True), (180.0, True), (270.0, True)]
    scores = {c: _score(*c) for c in candidates}
    best = max(scores, key=lambda c: scores[c])
    best_deg, best_reflect = best
    if best_reflect or abs(best_deg) > 1e-6:
        sin_r, cos_r = _transform(best_deg, best_reflect)
        bnd_feat[:, sin_idx] = sin_r
        bnd_feat[:, cos_idx] = cos_r
    # Report as a signed code: reflections are encoded as 1000 + deg so that
    # downstream logging that expects a float still works.
    code = (1000.0 + best_deg) if best_reflect else float(best_deg)
    readable = {(f"{'refl ' if rf else 'rot '}{d:+.0f}"): s for (d, rf), s in scores.items()}
    return code, readable


def _build_test_loader_from_legacy_wrapper_logic(legacy, tm, job: Dict[str, Any], data_path: str,
                                                 use_bnd: str, batch: int, workers: int,
                                                 test_len: int):
    """Build the exact test split used by the legacy wrapper without training.

    This is adapted from wrapper() in UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py:
    - split from raw Hs with the robust block-stratified cascade,
    - normalization from training target times,
    - 6 base input channels from load_and_preprocess_data(),
    - optional 4 BND channels when use_bnd='on',
    - WindWaveDataset + SubsetIndicesDataset(idx_te) + safe_collate.
    """
    import pandas as pd
    import xarray as xr
    from pathlib import Path as _Path

    hp = job.get("hyperparams", {})
    seq_length = int(hp.get("seq_length", 12))
    time_steps = int(job.get("time_steps", getattr(tm, "CONFIG", {}).get("time_steps", 17498)))
    ub = str(use_bnd or job.get("use_bnd", "on")).strip().lower()
    if ub not in {"on", "off", "auto"}:
        ub = "on" if ub in {"true", "1", "yes"} else "off"

    if not Path(data_path).is_file():
        raise FileNotFoundError(f"NetCDF not found: {data_path}")

    ds_sim = xr.open_dataset(data_path)
    try:
        N = time_steps - seq_length
        hs_raw = ds_sim["hsign"].values[:time_steps]
        Y, X = hs_raw.shape[-2], hs_raw.shape[-1]
        wave_data_for_split = np.zeros((time_steps, 4, Y, X), dtype=np.float32)
        wave_data_for_split[:, 0] = hs_raw
        idx_tr, idx_va, idx_te, split_tag = _robust_block_split_for_benchmark(
            legacy, wave_data_for_split, seq_length
        )

        global_norm_params = legacy.compute_params_with_indices(ds_sim, idx_train=idx_tr, seq_length=seq_length)
        print(f"[adapter-norm] Hs range={global_norm_params['hs']}  Tm range={global_norm_params['tm']}")

        input_data, wave_data, lon, lat, kcs = legacy.load_and_preprocess_data(
            ds_sim, global_norm_params, time_steps=time_steps
        )

        if "time" in ds_sim:
            tvals = pd.to_datetime(ds_sim["time"].values[:time_steps])
            time_index = pd.DatetimeIndex(tvals).tz_localize(None)
        else:
            time_index = pd.date_range(start="2019-01-01 00:00:00", periods=time_steps, freq="h", tz="UTC").tz_localize(None)

        use_bnd_features = (ub == "on")
        if use_bnd_features:
            try:
                from bnd_features import read_all_bnds, build_owner_label, make_boundary_feature_maps, assert_on_edges
                from boundspec_segments import SEGMENTS, M as SWAN_M, N as SWAN_N
            except Exception as exc:
                raise RuntimeError(
                    "use_bnd='on' was requested but bnd_features/boundspec_segments could not be imported"
                ) from exc

            kcs2d = kcs[0] if getattr(kcs, "ndim", 2) == 3 else kcs
            H, W = kcs2d.shape
            if H == SWAN_M and W == SWAN_N:
                swap_ij = False
            elif H == SWAN_N and W == SWAN_M:
                swap_ij = True
            else:
                raise ValueError(f"Grid mismatch: data(H,W)=({H},{W}) vs SWAN(M,N)=({SWAN_M},{SWAN_N})")
            assert_on_edges(SEGMENTS, M=SWAN_M, N=SWAN_N)

            cfg = getattr(tm, "CONFIG", {})
            bnd_dirs_by_year = {
                2019: os.environ.get("SWAN_BND_DIR_2019", str(cfg.get("bnd_dir_2019", ""))),
                2020: os.environ.get("SWAN_BND_DIR_2020", str(cfg.get("bnd_dir_2020", ""))),
            }
            years_needed = sorted(set(pd.DatetimeIndex(time_index).year))
            seg_dicts = []
            for y in years_needed:
                bdir = bnd_dirs_by_year.get(y)
                if bdir and os.path.isdir(bdir):
                    seg_y = read_all_bnds(_Path(bdir), direction="from")
                    seg_dicts.append(seg_y)
                    print(f"[adapter-BND] {y}: loaded .BND from {bdir}")
                else:
                    print(f"[adapter-BND][WARN] BND folder missing for {y}: {bdir}")
            if not seg_dicts:
                raise FileNotFoundError(f"No .BND files found for years {years_needed}")

            merge_fn = getattr(legacy, "merge_seg_series_dicts", _merge_seg_series_dicts_for_benchmark)
            seg_series = merge_fn(*seg_dicts)
            owner_label, id2name = build_owner_label(
                H, W, segments=SEGMENTS, exact_M=SWAN_M, exact_N=SWAN_N,
                kcs=kcs2d, swap_ij=swap_ij,
            )
            bnd_feat = make_boundary_feature_maps(
                time_index=time_index,
                owner_label=owner_label,
                seg_series=seg_series,
                id2name=id2name,
                kcs=kcs2d,
                norm_hs=global_norm_params["hs"],
                norm_tm=global_norm_params["tm"],
            )
            if bnd_feat.shape[0] != input_data.shape[0]:
                raise ValueError(f"BND T={bnd_feat.shape[0]} vs input T={input_data.shape[0]}")
            if bnd_feat.shape[2:] != input_data.shape[2:]:
                raise ValueError(f"BND HW={bnd_feat.shape[2:]} vs input HW={input_data.shape[2:]}")
            best_deg, scores = _auto_align_bnd_dir_for_benchmark(bnd_feat, ds_sim, kcs2d)
            if scores:
                msg = " ".join([f"{k}:{v:.4f}" for k, v in scores.items()])
                chosen = (f"reflection theta'={best_deg-1000:.0f}-theta" if best_deg >= 1000
                          else f"rotation {best_deg:+.0f}°")
                print(f"[adapter-BND] dir autocorrect -> chosen {chosen} | scores {msg}")
            input_data = np.concatenate([input_data, bnd_feat], axis=1)
            print(f"[adapter-BND] appended 4 BND channels -> input_channels={input_data.shape[1]}")

        base_ds = legacy.WindWaveDataset(input_data, wave_data, seq_length, 0, N)
        test_ds = legacy.SubsetIndicesDataset(base_ds, idx_te)
        if len(test_ds) != test_len:
            print(f"[adapter][WARN] test dataset length {len(test_ds)} != expected fingerprint {test_len}")
        loader = DataLoader(
            test_ds,
            batch_size=max(1, min(int(batch), len(test_ds))),
            shuffle=False,
            num_workers=int(workers),
            collate_fn=getattr(legacy, "safe_collate", None),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        kcs2d = kcs[0] if getattr(kcs, "ndim", 2) == 3 else kcs
        mask = np.asarray(kcs2d) > 0

        # Stash for debugging/figures-only inspection if desired.
        setattr(legacy, "global_norm_params", global_norm_params)
        setattr(legacy, "kcs_map", kcs2d)
        setattr(legacy, "lon_map", lon)
        setattr(legacy, "lat_map", lat)
        setattr(legacy, "idx_te", idx_te)
        setattr(legacy, "split_tag", split_tag)

        hint = f"direct_legacy_wrapper_logic:{split_tag}:bnd={ub}:n={len(test_ds)}"
        return loader, global_norm_params, mask, hint
    finally:
        try:
            ds_sim.close()
        except Exception:
            pass


def build_test_loader(legacy, tm, job: Dict[str, Any], data_path: str, use_bnd: str,
                      batch: int, workers: int, test_len: int, adapter: str):
    """Build the legacy test loader used by the trained checkpoints.

    The original legacy script only exposes wrapper(), so the default path below
    reconstructs the test split directly from wrapper()'s data-loading logic.
    If --adapter is explicitly supplied, the older auto-discovery fallback is
    still available for future variants.
    """
    # Align legacy globals with this job (paths, seq length, BND dirs, etc.).
    for fn_name, args in (("patch_legacy_globals", (legacy, job, 0)),
                          ("patch_legacy_external_paths", (legacy,))):
        fn = getattr(tm, fn_name, None)
        if callable(fn):
            try:
                fn(*args)
            except TypeError:
                try:
                    fn(legacy)
                except Exception as exc:
                    print(f"[adapter] {fn_name} skipped: {exc}")
            except Exception as exc:
                print(f"[adapter] {fn_name} skipped: {exc}")

    if not adapter:
        return _build_test_loader_from_legacy_wrapper_logic(
            legacy, tm, job, data_path, use_bnd, batch, workers, test_len
        )

    # Optional legacy-discovery fallback when user passes --adapter.
    found: List[Tuple[str, Any]] = []
    fn = getattr(legacy, adapter, None)
    if not callable(fn):
        raise RuntimeError(f"--adapter '{adapter}' is not a callable in the legacy module")
    for args, kwargs in _call_variants(data_path, use_bnd):
        try:
            res = fn(*args, **kwargs)
        except Exception:
            continue
        _hunt(res, test_len, adapter, found)
        if found:
            break
    if not found:
        raise RuntimeError(f"Adapter '{adapter}' did not return a dataset/DataLoader of length {test_len}.")

    test_named = [f for f in found if "test" in f[0].lower()]
    hint, chosen = (test_named[-1] if test_named else found[-1])
    if isinstance(chosen, DataLoader):
        ds, collate = chosen.dataset, chosen.collate_fn
    else:
        ds, collate = chosen, None
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=torch.cuda.is_available())
    norm = getattr(legacy, "global_norm_params", dict(FALLBACK_NORM))
    mask = None
    for name in ("kcs_map", "kcs", "KCS", "ocean_mask", "mask", "land_mask"):
        cand = getattr(legacy, name, None)
        if cand is None:
            continue
        arr = np.asarray(cand)
        if arr.ndim == 2:
            mask = arr > 0
            break
    return loader, norm, mask, hint


# ----------------------------------------------------------------------------
# Metrics machinery
# ----------------------------------------------------------------------------

def vec2deg(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Direction from sin/cos. Absolute convention cancels in pred-true diffs."""
    return (np.degrees(np.arctan2(s, c))) % 360.0


def circ_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


class PooledMoments:
    __slots__ = ("n", "sp", "st", "spp", "stt", "spt", "sd", "sad", "sdd")

    def __init__(self) -> None:
        self.n = self.sp = self.st = self.spp = self.stt = self.spt = 0.0
        self.sd = self.sad = self.sdd = 0.0

    def update(self, p: np.ndarray, t: np.ndarray) -> None:
        d = p - t
        self.n += p.size
        self.sp += float(p.sum()); self.st += float(t.sum())
        self.spp += float((p * p).sum()); self.stt += float((t * t).sum())
        self.spt += float((p * t).sum())
        self.sd += float(d.sum()); self.sad += float(np.abs(d).sum())
        self.sdd += float((d * d).sum())

    def finalize(self) -> Dict[str, float]:
        nan = float("nan")
        if self.n == 0:
            return {"rmse": nan, "mae": nan, "bias": nan, "r": nan, "r2": nan, "n": 0.0}
        n = self.n
        vp = self.spp - self.sp ** 2 / n
        vt = self.stt - self.st ** 2 / n
        cov = self.spt - self.sp * self.st / n
        r = cov / math.sqrt(vp * vt) if vp > 0 and vt > 0 else nan
        r2 = 1.0 - self.sdd / vt if vt > 0 else nan
        return {"rmse": math.sqrt(self.sdd / n), "mae": self.sad / n,
                "bias": self.sd / n, "r": r, "r2": r2, "n": n}


class RunAccumulator:
    """Per-run accumulation of stepwise, pooled, per-pixel, and binned errors."""

    def __init__(self, grid_shape: Tuple[int, int], base_mask: np.ndarray,
                 norm: Dict[str, Tuple[float, float]], keep_hex: bool = False,
                 hex_cap: int = 2_500_000) -> None:
        H, W = grid_shape
        self.base_mask = base_mask
        self.norm = norm
        self.step = {k: [] for k in ("hs_rmse", "hs_mae", "tm_rmse", "tm_mae",
                                     "dir_crmse", "dir_cmae")}
        self.pool = {"hs": PooledMoments(), "tm": PooledMoments()}
        self.dir_sums = dict(n=0.0, sd2=0.0, sad=0.0, ssin=0.0, scos=0.0)
        self.dirhi_sums = dict(n=0.0, sd2=0.0, sad=0.0)
        self.sse = {v: np.zeros((H, W)) for v in ("hs", "tm", "dir")}
        self.cnt = {v: np.zeros((H, W)) for v in ("hs", "tm", "dir")}
        nb = len(SEASTATE_EDGES) - 1
        self.bins_sse = {v: np.zeros(nb) for v in ("hs", "tm", "dir")}
        self.bins_n = {v: np.zeros(nb) for v in ("hs", "tm", "dir")}
        self.keep_hex = keep_hex
        self.hex_cap = hex_cap
        self.hex_p: List[np.ndarray] = []
        self.hex_t: List[np.ndarray] = []
        self.hex_n = 0
        self.n_steps = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        """pred/true: (4, H, W) in normalized space, channel order [Hs, Tm, sin, cos]."""
        hs_lo, hs_hi = float(self.norm["hs"][0]), float(self.norm["hs"][1])
        tm_lo, tm_hi = float(self.norm["tm"][0]), float(self.norm["tm"][1])
        p_hs = pred[0] * (hs_hi - hs_lo) + hs_lo
        t_hs = true[0] * (hs_hi - hs_lo) + hs_lo
        p_tm = pred[1] * (tm_hi - tm_lo) + tm_lo
        t_tm = true[1] * (tm_hi - tm_lo) + tm_lo
        p_dir = vec2deg(pred[2], pred[3])
        t_dir = vec2deg(true[2], true[3])

        bm = self.base_mask
        m_hs = bm & np.isfinite(p_hs) & np.isfinite(t_hs)
        m_tm = bm & np.isfinite(p_tm) & np.isfinite(t_tm)
        m_dir = (bm & np.isfinite(pred[2]) & np.isfinite(pred[3])
                 & np.isfinite(true[2]) & np.isfinite(true[3]))

        if m_hs.any():
            d = p_hs[m_hs] - t_hs[m_hs]
            self.step["hs_rmse"].append(float(np.sqrt((d * d).mean())))
            self.step["hs_mae"].append(float(np.abs(d).mean()))
            self.pool["hs"].update(p_hs[m_hs], t_hs[m_hs])
            self.sse["hs"][m_hs] += d * d
            self.cnt["hs"][m_hs] += 1
            idx = np.clip(np.digitize(t_hs[m_hs], SEASTATE_EDGES) - 1, 0, len(SEASTATE_EDGES) - 2)
            np.add.at(self.bins_sse["hs"], idx, d * d)
            np.add.at(self.bins_n["hs"], idx, 1)
            if self.keep_hex and self.hex_n < self.hex_cap and self.n_steps % 4 == 0:
                sub_p = p_hs[m_hs][::13].astype(np.float32)
                sub_t = t_hs[m_hs][::13].astype(np.float32)
                self.hex_p.append(sub_p)
                self.hex_t.append(sub_t)
                self.hex_n += sub_p.size

        if m_tm.any():
            d = p_tm[m_tm] - t_tm[m_tm]
            self.step["tm_rmse"].append(float(np.sqrt((d * d).mean())))
            self.step["tm_mae"].append(float(np.abs(d).mean()))
            self.pool["tm"].update(p_tm[m_tm], t_tm[m_tm])
            self.sse["tm"][m_tm] += d * d
            self.cnt["tm"][m_tm] += 1
            idx = np.clip(np.digitize(t_hs[m_tm], SEASTATE_EDGES) - 1, 0, len(SEASTATE_EDGES) - 2)
            np.add.at(self.bins_sse["tm"], idx, d * d)
            np.add.at(self.bins_n["tm"], idx, 1)

        if m_dir.any():
            dd = circ_diff_deg(p_dir[m_dir], t_dir[m_dir])
            self.step["dir_crmse"].append(float(np.sqrt((dd * dd).mean())))
            self.step["dir_cmae"].append(float(np.abs(dd).mean()))
            s = self.dir_sums
            s["n"] += dd.size
            s["sd2"] += float((dd * dd).sum())
            s["sad"] += float(np.abs(dd).sum())
            rad = np.radians(dd)
            s["ssin"] += float(np.sin(rad).sum())
            s["scos"] += float(np.cos(rad).sum())
            self.sse["dir"][m_dir] += dd * dd
            self.cnt["dir"][m_dir] += 1
            idx = np.clip(np.digitize(t_hs[m_dir], SEASTATE_EDGES) - 1, 0, len(SEASTATE_EDGES) - 2)
            np.add.at(self.bins_sse["dir"], idx, dd * dd)
            np.add.at(self.bins_n["dir"], idx, 1)
            hi = m_dir & (t_hs >= HS_MIN_FOR_DIR)
            if hi.any():
                ddh = circ_diff_deg(p_dir[hi], t_dir[hi])
                h = self.dirhi_sums
                h["n"] += ddh.size
                h["sd2"] += float((ddh * ddh).sum())
                h["sad"] += float(np.abs(ddh).sum())

        self.n_steps += 1

    def finalize(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        nan = float("nan")
        row: Dict[str, float] = {"n_steps": float(self.n_steps)}

        def stepmean(key: str) -> float:
            return float(np.mean(self.step[key])) if self.step[key] else nan

        row["hs_rmse_stepmean"] = stepmean("hs_rmse")
        row["hs_mae_stepmean"] = stepmean("hs_mae")
        row["tm_rmse_stepmean"] = stepmean("tm_rmse")
        row["tm_mae_stepmean"] = stepmean("tm_mae")
        row["dir_crmse_stepmean"] = stepmean("dir_crmse")
        row["dir_cmae_stepmean"] = stepmean("dir_cmae")

        for var in ("hs", "tm"):
            p = self.pool[var].finalize()
            for k, v in p.items():
                row[f"{var}_{k}_pooled"] = float(v)

        s = self.dir_sums
        if s["n"] > 0:
            row["dir_crmse_pooled"] = math.sqrt(s["sd2"] / s["n"])
            row["dir_cmae_pooled"] = s["sad"] / s["n"]
            row["dir_cbias_pooled"] = math.degrees(math.atan2(s["ssin"] / s["n"], s["scos"] / s["n"]))
        else:
            row["dir_crmse_pooled"] = row["dir_cmae_pooled"] = row["dir_cbias_pooled"] = nan
        h = self.dirhi_sums
        row["dir_crmse_pooled_hs05"] = math.sqrt(h["sd2"] / h["n"]) if h["n"] > 0 else nan
        row["dir_cmae_pooled_hs05"] = h["sad"] / h["n"] if h["n"] > 0 else nan

        arrays = {}
        for v in ("hs", "tm", "dir"):
            arrays[f"sse_{v}"] = self.sse[v].astype(np.float32)
            arrays[f"cnt_{v}"] = self.cnt[v].astype(np.float32)
            arrays[f"bins_sse_{v}"] = self.bins_sse[v]
            arrays[f"bins_n_{v}"] = self.bins_n[v]
        if self.keep_hex and self.hex_p:
            arrays["hex_p"] = np.concatenate(self.hex_p)
            arrays["hex_t"] = np.concatenate(self.hex_t)
        return row, arrays


def _split_batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        for kx, ky in (("x", "y"), ("input", "target"), ("inputs", "targets"), ("X", "Y")):
            if kx in batch and ky in batch:
                return batch[kx], batch[ky]
    raise RuntimeError(f"Unrecognized batch structure: {type(batch)}; adapt _split_batch().")


def evaluate_run(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                 norm: Dict[str, Tuple[float, float]], base_mask: Optional[np.ndarray],
                 limit_steps: int, keep_hex: bool) -> RunAccumulator:
    model = model.to(device).eval()
    acc: Optional[RunAccumulator] = None
    steps = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _split_batch(batch)
            x = x.to(device, non_blocking=True).float()
            out = model(x)
            pred_t = out[0] if isinstance(out, (list, tuple)) else out
            pred = pred_t.float().cpu().numpy().astype(np.float64)
            true = (y.float().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)).astype(np.float64)
            if acc is None:
                H, W = true.shape[-2], true.shape[-1]
                bm = base_mask if base_mask is not None else np.ones((H, W), dtype=bool)
                if bm.shape != (H, W):
                    print(f"[warn] mask shape {bm.shape} != grid {(H, W)}; ignoring mask")
                    bm = np.ones((H, W), dtype=bool)
                acc = RunAccumulator((H, W), bm, norm, keep_hex=keep_hex)
            for b in range(true.shape[0]):
                acc.update(pred[b], true[b])
                steps += 1
                if limit_steps and steps >= limit_steps:
                    return acc
    if acc is None:
        raise RuntimeError("Empty test loader; nothing evaluated.")
    return acc


# ----------------------------------------------------------------------------
# Statistics: paired t-tests with Holm correction
# ----------------------------------------------------------------------------

def t_pvalue_two_sided(t: float, df: int) -> float:
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        if df == 2:
            # Closed form for Student-t with df=2: two-sided p = 1 - |t|/sqrt(2+t^2)
            return float(1.0 - abs(t) / math.sqrt(2.0 + t * t))
        return float(math.erfc(abs(t) / math.sqrt(2.0)))  # normal approximation


def pairwise_holm(pivot, metric_label: str):
    """pivot: DataFrame indexed by model, columns = seeds, values = metric."""
    import pandas as pd
    from itertools import combinations
    models = sorted(pivot.index)
    rows = []
    for a, b in combinations(models, 2):
        common = pivot.columns[pivot.loc[a].notna() & pivot.loc[b].notna()]
        if len(common) < 2:
            continue
        d = pivot.loc[a, common].values - pivot.loc[b, common].values
        n = len(d)
        mean_d, sd_d = float(np.mean(d)), float(np.std(d, ddof=1))
        t = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else float("inf")
        p = t_pvalue_two_sided(t, n - 1)
        rows.append({"metric": metric_label, "A": a, "B": b, "n": n,
                     "mean_diff": mean_d, "sd_diff": sd_d, "t": t, "p": p})
    if not rows:
        return pd.DataFrame()
    res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(res)
    running = 0.0
    adj = np.empty(m)
    for rank, p in enumerate(res["p"].values):
        running = max(running, (m - rank) * p)
        adj[rank] = min(1.0, running)
    res["p_holm"] = adj
    res["distinct_holm"] = res["p_holm"] < 0.05
    return res


# ----------------------------------------------------------------------------
# Aggregation, tables, figures
# ----------------------------------------------------------------------------

SUMMARY_METRICS = ["hs_rmse_stepmean", "hs_mae_stepmean", "tm_rmse_stepmean",
                   "tm_mae_stepmean", "dir_crmse_stepmean", "dir_cmae_stepmean",
                   "hs_r2_pooled", "latency_ms", "params_M"]

PAIRWISE_VARS = [("hs_rmse_stepmean", "Hs RMSE (m)"),
                 ("tm_rmse_stepmean", "Tm RMSE (s)"),
                 ("dir_crmse_stepmean", "Dir cRMSE (deg)")]

BIN_LABELS = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4+"]


def model_colors(models: List[str]) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(models)}


def rmse_map(npz, var: str) -> np.ndarray:
    sse, cnt = npz[f"sse_{var}"], npz[f"cnt_{var}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.sqrt(sse / cnt)
    return np.ma.masked_invalid(np.where(cnt > 0, m, np.nan))


def make_figures(out_dir: Path, df, holm_frames, seed42_npz: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ms = df[df["stage"] == "stage2_multiseed"]
    if ms.empty:
        print("[fig] no multiseed rows; figures skipped")
        return
    order = ms.groupby("model")["hs_rmse_stepmean"].mean().sort_values().index.tolist()
    colors = model_colors(order)

    # 1) Accuracy bars: Hs / Tm / Dir with seed error bars
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        specs = [("hs_rmse_stepmean", "Hs RMSE (m)"), ("tm_rmse_stepmean", "Tm RMSE (s)"),
                 ("dir_crmse_stepmean", "Dir cRMSE (deg)")]
        for ax, (col, label) in zip(axes, specs):
            g = ms.groupby("model")[col].agg(["mean", "std"]).reindex(order)
            ax.bar(range(len(order)), g["mean"], yerr=g["std"], capsize=3,
                   color=[colors[m] for m in order])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Test-set accuracy by architecture (mean +/- sd over seeds)")
        fig.tight_layout()
        fig.savefig(figs / "fig_accuracy_bars.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[fig] accuracy bars failed: {exc}")

    # 2) Pareto: accuracy vs latency and vs parameters
    try:
        agg = ms.groupby("model").agg(hs=("hs_rmse_stepmean", "mean"),
                                      lat=("latency_ms", "mean"),
                                      par=("params_M", "mean")).reindex(order)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, xcol, xlabel in ((axes[0], "lat", "Inference latency (ms)"),
                                 (axes[1], "par", "Parameters (M)")):
            for m in order:
                ax.scatter(agg.loc[m, xcol], agg.loc[m, "hs"], s=60, color=colors[m], label=m)
                ax.annotate(m, (agg.loc[m, xcol], agg.loc[m, "hs"]), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Hs RMSE (m)")
            ax.grid(alpha=0.3, which="both")
        fig.suptitle("Accuracy-efficiency trade-off")
        fig.tight_layout()
        fig.savefig(figs / "fig_pareto.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[fig] pareto failed: {exc}")

    # 3) Boundary ablation: bndON vs bndOFF (seed 42)
    try:
        on = df[(df["stage"] == "stage2_multiseed") & (df["seed"] == 42)].set_index("model")
        off = df[df["stage"] == "stage2_bndoff"].set_index("model")
        common = [m for m in order if m in on.index and m in off.index]
        if common:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for ax, (col, label) in zip(axes, PAIRWISE_VARS):
                x = np.arange(len(common))
                ax.bar(x - 0.2, on.loc[common, col], width=0.4, label="bnd ON")
                ax.bar(x + 0.2, off.loc[common, col], width=0.4, label="bnd OFF")
                ax.set_xticks(x)
                ax.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=0.3)
            axes[0].legend()
            fig.suptitle("Boundary-feature ablation (seed 42, best config)")
            fig.tight_layout()
            fig.savefig(figs / "fig_ablation.png", dpi=200)
            plt.close(fig)
    except Exception as exc:
        print(f"[fig] ablation failed: {exc}")

    # 4) Spatial RMSE maps per variable (seed 42 multiseed runs)
    for var, label in (("hs", "Hs RMSE (m)"), ("tm", "Tm RMSE (s)"), ("dir", "Dir cRMSE (deg)")):
        try:
            avail = [m for m in order if m in seed42_npz]
            if not avail:
                break
            maps = {m: rmse_map(seed42_npz[m], var) for m in avail}
            vmax = np.nanpercentile(np.concatenate([np.ma.filled(v, np.nan).ravel() for v in maps.values()]), 99)
            ncol = 5
            nrow = int(math.ceil(len(avail) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow))
            axes = np.atleast_2d(axes)
            for i, m in enumerate(avail):
                ax = axes[i // ncol, i % ncol]
                im = ax.imshow(maps[m], origin="lower", vmin=0, vmax=vmax, cmap="viridis")
                ax.set_title(m, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            for j in range(len(avail), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=label)
            fig.suptitle(f"Per-pixel {label} over test set (seed 42)")
            fig.savefig(figs / f"fig_spatial_rmse_{var}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[fig] spatial {var} failed: {exc}")

    # 5) Spatial Hs RMSE difference vs the best model
    try:
        avail = [m for m in order if m in seed42_npz]
        if len(avail) >= 2:
            best = avail[0]
            base = rmse_map(seed42_npz[best], "hs")
            others = avail[1:]
            ncol = 3
            nrow = int(math.ceil(len(others) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
            axes = np.atleast_2d(axes)
            diffs = {m: rmse_map(seed42_npz[m], "hs") - base for m in others}
            lim = np.nanpercentile(np.abs(np.concatenate(
                [np.ma.filled(v, np.nan).ravel() for v in diffs.values()])), 98)
            for i, m in enumerate(others):
                ax = axes[i // ncol, i % ncol]
                im = ax.imshow(diffs[m], origin="lower", vmin=-lim, vmax=lim, cmap="RdBu_r")
                ax.set_title(f"{m} - {best}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            for j in range(len(others), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Delta Hs RMSE (m)")
            fig.suptitle(f"Spatial Hs RMSE difference relative to {best} (seed 42)")
            fig.savefig(figs / "fig_spatial_diff_hs.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"[fig] spatial diff failed: {exc}")

    # 6) Error vs sea state
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        centers = np.arange(len(BIN_LABELS))
        for ax, (var, label) in zip(axes, (("hs", "Hs RMSE (m)"), ("tm", "Tm RMSE (s)"),
                                           ("dir", "Dir cRMSE (deg)"))):
            for m in order:
                if m not in seed42_npz:
                    continue
                z = seed42_npz[m]
                sse, n = z[f"bins_sse_{var}"], z[f"bins_n_{var}"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.sqrt(sse / n)
                ax.plot(centers, r, marker="o", ms=3, lw=1.2, color=colors[m], label=m)
            ax.set_xticks(centers)
            ax.set_xticklabels(BIN_LABELS, fontsize=8)
            ax.set_xlabel("True Hs bin (m)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle("Error by sea state (seed 42)")
        fig.tight_layout()
        fig.savefig(figs / "fig_error_vs_seastate.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[fig] seastate failed: {exc}")

    # 7) Pairwise Holm-corrected significance heatmaps
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (col, label) in zip(axes, PAIRWISE_VARS):
            res = holm_frames.get(col)
            if res is None or res.empty:
                ax.axis("off")
                continue
            mat = pd.DataFrame(np.nan, index=order, columns=order)
            for _, r in res.iterrows():
                if r["A"] in order and r["B"] in order:
                    mat.loc[r["A"], r["B"]] = r["p_holm"]
                    mat.loc[r["B"], r["A"]] = r["p_holm"]
            im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="RdYlGn")
            ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
            ax.set_xticklabels(order, rotation=90, fontsize=7)
            ax.set_yticklabels(order, fontsize=7)
            for i in range(len(order)):
                for j in range(len(order)):
                    v = mat.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
            ax.set_title(f"Holm-adjusted p: {label}", fontsize=9)
        fig.tight_layout()
        fig.savefig(figs / "fig_pairwise_significance.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[fig] significance heatmap failed: {exc}")

    # 8) Pred-vs-true Hs density (hexbin)
    try:
        avail = [m for m in order if m in seed42_npz and "hex_p" in seed42_npz[m]]
        if avail:
            ncol = 5
            nrow = int(math.ceil(len(avail) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
            axes = np.atleast_2d(axes)
            hi = max(float(np.nanmax(seed42_npz[m]["hex_t"])) for m in avail)
            for i, m in enumerate(avail):
                ax = axes[i // ncol, i % ncol]
                z = seed42_npz[m]
                ax.hexbin(z["hex_t"], z["hex_p"], gridsize=60, bins="log",
                          extent=(0, hi, 0, hi), cmap="magma")
                ax.plot([0, hi], [0, hi], "w--", lw=0.8)
                ax.set_title(m, fontsize=9)
                ax.set_xlabel("SWAN Hs (m)", fontsize=7)
                ax.set_ylabel("Emulated Hs (m)", fontsize=7)
            for j in range(len(avail), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            fig.suptitle("Predicted vs. reference Hs density (seed 42, subsampled)")
            fig.tight_layout()
            fig.savefig(figs / "fig_scatter_density_hs.png", dpi=200)
            plt.close(fig)
    except Exception as exc:
        print(f"[fig] hexbin failed: {exc}")

    print(f"[fig] figures written to {figs}")


def aggregate_and_report(out_dir: Path, rows: List[Dict[str, Any]],
                         seed42_npz: Dict[str, Any], make_figs: bool) -> None:
    import pandas as pd
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables / "per_run_metrics.csv", index=False)

    ms = df[df["stage"] == "stage2_multiseed"]
    if not ms.empty:
        cols = [c for c in SUMMARY_METRICS if c in ms.columns]
        summary = ms.groupby("model")[cols].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        if "hs_rmse_stepmean_mean" in summary.columns:
            summary = summary.sort_values("hs_rmse_stepmean_mean")
        summary.to_csv(tables / "per_model_summary.csv")
        print("\n=== Per-model summary (multiseed) ===")
        show = [c for c in ("hs_rmse_stepmean_mean", "tm_rmse_stepmean_mean",
                            "dir_crmse_stepmean_mean") if c in summary.columns]
        print(summary[show].round(4).to_string())

    holm_frames: Dict[str, Any] = {}
    for col, label in PAIRWISE_VARS:
        if col not in ms.columns or ms.empty:
            continue
        pivot = ms.pivot_table(index="model", columns="seed", values=col)
        res = pairwise_holm(pivot, label)
        holm_frames[col] = res
        if not res.empty:
            res.to_csv(tables / f"pairwise_holm_{col}.csv", index=False)
            nd = int(res["distinct_holm"].sum())
            print(f"[stats] {label}: {nd}/{len(res)} pairs distinct after Holm")

    # Boundary ablation table (paired at seed 42)
    on = df[(df["stage"] == "stage2_multiseed") & (df["seed"] == 42)].set_index("model")
    off = df[df["stage"] == "stage2_bndoff"].set_index("model")
    common = sorted(set(on.index) & set(off.index))
    if common:
        abl_rows = []
        for m in common:
            r = {"model": m}
            for col, _ in PAIRWISE_VARS:
                r[f"{col}_on"] = float(on.loc[m, col])
                r[f"{col}_off"] = float(off.loc[m, col])
                r[f"{col}_ratio"] = float(off.loc[m, col] / on.loc[m, col])
            abl_rows.append(r)
        pd.DataFrame(abl_rows).to_csv(tables / "ablation_boundary.csv", index=False)

    # Reproduction check table
    rep_cols = ["run_name", "model", "stage", "seed", "hs_rmse_stepmean",
                "ref_hs_rmse", "hs_rel_diff", "validation"]
    rep = df[[c for c in rep_cols if c in df.columns]]
    rep.to_csv(tables / "reproduction_check.csv", index=False)
    bad = df[df["validation"].isin(["WARN", "FAIL"])] if "validation" in df.columns else df.iloc[0:0]
    if len(bad):
        print(f"\n[VALIDATION][WARN] {len(bad)} runs deviate from stored Hs RMSE; "
              f"inspect {tables/'reproduction_check.csv'} before using any numbers.")
    else:
        print("\n[VALIDATION] all evaluated runs reproduce the stored Hs RMSE.")

    if make_figs:
        make_figures(out_dir, df, holm_frames, seed42_npz)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-py", default="/home/jovyan/swan/train.py")
    ap.add_argument("--legacy-py", default="")
    ap.add_argument("--data-path", default="")
    ap.add_argument("--results-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--models", default="", help="comma-separated filter, e.g. ffno,fno")
    ap.add_argument("--stages", default="stage2_multiseed,stage2_bndoff")
    ap.add_argument("--limit-steps", type=int, default=0, help="smoke test: cap test samples")
    ap.add_argument("--test-len", type=int, default=TEST_LEN_DEFAULT)
    ap.add_argument("--adapter", default="", help="legacy function name that builds the loaders")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if per-run npz exists")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--reconstruct-only", action="store_true",
                    help="Only strict-load checkpoints and stop before building the data loader/evaluation.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    train_py = Path(args.train_py).resolve()
    assert_main_guard(train_py)
    tm = load_module_from_path("train_module_for_inference", train_py)
    cfg = getattr(tm, "CONFIG", {})

    legacy_py = Path(args.legacy_py or cfg.get("legacy_train_script", "")).resolve()
    data_path = str(args.data_path or cfg.get("data_path", ""))
    results_root = Path(args.results_root or cfg.get("results_root", "")).resolve()
    out_dir = Path(args.out) if args.out else results_root / "_inference_metrics"
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    set_bnd_env(tm)
    if not legacy_py.is_file():
        print(f"[FATAL] legacy module not found: {legacy_py}")
        return 1
    legacy = load_module_from_path("legacy_module_for_inference", legacy_py)

    if args.probe:
        probe_legacy(legacy)
        return 0

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    runs = discover_runs(results_root, stages, models)
    print(f"Discovered {len(runs)} runs under {results_root} (stages={stages})")
    if not runs:
        return 1

    seed42_npz: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    if args.figures_only:
        for run in runs:
            npz_path = per_run_dir / f"{run['run_name']}.npz"
            if not npz_path.exists():
                continue
            z = np.load(npz_path, allow_pickle=False)
            rows.append(json.loads(str(z["row"])))
            if run["stage"] == "stage2_multiseed" and run["seed"] == 42:
                seed42_npz[run["model"]] = z
        if not rows:
            print("[FATAL] --figures-only but no per-run npz found; run inference first.")
            return 1
        aggregate_and_report(out_dir, rows, seed42_npz, make_figs=not args.no_figures)
        return 0

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    loader_cache: Dict[str, Tuple[DataLoader, Dict, Optional[np.ndarray], str]] = {}

    for run in runs:
        npz_path = per_run_dir / f"{run['run_name']}.npz"
        if npz_path.exists() and not args.force:
            z = np.load(npz_path, allow_pickle=False)
            rows.append(json.loads(str(z["row"])))
            if run["stage"] == "stage2_multiseed" and run["seed"] == 42:
                seed42_npz[run["model"]] = z
            print(f"[skip] {run['run_name']} (cached)")
            continue

        print(f"\n=== {run['run_name']} ===")
        hp = run["job"].get("hyperparams", {})
        seq_len = int(hp.get("seq_length", 12))
        try:
            model, in_ch, kw = build_and_load_model(tm, run["model"], hp, run["weight_path"], seq_len)
        except Exception as exc:
            print(f"[FAIL] model reconstruction: {exc}")
            continue
        print(f"[model] {run['model']} rebuilt with input_channels={in_ch}")
        if args.reconstruct_only:
            row = {
                "run_name": run["run_name"],
                "model": run["model"],
                "stage": run["stage"],
                "seed": run["seed"],
                "config_id": run["config_id"],
                "use_bnd": run["use_bnd"],
                "in_channels": in_ch,
                "validation": "RECONSTRUCT-OK",
            }
            row.update(summary_efficiency(run["summary"]))
            rows.append(row)
            del model
            continue

        ub = run["use_bnd"]
        if ub not in loader_cache:
            try:
                loader_cache[ub] = build_test_loader(
                    legacy, tm, run["job"], data_path, ub,
                    args.batch, args.workers, args.test_len, args.adapter)
                print(f"[adapter] test loader for use_bnd={ub}: source='{loader_cache[ub][3]}', "
                      f"n={len(loader_cache[ub][0].dataset)}")
            except Exception as exc:
                print(f"[FAIL] test loader (use_bnd={ub}): {exc}")
                continue
        loader, norm, mask, _ = loader_cache[ub]

        keep_hex = run["stage"] == "stage2_multiseed" and run["seed"] == 42
        try:
            acc = evaluate_run(model, loader, device, norm, mask, args.limit_steps, keep_hex)
        except Exception:
            print(f"[FAIL] evaluation:\n{traceback.format_exc()}")
            continue
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row, arrays = acc.finalize()
        row.update({"run_name": run["run_name"], "model": run["model"],
                    "stage": run["stage"], "seed": run["seed"],
                    "config_id": run["config_id"], "use_bnd": ub,
                    "in_channels": in_ch})
        row.update(summary_efficiency(run["summary"]))

        ref = summary_reference_hs_rmse(run["summary"])
        row["ref_hs_rmse"] = float("nan") if ref is None else ref
        if args.limit_steps:
            row["hs_rel_diff"], row["validation"] = float("nan"), "SKIPPED(limit)"
        elif ref is None or not np.isfinite(row["hs_rmse_stepmean"]):
            row["hs_rel_diff"], row["validation"] = float("nan"), "NO-REF"
        else:
            rel = abs(row["hs_rmse_stepmean"] - ref) / ref
            row["hs_rel_diff"] = rel
            row["validation"] = "OK" if rel < 0.02 else ("WARN" if rel < 0.10 else "FAIL")
        print(f"[check] Hs RMSE ours={row['hs_rmse_stepmean']:.6f} "
              f"ref={row['ref_hs_rmse']:.6f} -> {row['validation']}"
              if np.isfinite(row["ref_hs_rmse"]) else "[check] no reference Hs RMSE")
        print(f"[metrics] Tm RMSE={row['tm_rmse_stepmean']:.4f} s, "
              f"Dir cRMSE={row['dir_crmse_stepmean']:.2f} deg "
              f"(Hs>=0.5m: {row['dir_crmse_pooled_hs05']:.2f} deg)")

        save_kwargs = {k: v for k, v in arrays.items()}
        np.savez_compressed(npz_path, row=json.dumps(row), **save_kwargs)
        rows.append(row)
        if keep_hex:
            seed42_npz[run["model"]] = np.load(npz_path, allow_pickle=False)

    if args.reconstruct_only:
        # Reconstruct-only intentionally produces no physical metrics.
        # Save a simple table and exit before the metric aggregator.
        try:
            import pandas as pd
            tables = out_dir / "tables"
            tables.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(tables / "reconstruction_check.csv", index=False)
            print(f"[RECONSTRUCT] {len(rows)} runs rebuilt successfully. Saved -> {tables / 'reconstruction_check.csv'}")
        except Exception as exc:
            print(f"[RECONSTRUCT] completed, but failed to write CSV: {exc}")
        return 0

    if rows:
        aggregate_and_report(out_dir, rows, seed42_npz, make_figs=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
