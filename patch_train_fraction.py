#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_train_fraction.py  (v4)
=============================
Create (or refresh) the follow-up copy of the legacy training script. The
original is never modified; the copy is regenerated from the CURRENT original
every run and compared with the existing copy, and SHA-256 of both files is
recorded.

Four edits are applied to the copy (each must match exactly once):

  E1  make_block_stratified_split(): optional SWAN_TRAIN_FRACTION hook.
      Training blocks are subsampled within each stratum; dropped blocks
      belong to no set (-1); validation and test are unchanged.
  E2  wrapper(): the fraction is validated at the start, OUTSIDE the split
      helper's try/except, so an invalid value stops the worker instead of
      being swallowed and falling through to a random split.
  E3  robust_block_split(): when a fraction is set, a split failure raises
      instead of trying the next candidate or the random fallback, so the
      data-scaling runs can only ever use the same (bh=168, q=5) split as
      the 100 % run.
  E4  _auto_align_bnd_dir(): the transform is scored on TRAINING target
      times only (no validation/test targets), and SWAN_BND_DIR_TRANSFORM
      (e.g. "refl+270") forces a fixed transform; the train-only scores are
      still printed as a diagnostic. The call site passes the training
      indices.

Run on the server:   cd ~/swan && python3 patch_train_fraction.py
"""
import hashlib
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path

SRC = Path("/home/jovyan/swan/UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py")
DST = SRC.with_name(SRC.stem + "_followup.py")
HASHES = DST.with_suffix(".sha256.json")

EDITS = []

# ---------------------------------------------------------------- E1
EDITS.append(("E1 split hook", '''    tr_blocks, va_blocks, te_blocks = set(tr_blocks), set(va_blocks), set(te_blocks)
    blk2set = {b:(0 if b in tr_blocks else (1 if b in va_blocks else 2)) for b in range(num_blocks)}
''', '''    # --- follow-up hook: optional training-data fraction -------------------
    _frac = _followup_fraction()
    if _frac < 1.0:
        _rng_f = np.random.default_rng(seed + 1000)
        _kept = []
        for lab in range(q):
            _b = sorted(b for b in tr_blocks if labels[b] == lab)
            _rng_f.shuffle(_b)
            _kept.extend(_b[:max(1, int(round(_frac * len(_b))))])
        _kept = sorted(int(b) for b in _kept)
        print(f"[split-fraction] SWAN_TRAIN_FRACTION={_frac}: training blocks "
              f"{len(tr_blocks)} -> {len(_kept)} (actual {len(_kept)/max(1,len(tr_blocks)):.3f}); "
              f"val/test unchanged; kept block ids: {_kept}")
        tr_blocks = _kept
    # -----------------------------------------------------------------------
    tr_blocks, va_blocks, te_blocks = set(tr_blocks), set(va_blocks), set(te_blocks)
    blk2set = {b:(0 if b in tr_blocks else (1 if b in va_blocks else (2 if b in te_blocks else -1)))
               for b in range(num_blocks)}
'''))

# ---------------------------------------------------------------- E2
EDITS.append(("E2 validate at wrapper start", '''def wrapper(data_path, use_bnd="on"):  # "on" | "off" | "auto"
''', '''def _followup_fraction():
    """SWAN_TRAIN_FRACTION as a float in (0, 1]; unset or empty means 1.0.
    Any other value raises, and the caller must not swallow that error."""
    import os as _os, math as _math
    _raw = _os.environ.get("SWAN_TRAIN_FRACTION")
    if _raw is None or _raw.strip() == "":
        return 1.0
    try:
        _f = float(_raw)
    except ValueError:
        raise ValueError(f"SWAN_TRAIN_FRACTION must be a number in (0, 1], got {_raw!r}")
    if not _math.isfinite(_f) or not (0.0 < _f <= 1.0):
        raise ValueError(f"SWAN_TRAIN_FRACTION must be in (0, 1], got {_f}")
    return _f


def wrapper(data_path, use_bnd="on"):  # "on" | "off" | "auto"
    # Follow-up: fail before any data is read if the fraction is invalid.
    _followup_fraction()
'''))

# ---------------------------------------------------------------- E3
EDITS.append(("E3a no candidate fallback under a fraction", '''                    except Exception:
                        continue
                    if len(idx_tr)>0 and len(idx_va)>0 and len(idx_te)>0:
                        print(f"[split-ok] bh={bh}, q={q}, emb={emb} -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
''', '''                    except Exception:
                        if _followup_fraction() < 1.0:
                            raise   # a data-scaling run must not fall through to another split
                        continue
                    if len(idx_tr)>0 and len(idx_va)>0 and len(idx_te)>0:
                        print(f"[split-ok] bh={bh}, q={q}, emb={emb} -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
'''))
EDITS.append(("E3b no random fallback under a fraction", '''        # fallback
        rng = np.random.default_rng(42)
''', '''        # fallback
        if _followup_fraction() < 1.0:
            raise RuntimeError("block split failed under SWAN_TRAIN_FRACTION; random fallback is not allowed")
        rng = np.random.default_rng(42)
'''))

# ---------------------------------------------------------------- E4
EDITS.append(("E4a autocorrect signature", '''    def _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index):
''', '''    def _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index, train_idx=None, seq_length=0):
'''))
EDITS.append(("E4b train-only scoring + forced transform", '''        def _score(deg, reflect):
            sin_r, cos_r = _transform(deg, reflect)
            v = (sin_r*tsin + cos_r*tcos)  # cos(Δθ)
            vv = v[:, mask]
            return float(np.nanmean(vv))

        # Four rotations and four reflections cover every axis-aligned convention
        # change between compass-from, compass-to, and Cartesian angles.
        candidates = [(0.0, False), (90.0, False), (-90.0, False), (180.0, False),
                      (0.0, True), (90.0, True), (180.0, True), (270.0, True)]
        scores = {c: _score(*c) for c in candidates}
        best = max(scores, key=lambda c: scores[c])
        best_deg, best_reflect = best
''', '''        # Follow-up: score the candidate transforms on TRAINING target times only.
        # Sample index t targets raw time t + seq_length (see make_block_stratified_split).
        if train_idx is not None and len(train_idx) > 0:
            _sel = np.asarray(train_idx, dtype=int) + int(seq_length)
            _sel = _sel[(_sel >= 0) & (_sel < T)]
        else:
            _sel = np.arange(T)
        print(f"[BND] dir autocorrect scoring on {len(_sel)} training target times out of {T}")

        def _score(deg, reflect):
            sin_r, cos_r = _transform(deg, reflect)
            v = (sin_r*tsin + cos_r*tcos)  # cos(Δθ)
            vv = v[_sel][:, mask]
            return float(np.nanmean(vv))

        # Four rotations and four reflections cover every axis-aligned convention
        # change between compass-from, compass-to, and Cartesian angles.
        candidates = [(0.0, False), (90.0, False), (-90.0, False), (180.0, False),
                      (0.0, True), (90.0, True), (180.0, True), (270.0, True)]
        scores = {c: _score(*c) for c in candidates}
        best = max(scores, key=lambda c: scores[c])
        # Follow-up: a fixed transform can be forced (documented convention);
        # the search result above is still printed for the record.
        import os as _os, re as _re
        _forced = _os.environ.get("SWAN_BND_DIR_TRANSFORM", "").strip()
        if _forced:
            _m = _re.fullmatch(r"(rot|refl)([+-]?\\d+(?:\\.\\d+)?)", _forced)
            if not _m:
                raise ValueError(f"SWAN_BND_DIR_TRANSFORM must look like rot-90 or refl+270, got {_forced!r}")
            _fc = (float(_m.group(2)), _m.group(1) == "refl")
            print(f"[BND] dir transform FORCED to {_forced} (train-only search would pick "
                  f"{'refl' if best[1] else 'rot'}{best[0]:+.0f})")
            best = _fc
        best_deg, best_reflect = best
'''))
EDITS.append(("E4c call site passes training indices", '''                    best_deg, scores = _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index)
''', '''                    best_deg, scores = _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index,
                                                           train_idx=idx_tr, seq_length=seq_length)
'''))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not SRC.exists():
        sys.exit(f"[STOP] source not found: {SRC}")
    src = SRC.read_text(encoding="utf-8")
    if "SWAN_TRAIN_FRACTION" in src:
        sys.exit(f"[STOP] the ORIGINAL {SRC.name} contains the hook; restore it from backup first.")
    patched = src
    for name, old, new in EDITS:
        n = patched.count(old)
        if n != 1:
            sys.exit(f"[STOP] {name}: expected exactly one match, found {n}. Nothing written.")
        patched = patched.replace(old, new)

    if DST.exists() and DST.read_text(encoding="utf-8") == patched:
        print(f"[OK] {DST.name} is up to date with the current original.")
    else:
        if DST.exists():
            print(f"[note] {DST.name} differs from what the current original produces; rebuilding.")
        fd, tmp = tempfile.mkstemp(prefix=DST.stem + ".", suffix=".py", dir=str(DST.parent))
        os.close(fd)
        try:
            Path(tmp).write_text(patched, encoding="utf-8")
            py_compile.compile(tmp, doraise=True)
            os.replace(tmp, DST)
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            sys.exit(f"[STOP] patched copy failed to compile; nothing written. {exc}")
        print(f"[OK] wrote {DST.name}  (original {SRC.name} untouched)")
    HASHES.write_text(json.dumps({"original": str(SRC), "original_sha256": sha(SRC),
                                  "followup_copy": str(DST), "followup_sha256": sha(DST)}, indent=2))
    print(f"[OK] hashes recorded in {HASHES.name}")


if __name__ == "__main__":
    main()
