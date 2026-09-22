"""Regression tests for the SWAN_TRAIN_FRACTION hook in the follow-up copy.

The hook subsamples training blocks inside each stratum while leaving the
validation and test sets untouched. These tests run on CPU with a synthetic
wave array and never read data.
"""
import os

import numpy as np
import pytest

from _legacy_source import LEGACY, LEGACY_FOLLOWUP, load_functions

SEQ = 12
BLOCK = 168
N_BLOCKS = 30


@pytest.fixture(scope="module")
def wave_data():
    # (T, 4, H, W): channel 0 is Hs. Give every block its own amplitude so the
    # q=5 stratification is non-degenerate.
    rng = np.random.default_rng(0)
    T = BLOCK * N_BLOCKS + SEQ
    amp = np.repeat(rng.uniform(0.5, 3.0, size=N_BLOCKS + 1), BLOCK)[:T]
    hs = amp[:, None, None] * (1.0 + 0.1 * rng.standard_normal((T, 3, 3)))
    wd = np.zeros((T, 4, 3, 3), dtype=np.float32)
    wd[:, 0] = hs
    return wd


@pytest.fixture(scope="module")
def original():
    return load_functions(LEGACY, ["make_block_stratified_split"])


@pytest.fixture(scope="module")
def followup():
    return load_functions(LEGACY_FOLLOWUP,
                          ["_followup_fraction", "make_block_stratified_split", "robust_block_split"],
                          os=os)


def _split(ns, wave_data):
    return ns["make_block_stratified_split"](wave_data, SEQ, block_hours=BLOCK, q=5, seed=42,
                                             embargo_hours=SEQ)


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_unset_env_matches_original(monkeypatch, wave_data, original, followup):
    monkeypatch.delenv("SWAN_TRAIN_FRACTION", raising=False)
    ref = _split(original, wave_data)
    assert all(len(x) > 0 for x in ref)
    assert _same(ref, _split(followup, wave_data))


def test_fraction_one_matches_original(monkeypatch, wave_data, original, followup):
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", "1.0")
    assert _same(_split(original, wave_data), _split(followup, wave_data))


def test_half_keeps_val_test_and_subsets_train(monkeypatch, wave_data, original, followup):
    monkeypatch.delenv("SWAN_TRAIN_FRACTION", raising=False)
    tr_full, va_full, te_full = _split(original, wave_data)
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", "0.5")
    tr_half, va_half, te_half = _split(followup, wave_data)
    assert np.array_equal(va_half, va_full)
    assert np.array_equal(te_half, te_full)
    assert set(tr_half) < set(tr_full)
    # Blocks are dropped whole, so the sample count scales with the kept blocks.
    frac = len(tr_half) / len(tr_full)
    assert 0.3 < frac < 0.7


def test_quarter_nested_in_half(monkeypatch, wave_data, followup):
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", "0.5")
    tr_half, va_half, te_half = _split(followup, wave_data)
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", "0.25")
    tr_q, va_q, te_q = _split(followup, wave_data)
    assert set(tr_q) < set(tr_half)
    assert np.array_equal(va_q, va_half) and np.array_equal(te_q, te_half)


def test_robust_split_uses_first_candidate(monkeypatch, wave_data, followup, capsys):
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", "0.5")
    idx_tr, idx_va, idx_te, tag = followup["robust_block_split"](wave_data, SEQ)
    assert tag == f"block(bh={BLOCK},q=5,emb={SEQ})"
    assert "[split-fraction]" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["0", "-1", "1.1", "nan", "abc"])
def test_invalid_fraction_raises_without_fallback(monkeypatch, wave_data, followup, bad):
    monkeypatch.setenv("SWAN_TRAIN_FRACTION", bad)
    with pytest.raises(ValueError):
        followup["_followup_fraction"]()
    with pytest.raises(ValueError):
        followup["robust_block_split"](wave_data, SEQ)
