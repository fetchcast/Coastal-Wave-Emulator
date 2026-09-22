"""Tests for the boundary-direction autocorrect in the follow-up copy.

Synthetic fields: the boundary direction channels follow the reflection
theta_true = 270 - theta_bnd at training target times and a plain rotation
by 0 degrees elsewhere. Scoring on every frame would pick the rotation; the
follow-up copy must score training target times only and pick refl+270.
"""
import os
import re
import types

import numpy as np
import pytest

from _legacy_source import LEGACY, LEGACY_FOLLOWUP, load_functions

T, H, W = 200, 6, 6
SEQ = 12
TRAIN_TARGETS = np.arange(SEQ, SEQ + 50)          # 50 training target frames
TRAIN_IDX = TRAIN_TARGETS - SEQ                    # sample index t targets t + SEQ


def _make_case(train_convention="refl270", other_convention="rot0"):
    rng = np.random.default_rng(1)
    theta_b = rng.uniform(0.0, 360.0, size=(T, H, W)).astype(np.float32)
    theta_true = np.where(np.isin(np.arange(T), TRAIN_TARGETS)[:, None, None],
                          _apply(theta_b, train_convention), _apply(theta_b, other_convention))
    bnd = np.zeros((T, 4, H, W), dtype=np.float32)
    bnd[:, 0] = 0.5
    bnd[:, 1] = 0.5
    bnd[:, 2] = np.sin(np.deg2rad(theta_b))
    bnd[:, 3] = np.cos(np.deg2rad(theta_b))
    ds_sim = {"dir": types.SimpleNamespace(values=theta_true.astype(np.float32))}
    kcs2d = np.ones((H, W), dtype=np.float32)
    return bnd, ds_sim, kcs2d, theta_b


def _apply(theta_b, convention):
    if convention == "refl270":
        return (270.0 - theta_b) % 360.0
    if convention == "rot0":
        return theta_b
    raise ValueError(convention)


@pytest.fixture(scope="module")
def patched():
    return load_functions(LEGACY_FOLLOWUP, ["_auto_align_bnd_dir"], os=os, re=re)["_auto_align_bnd_dir"]


@pytest.fixture(scope="module")
def original():
    return load_functions(LEGACY, ["_auto_align_bnd_dir"])["_auto_align_bnd_dir"]


def test_train_only_scoring_picks_reflection(monkeypatch, patched):
    monkeypatch.delenv("SWAN_BND_DIR_TRANSFORM", raising=False)
    bnd, ds_sim, kcs2d, theta_b = _make_case()
    best, scores = patched(bnd, ds_sim, kcs2d, np.arange(T), train_idx=TRAIN_IDX, seq_length=SEQ)
    assert best == pytest.approx(1000.0 + 270.0)          # reflection encoded as 1000 + deg
    assert max(scores, key=scores.get) == "refl+270"
    assert scores["refl+270"] == pytest.approx(1.0, abs=1e-5)
    # Applied in place: sin(270 - theta) = -cos(theta), cos(270 - theta) = -sin(theta)
    np.testing.assert_allclose(bnd[:, 2], -np.cos(np.deg2rad(theta_b)), atol=1e-5)
    np.testing.assert_allclose(bnd[:, 3], -np.sin(np.deg2rad(theta_b)), atol=1e-5)


def test_all_frame_scoring_would_pick_rotation(original):
    # The pre-patch function scores every frame; the 150 non-training frames
    # follow the identity rotation and dominate.
    bnd, ds_sim, kcs2d, _ = _make_case()
    best, scores = original(bnd, ds_sim, kcs2d, np.arange(T))
    assert best == pytest.approx(0.0)
    assert max(scores, key=scores.get) == "rot+0"


def test_forced_transform_overrides_search(monkeypatch, patched, capsys):
    # Data favour rot+0 everywhere, but the forced reflection must be applied.
    monkeypatch.setenv("SWAN_BND_DIR_TRANSFORM", "refl+270")
    bnd, ds_sim, kcs2d, theta_b = _make_case(train_convention="rot0", other_convention="rot0")
    best, scores = patched(bnd, ds_sim, kcs2d, np.arange(T), train_idx=TRAIN_IDX, seq_length=SEQ)
    assert best == pytest.approx(1000.0 + 270.0)
    out = capsys.readouterr().out
    assert "FORCED to refl+270" in out
    assert "train-only search would pick rot+0" in out
    np.testing.assert_allclose(bnd[:, 2], -np.cos(np.deg2rad(theta_b)), atol=1e-5)
    np.testing.assert_allclose(bnd[:, 3], -np.sin(np.deg2rad(theta_b)), atol=1e-5)


def test_forced_rotation_is_a_rotation(monkeypatch, patched):
    monkeypatch.setenv("SWAN_BND_DIR_TRANSFORM", "rot-90")
    bnd, ds_sim, kcs2d, theta_b = _make_case()
    best, _ = patched(bnd, ds_sim, kcs2d, np.arange(T), train_idx=TRAIN_IDX, seq_length=SEQ)
    assert best == pytest.approx(-90.0)
    np.testing.assert_allclose(bnd[:, 2], np.sin(np.deg2rad(theta_b - 90.0)), atol=1e-5)
    np.testing.assert_allclose(bnd[:, 3], np.cos(np.deg2rad(theta_b - 90.0)), atol=1e-5)


def test_bogus_transform_raises(monkeypatch, patched):
    monkeypatch.setenv("SWAN_BND_DIR_TRANSFORM", "bogus")
    bnd, ds_sim, kcs2d, _ = _make_case()
    with pytest.raises(ValueError):
        patched(bnd, ds_sim, kcs2d, np.arange(T), train_idx=TRAIN_IDX, seq_length=SEQ)
