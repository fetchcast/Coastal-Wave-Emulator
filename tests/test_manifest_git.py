"""The run manifest must record the commit hash and a dirty flag."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import train  # noqa: E402


def test_git_state_keys_present():
    st = train.git_state(REPO)
    assert set(st) == {"git_commit", "git_dirty"}
    assert isinstance(st["git_commit"], str) and st["git_commit"]


def test_git_state_matches_repository():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    st = train.git_state(REPO)
    assert st["git_commit"] == head.stdout.strip()
    assert st["git_dirty"] in (True, False)


def test_worker_manifest_includes_git_state():
    # The manifest dict is built inline in worker_main; confirm the keys are
    # spliced in next to the resolved paths.
    src = (REPO / "train.py").read_text(encoding="utf-8")
    block = src[src.index('"resolved_legacy_train_script"'):src.index('write_json(run_dir / "run_manifest.json"')]
    assert "**git_state()" in block


def test_git_state_outside_repository(tmp_path):
    st = train.git_state(tmp_path)
    assert st == {"git_commit": "unknown", "git_dirty": None}
