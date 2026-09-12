"""Extract functions from the legacy training scripts without importing them.

The legacy scripts select a CUDA device, build an AMP scaler and configure
matplotlib at import time, so the tests compile only the function bodies
they need. Nested functions (for example ``robust_block_split`` inside
``wrapper``) are located by walking the AST and dedented before compiling.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py"
LEGACY_FOLLOWUP = LEGACY.with_name(LEGACY.stem + "_followup.py")


def function_source(path: Path, name: str) -> str:
    """Return the dedented source of the first function called ``name``."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg is not None
            return textwrap.dedent(seg)
    raise KeyError(f"{name} not found in {path.name}")


def load_functions(path: Path, names: Iterable[str], **extra_globals: Any) -> Dict[str, Any]:
    """Compile the named functions into one shared namespace.

    ``extra_globals`` seeds the namespace (numpy, os, ...). Functions that
    call each other must be listed together so they resolve at call time.
    """
    import numpy as np

    ns: Dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    ns.update(extra_globals)
    for name in names:
        exec(compile(function_source(path, name), f"{path.name}:{name}", "exec"), ns)
    return ns
