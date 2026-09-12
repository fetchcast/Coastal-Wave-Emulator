"""Regression test for the two-block spectral convolutions in train.py.

With identity weights on the retained modes a spectral convolution must
return its input for any signal whose Fourier content lies inside the
retained block. cos(2*pi*(3x + 4y)) lives in the positive-frequency block of
the first axis; cos(2*pi*(-3x + 4y)) lives in the negative-frequency block
[-mx:, :my]. A single-block implementation drops the second signal.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402

HW = 32
MODES = 8
TOL = 1e-4


def _identity_weights(module):
    # Every spectral weight is stored as a real tensor (..., 2) = (re, im).
    with torch.no_grad():
        for p in module.parameters():
            p[..., 0] = 1.0
            p[..., 1] = 0.0


def _field2d(kx, ky, h=HW, w=HW):
    x = torch.arange(h, dtype=torch.float32)[:, None] / h
    y = torch.arange(w, dtype=torch.float32)[None, :] / w
    return torch.cos(2 * torch.pi * (kx * x + ky * y))[None, None]        # (1, 1, h, w)


def _field3d(kt, kx, ky, t=8, h=HW, w=HW):
    tt = torch.arange(t, dtype=torch.float32)[:, None, None] / t
    x = torch.arange(h, dtype=torch.float32)[None, :, None] / h
    y = torch.arange(w, dtype=torch.float32)[None, None, :] / w
    return torch.cos(2 * torch.pi * (kt * tt + kx * x + ky * y))[None, None]   # (1, 1, t, h, w)


@pytest.mark.parametrize("kx,ky", [(3, 4), (-3, 4)])
def test_spectral_conv2d_reconstructs_both_blocks(kx, ky):
    conv = train.SpectralConv2d(1, 1, MODES, MODES)
    _identity_weights(conv)
    x = _field2d(kx, ky)
    y = conv(x)
    assert torch.max(torch.abs(y - x)).item() < TOL


@pytest.mark.parametrize("kt,kx,ky", [(0, 3, 4), (0, -3, 4), (-2, 3, 4), (2, -3, 4)])
def test_spectral_conv3d_reconstructs_all_sign_blocks(kt, kx, ky):
    conv = train.SpectralConv3d(1, 1, 3, MODES, MODES)
    _identity_weights(conv)
    x = _field3d(kt, kx, ky)
    y = conv(x)
    assert torch.max(torch.abs(y - x)).item() < TOL


def test_odd_grid_does_not_overlap_blocks():
    # On an odd grid the positive and negative blocks must not overwrite each
    # other; identity weights must still reproduce a mixed-sign signal.
    conv = train.SpectralConv2d(1, 1, MODES, MODES)
    _identity_weights(conv)
    x = _field2d(3, 4, h=31, w=33) + _field2d(-3, 4, h=31, w=33)
    y = conv(x)
    assert torch.max(torch.abs(y - x)).item() < TOL
