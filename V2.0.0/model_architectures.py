# -*- coding: utf-8 -*-
"""
model_architectures.py
======================
Clean, side-effect-free module that contains ONLY the model class
definitions used to train the APOR revision experiments E01-E08.

Why this file exists
--------------------
The inference scripts (inference_and_plot_V5_*.py) previously defined the
model inline, and those inline definitions had drifted from the actual
training code (missing `log_vars`, missing `final_drop`, a different
`feat` default). Importing the training script directly is also unsafe,
because that script runs argparse, CUDA setup, font setup, and a training
sweep at import time.

This module solves both problems. It contains the EXACT class definitions
from the training script
`UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py`
(SEBlock, ImprovedConvBlock, ConvLSTMCell, UNetPlusPlus, UNetConvLSTM),
plus the three ablation variants (ConvLSTM_only, UNetPP_stack,
UNetPP_only) and a `build_model` factory. Nothing here executes on
import; there is no argparse, no file I/O, no CUDA calls, no training.

IMPORTANT - verification step
-----------------------------
Before trusting this file, confirm that the five base classes below are
byte-for-byte identical to the ones in your current training script. Run:

    python -c "
    import ast, sys
    def classes(path):
        src = open(path, encoding='utf-8').read()
        tree = ast.parse(src)
        return {n.name: ast.get_source_segment(src, n)
                for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    a = classes('UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py')
    b = classes('model_architectures.py')
    for name in ['SEBlock','ImprovedConvBlock','ConvLSTMCell','UNetPlusPlus','UNetConvLSTM']:
        same = a.get(name) == b.get(name)
        print(f'{name:20s} identical={same}')
        if not same:
            print('--- TRAIN ---'); print(a.get(name))
            print('--- HERE  ---'); print(b.get(name))
    "

Every line must print `identical=True`. If any prints False, replace the
corresponding class here with the version from the training script and
re-run the check. Do NOT run inference until all five match.

Notes on the dormant `feat` default
------------------------------------
In the training script, `UNetConvLSTM.__init__` carries the default
`feat=[24, 48, 96, 192, 384]`. That default was never exercised: the
training loop builds the model with an explicit
`feat=[32, 64, 128, 256, 512]` (see `unet_feat_list` and the
`UNetConvLSTM(..., feat=unet_feat)` call). To avoid any ambiguity, this
module makes `feat` a required argument with no default, so a caller
must always pass the value that matches the checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-level constant copied from the training script
#   DROPOUT_P = 0.1
# In the training script this is a global; here it is a plain constant so
# the dropout layers (which are not parameters and therefore do not appear
# in the checkpoint) construct with the same probability.
# ---------------------------------------------------------------------------
DROPOUT_P = 0.1


# ===========================================================================
# Base building blocks  -- copied verbatim from the training script
# ===========================================================================
class SEBlock(nn.Module):
    def __init__(self, c, red=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, c // red), nn.ReLU(True),
                                nn.Linear(c // red, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x)
        return x * w.view(x.size(0), x.size(1), 1, 1)


class ImprovedConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, pad=1):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_c, in_c, k, padding=pad, groups=in_c, bias=False)
        self.conv_pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        gn = 32
        while out_c % gn != 0 and gn > 1:
            gn //= 2
        self.norm = nn.GroupNorm(gn, out_c)
        self.se = SEBlock(out_c)
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout2d(p=DROPOUT_P)

    def forward(self, x):
        x = self.conv_pw(self.conv_dw(x))
        x = self.act(self.drop(self.se(self.norm(x))))
        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        pad = k // 2
        self.h = hid_c
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=pad)

    def forward(self, x, s):
        h, c = s
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], 1)), self.h, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, B, H, W, dev=None):
        # The training script's version reads a module-global `device`.
        # Here we accept an explicit device so this file has no globals.
        # Callers inside this module always pass dev=x.device.
        if dev is None:
            dev = torch.device("cpu")
        z = torch.zeros(B, self.h, H, W, device=dev)
        return z.clone(), z.clone()


class UNetPlusPlus(nn.Module):
    def __init__(self, in_c, out_c, feat):
        super().__init__()
        f = feat
        self.enc00 = ImprovedConvBlock(in_c, f[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.enc10 = ImprovedConvBlock(f[0], f[1])
        self.enc20 = ImprovedConvBlock(f[1], f[2])
        self.enc30 = ImprovedConvBlock(f[2], f[3])
        self.enc40 = ImprovedConvBlock(f[3], f[4])

        self.dec01 = ImprovedConvBlock(f[0] + f[1], f[0])
        self.dec11 = ImprovedConvBlock(f[1] + f[2], f[1])
        self.dec21 = ImprovedConvBlock(f[2] + f[3], f[2])
        self.dec31 = ImprovedConvBlock(f[3] + f[4], f[3])

        self.dec02 = ImprovedConvBlock(f[0] * 2 + f[1], f[0])
        self.dec12 = ImprovedConvBlock(f[1] * 2 + f[2], f[1])
        self.dec22 = ImprovedConvBlock(f[2] * 2 + f[3], f[2])

        self.dec03 = ImprovedConvBlock(f[0] * 3 + f[1], f[0])
        self.dec13 = ImprovedConvBlock(f[1] * 3 + f[2], f[1])

        self.dec04 = ImprovedConvBlock(f[0] * 4 + f[1], f[0])
        self.outs = nn.ModuleList([nn.Conv2d(f[0], out_c, 1) for _ in range(4)])

    def _u(self, x, y):
        return torch.cat(
            [F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False), y], 1)

    def forward(self, x):
        x00 = self.enc00(x)
        x10 = self.enc10(self.pool(x00))
        x20 = self.enc20(self.pool(x10))
        x30 = self.enc30(self.pool(x20))
        x40 = self.enc40(self.pool(x30))

        x01 = self.dec01(self._u(x10, x00))
        x11 = self.dec11(self._u(x20, x10))
        x21 = self.dec21(self._u(x30, x20))
        x31 = self.dec31(self._u(x40, x30))

        x02 = self.dec02(self._u(x11, torch.cat([x00, x01], 1)))
        x12 = self.dec12(self._u(x21, torch.cat([x10, x11], 1)))
        x22 = self.dec22(self._u(x31, torch.cat([x20, x21], 1)))

        x03 = self.dec03(self._u(x12, torch.cat([x00, x01, x02], 1)))
        x13 = self.dec13(self._u(x22, torch.cat([x10, x11, x12], 1)))
        x04 = self.dec04(self._u(x13, torch.cat([x00, x01, x02, x03], 1)))
        return [self.outs[0](x04), self.outs[1](x03), self.outs[2](x02), self.outs[3](x01)]


class UNetConvLSTM(nn.Module):
    # The training script declares the dormant default feat=[24,48,96,192,384];
    # it is never used because the training loop always passes feat explicitly.
    # Here feat has no default, so the caller must pass the checkpoint-matching
    # value. hidden_dim default is kept at 64 to match the training script's
    # signature, but the training loop always passes hidden_dim=128 explicitly.
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=64, feat=None):
        super().__init__()
        if feat is None:
            raise ValueError(
                "UNetConvLSTM requires an explicit `feat` list. The training "
                "runs used feat=[32, 64, 128, 256, 512]."
            )
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.lstm = ConvLSTMCell(output_channels, hidden_dim)
        self.final_drop = nn.Dropout2d(p=DROPOUT_P)
        self.head = nn.Conv2d(hidden_dim, output_channels, 1)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [Hs, Tm, Dir]

    def forward(self, x):
        B, T, _, H, W = x.shape
        h, c = self.lstm.init_state(B, H, W, dev=x.device)
        last_u = None
        for t in range(T):
            outs = self.unet(x[:, t])
            last_u = outs
            h, c = self.lstm(outs[0], (h, c))
        h = self.final_drop(h)
        return [self.head(h)] + last_u


# ===========================================================================
# Ablation variants
#   These mirror the variants used in run_single_experiment.py for E02/E03.
#   Their parameter names are chosen so that the saved checkpoints load with
#   missing=[] and unexpected=[] (no strict=False needed for these).
# ===========================================================================
class UNetConvLSTM_full(UNetConvLSTM):
    """Alias of the published baseline (variant 'full')."""
    pass


class ConvLSTM_only(nn.Module):
    """
    Pure ConvLSTM stack with four full-resolution 1x1 heads.

    This is the E02 ablation: it removes the UNet++ multi-scale spatial
    decoder. Two stacked ConvLSTM cells run over the input window, then
    four 1x1 heads produce the main output and three (intentionally
    redundant) auxiliary outputs at the same full resolution, so the
    deep-supervision loss interface is unchanged.
    """
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=128, feat=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = ConvLSTMCell(input_channels, hidden_dim)
        self.lstm2 = ConvLSTMCell(hidden_dim, hidden_dim)
        self.norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.drop = nn.Dropout2d(p=DROPOUT_P)
        self.head_main = nn.Conv2d(hidden_dim, output_channels, 1)
        self.head_aux1 = nn.Conv2d(hidden_dim, output_channels, 1)
        self.head_aux2 = nn.Conv2d(hidden_dim, output_channels, 1)
        self.head_aux3 = nn.Conv2d(hidden_dim, output_channels, 1)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        B, T, _, H, W = x.shape
        h1, c1 = self.lstm1.init_state(B, H, W, dev=x.device)
        h2, c2 = self.lstm2.init_state(B, H, W, dev=x.device)
        for t in range(T):
            h1, c1 = self.lstm1(x[:, t], (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
        h = self.drop(self.norm(h2))
        return [self.head_main(h),
                self.head_aux1(h),
                self.head_aux2(h),
                self.head_aux3(h)]


class UNetPP_stack(nn.Module):
    """
    UNet++ applied to the 12-hour input concatenated along the channel
    dimension (B, T, C, H, W) -> (B, T*C, H, W).

    This is the E03 ablation: it keeps all 12 hours of forcing information
    but removes the recurrent temporal module.
    """
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=None,
                 feat=None, seq_length=12):
        super().__init__()
        if feat is None:
            raise ValueError(
                "UNetPP_stack requires an explicit `feat` list. The training "
                "runs used feat=[32, 64, 128, 256, 512]."
            )
        self.seq_length = seq_length
        self.unet = UNetPlusPlus(input_channels * seq_length, output_channels, feat)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_stack = x.reshape(B, T * C, H, W)
        return self.unet(x_stack)


class UNetPP_only(nn.Module):
    """
    UNet++ applied to the last input frame only (no temporal aggregation).
    Kept for reference; not part of the main 8-experiment matrix.
    """
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=None,
                 feat=None):
        super().__init__()
        if feat is None:
            raise ValueError(
                "UNetPP_only requires an explicit `feat` list. The training "
                "runs used feat=[32, 64, 128, 256, 512]."
            )
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return self.unet(x[:, -1])


# ===========================================================================
# Factory
# ===========================================================================
def build_model(variant, input_channels, hidden_dim, feat, seq_length=12):
    """
    Build a model variant.

    Parameters
    ----------
    variant : str
        One of: 'full', 'convlstm_only', 'unetpp_stack', 'unetpp_only'.
    input_channels : int
        Number of input channels. 10 for BND-on runs (6 geophysical + 4
        boundary descriptors), 6 for BND-off runs.
    hidden_dim : int
        ConvLSTM hidden dimension. The training runs used 128.
    feat : list[int]
        UNet++ channel widths. The training runs used
        [32, 64, 128, 256, 512]. This MUST match the checkpoint.
    seq_length : int
        Input window length. Only used by 'unetpp_stack'. Training used 12.

    Returns
    -------
    torch.nn.Module
    """
    v = str(variant).lower()
    if v in ("full", "unet_convlstm", "baseline"):
        return UNetConvLSTM_full(input_channels=input_channels,
                                 output_channels=4,
                                 hidden_dim=hidden_dim,
                                 feat=feat)
    if v in ("convlstm_only", "convlstm-only", "convlstmonly"):
        return ConvLSTM_only(input_channels=input_channels,
                             output_channels=4,
                             hidden_dim=hidden_dim,
                             feat=feat)
    if v in ("unetpp_stack", "unetppstack", "unet_stack"):
        return UNetPP_stack(input_channels=input_channels,
                            output_channels=4,
                            hidden_dim=hidden_dim,
                            feat=feat,
                            seq_length=seq_length)
    if v in ("unetpp_only", "unet_only", "unetpponly"):
        return UNetPP_only(input_channels=input_channels,
                           output_channels=4,
                           hidden_dim=hidden_dim,
                           feat=feat)
    raise ValueError(f"Unknown variant: {variant!r}")


# ===========================================================================
# Safe checkpoint loader
# ===========================================================================
def load_checkpoint_strict_except(model, state_dict, allowed_missing=None,
                                  allowed_unexpected=("log_vars",)):
    """
    Load a state_dict with strict=False, but raise if any missing or
    unexpected key is NOT in the explicitly allowed list.

    The training-time `UNetConvLSTM` carries a `log_vars` parameter used
    only by the loss. Inference does not need it, so `log_vars` appearing
    as an unexpected key is fine. Any OTHER mismatch means the model
    definition and the checkpoint disagree, and we want a hard error
    rather than a silently wrong model.

    Returns
    -------
    (missing, unexpected) : the raw lists from load_state_dict, for logging.
    """
    if allowed_missing is None:
        allowed_missing = []
    allowed_missing = set(allowed_missing)
    allowed_unexpected = set(allowed_unexpected)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in missing if k not in allowed_missing]
    bad_unexpected = [k for k in unexpected if k not in allowed_unexpected]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint does not match model definition.\n"
            f"  unexpected (allowed: {sorted(allowed_unexpected)}): {bad_unexpected}\n"
            f"  missing   (allowed: {sorted(allowed_missing)}): {bad_missing}\n"
            "This means model_architectures.py has drifted from the training "
            "code, or the wrong --variant / --feat / --use_bnd was passed."
        )
    return missing, unexpected
