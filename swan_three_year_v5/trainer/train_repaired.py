from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Quiet the cosmetic "findfont: Font family 'Arial' not found" messages emitted
# by matplotlib when the figures are rendered. The fallback font is substituted
# automatically; this affects only figure typography, not any computed result.
warnings.filterwarnings("ignore", message="findfont")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =============================================================================
# HARD-CODED USER CONFIGURATION
# Edit only this block when you need to change paths or search space.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Core paths
    "legacy_train_script": "/home/jovyan/swan/UNET_LSTM_V64_fixes_ds_loss_peaksampler_boundary_input_9input.py",
    "data_path": "/home/jovyan/swan/wavm-Waves_2019_2020_v2.nc",
    "results_root": "/home/jovyan/swan/runs/v2_focused_all",
    "log_file": "/home/jovyan/swan/v2_focused_all.log",

    # Boundary settings
    "use_bnd": "on",
    "bnd_dir_2019": "/home/jovyan/swan/bnd_2019_v2",
    "bnd_dir_2020": "/home/jovyan/swan/bnd_2020_v2",
    # Folder holding the buoy/station validation CSVs. Defaults to the data
    # file's directory if left out. Used only for end-of-run buoy validation.
    "station_root": "/home/jovyan/swan",

    # GPU / distributed settings
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
    "gpus_per_trial": 1,
    "max_concurrent_jobs": 8,
    "master_port_base": 29500,
    "omp_num_threads": 8,

    # Training settings
    "epochs": 30,
    "time_steps": 17498,
    "seq_length": 12,
    "seed_list": [42],
    # Stage-1 representative multiseed is disabled. The meaningful multi-seed test
    # runs in stage-2 on each model's winning config, so a middle-of-grid rerun
    # here would be redundant and is not used in the paper.
    "extra_seeds": [],
    "multiseed_models": ["convlstm", "unet_lstm", "fno", "ffno", "tno", "u_ffno",
                         "swin", "vit", "convnext_lstm", "conv_swin"],
    # Naive persistence baseline (predict the last input frame). Computed once
    # over the same test split as the models, to anchor what the error numbers mean.
    "compute_persistence_baseline": True,
    # Automatic stage-2 pipeline. After the stage-1 sweep finishes and is
    # aggregated, the best config per model is extracted and two follow-ups are
    # run on those winners: a BND on/off ablation and a true multi-seed pass.
    # A significance summary is then written from the multi-seed spread.
    "run_stage2_after_sweep": True,
    "stage2_best_metric": "val_loss_final",   # lower is better; falls back to rmse_hs
    "stage2_bnd_off_ablation": True,
    "stage2_true_multiseed_seeds": [42, 123, 456, 789, 1011],
    "significance_test": True,
    "benchmark_after_train": True,
    "benchmark_precision": "bf16",
    "synthetic_benchmark_iters": 20,
    "synthetic_benchmark_warmup": 5,
    "skip_existing_completed_runs": True,

    # Runtime behavior
    "continue_on_error": True,
    "plot_after_finish": True,
    "aggregate_after_finish": True,
    "keep_rank_workdirs": False,

    # Model selection
    "models_to_run": ["convlstm", "unet_lstm", "fno", "ffno", "tno", "u_ffno",
                      "swin", "vit", "convnext_lstm", "conv_swin"],

    # Focused search space
    "search_space": {
        "convlstm": {
            "widths": [128, 256, 512],
            "depths": [2, 3, 4],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "unet_lstm": {
            "feat_variants": [
                [64, 128, 256, 512, 1024],
                [128, 256, 512, 1024, 2048],
            ],
            "hidden_dims": [256, 512, 768],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "fno": {
    "widths": [128, 256, 384],
    "depths": [4, 6, 8],
    "mode_pairs": [
        [24, 24],
        [32, 32],
        [48, 48],
        [64, 64],
    ],
    "lrs": [1.0e-4],
    "wds": [1.0e-4],
},
        "ffno": {
            "widths": [128, 256, 384, 512],
            "depths": [4, 6],
            "mode_pairs": [
                [24, 24],
                [32, 32],
                [48, 48],
                [64, 64],
            ],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "tno": {
            # The full modes_h x modes_t x modes_w grid is a 27x multiplier and
            # dominates total compute. Fix the spatial modes and vary only the
            # temporal modes, which is where TNO is most sensitive, keeping the
            # job count tractable without losing the architecture comparison.
            "widths": [64, 128, 192, 256],
            "depths": [3, 4],
            "modes_h": [24],
            "modes_t": [4, 6, 8],
            "modes_w": [24],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "u_ffno": {
            "feat_variants": [
                [64, 128, 256, 512, 1024],
                [128, 256, 512, 1024, 2048],
            ],
            "widths": [256, 384],
            "depths": [4, 8],
            "mode_pairs": [
                [8, 8],
                [16, 16],
            ],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "swin": {
            # embed_dim is the stage-0 channel width; stages double it.
            "embed_dims": [72, 96, 128],
            # Each variant is (depths, num_heads); lengths must match and the
            # window size must divide head_dim. Two- and three-stage variants
            # keep the receptive field large without exploding memory.
            "stage_variants": [
                {"depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
                {"depths": [2, 6, 2],    "num_heads": [3, 6, 12]},
                {"depths": [2, 2, 2],    "num_heads": [4, 8, 16]},
            ],
            "window_sizes": [8],
            "patch_sizes": [4],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "vit": {
            # Large patch keeps the token count low so global attention is cheap.
            "embed_dims": [256, 384],
            "depths": [6, 8],
            "num_heads": [4, 8],
            "patch_sizes": [16],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "convnext_lstm": {
            # Three encoder stages; lstm_hidden is the temporal state width.
            "dim_variants": [
                {"dims": [64, 128, 256], "depths": [2, 2, 2]},
                {"dims": [96, 192, 384], "depths": [2, 2, 6]},
            ],
            "lstm_hiddens": [256, 384],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
        "conv_swin": {
            # Conv encoder width plus a Swin block at the bottleneck.
            "base_widths": [48, 64],
            "swin_dims": [256, 384],
            "swin_depths": [4, 6],
            "swin_heads": [8],
            "window_sizes": [8],
            "lrs": [1.0e-4],
            "wds": [1.0e-4],
        },
    },
}

# Repair package paths are isolated from the original benchmark.
_PACKAGE=Path(__file__).resolve().parent
_SERVER=Path(os.environ.get('SWAN_SERVER_ROOT','/home/jovyan/swan')).resolve()
CONFIG.update(legacy_train_script=str(_PACKAGE/'legacy_repaired.py'),
    data_path=os.environ.get('SWAN_DATA_PATH',str(_SERVER/'wavm-Waves_2019_2020_v2.nc')),
    results_root=os.environ.get('SWAN_RESULTS_ROOT',str(_SERVER/'runs/repaired_v1')),
    log_file=str(_SERVER/'repaired_v1.log'),
    bnd_dir_2019=os.environ.get('SWAN_BND_DIR_2019',str(_SERVER/'bnd_2019_v2')),
    bnd_dir_2020=os.environ.get('SWAN_BND_DIR_2020',str(_SERVER/'bnd_2020_v2')),
    bnd_dir_2021=os.environ.get('SWAN_BND_DIR_2021',str(_SERVER/'bnd_2021_v2')),
    station_root=os.environ.get('SWAN_STATION_ROOT',str(_SERVER)),
    benchmark_after_train=os.environ.get('SWAN_SYNTHETIC_BENCH','1')=='1')
if str(_SERVER) not in sys.path: sys.path.insert(0,str(_SERVER))
import repair_support as repair


# =============================================================================
# Logging and monitoring helpers
# =============================================================================

def setup_file_logger(name: str, log_path: Path, level: int = logging.DEBUG) -> logging.Logger:
    """Create a logger that writes to both file and stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_system_memory(logger: logging.Logger, tag: str = "") -> Dict[str, Any]:
    """Log current system RAM and GPU memory usage."""
    info: Dict[str, Any] = {"tag": tag}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_GB"] = round(vm.total / 1e9, 1)
        info["ram_used_GB"] = round(vm.used / 1e9, 1)
        info["ram_avail_GB"] = round(vm.available / 1e9, 1)
        info["ram_percent"] = vm.percent
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            meminfo = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total = meminfo.get("MemTotal", 0) / 1e6
            avail = meminfo.get("MemAvailable", 0) / 1e6
            info["ram_total_GB"] = round(total, 1)
            info["ram_avail_GB"] = round(avail, 1)
            info["ram_used_GB"] = round(total - avail, 1)
        except Exception:
            info["ram"] = "unavailable"
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            gpu_info.append({"gpu": i, "alloc_GB": round(alloc, 2), "reserved_GB": round(reserved, 2)})
        info["gpus"] = gpu_info
    logger.info(f"[MEM {tag}] {json.dumps(info)}")
    return info


def install_signal_handlers(logger: logging.Logger, run_name: str) -> None:
    """Install signal handlers that log before exit."""
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.critical(f"[SIGNAL] {run_name} received {sig_name} (signum={signum})")
        logger.critical(f"[SIGNAL] Stack at signal:\n{''.join(traceback.format_stack(frame))}")
        # flush all handlers
        for h in logger.handlers:
            h.flush()
        sys.exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass


# =============================================================================
# Model registry and building blocks
# =============================================================================
MODEL_REGISTRY: Dict[str, type[nn.Module]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)


class BaseWaveEmulator(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, seq_length: int) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.seq_length = int(seq_length)

    @staticmethod
    def make_coord_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([yy, xx], dim=0)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def init_state(self, batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device, dtype=dtype)
        return h, c

    def forward(self, x: torch.Tensor, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class SpectralConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,modes_x,modes_y):
        super().__init__()
        self.out_channels=out_channels; self.modes_x=int(modes_x); self.modes_y=int(modes_y)
        scale=1./max(1,in_channels*out_channels)
        shape=(in_channels,out_channels,self.modes_x,self.modes_y,2)
        self.weight_pos=nn.Parameter(scale*torch.randn(*shape))
        self.weight_neg=nn.Parameter(scale*torch.randn(*shape))
    def forward(self,x):
        b,_,h,w=x.shape
        mx=min(self.modes_x,(h+1)//2); mn=min(self.modes_x,h//2); my=min(self.modes_y,w//2+1)
        ft=torch.fft.rfft2(x.float(),norm='ortho')
        out=torch.zeros(b,self.out_channels,h,w//2+1,dtype=torch.cfloat,device=x.device)
        out[:,:,:mx,:my]=torch.einsum('bixy,ioxy->boxy',ft[:,:,:mx,:my],torch.view_as_complex(self.weight_pos)[:,:,:mx,:my])
        if mn:
            out[:,:,-mn:,:my]=torch.einsum('bixy,ioxy->boxy',ft[:,:,-mn:,:my],torch.view_as_complex(self.weight_neg)[:,:,:mn,:my])
        return torch.fft.irfft2(out,s=(h,w),norm='ortho').to(x.dtype)


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        scale = 1.0 / max(1, in_channels * out_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        # Store as real-valued (..., 2) for NCCL compatibility.
        self.weight_x = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes_x, 2))
        self.weight_y = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes_y, 2))

    def _apply_x(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft(x.float(), dim=-2, norm="ortho")
        out_ft = torch.zeros(batch, self.out_channels, h // 2 + 1, w, dtype=torch.cfloat, device=x.device)
        mx = min(self.modes_x, h // 2 + 1)
        w_c = torch.view_as_complex(self.weight_x)
        out_ft[:, :, :mx, :] = torch.einsum("bixw,iox->boxw", x_ft[:, :, :mx, :], w_c[:, :, :mx])
        return torch.fft.irfft(out_ft, n=h, dim=-2, norm="ortho")

    def _apply_y(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
        out_ft = torch.zeros(batch, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        my = min(self.modes_y, w // 2 + 1)
        w_c = torch.view_as_complex(self.weight_y)
        out_ft[:, :, :, :my] = torch.einsum("bihy,ioy->bohy", x_ft[:, :, :, :my], w_c[:, :, :my])
        return torch.fft.irfft(out_ft, n=w, dim=-1, norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._apply_x(x) + self._apply_y(x)).to(x.dtype)


class SpectralConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,modes_t,modes_x,modes_y):
        super().__init__()
        self.out_channels=out_channels; self.modes_t=int(modes_t); self.modes_x=int(modes_x); self.modes_y=int(modes_y)
        shape=(in_channels,out_channels,self.modes_t,self.modes_x,self.modes_y,2)
        scale=1./max(1,in_channels*out_channels)
        self.weights=nn.ParameterList([nn.Parameter(scale*torch.randn(*shape)) for _ in range(4)])
    def forward(self,x):
        b,_,t,h,w=x.shape
        ft=torch.fft.rfftn(x.float(),dim=(-3,-2,-1),norm='ortho')
        out=torch.zeros(b,self.out_channels,t,h,w//2+1,dtype=torch.cfloat,device=x.device)
        my=min(self.modes_y,w//2+1)
        for i,(neg_t,neg_x) in enumerate([(False,False),(True,False),(False,True),(True,True)]):
            mt=min(self.modes_t,t//2 if neg_t else (t+1)//2)
            mx=min(self.modes_x,h//2 if neg_x else (h+1)//2)
            if not mt or not mx: continue
            ts=slice(-mt,None) if neg_t else slice(0,mt)
            xs=slice(-mx,None) if neg_x else slice(0,mx)
            out[:,:,ts,xs,:my]=torch.einsum('bixyz,ioxyz->boxyz',ft[:,:,ts,xs,:my],
                torch.view_as_complex(self.weights[i])[:,:,:mt,:mx,:my])
        return torch.fft.irfftn(out,s=(t,h,w),dim=(-3,-2,-1),norm='ortho').to(x.dtype)


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, dropout: float = 0.0, factorized: bool = False) -> None:
        super().__init__()
        conv_cls = FactorizedSpectralConv2d if factorized else SpectralConv2d
        self.spectral = conv_cls(width, width, modes_x, modes_y)
        self.skip = nn.Conv2d(width, width, kernel_size=1)
        groups = max(1, min(8, width if width < 8 else max(1, width // 8)))
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=width)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.spectral(x) + self.skip(x))
        return self.act(y + self.mlp(y))


class TNOBlock(nn.Module):
    def __init__(self, width: int, modes_t: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_t, modes_x, modes_y)
        self.skip = nn.Conv3d(width, width, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, width, kernel_size=1),
        )
        self.norm = nn.GroupNorm(num_groups=max(1, min(8, width)), num_channels=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.skip(x)
        b, c, t, h, w = y.shape
        y = self.norm(y.reshape(b, c, t * h, w)).reshape(b, c, t, h, w)
        return y + self.ffn(y)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w.view(x.shape[0], x.shape[1], 1, 1)


class ImprovedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        groups = out_channels
        while groups > 1 and out_channels % groups != 0:
            groups //= 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, min(32, groups)), num_channels=out_channels)
        self.se = SEBlock(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise(self.depthwise(x))
        x = self.norm(x)
        x = self.se(x)
        x = self.drop(x)
        return self.act(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, feat: Sequence[int]) -> None:
        super().__init__()
        f = list(feat)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc00 = ImprovedConvBlock(in_channels, f[0])
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
        self.outs = nn.ModuleList([nn.Conv2d(f[0], out_channels, kernel_size=1) for _ in range(4)])

    @staticmethod
    def _upsample_and_cat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, y], dim=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x00 = self.enc00(x)
        x10 = self.enc10(self.pool(x00))
        x20 = self.enc20(self.pool(x10))
        x30 = self.enc30(self.pool(x20))
        x40 = self.enc40(self.pool(x30))
        x01 = self.dec01(self._upsample_and_cat(x10, x00))
        x11 = self.dec11(self._upsample_and_cat(x20, x10))
        x21 = self.dec21(self._upsample_and_cat(x30, x20))
        x31 = self.dec31(self._upsample_and_cat(x40, x30))
        x02 = self.dec02(self._upsample_and_cat(x11, torch.cat([x00, x01], dim=1)))
        x12 = self.dec12(self._upsample_and_cat(x21, torch.cat([x10, x11], dim=1)))
        x22 = self.dec22(self._upsample_and_cat(x31, torch.cat([x20, x21], dim=1)))
        x03 = self.dec03(self._upsample_and_cat(x12, torch.cat([x00, x01, x02], dim=1)))
        x13 = self.dec13(self._upsample_and_cat(x22, torch.cat([x10, x11, x12], dim=1)))
        x04 = self.dec04(self._upsample_and_cat(x13, torch.cat([x00, x01, x02, x03], dim=1)))
        return [self.outs[0](x04), self.outs[1](x03), self.outs[2](x02), self.outs[3](x01)]


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = max(1, min(16, out_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = ConvGNAct(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvGNAct(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ScaleTemporalFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(out_channels, min(in_channels, out_channels * 2))
        groups = max(1, min(16, hidden))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, min(16, out_channels)), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UFFNOEncoder(nn.Module):
    def __init__(self, input_channels: int, feat: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = [int(v) for v in feat]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc0 = EncoderBlock(input_channels, f0, dropout=dropout)
        self.enc1 = EncoderBlock(f0, f1, dropout=dropout)
        self.enc2 = EncoderBlock(f1, f2, dropout=dropout)
        self.enc3 = EncoderBlock(f2, f3, dropout=dropout)
        self.enc4 = EncoderBlock(f3, f4, dropout=dropout)

    def forward(self, x: torch.Tensor):
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.enc4(self.pool(s3))
        return s0, s1, s2, s3, b


@register_model("convlstm")
class ConvLSTMModel(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12, hidden_dim: int = 128, width: int | None = None, depth: int = 2, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        width = int(width or hidden_dim)
        self.step_embed = nn.Sequential(
            nn.Conv2d(input_channels + 2, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cells = nn.ModuleList([ConvLSTMCell(width, width) for _ in range(int(depth))])
        self.main_head = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux0 = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux1 = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux2 = nn.Conv2d(width, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, _, h, w = x.shape
        states = None
        last = None
        for i in range(t):
            grid = self.make_coord_grid(b, h, w, x.device, x.dtype)
            step = self.step_embed(torch.cat([x[:, i], grid], dim=1))
            if states is None:
                states = [cell.init_state(b, h, w, x.device, step.dtype) for cell in self.cells]
            new_states = []
            current = step
            for cell, state in zip(self.cells, states):
                h_s, c_s = cell(current, state)
                new_states.append((h_s, c_s))
                current = h_s
            states = new_states
            last = current
        assert last is not None
        main = self.main_head(last)
        # Auxiliary outputs must stay at the same spatial resolution as 'main'
        # because the legacy deep_supervised_loss applies a full-resolution
        # spatial_weight to every output head.
        aux0 = self.aux0(last)
        aux1 = self.aux1(last)
        aux2 = self.aux2(last)
        return [main, aux0, aux1, aux2]


@register_model("unet_lstm")
@register_model("unet_lstm_baseline")
class UNetLSTMModel(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12, hidden_dim: int = 128, feat: Sequence[int] | None = None, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        feat = feat or [32, 64, 128, 256, 512]
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.temporal = ConvLSTMCell(output_channels, hidden_dim)
        self.final_drop = nn.Dropout2d(p=dropout)
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, _, h, w = x.shape
        h_s = c_s = None
        last_unet = None
        for i in range(t):
            last_unet = self.unet(x[:, i])
            if h_s is None or c_s is None:
                h_s, c_s = self.temporal.init_state(b, h, w, x.device, x.dtype)
            h_s, c_s = self.temporal(last_unet[0], (h_s, c_s))
        assert last_unet is not None
        main = self.head(self.final_drop(h_s))
        return [main] + list(last_unet)


class TemporalReducer(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FNOBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, depth: int, modes_x: int, modes_y: int, dropout: float, factorized: bool = False) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.blocks = nn.ModuleList([FourierBlock(width, modes_x, modes_y, dropout=dropout, factorized=factorized) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )
        self.aux_heads = nn.ModuleList([nn.Conv2d(width, out_channels, kernel_size=1) for _ in range(3)])

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> List[torch.Tensor]:
        h = self.lift(torch.cat([x, grid], dim=1))
        hidden_states = []
        for block in self.blocks:
            h = block(h)
            hidden_states.append(h)
        main = self.head(h)
        feat_list = hidden_states[-3:] if len(hidden_states) >= 3 else [h, h, h]
        # Auxiliary outputs must stay at the same spatial resolution as 'main'
        # because the legacy deep_supervised_loss applies a full-resolution
        # spatial_weight to every output head.
        aux_outs: List[torch.Tensor] = []
        for idx, feat in enumerate(reversed(feat_list)):
            aux_outs.append(self.aux_heads[idx](feat))
        return [main] + aux_outs


class FNOSequenceModel(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12, width: int = 64, depth: int = 4, modes_x: int = 24, modes_y: int = 24, dropout: float = 0.1, factorized: bool = False, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        self.reducer = TemporalReducer(input_channels * seq_length, width)
        self.backbone = FNOBackbone(width, output_channels, width, depth, modes_x, modes_y, dropout, factorized=factorized)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        enc = self.reducer(x.reshape(b, t * c, h, w))
        grid = self.make_coord_grid(b, h, w, x.device, enc.dtype)
        return self.backbone(enc, grid)


@register_model("fno")
class FNO2dModel(FNOSequenceModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(factorized=False, **kwargs)


@register_model("ffno")
class FFNO2dModel(FNOSequenceModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(factorized=True, **kwargs)


@register_model("tno")
class TNOModel(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12, width: int = 64, depth: int = 4, modes_x: int = 24, modes_y: int = 24, modes_t: int | None = None, use_checkpoint: bool = False, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        modes_t = int(modes_t or min(8, seq_length))
        self.use_checkpoint = bool(use_checkpoint)
        self.lift = nn.Conv3d(input_channels + 2, width, kernel_size=1)
        self.blocks = nn.ModuleList([TNOBlock(width, modes_t, modes_x, modes_y) for _ in range(depth)])
        self.main_head = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux0 = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux1 = nn.Conv2d(width, output_channels, kernel_size=1)
        self.aux2 = nn.Conv2d(width, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        grid = self.make_coord_grid(b * t, h, w, x.device, x.dtype).reshape(b, t, 2, h, w)
        x3 = torch.cat([x, grid], dim=2).permute(0, 2, 1, 3, 4).contiguous()
        h3 = self.lift(x3)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h3 = _grad_checkpoint(block, h3, use_reentrant=False)
            else:
                h3 = block(h3)
        last = h3[:, :, -1]
        main = self.main_head(last)
        # Auxiliary outputs must stay at the same spatial resolution as 'main'
        # because the legacy deep_supervised_loss applies a full-resolution
        # spatial_weight to every output head.
        aux0 = self.aux0(last)
        aux1 = self.aux1(last)
        aux2 = self.aux2(last)
        return [main, aux0, aux1, aux2]


@register_model("u_ffno")
class UFFNOModel(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12, feat: Sequence[int] | None = None, width: int = 256, depth: int = 4, unet_feat: Sequence[int] | None = None, fno_width: int | None = None, fno_depth: int | None = None, modes_x: int = 16, modes_y: int = 16, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        feat = list(unet_feat or feat or (64, 128, 256, 512, 1024))
        width = int(fno_width or width)
        depth = int(fno_depth or depth)
        self.encoder = UFFNOEncoder(input_channels=input_channels, feat=feat, dropout=dropout)
        f0, f1, f2, f3, f4 = feat
        self.fuse0 = ScaleTemporalFusion(f0 * seq_length, f0, dropout=dropout)
        self.fuse1 = ScaleTemporalFusion(f1 * seq_length, f1, dropout=dropout)
        self.fuse2 = ScaleTemporalFusion(f2 * seq_length, f2, dropout=dropout)
        self.fuse3 = ScaleTemporalFusion(f3 * seq_length, f3, dropout=dropout)
        self.fuseb = ScaleTemporalFusion(f4 * seq_length, f4, dropout=dropout)
        self.latent_in = nn.Conv2d(f4 + 2, width, kernel_size=1)
        self.ffno_blocks = nn.Sequential(*[FourierBlock(width, modes_x, modes_y, dropout=dropout, factorized=True) for _ in range(depth)])
        self.latent_out = nn.Conv2d(width, f4, kernel_size=1)
        self.dec3 = DecoderBlock(f4, f3, f3, dropout=dropout)
        self.dec2 = DecoderBlock(f3, f2, f2, dropout=dropout)
        self.dec1 = DecoderBlock(f2, f1, f1, dropout=dropout)
        self.dec0 = DecoderBlock(f1, f0, f0, dropout=dropout)
        self.main_head = nn.Conv2d(f0, output_channels, kernel_size=1)
        self.aux0_head = nn.Conv2d(f0, output_channels, kernel_size=1)
        self.aux1_head = nn.Conv2d(f1, output_channels, kernel_size=1)
        self.aux2_head = nn.Conv2d(f2, output_channels, kernel_size=1)

    @staticmethod
    def _flatten_time(features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=1)

    def _encode_sequence(self, x: torch.Tensor):
        seq_s0: List[torch.Tensor] = []
        seq_s1: List[torch.Tensor] = []
        seq_s2: List[torch.Tensor] = []
        seq_s3: List[torch.Tensor] = []
        seq_b: List[torch.Tensor] = []
        for t in range(x.shape[1]):
            s0, s1, s2, s3, b = self.encoder(x[:, t])
            seq_s0.append(s0)
            seq_s1.append(s1)
            seq_s2.append(s2)
            seq_s3.append(s3)
            seq_b.append(b)
        return (
            self.fuse0(self._flatten_time(seq_s0)),
            self.fuse1(self._flatten_time(seq_s1)),
            self.fuse2(self._flatten_time(seq_s2)),
            self.fuse3(self._flatten_time(seq_s3)),
            self.fuseb(self._flatten_time(seq_b)),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b = x.shape[0]
        s0, s1, s2, s3, bt = self._encode_sequence(x)
        grid = self.make_coord_grid(b, bt.shape[-2], bt.shape[-1], x.device, bt.dtype)
        latent = self.latent_in(torch.cat([bt, grid], dim=1))
        latent = self.ffno_blocks(latent)
        latent = self.latent_out(latent)
        y3 = self.dec3(latent, s3)
        y2 = self.dec2(y3, s2)
        y1 = self.dec1(y2, s1)
        y0 = self.dec0(y1, s0)
        main = self.main_head(y0)
        # All auxiliary outputs must be at the same spatial resolution as 'main'
        # because the legacy deep_supervised_loss applies a full-resolution
        # spatial_weight to every output head.
        aux0 = self.aux0_head(y0)
        aux1 = self.aux1_head(F.interpolate(y1, size=y0.shape[-2:], mode="bilinear", align_corners=False))
        aux2 = self.aux2_head(F.interpolate(y2, size=y0.shape[-2:], mode="bilinear", align_corners=False))
        return [main, aux0, aux1, aux2]


# =============================================================================
# Swin Transformer (windowed attention) - transformer-family benchmark entry
# =============================================================================
# A memory-safe spatiotemporal emulator. Time is folded into the patch-embedding
# channels (same idea as the FNO/FFNO temporal reducer), then a Swin-UNet
# encoder-decoder with shifted-window attention maps the field back to full
# resolution. Window attention keeps the cost linear in the number of tokens
# instead of quadratic, so the 261 x 256 grid fits comfortably on one GPU.
# Dynamic padding inside every block handles non-power-of-two grids, so no
# wasteful global padding of the input is needed.
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # (B, H, W, C) -> (num_windows * B, window_size, window_size, C)
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    # (num_windows * B, window_size, window_size, C) -> (B, H, W, C)
    n_windows = (h // window_size) * (w // window_size)
    b = windows.shape[0] // max(1, n_windows)
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        rel = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bn, n, c = x.shape
        qkv = self.qkv(x).reshape(bn, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0).to(attn.dtype)
        if mask is not None:
            n_win = mask.shape[0]
            attn = attn.view(bn // n_win, n_win, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0).to(attn.dtype)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj_drop(self.proj(out))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def _build_mask(self, hp: int, wp: int, device: torch.device) -> torch.Tensor:
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, hp, wp, 1), device=device)
        slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hsl in slices:
            for wsl in slices:
                img_mask[:, hsl, wsl, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        ws = self.window_size
        pad_b = (ws - h % ws) % ws
        pad_r = (ws - w % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = h + pad_b, w + pad_r
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._build_mask(hp, wp, x.device)
        else:
            attn_mask = None
        windows = window_partition(x, ws).view(-1, ws * ws, c)
        attn_windows = self.attn(windows, mask=attn_mask).view(-1, ws, ws, c)
        x = window_reverse(attn_windows, ws, hp, wp)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_b or pad_r:
            x = x[:, :h, :w, :].contiguous()
        x = shortcut + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int]:
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        hn, wn = h // 2, w // 2
        x = x.view(b, hn * wn, 4 * c)
        return self.reduction(self.norm(x)), hn, wn


class PatchExpand(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int]:
        b, _, c = x.shape
        x = self.expand(x).view(b, h, w, 2, 2, c // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 2, w * 2, c // 2)
        hn, wn = h * 2, w * 2
        return self.norm(x.view(b, hn * wn, c // 2)), hn, wn


class BasicLayer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = _grad_checkpoint(blk, x, h, w, use_reentrant=False)
            else:
                x = blk(x, h, w)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        ps = self.patch_size
        pad_b = (ps - h % ps) % ps
        pad_r = (ps - w % ps) % ps
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.proj(x)
        hp, wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), hp, wp


@register_model("swin")
class SwinUNetEmulator(BaseWaveEmulator):
    """Shifted-window transformer with a Swin-UNet encoder-decoder.

    Output contract matches every other benchmark model: a list of four
    full-resolution heads [main, aux0, aux1, aux2], each (B, output_channels, H, W),
    so the legacy deep-supervised loss applies unchanged.
    """

    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12,
                 embed_dim: int = 96, depths: Sequence[int] = (2, 2, 6, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24), window_size: int = 8,
                 patch_size: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_drop: float = 0.0, use_checkpoint: bool = False, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        depths = [int(d) for d in depths]
        num_heads = [int(h) for h in num_heads]
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have equal length")
        self.num_layers = len(depths)
        self.window_size = int(window_size)
        self.patch_size = int(patch_size)
        embed_dim = int(embed_dim)
        in_chans = input_channels * seq_length + 2  # folded time + 2 coordinate channels
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.pos_drop = nn.Dropout(dropout)

        dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.enc_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_layers):
            self.enc_layers.append(BasicLayer(
                dims[i], depths[i], num_heads[i], self.window_size,
                mlp_ratio, dropout, attn_drop, use_checkpoint))
            if i < self.num_layers - 1:
                self.downsamples.append(PatchMerging(dims[i]))

        self.expands = nn.ModuleList()
        self.concat_reduce = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            self.expands.append(PatchExpand(dims[i]))
            self.concat_reduce.append(nn.Linear(2 * dims[i - 1], dims[i - 1]))
            self.dec_layers.append(BasicLayer(
                dims[i - 1], depths[i - 1], num_heads[i - 1], self.window_size,
                mlp_ratio, dropout, attn_drop, use_checkpoint))

        self.norm = nn.LayerNorm(embed_dim)
        self.head_main = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux0 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux1 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux2 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)

    @staticmethod
    def _resize_tokens(tok: torch.Tensor, h: int, w: int, th: int, tw: int) -> torch.Tensor:
        b, _, c = tok.shape
        x = tok.transpose(1, 2).reshape(b, c, h, w)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        grid = self.make_coord_grid(b, h, w, x.device, x.dtype)
        folded = torch.cat([x.reshape(b, t * c, h, w), grid], dim=1)
        tok, cur_h, cur_w = self.patch_embed(folded)
        tok = self.pos_drop(tok)

        skips: List[torch.Tensor] = []
        res: List[Tuple[int, int]] = []
        for i in range(self.num_layers):
            tok = self.enc_layers[i](tok, cur_h, cur_w)
            skips.append(tok)
            res.append((cur_h, cur_w))
            if i < self.num_layers - 1:
                tok, cur_h, cur_w = self.downsamples[i](tok, cur_h, cur_w)

        for j, i in enumerate(range(self.num_layers - 1, 0, -1)):
            tok, cur_h, cur_w = self.expands[j](tok, cur_h, cur_w)
            sh, sw = res[i - 1]
            if (cur_h, cur_w) != (sh, sw):
                tok = self._resize_tokens(tok, cur_h, cur_w, sh, sw)
                cur_h, cur_w = sh, sw
            tok = self.concat_reduce[j](torch.cat([tok, skips[i - 1]], dim=-1))
            tok = self.dec_layers[j](tok, cur_h, cur_w)

        tok = self.norm(tok)
        feat = tok.transpose(1, 2).reshape(b, -1, cur_h, cur_w)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return [self.head_main(feat), self.head_aux0(feat),
                self.head_aux1(feat), self.head_aux2(feat)]


# =============================================================================
# Additional architectures: ViT, ConvNeXt-LSTM, Conv-Swin hybrid
# =============================================================================
# All three keep the shared output contract: forward takes x of shape
# (B, T, C, H, W) and returns a list of four full-resolution heads
# [main, aux0, aux1, aux2], each (B, output_channels, H, W). The legacy
# deep-supervised loss and the rest of the benchmark see the same interface
# as every other model, so the comparison stays controlled.


def _align_to(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize src to match the spatial size of ref. Removes off-by-one
    mismatches that arise from striding an odd grid such as 261 x 256."""
    if src.shape[-2:] != ref.shape[-2:]:
        src = F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return src


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm for (B, C, H, W) tensors, as used by ConvNeXt."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


# -----------------------------------------------------------------------------
# Vision Transformer (plain ViT). A reviewer asked explicitly for a ViT
# comparison. Time is folded into the channel dimension and a large patch size
# keeps the token count small, so global self-attention stays well within the
# memory budget (a 261 x 256 grid at patch 16 gives only a few hundred tokens).
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


@register_model("vit")
class ViTEmulator(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12,
                 embed_dim: int = 384, depth: int = 8, num_heads: int = 6, patch_size: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, attn_drop: float = 0.0,
                 ref_h: int = 261, ref_w: int = 256, use_checkpoint: bool = False, **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_checkpoint = bool(use_checkpoint)
        in_chans = input_channels * seq_length + 2
        self.patch_embed = PatchEmbed(in_chans, embed_dim, self.patch_size)
        gh = math.ceil(ref_h / self.patch_size)
        gw = math.ceil(ref_w / self.patch_size)
        self.base_grid = (gh, gw)
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_drop)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_main = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux0 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux1 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)
        self.head_aux2 = nn.Conv2d(embed_dim, output_channels, kernel_size=1)

    def _resized_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        if (gh, gw) == self.base_grid:
            return self.pos_embed
        bgh, bgw = self.base_grid
        pe = self.pos_embed.reshape(1, bgh, bgw, self.embed_dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(gh, gw), mode="bilinear", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, gh * gw, self.embed_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        grid = self.make_coord_grid(b, h, w, x.device, x.dtype)
        folded = torch.cat([x.reshape(b, t * c, h, w), grid], dim=1)
        tok, gh, gw = self.patch_embed(folded)
        tok = self.pos_drop(tok + self._resized_pos_embed(gh, gw))
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                tok = _grad_checkpoint(blk, tok, use_reentrant=False)
            else:
                tok = blk(tok)
        tok = self.norm(tok)
        feat = tok.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return [self.head_main(feat), self.head_aux0(feat),
                self.head_aux1(feat), self.head_aux2(feat)]


# -----------------------------------------------------------------------------
# ConvNeXt-LSTM. A modern convolutional backbone (large depthwise kernels,
# layer norm, inverted bottleneck) extracts per-frame features; a ConvLSTM at
# the coarsest scale carries the temporal state. Fully convolutional, so memory
# scales with grid area rather than with the square of the token count.
# -----------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        return inp + self.drop(x)


def _convnext_stage(dim: int, depth: int) -> nn.Sequential:
    return nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])


@register_model("convnext_lstm")
class ConvNeXtLSTMEmulator(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12,
                 dims: Sequence[int] = (64, 128, 256), depths: Sequence[int] = (2, 2, 2),
                 lstm_hidden: int = 256, dropout: float = 0.0, use_checkpoint: bool = False,
                 **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        dims = [int(d) for d in dims]
        depths = [int(d) for d in depths]
        if len(dims) != 3 or len(depths) != 3:
            raise ValueError("ConvNeXt-LSTM expects three encoder stages")
        self.use_checkpoint = bool(use_checkpoint)
        in_chans = input_channels + 2  # per-frame input plus coordinate channels

        # Encoder: stem (/2) then three stages with /2 downsampling between them.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm2d(dims[0]),
        )
        self.stage0 = _convnext_stage(dims[0], depths[0])
        self.down1 = nn.Sequential(LayerNorm2d(dims[0]),
                                   nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))
        self.stage1 = _convnext_stage(dims[1], depths[1])
        self.down2 = nn.Sequential(LayerNorm2d(dims[1]),
                                   nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.stage2 = _convnext_stage(dims[2], depths[2])

        self.lstm_hidden = int(lstm_hidden)
        self.convlstm = ConvLSTMCell(dims[2], self.lstm_hidden, kernel_size=3)

        # Decoder: upsample back to full resolution with skip connections taken
        # from the final frame's encoder features.
        self.up2 = nn.Conv2d(self.lstm_hidden, dims[1], kernel_size=3, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(dims[1] * 2, dims[1], kernel_size=3, padding=1),
                                  nn.GELU(), ConvNeXtBlock(dims[1]))
        self.up1 = nn.Conv2d(dims[1], dims[0], kernel_size=3, padding=1)
        self.dec0 = nn.Sequential(nn.Conv2d(dims[0] * 2, dims[0], kernel_size=3, padding=1),
                                  nn.GELU(), ConvNeXtBlock(dims[0]))
        self.head_main = nn.Conv2d(dims[0], output_channels, kernel_size=1)
        self.head_aux0 = nn.Conv2d(dims[0], output_channels, kernel_size=1)
        self.head_aux1 = nn.Conv2d(dims[0], output_channels, kernel_size=1)
        self.head_aux2 = nn.Conv2d(dims[0], output_channels, kernel_size=1)

    def _encode_frame(self, xt: torch.Tensor):
        s = self.stem(xt)
        e0 = self.stage0(s)
        e1 = self.stage1(self.down1(e0))
        e2 = self.stage2(self.down2(e1))
        return e0, e1, e2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        grid = self.make_coord_grid(b, h, w, x.device, x.dtype)
        e2_seq = []
        e0_last = e1_last = None
        for ti in range(t):
            xt = torch.cat([x[:, ti], grid], dim=1)
            if self.use_checkpoint and self.training:
                e0, e1, e2 = _grad_checkpoint(self._encode_frame, xt, use_reentrant=False)
            else:
                e0, e1, e2 = self._encode_frame(xt)
            e2_seq.append(e2)
            e0_last, e1_last = e0, e1
        hh, ww = e2_seq[0].shape[-2:]
        state = self.convlstm.init_state(b, hh, ww, x.device, x.dtype)
        for ti in range(t):
            state = self.convlstm(e2_seq[ti], state)
        hid = state[0]

        u2 = F.interpolate(hid, size=e1_last.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        u2 = self.dec1(torch.cat([u2, e1_last], dim=1))
        u1 = F.interpolate(u2, size=e0_last.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        u1 = self.dec0(torch.cat([u1, e0_last], dim=1))
        feat = F.interpolate(u1, size=(h, w), mode="bilinear", align_corners=False)
        return [self.head_main(feat), self.head_aux0(feat),
                self.head_aux1(feat), self.head_aux2(feat)]


# -----------------------------------------------------------------------------
# Conv-Swin hybrid (UNet encoder-decoder with a shifted-window transformer at
# the bottleneck). Convolutions handle local, bathymetry-driven detail; the
# Swin block at the coarsest scale supplies global context for long-range swell.
# Attention runs only at the downsampled bottleneck, so its cost is small.
# -----------------------------------------------------------------------------
def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.GELU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), nn.GELU(),
    )


@register_model("conv_swin")
class ConvSwinUNetEmulator(BaseWaveEmulator):
    def __init__(self, input_channels: int = 6, output_channels: int = 4, seq_length: int = 12,
                 base_width: int = 48, swin_dim: int = 256, swin_depth: int = 4,
                 swin_heads: int = 8, window_size: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_drop: float = 0.0, use_checkpoint: bool = False,
                 **kwargs) -> None:
        super().__init__(input_channels, output_channels, seq_length)
        bw = int(base_width)
        self.swin_dim = int(swin_dim)
        in_chans = input_channels * seq_length + 2

        self.enc0 = _conv_block(in_chans, bw)
        self.down1 = nn.Conv2d(bw, bw, kernel_size=2, stride=2)
        self.enc1 = _conv_block(bw, bw * 2)
        self.down2 = nn.Conv2d(bw * 2, bw * 2, kernel_size=2, stride=2)
        self.enc2 = _conv_block(bw * 2, bw * 4)
        self.down3 = nn.Conv2d(bw * 4, bw * 4, kernel_size=2, stride=2)
        self.enc3 = _conv_block(bw * 4, bw * 8)

        self.to_tokens = nn.Conv2d(bw * 8, self.swin_dim, kernel_size=1)
        self.tok_norm = nn.LayerNorm(self.swin_dim)
        self.swin = BasicLayer(self.swin_dim, int(swin_depth), int(swin_heads),
                               int(window_size), mlp_ratio, dropout, attn_drop, use_checkpoint)
        self.from_tokens = nn.Conv2d(self.swin_dim, bw * 8, kernel_size=1)

        self.up3 = nn.Conv2d(bw * 8, bw * 4, kernel_size=3, padding=1)
        self.dec3 = _conv_block(bw * 8, bw * 4)
        self.up2 = nn.Conv2d(bw * 4, bw * 2, kernel_size=3, padding=1)
        self.dec2 = _conv_block(bw * 4, bw * 2)
        self.up1 = nn.Conv2d(bw * 2, bw, kernel_size=3, padding=1)
        self.dec1 = _conv_block(bw * 2, bw)
        self.head_main = nn.Conv2d(bw, output_channels, kernel_size=1)
        self.head_aux0 = nn.Conv2d(bw, output_channels, kernel_size=1)
        self.head_aux1 = nn.Conv2d(bw, output_channels, kernel_size=1)
        self.head_aux2 = nn.Conv2d(bw, output_channels, kernel_size=1)

    def _upcat(self, x: torch.Tensor, skip: torch.Tensor, up: nn.Module, dec: nn.Module) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = up(x)
        return dec(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, t, c, h, w = x.shape
        grid = self.make_coord_grid(b, h, w, x.device, x.dtype)
        folded = torch.cat([x.reshape(b, t * c, h, w), grid], dim=1)

        e0 = self.enc0(folded)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))

        bh, bw_ = e3.shape[-2:]
        tok = self.to_tokens(e3).flatten(2).transpose(1, 2)
        tok = self.tok_norm(tok)
        tok = self.swin(tok, bh, bw_)
        bb = tok.transpose(1, 2).reshape(b, self.swin_dim, bh, bw_)
        bb = self.from_tokens(bb)

        d3 = self._upcat(bb, e2, self.up3, self.dec3)
        d2 = self._upcat(d3, e1, self.up2, self.dec2)
        d1 = self._upcat(d2, e0, self.up1, self.dec1)
        feat = _align_to(d1, x[:, 0])
        return [self.head_main(feat), self.head_aux0(feat),
                self.head_aux1(feat), self.head_aux2(feat)]


# =============================================================================
# Distributed and legacy patch helpers
# =============================================================================

def ddp_info() -> Dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_rank0": rank == 0,
    }


def init_distributed(backend: str = "nccl") -> Dict[str, Any]:
    info = ddp_info()
    if info["world_size"] > 1 and not dist.is_initialized():
        torch.cuda.set_device(info["local_rank"])
        dist.init_process_group(backend=backend, init_method="env://")
    return ddp_info()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank_workdir(run_dir: Path, rank: int) -> Path:
    return run_dir if rank == 0 else run_dir / f"rank{rank:02d}"


def derive_run_dir(job: Dict[str, Any]) -> Path:
    """Build the per-run output directory path from a job spec. Centralized so
    the coordinator and worker agree on the same folder, which lets the stdout
    and stderr logs live alongside the worker log and checkpoints."""
    hp = job["hyperparams"]
    run_name = (
        f"{job['stage']}_{job['model']}_{job['config_id']}_seed{job['seed']}_"
        f"seq{hp.get('seq_length', CONFIG['seq_length'])}_"
        f"lr{hp.get('max_lr', 1e-4):.0e}_wd{hp.get('weight_decay', 1e-4):.0e}"
    )
    return Path(CONFIG["results_root"]) / run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_module_from_path(module_name: str, file_path: str):
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LegacyCompatibleBenchmarkModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 128,
        feat: Optional[Sequence[int]] = None,
        seq_length: int = 12,
        fno_width: int = 64,
        fno_depth: int = 4,
        modes_x: int = 24,
        modes_y: int = 24,
        modes_t: int = 6,
        width: int | None = None,
        depth: int | None = None,
        embed_dim: int = 96,
        swin_depths: Optional[Sequence[int]] = None,
        swin_num_heads: Optional[Sequence[int]] = None,
        window_size: int = 8,
        patch_size: int = 4,
        use_checkpoint: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        extra = dict(extra or {})
        kwargs: Dict[str, Any] = {
            "input_channels": input_channels,
            "output_channels": output_channels,
            "seq_length": seq_length,
        }
        if model_name in {"fno", "ffno"}:
            kwargs.update({
                "width": fno_width,
                "depth": fno_depth,
                "modes_x": modes_x,
                "modes_y": modes_y,
            })
        elif model_name == "tno":
            kwargs.update({
                "width": width or hidden_dim,
                "depth": depth or fno_depth,
                "modes_x": modes_x,
                "modes_y": modes_y,
                "modes_t": modes_t,
                "use_checkpoint": use_checkpoint,
            })
        elif model_name == "u_ffno":
            kwargs.update({
                "unet_feat": feat,
                "hidden_dim": hidden_dim,
                "fno_width": fno_width,
                "fno_depth": fno_depth,
                "modes_x": modes_x,
                "modes_y": modes_y,
            })
        elif model_name == "swin":
            kwargs.update({
                "embed_dim": embed_dim,
                "depths": list(swin_depths) if swin_depths else [2, 2, 6, 2],
                "num_heads": list(swin_num_heads) if swin_num_heads else [3, 6, 12, 24],
                "window_size": window_size,
                "patch_size": patch_size,
                "use_checkpoint": use_checkpoint,
            })
        elif model_name == "vit":
            kwargs.update({
                "embed_dim": int(extra.get("embed_dim", embed_dim)),
                "depth": int(extra.get("vit_depth", 8)),
                "num_heads": int(extra.get("vit_heads", 6)),
                "patch_size": int(extra.get("patch_size", patch_size if patch_size >= 8 else 16)),
                "use_checkpoint": use_checkpoint,
            })
        elif model_name == "convnext_lstm":
            kwargs.update({
                "dims": list(extra.get("convnext_dims", [64, 128, 256])),
                "depths": list(extra.get("convnext_depths", [2, 2, 2])),
                "lstm_hidden": int(extra.get("lstm_hidden", 256)),
                "use_checkpoint": use_checkpoint,
            })
        elif model_name == "conv_swin":
            kwargs.update({
                "base_width": int(extra.get("base_width", 48)),
                "swin_dim": int(extra.get("swin_dim", 256)),
                "swin_depth": int(extra.get("swin_depth", 4)),
                "swin_heads": int(extra.get("swin_heads", 8)),
                "window_size": int(extra.get("window_size", window_size)),
                "use_checkpoint": use_checkpoint,
            })
        elif model_name in {"convlstm"}:
            kwargs.update({
                "hidden_dim": hidden_dim,
                "width": width or hidden_dim,
                "depth": depth or fno_depth,
            })
        else:
            kwargs.update({"hidden_dim": hidden_dim, "feat": feat})
        self.model_name = model_name
        self.model = create_model(model_name, **kwargs)
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)


def install_legacy_model_patch(module, job: Dict[str, Any], local_rank: int) -> None:
    model_name = job["model"]
    hp = job["hyperparams"]
    seq_length = hp.get("seq_length", CONFIG["seq_length"])
    feat = hp.get("unet_feat")
    fno_width = hp.get("fno_width", 64)
    fno_depth = hp.get("fno_depth", 4)
    modes_x = hp.get("modes_x", 24)
    modes_y = hp.get("modes_y", 24)
    modes_t = hp.get("modes_t", 6)
    hidden_dim = hp.get("hidden_dim", 128)
    width = hp.get("width")
    depth = hp.get("depth")
    embed_dim = hp.get("embed_dim", hp.get("width", 96))
    swin_depths = hp.get("swin_depths")
    swin_num_heads = hp.get("swin_num_heads")
    window_size = hp.get("window_size", 8)
    patch_size = hp.get("patch_size", 4)
    use_checkpoint = bool(hp.get("use_checkpoint", False))
    # All model-specific hyperparameters (vit_depth, convnext_dims, base_width, ...)
    # ride along in this dict so new architectures plug in without widening the
    # constructor signature further.
    extra = dict(hp)

    class PatchedModel(LegacyCompatibleBenchmarkModel):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            hidden_dim: int = 128,
            feat=None,
            *args,
            **kwargs,
        ):
            # Accept legacy constructor keywords exactly as the original script uses them.
            # Also tolerate extra args/kwargs so the patch layer does not fail before training starts.
            hidden_dim_resolved = kwargs.pop("hidden_dim_", hidden_dim)
            feat_resolved = kwargs.pop("feat_", feat)
            super().__init__(
                model_name=model_name,
                input_channels=input_channels,
                output_channels=output_channels,
                hidden_dim=hidden_dim if hidden_dim is not None else hidden_dim_resolved,
                feat=feat if feat is not None else feat_resolved,
                seq_length=seq_length,
                fno_width=fno_width,
                fno_depth=fno_depth,
                modes_x=modes_x,
                modes_y=modes_y,
                modes_t=modes_t,
                width=width,
                depth=depth,
                embed_dim=embed_dim,
                swin_depths=swin_depths,
                swin_num_heads=swin_num_heads,
                window_size=window_size,
                patch_size=patch_size,
                use_checkpoint=use_checkpoint,
                extra=extra,
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.to(torch.device(f"cuda:{local_rank}"))

    module.UNetConvLSTM = PatchedModel


def patch_legacy_globals(module, job: Dict[str, Any], local_rank: int) -> None:
    hp = job["hyperparams"]
    feat = hp.get("unet_feat", [32, 64, 128, 256, 512])
    setattr(module, "BATCH_SIZE", int(hp.get("batch_size", 1)))
    setattr(module, "batch_size", int(hp.get("batch_size", 1)))
    setattr(module, "ACC_STEPS", int(hp.get("acc_steps", 1)))
    setattr(module, "acc_steps", int(hp.get("acc_steps", 1)))
    setattr(module, "max_lr", float(hp.get("max_lr", 1.0e-4)))
    setattr(module, "weight_decay", float(hp.get("weight_decay", 1.0e-4)))
    setattr(module, "LOCAL_RANK", local_rank)
    setattr(module, "device", torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    setattr(module, "DEVICE", torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    setattr(module, "bnd_direction", None)
    setattr(module, "BND_DIRS_BY_YEAR", {
        2019: CONFIG["bnd_dir_2019"],
        2020: CONFIG["bnd_dir_2020"],
        2021: CONFIG["bnd_dir_2021"],
    })
    setattr(module, "time_steps_list", [job.get("time_steps", CONFIG["time_steps"])])
    setattr(module, "seq_length_list", [hp.get("seq_length", CONFIG["seq_length"])])
    setattr(module, "epochs_list", [job.get("epochs", CONFIG["epochs"])])
    setattr(module, "hidden_dim_list", [hp.get("hidden_dim", 128)])
    setattr(module, "unet_feat_list", [feat])

    # Environment hints for boundary helpers.
    os.environ["SWAN_BND_DIR_2019"] = CONFIG["bnd_dir_2019"]
    os.environ["SWAN_BND_DIR_2020"] = CONFIG["bnd_dir_2020"]
    os.environ["SWAN_BND_DIR_2021"] = CONFIG["bnd_dir_2021"]
    os.environ["SWAN_STATION_ROOT"] = CONFIG.get(
        "station_root", str(Path(CONFIG["data_path"]).resolve().parent))
    os.environ["PYTHONPATH"] = f"{Path(CONFIG['legacy_train_script']).resolve().parent}:{os.environ.get('PYTHONPATH', '')}"

    # Optional helper module import path.
    legacy_parent = str(Path(CONFIG["legacy_train_script"]).resolve().parent)
    if legacy_parent not in sys.path:
        sys.path.insert(0, legacy_parent)


def patch_nonzero_rank_io(module, rank: int) -> None:
    if rank == 0:
        return
    try:
        import builtins
        _orig_print = builtins.print
        def _quiet_print(*args, **kwargs):
            if kwargs.pop("force", False):
                _orig_print(*args, **kwargs)
        builtins.print = _quiet_print
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.savefig = lambda *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None
    except Exception:
        pass


def make_distributed_peak_sampler_class(rank: int, world_size: int, seed: int):
    class DistributedPeakSamplerRestricted(torch.utils.data.Sampler[int]):
        """
        Compatibility wrapper for the legacy PeakSamplerRestricted.

        The legacy training script instantiates the sampler like:
            PeakSamplerRestricted(
                allowed_indices=idx_tr,
                wave_data=wave_data,
                seq_len=seq_length,
                pct=95,
                up_factor=2,
            )

        Earlier versions of this one-file launcher replaced that class with a
        sampler that required a positional `data_source`, which immediately
        crashed. This implementation accepts both styles and shards only the
        allowed indices across ranks.
        """
        def __init__(self, *args, **kwargs) -> None:
            self.seed = int(kwargs.pop("seed", seed))
            self.rank = rank
            self.world_size = world_size

            allowed = kwargs.pop("allowed_indices", None)
            wave_data = kwargs.pop("wave_data", None)
            data_source = kwargs.pop("data_source", None)

            # Backward-compatible positional parsing.
            if allowed is None and len(args) >= 1:
                first = args[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    allowed = list(first)
                else:
                    data_source = first
            if wave_data is None and len(args) >= 2:
                wave_data = args[1]

            if allowed is None:
                if data_source is not None:
                    allowed = list(range(len(data_source)))
                elif wave_data is not None:
                    allowed = list(range(len(wave_data)))
                else:
                    allowed = []

            self.allowed_indices = [int(x) for x in allowed]
            self._indices = self.allowed_indices[self.rank::self.world_size]

        def __iter__(self):
            rng = random.Random(self.seed)
            indices = list(self._indices)
            rng.shuffle(indices)
            return iter(indices)

        def __len__(self):
            return len(self._indices)

    return DistributedPeakSamplerRestricted


# =============================================================================
# Benchmark / summary helpers
# =============================================================================
@dataclass
class BenchmarkResult:
    params: int
    mean_latency_ms: float
    std_latency_ms: float
    max_memory_mb: float
    ar_total_latency_ms: float
    ar_rollout_steps: int


@torch.no_grad()
def benchmark_model(model_name: str, job: Dict[str, Any], device: torch.device) -> BenchmarkResult:
    hp = job["hyperparams"]
    seq_length = int(hp.get("seq_length", CONFIG["seq_length"]))
    input_channels = 10 if job.get("use_bnd", CONFIG["use_bnd"]) == "on" else 6
    output_channels = 4
    height, width = 261, 256
    kwargs: Dict[str, Any] = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "seq_length": seq_length,
    }
    if model_name in {"fno", "ffno"}:
        kwargs.update({
            "width": int(hp.get("fno_width", 64)),
            "depth": int(hp.get("fno_depth", 4)),
            "modes_x": int(hp.get("modes_x", 24)),
            "modes_y": int(hp.get("modes_y", 24)),
        })
    elif model_name == "tno":
        kwargs.update({
            "width": int(hp.get("width", hp.get("hidden_dim", 64))),
            "depth": int(hp.get("depth", hp.get("fno_depth", 4))),
            "modes_x": int(hp.get("modes_x", 24)),
            "modes_y": int(hp.get("modes_y", 24)),
            "modes_t": int(hp.get("modes_t", 6)),
        })
    elif model_name == "u_ffno":
        kwargs.update({
            "unet_feat": hp.get("unet_feat", [64, 128, 256, 512, 1024]),
            "hidden_dim": int(hp.get("hidden_dim", 256)),
            "fno_width": int(hp.get("fno_width", 256)),
            "fno_depth": int(hp.get("fno_depth", 4)),
            "modes_x": int(hp.get("modes_x", 16)),
            "modes_y": int(hp.get("modes_y", 16)),
        })
    elif model_name == "swin":
        kwargs.update({
            "embed_dim": int(hp.get("embed_dim", hp.get("width", 96))),
            "depths": hp.get("swin_depths", [2, 2, 6, 2]),
            "num_heads": hp.get("swin_num_heads", [3, 6, 12, 24]),
            "window_size": int(hp.get("window_size", 8)),
            "patch_size": int(hp.get("patch_size", 4)),
            # Report true inference cost: no gradient checkpointing here.
            "use_checkpoint": False,
        })
    elif model_name == "vit":
        kwargs.update({
            "embed_dim": int(hp.get("embed_dim", 384)),
            "depth": int(hp.get("vit_depth", 8)),
            "num_heads": int(hp.get("vit_heads", 6)),
            "patch_size": int(hp.get("patch_size", 16)),
            "use_checkpoint": False,
        })
    elif model_name == "convnext_lstm":
        kwargs.update({
            "dims": hp.get("convnext_dims", [64, 128, 256]),
            "depths": hp.get("convnext_depths", [2, 2, 2]),
            "lstm_hidden": int(hp.get("lstm_hidden", 256)),
            "use_checkpoint": False,
        })
    elif model_name == "conv_swin":
        kwargs.update({
            "base_width": int(hp.get("base_width", 48)),
            "swin_dim": int(hp.get("swin_dim", 256)),
            "swin_depth": int(hp.get("swin_depth", 4)),
            "swin_heads": int(hp.get("swin_heads", 8)),
            "window_size": int(hp.get("window_size", 8)),
            "use_checkpoint": False,
        })
    elif model_name == "convlstm":
        kwargs.update({
            "hidden_dim": int(hp.get("hidden_dim", hp.get("width", 128))),
            "width": int(hp.get("width", hp.get("hidden_dim", 128))),
            "depth": int(hp.get("depth", hp.get("fno_depth", 2))),
        })
    else:
        kwargs.update({
            "hidden_dim": int(hp.get("hidden_dim", 128)),
            "feat": hp.get("unet_feat", [32, 64, 128, 256, 512]),
        })
    model = create_model(model_name, **kwargs).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    dtype = torch.float32
    if device.type == "cuda":
        if CONFIG["benchmark_precision"] == "fp16":
            dtype = torch.float16
        elif CONFIG["benchmark_precision"] == "bf16":
            dtype = torch.bfloat16
    x = torch.randn(1, seq_length, input_channels, height, width, device=device, dtype=torch.float32)

    def _run_once() -> None:
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(x)
        else:
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(CONFIG["synthetic_benchmark_warmup"])):
        _run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latencies = []
    for _ in range(int(CONFIG["synthetic_benchmark_iters"])):
        start = time.perf_counter()
        _run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    ar_steps = 24
    start = time.perf_counter()
    for _ in range(ar_steps):
        _run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_mem = 0.0
    ar_total = (time.perf_counter() - start) * 1000.0
    return BenchmarkResult(
        params=params,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        max_memory_mb=float(peak_mem),
        ar_total_latency_ms=float(ar_total),
        ar_rollout_steps=ar_steps,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def parse_unet_feat(text: str | Sequence[int]) -> List[int]:
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("unet_feat cannot be empty")
    return values


def find_single_csv(run_dir: Path, prefix: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(f"{prefix}*.csv"))
    return matches[-1] if matches else None


def find_single_file(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    return matches[-1] if matches else None


def parse_legacy_summary_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[-1]
    out: Dict[str, Any] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except Exception:
            out[key] = value
    return out


# =============================================================================
# Search plan generation
# =============================================================================

def convlstm_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["convlstm"]
    idx = 0
    for width in space["widths"]:
        for depth in space["depths"]:
            for lr in space["lrs"]:
                for wd in space["wds"]:
                    batch_size = 4 if width <= 256 else 2
                    acc_steps = 1 if width <= 256 else 2
                    jobs.append({
                        "model": "convlstm",
                        "config_id": f"convlstm_focus_{idx:03d}",
                        "hyperparams": {
                            "seq_length": CONFIG["seq_length"],
                            "hidden_dim": width,
                            "width": width,
                            "depth": depth,
                            "max_lr": lr,
                            "weight_decay": wd,
                            "batch_size": batch_size,
                            "acc_steps": acc_steps,
                        },
                    })
                    idx += 1
    return jobs


def unet_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["unet_lstm"]
    idx = 0
    for feat in space["feat_variants"]:
        for hidden_dim in space["hidden_dims"]:
            for lr in space["lrs"]:
                for wd in space["wds"]:
                    jobs.append({
                        "model": "unet_lstm",
                        "config_id": f"unet_lstm_focus_{idx:03d}",
                        "hyperparams": {
                            "seq_length": CONFIG["seq_length"],
                            "hidden_dim": hidden_dim,
                            "unet_feat": feat,
                            "max_lr": lr,
                            "weight_decay": wd,
                            "batch_size": 1,
                            "acc_steps": 4,
                        },
                    })
                    idx += 1
    return jobs


def operator_jobs(model: str) -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"][model]
    idx = 0
    for width in space["widths"]:
        for depth in space["depths"]:
            local_modes = [tuple(x) for x in space["mode_pairs"]]

            # Dense FNO scales much more aggressively than FFNO.
            if model == "fno":
                if width >= 384 and depth >= 8:
                    local_modes = [(24, 24), (32, 32), (48, 48)]
                elif width >= 384:
                    local_modes = [(24, 24), (32, 32), (48, 48), (32, 64)]
                elif depth >= 8:
                    local_modes = [(24, 24), (32, 32), (48, 48)]
            else:
                if width >= 512 and depth >= 8:
                    local_modes = [(24, 24), (32, 32), (48, 48), (64, 64)]
                elif width >= 512:
                    local_modes = [(24, 24), (32, 32), (48, 48), (64, 64), (32, 64)]

            for modes_x, modes_y in local_modes:
                for lr in space["lrs"]:
                    for wd in space["wds"]:
                        if model == "fno":
                            if width >= 384 or depth >= 8 or max(modes_x, modes_y) >= 64:
                                batch_size = 1
                                acc_steps = 4
                            elif width >= 256 or max(modes_x, modes_y) >= 48:
                                batch_size = 2
                                acc_steps = 2
                            else:
                                batch_size = 4
                                acc_steps = 1
                        else:
                            if width >= 512 or depth >= 8 or max(modes_x, modes_y) >= 64:
                                batch_size = 1
                                acc_steps = 4
                            elif width >= 384 or max(modes_x, modes_y) >= 48:
                                batch_size = 2
                                acc_steps = 2
                            else:
                                batch_size = 4
                                acc_steps = 1
                        jobs.append({
                            "model": model,
                            "config_id": f"{model}_focus_{idx:03d}",
                            "hyperparams": {
                                "seq_length": CONFIG["seq_length"],
                                "hidden_dim": 256,
                                "fno_width": width,
                                "fno_depth": depth,
                                "modes_x": modes_x,
                                "modes_y": modes_y,
                                "max_lr": lr,
                                "weight_decay": wd,
                                "batch_size": batch_size,
                                "acc_steps": acc_steps,
                            },
                        })
                        idx += 1
    return jobs


def tno_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["tno"]
    idx = 0
    for width in space["widths"]:
        for depth in space["depths"]:
            for mh in space["modes_h"]:
                for mt in space["modes_t"]:
                    for mw in space["modes_w"]:
                        if width >= 256 or depth >= 6:
                            batch_size = 1
                            acc_steps = 4
                        elif width >= 128:
                            batch_size = 2
                            acc_steps = 2
                        else:
                            batch_size = 4
                            acc_steps = 1
                        for lr in space["lrs"]:
                            for wd in space["wds"]:
                                jobs.append({
                                    "model": "tno",
                                    "config_id": f"tno_focus_{idx:03d}",
                                    "hyperparams": {
                                        "seq_length": CONFIG["seq_length"],
                                        "hidden_dim": width,
                                        "width": width,
                                        "depth": depth,
                                        "modes_x": mh,
                                        "modes_y": mw,
                                        "modes_t": mt,
                                        # Gradient checkpointing clears the known TNO
                                        # out-of-memory failure on the full 3D spectral
                                        # path. Training only; reported efficiency comes
                                        # from the checkpoint-free synthetic benchmark.
                                        "use_checkpoint": True,
                                        "max_lr": lr,
                                        "weight_decay": wd,
                                        "batch_size": batch_size,
                                        "acc_steps": acc_steps,
                                    },
                                })
                                idx += 1
    return jobs


def uffno_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["u_ffno"]
    idx = 0
    for feat in space["feat_variants"]:
        for width in space["widths"]:
            for depth in space["depths"]:
                local_modes = [(8, 8), (12, 12), (16, 16)] if (width >= 512 and depth >= 8) else [tuple(x) for x in space["mode_pairs"]]
                for modes_x, modes_y in local_modes:
                    for lr in space["lrs"]:
                        for wd in space["wds"]:
                            jobs.append({
                                "model": "u_ffno",
                                "config_id": f"u_ffno_focus_{idx:03d}",
                                "hyperparams": {
                                    "seq_length": CONFIG["seq_length"],
                                    "unet_feat": feat,
                                    "hidden_dim": 256,
                                    "fno_width": width,
                                    "fno_depth": depth,
                                    "modes_x": modes_x,
                                    "modes_y": modes_y,
                                    "max_lr": lr,
                                    "weight_decay": wd,
                                    "batch_size": 1,
                                    "acc_steps": 4,
                                },
                            })
                            idx += 1
    return jobs


def swin_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["swin"]
    idx = 0
    for embed_dim in space["embed_dims"]:
        for variant in space["stage_variants"]:
            depths = list(variant["depths"])
            num_heads = list(variant["num_heads"])
            # Every stage halves the grid and doubles the channels and heads,
            # so dims[i] % num_heads[i] == embed_dim % num_heads[0]. Skip any
            # width that would not divide evenly into the attention heads.
            if embed_dim % num_heads[0] != 0:
                continue
            num_stages = len(depths)
            for window_size in space["window_sizes"]:
                for patch_size in space["patch_sizes"]:
                    if embed_dim >= 128 or (num_stages >= 4 and embed_dim >= 96):
                        batch_size, acc_steps = 1, 4
                    elif embed_dim >= 96 or num_stages >= 4:
                        batch_size, acc_steps = 2, 2
                    else:
                        batch_size, acc_steps = 4, 1
                    for lr in space["lrs"]:
                        for wd in space["wds"]:
                            jobs.append({
                                "model": "swin",
                                "config_id": f"swin_focus_{idx:03d}",
                                "hyperparams": {
                                    "seq_length": CONFIG["seq_length"],
                                    "embed_dim": embed_dim,
                                    "swin_depths": depths,
                                    "swin_num_heads": num_heads,
                                    "window_size": window_size,
                                    "patch_size": patch_size,
                                    # Gradient checkpointing keeps long unattended
                                    # runs inside the memory budget; it affects only
                                    # training, not the reported synthetic benchmark.
                                    "use_checkpoint": True,
                                    "max_lr": lr,
                                    "weight_decay": wd,
                                    "batch_size": batch_size,
                                    "acc_steps": acc_steps,
                                },
                            })
                            idx += 1
    return jobs


def vit_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["vit"]
    idx = 0
    for embed_dim in space["embed_dims"]:
        for depth in space["depths"]:
            for num_heads in space["num_heads"]:
                if embed_dim % num_heads != 0:
                    continue
                for patch_size in space["patch_sizes"]:
                    batch_size, acc_steps = (2, 2) if (embed_dim >= 384 or depth >= 8) else (4, 1)
                    for lr in space["lrs"]:
                        for wd in space["wds"]:
                            jobs.append({
                                "model": "vit",
                                "config_id": f"vit_focus_{idx:03d}",
                                "hyperparams": {
                                    "seq_length": CONFIG["seq_length"],
                                    "embed_dim": embed_dim,
                                    "vit_depth": depth,
                                    "vit_heads": num_heads,
                                    "patch_size": patch_size,
                                    "max_lr": lr,
                                    "weight_decay": wd,
                                    "batch_size": batch_size,
                                    "acc_steps": acc_steps,
                                },
                            })
                            idx += 1
    return jobs


def convnext_lstm_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["convnext_lstm"]
    idx = 0
    for variant in space["dim_variants"]:
        dims = list(variant["dims"])
        depths = list(variant["depths"])
        for lstm_hidden in space["lstm_hiddens"]:
            batch_size, acc_steps = (2, 2) if dims[-1] >= 384 else (4, 1)
            for lr in space["lrs"]:
                for wd in space["wds"]:
                    jobs.append({
                        "model": "convnext_lstm",
                        "config_id": f"convnext_lstm_focus_{idx:03d}",
                        "hyperparams": {
                            "seq_length": CONFIG["seq_length"],
                            "convnext_dims": dims,
                            "convnext_depths": depths,
                            "lstm_hidden": lstm_hidden,
                            # Gradient checkpointing trades compute for memory on
                            # the per-frame encoder; training only, results unchanged.
                            "use_checkpoint": True,
                            "max_lr": lr,
                            "weight_decay": wd,
                            "batch_size": batch_size,
                            "acc_steps": acc_steps,
                        },
                    })
                    idx += 1
    return jobs


def conv_swin_jobs() -> List[Dict[str, Any]]:
    jobs = []
    space = CONFIG["search_space"]["conv_swin"]
    idx = 0
    for base_width in space["base_widths"]:
        for swin_dim in space["swin_dims"]:
            for swin_depth in space["swin_depths"]:
                for swin_heads in space["swin_heads"]:
                    if swin_dim % swin_heads != 0:
                        continue
                    for window_size in space["window_sizes"]:
                        batch_size, acc_steps = (2, 2) if (base_width >= 64 or swin_dim >= 384) else (4, 1)
                        for lr in space["lrs"]:
                            for wd in space["wds"]:
                                jobs.append({
                                    "model": "conv_swin",
                                    "config_id": f"conv_swin_focus_{idx:03d}",
                                    "hyperparams": {
                                        "seq_length": CONFIG["seq_length"],
                                        "base_width": base_width,
                                        "swin_dim": swin_dim,
                                        "swin_depth": swin_depth,
                                        "swin_heads": swin_heads,
                                        "window_size": window_size,
                                        "use_checkpoint": True,
                                        "max_lr": lr,
                                        "weight_decay": wd,
                                        "batch_size": batch_size,
                                        "acc_steps": acc_steps,
                                    },
                                })
                                idx += 1
    return jobs


def generate_plan() -> List[Dict[str, Any]]:
    base_jobs: List[Dict[str, Any]] = []
    model_set = set(CONFIG["models_to_run"])
    if "convlstm" in model_set:
        base_jobs.extend(convlstm_jobs())
    if "unet_lstm" in model_set:
        base_jobs.extend(unet_jobs())
    if "fno" in model_set:
        base_jobs.extend(operator_jobs("fno"))
    if "ffno" in model_set:
        base_jobs.extend(operator_jobs("ffno"))
    if "tno" in model_set:
        base_jobs.extend(tno_jobs())
    if "u_ffno" in model_set:
        base_jobs.extend(uffno_jobs())
    if "swin" in model_set:
        base_jobs.extend(swin_jobs())
    if "vit" in model_set:
        base_jobs.extend(vit_jobs())
    if "convnext_lstm" in model_set:
        base_jobs.extend(convnext_lstm_jobs())
    if "conv_swin" in model_set:
        base_jobs.extend(conv_swin_jobs())

    jobs: List[Dict[str, Any]] = []
    for base in base_jobs:
        for seed in CONFIG["seed_list"]:
            jobs.append({
                "stage": "focused_all_onefile",
                "model": base["model"],
                "epochs": CONFIG["epochs"],
                "time_steps": CONFIG["time_steps"],
                "use_bnd": CONFIG["use_bnd"],
                "seed": int(seed),
                "gpus_per_trial": CONFIG["gpus_per_trial"],
                "config_id": base["config_id"],
                "hyperparams": base["hyperparams"],
            })

    # Multi-seed pass. For each selected model, take one representative config
    # (the middle of that model's generated list) and also run it at the extra
    # seeds. The config_id is unchanged, so the seed_list run and the extra-seed
    # runs share a config_id and group together for error bars. skip_existing
    # removes any overlap, so this is safe to run alongside or after the sweep.
    extra_seeds = [int(s) for s in CONFIG.get("extra_seeds", []) if int(s) not in CONFIG["seed_list"]]
    ms_models = set(CONFIG.get("multiseed_models", []))
    if extra_seeds and ms_models:
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for base in base_jobs:
            by_model.setdefault(base["model"], []).append(base)
        for model, blist in by_model.items():
            if model not in ms_models or not blist:
                continue
            rep = blist[len(blist) // 2]  # representative = middle config
            for seed in extra_seeds:
                jobs.append({
                    "stage": "focused_all_onefile",
                    "model": rep["model"],
                    "epochs": CONFIG["epochs"],
                    "time_steps": CONFIG["time_steps"],
                    "use_bnd": CONFIG["use_bnd"],
                    "seed": int(seed),
                    "gpus_per_trial": CONFIG["gpus_per_trial"],
                    "config_id": rep["config_id"],
                    "hyperparams": rep["hyperparams"],
                })
    return jobs


# =============================================================================
# Worker main
# =============================================================================

def worker_main(job_file: str) -> int:
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    repair.verify_job(job, _PACKAGE)
    os.environ['SWAN_REPAIR_JOB']=json.dumps(job)
    os.environ['SWAN_TRAIN_FRACTION']=str(job.get('train_fraction',1.))
    os.environ['SWAN_BND_DIR_TRANSFORM']=job.get('bnd_dir_transform','train_auto')
    dist_info = init_distributed("nccl")
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    rank0 = dist_info["is_rank0"]
    set_seed(int(job["seed"]) + rank)

    hp = job["hyperparams"]
    run_name = (
        f"{job['stage']}_{job['model']}_{job['config_id']}_seed{job['seed']}_"
        f"seq{hp.get('seq_length', CONFIG['seq_length'])}_"
        f"lr{hp.get('max_lr', 1e-4):.0e}_wd{hp.get('weight_decay', 1e-4):.0e}"
    )
    run_dir = Path(CONFIG["results_root"]) / run_name
    work_dir = rank_workdir(run_dir, rank)
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-worker file logger (survives stdout capture loss) ---
    wlog = setup_file_logger(
        f"worker_{run_name}_r{rank}",
        run_dir / f"worker_rank{rank:02d}.log",
    )
    install_signal_handlers(wlog, f"{run_name}/rank{rank}")

    wlog.info(f"Worker started: {run_name}  rank={rank}/{world_size}  local_rank={local_rank}")
    wlog.info(f"  model={job['model']}  config_id={job['config_id']}  seed={job['seed']}")
    wlog.info(f"  hyperparams={json.dumps(hp, default=str)}")
    wlog.info(f"  pid={os.getpid()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    log_system_memory(wlog, "worker_start")

    summary_path = run_dir / "run_summary.json"
    if CONFIG["skip_existing_completed_runs"] and summary_path.exists():
        try:
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_obj = {}
        if not summary_obj.get("failed", False):
            wlog.info(f"[skip] {run_name} already completed successfully.")
            cleanup_distributed()
            return 0

    # Clean up stale error files from previous failed attempts so that
    # the coordinator does not mix old rank errors with this attempt.
    if rank0:
        for stale in run_dir.glob("rank*_error.json"):
            stale.unlink(missing_ok=True)
        stale_summary = run_dir / "run_summary.json"
        if stale_summary.exists():
            try:
                obj = json.loads(stale_summary.read_text(encoding="utf-8"))
                if obj.get("failed", False):
                    stale_summary.unlink(missing_ok=True)
            except Exception:
                pass
    barrier()

    try:
        wlog.info("STEP 1/6: Loading legacy module...")
        module = load_module_from_path(f"legacy_train_module_{run_name}_rank{rank}", CONFIG["legacy_train_script"])
        wlog.info("STEP 1/6: Legacy module loaded OK.")

        if world_size > 1:
            raise RuntimeError("Repair v1 supports one GPU per job; use independent jobs across GPUs")
            module.PeakSamplerRestricted = make_distributed_peak_sampler_class(rank=rank, world_size=world_size, seed=int(job["seed"]))

        wlog.info("STEP 2/6: Installing model patch...")
        install_legacy_model_patch(module, job, local_rank)
        patch_legacy_globals(module, job, local_rank)
        patch_nonzero_rank_io(module, rank)
        wlog.info("STEP 2/6: Patches installed OK.")

        if rank0:
            manifest = {
                "config": CONFIG,
                "job": job,
                "resolved_run_dir": str(run_dir.resolve()),
                "resolved_data_path": str(Path(CONFIG["data_path"]).resolve()),
                "resolved_legacy_train_script": str(Path(CONFIG["legacy_train_script"]).resolve()),
                "ddp_world_size": world_size,
            }
            write_json(run_dir / "run_manifest.json", manifest)

        wlog.info("STEP 3/6: Starting training (module.wrapper)...")
        log_system_memory(wlog, "before_training")
        start = time.perf_counter()
        with pushd(work_dir):
            module.wrapper(CONFIG["data_path"], use_bnd=job.get("use_bnd", CONFIG["use_bnd"]))
        # Do not synchronize here. The legacy script performs rank-specific I/O and
        # evaluation near the end of wrapper(), so one rank can legitimately finish
        # later or fail earlier. A monitored barrier here turns the real exception on
        # one rank into a misleading 'connection closed by peer' error on the other.
        elapsed_s = time.perf_counter() - start
        wlog.info(f"STEP 3/6: Training finished in {elapsed_s:.1f}s")
        log_system_memory(wlog, "after_training")

        if rank0:
            wlog.info("STEP 4/6: Collecting results...")
            summary_csv = find_single_csv(run_dir, "performance_summary_global_norm_v61_")
            summary = parse_legacy_summary_csv(summary_csv) if summary_csv is not None else {}
            best_weight = find_single_file(run_dir, "*_best_ema.pth")
            audit=json.loads((work_dir/"training_audit.json").read_text())
            train_txt = find_single_file(run_dir, "training_results_*.txt")
            train_png = find_single_file(run_dir, "training_plot_*.png")
            final_summary: Dict[str, Any] = {
                "run_name": run_name,
                "model": job["model"],
                "seed": job["seed"],
                "stage": job["stage"],
                "config_id": job["config_id"],
                "training_wallclock_s": elapsed_s,
                "summary_csv": str(summary_csv.resolve()) if summary_csv else None,
                "best_weight": str(best_weight.resolve()) if best_weight else None,
                "training_results_txt": str(train_txt.resolve()) if train_txt else None,
                "training_plot_png": str(train_png.resolve()) if train_png else None,
                "legacy_metrics": summary,
                "repair_audit": audit,
                "repair_job": job,
                "hyperparams": hp,
                "ddp": {"enabled": world_size > 1, "world_size": world_size},
            }
            if CONFIG["benchmark_after_train"]:
                wlog.info("STEP 5/6: Running synthetic benchmark...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                final_summary["synthetic_benchmark"] = asdict(benchmark_model(job["model"], job, device))
                wlog.info("STEP 5/6: Benchmark done.")
            write_json(run_dir / "run_summary.json", final_summary)
            wlog.info(f"STEP 6/6: run_summary.json written. SUCCESS.")

        cleanup_distributed()
        if rank != 0 and not CONFIG["keep_rank_workdirs"]:
            shutil.rmtree(work_dir, ignore_errors=True)
        log_system_memory(wlog, "worker_exit_ok")
        wlog.info(f"Worker finished OK: {run_name}")
        return 0

    except RuntimeError as exc:
        message = str(exc)
        is_oom = "out of memory" in message.lower() or "cuda error" in message.lower()
        wlog.error(f"RuntimeError (oom={is_oom}): {message}")
        tb_text = traceback.format_exc()
        wlog.error(tb_text)
        try:
            sys.stderr.write(tb_text + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        log_system_memory(wlog, "worker_runtime_error")
        payload = {
            "run_name": run_name,
            "model": job["model"],
            "stage": job["stage"],
            "config_id": job["config_id"],
            "seed": job["seed"],
            "failed": True,
            "rank": rank,
            "local_rank": local_rank,
            "error_type": type(exc).__name__,
            "error_message": message,
            "oom": bool(is_oom),
            "traceback": tb_text,
            "hyperparams": hp,
        }
        write_json(work_dir / f"rank{rank:02d}_error.json", payload)
        if rank0:
            write_json(run_dir / "run_summary.json", payload)
        cleanup_distributed()
        return 99 if is_oom else 1
    except Exception as exc:
        wlog.error(f"Exception: {exc}")
        tb_text = traceback.format_exc()
        wlog.error(tb_text)
        try:
            sys.stderr.write(tb_text + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        log_system_memory(wlog, "worker_exception")
        payload = {
            "run_name": run_name,
            "model": job["model"],
            "stage": job["stage"],
            "config_id": job["config_id"],
            "seed": job["seed"],
            "failed": True,
            "rank": rank,
            "local_rank": local_rank,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "oom": False,
            "traceback": tb_text,
            "hyperparams": hp,
        }
        write_json(work_dir / f"rank{rank:02d}_error.json", payload)
        if rank0:
            write_json(run_dir / "run_summary.json", payload)
        cleanup_distributed()
        return 1


# =============================================================================
# Aggregation and plotting
# =============================================================================

def load_summaries(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("run_summary.json")):
        if len(path.parts) >= 2 and path.parts[-2].startswith("rank"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path.resolve())
        items.append(data)
    return items


def normalize_value(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return v


def coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        return v
    return None


def flatten_record(data: Dict[str, Any]) -> Dict[str, Any]:
    legacy = data.get("legacy_metrics", {}) or {}
    hp = data.get("hyperparams", {}) or {}
    synth = data.get("synthetic_benchmark", {}) or {}
    row = {
        "run_name": data.get("run_name"),
        "model": data.get("model"),
        "stage": data.get("stage"),
        "config_id": data.get("config_id"),
        "seed": data.get("seed"),
        "failed": data.get("failed", False),
        "oom": data.get("oom", False),
        "training_wallclock_s": data.get("training_wallclock_s"),
        "rmse_hs": coalesce(legacy.get("rmse_hs"), legacy.get("rmse_m")),
        "mae_hs": coalesce(legacy.get("mae_hs"), legacy.get("mae_m"), legacy.get("val_mae_hs_m")),
        "bias_hs": legacy.get("bias_hs"),
        "r_hs": coalesce(legacy.get("r_hs"), legacy.get("cc_hs")),
        "smape_hs": coalesce(legacy.get("smape_hs"), legacy.get("smape_m")),
        "rmse_tm": legacy.get("rmse_tm"),
        "mae_tm": coalesce(legacy.get("mae_tm"), legacy.get("val_mae_tm_s")),
        "bias_tm": legacy.get("bias_tm"),
        "r_tm": coalesce(legacy.get("r_tm"), legacy.get("cc_tm")),
        "smape_tm": legacy.get("smape_tm"),
        "crmse_dir": legacy.get("crmse_dir"),
        "cmae_dir": coalesce(legacy.get("cmae_dir"), legacy.get("val_mae_dir_deg")),
        "cbias_dir": legacy.get("cbias_dir"),
        "r_dir": coalesce(legacy.get("r_dir"), legacy.get("cc_dir")),
        "rmse_m": legacy.get("rmse_m"),
        "mae_m": legacy.get("mae_m"),
        "r2_m": legacy.get("r2_m"),
        "smape_m": legacy.get("smape_m"),
        "train_loss_final": coalesce(legacy.get("train_loss_final"), legacy.get("train_loss_final_txt")),
        "val_loss_final": coalesce(legacy.get("val_loss_final"), legacy.get("val_loss_final_txt")),
        "params_M": (synth.get("params") / 1.0e6) if synth.get("params") is not None else None,
        "mean_ms": synth.get("mean_latency_ms"),
        "gpu_mem_GB": (synth.get("max_memory_mb") / 1024.0) if synth.get("max_memory_mb") is not None else None,
        "ar_total_ms": synth.get("ar_total_latency_ms"),
        "ar_steps": synth.get("ar_rollout_steps"),
        "_path": data.get("_path"),
    }
    for k, v in hp.items():
        row[f"hp_{k}"] = normalize_value(v)
    return row


def numeric_first(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.iloc[0]


def compute_persistence_baseline(root: Path) -> Optional[Path]:
    """Naive persistence baseline: predict the target frame at t with the last
    input frame (t-1). Computed over the same block-stratified test split and the
    same normalization the models use, by importing the legacy helpers, so the
    numbers are directly comparable to the model scores. Writes one CSV. This is a
    domain-wide anchor; it does not train anything. Returns the CSV path or None.
    """
    try:
        import xarray as xr  # available in the training env

        module = load_module_from_path("legacy_persist", CONFIG["legacy_train_script"])
        denorm = module.denorm
        load_and_preprocess_data = module.load_and_preprocess_data
        make_split = module.make_block_stratified_split

        seq_length = int(CONFIG["seq_length"])
        time_steps = int(CONFIG["time_steps"])

        ds = xr.open_dataset(CONFIG["data_path"])
        # Normalization params come from the training indices, mirroring the models.
        # A first pass needs wave_data to build the split; use a temporary full-range
        # param estimate, then the split, exactly as the legacy wrapper does.
        gp0 = module.compute_params_with_indices(ds, idx_train=np.arange(time_steps - seq_length),
                                                 seq_length=seq_length)
        _, wave_data, lon, lat, kcs = load_and_preprocess_data(ds, gp0, time_steps=time_steps)
        idx_tr, idx_va, idx_te = make_split(wave_data, seq_length, seed=int(CONFIG["seed_list"][0]))
        gp = module.compute_params_with_indices(ds, idx_train=idx_tr, seq_length=seq_length)
        _, wave_data, lon, lat, kcs = load_and_preprocess_data(ds, gp, time_steps=time_steps)
        ds.close()

        # Area weights: cos(latitude) over ocean cells only.
        lat2d = lat if lat.ndim == 2 else np.broadcast_to(lat[:, None], wave_data.shape[-2:])
        w = np.cos(np.deg2rad(lat2d)).astype(np.float64)
        ocean = (kcs[0] if kcs.ndim == 3 else kcs) > 0
        w = np.where(ocean, w, 0.0)
        wsum = w.sum() if w.sum() > 0 else 1.0

        def _aw_rmse_mae(pred, true):
            err = pred - true
            mae = float((np.abs(err) * w).sum() / (wsum * len(pred)))
            rmse = float(np.sqrt(((err ** 2) * w).sum() / (wsum * len(pred))))
            return rmse, mae

        # Test windows: target at t = idx + seq_length; persistence pred = frame t-1.
        t_targets = np.asarray(idx_te, dtype=int) + seq_length
        t_targets = t_targets[(t_targets >= 1) & (t_targets < wave_data.shape[0])]
        true = wave_data[t_targets]          # (Nte, C, H, W) normalized
        pred = wave_data[t_targets - 1]      # last input frame

        (hmin, hmax) = gp["hs"]; (tmin, tmax) = gp["tm"]
        hs_rmse, hs_mae = _aw_rmse_mae(denorm(pred[:, 0], hmin, hmax), denorm(true[:, 0], hmin, hmax))
        tm_rmse, tm_mae = _aw_rmse_mae(denorm(pred[:, 1], tmin, tmax), denorm(true[:, 1], tmin, tmax))

        # Direction from sin/cos channels (2,3): circular error in degrees.
        def _ang(c_sin, c_cos):
            return np.arctan2(c_sin, c_cos)
        ang_p = _ang(pred[:, 2], pred[:, 3]); ang_t = _ang(true[:, 2], true[:, 3])
        d = np.angle(np.exp(1j * (ang_p - ang_t)))
        dir_deg = np.abs(np.rad2deg(d))
        dir_rmse = float(np.sqrt(((dir_deg ** 2) * w).sum() / (wsum * len(pred))))
        dir_mae = float((dir_deg * w).sum() / (wsum * len(pred)))

        out = root / "persistence_baseline.csv"
        import csv as _csv
        with open(out, "w", newline="") as f:
            wri = _csv.writer(f)
            wri.writerow(["metric", "Hs", "Tm", "Dir"])
            wri.writerow(["rmse", f"{hs_rmse:.6f}", f"{tm_rmse:.6f}", f"{dir_rmse:.4f}"])
            wri.writerow(["mae", f"{hs_mae:.6f}", f"{tm_mae:.6f}", f"{dir_mae:.4f}"])
            wri.writerow(["n_test_windows", str(len(pred)), "", ""])
        print(f"[persistence] baseline saved -> {out}")
        print(f"[persistence] Hs RMSE={hs_rmse:.4f} m, Tm RMSE={tm_rmse:.4f} s, Dir RMSE={dir_rmse:.2f} deg")
        return out
    except Exception as exc:
        print(f"[persistence] skipped ({type(exc).__name__}: {exc})")
        return None


def aggregate_results(root: Path) -> Dict[str, Path]:
    items = load_summaries(root)
    if not items:
        raise FileNotFoundError(f"No run_summary.json files found under {root}")
    rows = [flatten_record(x) for x in items]
    df = pd.DataFrame(rows)
    raw_runs_csv = root / "raw_runs.csv"
    grouped_csv = root / "grouped_results.csv"
    aggregated_json = root / "aggregated_results.json"
    df.to_csv(raw_runs_csv, index=False)

    group_cols = ["model", "stage", "config_id"]
    protected_cols = group_cols + ["run_name", "seed", "_path"]
    numeric_cols = [c for c in df.columns if c not in protected_cols and pd.api.types.is_numeric_dtype(df[c])]
    nonnumeric_cols = [c for c in df.columns if c not in protected_cols and c not in numeric_cols]

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    agg_num = grouped[numeric_cols].agg(["mean", "std", "min", "max", "count"]) if numeric_cols else pd.DataFrame(index=grouped.size().index)
    if len(agg_num.columns):
        agg_num.columns = ["__".join([c for c in col if c]).strip("_") for col in agg_num.columns.to_flat_index()]
    agg_num = agg_num.reset_index()

    if nonnumeric_cols:
        first_df = grouped[nonnumeric_cols].agg(numeric_first).reset_index()
        agg = agg_num.merge(first_df, on=group_cols, how="left")
    else:
        agg = agg_num

    primary_col = "mae_hs__mean" if "mae_hs__mean" in agg.columns else ("rmse_m__mean" if "rmse_m__mean" in agg.columns else None)
    if primary_col is not None:
        agg["rank_within_stage"] = agg.groupby(["stage", "model"], dropna=False)[primary_col].rank(method="dense", ascending=True)

    agg.to_csv(grouped_csv, index=False)
    payload = {
        "n_runs": int(len(df)),
        "raw_runs_csv": str(raw_runs_csv.resolve()),
        "grouped_csv": str(grouped_csv.resolve()),
    }
    aggregated_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"raw_runs_csv": raw_runs_csv, "grouped_csv": grouped_csv, "aggregated_json": aggregated_json}


def _best_only(df: pd.DataFrame) -> pd.DataFrame:
    if "rank_within_stage" in df.columns:
        df = df[df["rank_within_stage"] == 1].copy()
    return df


def _drop_metric_nan(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df[df[metric].notna()].copy() if metric in df.columns else df.iloc[0:0].copy()


def _save_bar(df: pd.DataFrame, x: str, y: str, err: str | None, out_path: Path, title: str, ylabel: str) -> None:
    if df.empty or y not in df.columns:
        return
    work = _drop_metric_nan(df, y)
    if work.empty:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    vals = work[y].astype(float).values
    errs = work[err].astype(float).fillna(0.0).values if err and err in work.columns else None
    labels = [str(m) for m in work[x].astype(str)]
    plt.bar(labels, vals, yerr=errs, capsize=4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    work = df[df[x].notna() & df[y].notna()].copy()
    if work.empty:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    for _, row in work.iterrows():
        plt.scatter(float(row[x]), float(row[y]), s=60)
        label = f"{row['model']}:{row.get('stage', '')}".rstrip(":")
        plt.text(float(row[x]), float(row[y]), label, fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_grouped_results(grouped_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(grouped_csv)
    best = _best_only(df)
    _save_bar(best, "model", "mae_hs__mean", "mae_hs__std", out_dir / "accuracy_mae_hs_bar.png", "Best Hs MAE by model", "Hs MAE (m)")
    _save_bar(best, "model", "mae_tm__mean", "mae_tm__std", out_dir / "accuracy_mae_tm_bar.png", "Best Tm MAE by model", "Tm MAE (s)")
    _save_bar(best, "model", "cmae_dir__mean", "cmae_dir__std", out_dir / "accuracy_cmae_dir_bar.png", "Best direction MAE by model", "Dir MAE (deg)")
    _save_bar(best, "model", "mean_ms__mean", "mean_ms__std", out_dir / "efficiency_latency_bar.png", "Synthetic latency by model", "Latency (ms)")
    _save_bar(best, "model", "gpu_mem_GB__mean", "gpu_mem_GB__std", out_dir / "efficiency_memory_bar.png", "Peak memory by model", "Memory (GB)")
    _save_scatter(best, "mean_ms__mean", "mae_hs__mean", out_dir / "pareto_maehs_latency.png", "Hs MAE vs latency", "Latency (ms)", "Hs MAE (m)")
    _save_scatter(best, "params_M__mean", "mae_hs__mean", out_dir / "pareto_maehs_params.png", "Hs MAE vs parameters", "Parameters (M)", "Hs MAE (m)")


# =============================================================================
# Coordinator
# =============================================================================

def chunk_gpus(gpu_ids: List[str], width: int) -> List[List[str]]:
    return [gpu_ids[i:i + width] for i in range(0, len(gpu_ids), width) if len(gpu_ids[i:i + width]) == width]


def build_worker_command(job_file: Path, gpu_group: List[str], master_port: int) -> Tuple[List[str], Dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_group)),
        "--master_port",
        str(master_port),
        str(Path(__file__).resolve()),
        "--worker",
        "--job_file",
        str(job_file),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
    env["OMP_NUM_THREADS"] = str(CONFIG["omp_num_threads"])
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    # Enable torchrun traceback capture so that error_file is populated
    # instead of showing <N/A> in ChildFailedError messages.
    error_dir = Path(CONFIG["results_root"]) / "_error_files"
    error_dir.mkdir(parents=True, exist_ok=True)
    job_stem = Path(job_file).stem
    env["TORCHELASTIC_ERROR_FILE"] = str(error_dir / f"{job_stem}_port{master_port}.json")
    return cmd, env


def launch_jobs(plan: List[Dict[str, Any]], jobs_dir: Path, coord_logger: logging.Logger) -> List[Dict[str, Any]]:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = jobs_dir.parent / "_worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    free_gpus = [str(x) for x in CONFIG["gpus"]]
    max_concurrent = int(CONFIG.get("max_concurrent_jobs", len(free_gpus)))
    queue: Deque[Tuple[Dict[str, Any], Path]] = deque()
    for idx, job in enumerate(plan):
        job_file = jobs_dir / f"job_{idx:05d}_{job['config_id']}.json"
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        queue.append((job, job_file))

    # running: (proc, gpu_group, job, job_file, stdout_fh, stderr_fh)
    running: List[Tuple[subprocess.Popen, List[str], Dict[str, Any], Path, Any, Any]] = []
    events: List[Dict[str, Any]] = []
    _port_counter = 0

    while queue or running:
        still_running = []
        for proc, gpu_group, job, job_file, fh_out, fh_err in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu_group, job, job_file, fh_out, fh_err))
                continue
            # Retained legacy processing.
            fh_out.close()
            fh_err.close()
            free_gpus.extend(gpu_group)
            free_gpus.sort(key=int)
            run_tag = f"{job['model']}:{job['config_id']}:seed{job['seed']}"
            event = {
                "run": run_tag,
                "gpus": gpu_group,
                "return_code": int(ret),
                "status": "ok" if ret == 0 else ("oom" if ret == 99 else "failed"),
            }
            events.append(event)
            msg = f"[done] {run_tag} gpus={','.join(gpu_group)} rc={ret}"
            print(msg)
            coord_logger.info(msg)
            if ret != 0:
                # Retained legacy processing.
                # torchrun ChildFailedError boilerplate alone takes ~25 lines).
                err_log = derive_run_dir(job) / f"{job['config_id']}_seed{job['seed']}_stderr.log"
                if err_log.exists():
                    tail = err_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                    tail_msg = f"[FAIL-TAIL] {run_tag} rc={ret} last 80 stderr lines:\n" + "\n".join(tail)
                    coord_logger.error(tail_msg)
                # Surface detailed rank traceback if available.
                hp = job["hyperparams"]
                run_name = (
                    f"{job['stage']}_{job['model']}_{job['config_id']}_seed{job['seed']}_"
                    f"seq{hp.get('seq_length', CONFIG['seq_length'])}_"
                    f"lr{hp.get('max_lr', 1e-4):.0e}_wd{hp.get('weight_decay', 1e-4):.0e}"
                )
                run_dir = Path(CONFIG["results_root"]) / run_name
                detail_paths = sorted(run_dir.rglob("rank*_error.json"))
                if detail_paths:
                    for p in detail_paths:
                        try:
                            payload = json.loads(p.read_text(encoding="utf-8"))
                            tb = payload.get("traceback", "")
                            if tb:
                                coord_logger.error(f"[FAIL-DETAIL] {run_tag} rc={ret} {p.name}:\n{tb}")
                        except Exception:
                            pass
                # Also check worker log files for errors.
                for wlog_path in sorted(run_dir.glob("worker_rank*.log")):
                    try:
                        wlog_text = wlog_path.read_text(encoding="utf-8", errors="replace")
                        if "ERROR" in wlog_text or "Traceback" in wlog_text:
                            wlog_tail = wlog_text.splitlines()[-40:]
                            coord_logger.error(
                                f"[WORKER-LOG] {run_tag} {wlog_path.name}:\n" + "\n".join(wlog_tail)
                            )
                    except Exception:
                        pass
        running = still_running

        launched = False
        free_gpus.sort(key=int)
        while queue:
            if len(running) >= max_concurrent:
                break
            job, job_file = queue[0]
            need = int(job.get("gpus_per_trial", CONFIG["gpus_per_trial"]))
            groups = chunk_gpus(free_gpus, need)
            if not groups:
                break
            gpu_group = groups[0]
            for gid in gpu_group:
                free_gpus.remove(gid)
            port = int(CONFIG["master_port_base"]) + _port_counter
            _port_counter += 1
            cmd, env = build_worker_command(job_file, gpu_group, port)
            run_tag = f"{job['model']}:{job['config_id']}:seed{job['seed']}"
            # Open per-job log files inside the run's own directory, so stdout,
            # stderr, and worker_rank00.log all sit together with the checkpoints.
            job_run_dir = derive_run_dir(job)
            job_run_dir.mkdir(parents=True, exist_ok=True)
            fh_out = open(job_run_dir / f"{job['config_id']}_seed{job['seed']}_stdout.log", "w")
            fh_err = open(job_run_dir / f"{job['config_id']}_seed{job['seed']}_stderr.log", "w")
            msg = f"[launch] {run_tag} gpus={','.join(gpu_group)} port={port}"
            print(msg)
            coord_logger.info(msg)
            proc = subprocess.Popen(cmd, env=env, stdout=fh_out, stderr=fh_err)
            running.append((proc, gpu_group, job, job_file, fh_out, fh_err))
            queue.popleft()
            launched = True
        time.sleep(5 if launched else 20)

    return events


# =============================================================================
# Stage 2: best-config follow-ups (BND ablation, true multi-seed, significance)
# =============================================================================

def _stage2_metric_value(rec: Dict[str, Any], metric: str) -> Optional[float]:
    """Pull the selection metric from a flattened record, lower is better."""
    v = rec.get(metric)
    if v is None and metric == "val_loss_final":
        v = rec.get("rmse_hs")  # fallback when the loss was not recorded
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def select_best_configs(root: Path, metric: str) -> Dict[str, Dict[str, Any]]:
    """Group stage-1 runs by model and pick the best config per model by `metric`
    (lower is better), using only primary-seed, non-failed sweep runs. Returns
    {model: full_run_summary}, so the winner's hyperparameters and checkpoint path
    are available to build the follow-up jobs."""
    primary_seed = int(CONFIG["seed_list"][0])
    best: Dict[str, Dict[str, Any]] = {}
    best_val: Dict[str, float] = {}
    for summary in load_summaries(root):
        if summary.get("failed") or summary.get("oom"):
            continue
        if summary.get("stage") != "focused_all_onefile":
            continue  # stage-1 sweep only
        if int(summary.get("seed", -1)) != primary_seed:
            continue
        model = summary.get("model")
        rec = flatten_record(summary)
        val = _stage2_metric_value(rec, metric)
        if model is None or val is None:
            continue
        if model not in best_val or val < best_val[model]:
            best_val[model] = val
            best[model] = summary
    return best


def build_stage2_plan(best: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """From the per-model winners, build the BND-off ablation and true multi-seed
    follow-up jobs. Distinct stage tags and config_id suffixes keep these in their
    own run directories, separate from the stage-1 sweep."""
    jobs: List[Dict[str, Any]] = []
    ms_seeds = [int(s) for s in CONFIG.get("stage2_true_multiseed_seeds", [])]
    do_bndoff = bool(CONFIG.get("stage2_bnd_off_ablation", False))
    for model, summary in sorted(best.items()):
        hp = summary.get("hyperparams", {})
        base_cfg = summary.get("config_id", f"{model}_best")
        # True multi-seed on the winning config (BND on, as in the main benchmark).
        for seed in ms_seeds:
            jobs.append({
                "stage": "stage2_multiseed",
                "model": model,
                "epochs": CONFIG["epochs"],
                "time_steps": CONFIG["time_steps"],
                "use_bnd": "on",
                "seed": int(seed),
                "gpus_per_trial": CONFIG["gpus_per_trial"],
                "config_id": f"{base_cfg}_best",
                "hyperparams": hp,
            })
        # BND-off ablation on the winning config, at the primary seed only.
        if do_bndoff:
            jobs.append({
                "stage": "stage2_bndoff",
                "model": model,
                "epochs": CONFIG["epochs"],
                "time_steps": CONFIG["time_steps"],
                "use_bnd": "off",
                "seed": int(CONFIG["seed_list"][0]),
                "gpus_per_trial": CONFIG["gpus_per_trial"],
                "config_id": f"{base_cfg}_bndoff",
                "hyperparams": hp,
            })
    return jobs


def _t_crit_95(dof: int) -> float:
    """Two-sided 95% t critical value for small samples (no SciPy dependency)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if dof <= 0:
        return float("nan")
    return table.get(dof, 1.96)


def compute_significance(root: Path) -> Optional[Path]:
    """Summarize the multi-seed spread of the winning configs and write a CSV.
    Reports mean, std, n, and a t-based 95% CI per model for Hs RMSE, plus a
    paired-difference 95% CI between the best model and each other model over the
    shared seeds. With only a few seeds the test power is limited, and the CSV
    states that explicitly so the result is not over-read."""
    try:
        import statistics as st

        rows = []
        by_model: Dict[str, Dict[int, float]] = {}
        for summary in load_summaries(root):
            if summary.get("stage") != "stage2_multiseed" or summary.get("failed"):
                continue
            model = summary.get("model")
            seed = int(summary.get("seed", -1))
            rec = flatten_record(summary)
            v = rec.get("rmse_hs")
            if model is None or v is None:
                continue
            by_model.setdefault(model, {})[seed] = float(v)

        if not by_model:
            print("[significance] no stage2 multiseed runs found; skipping")
            return None

        per_model = {}
        for model, seedvals in by_model.items():
            vals = list(seedvals.values())
            n = len(vals)
            mean = st.fmean(vals)
            sd = st.stdev(vals) if n > 1 else 0.0
            half = _t_crit_95(n - 1) * sd / (n ** 0.5) if n > 1 else float("nan")
            per_model[model] = {"n": n, "mean": mean, "std": sd,
                                "ci_lo": mean - half, "ci_hi": mean + half,
                                "seedvals": seedvals}
            rows.append(["per_model", model, f"{mean:.6f}", f"{sd:.6f}", str(n),
                         f"{mean - half:.6f}", f"{mean + half:.6f}", ""])

        # Best model = lowest mean Hs RMSE. Paired differences over shared seeds.
        best_model = min(per_model, key=lambda m: per_model[m]["mean"])
        b = per_model[best_model]["seedvals"]
        for model, info in sorted(per_model.items()):
            if model == best_model:
                continue
            shared = sorted(set(b) & set(info["seedvals"]))
            diffs = [info["seedvals"][s] - b[s] for s in shared]  # other minus best
            n = len(diffs)
            if n >= 1:
                dmean = st.fmean(diffs)
                dsd = st.stdev(diffs) if n > 1 else 0.0
                half = _t_crit_95(n - 1) * dsd / (n ** 0.5) if n > 1 else float("nan")
                distinct = (n > 1) and (dmean - half > 0 or dmean + half < 0)
                rows.append(["paired_vs_best", f"{model}_minus_{best_model}",
                             f"{dmean:.6f}", f"{dsd:.6f}", str(n),
                             f"{dmean - half:.6f}", f"{dmean + half:.6f}",
                             "distinct" if distinct else "overlaps_zero"])

        out = root / "significance.csv"
        import csv as _csv
        with open(out, "w", newline="") as f:
            wri = _csv.writer(f)
            wri.writerow(["kind", "name", "mean", "std", "n", "ci95_lo", "ci95_hi", "note"])
            wri.writerows(rows)
            wri.writerow([])
            wri.writerow(["# metric: Hs RMSE (m). best_model =", best_model, "", "", "", "", "", ""])
            wri.writerow(["# CI is t-based; with a small seed count the test power is limited.",
                          "", "", "", "", "", "", ""])
        print(f"[significance] saved -> {out} (best={best_model})")
        return out
    except Exception as exc:
        print(f"[significance] skipped ({type(exc).__name__}: {exc})")
        return None


def coordinator_main() -> int:
    legacy_script = Path(CONFIG["legacy_train_script"]).resolve()
    if not legacy_script.is_file():
        print(f"[FATAL] legacy_train_script not found: {legacy_script}")
        return 1
    data_path = Path(CONFIG["data_path"]).resolve()
    if not data_path.is_file():
        print(f"[FATAL] data_path not found: {data_path}")
        return 1

    root = Path(CONFIG["results_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    coord_logger = setup_file_logger("coordinator", root / "coordinator.log")
    coord_logger.info("=" * 60)
    coord_logger.info("Coordinator started")
    coord_logger.info(f"results_root = {root}")
    coord_logger.info(f"max_concurrent_jobs = {CONFIG.get('max_concurrent_jobs', 'unlimited')}")
    coord_logger.info(f"gpus_per_trial = {CONFIG['gpus_per_trial']}")
    coord_logger.info(f"gpus = {CONFIG['gpus']}")
    log_system_memory(coord_logger, "coordinator_start")

    jobs_dir = root / "_jobs"
    figures_dir = root / "figures"
    plan_json = root / "plan.json"
    config_json = root / "config_snapshot.json"
    config_json.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

    plan = generate_plan()
    original_plan_len = len(plan)
    if CONFIG.get("skip_existing_completed_runs", False):
        filtered = []
        skipped = 0
        for job in plan:
            hp = job["hyperparams"]
            run_name = (
                f"{job['stage']}_{job['model']}_{job['config_id']}_seed{job['seed']}_"
                f"seq{hp.get('seq_length', CONFIG['seq_length'])}_"
                f"lr{hp.get('max_lr', 1e-4):.0e}_wd{hp.get('weight_decay', 1e-4):.0e}"
            )
            run_dir = Path(CONFIG["results_root"]) / run_name
            summary_path = run_dir / "run_summary.json"
            if summary_path.exists():
                try:
                    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary_obj = {}
                if not summary_obj.get("failed", False):
                    skipped += 1
                    print(f"[prefilter-skip] {run_name} already completed successfully.")
                    continue
            filtered.append(job)
        plan = filtered
        if skipped:
            print(f"Prefilter skipped {skipped} completed jobs.")
    plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    coord_logger.info(f"Generated {len(plan)} jobs (from {original_plan_len})")
    print(f"Generated {len(plan)} jobs (from {original_plan_len})")

    events = launch_jobs(plan, jobs_dir, coord_logger)
    write_json(root / "launcher_events.json", {"events": events})

    # Summary of all events
    n_ok = sum(1 for e in events if e["status"] == "ok")
    n_oom = sum(1 for e in events if e["status"] == "oom")
    n_fail = sum(1 for e in events if e["status"] == "failed")
    coord_logger.info(f"All jobs finished: ok={n_ok}, oom={n_oom}, failed={n_fail}")
    log_system_memory(coord_logger, "coordinator_end")

    if CONFIG["aggregate_after_finish"]:
        try:
            if CONFIG.get("compute_persistence_baseline", False):
                compute_persistence_baseline(root)
            outputs = aggregate_results(root)
            print(f"Aggregated results saved to {outputs['grouped_csv']}")
            if CONFIG["plot_after_finish"]:
                plot_grouped_results(outputs["grouped_csv"], figures_dir)
                print(f"Figures saved to {figures_dir}")
        except Exception as exc:
            coord_logger.error(f"aggregation/plotting failed: {exc}", exc_info=True)

    # ---- Stage 2: automatic follow-ups on the per-model winners ----
    if CONFIG.get("run_stage2_after_sweep", False):
        try:
            best = select_best_configs(root, CONFIG.get("stage2_best_metric", "val_loss_final"))
            if not best:
                coord_logger.warning("stage2: no completed stage-1 runs found; skipping follow-ups")
            else:
                names = ", ".join(f"{m}:{s.get('config_id')}" for m, s in sorted(best.items()))
                coord_logger.info(f"stage2: best configs -> {names}")
                stage2_plan = build_stage2_plan(best)
                write_json(root / "stage2_best_configs.json",
                           {m: s.get("config_id") for m, s in best.items()})
                print(f"Stage 2: launching {len(stage2_plan)} follow-up jobs "
                      f"(ablation + multi-seed) on {len(best)} winners")
                s2_events = launch_jobs(stage2_plan, jobs_dir, coord_logger)
                write_json(root / "stage2_launcher_events.json", {"events": s2_events})
                if CONFIG["aggregate_after_finish"]:
                    outputs = aggregate_results(root)
                    print(f"Re-aggregated with stage-2 runs -> {outputs['grouped_csv']}")
                    if CONFIG["plot_after_finish"]:
                        plot_grouped_results(outputs["grouped_csv"], figures_dir)
                if CONFIG.get("significance_test", False):
                    compute_significance(root)
        except Exception as exc:
            coord_logger.error(f"stage2 pipeline failed: {exc}", exc_info=True)
    return 0


# =============================================================================
# CLI entry
# =============================================================================

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file SWAN benchmark launcher and worker.", add_help=True)
    parser.add_argument("--worker", action="store_true", help="Internal worker mode. Do not set manually.")
    parser.add_argument("--job_file", type=str, default="", help="Internal worker job JSON path.")
    # torchrun may inject one of these depending on version/configuration.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    # Ignore unknown launcher args instead of crashing before worker_main starts.
    args, _unknown = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_cli()
    if args.worker:
        if not args.job_file:
            raise ValueError("--job_file is required in worker mode")
        return worker_main(args.job_file)
    return coordinator_main()


if __name__ == "__main__":
    if '--worker' not in sys.argv:
        raise SystemExit('Use run_repaired.py to plan or launch the corrected campaign')
    raise SystemExit(main())
