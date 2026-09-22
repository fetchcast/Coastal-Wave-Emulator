# -*- coding: utf-8 -*-
"""
UNet-ConvLSTM wave emulator (v57, 'all-patches')
- Includes: circular direction loss, deep supervision, TV regularizer,
  coastline-ring weighting, extra static & seasonal features, peak curriculum.

Tested with PyTorch 2.x / CUDA environment.
"""

import os, math, argparse, traceback, warnings, itertools
import json
import repair_support as repair
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import r2_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Quiet the cosmetic "findfont: Font family 'Arial' not found" messages. The
# fallback font is used automatically; this does not change any numbers and only
# affects the on-screen font of figures, not the data.
import logging as _logging
warnings.filterwarnings("ignore", message="findfont")
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

from scipy.ndimage import zoom as nd_zoom
from scipy.ndimage import binary_dilation
from pathlib import Path


# =========================
# Globals & Tiny Utilities
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (device.type == "cuda")
USE_BF16 = USE_CUDA and torch.cuda.get_device_capability(0)[0] >= 8
AMP_ENABLED = USE_CUDA
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
SCALER = GradScaler(enabled=AMP_ENABLED and not USE_BF16)
EPS = 1e-8

DTYPE = torch.float32
DROPOUT_P = 0.1
BATCH_SIZE = 1
ACC_STEPS = 2

# ---------------------------
# Hyperparam sweep examples
# ---------------------------
time_steps_list = [17498]
seq_length_list = [12]
epochs_list     = [20]
hidden_dim_list = [128]
unet_feat_list  = [[32, 64, 128, 256, 512]]
max_lr = 1e-4
weight_decay = 1e-4
pct_start = 0.3
div_factor = 10.0
final_div_factor = 100.0

# =========================
# Fonts (Korean safe)
# =========================
def _set_korean_font():
    # Try known CJK family names first. The list is broad so a usable font on a
    # Linux/conda system (Noto, Nanum, WenQuanYi, etc.) is picked up automatically.
    candidates = [
        'Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Nanum Barun Gothic',
        'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'Noto Sans CJK JP',
        'NanumGothicCoding', 'UnDotum', 'Baekmuk Gulim', 'WenQuanYi Zen Hei',
        'Droid Sans Fallback',
    ]
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in system_fonts:
            matplotlib.rcParams['font.family'] = name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return

    # Next, a NanumGothic.ttf placed beside the training script.
    local = os.path.join(os.path.dirname(__file__), 'NanumGothic.ttf')
    if os.path.isfile(local):
        fm.fontManager.addfont(local)
        matplotlib.rcParams['font.family'] = 'NanumGothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        return

    # Last, scan installed font files for anything CJK-capable and register it.
    try:
        for path in fm.findSystemFonts(fontext='ttf'):
            base = os.path.basename(path).lower()
            if any(tok in base for tok in ('nanum', 'notosanscjk', 'notosanskr',
                                           'cjk', 'gothic', 'gulim', 'batang',
                                           'wqy', 'unfonts', 'malgun')):
                fm.fontManager.addfont(path)
                matplotlib.rcParams['font.family'] = fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams['axes.unicode_minus'] = False
                return
    except Exception:
        pass

    # Nothing CJK-capable is installed. Fall back to a default font silently.
    # Korean labels in figures may render as boxes, which is cosmetic and does
    # not affect any computed result. Install a Korean font to restore glyphs.
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

_set_korean_font()

# =========================
# EMA
# =========================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False)}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for k, v in model.state_dict().items():
            if k in self.shadow and hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False):
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model):
        for k, v in self.backup.items():
            model.state_dict()[k].copy_(v)
        self.backup = {}

# =========================
# Station meta (unchanged)
# =========================
STATIONS = {
    "대한해협": {"lat": 34.933888, "lon": 129.1375, "file": "daehanhaehyup_kg_wave_1h.csv"},
    "제주해협": {"lat": 33.901944, "lon": 126.490555, "file": "jejuhaehyup_kg_wave_1h.csv"},
    "남해동부": {"lat": 34.223611, "lon": 128.420555, "file": "namhaedongbu_kg_wave_1h.csv"},
    "울릉도북서": {"lat": 37.7275, "lon": 130.578055, "file": "UlleungdoNW_kg_wave_1h.csv"},
    "대천해수욕장": {"lat": 36.28438490, "lon": 126.46236720, "file": "daechon_sig_wave_1H.csv"},
    "감천항": {"lat": 35.052806, "lon": 129.003083, "file": "gamcheon_sig_wave_1H.csv"},
    "경포대해수욕장": {"lat": 37.808840, "lon": 128.931980, "file": "gyeongpo_sig_wave_1H.csv"},
    "해운대해수욕장": {"lat": 35.148888, "lon": 129.169722, "file": "haeundae_sig_wave_1H.csv"},
    "임랑해수욕장": {"lat": 35.3025, "lon": 129.2925, "file": "imrang_sig_wave_1H.csv"},
    "중문해수욕장": {"lat": 33.234444, "lon": 126.409722, "file": "jungmun_sig_wave_1H.csv"},
    "생일도": {"lat": 34.258716, "lon": 126.960269, "file": "saengil_sig_wave_1H.csv"},
    "상왕등도": {"lat": 35.652458, "lon": 126.194255, "file": "sangwang_sig_wave_1H.csv"},
    "송정해수욕장": {"lat": 35.164722, "lon": 129.219444, "file": "songjung_sig_wave_1H.csv"}
}

# =========================
# Data utils
# =========================
def denorm(x, vmin, vmax):
    return x * (vmax - vmin) + vmin

def normalize_with_external_params(data, params):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmin, dmax = params
    if abs(dmax - dmin) < EPS:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - dmin) / (dmax - dmin)).astype(np.float32)

def compute_params_with_indices(
    ds,
    idx_train,
    seq_length,
    *,
    use_kcs: bool = True,
    hs_cap: float = 12.0,
    hs_q: float = 99.9,
    tm_q: float = 98.0,
    wind_q: tuple = (0.5, 99.9),
    vel_q: tuple = (0.5, 99.9),
    depth_q: tuple = (0.0, 100.0)
):
    """Legacy function retained except for the documented repair changes."""
    if idx_train is None or len(idx_train) == 0:
        raise ValueError("compute_params_with_indices: idx_train이 비어 있습니다.")
    t_idx = np.asarray(idx_train, dtype=int) + int(seq_length)

    def _valid_t_idx(da: xr.DataArray, t_idx_arr):
        if "time" in da.dims:
            T = da.sizes["time"]
            return t_idx_arr[(t_idx_arr >= 0) & (t_idx_arr < T)]
        return None

    def _sel_values(varname: str):
        if varname not in ds:
            raise KeyError(f"'{varname}' 변수를 NetCDF에서 찾을 수 없습니다.")
        da = ds[varname]
        t_valid = _valid_t_idx(da, t_idx)
        if t_valid is None:
            arr = da.values
        else:
            arr = da.isel(time=xr.DataArray(t_valid, dims="time_idx")).values
        return arr

    kcs_mask = None
    if use_kcs and ("kcs" in ds):
        kcs_arr = _sel_values("kcs")
        if kcs_arr.ndim == 2:
            if "time" in ds.get("hsign", xr.DataArray()).dims:
                Tlen = len(_valid_t_idx(ds["hsign"], t_idx)) if _valid_t_idx(ds["hsign"], t_idx) is not None else 1
            else:
                Tlen = 1
            kcs_mask = np.broadcast_to((kcs_arr == 1), (Tlen,) + kcs_arr.shape)
        else:
            kcs_mask = (kcs_arr == 1)

    def _robust_range(arr, qlo, qhi, mask=None):
        a = arr
        if mask is not None and mask.shape == arr.shape:
            a = a[mask]
        a = a[np.isfinite(a)]
        if a.size == 0: return (0.0, 1.0)
        if qlo <= 0.0 and qhi >= 100.0:
            lo = float(np.nanmin(a)); hi = float(np.nanmax(a))
        else:
            lo = float(np.nanpercentile(a, qlo)); hi = float(np.nanpercentile(a, qhi))
        if not np.isfinite(lo): lo = 0.0
        if (not np.isfinite(hi)) or (hi <= lo + 1e-12): hi = lo + 1.0
        return (lo, hi)

    params = {}
    windu = _sel_values("windu"); windv = _sel_values("windv")
    u_min, u_max = _robust_range(windu, wind_q[0], wind_q[1], kcs_mask)
    v_min, v_max = _robust_range(windv, wind_q[0], wind_q[1], kcs_mask)
    params["wind_u"] = (u_min, u_max); params["wind_v"] = (v_min, v_max)

    depth = _sel_values("depth")
    dm = ds["kcs"].values
    depth_mask = (dm == 1) if dm.ndim == 2 else (dm[0] == 1)
    d_min, d_max = _robust_range(depth, depth_q[0], depth_q[1], depth_mask)
    params["depth"] = (d_min, d_max)

    velx = _sel_values("veloc-x"); vely = _sel_values("veloc-y")
    vx_min, vx_max = _robust_range(velx, vel_q[0], vel_q[1], kcs_mask)
    vy_min, vy_max = _robust_range(vely, vel_q[0], vel_q[1], kcs_mask)
    params["veloc_x"] = (vx_min, vx_max); params["veloc_y"] = (vy_min, vy_max)

    hs = _sel_values("hsign"); _, hs_max_q = _robust_range(hs, 50.0, hs_q, kcs_mask)
    params["hs"] = (0.0, float(min(hs_max_q, hs_cap)))

    tm = _sel_values("period"); _, tm_max_q = _robust_range(tm, 50.0, tm_q, kcs_mask)
    params["tm"] = (0.0, float(max(tm_max_q, 15.0)))

    params["dir"] = (0.0, 360.0)
    return params

# =========================
# Feature engineering
# =========================
def _depth_grad_mag(depth2d):
    """Legacy function retained except for the documented repair changes."""
    if depth2d.ndim == 3: depth2d = depth2d[0]
    gy, gx = np.gradient(np.nan_to_num(depth2d.astype(np.float32)))
    g = np.hypot(gx, gy)
    lo, hi = np.nanpercentile(g, 1), np.nanpercentile(g, 99)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return g

# Retained legacy processing.

# =========================
# Data IO / preprocessing
# =========================
def load_and_preprocess_data(ds, global_norm_params, time_steps=100):
    """Legacy function retained except for the documented repair changes."""
    required_vars = ['windu','windv','depth','veloc-x','veloc-y','hsign','period','dir','x','y','kcs']
    for v in required_vars:
        if v not in ds: raise ValueError(f"Variable '{v}' not found in dataset")

    _DTYPE = np.float32
    T = time_steps

    def time_values(name):
        da=ds[name]
        a=da.isel(time=slice(0,T)).values if 'time' in da.dims else da.values
        if a.ndim==2:a=np.broadcast_to(a,(T,)+a.shape)
        if a.ndim!=3 or len(a)!=T:raise ValueError(f'Invalid time/spatial shape: {name}')
        return a
    wind_u = normalize_with_external_params(time_values('windu'), global_norm_params['wind_u']).astype(_DTYPE)
    wind_v = normalize_with_external_params(time_values('windv'), global_norm_params['wind_v']).astype(_DTYPE)
    depth   = normalize_with_external_params(time_values('depth'),  global_norm_params['depth']).astype(_DTYPE)
    veloc_x = normalize_with_external_params(time_values('veloc-x'), global_norm_params['veloc_x']).astype(_DTYPE)
    veloc_y = normalize_with_external_params(time_values('veloc-y'), global_norm_params['veloc_y']).astype(_DTYPE)

    # Retained legacy processing.
    hs  = normalize_with_external_params(time_values('hsign'), global_norm_params['hs']).astype(_DTYPE)
    tm  = normalize_with_external_params(time_values('period'), global_norm_params['tm']).astype(_DTYPE)
    rad = np.deg2rad(time_values('dir'))
    dsin, dcos = np.sin(rad).astype(_DTYPE), np.cos(rad).astype(_DTYPE)

    lon = ds['x'].values; lat = ds['y'].values; kcs = ds['kcs'].values
    if lat.ndim == 3: lat = lat[0]
    if kcs.ndim == 3: kcs = kcs[0]

    # Retained legacy processing.
    depth2d = ds['depth'].values if ds['depth'].values.ndim == 2 else ds['depth'].values[0]
    depth_grad = _depth_grad_mag(depth2d)                    # (Y,X)

    # Retained legacy processing.
    H, W = hs.shape[-2], hs.shape[-1]
    depth_grad_3d = np.broadcast_to(depth_grad[None, ...], (T, H, W)).astype(_DTYPE)

    # Retained legacy processing.
    input_data = np.stack(
        [wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad_3d], axis=1
    )
    wave_data = np.stack([hs, tm, dsin, dcos], axis=1)
    # Undefined land directions are excluded from the objective and set to zero.
    wave_data=np.where((kcs>0)[None,None],wave_data,0.).astype(np.float32)
    return input_data, wave_data, lon, lat, kcs

# =========================
# Station CSV loader (unchanged)
# =========================
def load_all_station_data(root_dir, norm_params, time_index):
    col_hs = ['유의파고(MOSE.HF)(m)', '유의파고(m)', 'Hs(m)', 'HS', 'hs']
    col_tm = ['유의파주기(MOSE.HF)(sec)', '유의파주기(sec)', 'Tm(sec)', 'TP', 'tm']
    col_dir = ['파향(deg)', 'Dir(deg)', '파향', 'dir']
    col_time = ['관측시간', 'datetime', 'time', 'date', 'DateTime', 'DATE']

    station_data = {}
    for name, meta in STATIONS.items():
        fp = os.path.join(root_dir, meta["file"])
        if not os.path.isfile(fp):
            print(f"[warn] CSV not found → {fp}")
            station_data[name] = np.full((len(time_index), 3), np.nan, dtype=np.float32)
            continue

        df = pd.read_csv(fp, na_values=['', ' ', 'NaN'])
        tcol = next((c for c in col_time if c in df.columns), None)
        if tcol is None:
            raise ValueError(f"{fp}: 시간 컬럼을 찾지 못했습니다.")
        df[tcol] = pd.to_datetime(df[tcol], errors='coerce').dt.tz_localize('Asia/Seoul',
                                                                            nonexistent='shift_forward').dt.tz_convert(
            'UTC').dt.tz_localize(None)
        df = df.set_index(tcol)

        hcol = next((c for c in col_hs if c in df.columns), None)
        pcol = next((c for c in col_tm if c in df.columns), None)
        dcol = next((c for c in col_dir if c in df.columns), None)

        df_std = pd.DataFrame(index=df.index)
        df_std['hs'] = df[hcol] if hcol else np.nan
        df_std['tm'] = df[pcol] if pcol else np.nan
        df_std['dir'] = df[dcol] if dcol else np.nan

        df_std['hs'] = df_std['hs'].replace(0, np.nan)
        df_std['hs'] = (df_std['hs'] - norm_params['hs'][0]) / (norm_params['hs'][1] - norm_params['hs'][0])
        df_std['tm'] = (df_std['tm'] - norm_params['tm'][0]) / (norm_params['tm'][1] - norm_params['tm'][0])
        df_std['dir'] = (df_std['dir'] - norm_params['dir'][0]) / (norm_params['dir'][1] - norm_params['dir'][0])

        df_std = df_std.reindex(time_index, method=None)
        station_data[name] = df_std[['hs', 'tm', 'dir']].values.astype(np.float32)
    return station_data

# =========================
# Dataset / Samplers
# =========================
class WindWaveDataset(Dataset):
    def __init__(self, input_data, wave_data, seq_length, start_idx, end_idx):
        self.input_data = input_data
        self.wave_data = wave_data
        self.seq_length = seq_length
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        seq_X = self.input_data[actual_idx:actual_idx + self.seq_length]
        seq_y = self.wave_data[actual_idx + self.seq_length]
        return torch.from_numpy(seq_X).float(), torch.from_numpy(seq_y).float()

class SubsetIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds; self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]

class PeakSamplerRestricted(repair.PeakSampler):
    pass

def safe_collate(batch):
    return repair.strict_collate(batch)

# =========================
# Geometry utils
# =========================
def _vec2deg(sin_, cos_):
    ang = np.rad2deg(np.arctan2(sin_, cos_)) % 360
    return ang

# =========================
# Model blocks
# =========================
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
        while out_c % gn != 0 and gn > 1: gn //= 2
        self.norm = nn.GroupNorm(gn, out_c)
        self.se  = SEBlock(out_c)
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout2d(p=DROPOUT_P)
    def forward(self, x):
        x = self.conv_pw(self.conv_dw(x))
        x = self.act(self.drop(self.se(self.norm(x))))
        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        pad = k // 2; self.h = hid_c
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=pad)
    def forward(self, x, s):
        h, c = s
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], 1)), self.h, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g); c = f * c + i * g; h = o * torch.tanh(c)
        return h, c
    def init_state(self, B, H, W):
        z = torch.zeros(B, self.h, H, W, device=device)
        return z.clone(), z.clone()

class UNetPlusPlus(nn.Module):
    def __init__(self, in_c, out_c, feat):
        super().__init__()
        f = feat
        self.enc00 = ImprovedConvBlock(in_c, f[0]); self.pool = nn.MaxPool2d(2,2)
        self.enc10 = ImprovedConvBlock(f[0], f[1])
        self.enc20 = ImprovedConvBlock(f[1], f[2])
        self.enc30 = ImprovedConvBlock(f[2], f[3])
        self.enc40 = ImprovedConvBlock(f[3], f[4])

        self.dec01 = ImprovedConvBlock(f[0] + f[1], f[0])
        self.dec11 = ImprovedConvBlock(f[1] + f[2], f[1])
        self.dec21 = ImprovedConvBlock(f[2] + f[3], f[2])
        self.dec31 = ImprovedConvBlock(f[3] + f[4], f[3])

        self.dec02 = ImprovedConvBlock(f[0]*2 + f[1], f[0])
        self.dec12 = ImprovedConvBlock(f[1]*2 + f[2], f[1])
        self.dec22 = ImprovedConvBlock(f[2]*2 + f[3], f[2])

        self.dec03 = ImprovedConvBlock(f[0]*3 + f[1], f[0])
        self.dec13 = ImprovedConvBlock(f[1]*3 + f[2], f[1])

        self.dec04 = ImprovedConvBlock(f[0]*4 + f[1], f[0])
        self.outs  = nn.ModuleList([nn.Conv2d(f[0], out_c, 1) for _ in range(4)])

    def _u(self, x, y):
        return torch.cat([F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False), y], 1)

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
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=64, feat=[24,48,96,192,384]):
        super().__init__()
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.lstm = ConvLSTMCell(output_channels, hidden_dim)
        self.final_drop = nn.Dropout2d(p=DROPOUT_P)
        self.head = nn.Conv2d(hidden_dim, output_channels, 1)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [Hs, Tm, Dir]
    def forward(self, x):
        B, T, _, H, W = x.shape
        h, c = self.lstm.init_state(B, H, W); last_u = None
        for t in range(T):
            outs = self.unet(x[:, t]); last_u = outs
            h, c = self.lstm(outs[0], (h, c))
        h = self.final_drop(h)
        return [self.head(h)] + last_u

# =========================
# Losses (incl. circular dir, deep supervision, TV)
# =========================

def ds_loss(
    pred, target,
    spatial_weight=None, valid_mask=None,
    log_vars=None, use_huber=False, huber_beta=0.1, eps=1e-8
):
    """Legacy function retained except for the documented repair changes."""
    if pred.ndim != 4 or target.ndim != 4 or pred.shape[1] != 4 or target.shape[1] != 4:
        raise ValueError('ds_loss requires BCHW tensors with four channels')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite prediction or target in ds_loss')
    pred=pred.float().movedim(1,-1); target=target.float().movedim(1,-1)
    phs,ptm=pred[...,0],pred[...,1]; ths,ttm=target[...,0],target[...,1]
    pvec=F.normalize(pred[...,2:4],dim=-1,eps=1e-6)
    tvec=F.normalize(target[...,2:4],dim=-1,eps=1e-6)
    psin,pcos=pvec[...,0],pvec[...,1]; tsin,tcos=tvec[...,0],tvec[...,1]

    if use_huber:
        def crit(a,b): return F.smooth_l1_loss(a,b,beta=huber_beta,reduction="none")
    else:
        def crit(a,b): return F.l1_loss(a,b,reduction="none")

    w = 1.0
    if spatial_weight is not None: w = w * spatial_weight
    if valid_mask    is not None: w = w * valid_mask
    w = torch.as_tensor(w, dtype=phs.dtype, device=phs.device)

    def wmean(x):
        num = (x * w).nan_to_num(0.0).sum()
        den = (torch.ones_like(x) * w).nan_to_num(0.0).sum().clamp_min(eps)
        return num / den

    loss_hs = wmean(crit(phs, ths))
    loss_tm = wmean(crit(ptm, ttm))
    # circular direction loss
    cos_delta = (pcos * tcos + psin * tsin).clamp(-1.0, 1.0)   # Retained legacy processing.
    direction_valid = (torch.linalg.vector_norm(target[...,2:4],dim=-1) > 1e-6)
    dw=w*direction_valid
    loss_dir=((1.0-cos_delta)*dw).sum()/((torch.ones_like(cos_delta)*dw).sum().clamp_min(eps))

    if (log_vars is not None) and isinstance(log_vars, torch.nn.Parameter):
        total = 0.0
        for i, Li in enumerate([loss_hs, loss_tm, loss_dir]):
            s2 = torch.exp(-log_vars[i])
            total = total + 0.5 * (s2 * Li + log_vars[i])
    else:
        total = loss_hs + loss_tm + loss_dir
    return total, (loss_hs.detach(), loss_tm.detach(), loss_dir.detach())

def tv2d(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

def deep_supervised_loss(main_pred, aux_preds, target, w, log_vars, lambda_tv=2e-3):
    """
    Deep supervision + TV regularization(Hs/Tm)
    """
    weights = [1.0, 0.5, 0.25, 0.125]   # main + x03 + x02 + x01
    preds = [main_pred] + list(aux_preds)
    total = 0.0; last_parts = None
    for pr, wt in zip(preds, weights):
        tgt = F.interpolate(target, size=pr.shape[-2:], mode='bilinear', align_corners=False)
        loss_tot, parts = ds_loss(pr, tgt, spatial_weight=w, log_vars=log_vars)
        total = total + wt * loss_tot
        last_parts = parts

    # TV on Hs/Tm of main prediction
    tv_loss = tv2d(main_pred[:,0]) + tv2d(main_pred[:,1])
    total = total + lambda_tv * tv_loss
    return total, last_parts

# =========================
# Metrics (phys MAE)
# =========================
def _vec2deg_np(sin_, cos_):
    return (np.rad2deg(np.arctan2(sin_, cos_)) % 360.0)

def compute_mae_phys(model, loader, w, norm_params):
    model.eval()
    total_weighted_error = {"hs": 0.0, "tm": 0.0, "dir": 0.0}
    total_weights = {"hs": 0.0, "tm": 0.0, "dir": 0.0}
    vmin_hs, vmax_hs = norm_params["hs"]; vmin_tm, vmax_tm = norm_params["tm"]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device).float()
            pred_maps = model(xb)
            pred = pred_maps[0] if isinstance(pred_maps, list) else pred_maps
            pred = pred.float()
            B, _, H, W = pred.shape
            w_2d = w.detach().float().cpu().numpy()
            if w_2d.ndim == 2: pass
            elif w_2d.ndim == 3: w_2d = w_2d[0]
            elif w_2d.ndim == 4: w_2d = w_2d[0,0]
            else: w_2d = np.ones((H,W), np.float32)
            if w_2d.shape != (H,W):
                w_2d = nd_zoom(w_2d, (H/w_2d.shape[0], W/w_2d.shape[1]), order=1)
            w_2d = np.nan_to_num(w_2d, nan=0.0).astype(np.float32)
            if np.sum(w_2d) < 1e-12: w_2d = np.ones_like(w_2d)/w_2d.size

            pred_hs = denorm(pred[:,0].detach().cpu().numpy().astype(np.float32), vmin_hs, vmax_hs)
            true_hs = denorm(yb[:,0].detach().cpu().numpy().astype(np.float32), vmin_hs, vmax_hs)
            pred_tm = denorm(pred[:,1].detach().cpu().numpy().astype(np.float32), vmin_tm, vmax_tm)
            true_tm = denorm(yb[:,1].detach().cpu().numpy().astype(np.float32), vmin_tm, vmax_tm)
            pred_dir = _vec2deg_np(pred[:,2].detach().cpu().numpy(), pred[:,3].detach().cpu().numpy())
            true_dir = _vec2deg_np(yb[:,2].detach().cpu().numpy(),  yb[:,3].detach().cpu().numpy())

            for b in range(B):
                valid = np.isfinite(w_2d) & (w_2d>1e-12)
                if not valid.any(): continue
                ws = w_2d[valid]; wsum = float(ws.sum())
                hs_err = np.abs(pred_hs[b][valid] - true_hs[b][valid])
                tm_err = np.abs(pred_tm[b][valid] - true_tm[b][valid])
                diff = (pred_dir[b][valid] - true_dir[b][valid] + 180.0) % 360.0 - 180.0
                dir_err = np.abs(diff)
                total_weighted_error["hs"]  += float(np.sum(hs_err  * ws))
                total_weighted_error["tm"]  += float(np.sum(tm_err  * ws))
                total_weighted_error["dir"] += float(np.sum(dir_err * ws))
                total_weights["hs"] += wsum; total_weights["tm"] += wsum; total_weights["dir"] += wsum

    out = {}
    for k in ["hs","tm","dir"]:
        out[k] = total_weighted_error[k] / max(total_weights[k], 1e-12)
    return out

# =========================
# Train / Eval
# =========================
def compute_loss(model, loader, w, lambda_tv=2e-3):
    model.eval(); total = 0.0; n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                outs = model(xb)
                main_pred = outs[0]; aux_preds = outs[1:]
                loss_tot, _ = deep_supervised_loss(main_pred, aux_preds, yb, w, getattr(model, "log_vars", None), lambda_tv)
            total += loss_tot.item() * xb.size(0); n += xb.size(0)
    return total / max(n,1)

def train(model, dl_tr, dl_va, w, *, epochs, acc_steps, norm_params, ckpt_prefix="ckpt", freeze_logvars_epochs=1, early_stop_patience=0, lambda_tv=2e-3):
    return repair.train_loop(globals(), model, dl_tr, dl_va, w, epochs=epochs, acc_steps=acc_steps,
        norm_params=norm_params, ckpt_prefix=ckpt_prefix, freeze_logvars_epochs=freeze_logvars_epochs,
        early_stop_patience=early_stop_patience, lambda_tv=lambda_tv)

# =========================
# Split maker (block stratified)
# =========================
def make_block_stratified_split(
    wave_data, seq_length, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
    block_hours=168, q=5, seed=42, embargo_hours=None
):
    if embargo_hours is None: embargo_hours = seq_length
    T = wave_data.shape[0]; N = T - seq_length
    if N <= 0: return np.array([],int), np.array([],int), np.array([],int)

    block_len = int(block_hours)
    group = np.floor_divide(np.arange(N), block_len)
    num_blocks = int(group.max()) + 1

    hs_t = wave_data[seq_length:, 0]
    hs_max_t = hs_t.reshape(N, -1).max(axis=1)
    block_scores = np.full(num_blocks, -np.inf, dtype=np.float32)
    for b in range(num_blocks):
        t_idx = np.where(group == b)[0]
        if t_idx.size > 0:
            block_scores[b] = np.nanpercentile(hs_max_t[t_idx], 95)

    valid_scores = block_scores[np.isfinite(block_scores)]
    if valid_scores.size == 0:
        labels = np.zeros(num_blocks, dtype=int)
    else:
        qbins = np.quantile(valid_scores, np.linspace(0, 1, q + 1))
        labels = np.digitize(block_scores, qbins[1:-1], right=True).astype(int)

    rng = np.random.default_rng(seed)
    tr_blocks, va_blocks, te_blocks = [], [], []
    for lab in range(q):
        b_lab = np.where(labels == lab)[0]; rng.shuffle(b_lab)
        n = len(b_lab); n_tr = int(round(n*train_ratio)); n_va = int(round(n*val_ratio)); n_te = n - n_tr - n_va
        if n_te < 0: n_te = max(0, n - n_tr - n_va); n_va = max(0, n - n_tr - n_te)
        tr_blocks.extend(b_lab[:n_tr]); va_blocks.extend(b_lab[n_tr:n_tr+n_va]); te_blocks.extend(b_lab[n_tr+n_va:])
    tr_blocks = repair.split_hook(tr_blocks, labels, q, seed)
    tr_blocks, va_blocks, te_blocks = set(tr_blocks), set(va_blocks), set(te_blocks)
    blk2set = {b:(0 if b in tr_blocks else (1 if b in va_blocks else (2 if b in te_blocks else -1))) for b in range(num_blocks)}
    idx_tr = np.where([blk2set[g]==0 for g in group])[0]
    idx_va = np.where([blk2set[g]==1 for g in group])[0]
    idx_te = np.where([blk2set[g]==2 for g in group])[0]

    def _same_set_ok(t):
        if t - seq_length < 0: return False
        return blk2set[group[t]] == blk2set[group[t - seq_length]]

    idx_tr = np.array([t for t in idx_tr if _same_set_ok(t)], dtype=int)
    idx_va = np.array([t for t in idx_va if _same_set_ok(t)], dtype=int)
    idx_te = np.array([t for t in idx_te if _same_set_ok(t)], dtype=int)

    if embargo_hours and embargo_hours > 0:
        emb = int(embargo_hours)
        def _filter_embargo(idxs):
            keep = []
            for t in idxs:
                pos = t % block_len
                left_ok  = pos >= emb
                right_ok = pos <= (block_len - seq_length - emb - 1)
                if left_ok and right_ok: keep.append(t)
            return np.array(keep, dtype=int)
        idx_tr = _filter_embargo(idx_tr); idx_va = _filter_embargo(idx_va); idx_te = _filter_embargo(idx_te)
    return idx_tr, idx_va, idx_te

# =========================
# Retained legacy processing.
# =========================
def find_nearest_index(lon_map, lat_map, kcs_map, target_lon, target_lat):
    valid_indices = np.where(kcs_map > 0)
    if valid_indices[0].size == 0: return 0, 0
    valid_lons = lon_map[valid_indices]; valid_lats = lat_map[valid_indices]
    distances = np.sqrt((valid_lons - target_lon)**2 + (valid_lats - target_lat)**2)
    nearest_idx = np.argmin(distances)
    m_idx = valid_indices[0][nearest_idx]; n_idx = valid_indices[1][nearest_idx]
    return m_idx, n_idx

def _plot_spatial_sample(pred, true, kcs, extent, fname, var_name="Variable", title_suffix=""):
    plt.figure(figsize=(18,5))
    valid_mask = (kcs > 0) & np.isfinite(true)
    vmin = np.nanmin(true[valid_mask]) if valid_mask.any() else 0
    vmax = np.nanmax(true[valid_mask]) if valid_mask.any() else 1
    is_dir=var_name.lower()=='dir'
    err=np.abs((pred-true+180.)%360.-180.) if is_dir else np.abs(pred-true)
    if is_dir: vmin,vmax=0.,360.
    cmap='twilight' if is_dir else 'jet'
    items=[('Pred',pred,cmap),('True',true,cmap),('|Err|',err,'magma')]
    for i, (ttl, dat, cmap) in enumerate(items, 1):
        ax = plt.subplot(1,3,i)
        d = np.ma.masked_where(kcs <= 0, dat)
        err_max = np.nanmax(d)
        lo, hi = (vmin, vmax) if i < 3 else (0, 180. if is_dir else (err_max + 1e-6 if err_max is not np.ma.masked else 1))
        im = ax.imshow(d, cmap=cmap, origin="lower", extent=extent, vmin=lo, vmax=hi)
        ax.set_title(ttl); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f"Spatial sample for {var_name}" + title_suffix)
    plt.tight_layout(); plt.savefig(fname); plt.close()

def _plot_timeseries(ts_dict, station_name, var_name, fname, norm_params=(0.0,1.0)):
    import re
    dir_, base = os.path.split(fname)
    base = re.sub(r'[\\/:\"*?<>|]+', '_', base); fname = os.path.join(dir_, base)
    os.makedirs(dir_ or '.', exist_ok=True)
    vmin, vmax = norm_params
    if var_name == 'dir':
        true_denorm = ts_dict["true"]; pred_denorm = ts_dict["pred"]
        meas_denorm = [denorm(x, vmin, vmax) if np.isfinite(x) else np.nan for x in ts_dict["meas"]]
    else:
        true_denorm = [denorm(x, vmin, vmax) for x in ts_dict["true"]]
        pred_denorm = [denorm(x, vmin, vmax) for x in ts_dict["pred"]]
        meas_denorm = [denorm(x, vmin, vmax) if np.isfinite(x) else np.nan for x in ts_dict["meas"]]
    plt.figure(figsize=(12,5))
    plt.plot(true_denorm, "k-", label="simulation")
    plt.plot(pred_denorm, "b--", label="predicted")
    plt.plot(meas_denorm, "r:", label="measured")
    plt.title(f"{station_name} – {var_name} (denormalized)")
    plt.xlabel("Time index (test)"); plt.ylabel(f"{var_name} (physical unit)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(fname); plt.close()

def evaluate_and_visualize(model, loader, lon_map, lat_map, kcs_map, station_meta, station_data,
                           test_start_idx, seq_length, global_norm_params, out_prefix="eval",
                           save_limit=5, time_steps_info="N/A", seq_length_info="N/A",
                           epochs_info="N/A", alpha="N/A", base_indices=None):
    model.eval(); os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    def metrics(pred, true, vmin, vmax):
        p = denorm(pred, vmin, vmax); t = denorm(true, vmin, vmax)
        mask = np.isfinite(p) & np.isfinite(t); p, t = p[mask], t[mask]
        if len(p)==0: return dict(rmse=np.nan, mae=np.nan, r2=np.nan, smape=np.nan)
        rmse = np.sqrt(((p - t) ** 2).mean()); mae = np.abs(p - t).mean()
        r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        smape = (np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-6)).mean()
        return dict(rmse=rmse, mae=mae, r2=r2, smape=smape)
    if base_indices is not None: order = np.argsort(base_indices)
    else: order = None

    st_indices = {n: find_nearest_index(lon_map, lat_map, kcs_map, m["lon"], m["lat"]) for n, m in station_meta.items()}
    cos_lat = np.cos(np.deg2rad(lat_map)); cos_lat[(kcs_map<=0) | ~np.isfinite(cos_lat)] = 0
    spatial_w = cos_lat / (cos_lat.sum() + 1e-12)

    lon_f = lon_map[np.isfinite(lon_map)]; lat_f = lat_map[np.isfinite(lat_map)]
    extent = [lon_f.min(), lon_f.max(), lat_f.min(), lat_f.max()]

    variables = ['hs','tm','dir']
    st_ts = {n:{v: {"pred":[], "true":[], "meas":[]} for v in variables} for n in station_meta}
    metric_results = {k: [] for k in ["rmse","bias","mae","pred_wmean","true_wmean","cc","r2","nse","mape","smape","rmse_m","mae_m","r2_m","smape_m","acc"]}

    save_cnt = 0; seen=0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate", leave=False):
            x, y = batch; x, y = x.to(device), y.to(device)
            pred_maps = model(x); Bcur = x.size(0)
            for b in range(Bcur):
                full_t = (int(base_indices[seen+b]) + seq_length) if base_indices is not None else (int(test_start_idx) + seq_length + (seen+b))
                pred_hs = pred_maps[0][b,0].detach().cpu().numpy()
                pred_tm = pred_maps[0][b,1].detach().cpu().numpy()
                pred_sin= pred_maps[0][b,2].detach().cpu().numpy()
                pred_cos= pred_maps[0][b,3].detach().cpu().numpy()
                true_hs = y[b,0].detach().cpu().numpy()
                true_tm = y[b,1].detach().cpu().numpy()
                true_sin= y[b,2].detach().cpu().numpy()
                true_cos= y[b,3].detach().cpu().numpy()

                pred_dir = _vec2deg_np(pred_sin, pred_cos); true_dir = _vec2deg_np(true_sin, true_cos)

                oce = (kcs_map>0) & np.isfinite(pred_hs) & np.isfinite(true_hs)
                if oce.any():
                    diff = pred_hs[oce] - true_hs[oce]
                    phys = metrics(pred_hs[oce], true_hs[oce], global_norm_params['hs'][0], global_norm_params['hs'][1])
                    metric_results["rmse"].append(np.sqrt((diff**2).mean()))
                    metric_results["bias"].append(diff.mean()); metric_results["mae"].append(np.abs(diff).mean())
                    metric_results["rmse_m"].append(phys["rmse"]); metric_results["mae_m"].append(phys["mae"])
                    metric_results["r2_m"].append(phys["r2"]); metric_results["smape_m"].append(phys["smape"])
                    metric_results["pred_wmean"].append(np.nansum(pred_hs*spatial_w))
                    metric_results["true_wmean"].append(np.nansum(true_hs*spatial_w))

                for n, (m, n_) in st_indices.items():
                    is_valid_time = 0 <= full_t < station_data[n].shape[0]
                    st_ts[n]['hs']["true"].append(true_hs[m, n_]); st_ts[n]['hs']["pred"].append(pred_hs[m, n_])
                    st_ts[n]['hs']["meas"].append(station_data[n][full_t, 0] if is_valid_time else np.nan)
                    st_ts[n]['tm']["true"].append(true_tm[m, n_]); st_ts[n]['tm']["pred"].append(pred_tm[m, n_])
                    st_ts[n]['tm']["meas"].append(station_data[n][full_t, 1] if is_valid_time else np.nan)
                    st_ts[n]['dir']["true"].append(true_dir[m, n_]); st_ts[n]['dir']["pred"].append(pred_dir[m, n_])
                    st_ts[n]['dir']["meas"].append(station_data[n][full_t, 2] if is_valid_time else np.nan)

                if save_cnt < save_limit:
                    np.savez_compressed(f'{out_prefix}_raw_{full_t}.npz',
                        pred=pred_maps[0][b].detach().float().cpu().numpy(),
                        true=y[b].detach().float().cpu().numpy(),kcs=kcs_map,
                        direction_radius=np.hypot(pred_sin,pred_cos),time_index=full_t)
                    _plot_spatial_sample(pred_hs, true_hs, kcs_map, extent, f"{out_prefix}_spatial_hs_{full_t}.png", "Hs", f" (t={full_t}, ep={epochs_info})")
                    _plot_spatial_sample(pred_tm, true_tm, kcs_map, extent, f"{out_prefix}_spatial_tm_{full_t}.png", "Tm", f" (t={full_t}, ep={epochs_info})")
                    _plot_spatial_sample(pred_dir, true_dir, kcs_map, extent, f"{out_prefix}_spatial_dir_{full_t}.png", "Dir", f" (t={full_t}, ep={epochs_info})")
                    save_cnt += 1
            seen += Bcur

    if order is not None:
        expected_len = len(order)
        for n, station_vars in st_ts.items():
            for var_name, ts in station_vars.items():
                for key in ("pred","true","meas"):
                    if len(ts[key]) == expected_len:
                        ts[key] = [ts[key][i] for i in order]

    for n, station_vars in st_ts.items():
        for var_name, ts_dict in station_vars.items():
            if np.isfinite(ts_dict["meas"]).sum() == 0: continue
            file_path = f"{out_prefix}_ts_{n}_{var_name}_ts{time_steps_info}_seq{seq_length_info}_ep{epochs_info}_a{alpha}.png"
            np_norm_params = global_norm_params[var_name]
            _plot_timeseries(ts_dict, n, var_name, file_path, np_norm_params)

    summary = {k: (np.nanmean(v) if v else np.nan) for k, v in metric_results.items()}
    print("\n###  test-set summary (Hs)  ###")
    for k, v in summary.items(): print(f"{k:>12}: {v:8.4f}")
    return summary

# =========================
# Save/plot training curves
# =========================
def save_training_results(training_results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Training and Validation Results\n" + "="*50 + "\n\n")
        f.write(f"{'Epoch':<6} {'TrainLoss':<12} {'ValLoss':<12} "
                f"{'HsMAE_tr(m)':<12} {'HsMAE_va(m)':<12} "
                f"{'TmMAE_tr(s)':<12} {'TmMAE_va(s)':<12} "
                f"{'DirMAE_tr(°)':<12} {'DirMAE_va(°)':<12}\n")
        f.write("-"*100 + "\n")
        for i, epoch in enumerate(training_results['epochs']):
            f.write(f"{epoch:<6} {training_results['train_losses'][i]:<12.6f} "
                    f"{training_results['val_losses'][i]:<12.6f} "
                    f"{training_results['train_mae_hs'][i]:<12.6f} {training_results['val_mae_hs'][i]:<12.6f} "
                    f"{training_results['train_mae_tm'][i]:<12.6f} {training_results['val_mae_tm'][i]:<12.6f} "
                    f"{training_results['train_mae_dir'][i]:<12.6f} {training_results['val_mae_dir'][i]:<12.6f}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Final Results:\n")
        f.write(f"Final Training Loss: {training_results['train_losses'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {training_results['val_losses'][-1]:.6f}\n")
        f.write(f"Final Hs MAE (train/val): {training_results['train_mae_hs'][-1]:.6f} / {training_results['val_mae_hs'][-1]:.6f} m\n")
        f.write(f"Final Tm MAE (train/val): {training_results['train_mae_tm'][-1]:.6f} / {training_results['val_mae_tm'][-1]:.6f} s\n")
        f.write(f"Final Dir MAE (train/val): {training_results['train_mae_dir'][-1]:.6f} / {training_results['val_mae_dir'][-1]:.6f} °\n")
    print(f"Training results saved to: {filename}")

def plot_training_results(training_results, filename):
    plt.figure(figsize=(12, 8)); epochs = training_results['epochs']
    plt.plot(epochs, training_results['train_losses'], 'b-o', linewidth=2, markersize=4, label='Train Loss')
    plt.plot(epochs, training_results['val_losses'], 'r-s', linewidth=2, markersize=4, label='Val Loss')
    plt.plot(epochs, training_results['train_mae_hs'], linewidth=1.5, label='Hs MAE (train) [m]')
    plt.plot(epochs, training_results['val_mae_hs'], linewidth=1.5, label='Hs MAE (val) [m]')
    plt.plot(epochs, training_results['train_mae_tm'], linewidth=1.5, label='Tm MAE (train) [s]')
    plt.plot(epochs, training_results['val_mae_tm'], linewidth=1.5, label='Tm MAE (val) [s]')
    plt.plot(epochs, training_results['train_mae_dir'], linewidth=1.5, label='Dir MAE (train) [deg]')
    plt.plot(epochs, training_results['val_mae_dir'], linewidth=1.5, label='Dir MAE (val) [deg]')
    plt.title('Training / Validation: Loss + Physical MAEs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss / MAE'); plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3); plt.ylim(0, None); plt.xlim(0, max(epochs))
    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Training plot saved to: {filename}")

# =========================================================
# Retained legacy processing.
# =========================================================
def merge_seg_series_dicts(*dicts):
    out = {}
    all_names = set().union(*[d.keys() for d in dicts if d])
    for name in all_names:
        dfs = [d[name] for d in dicts if (d and name in d)]
        if not dfs:
            continue
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[name] = df
    return out

# =========================
# Wrapper
# =========================
def wrapper(data_path, use_bnd="on"):  # "on" | "off" | "auto"
    repair.fraction()
    import itertools, os, traceback
    import numpy as np
    import pandas as pd
    import xarray as xr
    import torch
    from torch.utils.data import DataLoader
    from scipy.ndimage import binary_dilation

    # Retained legacy processing.
    try:
        from bnd_features import (
            read_all_bnds,
            build_owner_label,
            make_boundary_feature_maps,
            assert_on_edges,
        )
        from boundspec_segments import SEGMENTS, M as SWAN_M, N as SWAN_N
        _HAS_BND = True
    except Exception as _e:
        print(f"[BND] 경계 특성 모듈을 불러오지 못했습니다 → {type(_e).__name__}: {_e}")
        _HAS_BND = False

    # Retained legacy processing.
    env_flag = os.getenv("USE_BND_FEATURES", "auto").strip().lower()
    flag = (use_bnd or "auto").strip().lower()
    if flag not in ("on", "off", "auto"):
        flag = "auto"
    if flag == "auto":
        flag = env_flag if env_flag in ("on","off","auto") else "auto"

    if flag == "on":
        # Explicit request. If the helper modules are missing, do not quietly fall
        # back to bndOFF, since the run would then train on 6-channel input under a
        # bndON label. Stop loudly so the missing dependency is fixed first.
        if not _HAS_BND:
            raise RuntimeError(
                "use_bnd='on' was requested but the BND helper modules "
                "(bnd_features, boundspec_segments) could not be imported. Refusing to "
                "continue, because this would silently train without BND. Make sure both "
                "files are on PYTHONPATH next to the training script."
            )
        USE_BND_FEATURES = _HAS_BND
    elif flag == "off":
        USE_BND_FEATURES = False
    else:  # auto
        USE_BND_FEATURES = _HAS_BND

    bnd_tag = "bndON" if USE_BND_FEATURES else "bndOFF"
    print(f"[BND] boundary features: {bnd_tag} (request='{flag}', has_bnd={_HAS_BND})")

    # Retained legacy processing.
    # Retained legacy processing.
    # Retained legacy processing.
    combinations = list(itertools.product(time_steps_list, seq_length_list, epochs_list, hidden_dim_list, unet_feat_list))

    # Retained legacy processing.
    csv_filename = f"performance_summary_global_norm_v61_{bnd_tag}.csv"
    HEADER = ["time_steps", "seq_length", "epochs", "hidden_dim", "rmse_m", "mae_m", "r2_m", "smape_m",
              "pred_weighted_mean", "true_weighted_mean", "train_loss_final", "val_loss_final"]
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        import csv; csv.writer(f).writerow(HEADER)

    # Retained legacy processing.
    # Retained legacy processing.
    # Retained legacy processing.
    def robust_block_split(wave_data_for_split, seq_length, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, times=None):
        return repair.strict_split(make_block_stratified_split, wave_data_for_split, seq_length,
            train_ratio=train_ratio,val_ratio=val_ratio,test_ratio=test_ratio,times=times)

    # Retained legacy processing.
    # Retained legacy processing.
    # Retained legacy processing.
    BND_DIRS_BY_YEAR = {
        2021: os.environ["SWAN_BND_DIR_2021"],
        2019: os.environ.get(
            "SWAN_BND_DIR_2019",
            r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2019"),
        2020: os.environ.get(
            "SWAN_BND_DIR_2020",
            r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2020"),
    }
    bnd_direction = "from"  # Retained legacy processing.

    # Retained legacy processing.
    # Retained legacy processing.
    # Retained legacy processing.
    def _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index, seq_length):
        return repair.align_bnd(bnd_feat,ds_sim,kcs2d,time_index,seq_length)

    # Retained legacy processing.
    # Retained legacy processing.
    # Retained legacy processing.
    for ci, (time_steps, seq_length, epochs, hidden_dim, unet_feat) in enumerate(combinations, 1):
        print(f"\n===== Comb {ci}/{len(combinations)} =====")
        print(f"ts={time_steps}  L={seq_length}  ep={epochs}  hid={hidden_dim}")
        torch.cuda.empty_cache()
        try:
            if not os.path.isfile(data_path): raise FileNotFoundError(f"NetCDF not found: {data_path}")
            ds_sim = xr.open_dataset(data_path)

            # Retained legacy processing.
            N = time_steps - seq_length
            repair.validate_source(ds_sim,time_steps)
            hs_raw = ds_sim["hsign"].values[:time_steps]
            Y, X = hs_raw.shape[-2], hs_raw.shape[-1]
            wave_data_for_split = np.zeros((time_steps, 1, Y, X), dtype=np.float32); wave_data_for_split[:,0] = hs_raw
            idx_tr, idx_va, idx_te, split_tag = robust_block_split(wave_data_for_split, seq_length, times=ds_sim["time"].values[:time_steps])

            # Retained legacy processing.
            global_norm_params = compute_params_with_indices(ds_sim, idx_train=idx_tr, seq_length=seq_length)
            repair.atomic_json('normalization.json',global_norm_params)
            print("Hs range:", global_norm_params['hs'], "  Tm range:", global_norm_params['tm'])

            # Retained legacy processing.
            input_data, wave_data, lon, lat, kcs = load_and_preprocess_data(ds_sim, global_norm_params, time_steps=time_steps)

            # Retained legacy processing.
            if 'time' in ds_sim:
                tvals = pd.to_datetime(ds_sim['time'].values[:time_steps])
                time_index = pd.DatetimeIndex(tvals).tz_localize(None)
            else:
                time_index = pd.date_range(start="2019-01-01 00:00:00", periods=time_steps, freq="h", tz="UTC").tz_localize(None)

            # Retained legacy processing.
            # Retained legacy processing.
            # Retained legacy processing.
            if USE_BND_FEATURES:
                try:
                    # Retained legacy processing.
                    kcs2d = kcs[0] if kcs.ndim == 3 else kcs
                    H, W = kcs2d.shape
                    if   (H == SWAN_M and W == SWAN_N):   swap_ij = False
                    elif (H == SWAN_N and W == SWAN_M):   swap_ij = True
                    else:
                        raise ValueError(f"Grid mismatch: data(H,W)=({H},{W}) vs SWAN(M,N)=({SWAN_M},{SWAN_N}). 자동 스케일 금지.")

                    # Retained legacy processing.
                    assert_on_edges(SEGMENTS, M=SWAN_M, N=SWAN_N)

                    # Retained legacy processing.
                    years_needed = sorted(set(pd.DatetimeIndex(time_index).year))
                    seg_dicts = []
                    for y in years_needed:
                        bdir = BND_DIRS_BY_YEAR.get(y, None)
                        if bdir and os.path.isdir(bdir):
                            seg_y = read_all_bnds(Path(bdir), direction=bnd_direction)
                            seg_dicts.append(seg_y)
                            print(f"[BND] {y}: {bdir} 에서 .BND 로딩 완료")
                        else:
                            print(f"[BND] 경고: {y}년 BND 폴더가 설정되지 않았거나 존재하지 않습니다 → {bdir}")
                    if not seg_dicts:
                        raise FileNotFoundError(f"[BND] 필요한 연도 {years_needed}에 해당하는 .BND를 찾지 못했습니다.")

                    seg_series = merge_seg_series_dicts(*seg_dicts)

                    # Retained legacy processing.
                    owner_label, id2name = build_owner_label(
                        H, W, segments=SEGMENTS, exact_M=SWAN_M, exact_N=SWAN_N,
                        kcs=kcs2d, swap_ij=swap_ij
                    )

                    # Retained legacy processing.
                    bnd_feat = make_boundary_feature_maps(
                        time_index=time_index,
                        owner_label=owner_label,
                        seg_series=seg_series,
                        id2name=id2name,
                        kcs=kcs2d,
                        norm_hs=global_norm_params['hs'],
                        norm_tm=global_norm_params['tm'],
                    )

                    if bnd_feat.shape[0] != input_data.shape[0]:
                        raise ValueError(f"BND T={bnd_feat.shape[0]} vs input T={input_data.shape[0]}")
                    if bnd_feat.shape[2:] != input_data.shape[2:]:
                        raise ValueError(f"BND HW={bnd_feat.shape[2:]} vs input HW={input_data.shape[2:]}")

                    # Retained legacy processing.
                    best_deg, scores = _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index, seq_length)
                    msg = " ".join([f"{k}:{v:.4f}" for k,v in scores.items()])
                    chosen = (f"reflection theta'={best_deg-1000:.0f}-theta" if best_deg >= 1000
                              else f"rotation {best_deg:+.0f}°")
                    print(f"[BND] dir autocorrect → chosen {chosen} | scores {msg}")

                    # Retained legacy processing.
                    input_data = np.concatenate([input_data, bnd_feat], axis=1)
                    print(f"[BND] 경계 특성 4채널 추가 완료 → input channels = {input_data.shape[1]}")
                except Exception as be:
                    # If BND was explicitly requested (flag == "on"), a failure here would
                    # silently train on 6-channel input while the run is labeled bndON, which
                    # invalidates the experiment. Stop loudly instead of continuing. For
                    # "auto", best-effort skipping is still acceptable.
                    if flag == "on":
                        raise RuntimeError(
                            f"use_bnd='on' was requested but boundary features could not be "
                            f"combined ({type(be).__name__}: {be}). Refusing to continue, because "
                            f"this would train on inputs without BND under a bndON label. "
                            f"Check SWAN_BND_DIR_2019 and SWAN_BND_DIR_2020."
                        ) from be
                    print(f"[BND] 경계 특성 결합을 건너뜁니다: {type(be).__name__}: {be}")

            # Retained legacy processing.
            station_root_dir = os.environ.get(
                "SWAN_STATION_ROOT",
                r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET")
            station_data = load_all_station_data(station_root_dir, global_norm_params, time_index)
            ds_sim.close()
        except Exception as e:
            raise RuntimeError("Data preparation failed; refusing to continue with another configuration") from e

        # Record the full-data reference sampler length before releasing split data.
        base_indices = repair.SPLIT_CONTEXT['full_train_indices']
        base_sampler = PeakSamplerRestricted(base_indices,wave_data_for_split,seq_length,pct=95,up_factor=2)
        split_record = json.loads(Path('split_manifest.json').read_text())
        split_record['base_sampler_length'] = len(base_sampler)
        repair.atomic_json('split_manifest.json',split_record)
        del base_sampler, wave_data_for_split, hs_raw
        # Dataset / Loader
        base_ds = WindWaveDataset(input_data, wave_data, seq_length, 0, N)
        train_sampler = PeakSamplerRestricted(allowed_indices=idx_tr, wave_data=wave_data, seq_len=seq_length, pct=95, up_factor=2, seed=int(os.getenv("SWAN_SAMPLER_SEED","42")))
        dl_tr = DataLoader(base_ds, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=safe_collate,
                           num_workers=0, pin_memory=(torch.cuda.is_available()), drop_last=True, generator=torch.Generator().manual_seed(71))
        val_ds = SubsetIndicesDataset(base_ds, idx_va); test_ds = SubsetIndicesDataset(base_ds, idx_te)
        bs_val = max(1, min(BATCH_SIZE, len(val_ds))); bs_te = max(1, min(BATCH_SIZE, len(test_ds)))
        dl_va = DataLoader(val_ds, batch_size=bs_val, shuffle=False, collate_fn=safe_collate, num_workers=0,
                           pin_memory=(torch.cuda.is_available()), drop_last=False, generator=torch.Generator().manual_seed(72))
        dl_te = DataLoader(test_ds, batch_size=bs_te, shuffle=False, collate_fn=safe_collate, num_workers=0,
                           pin_memory=(torch.cuda.is_available()), drop_last=False, generator=torch.Generator().manual_seed(73))

        # Smoke evaluation is diagnostic only and uses eight test samples.
        repair_job = json.loads(os.getenv('SWAN_REPAIR_JOB','{}'))
        if repair_job.get('stage') == 'smoke':
            idx_te = idx_te[:8]
            dl_te = DataLoader(SubsetIndicesDataset(base_ds,idx_te),batch_size=1,
                shuffle=False,collate_fn=safe_collate,num_workers=0,
                generator=torch.Generator().manual_seed(73))
        # Retained legacy processing.
        in_ch = int(input_data.shape[1])
        model = UNetConvLSTM(input_channels=in_ch, output_channels=4, hidden_dim=hidden_dim, feat=unet_feat).to(device)

        # Spatial weights (cos(lat)) + coastline ring boost
        if lat.ndim == 3: lat2d = lat[0]
        else:             lat2d = lat
        if kcs.ndim == 3: kcs2d_for_w = kcs[0]
        else:             kcs2d_for_w = kcs
        cos_lat = np.cos(np.deg2rad(lat2d))
        ocean_mask = (kcs2d_for_w > 0) & np.isfinite(cos_lat) & (cos_lat > 0)
        base_w = np.where(ocean_mask, cos_lat, 0.0).astype(np.float32)
        land = (kcs2d_for_w <= 0); ring = binary_dilation(land, iterations=1) & (kcs2d_for_w > 0)
        base_w[ring] *= 1.2
        s = base_w.sum()
        spatial_w_np = (base_w / (s + 1e-12)) if s > 0 else (ocean_mask.astype(np.float32) / (ocean_mask.sum() + 1e-12))
        spatial_w = torch.from_numpy(spatial_w_np).float().to(device)

        # Retained legacy processing.
        ckpt_prefix      = f"ckpt_ts{time_steps}_seq{seq_length}_hid{hidden_dim}_{split_tag}_{bnd_tag}"
        training_results = train(model, dl_tr, dl_va, spatial_w, epochs=epochs, acc_steps=ACC_STEPS,
                                 norm_params=global_norm_params, ckpt_prefix=ckpt_prefix, lambda_tv=2e-3)

        results_filename = f"training_results_ts{time_steps}_seq{seq_length}_ep{epochs}_hid{hidden_dim}_{bnd_tag}.txt"
        save_training_results(training_results, results_filename)
        plot_filename    = f"training_plot_ts{time_steps}_seq{seq_length}_ep{epochs}_hid{hidden_dim}_{bnd_tag}.png"
        plot_training_results(training_results, plot_filename)

        final_val_loss = training_results['val_losses'][-1] if training_results['val_losses'] else np.nan
        final_train_loss = training_results['train_losses'][-1] if training_results['train_losses'] else np.nan

        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = f"{ts_now}_model_weights_{time_steps}_seq{seq_length}_epochs{epochs}_hid{hidden_dim}_UNET{unet_feat[0]}_{bnd_tag}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"모델이 '{model_file}' 파일로 저장되었습니다.")

        eval_metrics = evaluate_and_visualize(
            model, dl_te, lon, lat, kcs, STATIONS, station_data,
            test_start_idx=None, seq_length=seq_length, global_norm_params=global_norm_params,
            out_prefix=f"eval_v57_ts{time_steps}_seq{seq_length}_{bnd_tag}",
            time_steps_info=str(time_steps), seq_length_info=str(seq_length),
            epochs_info=str(epochs), alpha="NA", base_indices=idx_te
        )

        import csv
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                time_steps, seq_length, epochs, hidden_dim,
                eval_metrics.get("rmse_m", np.nan), eval_metrics.get("mae_m", np.nan),
                eval_metrics.get("r2_m", np.nan), eval_metrics.get("smape_m", np.nan),
                eval_metrics.get("pred_wmean", np.nan), eval_metrics.get("true_wmean", np.nan),
                final_train_loss, final_val_loss
            ])
        print(f"→ Comb {ci} done.\n")

    print(f"All combinations complete. Results in '{csv_filename}'.")


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet-ConvLSTM with extended features & losses")
    parser.add_argument('--data_path', type=str,
                        default=r'C:\DELFT3DFM\South_Korea_emulator_2020_ST6_bnd_test\wave\wavm-Waves_2019_2020_final.nc',
                        help='Path to NetCDF data file')
    parser.add_argument('--use_bnd', type=str, default=os.getenv("USE_BND_FEATURES", "on"),
                        choices=["on","off","auto"],
                        help="Boundary features switch: on/off/auto (default=auto)")
    args = parser.parse_args()
    wrapper(args.data_path, use_bnd=args.use_bnd)
