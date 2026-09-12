# -*- coding: utf-8 -*-
"""
UNet-ConvLSTM wave emulator (v57, 'all-patches')
- Includes: circular direction loss, deep supervision, TV regularizer,
  coastline-ring weighting, extra static & seasonal features, peak curriculum.

Tested with PyTorch 2.x / CUDA environment.
"""

import os, math, argparse, traceback, warnings, itertools
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
    """Train-target 시점만 사용한 정규화 파라미터 계산(기존 로직 유지)."""
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
    """|∇depth| → 0~1 정규화 (robust)"""
    if depth2d.ndim == 3: depth2d = depth2d[0]
    gy, gx = np.gradient(np.nan_to_num(depth2d.astype(np.float32)))
    g = np.hypot(gx, gy)
    lo, hi = np.nanpercentile(g, 1), np.nanpercentile(g, 99)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return g

# (REMOVED: _distance_to_coast, _seasonal_sin_cos)  # ← 요청대로 제거

# =========================
# Data IO / preprocessing
# =========================
def load_and_preprocess_data(ds, global_norm_params, time_steps=100):
    """
    NetCDF → (input_data, wave_data, lon, lat, kcs)
    input_data: (T, 6, Y, X)  = wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad   # (변경됨)
    wave_data : (T, 4, Y, X)  = Hs, Tm, sinθ, cosθ
    """
    required_vars = ['windu','windv','depth','veloc-x','veloc-y','hsign','period','dir','x','y','kcs']
    for v in required_vars:
        if v not in ds: raise ValueError(f"Variable '{v}' not found in dataset")

    _DTYPE = np.float32
    T = time_steps

    # 기본 필드 정규화
    wind_u = normalize_with_external_params(ds['windu'].values[:T], global_norm_params['wind_u']).astype(_DTYPE)
    wind_v = normalize_with_external_params(ds['windv'].values[:T], global_norm_params['wind_v']).astype(_DTYPE)
    depth   = normalize_with_external_params(ds['depth'].values[:T],  global_norm_params['depth']).astype(_DTYPE)
    veloc_x = normalize_with_external_params(ds['veloc-x'].values[:T], global_norm_params['veloc_x']).astype(_DTYPE)
    veloc_y = normalize_with_external_params(ds['veloc-y'].values[:T], global_norm_params['veloc_y']).astype(_DTYPE)

    # 목표 필드 정규화
    hs  = normalize_with_external_params(ds['hsign'].values[:T], global_norm_params['hs']).astype(_DTYPE)
    tm  = normalize_with_external_params(ds['period'].values[:T], global_norm_params['tm']).astype(_DTYPE)
    rad = np.deg2rad(ds['dir'].values[:T])
    dsin, dcos = np.sin(rad).astype(_DTYPE), np.cos(rad).astype(_DTYPE)

    lon = ds['x'].values; lat = ds['y'].values; kcs = ds['kcs'].values
    if lat.ndim == 3: lat = lat[0]
    if kcs.ndim == 3: kcs = kcs[0]

    # ---------- 추가 특성 (정적 1: depth_grad) ----------
    depth2d = ds['depth'].values if ds['depth'].values.ndim == 2 else ds['depth'].values[0]
    depth_grad = _depth_grad_mag(depth2d)                    # (Y,X)

    # (T,H,W)로 브로드캐스트
    H, W = hs.shape[-2], hs.shape[-1]
    depth_grad_3d = np.broadcast_to(depth_grad[None, ...], (T, H, W)).astype(_DTYPE)

    # ---------- 스택 ----------
    input_data = np.stack(
        [wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad_3d], axis=1
    )
    wave_data = np.stack([hs, tm, dsin, dcos], axis=1)
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

class PeakSamplerRestricted(torch.utils.data.Sampler):
    def __init__(self, allowed_indices, wave_data, seq_len, pct=95, up_factor=2, seed=42):
        self.allowed = np.asarray(allowed_indices, dtype=np.int64)
        self.seq_len = seq_len; self.up_factor = int(up_factor)
        self.rng = np.random.default_rng(seed)
        hs = wave_data[seq_len:, 0]
        hs_max = hs.reshape(hs.shape[0], -1).max(axis=1)
        hs_sub = hs_max[self.allowed]
        thresh = np.percentile(hs_sub, pct)
        self.peak_idx = self.allowed[hs_sub >= thresh]
        self.norm_idx = self.allowed[hs_sub <  thresh]
        self.length = len(self.norm_idx) + self.up_factor * max(1, len(self.peak_idx))
    def __iter__(self):
        normal = self.rng.permutation(self.norm_idx) if len(self.norm_idx) else np.array([], dtype=int)
        if len(self.peak_idx):
            peaks = self.rng.choice(self.peak_idx, size=self.up_factor * len(self.peak_idx), replace=True)
        else:
            peaks = np.array([], dtype=int)
        idx_all = np.concatenate([normal, peaks]); self.rng.shuffle(idx_all)
        return iter(idx_all.tolist())
    def __len__(self): return self.length

def safe_collate(batch):
    clean_x, clean_y = [], []
    for item in batch:
        try:
            if item is None: continue
            x, y = item
            x = torch.as_tensor(x).contiguous().float()
            y = torch.as_tensor(y).contiguous().float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all(): continue
            clean_x.append(x); clean_y.append(y)
        except Exception:
            continue
    if len(clean_x) == 0:
        try:
            x0, y0 = batch[0]; x0 = torch.as_tensor(x0); y0 = torch.as_tensor(y0)
            T, Cx, H, W = x0.shape; Cy, Hy, Wy = y0.shape
        except Exception:
            T, Cx, H, W = 6, 6, 64, 64; Cy, Hy, Wy = 4, 64, 64
        return torch.zeros(1, T, Cx, H, W), torch.zeros(1, Cy, Hy, Wy)
    try:
        return torch.stack(clean_x, 0), torch.stack(clean_y, 0)
    except Exception as e:
        shapes_x = [tuple(t.shape) for t in clean_x]
        shapes_y = [tuple(t.shape) for t in clean_y]
        raise RuntimeError(f"[collate] shape mismatch: X={shapes_x}, Y={shapes_y}") from e

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
    """
    Dir 손실: 1 - cos(Δθ) (원형 오차)
    나머지는 L1/Huber 가중 평균
    """
    def _to_last(x):
        if x.shape[-1] in (3,4,5): return x
        elif x.dim() >= 4 and x.shape[-3] in (3,4,5): return x.movedim(-3, -1)
        elif x.dim() >= 4 and x.shape[1] in (3,4,5):  return x.movedim(1, -1)
        elif x.dim() >= 3 and x.shape[1] in (3,4,5):  return x.movedim(1, -1)
        return x
    pred  = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    target= torch.nan_to_num(target,nan=0.0, posinf=1e6, neginf=-1e6)
    pred  = _to_last(pred); target = _to_last(target)

    phs, ptm, psin, pcos = pred[...,0], pred[...,1], pred[...,2].clamp(-1,1), pred[...,3].clamp(-1,1)
    ths, ttm, tsin, tcos = target[...,0], target[...,1], target[...,2].clamp(-1,1), target[...,3].clamp(-1,1)

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
    cos_delta = (pcos * tcos + psin * tsin).clamp(-1.0, 1.0)   # cos(Δθ)
    loss_dir  = wmean(1.0 - cos_delta)

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

def train(
    model, dl_tr, dl_va, w, * ,
    epochs, acc_steps, norm_params, ckpt_prefix="ckpt",
    freeze_logvars_epochs=1, early_stop_patience=3, lambda_tv=2e-3
):
    best_val = float('inf')
    opt = optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if n != "log_vars"], "lr": max_lr, "weight_decay": weight_decay},
        {"params": [model.log_vars], "lr": max_lr*0.1, "weight_decay": 0.0}
    ])
    steps_per_epoch = max(1, math.ceil(len(dl_tr)/acc_steps))
    total_steps = epochs * steps_per_epoch
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start,
        div_factor=div_factor, final_div_factor=final_div_factor
    )
    ema = EMA(model, decay=0.999)

    # log_vars warmup
    had_logvars = hasattr(model,"log_vars") and isinstance(model.log_vars, torch.nn.Parameter)
    if had_logvars: model.log_vars.requires_grad_(False)

    train_losses, val_losses = [], []
    train_mae_hs, train_mae_tm, train_mae_dir = [], [], []
    val_mae_hs, val_mae_tm, val_mae_dir       = [], [], []

    bad_epochs = 0; step_cnt = 0
    start_epoch = 1

    # ---- Resume from the last full-state checkpoint, if one exists. ----
    # A killed job (node reboot, preemption, OOM kill) restarts from here instead
    # of epoch 1. The checkpoint carries model, optimizer, scheduler, AMP scaler,
    # EMA shadow, epoch counter, and loss history, so the run continues exactly
    # where it stopped. Anything wrong with the file falls back to a clean start.
    resume_path = f"{ckpt_prefix}_resume.pt"
    if os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if ck.get("scaler") is not None:
                try: SCALER.load_state_dict(ck["scaler"])
                except Exception as _se: print("Warn: scaler resume skipped:", _se)
            if ck.get("ema_shadow") is not None:
                ema.shadow = {k: v.to(device) for k, v in ck["ema_shadow"].items()}
            best_val   = ck.get("best_val", best_val)
            bad_epochs = ck.get("bad_epochs", 0)
            step_cnt   = ck.get("step_cnt", 0)
            hist = ck.get("history", {})
            train_losses  = list(hist.get("train_losses", []))
            val_losses    = list(hist.get("val_losses", []))
            train_mae_hs  = list(hist.get("train_mae_hs", []))
            train_mae_tm  = list(hist.get("train_mae_tm", []))
            train_mae_dir = list(hist.get("train_mae_dir", []))
            val_mae_hs    = list(hist.get("val_mae_hs", []))
            val_mae_tm    = list(hist.get("val_mae_tm", []))
            val_mae_dir   = list(hist.get("val_mae_dir", []))
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[RESUME] {resume_path}: continuing from epoch {start_epoch} "
                  f"(best_val={best_val:.6f}, bad_epochs={bad_epochs})")
        except Exception as e:
            print(f"[RESUME] failed to load {resume_path} ({type(e).__name__}: {e}); starting fresh.")
            start_epoch = 1

    # If we resumed past the log_vars warmup window, unfreeze them now.
    if had_logvars and start_epoch > freeze_logvars_epochs:
        model.log_vars.requires_grad_(True)
    if start_epoch > epochs:
        print(f"[RESUME] checkpoint already at epoch {start_epoch-1} >= {epochs}; nothing to train.")

    for ep in range(start_epoch, epochs+1):
        model.train(); running = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)

        if had_logvars and ep == freeze_logvars_epochs + 1:
            model.log_vars.requires_grad_(True)

        # Peak curriculum
        if ep < 5:
            use_peak = False; q_val = 0.95
        elif ep < 8:
            use_peak = True; q_val = 0.90
        else:
            use_peak = True; q_val = 0.95
        boost = 1.2

        for i, (xb, yb) in enumerate(tqdm(dl_tr, desc=f"Epoch {ep}")):
            xb = xb.to(device); yb = yb.to(device)
            # w_eff 준비(+피크 가중)
            w_batch = w
            if w_batch.dim() == 2: w_batch = w_batch.unsqueeze(0).unsqueeze(0)
            elif w_batch.dim() == 3: w_batch = w_batch.unsqueeze(0)
            if use_peak:
                with torch.no_grad():
                    hs_target = yb[:,0]
                    hs_flat = hs_target.reshape(hs_target.size(0), -1).float()
                    q = torch.tensor([q_val], device=hs_flat.device, dtype=torch.float32)
                    qk = torch.quantile(hs_flat, q, dim=1, keepdim=True).squeeze(-1).view(-1,1,1)
                    peak_mask = (hs_target >= qk).float()
                    w_eff = (w_batch.expand(hs_target.size(0), -1, -1, -1) * (1.0 + (boost-1.0)*peak_mask.unsqueeze(1))).squeeze(1)
            else:
                w_eff = w_batch.expand(xb.size(0), -1, -1, -1).squeeze(1)

            with autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                outs = model(xb)
                main_pred = outs[0]; aux_preds = outs[1:]
                loss_total, parts = deep_supervised_loss(
                    main_pred, aux_preds, yb, w_eff,
                    (model.log_vars if (had_logvars and model.log_vars.requires_grad) else None),
                    lambda_tv=lambda_tv
                )
                loss = loss_total / acc_steps

            SCALER.scale(loss).backward()

            if (i+1) % acc_steps == 0:
                SCALER.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                SCALER.step(opt); SCALER.update()
                opt.zero_grad(set_to_none=True); sched.step(); step_cnt += 1
                with torch.no_grad():
                    if hasattr(model,"log_vars"):
                        model.log_vars.data = torch.nan_to_num(model.log_vars.data, nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0,5.0)
                ema.update(model)

            running += loss.item() * acc_steps; nb += 1

        # Apply the final partial accumulation window. Without this, the last
        # mini-batches are dropped from the optimizer step whenever the number of
        # batches is not divisible by acc_steps. Harmless when acc_steps == 1.
        if nb > 0 and (nb % acc_steps) != 0:
            SCALER.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            SCALER.step(opt); SCALER.update()
            opt.zero_grad(set_to_none=True); sched.step(); step_cnt += 1
            with torch.no_grad():
                if hasattr(model,"log_vars"):
                    model.log_vars.data = torch.nan_to_num(model.log_vars.data, nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0,5.0)
            ema.update(model)

        train_loss = running / max(nb,1)

        # Validation with EMA
        ema.apply_to(model)
        val_loss = compute_loss(model, dl_va, w, lambda_tv=lambda_tv)
        va_mae  = compute_mae_phys(model, dl_va, w, norm_params)
        ema.restore(model)

        # Train MAE (non-EMA)
        tr_mae  = compute_mae_phys(model, dl_tr, w, norm_params)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_mae_hs.append(tr_mae["hs"]); train_mae_tm.append(tr_mae["tm"]); train_mae_dir.append(tr_mae["dir"])
        val_mae_hs.append(va_mae["hs"]);   val_mae_tm.append(va_mae["tm"]);   val_mae_dir.append(va_mae["dir"])

        last_lr = sched.get_last_lr()[0] if hasattr(sched, 'get_last_lr') else max_lr
        print(f"Ep{ep} Train {train_loss:.6f} | Val {val_loss:.6f} | MAE(Hs/Tm/Dir tr→va) "
              f"{tr_mae['hs']:.3f}/{va_mae['hs']:.3f} m, {tr_mae['tm']:.3f}/{va_mae['tm']:.3f} s, "
              f"{tr_mae['dir']:.1f}/{va_mae['dir']:.1f} deg | LR {last_lr:.2e}")

        # save last/best
        try: torch.save(model.state_dict(), f"{ckpt_prefix}_last_raw.pth")
        except Exception as e: print("Warn: save last failed:", e)

        if val_loss < best_val - 1e-6:
            best_val = val_loss; bad_epochs = 0
            try:
                torch.save(model.state_dict(), f"{ckpt_prefix}_best_raw.pth")
                ema.apply_to(model)
                torch.save(model.state_dict(), f"{ckpt_prefix}_best_ema.pth")
                ema.restore(model)
                print(f"[CHECKPOINT] NEW BEST ep {ep}: {val_loss:.6f}")
            except Exception as e:
                print("Warn: save best failed:", e)
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"[EARLY STOP] {bad_epochs} epochs without improvement. Stop at ep {ep}.")
                break

        # Full-state resume checkpoint, written atomically every epoch so a kill
        # mid-write cannot corrupt it (write tmp, then os.replace).
        try:
            _resume_obj = {
                "epoch": ep,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "step_cnt": step_cnt,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": (SCALER.state_dict() if SCALER.is_enabled() else None),
                "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                "history": {
                    "train_losses": train_losses, "val_losses": val_losses,
                    "train_mae_hs": train_mae_hs, "train_mae_tm": train_mae_tm, "train_mae_dir": train_mae_dir,
                    "val_mae_hs": val_mae_hs, "val_mae_tm": val_mae_tm, "val_mae_dir": val_mae_dir,
                },
            }
            _tmp = f"{ckpt_prefix}_resume.pt.tmp"
            torch.save(_resume_obj, _tmp)
            os.replace(_tmp, f"{ckpt_prefix}_resume.pt")
        except Exception as e:
            print("Warn: resume checkpoint save failed:", e)

    # Training finished (completed or early-stopped); drop the resume file so a
    # later re-dispatch of this exact config starts clean rather than no-op'ing.
    try:
        if os.path.exists(f"{ckpt_prefix}_resume.pt"):
            os.remove(f"{ckpt_prefix}_resume.pt")
    except Exception:
        pass

    return {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_mae_hs': train_mae_hs, 'train_mae_tm': train_mae_tm, 'train_mae_dir': train_mae_dir,
        'val_mae_hs': val_mae_hs, 'val_mae_tm': val_mae_tm, 'val_mae_dir': val_mae_dir,
        'epochs': list(range(1, len(train_losses)+1))
    }

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
    tr_blocks, va_blocks, te_blocks = set(tr_blocks), set(va_blocks), set(te_blocks)
    blk2set = {b:(0 if b in tr_blocks else (1 if b in va_blocks else 2)) for b in range(num_blocks)}
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
# Plot / Evaluate (핵심만)
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
    items = [("Pred", pred, "jet"), ("True", true, "jet"), ("|Err|", np.abs(pred - true), "coolwarm")]
    for i, (ttl, dat, cmap) in enumerate(items, 1):
        ax = plt.subplot(1,3,i)
        d = np.ma.masked_where(kcs <= 0, dat)
        err_max = np.nanmax(d)
        lo, hi = (vmin, vmax) if i < 3 else (0, err_max + 1e-6 if err_max is not np.ma.masked else 1)
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
# (B) 연도별 dict를 시간축 기준으로 합치는 유틸 (새로 추가)
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
    import itertools, os, traceback
    import numpy as np
    import pandas as pd
    import xarray as xr
    import torch
    from torch.utils.data import DataLoader
    from scipy.ndimage import binary_dilation

    # ── Lazy imports for BND features / segments
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

    # --- 스위치 결정: CLI 인자 > 환경변수 > auto ---
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

    # ─────────────────────────────────────────────
    # 실험 조합(전역 리스트 사용: time_steps_list 등)
    # ─────────────────────────────────────────────
    combinations = list(itertools.product(time_steps_list, seq_length_list, epochs_list, hidden_dim_list, unet_feat_list))

    # 요약 CSV(모드별 파일명 분리)
    csv_filename = f"performance_summary_global_norm_v61_{bnd_tag}.csv"
    HEADER = ["time_steps", "seq_length", "epochs", "hidden_dim", "rmse_m", "mae_m", "r2_m", "smape_m",
              "pred_weighted_mean", "true_weighted_mean", "train_loss_final", "val_loss_final"]
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        import csv; csv.writer(f).writerow(HEADER)

    # ─────────────────────────────────────────────
    # Split 생성 보조
    # ─────────────────────────────────────────────
    def robust_block_split(wave_data_for_split, seq_length, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        T = wave_data_for_split.shape[0]; N = T - seq_length
        block_hours_list = [168, 96, 72, 48, 336]
        q_list = [5,4,3,2]; embargo_list = [seq_length, max(seq_length//2,1), 0]
        for bh in block_hours_list:
            for q in q_list:
                for emb in embargo_list:
                    try:
                        idx_tr, idx_va, idx_te = make_block_stratified_split(
                            wave_data_for_split, seq_length, train_ratio=train_ratio, val_ratio=val_ratio,
                            test_ratio=test_ratio, block_hours=bh, q=q, seed=42, embargo_hours=emb
                        )
                    except Exception:
                        continue
                    if len(idx_tr)>0 and len(idx_va)>0 and len(idx_te)>0:
                        print(f"[split-ok] bh={bh}, q={q}, emb={emb} -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
                        return idx_tr, idx_va, idx_te, f"block(bh={bh},q={q},emb={emb})"
        # fallback
        rng = np.random.default_rng(42)
        all_idx = np.arange(0, N, dtype=int); rng.shuffle(all_idx)
        n_tr = max(1, int(round(N*0.70))); n_va = max(1, int(round(N*0.15))); n_te = N - n_tr - n_va
        if n_te <= 0: n_te = 1; n_va = max(1, n_va-1)
        idx_tr = np.sort(all_idx[:n_tr]); idx_va = np.sort(all_idx[n_tr:n_tr+n_va]); idx_te = np.sort(all_idx[n_tr+n_va:n_tr+n_va+n_te])
        print(f"[split-fallback] random -> tr/va/te={len(idx_tr)}/{len(idx_va)}/{len(idx_te)} (N={N})")
        return idx_tr, idx_va, idx_te, "random-fallback"

    # ─────────────────────────────────────────────
    # BND 폴더 매핑 / 방향
    # ─────────────────────────────────────────────
    BND_DIRS_BY_YEAR = {
        2019: os.environ.get(
            "SWAN_BND_DIR_2019",
            r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2019"),
        2020: os.environ.get(
            "SWAN_BND_DIR_2020",
            r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2020"),
    }
    bnd_direction = "from"  # 또는 "toward"

    # ─────────────────────────────────────────────
    # (NEW) BND 파향 자동 보정 유틸  [CHANGE-BND-AUTOCORR]
    # ─────────────────────────────────────────────
    def _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index):
        """
        bnd_feat: (T, 4, H, W) with channels [Hs_norm, Tm_norm, sin, cos] 가정
        ds_sim['dir']: degrees (T,H,W)
        kcs2d: (H,W) ocean mask (>0)
        time_index: length=T
        return: rotated bnd_feat (in-place 수정), chosen_deg, scores dict
        """
        T = bnd_feat.shape[0]
        # 타깃 파향(sin, cos)
        if 'dir' not in ds_sim:
            return 0.0, {}
        rad = np.deg2rad(ds_sim['dir'].values[:T])
        tsin = np.sin(rad).astype(np.float32)
        tcos = np.cos(rad).astype(np.float32)

        # BND 파향 채널 추정(기본: 2,3). 값 범위 체크(음수 존재)로 방어.
        sin_idx, cos_idx = 2, 3
        for try_s, try_c in [(2,3), (3,2)]:
            smin = np.nanmin(bnd_feat[:,try_s])
            cmin = np.nanmin(bnd_feat[:,try_c])
            if (smin < -0.1) and (cmin < -0.1):
                sin_idx, cos_idx = try_s, try_c
                break

        sin_b = bnd_feat[:, sin_idx]
        cos_b = bnd_feat[:, cos_idx]

        mask = (kcs2d > 0)
        if mask.ndim != 2:
            mask = mask[0] if mask.ndim == 3 else mask
        mask = np.asarray(mask, bool)

        def _score(deg):
            r = np.deg2rad(deg)
            sin_r = sin_b*np.cos(r) + cos_b*np.sin(r)
            cos_r = cos_b*np.cos(r) - sin_b*np.sin(r)
            v = (sin_r*tsin + cos_r*tcos)  # cos(Δθ)
            vv = v[:, mask]
            return float(np.nanmean(vv))

        candidates = [0.0, 90.0, -90.0, 180.0]
        scores = {deg: _score(deg) for deg in candidates}
        best_deg = max(scores, key=lambda d: scores[d])

        # 실제 적용
        if abs(best_deg) > 1e-6:
            r = np.deg2rad(best_deg)
            sin_r = sin_b*np.cos(r) + cos_b*np.sin(r)
            cos_r = cos_b*np.cos(r) - sin_b*np.sin(r)
            bnd_feat[:, sin_idx] = sin_r
            bnd_feat[:, cos_idx] = cos_r
        return best_deg, scores

    # ─────────────────────────────────────────────
    # 실험 루프
    # ─────────────────────────────────────────────
    for ci, (time_steps, seq_length, epochs, hidden_dim, unet_feat) in enumerate(combinations, 1):
        print(f"\n===== Comb {ci}/{len(combinations)} =====")
        print(f"ts={time_steps}  L={seq_length}  ep={epochs}  hid={hidden_dim}")
        torch.cuda.empty_cache()
        try:
            if not os.path.isfile(data_path): raise FileNotFoundError(f"NetCDF not found: {data_path}")
            ds_sim = xr.open_dataset(data_path)

            # split 준비
            N = time_steps - seq_length
            hs_raw = ds_sim["hsign"].values[:time_steps]
            Y, X = hs_raw.shape[-2], hs_raw.shape[-1]
            wave_data_for_split = np.zeros((time_steps, 4, Y, X), dtype=np.float32); wave_data_for_split[:,0] = hs_raw
            idx_tr, idx_va, idx_te, split_tag = robust_block_split(wave_data_for_split, seq_length)

            # 정규화 파라미터(훈련 타겟 시점 기반)
            global_norm_params = compute_params_with_indices(ds_sim, idx_train=idx_tr, seq_length=seq_length)
            print("Hs range:", global_norm_params['hs'], "  Tm range:", global_norm_params['tm'])

            # 전처리(+정적 경사 특성)
            input_data, wave_data, lon, lat, kcs = load_and_preprocess_data(ds_sim, global_norm_params, time_steps=time_steps)

            # 공통 time_index
            if 'time' in ds_sim:
                tvals = pd.to_datetime(ds_sim['time'].values[:time_steps])
                time_index = pd.DatetimeIndex(tvals).tz_localize(None)
            else:
                time_index = pd.date_range(start="2019-01-01 00:00:00", periods=time_steps, freq="h", tz="UTC").tz_localize(None)

            # ─────────────────────────────────────────────
            # (C) BND 경계 특성 결합 (+ 파향 자동 보정)  [CHANGE-BND-AUTOCORR]
            # ─────────────────────────────────────────────
            if USE_BND_FEATURES:
                try:
                    # kcs 크기와 SWAN(M,N) 비교 → 전치 여부(swap_ij) 자동 판정
                    kcs2d = kcs[0] if kcs.ndim == 3 else kcs
                    H, W = kcs2d.shape
                    if   (H == SWAN_M and W == SWAN_N):   swap_ij = False
                    elif (H == SWAN_N and W == SWAN_M):   swap_ij = True
                    else:
                        raise ValueError(f"Grid mismatch: data(H,W)=({H},{W}) vs SWAN(M,N)=({SWAN_M},{SWAN_N}). 자동 스케일 금지.")

                    # 세그먼트 정의 유효성 검사(끝점이 외곽인지)
                    assert_on_edges(SEGMENTS, M=SWAN_M, N=SWAN_N)

                    # 필요한 연도만 골라서 dict 병합
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

                    # 경계 라벨 생성
                    owner_label, id2name = build_owner_label(
                        H, W, segments=SEGMENTS, exact_M=SWAN_M, exact_N=SWAN_N,
                        kcs=kcs2d, swap_ij=swap_ij
                    )

                    # (T,4,H,W) 경계 특성맵 생성
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

                    # === 파향 자동 보정 적용 ===
                    best_deg, scores = _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index)
                    msg = " ".join([f"{k:+.0f}°:{v:.4f}" for k,v in sorted(scores.items())])
                    print(f"[BND] dir autocorrect → chosen {best_deg:+.0f}° | scores {msg}")

                    # 입력 결합
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

            # 관측소 자료
            station_root_dir = os.environ.get(
                "SWAN_STATION_ROOT",
                r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET")
            station_data = load_all_station_data(station_root_dir, global_norm_params, time_index)
            ds_sim.close()
        except Exception as e:
            print("Data loading error:", e); print(traceback.format_exc()); continue

        # Dataset / Loader
        base_ds = WindWaveDataset(input_data, wave_data, seq_length, 0, N)
        train_sampler = PeakSamplerRestricted(allowed_indices=idx_tr, wave_data=wave_data, seq_len=seq_length, pct=95, up_factor=2)
        dl_tr = DataLoader(base_ds, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=safe_collate,
                           num_workers=0, pin_memory=(torch.cuda.is_available()), drop_last=True)
        val_ds = SubsetIndicesDataset(base_ds, idx_va); test_ds = SubsetIndicesDataset(base_ds, idx_te)
        bs_val = max(1, min(BATCH_SIZE, len(val_ds))); bs_te = max(1, min(BATCH_SIZE, len(test_ds)))
        dl_va = DataLoader(val_ds, batch_size=bs_val, shuffle=False, collate_fn=safe_collate, num_workers=0,
                           pin_memory=(torch.cuda.is_available()), drop_last=True)
        dl_te = DataLoader(test_ds, batch_size=bs_te, shuffle=False, collate_fn=safe_collate, num_workers=0,
                           pin_memory=(torch.cuda.is_available()), drop_last=False)

        # Model (입력 채널 수 자동 반영)
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

        # 파일명에 bnd_tag 부착
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
