# -*- coding: utf-8 -*-
"""
UNet-ConvLSTM wave emulator — INFERENCE ONLY (Windowed stacking ver.)
- Memory-efficient: does not load the full time axis into memory at once
- All preprocessing and stack construction is done per 'window' (chunk)
- BND (boundary features) are also built and merged on a per-window basis
- Typhoon-window evaluation, general evaluation, and speed benchmarking all use the window approach

* This version FIXES the BND direction rotation angle to -90 deg for typhoon inference.
"""

import os, re, argparse, warnings, datetime, time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Tuple
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import matplotlib.dates as mdates

# -- [NEW] Cartopy (optional, for visualization)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAVE_CARTOPY = True
except Exception:
    _HAVE_CARTOPY = False

# ----------------------------
# Fonts (force Times New Roman)
# ----------------------------
def _set_global_font():
    # Try Arial first, with Korean-glyph / symbol fallback
    matplotlib.rcParams.update({
        "font.family": ["Arial", "Arial Unicode MS", "Noto Sans CJK KR",
                        "Malgun Gothic", "AppleGothic", "DejaVu Sans"],
        "axes.unicode_minus": False,  # prevent minus-sign rendering issues
        "pdf.fonttype": 42,           # TrueType embedding -> easier to edit in vector editors
        "ps.fonttype": 42,
        # Slightly increase default font size
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    })
    warnings.filterwarnings("ignore", category=UserWarning, message="Glyph .* missing from font")

_set_global_font()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ----------------------------
# Station metadata (preserve English names)
# ----------------------------
STATIONS = {
    "Korea Strait":{"lat":34.933888,"lon":129.1375,"file":"daehanhaehyup_kg_wave_1h.csv"},
    "Jeju Strait":{"lat":33.901944,"lon":126.490555,"file":"jejuhaehyup_kg_wave_1h.csv"},
    "South Sea East":{"lat":34.223611,"lon":128.420555,"file":"namhaedongbu_kg_wave_1h.csv"},
    "Daecheon Beach":{"lat":36.28438490,"lon":126.46236720,"file":"daechon_sig_wave_1H.csv"},
    "Haeundae Beach":{"lat":35.148888,"lon":129.169722,"file":"haeundae_sig_wave_1H.csv"},
    "Imrang Beach":{"lat":35.3025,"lon":129.2925,"file":"imrang_sig_wave_1H.csv"},
    "Jungmun Beach":{"lat":33.234444,"lon":126.409722,"file":"jungmun_sig_wave_1H.csv"},
    "Saengil Island":{"lat":34.258716,"lon":126.960269,"file":"saengil_sig_wave_1H.csv"},
    "Sangwangdeungdo":{"lat":35.652458,"lon":126.194255,"file":"sangwang_sig_wave_1H.csv"},
}
# Mapping from Korean station names (as stored in KHOA CSV files) to their English equivalents.
# Korean keys are required because the source CSV files use Korean station identifiers.
STATION_EN = {"대한해협":"Korea Strait","제주해협":"Jeju Strait","남해동부":"South Sea East","대천해수욕장":"Daecheon Beach",
              "해운대해수욕장":"Haeundae Beach","임랑해수욕장":"Imrang Beach","중문해수욕장":"Jungmun Beach",
              "생일도":"Saengil Island","상왕등도":"Sangwangdeungdo"}

# ----------------------------
# Typhoon windows
# ----------------------------
TYPHOONS = {
    "lingling": {"start": pd.Timestamp("2019-09-02 00:00:00"), "end": pd.Timestamp("2019-09-08 00:00:00"), "prefix":"lingling"},
    "bavi":     {"start": pd.Timestamp("2020-08-22 00:00:00"), "end": pd.Timestamp("2020-08-27 00:00:00"), "prefix":"bavi"},
    "maysak":   {"start": pd.Timestamp("2020-08-28 00:00:00"), "end": pd.Timestamp("2020-09-04 00:00:00"), "prefix":"maysak"},
    "haishen":  {"start": pd.Timestamp("2020-09-04 00:00:00"), "end": pd.Timestamp("2020-09-09 00:00:00"), "prefix":"haishen"},
}

# =========================================================
# Small utils
# =========================================================
def _safe_fname(path: str) -> str:
    directory, filename = os.path.split(path)
    safe_filename = re.sub(r"[^0-9A-Za-z_\-\.]", "_", filename)
    if directory: os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, safe_filename)

def denorm(x, vmin, vmax): return x * (vmax - vmin) + vmin

def normalize_with_external_params(data, params):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmin, dmax = params
    if dmax == dmin: return np.zeros_like(data, dtype=np.float32)
    return ((data - dmin) / (dmax - dmin)).astype(np.float32)

def circular_diff_deg(pred_deg, true_deg):
    pred_deg = np.asarray(pred_deg); true_deg = np.asarray(true_deg)
    diff = ((pred_deg - true_deg + 180) % 360) - 180
    return diff

def circular_rmse_deg(pred_deg, true_deg):
    diff = circular_diff_deg(pred_deg, true_deg); return np.sqrt(np.mean(diff**2))

def circular_mae_deg(pred_deg, true_deg):
    diff = circular_diff_deg(pred_deg, true_deg); return np.mean(np.abs(diff))

def circular_correlation(pred_deg, true_deg):
    pr = np.deg2rad(np.asarray(pred_deg)); tr = np.deg2rad(np.asarray(true_deg))
    sp, cp = np.sin(pr), np.cos(pr); st, ct = np.sin(tr), np.cos(tr)
    num = np.mean(sp*st) + np.mean(cp*ct)
    den = np.sqrt((np.mean(sp**2)+np.mean(cp**2))*(np.mean(st**2)+np.mean(ct**2)))
    return num/(den+1e-8)

def circular_r2(pred_deg, true_deg):
    diff = circular_diff_deg(pred_deg, true_deg)
    var_true = (np.rad2deg(np.std(np.deg2rad(true_deg)))**2)
    mse = np.mean(diff**2)
    return 1 - (mse/var_true) if var_true > 0 else 0

def _safe_acc(pred, true, thresh=0.25):
    if len(pred) == 0: return np.nan
    return np.mean(np.abs(pred - true) <= thresh)

def _safe_mape(pred, true, eps=1e-6, thresh=0.25):
    mask = (np.abs(true) >= thresh) & np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0: return np.nan
    return (np.abs(pred[mask]-true[mask])/(np.abs(true[mask])+eps)).mean()*100

def _event_peak_metrics(model_phys, obs_phys, exc_thresh=3.0):
    """Event-peak diagnostics for one window (physical units), model vs obs."""
    m = np.asarray(model_phys, dtype=float); o = np.asarray(obs_phys, dtype=float)
    n = min(len(m), len(o)); m = m[:n]; o = o[:n]
    res = {k: np.nan for k in ['obs_peak','model_peak','peak_bias','rel_peak_err_pct',
                               'timing_err_h','p95_err','p99_err','exc_f1']}
    fo = np.isfinite(o); fm = np.isfinite(m)
    if fo.sum() == 0 or fm.sum() == 0: return res
    o_peak = float(np.nanmax(o)); m_peak = float(np.nanmax(m))
    res['obs_peak'] = o_peak; res['model_peak'] = m_peak
    res['peak_bias'] = m_peak - o_peak
    if o_peak != 0: res['rel_peak_err_pct'] = 100.0*(m_peak - o_peak)/o_peak
    try:
        i_o = int(np.argmax(np.where(fo, o, -np.inf)))
        i_m = int(np.argmax(np.where(fm, m, -np.inf)))
        res['timing_err_h'] = float(i_m - i_o)   # hourly sampling; +ve = model peak later
    except ValueError:
        pass
    pair = fo & fm
    if pair.sum() >= 1:
        res['p95_err'] = float(np.nanpercentile(m[pair],95) - np.nanpercentile(o[pair],95))
        res['p99_err'] = float(np.nanpercentile(m[pair],99) - np.nanpercentile(o[pair],99))
        eo = o[pair] >= exc_thresh; em = m[pair] >= exc_thresh
        tp = int(np.sum(eo & em)); fp = int(np.sum(~eo & em)); fn = int(np.sum(eo & ~em))
        if tp > 0 and (tp+fp) > 0 and (tp+fn) > 0:
            prec = tp/(tp+fp); rec = tp/(tp+fn)
            res['exc_f1'] = float(2*prec*rec/(prec+rec))
    return res

def _obs_agreement_metrics(model_phys, obs_phys):
    """RMSE, RMSE/mean(%), Pearson r(+p), Willmott index of agreement, model vs obs."""
    m = np.asarray(model_phys, dtype=float); o = np.asarray(obs_phys, dtype=float)
    n = min(len(m), len(o)); m = m[:n]; o = o[:n]
    res = {k: np.nan for k in ['n','rmse','rmse_over_mean_pct','pearson_r','p_value','willmott_d']}
    pair = np.isfinite(m) & np.isfinite(o); npair = int(pair.sum())
    res['n'] = npair
    if npair < 3: return res
    mp = m[pair]; op = o[pair]
    rmse = float(np.sqrt(np.mean((mp - op)**2))); res['rmse'] = rmse
    omean = float(np.mean(op))
    if omean != 0: res['rmse_over_mean_pct'] = 100.0*rmse/omean
    try:
        from scipy.stats import pearsonr
        r, pv = pearsonr(mp, op)
        res['pearson_r'] = float(r); res['p_value'] = float(pv)
    except Exception:
        res['pearson_r'] = float(np.corrcoef(mp, op)[0,1])
    denom = float(np.sum((np.abs(mp - omean) + np.abs(op - omean))**2))
    if denom > 0: res['willmott_d'] = float(1.0 - np.sum((mp - op)**2)/denom)
    return res


def _depth_grad_mag(depth2d):
    """Normalize the magnitude of the depth gradient to [0, 1]."""
    if depth2d.ndim == 3: depth2d = depth2d[0]
    gy, gx = np.gradient(np.nan_to_num(depth2d.astype(np.float32)))
    g = np.hypot(gx, gy)
    lo, hi = np.nanpercentile(g, 1), np.nanpercentile(g, 99)
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

# =========================================================
# BND features (same as train) -- built per-window
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

# BND file paths (per year)
BND_DIRS_BY_YEAR = {
    2019: r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2019",
    2020: r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2020",
}
BND_DIRECTION = "from"  # or "toward"

def build_bnd_features(ds_sim, kcs, time_index, global_norm_params, *, force_rotation_deg: float | None = None):
    """
    Returns: bnd_feat (T, 4, H, W)  [channels: Hs_norm, Tm_norm, sin, cos]
    Required local modules: bnd_features.py, boundspec_segments.py

    * If force_rotation_deg is given, automatic alignment is BYPASSED and sin/cos are rotated by that angle.
    """
    try:
        from bnd_features import (
            read_all_bnds, build_owner_label, make_boundary_feature_maps, assert_on_edges
        )
        from boundspec_segments import SEGMENTS, M as SWAN_M, N as SWAN_N
    except Exception as e:
        raise RuntimeError(f"[BND] Failed to load module: {type(e).__name__}: {e}")

    kcs2d = kcs[0] if kcs.ndim == 3 else kcs
    H, W = kcs2d.shape

    if   (H == SWAN_M and W == SWAN_N):   swap_ij = False
    elif (H == SWAN_N and W == SWAN_M):   swap_ij = True
    else:
        raise RuntimeError(f"[BND] Grid mismatch: data(H,W)=({H},{W}) vs SWAN(M,N)=({SWAN_M},{SWAN_N})")

    assert_on_edges(SEGMENTS, M=SWAN_M, N=SWAN_N)

    years_needed = sorted(set(pd.DatetimeIndex(time_index).year))
    seg_dicts = []
    for y in years_needed:
        bdir = BND_DIRS_BY_YEAR.get(y, None)
        if bdir and os.path.isdir(bdir):
            seg_y = read_all_bnds(Path(bdir), direction=BND_DIRECTION)
            seg_dicts.append(seg_y)
            print(f"[BND] {y}: loaded from {bdir}")
        else:
            raise RuntimeError(f"[BND] {y}: BND folder is missing or the path is invalid: {bdir}")

    seg_series = merge_seg_series_dicts(*seg_dicts)

    owner_label, id2name = build_owner_label(
        H, W, segments=SEGMENTS, exact_M=SWAN_M, exact_N=SWAN_N,
        kcs=kcs2d, swap_ij=swap_ij
    )

    bnd_feat = make_boundary_feature_maps(
        time_index=time_index,
        owner_label=owner_label,
        seg_series=seg_series,
        id2name=id2name,
        kcs=kcs2d,
        norm_hs=global_norm_params['hs'],
        norm_tm=global_norm_params['tm'],
    )  # (T, 4, H, W)

    # --- sin/cos rotation utility
    def _rotate(sin_arr, cos_arr, deg):
        r = np.deg2rad(deg)
        sin_r = sin_arr*np.cos(r) + cos_arr*np.sin(r)
        cos_r = cos_arr*np.cos(r) - sin_arr*np.sin(r)
        return sin_r.astype(np.float32), cos_r.astype(np.float32)

    # --- channel indices (Hs, Tm, sin, cos)
    sin_idx, cos_idx = 2, 3

    # * 1) If a forced rotation is specified, ALWAYS apply it
    if force_rotation_deg is not None:
        bnd_feat[:, sin_idx], bnd_feat[:, cos_idx] = _rotate(bnd_feat[:, sin_idx], bnd_feat[:, cos_idx], force_rotation_deg)
        print(f"[BND] force rotation applied: {force_rotation_deg:+.0f}° (auto-align skipped)")
        return bnd_feat.astype(np.float32)

    # 2) Otherwise use automatic alignment (previous behaviour)
    try:
        # Automatic alignment: best match against sim 'dir' (0 / +-90 / 180)
        if 'dir' in ds_sim:
            T = bnd_feat.shape[0]
            rad = np.deg2rad(ds_sim['dir'].values[:T])
            tsin = np.sin(rad).astype(np.float32); tcos = np.cos(rad).astype(np.float32)
            mask = (kcs2d > 0)
            def _score(deg):
                sr, cr = _rotate(bnd_feat[:, sin_idx], bnd_feat[:, cos_idx], deg)
                v = (sr*tsin + cr*tcos)  # cos(Δθ)
                return float(np.nanmean(v[:, mask]))
            candidates = [0.0, 90.0, -90.0, 180.0]
            scores = {deg: _score(deg) for deg in candidates}
            best_deg = max(scores, key=lambda d: scores[d])
            if abs(best_deg) > 1e-6:
                bnd_feat[:, sin_idx], bnd_feat[:, cos_idx] = _rotate(bnd_feat[:, sin_idx], bnd_feat[:, cos_idx], best_deg)
            msg = " ".join([f"{k:+.0f}°:{v:.4f}" for k, v in sorted(scores.items())])
            print(f"[BND] dir autocorrect → chosen {best_deg:+.0f}° | scores {msg}")
    except Exception as e:
        print(f"[BND] dir autocorrect skipped: {type(e).__name__}: {e}")

    return bnd_feat.astype(np.float32)

# =========================================================
# Norm params (train-consistent)
# =========================================================
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
    if idx_train is None or len(idx_train) == 0:
        raise ValueError("compute_params_with_indices: idx_train is empty.")
    t_idx = np.asarray(idx_train, dtype=int) + int(seq_length)

    def _valid_t_idx(da: xr.DataArray, t_idx_arr):
        if "time" in da.dims:
            T = da.sizes["time"]
            return t_idx_arr[(t_idx_arr >= 0) & (t_idx_arr < T)]
        return None

    def _sel_values(varname: str):
        if varname not in ds:
            raise KeyError(f"Variable '{varname}' not found in NetCDF.")
        da = ds[varname]
        t_valid = _valid_t_idx(da, t_idx)
        if t_valid is None: arr = da.values
        else:               arr = da.isel(time=xr.DataArray(t_valid, dims="time_idx")).values
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

# === Load & preprocess (WINDOW VERSION) ===
def load_and_preprocess_window(ds_window: xr.Dataset, global_norm_params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_vars = ['windu','windv','depth','veloc-x','veloc-y','hsign','period','dir','x','y','kcs']
    for v in required_vars:
        if v not in ds_window:
            raise ValueError(f"[window] Variable '{v}' not found in dataset")

    _DTYPE = np.float32
    T = ds_window.sizes.get('time', None)
    if T is None or T <= 0:
        raise ValueError("[window] empty time window")

    # Input 6 channels (normalized)
    wind_u = normalize_with_external_params(ds_window['windu'].values, global_norm_params['wind_u']).astype(_DTYPE)
    wind_v = normalize_with_external_params(ds_window['windv'].values, global_norm_params['wind_v']).astype(_DTYPE)
    depth   = normalize_with_external_params(ds_window['depth'].values,  global_norm_params['depth']).astype(_DTYPE)
    veloc_x = normalize_with_external_params(ds_window['veloc-x'].values, global_norm_params['veloc_x']).astype(_DTYPE)
    veloc_y = normalize_with_external_params(ds_window['veloc-y'].values, global_norm_params['veloc_y']).astype(_DTYPE)

    # Target 4 channels (normalized/transformed)
    hs  = normalize_with_external_params(ds_window['hsign'].values, global_norm_params['hs']).astype(_DTYPE)
    tm  = normalize_with_external_params(ds_window['period'].values, global_norm_params['tm']).astype(_DTYPE)
    rad = np.deg2rad(ds_window['dir'].values)
    dsin, dcos = np.sin(rad).astype(_DTYPE), np.cos(rad).astype(_DTYPE)

    lon = ds_window['x'].values
    lat = ds_window['y'].values
    kcs = ds_window['kcs'].values
    if lat.ndim == 3: lat = lat[0]
    if kcs.ndim == 3: kcs = kcs[0]

    depth2d = ds_window['depth'].values if ds_window['depth'].values.ndim == 2 else ds_window['depth'].values[0]
    depth_grad = _depth_grad_mag(depth2d)
    H, W = hs.shape[-2], hs.shape[-1]
    depth_grad_3d = np.broadcast_to(depth_grad[None, ...], (T, H, W)).astype(_DTYPE)

    input_data = np.stack([wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad_3d], axis=1)  # (T, 6, H, W)
    wave_data  = np.stack([hs, tm, dsin, dcos], axis=1)                                      # (T, 4, H, W)
    return input_data, wave_data, lon, lat, kcs

# =========================================================
# Models
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, c, red=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, max(c//red,1)), nn.ReLU(True),
            nn.Linear(max(c//red,1), c), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w.view(x.size(0), x.size(1), 1, 1)

class ImprovedConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, pad=1):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_c, in_c, k, padding=pad, groups=in_c, bias=False)
        self.conv_pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.norm = nn.GroupNorm(32 if out_c % 32 == 0 else 16, out_c)
        self.se = SEBlock(out_c)
        self.act = nn.ReLU(True)
    def forward(self, x):
        x = self.conv_pw(self.conv_dw(x))
        return self.act(self.se(self.norm(x)))

class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__(); pad = k // 2; self.h = hid_c
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=pad)
    def forward(self, x, s):
        h, c = s
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], 1)), self.h, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o); g = torch.tanh(g)
        c = f*c + i*g; h = o*torch.tanh(c); return h, c
    def init_state(self, B, H, W, dev):
        z = torch.zeros(B, self.h, H, W, device=dev); return z.clone(), z.clone()

class UNetPlusPlus(nn.Module):
    def __init__(self, in_c, out_c, feat=[32,64,128,256,512]):
        super().__init__(); f=feat
        self.enc00 = ImprovedConvBlock(in_c, f[0]); self.pool = nn.MaxPool2d(2,2)
        self.enc10 = ImprovedConvBlock(f[0], f[1]); self.enc20 = ImprovedConvBlock(f[1], f[2])
        self.enc30 = ImprovedConvBlock(f[2], f[3]); self.enc40 = ImprovedConvBlock(f[3], f[4])
        self.dec01 = ImprovedConvBlock(f[0]+f[1], f[0]); self.dec11 = ImprovedConvBlock(f[1]+f[2], f[1])
        self.dec21 = ImprovedConvBlock(f[2]+f[3], f[2]); self.dec31 = ImprovedConvBlock(f[3]+f[4], f[3])
        self.dec02 = ImprovedConvBlock(f[0]*2+f[1], f[0]); self.dec12 = ImprovedConvBlock(f[1]*2+f[2], f[1])
        self.dec22 = ImprovedConvBlock(f[2]*2+f[3], f[2]); self.dec03 = ImprovedConvBlock(f[0]*3+f[1], f[0])
        self.dec13 = ImprovedConvBlock(f[1]*3+f[2], f[1]); self.dec04 = ImprovedConvBlock(f[0]*4+f[1], f[0])
        self.outs = nn.ModuleList([nn.Conv2d(f[0], out_c, 1) for _ in range(4)])
    def _u(self, x, y):
        return torch.cat([F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False), y], 1)
    def forward(self, x):
        x00 = self.enc00(x); x10 = self.enc10(self.pool(x00)); x20 = self.enc20(self.pool(x10))
        x30 = self.enc30(self.pool(x20)); x40 = self.enc40(self.pool(x30))
        x01 = self.dec01(self._u(x10, x00)); x11 = self.dec11(self._u(x20, x10))
        x21 = self.dec21(self._u(x30, x20)); x31 = self.dec31(self._u(x40, x30))
        x02 = self.dec02(self._u(x11, torch.cat([x00, x01], 1)))
        x12 = self.dec12(self._u(x21, torch.cat([x10, x11], 1)))
        x22 = self.dec22(self._u(x31, torch.cat([x20, x21], 1)))
        x03 = self.dec03(self._u(x12, torch.cat([x00, x01, x02], 1)))
        x13 = self.dec13(self._u(x22, torch.cat([x10, x11, x12], 1)))
        x04 = self.dec04(self._u(x13, torch.cat([x00, x01, x02, x03], 1)))
        return [self.outs[0](x04), self.outs[1](x03), self.outs[2](x02), self.outs[3](x01)]

class UNetConvLSTM(nn.Module):
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=64, feat=[32,64,128,256,512]):
        super().__init__()
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.lstm = ConvLSTMCell(output_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, output_channels, 1)
    def forward(self, x):
        B, T, _, H, W = x.shape
        h, c = self.lstm.init_state(B, H, W, dev=x.device)
        last_u = None
        for t in range(T):
            outs = self.unet(x[:, t]); last_u = outs
            h, c = self.lstm(outs[0], (h, c))
        return [self.head(h)] + last_u

# =========================================================
# Data classes & loaders
# =========================================================
class WindWaveDataset(Dataset):
    def __init__(self, input_stack: np.ndarray, wave_stack: np.ndarray, seq_length: int, start_idx: int, end_idx: int):
        self.X = input_stack; self.Y = wave_stack
        self.seq_length = int(seq_length); self.start_idx = int(start_idx); self.end_idx = int(end_idx)
    def __len__(self): return max(0, self.end_idx - self.start_idx)
    def __getitem__(self, idx):
        i = self.start_idx + idx
        seq_X = self.X[i:i + self.seq_length]           # (S, C, H, W)
        seq_y = self.Y[i + self.seq_length]             # (4, H, W)
        return torch.from_numpy(seq_X).float(), torch.from_numpy(seq_y).float()

def collate_fn(batch):
    X = torch.stack([b[0] for b in batch])   # (B, S, C, H, W)
    Y = torch.stack([b[1] for b in batch])   # (B, 4, H, W)
    return X, Y

def find_nearest_index(lon_map, lat_map, kcs_map, target_lon, target_lat):
    kcs2d = kcs_map if kcs_map.ndim == 2 else kcs_map[0]
    valid = np.where((kcs2d == 1) | (kcs2d > 0))
    if valid[0].size == 0: return 0, 0
    lons = lon_map[valid]; lats = lat_map[valid]
    d = np.sqrt((lons - target_lon)**2 + (lats - target_lat)**2); j = np.argmin(d)
    return valid[0][j], valid[1][j]

def load_all_station_data(root_dir, norm_params, time_index):
    # Candidate column-name lists used to locate fields in KHOA buoy CSV files.
    # Korean keywords: '유의파고' = significant wave height (Hs), '유의파주기' / '파주기' = wave period (Tm),
    # '파향' = wave direction, '관측시간' = observation time.
    col_hs = ['유의파고(MOSE.HF)(m)','유의파고(m)','Hs(m)','HS','hs']
    col_tm = ['유의파주기(MOSE.HF)(sec)','유의파주기(sec)','Tm(sec)','TP','tm']
    col_dir = ['파향(deg)','Dir(deg)','파향','dir']; col_time = ['관측시간','datetime','time','date','DateTime','DATE']
    station_data = {}
    for name, meta in STATIONS.items():
        fp = os.path.join(root_dir, meta["file"])
        if not os.path.isfile(fp):
            print(f"[warn] CSV not found → {fp}")
            station_data[name] = np.full((len(time_index), 3), np.nan, dtype=np.float32); continue
        df = pd.read_csv(fp, na_values=['','',' ','NaN'])
        tcol = next((c for c in col_time if c in df.columns), None)
        if tcol is None: raise ValueError(f"{fp}: time column missing.")
        df[tcol] = pd.to_datetime(df[tcol], errors='coerce').dt.tz_localize('Asia/Seoul',nonexistent='shift_forward').dt.tz_convert('UTC').dt.tz_localize(None)
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
        df_std['dir'] = df_std['dir'] % 360
        df_std = df_std.reindex(time_index, method=None)
        station_data[name] = df_std[['hs','tm','dir']].values.astype(np.float32)
    return station_data

# =========================================================
# NaN-safe helpers for map drawing
# =========================================================
def _as_plain_array(a):
    a = np.asanyarray(a)
    if np.ma.isMaskedArray(a): a = a.filled(np.nan)
    return a

def _fill_invalid_with_nearest(a):
    a = _as_plain_array(a)
    if a.ndim != 2: return a
    mask = ~np.isfinite(a)
    if not np.any(mask) or np.all(mask): return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    out = a.copy(); out[mask] = a[tuple(ind[mask] for ind in idx)]
    return out

def _extent_from_lonlat(lon_map, lat_map):
    lon_f = _as_plain_array(lon_map); lat_f = _as_plain_array(lat_map)
    lon_f = lon_f[np.isfinite(lon_f)]; lat_f = lat_f[np.isfinite(lat_f)]
    return [float(lon_f.min()), float(lon_f.max()), float(lat_f.min()), float(lat_f.max())]

def _new_map_axes(ncols=1, nrows=1, figsize=(8,6), lon_map=None, lat_map=None):
    axes = []
    if _HAVE_CARTOPY:
        fig = plt.figure(figsize=figsize)
        for i in range(1, nrows*ncols + 1):
            ax = plt.subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
            if lon_map is not None and lat_map is not None:
                ax.set_extent(_extent_from_lonlat(lon_map, lat_map), crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m'),
                           facecolor='0.92', edgecolor='none', zorder=3)
            ax.coastlines(resolution='10m', linewidth=0.4, zorder=4)
            axes.append(ax)
        return fig, axes
    else:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if isinstance(ax, np.ndarray): axes = list(ax.flat)
        else: axes = [ax]
        return fig, axes

def _draw_field_on_ax(ax, data, lon_map, lat_map, kcs_map, cmap='viridis', vmin=None, vmax=None, title=None):
    kcs2d = kcs_map if kcs_map.ndim == 2 else kcs_map[0]
    d = np.ma.masked_where(((kcs2d!=1)&(kcs2d<=0)) | ~np.isfinite(data), data)
    lon2 = _as_plain_array(lon_map); lat2 = _as_plain_array(lat_map)
    is_curvi = (lon2.ndim == 2) or (lat2.ndim == 2)

    if is_curvi:
        use_pcolor = False
        if np.isfinite(lon2).all() and np.isfinite(lat2).all(): use_pcolor = True
        else:
            if np.isfinite(lon2).any() and np.isfinite(lat2).any():
                lon2 = _fill_invalid_with_nearest(lon2); lat2 = _fill_invalid_with_nearest(lat2)
                use_pcolor = np.isfinite(lon2).all() and np.isfinite(lat2).all()
        if use_pcolor:
            kw = dict(shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
            if _HAVE_CARTOPY: im = ax.pcolormesh(lon2, lat2, d, transform=ccrs.PlateCarree(), **kw)
            else:             im = ax.pcolormesh(lon2, lat2, d, **kw)
            if title: ax.set_title(title)
            return im

    extent = _extent_from_lonlat(lon2, lat2)
    kw = dict(extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear', zorder=2)
    if _HAVE_CARTOPY: im = ax.imshow(d, transform=ccrs.PlateCarree(), **kw)
    else:             im = ax.imshow(d, **kw)
    if title: ax.set_title(title)
    return im

# --- Dir sample plots (fixed 0-360 deg + HSV)
def _plot_dir_sample(pdir_deg, tdir_deg, lon, lat, kcs, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    pdir = np.mod(pdir_deg, 360.0).astype(float)
    tdir = np.mod(tdir_deg, 360.0).astype(float)
    cerr = np.abs(((pdir - tdir + 180.0) % 360.0) - 180.0)  # [0,180]

    fig, axes = _new_map_axes(ncols=3, nrows=1, figsize=(18,5), lon_map=lon, lat_map=lat)
    im0 = _draw_field_on_ax(axes[0], pdir, lon, lat, kcs, cmap='hsv',   vmin=0.0, vmax=360.0, title="Pred Dir (deg)")
    im1 = _draw_field_on_ax(axes[1], tdir, lon, lat, kcs, cmap='hsv',   vmin=0.0, vmax=360.0, title="True Dir (deg)")
    im2 = _draw_field_on_ax(axes[2], cerr, lon, lat, kcs, cmap='magma', vmin=0.0, vmax=180.0, title="|Circular err| (deg)")
    for ax, im in zip(axes, (im0, im1, im2)):
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout(); plt.savefig(_safe_fname(fname), dpi=300, bbox_inches='tight'); plt.close()
    print("saved:", fname)

# -----------------------------
# Output dir, plots
# -----------------------------
def create_output_directory(pth_filename="model_weights_default.pth"):
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    pth_basename = os.path.splitext(os.path.basename(pth_filename))[0]
    dir_name = re.sub(r"[^0-9A-Za-z_\-.]", "_", f"{current_date}_{pth_basename}")
    os.makedirs(dir_name, exist_ok=True); return dir_name

def _plot_spatial_sample(pred, true, lon_map, lat_map, kcs, fname, var_name="Hs", title_suffix="", norm_params=None):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if norm_params is not None:
        vmin, vmax = norm_params; pred = denorm(pred, vmin, vmax); true = denorm(true, vmin, vmax)
    kcs2d = kcs if kcs.ndim == 2 else kcs[0]
    valid = ((kcs2d==1) | (kcs2d>0)) & np.isfinite(true)
    vmin = np.nanmin(true[valid]) if valid.any() else 0.0
    vmax = np.nanmax(true[valid]) if valid.any() else 1.0
    fig, axes = _new_map_axes(ncols=3, nrows=1, figsize=(18,5), lon_map=lon_map, lat_map=lat_map)
    items=[("Pred",pred,"jet",vmin,vmax),("True",true,"jet",vmin,vmax),("|Err|",np.abs(pred-true),"coolwarm",0,None)]
    for ax,(ttl,dat,cmap,lo,hi) in zip(axes, items):
        if hi is None:
            finite = np.asarray(dat)[np.isfinite(dat)]
            hi = float(np.nanmax(finite)) if finite.size else 1.0
        im = _draw_field_on_ax(ax, dat, lon_map, lat_map, kcs, cmap=cmap, vmin=lo, vmax=hi, title=ttl)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.suptitle(f"Spatial sample for {var_name}{title_suffix}")
    plt.tight_layout(); plt.savefig(_safe_fname(fname), dpi=300, bbox_inches='tight'); plt.close(); print("saved:", fname)

def _plot_spatial_rmse_maps(rmse_hs, rmse_tm, rmse_dir, lon_map, lat_map, kcs_map, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig, axes = _new_map_axes(ncols=3, figsize=(18,5), lon_map=lon_map, lat_map=lat_map)
    items=[("Hs RMSE (m)", rmse_hs, "viridis"), ("Tm RMSE (s)", rmse_tm, "viridis"), ("Dir cRMSE (°)", rmse_dir, "magma")]
    for ax,(ttl,dat,cmap) in zip(axes, items):
        vmax = np.nanpercentile(dat,99) if np.isfinite(dat).any() else 1.0
        im = _draw_field_on_ax(ax, dat, lon_map, lat_map, kcs_map, cmap=cmap, vmin=0, vmax=vmax, title=ttl)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label(ttl.split()[-1].strip("()"))
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    plt.suptitle("Spatial RMSE maps (window)", y=1.03, fontsize=13)
    plt.tight_layout()
    plt.savefig(_safe_fname(fname), dpi=300, bbox_inches='tight'); plt.close(); print("saved:", fname)

def _plot_timeseries(ts_dict: dict, station_name: str, var_name: str, fname: str, norm_params=(0.0,1.0), date_index=None, first_hour=0):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if var_name in ('hs','tm'):
        vmin,vmax=norm_params
        true_denorm=[denorm(x,vmin,vmax) for x in ts_dict["true"]]
        pred_denorm=[denorm(x,vmin,vmax) for x in ts_dict.get("pred",[])]
        meas_phys=[denorm(x,vmin,vmax) if np.isfinite(x) else np.nan for x in ts_dict.get("meas",[])]
    else:
        true_denorm=ts_dict["true"]; pred_denorm=ts_dict.get("pred_corrected", ts_dict.get("pred",[])); meas_phys=ts_dict.get("meas",[])
    base_dates = date_index[first_hour:first_hour+len(true_denorm)] if date_index is not None else np.arange(len(true_denorm))
    meas_aligned = meas_phys[:len(true_denorm)]
    eng_name = STATION_EN.get(station_name, station_name)
    plt.figure(figsize=(12,5))
    plt.plot(base_dates,true_denorm,"k-",label="Simulation",linewidth=2.2)
    if len(pred_denorm)>0: plt.plot(base_dates,pred_denorm,"b--",label="Predicted",linewidth=2.2)
    if len(meas_aligned)>0: plt.plot(base_dates,meas_aligned,"r:",label="Observed",linewidth=1.8,alpha=0.8)
    name_map = {'hs': 'Hs', 'tm': 'Tm', 'dir': 'Dir'}
    unit_map = {'hs': '(m)', 'tm': '(s)', 'dir': '(°)'}
    disp = name_map.get(var_name, var_name)
    ylabel = f"{disp} {unit_map.get(var_name, '')}".strip()
    LABEL_FONTSIZE = 22
    TICK_FONTSIZE  = 16
    LEGEND_FONTSIZE = 20
    TITLE_FONTSIZE  = 22
    plt.title(f"{eng_name} – {disp} (time series)", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Date/Time (UTC)", fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=LEGEND_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout(); plt.savefig(_safe_fname(fname), dpi=300); plt.close(); print("saved:", fname)

def create_cdf_error_plots(station_results, norm_params, output_dir):
    os.makedirs(output_dir, exist_ok=True); variables=['hs','tm','dir']; plt.figure(figsize=(18,5))
    for i,var in enumerate(variables,1):
        errors=[]
        for st in station_results.values():
            if var not in st: continue
            if var=='dir':
                p=np.array(st[var].get('pred_corrected', st[var].get('pred',[]))); t=np.array(st[var]['true'])
                m=np.isfinite(p)&np.isfinite(t);
                if m.any(): errors.extend(circular_diff_deg(p[m], t[m]))
            else:
                p=denorm(np.array(st[var].get('pred',[])),*norm_params[var]); t=denorm(np.array(st[var]['true']),*norm_params[var])
                m=np.isfinite(p)&np.isfinite(t);
                if m.any(): errors.extend((p-t)[m])
        if not errors: continue
        errors=np.asarray(errors); ecdf=np.sort(errors); cdf=np.linspace(0,1,len(ecdf),endpoint=False)
        ax=plt.subplot(1,3,i); ax.plot(ecdf,cdf); ax.axvline(0,color='k',lw=1)
        ax.set_title(f'CDF Error – {var.upper()}'); ax.set_xlabel('Error' + (' (deg)' if var=='dir' else '')); ax.set_ylabel('Probability'); ax.grid(True,alpha=0.3)
    fname=os.path.join(output_dir,"error_cdf.png"); plt.tight_layout(); plt.savefig(_safe_fname(fname), dpi=300); plt.close(); print("saved:", fname)

# -----------------------------
# Core evaluation — SINGLE WINDOW (build stack on the fly)
# -----------------------------
def evaluate_window(
    *,
    model,
    ds_win: xr.Dataset,
    time_index_full: pd.DatetimeIndex,
    global_norm_params: dict,
    seq_length: int,
    batch_size: int,
    station_data_full: dict,
    out_model_path: str,
    window_prefix: str = "win",
    save_limit: int = 3,
    use_bnd: bool = True,
    force_bnd_rotation_deg: float | None = None,   # * Added: fixed BND direction angle
):
    # 1) Preprocess only this window
    input_win, wave_win, lon, lat, kcs = load_and_preprocess_window(ds_win, global_norm_params)

    # 2) (optional) BND for this window
    if use_bnd:
        try:
            bnd_feat = build_bnd_features(
                ds_win, kcs, pd.to_datetime(ds_win['time'].values).tz_localize(None),
                global_norm_params, force_rotation_deg=force_bnd_rotation_deg  # * pass the fixed -90 deg through
            )
            if bnd_feat.shape[0] != input_win.shape[0]:
                Tuse = min(bnd_feat.shape[0], input_win.shape[0])
                input_win = input_win[:Tuse]; wave_win  = wave_win[:Tuse]; bnd_feat  = bnd_feat[:Tuse]
            input_win = np.concatenate([input_win, bnd_feat], axis=1)
            print(f"[BND] window appended → input channels = {input_win.shape[1]}")
        except Exception as e:
            warnings.warn(f"[BND] window failed ({type(e).__name__}: {e}) -> using 6 channels")
            use_bnd = False

    # 3) Dataset/Loader for this window
    T = input_win.shape[0]
    start_idx = 0
    end_idx = max(0, T - seq_length)
    if end_idx <= start_idx:
        print("[EVAL] window too short for sequence length.")
        return None

    ds_local = WindWaveDataset(input_win, wave_win, seq_length, start_idx, end_idx)
    loader = DataLoader(ds_local, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # 4) Evaluation
    model.eval()
    dev = next(model.parameters()).device
    out_dir = create_output_directory(out_model_path)
    print(f"[EVAL] Saving figures → {out_dir}")

    st_indices = {n: find_nearest_index(lon, lat, kcs, m["lon"], m["lat"]) for n,m in STATIONS.items()}
    cos_lat = np.cos(np.deg2rad(lat)); cos_lat[((kcs != 1) & (kcs <= 0)) | ~np.isfinite(cos_lat)] = 0
    spatial_w = cos_lat / (cos_lat.sum() + 1e-12)

    H, W = (kcs.shape if kcs.ndim == 2 else kcs[0].shape)
    rmse_acc = {'hs':{'sum':np.zeros((H,W),dtype=np.float64),'cnt':np.zeros((H,W),dtype=np.int64)},
                'tm':{'sum':np.zeros((H,W),dtype=np.float64),'cnt':np.zeros((H,W),dtype=np.int64)},
                'dir':{'sum':np.zeros((H,W),dtype=np.float64),'cnt':np.zeros((H,W),dtype=np.int64)}}

    variables = ['hs','tm','dir']
    st_ts = {n:{v:{"pred":[], "true":[], "meas":[]} for v in variables} for n in STATIONS}
    metric_results = {k:[] for k in ["rmse_hs","rmse_tm","rmse_dir","mae_hs","mae_tm","mae_dir","bias_hs","bias_tm","bias_dir",
                                     "cc_hs","cc_tm","cc_dir","r2_hs","r2_tm","r2_dir","acc_hs","mape_hs","pred_wmean","true_wmean"]}

    save_cnt=0
    t0 = int(np.searchsorted(time_index_full.values, np.datetime64(pd.to_datetime(ds_win['time'].values)[0].tz_localize(None))))
    with torch.no_grad():
        for bidx, (x,y) in enumerate(loader):
            x, y = x.to(dev), y.to(dev)
            pred_maps = model(x)
            for b in range(pred_maps[0].size(0)):
                pred_hs = pred_maps[0][b,0].cpu().numpy()
                pred_tm = pred_maps[0][b,1].cpu().numpy()
                pred_sin = pred_maps[0][b,2].cpu().numpy()
                pred_cos = pred_maps[0][b,3].cpu().numpy()
                pred_dir = (np.rad2deg(np.arctan2(pred_sin, pred_cos)) + 360) % 360

                true_hs = y[b,0].cpu().numpy(); true_tm = y[b,1].cpu().numpy()
                true_sin = y[b,2].cpu().numpy(); true_cos = y[b,3].cpu().numpy()
                true_dir = (np.rad2deg(np.arctan2(true_sin, true_cos)) + 360) % 360

                hs_min, hs_max = global_norm_params['hs']; tm_min, tm_max = global_norm_params['tm']
                pred_hs_phys = denorm(pred_hs, hs_min, hs_max); true_hs_phys = denorm(true_hs, hs_min, hs_max)
                pred_tm_phys = denorm(pred_tm, tm_min, tm_max); true_tm_phys = denorm(true_tm, tm_min, tm_max)
                pred_dir_phys = pred_dir; true_dir_phys = true_dir

                kcs2d = kcs if kcs.ndim == 2 else kcs[0]
                mask_h = ((kcs2d == 1) | (kcs2d > 0)) & np.isfinite(true_hs_phys) & np.isfinite(pred_hs_phys)
                rmse_acc['hs']['sum'][mask_h] += (pred_hs_phys[mask_h] - true_hs_phys[mask_h]) ** 2
                rmse_acc['hs']['cnt'][mask_h] += 1

                mask_t = ((kcs2d == 1) | (kcs2d > 0)) & np.isfinite(true_tm_phys) & np.isfinite(pred_tm_phys)
                rmse_acc['tm']['sum'][mask_t] += (pred_tm_phys[mask_t] - true_tm_phys[mask_t]) ** 2
                rmse_acc['tm']['cnt'][mask_t] += 1

                mask_d = ((kcs2d == 1) | (kcs2d > 0)) & np.isfinite(true_dir_phys) & np.isfinite(pred_dir_phys)
                diff_dir = circular_diff_deg(pred_dir_phys[mask_d], true_dir_phys[mask_d])
                rmse_acc['dir']['sum'][mask_d] += diff_dir ** 2
                rmse_acc['dir']['cnt'][mask_d] += 1

                test_t = t0 + (bidx*loader.batch_size + b) + seq_length
                for st_name, (i,j) in st_indices.items():
                    if 0 <= test_t < len(station_data_full[st_name]):
                        meas_hs, meas_tm, meas_dir = station_data_full[st_name][test_t]
                    else:
                        meas_hs, meas_tm, meas_dir = (np.nan, np.nan, np.nan)
                    st_ts[st_name]['hs']['pred'].append(pred_hs[i,j]); st_ts[st_name]['hs']['true'].append(true_hs[i,j]); st_ts[st_name]['hs']['meas'].append(meas_hs)
                    st_ts[st_name]['tm']['pred'].append(pred_tm[i,j]); st_ts[st_name]['tm']['true'].append(true_tm[i,j]); st_ts[st_name]['tm']['meas'].append(meas_tm)
                    st_ts[st_name]['dir']['pred'].append(pred_dir[i,j]); st_ts[st_name]['dir']['true'].append(true_dir[i,j]); st_ts[st_name]['dir']['meas'].append(meas_dir % 360 if np.isfinite(meas_dir) else np.nan)

                # aggregate scalar metrics (area)
                if mask_h.sum() > 0:
                    w = spatial_w[mask_h]
                    rmse_hs = np.sqrt(np.mean((pred_hs_phys[mask_h] - true_hs_phys[mask_h])**2))
                    mae_hs = np.mean(np.abs(pred_hs_phys[mask_h] - true_hs_phys[mask_h]))
                    bias_hs = np.mean(pred_hs_phys[mask_h] - true_hs_phys[mask_h])
                    cc_hs = np.corrcoef(pred_hs_phys[mask_h], true_hs_phys[mask_h])[0,1]
                    r2_hs = 1 - np.sum((true_hs_phys[mask_h] - pred_hs_phys[mask_h])**2) / np.sum((true_hs_phys[mask_h] - np.mean(true_hs_phys[mask_h]))**2)
                    acc_hs = _safe_acc(pred_hs_phys[mask_h], true_hs_phys[mask_h]); mape_hs = _safe_mape(pred_hs_phys[mask_h], true_hs_phys[mask_h])
                    pred_wmean_hs = np.sum(pred_hs_phys[mask_h] * w); true_wmean_hs = np.sum(true_hs_phys[mask_h] * w)
                    metric_results["rmse_hs"].append(rmse_hs); metric_results["mae_hs"].append(mae_hs); metric_results["bias_hs"].append(bias_hs)
                    metric_results["cc_hs"].append(cc_hs); metric_results["r2_hs"].append(r2_hs); metric_results["acc_hs"].append(acc_hs); metric_results["mape_hs"].append(mape_hs)
                    metric_results["pred_wmean"].append(pred_wmean_hs); metric_results["true_wmean"].append(true_wmean_hs)

                if mask_t.sum() > 0:
                    rmse_tm = np.sqrt(np.mean((pred_tm_phys[mask_t] - true_tm_phys[mask_t])**2))
                    mae_tm = np.mean(np.abs(pred_tm_phys[mask_t] - true_tm_phys[mask_t]))
                    bias_tm = np.mean(pred_tm_phys[mask_t] - true_tm_phys[mask_t])
                    cc_tm = np.corrcoef(pred_tm_phys[mask_t], true_tm_phys[mask_t])[0,1]
                    r2_tm = 1 - np.sum((true_tm_phys[mask_t] - pred_tm_phys[mask_t])**2) / np.sum((true_tm_phys[mask_t] - np.mean(true_tm_phys[mask_t]))**2)
                    metric_results["rmse_tm"].append(rmse_tm); metric_results["mae_tm"].append(mae_tm); metric_results["bias_tm"].append(bias_tm)
                    metric_results["cc_tm"].append(cc_tm); metric_results["r2_tm"].append(r2_tm)

                if mask_d.sum() > 0:
                    rmse_dir = circular_rmse_deg(pred_dir_phys[mask_d], true_dir_phys[mask_d])
                    mae_dir = circular_mae_deg(pred_dir_phys[mask_d], true_dir_phys[mask_d])
                    bias_dir = np.mean(circular_diff_deg(pred_dir_phys[mask_d], true_dir_phys[mask_d]))
                    cc_dir = circular_correlation(pred_dir_phys[mask_d], true_dir_phys[mask_d])
                    r2_dir = circular_r2(pred_dir_phys[mask_d], true_dir_phys[mask_d])
                    metric_results["rmse_dir"].append(rmse_dir); metric_results["mae_dir"].append(mae_dir); metric_results["bias_dir"].append(bias_dir)
                    metric_results["cc_dir"].append(cc_dir); metric_results["r2_dir"].append(r2_dir)

                if save_cnt < save_limit:
                    _plot_spatial_sample(pred_hs_phys, true_hs_phys, lon, lat, kcs,
                                         os.path.join(out_dir, f"{window_prefix}_spatial_hs_{save_cnt}.png"),
                                         var_name="Hs", title_suffix=f" (step {bidx*loader.batch_size + b})", norm_params=None)
                    _plot_spatial_sample(pred_tm_phys, true_tm_phys, lon, lat, kcs,
                                         os.path.join(out_dir, f"{window_prefix}_spatial_tm_{save_cnt}.png"),
                                         var_name="Tm", title_suffix=f" (step {bidx*loader.batch_size + b})", norm_params=None)
                    # * Dir samples use HSV with fixed 0-360 deg
                    _plot_dir_sample(pred_dir_phys, true_dir_phys, lon, lat, kcs,
                                     os.path.join(out_dir, f"{window_prefix}_spatial_dir_{save_cnt}.png"))
                    save_cnt += 1

    # 5) spatial RMSE maps (for this window)
    rmse_hs_map = np.full((H,W), np.nan); m = rmse_acc['hs']['cnt'] > 0
    rmse_hs_map[m] = np.sqrt(rmse_acc['hs']['sum'][m] / np.maximum(rmse_acc['hs']['cnt'][m], 1))
    rmse_tm_map = np.full((H,W), np.nan); m = rmse_acc['tm']['cnt'] > 0
    rmse_tm_map[m] = np.sqrt(rmse_acc['tm']['sum'][m] / np.maximum(rmse_acc['tm']['cnt'][m], 1))
    rmse_dir_map = np.full((H,W), np.nan); m = rmse_acc['dir']['cnt'] > 0
    rmse_dir_map[m] = np.sqrt(rmse_acc['dir']['sum'][m] / np.maximum(rmse_acc['dir']['cnt'][m], 1))
    _plot_spatial_rmse_maps(rmse_hs_map, rmse_tm_map, rmse_dir_map, lon, lat, kcs,
                            os.path.join(out_dir, f"{window_prefix}_spatial_rmse_maps.png"))

    # 6) station-wise plots
    start_pos = int(np.searchsorted(time_index_full.values, np.datetime64(pd.to_datetime(ds_win['time'].values)[0].tz_localize(None))))
    for st_name, ts_data in st_ts.items():
        for var in variables:
            _plot_timeseries(ts_data[var], st_name, var,
                             os.path.join(out_dir, f"{window_prefix}_timeseries_{st_name}_{var}.png"),
                             norm_params=global_norm_params[var], date_index=time_index_full,
                             first_hour=start_pos + seq_length)

    create_cdf_error_plots(st_ts, global_norm_params, out_dir)

    # ---- B-1/B-2: event-peak & observation-agreement metrics at Haeundae ----
    peak_metrics = {}; obs_metrics = {}
    try:
        HAE = "Haeundae Beach"
        hs_lo, hs_hi = global_norm_params['hs']
        _ts = st_ts.get(HAE, {}).get('hs', {})
        _pred = np.array([denorm(x, hs_lo, hs_hi) for x in _ts.get('pred', [])], dtype=float)
        _true = np.array([denorm(x, hs_lo, hs_hi) for x in _ts.get('true', [])], dtype=float)
        _meas = np.array([denorm(x, hs_lo, hs_hi) if np.isfinite(x) else np.nan
                          for x in _ts.get('meas', [])], dtype=float)
        peak_metrics = {'Emulator_vs_Obs': _event_peak_metrics(_pred, _meas),
                        'SWAN_vs_Obs':     _event_peak_metrics(_true, _meas)}
        obs_metrics  = {'Emulator_vs_Obs': _obs_agreement_metrics(_pred, _meas),
                        'SWAN_vs_Obs':     _obs_agreement_metrics(_true, _meas)}
    except Exception as e:
        warnings.warn(f"[PEAK] Haeundae peak/obs metrics failed: {type(e).__name__}: {e}")

    def _mean_or_nan(x): return float(np.nanmean(x)) if (x and len(x)>0) else float('nan')
    summ = {
        'rmse_hs': _mean_or_nan(metric_results['rmse_hs']),
        'mae_hs' : _mean_or_nan(metric_results['mae_hs']),
        'cc_hs'  : _mean_or_nan(metric_results['cc_hs']),
        'station_results': st_ts,
        'metric_results' : metric_results,
        'peak': peak_metrics, 'obs_agree': obs_metrics
    }
    return summ

# -----------------------------
# Typhoon loop + tables
# -----------------------------
def _write_typhoon_tables(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    simple_cols = ["Typhoon", "RMSE (m)", "MAE (m)", "r"]
    df_simple = pd.DataFrame([{k: r[k] for k in simple_cols} for r in results], columns=simple_cols)
    simple_csv = os.path.join(out_dir, "typhoon_skill_simple.csv")
    df_simple.to_csv(simple_csv, index=False)
    simple_tex = os.path.join(out_dir, "typhoon_skill_simple.tex")
    with open(simple_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp] \\centering\\small "
                "\\caption{Skill during major typhoons (2019–2020). Bold indicates best among models.} "
                "\\label{tab:typhoon_skill} "
                "\\begin{tabular}{lccc} \\toprule "
                "Typhoon & RMSE (m) & MAE (m) & $r$ \\\\ \\midrule\n")
        for r in results:
            f.write(f"{r['Typhoon']} & {r['RMSE (m)']:.2f} & {r['MAE (m)']:.2f} & {r['r']:.2f}  \\\\\n")
        f.write("\\bottomrule \\end{tabular} \\end{table}\n")
    print(f"[TY] saved: {simple_csv}, {simple_tex}")

def _write_typhoon_tables_full(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=["Typhoon","RMSE (m)","MAE (m)","r","PICP (%)","F1","MPIW (m)"])
    csv_path = os.path.join(out_dir, "typhoon_skill.csv")
    tex_path = os.path.join(out_dir, "typhoon_skill.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\\centering\\small\n")
        f.write("\\caption{Skill during major typhoons (2019--2020).}\\label{tab:typhoon_skill_full}\n")
        f.write("\\begin{tabular}{lcccccc}\\toprule\n")
        f.write("Typhoon & RMSE (m) & MAE (m) & $r$ & PICP (\\%) & F1 & MPIW (m) \\\\\\midrule\n")
        for r in results:
            f.write(f"{r['Typhoon']} & {r['RMSE (m)']:.2f} & {r['MAE (m)']:.2f} & {r['r']:.2f} & -- & -- & -- \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"[TY] saved: {csv_path}, {tex_path}")

def _write_typhoon_peak_tables(peak_rows, out_dir):
    cols = ["Typhoon","Model","Obs peak (m)","Model peak (m)","Peak bias (m)",
            "Rel. peak err (%)","Timing err (h)","P95 err (m)","P99 err (m)","Exceed. F1 (>=3 m)"]
    pd.DataFrame(peak_rows, columns=cols).to_csv(os.path.join(out_dir,"typhoon_peak_skill.csv"), index=False)
    def _f(x, nd=2):
        return "--" if (x is None or (isinstance(x,float) and not np.isfinite(x))) else f"{x:.{nd}f}"
    with open(os.path.join(out_dir,"typhoon_peak_skill.tex"),"w") as f:
        f.write("\\begin{table}[htbp]\\centering\\small\n")
        f.write("\\caption{Event-peak diagnostics at the Haeundae buoy-nearest grid cell for $H_s$ "
                "during major typhoons (2019--2020). Peak bias and relative peak error are model minus "
                "observed. Timing error is the signed difference between model and observed times of "
                "maximum $H_s$ (hours; positive means the model peak is later). Exceedance F1 uses a "
                "3~m $H_s$ threshold.}\\label{tab:typhoon_peak}\n")
        f.write("\\begin{tabular}{llrrrrrrrr}\\toprule\n")
        f.write("Typhoon & Model & Obs peak (m) & Model peak (m) & Peak bias (m) & Rel. peak err (\\%) "
                "& Timing err (h) & P95 err (m) & P99 err (m) & Exceed.\\ F1 \\\\\\midrule\n")
        for r in peak_rows:
            f.write(f"{r[0]} & {r[1]} & {_f(r[2])} & {_f(r[3])} & {_f(r[4])} & {_f(r[5],1)} "
                    f"& {_f(r[6],1)} & {_f(r[7])} & {_f(r[8])} & {_f(r[9],2)} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print("saved: typhoon_peak_skill.csv / .tex")

def _write_typhoon_obs_tables(obs_rows, out_dir):
    cols = ["Typhoon","Comparison","N","RMSE (m)","RMSE/mean (%)","Pearson r","p-value","Willmott d"]
    pd.DataFrame(obs_rows, columns=cols).to_csv(os.path.join(out_dir,"typhoon_obs_skill.csv"), index=False)
    def _f(x, nd=3):
        return "--" if (x is None or (isinstance(x,float) and not np.isfinite(x))) else f"{x:.{nd}f}"
    with open(os.path.join(out_dir,"typhoon_obs_skill.tex"),"w") as f:
        f.write("\\begin{table}[htbp]\\centering\\small\n")
        f.write("\\caption{Agreement with buoy observations at the Haeundae buoy-nearest grid cell for "
                "$H_s$ during major typhoons (2019--2020). $d$ is the Willmott index of agreement. "
                "Sim denotes the parent SWAN/Delft3D-FM WAVE field and Pred the emulator.}"
                "\\label{tab:typhoon_obs}\n")
        f.write("\\begin{tabular}{llrrrrrr}\\toprule\n")
        f.write("Typhoon & Comparison & $N$ & RMSE (m) & RMSE/mean (\\%) & $r$ & $p$ & $d$ \\\\\\midrule\n")
        for r in obs_rows:
            pv = r[6]
            if pv is None or (isinstance(pv,float) and not np.isfinite(pv)): pv_s = "--"
            elif pv < 1e-3: pv_s = f"{pv:.1e}"
            else: pv_s = f"{pv:.3f}"
            nn = int(r[2]) if (isinstance(r[2],(int,float)) and np.isfinite(r[2])) else "--"
            f.write(f"{r[0]} & {r[1]} & {nn} & {_f(r[3])} & {_f(r[4],1)} & {_f(r[5])} & {pv_s} & {_f(r[7])} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print("saved: typhoon_obs_skill.csv / .tex")


def run_typhoon_windows(
    *,
    model,
    ds_sim: xr.Dataset,
    lon_map, lat_map, kcs_map,
    time_index: pd.DatetimeIndex,
    global_norm_params: dict,
    seq_length: int,
    batch_size: int,
    station_data: dict,
    out_model_path: str,
    typhoon_dict: dict = TYPHOONS,
    save_limit: int = 3,
    use_bnd: bool = True,
    force_bnd_rotation_deg: float | None = -90.0,   # * default -90 deg fixed
):
    results = []
    peak_rows = []
    obs_rows = []
    for name, info in typhoon_dict.items():
        print(f"[TY] Processing {name} ...")
        # Window computation
        def _window_indices(time_index: pd.DatetimeIndex, start_utc: pd.Timestamp, end_utc: pd.Timestamp, seq_len: int):
            start_utc = pd.Timestamp(start_utc).tz_localize(None)
            end_utc   = pd.Timestamp(end_utc).tz_localize(None)
            s = int(np.searchsorted(time_index.values, np.datetime64(start_utc))) - seq_len
            e = int(np.searchsorted(time_index.values, np.datetime64(end_utc)))
            return max(0, s), max(0, e)
        s_idx, e_idx = _window_indices(time_index, info["start"], info["end"], seq_length)
        s_load = max(0, s_idx - seq_length); e_load = e_idx
        ds_win = ds_sim.isel(time=slice(s_load, e_load))
        summ = evaluate_window(
            model=model, ds_win=ds_win, time_index_full=time_index, global_norm_params=global_norm_params,
            seq_length=seq_length, batch_size=batch_size, station_data_full=station_data,
            out_model_path=out_model_path, window_prefix=info['prefix'], save_limit=save_limit,
            use_bnd=use_bnd, force_bnd_rotation_deg=force_bnd_rotation_deg  # * pass -90 deg through
        )
        if summ is None:
            results.append({"Typhoon": name.capitalize(), "RMSE (m)": np.nan, "MAE (m)": np.nan, "r": np.nan,
                            "PICP (%)": np.nan, "F1": np.nan, "MPIW (m)": np.nan})
            for _lbl in ("SWAN","Emulator"):
                peak_rows.append([name.capitalize(), _lbl] + [np.nan]*8)
            for _lbl in ("Sim vs Obs","Pred vs Obs"):
                obs_rows.append([name.capitalize(), _lbl] + [np.nan]*6)
            continue
        results.append({
            "Typhoon": name.capitalize(),
            "RMSE (m)": float(summ['rmse_hs']),
            "MAE (m)":  float(summ['mae_hs']),
            "r":        float(summ['cc_hs']),
            "PICP (%)": np.nan, "F1": np.nan, "MPIW (m)": np.nan
        })
        _pk = summ.get("peak", {}) or {}
        for _tag, _lbl in (('SWAN_vs_Obs','SWAN'), ('Emulator_vs_Obs','Emulator')):
            _d = _pk.get(_tag, {}) or {}
            peak_rows.append([name.capitalize(), _lbl,
                _d.get('obs_peak',np.nan), _d.get('model_peak',np.nan), _d.get('peak_bias',np.nan),
                _d.get('rel_peak_err_pct',np.nan), _d.get('timing_err_h',np.nan),
                _d.get('p95_err',np.nan), _d.get('p99_err',np.nan), _d.get('exc_f1',np.nan)])
        _oa = summ.get("obs_agree", {}) or {}
        for _tag, _lbl in (('SWAN_vs_Obs','Sim vs Obs'), ('Emulator_vs_Obs','Pred vs Obs')):
            _d = _oa.get(_tag, {}) or {}
            obs_rows.append([name.capitalize(), _lbl, _d.get('n',np.nan),
                _d.get('rmse',np.nan), _d.get('rmse_over_mean_pct',np.nan),
                _d.get('pearson_r',np.nan), _d.get('p_value',np.nan), _d.get('willmott_d',np.nan)])

    out_dir = create_output_directory(out_model_path)
    _write_typhoon_tables(results, out_dir)
    _write_typhoon_tables_full(results, out_dir)
    _write_typhoon_peak_tables(peak_rows, out_dir)
    _write_typhoon_obs_tables(obs_rows, out_dir)

# -----------------------------
# Speed benchmark (optional)
# -----------------------------
def run_speed_benchmark(
    *,
    model: nn.Module,
    ds_sim: xr.Dataset,
    seq_length: int,
    batch_size: int,
    global_norm_params: dict,
    out_model_path: str,
    repeats: int = 1,
    baseline_min_per_hour: float | None = None,
    bench_hours: int = 7*24,
    use_bnd: bool = True,
    force_bnd_rotation_deg: float | None = None,
):
    model.eval()
    dev = next(model.parameters()).device
    if 'time' in ds_sim:
        t_all = pd.to_datetime(ds_sim['time'].values)
        t0 = 0
        t1 = min(len(t_all), t0 + bench_hours + seq_length + 1)
        ds_win = ds_sim.isel(time=slice(t0, t1))
    else:
        ds_win = ds_sim

    input_win, wave_win, lon, lat, kcs = load_and_preprocess_window(ds_win, global_norm_params)
    if use_bnd:
        try:
            bnd_feat = build_bnd_features(
                ds_win, kcs, pd.to_datetime(ds_win['time'].values).tz_localize(None),
                global_norm_params, force_rotation_deg=force_bnd_rotation_deg
            )
            Tuse = min(bnd_feat.shape[0], input_win.shape[0])
            input_win = input_win[:Tuse]; wave_win = wave_win[:Tuse]; bnd_feat = bnd_feat[:Tuse]
            input_win = np.concatenate([input_win, bnd_feat], axis=1)
            print(f"[SPD] BND on (C={input_win.shape[1]})")
        except Exception as e:
            warnings.warn(f"[SPD] BND failed ({type(e).__name__}: {e}) -> using 6 channels")
            use_bnd = False

    T = input_win.shape[0]
    start_idx, end_idx = 0, max(0, T - seq_length)
    if end_idx <= start_idx:
        print("[SPD] Not enough timesteps for speed benchmark."); return

    ds_local = WindWaveDataset(input_win, np.zeros_like(wave_win), seq_length, start_idx, end_idx)
    loader = DataLoader(ds_local, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=(dev.type == "cuda"))

    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(dev, non_blocking=True)
            _ = model(xb)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            break

    n_steps = len(ds_local)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _r in range(repeats):
            for xb, _ in loader:
                xb = xb.to(dev, non_blocking=True)
                _ = model(xb)
                if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    sec_per_step = elapsed / (n_steps * repeats)
    steps_per_sec = 1.0 / sec_per_step
    H, W = input_win.shape[-2], input_win.shape[-1]
    cell_steps_per_sec = H * W * steps_per_sec
    speedup = None
    if baseline_min_per_hour is not None and baseline_min_per_hour > 0:
        speedup = (baseline_min_per_hour * 60.0) / sec_per_step

    out_dir = create_output_directory(out_model_path)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "speed_benchmark.csv")
    tex_path = os.path.join(out_dir, "speed_benchmark.tex")
    row = {
        "Device": str(dev), "Batch": batch_size, "SeqLen": seq_length, "Grid": f"{H}x{W}",
        "C_in": input_win.shape[1], "s/step": round(sec_per_step, 6), "steps/s": round(steps_per_sec, 3),
        "cell-steps/s": int(cell_steps_per_sec),
        "baseline_min/h": baseline_min_per_hour if baseline_min_per_hour is not None else None,
        "speedup_x": round(speedup, 2) if speedup is not None else None,
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\\centering\\small\n")
        f.write("\\caption{Inference throughput and speedup.}\\label{tab:speedup}\n")
        f.write("\\begin{tabular}{lcccccc}\\toprule\n")
        f.write("Batch & Seq & Grid & C$_{in}$ & s/step & steps/s & speedup ($\\times$) \\\\\\midrule\n")
        f.write(f"{batch_size} & {seq_length} & {H}x{W} & {input_win.shape[1]} & {sec_per_step:.4f} & {steps_per_sec:.1f} & "
                f"{(f'{speedup:.1f}' if speedup is not None else '--')} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"[SPD] saved: {csv_path}, {tex_path}")

# -----------------------------
# main
# -----------------------------
def _find_latest_file(patterns, roots):
    import glob
    cand = []
    for root in roots:
        if not os.path.isdir(root): continue
        for pat in patterns:
            cand.extend(glob.glob(os.path.join(root, pat)))
    if not cand: return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def _autofill_paths(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [script_dir, os.path.join(script_dir, "data"), os.path.join(script_dir, "datasets"),
                   os.path.join(script_dir, "checkpoints"), os.path.join(script_dir, "weights"),
                   os.path.join(script_dir, "models"), os.getcwd()]
    if not args.data_path:
        args.data_path = _find_latest_file(["*.nc", "*.nc4", "*.netcdf"], search_dirs)
        if args.data_path: print(f"[AUTO] Using data_path={args.data_path}")
    if not args.model_path:
        args.model_path = _find_latest_file(["*.pth", "*.pt", "*.ckpt"], search_dirs)
        if args.model_path: print(f"[AUTO] Using model_path={args.model_path}")
    if not args.data_path or not os.path.isfile(args.data_path):
        raise FileNotFoundError("[AUTO] No NetCDF (.nc) file found. Please specify --data_path.")
    if not args.model_path or not os.path.isfile(args.model_path):
        raise FileNotFoundError("[AUTO] No model checkpoint (.pth/.pt/.ckpt) found. Please specify --model_path.")

def main():
    parser = argparse.ArgumentParser(description="UNet-ConvLSTM Inference (Windowed stacking ver.; Typhoon BND=-90° fixed)")
    # Paths
    parser.add_argument('--data_path', type=str, default=r'C:\DELFT3DFM\South_Korea_emulator_2020_ST6_bnd_test\wave\wavm-Waves_2019_2020_final.nc')
    parser.add_argument('--model_path', type=str, default=r'20250919_110000_model_weights_17498_seq12_epochs20_hid128_UNET32_bndON.pth')

    # Core hyper-params
    parser.add_argument('--seq_length', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--feat', type=str, default='32,64,128,256,512')

    # BND switch + * forced rotation angle
    parser.add_argument('--bnd', dest='bnd', action='store_true', help='Enable BND boundary features (default ON)')
    parser.add_argument('--no-bnd', dest='bnd', action='store_false', help='Disable BND (use 6ch)')
    parser.set_defaults(bnd=True)
    parser.add_argument('--bnd_force_deg', type=float, default=-90.0,
                        help='Force rotation (deg) for BND sin/cos. Use -90.0 to match training; set 0 to disable forcing.')

    # Typhoon / speed bench
    parser.add_argument('--typhoon_table', action='store_true', help='Generate typhoon-window skill tables/figures')
    parser.add_argument('--speed_benchmark', action='store_true', help='Run inference throughput benchmark')
    parser.set_defaults(typhoon_table=True, speed_benchmark=True)

    parser.add_argument('--baseline_min_per_hour', type=float, default=None, help='Baseline (e.g., SWAN) minutes per 1-hour forecast')
    parser.add_argument('--speed_repeats', type=int, default=1, help='Repeats for smoothing timing')
    parser.add_argument('--bench_hours', type=int, default=7*24, help='Hours for speed benchmark window')

    args = parser.parse_args()
    _autofill_paths(args)

    print(f"[INFO] Loading NetCDF: {args.data_path}")
    ds_sim = xr.open_dataset(args.data_path)

    # align to hourly
    def _pad_to_hourly(ds):
        if 'time' not in ds: return ds
        t = pd.to_datetime(ds['time'].values)
        full = pd.date_range(t[0], t[-1], freq='H')
        return ds if len(full) == len(t) else ds.reindex({'time': full}, fill_value=np.nan)
    ds_sim = _pad_to_hourly(ds_sim)
    time_index = pd.to_datetime(ds_sim['time'].values).tz_localize(None) if 'time' in ds_sim else pd.date_range('2000-01-01', periods=1, freq='H')

    # normalization params (train-consistent)
    T = len(time_index)
    idx_train = np.arange(0, max(0, T - args.seq_length - 1), dtype=int)
    global_norm_params = compute_params_with_indices(ds_sim, idx_train, args.seq_length)

    # station CSVs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    station_data = load_all_station_data(script_dir, global_norm_params, time_index=time_index)

    # model init & weights
    feat_list = [int(x) for x in args.feat.split(',')]
    in_channels = 10 if args.bnd else 6
    model = UNetConvLSTM(input_channels=in_channels, output_channels=4, hidden_dim=args.hidden_dim, feat=feat_list).to(device)
    print(f"[INFO] Using input_channels={in_channels}, hidden_dim={args.hidden_dim}, feat={feat_list}")
    print(f"[INFO] Loading weights: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model_keys = dict(model.state_dict())
    drop = [k for k, v in state.items() if (k not in model_keys) or (model_keys[k].shape != v.shape)]
    for k in drop: state.pop(k, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

    # speed benchmark (optional)
    if args.speed_benchmark:
        run_speed_benchmark(
            model=model, ds_sim=ds_sim, seq_length=args.seq_length, batch_size=args.batch_size,
            global_norm_params=global_norm_params, out_model_path=args.model_path,
            repeats=args.speed_repeats, baseline_min_per_hour=args.baseline_min_per_hour,
            bench_hours=args.bench_hours, use_bnd=args.bnd, force_bnd_rotation_deg=args.bnd_force_deg
        )

    # typhoon windows (* default -90 deg fixed here)
    if args.typhoon_table:
        lon = ds_sim.get('x', xr.DataArray()).values if 'x' in ds_sim else None
        lat = ds_sim.get('y', xr.DataArray()).values if 'y' in ds_sim else None
        kcs = ds_sim.get('kcs', xr.DataArray()).values if 'kcs' in ds_sim else None
        run_typhoon_windows(
            model=model, ds_sim=ds_sim, lon_map=lon, lat_map=lat, kcs_map=kcs,
            time_index=time_index, global_norm_params=global_norm_params, seq_length=args.seq_length,
            batch_size=args.batch_size, station_data=station_data, out_model_path=args.model_path,
            typhoon_dict=TYPHOONS, save_limit=60, use_bnd=args.bnd, force_bnd_rotation_deg=args.bnd_force_deg
        )
    else:
        print("[INFO] --typhoon_table off; skipping typhoon skills.")

    ds_sim.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()
