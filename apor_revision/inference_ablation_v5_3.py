# -*- coding: utf-8 -*-
"""
UNet-ConvLSTM wave emulator INFERENCE ONLY
(Enhanced v49 → synchronized with v57 TRAIN changes)
- [SYNC v57] Input channels: wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad (+ optional 4x BND features)
- [ADD] Optional boundary features (BND) with auto direction alignment (same as train)
- [FIX] Completed truncated call for create_cdf_error_plots(...)
- [CHANGE] Memmap writer now adapts to dynamic input channel count (6 or 10)
- [SYNC v57] Model ctor now accepts feat list and input_channels to match training checkpoints
- [NEW] High-res coastline/land overlay with Cartopy (10m) + pcolormesh/imshow(bilinear) to soften blocky land mask
- [NEW] Buoy validation tables (CSV + LaTeX) & plot style aligned to manuscript (SWAN grey / Emulator orange / Buoy black)

★ FIX-MAPE:
  - Add robust MAPE with denominator threshold to avoid explosion near zeros
  - Use in evaluation loop (see part 2/3)

★ NEW (CSV):
  - Save aggregated Emulator-vs-SWAN metrics to CSV and LaTeX
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
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from tqdm import tqdm
from typing import Dict, Tuple
from scipy.ndimage import distance_transform_edt, binary_dilation
import seaborn as sns  # noqa
from pathlib import Path
import logging
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

# -- [NEW] Cartopy (optional)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAVE_CARTOPY = True
except Exception:
    _HAVE_CARTOPY = False

# ----------------------------
# Fonts (Korean-safe)
# ----------------------------
def _set_arial_font():
    # (선택) 스크립트 폴더에 Arial.ttf 등이 있다면 등록
    for fname in ("Arial.ttf", "arial.ttf", "Arial Unicode MS.ttf", "ArialUnicodeMS.ttf"):
        p = os.path.join(os.path.dirname(__file__), fname)
        if os.path.isfile(p):
            fm.fontManager.addfont(p)

    # 사용 가능한 폰트 확인
    available = {f.name for f in fm.fontManager.ttflist}

    # 기본: Arial, 없으면 Liberation Sans/Helvetica/DejaVu Sans로 유사 폰트 대체
    arial_like = [f for f in ("Arial", "ArialMT", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in available]
    # 한글 폴백(시스템에 있는 것만)
    korean_fallback = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if f in available]

    # 폰트 패밀리 우선순위(앞에서부터 시도)
    matplotlib.rcParams["font.family"] = arial_like + korean_fallback
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 수식 폰트가 산세리프(Arial과 어울리는 계열)로 보이도록
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"  # 또는 'stixsans'

    # PDF/SVG에 실제 글꼴 임베딩
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

_set_arial_font()

# ----------------------------
# [VECTOR EXPORT] 전역 설정 + 래스터화 개선
# ----------------------------
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['agg.path.chunksize'] = 20000

def _safe_fname(path: str) -> str:
    directory, filename = os.path.split(path)
    safe_filename = re.sub(r"[^0-9A-Za-z_\-\.]", "_", filename)
    if directory: os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, safe_filename)

from matplotlib.collections import (PathCollection, PolyCollection, LineCollection, QuadMesh)
from matplotlib.lines import Line2D
def _rasterize_axes_data(ax):
    ax.set_rasterization_zorder(0.1)
    for artist in ax.get_children():
        if isinstance(artist, (matplotlib.text.Text, matplotlib.legend.Legend, matplotlib.spines.Spine)):
            continue
        if isinstance(artist, (PathCollection, PolyCollection, LineCollection, QuadMesh, Line2D)):
            try:
                artist.set_zorder(0.0)
                artist.set_rasterized(True)
            except Exception:
                pass

def _prepare_figure_for_rasterized_export(fig=None):
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        _rasterize_axes_data(ax)

# SVG export is optional because it is slow and produces large files.
SAVE_SVG = False

def _savefig_vector(path_like_png: str):
    safe = _safe_fname(path_like_png)
    base, _ = os.path.splitext(safe)
    png_path = base + ".png"
    pdf_path = base + ".pdf"
    _prepare_figure_for_rasterized_export(plt.gcf())
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    if SAVE_SVG:
        svg_path = base + ".svg"
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()
        print(f"saved(raster+vector): {png_path}, {pdf_path}, {svg_path}")
    else:
        plt.close()
        print(f"saved(raster+pdf only, svg skipped): {png_path}, {pdf_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===== Global knobs =====
# 하한(threshold)보다 작은 참값은 MAPE 평균에서 제외해 폭주를 방지
MAPE_THRESH = {'hs': 0.25,  'tm': 1.0}  # units: m, s

# ======= Missing pieces: drop-in implementations =======

def _depth_grad_mag(depth2d: np.ndarray) -> np.ndarray:
    """
    수심(deptn) 2D 배열의 기울기 크기 |∇depth| 를 [0,1]로 강건 정규화.
    NaN은 주변 최근접값으로 채운 뒤 계산, 최종 NaN은 0으로.
    """
    a = np.asarray(depth2d, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    valid = np.isfinite(a)
    if not valid.any():
        return np.zeros_like(a, dtype=np.float32)

    # 빈 곳 메우기
    a_filled = _fill_invalid_with_nearest(a)

    # 단순 그래디언트
    gy, gx = np.gradient(a_filled.astype(np.float32))
    g = np.hypot(gx, gy)

    # 강건 정규화(1–99백분위)
    p1, p99 = np.nanpercentile(g[valid], [1.0, 99.0])
    if not np.isfinite(p1): p1 = 0.0
    if not np.isfinite(p99) or p99 <= p1:
        p99 = p1 + 1.0
    g = (g - p1) / (p99 - p1)
    g = np.clip(g, 0.0, 1.0).astype(np.float32)
    g[~valid] = 0.0
    return g

def load_all_station_data(station_root: str, norm_params: dict, time_index: pd.DatetimeIndex) -> dict:
    """
    Read all station CSV files listed in STATIONS and return arrays aligned
    to the model time axis. Each array has columns [hs_norm, tm_norm, dir_deg].

    The KHOA CSV timestamps are recorded in KST. The SWAN NetCDF time axis is
    UTC, so the timestamps are converted from Asia/Seoul to UTC before
    reindexing. Missing station values are preserved as NaN for skill metrics.
    """
    def _pick_column(df, keywords, exclude=()):
        """Pick a column by case-insensitive substring matching."""
        cols = list(df.columns)
        for kw in keywords:
            kw_l = str(kw).lower()
            for c in cols:
                c_l = str(c).lower()
                if kw_l in c_l and not any(str(x).lower() in c_l for x in exclude):
                    return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=np.float32)

    def _normalize_station_preserve_nan(data, params):
        """Min-max normalize station observations while keeping NaNs."""
        arr = np.asarray(data, dtype=np.float32)
        dmin, dmax = params
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        if abs(float(dmax) - float(dmin)) < 1e-12:
            return out
        valid = np.isfinite(arr)
        out[valid] = (arr[valid] - dmin) / (dmax - dmin)
        return out

    out = {}
    for name, meta in STATIONS.items():
        fpath = os.path.join(station_root, meta["file"])
        if not os.path.isfile(fpath):
            print(f"[station] missing: {fpath} -> fill NaN")
            out[name] = np.full((len(time_index), 3), np.nan, dtype=np.float32)
            continue

        try:
            try:
                df = pd.read_csv(fpath, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(fpath, encoding="cp949")
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

            time_col = None
            for c in df.columns:
                c_l = str(c).lower()
                if ("관측시간" in str(c) or "시간" in str(c)
                        or c_l in ("time", "date", "datetime", "obs_time",
                                   "ymdhm", "ymdh", "yyyymmddhh", "timestamp")):
                    time_col = c
                    break
            if time_col is None:
                time_col = df.columns[0]

            # KHOA buoy times are KST. Convert to UTC before alignment.
            t = pd.to_datetime(df[time_col], errors="coerce")
            t = (t.dt.tz_localize("Asia/Seoul", nonexistent="shift_forward", ambiguous="NaT")
                   .dt.tz_convert("UTC")
                   .dt.tz_localize(None))
            df = (df.assign(__t__=t)
                    .dropna(subset=["__t__"])
                    .set_index("__t__")
                    .sort_index())
            df = df[~df.index.duplicated(keep="first")]

            hs_obs = _pick_column(df, ["유의파고", "hsig", "hs"], exclude=["주기", "period", "tm"])
            tm_obs = _pick_column(df, ["유의파주기", "파주기", "period", "tm"])
            dir_obs = _pick_column(df, ["파향", "dir", "wdir", "direction"], exclude=["주기", "period", "tm"])

            # KHOA commonly encodes missing Hs as 0.0. Keep it out of metrics.
            hs_obs = hs_obs.replace(0, np.nan)

            hs_aligned = hs_obs.reindex(time_index).to_numpy()
            tm_aligned = tm_obs.reindex(time_index).to_numpy()
            dir_aligned = dir_obs.reindex(time_index).to_numpy().astype(np.float32)

            n_hs = int(np.isfinite(hs_aligned).sum())
            n_tm = int(np.isfinite(tm_aligned).sum())
            n_dir = int(np.isfinite(dir_aligned).sum())
            print(f"[station] {name:16s} <- {meta['file']:32s} "
                  f"finite Hs={n_hs} Tm={n_tm} Dir={n_dir}")
            if n_hs == 0 and n_tm == 0 and n_dir == 0:
                print(f"[station]   WARNING: no overlapping finite data for {name}. "
                      f"Columns were: {list(df.columns)}")

            hs_norm = _normalize_station_preserve_nan(hs_aligned, norm_params["hs"])
            tm_norm = _normalize_station_preserve_nan(tm_aligned, norm_params["tm"])
            dir_deg = dir_aligned
            out[name] = np.stack([hs_norm, tm_norm, dir_deg], axis=1).astype(np.float32)

        except Exception as e:
            print(f"[station] read failed: {fpath} -> {type(e).__name__}: {e}")
            out[name] = np.full((len(time_index), 3), np.nan, dtype=np.float32)
    return out

def _scatter_ax(ax, x, y, title, xlabel, ylabel):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.scatter(x[m], y[m], s=6, alpha=0.35, edgecolors="none")
    lo = np.nanmin([x[m].min(), y[m].min()])
    hi = np.nanmax([x[m].max(), y[m].max()])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    # 간단 통계
    rmse = np.sqrt(np.mean((x[m]-y[m])**2))
    r = np.corrcoef(x[m], y[m])[0,1] if m.sum()>1 else np.nan
    ax.text(0.02, 0.98, f"RMSE={rmse:.3f}\nr={r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FONT_SIZES["legend"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))

def create_scatter_plots(station_results: dict, norm_params: dict, output_dir: str):
    """
    Emulator vs SWAN 산점도 (Hs, Tm) + Buoy vs SWAN 산점도 (참고용).
    """
    os.makedirs(output_dir, exist_ok=True)
    # 모으기
    hs_p, hs_t, hs_m = [], [], []
    tm_p, tm_t, tm_m = [], [], []
    for d in station_results.values():
        hs_p += d["hs"]["pred"]; hs_t += d["hs"]["true"]; hs_m += d["hs"]["meas"]
        tm_p += d["tm"]["pred"]; tm_t += d["tm"]["true"]; tm_m += d["tm"]["meas"]

    hs_p = denorm(np.asarray(hs_p, float), *norm_params["hs"])
    hs_t = denorm(np.asarray(hs_t, float), *norm_params["hs"])
    hs_m = denorm(np.asarray(hs_m, float), *norm_params["hs"])
    tm_p = denorm(np.asarray(tm_p, float), *norm_params["tm"])
    tm_t = denorm(np.asarray(tm_t, float), *norm_params["tm"])
    tm_m = denorm(np.asarray(tm_m, float), *norm_params["tm"])

    # Emulator vs SWAN
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1); _scatter_ax(ax1, hs_t, hs_p, "Hs: Emulator vs SWAN", "SWAN (m)", "Emulator (m)")
    ax2 = plt.subplot(1,2,2); _scatter_ax(ax2, tm_t, tm_p, "Tm: Emulator vs SWAN", "SWAN (s)", "Emulator (s)")
    plt.tight_layout()
    _savefig_vector(os.path.join(output_dir, "scatter_emulator_vs_swan.png"))

    # Buoy vs SWAN (참고)
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1); _scatter_ax(ax1, hs_t, hs_m, "Hs: Buoy vs SWAN", "SWAN (m)", "Buoy (m)")
    ax2 = plt.subplot(1,2,2); _scatter_ax(ax2, tm_t, tm_m, "Tm: Buoy vs SWAN", "SWAN (s)", "Buoy (s)")
    plt.tight_layout()
    _savefig_vector(os.path.join(output_dir, "scatter_buoy_vs_swan.png"))

def create_error_distribution_plots(
    station_results: dict,
    norm_params: dict,
    output_dir: str,
    *,
    center_on_median: bool = True,   # ← 중간값을 x축 중앙에 배치
    show_median_line: bool = True    # ← 중앙값 수직선 표시
):
    """
    |error| CDF (Hs, Tm) 와 Dir의 원형 절댓값 오차 CDF.
    옵션:
      - center_on_median=True  : 각 패널의 x축을 median을 중심으로 좌우 대칭 배치
      - show_median_line=True  : median 위치에 수직선 + 숫자 표시
    """
    os.makedirs(output_dir, exist_ok=True)

    # 수집
    hs_e, tm_e, dr_e = [], [], []
    for d in station_results.values():
        # Hs/Tm: 정규화→물리단위로 복원 후 |error|
        ph = denorm(np.asarray(d["hs"]["pred"], float), *norm_params["hs"])
        th = denorm(np.asarray(d["hs"]["true"], float), *norm_params["hs"])
        pt = denorm(np.asarray(d["tm"]["pred"], float), *norm_params["tm"])
        tt = denorm(np.asarray(d["tm"]["true"], float), *norm_params["tm"])
        hs_e.append(np.abs(ph - th))
        tm_e.append(np.abs(pt - tt))

        # Dir (원형 |오차|)
        pd = np.asarray(d["dir"]["pred"], float) % 360.0
        td = np.asarray(d["dir"]["true"], float) % 360.0
        diff = np.abs(((pd - td + 180.0) % 360.0) - 180.0)
        dr_e.append(diff)

    def _stack(a_list):
        return np.concatenate(a_list) if a_list else np.array([])

    hs_e = _stack(hs_e)
    tm_e = _stack(tm_e)
    dr_e = _stack(dr_e)

    def _cdf(ax, arr, label, xlabel, unit):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=FONT_SIZES["title"])
            ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
            ax.set_ylabel("CDF", fontsize=FONT_SIZES["label"])
            ax.tick_params(labelsize=FONT_SIZES["tick"])
            return

        x = np.sort(a)
        y = np.linspace(0, 1, len(x), endpoint=True)

        ax.plot(x, y, lw=2)
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=FONT_SIZES["title"])
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
        ax.set_ylabel("CDF", fontsize=FONT_SIZES["label"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        # 중앙값 라인 + x축 중앙 정렬(옵션)
        if show_median_line or center_on_median:
            med = float(np.nanmedian(a))
            if show_median_line:
                ax.axvline(med, ls="-.", lw=1.5, color="k", alpha=0.8)
                ax.text(med, 0.02, f" median = {med:.3f} {unit}",
                        rotation=90, va="bottom", ha="right",
                        fontsize=FONT_SIZES["legend"], color="k",
                        transform=ax.get_xaxis_transform())

            if center_on_median:
                xmin = float(np.nanmin(a))
                xmax = float(np.nanmax(a))
                # median을 정확히 중앙에 두도록 반경을 좌/우 중 큰 쪽으로 설정
                r = max(med - xmin, xmax - med)
                # 음수 x도 허용(데이터는 없지만 '시각적으로 중앙'을 맞추기 위함)
                ax.set_xlim(med - r, med + r)

    plt.figure(figsize=(12, 4))
    _cdf(plt.subplot(1, 3, 1), hs_e, "Hs |error|", "m", "m")
    _cdf(plt.subplot(1, 3, 2), tm_e, "Tm |error|", "s", "s")
    _cdf(plt.subplot(1, 3, 3), dr_e, "Dir |circular error|", "deg", "°")
    plt.tight_layout()
    _savefig_vector(os.path.join(output_dir, "error_cdf.png"))

def create_circular_scatter_plots(station_results: dict, norm_params: dict, output_dir: str, max_points: int = 20000):
    """
    Dir: true vs pred 산점도(도수 wrap). 점수가 많으면 서브샘플.
    """
    os.makedirs(output_dir, exist_ok=True)
    pd_all, td_all = [], []
    for d in station_results.values():
        pd_all += d["dir"]["pred"]
        td_all += d["dir"]["true"]
    pd_all = np.asarray(pd_all, float) % 360.0
    td_all = np.asarray(td_all, float) % 360.0
    m = np.isfinite(pd_all) & np.isfinite(td_all)
    pd_all, td_all = pd_all[m], td_all[m]
    if pd_all.size > max_points:
        idx = np.random.default_rng(0).choice(pd_all.size, size=max_points, replace=False)
        pd_all, td_all = pd_all[idx], td_all[idx]

    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.scatter(td_all, pd_all, s=6, alpha=0.35, edgecolors="none")
    ax.plot([0,360],[0,360],"k--",lw=1)
    ax.set_xlim(0,360); ax.set_ylim(0,360)
    ax.set_xlabel("SWAN Dir (deg)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Emulator Dir (deg)", fontsize=FONT_SIZES["label"])
    ax.set_title("Direction: Emulator vs SWAN", fontsize=FONT_SIZES["title"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    # 간단 통계(원형 RMSE)
    diff = ((pd_all - td_all + 180.0) % 360.0) - 180.0
    crmse = np.sqrt(np.mean(diff**2)) if diff.size else np.nan
    ax.text(0.02,0.98,f"cRMSE={crmse:.2f}°",transform=ax.transAxes,ha="left",va="top",
            fontsize=FONT_SIZES["legend"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))
    plt.tight_layout()
    _savefig_vector(os.path.join(output_dir, "dir_scatter.png"))
# =======================================================

def create_output_directory(pth_filename: str, tag: str = None) -> str:
    base = os.path.splitext(os.path.basename(pth_filename))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_" if tag else ""
    out = os.path.join("outputs", f"{prefix}{base}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out

def apply_bias_correction(station_results: dict, norm_params: dict) -> dict:
    out = {k: {vv: {kk: list(vvv) for kk, vvv in d[vv].items()} for vv in d} for k, d in station_results.items()}
    all_p, all_t = [], []
    for st in station_results.values():
        if 'dir' in st:
            p = np.asarray(st['dir'].get('pred', []), float)
            t = np.asarray(st['dir'].get('true', []), float)
            m = np.isfinite(p) & np.isfinite(t)
            all_p.extend(p[m]); all_t.extend(t[m])
    if len(all_p) and len(all_t):
        all_p = np.asarray(all_p); all_t = np.asarray(all_t)
        bias = float(np.mean(circular_diff_deg(all_p, all_t)))
    else:
        bias = 0.0
    for st in out.values():
        if 'dir' in st and 'pred' in st['dir']:
            pc = (np.asarray(st['dir']['pred'], float) - bias) % 360.0
            st['dir']['pred_corrected'] = pc.tolist()
    print(f"[DIR bias-correct] global bias = {bias:+.2f}°")
    return out

def create_enhanced_journal_figures(metric_results: dict, station_results: dict,
                                    norm_params: dict, output_dir: str,
                                    kcs_map=None, lon_map=None, lat_map=None):
    create_scatter_plots(station_results, norm_params, output_dir)
    create_error_distribution_plots(station_results, norm_params, output_dir)
    create_circular_scatter_plots(station_results, norm_params, output_dir)

def create_cdf_error_plots(station_results: dict, norm_params: dict, output_dir: str):
    # median을 중앙에 두고, 수직선도 표시
    return create_error_distribution_plots(
        station_results, norm_params, output_dir,
        center_on_median=True, show_median_line=True
    )

def compute_station_skill_tables(station_results: dict, norm_params: dict):
    """
    Buoy(meas) 기준 스킬 테이블
    - Hs & Tm: SWAN(true) vs Buoy, Emulator(pred) vs Buoy 둘 다 출력
    - Dir: (원형) SWAN(true) vs Buoy, Emulator(pred_corrected) vs Buoy 둘 다 출력
    반환: df_htm( Hs+Tm ), df_dir( Dir )
    """
    def _stats_linear(pred, obs):
        m = np.isfinite(pred) & np.isfinite(obs)
        if m.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        rmse = float(np.sqrt(np.mean((pred[m] - obs[m])**2)))
        mae  = float(np.mean(np.abs(pred[m] - obs[m])))
        bias = float(np.mean(pred[m] - obs[m]))
        r    = float(np.corrcoef(pred[m], obs[m])[0,1]) if m.sum() > 1 else np.nan
        return r, rmse, mae, bias

    def _stats_dir(pred_deg, obs_deg):
        m = np.isfinite(pred_deg) & np.isfinite(obs_deg)
        if m.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        rmse = float(circular_rmse_deg(pred_deg[m], obs_deg[m]))
        mae  = float(circular_mae_deg(pred_deg[m], obs_deg[m]))
        bias = float(np.mean(circular_diff_deg(pred_deg[m], obs_deg[m])))
        r    = float(circular_correlation(pred_deg[m], obs_deg[m]))
        return r, rmse, mae, bias

    rows_htm, rows_dir = [], []

    for name, d in station_results.items():
        # ---------- Hs ----------
        ph = denorm(np.asarray(d['hs']['pred'], float),  *norm_params['hs'])
        th = denorm(np.asarray(d['hs']['true'], float),  *norm_params['hs'])
        mh = denorm(np.asarray(d['hs']['meas'], float),  *norm_params['hs'])

        r_hs_swan, rmse_hs_swan, mae_hs_swan, bias_hs_swan = _stats_linear(th, mh)
        r_hs_emul, rmse_hs_emul, mae_hs_emul, bias_hs_emul = _stats_linear(ph, mh)

        # ---------- Tm ----------
        pt = denorm(np.asarray(d['tm']['pred'], float),  *norm_params['tm'])
        tt = denorm(np.asarray(d['tm']['true'], float),  *norm_params['tm'])
        mt = denorm(np.asarray(d['tm']['meas'], float),  *norm_params['tm'])

        r_tm_swan, rmse_tm_swan, mae_tm_swan, bias_tm_swan = _stats_linear(tt, mt)
        r_tm_emul, rmse_tm_emul, mae_tm_emul, bias_tm_emul = _stats_linear(pt, mt)

        rows_htm.append({
            'station': name,
            'Hs_r_SWAN': r_hs_swan,     'Hs_RMSE_SWAN': rmse_hs_swan,
            'Hs_MAE_SWAN': mae_hs_swan, 'Hs_Bias_SWAN': bias_hs_swan,
            'Hs_r_Emul': r_hs_emul,     'Hs_RMSE_Emul': rmse_hs_emul,
            'Hs_MAE_Emul': mae_hs_emul, 'Hs_Bias_Emul': bias_hs_emul,
            'Tm_r_SWAN': r_tm_swan,     'Tm_RMSE_SWAN': rmse_tm_swan,
            'Tm_MAE_SWAN': mae_tm_swan, 'Tm_Bias_SWAN': bias_tm_swan,
            'Tm_r_Emul': r_tm_emul,     'Tm_RMSE_Emul': rmse_tm_emul,
            'Tm_MAE_Emul': mae_tm_emul, 'Tm_Bias_Emul': bias_tm_emul,
        })

        # ---------- Dir (원형) ----------
        pd0 = np.asarray(d['dir'].get('pred_corrected', d['dir']['pred']), float) % 360.0
        td0 = np.asarray(d['dir']['true'], float) % 360.0
        md0 = np.asarray(d['dir']['meas'], float) % 360.0

        r_dir_swan, rmse_dir_swan, mae_dir_swan, bias_dir_swan = _stats_dir(td0, md0)
        r_dir_emul, rmse_dir_emul, mae_dir_emul, bias_dir_emul = _stats_dir(pd0, md0)

        rows_dir.append({
            'station': name,
            'cR_SWAN': r_dir_swan,     'cRMSE_SWAN': rmse_dir_swan,
            'cMAE_SWAN': mae_dir_swan, 'cBias_SWAN': bias_dir_swan,
            'cR_Emul': r_dir_emul,     'cRMSE_Emul': rmse_dir_emul,
            'cMAE_Emul': mae_dir_emul, 'cBias_Emul': bias_dir_emul,
        })

    df_htm = pd.DataFrame(rows_htm).set_index('station')
    df_dir = pd.DataFrame(rows_dir).set_index('station')
    return df_htm, df_dir

def save_skill_tables_csv_and_latex(df_htm, df_dir, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv1 = os.path.join(output_dir, 'buoy_skill_htm.csv')
    csv2 = os.path.join(output_dir, 'buoy_skill_dir.csv')
    tex1 = os.path.join(output_dir, 'station_metrics_htm.tex')
    tex2 = os.path.join(output_dir, 'station_metrics_dir.tex')
    df_htm.to_csv(csv1); df_dir.to_csv(csv2)
    with open(tex1, 'w', encoding='utf-8') as f:
        f.write(df_htm.to_latex(float_format="%.3f"))
    with open(tex2, 'w', encoding='utf-8') as f:
        f.write(df_dir.to_latex(float_format="%.3f"))
    print(f"[TABLES] saved: {csv1}, {csv2}, {tex1}, {tex2}")

# ---------- ★ FIX-MAPE helpers ----------
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
    """
    Robust MAPE [%]: mean( |p - t| / |t| ) * 100
    - excludes samples with |t| < thresh to avoid division explosion near zeros
    - eps only prevents literal division-by-zero; 'thresh' controls stability.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = (np.abs(true) >= thresh) & np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return np.nan
    return (np.abs(pred[mask]-true[mask])/(np.abs(true[mask])+eps)).mean()*100.0

def _safe_smape(pred, true, eps=1e-6, thresh=0.25):
    """
    Symmetric MAPE [%] with stability threshold on (|p|+|t|).
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    denom = np.abs(pred) + np.abs(true)
    mask = np.isfinite(pred) & np.isfinite(true) & (denom >= thresh)
    if mask.sum() == 0:
        return np.nan
    return (np.abs(pred[mask] - true[mask]) / (denom[mask] + eps)).mean() * 100.0

# ---------- ★ NEW: overall Emulator-vs-SWAN table (CSV + LaTeX) ----------
def _nanmean(lst):
    arr = np.asarray(lst, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan

def build_overall_metrics_dataframe(metric_results: dict) -> pd.DataFrame:
    """
    Build the manuscript-style aggregated table using metric_results lists.
    Dir has no MAPE/sMAPE by design (NaN).
    """
    hs = {
        "RMSE":       _nanmean(metric_results.get("rmse_hs", [])),
        "MAE":        _nanmean(metric_results.get("mae_hs", [])),
        "Mean Bias":  _nanmean(metric_results.get("bias_hs", [])),
        "Pearson r":  _nanmean(metric_results.get("cc_hs", [])),
        "R^2":        _nanmean(metric_results.get("r2_hs", [])),
        "MAPE (%)":   _nanmean(metric_results.get("mape_hs", [])),
        "sMAPE (%)":  _nanmean(metric_results.get("smape_hs", [])),
        "Area-weighted mean": _nanmean(metric_results.get("true_wmean", [])),
    }
    tm = {
        "RMSE":       _nanmean(metric_results.get("rmse_tm", [])),
        "MAE":        _nanmean(metric_results.get("mae_tm", [])),
        "Mean Bias":  _nanmean(metric_results.get("bias_tm", [])),
        "Pearson r":  _nanmean(metric_results.get("cc_tm", [])),
        "R^2":        _nanmean(metric_results.get("r2_tm", [])),
        "MAPE (%)":   _nanmean(metric_results.get("mape_tm", [])),
        "sMAPE (%)":  _nanmean(metric_results.get("smape_tm", [])),
        "Area-weighted mean": np.nan,  # 표 규칙상 Hs만 표기
    }
    dr = {
        "RMSE":       _nanmean(metric_results.get("rmse_dir", [])),
        "MAE":        _nanmean(metric_results.get("mae_dir", [])),
        "Mean Bias":  _nanmean(metric_results.get("bias_dir", [])),
        "Pearson r":  _nanmean(metric_results.get("cc_dir", [])),
        "R^2":        _nanmean(metric_results.get("r2_dir", [])),
        "MAPE (%)":   np.nan,
        "sMAPE (%)":  np.nan,
        "Area-weighted mean": np.nan,
    }
    df = pd.DataFrame(
        {"Hs (m)": hs, "Tm (s)": tm, "Dir (deg)": dr}
    )
    return df

def save_overall_metrics_csv_and_latex(metric_results: dict, output_dir: str):
    """
    Writes 'overall_metrics_vs_swan.csv' and a LaTeX table 'aggregated_metrics.tex'.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = build_overall_metrics_dataframe(metric_results)
    csv_path = os.path.join(output_dir, "overall_metrics_vs_swan.csv")
    tex_path = os.path.join(output_dir, "aggregated_metrics.tex")
    df.to_csv(csv_path, float_format="%.6f")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(float_format="%.3f"))
    print(f"[OVERALL] saved: {csv_path}, {tex_path}")

# ----------------------------
# Station meta (영문 이름 유지)
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
STATION_EN = {"대한해협":"Korea Strait","제주해협":"Jeju Strait","남해동부":"South Sea East","대천해수욕장":"Daecheon Beach",
              "해운대해수욕장":"Haeundae Beach","임랑해수욕장":"Imrang Beach","중문해수욕장":"Jungmun Beach",
              "생일도":"Saengil Island","상왕등도":"Sangwangdeungdo"}

# ----------------------------
# Typhoon windows (optional)
# ----------------------------
TYPHOONS = {
    "lingling": {"start": pd.Timestamp("2019-09-02 00:00:00"), "end": pd.Timestamp("2019-09-08 00:00:00"), "prefix":"lingling"},
    "bavi":     {"start": pd.Timestamp("2020-08-22 00:00:00"), "end": pd.Timestamp("2020-08-27 00:00:00"), "prefix":"bavi"},
    "maysak":   {"start": pd.Timestamp("2020-08-28 00:00:00"), "end": pd.Timestamp("2020-09-03 00:00:00"), "prefix":"maysak"},
    "haishen":  {"start": pd.Timestamp("2020-09-01 00:00:00"), "end": pd.Timestamp("2020-09-07 00:00:00"), "prefix":"haishen"},
}

# =========================================================
# Small utils for BND & direction transforms
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

def _auto_align_bnd_dir(bnd_feat, ds_sim, kcs2d, time_index):
    """경계 파향(sin, cos) 축을 타깃(sim dir)에 맞춰 0/±90/180 중 최적 회전 선택."""
    T = bnd_feat.shape[0]
    if 'dir' not in ds_sim:
        return 0.0, {}
    rad = np.deg2rad(ds_sim['dir'].values[:T])
    tsin = np.sin(rad).astype(np.float32)
    tcos = np.cos(rad).astype(np.float32)

    sin_idx, cos_idx = 2, 3
    for try_s, try_c in [(2,3), (3,2)]:
        if (np.nanmin(bnd_feat[:, try_s]) < -0.1) and (np.nanmin(bnd_feat[:, try_c]) < -0.1):
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

    if abs(best_deg) > 1e-6:
        r = np.deg2rad(best_deg)
        sin_r = sin_b*np.cos(r) + cos_b*np.sin(r)
        cos_r = cos_b*np.cos(r) - sin_b*np.sin(r)
        bnd_feat[:, sin_idx] = sin_r
        bnd_feat[:, cos_idx] = cos_r
    return best_deg, scores

def _apply_dir_transform(arr_deg: np.ndarray, *, flip_sign=False, add_deg=0.0, swap_from_toward=False):
    """
    arr_deg: [deg], '북-시계(Bearing, 0°=North, CW+)'로 들어온다고 가정한 각도에 대해
    - flip_sign=True  → 반시계로 뒤집기( a → -a )
    - add_deg=...,    → ±90 / ±180 보정
    - swap_from_toward=True → from↔toward 전환 (+180)
    """
    a = np.asarray(arr_deg, dtype=float)
    if swap_from_toward:
        a = (a + 180.0) % 360.0
    if flip_sign:
        a = (-a)
    a = (a + add_deg) % 360.0
    return a

def decide_and_apply_best_buoy_dir_transform(station_ts_dict: dict, *, save_dir: str = None):
    """
    부이 Dir을 전역적으로 하나의 변환(set)으로 보정해 'meas_best'에 저장
    """
    meas_all, true_all = [], []
    for d in station_ts_dict.values():
        if 'dir' not in d:
            continue
        m = np.asarray(d['dir'].get('meas', []), float)
        t = np.asarray(d['dir'].get('true', []), float) % 360.0
        ok = np.isfinite(m) & np.isfinite(t)
        if ok.any():
            meas_all.append(m[ok]); true_all.append(t[ok])
    if not meas_all:
        return None, []

    meas_all = np.concatenate(meas_all)
    true_all = np.concatenate(true_all)

    candidates = []
    for add in (0.0, 90.0, -90.0, 180.0):
        for flip in (False, True):
            for swap in (False, True):
                key = f"add={add:+.0f}, flip={'Y' if flip else 'N'}, swap_from_toward={'Y' if swap else 'N'}"
                candidates.append((key, dict(add_deg=add, flip_sign=flip, swap_from_toward=swap)))

    results = []
    for key, opts in candidates:
        meas_hat = _apply_dir_transform(meas_all, **opts)
        rmse = circular_rmse_deg(meas_hat, true_all)
        results.append({'key': key, 'rmse': rmse})

    results_sorted = sorted(results, key=lambda x: x['rmse'])
    best = results_sorted[0]; best_key = best['key']
    best_opts = dict([c for c in candidates if c[0] == best_key][0][1])

    for d in station_ts_dict.values():
        if 'dir' not in d:
            continue
        m = np.asarray(d['dir'].get('meas', []), float)
        with np.errstate(invalid='ignore'):
            d['dir']['meas_best'] = _apply_dir_transform(m, **best_opts).tolist()
            d['dir']['meas_transform_key'] = best_key

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(results_sorted).to_csv(os.path.join(save_dir, "buoy_dir_transform_metrics.csv"), index=False)
        with open(os.path.join(save_dir, "buoy_dir_transform_choice.txt"), "w", encoding="utf-8") as f:
            f.write(f"BEST: {best_key}  (global cRMSE={best['rmse']:.3f} deg)\n")
    return best_key, results_sorted

# 연도별 BND 폴더 매핑 (필요시 본인 경로에 맞게 수정)
BND_DIRS_BY_YEAR = {
    2019: r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2019",
    2020: r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET\SWAN_BND_FILES\bnd_2020",
}
BND_DIRECTION = "from"  # or "toward"

def build_bnd_features(ds_sim, kcs, time_index, global_norm_params, idx_tr, seq_length):
    """
    Build BND feature maps with channels [Hs_norm, Tm_norm, sin_dir, cos_dir].
    Directional alignment is selected using training target indices only and
    is then applied to the full period without validation or test leakage.
    """
    try:
        from bnd_features import (
            read_all_bnds, build_owner_label, make_boundary_feature_maps, assert_on_edges
        )
        from boundspec_segments import SEGMENTS, M as SWAN_M, N as SWAN_N
    except Exception as e:
        raise RuntimeError(f"[BND] Failed to import BND modules: {type(e).__name__}: {e}")

    kcs2d = kcs[0] if kcs.ndim == 3 else kcs
    H, W = kcs2d.shape

    if (H == SWAN_M and W == SWAN_N):
        swap_ij = False
    elif (H == SWAN_N and W == SWAN_M):
        swap_ij = True
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
            print(f"[BND] {y}: loaded {bdir}")
        else:
            raise RuntimeError(f"[BND] Missing BND folder for {y}: {bdir}")

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
    )

    try:
        train_target_idx = np.asarray(idx_tr, dtype=np.int64) + int(seq_length)
        best_deg, scores = auto_align_bnd_dir_trainonly(
            bnd_feat, ds_sim, kcs2d, train_target_idx
        )
        msg = " ".join([f"{k:+.0f}deg:{v:.4f}" for k, v in sorted(scores.items())])
        print(f"[BND] train-only dir autocorrect -> chosen {best_deg:+.0f}deg | scores {msg}")
    except Exception as e:
        print(f"[BND] train-only dir autocorrect skipped: {type(e).__name__}: {e}")

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

# === Load & preprocess (dense arrays) ===
def load_and_preprocess_data(ds, global_norm_params, time_steps=100):
    """
    NetCDF → (input_data, wave_data, lon, lat, kcs)
    input_data: (T, 6, Y, X)  = wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad
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

    # depth gradient (정적 1채널)
    depth2d = ds['depth'].values if ds['depth'].values.ndim == 2 else ds['depth'].values[0]
    depth_grad = _depth_grad_mag(depth2d)  # (Y,X)

    H, W = hs.shape[-2], hs.shape[-1]
    depth_grad_3d = np.broadcast_to(depth_grad[None, ...], (T, H, W)).astype(_DTYPE)

    input_data = np.stack(
        [wind_u, wind_v, depth, veloc_x, veloc_y, depth_grad_3d], axis=1
    )
    wave_data = np.stack([hs, tm, dsin, dcos], axis=1)
    return input_data, wave_data, lon, lat, kcs

# =========================================================
# Models  --  imported from the clean, side-effect-free module
# =========================================================
# The model class definitions used to be inlined here, and they had
# drifted from the training code (missing log_vars, missing final_drop,
# a different feat default). They are now imported from
# model_architectures.py, which holds the EXACT training-time class
# definitions plus the ablation variants and a build_model factory.
# Importing that module runs no argparse, no CUDA setup, no training.
from model_architectures import (
    SEBlock, ImprovedConvBlock, ConvLSTMCell, UNetPlusPlus,
    UNetConvLSTM, UNetConvLSTM_full, ConvLSTM_only, UNetPP_stack,
    UNetPP_only, build_model, load_checkpoint_strict_except,
)
from revision_patches import (
    auto_align_bnd_dir_trainonly,
    compute_test_metrics_all,
    make_chronological_split,
)


# =========================================================
# Data classes & loaders
# =========================================================
class WindWaveDataset(Dataset):
    def __init__(self, input_data, wave_data, seq_length, start_idx, end_idx):
        self.input_data = input_data; self.wave_data = wave_data
        self.seq_length = seq_length; self.start_idx = start_idx; self.end_idx = end_idx
    def __len__(self): return self.end_idx - self.start_idx
    def __getitem__(self, idx):
        i = self.start_idx + idx
        seq_X = self.input_data[i:i + self.seq_length]; seq_y = self.wave_data[i + self.seq_length]
        return torch.from_numpy(seq_X).float(), torch.from_numpy(seq_y).float()

def collate_fn(batch):
    X = torch.stack([b[0] for b in batch]); Y = torch.stack([b[1] for b in batch]); return X, Y

class SubsetIndicesDataset(Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        return self.base[self.indices[i]]

def find_nearest_index(lon_map, lat_map, kcs_map, target_lon, target_lat):
    kcs2d = kcs_map if kcs_map.ndim == 2 else kcs_map[0]
    valid = np.where((kcs2d == 1) | (kcs2d > 0))
    if valid[0].size == 0: return 0, 0
    lons = lon_map[valid]; lats = lat_map[valid]
    d = np.sqrt((lons - target_lon)**2 + (lats - target_lat)**2); j = np.argmin(d)
    return valid[0][j], valid[1][j]

# =========================================================
# Buoy direction convention for station validation
# =========================================================
# KHOA wave direction is recorded as the direction from which waves
# approach. In this inference script, the buoy direction is used as
# recorded after the timestamp correction from KST to UTC.
#
# No fixed angular rotation is applied here. Earlier compass-to-Cartesian
# conversion code must remain removed because it rotates the buoy record
# away from the SWAN/emulator direction used in the station-validation
# table.
#
# The automatic best-transform search is disabled so that no rotation is
# fitted to the buoy observations.
BUOY_DIR_DEFINITION = "as_is"
MODEL_DIR_DEFINITION = "as_is"
ENABLE_BUOY_DIR_AUTO_TRANSFORM = False


def _convert_dir_deg(angle_deg, src="as_is", dst="as_is"):
    """Convert direction only when an explicit from/toward conversion is requested."""
    ang = np.asarray(angle_deg, dtype=float)

    if src == dst or src == "as_is" or dst == "as_is":
        return np.mod(ang, 360.0)

    if (src, dst) in (("from", "toward"), ("toward", "from")):
        return (ang + 180.0) % 360.0

    return np.mod(ang, 360.0)


def adjust_buoy_direction_convention_inplace(station_data: dict, src="as_is", dst="as_is"):
    """
    Apply a fixed direction-convention conversion to station_data in place.

    For the final station-validation run, src=dst='as_is', so this function
    is intentionally a no-op. KHOA direction is used as recorded after the
    KST-to-UTC timestamp alignment.
    """
    if src == dst or src == "as_is" or dst == "as_is":
        print(
            f"[Buoy Dir] No angular rotation applied "
            f"(src={src}, dst={dst}). Buoy directions are used as recorded "
            f"after KST-to-UTC timestamp alignment."
        )
        return station_data

    for k, arr in station_data.items():
        if arr is None or not isinstance(arr, np.ndarray) or arr.shape[-1] < 3:
            continue

        dir_col = arr[:, 2].astype(float)

        with np.errstate(invalid="ignore"):
            dir_col = _convert_dir_deg(dir_col, src=src, dst=dst)

        arr[:, 2] = dir_col
        station_data[k] = arr

    print(f"[Buoy Dir] Converted buoy direction: {src} -> {dst}")
    return station_data

# =========================================================
# [NEW] NaN-safe helpers for map drawing
# =========================================================
def _as_plain_array(a):
    a = np.asanyarray(a)
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    return a

def _fill_invalid_with_nearest(a):
    a = _as_plain_array(a)
    if a.ndim != 2:
        return a
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return a
    if np.all(mask):
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = a.copy()
    filled[mask] = a[tuple(ind[mask] for ind in idx)]
    return filled

FONT_SIZES = {"title": 28, "label": 22, "tick": 20, "legend": 20}

# =========================================================
# [NEW] Cartopy-friendly plotting helpers
# =========================================================
def _extent_from_lonlat(lon_map, lat_map):
    lon_f = _as_plain_array(lon_map)
    lat_f = _as_plain_array(lat_map)
    lon_f = lon_f[np.isfinite(lon_f)]
    lat_f = lat_f[np.isfinite(lat_f)]
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
                           facecolor='0.92', edgecolor='none', zorder=5)
            ax.coastlines(resolution='10m', linewidth=0.4, zorder=6)
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

    lon2 = _as_plain_array(lon_map)
    lat2 = _as_plain_array(lat_map)
    is_curvi = (lon2.ndim == 2) or (lat2.ndim == 2)

    if is_curvi:
        use_pcolor = False
        if np.isfinite(lon2).all() and np.isfinite(lat2).all():
            use_pcolor = True
        else:
            if np.isfinite(lon2).any() and np.isfinite(lat2).any():
                lon2 = _fill_invalid_with_nearest(lon2)
                lat2 = _fill_invalid_with_nearest(lat2)
                use_pcolor = np.isfinite(lon2).all() and np.isfinite(lat2).all()

        if use_pcolor:
            kw = dict(shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
            if _HAVE_CARTOPY:
                im = ax.pcolormesh(lon2, lat2, d, transform=ccrs.PlateCarree(), **kw)
            else:
                im = ax.pcolormesh(lon2, lat2, d, **kw)
            if title: ax.set_title(title, fontsize=FONT_SIZES["title"])
            return im

    extent = _extent_from_lonlat(lon2, lat2)
    kw = dict(extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='bilinear', zorder=2)
    if _HAVE_CARTOPY:
        im = ax.imshow(d, transform=ccrs.PlateCarree(), **kw)
    else:
        im = ax.imshow(d, **kw)
    if title: ax.set_title(title, fontsize=FONT_SIZES["title"])
    return im

# =========================================================
# Plot helpers (UPDATED)
# =========================================================
def _plot_spatial_sample(pred, true, lon_map, lat_map, kcs, fname, var_name="Hs", title_suffix="", norm_params=None):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if norm_params is not None:
        vmin, vmax = norm_params; pred = denorm(pred, vmin, vmax); true = denorm(true, vmin, vmax)

    kcs2d = kcs if kcs.ndim == 2 else kcs[0]
    valid = ((kcs2d==1) | (kcs2d>0)) & np.isfinite(true)
    vmin = np.nanmin(true[valid]) if valid.any() else 0.0
    vmax = np.nanmax(true[valid]) if valid.any() else 1.0

    fig, axes = _new_map_axes(ncols=3, nrows=1, figsize=(18,5), lon_map=lon_map, lat_map=lat_map)
    items=[("Pred",pred,"jet",vmin,vmax),("True",true,"jet",vmin,vmax),("|Err|",np.abs(pred-true),"coolwarm",0, None)]
    for ax,(ttl,dat,cmap,lo,hi) in zip(axes, items):
        if hi is None:
            finite = np.asarray(dat)[np.isfinite(dat)]
            hi = float(np.nanmax(finite)) if finite.size else 1.0
        im = _draw_field_on_ax(ax, dat, lon_map, lat_map, kcs, cmap=cmap, vmin=lo, vmax=hi, title=ttl)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("lon", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("lat", fontsize=FONT_SIZES["label"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])

    key = str(var_name).strip().lower()
    var_label = {"hs": r"$H_s$", "tm": r"$T_m$", "dir": "Dir"}.get(key, var_name)
    unit_label = {"hs": "(m)", "tm": "(s)", "dir": r"($^\circ$)"}.get(key, "")
    plt.suptitle(f"Spatial sample for {var_label} {unit_label}", fontsize=FONT_SIZES["title"])
    plt.tight_layout()
    _savefig_vector(fname)

def _plot_spatial_rmse_maps(rmse_hs, rmse_tm, rmse_dir, lon_map, lat_map, kcs_map, fname_base, cmaps=None):
    if cmaps is None:
        cmaps = ["Greys", "cividis", "magma", "plasma"]
    os.makedirs(os.path.dirname(fname_base), exist_ok=True)

    for cmap in cmaps:
        fig, axes = _new_map_axes(ncols=3, figsize=(18,5), lon_map=lon_map, lat_map=lat_map)
        items = [
            ("RMSE (m)", rmse_hs, cmap),
            ("RMSE (s)", rmse_tm, cmap),
            ("cRMSE (°)", rmse_dir, cmap),
        ]
        for ax, (cbar_label, dat, cmap_one) in zip(axes, items):
            vmax = np.nanpercentile(dat, 99) if np.isfinite(dat).any() else 1.0
            im = _draw_field_on_ax(ax, dat, lon_map, lat_map, kcs_map,
                                   cmap=cmap_one, vmin=0, vmax=vmax, title=None)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=FONT_SIZES["label"])
            cbar.ax.tick_params(labelsize=FONT_SIZES["tick"])
            ax.set_xlabel('Longitude', fontsize=FONT_SIZES["label"])
            ax.set_ylabel('Latitude',  fontsize=FONT_SIZES["label"])
            ax.tick_params(labelsize=FONT_SIZES["tick"])
        plt.tight_layout()
        out_name = os.path.splitext(fname_base)[0] + f"_{cmap}.png"
        _savefig_vector(out_name)

def _plot_timeseries_split_years(ts_dict: dict, station_name: str, var_name: str, fname_base: str,
                                 norm_params=(0.0,1.0), date_index=None, first_hour=0, seq_length=6):
    os.makedirs(os.path.dirname(fname_base), exist_ok=True)
    if var_name in ('hs','tm'):
        vmin,vmax=norm_params
        true_denorm=[denorm(x,vmin,vmax) for x in ts_dict["true"]]
        pred_denorm=[denorm(x,vmin,vmax) for x in ts_dict["pred"]]
        meas_phys=[denorm(x,vmin,vmax) if np.isfinite(x) else np.nan for x in ts_dict["meas"]]
    else:
        true_denorm=ts_dict["true"]
        pred_denorm=ts_dict.get("pred_corrected", ts_dict["pred"])
        meas_source = ts_dict.get("meas", [])
        meas_phys   = meas_source

    base_dates = date_index[first_hour:first_hour+len(true_denorm)] if date_index is not None else np.arange(len(true_denorm))
    meas_aligned = meas_phys[:len(true_denorm)]
    eng_name = STATION_EN.get(station_name, station_name)

    years = [(2019, '2019-01-01', '2019-12-31'), (2020, '2020-01-01', '2020-12-31')]
    for y, start_str, end_str in years:
        if date_index is None:
            break
        start = pd.Timestamp(start_str); end = pd.Timestamp(end_str) + pd.Timedelta(days=1)
        mask = (base_dates >= start) & (base_dates < end)
        if not np.any(mask):
            continue

        dates_y = base_dates[mask]
        true_y  = np.asarray(true_denorm)[mask]
        pred_y  = np.asarray(pred_denorm)[mask]
        meas_y  = np.asarray(meas_aligned)[mask]

        plt.figure(figsize=(12,5))
        plt.plot(dates_y, true_y, "-", color="k", label="SWAN (hindcast)", linewidth=1.8)
        plt.plot(dates_y, pred_y, "--", color="#1f77b4", label="Emulator", linewidth=1.8)
        plt.plot(dates_y, meas_y, ":", color="#d62728", label="Buoy (obs.)", linewidth=1.8)

        if var_name=='hs':
            ylabel = r'$H_s$ (m)'
        elif var_name=='tm':
            ylabel = r'$T_m$ (s)'
        else:
            ylabel = 'Dir (°)'
        plt.title(f"{eng_name} – {var_name.upper()} (time series {y})", fontsize=FONT_SIZES["title"])
        plt.xlabel("Date/Time (UTC)", fontsize=FONT_SIZES["label"])
        plt.ylabel(ylabel, fontsize=FONT_SIZES["label"])
        plt.grid(True, alpha=0.3)

        ax = plt.gca()
        y_stack = np.concatenate([
            np.asarray(true_y, dtype=float),
            np.asarray(pred_y, dtype=float),
            np.asarray(meas_y, dtype=float)
        ])
        y_stack = y_stack[np.isfinite(y_stack)]
        if y_stack.size:
            ymin, ymax = float(np.nanmin(y_stack)), float(np.nanmax(y_stack))
            yrange = ymax - ymin if (ymax - ymin) > 0 else max(abs(ymax), 1.0)
            pad = yrange * 0.15
            ax.set_ylim(ymin, ymax + pad)

        leg = ax.legend(loc='upper right', frameon=True)
        for txt in leg.get_texts():
            txt.set_fontsize(FONT_SIZES["legend"])

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        plt.tight_layout()
        fname = os.path.splitext(fname_base)[0] + f"_{y}.png"
        _savefig_vector(fname)

# =========================================================
# Evaluation & figure bundle (+ Buoy Validation tables)
#  ★ MOD in this block:
#    - use _safe_mape/_safe_smape with thresholds (MAPE_THRESH) → stable MAPE
#    - save_overall_metrics_csv_and_latex(...) to write Emulator-vs-SWAN CSV/TeX
# =========================================================
def evaluate_and_visualize(
    model, loader, lon_map, lat_map, kcs_map, station_meta, station_data, test_start_idx,
    seq_length, global_norm_params, out_prefix="eval", save_limit=5,
    time_steps_info="N/A", seq_length_info="N/A", epochs_info="N/A", alpha="N/A",
    sim_start=None, pth_filename="model_weights_default.pth", date_index=None,
    window_start_utc=None, window_end_utc=None, window_prefix="",
    base_indices=None,
    run_tag=None):
    import os, time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- 내부: Dir 샘플 지도 (0–360 고정) ------------------------------------
    def _plot_dir_sample(pdir_deg, tdir_deg, lon, lat, kcs, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        pdir = np.mod(pdir_deg, 360.0)
        tdir = np.mod(tdir_deg, 360.0)
        cerr = np.abs(((pdir - tdir + 180.0) % 360.0) - 180.0)  # [0,180]

        fig, axes = _new_map_axes(ncols=3, nrows=1, figsize=(18,5), lon_map=lon, lat_map=lat)
        im0 = _draw_field_on_ax(axes[0], pdir, lon, lat, kcs, cmap='hsv', vmin=0.0, vmax=360.0, title="Pred Dir (deg)")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        im1 = _draw_field_on_ax(axes[1], tdir, lon, lat, kcs, cmap='hsv', vmin=0.0, vmax=360.0, title="True Dir (deg)")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)
        im2 = _draw_field_on_ax(axes[2], cerr, lon, lat, kcs, cmap='magma', vmin=0.0, vmax=180.0, title="|Circular err| (deg)")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        for ax in axes:
            ax.set_xlabel("lon", fontsize=FONT_SIZES["label"])
            ax.set_ylabel("lat", fontsize=FONT_SIZES["label"])
            ax.tick_params(labelsize=FONT_SIZES["tick"])

        plt.tight_layout()
        _savefig_vector(fname)

    # --- 출력 폴더 ---------------------------------------------------------
    out_dir = create_output_directory(pth_filename, tag=run_tag)
    print(f"[eval] outputs → {out_dir}")

    # --- 공간 가중치 (cos(lat)) -------------------------------------------
    lat2d = lat_map if lat_map.ndim == 2 else lat_map[0]
    kcs2d = kcs_map if kcs_map.ndim == 2 else kcs_map[0]
    cos_lat = np.cos(np.deg2rad(lat2d)).astype(np.float32)
    cos_lat[(kcs2d <= 0) | ~np.isfinite(cos_lat)] = 0.0
    spatial_w = cos_lat / (cos_lat.sum() + 1e-12)

    # --- 스테이션 격자 인덱스 ---------------------------------------------
    st_indices = {}
    for name, meta in station_meta.items():
        i, j = find_nearest_index(lon_map, lat_map, kcs_map, meta["lon"], meta["lat"])
        st_indices[name] = (i, j)

    # --- 메트릭 누적 컨테이너 ---------------------------------------------
    metric_results = {k: [] for k in [
        "rmse_hs", "mae_hs", "bias_hs", "cc_hs", "r2_hs", "acc_hs", "mape_hs", "smape_hs",
        "rmse_tm", "mae_tm", "bias_tm", "cc_tm", "r2_tm", "mape_tm", "smape_tm",
        "rmse_dir", "mae_dir", "bias_dir", "cc_dir", "r2_dir",
        "pred_wmean", "true_wmean", "pred_wmean_tm", "true_wmean_tm"
    ]}
    variables = ['hs', 'tm', 'dir']
    st_ts = {n: {v: {"pred": [], "true": [], "meas": []} for v in variables} for n in station_meta}

    hs_min, hs_max = global_norm_params['hs']
    tm_min, tm_max = global_norm_params['tm']

    # --- Spatial RMSE 누적 버퍼 -------------------------------------------
    H, W = (kcs2d).shape
    hs_se = np.zeros((H, W), dtype=np.float64); hs_n = np.zeros((H, W), dtype=np.int32)
    tm_se = np.zeros((H, W), dtype=np.float64); tm_n = np.zeros((H, W), dtype=np.int32)
    dir_se = np.zeros((H, W), dtype=np.float64); dir_n = np.zeros((H, W), dtype=np.int32)

    # --- 평가 루프 ---------------------------------------------------------
    model.eval()
    save_cnt = 0
    seen = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for bidx, (x, y) in enumerate(loader):
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)

            outs = model(x)           # [main, aux...]
            pred = outs[0].float()    # (B,4,H,W)

            B = pred.size(0)
            for b in range(B):
                # 시간 인덱스 복원
                if base_indices is not None:
                    test_t = int(base_indices[seen + b]) + int(seq_length)
                else:
                    test_t = (bidx * loader.batch_size + b) + int(test_start_idx or 0) + int(seq_length)

                # 예측/정답 (정규화 → 물리량 복원)
                phs = pred[b, 0].detach().cpu().numpy()
                ptm = pred[b, 1].detach().cpu().numpy()
                psin = pred[b, 2].detach().cpu().numpy()
                pcos = pred[b, 3].detach().cpu().numpy()
                pdir = (np.rad2deg(np.arctan2(psin, pcos)) + 360.0) % 360.0

                ths = y[b, 0].detach().cpu().numpy()
                ttm = y[b, 1].detach().cpu().numpy()
                tsin = y[b, 2].detach().cpu().numpy()
                tcos = y[b, 3].detach().cpu().numpy()
                tdir = (np.rad2deg(np.arctan2(tsin, tcos)) + 360.0) % 360.0

                phs_m = denorm(phs, hs_min, hs_max)
                ths_m = denorm(ths, hs_min, hs_max)
                ptm_s = denorm(ptm, tm_min, tm_max)
                ttm_s = denorm(ttm, tm_min, tm_max)

                # --- 공간 메트릭(스칼라, 면적가중 포함)
                oce_hs  = (kcs2d > 0) & np.isfinite(ths_m) & np.isfinite(phs_m)
                oce_tm  = (kcs2d > 0) & np.isfinite(ttm_s) & np.isfinite(ptm_s)
                oce_dir = (kcs2d > 0) & np.isfinite(tdir)  & np.isfinite(pdir)

                if np.any(oce_hs):
                    w = spatial_w[oce_hs]
                    rmse_hs = float(np.sqrt(np.mean((phs_m[oce_hs] - ths_m[oce_hs])**2)))
                    mae_hs  = float(np.mean(np.abs (phs_m[oce_hs] - ths_m[oce_hs])))
                    bias_hs = float(np.mean(        phs_m[oce_hs] - ths_m[oce_hs]))
                    cc_hs   = float(np.corrcoef(phs_m[oce_hs], ths_m[oce_hs])[0, 1]) if np.sum(oce_hs) > 1 else np.nan
                    r2_hs   = float(1 - np.sum((ths_m[oce_hs] - phs_m[oce_hs])**2) /
                                       (np.sum((ths_m[oce_hs] - np.mean(ths_m[oce_hs]))**2) + 1e-12))
                    acc_hs  = float(np.mean(np.abs(phs_m[oce_hs] - ths_m[oce_hs]) <= 0.25))
                    # ★ FIX: robust MAPE/sMAPE
                    mape_hs  = float(_safe_mape(phs_m[oce_hs], ths_m[oce_hs], thresh=MAPE_THRESH['hs']))
                    smape_hs = float(_safe_smape(phs_m[oce_hs], ths_m[oce_hs], thresh=MAPE_THRESH['hs']))
                    pred_wmean = float(np.sum(phs_m[oce_hs] * w))
                    true_wmean = float(np.sum(ths_m[oce_hs] * w))
                    for k, v in dict(
                        rmse_hs=rmse_hs, mae_hs=mae_hs, bias_hs=bias_hs, cc_hs=cc_hs, r2_hs=r2_hs,
                        acc_hs=acc_hs, mape_hs=mape_hs, smape_hs=smape_hs,
                        pred_wmean=pred_wmean, true_wmean=true_wmean
                    ).items():
                        metric_results[k].append(v)

                if np.any(oce_tm):
                    w2 = spatial_w[oce_tm]
                    rmse_tm = float(np.sqrt(np.mean((ptm_s[oce_tm] - ttm_s[oce_tm])**2)))
                    mae_tm  = float(np.mean(np.abs (ptm_s[oce_tm] - ttm_s[oce_tm])))
                    bias_tm = float(np.mean(        ptm_s[oce_tm] - ttm_s[oce_tm]))
                    cc_tm   = float(np.corrcoef(ptm_s[oce_tm], ttm_s[oce_tm])[0, 1]) if np.sum(oce_tm) > 1 else np.nan
                    r2_tm   = float(1 - np.sum((ttm_s[oce_tm] - ptm_s[oce_tm])**2) /
                                       (np.sum((ttm_s[oce_tm] - np.mean(ttm_s[oce_tm]))**2) + 1e-12))
                    denom_tm = np.abs(ptm_s[oce_tm]) + np.abs(ttm_s[oce_tm])
                    # ★ FIX: robust MAPE/sMAPE
                    mape_tm  = float(_safe_mape(ptm_s[oce_tm], ttm_s[oce_tm], thresh=MAPE_THRESH['tm']))
                    smape_tm = float(_safe_smape(ptm_s[oce_tm], ttm_s[oce_tm], thresh=MAPE_THRESH['tm']))
                    pred_wmean_tm = float(np.sum(ptm_s[oce_tm] * w2))
                    true_wmean_tm = float(np.sum(ttm_s[oce_tm] * w2))
                    for k, v in dict(
                        rmse_tm=rmse_tm, mae_tm=mae_tm, bias_tm=bias_tm, cc_tm=cc_tm, r2_tm=r2_tm,
                        mape_tm=mape_tm, smape_tm=smape_tm,
                        pred_wmean_tm=pred_wmean_tm, true_wmean_tm=true_wmean_tm
                    ).items():
                        metric_results[k].append(v)

                if np.any(oce_dir):
                    diff = ((pdir[oce_dir] - tdir[oce_dir] + 180.0) % 360.0) - 180.0
                    rmse_dir = float(np.sqrt(np.mean(diff**2)))
                    mae_dir  = float(np.mean(np.abs(diff)))
                    bias_dir = float(np.mean(diff))
                    pr = np.deg2rad(pdir[oce_dir]); tr = np.deg2rad(tdir[oce_dir])
                    sp, cp = np.sin(pr), np.cos(pr); st, ct = np.sin(tr), np.cos(tr)
                    num = float(np.mean(sp*st) + np.mean(cp*ct))
                    den = float(np.sqrt((np.mean(sp**2)+np.mean(cp**2))*(np.mean(st**2)+np.mean(ct**2))) + 1e-12)
                    cc_dir = num / den
                    var_true = float((np.rad2deg(np.std(tr)))**2)
                    r2_dir = float(1 - (np.mean(diff**2) / (var_true + 1e-12)))
                    for k, v in dict(rmse_dir=rmse_dir, mae_dir=mae_dir, bias_dir=bias_dir, cc_dir=cc_dir, r2_dir=r2_dir).items():
                        metric_results[k].append(v)

                # --- Spatial RMSE 누적 (per-grid)
                if np.any(oce_hs):
                    e = (phs_m - ths_m)**2
                    mask = oce_hs
                    hs_se[mask] += e[mask]; hs_n[mask] += 1
                if np.any(oce_tm):
                    e = (ptm_s - ttm_s)**2
                    mask = oce_tm
                    tm_se[mask] += e[mask]; tm_n[mask] += 1
                if np.any(oce_dir):
                    diff_all = ((pdir - tdir + 180.0) % 360.0) - 180.0
                    e = diff_all**2
                    mask = oce_dir
                    dir_se[mask] += e[mask]; dir_n[mask] += 1

                # --- 스테이션 시계열 적재 (정규화 스페이스)
                for st_name, (i, j) in st_indices.items():
                    if 0 <= test_t < len(station_data[st_name]):
                        meas_hs, meas_tm, meas_dir = station_data[st_name][test_t]
                    else:
                        meas_hs, meas_tm, meas_dir = (np.nan, np.nan, np.nan)
                    st_ts[st_name]['hs']['pred'].append(phs[i, j])
                    st_ts[st_name]['hs']['true'].append(ths[i, j])
                    st_ts[st_name]['hs']['meas'].append(meas_hs)

                    st_ts[st_name]['tm']['pred'].append(ptm[i, j])
                    st_ts[st_name]['tm']['true'].append(ttm[i, j])
                    st_ts[st_name]['tm']['meas'].append(meas_tm)

                    st_ts[st_name]['dir']['pred'].append(pdir[i, j] % 360.0)
                    st_ts[st_name]['dir']['true'].append(tdir[i, j] % 360.0)
                    st_ts[st_name]['dir']['meas'].append(meas_dir % 360.0 if np.isfinite(meas_dir) else np.nan)

                # --- 샘플 플롯 저장 (첫 몇 장)
                if save_cnt < save_limit:
                    _plot_spatial_sample(phs_m, ths_m, lon_map, lat_map, kcs_map,
                                         fname=os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_hs_{save_cnt}.png"),
                                         var_name="Hs", title_suffix="", norm_params=None)
                    _plot_spatial_sample(ptm_s, ttm_s, lon_map, lat_map, kcs_map,
                                         fname=os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_tm_{save_cnt}.png"),
                                         var_name="Tm", title_suffix="", norm_params=None)
                    _plot_dir_sample(pdir, tdir, lon_map, lat_map, kcs_map,
                                     fname=os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_dir_{save_cnt}.png"))
                    save_cnt += 1

            seen += B

    # --- 요약 스칼라 (평균) ------------------------------------------------
    def _nanmean_local(lst): return float(np.nanmean(lst)) if len(lst) else np.nan
    out = {
        'rmse_hs': _nanmean_local(metric_results['rmse_hs']),
        'mae_hs':  _nanmean_local(metric_results['mae_hs']),
        'cc_hs':   _nanmean_local(metric_results['cc_hs']),
        'rmse_tm': _nanmean_local(metric_results['rmse_tm']),
        'mae_tm':  _nanmean_local(metric_results['mae_tm']),
        'cc_tm':   _nanmean_local(metric_results['cc_tm']),
        'rmse_dir': _nanmean_local(metric_results['rmse_dir']),
        'mae_dir':  _nanmean_local(metric_results['mae_dir']),
        'cc_dir':   _nanmean_local(metric_results['cc_dir']),
        'out_dir': out_dir,
    }

    # --- Spatial RMSE map 계산 및 저장 ------------------------------------
    def _safe_rmse(se, n):
        rmse = np.full_like(se, np.nan, dtype=np.float64)
        m = n > 0
        rmse[m] = np.sqrt(se[m] / n[m])
        return rmse.astype(np.float32)

    rmse_hs_map  = _safe_rmse(hs_se,  hs_n)   # (m)
    rmse_tm_map  = _safe_rmse(tm_se,  tm_n)   # (s)
    rmse_dir_map = _safe_rmse(dir_se, dir_n)  # (°)

    _plot_spatial_rmse_maps(
        rmse_hs_map, rmse_tm_map, rmse_dir_map,
        lon_map, lat_map, kcs_map,
        os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_rmse_maps.png")
    )

    # 숫자 요약(면적가중 평균/백분위) + NetCDF 저장 시도
    valid_hs  = (kcs2d > 0) & np.isfinite(rmse_hs_map)
    valid_tm  = (kcs2d > 0) & np.isfinite(rmse_tm_map)
    valid_dir = (kcs2d > 0) & np.isfinite(rmse_dir_map)

    mean_rmse_hs  = float(np.average(rmse_hs_map[valid_hs],  weights=spatial_w[valid_hs]))  if np.any(valid_hs)  else np.nan
    mean_rmse_tm  = float(np.average(rmse_tm_map[valid_tm],  weights=spatial_w[valid_tm]))  if np.any(valid_tm)  else np.nan
    mean_rmse_dir = float(np.average(rmse_dir_map[valid_dir], weights=spatial_w[valid_dir])) if np.any(valid_dir) else np.nan

    p95_rmse_hs  = float(np.nanpercentile(rmse_hs_map[valid_hs],  95)) if np.any(valid_hs)  else np.nan
    p95_rmse_tm  = float(np.nanpercentile(rmse_tm_map[valid_tm],  95)) if np.any(valid_tm)  else np.nan
    p95_rmse_dir = float(np.nanpercentile(rmse_dir_map[valid_dir], 95)) if np.any(valid_dir) else np.nan

    print(f"[spatial RMSE] area-weighted mean → Hs={mean_rmse_hs:.3f} m, Tm={mean_rmse_tm:.3f} s, Dir(cRMSE)={mean_rmse_dir:.1f}°")
    print(f"[spatial RMSE] p95                 → Hs={p95_rmse_hs:.3f} m, Tm={p95_rmse_tm:.3f} s, Dir(cRMSE)={p95_rmse_dir:.1f}°")

    rmse_nc_path = None
    try:
        rmse_nc_path = os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_rmse_maps.nc")
        Hm, Wm = rmse_hs_map.shape
        lon_arr = np.asarray(lon_map)
        lat_arr = np.asarray(lat_map)
        kcs_arr = np.asarray(kcs_map if kcs_map.ndim == 2 else kcs_map[0])

        if lon_arr.ndim == 1 and lat_arr.ndim == 1 and lon_arr.shape[0] == Wm and lat_arr.shape[0] == Hm:
            ds_rmse = xr.Dataset(
                {"rmse_hs":  (("y","x"), rmse_hs_map.astype(np.float32)),
                 "rmse_tm":  (("y","x"), rmse_tm_map.astype(np.float32)),
                 "rmse_dir": (("y","x"), rmse_dir_map.astype(np.float32)),
                 "kcs":      (("y","x"), kcs_arr.astype(np.float32)),},
                coords={"y": lat_arr, "x": lon_arr}
            )
        else:
            ds_rmse = xr.Dataset(
                {"rmse_hs":  (("y","x"), rmse_hs_map.astype(np.float32)),
                 "rmse_tm":  (("y","x"), rmse_tm_map.astype(np.float32)),
                 "rmse_dir": (("y","x"), rmse_dir_map.astype(np.float32)),
                 "lon":      (("y","x"), np.broadcast_to(lon_arr if lon_arr.ndim==2 else np.meshgrid(lon_arr, np.arange(Hm))[0], (Hm,Wm)).astype(np.float32)),
                 "lat":      (("y","x"), np.broadcast_to(lat_arr if lat_arr.ndim==2 else np.meshgrid(np.arange(Hm), lon_arr)[0], (Hm,Wm)).astype(np.float32)),
                 "kcs":      (("y","x"), kcs_arr.astype(np.float32)),},
                coords={"y": np.arange(Hm), "x": np.arange(Wm)}
            )
        ds_rmse.to_netcdf(rmse_nc_path)
        print(f"[spatial RMSE] saved: {rmse_nc_path}")
    except Exception as e:
        print(f"[warn] save spatial RMSE .nc skipped: {e}")

    out.setdefault('spatial_rmse', {})
    out['spatial_rmse'].update({
        "hs_mean":  mean_rmse_hs,  "tm_mean":  mean_rmse_tm,  "dir_mean":  mean_rmse_dir,
        "hs_p95":   p95_rmse_hs,   "tm_p95":   p95_rmse_tm,   "dir_p95":   p95_rmse_dir,
    })

    # --- Buoy validation tables ------------------------------------------------
    if ENABLE_BUOY_DIR_AUTO_TRANSFORM:
        try:
            decide_and_apply_best_buoy_dir_transform(st_ts, save_dir=out_dir)
            print("[buoy dir] auto best-transform search saved as DIAGNOSTIC only; skill table is unaffected.")
        except Exception as e:
            print(f"[warn] buoy dir transform diagnostic skipped: {e}")
    else:
        print("[buoy dir] auto best-transform search is DISABLED "
              "(ENABLE_BUOY_DIR_AUTO_TRANSFORM=False). Buoy directions are "
              "used as recorded after KST-to-UTC timestamp alignment.")

    try:
        df_htm, df_dir = compute_station_skill_tables(st_ts, global_norm_params)
        save_skill_tables_csv_and_latex(df_htm, df_dir, out_dir)
    except Exception as e:
        print(f"[warn] station skill tables skipped: {e}")

    # --- Journal figure bundle --------------------------------------------------
    try:
        create_enhanced_journal_figures(metric_results, st_ts, global_norm_params, out_dir,
                                        kcs_map=kcs_map, lon_map=lon_map, lat_map=lat_map)
    except Exception as e:
        print(f"[warn] journal figures skipped: {e}")

    # ★ NEW: Emulator-vs-SWAN overall metrics CSV + LaTeX
    try:
        save_overall_metrics_csv_and_latex(metric_results, out_dir)
    except Exception as e:
        print(f"[warn] overall metrics save skipped: {e}")

    # Area-weighted ablation metrics, using the same definition as training.
    try:
        import json as _json
        import csv as _csv
        spatial_w_t = torch.as_tensor(spatial_w, dtype=torch.float32)
        abl = compute_test_metrics_all(
            model, loader, global_norm_params, spatial_w_t,
            next(model.parameters()).device
        )
        abl_tagged = dict(abl)
        abl_tagged["run_tag"] = run_tag
        abl_tagged["n_test"] = int(len(base_indices)) if base_indices is not None else None
        with open(os.path.join(out_dir, "ablation_metrics.json"), "w", encoding="utf-8") as _f:
            _json.dump(abl_tagged, _f, indent=2)
        with open(os.path.join(out_dir, "ablation_metrics.csv"), "w", newline="", encoding="utf-8") as _f:
            _w = _csv.writer(_f)
            _w.writerow(list(abl_tagged.keys()))
            _w.writerow(list(abl_tagged.values()))
        print(f"[ablation] area-weighted metrics saved: "
              f"Hs RMSE={abl['hs_rmse_m']:.4f} m  "
              f"Tm RMSE={abl['tm_rmse_s']:.4f} s  "
              f"Dir cRMSE={abl['dir_crmse_deg']:.3f} deg "
              f"(-> ablation_metrics.json / .csv)")
    except Exception as e:
        print(f"[warn] ablation metrics (compute_test_metrics_all) skipped: {type(e).__name__}: {e}")

    t1 = time.perf_counter()
    print(f"[eval] done in {t1 - t0:.2f}s")

    out['files'] = {
        'spatial_rmse_maps': os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_rmse_maps.png"),
        'example_spatial_hs': os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_hs_0.png"),
        'example_spatial_tm': os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_tm_0.png"),
        'example_spatial_dir': os.path.join(out_dir, f"{window_prefix}{out_prefix}_spatial_dir_0.png"),
        'buoy_htm_csv': os.path.join(out_dir, 'buoy_skill_htm.csv'),
        'buoy_dir_csv': os.path.join(out_dir, 'buoy_skill_dir.csv'),
        'station_htm_tex': os.path.join(out_dir, 'station_metrics_htm.tex'),
        'station_dir_tex': os.path.join(out_dir, 'station_metrics_dir.tex'),
        'overall_metrics_csv': os.path.join(out_dir, 'overall_metrics_vs_swan.csv'),
        'aggregated_metrics_tex': os.path.join(out_dir, 'aggregated_metrics.tex'),
        'spatial_rmse_nc': rmse_nc_path if 'rmse_nc_path' in locals() else None,
        'ablation_metrics_json': os.path.join(out_dir, 'ablation_metrics.json'),
        'ablation_metrics_csv': os.path.join(out_dir, 'ablation_metrics.csv')
    }

    create_error_distribution_plots(st_ts, global_norm_params, out_dir,
                                    center_on_median=True, show_median_line=True)
    return out

# =========================================================
# 3/3 — CSV/LaTeX 저장 유틸 + main()
#   - save_overall_metrics_csv_and_latex(): SWAN 대비 전체 성능표 저장
#   - MAPE_THRESH: 0에 가까운 값 배제 임계치(과도한 % 폭주 방지)
#   - main(): 평가 실행 + 파일 경로 로그
# =========================================================

# MAPE가 비정상적으로 커지는 것을 막기 위한 물리단위 임계치
#  - Hs: 0.25 m 미만 구간은 MAPE 계산에서 제외 (관측치가 너무 작아 %가 폭주)
#  - Tm: 1.0 s 미만 구간 제외 (짧은 주기는 분모가 너무 작아짐)
MAPE_THRESH = {
    "hs": 0.25,
    "tm": 1.0,
}

def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"

def save_overall_metrics_csv_and_latex(metric_results: dict, output_dir: str):
    """
    Emulator vs SWAN의 '전체' 성능 요약을 CSV + LaTeX 테이블로 저장.
    metric_results는 evaluate_and_visualize()에서 step별/타일별 평균들을 누적한 컨테이너.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _agg(key):
        arr = np.asarray(metric_results.get(key, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan

    # --- 집계 ---
    hs = {
        "rmse": _agg("rmse_hs"),
        "mae":  _agg("mae_hs"),
        "bias": _agg("bias_hs"),
        "r":    _agg("cc_hs"),
        "r2":   _agg("r2_hs"),
        "mape": _agg("mape_hs"),
        "smape":_agg("smape_hs"),
        "area_w_mean": _agg("true_wmean"),
    }
    tm = {
        "rmse": _agg("rmse_tm"),
        "mae":  _agg("mae_tm"),
        "bias": _agg("bias_tm"),
        "r":    _agg("cc_tm"),
        "r2":   _agg("r2_tm"),
        "mape": _agg("mape_tm"),
        "smape":_agg("smape_tm"),
    }
    dr = {
        "rmse": _agg("rmse_dir"),
        "mae":  _agg("mae_dir"),
        "bias": _agg("bias_dir"),
        "r":    _agg("cc_dir"),
        "r2":   _agg("r2_dir"),
    }

    # --- CSV ---
    df = pd.DataFrame({
        "Metric": [
            "RMSE", "MAE", "Mean Bias", "Pearson r", "R^2", "MAPE (%)", "sMAPE (%)", "Area-weighted mean"
        ],
        "Hs (m)": [
            hs["rmse"], hs["mae"], hs["bias"], hs["r"], hs["r2"], hs["mape"], hs["smape"], hs["area_w_mean"]
        ],
        "Tm (s)": [
            tm["rmse"], tm["mae"], tm["bias"], tm["r"], tm["r2"], tm["mape"], tm["smape"], np.nan
        ],
        "Dir (deg)": [
            dr["rmse"], dr["mae"], dr["bias"], dr["r"], dr["r2"], np.nan, np.nan, np.nan
        ],
    })
    csv_path = os.path.join(output_dir, "overall_metrics_vs_swan.csv")
    df.to_csv(csv_path, index=False)

    # --- LaTeX (논문 표 스타일) ---
    tex_path = os.path.join(output_dir, "aggregated_metrics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]"+"\n")
        f.write(r"\centering"+"\n")
        f.write(r"\scriptsize"+"\n")
        f.write(r"\caption{Overall performance statistics on the test set (vs.\ SWAN). "
                r"For direction, RMSE/MAE/Bias are circular errors in degrees (cRMSE, cMAE, cBias).}"+"\n")
        f.write(r"\label{tab:aggregated_metrics}"+"\n")
        f.write(r"\begin{tabular}{lccc}"+"\n")
        f.write(r"\toprule"+"\n")
        f.write(r"Metric & $H_s$ (m) & $T_m$ (s) & $Dir$ (°) \\"+"\n")
        f.write(r"\midrule"+"\n")
        f.write(f"RMSE              & {_fmt(hs['rmse'])} & {_fmt(tm['rmse'])} & {_fmt(dr['rmse'])} \\\\\n")
        f.write(f"MAE               & {_fmt(hs['mae'])} & {_fmt(tm['mae'])} & {_fmt(dr['mae'])} \\\\\n")
        f.write(f"Mean Bias         & {_fmt(hs['bias'])} & {_fmt(tm['bias'])} & {_fmt(dr['bias'])} \\\\\n")
        f.write(f"Pearson $r$       & {_fmt(hs['r'])} & {_fmt(tm['r'])} & {_fmt(dr['r'])} \\\\\n")
        f.write(f"$R^2$             & {_fmt(hs['r2'])} & {_fmt(tm['r2'])} & {_fmt(dr['r2'])} \\\\\n")
        f.write(f"MAPE (\\%)         & {_fmt(hs['mape'])} & {_fmt(tm['mape'])} & — \\\\\n")
        f.write(f"sMAPE (\\%)        & {_fmt(hs['smape'])} & {_fmt(tm['smape'])} & — \\\\\n")
        f.write(f"Area-weighted mean & {_fmt(hs['area_w_mean'])} & — & — \\\\\n")
        f.write(r"\bottomrule"+"\n")
        f.write(r"\end{tabular}"+"\n")
        f.write(r"\end{table}"+"\n")

    print(f"[overall] CSV saved: {csv_path}")
    print(f"[overall] LaTeX saved: {tex_path}")

# =========================================================
# main()
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="UNet-ConvLSTM INFERENCE ONLY (Enhanced + Buoy Validation + Embargo)")
    parser.add_argument('--data_path', type=str,
                        default=r'C:\DELFT3DFM\South_Korea_emulator_2020_ST6_bnd_test\wave\wavm-Waves_2019_2020_final.nc',
                        help='Path to NetCDF data file (simulation NetCDF)')
    parser.add_argument('--weights', type=str,
                        default=r'ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth',
                        help='Path to .pth weights file')
    parser.add_argument('--time_steps', type=int, default=17498, help='Number of time steps to use')
    parser.add_argument('--seq_length', type=int, default=12, help='Sequence length (must match training)')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
    parser.add_argument('--station_root', type=str,
                        default=r"C:\Users\User\PycharmProjects\CUDA_emulator_LSTM_UNET",
                        help='Folder containing station CSVs')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'convlstm_only', 'unetpp_stack', 'unetpp_only'],
                        help='Architecture variant of this checkpoint. '
                             'Must match how the weights were trained.')
    parser.add_argument('--feat', type=int, nargs='+', default=[32, 64, 128, 256, 512],
                        help='UNet++ channel widths. Must match the checkpoint. '
                             'The E01-E08 runs used 32 64 128 256 512.')
    parser.add_argument('--use_bnd', type=str, default='auto',
                        choices=['auto', 'on', 'off'],
                        help="Boundary descriptors. 'auto' decides from the "
                             "weight filename and the checkpoint's input "
                             "channel count; 'on' forces 10ch, 'off' forces 6ch.")
    parser.add_argument('--tag', type=str, default='inference',
                        help='Tag added to all output filenames so multiple '
                             'runs (E01..E08) do not overwrite each other.')
    parser.add_argument('--save_svg', action='store_true', default=False,
                        help='Also write SVG files. Off by default for fast runs.')
    parser.add_argument('--split', type=str, default='block',
                        choices=['block', 'chrono_2019_train_2020_test', 'chrono_2020_train_2019_test'],
                        help='Data split. Use chrono_2019_train_2020_test for E04.')
    args = parser.parse_args()

    global SAVE_SVG
    SAVE_SVG = bool(args.save_svg)
    print(f"[fig] save_svg = {SAVE_SVG}")

    # ---- open data ----
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"NetCDF not found: {args.data_path}")
    ds = xr.open_dataset(args.data_path)

    total_T = int(ds['windu'].shape[0])
    T = int(min(args.time_steps, total_T))
    if 'time' in ds:
        tvals = pd.to_datetime(ds['time'].values[:T])
        time_index = pd.DatetimeIndex(tvals).tz_localize(None)
    else:
        time_index = pd.date_range(start="2019-01-01 00:00:00", periods=T, freq="h", tz="UTC").tz_localize(None)

    # ---- split ----
    hs_raw = ds["hsign"].values[:T]
    Y, X = hs_raw.shape[-2], hs_raw.shape[-1]
    wave_for_split = np.zeros((T, 4, Y, X), dtype=np.float32)
    wave_for_split[:, 0] = hs_raw

    if args.split == 'block':
        idx_tr, idx_va, idx_te = make_block_stratified_split(
            wave_for_split,
            seq_length=args.seq_length,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            block_hours=168, q=5, seed=42,
            embargo_hours=args.seq_length
        )
        split_tag = "block(bh=168,q=5,emb=seq)"
    elif args.split == 'chrono_2019_train_2020_test':
        idx_tr, idx_va, idx_te, split_tag = make_chronological_split(
            time_index, seq_length=args.seq_length, holdout_year=2020
        )
    elif args.split == 'chrono_2020_train_2019_test':
        idx_tr, idx_va, idx_te, split_tag = make_chronological_split(
            time_index, seq_length=args.seq_length, holdout_year=2019
        )
    else:
        raise ValueError(f"Unknown --split: {args.split}")

    print(f"[split] {args.split} ({split_tag}) tr/va/te = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")
    if (set(idx_tr) & set(idx_va)) or (set(idx_tr) & set(idx_te)) or (set(idx_va) & set(idx_te)):
        raise RuntimeError("Split sets are not disjoint.")

    # ---- norm params (train-consistent) ----
    global_norm_params = compute_params_with_indices(
        ds, idx_train=idx_tr, seq_length=args.seq_length
    )
    print("[norm] hs:", global_norm_params['hs'], " tm:", global_norm_params['tm'])

    # ---- preprocess (base 6ch) ----
    input_data, wave_data, lon, lat, kcs = load_and_preprocess_data(ds, global_norm_params, time_steps=T)

    # ---- peek weights to decide input channels ----
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights .pth not found: {args.weights}")
    raw = torch.load(args.weights, map_location=device)
    sd = raw['state_dict'] if (isinstance(raw, dict) and 'state_dict' in raw) else raw
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    def _infer_expected_logical_in(sd, variant, seq_length, hidden_dim):
        """Infer logical input channels (6 or 10) from checkpoint weights."""
        v = str(variant).lower()
        if v in ("full", "unet_convlstm", "baseline", "unetpp_only", "unet_only", "unetpponly"):
            w = sd.get("unet.enc00.conv_pw.weight", None)
            return int(w.shape[1]) if w is not None else None
        if v in ("unetpp_stack", "unetppstack", "unet_stack"):
            w = sd.get("unet.enc00.conv_pw.weight", None)
            if w is None:
                return None
            first_in = int(w.shape[1])
            if first_in % int(seq_length) != 0:
                raise RuntimeError(
                    f"unetpp_stack first conv has in_channels={first_in}, "
                    f"not divisible by seq_length={seq_length}."
                )
            return first_in // int(seq_length)
        if v in ("convlstm_only", "convlstm-only", "convlstmonly"):
            w = sd.get("lstm1.conv.weight", None)
            if w is None:
                return None
            return int(w.shape[1]) - int(hidden_dim)
        return None

    _HIDDEN_DIM_PEEK = 128
    expected_in = _infer_expected_logical_in(sd, args.variant, args.seq_length, _HIDDEN_DIM_PEEK)
    print(f"[peek] variant={args.variant} checkpoint logical input channels = {expected_in}")

    # ---- optional BND ----
    # --use_bnd auto : decide from filename + checkpoint input channels (V5.3 behavior)
    # --use_bnd on   : force boundary channels (input = 10)
    # --use_bnd off  : force no boundary channels (input = 6)
    if args.use_bnd == 'on':
        use_bnd = True
    elif args.use_bnd == 'off':
        use_bnd = False
    else:  # 'auto'
        use_bnd = (("bndon" in os.path.basename(args.weights).lower())
                   or (expected_in == 10))
    print(f"[BND] use_bnd={use_bnd} (mode={args.use_bnd}, "
          f"expected_in={expected_in})")
    in_ch = int(input_data.shape[1])
    if use_bnd and in_ch == 6:
        print("[BND] Building boundary feature channels (4ch) ...")
        bnd_feat = build_bnd_features(
            ds_sim=ds, kcs=kcs, time_index=time_index,
            global_norm_params=global_norm_params,
            idx_tr=idx_tr, seq_length=args.seq_length
        )  # (T,4,H,W)
        if bnd_feat.shape[0] != input_data.shape[0] or bnd_feat.shape[2:] != input_data.shape[2:]:
            raise RuntimeError(f"[BND] shape mismatch: bnd {bnd_feat.shape} vs input {input_data.shape}")
        input_data = np.concatenate([input_data, bnd_feat], axis=1)  # (T,10,H,W)
        in_ch = 10
        print(f"[BND] Added 4 channels → input channels = {in_ch}")
    else:
        print(f"[BND] Skipped (use_bnd={use_bnd}, current_in={in_ch}, expected_in={expected_in})")

    # dataset은 더 이상 필요 없으므로 해제
    ds.close()

    # ---- dataset/loader (TEST ONLY via holdout indices) ----
    N = T - args.seq_length
    if N <= 0:
        raise RuntimeError(f"Not enough time steps: T={T}, seq_length={args.seq_length}")
    base_ds = WindWaveDataset(input_data, wave_data, args.seq_length, start_idx=0, end_idx=N)
    test_ds = SubsetIndicesDataset(base_ds, idx_te)
    dl_te = DataLoader(test_ds,
                       batch_size=max(1, min(args.batch_size, len(test_ds))),
                       shuffle=False, collate_fn=collate_fn, num_workers=0,
                       pin_memory=torch.cuda.is_available())

    # ---- model ----
    # Built through the shared factory in model_architectures.py so the
    # exact training-time class is used. --variant selects full /
    # convlstm_only / unetpp_stack / unetpp_only; --feat must match the
    # checkpoint (the E01-E08 runs used 32 64 128 256 512).
    hidden_dim = 128
    print(f"[model] variant={args.variant}  in_ch={in_ch}  "
          f"hidden_dim={hidden_dim}  feat={args.feat}")
    model = build_model(
        args.variant,
        input_channels=in_ch,
        hidden_dim=hidden_dim,
        feat=list(args.feat),
        seq_length=args.seq_length,
    ).to(device)
    model.eval()

    # ---- load weights ----
    # Sanity: the checkpoint's input-channel count must match the
    # preprocessed input. A mismatch here usually means the wrong
    # --use_bnd was passed (BND-on weights need 10ch, BND-off need 6ch).
    if expected_in is not None and expected_in != in_ch:
        raise RuntimeError(
            f"Logical input channels mismatch: weights expect {expected_in}, but "
            f"preprocessed data has {in_ch}. Check --use_bnd "
            f"(on=10ch, off=6ch) for this checkpoint."
        )
    # strict-except-log_vars: log_vars is a loss-only parameter that the
    # training checkpoint carries but inference does not need. ANY other
    # missing or unexpected key is a hard error, because it means
    # model_architectures.py disagrees with the checkpoint (wrong
    # --variant, wrong --feat, or a real definition drift).
    missing, unexpected = load_checkpoint_strict_except(
        model, sd,
        allowed_missing=[],
        allowed_unexpected=("log_vars",),
    )
    if unexpected:
        print(f"[load] unexpected keys (allowed, ignored): {unexpected}")
    if missing:
        print(f"[load] missing keys (allowed): {missing}")
    print("[load] weights loaded successfully "
          f"(variant={args.variant}, in_ch={in_ch}).")

    # ---- station CSV (Buoy) ----
    station_data = load_all_station_data(args.station_root, global_norm_params, time_index)

    # ---- unify buoy direction (from → toward) ----
    adjust_buoy_direction_convention_inplace(
        station_data,
        src=BUOY_DIR_DEFINITION,
        dst=MODEL_DIR_DEFINITION
    )

    # ---- evaluate + visualize ----
    metrics = evaluate_and_visualize(
        model, dl_te, lon, lat, kcs, STATIONS, station_data,
        test_start_idx=None, seq_length=args.seq_length, global_norm_params=global_norm_params,
        out_prefix="inference_eval_embargo", pth_filename=args.weights, date_index=time_index,
        base_indices=idx_te, run_tag=args.tag
    )

    if isinstance(metrics, dict):
        print("\n=== Inference summary (Hs) ===")
        for k in ("rmse_hs", "mae_hs", "cc_hs"):
            v = metrics.get(k, None)
            if v is not None and np.isfinite(v):
                print(f"{k}: {v:.4f}")

        if 'spatial_rmse' in metrics and isinstance(metrics['spatial_rmse'], dict):
            s = metrics['spatial_rmse']
            print("\n=== Spatial RMSE (area-weighted mean) ===")
            print(f"Hs:  {s.get('hs_mean',  np.nan):.3f} m   (p95 {s.get('hs_p95',  np.nan):.3f} m)")
            print(f"Tm:  {s.get('tm_mean',  np.nan):.3f} s   (p95 {s.get('tm_p95',  np.nan):.3f} s)")
            print(f"Dir: {s.get('dir_mean', np.nan):.1f} °   (p95 {s.get('dir_p95', np.nan):.1f} °)")
            if metrics.get('files', {}).get('spatial_rmse_nc'):
                print(f"Spatial RMSE maps (NetCDF): {metrics['files']['spatial_rmse_nc']}")

        out_dir = metrics.get('out_dir', None)
        if out_dir:
            print(f"\n[Buoy Validation] Tables saved to:\n"
                  f" - {os.path.join(out_dir, 'buoy_skill_htm.csv')}\n"
                  f" - {os.path.join(out_dir, 'buoy_skill_dir.csv')}\n"
                  f" - {os.path.join(out_dir, 'station_metrics_htm.tex')}\n"
                  f" - {os.path.join(out_dir, 'station_metrics_dir.tex')}\n"
                  f" - {os.path.join(out_dir, 'overall_metrics_vs_swan.csv')}\n"
                  f" - {os.path.join(out_dir, 'aggregated_metrics.tex')}"
                  )

if __name__ == "__main__":
    main()
