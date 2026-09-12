# bnd_features.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.ndimage import distance_transform_edt
import wavespectra as ws

# ──────────────────────────────────────────────────────────────
# 1) Read .bnd files and summarize per-time Hs, Tm01, mean direction (sin, cos)
# ──────────────────────────────────────────────────────────────
def _tm01_numint(efth: np.ndarray, freq: np.ndarray, dtheta_rad: np.ndarray):
    """Tm01 = m0/m1, m0=∫∫E df dθ, m1=∫∫ fE df dθ   (efth: (T,F,D))"""
    df = np.gradient(freq)  # (F,)
    w0 = df[None, :, None] * dtheta_rad[None, None, :]  # (1,F,1)*(1,1,D)
    m0 = np.sum(efth * w0, axis=(1, 2))                         # (T,)
    m1 = np.sum(efth * (freq[None, :, None] * w0), axis=(1, 2)) # (T,)
    tm01 = m0 / np.maximum(m1, 1e-12)
    return m0, tm01

def _mean_dir_sincos(efth: np.ndarray, dirs_deg: np.ndarray, freq: np.ndarray, dtheta_rad: np.ndarray):
    """Spectral mean direction: atan2(integral E*sin(theta) df dtheta, integral E*cos(theta) df dtheta) -> sin, cos"""
    df = np.gradient(freq)                # (F,)
    theta = np.deg2rad(dirs_deg)          # (D,)
    s = np.sin(theta)[None, None, :]      # (1,1,D)
    c = np.cos(theta)[None, None, :]      # (1,1,D)
    w = df[None, :, None] * dtheta_rad[None, None, :]
    Ey = np.sum(efth * s * w, axis=(1, 2))  # (T,)
    Ex = np.sum(efth * c * w, axis=(1, 2))  # (T,)
    ang = np.arctan2(Ey, Ex)                # (-pi, pi]
    return np.sin(ang), np.cos(ang)         # (T,), (T,)

def read_bnd_to_series(bnd_path: Path, direction: str = "from") -> pd.DataFrame:
    """
    Read a single SWAN .bnd file and return per-time summary:
      columns = ['hs','tm','sin','cos']  (index = time UTC)
    direction:
      - "from"  : file is already in 'coming-from direction' (nautical-from); use as-is
      - "toward": file is in 'going-toward direction'; rotate by +180 deg to convert to 'from'
    """
    spec = ws.read_swan(str(bnd_path))             # efth(time,freq,dir[,(site)])
    da = spec.efth if hasattr(spec, "efth") else spec["efth"]
    da = da.squeeze().transpose("time", "freq", "dir")  # (T,F,D)

    efth = da.values.astype("float64")
    time = pd.to_datetime(da["time"].values)
    freq = da["freq"].values.astype("float64")  # Hz
    dirs = da["dir"].values.astype("float64")   # degrees (0..360)
    dtheta = np.gradient(dirs)
    dtheta_rad = np.deg2rad(dtheta)

    m0, tm01 = _tm01_numint(efth, freq, dtheta_rad)
    hs = 4.0 * np.sqrt(np.maximum(m0, 0.0))
    sdir, cdir = _mean_dir_sincos(efth, dirs, freq, dtheta_rad)

    if direction.lower() == "toward":
        # toward -> from (rotate mean angle by +180 deg)
        sdir = -sdir
        cdir = -cdir

    df = pd.DataFrame({"hs": hs, "tm": tm01, "sin": sdir, "cos": cdir}, index=time)
    return df

def read_all_bnds(bnd_dir: Path, direction: str = "from") -> Dict[str, pd.DataFrame]:
    """
    Read all *.bnd files in a directory and return as dict.
    keys = filename (without extension, uppercased)
    values = result of read_bnd_to_series
    """
    out = {}
    files = sorted(Path(bnd_dir).glob("*.bnd"))
    if not files:
        raise FileNotFoundError(f"No .bnd files in {bnd_dir}")
    for fp in files:
        name = fp.stem.upper()
        out[name] = read_bnd_to_series(fp, direction=direction)
    return out

# ──────────────────────────────────────────────────────────────
# 2) Build boundary label map (owner_label) from BOUNDSPEC SEGMENTs (absolute indices)
# ──────────────────────────────────────────────────────────────
def segment_pixels(i1: int, j1: int, i2: int, j2: int) -> List[Tuple[int, int]]:
    """Integer-coordinate interpolation (horizontal/vertical segments only). SWAN: I=x(col, n), J=y(row, m)"""
    pts = []
    if i1 == i2:  # vertical
        i = i1
        for j in range(min(j1, j2), max(j1, j2) + 1):
            pts.append((i, j))
    elif j1 == j2:  # horizontal
        j = j1
        for i in range(min(i1, i2), max(i1, i2) + 1):
            pts.append((i, j))
    else:
        raise ValueError(f"Only vertical/horizontal segments supported: {(i1, j1, i2, j2)}")
    return pts

def assert_on_edges(segments: Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]], M: int, N: int):
    """Check that segment endpoints lie on the actual outer boundary (sanity check)."""
    for name, ((i1, j1), (i2, j2)) in segments.items():
        for (i, j) in [(i1, j1), (i2, j2)]:
            ok = (i in (0, N-1)) or (j in (0, M-1))
            if not ok:
                raise ValueError(f"{name}: (I,J)=({i},{j}) is not on the outer boundary. "
                                 f"Allowed: I in {{0,{N-1}}} or J in {{0,{M-1}}}")

def build_owner_label(
    H: int, W: int,
    segments: Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]],
    exact_M: int, exact_N: int,
    kcs: np.ndarray = None,
    swap_ij: bool = False
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Returns:
      owner_label: (H,W) int array (0=unassigned/land, 1..K=segment label)
      id2name: {label_id: segment_name}

    Absolute-index mode (no scaling):
      - default (no transpose):  (H,W)==(exact_M, exact_N), mapping m=J, n=I
      - transposed (swap_ij):    (H,W)==(exact_N, exact_M), mapping m=I, n=J
    """
    if not swap_ij:
        if (H != exact_M) or (W != exact_N):
            raise ValueError(
                f"Grid mismatch: data(H,W)=({H},{W}) != SWAN(M,N)=({exact_M},{exact_N}). Automatic scaling is disabled."
            )
    else:
        if (H != exact_N) or (W != exact_M):
            raise ValueError(
                f"Grid mismatch (transposed expected): data(H,W)=({H},{W}) != (N,M)=({exact_N},{exact_M}). Automatic scaling is disabled."
            )

    owner = np.zeros((H, W), dtype=np.int32)
    id2name: Dict[int, str] = {}
    k = 0

    for name, ((i1, j1), (i2, j2)) in segments.items():
        k += 1
        id2name[k] = name
        pts = segment_pixels(i1, j1, i2, j2)
        for (ii, jj) in pts:
            if not swap_ij:
                n, m = ii, jj  # default: n=I, m=J
            else:
                n, m = jj, ii  # transposed: n=J, m=I
            if not (0 <= n < W and 0 <= m < H):
                raise ValueError(
                    f"{name}: (mapped m,n)=({m},{n}) is out of range. "
                    f"Valid range: n in [0,{W-1}], m in [0,{H-1}]"
                )
            owner[m, n] = k

    # Nearest-segment Voronoi assignment (propagate the nearest boundary-segment label to each interior pixel)
    seed_mask = owner > 0
    if not seed_mask.any():
        raise RuntimeError("No boundary seeds painted; check segments and sizes.")
    _, (iy, ix) = distance_transform_edt(~seed_mask, return_indices=True)
    nearest = owner[iy, ix]

    # Keep land pixels (kcs<=0) labeled as 0
    if kcs is not None:
        nearest = np.where((kcs > 0), nearest, 0)

    return nearest.astype(np.int32), id2name

# ──────────────────────────────────────────────────────────────
# 3) Build per-time 4-channel boundary feature map
# ──────────────────────────────────────────────────────────────
def make_boundary_feature_maps(
    time_index: pd.DatetimeIndex,
    owner_label: np.ndarray,
    seg_series: Dict[str, pd.DataFrame],
    id2name: Dict[int, str],
    kcs: np.ndarray,
    norm_hs: Tuple[float, float],
    norm_tm: Tuple[float, float]
) -> np.ndarray:
    """
    Returns: feat  (T, 4, H, W)  = Hs_bnd, Tm_bnd, sin_bnd, cos_bnd  (hs and tm are already normalized)
    """
    H, W = owner_label.shape
    K = max(id2name.keys())

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        s = df.reindex(time_index)

        # If the index is time-based, use time interpolation; otherwise use default interpolation
        if isinstance(s.index, pd.DatetimeIndex):
            s = s.interpolate(method="time", limit_direction="both")
        else:
            s = s.interpolate(limit_direction="both")

        # deprecated: fillna(method="...")  ->  use .bfill() / .ffill() instead
        s = s.bfill().ffill()
        return s.fillna(0.0)

    hs_mat = np.zeros((len(time_index), K), "float32")
    tm_mat = np.zeros((len(time_index), K), "float32")
    si_mat = np.zeros((len(time_index), K), "float32")
    co_mat = np.zeros((len(time_index), K), "float32")

    for k in range(1, K + 1):
        name = id2name[k]
        if name not in seg_series:
            continue
        df = _align(seg_series[name])
        hs_mat[:, k - 1] = df["hs"].astype("float32").values
        tm_mat[:, k - 1] = df["tm"].astype("float32").values
        si_mat[:, k - 1] = df["sin"].astype("float32").values
        co_mat[:, k - 1] = df["cos"].astype("float32").values

    def _norm(x, lo, hi):
        den = max(hi - lo, 1e-8)
        return (x - lo) / den

    hs_mat = _norm(hs_mat, *norm_hs)
    tm_mat = _norm(tm_mat, *norm_tm)

    idx_map = np.clip(owner_label - 1, 0, max(K - 1, 0)).astype(np.int32)
    feat = np.zeros((len(time_index), 4, H, W), "float32")
    ocean = (kcs > 0)

    for t in range(len(time_index)):
        hs_map = np.take(hs_mat[t], idx_map)
        tm_map = np.take(tm_mat[t], idx_map)
        si_map = np.take(si_mat[t], idx_map)
        co_map = np.take(co_mat[t], idx_map)

        hs_map = np.where(ocean, hs_map, 0.0)
        tm_map = np.where(ocean, tm_map, 0.0)
        si_map = np.where(ocean, si_map, 0.0)
        co_map = np.where(ocean, co_map, 0.0)

        feat[t, 0] = hs_map
        feat[t, 1] = tm_map
        feat[t, 2] = si_map
        feat[t, 3] = co_map

    return feat
