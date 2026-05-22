# bnd_features.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.ndimage import distance_transform_edt
import wavespectra as ws

# ──────────────────────────────────────────────────────────────
# 1) .bnd 읽어서 시간별 Hs, Tm01, mean dir(sin, cos)로 요약
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
    """스펙트럼 평균방향: atan2(∫∫E sinθ df dθ, ∫∫E cosθ df dθ) → sin, cos"""
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
    SWAN .bnd 한 개 파일을 읽어 시간별 요약 반환:
      columns = ['hs','tm','sin','cos']  (index = time UTC)
    direction:
      - "from"  : 파일이 '오는 방향(nautical-from)'이면 그대로 사용
      - "toward": '가는 방향(toward)'이면 +180° 회전해서 'from'으로 변환
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
        # toward → from (평균각 +180°)
        sdir = -sdir
        cdir = -cdir

    df = pd.DataFrame({"hs": hs, "tm": tm01, "sin": sdir, "cos": cdir}, index=time)
    return df

def read_all_bnds(bnd_dir: Path, direction: str = "from") -> Dict[str, pd.DataFrame]:
    """
    디렉토리 내 *.bnd 전부 읽어 dict 반환.
    키 = 파일명(확장자 제외, 대문자), 값 = read_bnd_to_series 결과
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
# 2) BOUNDSPEC SEGMENT → 경계 라벨맵(owner_label) 만들기 (절대 인덱스)
# ──────────────────────────────────────────────────────────────
def segment_pixels(i1: int, j1: int, i2: int, j2: int) -> List[Tuple[int, int]]:
    """정수 좌표 보간(수평/수직 세그먼트만 지원). SWAN: I=x(열, n), J=y(행, m)"""
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
    """세그먼트 끝점이 실제 외곽 경계 위에 있는지 검사(실수 방지용)."""
    for name, ((i1, j1), (i2, j2)) in segments.items():
        for (i, j) in [(i1, j1), (i2, j2)]:
            ok = (i in (0, N-1)) or (j in (0, M-1))
            if not ok:
                raise ValueError(f"{name}: (I,J)=({i},{j})가 외곽 경계가 아닙니다. "
                                 f"허용: I∈{{0,{N-1}}} 또는 J∈{{0,{M-1}}}")

def build_owner_label(
    H: int, W: int,
    segments: Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]],
    exact_M: int, exact_N: int,
    kcs: np.ndarray = None,
    swap_ij: bool = False
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    반환:
      owner_label: (H,W) int 배열 (0=비할당/육지, 1..K=세그먼트 라벨)
      id2name: {라벨번호: 세그먼트이름}

    절대 인덱스 모드(무스케일):
      - 기본(미전치):   (H,W)==(exact_M, exact_N), 맵핑 m=J, n=I
      - 전치(swap_ij): (H,W)==(exact_N, exact_M), 맵핑 m=I, n=J
    """
    if not swap_ij:
        if (H != exact_M) or (W != exact_N):
            raise ValueError(
                f"Grid mismatch: data(H,W)=({H},{W}) != SWAN(M,N)=({exact_M},{exact_N}). 자동 스케일링은 금지되어 있습니다."
            )
    else:
        if (H != exact_N) or (W != exact_M):
            raise ValueError(
                f"Grid mismatch (transposed expected): data(H,W)=({H},{W}) != (N,M)=({exact_N},{exact_M}). 자동 스케일링은 금지되어 있습니다."
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
                n, m = ii, jj  # 기본: n=I, m=J
            else:
                n, m = jj, ii  # 전치: n=J, m=I
            if not (0 <= n < W and 0 <= m < H):
                raise ValueError(
                    f"{name}: (mapped m,n)=({m},{n})가 범위를 벗어납니다. "
                    f"유효범위 n∈[0,{W-1}], m∈[0,{H-1}]"
                )
            owner[m, n] = k

    # 최근접 세그먼트 보로노이 분할(내해 픽셀에 가장 가까운 경계 세그먼트 라벨 전파)
    seed_mask = owner > 0
    if not seed_mask.any():
        raise RuntimeError("No boundary seeds painted; check segments and sizes.")
    _, (iy, ix) = distance_transform_edt(~seed_mask, return_indices=True)
    nearest = owner[iy, ix]

    # 육지(kcs<=0)면 0으로 남기기
    if kcs is not None:
        nearest = np.where((kcs > 0), nearest, 0)

    return nearest.astype(np.int32), id2name

# ──────────────────────────────────────────────────────────────
# 3) 시간×4채널 경계 맵 만들기
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
    반환: feat  (T, 4, H, W)  = Hs_bnd, Tm_bnd, sin_bnd, cos_bnd  (hs, tm은 정규화 완료)
    """
    H, W = owner_label.shape
    K = max(id2name.keys())

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        s = df.reindex(time_index)

        # 시간 인덱스면 time 보간, 아니면 기본 보간
        if isinstance(s.index, pd.DatetimeIndex):
            s = s.interpolate(method="time", limit_direction="both")
        else:
            s = s.interpolate(limit_direction="both")

        # deprecated: fillna(method="...")  →  대체: .bfill() / .ffill()
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
