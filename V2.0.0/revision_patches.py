# -*- coding: utf-8 -*-
"""
revision_patches.py  (revised after smoke-test feedback)
========================================================
Targeted patches for the APOR revision.

1) install_unit_normalized_direction_loss()
   - Monkey-patches base.ds_loss so that predicted (sin, cos) AND
     target (sin, cos) are unit-normalized to length 1 BEFORE computing
     the circular loss 1 - cos(delta theta). This addresses Reviewer 1
     (item 3.7) directly: the trained model now optimizes a true unit-
     vector circular loss, and we can state so unambiguously in the
     manuscript.

2) auto_align_bnd_dir_trainonly(bnd_feat, ds_sim, kcs2d, train_idx)
   - Train-only counterpart of base._auto_align_bnd_dir.
   - Score computed on TRAIN indices only; the chosen rotation is then
     applied unchanged to the full bnd_feat array, so val/test inherit
     the same fixed rotation. (Reviewer 2, 2.4.)

3) make_chronological_split(time_index, seq_length, holdout_year)
   - Trains on one year and tests on the other with an L-hour embargo
     at the year boundary.

4) score_directions_train_only(...)
   - Returns the {0, +90, -90, +180} score dictionary for the bar plot.

5) compute_test_metrics_all(model, loader, norm_params, spatial_w, device)
   - Area-weighted RMSE, MAE for Hs (m), Tm (s), and circular Dir
     (degrees) on the loader. Used to populate full ablation tables.
"""

import numpy as np
import pandas as pd
import torch


# ===========================================================
# 1) Unit-normalized direction loss (monkey patch)
# ===========================================================
def install_unit_normalized_direction_loss(base_module):
    """
    Wrap base_module.ds_loss so that:
      - predicted (psin, pcos) are normalized to unit length
      - target    (tsin, tcos) are normalized to unit length
    BEFORE computing the cos(delta theta) circular loss. All other
    behavior of ds_loss is preserved.

    Call once, BEFORE base.train() is invoked.
    """
    import torch.nn.functional as F

    original_ds_loss = base_module.ds_loss
    if getattr(original_ds_loss, "_unit_norm_patched", False):
        return  # idempotent

    def patched_ds_loss(pred, target, spatial_weight=None, valid_mask=None,
                       log_vars=None, use_huber=False, huber_beta=0.05,
                       eps=1e-6):
        # Reproduce the channel-ordering inference used in the original.
        def _to_last(x):
            if x.shape[-1] in (3, 4, 5):
                return x
            elif x.dim() >= 4 and x.shape[-3] in (3, 4, 5):
                return x.movedim(-3, -1)
            elif x.dim() >= 4 and x.shape[1] in (3, 4, 5):
                return x.movedim(1, -1)
            elif x.dim() >= 3 and x.shape[1] in (3, 4, 5):
                return x.movedim(1, -1)
            return x

        pred   = torch.nan_to_num(pred,   nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        pred   = _to_last(pred)
        target = _to_last(target)

        phs, ptm = pred[..., 0], pred[..., 1]
        psin     = pred[..., 2].clamp(-1.0, 1.0)
        pcos     = pred[..., 3].clamp(-1.0, 1.0)
        ths, ttm = target[..., 0], target[..., 1]
        tsin     = target[..., 2].clamp(-1.0, 1.0)
        tcos     = target[..., 3].clamp(-1.0, 1.0)

        # UNIT-NORMALIZE both predicted and target (sin, cos)
        pmag = torch.sqrt(psin * psin + pcos * pcos + eps)
        tmag = torch.sqrt(tsin * tsin + tcos * tcos + eps)
        psin = psin / pmag
        pcos = pcos / pmag
        tsin = tsin / tmag
        tcos = tcos / tmag

        if use_huber:
            def crit(a, b):
                return F.smooth_l1_loss(a, b, beta=huber_beta, reduction="none")
        else:
            def crit(a, b):
                return F.l1_loss(a, b, reduction="none")

        w = 1.0
        if spatial_weight is not None: w = w * spatial_weight
        if valid_mask    is not None: w = w * valid_mask
        w = torch.as_tensor(w, dtype=phs.dtype, device=phs.device)

        def wmean(x):
            num = (x * w).nan_to_num(0.0).sum()
            den = (torch.ones_like(x) * w).nan_to_num(0.0).sum().clamp_min(eps)
            return num / den

        loss_hs  = wmean(crit(phs, ths))
        loss_tm  = wmean(crit(ptm, ttm))
        cos_delta = (pcos * tcos + psin * tsin).clamp(-1.0, 1.0)
        loss_dir = wmean(1.0 - cos_delta)

        if (log_vars is not None) and isinstance(log_vars, torch.nn.Parameter):
            total = 0.0
            for i, Li in enumerate([loss_hs, loss_tm, loss_dir]):
                s2 = torch.exp(-log_vars[i])
                total = total + 0.5 * (s2 * Li + log_vars[i])
        else:
            total = loss_hs + loss_tm + loss_dir
        return total, (loss_hs.detach(), loss_tm.detach(), loss_dir.detach())

    patched_ds_loss._unit_norm_patched = True
    base_module.ds_loss = patched_ds_loss
    print("[patch] ds_loss replaced with unit-normalized direction loss "
          "(predicted and target (sin, cos) normalized to unit length).")


# ===========================================================
# 2) Train-only boundary-direction rotation
# ===========================================================
def _detect_sincos_indices(bnd_feat):
    sin_idx, cos_idx = 2, 3
    for try_s, try_c in [(2, 3), (3, 2)]:
        smin = np.nanmin(bnd_feat[:, try_s])
        cmin = np.nanmin(bnd_feat[:, try_c])
        if (smin < -0.1) and (cmin < -0.1):
            sin_idx, cos_idx = try_s, try_c
            break
    return sin_idx, cos_idx


def score_directions_train_only(bnd_feat, ds_sim, kcs2d, train_idx,
                                candidates=(0.0, 90.0, -90.0, 180.0)):
    """Mean cos(delta theta) per candidate rotation, on TRAIN indices only."""
    if 'dir' not in ds_sim:
        return {deg: float('nan') for deg in candidates}, None, None

    T = bnd_feat.shape[0]
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_idx = train_idx[train_idx < T]

    rad = np.deg2rad(ds_sim['dir'].values[:T][train_idx])
    tsin = np.sin(rad).astype(np.float32)
    tcos = np.cos(rad).astype(np.float32)

    sin_idx, cos_idx = _detect_sincos_indices(bnd_feat)
    sin_b = bnd_feat[train_idx, sin_idx]
    cos_b = bnd_feat[train_idx, cos_idx]

    mask = (kcs2d > 0)
    if mask.ndim != 2:
        mask = mask[0] if mask.ndim == 3 else mask
    mask = np.asarray(mask, bool)

    scores = {}
    for deg in candidates:
        r = np.deg2rad(deg)
        sin_r = sin_b * np.cos(r) + cos_b * np.sin(r)
        cos_r = cos_b * np.cos(r) - sin_b * np.sin(r)
        v = (sin_r * tsin + cos_r * tcos)
        vv = v[:, mask]
        scores[deg] = float(np.nanmean(vv))
    return scores, sin_idx, cos_idx


def auto_align_bnd_dir_trainonly(bnd_feat, ds_sim, kcs2d, train_idx,
                                 candidates=(0.0, 90.0, -90.0, 180.0)):
    """Select rotation on TRAIN indices only, apply to full bnd_feat."""
    scores, sin_idx, cos_idx = score_directions_train_only(
        bnd_feat, ds_sim, kcs2d, train_idx, candidates=candidates)
    if sin_idx is None:
        return 0.0, scores
    best_deg = max(scores, key=lambda d: scores[d])
    if abs(best_deg) > 1e-6:
        r = np.deg2rad(best_deg)
        sin_b = bnd_feat[:, sin_idx].copy()
        cos_b = bnd_feat[:, cos_idx].copy()
        bnd_feat[:, sin_idx] = sin_b * np.cos(r) + cos_b * np.sin(r)
        bnd_feat[:, cos_idx] = cos_b * np.cos(r) - sin_b * np.sin(r)
    return best_deg, scores


# ===========================================================
# 3) Chronological split
# ===========================================================
def make_chronological_split(time_index, seq_length, holdout_year,
                             val_frac_of_train=0.15):
    ti = pd.DatetimeIndex(time_index)
    T = len(ti); N = T - seq_length
    all_target_t = np.arange(seq_length, T, dtype=np.int64)
    years = ti.year.values
    is_holdout = (years[all_target_t] == int(holdout_year))
    boundary = np.where(np.diff(years[all_target_t].astype(np.int64)) != 0)[0]
    embargo_mask = np.zeros(all_target_t.shape[0], dtype=bool)
    for b in boundary:
        lo = max(0, b - seq_length + 1)
        hi = min(all_target_t.shape[0], b + seq_length + 1)
        embargo_mask[lo:hi] = True
    train_pool_mask = (~is_holdout) & (~embargo_mask)
    test_mask       = is_holdout & (~embargo_mask)
    train_pool = all_target_t[train_pool_mask]
    test_idx   = all_target_t[test_mask]
    if len(train_pool) == 0:
        raise RuntimeError("Empty training pool.")
    n_val = max(1, int(round(len(train_pool) * val_frac_of_train)))
    train_idx = np.sort(train_pool[:-n_val]).astype(np.int64)
    val_idx   = np.sort(train_pool[-n_val:]).astype(np.int64)
    test_idx  = np.sort(test_idx).astype(np.int64)
    train_idx = train_idx - seq_length
    val_idx   = val_idx   - seq_length
    test_idx  = test_idx  - seq_length
    train_idx = train_idx[(train_idx >= 0) & (train_idx < N)]
    val_idx   = val_idx[(val_idx   >= 0) & (val_idx   < N)]
    test_idx  = test_idx[(test_idx  >= 0) & (test_idx  < N)]
    return train_idx, val_idx, test_idx, f"chronological(holdout={holdout_year})"


# ===========================================================
# 4) Predicted (sin, cos) magnitude stats
# ===========================================================
@torch.no_grad()
def predicted_sincos_magnitude_stats(model, loader, device):
    model.eval()
    s_mag_sum = 0.0; s_mag_sq = 0.0; n_pixels = 0; n_in_band = 0
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb)
        main = pred[0] if isinstance(pred, list) else pred
        main = main.float()
        sin_ = main[:, 2].clamp(-1, 1)
        cos_ = main[:, 3].clamp(-1, 1)
        mag  = torch.sqrt(sin_ * sin_ + cos_ * cos_ + 1e-12)
        m_np = mag.detach().cpu().numpy().reshape(-1)
        s_mag_sum += float(m_np.sum())
        s_mag_sq  += float((m_np ** 2).sum())
        n_pixels  += int(m_np.size)
        n_in_band += int(((m_np >= 0.9) & (m_np <= 1.1)).sum())
    mean = s_mag_sum / max(1, n_pixels)
    var  = max(0.0, s_mag_sq / max(1, n_pixels) - mean ** 2)
    std  = var ** 0.5
    frac = n_in_band / max(1, n_pixels)
    return {"mean": mean, "std": std, "frac_in_band_0p9_1p1": frac,
            "n_pixels": int(n_pixels)}


# ===========================================================
# 5) Full per-variable test metrics (Hs, Tm, Dir)
# ===========================================================
@torch.no_grad()
def compute_test_metrics_all(model, loader, norm_params, spatial_w, device):
    """
    Area-weighted RMSE / MAE / mean-bias on the test loader.

    Returns physical-unit metrics:
      hs_rmse_m, hs_mae_m, hs_bias_m
      tm_rmse_s, tm_mae_s, tm_bias_s
      dir_crmse_deg, dir_cmae_deg, dir_cbias_deg
    """
    model.eval()
    hs_err2 = hs_abs = hs_bias = 0.0
    tm_err2 = tm_abs = tm_bias = 0.0
    dir_err2 = dir_abs = dir_bias = 0.0
    den = 0.0

    hs_min, hs_max = norm_params["hs"]
    tm_min, tm_max = norm_params["tm"]

    w_t = spatial_w.to(device).float()
    if w_t.dim() == 2:
        w_t = w_t.unsqueeze(0)   # broadcast over batch

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        out = model(xb)
        pred = out[0] if isinstance(out, list) else out
        pred = pred.float()

        phs = pred[:, 0] * (hs_max - hs_min) + hs_min
        ths = yb[:, 0]  * (hs_max - hs_min) + hs_min
        ptm = pred[:, 1] * (tm_max - tm_min) + tm_min
        ttm = yb[:, 1]  * (tm_max - tm_min) + tm_min

        ps = pred[:, 2]; pc = pred[:, 3]
        ts = yb[:, 2];  tc = yb[:, 3]
        eps = 1e-12
        pmag = torch.sqrt(ps * ps + pc * pc + eps)
        tmag = torch.sqrt(ts * ts + tc * tc + eps)
        ps = ps / pmag; pc = pc / pmag
        ts = ts / tmag; tc = tc / tmag
        pred_ang = torch.atan2(ps, pc)
        true_ang = torch.atan2(ts, tc)
        dtheta = torch.atan2(torch.sin(pred_ang - true_ang),
                             torch.cos(pred_ang - true_ang))
        dtheta_deg = dtheta * 180.0 / 3.141592653589793

        hs_diff = phs - ths
        tm_diff = ptm - ttm

        wb = w_t.expand_as(phs)
        hs_err2 += float((hs_diff * hs_diff * wb).sum().cpu())
        hs_abs  += float((hs_diff.abs()   * wb).sum().cpu())
        hs_bias += float((hs_diff         * wb).sum().cpu())
        tm_err2 += float((tm_diff * tm_diff * wb).sum().cpu())
        tm_abs  += float((tm_diff.abs()   * wb).sum().cpu())
        tm_bias += float((tm_diff         * wb).sum().cpu())
        dir_err2 += float((dtheta_deg * dtheta_deg * wb).sum().cpu())
        dir_abs  += float((dtheta_deg.abs()        * wb).sum().cpu())
        dir_bias += float((dtheta_deg              * wb).sum().cpu())
        den += float((torch.ones_like(phs) * wb).sum().cpu())

    den = max(den, 1e-12)
    return {
        "hs_rmse_m":     (hs_err2 / den) ** 0.5,
        "hs_mae_m":      hs_abs / den,
        "hs_bias_m":     hs_bias / den,
        "tm_rmse_s":     (tm_err2 / den) ** 0.5,
        "tm_mae_s":      tm_abs / den,
        "tm_bias_s":     tm_bias / den,
        "dir_crmse_deg": (dir_err2 / den) ** 0.5,
        "dir_cmae_deg":  dir_abs / den,
        "dir_cbias_deg": dir_bias / den,
    }
