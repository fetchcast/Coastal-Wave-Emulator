import numpy as np, xarray as xr
from .norm import norm, depth_grad_mag

REQ = ["windu","windv","depth","veloc-x","veloc-y","hsign","period","dir","x","y","kcs"]

def load_xr(path):
    ds = xr.open_dataset(path)
    for v in REQ:
        if v not in ds: raise ValueError(f"missing var: {v}")
    return ds

def build_inputs(ds, normp, T=None, bnd_mode="auto"):
    if T is None: T = ds.dims.get("time", None)
    u  = norm(ds["windu"].values[:T],   *normp["wind_u"])
    v  = norm(ds["windv"].values[:T],   *normp["wind_v"])
    d  = norm(ds["depth"].values[:T],   *normp["depth"])
    ux = norm(ds["veloc-x"].values[:T], *normp["veloc_x"])
    uy = norm(ds["veloc-y"].values[:T], *normp["veloc_y"])

    rad = np.deg2rad(ds["dir"].values[:T].astype("float32"))
    s, c = np.sin(rad), np.cos(rad)
    hs = norm(ds["hsign"].values[:T],  *normp["hs"])
    tm = norm(ds["period"].values[:T], *normp["tm"])

    dep2 = ds["depth"].values
    dep2 = dep2[0] if dep2.ndim == 3 else dep2
    g = depth_grad_mag(dep2)
    H, W = hs.shape[-2], hs.shape[-1]
    g3 = np.broadcast_to(g[None, ...], (len(hs), H, W)).astype("float32")

    X = [u, v, d, ux, uy, g3]

    has_bnd = all(k in ds for k in ("bnd_hs","bnd_tm","bnd_dir"))
    if bnd_mode == "on" and not has_bnd:
        raise ValueError("bnd_mode=on, but no bnd files found")
    if bnd_mode in ("on","auto") and has_bnd:
        rb = np.deg2rad(ds["bnd_dir"].values[:T].astype("float32"))
        sb, cb = np.sin(rb), np.cos(rb)
        hb = norm(ds["bnd_hs"].values[:T], *normp["hs"])
        tb = norm(ds["bnd_tm"].values[:T], *normp["tm"])
        X += [hb, tb, sb, cb]
    elif bnd_mode in ("on","auto"):
        zero = np.zeros_like(hs, dtype="float32")
        X += [zero, zero, zero, zero]
    # bnd_mode == "off" 

    X = np.stack(X, 1).astype("float32")
    Y = np.stack([hs, tm, s, c], 1).astype("float32")
    lon = ds["x"].values; lat = ds["y"].values; kcs = ds["kcs"].values
    return X, Y, lon, lat, kcs
