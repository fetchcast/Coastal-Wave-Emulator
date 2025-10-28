import argparse, json, os, numpy as np
from pathlib import Path
from src.swan_emul.model import load_model
from src.swan_emul.norm import load_norm_params, denorm
from src.swan_emul.dataio import load_xr, build_inputs
from src.swan_emul.inference import run_inference

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_nc",   required=True)
    p.add_argument("--norm_json",  required=True)
    p.add_argument("--seq_len",    type=int, default=12)
    p.add_argument("--batch",      type=int, default=1)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--outdir",     default="outputs")
    p.add_argument("--bnd", choices=["on","auto","off"], default="auto")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save_png", type=str, default=None)
    return p.parse_args()

def _safe_extent(lon, lat):
    try:
        if lon is None or lat is None:
            return None
        lon = np.asarray(lon); lat = np.asarray(lat)
        mask = np.isfinite(lon) & np.isfinite(lat)
        if not mask.any():
            return None
        xmin = float(np.nanmin(lon[mask])); xmax = float(np.nanmax(lon[mask]))
        ymin = float(np.nanmin(lat[mask])); ymax = float(np.nanmax(lat[mask]))
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            return None
        return [xmin, xmax, ymin, ymax]
    except Exception:
        return None

def main():
    a = parse()
    os.makedirs(a.outdir, exist_ok=True)
    normp = load_norm_params(a.norm_json)
    ds = load_xr(a.input_nc)
    X, Y, lon, lat, kcs = build_inputs(ds, normp, bnd_mode=a.bnd)
    m = load_model(a.checkpoint, in_ch=X.shape[1], out_ch=4, hidden=128, maploc=a.device)
    pred = run_inference(m, X, L=a.seq_len, batch=a.batch, device=a.device)

    hs = denorm(pred[:,0], *normp["hs"])
    tm = denorm(pred[:,1], *normp["tm"])
    ang = (np.rad2deg(np.arctan2(pred[:,2], pred[:,3])) % 360.0)

    out_npz = os.path.join(a.outdir, "predictions.npz")
    np.savez(out_npz, hs=hs, tm=tm, dir=ang)
    print(f"saved: {out_npz}")

    if a.show or a.save_png:
        if hs.shape[0] == 0:
            print("no frames to plot (T - seq_len <= 0)")
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            extent = _safe_extent(lon, lat)
            if extent is None:
                plt.imshow(hs[0], origin="lower")
            else:
                plt.imshow(hs[0], origin="lower", extent=extent, aspect="auto")
                plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.title("Hs [m] @ t0"); plt.colorbar()
            if a.save_png:
                Path(a.save_png).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(a.save_png, dpi=150, bbox_inches="tight")
                print(f"saved: {a.save_png}")
            if a.show:
                plt.show()
            else:
                plt.close(fig)

if __name__ == "__main__":
    main()
