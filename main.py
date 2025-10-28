import argparse, json, os, numpy as np
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
    return p.parse_args()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_nc", required=True)
    p.add_argument("--norm_json", required=True)
    p.add_argument("--seq_len", type=int, default=6)
    p.add_argument("--bnd", choices=["on","off","auto"], default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", required=True)
    p.add_argument("--save_nc", type=str, default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    out = run_inference(
        checkpoint=args.checkpoint,
        input_nc=args.input_nc,
        norm_json=args.norm_json,
        seq_len=args.seq_len,
        bnd_mode=args.bnd,
        device=args.device,
    )
    hs, tm, dr = out["hs"], out["tm"], out["dir"]   # (N,H,W)
    lon = out.get("lon", None)
    lat = out.get("lat", None)

    out_npz = Path(args.outdir) / "predictions.npz"
    np.savez_compressed(
        out_npz,
        hs=hs, tm=tm, dir=dr,
        **({"lon": lon, "lat": lat} if lon is not None and lat is not None else {})
    )
    print(f"saved: {out_npz}")

    if args.save_nc:
        from src.swan_emul.dataio import save_nc 
        save_nc(hs, tm, dr, args.save_nc, lon=lon, lat=lat)
        print(f"saved: {args.save_nc}")

    if args.show:
        import matplotlib.pyplot as plt
        if hs.shape[0] == 0:
            print("no frames to show"); return
        img = hs[0]
        if lon is not None and lat is not None:
            extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
            plt.imshow(img, origin="lower", extent=extent, aspect="auto")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
        else:
            plt.imshow(img, origin="lower")
        plt.title("Hs [m] @ t0"); plt.colorbar(); plt.show()

if __name__ == "__main__":
    main()
