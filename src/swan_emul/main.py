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
    np.savez(os.path.join(a.outdir, "predictions.npz"), hs=hs, tm=tm, dir=ang)

if __name__ == "__main__":
    main()
