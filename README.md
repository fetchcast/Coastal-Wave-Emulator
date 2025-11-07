# Coastal-Wave-Emulator (Inference Only)

A minimal, inference-only pipeline for a UNet++–ConvLSTM emulator of nearshore waves.

1.What you get: lightweight code to run inference, a small sample dataset (10 steps), and final weights trained with sequence length L=6.

2.What’s not here: training scripts and large datasets (kept out to keep things simple)

**📂 Dataset Structure:**

    ├── assets/                       # normalization params
    │   └── norm_params_pctl.json
    ├── weights/                      # Weights
    │   └── 20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth
    ├── data/                         # Input NetCDFs 
    │   ├── sample_0010.zip           # Compressed demo (10 steps) — unzip first
    │   └── sample_0010_with_bnd.zip  # Compressed demo with boundary information(10 steps) — unzip first
    ├── src/ 
    │   └── swan_emul/
    │       ├── __init__.py
    │       ├── model.py              # UNet++–ConvLSTM model
    │       ├── dataio.py             # NetCDF → tensors, masks, channels
    │       ├── norm.py               # Normalization utils
    │       └── inference.py          # Inference helpers
    ├── main.py                       # CLI entry point
    ├── requirements.txt              # Python dependencies
    └── README.md
    
## Study Region & final results from weight (Typhoon Maysak)

<p align="center">
  <img src="https://github.com/fetchcast/Coastal-Wave-Emulator/blob/main/figure/maysak_hs.gif" alt="SR" width="600"/>
</p>


    
# Quick start
# 1) Install

python -m venv .venv
activate your venv, then:
pip install -r requirements.txt

# 2) Unzip the sample data

The sample is compressed as data/sample_0010.zip.
Unzip it first to get data/sample_0010.nc (10 time steps).

# 3) Run inference

Case A — .nc already contains boundary channels
(Example: data/sample_0010_with_bnd.nc has hs_bnd, tm_bnd, sin_dir_bnd, cos_dir_bnd)

python main.py --checkpoint weights/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth --input_nc data/sample_0010_with_bnd.nc --norm_json assets/norm_params_pctl.json --seq_len 6 --bnd on --device cpu --outdir outputs/demo --denorm off --show

Case B — .nc does not contain boundary channels
(Example: data/sample_0010.nc has no boundary channels)

python main.py --checkpoint weights/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth --input_nc data/sample_0010.nc --norm_json assets/norm_params_pctl.json --seq_len 6 --bnd auto --device cpu --outdir outputs/demo --denorm off --show

<Notes>
The checkpoint was trained with L = 6. If your input length is T = 10, the model produces T − L = 4 prediction frames.

--bnd
on: requires boundary channels in the input and uses them.
auto: uses them if present; otherwise fills zeros (OK for smoke tests).
off: ignores boundary channels even if present.

Outputs are already in physical units (Hs in meters, Tm in seconds, direction via sin/cos), so --denorm off is essential.
Use --device cuda if you have a GPU; otherwise use --device cpu.

# 4) Outputs

hs — (N, H, W) significant wave height [m],
tm — (N, H, W) mean wave period [s],
dir — (N, H, W) mean wave direction [deg, 0–360)
