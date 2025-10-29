# Coastal-Wave-Emulator (Inference Only)

A minimal, inference-only pipeline for a UNet++–ConvLSTM emulator of nearshore waves.

1.What you get: lightweight code to run inference, a small sample dataset (10 steps), and final weights trained with sequence length L=6.

2.What’s not here: training scripts and large datasets (kept out to keep things simple)

**📂 Dataset Structure:**

    ├── assets/                     # normalization params
    │   └── norm_params.json
    ├── weights/                     # Weights
    │   └── 20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth
    ├── data/                       # Input NetCDFs 
    │   └── sample_0010.zip         # Compressed demo (10 steps) — unzip first
    ├── src/ 
    │   └── swan_emul/
    │       ├── __init__.py
    │       ├── model.py            # UNet++–ConvLSTM model
    │       ├── dataio.py           # NetCDF → tensors, masks, channels
    │       ├── norm.py             # Normalization utils
    │       └── inference.py        # Inference helpers
    ├── main.py                     # CLI entry point
    ├── requirements.txt            # Python dependencies
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

python main.py --checkpoint weights/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth --input_nc data/sample_0010.nc --norm_json assets/norm_params.json --seq_len 6 --bnd auto --device cpu --outdir outputs/demo --show

These weights were trained with L=6. The 10-step sample works out of the box (it will produce 10 − 6 = 4 predictions).

--bnd auto: if boundary channels are missing in the sample, they are zero-filled internally (fine for smoke tests).

Use --device cuda if you have a GPU.

# 4) Outputs

hs — (N, H, W) significant wave height [m]
tm — (N, H, W) mean wave period [s]
dir — (N, H, W) mean wave direction [deg, 0–360)
