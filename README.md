# Coastal-Wave-Emulator (Inference Only)

A minimal, inference-only pipeline for a UNet++–ConvLSTM emulator of nearshore waves.

1.What you get: lightweight code to run inference, a small sample dataset (10 steps), and final weights trained with sequence length L=6.

2.What’s not here: training scripts and large datasets (kept out to keep things simple)

|   LICENSE
|   README.md
|   requirements.txt
|   tree.txt
|   
+---assets
|       norm_params.json
|       
+---data
|       sample_0010.zip
|       
+---src
|   \---swan_emul
|           dataio.py
|           inference
|           main
|           model
|           norm
|           __init__.py
|           
\---weights
        20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth
        20250919_110000_model_weights_17498_seq12_epochs20_hid128_UNET32_bndON.pth
        model.pt
        


# Quick start
# 1) Install

python -m venv .venv
activate your venv, then:
pip install -r requirements.txt

# 2) Unzip the sample data

The sample is compressed as data/sample_0010.zip.
Unzip it first to get data/sample_0010.nc (10 time steps).

# 3) Run inference

python main.py \
  --checkpoint assets/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth \
  --input_nc data/sample_0010.nc \
  --norm_json assets/norm_params.json \
  --seq_len 6 \
  --bnd auto \
  --device cpu \
  --outdir outputs/demo

These weights were trained with L=6. The 10-step sample works out of the box (it will produce 10 − 6 = 4 predictions).

--bnd auto: if boundary channels are missing in the sample, they are zero-filled internally (fine for smoke tests).

Use --device cuda if you have a GPU.

# 4) Outputs

outputs/demo/predictions.npz containing:

hs — (N, H, W) significant wave height [m]
tm — (N, H, W) mean wave period [s]
dir — (N, H, W) mean wave direction [deg, 0–360)
