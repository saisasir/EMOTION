# prestart.py
import os
import gdown

os.makedirs("models", exist_ok=True)

MODEL_PATH = "./models/cnn_transformer_ser.pt"
LABEL_ENCODER_PATH = "./models/label_encoder.npy"

if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1LorueP9xWUG4dwGAwV2nwVRUqUpNdYvR", MODEL_PATH, quiet=False)

if not os.path.exists(LABEL_ENCODER_PATH):
    gdown.download("https://drive.google.com/uc?id=1AzmDvf2sQ5nxWBkEFh0g_geEi5Rkmg7w", LABEL_ENCODER_PATH, quiet=False)
