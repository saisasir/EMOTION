from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import librosa
import os
from gtts import gTTS
import base64
import tempfile
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment  # For webm to wav conversion

# ---------------- Configuration ----------------
SAMPLE_RATE = 16000
MAX_TIME = 256

# Load model and label encoder paths from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'default_model_path')  # Default path for local testing
LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', 'default_label_encoder_path')  # Default path for local testing

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------------- Model Definition ----------------
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].to(x.device)

class CNNTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc_cnn = torch.nn.Linear(32 * (input_dim // 4) * (MAX_TIME // 4), 128)
        self.pos_enc = PositionalEncoding(128)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_cnn(x)
        x = x.unsqueeze(1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.classifier(x.squeeze(1))

# ---------------- Load Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
input_dim = 40 + 40 + 40 + 128 + 1 + 1 + 1
model = CNNTransformer(input_dim=input_dim, num_classes=len(label_classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- Utilities ----------------
def extract_features(y, sr):
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.concatenate([mfcc, delta, delta2, mel_db, zcr, rmse, rolloff], axis=0)
    if features.shape[1] < MAX_TIME:
        pad = MAX_TIME - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)), mode='constant')
    else:
        features = features[:, :MAX_TIME]

    return features

def generate_response_text(emotion):
    responses = {
        "happy": "That's wonderful! I'm here to make your day even brighter.",
        "sad": "It's okay to feel down. Let's find something uplifting together.",
        "angry": "I hear your frustration. Let's work through it calmly.",
        "neutral": "Ready to assist. How can I help you today?",
        "fear": "You're safe. I'm right here with you. Let's figure it out.",
        "surprised": "Wow! That caught you off guard, huh? Let's explore it.",
        "disgusted": "Hmm, that didn't sit well with you. I understand. Let me help."
    }
    return responses.get(emotion.lower(), "I'm here to assist, whatever you feel. Let's begin.")

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    with open(temp_audio_path, "rb") as audio_file:
        audio_binary = audio_file.read()
        audio_base64 = base64.b64encode(audio_binary).decode("utf-8")

    os.remove(temp_audio_path)
    return audio_base64

# ---------------- Routes ----------------
@app.route('/')
def home():
    return jsonify({"message": "Emotion Recognition API is running"})

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    try:
        audio_file = request.files['audio_file']
        temp_audio_path = "temp_audio.webm"
        audio_file.save(temp_audio_path)

        AudioSegment.converter = "ffmpeg"  # Only needed if ffmpeg isn't already in PATH

        try:
            sound = AudioSegment.from_file(temp_audio_path)
            temp_wav_path = "converted.wav"
            sound.export(temp_wav_path, format="wav")
            y, sr = librosa.load(temp_wav_path, sr=SAMPLE_RATE)
            os.remove(temp_wav_path)
        except Exception as e:
            print("Audio conversion failed:", e)
            return jsonify({"error": "Failed to convert audio file"}), 400

        os.remove(temp_audio_path)

        features = extract_features(y, sr)
        features_tensor = torch.tensor(features).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            emotion = label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx] * 100

        response_text = generate_response_text(emotion)
        audio_base64 = text_to_speech(response_text)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "text_response": response_text,
            "audio_base64": audio_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
