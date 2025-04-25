import os
import torch
import streamlit as st
import librosa
import requests
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Optional mic input (commented out to avoid cloud errors)
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    MIC_ENABLED = True
except:
    MIC_ENABLED = False

# Directory setup
SCRIPT_DIR = os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "wav2vec2_tiny_quantized")
os.makedirs(MODEL_PATH, exist_ok=True)

# Function to download files from GitHub
def download_from_github(file_name):
    url = f"https://github.com/your-username/your-repo/raw/main/{file_name}"
    response = requests.get(url)
    with open(os.path.join(MODEL_PATH, file_name), "wb") as f:
        f.write(response.content)
    return os.path.join(MODEL_PATH, file_name)

# Load/download model and processor
@st.cache_resource
def load_model():
    # Download necessary files from GitHub if not already present
    required_files = ["pytorch_model_quantized.pt", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]

    for file in required_files:
        file_path = os.path.join(MODEL_PATH, file)
        if not os.path.exists(file_path):
            st.info(f"🔄 Downloading {file} from GitHub...")
            download_from_github(file)

    try:
        # Load the processor using preprocessor_config.json file
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

        # Load the quantized model
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
        model.eval()  # Set the model to evaluation mode
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Optional: Record audio using mic
def record_audio(duration=5, filename="recorded.wav"):
    fs = 16000
    st.info("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    st.success(f"✅ Recorded and saved: {filename}")
    return filename

# Transcription function
def transcribe(file_path, processor, model):
    try:
        audio, _ = librosa.load(file_path, sr=16000)
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🗣️ Wav2Vec2 Transcriber", layout="centered")
st.title("🎧 Tiny Wav2Vec2 Transcriber")

# Load model
processor, model = load_model()

if processor is None or model is None:
    st.error("❌ Failed to load the model. Please check the logs.")
else:
    if MIC_ENABLED:
        option = st.radio("Choose input method:", ("🎤 Record Audio", "📁 Upload File"))
    else:
        st.warning("⚠️ Microphone input disabled (PortAudio not available). Only upload supported.")
        option = "📁 Upload File"

    if option == "🎤 Record Audio":
        if st.button("⏺️ Record 5s"):
            file_path = record_audio()
            transcription = transcribe(file_path, processor, model)
            st.subheader("📝 Transcription:")
            st.success(transcription)

    elif option == "📁 Upload File":
        uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
        if uploaded_file is not None:
            with open("uploaded.wav", "wb") as f:
                f.write(uploaded_file.read())
            transcription = transcribe("uploaded.wav", processor, model)
            st.subheader("📝 Transcription:")
            st.success(transcription)
